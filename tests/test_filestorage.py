"""Tests for `raft.net.filestorage.FileStorage`.

The core check is a model test: Hypothesis drives random sequences of
appends, truncations, vote saves, syncs, clean reopens and crashes
against both `FileStorage` and `MemoryStorage`, and after every step the
two must agree. A crash is modelled on the `MemoryStorage` side by
rolling it back to its state at the last point `FileStorage` promised
durability -- every write in ``per_write`` mode, every `sync()` in
``deferred`` mode.

The rest pins down recovery: torn final records are cut, damage with
valid data after it raises, and a crash part-way through a deferred
`sync()` loses nothing that an earlier `sync()` made durable.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from raft.net import raft_pb2 as pb
from raft.net.codec import decode_log_entry, encode_log_entry
from raft.net.filestorage import LOCK_FILE, LOG_FILE, STATE_FILE, FileStorage, FsyncMode
from raft.statemachine import Put
from raft.storage import SENTINEL, LogEntry, MemoryStorage, Storage, StorageError

MODES: list[FsyncMode] = ["per_write", "deferred"]
HEADER = 8


def encode(entry: LogEntry) -> bytes:
    return encode_log_entry(entry).SerializeToString()


def decode(data: bytes) -> LogEntry:
    return decode_log_entry(pb.LogEntry.FromString(data))


def open_storage(directory: Path, fsync: FsyncMode = "per_write") -> FileStorage:
    return FileStorage(directory, encode=encode, decode=decode, fsync=fsync)


def crash(storage: FileStorage) -> None:
    """Drop `storage` the way a killed process would: no sync, no flush,
    and the kernel releases its lock."""
    os.close(storage._fd)
    storage._fd = -1
    os.close(storage._lock_fd)


def entries(*commands: str, term: int = 1) -> list[LogEntry]:
    return [LogEntry(term=term, command=Put(key=c, value=c)) for c in commands]


def record_offsets(path: Path) -> list[tuple[int, int]]:
    """(start, end) byte range of every record in an undamaged log file."""
    data = path.read_bytes()
    offsets = []
    pos = 0
    while pos < len(data):
        end = pos + HEADER + int.from_bytes(data[pos : pos + 4], "little")
        offsets.append((pos, end))
        pos = end
    return offsets


def flip(data: bytes, pos: int) -> bytes:
    return data[:pos] + bytes([data[pos] ^ 0xFF]) + data[pos + 1 :]


# -- Model test --

log_entries = st.builds(
    LogEntry,
    term=st.integers(0, 5),
    command=st.one_of(
        st.none(), st.builds(Put, key=st.text(max_size=5), value=st.text(max_size=5))
    ),
)
operations = st.lists(
    st.one_of(
        st.tuples(st.just("append"), st.lists(log_entries, max_size=4)),
        st.tuples(st.just("truncate"), st.integers(-1, 12)),
        st.tuples(
            st.just("vote"),
            st.tuples(st.integers(-1, 5), st.one_of(st.none(), st.sampled_from(["a", "b", 5]))),
        ),
        st.tuples(st.just("sync"), st.none()),
        st.tuples(st.just("read"), st.tuples(st.integers(-2, 14), st.integers(-1, 6))),
        st.tuples(st.just("reopen"), st.none()),
        st.tuples(st.just("crash"), st.none()),
    ),
    max_size=30,
)


def snapshot(storage: MemoryStorage) -> tuple[list[LogEntry], tuple[int, str | None]]:
    return storage.load_log(), storage.load_term_and_vote()


def restore(snap: tuple[list[LogEntry], tuple[int, str | None]]) -> MemoryStorage:
    storage = MemoryStorage()
    log, (term, voted_for) = snap
    storage.save_term_and_vote(term, voted_for)
    storage.append_entries(log[1:])
    return storage


def outcome(call: Callable[[Storage], object], storage: Storage) -> object:
    """What `call(storage)` returned, or the type of what it raised."""
    try:
        return call(storage)
    except Exception as exc:
        return type(exc)


@pytest.mark.parametrize("mode", MODES)
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(ops=operations)
def test_matches_memory_storage(mode: FsyncMode, ops: list[tuple[str, object]]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        files = open_storage(directory, mode)
        model = MemoryStorage()
        durable = snapshot(model)
        try:
            for op, arg in ops:
                if op == "append":
                    assert isinstance(arg, list)
                    calls = [lambda s, a=arg: s.append_entries(a)]
                elif op == "truncate":
                    calls = [lambda s, a=arg: s.truncate_from(a)]
                elif op == "vote":
                    assert isinstance(arg, tuple)
                    calls = [lambda s, a=arg: s.save_term_and_vote(*a)]
                elif op == "read":  # bounded reads, including out-of-range ones
                    assert isinstance(arg, tuple)
                    calls = [
                        lambda s, a=arg: s.entry_at(a[0]),
                        lambda s, a=arg: s.term_at(a[0]),
                        lambda s, a=arg: s.entries_from(*a),
                    ]
                elif op == "sync":
                    files.sync()
                    durable = snapshot(model)
                    calls = []
                elif op == "reopen":
                    files.close()
                    files = open_storage(directory, mode)
                    durable = snapshot(model)
                    calls = []
                else:  # crash
                    crash(files)
                    files = open_storage(directory, mode)
                    model = restore(durable)
                    calls = []

                for call in calls:
                    assert outcome(call, files) == outcome(call, model)
                    if mode == "per_write":
                        durable = snapshot(model)

                assert files.load_log() == model.load_log()
                assert files.load_term_and_vote() == model.load_term_and_vote()
                assert files.last_log_index() == model.last_log_index()
                assert files.last_log_term() == model.last_log_term()
                # The bounded reads agree with the full copy, everywhere.
                log = files.load_log()
                assert [files.entry_at(i) for i in range(len(log))] == log
                assert [files.term_at(i) for i in range(len(log))] == [e.term for e in log]
                assert files.entries_from(0, len(log)) == log
                assert files.entries_from(len(log), 3) == []
        finally:
            files.close()


# -- Reopen --


@pytest.mark.parametrize("mode", MODES)
def test_reopen_after_writes(tmp_path: Path, mode: FsyncMode) -> None:
    with open_storage(tmp_path, mode) as storage:
        storage.save_term_and_vote(3, "n2")
        storage.append_entries(entries("a", "b", "c", "d"))
        storage.truncate_from(3)
        storage.append_entries(entries("x", term=3))
        storage.append_entries([LogEntry(term=3, command=None)])

    with open_storage(tmp_path, mode) as storage:
        assert storage.load_term_and_vote() == (3, "n2")
        assert storage.load_log() == [
            SENTINEL,
            *entries("a", "b"),
            *entries("x", term=3),
            LogEntry(term=3, command=None),
        ]
        assert storage.last_log_index() == 4
        assert storage.last_log_term() == 3


def test_fresh_directory_is_empty(tmp_path: Path) -> None:
    with open_storage(tmp_path / "new" / "dir") as storage:
        assert storage.load_term_and_vote() == (0, None)
        assert storage.load_log() == [SENTINEL]


def test_state_file_is_replaced_whole(tmp_path: Path) -> None:
    with open_storage(tmp_path) as storage:
        storage.save_term_and_vote(7, None)
        assert json.loads((tmp_path / STATE_FILE).read_text()) == {"term": 7, "voted_for": None}
        assert sorted(p.name for p in tmp_path.iterdir()) == [LOCK_FILE, LOG_FILE, STATE_FILE]


# -- Locking --


def test_second_open_of_a_directory_is_refused(tmp_path: Path) -> None:
    first = open_storage(tmp_path)
    first.append_entries(entries("a"))
    with pytest.raises(StorageError, match="already open"):
        open_storage(tmp_path)
    first.append_entries(entries("b"))  # the holder is unaffected
    first.close()

    with open_storage(tmp_path) as second:  # released on close
        assert second.load_log() == [SENTINEL, *entries("a", "b")]


def test_lock_is_released_when_a_crashed_process_exits(tmp_path: Path) -> None:
    crash(open_storage(tmp_path))
    open_storage(tmp_path).close()


def test_lock_is_released_when_open_fails(tmp_path: Path) -> None:
    (tmp_path / STATE_FILE).write_bytes(b"{not json")
    for _ in range(2):  # not "already open" the second time
        with pytest.raises(StorageError, match="not valid JSON"):
            open_storage(tmp_path)


# -- Torn tails --


def _cut_into_header(data: bytes, start: int, end: int) -> bytes:
    return data[: start + 3]


def _header_only(data: bytes, start: int, end: int) -> bytes:
    return data[: start + HEADER]


def _cut_into_payload(data: bytes, start: int, end: int) -> bytes:
    return data[: end - 1]


def _bad_payload_crc(data: bytes, start: int, end: int) -> bytes:
    return flip(data, end - 1)


def _bad_crc_field(data: bytes, start: int, end: int) -> bytes:
    return flip(data, start + 5)


def _length_too_long(data: bytes, start: int, end: int) -> bytes:
    return flip(data, start + 3)  # top byte of the length: now ~4 GiB


def _zero_tail(data: bytes, start: int, end: int) -> bytes:
    """A filesystem extended the file but the data never landed."""
    return data[:start] + bytes(end - start + 64)


@pytest.mark.parametrize(
    "damage",
    [
        _cut_into_header,
        _header_only,
        _cut_into_payload,
        _bad_payload_crc,
        _bad_crc_field,
        _length_too_long,
        _zero_tail,
    ],
)
def test_torn_final_record_is_cut(
    tmp_path: Path, damage: Callable[[bytes, int, int], bytes]
) -> None:
    with open_storage(tmp_path) as storage:
        storage.append_entries(entries("a", "b", "c"))
    log_path = tmp_path / LOG_FILE
    start, end = record_offsets(log_path)[-1]
    log_path.write_bytes(damage(log_path.read_bytes(), start, end))

    with open_storage(tmp_path) as storage:
        assert storage.load_log() == [SENTINEL, *entries("a", "b")]
        assert log_path.stat().st_size == start
        storage.append_entries(entries("d"))

    with open_storage(tmp_path) as storage:
        assert storage.load_log() == [SENTINEL, *entries("a", "b", "d")]


def test_every_truncation_point_recovers(tmp_path: Path) -> None:
    with open_storage(tmp_path) as storage:
        storage.append_entries(entries("a", "b", "c"))
    log_path = tmp_path / LOG_FILE
    data = log_path.read_bytes()
    ends = [0] + [end for _, end in record_offsets(log_path)]

    for cut in range(len(data) + 1):
        log_path.write_bytes(data[:cut])
        kept = max(i for i, end in enumerate(ends) if end <= cut)
        with open_storage(tmp_path) as storage:
            assert storage.load_log() == [SENTINEL, *entries("a", "b", "c")[:kept]]


# -- Corruption --


def _bad_first_payload(data: bytes, records: list[tuple[int, int]]) -> bytes:
    return flip(data, records[0][1] - 1)


def _bad_first_length(data: bytes, records: list[tuple[int, int]]) -> bytes:
    return flip(data, records[0][0] + 3)


def _zeroed_middle_record(data: bytes, records: list[tuple[int, int]]) -> bytes:
    start, end = records[1]
    return data[:start] + bytes(end - start) + data[end:]


def _garbage_between_records(data: bytes, records: list[tuple[int, int]]) -> bytes:
    split = records[0][1]
    return data[:split] + b"\x01\x02\x03" + data[split:]


@pytest.mark.parametrize(
    "damage",
    [_bad_first_payload, _bad_first_length, _zeroed_middle_record, _garbage_between_records],
)
def test_damage_followed_by_valid_data_raises(
    tmp_path: Path, damage: Callable[[bytes, list[tuple[int, int]]], bytes]
) -> None:
    with open_storage(tmp_path) as storage:
        storage.append_entries(entries("a", "b", "c"))
    log_path = tmp_path / LOG_FILE
    damaged = damage(log_path.read_bytes(), record_offsets(log_path))
    log_path.write_bytes(damaged)

    with pytest.raises(StorageError, match="followed by a valid record"):
        open_storage(tmp_path)
    assert log_path.read_bytes() == damaged  # never "repaired"


def test_undecodable_entry_raises(tmp_path: Path) -> None:
    with open_storage(tmp_path) as storage:
        storage.append_entries(entries("a"))

    def broken_decode(data: bytes) -> LogEntry:
        raise ValueError("nope")

    with pytest.raises(StorageError, match="entry 1 does not decode"):
        FileStorage(tmp_path, encode=encode, decode=broken_decode)


@pytest.mark.parametrize(
    "contents",
    [
        b"{not json",
        b"[1, 2]",
        b'{"term": -1, "voted_for": null}',
        b'{"term": true, "voted_for": null}',
        b'{"voted_for": null}',
        b'{"term": 1, "voted_for": 5}',
    ],
)
def test_bad_state_file_raises(tmp_path: Path, contents: bytes) -> None:
    (tmp_path / STATE_FILE).write_bytes(contents)
    with pytest.raises(StorageError):
        open_storage(tmp_path)


# -- Deferred mode and crashes --


def test_deferred_writes_nothing_before_sync(tmp_path: Path) -> None:
    storage = open_storage(tmp_path, "deferred")
    storage.save_term_and_vote(2, "n1")
    storage.append_entries(entries("a", "b"))
    assert not (tmp_path / STATE_FILE).exists()
    assert (tmp_path / LOG_FILE).stat().st_size == 0

    storage.sync()
    storage.append_entries(entries("c"))
    storage.truncate_from(1)
    storage.append_entries(entries("d", term=2))
    storage.save_term_and_vote(3, None)
    crash(storage)

    with open_storage(tmp_path, "deferred") as storage:
        assert storage.load_term_and_vote() == (2, "n1")
        assert storage.load_log() == [SENTINEL, *entries("a", "b")]


class SimulatedCrash(Exception):
    pass


@contextmanager
def write_budget(monkeypatch: pytest.MonkeyPatch, budget: int | None) -> Iterator[list[int]]:
    """Patch `os.pwrite` to write `budget` bytes in total, then crash.

    With `budget=None` it never crashes; the yielded list collects how
    many bytes each call wrote.
    """
    real_pwrite = os.pwrite
    written: list[int] = []

    def pwrite(fd: int, data: bytes, offset: int) -> int:
        allowed = len(data) if budget is None else min(len(data), budget - sum(written))
        if allowed:
            written.append(real_pwrite(fd, bytes(data)[:allowed], offset))
        if allowed < len(data):
            raise SimulatedCrash
        return allowed

    with monkeypatch.context() as patch:
        patch.setattr(os, "pwrite", pwrite)
        yield written


def _deferred_crash_scenario(
    directory: Path, monkeypatch: pytest.MonkeyPatch, budget: int | None
) -> int:
    storage = open_storage(directory, "deferred")
    storage.save_term_and_vote(1, "n1")
    storage.append_entries(entries("a", "b"))
    storage.sync()

    storage.truncate_from(2)
    storage.append_entries(entries("x", "y", "z", term=2))
    storage.save_term_and_vote(2, "n2")
    with write_budget(monkeypatch, budget) as written, suppress(SimulatedCrash):
        storage.sync()
    crash(storage)
    return sum(written)


def test_crash_during_deferred_sync_loses_only_unsynced_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    total = _deferred_crash_scenario(tmp_path / "dry-run", monkeypatch, None)
    synced_log = [SENTINEL, *entries("a", "b")]
    new_log = [SENTINEL, *entries("a"), *entries("x", "y", "z", term=2)]

    seen_logs = set()
    for budget in range(total + 1):
        directory = tmp_path / str(budget)
        _deferred_crash_scenario(directory, monkeypatch, budget)

        with open_storage(directory, "deferred") as storage:
            log = storage.load_log()
            assert storage.load_term_and_vote() in [(1, "n1"), (2, "n2")]
            # Either untouched, or the new log's prefix past the truncation
            # point -- never anything synced that was later lost, never a
            # torn record.
            assert log == synced_log or (log == new_log[: len(log)] and len(log) >= 2)
            seen_logs.add(len(log) if log != synced_log else -1)
        if budget == 0:
            assert (storage.load_term_and_vote(), log) == ((1, "n1"), synced_log)
        if budget == total:
            assert (storage.load_term_and_vote(), log) == ((2, "n2"), new_log)

    assert seen_logs == {-1, 2, 3, 4, 5}  # every intermediate state was reached


def test_failed_sync_makes_storage_unusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = open_storage(tmp_path, "deferred")
    storage.append_entries(entries("a"))
    with write_budget(monkeypatch, 0), pytest.raises(SimulatedCrash):
        storage.sync()

    for call in (
        storage.sync,
        lambda: storage.append_entries(entries("b")),
        lambda: storage.truncate_from(1),
        lambda: storage.save_term_and_vote(1, None),
    ):
        with pytest.raises(StorageError, match="reopen"):
            call()
    storage.close()  # releases the file without trying to sync


# -- Misuse --


def test_unencodable_entry_changes_nothing(tmp_path: Path) -> None:
    with open_storage(tmp_path, "deferred") as storage:
        storage.append_entries(entries("a"))
        with pytest.raises(TypeError):
            storage.append_entries([*entries("b"), LogEntry(term=1, command=Put("k", 5))])
        assert storage.load_log() == [SENTINEL, *entries("a")]
    with open_storage(tmp_path) as storage:
        assert storage.load_log() == [SENTINEL, *entries("a")]


def test_invalid_fsync_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fsync"):
        FileStorage(tmp_path, encode=encode, decode=decode, fsync="never")  # type: ignore[arg-type]


def test_closed_storage_rejects_writes(tmp_path: Path) -> None:
    storage = open_storage(tmp_path)
    storage.close()
    storage.close()  # idempotent
    with pytest.raises(StorageError, match="closed"):
        storage.append_entries(entries("a"))
