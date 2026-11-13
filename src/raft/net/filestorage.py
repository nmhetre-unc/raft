"""`Storage` on disk: what `MemoryStorage` simulates, made real.

A `FileStorage` owns one directory holding three files:

``LOCK``
    Empty. Held with an exclusive, non-blocking ``flock`` from open to
    `close()`, so a second `FileStorage` on the same directory -- in
    another process or this one -- raises `StorageError` instead of two
    writers interleaving records. The kernel drops the lock when the
    holder exits, however it exits, so a crash never leaves a stale
    lock behind. The file itself is never deleted: unlinking a lock
    file another opener may already have open races.

``state.json``
    ``{"term": int, "voted_for": str | null}``. Never written in place:
    each save writes ``state.json.tmp``, fsyncs it, ``os.replace``\\ s it
    over ``state.json`` and fsyncs the directory, so a crash leaves
    either the old file or the new one, never a mix. A missing file
    means ``(0, None)``.

``log.bin``
    Entries 1..n (the index-0 sentinel is never written) as append-only
    records ``[u32 length][u32 crc][payload]``, little-endian. The CRC is
    CRC-32 over the length field *and* the payload, so a corrupted length
    is caught, and so an all-zero header -- what a filesystem typically
    leaves after extending a file whose data never landed -- can never
    pass as a valid empty record.

Payloads are opaque. `encode`/`decode` are injected, so this module
never imports protobuf or knows what a command is; the only thing it
reads from a decoded entry is `term`, for `last_log_term`. Every call is
answered from an in-memory copy of the log kept in step with the file,
along with each record's end offset so `truncate_from` is a single
``ftruncate`` at a recorded byte position.

Durability (the ``fsync`` argument):

``"per_write"``
    Each write call is durable when it returns.

``"deferred"``
    Write calls only update the in-memory state and buffer the change;
    *nothing* -- the log or ``state.json`` -- reaches disk until `sync()`
    returns. This is what lets a driver batch: handle a burst of
    messages, `sync()` once, and only then send the replies. Raft's
    persist-before-reply rule becomes the driver's job, discharged by
    calling `sync()` before any message leaves.

Both modes share one path: ``per_write`` is ``deferred`` with a `sync()`
at the end of every write call. `sync()` writes ``state.json`` before
the log, so a crash between the two never leaves log entries on disk
from a term newer than the persisted currentTerm.

Recovery, on open, scans ``log.bin`` to the first record that fails to
parse (short header, short payload, or bad CRC), then searches every
later byte offset for a record that does parse:

- none found: the damage is a torn final write, and the file is cut
  back to the last good record;
- one found: valid data follows the damage, which no crash during an
  append can produce, so it is corruption and `StorageError` is raised
  rather than silently dropping entries that were once durable.

A `sync()` that raises leaves the files in an unknown state (a failed
fsync may already have dropped dirty pages), so the instance refuses
every later write and `sync()`; reopen the directory to carry on from
whatever actually reached disk.
"""

from __future__ import annotations

import fcntl
import json
import os
import struct
import time
import zlib
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Literal

from raft.storage import SENTINEL, LogEntry, Storage, StorageError, bounded_slice, check_index

FsyncMode = Literal["per_write", "deferred"]

STATE_FILE = "state.json"
LOG_FILE = "log.bin"
LOCK_FILE = "LOCK"

_LENGTH = struct.Struct("<I")
_HEADER = struct.Struct("<II")  # length, crc


def _crc(length_field: bytes | memoryview, payload: bytes | memoryview) -> int:
    return zlib.crc32(payload, zlib.crc32(length_field))


def _record(payload: bytes) -> bytes:
    length_field = _LENGTH.pack(len(payload))
    return length_field + _LENGTH.pack(_crc(length_field, payload)) + payload


def _parse_at(data: memoryview, pos: int) -> int | None:
    """End offset of a valid record starting at `pos`, or None."""
    if len(data) - pos < _HEADER.size:
        return None
    length, crc = _HEADER.unpack_from(data, pos)
    end = pos + _HEADER.size + length
    if end > len(data):
        return None
    if _crc(data[pos : pos + _LENGTH.size], data[pos + _HEADER.size : end]) != crc:
        return None
    return end


def _scan(data: memoryview) -> list[int]:
    """End offsets of the valid records in `data`, a torn tail excluded.

    Raises `StorageError` if a valid record follows a damaged one.
    """
    ends: list[int] = []
    pos = 0
    while pos < len(data):
        end = _parse_at(data, pos)
        if end is None:
            for later in range(pos + 1, len(data)):
                if _parse_at(data, later) is not None:
                    raise StorageError(
                        f"{LOG_FILE}: damaged record at byte {pos} is followed by a "
                        f"valid record at byte {later}"
                    )
            break
        ends.append(end)
        pos = end
    return ends


def _write_all(fd: int, data: bytes | bytearray, offset: int) -> None:
    view = memoryview(data)
    while view:
        written = os.pwrite(fd, view, offset)
        view = view[written:]
        offset += written


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _lock(path: Path) -> int:
    """Open `path` and take an exclusive flock on it; return the fd."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise StorageError(f"{path.parent} is already open in another FileStorage") from exc
    return fd


def _load_state(path: Path) -> tuple[int, str | None]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return (0, None)
    try:
        state = json.loads(raw)
    except ValueError as exc:
        raise StorageError(f"{path}: not valid JSON") from exc
    if not isinstance(state, dict):
        raise StorageError(f"{path}: expected a JSON object")
    term = state.get("term")
    voted_for = state.get("voted_for")
    if not isinstance(term, int) or isinstance(term, bool) or term < 0:
        raise StorageError(f"{path}: term must be a non-negative int, got {term!r}")
    if voted_for is not None and not isinstance(voted_for, str):
        raise StorageError(f"{path}: voted_for must be a str or null, got {voted_for!r}")
    return (term, voted_for)


class FileStorage(Storage):
    """A directory-backed `Storage`. See the module docstring for the format."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        encode: Callable[[LogEntry], bytes],
        decode: Callable[[bytes], LogEntry],
        fsync: FsyncMode = "per_write",
    ) -> None:
        if fsync not in ("per_write", "deferred"):
            raise ValueError(f"fsync must be 'per_write' or 'deferred', got {fsync!r}")
        self._dir = Path(directory)
        self._encode = encode
        self._per_write = fsync == "per_write"
        self._broken = False
        # How much full-log copying callers do (`load_log`), for stats.
        self.load_log_calls = 0
        self.load_log_entries = 0
        self.load_log_seconds = 0.0
        # Log fsyncs, and the entries they made durable, for stats: in
        # "deferred" mode one fsync can cover many writes.
        self.log_fsyncs = 0
        self.log_fsync_seconds = 0.0
        self.entries_synced = 0
        self._appended_since_sync = 0

        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock_fd = _lock(self._dir / LOCK_FILE)
        self._fd = -1
        try:
            self._term, self._voted_for = _load_state(self._dir / STATE_FILE)
            self._state_dirty = False
            self._fd = os.open(self._dir / LOG_FILE, os.O_RDWR | os.O_CREAT, 0o644)
            self._log, self._ends = self._recover(decode)
            _fsync_dir(self._dir)  # new files' directory entries
        except BaseException:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
            os.close(self._lock_fd)  # releases the lock
            raise

        # The log file as the in-memory log sees it is
        # disk[:self._keep] + self._pending; only self._disk_len bytes are
        # actually on disk.
        self._disk_len = self._ends[-1]
        self._keep = self._disk_len
        self._pending = bytearray()

    def _recover(self, decode: Callable[[bytes], LogEntry]) -> tuple[list[LogEntry], list[int]]:
        size = os.fstat(self._fd).st_size
        data = os.pread(self._fd, size, 0)
        ends = _scan(memoryview(data))
        valid_len = ends[-1] if ends else 0
        if valid_len < size:  # a torn final record: cut it
            os.ftruncate(self._fd, valid_len)
            os.fsync(self._fd)
        log = [SENTINEL]
        start = 0
        for index, end in enumerate(ends, start=1):
            try:
                log.append(decode(data[start + _HEADER.size : end]))
            except Exception as exc:
                raise StorageError(f"{LOG_FILE}: entry {index} does not decode") from exc
            start = end
        return log, [0, *ends]

    # -- Storage --

    def save_term_and_vote(self, term: int, voted_for: str | None) -> None:
        self._check_usable()
        if not isinstance(term, int) or isinstance(term, bool) or term < 0:
            raise ValueError("term must be a non-negative int")
        if voted_for is not None and not isinstance(voted_for, str):
            raise TypeError("voted_for must be a str or None")
        self._term = term
        self._voted_for = voted_for
        self._state_dirty = True
        self._sync_if_per_write()

    def load_term_and_vote(self) -> tuple[int, str | None]:
        return (self._term, self._voted_for)

    def append_entries(self, entries: list[LogEntry]) -> None:
        self._check_usable()
        # Encode everything before touching any state, so an entry that
        # doesn't encode leaves the storage exactly as it was.
        records = [_record(self._encode(entry)) for entry in entries]
        for entry, record in zip(entries, records, strict=True):
            self._log.append(entry)
            self._ends.append(self._ends[-1] + len(record))
            self._pending += record
        self._appended_since_sync += len(entries)
        self._sync_if_per_write()

    def truncate_from(self, index: int) -> None:
        self._check_usable()
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("index must be a non-negative int")
        if index == 0:
            raise ValueError("cannot truncate the sentinel at index 0")
        if index >= len(self._log):
            return
        offset = self._ends[index - 1]
        del self._log[index:]
        del self._ends[index:]
        if offset >= self._keep:
            del self._pending[offset - self._keep :]
        else:
            self._keep = offset
            self._pending.clear()
        self._sync_if_per_write()

    def load_log(self) -> list[LogEntry]:
        start = time.perf_counter()
        log = list(self._log)
        self.load_log_seconds += time.perf_counter() - start
        self.load_log_calls += 1
        self.load_log_entries += len(log)
        return log

    def entry_at(self, index: int) -> LogEntry:
        check_index(len(self._log), index)
        return self._log[index]

    def term_at(self, index: int) -> int:
        return self.entry_at(index).term

    def entries_from(self, start: int, max_count: int) -> list[LogEntry]:
        return bounded_slice(self._log, start, max_count)

    def last_log_index(self) -> int:
        return len(self._log) - 1

    def last_log_term(self) -> int:
        return self._log[-1].term

    # -- Durability --

    def sync(self) -> None:
        """Make every earlier write call durable.

        If this raises, the instance is unusable: reopen the directory.
        """
        job = self.begin_sync()
        if job is not None:
            job()

    def begin_sync(self) -> Callable[[], None] | None:
        """Take what `sync()` would write, and return a job that writes it
        and fsyncs -- or None if nothing has changed since the last sync.

        This is `sync()` split in two, so the slow half can run on another
        thread while the caller's thread carries on. This half, on the
        caller's thread, snapshots everything the job needs -- the encoded
        term and vote, the log bytes, where they go, any truncation -- and
        updates the bookkeeping as though the job had already succeeded.
        The job touches only the files and that snapshot, never the
        in-memory log, so reads (and even writes) can go on while it runs.

        Jobs must run one at a time, in the order they were begun: each
        writes from where the previous one ended. Run the job before
        `close()`. If the job raises, the instance is unusable, exactly as
        if `sync()` had failed; if it is never run, nothing it covers is
        durable.
        """
        self._check_usable()
        state = None
        if self._state_dirty:
            state = json.dumps({"term": self._term, "voted_for": self._voted_for}).encode()
            self._state_dirty = False
        truncate_to = self._keep if self._keep < self._disk_len else None
        data = None
        if truncate_to is not None or self._pending:
            data = bytes(self._pending)
            offset = self._keep
            appended = self._appended_since_sync
            self._appended_since_sync = 0
            self._disk_len = self._keep = self._ends[-1]
            self._pending.clear()
        if state is None and data is None:
            return None
        fd = self._fd

        def job() -> None:
            try:
                if state is not None:
                    self._write_state(state)
                if data is not None:
                    if truncate_to is not None:
                        os.ftruncate(fd, truncate_to)
                    _write_all(fd, data, offset)
                    start = time.perf_counter()
                    os.fsync(fd)
                    self.log_fsync_seconds += time.perf_counter() - start
                    self.log_fsyncs += 1
                    self.entries_synced += appended
            except BaseException:
                self._broken = True
                raise

        return job

    def close(self) -> None:
        """`sync()`, then release the log file and the lock. Idempotent."""
        if self._fd < 0:
            return
        try:
            if not self._broken:
                self.sync()
        finally:
            os.close(self._fd)
            self._fd = -1
            os.close(self._lock_fd)

    def __enter__(self) -> FileStorage:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- Internals --

    def _write_state(self, data: bytes) -> None:
        tmp = self._dir / (STATE_FILE + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            _write_all(fd, data, 0)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, self._dir / STATE_FILE)
        _fsync_dir(self._dir)

    def _sync_if_per_write(self) -> None:
        if self._per_write:
            self.sync()

    def _check_usable(self) -> None:
        if self._fd < 0:
            raise StorageError("storage is closed")
        if self._broken:
            raise StorageError("an earlier sync() failed; reopen the storage directory")
