import pytest

from raft.storage import SENTINEL, LogEntry, MemoryStorage, StorageError


def _entries(*commands: str, term: int = 1) -> list[LogEntry]:
    return [LogEntry(term=term, command=c) for c in commands]


def _log(*commands: str, term: int = 1) -> list[LogEntry]:
    """The expected `load_log()` result for a fresh log with these commands."""
    return [SENTINEL, *_entries(*commands, term=term)]


def test_term_and_vote_survive_a_simulated_crash() -> None:
    storage = MemoryStorage()

    # Before the crash: normal operation.
    storage.save_term_and_vote(5, "node-2")
    storage.append_entries(_entries("a", "b", "c"))
    volatile_role = "leader"
    volatile_commit_index = 3

    # The crash: everything volatile is thrown away and rebuilt from
    # scratch by whatever restarts the node -- Storage is not part of
    # that, it is handed, as-is, to the fresh node.
    volatile_role = "follower"  # a restarted node always starts as follower
    volatile_commit_index = 0  # and with no committed entries known yet

    # After the crash: the durable state is exactly what was last
    # persisted, unaffected by the crash that just wiped everything else.
    assert storage.load_term_and_vote() == (5, "node-2")
    assert storage.load_log() == _log("a", "b", "c")
    assert volatile_role == "follower"
    assert volatile_commit_index == 0


def test_first_appended_entry_lands_at_index_1() -> None:
    storage = MemoryStorage()

    storage.append_entries(_entries("e1"))

    assert storage.last_log_index() == 1
    assert storage.load_log()[1] == LogEntry(term=1, command="e1")


def test_truncate_from_removes_exactly_the_right_entries() -> None:
    storage = MemoryStorage()
    storage.append_entries(_entries("e1", "e2", "e3", "e4", "e5"))

    storage.truncate_from(3)  # discard index 3 (e3) onward

    assert storage.load_log() == _log("e1", "e2")


def test_truncate_from_past_the_end_is_a_noop() -> None:
    storage = MemoryStorage()
    storage.append_entries(_entries("e1", "e2"))

    storage.truncate_from(10)  # must not raise

    assert storage.load_log() == _log("e1", "e2")


def test_truncate_from_0_refuses_to_remove_the_sentinel() -> None:
    storage = MemoryStorage()
    storage.append_entries(_entries("e1", "e2"))

    with pytest.raises(ValueError, match="sentinel"):
        storage.truncate_from(0)

    assert storage.load_log() == _log("e1", "e2")  # untouched


def test_last_log_index_on_empty_log_is_0() -> None:
    storage = MemoryStorage()

    assert storage.last_log_index() == 0


def test_last_log_term_on_empty_log_is_0() -> None:
    storage = MemoryStorage()

    assert storage.last_log_term() == 0


def test_last_log_index_and_term_reflect_the_most_recent_entry() -> None:
    storage = MemoryStorage()
    storage.append_entries(_entries("e1", term=3))
    storage.append_entries(_entries("e2", term=7))

    assert storage.last_log_index() == 2
    assert storage.last_log_term() == 7


def test_fail_after_raises_on_the_nth_write_and_not_earlier() -> None:
    storage = MemoryStorage()
    storage.fail_after(3)

    storage.save_term_and_vote(1, None)  # write 1: succeeds
    storage.append_entries(_entries("e1"))  # write 2: succeeds
    with pytest.raises(StorageError):
        storage.truncate_from(1)  # write 3: raises


def test_state_written_before_a_failed_write_is_still_readable() -> None:
    storage = MemoryStorage()
    storage.save_term_and_vote(2, "node-1")
    storage.append_entries(_entries("e1", "e2"))

    storage.fail_after(1)
    with pytest.raises(StorageError):
        storage.append_entries(_entries("e3"))

    assert storage.load_term_and_vote() == (2, "node-1")
    assert storage.load_log() == _log("e1", "e2")


class _UninspectableEntry:
    """A log entry command that blows up if Storage ever looks inside it."""

    def __eq__(self, other: object) -> bool:
        raise AssertionError("Storage must never compare log entry contents")

    def __hash__(self) -> int:
        raise AssertionError("Storage must never hash log entry contents")


def test_storage_never_inspects_log_entry_contents() -> None:
    storage = MemoryStorage()
    payloads = [_UninspectableEntry(), _UninspectableEntry()]
    entries = [LogEntry(term=1, command=p) for p in payloads]

    storage.append_entries(entries)
    storage.truncate_from(5)  # past the end: still must not touch the entries
    loaded = storage.load_log()

    assert loaded[0] is SENTINEL
    assert loaded[1].command is payloads[0]
    assert loaded[2].command is payloads[1]


def test_fail_after_is_one_shot_and_then_disarms() -> None:
    storage = MemoryStorage()
    storage.fail_after(1)

    with pytest.raises(StorageError):
        storage.save_term_and_vote(1, "a")

    storage.save_term_and_vote(1, "a")  # succeeds now; the failure already fired

    assert storage.load_term_and_vote() == (1, "a")


def test_fail_after_recounts_from_the_call_that_arms_it() -> None:
    storage = MemoryStorage()
    storage.save_term_and_vote(0, None)  # a write before arming, uncounted
    storage.fail_after(1)

    with pytest.raises(StorageError):
        storage.append_entries(_entries("e1"))  # the 1st write since arming


def test_reads_never_count_toward_fail_after() -> None:
    storage = MemoryStorage()
    storage.fail_after(1)

    for _ in range(10):
        storage.load_term_and_vote()
        storage.load_log()
        storage.last_log_index()
        storage.last_log_term()

    with pytest.raises(StorageError):
        storage.save_term_and_vote(1, None)  # the 1st write; still fires here


def test_load_log_returns_a_snapshot_not_a_live_view() -> None:
    storage = MemoryStorage()
    storage.append_entries(_entries("e1"))

    snapshot = storage.load_log()
    snapshot.append(LogEntry(term=1, command="mutated-externally"))

    assert storage.load_log() == _log("e1")


def test_save_term_and_vote_rejects_negative_term() -> None:
    storage = MemoryStorage()
    with pytest.raises(ValueError, match="term"):
        storage.save_term_and_vote(-1, None)


def test_truncate_from_rejects_negative_index() -> None:
    storage = MemoryStorage()
    with pytest.raises(ValueError, match="index"):
        storage.truncate_from(-1)


def test_fail_after_rejects_non_positive_n_writes() -> None:
    storage = MemoryStorage()
    with pytest.raises(ValueError, match="positive"):
        storage.fail_after(0)


def test_save_term_and_vote_rejects_non_str_voted_for() -> None:
    storage = MemoryStorage()
    with pytest.raises(TypeError, match="voted_for"):
        storage.save_term_and_vote(1, 42)  # type: ignore[arg-type]
