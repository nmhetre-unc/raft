"""Tests for the Raft RPC message types.

Immutability and the heartbeat shape are checked directly at runtime.
"A wrong field type is caught by mypy" isn't a runtime property at all --
nothing stops `RequestVote(term="oops", ...)` from constructing happily
at runtime, since dataclasses don't validate field types on their own.
The only way to actually test that claim is to hand mypy some
intentionally wrong code and check that it objects, which is what the
two subprocess-based tests below do. They rely on `src/raft/py.typed`
(added alongside this file) so mypy will type-check *into* the installed
`raft` package from an unrelated file, rather than treating it as an
untyped dependency and silently accepting anything.
"""

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from raft.messages import AppendEntries, AppendEntriesReply, RequestVote, RequestVoteReply
from raft.storage import LogEntry


def test_request_vote_is_immutable() -> None:
    msg = RequestVote(term=1, candidate_id="a", last_log_index=0, last_log_term=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        msg.term = 2  # type: ignore[misc]


def test_request_vote_reply_is_immutable() -> None:
    msg = RequestVoteReply(term=1, vote_granted=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        msg.vote_granted = False  # type: ignore[misc]


def test_append_entries_is_immutable() -> None:
    msg = AppendEntries(
        term=1,
        leader_id="a",
        prev_log_index=0,
        prev_log_term=0,
        entries=(),
        leader_commit=0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        msg.term = 2  # type: ignore[misc]


def test_append_entries_reply_is_immutable() -> None:
    msg = AppendEntriesReply(term=1, success=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        msg.success = False  # type: ignore[misc]


def test_heartbeat_is_append_entries_with_empty_entries() -> None:
    heartbeat = AppendEntries(
        term=3,
        leader_id="leader-1",
        prev_log_index=5,
        prev_log_term=2,
        entries=(),
        leader_commit=5,
    )

    assert isinstance(heartbeat, AppendEntries)
    assert heartbeat.entries == ()


def test_append_entries_also_carries_real_log_entries() -> None:
    entry = LogEntry(term=3, command="set x=1")

    msg = AppendEntries(
        term=3,
        leader_id="leader-1",
        prev_log_index=0,
        prev_log_term=0,
        entries=(entry,),
        leader_commit=0,
    )

    assert msg.entries == (entry,)


def _run_mypy_on(tmp_path: Path, name: str, source: str) -> subprocess.CompletedProcess[str]:
    probe = tmp_path / name
    probe.write_text(source)
    return subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", str(probe)],
        capture_output=True,
        text=True,
        timeout=60,
    )


_BAD_FIELD_TYPES = '''\
from raft.messages import AppendEntries, AppendEntriesReply, RequestVote, RequestVoteReply

RequestVote(term="bad", candidate_id="c", last_log_index=0, last_log_term=0)
RequestVoteReply(term=1, vote_granted="bad")
AppendEntries(
    term=1,
    leader_id="a",
    prev_log_index=0,
    prev_log_term=0,
    entries=[],  # must be a tuple, not a list
    leader_commit=0,
)
AppendEntriesReply(term=1, success="bad")
'''


def test_constructing_with_a_wrong_field_type_is_caught_by_mypy(tmp_path: Path) -> None:
    result = _run_mypy_on(tmp_path, "bad_message_types.py", _BAD_FIELD_TYPES)

    assert result.returncode != 0, result.stdout
    error_lines = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(error_lines) == 4, result.stdout
    for name in ("RequestVote", "RequestVoteReply", "AppendEntries", "AppendEntriesReply"):
        assert name in result.stdout


_GOOD_FIELD_TYPES = '''\
from raft.messages import AppendEntries, AppendEntriesReply, Message, RequestVote, RequestVoteReply
from raft.storage import LogEntry

rv: Message = RequestVote(term=1, candidate_id="c", last_log_index=0, last_log_term=0)
rvr: Message = RequestVoteReply(term=1, vote_granted=True)
ae: Message = AppendEntries(
    term=1,
    leader_id="a",
    prev_log_index=0,
    prev_log_term=0,
    entries=(LogEntry(term=1, command="x"),),
    leader_commit=0,
)
aer: Message = AppendEntriesReply(term=1, success=True)
'''


def test_correctly_typed_messages_pass_mypy_cleanly(tmp_path: Path) -> None:
    # A positive control: proves the failing test above is catching real
    # type errors, not just failing to import `raft.messages` at all.
    result = _run_mypy_on(tmp_path, "good_message_types.py", _GOOD_FIELD_TYPES)

    assert result.returncode == 0, result.stdout
