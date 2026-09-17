"""A minimal, deterministic in-memory key/value state machine.

Raft's own guarantee -- every node applies the same sequence of
committed entries, in the same order -- is worthless without a state
machine that itself behaves deterministically once handed that
sequence: two `KVStateMachine`s fed the identical `apply()` calls, in
the identical order, always end up holding identical state. There is no
hidden nondeterminism here (no dict-ordering dependency on insertion
order across separately-constructed dicts, no wall-clock, no randomness)
for the same reason `raft.storage.LogEntry` never touches any of those
either.

`Command` is a closed union of the three operations this milestone
supports -- `Put`, `Get`, `Delete` -- each a frozen dataclass, matching
`raft.messages`' own style for exactly the same reason: a real field to
check against, not a dict key that silently becomes `None` on a typo.

`apply()` is the only thing this class needs for the replicated path:
`RaftNode` calls it once per committed `LogEntry.command`, in log order,
as `last_applied` advances (see `raft.node`'s module docstring). `get`
is deliberately NOT something `RaftNode` ever puts through the log at
this milestone -- see `read()` below for why, and for the limitation
that comes with answering it the way this file does instead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Put:
    """Store `value` under `key`, overwriting whatever was there."""

    key: str
    value: object


@dataclass(frozen=True)
class Get:
    """Read the value under `key`. Supported by `apply()` for
    completeness and symmetry with `Put`/`Delete` -- never mutates state
    either way -- but this milestone's `RaftNode` never constructs a
    `Get` as a log entry; see `KVStateMachine.read` for the read path it
    actually uses instead.
    """

    key: str


@dataclass(frozen=True)
class Delete:
    """Remove `key`, if present. A no-op, not an error, if it wasn't."""

    key: str


Command = Put | Get | Delete


class _NotFound:
    """Sentinel distinguishing "no such key" from a stored value of
    `None`, which is a perfectly legitimate thing to `Put`. A dedicated
    class (rather than reusing an arbitrary object()) so `repr()` reads
    clearly in test failures and log output.
    """

    def __repr__(self) -> str:
        return "NOT_FOUND"


NOT_FOUND = _NotFound()
"""Returned by `apply(Get(...))`/`apply(Delete(...))`/`read(...)` for a missing key."""


class KVStateMachine:
    """An in-memory, dict-backed key/value store.

    Never touches a clock, storage, or anything about Raft -- this class
    knows nothing about logs, terms, or commitment. It only ever sees
    whatever `command` `RaftNode` hands it, already decided to be safe to
    apply.
    """

    def __init__(self) -> None:
        self._data: dict[str, object] = {}

    def apply(self, command: Command) -> object:
        """Apply one command and return its result.

        `Put` returns `None`. `Get`/`Delete` return the key's prior value,
        or `NOT_FOUND` if it wasn't present -- `Delete` reports what it
        removed for the same reason a real KV store's DEL usually does:
        the caller often wants to know whether anything was actually
        there.
        """
        if isinstance(command, Put):
            self._data[command.key] = command.value
            return None
        if isinstance(command, Get):
            return self._data.get(command.key, NOT_FOUND)
        if isinstance(command, Delete):
            return self._data.pop(command.key, NOT_FOUND)
        raise TypeError(f"unknown command: {command!r}")

    def read(self, key: str) -> object:
        """A direct, local, unreplicated read of this node's own current state.

        KNOWN LIMITATION, not an oversight: this is a plain dict lookup
        against whatever this node has applied so far -- it does not go
        through the log or commitment at all, and it is therefore NOT
        linearizable. A client reading from a leader that has just been
        partitioned away (and doesn't know it yet) can get stale data
        from a soon-to-be-deposed leader, and a client reading from a
        lagging follower can see state older than what's already
        committed elsewhere. Making reads linearizable without paying for
        a full log round trip on every read needs real, separate
        machinery (e.g. a leader lease or Raft's own read-index protocol)
        that is out of scope for this milestone -- see `raft.node`'s
        module docstring.
        """
        return self._data.get(key, NOT_FOUND)

    def snapshot(self) -> dict[str, object]:
        """A shallow copy of all current state, for inspection/tests.

        A copy, not a live view: callers (including this project's own
        tests comparing two nodes' final state) must never be able to
        mutate a `KVStateMachine`'s internals just by holding a reference
        to what this returns.
        """
        return dict(self._data)
