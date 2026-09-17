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

`apply()` is the low-level entry point for the replicated path: `RaftNode`
calls it once per committed `LogEntry.command` that's a bare `Command`,
in log order, as `last_applied` advances (see `raft.node`'s module
docstring). `get` is deliberately NOT something `RaftNode` ever puts
through the log at this milestone -- see `read()` below for why, and for
the limitation that comes with answering it the way this file does
instead.

`ClientRequest` wraps a `Command` with `client_id`/`serial_number` and is
the shape a *client's* request actually takes through the log --
`apply_client_request()` is what `RaftNode` calls for one of these,
instead of `apply()` directly. The dedup this buys (see
`apply_client_request`'s own docstring for the exact rule) is tracked in
`_sessions`, a dict living on `KVStateMachine` itself, right alongside
`_data` -- not anywhere leader-specific. That placement is the entire
point: a session recorded here is reconstructed by *every* node that
applies the same log, survives a leader crash exactly as well as any
other applied state does, and never has to be explicitly handed off
between leaders the way something tracked only in one leader's own
memory would.

KNOWN LIMITATION, not an oversight: `_sessions` grows by one entry per
distinct `client_id` ever seen and is never pruned. A real
implementation bounds this by folding old sessions into a periodic
snapshot and discarding the log (and the sessions map) before it --
Raft's own log-compaction mechanism -- but that milestone doesn't exist
in this project yet and may never be built at all, per this project's
own stated priorities. Safely evicting a session before that exists
would need real client-lifecycle machinery this project has no model of
at all (an explicit open/close, or a lease tied to some liveness signal)
-- evicting a session a client might still legitimately retry against
would silently reopen the exact double-apply hazard this mechanism
exists to prevent, which would be worse than the unbounded growth it
was meant to fix. Until compaction exists, this is accepted as a
permanent characteristic of an in-memory, non-snapshotting
implementation, not a defect to patch around piecemeal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


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


@dataclass(frozen=True)
class ClientRequest:
    """A client's request to apply `command` at most once.

    `client_id` and `serial_number` follow the Raft paper's own client-
    interaction scheme: a client picks a `client_id` unique to itself
    (once, e.g. when it first connects) and a `serial_number` it
    increases by exactly one for every new *logical* request it ever
    issues -- never for a retry of one already sent. A retry of the same
    logical request (because the client never got a reply, or its
    leader crashed before replying, or the reply itself was lost) resends
    the identical `client_id`, `serial_number`, and `command` -- that
    triple is what `apply_client_request` uses to recognize "I've already
    done this" and answer without doing it again.
    """

    client_id: str
    serial_number: int
    command: Command


class _ClientSession(NamedTuple):
    """The last request `KVStateMachine` actually applied for one client."""

    serial_number: int
    result: object


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
        self._sessions: dict[str, _ClientSession] = {}

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

    def apply_client_request(self, request: ClientRequest) -> object:
        """Apply `request.command` at most once for `request.client_id`.

        The rule: if `request.serial_number` is less than or equal to the
        highest serial number already recorded for this client, `command`
        is NOT applied again -- the result already recorded for that
        client's session is returned instead, unconditionally. There is
        no notion here of a client legitimately reusing an
        already-superseded serial number for a genuinely new request, so
        "the highest one ever recorded" is the only comparison that
        matters; a `<=` comparison (not just `==`) also correctly answers
        a very stale retry the same way it answers the most recent one.

        This is the entire dedup mechanism, and it lives here -- on the
        state machine every node reconstructs identically by applying the
        same log -- specifically so it is safe across a leader change:
        whichever node applies this entry, whenever it applies it,
        computes the identical answer, because the check is against
        state produced by applying the log, never against anything only
        one leader ever held in memory.
        """
        session = self._sessions.get(request.client_id)
        if session is not None and request.serial_number <= session.serial_number:
            return session.result

        result = self.apply(request.command)
        self._sessions[request.client_id] = _ClientSession(request.serial_number, result)
        return result

    def last_applied_serial(self, client_id: str) -> int | None:
        """The highest serial_number recorded as applied for `client_id`,
        or `None` if this client has no recorded session yet."""
        session = self._sessions.get(client_id)
        return session.serial_number if session is not None else None

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

    def sessions_snapshot(self) -> dict[str, tuple[int, object]]:
        """A shallow copy of every client's recorded session, for
        inspection/tests -- same copy-not-live-view reasoning as
        `snapshot()`. `(serial_number, result)` per client_id.
        """
        return {
            client_id: (session.serial_number, session.result)
            for client_id, session in self._sessions.items()
        }
