"""Persistence for Raft's durable state: currentTerm, votedFor, and the log.

`Storage` is the interface every node's durable state goes through.
`MemoryStorage` is the in-memory implementation used by the simulation: it
holds everything in plain Python objects, but -- and this is the whole
point -- a simulated node crash never touches it. A crash wipes a node's
*volatile* state (commitIndex, nextIndex, matchIndex, current role -- all
of which live outside this module, on whatever holds a node together);
the `Storage` object itself is simply handed, unchanged, to whatever comes
back up. That is what makes crash tests exercise something real instead
of being decorative: state that was actually persisted before the crash
is still there after it, and nothing else survives.

Indexing convention: the log is 1-indexed, matching the Raft paper
(Figure 2), which uses prevLogIndex, nextIndex, matchIndex, and
lastLogIndex throughout as 1-based quantities. Index 0 is reserved for a
fixed sentinel entry (`term=0`, no command) rather than being left
unused, so that `self._log[i]` -- the underlying Python list position --
*is* the entry at Raft index `i`, for every `i` including 0. There is no
separate offset to track or get wrong in a consistency-check expression.
`load_log()` returns the log including that sentinel; the first real
entry any caller appends lands at index 1. The sentinel can never be
truncated away: `truncate_from(0)` raises rather than emptying the log
back past its anchor.

`MemoryStorage` never inspects a log entry's command. Entries are
`LogEntry(term, command)` pairs; Storage legitimately reads `term` (Raft's
consistency check is built on it), but `command` is an opaque payload it
only ever stores and returns -- never compared, hashed, or inspected.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol


class LogEntry(NamedTuple):
    """A single Raft log entry: a term and an opaque command.

    Storage reads `term` -- it has to, to answer `last_log_term()` -- but
    never inspects, compares, or hashes `command`.
    """

    term: int
    command: Any = None


SENTINEL: LogEntry = LogEntry(term=0, command=None)
"""The fixed entry occupying log index 0: term 0, no command."""


class StorageError(Exception):
    """Raised by a write call that `MemoryStorage.fail_after` has armed."""


class Storage(Protocol):
    """The durable-state interface a Raft node writes through.

    Raft requires currentTerm and votedFor to be persisted *before* the
    node replies to any RPC that changed them. Nothing in this interface
    enforces that ordering -- it can't, from in here -- but `fail_after`
    on `MemoryStorage` exists precisely so a test can arm a failure on
    exactly the write that matters and confirm the reply never happened.

    The log is 1-indexed with a sentinel at index 0; see the module
    docstring for the full convention.
    """

    def save_term_and_vote(self, term: int, voted_for: str | None) -> None:
        """Durably record currentTerm and votedFor."""
        ...

    def load_term_and_vote(self) -> tuple[int, str | None]:
        """Return the last persisted (currentTerm, votedFor), or (0, None)."""
        ...

    def append_entries(self, entries: list[LogEntry]) -> None:
        """Durably append `entries`, in order, after the current last entry.

        The first entry ever appended lands at index 1 -- index 0 is the
        sentinel and is never itself a target of appending.
        """
        ...

    def truncate_from(self, index: int) -> None:
        """Durably discard every log entry at or after `index`.

        A no-op, not an error, if `index` is at or past the end of the
        log. Raises if `index` is 0: the sentinel anchors the empty-log
        state and must never be removed.
        """
        ...

    def load_log(self) -> list[LogEntry]:
        """Return the full persisted log, including the index-0 sentinel."""
        ...

    def last_log_index(self) -> int:
        """Return the index of the last log entry, or 0 if the log is empty."""
        ...

    def last_log_term(self) -> int:
        """Return the term of the last log entry, or 0 if the log is empty."""
        ...


class MemoryStorage(Storage):
    """An in-memory `Storage` that survives a simulated crash by design.

    Nothing in this class's public API resets its state -- there is no
    `crash()` method here. A "crash" in the simulation means the *rest*
    of a node (its role, its commit/next/match indexes) is thrown away
    and rebuilt from scratch, while the very same `MemoryStorage`
    instance is handed to the fresh node, exactly as a restarted process
    would keep its handle to the same disk.

    `fail_after(n)` arms a one-shot failure on the n-th write call from
    then on (reads never count); that write raises `StorageError` and the
    storage disarms itself. Every write method checks the failure budget
    *before* touching any state, so a triggered failure changes nothing:
    it never leaves a partial mutation behind, and every earlier,
    successful write remains exactly as it was.
    """

    def __init__(self) -> None:
        self._term: int = 0
        self._voted_for: str | None = None
        self._log: list[LogEntry] = [SENTINEL]
        self._fail_after: int | None = None
        self._write_count: int = 0

    def fail_after(self, n_writes: int) -> None:
        """Arm a one-shot `StorageError` on the n_writes-th write from now.

        Counting restarts at zero every time this is called, so only
        writes made *after* this call count toward `n_writes`.
        """
        if not isinstance(n_writes, int) or isinstance(n_writes, bool) or n_writes < 1:
            raise ValueError("n_writes must be a positive int")
        self._fail_after = n_writes
        self._write_count = 0

    def save_term_and_vote(self, term: int, voted_for: str | None) -> None:
        self._consume_write_budget()
        if not isinstance(term, int) or isinstance(term, bool) or term < 0:
            raise ValueError("term must be a non-negative int")
        if voted_for is not None and not isinstance(voted_for, str):
            raise TypeError("voted_for must be a str or None")
        self._term = term
        self._voted_for = voted_for

    def load_term_and_vote(self) -> tuple[int, str | None]:
        return (self._term, self._voted_for)

    def append_entries(self, entries: list[LogEntry]) -> None:
        self._consume_write_budget()
        self._log.extend(entries)

    def truncate_from(self, index: int) -> None:
        self._consume_write_budget()
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("index must be a non-negative int")
        if index == 0:
            raise ValueError("cannot truncate the sentinel at index 0")
        del self._log[index:]  # no-op if index is already past the end

    def load_log(self) -> list[LogEntry]:
        return list(self._log)

    def last_log_index(self) -> int:
        return len(self._log) - 1

    def last_log_term(self) -> int:
        return self._log[-1].term

    def _consume_write_budget(self) -> None:
        """Count one write attempt; raise if `fail_after` says this is the one.

        Called before any state is touched, so a raise here always leaves
        the storage exactly as it was before the call.
        """
        if self._fail_after is None:
            return
        self._write_count += 1
        if self._write_count == self._fail_after:
            self._fail_after = None  # one-shot: disarm once it has fired
            raise StorageError(f"simulated storage failure on write #{self._write_count}")
