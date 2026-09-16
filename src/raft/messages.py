"""The RPC types Raft leader election runs on.

Every message is a frozen dataclass, not a dict: a dict payload would let
a typo in a key silently produce `None` at runtime, while these give
mypy a real field to check `msg.term` or `msg.vote_granted` against, and
`frozen=True` makes every message immutable once constructed, matching
how an RPC that already went out on the wire can't be mutated after the
fact. Field names match the Raft paper's Figure 2 exactly, so there is no
translation to keep straight while implementing the node against it.

`AppendEntries` is included now even though log replication comes later:
a heartbeat *is* an `AppendEntries` with `entries=()`, and leader election
depends on heartbeats to suppress followers' election timers. `entries`
is a `tuple[LogEntry, ...]`, not a `list`, so the message stays hashable
and immutable all the way through -- a `list` field would make `==`
still work but `frozen=True` a half-truth, since the list itself would
stay mutable in place.
"""

from __future__ import annotations

from dataclasses import dataclass

from raft.storage import LogEntry


@dataclass(frozen=True)
class RequestVote:
    """Sent by a candidate to gather votes (Figure 2, RequestVote RPC)."""

    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int


@dataclass(frozen=True)
class RequestVoteReply:
    """A voter's response to `RequestVote`."""

    term: int
    vote_granted: bool


@dataclass(frozen=True)
class AppendEntries:
    """Sent by the leader to replicate entries and as a heartbeat.

    A heartbeat is exactly this with `entries=()`.
    """

    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: tuple[LogEntry, ...]
    leader_commit: int


@dataclass(frozen=True)
class AppendEntriesReply:
    """A follower's response to `AppendEntries`.

    `match_index` is the index this reply's follower has actually
    verified it holds -- `prev_log_index + len(entries)` from the
    specific `AppendEntries` being answered, computed by the follower
    itself, which has full knowledge of exactly what it just accepted.
    Default 0 on a failure reply, where it's meaningless: nothing was
    accepted, so there's nothing to report.

    This field exists specifically so the leader never has to *guess*
    which of several possibly-overlapping in-flight requests a given
    reply answers (see `RaftNode`'s module docstring for the bug this
    replaced: a leader that instead tracked "what I most recently sent"
    per peer could misattribute an old reply against a newer, larger
    send and believe more was replicated than actually was).
    """

    term: int
    success: bool
    match_index: int = 0


Message = RequestVote | RequestVoteReply | AppendEntries | AppendEntriesReply
