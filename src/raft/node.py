"""The Raft node as a pure state machine.

`RaftNode` implements leader election plus log replication (Figure 2's
`AppendEntries` receiver implementation, in full) -- but no commit
tracking. A follower now actually applies what it's sent: it runs the
consistency check against `prev_log_index`/`prev_log_term`, deletes a
conflicting entry and everything after it, and appends whatever's new.
There is still no `commitIndex` anywhere in this file, and nothing here
ever decides an entry is safe to hand to a state machine -- that's
Milestone 4.

A leader tracks two things per peer, and they are not interchangeable:

- `next_index[peer]` is a *guess* -- the next log index this leader
  believes it should try sending that peer. It starts optimistic (one
  past this leader's own last entry) and walks backward only when a
  peer rejects an AppendEntries, which isn't implemented yet either.
- `match_index[peer]` is *known truth* -- the highest index this leader
  has actual confirmation the peer has replicated. It starts at 0
  (known nothing) and only ever moves forward, once a real
  acknowledgement arrives.

Neither is derived from the other, and nothing in this file computes one
from the other -- an optimistic guess and a confirmed fact answer
different questions, and conflating them is exactly how a leader would
end up believing a follower has entries it never actually got.

`RaftNode` never sends anything and never reads a clock. `tick(now)` and
`handle(msg, src, now)` both return `(destination, Message)` pairs for
whatever's driving the simulation -- a `Cluster`, a test, or eventually a
real transport -- to actually deliver. `now` always arrives as a plain
integer from the caller, exactly like `Clock` and `Network` elsewhere in
this project; the node has no other notion of time.

Persistence: `currentTerm` and `votedFor` are loaded from `Storage` once,
at construction, and written back through `Storage.save_term_and_vote`
before either method returns -- but only when this call actually changed
one of them, so a stale or rejected message that leaves them untouched
never produces a wasted write. That write happens before the return
value is ever constructed, so if `Storage` is armed with `fail_after` and
raises, the caller sees the exception and no reply, never a reply that
was never actually made durable.

Election timeouts are the one place randomness enters this node, and the
constructor takes an injected `random.Random` for exactly that reason:
seed it from `Cluster`'s own seed and the whole election sequence
reproduces. The timeout is redrawn from `election_timeout_range` on
*every* reset -- a node's first tick, granting a vote, accepting a
heartbeat, and starting a new election all redraw independently. Drawing
it once and reusing that fixed value everywhere would just replace one
kind of livelock (identical timeouts) with another (a fixed-but-different
timeout per node still produces the same repeating race in an adversarial
split-vote scenario); only redrawing every time actually avoids that.
The first draw happens on the first `tick()` call rather than in the
constructor, because a node doesn't know what "now" actually is until
then -- anchoring it to a constructor-time guess of 0 would be fine for a
node's initial boot but wrong for one rebuilt by `Cluster.restart()` well
into a simulation, which would otherwise inherit an already-expired
deadline and fire a disruptive election on its very first tick.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from enum import Enum

from raft.messages import (
    AppendEntries,
    AppendEntriesReply,
    Message,
    RequestVote,
    RequestVoteReply,
)
from raft.storage import LogEntry, Storage


class Role(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class RaftNode:
    """A single Raft node's election-only state machine.

    Persistent (loaded from `storage` at construction, saved back to it
    before any reply that changed them is returned): `current_term`,
    `voted_for`.

    Volatile (never touches `storage`, reset to nothing meaningful by a
    fresh construction -- i.e. by a simulated crash and restart):
    `role`, `leader_id`, `election_deadline`, `votes_received`, and --
    only meaningful while `role is Role.LEADER`, reinitialized fresh on
    every election win -- `next_index`, `match_index`.
    """

    def __init__(
        self,
        node_id: str,
        peers: list[str],
        storage: Storage,
        election_timeout_range: tuple[int, int] = (150, 300),
        heartbeat_interval: int = 50,
        rng: random.Random | None = None,
    ) -> None:
        lo, hi = election_timeout_range
        if lo <= 0 or hi < lo:
            raise ValueError("election_timeout_range must be a positive, non-empty range")
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")

        self.node_id = node_id
        self.peers = list(peers)
        self.storage = storage
        self.election_timeout_range = election_timeout_range
        self.heartbeat_interval = heartbeat_interval
        self._rng = rng if rng is not None else random.Random()

        # Persistent state -- loaded now, written back by _persist_if_dirty().
        self.current_term, self.voted_for = storage.load_term_and_vote()
        self._dirty = False

        # Volatile state -- gone and rebuilt on every crash/restart.
        self.role = Role.FOLLOWER
        self.leader_id: str | None = None
        self.votes_received: set[str] = set()
        self.election_deadline: int = 0
        self._next_heartbeat: int = 0
        # Leader-only, meaningless until the first election win; see the
        # module docstring for why these are never derived from each other.
        self.next_index: dict[str, int] = {}
        self.match_index: dict[str, int] = {}
        # The deadline's first real draw is deferred to the first tick()
        # (see there for why): at construction time this node has no idea
        # what "now" actually is, and a restart can happen arbitrarily far
        # into a simulation, not just at t=0.
        self._deadline_initialized = False

    def tick(self, now: int) -> list[tuple[str, Message]]:
        """Called periodically by whatever drives this node's clock."""
        if not self._deadline_initialized:
            # Drawing this at construction would anchor it to t=0, which is
            # only correct for a node's very first boot. A node rebuilt by
            # `Cluster.restart()` mid-simulation would otherwise inherit an
            # already-long-expired deadline and fire an election on its
            # very first tick, potentially disrupting a perfectly healthy
            # leader. Anchoring to this node's actual first `now` instead
            # makes construction and restart behave identically.
            self._reset_election_deadline(now)
            self._deadline_initialized = True

        out: list[tuple[str, Message]] = []

        if self.role in (Role.FOLLOWER, Role.CANDIDATE) and now >= self.election_deadline:
            out.extend(self._start_election(now))

        if self.role is Role.LEADER and now >= self._next_heartbeat:
            out.extend(self._send_append_entries(now))

        self._persist_if_dirty()
        return out

    def handle(self, msg: Message, src: str, now: int) -> list[tuple[str, Message]]:
        """Called with one incoming message, addressed to this node, from `src`."""
        self._maybe_adopt_newer_term(msg.term)

        if isinstance(msg, RequestVote):
            out = self._handle_request_vote(msg, src, now)
        elif isinstance(msg, RequestVoteReply):
            out = self._handle_request_vote_reply(msg, src, now)
        elif isinstance(msg, AppendEntries):
            out = self._handle_append_entries(msg, src, now)
        elif isinstance(msg, AppendEntriesReply):
            out = self._handle_append_entries_reply(msg, src, now)
        else:  # pragma: no cover - exhaustive over the Message union
            raise TypeError(f"unknown message type: {type(msg)!r}")

        self._persist_if_dirty()
        return out

    # -- Figure 2, rule 1: applied first, before any message-specific logic --

    def _maybe_adopt_newer_term(self, term: int) -> None:
        if term > self.current_term:
            self.current_term = term
            self.voted_for = None
            self.role = Role.FOLLOWER
            self._dirty = True

    # -- RequestVote --

    def _handle_request_vote(
        self, msg: RequestVote, src: str, now: int
    ) -> list[tuple[str, Message]]:
        if msg.term < self.current_term:
            return [(src, RequestVoteReply(term=self.current_term, vote_granted=False))]

        already_voted_elsewhere = self.voted_for is not None and self.voted_for != msg.candidate_id
        grant = not already_voted_elsewhere and self._challenger_log_is_current(msg)

        if grant:
            self.voted_for = msg.candidate_id
            self._dirty = True
            self._reset_election_deadline(now)

        return [(src, RequestVoteReply(term=self.current_term, vote_granted=grant))]

    def _challenger_log_is_current(self, msg: RequestVote) -> bool:
        """Figure 2's up-to-date check: compare last_log_term, then last_log_index."""
        our_term = self.storage.last_log_term()
        if msg.last_log_term != our_term:
            return msg.last_log_term > our_term
        return msg.last_log_index >= self.storage.last_log_index()

    def _handle_request_vote_reply(
        self, msg: RequestVoteReply, src: str, now: int
    ) -> list[tuple[str, Message]]:
        if msg.term < self.current_term:
            return []  # stale reply, from a term we've since moved past
        if self.role is not Role.CANDIDATE or msg.term != self.current_term:
            return []  # not campaigning, or not for our current election
        if not msg.vote_granted:
            return []

        self.votes_received.add(src)
        if not self._has_majority():
            return []
        return self._become_leader(now)

    # -- AppendEntries (Figure 2's receiver implementation, minus commit tracking) --

    def _handle_append_entries(
        self, msg: AppendEntries, src: str, now: int
    ) -> list[tuple[str, Message]]:
        # 1. Reply false if term < currentTerm.
        if msg.term < self.current_term:
            return [(src, AppendEntriesReply(term=self.current_term, success=False))]

        self.role = Role.FOLLOWER
        self.leader_id = msg.leader_id
        self._reset_election_deadline(now)

        log = self.storage.load_log()

        # 2. Reply false if the log has no entry at prev_log_index, or has
        # one whose term differs from prev_log_term -- the consistency
        # check. Index 0 is the sentinel: every log has one, its term is
        # always 0 everywhere, and any correctly-formed message naming
        # prev_log_index=0 carries prev_log_term=0 too, so this passes
        # unconditionally there without needing a special case for it.
        if msg.prev_log_index >= len(log) or log[msg.prev_log_index].term != msg.prev_log_term:
            return [(src, AppendEntriesReply(term=self.current_term, success=False))]

        # 3 & 4. Find the first sent entry (if any) this follower doesn't
        # already hold at that index and term -- Raft's Log Matching
        # Property guarantees that wherever the term already matches, the
        # command does too, so comparing term alone is exactly "already
        # have this entry", not an approximation of it. Everything up to
        # that point is left completely untouched: storage is only ever
        # written to from the first genuine divergence onward, never
        # truncated and rewritten over entries that already match. That
        # matters because messages get duplicated in this system, and a
        # resent, already-matching AppendEntries must be a true no-op --
        # truncating and reappending would look identical afterward while
        # having briefly discarded entries the leader already believes
        # replicated (match_index says so), which is exactly the kind of
        # gap a real crash could land in.
        first_new = 0
        while first_new < len(msg.entries):
            index = msg.prev_log_index + 1 + first_new
            if index >= len(log) or log[index].term != msg.entries[first_new].term:
                break
            first_new += 1

        new_entries = msg.entries[first_new:]
        if new_entries:
            conflict_index = msg.prev_log_index + 1 + first_new
            if conflict_index < len(log):
                self.storage.truncate_from(conflict_index)  # 3: delete the conflict onward
            self.storage.append_entries(list(new_entries))  # 4: append what's new
        # 5. Commit tracking: skipped -- Milestone 4.

        return [(src, AppendEntriesReply(term=self.current_term, success=True))]

    def _handle_append_entries_reply(
        self, msg: AppendEntriesReply, src: str, now: int
    ) -> list[tuple[str, Message]]:
        del msg, src, now  # nothing to do without replication tracking
        return []

    # -- Elections and heartbeats --

    def _start_election(self, now: int) -> list[tuple[str, Message]]:
        self.current_term += 1
        self.role = Role.CANDIDATE
        self.voted_for = self.node_id
        self.votes_received = {self.node_id}
        self._dirty = True
        self._reset_election_deadline(now)

        if self._has_majority():  # a lone node in a single-node cluster
            return self._become_leader(now)

        request = RequestVote(
            term=self.current_term,
            candidate_id=self.node_id,
            last_log_index=self.storage.last_log_index(),
            last_log_term=self.storage.last_log_term(),
        )
        return [(peer, request) for peer in self.peers]

    def _become_leader(self, now: int) -> list[tuple[str, Message]]:
        self.role = Role.LEADER
        self.leader_id = self.node_id

        # Fresh per-peer state for this leadership stint -- never carried
        # over from a previous one. next_index starts optimistic (this
        # leader's own log, one past the end); match_index starts at the
        # only thing actually known about a peer at this point: nothing.
        last_index = self.storage.last_log_index()
        self.next_index = dict.fromkeys(self.peers, last_index + 1)
        self.match_index = dict.fromkeys(self.peers, 0)

        return self._send_append_entries(now)  # sent immediately, per Figure 2

    def _send_append_entries(self, now: int) -> list[tuple[str, Message]]:
        """Send every peer an AppendEntries built from *its own* next_index.

        Each peer gets a distinct message -- not one shared object -- since
        `prev_log_index`/`prev_log_term` anchor the log-matching check at
        wherever *that* peer is believed to be, and `entries` is whatever
        this leader has from there onward. A peer already caught up (its
        next_index one past this leader's last entry) simply gets `entries
        = ()`: a heartbeat is nothing more than that case.
        """
        log = self.storage.load_log()
        out: list[tuple[str, Message]] = []
        for peer in self.peers:
            prev_log_index = self.next_index[peer] - 1
            message = AppendEntries(
                term=self.current_term,
                leader_id=self.node_id,
                prev_log_index=prev_log_index,
                prev_log_term=log[prev_log_index].term,
                entries=tuple(log[self.next_index[peer] :]),
                leader_commit=0,  # no commit tracking without replication
            )
            out.append((peer, message))

        self._next_heartbeat = now + self.heartbeat_interval
        return out

    def append_command(self, command: object, now: int) -> list[tuple[str, Message]]:
        """A client's request to replicate `command`.

        Returns `[]` and does nothing if this node isn't currently
        leader -- the caller is expected to redirect to `leader_id`
        instead of retrying here. Otherwise appends `command` (at this
        leader's current term) to its own log and immediately sends it
        on to every peer via the same per-peer AppendEntries logic a
        heartbeat uses.
        """
        if self.role is not Role.LEADER:
            return []

        self.storage.append_entries([LogEntry(term=self.current_term, command=command)])
        return self._send_append_entries(now)

    def _has_majority(self) -> bool:
        cluster_size = len(self.peers) + 1  # +1 for this node
        return len(self.votes_received) * 2 > cluster_size

    # -- Shared bookkeeping --

    def _reset_election_deadline(self, now: int) -> None:
        lo, hi = self.election_timeout_range
        self.election_deadline = now + self._rng.randint(lo, hi)

    def _persist_if_dirty(self) -> None:
        if self._dirty:
            self.storage.save_term_and_vote(self.current_term, self.voted_for)
            self._dirty = False


class RaftClusterNode:
    """Bridges `RaftNode`'s string node ids to `Cluster`'s integer ones.

    `RaftNode` addresses peers as `str` (`candidate_id`, `leader_id`, ...
    are all `str` in `raft.messages`, matching Figure 2). `Cluster` numbers
    its nodes as `int` (0..n-1) and uses that as the key for its own
    routing and liveness bookkeeping. Without a translation at this
    boundary, every message a `RaftNode` produces would be addressed to a
    destination `Cluster` has never heard of -- a `str` can never equal
    one of `Cluster`'s `int` ids -- and `Cluster._route` would silently
    drop it as if the destination were permanently down. This wrapper is
    that translation: it converts `int -> str` for outgoing `src`/`now`
    calls into the wrapped `RaftNode`, and `str -> int` for the
    `(dst, payload)` pairs it produces, and otherwise gets entirely out of
    the way. The handful of fields external callers (tests, invariant
    checkers) actually need to inspect -- `role`, `current_term`,
    `voted_for`, `leader_id`, `node_id`, `storage` -- are typed
    properties reading straight through to the real node; anything else
    falls back through `__getattr__`, untyped but still reachable.
    """

    def __init__(self, raft_node: RaftNode) -> None:
        self.raft_node = raft_node

    def __getattr__(self, name: str) -> object:
        return getattr(self.raft_node, name)

    @property
    def node_id(self) -> str:
        return self.raft_node.node_id

    @property
    def role(self) -> Role:
        return self.raft_node.role

    @property
    def current_term(self) -> int:
        return self.raft_node.current_term

    @property
    def voted_for(self) -> str | None:
        return self.raft_node.voted_for

    @property
    def leader_id(self) -> str | None:
        return self.raft_node.leader_id

    @property
    def storage(self) -> Storage:
        return self.raft_node.storage

    def tick(self, now: int) -> list[tuple[int, object]]:
        return [(int(dst), msg) for dst, msg in self.raft_node.tick(now)]

    def handle(self, payload: object, src: int, now: int) -> list[tuple[int, object]]:
        assert isinstance(payload, Message), f"unexpected payload type: {type(payload)!r}"
        produced = self.raft_node.handle(payload, str(src), now)
        return [(int(dst), msg) for dst, msg in produced]


def raft_node_factory(
    n: int,
    seed: int,
    *,
    election_timeout_range: tuple[int, int] = (150, 300),
    heartbeat_interval: int = 50,
) -> Callable[[int, Storage], RaftClusterNode]:
    """Build a `Cluster`-compatible `node_factory` that produces `RaftNode`s.

    `Cluster` numbers its nodes 0..n-1 and only ever calls the factory
    with `(node_id, storage)`; it has no notion of a peer list or of
    per-node randomness. This closes over both: node ids become their
    string form ("0", "1", ...), each node's peer list is every other id,
    and each node's `random.Random` is drawn from one master RNG seeded
    with `seed`, in ascending node-id order -- so the same `(n, seed)`
    always produces the same per-node election timeouts, and therefore
    the same election outcome, regardless of process, platform, or
    Python's hash randomization (which a `hash()`-based derivation would
    have been vulnerable to). Each `RaftNode` comes back wrapped in a
    `RaftClusterNode` so it can actually talk to `Cluster`'s integer-keyed
    routing; `RaftNode`-specific attributes remain reachable straight off
    the wrapper.
    """
    node_ids = [str(i) for i in range(n)]
    master_rng = random.Random(seed)
    node_seeds = {name: master_rng.getrandbits(64) for name in node_ids}

    def factory(node_id: int, storage: Storage) -> RaftClusterNode:
        name = str(node_id)
        peers = [peer for peer in node_ids if peer != name]
        raft_node = RaftNode(
            name,
            peers,
            storage,
            election_timeout_range=election_timeout_range,
            heartbeat_interval=heartbeat_interval,
            rng=random.Random(node_seeds[name]),
        )
        return RaftClusterNode(raft_node)

    return factory
