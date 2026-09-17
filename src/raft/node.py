"""The Raft node as a pure state machine.

`RaftNode` implements leader election, log replication in both
directions, commitment -- deciding when an entry is safe -- and
applying committed entries to a replicated key/value state machine
(`raft.statemachine.KVStateMachine`). There is still no log compaction;
that remains a later milestone.

Applying is Figure 2's separate step, deliberately kept separate here
too: `_apply_committed_entries` never decides *whether* something is
safe (that's `_advance_commit_index`'s job alone, untouched by this),
only *replays* whatever is already committed, one entry at a time, in
order, toward whatever the state machine hasn't seen yet. It is pure
state-machine-side logic -- a leader gets no special treatment here, and
every node, in every role, applies its own committed entries identically.
`last_applied` (like `commit_index`) stays volatile, not persisted, for
the same reason: a restarted node's `Storage` is untouched by the crash
(see `raft.storage`'s module docstring), so replaying the same committed
log from index 1 into a fresh `KVStateMachine` after a restart
reconstructs the identical final state a real disk-backed state machine
would have kept -- redundant durability, not a gap. A real client
request is logged as a `raft.statemachine.ClientRequest` (carrying the
client's own `client_id`/`serial_number` alongside its `Command`) and
applied through `state_machine.apply_client_request`, which deduplicates
a retried request instead of re-running it -- see that method's own
docstring for the exact rule, and for why living on `KVStateMachine`
itself, not anywhere leader-specific, is what makes it safe across a
leader change. A bare `Command` (no client wrapper) still goes straight
through `state_machine.apply`, exactly as before `ClientRequest` existed.
A command reaching `_apply_committed_entries` that is neither of those
(the log's own index-0 sentinel, `_become_leader`'s content-free no-op,
or -- outside this file -- a test using an opaque placeholder command to
exercise replication without caring about state-machine semantics) is
treated as inert: `last_applied` still advances past it, one index at a
time, exactly like any other entry, but nothing is fed to the state
machine for it. `get` is NOT put through this path at all in this
milestone -- see `raft.statemachine.KVStateMachine.read` for the
local-read alternative used instead, and the linearizability limitation
that comes with it.

Commitment's one rule matters more than everything else in this file
combined, and it is stated here in full rather than assumed known:
**a leader may only ever advance `commit_index` to an index whose entry
was written in its own current term.** A leader that instead commits the
moment a majority merely *holds* an entry -- regardless of which term
wrote it -- is unsafe: Figure 8 of the Raft paper shows a concrete
five-node execution where doing so lets a later leader silently
overwrite an entry an earlier leader had already told a client was
committed. Entries from earlier terms are never committed *directly* --
only *indirectly*, by riding along beneath a current-term entry once
that one commits, via the Log Matching Property guaranteeing everything
below it is already identical everywhere it exists. `_advance_commit_index`
enforces this with an explicit, separately-commented guard rather than
folding it into the majority arithmetic, specifically so it can never be
mistaken for an incidental detail.

A leader tracks two things per peer, and they are not interchangeable:

- `next_index[peer]` is a *guess* -- the next log index this leader
  believes it should try sending that peer. It starts optimistic (one
  past this leader's own last entry), decrements by exactly one on a
  rejection (no conflict-term optimization yet), and jumps forward again
  on an acceptance.
- `match_index[peer]` is *known truth* -- the highest index this leader
  has actual confirmation the peer has replicated. It starts at 0
  (known nothing) and only ever moves forward, once a real
  acknowledgement arrives -- enforced with an assertion, not a `max()`,
  because a would-be regression means a reply got misattributed
  somewhere, and that is a bug to surface, not paper over.

Neither is derived from the other, and nothing in this file computes one
from the other -- an optimistic guess and a confirmed fact answer
different questions, and conflating them is exactly how a leader would
end up believing a follower has entries it never actually got.

`AppendEntriesReply` carries the follower's own `match_index`: the index
it has actually verified it holds, computed by the follower itself from
the specific request it's answering (see `raft.messages`). An earlier
design instead had the *leader* track, per peer, the `(prev_log_index,
entry_count)` it most recently *sent* -- overwritten on every send -- and
used that recorded value to interpret whatever reply came back, on the
theory that a second send before the first's reply arrives only ever
carries a superset of what the first one sent (the log only grows), so
whichever reply got (mis)attributed to the latest record would still land
on the correct final `match_index`. That theory was wrong: the fuzzer
found a live counterexample (see `BUGS.md`) where the reply to the
*first*, *smaller* send arrived after a second, larger send had already
overwritten the record, so the small reply got interpreted against the
large one -- inflating `match_index` past what the peer had actually
confirmed, in one seed enough to commit an entry with only two of five
nodes truly holding it. Having the follower report its own confirmed
index removes the guesswork entirely: whatever a given reply says, it says
about *itself*, correctly, regardless of what other requests are also in
flight. The leader's only remaining job is to never let a reply move
`match_index` backward (`_handle_append_entries_reply` takes the max, not
an overwrite) -- an old, delayed reply reporting a smaller confirmed index
than one already recorded is simply stale, not corrupt, and is expected
to arrive that way sometimes.

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
from raft.statemachine import ClientRequest, Delete, Get, KVStateMachine, Put
from raft.storage import LogEntry, Storage


class Role(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class RaftNode:
    """A single Raft node's state machine: election, replication, commitment,
    and applying committed entries to a replicated key/value store.

    Persistent (loaded from `storage` at construction, saved back to it
    before any reply that changed them is returned): `current_term`,
    `voted_for`.

    Volatile (never touches `storage`, reset to nothing meaningful by a
    fresh construction -- i.e. by a simulated crash and restart):
    `role`, `leader_id`, `election_deadline`, `votes_received`,
    `commit_index`, `last_applied`, `state_machine`, and -- only
    meaningful while `role is Role.LEADER`, reinitialized fresh on every
    election win -- `next_index`, `match_index`. `commit_index` staying
    volatile is deliberate, not an oversight: it is fully reconstructible
    from the current leader's next heartbeat after a restart (see
    `_handle_append_entries`'s step 5), so persisting it would just be
    redundant durability for a value every live node already recomputes
    on its own. `last_applied`/`state_machine` are volatile for the
    matching reason: replaying the untouched, persisted log back into a
    fresh `KVStateMachine` after a restart reconstructs the identical
    final state (see the module docstring).
    """

    def __init__(
        self,
        node_id: str,
        peers: list[str],
        storage: Storage,
        election_timeout_range: tuple[int, int] = (150, 300),
        heartbeat_interval: int = 50,
        rng: random.Random | None = None,
        append_noop_on_election: bool = False,
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
        # Whether _become_leader appends a term-stamped, content-free entry
        # on every election win. A flag, not always-on, specifically so a
        # test can run the same node both ways: see _become_leader for why
        # a leader with nothing at its own current term can never commit
        # anything at all, including entries inherited from earlier terms.
        self.append_noop_on_election = append_noop_on_election

        # Persistent state -- loaded now, written back by _persist_if_dirty().
        self.current_term, self.voted_for = storage.load_term_and_vote()
        self._dirty = False

        # Volatile state -- gone and rebuilt on every crash/restart.
        self.role = Role.FOLLOWER
        self.leader_id: str | None = None
        self.votes_received: set[str] = set()
        self.election_deadline: int = 0
        self._next_heartbeat: int = 0
        # The highest index this node believes is committed, and the
        # highest index actually applied to state_machine so far. Both
        # volatile -- see the class docstring for why that's safe.
        self.commit_index: int = 0
        self.last_applied: int = 0
        self.state_machine = KVStateMachine()
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
        self._apply_committed_entries()
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
        self._apply_committed_entries()
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

    # -- AppendEntries (Figure 2's receiver implementation, commit tracking included) --

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

        # 5. If leaderCommit > commitIndex, set commitIndex = min(leaderCommit,
        # index of last new entry). "Index of last new entry" means the last
        # index *this message* covers -- prev_log_index + len(entries) --
        # regardless of whether those entries needed writing or already
        # matched; a bare heartbeat (entries=()) still legitimately advances
        # commit_index up to prev_log_index this way. The min() is not
        # optional: without it, a follower would happily adopt a leader's
        # higher commit_index even when its own log doesn't yet reach that
        # far (it could still be catching up from an earlier, smaller
        # message), which would mean claiming an entry is committed before
        # this follower has even seen it.
        last_new_index = msg.prev_log_index + len(msg.entries)
        if msg.leader_commit > self.commit_index:
            self.commit_index = min(msg.leader_commit, last_new_index)

        reply = AppendEntriesReply(term=self.current_term, success=True, match_index=last_new_index)
        return [(src, reply)]

    def _handle_append_entries_reply(
        self, msg: AppendEntriesReply, src: str, now: int
    ) -> list[tuple[str, Message]]:
        if msg.term < self.current_term:
            return []  # stale reply, from a term we've since moved past
        if self.role is not Role.LEADER:
            return []  # no longer leading; not ours to act on anymore

        if msg.success:
            # msg.match_index is the follower's own report of what IT
            # verified, computed from the specific request this reply
            # answers -- not this leader's guess about what it sent. Only
            # advance, never overwrite: replies can arrive out of order in
            # this simulator, and an old, delayed reply reporting a
            # smaller confirmed index than one already recorded is simply
            # stale, not a sign anything went backward on the follower
            # (see the module docstring for the bug this design replaced).
            if msg.match_index > self.match_index[src]:
                self.match_index[src] = msg.match_index
                self.next_index[src] = msg.match_index + 1
                self._advance_commit_index()
            return []

        # Failure, and (per the checks above) at our current term, not a
        # stale-term step-down: back off by exactly one and retry
        # immediately with the earlier prev_log_index. No conflict-term
        # optimization yet -- a later refinement, not this milestone.
        self.next_index[src] = max(1, self.next_index[src] - 1)
        log = self.storage.load_log()
        return [(src, self._build_append_entries_for(src, log))]

    def _advance_commit_index(self) -> None:
        """Figure 2's leader commit rule -- with Figure 2's own last-paragraph
        restriction (see also Figure 8) applied as an explicit, separate
        guard, not folded into the majority check it sits beside.

        The rule in full: a leader may advance commit_index to N if (a) a
        majority of match_index values -- counting this leader's own log,
        which trivially "matches" itself completely -- are >= N, AND (b)
        the entry actually stored at index N was written in this leader's
        OWN CURRENT TERM. Both conditions are required; neither implies
        the other. Search from this leader's own last index downward, so
        the first N found (if any) satisfying both is the highest such N.
        """
        match_counts = [*self.match_index.values(), self.storage.last_log_index()]
        majority_needed = len(match_counts) // 2 + 1

        log = self.storage.load_log()
        for candidate in range(self.storage.last_log_index(), self.commit_index, -1):
            confirmations = sum(1 for match_index in match_counts if match_index >= candidate)
            if confirmations < majority_needed:
                continue  # not enough nodes have reached this index yet

            # THE CRITICAL RESTRICTION (Figure 2, last paragraph; Figure 8):
            # never commit an entry from a term earlier than this leader's
            # own current term, no matter how large the majority already
            # holding it is. An entry only becomes safe to commit this way
            # once a *current-term* entry above it commits -- everything
            # below rides along via the Log Matching Property. Committing
            # an old-term entry directly, on majority alone, is exactly
            # the unsafe case Figure 8 exists to rule out: a later leader
            # that never saw it as committed can still overwrite it.
            if log[candidate].term != self.current_term:
                continue

            self.commit_index = candidate
            return

    def _apply_committed_entries(self) -> None:
        """Figure 2's separate "apply" step: replay every not-yet-applied
        entry, in order, from `last_applied + 1` through `commit_index`,
        advancing `last_applied` by exactly one per entry -- never
        jumping straight to `commit_index`, never skipping one. Called
        unconditionally at the end of `tick`, `handle`, and
        `append_command` -- the same three (and only) places `commit_index`
        can ever change, directly or transitively -- mirroring
        `_persist_if_dirty`'s own pattern of a single, un-skippable
        checkpoint at the end of every public entry point rather than a
        call threaded through each individual commit_index-changing
        branch. The `if` guard below makes the common case (nothing new
        committed since last time) a cheap no-op, so calling this
        unconditionally costs nothing when there's nothing to do.

        Never decides whether an index is safe to apply -- that's
        `_advance_commit_index`'s job alone, and by the time this runs,
        `commit_index` has already been decided. This treats a leader no
        differently than a follower: whoever is calling this just replays
        the committed log it already has.

        `entry.command` reaching an unrecognized type (not a real
        `raft.statemachine.Command` or `ClientRequest`) is treated as
        inert, not an error -- see the module docstring for exactly which
        entries take this path and why applying nothing for them is still
        correct. A `ClientRequest` goes through `state_machine.
        apply_client_request` (the deduplicating path); a bare `Command`
        goes through `state_machine.apply` directly (no session tracking
        at all -- this is what every entry built before `ClientRequest`
        existed still looks like, and stays fully supported).
        """
        if self.last_applied >= self.commit_index:
            return

        log = self.storage.load_log()
        for index in range(self.last_applied + 1, self.commit_index + 1):
            command = log[index].command
            if isinstance(command, ClientRequest):
                self.state_machine.apply_client_request(command)
            elif isinstance(command, Put | Get | Delete):
                self.state_machine.apply(command)
            self.last_applied = index

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

        if self.append_noop_on_election:
            # A leader can only ever commit an entry from its OWN current
            # term directly (_advance_commit_index's guard); everything
            # from an earlier term only ever commits indirectly, by riding
            # along beneath one. Without anything stamped at this leader's
            # own term, it can never commit *anything* at all -- including
            # entries it inherited from earlier leaders that a majority
            # may already hold. This no-op is exactly what closes that
            # gap: term-stamped, content-free, here purely so there is
            # something this leader itself wrote that it can commit.
            self.storage.append_entries([LogEntry(term=self.current_term, command=None)])

        # A lone leader in a single-node cluster (self.peers == []) will
        # never receive an AppendEntriesReply to trigger the usual
        # advancement path, so it gets one explicit check here too --
        # harmless for any multi-node cluster, where every peer's
        # match_index has just been reset to 0 and so can never yet
        # satisfy a majority at any new index.
        self._advance_commit_index()

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
        out: list[tuple[str, Message]] = [
            (peer, self._build_append_entries_for(peer, log)) for peer in self.peers
        ]
        self._next_heartbeat = now + self.heartbeat_interval
        return out

    def _build_append_entries_for(self, peer: str, log: list[LogEntry]) -> AppendEntries:
        """Build the AppendEntries `peer` should get right now, from its next_index."""
        prev_log_index = self.next_index[peer] - 1
        entries = tuple(log[self.next_index[peer] :])
        return AppendEntries(
            term=self.current_term,
            leader_id=self.node_id,
            prev_log_index=prev_log_index,
            prev_log_term=log[prev_log_index].term,
            entries=entries,
            leader_commit=self.commit_index,
        )

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
        # Same reasoning as the lone-leader check in _become_leader: this
        # entry is at self.current_term, so if this node has no peers (or
        # somehow already has a standing majority at this new index), it
        # can commit immediately without waiting on a reply that will
        # never come.
        self._advance_commit_index()
        out = self._send_append_entries(now)
        self._apply_committed_entries()
        return out

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
    one of `Cluster`'s `int` ids -- and `Cluster.route` would silently
    drop it as if the destination were permanently down. This wrapper is
    that translation: it converts `int -> str` for outgoing `src`/`now`
    calls into the wrapped `RaftNode`, and `str -> int` for the
    `(dst, payload)` pairs it produces, and otherwise gets entirely out of
    the way. The handful of fields external callers (tests, invariant
    checkers) actually need to inspect -- `role`, `current_term`,
    `voted_for`, `leader_id`, `node_id`, `storage`, `commit_index`,
    `last_applied`, `state_machine` -- are typed properties reading
    straight through to the real node; anything else falls back through
    `__getattr__`, untyped but still reachable.
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

    @property
    def commit_index(self) -> int:
        return self.raft_node.commit_index

    @property
    def last_applied(self) -> int:
        return self.raft_node.last_applied

    @property
    def state_machine(self) -> KVStateMachine:
        return self.raft_node.state_machine

    def tick(self, now: int) -> list[tuple[int, object]]:
        return [(int(dst), msg) for dst, msg in self.raft_node.tick(now)]

    def handle(self, payload: object, src: int, now: int) -> list[tuple[int, object]]:
        assert isinstance(payload, Message), f"unexpected payload type: {type(payload)!r}"
        produced = self.raft_node.handle(payload, str(src), now)
        return [(int(dst), msg) for dst, msg in produced]

    def append_command(self, command: object, now: int) -> list[tuple[int, object]]:
        """Same translation as `tick`/`handle`, for the client entry point.

        Without this, `__getattr__` would forward straight to
        `raft_node.append_command`, which addresses peers as `str` --
        exactly the mismatch this whole class exists to paper over.
        """
        return [(int(dst), msg) for dst, msg in self.raft_node.append_command(command, now)]


def raft_node_factory(
    n: int,
    seed: int,
    *,
    election_timeout_range: tuple[int, int] = (150, 300),
    heartbeat_interval: int = 50,
    append_noop_on_election: bool = False,
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
            append_noop_on_election=append_noop_on_election,
        )
        return RaftClusterNode(raft_node)

    return factory
