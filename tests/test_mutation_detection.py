"""Mutation testing for the fuzzer's invariant coverage.

Each test here deliberately breaks one specific piece of replication
correctness -- via `monkeypatch`, never committed to `src/raft/node.py`
itself -- and runs the same 50-seed sweep `tests/test_fuzz_sweep.py`
runs, to answer one question: does the fuzzer actually notice?

The honest answer, found by actually running this, is "not always." One
of the four mutations here is a **known, tracked gap**: the sweep passes
all 50 seeds even though the mutated code is genuinely wrong. That test
asserts the current (unfortunate) truth -- zero detections -- on
purpose, so that this file itself is the record of the gap, and so that
the day some future invariant check starts catching it, this test starts
failing and demands to be updated rather than staying silently stale. A
test that always passes regardless of what `check_*` actually does would
be worse than no test at all here.

Summary from the actual run (n=5, steps=300, seeds 0..49), after
`check_leader_completeness` and `check_state_machine_safety` (Milestone
4's commit-index-dependent checks) were wired in:

- Always truncating on a resend, even when it fully matches (the
  duplicate-message bug): **43/50 detected, all via
  `check_no_spurious_truncation`** (0 via `check_leader_completeness`,
  which runs later in the fixed check order and never gets a chance to
  fire on these seeds -- a `SafetyViolation` stops the run at the first
  property that catches it). This supersedes an earlier finding recorded
  here (and in `BUGS.md`) that read 2/50, always via
  `check_leader_completeness`, and treated `check_no_spurious_
  truncation`'s own trigger as a fuzzer-reachability gap. That reading
  was wrong about the checker, not the mutation: `check_no_spurious_
  truncation` used to compare log entries by *object identity*
  ("different object, same value"), a signature that really is
  unreachable through this simulator -- no `LogEntry` is ever cloned, so
  a resend built from a leader's own unchanged storage always carries
  the exact same object references a follower already stored, regardless
  of delivery order. But the mutation's actual damage was never about
  identity: always truncating from `prev_log_index + 1`, even when
  nothing conflicts, means a *stale, shorter* `AppendEntries` -- built
  from an earlier, smaller `next_index`, delivered by the (reordering)
  `Network` *after* a longer one already extended a follower past it --
  discards everything that follower held beyond the stale message's own
  range, with nothing put back. That is a genuine, *value*-level loss
  (an index present before is simply gone after, or changed with no term
  conflict to justify it), not an identity mismatch, and it went
  undetected by the old identity-based checker for the same structural
  reason its own designed-for signature was unreachable: no cloning means
  no "different object" ever appears, whether an entry survives, changes
  legitimately, or is silently dropped. `check_no_spurious_truncation`
  was rewritten to compare by value instead (see `raft/invariants.py` and
  `BUGS.md`), and now catches this directly -- including seeds 3 and 35,
  the two seeds the old `check_leader_completeness`-only reading found,
  now caught 140 and 37 steps earlier respectively, right when the loss
  happens instead of only once a much later election exposes it.
  `test_mutation_1_stale_shorter_delivery_is_caught_by_new_checker_not_old`
  constructs that exact large-then-small delivery by hand and proves the
  old checker misses it while the new one doesn't, so this isn't just a
  sweep-count coincidence. The remaining 7/50 undetected seeds
  (1, 2, 8, 9, 10, 43, 45) simply never happen to produce the triggering
  reorder within 300 steps -- a fuzzer-coverage question, not a checker
  blind spot.
- Advancing match_index/next_index on a rejected reply, trusting it
  regardless of the reply's success flag: 4/50 detected, always via
  `check_leader_completeness` -- some later, honestly elected leader
  ends up missing an entry an earlier leader believed it had committed,
  purely because of the corrupted match_index. This is exactly the gap
  Leader Completeness was added to close, and it does. (The original
  version of this mutation directly corrupted a leader-side
  `_outstanding` record that no longer exists: fixing the real bug that
  record turned out to enable, found via this exact checker on
  *unmutated* code, removed it -- see BUGS.md. The mutation is rewritten
  here to recreate the same class of bug against the message shape that
  replaced it.)
- Skipping the AppendEntries consistency check entirely: 33/50 detected
  -- 32 via check_log_matching, 1 via check_leader_completeness (see that
  test for why one seed's report differs from the rest).
- Leader-side next_index initialized to 1 instead of last_log_index + 1:
  0/50 detected -- GAP, and on reflection this one may not be a safety
  gap at all, just a wasteful one (see its test for why).
"""

import pytest

from raft.invariants import SafetyViolation, check_no_spurious_truncation
from raft.messages import AppendEntries, AppendEntriesReply, Message
from raft.node import RaftNode, Role, raft_node_factory
from raft.sim.cluster import Cluster
from raft.sim.fuzz import Fuzzer
from raft.storage import LogEntry

N_NODES = 5
STEPS_PER_SEED = 300
SEED_COUNT = 50


def make_cluster(n: int, seed: int) -> Cluster:
    return Cluster(n=n, seed=seed, node_factory=raft_node_factory(n, seed))


def _sweep() -> dict[int, str]:
    """Run the 50-seed sweep; return {seed: violation.reason} for every hit."""
    violations: dict[int, str] = {}
    for seed in range(SEED_COUNT):
        fuzzer = Fuzzer(n=N_NODES, seed=seed, steps=STEPS_PER_SEED)
        try:
            fuzzer.run()
        except SafetyViolation as violation:
            violations[seed] = violation.reason
    return violations


def _mutation_1_always_truncate_on_resend(
    self: RaftNode, msg: Message, src: str, now: int
) -> list[tuple[str, Message]]:
    """Mutation 1: always truncate+reappend from prev_log_index+1, even
    when the resent entries already match what's stored -- the exact
    duplicate-message case `_handle_append_entries`'s real implementation
    goes out of its way to make a no-op. Shared by both tests below so
    the sweep test and the direct-construction test are provably
    exercising the identical mutation, not two that have drifted apart.
    """
    assert isinstance(msg, AppendEntries)
    if msg.term < self.current_term:
        return [(src, AppendEntriesReply(term=self.current_term, success=False))]

    self.role = Role.FOLLOWER
    self.leader_id = msg.leader_id
    self._reset_election_deadline(now)

    log = self.storage.load_log()
    if msg.prev_log_index >= len(log) or log[msg.prev_log_index].term != msg.prev_log_term:
        return [(src, AppendEntriesReply(term=self.current_term, success=False))]

    # MUTATION: no "already matches" check -- always truncate+append.
    if msg.entries:
        conflict_index = msg.prev_log_index + 1
        if conflict_index < len(log):
            self.storage.truncate_from(conflict_index)
        self.storage.append_entries(list(msg.entries))

    match_index = msg.prev_log_index + len(msg.entries)
    reply = AppendEntriesReply(term=self.current_term, success=True, match_index=match_index)
    return [(src, reply)]


def _old_identity_based_check_no_spurious_truncation(
    cluster: Cluster, history: dict[str, list[LogEntry]]
) -> None:
    """A frozen copy of `check_no_spurious_truncation` as it existed before
    the value-based rewrite -- kept here ONLY so
    `test_mutation_1_stale_shorter_delivery_is_caught_by_new_checker_not_old`
    can prove the rewrite is actually discriminating (catches something
    real the old version missed), not just differently worded. Never
    imported from `raft.invariants`; this is a historical artifact, not
    live code.
    """
    for node_id in cluster.node_ids():
        node = cluster.get_node(node_id)
        if node is None:
            continue
        current_log = node.storage.load_log()
        previous_log = history.get(node.node_id)
        if previous_log is not None:
            for index in range(min(len(current_log), len(previous_log))):
                old_entry = previous_log[index]
                new_entry = current_log[index]
                if new_entry is not old_entry and new_entry == old_entry:
                    raise SafetyViolation(
                        f"node {node.node_id!r} log entry at index {index} was "
                        f"replaced by a different (but equal) object -- torn down "
                        f"and rebuilt rather than left alone",
                        seed=cluster.seed,
                        step=cluster.step_count,
                    )
        history[node.node_id] = current_log


def test_mutation_always_truncate_on_resend_is_now_caught_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation 1 against the 50-seed sweep: relies on the fuzzer's chaos
    to construct the triggering condition on its own.

    `check_no_spurious_truncation` is now value-based (see
    `raft/invariants.py` and `BUGS.md`): it no longer looks for a
    "different object, same value" swap (a signature this simulator
    structurally never produces, since no `LogEntry` is ever cloned), and
    instead flags any index that lost or changed value with no term
    conflict to justify it. That is exactly what always truncating from
    `prev_log_index + 1` produces the moment a stale, shorter
    `AppendEntries` -- built from an earlier, smaller `next_index` -- is
    delivered by the (reordering) `Network` after a longer one already
    extended a follower past it: everything beyond the stale message's
    own range is discarded and never restored.

    43/50 seeds now catch this, all via `check_no_spurious_truncation`
    (0 via `check_leader_completeness`, which runs later in the fixed
    check order in `Fuzzer._check_invariants` and never gets a chance --
    a `SafetyViolation` stops the run at the first property that catches
    it). Seeds 3 and 35 -- the two seeds the previous, identity-based
    checker missed entirely and only `check_leader_completeness` used to
    catch, 168 and 164 steps in respectively -- are both in this set,
    caught at step 28 and 127: 140 and 37 steps earlier, right when the
    loss happens instead of only once a much later election exposes it.
    See `test_mutation_1_stale_shorter_delivery_is_caught_by_new_checker_not_old`
    for a hand-constructed proof that this is the new checker actually
    discriminating, not a coincidence of the sweep.
    """
    monkeypatch.setattr(RaftNode, "_handle_append_entries", _mutation_1_always_truncate_on_resend)

    violations = _sweep()

    assert len(violations) == 43, f"detection rate shifted: {violations}"
    assert all(
        "log entry at index" in reason and "lost or changed" in reason
        for reason in violations.values()
    )
    assert 3 in violations and 35 in violations


def test_mutation_1_stale_shorter_delivery_is_caught_by_new_checker_not_old(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The diagnosis's hand-constructed scenario, driven through the real,
    mutated `_handle_append_entries` entry point: a leader's own log grows
    from 2 entries to 4, and two `AppendEntries` built at those two
    different moments are delivered out of order -- the longer one first
    (as `Network` reordering can always do), then the shorter, now-stale
    one. No rejection, no retry, no leadership change: the follower's own
    `next_index`-driven growth alone produces two legitimately-built
    messages that a real leader would send, and delivery order does the
    rest.

    Proves the rewrite is actually discriminating, not just differently
    worded: the OLD, identity-based checker (frozen above as
    `_old_identity_based_check_no_spurious_truncation`) does NOT raise on
    this exact sequence, because the reappended entries are the same
    `LogEntry` objects the leader already held (nothing is ever cloned) --
    while the NEW, value-based `check_no_spurious_truncation` does,
    because two of the follower's entries are simply gone with no term
    conflict to justify it.
    """
    monkeypatch.setattr(RaftNode, "_handle_append_entries", _mutation_1_always_truncate_on_resend)

    cluster = make_cluster(n=2, seed=1)
    follower = cluster.get_node(1)
    assert follower is not None

    e1 = LogEntry(term=1, command="a")
    e2 = LogEntry(term=1, command="b")
    e3 = LogEntry(term=1, command="c")
    e4 = LogEntry(term=1, command="d")

    send_later_large = AppendEntries(
        term=1, leader_id="0", prev_log_index=0, prev_log_term=0,
        entries=(e1, e2, e3, e4), leader_commit=0,
    )
    send_early_small = AppendEntries(
        term=1, leader_id="0", prev_log_index=0, prev_log_term=0,
        entries=(e1, e2), leader_commit=0,
    )

    # Delivered large-then-small: the Network reordering that produced
    # seeds 3 and 35.
    follower.raft_node.handle(send_later_large, src="0", now=0)
    assert follower.storage.last_log_index() == 4

    history_old: dict[str, list[LogEntry]] = {follower.node_id: follower.storage.load_log()}
    history_new: dict[str, list[LogEntry]] = {follower.node_id: follower.storage.load_log()}

    follower.raft_node.handle(send_early_small, src="0", now=1)
    assert follower.storage.last_log_index() == 2, "e3 and e4 should be silently discarded"

    # OLD checker: identity-based, misses it.
    _old_identity_based_check_no_spurious_truncation(cluster, history_old)  # must not raise

    # NEW checker: value-based, catches it.
    with pytest.raises(SafetyViolation, match="lost or changed"):
        check_no_spurious_truncation(cluster, history_new)


def test_mutation_match_index_advances_on_failure_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation 2: advance match_index/next_index on a *failed* reply too,
    not only on success -- so the leader believes entries are replicated
    that the follower actually rejected.

    Rewritten for the post-fix message shape: `AppendEntriesReply` now
    carries the follower's own honestly-computed `match_index`, so there
    is no leader-side "what did I last send this peer" record left to
    corrupt the way the original version of this mutation did (it patched
    `_outstanding`, which no longer exists -- see BUGS.md for the real bug
    that removed it, found via this exact checker). Reproducing the same
    *class* of bug -- a leader crediting a peer with an index nothing
    actually confirmed -- now means ignoring the reply's honestly-reported
    match_index and instead assuming this leader's own current log is
    fully replicated to the peer, regardless of what the reply actually
    says.

    DETECTED, always via check_leader_completeness: some later, honestly
    elected leader lacks an entry an earlier leader believed committed
    only because of this corruption. This is exactly the gap
    `check_leader_completeness` was added to close: nothing about
    election safety, log matching, or append-only-ness inherently
    depends on match_index being trustworthy, but commitment does.
    """

    def mutated(self: RaftNode, msg: Message, src: str, now: int) -> list[tuple[str, Message]]:
        assert isinstance(msg, AppendEntriesReply)
        if msg.term < self.current_term:
            return []
        if self.role is not Role.LEADER:
            return []

        # MUTATION: trust that this peer has fully caught up to this
        # leader's own current log -- ignoring what the reply actually
        # confirms -- unconditionally, not gated on msg.success.
        believed_match_index = self.storage.last_log_index()
        if believed_match_index > self.match_index[src]:
            self.match_index[src] = believed_match_index
            self.next_index[src] = believed_match_index + 1

        if msg.success:
            return []

        self.next_index[src] = max(1, self.next_index[src] - 1)
        log = self.storage.load_log()
        return [(src, self._build_append_entries_for(src, log))]

    monkeypatch.setattr(RaftNode, "_handle_append_entries_reply", mutated)

    violations = _sweep()

    assert len(violations) > 0, "expected this to be detected; check_leader_completeness regressed"
    assert all("missing entry" in reason for reason in violations.values())


def test_mutation_skipping_the_consistency_check_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation 3: skip the prev_log_index/prev_log_term consistency
    check entirely and always accept -- so entries land at whatever list
    position the follower's log happens to currently end at, regardless
    of what Raft index the leader thinks they're at.

    DETECTED: 33/50 seeds -- 32 via check_log_matching (the index
    misalignment this causes reliably produces two nodes disagreeing on
    what's stored at a shared (index, term)), and 1 via
    check_leader_completeness, where the same misalignment happens to
    surface as a later leader missing an entry before the log-matching
    divergence it would also eventually cause ever gets checked (a
    `SafetyViolation` stops the run at the first property that catches
    it, so which one fires first for a given seed depends on exactly
    when each condition becomes checkable, not which bug is "more
    real"). Both are genuine invariant coverage of the same underlying
    mutation, not a coincidence.
    """

    def mutated(self: RaftNode, msg: Message, src: str, now: int) -> list[tuple[str, Message]]:
        assert isinstance(msg, AppendEntries)
        if msg.term < self.current_term:
            return [(src, AppendEntriesReply(term=self.current_term, success=False))]

        self.role = Role.FOLLOWER
        self.leader_id = msg.leader_id
        self._reset_election_deadline(now)

        log = self.storage.load_log()
        # MUTATION: no consistency check at all -- straight to appending.

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
                self.storage.truncate_from(conflict_index)
            self.storage.append_entries(list(new_entries))

        match_index = msg.prev_log_index + len(msg.entries)
        reply = AppendEntriesReply(term=self.current_term, success=True, match_index=match_index)
        return [(src, reply)]

    monkeypatch.setattr(RaftNode, "_handle_append_entries", mutated)

    violations = _sweep()

    assert len(violations) > 0, "expected this to be reliably caught; detection ability regressed"
    assert all(
        "log matching violated" in reason or "missing entry" in reason
        for reason in violations.values()
    )


def test_mutation_pessimistic_next_index_is_not_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation 4: initialize next_index to 1 for every peer on becoming
    leader, instead of last_log_index + 1.

    NOT DETECTED (0/50) -- and unlike the other two gaps, this one is
    arguably not a safety bug at all under the current invariants: index
    1 means prev_log_index=0, the sentinel, which always passes the
    consistency check, so every peer just accepts a redundant resend of
    entries it may already have (correctly recognized as a no-op by the
    real, unmutated follower-side handler) and converges to the same
    correct state, just less efficiently. It's recorded here anyway,
    since "not currently reachable as a violation" and "provably
    harmless" are different claims, and only testing settles which one
    this is for a given invariant set.
    """

    def mutated(self: RaftNode, now: int) -> list[tuple[str, Message]]:
        self.role = Role.LEADER
        self.leader_id = self.node_id
        # MUTATION: pessimistic (1) instead of optimistic (last_log_index + 1).
        self.next_index = dict.fromkeys(self.peers, 1)
        self.match_index = dict.fromkeys(self.peers, 0)
        return self._send_append_entries(now)

    monkeypatch.setattr(RaftNode, "_become_leader", mutated)

    violations = _sweep()

    assert violations == {}, f"expected this known gap to stay undetected; found: {violations}"
