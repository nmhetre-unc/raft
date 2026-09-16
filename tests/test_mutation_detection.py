"""Mutation testing for the fuzzer's invariant coverage.

Each test here deliberately breaks one specific piece of replication
correctness -- via `monkeypatch`, never committed to `src/raft/node.py`
itself -- and runs the same 50-seed sweep `tests/test_fuzz_sweep.py`
runs, to answer one question: does the fuzzer actually notice?

The honest answer, found by actually running this, is "not always." Two
of the four mutations here are **known, tracked gaps**: the sweep passes
all 50 seeds even though the mutated code is genuinely wrong. Those
tests assert the current (unfortunate) truth -- zero detections -- on
purpose, so that this file itself is the record of the gap, and so that
the day some future invariant check starts catching one, this test
starts failing and demands to be updated rather than staying silently
stale. A test that always passes regardless of what `check_*` actually
does would be worse than no test at all here.

Summary from the actual run (n=5, steps=300, seeds 0..49):

- Always truncating on a resend, even when it fully matches (the
  duplicate-message bug): still 0/50 detected via the sweep, *even with
  `check_no_spurious_truncation` now wired in*. But this isn't a blind
  spot in the checker: constructing its triggering condition directly
  (`test_mutation_1_triggering_condition_is_caught_when_constructed_directly`)
  and running the real, mutated `_handle_append_entries` against it shows
  it fires immediately. The gap is specifically that the fuzzer's chaos
  never produces that condition on its own -- confirmed by directly
  instrumenting `Storage.truncate_from` across thousands of fuzzer-driven
  truncations. A narrower, better-understood claim than "the checker
  can't see this": the checker can: the fuzzer doesn't reach it.
- Advancing match_index on a *failed* reply, not just success: 0/50
  detected. GAP.
- Skipping the AppendEntries consistency check entirely: 42/50 detected,
  always via check_log_matching.
- Leader-side next_index initialized to 1 instead of last_log_index + 1:
  0/50 detected -- and on reflection this one may not be a safety gap at
  all, just a wasteful one (see its test for why).
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

    return [(src, AppendEntriesReply(term=self.current_term, success=True))]


def test_mutation_always_truncate_on_resend_is_still_not_detected_by_the_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation 1 against the 50-seed sweep: relies on the fuzzer's chaos
    to construct the triggering condition on its own.

    `check_no_spurious_truncation` (added specifically for this gap,
    comparing log entry object identity, not just value) is wired into
    every sweep run. It works: `tests/test_invariants.py` proves it fires
    on a hand-constructed "different object, same value" swap, and
    `test_mutation_1_triggering_condition_is_caught_when_constructed_directly`
    below proves it fires on this *exact* mutated handler, not just the
    checker in isolation. But this sweep still finds 0/50 -- because that
    specific condition never actually arises here. Instrumenting
    `Storage.truncate_from` directly across thousands of fuzzer-driven
    truncations (50 seeds heavy on every action, then 200 more at n=7
    with chaos weights skewed hard toward crash/restart/partition/
    client-request) turned up plenty of real truncations, but every
    single one was either (a) the exact same object being removed and put
    back -- a resend from the *same* leader's own unchanged storage
    always carries the same references, since nothing ever clones a
    `LogEntry` -- or (b) a genuine value change from real conflict
    resolution. Never (c), a different object holding an equal value.

    The reason turns out to be structural, not just unlucky: `next_index`
    only ever walks backward one step at a time, on rejection, and a
    rejection means the follower's entry at `prev_log_index` doesn't
    match -- so the walk necessarily lands exactly on the point of
    genuine agreement before any entries are ever included in a message.
    Whatever a leader sends past that point is therefore always either
    new or genuinely conflicting, never "the follower already has this,
    just from someone else" -- which is exactly the case
    `check_no_spurious_truncation` needs to fire. Given the test below
    proves the checker itself is sound, the gap here is specifically one
    of *fuzzer reachability*: closing it for real would need a fuzzer
    action built to construct that condition directly, not a better
    checker.
    """
    monkeypatch.setattr(RaftNode, "_handle_append_entries", _mutation_1_always_truncate_on_resend)

    violations = _sweep()

    assert violations == {}, f"expected this known gap to stay undetected; found: {violations}"


def test_mutation_1_triggering_condition_is_caught_when_constructed_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation 1, given its exact triggering condition by hand instead of
    hoping the fuzzer's chaos produces it: does check_no_spurious_truncation
    actually fire?

    The condition is an AppendEntries carrying an entry the follower
    already holds, where the entry in the message and the one already
    stored are *different Python objects* despite being value-equal --
    which the sweep test above establishes essentially never arises
    through this simulator's normal leader/follower traffic (a resend
    from the same leader's own storage always carries the same object
    references). The most plausible real-world shape for it anyway is a
    duplicated message redelivered after the follower already applied an
    earlier copy of the same logical entry, but that redelivered copy
    was reconstructed from a source other than what the follower already
    has -- e.g. two leaders who each independently inherited the same
    already-committed entry from a common ancestor, one of them now
    retrying what it believes is still an unacknowledged send. This test
    constructs exactly that end state directly: no election, no second
    node's storage involved, just a follower that already has an entry
    and a message that resends "the same" one as a distinct object.

    If this fires, `check_no_spurious_truncation` is proven sound end to
    end -- through the real, mutated `_handle_append_entries` entry
    point, not just against a hand-edited `Storage` -- and the sweep
    test's gap is purely about the fuzzer never reaching this state, a
    materially narrower claim than "the checker can't see this bug."
    """
    monkeypatch.setattr(RaftNode, "_handle_append_entries", _mutation_1_always_truncate_on_resend)

    cluster = make_cluster(n=2, seed=1)
    follower = cluster.get_node(1)
    assert follower is not None

    # The follower already applied this entry, from an earlier delivery.
    already_applied = LogEntry(term=1, command="x")
    follower.storage.append_entries([already_applied])

    # check_no_spurious_truncation's baseline, exactly as the fuzzer would
    # have recorded it after observing this state on an earlier step.
    history = {follower.node_id: follower.storage.load_log()}
    check_no_spurious_truncation(cluster, history)  # sanity: nothing wrong yet

    # A redelivered "duplicate" of the same logical entry -- but a
    # genuinely different object, as if reconstructed from a different
    # origin than what the follower already has. Value-equal,
    # object-different: exactly what a message duplicated in flight looks
    # like from the receiving end.
    resent = LogEntry(term=1, command="x")
    assert resent is not already_applied
    assert resent == already_applied

    retry_message = AppendEntries(
        term=1,
        leader_id="0",
        prev_log_index=0,
        prev_log_term=0,
        entries=(resent,),
        leader_commit=0,
    )
    follower.raft_node.handle(retry_message, src="0", now=0)

    # Confirm the mutated handler actually did tear the entry down and
    # rebuild it -- same value, different object -- before checking.
    rebuilt = follower.storage.load_log()[1]
    assert rebuilt == already_applied
    assert rebuilt is not already_applied

    with pytest.raises(SafetyViolation, match="torn down and rebuilt"):
        check_no_spurious_truncation(cluster, history)


def test_mutation_match_index_advances_on_failure_is_not_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation 2: advance match_index/next_index on a *failed* reply too,
    not only on success -- so the leader believes entries are replicated
    that the follower actually rejected.

    KNOWN GAP: 0/50 seeds detect this, with or without keeping the
    original "match_index must not move backward" assertion (checked
    both ways). None of the three implemented checkers inspect
    match_index directly -- it doesn't yet feed into anything
    safety-critical, since there is no commitIndex until Milestone 4.
    This is exactly the kind of gap that commit-index-dependent checks
    (Leader Completeness, State Machine Safety) would need to close,
    since committing based on a corrupted match_index is where this
    would actually cause observable harm.
    """

    def mutated(self: RaftNode, msg: Message, src: str, now: int) -> list[tuple[str, Message]]:
        assert isinstance(msg, AppendEntriesReply)
        if msg.term < self.current_term:
            return []
        if self.role is not Role.LEADER:
            return []
        outstanding = self._outstanding.get(src)
        if outstanding is None:
            return []
        prev_log_index, sent_count = outstanding

        # MUTATION: unconditional, no longer gated on msg.success.
        new_match_index = prev_log_index + sent_count
        self.match_index[src] = new_match_index
        self.next_index[src] = new_match_index + 1

        if msg.success:
            return []

        self.next_index[src] = max(1, self.next_index[src] - 1)
        log = self.storage.load_log()
        return [(src, self._build_append_entries_for(src, log))]

    monkeypatch.setattr(RaftNode, "_handle_append_entries_reply", mutated)

    violations = _sweep()

    assert violations == {}, f"expected this known gap to stay undetected; found: {violations}"


def test_mutation_skipping_the_consistency_check_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation 3: skip the prev_log_index/prev_log_term consistency
    check entirely and always accept -- so entries land at whatever list
    position the follower's log happens to currently end at, regardless
    of what Raft index the leader thinks they're at.

    DETECTED: 42/50 seeds, always via check_log_matching -- the index
    misalignment this causes reliably produces two nodes disagreeing on
    what's stored at a shared (index, term). Genuine invariant coverage,
    not a coincidence.
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

        return [(src, AppendEntriesReply(term=self.current_term, success=True))]

    monkeypatch.setattr(RaftNode, "_handle_append_entries", mutated)

    violations = _sweep()

    assert len(violations) > 0, "expected this to be reliably caught; detection ability regressed"
    assert all("log matching violated" in reason for reason in violations.values())


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
        self._outstanding = {}
        return self._send_append_entries(now)

    monkeypatch.setattr(RaftNode, "_become_leader", mutated)

    violations = _sweep()

    assert violations == {}, f"expected this known gap to stay undetected; found: {violations}"
