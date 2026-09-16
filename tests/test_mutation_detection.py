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
  duplicate-message bug): `check_no_spurious_truncation`'s own specific
  signature ("different object, same value") still never arises through
  the sweep -- confirmed exactly as before, by directly instrumenting
  `Storage.truncate_from` and by
  `test_mutation_1_triggering_condition_is_caught_when_constructed_directly`,
  which proves the checker fires on this exact mutated handler once that
  condition is constructed by hand. But `check_leader_completeness` now
  catches 2/50 seeds anyway, via a *different* consequence of the same
  mutation: always truncating from `prev_log_index + 1` discards
  anything a follower holds beyond the resent range too, and on those 2
  seeds that collateral damage later mattered to an election. Still a
  narrower claim than "the checker can't see this": `check_no_spurious_
  truncation`'s own trigger remains a fuzzer-reachability gap; it's just
  no longer true that *nothing* catches this mutation.
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


def test_mutation_always_truncate_on_resend_is_occasionally_caught_a_different_way(
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
    checker in isolation. But *that specific signature* still never
    arises here, confirmed the same way as before: instrumenting
    `Storage.truncate_from` directly across thousands of fuzzer-driven
    truncations turned up plenty of real truncations, but every single
    one was either (a) the exact same object being removed and put back
    -- a resend from the *same* leader's own unchanged storage always
    carries the same references, since nothing ever clones a `LogEntry`
    -- or (b) a genuine value change from real conflict resolution.
    Never (c), a different object holding an equal value. That part of
    the original finding stands: `next_index` only ever walks backward
    one step at a time, on rejection, so the walk necessarily lands
    exactly on the point of genuine agreement before any entries are
    ever included in a message -- structurally, not just by chance,
    `check_no_spurious_truncation`'s own trigger is unreachable here.

    What's changed since `check_leader_completeness` was wired in: this
    mutation also always truncates from `prev_log_index + 1` even when
    nothing conflicts, discarding anything a follower holds *beyond* the
    resent range too -- collateral damage the real handler's "already
    matches" check exists specifically to prevent. 2/50 seeds now catch
    that collateral damage, via `check_leader_completeness`, when the
    discarded entries on that follower turn out to matter to a later
    election. This is a materially narrower finding than "the checker
    can't see this bug": `check_no_spurious_truncation`'s own signature
    remains unreached, and this new detection is a *different* checker
    catching a *different* (also real) consequence of the same mutation.
    """
    monkeypatch.setattr(RaftNode, "_handle_append_entries", _mutation_1_always_truncate_on_resend)

    violations = _sweep()

    assert 0 < len(violations) < 10, f"detection rate shifted materially: {violations}"
    assert all("missing entry" in reason for reason in violations.values())


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
