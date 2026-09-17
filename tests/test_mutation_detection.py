"""Mutation testing for the fuzzer's invariant coverage.

Each test here deliberately breaks one specific piece of replication
correctness -- via `monkeypatch`, never committed to `src/raft/node.py`
itself -- and runs the same 50-seed sweep `tests/test_fuzz_sweep.py`
runs, to answer one question: does the fuzzer actually notice?

The honest answer, found by actually running this, is "not always." One
of the five mutations here is a **known, tracked gap**: the sweep passes
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
  duplicate-message bug): **45/50 detected, all via
  `check_no_spurious_truncation`** (0 via `check_leader_completeness`,
  which runs later in the fixed check order and never gets a chance to
  fire on these seeds -- a `SafetyViolation` stops the run at the first
  property that catches it). This supersedes two earlier findings: 2/50
  (always via `check_leader_completeness`, before `check_no_spurious_
  truncation` was rewritten to compare by value instead of object
  identity -- see `raft/invariants.py` and `BUGS.md` for that fix) and
  then 43/50 (once the rewrite landed, but before `STALE_REDELIVER`
  joined `DEFAULT_WEIGHTS`). `check_no_spurious_truncation`'s own
  signature was never actually unreachable -- object identity was; no
  `LogEntry` is ever cloned, so a resend built from a leader's own
  unchanged storage always carries the exact same object references a
  follower already stored, regardless of delivery order, but a *stale,
  shorter* `AppendEntries` -- built from an earlier, smaller `next_index`,
  delivered by the (reordering) `Network` *after* a longer one already
  extended a follower past it -- still discards everything that follower
  held beyond the stale message's own range, with nothing put back: a
  genuine *value*-level loss the identity check couldn't see, not because
  it was rare, but because it was watching the wrong thing.
  `STALE_REDELIVER` (see `raft/sim/fuzz.py`) exists specifically to stop
  relying on that reordering happening by luck: it deliberately
  redelivers an already-delivered `AppendEntries` a destination has since
  moved past, reconstructing the exact precondition on purpose. With it
  wired into the default action mix, detection rose from 43/50 to 45/50.
  The *specific set* of caught seeds shifted, not just grew -- weaving a
  new action into the shared, weighted `self._rng` stream perturbs every
  seed's entire subsequent random walk, not only the ones that need the
  new action, so a handful of seeds that used to trigger this via lucky
  native reordering no longer do, while a larger number that never did
  now do via deliberate redelivery instead. That's expected fuzzer
  behavior, not a regression: what matters is the aggregate count went up
  and seeds 3 and 35 -- the two this diagnosis was built around -- are
  still both in the caught set.
  `test_mutation_1_stale_shorter_delivery_is_caught_by_new_checker_not_old`
  constructs the exact large-then-small delivery by hand (independent of
  any fuzzer randomness) and proves the old, identity-based checker
  misses it while the new, value-based one doesn't, so none of this is a
  sweep-count coincidence. Detection then moved again, 45/50 -> 41/50,
  once CLIENT_REQUEST stopped omnisciently targeting the actual leader
  and started following `raft.node.NotLeader`'s own `leader_hint`
  instead, the way a real client has to (see `raft/sim/fuzz.py`'s
  `_choose_client_request_target`). No randomness was added anywhere --
  but a request that now needs one or more NOT_LEADER round trips before
  it lands on the real leader replicates on a different, generally
  slower cadence, which shifts which seeds happen to land a stale,
  shorter `AppendEntries` inside the reordering window this mutation
  needs. Seed 3 -- one of the two this diagnosis was originally built
  around -- is still in the caught set; seed 35, the other, no longer is,
  for the same reason: its own traffic pattern shifted enough that the
  precondition this mutation needs no longer arises within 300 steps.
  That's expected fuzzer behavior, not a regression in either the
  mutation or the checker -- the hand-constructed proof above is
  untouched by any of this, and remains the actual guarantee that the
  new checker discriminates correctly.
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
- Skipping the AppendEntries consistency check entirely: 31/50 detected
  (was 33/50 before `STALE_REDELIVER` joined `DEFAULT_WEIGHTS` -- see
  that entry above for why a new action shifts counts for mutations it
  has nothing to do with, by perturbing the shared RNG stream, not a
  regression) -- 30 via check_log_matching, 1 via check_leader_completeness
  (see that test for why one seed's report differs from the rest).
- Leader-side next_index initialized to 1 instead of last_log_index + 1:
  0/50 detected -- GAP, and on reflection this one may not be a safety
  gap at all, just a wasteful one (see its test for why).
- Applying past commit_index -- ignoring Figure 2's "apply" boundary
  entirely and replaying straight to the end of the log, committed or
  not (Milestone 5's replicated state machine and `last_applied`
  wiring): 8/50 detected (was 11/50 before CLIENT_REQUEST started
  following `raft.node.NotLeader`'s own `leader_hint` instead of
  omnisciently targeting the actual leader -- see the truncate-on-resend
  entry above for why that traffic-pattern change shifts sweep counts
  without touching either the mutation or the checker), always via
  `check_state_machine_safety` -- the **first real detection this
  checker has ever produced** in this project (see BUGS.md). An entry
  applied before it was actually safe can still be truncated and
  overwritten by a later, honestly elected leader that never saw it as
  committed; when that happens, whatever next applies that same index
  sees different content than what an earlier, premature application
  already recorded there, and the checker's cross-observation comparison
  catches exactly that.
"""

import pytest

from raft.invariants import SafetyViolation, check_no_spurious_truncation
from raft.messages import AppendEntries, AppendEntriesReply, Message
from raft.node import RaftNode, Role, raft_node_factory
from raft.sim.cluster import Cluster
from raft.sim.fuzz import Fuzzer
from raft.statemachine import Delete, Get, Put
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

    41/50 seeds now catch this (was 45/50 before CLIENT_REQUEST stopped
    omnisciently targeting the actual leader and started following
    `raft.node.NotLeader`'s own `leader_hint` instead -- see the module
    docstring for why that traffic-pattern change shifts sweep counts,
    same as `STALE_REDELIVER` joining `DEFAULT_WEIGHTS` did before it,
    without touching either the mutation or the checker), all via
    `check_no_spurious_truncation` (0 via `check_leader_completeness`,
    which runs later in the fixed check order in
    `Fuzzer._check_invariants` and never gets a chance -- a
    `SafetyViolation` stops the run at the first property that catches
    it). Seed 3 -- one of the two seeds the previous, identity-based
    checker missed entirely and only `check_leader_completeness` used to
    catch -- is still in this set; seed 35, the other, no longer is, its
    own traffic pattern having shifted enough that this mutation's
    precondition no longer arises for it within 300 steps. See
    `test_mutation_1_stale_shorter_delivery_is_caught_by_new_checker_not_old`
    for a hand-constructed proof (independent of any fuzzer randomness)
    that this is the new checker actually discriminating, and
    `test_stale_redeliver_reproduces_the_diagnosis_scenario` for the same
    proof driven through `STALE_REDELIVER` itself rather than native
    delivery.
    """
    monkeypatch.setattr(RaftNode, "_handle_append_entries", _mutation_1_always_truncate_on_resend)

    violations = _sweep()

    assert len(violations) == 41, f"detection rate shifted: {violations}"
    assert all(
        "log entry at index" in reason and "lost or changed" in reason
        for reason in violations.values()
    )
    assert 3 in violations
    assert 35 not in violations


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


def _drain_until_idle(fuzzer: Fuzzer) -> None:
    """Step until nothing is queued in the Network.

    Each delivery this test injects produces a reply, which would
    otherwise sit pending and compete (by delivery time, resolved by
    Network's own RNG when tied) with the *next* message this test
    injects -- exactly the kind of nondeterminism a controlled,
    hand-built scenario needs to avoid. Bounded by construction: a
    delivery here produces at most one reply, and that reply produces
    nothing further (see the calling tests), so this always terminates
    long before the cluster-wide tick timer (default first tick at
    t=100) would ever come due and start ticking nodes on its own.
    """
    while fuzzer.cluster.network.pending():
        assert fuzzer.cluster.step()


def _construct_large_then_small_precondition(fuzzer: Fuzzer):
    """Deliver a small AppendEntries, then a larger one, both natively
    through the real Network -- so its own delivery history ends up with
    exactly one stale-and-superseded candidate (the small one) for
    STALE_REDELIVER to find and redeliver a second time. Returns the
    follower wrapper node.
    """
    e1 = LogEntry(term=1, command="a")
    e2 = LogEntry(term=1, command="b")
    e3 = LogEntry(term=1, command="c")
    e4 = LogEntry(term=1, command="d")

    small = AppendEntries(
        term=1, leader_id="0", prev_log_index=0, prev_log_term=0, entries=(e1, e2), leader_commit=0
    )
    large = AppendEntries(
        term=1, leader_id="0", prev_log_index=0, prev_log_term=0,
        entries=(e1, e2, e3, e4), leader_commit=0,
    )

    fuzzer.cluster.route(0, [(1, small)], now=0)
    _drain_until_idle(fuzzer)  # delivers "small" (and its reply); recorded into history
    fuzzer.cluster.route(0, [(1, large)], now=1)
    _drain_until_idle(fuzzer)  # delivers "large" (and its reply); "small" now superseded

    follower = fuzzer.cluster.get_node(1)
    assert follower is not None
    assert follower.storage.last_log_index() == 4
    return follower


def test_stale_redeliver_reproduces_the_diagnosis_scenario_on_unmutated_code() -> None:
    """The diagnosis's large-then-small scenario, this time constructed
    the way STALE_REDELIVER itself would encounter it -- through
    Network's real delivery history and `Fuzzer._do_stale_redeliver` --
    rather than by calling `RaftNode.handle()` twice by hand (see
    `test_mutation_1_stale_shorter_delivery_is_caught_by_new_checker_not_old`
    for that hand-built version). On unmutated code, the real handler's
    "already matches" check makes the redelivery a pure no-op, exactly as
    verified independently before this action was ever wired into the
    fuzzer (see BUGS.md's diagnosis) -- confirmed here end-to-end through
    the actual action machinery.
    """
    fuzzer = Fuzzer(n=2, seed=1, steps=1)
    follower = _construct_large_then_small_precondition(fuzzer)

    history: dict[str, list[LogEntry]] = {follower.node_id: follower.storage.load_log()}
    check_no_spurious_truncation(fuzzer.cluster, history)  # baseline, nothing wrong yet

    entry = fuzzer._do_stale_redeliver(0, forced=None)
    assert not entry.detail.startswith("skipped"), entry.detail
    assert entry.msg_id is not None
    _drain_until_idle(fuzzer)  # actually processes the redelivered message

    assert follower.storage.last_log_index() == 4  # unchanged -- a real no-op
    check_no_spurious_truncation(fuzzer.cluster, history)  # must not raise


def test_stale_redeliver_reproduces_mutation_1_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same construction as the test above, under Mutation 1: the
    redelivered, now-stale message causes the real damage the diagnosis
    found -- e3 and e4 silently discarded -- and the (now value-based)
    check_no_spurious_truncation catches it, entirely through
    STALE_REDELIVER's own machinery rather than a hand-built message pair.
    """
    monkeypatch.setattr(RaftNode, "_handle_append_entries", _mutation_1_always_truncate_on_resend)

    fuzzer = Fuzzer(n=2, seed=1, steps=1)
    follower = _construct_large_then_small_precondition(fuzzer)

    history: dict[str, list[LogEntry]] = {follower.node_id: follower.storage.load_log()}
    check_no_spurious_truncation(fuzzer.cluster, history)  # baseline, nothing wrong yet

    entry = fuzzer._do_stale_redeliver(0, forced=None)
    assert not entry.detail.startswith("skipped"), entry.detail
    assert entry.msg_id is not None
    _drain_until_idle(fuzzer)  # actually processes the redelivered (stale) message

    assert follower.storage.last_log_index() == 2, "e3 and e4 should be silently discarded"

    with pytest.raises(SafetyViolation, match="lost or changed"):
        check_no_spurious_truncation(fuzzer.cluster, history)


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

    DETECTED: 31/50 seeds (was 33/50 before `STALE_REDELIVER` joined
    `DEFAULT_WEIGHTS` -- see the module docstring for why weaving a new
    action into the shared, weighted `self._rng` stream shifts *every*
    seed's subsequent random walk, not just the ones that need it, so a
    small count change here reflects a different-but-equally-valid
    sequence of events, not a regression) -- 30 via check_log_matching
    (the index misalignment this causes reliably produces two nodes
    disagreeing on what's stored at a shared (index, term)), and 1 via
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


def _mutation_5_apply_past_commit_index(self: RaftNode) -> None:
    """Mutation 5: ignore commit_index's safety boundary entirely and
    apply straight to the end of the log -- whatever is currently there,
    committed or not -- instead of stopping at commit_index. Otherwise
    identical to the real `_apply_committed_entries`: still advances
    `last_applied` one index at a time, still skips non-`Command`
    entries. The one thing missing is the one thing that makes applying
    safe at all.
    """
    unsafe_bound = self.storage.last_log_index()
    if self.last_applied >= unsafe_bound:
        return
    log = self.storage.load_log()
    for index in range(self.last_applied + 1, unsafe_bound + 1):
        command = log[index].command
        if isinstance(command, Put | Get | Delete):
            self.state_machine.apply(command)
        self.last_applied = index


def test_mutation_apply_past_commit_index_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation 5: apply everything currently in the log, not just what's
    actually committed -- the one bug this milestone's own new checker,
    `check_state_machine_safety`, exists to catch.

    Why this is unsafe and "out of order" (the example given for this
    positive control) wouldn't be, given how the checker actually works:
    `check_state_machine_safety` compares `node.storage.load_log()[index]`
    -- the log's own content -- against whatever any node has previously
    recorded as applied at that index; it never inspects the state
    machine's own resulting dict. Calling `state_machine.apply()` in the
    wrong order (but still bounded by commit_index) can produce a wrong
    *value* in the KV store, but produces no *log*-content divergence at
    any index for the checker to ever see -- Raft's Log Matching Property
    already guarantees the log itself agrees at any shared, committed
    index. Applying PAST commit_index is different: an entry applied
    before it was actually safe can still be truncated and overwritten
    by a later, honestly elected leader that never saw it as committed
    (Figure 8's whole reason for existing) -- and when that happens,
    whatever next reads that index sees genuinely different log content
    than an earlier, premature application already recorded.

    DETECTED: 8/50 seeds (was 11/50 before CLIENT_REQUEST started
    following `raft.node.NotLeader`'s own `leader_hint` instead of
    omnisciently targeting the actual leader -- see the module docstring
    and the truncate-on-resend mutation's own test for why that
    traffic-pattern change shifts sweep counts without touching either
    the mutation or the checker), always via `check_state_machine_safety`
    -- the first real detection this checker has ever produced (see
    BUGS.md; every prior mention of it in this project was either
    "vacuously satisfied" or a hand-constructed unit test in
    test_invariants.py that never went through real `RaftNode` apply
    logic at all).
    """
    monkeypatch.setattr(RaftNode, "_apply_committed_entries", _mutation_5_apply_past_commit_index)

    violations = _sweep()

    assert len(violations) == 8, f"detection rate shifted: {violations}"
    assert all(
        "applied" in reason and "already applied there" in reason for reason in violations.values()
    )
