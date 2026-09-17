"""Tests for the Fuzzer: reproducibility, the crash cap, and action coverage.

`SafetyViolation` was not raised by any run used to calibrate these
tests (dozens of seeds, 500-1000 steps each) -- a promising sign for
RaftNode's election implementation, though these tests are about the
Fuzzer's own mechanics, not a substitute for it never finding anything.
"""

import pytest

import raft.sim.fuzz as fuzz_module
from raft.invariants import SafetyViolation
from raft.node import raft_node_factory
from raft.sim.cluster import Cluster
from raft.sim.fuzz import Action, Fuzzer, TraceEntry

STEPS = 1000


def _effectful(fuzzer: Fuzzer, action: Action) -> list[str]:
    """Details of every occurrence of `action` that actually did something."""
    return [
        entry.detail
        for entry in fuzzer.trace
        if entry.action is action and not entry.detail.startswith("skipped")
    ]


def test_fixed_seed_produces_an_identical_trace_across_two_runs() -> None:
    def run_once() -> list:
        fuzzer = Fuzzer(n=5, seed=42, steps=STEPS)
        fuzzer.run()
        return fuzzer.trace

    assert run_once() == run_once()


def test_zero_fault_weight_behaves_like_a_plain_cluster_run() -> None:
    fuzzer = Fuzzer(n=5, seed=7, steps=STEPS, weights={Action.STEP: 1.0})
    fuzzer.run()

    plain = Cluster(n=5, seed=7, node_factory=raft_node_factory(5, 7))
    plain.run(STEPS)

    # Every drawn action must have been STEP -- nothing else had any weight.
    assert {entry.action for entry in fuzzer.trace} == {Action.STEP}

    for node_id in plain.node_ids():
        fuzzed_node = fuzzer.cluster.get_node(node_id)
        plain_node = plain.get_node(node_id)
        assert (fuzzed_node is None) == (plain_node is None)
        if fuzzed_node is not None and plain_node is not None:
            assert fuzzed_node.role is plain_node.role
            assert fuzzed_node.current_term == plain_node.current_term
            assert fuzzed_node.leader_id == plain_node.leader_id

    assert fuzzer.cluster.clock.now() == plain.clock.now()


def test_crash_cap_is_never_exceeded() -> None:
    n = 7
    max_crashed = (n - 1) // 2
    fuzzer = Fuzzer(n=n, seed=3, steps=STEPS)
    fuzzer.run()

    # Replay the trace's crash/restart entries to reconstruct how many
    # nodes were down at every point in the run, not just at the end.
    crashed_count = 0
    for entry in fuzzer.trace:
        if entry.action is Action.CRASH and not entry.detail.startswith("skipped"):
            crashed_count += 1
        elif entry.action is Action.RESTART and not entry.detail.startswith("skipped"):
            crashed_count -= 1
        assert 0 <= crashed_count <= max_crashed, (
            f"crash cap violated at step {entry.step}: {crashed_count} down, cap is {max_crashed}"
        )

    # The cap must have actually mattered -- otherwise this run never got
    # close enough to be a meaningful check of it.
    assert len(_effectful(fuzzer, Action.CRASH)) > 0


def test_every_action_type_actually_appears_across_a_sample_of_seeds() -> None:
    # A fuzzer that never partitions (or never does anything else) would
    # pass every other test in this file trivially; this is what actually
    # rules that out.
    seen_effectful: set[Action] = set()

    for seed in range(20):
        fuzzer = Fuzzer(n=5, seed=seed, steps=STEPS)
        fuzzer.run()
        for entry in fuzzer.trace:
            if not entry.detail.startswith("skipped"):
                seen_effectful.add(entry.action)

    assert seen_effectful == set(Action)


def test_a_safety_violation_carries_seed_step_and_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_violates(cluster: object) -> None:
        raise SafetyViolation("synthetic violation for testing", seed=-1, step=-1)

    monkeypatch.setattr(fuzz_module, "check_election_safety", always_violates)

    fuzzer = Fuzzer(n=3, seed=5, steps=STEPS)
    with pytest.raises(SafetyViolation) as exc_info:
        fuzzer.run()

    violation = exc_info.value
    assert violation.seed == -1  # untouched -- the checker's own seed report
    assert violation.step == 0  # overwritten with the fuzzer's own step index
    assert violation.trace == tuple(fuzzer.trace)
    assert len(fuzzer.trace) == 1  # run() stopped at the first violation


def test_constructor_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="n must be"):
        Fuzzer(n=0, seed=1, steps=10)
    with pytest.raises(ValueError, match="steps must be"):
        Fuzzer(n=3, seed=1, steps=-1)
    with pytest.raises(ValueError, match="positive weight"):
        Fuzzer(n=3, seed=1, steps=10, weights={Action.STEP: 0.0})


def test_crash_and_partition_skip_when_structurally_impossible() -> None:
    # n=1 can never form a partition (needs 2+ nodes) and can never crash
    # anyone (cap is 0) -- exercising both defensive skip paths directly.
    solo = Fuzzer(n=1, seed=1, steps=1)
    assert solo._do_partition(0, forced=None).detail.startswith("skipped")
    assert solo._do_crash(0, forced=None).detail.startswith("skipped")

    # A cluster with every node already crashed (bypassing the fuzzer's
    # own cap-respecting _do_crash) has no live node left to crash either,
    # nor anything scheduled at all -- so advance_clock takes its
    # idle-fallback path rather than bounding itself to a next event.
    cluster = Fuzzer(n=3, seed=1, steps=1)
    for node_id in cluster.cluster.node_ids():
        cluster.cluster.crash(node_id)
    assert cluster._do_crash(0, forced=None).detail == "skipped (no live node to crash)"
    assert cluster.cluster.next_event_time() is None
    assert cluster._do_advance_clock(0, forced=None).detail.startswith("advanced clock by")


def test_apply_client_request_skips_a_recorded_leader_that_has_since_crashed() -> None:
    # Only reachable on a forced (replay) path: a shrunk candidate can
    # legitimately no longer share the original run's crash history, so
    # a recorded leader id that was alive when this entry was captured
    # may not be by the time it's replayed.
    fuzzer = Fuzzer(n=3, seed=1, steps=1)
    fuzzer.cluster.crash(0)

    assert fuzzer._apply_client_request(0, "cmd") == "skipped (recorded leader is no longer alive)"


# -- replay() and shrink() --
#
# RaftNode's election implementation never actually violates Election
# Safety in any run calibrated for this file, so these tests can't rely
# on a naturally-occurring failure to shrink or replay. Instead they
# monkeypatch check_election_safety with a synthetic, easily-triggered
# condition -- "node 0 is down" -- tied to real, observable cluster
# state (so a real crash genuinely causes it, not a canned string match),
# giving full control over exactly when a "violation" fires without
# depending on RaftNode having a bug.


def _node_down_check(node_id: int):
    def check(cluster: Cluster) -> None:
        if cluster.get_node(node_id) is None:
            raise SafetyViolation(
                f"node {node_id} is down", seed=cluster.seed, step=cluster.step_count
            )

    return check


def test_shrink_reduces_a_padded_violation_to_just_the_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fuzz_module, "check_election_safety", _node_down_check(0))

    fuzzer = Fuzzer(n=5, seed=3, steps=2000)
    with pytest.raises(SafetyViolation) as exc_info:
        fuzzer.run()
    padded_trace = list(exc_info.value.trace)
    assert len(padded_trace) > 1  # confirms there was real padding to strip away

    shrunk = fuzzer.shrink(padded_trace)

    assert len(shrunk) == 1
    assert shrunk[0].action is Action.CRASH
    assert shrunk[0].node_id == 0
    # The shrunk trace must still genuinely reproduce the same violation,
    # not just look plausible.
    with pytest.raises(SafetyViolation, match="node 0 is down"):
        fuzzer.replay(shrunk)


def test_shrink_finds_the_true_minimum_when_two_actions_are_both_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def both_down(cluster: Cluster) -> None:
        if cluster.get_node(0) is None and cluster.get_node(1) is None:
            raise SafetyViolation(
                "nodes 0 and 1 both down", seed=cluster.seed, step=cluster.step_count
            )

    monkeypatch.setattr(fuzz_module, "check_election_safety", both_down)

    fuzzer = Fuzzer(n=5, seed=11, steps=5000)
    with pytest.raises(SafetyViolation) as exc_info:
        fuzzer.run()
    padded_trace = list(exc_info.value.trace)

    shrunk = fuzzer.shrink(padded_trace)

    assert len(shrunk) == 2  # neither crash alone reproduces this one
    crashed_ids = {entry.node_id for entry in shrunk}
    assert crashed_ids == {0, 1}


def test_shrinking_a_passing_trace_raises_rather_than_returning_silently() -> None:
    fuzzer = Fuzzer(n=3, seed=1, steps=0)
    passing_trace = [TraceEntry(step=i, action=Action.STEP, detail="stepped") for i in range(5)]

    with pytest.raises(ValueError, match="does not reproduce"):
        fuzzer.shrink(passing_trace)


def test_replay_of_a_recorded_failing_trace_reproduces_the_original_failure_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fuzz_module, "check_election_safety", _node_down_check(0))

    fuzzer = Fuzzer(n=5, seed=3, steps=2000)
    with pytest.raises(SafetyViolation) as original:
        fuzzer.run()
    original_trace = list(fuzzer.trace)

    with pytest.raises(SafetyViolation) as replayed:
        fuzzer.replay(original_trace)

    assert replayed.value.reason == original.value.reason
    assert replayed.value.step == original.value.step
    assert fuzzer.trace == original_trace


def test_replay_is_idempotent() -> None:
    # No monkeypatching needed here: a plain, non-violating run is enough
    # to show replaying the same trace twice lands on the same result.
    fuzzer = Fuzzer(n=5, seed=7, steps=STEPS)
    fuzzer.run()
    trace = list(fuzzer.trace)

    fuzzer.replay(trace)
    first_trace = list(fuzzer.trace)
    first_terms = [
        node.current_term if (node := fuzzer.cluster.get_node(i)) else None
        for i in fuzzer.cluster.node_ids()
    ]
    first_clock = fuzzer.cluster.clock.now()

    fuzzer.replay(trace)
    second_trace = list(fuzzer.trace)
    second_terms = [
        node.current_term if (node := fuzzer.cluster.get_node(i)) else None
        for i in fuzzer.cluster.node_ids()
    ]
    second_clock = fuzzer.cluster.clock.now()

    assert first_trace == second_trace == trace
    assert first_terms == second_terms
    assert first_clock == second_clock
