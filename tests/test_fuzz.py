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
from raft.sim.fuzz import Action, Fuzzer

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
    assert solo._do_partition(0).startswith("skipped")
    assert solo._do_crash().startswith("skipped")

    # A cluster with every node already crashed (bypassing the fuzzer's
    # own cap-respecting _do_crash) has no live node left to crash either,
    # nor anything scheduled at all -- so advance_clock takes its
    # idle-fallback path rather than bounding itself to a next event.
    cluster = Fuzzer(n=3, seed=1, steps=1)
    for node_id in cluster.cluster.node_ids():
        cluster.cluster.crash(node_id)
    assert cluster._do_crash() == "skipped (no live node to crash)"
    assert cluster.cluster.next_event_time() is None
    assert cluster._do_advance_clock().startswith("advanced clock by")
