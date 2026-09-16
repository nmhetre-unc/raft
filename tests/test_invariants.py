"""Tests for Raft's safety-property checkers."""

import pytest

from raft.invariants import (
    SafetyViolation,
    check_election_safety,
    check_leader_append_only,
    check_leader_completeness,
    check_log_matching,
    check_no_spurious_truncation,
    check_state_machine_safety,
)
from raft.node import Role, raft_node_factory
from raft.sim.cluster import Cluster
from raft.storage import LogEntry

RUN_STEPS = 3000  # see tests/test_election.py -- same calibrated budget


def make_cluster(n: int, seed: int, tick_interval_ms: int = 10) -> Cluster:
    return Cluster(
        n=n,
        seed=seed,
        node_factory=raft_node_factory(n, seed),
        tick_interval_ms=tick_interval_ms,
    )


# -- Election Safety --


def test_check_election_safety_passes_on_a_healthy_cluster() -> None:
    cluster = make_cluster(n=5, seed=1)
    cluster.run(RUN_STEPS)

    check_election_safety(cluster)  # must not raise


def test_check_election_safety_raises_on_two_leaders_in_one_term() -> None:
    cluster = make_cluster(n=3, seed=1)
    node_a = cluster.get_node(0)
    node_b = cluster.get_node(1)
    assert node_a is not None
    assert node_b is not None

    node_a.raft_node.role = Role.LEADER
    node_a.raft_node.current_term = 7
    node_b.raft_node.role = Role.LEADER
    node_b.raft_node.current_term = 7

    with pytest.raises(SafetyViolation) as exc_info:
        check_election_safety(cluster)

    assert exc_info.value.seed == cluster.seed
    assert exc_info.value.step == cluster.step_count


# -- Leader Append-Only --


def test_check_leader_append_only_passes_on_a_healthy_cluster() -> None:
    cluster = make_cluster(n=5, seed=1)
    history: dict[str, list[LogEntry]] = {}

    for _ in range(RUN_STEPS):
        if not cluster.step():
            break
        check_leader_append_only(cluster, history)  # must never raise


def test_check_leader_append_only_raises_when_a_leaders_log_shrinks() -> None:
    cluster = make_cluster(n=3, seed=1)
    history: dict[str, list[LogEntry]] = {}

    node = cluster.get_node(0)
    assert node is not None
    node.raft_node.role = Role.LEADER
    node.storage.append_entries([LogEntry(term=1, command="a"), LogEntry(term=1, command="b")])
    check_leader_append_only(cluster, history)  # records the 3-entry (incl. sentinel) baseline

    node.storage.truncate_from(2)  # shrink it -- must never happen to a real leader's own log

    with pytest.raises(SafetyViolation, match="shrank"):
        check_leader_append_only(cluster, history)


def test_check_leader_append_only_raises_when_a_leader_rewrites_an_entry() -> None:
    cluster = make_cluster(n=3, seed=1)
    history: dict[str, list[LogEntry]] = {}

    node = cluster.get_node(0)
    assert node is not None
    node.raft_node.role = Role.LEADER
    node.storage.append_entries([LogEntry(term=1, command="a")])
    check_leader_append_only(cluster, history)

    # Same length, but the existing entry's content changed underneath it.
    node.storage.truncate_from(1)
    node.storage.append_entries([LogEntry(term=1, command="different")])

    with pytest.raises(SafetyViolation, match="rewrote"):
        check_leader_append_only(cluster, history)


def test_check_leader_append_only_resets_baseline_when_no_longer_leader() -> None:
    """A follower being truncated by a new leader is legitimate, not a violation.

    Without resetting the baseline on step-down, a node that steps down,
    is truncated (correctly, as a follower), and later wins another
    election would be wrongly flagged as having violated append-only --
    even though the truncation happened while it held no leadership to
    violate. This is exactly the scenario the reset guards against.
    """
    cluster = make_cluster(n=3, seed=1)
    history: dict[str, list[LogEntry]] = {}

    node = cluster.get_node(0)
    assert node is not None
    node.raft_node.role = Role.LEADER
    node.storage.append_entries([LogEntry(term=1, command="a"), LogEntry(term=1, command="b")])
    check_leader_append_only(cluster, history)

    node.raft_node.role = Role.FOLLOWER
    node.storage.truncate_from(1)  # legitimate as a follower
    check_leader_append_only(cluster, history)  # must not raise

    node.raft_node.role = Role.LEADER  # wins a later election with the shorter log
    check_leader_append_only(cluster, history)  # must not raise: fresh baseline


# -- No Spurious Truncation --


def test_check_no_spurious_truncation_passes_on_a_healthy_cluster() -> None:
    cluster = make_cluster(n=5, seed=1)
    history: dict[str, list[LogEntry]] = {}

    for _ in range(RUN_STEPS):
        if not cluster.step():
            break
        check_no_spurious_truncation(cluster, history)  # must never raise


def test_check_no_spurious_truncation_raises_when_an_equal_object_is_swapped_in() -> None:
    cluster = make_cluster(n=3, seed=1)
    history: dict[str, list[LogEntry]] = {}

    node = cluster.get_node(0)
    assert node is not None
    original_entry = LogEntry(term=1, command="a")
    node.storage.append_entries([original_entry])
    check_no_spurious_truncation(cluster, history)  # records the baseline

    # A *different* LogEntry object, but value-identical to what's already
    # there -- exactly what a blind truncate-and-reappend of an
    # already-matching resend would produce.
    replacement = LogEntry(term=1, command="a")
    assert replacement is not original_entry
    assert replacement == original_entry
    node.storage.truncate_from(1)
    node.storage.append_entries([replacement])

    with pytest.raises(SafetyViolation, match="torn down and rebuilt"):
        check_no_spurious_truncation(cluster, history)


def test_check_no_spurious_truncation_allows_a_genuine_value_change() -> None:
    """Replacing an entry with a *different* value is legitimate conflict
    resolution (Figure 2, step 3) -- not what this checker watches for."""
    cluster = make_cluster(n=3, seed=1)
    history: dict[str, list[LogEntry]] = {}

    node = cluster.get_node(0)
    assert node is not None
    node.storage.append_entries([LogEntry(term=1, command="a")])
    check_no_spurious_truncation(cluster, history)

    node.storage.truncate_from(1)
    node.storage.append_entries([LogEntry(term=2, command="different")])

    check_no_spurious_truncation(cluster, history)  # must not raise


def test_check_no_spurious_truncation_watches_followers_not_just_leaders() -> None:
    """Unlike check_leader_append_only, this has no "only while leader"
    exception -- the bug it exists for (a follower's blind receiver) lives
    specifically in a role check_leader_append_only never looks at."""
    cluster = make_cluster(n=3, seed=1)
    history: dict[str, list[LogEntry]] = {}

    node = cluster.get_node(0)
    assert node is not None
    assert node.role is Role.FOLLOWER

    original_entry = LogEntry(term=1, command="a")
    node.storage.append_entries([original_entry])
    check_no_spurious_truncation(cluster, history)

    replacement = LogEntry(term=1, command="a")
    node.storage.truncate_from(1)
    node.storage.append_entries([replacement])

    with pytest.raises(SafetyViolation, match="torn down and rebuilt"):
        check_no_spurious_truncation(cluster, history)


# -- Log Matching --


def test_check_log_matching_passes_on_a_healthy_cluster() -> None:
    cluster = make_cluster(n=5, seed=1)
    cluster.run(RUN_STEPS)

    check_log_matching(cluster)  # must not raise


def test_check_log_matching_allows_differing_terms_at_the_same_index() -> None:
    """Two logs disagreeing on the term at an index is not itself a violation.

    The property only fires when the terms *match*; a term mismatch is
    exactly what you'd expect from an in-progress election and must not
    be flagged.
    """
    cluster = make_cluster(n=2, seed=1)
    node_a = cluster.get_node(0)
    node_b = cluster.get_node(1)
    assert node_a is not None
    assert node_b is not None

    node_a.storage.append_entries([LogEntry(term=1, command="x")])
    node_b.storage.append_entries([LogEntry(term=2, command="y")])

    check_log_matching(cluster)  # must not raise


def test_check_log_matching_raises_when_shared_index_and_term_disagree_earlier() -> None:
    cluster = make_cluster(n=2, seed=1)
    node_a = cluster.get_node(0)
    node_b = cluster.get_node(1)
    assert node_a is not None
    assert node_b is not None

    node_a.storage.append_entries(
        [LogEntry(term=1, command="x"), LogEntry(term=2, command="agree")]
    )
    node_b.storage.append_entries(
        [LogEntry(term=1, command="DIFFERENT"), LogEntry(term=2, command="agree")]
    )
    # Both logs agree at index 2 (term=2), which should imply identical
    # logs through index 2 -- but they disagree at index 1.

    with pytest.raises(SafetyViolation, match="differ"):
        check_log_matching(cluster)


# -- Crashed nodes are skipped, not treated as a violation or an error --


def test_checkers_skip_crashed_nodes_without_erroring() -> None:
    cluster = make_cluster(n=5, seed=1)
    cluster.run(RUN_STEPS)
    cluster.crash(0)

    check_election_safety(cluster)
    check_log_matching(cluster)
    check_leader_append_only(cluster, {})
    check_no_spurious_truncation(cluster, {})


# -- Not yet implemented --


def test_check_leader_completeness_is_not_yet_implemented() -> None:
    cluster = make_cluster(n=3, seed=1)
    with pytest.raises(NotImplementedError):
        check_leader_completeness(cluster)


def test_check_state_machine_safety_is_not_yet_implemented() -> None:
    cluster = make_cluster(n=3, seed=1)
    with pytest.raises(NotImplementedError):
        check_state_machine_safety(cluster)
