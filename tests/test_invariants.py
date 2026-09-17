"""Tests for Raft's safety-property checkers."""

import pytest

from raft.invariants import (
    AppliedEntryHistory,
    CommittedEntry,
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


def test_check_no_spurious_truncation_allows_an_equal_object_swap() -> None:
    """This checker used to compare by object identity and would raise
    here -- see BUGS.md for why that signature was structurally
    unreachable through the simulator (no `LogEntry` is ever cloned) and
    why the checker is value-based now. A different object holding an
    equal value changes nothing a safety property cares about, so this
    must pass.
    """
    cluster = make_cluster(n=3, seed=1)
    history: dict[str, list[LogEntry]] = {}

    node = cluster.get_node(0)
    assert node is not None
    original_entry = LogEntry(term=1, command="a")
    node.storage.append_entries([original_entry])
    check_no_spurious_truncation(cluster, history)  # records the baseline

    replacement = LogEntry(term=1, command="a")
    assert replacement is not original_entry
    assert replacement == original_entry
    node.storage.truncate_from(1)
    node.storage.append_entries([replacement])

    check_no_spurious_truncation(cluster, history)  # must not raise


def test_check_no_spurious_truncation_allows_a_genuine_value_change() -> None:
    """Replacing an entry with a *different* value, at a term conflict, is
    legitimate conflict resolution (Figure 2, rules 3 and 4) -- exactly
    what this checker's one exemption is for."""
    cluster = make_cluster(n=3, seed=1)
    history: dict[str, list[LogEntry]] = {}

    node = cluster.get_node(0)
    assert node is not None
    node.storage.append_entries([LogEntry(term=1, command="a")])
    check_no_spurious_truncation(cluster, history)

    node.storage.truncate_from(1)
    node.storage.append_entries([LogEntry(term=2, command="different")])

    check_no_spurious_truncation(cluster, history)  # must not raise


def test_check_no_spurious_truncation_raises_when_an_entry_is_lost_with_no_term_conflict() -> None:
    """The bug this checker is actually for: a follower's log gets shorter
    (or an entry silently changes) at a point where the term does NOT
    differ -- so there is no Figure 2 justification for having touched it
    at all. This is the shape seeds 3 and 35 hit: a stale, reordered,
    shorter AppendEntries truncates a follower's log past its own range
    and nothing puts the discarded tail back."""
    cluster = make_cluster(n=3, seed=1)
    history: dict[str, list[LogEntry]] = {}

    node = cluster.get_node(0)
    assert node is not None
    node.storage.append_entries(
        [LogEntry(term=1, command="a"), LogEntry(term=1, command="b")]
    )
    check_no_spurious_truncation(cluster, history)  # records the 2-entry baseline

    # Shortened with nothing appended back, and no term conflict anywhere
    # -- index 1 ("a") simply vanishes.
    node.storage.truncate_from(1)

    with pytest.raises(SafetyViolation, match="lost or changed"):
        check_no_spurious_truncation(cluster, history)


def test_check_no_spurious_truncation_raises_on_same_term_different_command() -> None:
    """An entry changing value while its term stays THE SAME at the first
    point of divergence is never justified by Figure 2 -- a term match
    with a different command is itself a Log Matching Property violation,
    not a conflict truncation, so this must raise too, not just an outright
    shrink."""
    cluster = make_cluster(n=3, seed=1)
    history: dict[str, list[LogEntry]] = {}

    node = cluster.get_node(0)
    assert node is not None
    node.storage.append_entries([LogEntry(term=1, command="a")])
    check_no_spurious_truncation(cluster, history)

    node.storage.truncate_from(1)
    node.storage.append_entries([LogEntry(term=1, command="different")])  # same term!

    with pytest.raises(SafetyViolation, match="lost or changed"):
        check_no_spurious_truncation(cluster, history)


def test_check_no_spurious_truncation_watches_followers_not_just_leaders() -> None:
    """Unlike check_leader_append_only, this has no "only while leader"
    exception -- the bug it exists for (a follower's blind receiver) lives
    specifically in a role check_leader_append_only never looks at."""
    cluster = make_cluster(n=3, seed=1)
    history: dict[str, list[LogEntry]] = {}

    node = cluster.get_node(0)
    assert node is not None
    assert node.role is Role.FOLLOWER

    node.storage.append_entries(
        [LogEntry(term=1, command="a"), LogEntry(term=1, command="b")]
    )
    check_no_spurious_truncation(cluster, history)

    node.storage.truncate_from(1)  # shortened, no term conflict, nothing appended back

    with pytest.raises(SafetyViolation, match="lost or changed"):
        check_no_spurious_truncation(cluster, history)


def test_check_no_spurious_truncation_crash_mid_write_is_not_exempted() -> None:
    """Design note (see BUGS.md and this function's own docstring):
    `MemoryStorage.fail_after` can raise between `truncate_from` and the
    `append_entries` Figure 2 expects to follow it, leaving a real node
    with a genuinely shorter log and no term conflict to justify it --
    exactly the shape this checker raises on. Nothing in `Cluster` or
    `Fuzzer` ever arms `fail_after`, so no fuzzer run reaches this state,
    but the checker deliberately does not special-case it anyway: if it
    ever did happen, `Cluster` has no mechanism to mark that node as
    crashed (nothing calls `Cluster.crash()`), so it would stay live with
    an unrecoverable gap in its log -- the exact hazard this function
    exists to catch. This test constructs that half-completed write by
    hand (calling `truncate_from` without a following `append_entries`,
    exactly what a crash between the two would leave behind) and confirms
    it raises rather than being silently trusted.
    """
    cluster = make_cluster(n=3, seed=1)
    history: dict[str, list[LogEntry]] = {}

    node = cluster.get_node(0)
    assert node is not None
    node.storage.append_entries(
        [LogEntry(term=1, command="a"), LogEntry(term=1, command="b")]
    )
    check_no_spurious_truncation(cluster, history)

    # truncate_from succeeds; the compensating append_entries never runs,
    # exactly as if StorageError had fired in between.
    node.storage.truncate_from(1)

    with pytest.raises(SafetyViolation, match="lost or changed"):
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
    check_leader_completeness(cluster, {})
    check_state_machine_safety(cluster, AppliedEntryHistory())


# -- Leader Completeness --


def test_check_leader_completeness_passes_on_a_healthy_cluster() -> None:
    # append_noop_on_election=True so every elected leader actually has
    # something at its own current term to commit -- without it, nothing
    # in an election-only run ever advances commit_index at all (see
    # raft.node's module docstring), and the test below would exercise
    # nothing.
    cluster = Cluster(
        n=5,
        seed=1,
        node_factory=raft_node_factory(5, 1, append_noop_on_election=True),
        tick_interval_ms=10,
    )
    history: dict[int, CommittedEntry] = {}

    for _ in range(RUN_STEPS):
        if not cluster.step():
            break
        check_leader_completeness(cluster, history)  # must never raise

    assert history  # sanity: something was actually committed and checked


def test_check_leader_completeness_raises_when_a_later_leader_lacks_a_committed_entry() -> None:
    cluster = make_cluster(n=3, seed=1)
    history: dict[int, CommittedEntry] = {}

    committed_node = cluster.get_node(0)
    assert committed_node is not None
    committed_node.storage.append_entries([LogEntry(term=1, command="a")])
    committed_node.raft_node.current_term = 1
    committed_node.raft_node.commit_index = 1
    check_leader_completeness(cluster, history)  # records index 1 as committed at term 1
    assert history == {1: CommittedEntry(term=1, established_at_term=1)}

    later_leader = cluster.get_node(1)
    assert later_leader is not None
    later_leader.raft_node.role = Role.LEADER
    later_leader.raft_node.current_term = 5  # later than the term index 1 was committed at
    # later_leader's log is still empty -- missing the committed entry entirely.

    with pytest.raises(SafetyViolation, match="missing entry"):
        check_leader_completeness(cluster, history)


def test_check_leader_completeness_ignores_a_leader_at_or_below_the_committed_term() -> None:
    """Only a leader from a LATER term is bound by an earlier commitment --
    Figure 8's restriction is about committing forward, not a constraint
    on what a same-or-earlier-term node's log looks like."""
    cluster = make_cluster(n=3, seed=1)
    history: dict[int, CommittedEntry] = {}

    committed_node = cluster.get_node(0)
    assert committed_node is not None
    committed_node.storage.append_entries([LogEntry(term=3, command="a")])
    committed_node.raft_node.current_term = 3
    committed_node.raft_node.commit_index = 1
    check_leader_completeness(cluster, history)
    assert history == {1: CommittedEntry(term=3, established_at_term=3)}

    leader = cluster.get_node(1)
    assert leader is not None
    leader.raft_node.role = Role.LEADER
    leader.raft_node.current_term = 3  # same term as the commitment, not later
    # leader's log is empty -- would be missing it, but isn't bound by it.

    check_leader_completeness(cluster, history)  # must not raise


def test_check_leader_completeness_ignores_a_leader_that_predates_a_ridden_along_entry() -> None:
    """A leader elected between an entry's own term and the later term
    whose commit decision actually rode it along is not bound by it --
    it was never part of the guarantee that made the entry safe.

    This is the regression the term/established_at_term split in
    `CommittedEntry` exists for: gating on the entry's own term (3, here)
    instead of the term whose decision actually committed it (5) would
    wrongly flag `leader`, elected at term 4 -- honestly, before index 1
    was ever protected -- as having "lost" something it was never
    guaranteed to have in the first place.
    """
    cluster = make_cluster(n=3, seed=1)
    history: dict[int, CommittedEntry] = {}

    committed_node = cluster.get_node(0)
    assert committed_node is not None
    # index 1 (term 3) only becomes safe once index 2 (term 5, THIS node's
    # current term) commits alongside it -- Figure 8's rule, riding an
    # earlier-term entry along beneath a current-term one.
    committed_node.storage.append_entries(
        [LogEntry(term=3, command="old"), LogEntry(term=5, command="new")]
    )
    committed_node.raft_node.current_term = 5
    committed_node.raft_node.commit_index = 2
    check_leader_completeness(cluster, history)
    assert history == {
        1: CommittedEntry(term=3, established_at_term=5),
        2: CommittedEntry(term=5, established_at_term=5),
    }

    leader = cluster.get_node(1)
    assert leader is not None
    leader.raft_node.role = Role.LEADER
    leader.raft_node.current_term = 4  # between the entry's term (3) and term 5
    # leader's log is empty -- would be "missing" index 1 under the wrong gate.

    check_leader_completeness(cluster, history)  # must not raise


# -- State Machine Safety --


def test_check_state_machine_safety_passes_when_nothing_has_been_applied_yet() -> None:
    """Nothing in this codebase advances last_applied yet (see raft.node's
    module docstring) -- so on a real, healthy run this checker currently
    has nothing to ever observe. Implemented now regardless, so it's
    ready the moment something starts advancing it."""
    cluster = make_cluster(n=5, seed=1)
    cluster.run(RUN_STEPS)
    history = AppliedEntryHistory()

    check_state_machine_safety(cluster, history)  # must not raise

    assert history.applied == {}


def test_check_state_machine_safety_raises_when_two_nodes_apply_different_entries() -> None:
    cluster = make_cluster(n=3, seed=1)
    history = AppliedEntryHistory()

    node_a = cluster.get_node(0)
    node_b = cluster.get_node(1)
    assert node_a is not None
    assert node_b is not None

    node_a.storage.append_entries([LogEntry(term=1, command="a")])
    node_a.raft_node.last_applied = 1
    check_state_machine_safety(cluster, history)
    assert history.applied == {1: LogEntry(term=1, command="a")}

    node_b.storage.append_entries([LogEntry(term=2, command="DIFFERENT")])
    node_b.raft_node.last_applied = 1

    with pytest.raises(SafetyViolation, match="applied"):
        check_state_machine_safety(cluster, history)


def test_check_state_machine_safety_allows_the_same_node_reapplying_unchanged() -> None:
    cluster = make_cluster(n=3, seed=1)
    history = AppliedEntryHistory()

    node = cluster.get_node(0)
    assert node is not None
    node.storage.append_entries([LogEntry(term=1, command="a")])
    node.raft_node.last_applied = 1
    check_state_machine_safety(cluster, history)

    check_state_machine_safety(cluster, history)  # nothing new -- must not raise


def test_check_state_machine_safety_tracks_each_nodes_progress_independently() -> None:
    """A slower node's own catch-up range must still get checked against
    what a faster node already recorded -- not silently skipped because a
    shared high-water mark had already moved past its target index. This
    is exactly why AppliedEntryHistory keeps a per-node cursor rather than
    one shared mark the way check_leader_completeness's history does."""
    cluster = make_cluster(n=3, seed=1)
    history = AppliedEntryHistory()

    fast_node = cluster.get_node(0)
    slow_node = cluster.get_node(1)
    assert fast_node is not None
    assert slow_node is not None

    fast_node.storage.append_entries(
        [LogEntry(term=1, command="a"), LogEntry(term=1, command="b")]
    )
    fast_node.raft_node.last_applied = 2
    check_state_machine_safety(cluster, history)
    assert history.applied == {
        1: LogEntry(term=1, command="a"),
        2: LogEntry(term=1, command="b"),
    }

    # The slow node only now catches up to index 1 -- with a CONFLICTING value.
    slow_node.storage.append_entries([LogEntry(term=1, command="DIFFERENT")])
    slow_node.raft_node.last_applied = 1

    with pytest.raises(SafetyViolation, match="applied"):
        check_state_machine_safety(cluster, history)
