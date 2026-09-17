"""Tests for applying committed entries to the replicated state machine:
`RaftNode._apply_committed_entries`'s own contract (exactly one entry at
a time, never skipping, never exceeding commit_index), each of its three
call sites (`tick`, `handle`, `append_command`), backward compatibility
with non-`Command` log entries (the sentinel, `_become_leader`'s no-op,
and the fuzzer's own opaque string commands), and cross-node convergence
to identical final state.
"""

import random

from raft.messages import AppendEntries
from raft.node import RaftNode, Role, raft_node_factory
from raft.sim.cluster import Cluster
from raft.sim.fuzz import Fuzzer
from raft.statemachine import Put
from raft.storage import LogEntry, MemoryStorage

N_NODES = 5
STEPS_PER_SEED = 300
SEED_COUNT = 50


def make_cluster(n: int, seed: int, tick_interval_ms: int = 10) -> Cluster:
    return Cluster(
        n=n, seed=seed, node_factory=raft_node_factory(n, seed), tick_interval_ms=tick_interval_ms
    )


# -- _apply_committed_entries' own contract, tested directly --


def test_apply_advances_last_applied_by_exactly_one_entry_at_a_time() -> None:
    storage = MemoryStorage()
    storage.append_entries(
        [LogEntry(term=1, command=Put(key=str(i), value=i)) for i in range(1, 6)]
    )
    node = RaftNode("0", [], storage, rng=random.Random(1))
    node.commit_index = 5

    applied_order: list[object] = []
    original_apply = node.state_machine.apply

    def spy_apply(command: object) -> object:
        applied_order.append(command)
        return original_apply(command)

    node.state_machine.apply = spy_apply  # type: ignore[method-assign]

    node._apply_committed_entries()

    assert node.last_applied == 5
    # In order, none skipped, none repeated:
    assert applied_order == [Put(key=str(i), value=i) for i in range(1, 6)]
    assert node.state_machine.snapshot() == {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}


def test_apply_only_applies_the_newly_committed_delta_on_repeated_calls() -> None:
    storage = MemoryStorage()
    storage.append_entries(
        [LogEntry(term=1, command=Put(key=str(i), value=i)) for i in range(1, 4)]
    )
    node = RaftNode("0", [], storage, rng=random.Random(1))
    node.commit_index = 2

    node._apply_committed_entries()
    assert node.last_applied == 2
    assert node.state_machine.snapshot() == {"1": 1, "2": 2}

    node.commit_index = 3
    node._apply_committed_entries()
    assert node.last_applied == 3
    assert node.state_machine.snapshot() == {"1": 1, "2": 2, "3": 3}


def test_apply_is_a_no_op_when_nothing_new_is_committed() -> None:
    storage = MemoryStorage()
    storage.append_entries([LogEntry(term=1, command=Put(key="x", value=1))])
    node = RaftNode("0", [], storage, rng=random.Random(1))
    node.commit_index = 1
    node._apply_committed_entries()
    assert node.last_applied == 1

    node._apply_committed_entries()  # nothing new committed

    assert node.last_applied == 1
    assert node.state_machine.snapshot() == {"x": 1}


def test_non_command_entries_are_inert_but_still_advance_last_applied() -> None:
    """The index-0 sentinel (command=None), `_become_leader`'s
    content-free no-op (also command=None), and an opaque payload that
    isn't a real `raft.statemachine.Command` (exactly what the fuzzer's
    own CLIENT_REQUEST action sends -- a plain string, see
    `raft.sim.fuzz`) must not be fed to the state machine, but
    `last_applied` must still advance past them like any other entry.
    This is what keeps this change from breaking every existing test and
    fuzzer run that predates `raft.statemachine`.
    """
    storage = MemoryStorage()
    storage.append_entries(
        [
            LogEntry(term=1, command=None),  # e.g. _become_leader's no-op
            LogEntry(term=1, command="cmd-16"),  # e.g. the fuzzer's CLIENT_REQUEST
            LogEntry(term=1, command=Put(key="x", value=1)),
        ]
    )
    node = RaftNode("0", [], storage, rng=random.Random(1))
    node.commit_index = 3

    node._apply_committed_entries()

    assert node.last_applied == 3
    assert node.state_machine.snapshot() == {"x": 1}


# -- Each of the three call sites, individually --


def test_append_command_applies_immediately_for_a_lone_leader() -> None:
    storage = MemoryStorage()
    node = RaftNode("0", [], storage, rng=random.Random(1))
    node.election_deadline = 0
    node._deadline_initialized = True
    node.tick(now=0)
    assert node.role is Role.LEADER

    node.append_command(Put(key="x", value=1), now=1)

    assert node.commit_index == 1
    assert node.last_applied == 1
    assert node.state_machine.snapshot() == {"x": 1}


def test_tick_applies_after_a_lone_leader_commits_its_own_noop_on_election() -> None:
    storage = MemoryStorage()
    node = RaftNode("0", [], storage, rng=random.Random(1), append_noop_on_election=True)
    node.election_deadline = 0
    node._deadline_initialized = True

    node.tick(now=0)

    assert node.role is Role.LEADER
    assert node.commit_index == 1
    assert node.last_applied == 1  # advanced within this same tick() call
    assert node.state_machine.snapshot() == {}  # the no-op has nothing to apply


def test_handle_applies_after_a_follower_adopts_a_higher_leader_commit() -> None:
    storage = MemoryStorage()
    storage.append_entries([LogEntry(term=1, command=Put(key="x", value=1))])
    follower = RaftNode("1", ["0"], storage, rng=random.Random(1))

    msg = AppendEntries(
        term=1, leader_id="0", prev_log_index=1, prev_log_term=1, entries=(), leader_commit=1
    )
    follower.handle(msg, src="0", now=0)

    assert follower.commit_index == 1
    assert follower.last_applied == 1
    assert follower.state_machine.snapshot() == {"x": 1}


# -- last_applied's own invariant, and cross-node convergence, on a real cluster --


def _current_leader(cluster: Cluster) -> tuple[int, object]:
    for node_id in cluster.node_ids():
        node = cluster.get_node(node_id)
        if node is not None and node.role is Role.LEADER:
            return node_id, node
    raise AssertionError("no leader elected -- test setup is broken")


def _submit(cluster: Cluster, leader_id: int, leader: object, command: object) -> None:
    now = cluster.clock.now()
    produced = leader.append_command(command, now)  # type: ignore[attr-defined]
    cluster.route(leader_id, produced, now)


def test_multiple_nodes_converge_to_identical_state_after_replication() -> None:
    cluster = make_cluster(n=5, seed=1)
    cluster.run(2000)  # elect a stable leader

    leader_id, leader = _current_leader(cluster)

    _submit(cluster, leader_id, leader, Put(key="x", value=1))
    _submit(cluster, leader_id, leader, Put(key="y", value=2))
    _submit(cluster, leader_id, leader, Put(key="x", value=3))

    cluster.run(2000)  # let replication, commitment, and apply all settle

    live_nodes = [cluster.get_node(nid) for nid in cluster.node_ids()]
    live_nodes = [n for n in live_nodes if n is not None]
    assert len(live_nodes) == 5  # no faults injected here -- nothing should be down

    for node in live_nodes:
        assert node.last_applied == node.commit_index

    snapshots = [node.state_machine.snapshot() for node in live_nodes]
    assert all(s == snapshots[0] for s in snapshots), snapshots
    assert snapshots[0] == {"x": 3, "y": 2}


def test_last_applied_never_exceeds_commit_index_across_a_running_cluster() -> None:
    cluster = make_cluster(n=5, seed=1)
    checked = 0
    for _ in range(3000):
        if not cluster.step():
            break
        for node_id in cluster.node_ids():
            node = cluster.get_node(node_id)
            if node is not None:
                assert node.last_applied <= node.commit_index
                checked += 1

    assert checked > 0  # sanity: this actually observed something


def test_last_applied_is_bounded_and_nonzero_across_the_fixed_seed_sweep() -> None:
    """The concrete proof check_state_machine_safety moved from dormant
    to functioning: across the fixed 50-seed sweep, last_applied actually
    advances on real nodes during ordinary fuzzing, never exceeding
    commit_index anywhere.
    """
    seeds_with_progress = 0
    max_last_applied = 0
    live_observations = 0

    for seed in range(SEED_COUNT):
        fuzzer = Fuzzer(n=N_NODES, seed=seed, steps=STEPS_PER_SEED)
        fuzzer.run()  # must not raise -- also covered by test_fuzz_sweep.py

        any_progress = False
        for node_id in fuzzer.cluster.node_ids():
            node = fuzzer.cluster.get_node(node_id)
            if node is None:
                continue
            live_observations += 1
            assert node.last_applied <= node.commit_index, (
                f"seed={seed} node={node_id}: last_applied {node.last_applied} "
                f"exceeds commit_index {node.commit_index}"
            )
            max_last_applied = max(max_last_applied, node.last_applied)
            any_progress = any_progress or node.last_applied > 0
        if any_progress:
            seeds_with_progress += 1

    assert live_observations > 0
    assert seeds_with_progress >= 20, (
        f"only {seeds_with_progress}/{SEED_COUNT} seeds ever advanced last_applied -- "
        f"suspiciously low for a checker meant to actually exercise this now"
    )
    assert max_last_applied > 0, "last_applied never advanced anywhere across the whole sweep"
