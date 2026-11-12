"""RaftNode never copies its whole log on a per-message path.

`Storage.load_log()` copies the entire log, so any per-message caller
makes the cost of every message grow with uptime -- which the benchmark
measured (BUGS.md): throughput fell from about 840 to 165 entries/s as the
log grew to 94,000 entries. RaftNode reads through `entry_at`, `term_at`
and `entries_from` instead. This drives elections, replication, a
partition that leaves divergent logs to be overwritten (the conflict
paths), crashes and restarts, with `load_log` made to fail if called,
then the two conflict searches directly -- now binary searches, checked
against a linear scan.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from raft.messages import AppendEntries, AppendEntriesReply
from raft.node import RaftClusterNode, RaftNode, Role, raft_node_factory
from raft.sim.cluster import Cluster
from raft.statemachine import ClientRequest, Put
from raft.storage import LogEntry, MemoryStorage

N, SEED = 5, 7


def leader_of(cluster: Cluster[RaftClusterNode]) -> int | None:
    found = [
        (n.raft_node.current_term, i)
        for i in cluster.node_ids()
        if (n := cluster.get_node(i)) is not None and n.raft_node.role is Role.LEADER
    ]
    return max(found)[1] if found else None


def run(cluster: Cluster[RaftClusterNode], ms: int) -> None:
    end = cluster.clock.now() + ms
    while cluster.clock.now() < end and cluster.step():
        pass


def write(cluster: Cluster[RaftClusterNode], node_id: int, serial: int) -> None:
    node = cluster.get_node(node_id)
    assert node is not None
    out = node.append_command(
        ClientRequest("c", serial, Put("k", str(serial))), cluster.clock.now()
    )
    if isinstance(out, list):
        cluster.route(node_id, out, cluster.clock.now())


def test_no_full_log_copy_through_elections_conflicts_and_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(self: MemoryStorage) -> list[LogEntry]:
        raise AssertionError("RaftNode copied the whole log")

    monkeypatch.setattr(MemoryStorage, "load_log", forbidden)
    cluster = Cluster(
        n=N,
        seed=SEED,
        node_factory=raft_node_factory(N, SEED, pre_vote=True, append_noop_on_election=True),
    )
    for a in cluster.node_ids():
        for b in cluster.node_ids():
            if a != b:
                cluster.network.set_delay(a, b, 3)
    serial = 0
    run(cluster, 1000)
    old = leader_of(cluster)
    assert old is not None
    for _ in range(50):
        serial += 1
        write(cluster, old, serial)
        run(cluster, 2)

    # Cut the leader off with one follower and keep writing to it: those
    # entries never commit, and the majority side elects a new leader and
    # writes its own -- two divergent logs.
    follower = next(i for i in cluster.node_ids() if i != old)
    minority = {old, follower}
    cluster.partition(minority, set(cluster.node_ids()) - minority)
    for _ in range(60):  # more than the majority side will write, so logs conflict by term
        serial += 1
        write(cluster, old, serial)
        run(cluster, 2)
    run(cluster, 1000)
    new = leader_of(cluster)
    assert new is not None and new not in minority
    for _ in range(10):
        serial += 1
        write(cluster, new, serial)
        run(cluster, 2)

    # Heal: the old side's uncommitted entries are overwritten (conflicts),
    # then a crash and restart catch a node up from scratch of its memory.
    cluster.heal()
    run(cluster, 1000)
    cluster.crash(follower)
    for _ in range(20):
        serial += 1
        write(cluster, new, serial)
        run(cluster, 2)
    cluster.restart(follower)
    run(cluster, 1500)

    lead = leader_of(cluster)
    assert lead is not None
    target = cluster.get_node(lead)
    assert target is not None
    commit = target.raft_node.commit_index
    assert commit >= 50 + 10 + 20
    terms = [target.raft_node.storage.term_at(i) for i in range(commit + 1)]
    for i in cluster.node_ids():
        node = cluster.get_node(i)
        assert node is not None
        assert node.raft_node.commit_index == commit
        assert [node.raft_node.storage.term_at(j) for j in range(commit + 1)] == terms


def node_with_terms(terms: list[int], peers: tuple[str, ...] = ("b",)) -> RaftNode:
    storage = MemoryStorage()
    storage.append_entries([LogEntry(t, f"e{i}") for i, t in enumerate(terms, 1)])
    node = RaftNode("a", list(peers), storage, rng=random.Random(1))
    node.current_term = max(terms, default=0) + 1
    return node


@pytest.fixture
def no_full_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(self: MemoryStorage) -> list[LogEntry]:
        raise AssertionError("RaftNode copied the whole log")

    monkeypatch.setattr(MemoryStorage, "load_log", forbidden)


@pytest.mark.usefixtures("no_full_copies")
def test_follower_term_conflict_reports_the_start_of_the_run() -> None:
    follower = node_with_terms([1, 1, 1, 2, 2, 2])
    msg = AppendEntries(follower.current_term, "b", 5, 3, (), 0)  # it has term 2 at 5
    [(_, reply)] = follower.handle(msg, "b", 0)
    assert isinstance(reply, AppendEntriesReply)
    assert (reply.success, reply.conflict_term, reply.conflict_index) == (False, 2, 4)


@pytest.mark.usefixtures("no_full_copies")
@pytest.mark.parametrize(
    ("conflict_term", "conflict_index", "expected"),
    [(1, 2, 4), (2, 4, 4), (3, 6, 6)],  # holds term 1 (ends at 3), lacks 2, holds 3 (ends at 5)
)
def test_leader_conflict_term_lookup(
    conflict_term: int, conflict_index: int, expected: int
) -> None:
    leader = node_with_terms([1, 1, 1, 3, 3])
    leader.role = Role.LEADER
    leader.next_index = {"b": 6}
    leader.match_index = {"b": 0}
    reply = AppendEntriesReply(
        leader.current_term, False, conflict_index=conflict_index, conflict_term=conflict_term
    )
    [(_, retry)] = leader.handle(reply, "b", 0)
    assert isinstance(retry, AppendEntries)
    assert retry.prev_log_index == expected - 1


@given(
    runs=st.lists(st.integers(1, 30), min_size=0, max_size=6),
    term=st.integers(0, 8),
    data=st.data(),
)
def test_binary_search_matches_a_linear_scan(
    runs: list[int], term: int, data: st.DataObject
) -> None:
    terms = [t for t, length in enumerate(runs, 1) for _ in range(length)]  # non-decreasing
    node = node_with_terms(terms)
    hi = data.draw(st.integers(0, len(terms)))
    linear = next((i for i in range(1, hi + 1) if terms[i - 1] >= term), hi + 1)
    assert node._first_index_with_term_at_least(term, hi) == linear
