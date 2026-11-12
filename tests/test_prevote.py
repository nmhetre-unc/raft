"""PreVote (Ongaro's thesis, section 9.6): scenario tests in the simulator,
then unit tests for each grant/reject rule."""

from __future__ import annotations

import random

import pytest

from raft.messages import AppendEntries, PreVote, PreVoteReply, RequestVote
from raft.node import RaftClusterNode, RaftNode, Role, raft_node_factory
from raft.sim.cluster import Cluster
from raft.storage import LogEntry, MemoryStorage

SEED = 11
N = 5

# -- Scenarios --


def make_cluster(pre_vote: bool) -> Cluster[RaftClusterNode]:
    return Cluster(
        n=N,
        seed=SEED,
        node_factory=raft_node_factory(N, SEED, append_noop_on_election=True, pre_vote=pre_vote),
    )


def raft(cluster: Cluster[RaftClusterNode], node_id: int) -> RaftNode:
    node = cluster.get_node(node_id)
    assert node is not None
    return node.raft_node


def leaders(cluster: Cluster[RaftClusterNode]) -> list[int]:
    return [
        i
        for i in cluster.node_ids()
        if cluster.get_node(i) is not None and raft(cluster, i).role is Role.LEADER
    ]


def run_until(cluster: Cluster[RaftClusterNode], t: int) -> None:
    while cluster.clock.now() < t:
        assert cluster.step()


def stable_leader(cluster: Cluster[RaftClusterNode]) -> int:
    for _ in range(100_000):
        found = leaders(cluster)
        if len(found) == 1 and raft(cluster, found[0]).commit_index > 0:
            return found[0]
        cluster.step()
    raise AssertionError("no leader")


@pytest.mark.parametrize("pre_vote", [True, False], ids=["prevote", "control-without-prevote"])
def test_partitioned_node_rejoining_does_not_disrupt_a_stable_leader(pre_vote: bool) -> None:
    cluster = make_cluster(pre_vote)
    leader = stable_leader(cluster)
    term = raft(cluster, leader).current_term
    loner = next(i for i in cluster.node_ids() if i != leader)

    cluster.partition({loner}, set(cluster.node_ids()) - {loner})
    run_until(cluster, cluster.clock.now() + 3000)  # ten or more election timeouts
    cluster.heal()
    run_until(cluster, cluster.clock.now() + 2000)

    if pre_vote:
        assert raft(cluster, loner).current_term == term  # never inflated while cut off
        assert leaders(cluster) == [leader]
        assert raft(cluster, leader).current_term == term  # the term never changed
        assert raft(cluster, loner).role is Role.FOLLOWER
    else:
        # The control: without PreVote the same run does disrupt -- the
        # loner comes back with an inflated term and the leader steps down.
        assert raft(cluster, leader).current_term > term


@pytest.mark.parametrize("pre_vote", [True, False], ids=["prevote", "control-without-prevote"])
def test_elections_still_succeed_when_the_leader_is_dead(pre_vote: bool) -> None:
    cluster = make_cluster(pre_vote)
    old = stable_leader(cluster)
    term = raft(cluster, old).current_term
    crashed_at = cluster.clock.now()

    cluster.crash(old)
    new = stable_leader(cluster)

    assert new != old
    assert raft(cluster, new).current_term > term
    # One pre-vote round trip costs nothing measurable next to an
    # election timeout: both modes elect within a few timeouts.
    assert cluster.clock.now() - crashed_at < 2000
    for i in cluster.node_ids():
        if i != old:
            assert raft(cluster, i).current_term == raft(cluster, new).current_term


def test_pre_vote_cluster_survives_repeated_leader_crashes() -> None:
    cluster = make_cluster(pre_vote=True)
    down: list[int] = []
    for _ in range(4):
        leader = stable_leader(cluster)
        cluster.crash(leader)
        down.append(leader)
        if len(down) > 2:  # keep a majority alive: restart the oldest crash
            cluster.restart(down.pop(0))
    stable_leader(cluster)


# -- Unit tests --


def node(
    peers: tuple[str, ...] = ("b", "c"),
    entries: tuple[int, ...] = (),
    term: int = 0,
) -> RaftNode:
    storage = MemoryStorage()
    storage.append_entries([LogEntry(t, f"e{i}") for i, t in enumerate(entries, 1)])
    n = RaftNode("a", list(peers), storage, rng=random.Random(1), pre_vote=True)
    n.current_term = term
    return n


def pre_vote(term: int, last_log_index: int = 0, last_log_term: int = 0) -> PreVote:
    return PreVote(term, "b", last_log_index, last_log_term)


def test_timeout_asks_for_pre_votes_without_changing_anything() -> None:
    n = node(entries=(1, 2), term=2)
    n.tick(0)  # draws the first deadline
    out = n.tick(n.election_deadline)
    assert out == [(p, PreVote(3, "a", 2, 2)) for p in ("b", "c")]
    assert (n.current_term, n.voted_for, n.role, n.pre_voting) == (2, None, Role.FOLLOWER, True)


def test_single_node_cluster_goes_straight_to_a_real_election() -> None:
    n = node(peers=())
    n.tick(0)
    n.tick(n.election_deadline)
    assert (n.role, n.current_term, n.pre_voting) == (Role.LEADER, 1, False)


def test_majority_of_pre_votes_starts_a_real_election() -> None:
    n = node(peers=("b", "c", "d", "e"), term=4)
    n.tick(0)
    n.tick(n.election_deadline)
    assert n.handle(PreVoteReply(4, vote_granted=True), "b", 1) == []
    assert n.handle(PreVoteReply(4, vote_granted=False), "c", 1) == []  # refusals don't count
    out = n.handle(PreVoteReply(4, vote_granted=True), "d", 1)  # a, b, d: 3 of 5
    assert (n.role, n.current_term, n.voted_for, n.pre_voting) == (Role.CANDIDATE, 5, "a", False)
    assert {dst for dst, _ in out} == {"b", "c", "d", "e"}
    assert all(isinstance(msg, RequestVote) and msg.term == 5 for _, msg in out)


def test_late_pre_vote_grants_outside_a_round_are_ignored() -> None:
    n = node(term=1)
    assert n.handle(PreVoteReply(1, vote_granted=True), "b", 0) == []
    assert n.handle(PreVoteReply(1, vote_granted=True), "c", 0) == []
    assert (n.role, n.current_term) == (Role.FOLLOWER, 1)


def test_reply_with_a_newer_term_ends_the_round() -> None:
    n = node(term=1)
    n.tick(0)
    n.tick(n.election_deadline)
    n.handle(PreVoteReply(7, vote_granted=False), "b", 1)
    assert (n.current_term, n.pre_voting, n.role) == (7, False, Role.FOLLOWER)
    assert n.handle(PreVoteReply(7, vote_granted=True), "c", 1) == []  # round is over


def test_append_entries_from_a_leader_ends_the_round() -> None:
    n = node(term=1)
    n.tick(0)
    n.tick(n.election_deadline)
    n.handle(AppendEntries(1, "b", 0, 0, (), 0), "b", 1)
    assert not n.pre_voting


def test_voter_grants_without_changing_any_state() -> None:
    n = node(entries=(1,), term=3)
    n.tick(0)
    deadline = n.election_deadline
    assert n.handle(pre_vote(term=4, last_log_index=1, last_log_term=1), "b", 5) == [
        ("b", PreVoteReply(3, vote_granted=True))
    ]
    assert (n.current_term, n.voted_for, n.election_deadline) == (3, None, deadline)
    assert n.leader_hint is None  # a pre-candidate is nobody's guess at the leader


def test_hypothetical_term_is_never_adopted_even_when_refused() -> None:
    n = node(entries=(5,), term=5)
    n.handle(pre_vote(term=99, last_log_index=0, last_log_term=0), "b", 0)  # stale log
    assert n.current_term == 5


@pytest.mark.parametrize(
    ("why", "msg", "term"),
    [
        ("term not newer", pre_vote(term=3, last_log_index=1, last_log_term=1), 3),
        ("stale last term", pre_vote(term=4, last_log_index=9, last_log_term=0), 3),
        ("shorter log", pre_vote(term=4, last_log_index=0, last_log_term=1), 3),
    ],
)
def test_voter_refuses(why: str, msg: PreVote, term: int) -> None:
    n = node(entries=(1,), term=term)
    assert n.handle(msg, "b", 0) == [("b", PreVoteReply(term, vote_granted=False))], why


def test_voter_refuses_while_it_hears_from_a_leader() -> None:
    n = node(term=2)
    n.handle(AppendEntries(2, "c", 0, 0, (), 0), "c", 1000)
    lo = n.election_timeout_range[0]
    assert n.handle(pre_vote(3), "b", 1000 + lo - 1) == [("b", PreVoteReply(2, False))]
    assert n.handle(pre_vote(3), "b", 1000 + lo) == [("b", PreVoteReply(2, True))]


def test_leader_refuses() -> None:
    n = node(peers=(), term=0)
    n.tick(0)
    n.tick(n.election_deadline)
    assert n.role is Role.LEADER
    assert n.handle(pre_vote(5), "b", n.election_deadline) == [("b", PreVoteReply(1, False))]
