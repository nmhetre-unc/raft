"""Tests for RaftNode's leader election, wired into Cluster via raft_node_factory."""

import random

import pytest

from raft.messages import AppendEntries, AppendEntriesReply, RequestVote, RequestVoteReply
from raft.node import RaftNode, Role, raft_node_factory
from raft.sim.cluster import Cluster
from raft.storage import LogEntry, MemoryStorage, StorageError

# Generous but cheap: calibration showed 5-node clusters converge on a
# leader within ~35 steps for every seed tried, so this leaves ample
# headroom (including for a second, post-crash election) without the
# suite being slow.
RUN_STEPS = 3000


def make_cluster(n: int, seed: int, tick_interval_ms: int = 10) -> Cluster:
    return Cluster(
        n=n,
        seed=seed,
        node_factory=raft_node_factory(n, seed),
        tick_interval_ms=tick_interval_ms,
    )


def _force_election_timeout(node: RaftNode) -> None:
    """Make `node`'s very next tick() start an election.

    `election_deadline`'s real first value is only drawn on a node's
    first tick() call (see RaftNode's docstring for why), so setting
    `election_deadline` alone before that first call would just be
    overwritten. Marking it already-initialized keeps this override in
    place.
    """
    node.election_deadline = 0
    node._deadline_initialized = True


def leaders(cluster: Cluster) -> list[RaftNode]:
    """Every currently-live node whose role is LEADER."""
    result = []
    for node_id in cluster.node_ids():
        node = cluster.get_node(node_id)
        if node is not None and node.role is Role.LEADER:
            result.append(node)
    return result


def test_five_nodes_boot_with_no_leader_then_elect_exactly_one() -> None:
    cluster = make_cluster(n=5, seed=1)

    assert leaders(cluster) == []  # nobody is leader right after boot

    cluster.run(RUN_STEPS)

    won = leaders(cluster)
    assert len(won) == 1


def test_every_follower_agrees_on_the_leader_id_and_term() -> None:
    cluster = make_cluster(n=5, seed=1)
    cluster.run(RUN_STEPS)

    [leader] = leaders(cluster)

    for node_id in cluster.node_ids():
        node = cluster.get_node(node_id)
        assert node is not None
        if node is leader:
            continue
        assert node.role is Role.FOLLOWER
        assert node.leader_id == leader.node_id
        assert node.current_term == leader.current_term


def test_killing_the_leader_elects_a_new_one_in_a_strictly_higher_term() -> None:
    cluster = make_cluster(n=5, seed=2)
    cluster.run(RUN_STEPS)
    [old_leader] = leaders(cluster)
    old_term = old_leader.current_term
    old_id = old_leader.node_id

    cluster.crash(int(old_id))
    cluster.run(RUN_STEPS)

    won = leaders(cluster)
    assert len(won) == 1
    new_leader = won[0]
    assert new_leader.node_id != old_id
    assert new_leader.current_term > old_term


def test_restarting_a_dead_follower_rejoins_without_disrupting_the_leader() -> None:
    cluster = make_cluster(n=5, seed=3)
    cluster.run(RUN_STEPS)
    [leader] = leaders(cluster)

    follower_id = next(nid for nid in cluster.node_ids() if str(nid) != leader.node_id)

    cluster.crash(follower_id)
    cluster.run(RUN_STEPS)
    assert leaders(cluster) == [leader]  # losing one follower doesn't cost the leader

    cluster.restart(follower_id)
    cluster.run(RUN_STEPS)

    restarted = cluster.get_node(follower_id)
    assert restarted is not None
    assert restarted.role is Role.FOLLOWER
    assert restarted.leader_id == leader.node_id
    assert restarted.current_term == leader.current_term
    assert leaders(cluster) == [leader]  # still the same leader, undisturbed


def test_candidate_with_a_shorter_or_older_log_is_denied_a_vote() -> None:
    voter_storage = MemoryStorage()
    voter_storage.append_entries([LogEntry(term=5, command="x")])  # voter's log is ahead
    voter = RaftNode("voter", ["candidate"], voter_storage, rng=random.Random(1))

    candidate_storage = MemoryStorage()  # empty log -- behind
    candidate = RaftNode("candidate", ["voter"], candidate_storage, rng=random.Random(2))
    _force_election_timeout(candidate)
    produced = candidate.tick(now=1000)

    assert len(produced) == 1
    dst, request = produced[0]
    assert dst == "voter"
    assert isinstance(request, RequestVote)

    replies = voter.handle(request, src="candidate", now=1001)

    assert replies == [("candidate", RequestVoteReply(term=request.term, vote_granted=False))]
    assert voter.voted_for is None


def test_a_vote_persisted_in_a_term_survives_crash_and_is_not_cast_twice() -> None:
    storage = MemoryStorage()
    voter = RaftNode("voter", ["candidate-a", "candidate-b"], storage, rng=random.Random(1))

    first_request = RequestVote(
        term=5, candidate_id="candidate-a", last_log_index=0, last_log_term=0
    )
    first_reply = voter.handle(first_request, src="candidate-a", now=0)
    assert first_reply == [("candidate-a", RequestVoteReply(term=5, vote_granted=True))]
    assert storage.load_term_and_vote() == (5, "candidate-a")

    # Simulate a crash + restart: a *fresh* RaftNode built from the same,
    # untouched Storage -- exactly what Cluster.restart() does.
    restarted = RaftNode("voter", ["candidate-a", "candidate-b"], storage, rng=random.Random(2))
    assert restarted.current_term == 5
    assert restarted.voted_for == "candidate-a"

    second_request = RequestVote(
        term=5, candidate_id="candidate-b", last_log_index=0, last_log_term=0
    )
    second_reply = restarted.handle(second_request, src="candidate-b", now=0)

    assert second_reply == [("candidate-b", RequestVoteReply(term=5, vote_granted=False))]


def test_persisted_vote_guard_actually_depends_on_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mutation check: disable persistence and confirm the test above would fail.

    This isn't testing RaftNode's real behavior -- persistence is always on
    in every other test in this file -- it exists purely to prove that
    `test_a_vote_persisted_in_a_term_survives_crash_and_is_not_cast_twice`
    is actually exercising the persistence guarantee, not passing for some
    unrelated reason.
    """
    monkeypatch.setattr(RaftNode, "_persist_if_dirty", lambda self: None)

    storage = MemoryStorage()
    voter = RaftNode("voter", ["candidate-a", "candidate-b"], storage, rng=random.Random(1))
    voter.handle(
        RequestVote(term=5, candidate_id="candidate-a", last_log_index=0, last_log_term=0),
        src="candidate-a",
        now=0,
    )
    # Storage was never actually written, since persistence is disabled.
    assert storage.load_term_and_vote() == (0, None)

    restarted = RaftNode("voter", ["candidate-a", "candidate-b"], storage, rng=random.Random(2))
    second_reply = restarted.handle(
        RequestVote(term=5, candidate_id="candidate-b", last_log_index=0, last_log_term=0),
        src="candidate-b",
        now=0,
    )

    # Without persistence, the restarted node has no memory of its first
    # vote and grants a second one in the same term -- exactly the bug
    # the real test guards against.
    assert second_reply == [("candidate-b", RequestVoteReply(term=5, vote_granted=True))]


def test_fixed_seed_elects_the_same_leader_in_the_same_term_across_two_runs() -> None:
    def run_once() -> tuple[str, int]:
        cluster = make_cluster(n=5, seed=99)
        cluster.run(RUN_STEPS)
        [leader] = leaders(cluster)
        return leader.node_id, leader.current_term

    assert run_once() == run_once()


def test_persist_before_reply_storage_failure_prevents_the_reply() -> None:
    storage = MemoryStorage()
    node = RaftNode("n", ["peer"], storage, rng=random.Random(1))
    storage.fail_after(1)

    higher_term_request = RequestVote(
        term=node.current_term + 1, candidate_id="peer", last_log_index=0, last_log_term=0
    )

    with pytest.raises(StorageError):
        node.handle(higher_term_request, src="peer", now=0)

    # The write itself failed, so nothing was durably recorded either.
    assert storage.load_term_and_vote() == (0, None)


def test_a_lone_node_with_no_peers_becomes_leader_immediately() -> None:
    storage = MemoryStorage()
    node = RaftNode("solo", [], storage, rng=random.Random(1))
    _force_election_timeout(node)

    out = node.tick(now=10)

    assert node.role is Role.LEADER
    assert out == []  # no peers to heartbeat


def test_leader_sends_heartbeats_and_followers_record_it() -> None:
    cluster = make_cluster(n=3, seed=4)
    cluster.run(RUN_STEPS)
    [leader] = leaders(cluster)

    # Run further so at least one more heartbeat round goes out and back.
    cluster.run(RUN_STEPS)

    for node_id in cluster.node_ids():
        node = cluster.get_node(node_id)
        assert node is not None
        if node is leader:
            continue
        assert node.leader_id == leader.node_id


def test_constructor_rejects_invalid_election_timeout_range() -> None:
    storage = MemoryStorage()
    with pytest.raises(ValueError, match="election_timeout_range"):
        RaftNode("n", [], storage, election_timeout_range=(300, 150))


def test_constructor_rejects_non_positive_heartbeat_interval() -> None:
    storage = MemoryStorage()
    with pytest.raises(ValueError, match="heartbeat_interval"):
        RaftNode("n", [], storage, heartbeat_interval=0)


def test_stale_append_entries_is_rejected_with_current_term() -> None:
    storage = MemoryStorage()
    storage.save_term_and_vote(5, None)
    node = RaftNode("n", ["leader"], storage, rng=random.Random(1))

    stale_heartbeat = AppendEntries(
        term=2, leader_id="leader", prev_log_index=0, prev_log_term=0, entries=(), leader_commit=0
    )
    reply = node.handle(stale_heartbeat, src="leader", now=0)

    assert reply == [("leader", AppendEntriesReply(term=5, success=False))]
    assert node.role is Role.FOLLOWER
    assert node.leader_id is None  # a stale heartbeat must not be recorded as legitimate


def test_stale_request_vote_is_rejected_with_current_term() -> None:
    storage = MemoryStorage()
    storage.save_term_and_vote(5, None)
    node = RaftNode("n", ["candidate"], storage, rng=random.Random(1))

    stale_request = RequestVote(term=2, candidate_id="candidate", last_log_index=0, last_log_term=0)
    reply = node.handle(stale_request, src="candidate", now=0)

    assert reply == [("candidate", RequestVoteReply(term=5, vote_granted=False))]
    assert node.voted_for is None


def test_stale_request_vote_reply_is_ignored() -> None:
    storage = MemoryStorage()
    storage.save_term_and_vote(5, None)
    node = RaftNode("n", ["peer"], storage, rng=random.Random(1))

    stale_reply = RequestVoteReply(term=2, vote_granted=True)
    out = node.handle(stale_reply, src="peer", now=0)

    assert out == []
    assert node.role is Role.FOLLOWER  # a stale vote can't make us leader
