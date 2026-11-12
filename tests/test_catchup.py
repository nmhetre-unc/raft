"""A follower that comes back thousands of entries behind must catch up
in a bounded number of round trips.

The benchmark found this one (see BUGS.md): a restarted node never caught
up because the leader backed `next_index` up one entry per round trip,
and every one of those retries carried the whole log tail. This pins it
down deterministically in the simulator.

The precondition matters. A leader that was already leading when the
follower crashed keeps `next_index` at the crash point -- it never got a
reply to move it -- so one AppendEntries closes the gap and nothing is
exercised. What makes backtracking necessary is a leader elected *while
the follower was down*: it initializes `next_index` optimistically to
its own log end, thousands of entries past the follower's. So after
building the log, the old leader is crashed and a new one elected among
the remaining three (still a majority of five) before the follower
returns.

Links get a 5 ms delay for the catch-up phase so a round trip costs
simulated time; building the log runs with zero delay so it's fast.
"""

from __future__ import annotations

from raft.messages import Message
from raft.node import RaftClusterNode, Role, raft_node_factory
from raft.sim.cluster import Cluster
from raft.statemachine import ClientRequest, Put

SEED = 4
N = 5
ENTRIES = 5000
LINK_DELAY_MS = 5
# One election-timeout's worth of waiting for the first heartbeat, plus a
# few dozen 10 ms round trips -- generous for any backtracking that isn't
# linear in the gap, and orders of magnitude short of one-per-entry.
CATCH_UP_BOUND_MS = 1000
CATCH_UP_BOUND_MESSAGES = 200


def node(cluster: Cluster[RaftClusterNode], node_id: int) -> RaftClusterNode:
    found = cluster.get_node(node_id)
    assert found is not None
    return found


def leader_of(cluster: Cluster[RaftClusterNode]) -> int | None:
    leaders = [
        (n.raft_node.current_term, i)
        for i in cluster.node_ids()
        if (n := cluster.get_node(i)) is not None and n.raft_node.role is Role.LEADER
    ]
    return max(leaders)[1] if leaders else None


def run_until_leader(cluster: Cluster[RaftClusterNode], exclude: set[int]) -> int:
    for _ in range(100_000):
        leader = leader_of(cluster)
        if leader is not None and leader not in exclude:
            return leader
        cluster.step()
    raise AssertionError("no leader elected")


def drain(cluster: Cluster[RaftClusterNode]) -> None:
    """Deliver everything in flight (zero-delay links: all due now)."""
    while cluster.network.pending():
        cluster.step()


def test_restarted_follower_catches_up_in_bounded_time_and_messages() -> None:
    cluster = Cluster(
        n=N, seed=SEED, node_factory=raft_node_factory(N, SEED, append_noop_on_election=True)
    )
    old_leader = run_until_leader(cluster, exclude=set())
    follower = next(i for i in cluster.node_ids() if i != old_leader)
    cluster.crash(follower)
    follower_had = node(cluster, old_leader).raft_node.storage.last_log_index()

    for serial in range(1, ENTRIES + 1):
        now = cluster.clock.now()
        request = ClientRequest("c", serial, Put(f"k{serial}", str(serial)))
        produced = node(cluster, old_leader).append_command(request, now)
        assert isinstance(produced, list)
        cluster.route(old_leader, produced, now)
        drain(cluster)
    assert node(cluster, old_leader).raft_node.commit_index >= ENTRIES

    # A leader elected while the follower is down.
    cluster.crash(old_leader)
    new_leader = run_until_leader(cluster, exclude={old_leader})
    drain(cluster)

    for a in cluster.node_ids():
        for b in cluster.node_ids():
            if a != b:
                cluster.network.set_delay(a, b, LINK_DELAY_MS)
    messages = 0
    send = cluster.network.send

    def counting_send(src: int, dst: int, msg: Message, now: int) -> None:
        nonlocal messages
        if follower in (src, dst):
            messages += 1
        send(src, dst, msg, now)

    cluster.network.send = counting_send  # type: ignore[assignment,method-assign]
    cluster.restart(follower)
    start = cluster.clock.now()

    def caught_up() -> bool:
        leader = node(cluster, new_leader).raft_node
        mine = node(cluster, follower).raft_node
        return (
            mine.storage.last_log_index() == leader.storage.last_log_index()
            and mine.commit_index == leader.commit_index
        )

    while not caught_up():
        elapsed = cluster.clock.now() - start
        assert elapsed <= CATCH_UP_BOUND_MS and messages <= CATCH_UP_BOUND_MESSAGES, (
            f"follower {follower} still behind after {elapsed} ms and {messages} messages: "
            f"has {node(cluster, follower).raft_node.storage.last_log_index()} entries "
            f"(had {follower_had} when it crashed), leader "
            f"{new_leader} has {node(cluster, new_leader).raft_node.storage.last_log_index()}"
        )
        cluster.step()

    assert leader_of(cluster) == new_leader  # caught up without disrupting anyone
    assert (
        node(cluster, follower).raft_node.storage.load_log()
        == node(cluster, new_leader).raft_node.storage.load_log()
    )
