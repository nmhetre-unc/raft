"""A follower's commit_index never decreases.

Figure 2's AppendEntries rule 5, taken literally -- "if leaderCommit >
commitIndex, set commitIndex = min(leaderCommit, index of last new entry)"
-- lowers commitIndex whenever a message's last covered index is below it
while its leaderCommit is above it. The leader produces exactly that: to a
follower with entries in flight it sends an *empty* heartbeat anchored at
the last index the follower has *acknowledged* (flow control), carrying
the leader's *current* commit index. A lagging follower that has already
received the in-flight chunk -- and committed up to the commit index that
chunk carried -- then drops back to its old acknowledged index as soon as
the commit index moves on before the chunk's reply gets home.

Found on the Docker cluster (BUGS.md): after a partition, the rejoined
node's commit index flipped between 713 and 1225 for seconds.
"""

from __future__ import annotations

import pytest

from raft.invariants import SafetyViolation, check_commit_index_monotonic
from raft.messages import AppendEntries
from raft.node import RaftClusterNode, RaftNode, Role, raft_node_factory
from raft.sim.cluster import Cluster
from raft.statemachine import ClientRequest, Put
from raft.storage import LogEntry, MemoryStorage

# -- RaftNode on its own --


def test_heartbeat_anchored_below_the_commit_index_does_not_lower_it() -> None:
    follower = RaftNode("f", ["L", "x"], MemoryStorage())
    entries = tuple(LogEntry(1, f"c{i}") for i in range(1, 11))
    follower.handle(AppendEntries(1, "L", 0, 0, entries, leader_commit=10), "L", 0)
    assert follower.commit_index == 10
    # The leader has seen no reply yet, so it still anchors at 5; its own
    # commit index has meanwhile moved on to 12.
    follower.handle(AppendEntries(1, "L", 5, 1, (), leader_commit=12), "L", 1)
    assert follower.commit_index == 10
    # It still advances when a message does cover more.
    follower.handle(AppendEntries(1, "L", 10, 1, (), leader_commit=12), "L", 2)
    assert follower.commit_index == 10  # covers only through 10
    more = (LogEntry(1, "c11"), LogEntry(1, "c12"))
    follower.handle(AppendEntries(1, "L", 10, 1, more, leader_commit=12), "L", 3)
    assert follower.commit_index == 12


# -- The scenario, end to end in the simulator --

SEED = 3
N = 3
BACKLOG = 20
CHUNK = 5  # entries per AppendEntries: catching up takes several
# follower -> leader only. Under the 50 ms heartbeat interval, and not a
# multiple of the 10 ms tick: each reply makes the leader send the next
# chunk at once, mid-interval, so the next heartbeat finds it in flight.
SLOW_ACK_MS = 33


def node(cluster: Cluster[RaftClusterNode], node_id: int) -> RaftClusterNode:
    found = cluster.get_node(node_id)
    assert found is not None
    return found


def run_until_leader(cluster: Cluster[RaftClusterNode]) -> int:
    for _ in range(100_000):
        leaders = [
            i
            for i in cluster.node_ids()
            if (n := cluster.get_node(i)) is not None and n.raft_node.role is Role.LEADER
        ]
        if leaders:
            return leaders[0]
        cluster.step()
    raise AssertionError("no leader elected")


def submit(cluster: Cluster[RaftClusterNode], leader: int, serial: int) -> None:
    now = cluster.clock.now()
    request = ClientRequest("c", serial, Put("k", str(serial)))
    produced = node(cluster, leader).append_command(request, now)
    assert isinstance(produced, list)
    cluster.route(leader, produced, now)


def test_follower_catching_up_in_chunks_never_lowers_its_commit_index() -> None:
    """A follower BACKLOG entries behind catches up CHUNK entries per
    AppendEntries. Each chunk carries the leader's commit index, so the
    follower's commit index climbs chunk by chunk; meanwhile the leader's
    heartbeat to it -- empty, the next chunk being in flight -- is anchored
    one chunk lower and carries a commit index above the follower's. With
    rule 5 taken literally, that heartbeat sent the commit index back down
    one chunk, every time."""
    # Ticks every 10 ms (the default 100 ms tick always lands after a
    # chunk's one-heartbeat "in flight" window, and resends it instead).
    cluster = Cluster(
        n=N,
        seed=SEED,
        node_factory=raft_node_factory(
            N, SEED, append_noop_on_election=True, max_append_entries=CHUNK
        ),
        tick_interval_ms=10,
    )
    leader = run_until_leader(cluster)
    follower = next(i for i in cluster.node_ids() if i != leader)

    # The follower falls behind: down while BACKLOG entries commit.
    cluster.crash(follower)
    for serial in range(1, BACKLOG + 1):
        submit(cluster, leader, serial)
        while cluster.network.pending():
            cluster.step()
    leading = node(cluster, leader).raft_node
    assert leading.commit_index > BACKLOG

    cluster.network.set_delay(follower, leader, SLOW_ACK_MS)
    cluster.restart(follower)
    lagging = node(cluster, follower).raft_node
    history: dict[int, tuple[RaftClusterNode, int]] = {}
    seen: list[int] = []
    for _ in range(5000):
        cluster.step()
        check_commit_index_monotonic(cluster, history)
        if not seen or seen[-1] != lagging.commit_index:
            seen.append(lagging.commit_index)
        if lagging.commit_index == leading.commit_index:
            break
    else:
        raise AssertionError(f"the follower never caught up: {seen}")
    assert len(seen) > 3  # it really did climb chunk by chunk
    assert seen == sorted(seen)


def test_checker_flags_a_decrease_but_not_a_restart() -> None:
    cluster = Cluster(n=N, seed=SEED, node_factory=raft_node_factory(N, SEED))
    history: dict[int, tuple[RaftClusterNode, int]] = {}
    check_commit_index_monotonic(cluster, history)
    victim = node(cluster, 0).raft_node
    victim.commit_index = 3
    check_commit_index_monotonic(cluster, history)

    cluster.crash(0)  # forgotten while down ...
    check_commit_index_monotonic(cluster, history)
    assert 0 not in history
    cluster.restart(0)  # ... and a new incarnation starts again from 0
    check_commit_index_monotonic(cluster, history)

    cluster.crash(1)
    cluster.restart(1)  # restarted between two checks: told apart by identity
    node(cluster, 2).raft_node.commit_index = 2
    check_commit_index_monotonic(cluster, history)
    node(cluster, 2).raft_node.commit_index = 1
    with pytest.raises(SafetyViolation, match="node 2 commit_index decreased from 2 to 1"):
        check_commit_index_monotonic(cluster, history)
