"""Per-peer flow control: at most one AppendEntries carrying entries in
flight to each peer. Entries appended meanwhile wait and go out together
when the reply arrives; heartbeats still go out on schedule, empty for a
peer with something in flight; and a message or reply that is lost is
recovered by re-sending once it is a heartbeat interval old.

The benchmark found why this matters (BUGS.md): without it, every client
write sent every peer everything not yet acknowledged, so under load each
entry was encoded and sent dozens of times.
"""

from __future__ import annotations

import random

from raft.messages import AppendEntries, AppendEntriesReply, Message
from raft.node import NotLeader, RaftClusterNode, RaftNode, Role, raft_node_factory
from raft.sim.cluster import Cluster
from raft.statemachine import ClientRequest, Put
from raft.storage import LogEntry, MemoryStorage

HEARTBEAT = 50


def leader(entries: int = 0, peers: tuple[str, ...] = ("b", "c")) -> RaftNode:
    storage = MemoryStorage()
    storage.append_entries([LogEntry(1, f"e{i}") for i in range(1, entries + 1)])
    node = RaftNode("a", list(peers), storage, rng=random.Random(1), heartbeat_interval=HEARTBEAT)
    node.role = Role.LEADER
    node.current_term = 1
    node.next_index = dict.fromkeys(peers, entries + 1)
    node.match_index = dict.fromkeys(peers, entries)
    return node


def to(out: list[tuple[str, Message]] | NotLeader, peer: str) -> list[AppendEntries]:
    assert isinstance(out, list)
    msgs = [m for dst, m in out if dst == peer]
    assert all(isinstance(m, AppendEntries) for m in msgs)
    return msgs  # type: ignore[return-value]


def commands(msg: AppendEntries) -> list[object]:
    return [e.command for e in msg.entries]


def ack(node: RaftNode, peer: str, match: int, now: int) -> list[tuple[str, Message]]:
    return node.handle(AppendEntriesReply(1, success=True, match_index=match), peer, now)


def test_writes_while_in_flight_wait_and_then_go_out_together() -> None:
    node = leader()
    [first] = to(node.append_command("x1", 0), "b")
    assert commands(first) == ["x1"]
    for i, t in ((2, 1), (3, 2), (4, 3)):
        assert to(node.append_command(f"x{i}", t), "b") == []  # held back
    [batch] = to(ack(node, "b", 1, 4), "b")
    assert (batch.prev_log_index, commands(batch)) == (1, ["x2", "x3", "x4"])
    assert to(ack(node, "b", 4, 5), "b") == []  # caught up: nothing more


def test_heartbeat_to_an_in_flight_peer_is_empty_and_leaves_the_flight_alone() -> None:
    node = leader()
    node.append_command("x1", 0)  # to both peers; the heartbeat is due at 50
    node.append_command("x2", 1)  # held
    ack(node, "c", 2, 2)
    [batch] = to(ack(node, "b", 1, 30), "b")  # x2 goes out at 30
    assert commands(batch) == ["x2"]
    [beat] = to(node.tick(HEARTBEAT), "b")  # x2 is only 20 ms old: don't resend it
    assert (beat.prev_log_index, beat.entries) == (1, ())
    assert to(node.append_command("x3", HEARTBEAT), "b") == []  # still in flight
    assert ack(node, "b", 1, HEARTBEAT + 1) == []  # the heartbeat's reply confirms nothing new
    [batch] = to(ack(node, "b", 2, HEARTBEAT + 2), "b")
    assert commands(batch) == ["x3"]


def test_lost_reply_is_recovered_by_the_heartbeat_resending() -> None:
    node = leader()
    node.append_command("x1", 0)
    node.append_command("x2", 1)  # held; then the reply to x1 is lost
    [resend] = to(node.tick(HEARTBEAT), "b")  # x1 is a heartbeat interval old
    assert (resend.prev_log_index, commands(resend)) == (0, ["x1", "x2"])
    assert to(ack(node, "b", 2, HEARTBEAT + 1), "b") == []


def test_dead_peer_costs_one_chunk_per_heartbeat_not_one_per_write() -> None:
    node = leader(peers=("b", "c"))
    carrying = 0
    for now in range(0, 1000):  # a write every millisecond for one second
        out = node.append_command(f"x{now}", now)
        assert isinstance(out, list)
        out += node.tick(now)
        carrying += sum(1 for m in to(out, "b") if m.entries)
        if now % 5 == 0:  # c answers everything; b is dead
            node.handle(
                AppendEntriesReply(1, success=True, match_index=node.storage.last_log_index()),
                "c",
                now,
            )
    assert carrying <= 1000 // HEARTBEAT + 1
    assert node.commit_index > 900  # c and the leader are a majority of three


def test_failure_reply_is_retried_at_once_even_while_writes_are_held() -> None:
    node = leader(entries=4)
    node.next_index["b"] = 5
    node.match_index["b"] = 0
    node.append_command("x5", 0)  # in flight to b at prev 4
    rejected = AppendEntriesReply(1, success=False, conflict_index=2, conflict_term=0)
    [retry] = to(node.handle(rejected, "b", 1), "b")
    assert (retry.prev_log_index, len(retry.entries)) == (1, 4)


# -- In the simulator, with a burst of writes faster than a round trip --


def test_burst_sends_each_entry_to_each_peer_about_once() -> None:
    n, seed, writes = 5, 3, 200
    cluster = Cluster(
        n=n, seed=seed, node_factory=raft_node_factory(n, seed, append_noop_on_election=True)
    )
    while not any(
        (x := cluster.get_node(i)) is not None and x.raft_node.role is Role.LEADER
        for i in cluster.node_ids()
    ):
        cluster.step()
    lead = next(
        i
        for i in cluster.node_ids()
        if (x := cluster.get_node(i)) is not None and x.raft_node.role is Role.LEADER
    )
    while cluster.network.pending():
        cluster.step()
    for a in cluster.node_ids():
        for b in cluster.node_ids():
            if a != b:
                cluster.network.set_delay(a, b, 5)
    sent_entries = 0
    send = cluster.network.send

    def counting(src: int, dst: int, msg: Message, now: int) -> None:
        nonlocal sent_entries
        if src == lead and isinstance(msg, AppendEntries):
            sent_entries += len(msg.entries)
        send(src, dst, msg, now)

    cluster.network.send = counting  # type: ignore[assignment,method-assign]
    node: RaftClusterNode | None = cluster.get_node(lead)
    assert node is not None
    start = cluster.clock.now()
    for serial in range(1, writes + 1):  # one write per simulated millisecond
        while cluster.clock.now() < start + serial and cluster.step():
            pass
        produced = node.append_command(
            ClientRequest("c", serial, Put("k", str(serial))), cluster.clock.now()
        )
        assert isinstance(produced, list)
        cluster.route(lead, produced, cluster.clock.now())
    target = node.raft_node.storage.last_log_index()
    while any(
        (x := cluster.get_node(i)) is not None and x.raft_node.commit_index < target
        for i in cluster.node_ids()
    ):
        cluster.step()
    # Without flow control every write resends everything unacknowledged
    # -- about 10 entries per peer per write at this rate.
    assert sent_entries <= (n - 1) * writes * 1.1
