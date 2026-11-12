"""Chunked catch-up: one AppendEntries carries a bounded number of entries
and bytes, and a follower still behind gets the next chunk as soon as it
acknowledges the last one.

Without the cap, the AppendEntries that catches up a far-behind follower
carries the whole gap; over gRPC that message passes the receiver's
4 MiB limit and is rejected every time (BUGS.md). The simulator has no
message size limit, so the scenario here checks the shape -- every
message within the caps, and chunks pipelined at round-trip speed rather
than heartbeat speed -- and the unit tests pin each rule.
"""

from __future__ import annotations

import random

import pytest

from raft.messages import AppendEntries, AppendEntriesReply, Message
from raft.node import (
    DEFAULT_MAX_APPEND_ENTRIES,
    RaftClusterNode,
    RaftNode,
    Role,
    approximate_entry_size,
    raft_node_factory,
)
from raft.sim.cluster import Cluster
from raft.statemachine import ClientRequest, Put
from raft.storage import LogEntry, MemoryStorage

# -- Scenario --

SEED = 4
N = 3
ENTRIES = 3000
LINK_DELAY_MS = 5  # a 10 ms round trip
HEARTBEAT_MS = 50


def get(cluster: Cluster[RaftClusterNode], node_id: int) -> RaftNode:
    found = cluster.get_node(node_id)
    assert found is not None
    return found.raft_node


def test_far_behind_follower_catches_up_in_capped_chunks_at_round_trip_speed() -> None:
    cluster = Cluster(
        n=N, seed=SEED, node_factory=raft_node_factory(N, SEED, append_noop_on_election=True)
    )
    while not any(
        (n := cluster.get_node(i)) is not None and n.raft_node.role is Role.LEADER
        for i in cluster.node_ids()
    ):
        cluster.step()
    leader = next(i for i in cluster.node_ids() if get(cluster, i).role is Role.LEADER)
    follower = next(i for i in cluster.node_ids() if i != leader)
    cluster.crash(follower)

    for serial in range(1, ENTRIES + 1):
        node = cluster.get_node(leader)
        assert node is not None
        produced = node.append_command(
            ClientRequest("c", serial, Put("k", str(serial))), cluster.clock.now()
        )
        assert isinstance(produced, list)
        cluster.route(leader, produced, cluster.clock.now())
        while cluster.network.pending():
            cluster.step()
    target = get(cluster, leader).storage.last_log_index()

    for a in cluster.node_ids():
        for b in cluster.node_ids():
            if a != b:
                cluster.network.set_delay(a, b, LINK_DELAY_MS)
    sizes: list[int] = []
    send = cluster.network.send

    def recording_send(src: int, dst: int, msg: Message, now: int) -> None:
        if dst == follower and isinstance(msg, AppendEntries):
            sizes.append(len(msg.entries))
        send(src, dst, msg, now)

    cluster.network.send = recording_send  # type: ignore[assignment,method-assign]
    cluster.restart(follower)
    first_chunk_at = None
    while get(cluster, follower).commit_index < target:
        if first_chunk_at is None and any(sizes):
            first_chunk_at = cluster.clock.now()
        assert cluster.clock.now() < 5000, f"follower still behind: {sizes}"
        cluster.step()
    assert first_chunk_at is not None

    assert max(sizes) == DEFAULT_MAX_APPEND_ENTRIES  # capped, and the cap was used
    chunks = -(-ENTRIES // DEFAULT_MAX_APPEND_ENTRIES)
    assert sum(1 for s in sizes if s) <= chunks + 2  # no chunk sent many times over
    # Pipelined: about one round trip per chunk. Waiting for a heartbeat
    # per chunk would need (chunks - 1) * HEARTBEAT_MS = 250 ms.
    assert cluster.clock.now() - first_chunk_at < (chunks - 1) * HEARTBEAT_MS // 2
    assert get(cluster, follower).storage.load_log() == get(cluster, leader).storage.load_log()


# -- Unit tests --


def leader(
    entries: int,
    *,
    max_entries: int = DEFAULT_MAX_APPEND_ENTRIES,
    max_bytes: int = 1 << 20,
    entry_size: object = approximate_entry_size,
) -> RaftNode:
    storage = MemoryStorage()
    storage.append_entries([LogEntry(1, f"e{i}") for i in range(1, entries + 1)])
    node = RaftNode(
        "a",
        ["b", "c"],
        storage,
        rng=random.Random(1),
        max_append_entries=max_entries,
        max_append_bytes=max_bytes,
        entry_size=entry_size,  # type: ignore[arg-type]
    )
    node.role = Role.LEADER
    node.current_term = 1
    node.next_index = {"b": 1, "c": entries + 1}
    node.match_index = {"b": 0, "c": entries}
    return node


def append_entries_to(out: list[tuple[str, Message]], peer: str) -> AppendEntries:
    [msg] = [m for dst, m in out if dst == peer]
    assert isinstance(msg, AppendEntries)
    return msg


def test_chunk_is_capped_by_count() -> None:
    node = leader(1300, max_entries=512)
    msg = append_entries_to(node._send_append_entries(0), "b")
    assert (msg.prev_log_index, len(msg.entries)) == (0, 512)
    assert msg.entries[0].command == "e1"


def test_chunk_is_capped_by_size() -> None:
    node = leader(10, max_bytes=100, entry_size=lambda e: 30)
    assert len(append_entries_to(node._send_append_entries(0), "b").entries) == 3  # 90 <= 100


def test_one_oversized_entry_is_still_sent_alone() -> None:
    node = leader(3, max_bytes=10, entry_size=lambda e: 1000)
    assert len(append_entries_to(node._send_append_entries(0), "b").entries) == 1


def test_short_tail_is_sent_whole_and_caught_up_peer_gets_a_heartbeat() -> None:
    node = leader(5)
    out = node._send_append_entries(0)
    assert len(append_entries_to(out, "b").entries) == 5
    assert append_entries_to(out, "c").entries == ()


def test_reply_confirming_a_chunk_sends_the_next_one_at_once() -> None:
    node = leader(1300, max_entries=512)
    node._send_append_entries(0)
    out = node.handle(AppendEntriesReply(1, success=True, match_index=512), "b", 10)
    msg = append_entries_to(out, "b")
    assert (msg.prev_log_index, len(msg.entries)) == (512, 512)
    out = node.handle(AppendEntriesReply(1, success=True, match_index=1024), "b", 20)
    msg = append_entries_to(out, "b")
    assert (msg.prev_log_index, len(msg.entries)) == (1024, 276)  # the rest
    assert node.handle(AppendEntriesReply(1, success=True, match_index=1300), "b", 30) == []


def test_no_extra_send_while_newer_entries_are_already_in_flight() -> None:
    node = leader(10)
    node._send_append_entries(0)  # carries 1..10
    node.append_command("e11", 1)  # carries 1..11
    # The reply to the first message: behind, but 11 is already on its way.
    assert node.handle(AppendEntriesReply(1, success=True, match_index=10), "b", 2) == []
    assert node.match_index["b"] == 10


def test_stale_success_reply_sends_nothing() -> None:
    node = leader(1300, max_entries=512)
    node._send_append_entries(0)
    node.handle(AppendEntriesReply(1, success=True, match_index=512), "b", 10)
    assert node.handle(AppendEntriesReply(1, success=True, match_index=512), "b", 11) == []


@pytest.mark.parametrize("kwargs", [{"max_append_entries": 0}, {"max_append_bytes": 0}])
def test_caps_must_be_positive(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        RaftNode("a", [], MemoryStorage(), **kwargs)  # type: ignore[arg-type]


def test_approximate_size_grows_with_the_command() -> None:
    assert approximate_entry_size(LogEntry(1, "x" * 1000)) > approximate_entry_size(
        LogEntry(1, "x")
    )
