"""Tests for Cluster, exercised against a trivial echo node.

There is no Raft node yet, so these tests define one inline: EchoNode
replies to every message it receives with a tagged `("reply", payload)`
message, and ignores replies to its own replies. That one safeguard is
necessary to keep test scenarios finite -- two nodes that *unconditionally*
echoed everything, including each other's replies, would ping-pong
forever. It does not change what's under test: Cluster's routing, timing,
and crash/restart plumbing.
"""

from collections.abc import Callable

import pytest

from raft.sim.cluster import Cluster, Message, NodeId
from raft.storage import Storage


class EchoNode:
    """Replies to every non-reply message; ticks produce nothing.

    `events_seen` is deliberately volatile (an ordinary instance
    attribute, reset to 0 whenever a fresh EchoNode is constructed).
    `term`, by contrast, is loaded from `storage` at construction and
    written back to it whenever a message is handled, standing in for
    the kind of durable state a real Raft node would persist.
    """

    def __init__(self, node_id: NodeId, storage: Storage, log: list[tuple] | None = None) -> None:
        self.node_id = node_id
        self.storage = storage
        self.log = log if log is not None else []
        self.events_seen = 0
        term, _voted_for = storage.load_term_and_vote()
        self.term = term

    def tick(self, now: int) -> list[Message]:
        self.events_seen += 1
        self.log.append(("tick", self.node_id, now))
        return []

    def handle(self, msg: Message, now: int) -> list[Message]:
        self.events_seen += 1
        src, payload = msg
        self.log.append(("handle", self.node_id, now, src, payload))
        if isinstance(payload, tuple) and payload and payload[0] == "reply":
            return []  # don't reply to a reply -- keeps this finite
        self.term += 1
        self.storage.save_term_and_vote(self.term, f"node-{self.node_id}")
        return [(src, ("reply", payload))]


def _make_factory(log: list[tuple]) -> Callable[[NodeId, Storage], EchoNode]:
    def factory(node_id: NodeId, storage: Storage) -> EchoNode:
        return EchoNode(node_id, storage, log)

    return factory


def test_fixed_seed_produces_identical_event_sequence_across_two_runs() -> None:
    def build_and_run() -> list[tuple]:
        log: list[tuple] = []
        cluster = Cluster(n=3, seed=42, node_factory=_make_factory(log), tick_interval_ms=10)
        cluster.network.set_delay(0, 1, 5)
        cluster.network.send(0, 1, "hello", now=0)
        cluster.run(50)
        return log

    assert build_and_run() == build_and_run()


def test_crash_then_restart_preserves_storage_and_resets_volatile_state() -> None:
    cluster = Cluster(n=2, seed=1, node_factory=_make_factory([]), tick_interval_ms=1000)

    cluster.network.send(1, 0, "ping", now=0)
    cluster.step()  # deliver "ping" to node 0: term -> 1, persisted; events_seen -> 1

    before = cluster.get_node(0)
    assert before is not None
    assert before.term == 1
    assert before.events_seen == 1

    cluster.crash(0)
    assert cluster.get_node(0) is None  # gone while down

    cluster.restart(0)
    after = cluster.get_node(0)

    assert after is not None
    assert after is not before  # a genuinely new instance
    assert after.events_seen == 0  # volatile state: reset
    assert after.term == 1  # durable state: survived the crash
    assert after.storage.load_term_and_vote() == (1, "node-0")


def test_messages_to_a_crashed_node_are_dropped() -> None:
    log: list[tuple] = []
    cluster = Cluster(n=2, seed=1, node_factory=_make_factory(log), tick_interval_ms=1000)
    cluster.crash(1)

    cluster.network.send(0, 1, "hello", now=0)
    happened = cluster.step()  # the delivery attempt is still "an event"

    assert happened is True
    assert cluster.network.pending() == []  # gone, not sitting queued for a restart
    assert log == []  # node 1 never saw it -- it didn't exist to receive it


def test_partition_prevents_cross_group_delivery_but_allows_within_group() -> None:
    log: list[tuple] = []
    cluster = Cluster(n=3, seed=1, node_factory=_make_factory(log), tick_interval_ms=1000)
    cluster.partition({0, 1}, {2})

    cluster.network.send(0, 1, "within-group", now=0)
    cluster.network.send(0, 2, "cross-group", now=0)

    cluster.run(2)  # both sends are due at t=0; two steps drains both

    handled_by = {node_id for kind, node_id, *_rest in log if kind == "handle"}
    assert 1 in handled_by  # within-group delivery went through
    assert 2 not in handled_by  # cross-group delivery was cut


def test_heal_restores_cross_group_delivery() -> None:
    log: list[tuple] = []
    cluster = Cluster(n=2, seed=1, node_factory=_make_factory(log), tick_interval_ms=1000)
    cluster.partition({0}, {1})
    cluster.heal()

    cluster.network.send(0, 1, "hello", now=0)
    cluster.step()

    handled_by = {node_id for kind, node_id, *_rest in log if kind == "handle"}
    assert 1 in handled_by


def test_step_returns_false_when_nothing_is_pending() -> None:
    cluster = Cluster(n=0, seed=1, node_factory=EchoNode)

    assert cluster.step() is False


def test_a_reply_to_a_now_crashed_node_is_dropped_before_being_queued() -> None:
    log: list[tuple] = []
    cluster = Cluster(n=2, seed=1, node_factory=_make_factory(log), tick_interval_ms=1000)

    cluster.network.send(1, 0, "ping", now=0)  # from node 1, while it's still alive
    cluster.crash(1)  # node 1 goes down before its own message is even delivered

    cluster.step()  # node 0 handles "ping" and tries to reply to node 1

    assert cluster.network.pending() == []  # the reply was never queued
    assert any(kind == "handle" and node_id == 0 for kind, node_id, *_rest in log)


def test_run_stops_early_once_idle_rather_than_looping_pointlessly() -> None:
    cluster = Cluster(n=0, seed=1, node_factory=EchoNode)

    cluster.run(5)  # nothing pending, ever; must not loop or raise

    assert cluster.clock.now() == 0


def test_a_message_dropped_entirely_by_partition_still_counts_as_a_step() -> None:
    # A single due message whose link is cut has no deliverable candidate
    # at all -- Network.deliver_next returns None -- which is distinct
    # from nothing being pending in the first place.
    log: list[tuple] = []
    cluster = Cluster(n=2, seed=1, node_factory=_make_factory(log), tick_interval_ms=1000)
    cluster.partition({0}, {1})

    cluster.network.send(0, 1, "lost", now=0)
    happened = cluster.step()

    assert happened is True
    assert cluster.network.pending() == []
    assert log == []


def test_cluster_rejects_negative_n() -> None:
    with pytest.raises(ValueError, match="n must be"):
        Cluster(n=-1, seed=1, node_factory=EchoNode)


def test_cluster_rejects_non_positive_tick_interval() -> None:
    with pytest.raises(ValueError, match="tick_interval_ms"):
        Cluster(n=1, seed=1, node_factory=EchoNode, tick_interval_ms=0)


def test_crash_on_unknown_node_id_raises() -> None:
    cluster = Cluster(n=1, seed=1, node_factory=EchoNode)
    with pytest.raises(KeyError):
        cluster.crash(99)


def test_restart_on_unknown_node_id_raises() -> None:
    cluster = Cluster(n=1, seed=1, node_factory=EchoNode)
    with pytest.raises(KeyError):
        cluster.restart(99)


def test_run_is_equivalent_to_n_calls_to_step() -> None:
    def via_run(steps: int) -> tuple[list[tuple], int]:
        log: list[tuple] = []
        cluster = Cluster(n=3, seed=7, node_factory=_make_factory(log), tick_interval_ms=10)
        cluster.network.send(0, 1, "hello", now=0)
        cluster.run(steps)
        return log, cluster.clock.now()

    def via_step_loop(steps: int) -> tuple[list[tuple], int]:
        log: list[tuple] = []
        cluster = Cluster(n=3, seed=7, node_factory=_make_factory(log), tick_interval_ms=10)
        cluster.network.send(0, 1, "hello", now=0)
        for _ in range(steps):
            cluster.step()
        return log, cluster.clock.now()

    steps = 30
    assert via_run(steps) == via_step_loop(steps)
