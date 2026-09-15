"""Ties the virtual `Clock`, the fault-injecting `Network`, and per-node
`Storage` together into one deterministic simulation harness.

`Cluster` is generic over any node that speaks a tiny protocol --

    tick(now) -> list[(dst, payload)]
    handle(payload, src, now) -> list[(dst, payload)]

-- so it works equally well with a trivial echo node (see
`tests/test_cluster.py`) or a real `RaftNode` (see `raft.node`).
`handle` takes `payload` and `src` as separate arguments rather than a
combined `(src, payload)` tuple, because that's what typed node
implementations actually want to write against: a real node's `handle`
has a specific message type for `payload` and a plain `str`/`int` for
`src`, and a tuple parameter can't express that distinction to mypy the
way two separate ones can.

Cluster is the only thing that ever reads `Clock.now()` and hands `now`
to `Network.send`/`Network.deliver_next`; nodes never see a clock or a
network, only `tick`/`handle` calls with a `now` and whatever messages
they choose to produce in return. `Cluster.clock` and `Cluster.network`
are left public for tests and future callers to inspect or drive
directly -- there is nothing private about a simulation harness.

Two kinds of event compete for "what happens next": a message becoming
due for delivery on the `Network`, and the cluster-wide tick timer
becoming due. Every live node is ticked together, at a fixed interval,
the way a real Raft node's own `tick()` would be driven in production --
Cluster has no notion of individual election or heartbeat timeouts; it
just supplies a steady heartbeat and lets the node decide what to do with
it. `step()` always picks whichever of those two events is due first,
advances `Clock` to exactly that instant, and processes exactly that one
event. Ties (message and timer due at the same instant) resolve in favor
of the message, and ties among nodes sharing a tick instant resolve in
ascending node-id order -- both fixed, documented rules, not something
left to iteration order or a dict's hash order. Combined with `Network`'s
own seeded randomness, that is what makes the whole sequence of events
reproducible from a seed alone.

A crashed node is simply removed from the live set: it produces nothing
(no more ticks), and any message addressed to it -- whether already
in flight or sent afterward -- is dropped without error the moment
Cluster notices the node is down, either when routing an outbound
message to it or when a message addressed to it comes due for delivery.
Restarting a node discards that removal and reconstructs a fresh node
from its (untouched) `Storage`; restarting never causes messages that
arrived while it was down to be replayed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from raft.sim.clock import Clock
from raft.sim.network import Network
from raft.storage import MemoryStorage, Storage

NodeId = int
Message = tuple[NodeId, object]
"""An addressed, opaque payload produced by a node via `tick`/`handle`: `(dst, payload)`."""


class Node(Protocol):
    """The protocol any node -- echo, Raft, or otherwise -- must satisfy."""

    def tick(self, now: int) -> list[Message]:
        """Called for every live node at each cluster-wide tick instant."""
        ...

    def handle(self, payload: object, src: NodeId, now: int) -> list[Message]:
        """Called with one incoming payload addressed to this node, from `src`."""
        ...


class Cluster[N: Node]:
    """Drives `n` nodes of type `N` through a shared virtual `Clock` and `Network`.

    `node_factory(node_id, storage)` builds a node; Cluster calls it once
    per node at construction and again, with that node's original
    `Storage`, on every `restart`. `tick_interval_ms` sets the fixed
    cadence at which every live node is ticked together (a Raft node
    built on this is expected to count ticks itself, the way it would
    count real timer callbacks in production).

    `seed` (the value the caller passed in) and `step_count` (the number
    of `step()` calls that actually processed an event) are both public,
    so anything that needs to name exactly where a run is -- an invariant
    checker reporting a violation, a fuzzer reproducing one -- can do so
    without threading that bookkeeping through separately.
    """

    def __init__(
        self,
        n: int,
        seed: int,
        node_factory: Callable[[NodeId, Storage], N],
        tick_interval_ms: int = 100,
    ) -> None:
        if n < 0:
            raise ValueError("n must be >= 0")
        if tick_interval_ms < 1:
            raise ValueError("tick_interval_ms must be >= 1")

        self.clock = Clock()
        self.network = Network(seed)
        self.seed = seed
        self.step_count = 0
        self._node_factory = node_factory
        self._tick_interval_ms = tick_interval_ms
        self._next_tick_time = tick_interval_ms

        self._storage: dict[NodeId, MemoryStorage] = {i: MemoryStorage() for i in range(n)}
        self._nodes: dict[NodeId, N] = {
            node_id: node_factory(node_id, storage) for node_id, storage in self._storage.items()
        }

    def step(self) -> bool:
        """Process exactly one event: the next due message or the next due tick.

        Returns False, doing nothing, if no event is pending -- no
        messages in flight and no live node left to tick.
        """
        next_msg = self._next_message_time()
        next_timer = self._next_timer_time()

        if next_msg is None and next_timer is None:
            return False

        if next_timer is None or (next_msg is not None and next_msg <= next_timer):
            t = next_msg
            assert t is not None
            self.clock.run_until(t)
            self._deliver_one(t)
        else:
            t = next_timer
            self.clock.run_until(t)
            self._fire_timer(t)

        self.step_count += 1
        return True

    def run(self, steps: int) -> None:
        """Call `step()` up to `steps` times, stopping early once idle."""
        for _ in range(steps):
            if not self.step():
                break

    def get_node(self, node_id: NodeId) -> N | None:
        """Return the currently live node for `node_id`, or None if it's down."""
        return self._nodes.get(node_id)

    def node_ids(self) -> list[NodeId]:
        """Every node id this cluster knows about, whether live or crashed."""
        return list(self._storage)

    def crash(self, node_id: NodeId) -> None:
        """Discard the node's volatile state; its `Storage` is left untouched.

        A no-op if the node is already down.
        """
        self._require_known(node_id)
        self._nodes.pop(node_id, None)

    def restart(self, node_id: NodeId) -> None:
        """Rebuild the node from scratch using its original, undisturbed `Storage`."""
        self._require_known(node_id)
        storage = self._storage[node_id]
        self._nodes[node_id] = self._node_factory(node_id, storage)

    def partition(self, group_a: set[NodeId], group_b: set[NodeId]) -> None:
        """Cut delivery between `group_a` and `group_b`; see `Network.partition`."""
        self.network.partition(group_a, group_b)

    def heal(self) -> None:
        """Remove any active partition; see `Network.heal`."""
        self.network.heal()

    def _require_known(self, node_id: NodeId) -> None:
        if node_id not in self._storage:
            raise KeyError(f"no such node: {node_id!r}")

    def _next_message_time(self) -> int | None:
        pending = self.network.pending()
        if not pending:
            return None
        return min(delivery_time for _src, _dst, _msg, delivery_time in pending)

    def _next_timer_time(self) -> int | None:
        if not self._nodes:  # nothing left alive to ever tick again
            return None
        return self._next_tick_time

    def _deliver_one(self, t: int) -> None:
        delivered = self.network.deliver_next(now=t)
        if delivered is None:
            return  # was due, but dropped in-flight (e.g. a partition)
        src, dst, payload = delivered
        node = self._nodes.get(dst)
        if node is None:
            return  # dst is down; the message is gone, not redelivered later
        produced = node.handle(payload, src, t)
        self._route(dst, produced, t)

    def _fire_timer(self, t: int) -> None:
        for node_id in sorted(self._nodes):  # deterministic tie-break by id
            produced = self._nodes[node_id].tick(t)
            self._route(node_id, produced, t)
        self._next_tick_time = t + self._tick_interval_ms

    def _route(self, src: NodeId, produced: list[Message], now: int) -> None:
        for dst, payload in produced:
            if dst not in self._nodes:
                continue  # dst is down; dropped here, never even queued
            self.network.send(src, dst, payload, now=now)
