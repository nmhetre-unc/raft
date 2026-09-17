"""A simulated, fault-injecting message bus.

The Network moves opaque messages between named endpoints. It never
inspects a message's contents -- it only routes it -- and it never reads
a real clock or the wall clock: every notion of "now" is an integer the
caller (the Cluster) hands it explicitly. That keeps the Network and the
Cluster's virtual `Clock` fully decoupled from each other.

Every fault this module injects -- reordering and probabilistic drops --
is driven by a single `random.Random` owned by the Network and seeded in
`__init__`. Given the same seed and the same sequence of calls, two
Networks make exactly the same random decisions in exactly the same order.

Partitions are a separate, non-random fault. Whether a link is cut is a
pure function of the current partition state, checked at the moment a
message would actually be delivered -- not when it was sent. A message
already in flight when a partition forms is dropped when it comes due,
exactly as it would be lost in a real network.

Every message that is actually delivered (never a dropped or partitioned
one) is also recorded into a bounded, per-`(src, dst)` link history --
oldest evicted first once a link's buffer is full. This exists so a
caller (`raft.sim.fuzz.Fuzzer`'s `STALE_REDELIVER` action) can later
redeliver an already-delivered message a destination has since moved
past, the way a duplicated or delayed-then-finally-arriving real packet
would. The Network stays payload-agnostic here too -- it records
`msg_id`, `src`, `dst`, `msg`, and the delivery time, and never inspects
`msg`'s contents to decide what's worth keeping; that interpretation
(which records are usefully "stale") belongs to whoever reads the
history back out, not to the Network.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, NamedTuple

Endpoint = Any  # any hashable node identifier

DEFAULT_HISTORY_DEPTH = 32
"""Default per-link delivered-message history depth; see `Network.__init__`.

Measured, not guessed: an initial estimate of 16 was checked empirically
against `Fuzzer`'s `STALE_REDELIVER` action (see `raft/sim/fuzz.py` and
BUGS.md) over the 50-seed and 1000-seed sweeps, comparing the candidates
this depth actually retains against an unbounded ground truth computed
from the identical delivery sequence. 16 measurably lost real candidates
to eviction before a draw could use them -- 90.9% survival over 50 seeds,
94.2% over 1000 (11 draws over 1000 seeds found zero candidates despite
some genuinely existing). 32 recovers effectively all of it: 99.9% and
99.97% survival respectively, zero fully-evicted draws in either sweep.
64 and 128 showed no further measurable gain over 32 in the 50-seed
measurement, so 32 is the point past which more depth stops buying
anything -- not an arbitrary round number one step up from the miss.
"""


def _require_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")


class _Pending(NamedTuple):
    delivery_time: int
    src: Endpoint
    dst: Endpoint
    msg: object
    msg_id: int


class DeliveryRecord(NamedTuple):
    """One message the Network has actually delivered, retained in a
    link's bounded history. `msg_id` is assigned once, at `send()` time,
    and is unique and monotonic for the lifetime of a `Network` instance
    -- it is how `recall()` finds this exact delivery again later,
    deterministically, with no RNG involved.
    """

    msg_id: int
    src: Endpoint
    dst: Endpoint
    msg: object
    delivered_at: int


class Network:
    """A seeded, fault-injecting bus for opaque messages between endpoints.

    The Network holds no `Clock` and has no notion of wall-clock time.
    Every method that needs "now" takes it as a plain integer, supplied
    by whatever is driving the simulation.
    """

    def __init__(self, seed: int, history_depth: int = DEFAULT_HISTORY_DEPTH) -> None:
        if history_depth < 1:
            raise ValueError("history_depth must be >= 1")

        self._rng = random.Random(seed)
        self._pending: list[_Pending] = []
        self._delays: dict[tuple[Endpoint, Endpoint], int] = {}
        self._drop_rates: dict[tuple[Endpoint, Endpoint], float] = {}
        self._partition: tuple[frozenset[Endpoint], frozenset[Endpoint]] | None = None
        self._next_msg_id = 0
        self._history_depth = history_depth
        self._history: dict[tuple[Endpoint, Endpoint], deque[DeliveryRecord]] = {}

    def send(self, src: Endpoint, dst: Endpoint, msg: object, now: int) -> None:
        """Hand `msg` to the network to route from `src` to `dst`.

        `now` is the current virtual time, supplied by the caller. The
        message may be lost immediately per `set_drop_rate` -- a
        partition is deliberately *not* checked here, only at actual
        delivery time, since a partition that forms after this call must
        still be able to catch the message in flight. Every call assigns
        a fresh `msg_id`, whether this is a node's original send or a
        deliberate redelivery of something already delivered before --
        each transit attempt is its own event.
        """
        _require_int(now, "now")
        p = self._drop_rates.get((src, dst), 0.0)
        if p > 0.0 and self._rng.random() < p:
            return  # lost on the wire, silently, exactly like a real drop
        delay = self._delays.get((src, dst), 0)
        msg_id = self._next_msg_id
        self._next_msg_id += 1
        self._pending.append(_Pending(now + delay, src, dst, msg, msg_id))

    def deliver_next(self, now: int) -> tuple[Endpoint, Endpoint, object] | None:
        """Deliver one message due at or before `now`, or return `None`.

        If several messages are due, which one is delivered is chosen at
        random -- delivery order on a link is never guaranteed, so
        reordering is possible on every call. A due message whose link is
        currently partitioned is dropped silently as a side effect of
        this call: it is removed from the queue, never returned, and
        never raises. Return shape is unchanged from before per-message
        ids existed -- callers that only need `(src, dst, msg)` are
        unaffected; the assigned id is retained internally and reachable
        through `recall()` and `history_by_link()`, not through this
        return value.
        """
        _require_int(now, "now")
        due = [i for i, p in enumerate(self._pending) if p.delivery_time <= now]
        deliverable = [
            i for i in due if not self._is_cut(self._pending[i].src, self._pending[i].dst)
        ]
        dropped = [i for i in due if i not in deliverable]

        result: tuple[Endpoint, Endpoint, object] | None = None
        chosen = self._rng.choice(deliverable) if deliverable else None
        if chosen is not None:
            entry = self._pending[chosen]
            result = (entry.src, entry.dst, entry.msg)
            link = (entry.src, entry.dst)
            self._history.setdefault(link, deque(maxlen=self._history_depth)).append(
                DeliveryRecord(entry.msg_id, entry.src, entry.dst, entry.msg, now)
            )

        to_remove = dropped + ([chosen] if chosen is not None else [])
        for i in sorted(to_remove, reverse=True):
            del self._pending[i]

        return result

    def recall(self, msg_id: int) -> DeliveryRecord | None:
        """Look up a previously delivered message by its `msg_id`.

        Deterministic: a plain search over retained history, no RNG. May
        return `None` for a real, past id if it has since been evicted
        from its link's bounded history -- there is no separate all-time
        record kept anywhere.
        """
        for records in self._history.values():
            for record in records:
                if record.msg_id == msg_id:
                    return record
        return None

    def history_by_link(self) -> dict[tuple[Endpoint, Endpoint], list[DeliveryRecord]]:
        """A snapshot of the retained delivery history, per `(src, dst)`
        link, oldest first. A plain list copy per link -- mutating it has
        no effect on the Network's own retained history.
        """
        return {link: list(records) for link, records in self._history.items()}

    def partition(self, group_a: set[Endpoint], group_b: set[Endpoint]) -> None:
        """Cut every link between `group_a` and `group_b`, in both directions.

        Nodes within the same group can still reach each other. This does
        not touch already-queued messages -- it only changes what happens
        the next time each one comes up for delivery.
        """
        a, b = frozenset(group_a), frozenset(group_b)
        if a & b:
            raise ValueError("group_a and group_b must be disjoint")
        self._partition = (a, b)

    def heal(self) -> None:
        """Remove any active partition, restoring delivery on every link."""
        self._partition = None

    def set_delay(self, src: Endpoint, dst: Endpoint, ms: int) -> None:
        """Set the delivery delay, in integer milliseconds, for src -> dst."""
        _require_int(ms, "ms")
        if ms < 0:
            raise ValueError("delay cannot be negative")
        self._delays[(src, dst)] = ms

    def set_drop_rate(self, src: Endpoint, dst: Endpoint, p: float) -> None:
        """Set the probability, in [0, 1], that a src -> dst send is lost."""
        if not 0.0 <= p <= 1.0:
            raise ValueError("drop rate must be between 0 and 1")
        self._drop_rates[(src, dst)] = p

    def pending(self) -> list[tuple[Endpoint, Endpoint, object, int]]:
        """Return a snapshot of queued, undelivered messages.

        Each entry is `(src, dst, msg, delivery_time)`. The order of the
        list reflects internal bookkeeping, not delivery order.
        """
        return [(p.src, p.dst, p.msg, p.delivery_time) for p in self._pending]

    def _is_cut(self, src: Endpoint, dst: Endpoint) -> bool:
        if self._partition is None:
            return False
        a, b = self._partition
        return (src in a and dst in b) or (src in b and dst in a)
