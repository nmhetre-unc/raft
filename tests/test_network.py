import pytest

from raft.sim.network import Network


def test_delivery_works_on_clean_link() -> None:
    net = Network(seed=1)

    net.send("a", "b", "hello", now=0)

    assert net.deliver_next(now=0) == ("a", "b", "hello")
    assert net.deliver_next(now=0) is None


def test_delayed_messages_arrive_after_their_delay() -> None:
    net = Network(seed=1)
    net.set_delay("a", "b", 10)

    net.send("a", "b", "hi", now=0)

    assert net.deliver_next(now=5) is None
    assert net.deliver_next(now=9) is None
    assert net.deliver_next(now=10) == ("a", "b", "hi")


def test_partitioned_messages_are_dropped_with_no_error() -> None:
    net = Network(seed=1)
    net.partition({"a"}, {"b"})

    net.send("a", "b", "msg", now=0)
    result = net.deliver_next(now=0)  # must not raise

    assert result is None
    assert net.pending() == []


def test_heal_restores_delivery() -> None:
    net = Network(seed=1)
    net.partition({"a"}, {"b"})
    net.heal()

    net.send("a", "b", "msg", now=0)

    assert net.deliver_next(now=0) == ("a", "b", "msg")


def _drop_pattern(seed: int, n: int = 50) -> list[bool]:
    """True at index i means the i-th send was queued; False means dropped."""
    net = Network(seed=seed)
    net.set_drop_rate("a", "b", 0.5)
    pattern = []
    prev_len = 0
    for i in range(n):
        net.send("a", "b", i, now=0)
        new_len = len(net.pending())
        pattern.append(new_len > prev_len)
        prev_len = new_len
    return pattern


def test_same_seed_reproduces_identical_drop_pattern() -> None:
    assert _drop_pattern(seed=7) == _drop_pattern(seed=7)


def test_different_seed_produces_a_different_drop_pattern() -> None:
    assert _drop_pattern(seed=7) != _drop_pattern(seed=8)


def test_pending_reflects_queued_undelivered_messages() -> None:
    net = Network(seed=1)
    net.set_delay("a", "b", 5)

    net.send("a", "b", "one", now=0)
    net.send("a", "c", "two", now=0)

    pending = net.pending()
    assert len(pending) == 2
    assert ("a", "b", "one", 5) in pending
    assert ("a", "c", "two", 0) in pending

    delivered = net.deliver_next(now=0)  # only "two" is due at t=0

    assert delivered == ("a", "c", "two")
    assert net.pending() == [("a", "b", "one", 5)]


def test_message_sent_before_partition_delivered_after_is_still_dropped() -> None:
    net = Network(seed=1)
    net.set_delay("a", "b", 10)

    net.send("a", "b", "msg", now=0)  # in flight, link still clean
    net.partition({"a"}, {"b"})  # partition forms while message is in flight

    result = net.deliver_next(now=10)  # due now, but the link is cut

    assert result is None
    assert net.pending() == []  # gone, not stuck in the queue forever


def test_partition_only_cuts_links_between_the_two_groups() -> None:
    net = Network(seed=1)
    net.partition({"a", "b"}, {"c"})

    net.send("a", "b", "within-group", now=0)  # both in group_a
    net.send("a", "c", "cross-group", now=0)  # a in group_a, c in group_b

    assert net.deliver_next(now=0) == ("a", "b", "within-group")
    assert net.deliver_next(now=0) is None  # a -> c was dropped
    assert net.pending() == []


def test_partition_rejects_overlapping_groups() -> None:
    net = Network(seed=1)
    with pytest.raises(ValueError, match="disjoint"):
        net.partition({"a"}, {"a", "b"})


def test_reordering_is_possible_for_same_time_deliveries() -> None:
    # With enough same-instant candidates and a fixed seed, at least one
    # seed must deliver them out of send order -- that's what makes
    # reordering an available fault, not just a theoretical one.
    orders = set()
    for seed in range(200):
        net = Network(seed=seed)
        net.send("a", "b", "first", now=0)
        net.send("a", "b", "second", now=0)
        net.send("a", "b", "third", now=0)
        delivered = tuple(net.deliver_next(now=0)[2] for _ in range(3))  # type: ignore[index]
        orders.add(delivered)

    assert ("first", "second", "third") in orders
    assert len(orders) > 1  # some seed reordered them


def test_send_rejects_float_time() -> None:
    net = Network(seed=1)
    with pytest.raises(TypeError):
        net.send("a", "b", "msg", now=0.5)  # type: ignore[arg-type]


def test_set_delay_rejects_negative_ms() -> None:
    net = Network(seed=1)
    with pytest.raises(ValueError, match="negative"):
        net.set_delay("a", "b", -1)


def test_set_drop_rate_rejects_out_of_range_probability() -> None:
    net = Network(seed=1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        net.set_drop_rate("a", "b", 1.5)


def test_messages_are_never_inspected_only_routed() -> None:
    class Unopenable:
        def __eq__(self, other: object) -> bool:
            raise AssertionError("message contents must never be inspected")

        def __hash__(self) -> int:
            raise AssertionError("message contents must never be inspected")

    net = Network(seed=1)
    payload = Unopenable()

    net.send("a", "b", payload, now=0)
    src, dst, msg = net.deliver_next(now=0)  # type: ignore[misc]

    assert src == "a"
    assert dst == "b"
    assert msg is payload
