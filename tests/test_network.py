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


# -- Delivery history (msg_id / recall / history_by_link) --
#
# Added for Fuzzer's STALE_REDELIVER action. deliver_next() and pending()
# keep their exact pre-existing return shapes throughout (every assertion
# above this section is unmodified and still passes) -- history tracking
# is purely additive, internal bookkeeping plus new, separate accessors.


def test_deliver_next_return_shape_is_unchanged_by_history_tracking() -> None:
    net = Network(seed=1)
    net.send("a", "b", "hello", now=0)

    result = net.deliver_next(now=0)

    assert result == ("a", "b", "hello")  # still a plain 3-tuple


def test_recall_finds_a_delivered_message_by_its_id() -> None:
    net = Network(seed=1)
    net.send("a", "b", "first", now=0)
    net.send("a", "b", "second", now=0)
    net.deliver_next(now=0)
    net.deliver_next(now=0)

    history = net.history_by_link()[("a", "b")]
    assert [record.msg for record in history] == ["first", "second"]  # oldest first

    first_record = history[0]
    recalled = net.recall(first_record.msg_id)
    assert recalled == first_record
    assert recalled is not None
    assert recalled.src == "a"
    assert recalled.dst == "b"
    assert recalled.msg == "first"


def test_recall_returns_none_for_an_id_never_assigned() -> None:
    net = Network(seed=1)
    net.send("a", "b", "hello", now=0)
    net.deliver_next(now=0)

    assert net.recall(999) is None


def test_recall_returns_none_for_a_message_still_pending_not_yet_delivered() -> None:
    # History is populated on DELIVERY, not on send -- a queued-but-not-
    # yet-delivered message has no history entry to recall, by design
    # (it isn't "already delivered and superseded" yet; it hasn't been
    # delivered at all).
    net = Network(seed=1)
    net.set_delay("a", "b", 10)
    net.send("a", "b", "hello", now=0)

    assert net.pending() != []
    assert net.recall(0) is None


def test_recall_returns_none_for_a_dropped_message() -> None:
    net = Network(seed=1)
    net.partition({"a"}, {"b"})
    net.send("a", "b", "hello", now=0)
    net.deliver_next(now=0)  # dropped, per the partition -- never delivered

    assert net.recall(0) is None


def test_msg_id_is_assigned_monotonically_per_send_call() -> None:
    net = Network(seed=1)
    net.send("a", "b", "one", now=0)
    net.send("a", "b", "two", now=0)
    net.send("a", "b", "three", now=0)

    ids = []
    for _ in range(3):
        src, dst, msg = net.deliver_next(now=0)  # type: ignore[misc]
        ids.append(net.history_by_link()[(src, dst)][-1].msg_id)

    assert ids == sorted(ids)
    assert len(set(ids)) == 3  # all distinct


def test_history_by_link_snapshot_does_not_expose_internal_state() -> None:
    net = Network(seed=1)
    net.send("a", "b", "hello", now=0)
    net.deliver_next(now=0)

    snapshot = net.history_by_link()
    snapshot[("a", "b")].clear()  # mutate the returned copy

    assert len(net.history_by_link()[("a", "b")]) == 1  # internal state untouched


def test_history_evicts_oldest_once_a_links_buffer_is_full() -> None:
    net = Network(seed=1, history_depth=2)
    net.send("a", "b", "one", now=0)
    net.send("a", "b", "two", now=0)
    net.send("a", "b", "three", now=0)
    for _ in range(3):
        net.deliver_next(now=0)

    retained = net.history_by_link()[("a", "b")]
    assert [r.msg for r in retained] == ["two", "three"]  # "one" evicted

    assert net.recall(0) is None  # "one"'s id, evicted
    assert net.recall(2) is not None  # "three"'s id, still retained


def test_history_depth_rejects_less_than_one() -> None:
    with pytest.raises(ValueError, match="history_depth"):
        Network(seed=1, history_depth=0)
