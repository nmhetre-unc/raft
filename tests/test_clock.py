import pytest

from raft.sim.clock import Clock


def test_callbacks_fire_in_time_order() -> None:
    clock = Clock()
    fired: list[int] = []
    clock.schedule(30, lambda: fired.append(30))
    clock.schedule(10, lambda: fired.append(10))
    clock.schedule(20, lambda: fired.append(20))

    clock.advance(100)

    assert fired == [10, 20, 30]


def test_same_time_callbacks_fire_in_insertion_order() -> None:
    clock = Clock()
    fired: list[str] = []
    clock.schedule(5, lambda: fired.append("first"))
    clock.schedule(5, lambda: fired.append("second"))
    clock.schedule(5, lambda: fired.append("third"))

    clock.advance(5)

    assert fired == ["first", "second", "third"]


def test_cancelled_callbacks_never_fire() -> None:
    clock = Clock()
    fired: list[str] = []
    clock.schedule(10, lambda: fired.append("kept"))
    handle = clock.schedule(10, lambda: fired.append("cancelled"))
    clock.cancel(handle)

    clock.advance(10)

    assert fired == ["kept"]


def test_schedule_in_past_still_fires() -> None:
    clock = Clock()
    clock.advance(50)
    fired: list[int] = []

    clock.schedule(10, lambda: fired.append(10))

    assert fired == []  # not fired immediately

    clock.advance(1)

    assert fired == [10]


def test_advance_past_several_callbacks_fires_all_in_order() -> None:
    clock = Clock()
    fired: list[int] = []
    for t in (40, 10, 30, 20, 5):
        clock.schedule(t, lambda t=t: fired.append(t))

    clock.advance(1000)

    assert fired == [5, 10, 20, 30, 40]


def test_cancel_from_inside_callback_is_safe() -> None:
    clock = Clock()
    fired: list[str] = []
    other_handle = clock.schedule(10, lambda: fired.append("other"))

    def cancel_other() -> None:
        fired.append("canceller")
        clock.cancel(other_handle)

    # Runs before "other" (t=5 < t=10), so it cancels "other" while that
    # callback is still pending -- the case this test exists to cover.
    clock.schedule(5, cancel_other)

    clock.advance(10)

    assert fired == ["canceller"]


def test_cancel_self_from_inside_callback_is_safe() -> None:
    clock = Clock()
    fired: list[str] = []
    self_handle_box: list[object] = []

    def cancel_self() -> None:
        fired.append("ran")
        clock.cancel(self_handle_box[0])  # type: ignore[arg-type]

    handle = clock.schedule(10, cancel_self)
    self_handle_box.append(handle)

    clock.advance(10)

    assert fired == ["ran"]


def test_now_reflects_time_during_firing_and_after_advance() -> None:
    clock = Clock()
    seen: list[int] = []
    clock.schedule(15, lambda: seen.append(clock.now()))

    clock.advance(100)

    assert seen == [15]
    assert clock.now() == 100


def test_run_until_fires_callbacks_scheduled_by_other_callbacks() -> None:
    clock = Clock()
    fired: list[int] = []

    def schedule_next() -> None:
        fired.append(10)
        clock.schedule(20, lambda: fired.append(20))

    clock.schedule(10, schedule_next)

    clock.run_until(20)

    assert fired == [10, 20]


def test_cancelling_already_fired_handle_is_a_noop() -> None:
    clock = Clock()
    fired: list[int] = []
    handle = clock.schedule(5, lambda: fired.append(5))

    clock.advance(5)
    clock.cancel(handle)  # already fired; must not raise or affect anything

    assert fired == [5]


def test_schedule_rejects_float_time() -> None:
    clock = Clock()
    with pytest.raises(TypeError):
        clock.schedule(1.5, lambda: None)  # type: ignore[arg-type]


def test_advance_rejects_negative_ms() -> None:
    clock = Clock()
    with pytest.raises(ValueError, match="backward"):
        clock.advance(-1)


def test_run_until_rejects_moving_time_backward() -> None:
    clock = Clock()
    clock.advance(10)
    with pytest.raises(ValueError, match="backward"):
        clock.run_until(5)
