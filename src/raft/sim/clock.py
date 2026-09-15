"""A deterministic virtual clock for simulation.

Time here is purely a logical construct: an integer count of milliseconds
that only moves when something explicitly advances it. Nothing in this
module reads the wall clock, and nothing here is a float. Two runs that
make the same calls in the same order fire the same callbacks in the same
order, every time -- that determinism is the whole point.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable
from typing import NewType

Handle = NewType("Handle", int)
"""Opaque token returned by `Clock.schedule`, used to `Clock.cancel` it."""

_Callback = Callable[[], None]


def _require_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int (milliseconds), got {type(value).__name__}")


class Clock:
    """A virtual clock with millisecond-integer time and deterministic ordering.

    Callbacks scheduled for the same instant fire in the order they were
    scheduled (insertion order). Scheduling a callback in the past is not
    an error and does not fire it immediately: it fires the next time the
    clock is advanced past (or to) the current time.
    """

    def __init__(self) -> None:
        self._now: int = 0
        self._seq = itertools.count()
        # Min-heap of (fire_time, handle). `handle` is monotonically
        # increasing in scheduling order, so it doubles as the tiebreaker
        # that gives same-instant callbacks insertion-order firing.
        self._heap: list[tuple[int, int]] = []
        self._callbacks: dict[int, _Callback] = {}

    def now(self) -> int:
        """Return the current virtual time in milliseconds."""
        return self._now

    def schedule(self, at: int, callback: _Callback) -> Handle:
        """Schedule `callback` to fire at virtual time `at` (milliseconds).

        `at` may be less than `now()`; such a callback is not dropped and
        is not fired immediately -- it fires on the next `advance` or
        `run_until` call, exactly as if it had been due all along.
        """
        _require_int(at, "at")
        handle = Handle(next(self._seq))
        self._callbacks[handle] = callback
        heapq.heappush(self._heap, (at, handle))
        return handle

    def cancel(self, handle: Handle) -> None:
        """Cancel a scheduled callback so it will never fire.

        A no-op if `handle` has already fired, already been cancelled, or
        is the handle of the callback currently executing -- so it is
        always safe to call from inside a callback, including on itself.
        """
        self._callbacks.pop(handle, None)

    def advance(self, ms: int) -> None:
        """Move virtual time forward by `ms` milliseconds, firing due callbacks.

        Equivalent to `run_until(now() + ms)`.
        """
        _require_int(ms, "ms")
        if ms < 0:
            raise ValueError("advance() cannot move time backward")
        self.run_until(self._now + ms)

    def run_until(self, t: int) -> None:
        """Advance virtual time to `t`, firing every callback due at or before it.

        Callbacks due at the same instant fire in the order they were
        scheduled. A callback that schedules another callback at or
        before `t` causes that new callback to fire too, before this call
        returns. `t` must not be before the current time.
        """
        _require_int(t, "t")
        if t < self._now:
            raise ValueError("run_until() cannot move time backward")
        while self._heap and self._heap[0][0] <= t:
            fire_time, handle = heapq.heappop(self._heap)
            callback = self._callbacks.pop(handle, None)
            if callback is None:
                continue  # cancelled before it got a chance to fire
            self._now = fire_time
            callback()
        self._now = t
