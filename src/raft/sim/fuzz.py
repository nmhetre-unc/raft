"""A randomized scheduler that drives a `Cluster` through a hostile,
reproducible sequence of events, checking Raft's safety properties after
every one -- plus the machinery to replay a recorded run exactly, and to
shrink a failing one down to the smallest sequence that still fails.

`Fuzzer` owns a `Cluster[RaftClusterNode]` built internally from
`raft_node_factory` -- it isn't generic like `Cluster` itself, since
running the invariant checks is the whole point and those are
Raft-specific. Every action -- stepping the cluster, crashing or
restarting a node, forming or healing a partition, jumping the clock
forward -- is chosen by one seeded `random.Random`, so the entire
sequence of events (and any violation it finds) reproduces exactly from
`(n, seed, steps)` alone.

Two constraints keep a run meaningful instead of degenerate, both
enforced structurally rather than left to chance:

- Crashing is capped at `(n - 1) // 2` live nodes down at once. Past
  that, no majority is reachable and Raft is *supposed* to make no
  progress -- a run that let the fuzzer crash a majority would spend all
  its time testing that correctly-inert state, not anything interesting.
- A partition, once formed, persists for a randomly drawn number of
  steps before a `heal` draw is allowed to take effect. Without this,
  `heal` could immediately undo whatever `partition` just did on the very
  next draw, on average, and the cluster would rarely see partitioned
  behavior for long enough to actually react to it.

Replay and shrinking (`replay`, `shrink`) both work on `list[TraceEntry]`,
not `list[Action]`. A bare action *kind* -- "crash", with no word on
which node -- isn't enough to replay anything exactly: which node
crashed, which two groups a partition split into, how many milliseconds
the clock jumped are all essential, recorded state, not something a
re-seeded generator could regenerate on demand (a fresh `random.Random`
draws from the same seed, but replay skips the action-*selection* draws
entirely, since the trace already says what happened -- so its later
draws land in different positions in the stream and produce different
values). `TraceEntry` already carries a `detail` string for humans, and
each action here now writes its resolved parameters (`node_id`,
`group_a`/`group_b`, `ms`) into the same record, so `Fuzzer.trace` -- the
thing you actually have on hand after a run -- is directly replayable
with no extra bookkeeping.
"""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from enum import Enum

from raft.invariants import (
    SafetyViolation,
    check_election_safety,
    check_leader_append_only,
    check_log_matching,
)
from raft.node import RaftClusterNode, raft_node_factory
from raft.sim.cluster import Cluster
from raft.storage import LogEntry


class Action(Enum):
    STEP = "step"
    CRASH = "crash"
    RESTART = "restart"
    PARTITION = "partition"
    HEAL = "heal"
    ADVANCE_CLOCK = "advance_clock"


DEFAULT_WEIGHTS: dict[Action, float] = {
    Action.STEP: 70.0,
    Action.CRASH: 7.0,
    Action.RESTART: 7.0,
    Action.PARTITION: 6.0,
    Action.HEAL: 8.0,
    Action.ADVANCE_CLOCK: 2.0,
}

# How long (in fuzzer steps) a newly-formed partition refuses to heal.
PARTITION_DURATION_RANGE = (3, 20)

# Fallback range for advance_clock when nothing is scheduled at all (only
# possible transiently, since the crash cap always leaves >=1 node alive).
IDLE_ADVANCE_RANGE_MS = (1, 50)


@dataclass(frozen=True)
class TraceEntry:
    """One fuzzer step: which action was drawn, and what actually happened.

    `detail` always starts with "skipped" when the drawn action turned
    out to be ineligible (nothing to crash, a partition still in its
    persistence window, ...) and was a no-op; anything else means it
    actually took effect. The remaining fields hold whichever resolved
    parameters that action needed -- `node_id` for CRASH/RESTART,
    `group_a`/`group_b`/`duration` for PARTITION, `ms` for
    ADVANCE_CLOCK -- and stay `None` both for STEP/HEAL (never
    parameterized; HEAL's outcome is fully determined by state this
    record's predecessors already pin down) and for any no-op occurrence
    of an action that normally would carry one.
    """

    step: int
    action: Action
    detail: str
    node_id: int | None = None
    group_a: frozenset[int] | None = None
    group_b: frozenset[int] | None = None
    duration: int | None = None
    ms: int | None = None


class Fuzzer:
    """Drives an n-node `Cluster` through `steps` randomized events."""

    def __init__(
        self,
        n: int,
        seed: int,
        steps: int,
        weights: dict[Action, float] | None = None,
    ) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        if steps < 0:
            raise ValueError("steps must be >= 0")
        effective_weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        if sum(effective_weights.get(a, 0.0) for a in Action) <= 0:
            raise ValueError("at least one action must have a positive weight")

        self.n = n
        self.seed = seed
        self.steps = steps
        self.weights = effective_weights
        self.trace: list[TraceEntry] = []

        self._rng = random.Random(seed)
        self._max_crashed = (n - 1) // 2
        self.cluster: Cluster[RaftClusterNode]
        self._reset()

    def run(self) -> None:
        """Run all `steps` fuzzer iterations, checking invariants after each.

        Raises `SafetyViolation` (with `.seed`, `.step`, and `.trace` all
        set) the moment a check fails, without running any further steps.
        """
        for step_index in range(self.steps):
            action = self._choose_action()
            entry = self._perform(action, step_index)
            self.trace.append(entry)
            self._check_invariants(step_index)

    def replay(self, trace: list[TraceEntry]) -> None:
        """Execute exactly this recorded sequence against a fresh cluster.

        Nothing is drawn from any RNG: every action's parameters come
        from `trace` itself. Resets `self.cluster` and `self.trace`
        first, then re-runs `trace` entry by entry, re-indexing steps
        from 0 regardless of what `.step` the entries originally carried
        (shrinking removes entries, so those numbers would have gaps
        anyway) -- what matters is the *order* and the *resolved
        parameters*, not the original step numbering.

        If `trace` reproduces a violation, this raises `SafetyViolation`
        exactly as `run()` would, and it must be the identical violation
        (same `.reason`) every time -- that's the whole premise `shrink`
        relies on. If it doesn't, this returns normally.
        """
        self._reset()
        for step_index, recorded in enumerate(trace):
            entry = self._perform(recorded.action, step_index, forced=recorded)
            self.trace.append(entry)
            self._check_invariants(step_index)

    def shrink(self, trace: list[TraceEntry]) -> list[TraceEntry]:
        """Delta-debug `trace` down to the shortest sequence that still fails.

        Classic ddmin: try removing progressively smaller contiguous
        chunks, keep any removal that still reproduces the identical
        violation (matched by `SafetyViolation.reason`, since `.step`
        legitimately shifts once the trace has been edited), and give up
        shrinking a chunk size only once a full sweep at that size makes
        no more progress. The result is 1-minimal: no single remaining
        entry can be dropped without losing the repro.

        Raises `ValueError` if `trace` doesn't reproduce a violation in
        the first place -- there is nothing to shrink toward, and
        returning some arbitrary trace instead would silently discard
        that this trace never failed.
        """
        target = self._signature(trace)
        if target is None:
            raise ValueError("trace does not reproduce a violation; nothing to shrink")

        current = list(trace)
        chunk_size = max(1, len(current) // 2)
        while True:
            progressed = False
            start = 0
            while start < len(current):
                end = min(start + chunk_size, len(current))
                candidate = current[:start] + current[end:]
                if candidate and self._signature(candidate) == target:
                    current = candidate
                    progressed = True
                    # don't advance `start` -- re-try the same position
                    # against the now-shorter list
                else:
                    start += chunk_size

            if progressed:
                continue
            if chunk_size == 1:
                break
            chunk_size = max(1, chunk_size // 2)

        # `current` is guaranteed (by the loop above) to still reproduce
        # the violation, so this replay always raises -- deliberately: it
        # exists only to leave self.trace holding a properly re-indexed
        # (0, 1, 2, ...) minimal trace rather than fragments of the
        # original numbering, and self.cluster matching it. shrink()
        # returns a value; it doesn't raise.
        with contextlib.suppress(SafetyViolation):
            self.replay(current)
        return list(self.trace)

    def _signature(self, trace: list[TraceEntry]) -> tuple[str, str] | None:
        """None if `trace` passes; otherwise a value identifying which violation."""
        try:
            self.replay(trace)
        except SafetyViolation as violation:
            return (type(violation).__name__, violation.reason)
        except Exception:
            # A candidate reduced by shrink() can hit a structural
            # inconsistency replay() itself doesn't guard against (e.g. a
            # forced clock jump that overshoots a due event only because
            # some earlier, now-removed action would have consumed it
            # first). That candidate simply isn't a valid reproduction --
            # not a reason for shrink() itself to blow up.
            return None
        return None

    def _reset(self) -> None:
        self.cluster = Cluster(
            n=self.n, seed=self.seed, node_factory=raft_node_factory(self.n, self.seed)
        )
        self.trace = []
        self._partition_active = False
        self._heal_at = 0
        self._append_only_history: dict[str, list[LogEntry]] = {}

    def _choose_action(self) -> Action:
        actions = list(Action)
        weights = [self.weights.get(a, 0.0) for a in actions]
        return self._rng.choices(actions, weights=weights, k=1)[0]

    def _perform(
        self, action: Action, step_index: int, forced: TraceEntry | None = None
    ) -> TraceEntry:
        if action is Action.STEP:
            return self._do_step(step_index)
        if action is Action.CRASH:
            return self._do_crash(step_index, forced)
        if action is Action.RESTART:
            return self._do_restart(step_index, forced)
        if action is Action.PARTITION:
            return self._do_partition(step_index, forced)
        if action is Action.HEAL:
            return self._do_heal(step_index)
        if action is Action.ADVANCE_CLOCK:
            return self._do_advance_clock(step_index, forced)
        raise AssertionError(f"unhandled action: {action!r}")  # pragma: no cover

    def _do_step(self, step_index: int) -> TraceEntry:
        happened = self.cluster.step()
        detail = "stepped" if happened else "idle (nothing pending)"
        return TraceEntry(step=step_index, action=Action.STEP, detail=detail)

    def _apply_crash(self, node_id: int) -> str:
        self.cluster.crash(node_id)
        return f"crashed node {node_id}"

    def _do_crash(self, step_index: int, forced: TraceEntry | None) -> TraceEntry:
        if forced is not None:
            if forced.node_id is not None:
                detail = self._apply_crash(forced.node_id)
            else:
                detail = forced.detail
            return TraceEntry(
                step=step_index, action=Action.CRASH, detail=detail, node_id=forced.node_id
            )

        live_ids = [nid for nid in self.cluster.node_ids() if self.cluster.get_node(nid)]
        crashed_count = len(self.cluster.node_ids()) - len(live_ids)
        if not live_ids:
            detail = "skipped (no live node to crash)"
            return TraceEntry(step=step_index, action=Action.CRASH, detail=detail)
        if crashed_count >= self._max_crashed:
            detail = f"skipped (crash cap of {self._max_crashed} already reached)"
            return TraceEntry(step=step_index, action=Action.CRASH, detail=detail)

        node_id = self._rng.choice(live_ids)
        detail = self._apply_crash(node_id)
        return TraceEntry(step=step_index, action=Action.CRASH, detail=detail, node_id=node_id)

    def _apply_restart(self, node_id: int) -> str:
        self.cluster.restart(node_id)
        return f"restarted node {node_id}"

    def _do_restart(self, step_index: int, forced: TraceEntry | None) -> TraceEntry:
        if forced is not None:
            detail = (
                self._apply_restart(forced.node_id) if forced.node_id is not None else forced.detail
            )
            return TraceEntry(
                step=step_index, action=Action.RESTART, detail=detail, node_id=forced.node_id
            )

        crashed_ids = [nid for nid in self.cluster.node_ids() if self.cluster.get_node(nid) is None]
        if not crashed_ids:
            detail = "skipped (no crashed node to restart)"
            return TraceEntry(step=step_index, action=Action.RESTART, detail=detail)

        node_id = self._rng.choice(crashed_ids)
        detail = self._apply_restart(node_id)
        return TraceEntry(step=step_index, action=Action.RESTART, detail=detail, node_id=node_id)

    def _apply_partition(
        self, group_a: frozenset[int], group_b: frozenset[int], duration: int, step_index: int
    ) -> str:
        self.cluster.partition(set(group_a), set(group_b))
        self._partition_active = True
        self._heal_at = step_index + duration
        return f"partitioned {sorted(group_a)} | {sorted(group_b)} for {duration} steps"

    def _do_partition(self, step_index: int, forced: TraceEntry | None) -> TraceEntry:
        if forced is not None:
            if forced.group_a is None or forced.group_b is None or forced.duration is None:
                detail = forced.detail
            else:
                detail = self._apply_partition(
                    forced.group_a, forced.group_b, forced.duration, step_index
                )
            return TraceEntry(
                step=step_index,
                action=Action.PARTITION,
                detail=detail,
                group_a=forced.group_a,
                group_b=forced.group_b,
                duration=forced.duration,
            )

        if self._partition_active:
            detail = "skipped (a partition is already active)"
            return TraceEntry(step=step_index, action=Action.PARTITION, detail=detail)
        all_ids = self.cluster.node_ids()
        if len(all_ids) < 2:
            detail = "skipped (need at least 2 nodes to partition)"
            return TraceEntry(step=step_index, action=Action.PARTITION, detail=detail)

        shuffled = list(all_ids)
        self._rng.shuffle(shuffled)
        split = self._rng.randint(1, len(shuffled) - 1)  # both groups end up non-empty
        group_a = frozenset(shuffled[:split])
        group_b = frozenset(shuffled[split:])
        duration = self._rng.randint(*PARTITION_DURATION_RANGE)
        detail = self._apply_partition(group_a, group_b, duration, step_index)
        return TraceEntry(
            step=step_index,
            action=Action.PARTITION,
            detail=detail,
            group_a=group_a,
            group_b=group_b,
            duration=duration,
        )

    def _do_heal(self, step_index: int) -> TraceEntry:
        # Nothing to force: HEAL draws no parameters of its own. Its
        # outcome is fully pinned down by _partition_active/_heal_at,
        # both already faithfully reproduced by whatever PARTITION entry
        # (forced or drawn) came before this one in the same replay.
        if not self._partition_active:
            detail = "skipped (no active partition)"
        elif step_index < self._heal_at:
            detail = f"skipped (partition persists until step {self._heal_at})"
        else:
            self.cluster.heal()
            self._partition_active = False
            detail = "healed"
        return TraceEntry(step=step_index, action=Action.HEAL, detail=detail)

    def _apply_advance_clock(self, ms: int) -> str:
        self.cluster.clock.advance(ms)
        return f"advanced clock by {ms}ms (now={self.cluster.clock.now()})"

    def _do_advance_clock(self, step_index: int, forced: TraceEntry | None) -> TraceEntry:
        if forced is not None:
            if forced.ms is not None:
                detail = self._apply_advance_clock(forced.ms)
            else:
                detail = forced.detail
            return TraceEntry(
                step=step_index, action=Action.ADVANCE_CLOCK, detail=detail, ms=forced.ms
            )

        now = self.cluster.clock.now()
        next_event = self.cluster.next_event_time()
        if next_event is None:
            ms = self._rng.randint(*IDLE_ADVANCE_RANGE_MS)
        else:
            headroom = next_event - now
            if headroom <= 0:
                detail = "skipped (already at or past the next due event)"
                return TraceEntry(step=step_index, action=Action.ADVANCE_CLOCK, detail=detail)
            ms = self._rng.randint(1, headroom)

        detail = self._apply_advance_clock(ms)
        return TraceEntry(step=step_index, action=Action.ADVANCE_CLOCK, detail=detail, ms=ms)

    def _check_invariants(self, step_index: int) -> None:
        try:
            check_election_safety(self.cluster)
            check_leader_append_only(self.cluster, self._append_only_history)
            check_log_matching(self.cluster)
        except SafetyViolation as violation:
            violation.step = step_index
            violation.trace = tuple(self.trace)
            raise
