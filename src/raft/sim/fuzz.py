"""A randomized scheduler that drives a `Cluster` through a hostile,
reproducible sequence of events, checking Raft's safety properties after
every one.

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
"""

from __future__ import annotations

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
    actually took effect.
    """

    step: int
    action: Action
    detail: str


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
        self.cluster: Cluster[RaftClusterNode] = Cluster(
            n=n, seed=seed, node_factory=raft_node_factory(n, seed)
        )
        self.trace: list[TraceEntry] = []

        self._rng = random.Random(seed)
        self._max_crashed = (n - 1) // 2
        self._partition_active = False
        self._heal_at = 0
        self._append_only_history: dict[str, list[LogEntry]] = {}

    def run(self) -> None:
        """Run all `steps` fuzzer iterations, checking invariants after each.

        Raises `SafetyViolation` (with `.seed`, `.step`, and `.trace` all
        set) the moment a check fails, without running any further steps.
        """
        for step_index in range(self.steps):
            action = self._choose_action()
            detail = self._perform(action, step_index)
            self.trace.append(TraceEntry(step=step_index, action=action, detail=detail))
            self._check_invariants(step_index)

    def _choose_action(self) -> Action:
        actions = list(Action)
        weights = [self.weights.get(a, 0.0) for a in actions]
        return self._rng.choices(actions, weights=weights, k=1)[0]

    def _perform(self, action: Action, step_index: int) -> str:
        if action is Action.STEP:
            return self._do_step()
        if action is Action.CRASH:
            return self._do_crash()
        if action is Action.RESTART:
            return self._do_restart()
        if action is Action.PARTITION:
            return self._do_partition(step_index)
        if action is Action.HEAL:
            return self._do_heal(step_index)
        if action is Action.ADVANCE_CLOCK:
            return self._do_advance_clock()
        raise AssertionError(f"unhandled action: {action!r}")  # pragma: no cover

    def _do_step(self) -> str:
        happened = self.cluster.step()
        return "stepped" if happened else "idle (nothing pending)"

    def _do_crash(self) -> str:
        live_ids = [nid for nid in self.cluster.node_ids() if self.cluster.get_node(nid)]
        crashed_count = len(self.cluster.node_ids()) - len(live_ids)
        if not live_ids:
            return "skipped (no live node to crash)"
        if crashed_count >= self._max_crashed:
            return f"skipped (crash cap of {self._max_crashed} already reached)"
        node_id = self._rng.choice(live_ids)
        self.cluster.crash(node_id)
        return f"crashed node {node_id}"

    def _do_restart(self) -> str:
        crashed_ids = [nid for nid in self.cluster.node_ids() if self.cluster.get_node(nid) is None]
        if not crashed_ids:
            return "skipped (no crashed node to restart)"
        node_id = self._rng.choice(crashed_ids)
        self.cluster.restart(node_id)
        return f"restarted node {node_id}"

    def _do_partition(self, step_index: int) -> str:
        if self._partition_active:
            return "skipped (a partition is already active)"
        all_ids = self.cluster.node_ids()
        if len(all_ids) < 2:
            return "skipped (need at least 2 nodes to partition)"

        shuffled = list(all_ids)
        self._rng.shuffle(shuffled)
        split = self._rng.randint(1, len(shuffled) - 1)  # both groups end up non-empty
        group_a, group_b = set(shuffled[:split]), set(shuffled[split:])
        self.cluster.partition(group_a, group_b)

        self._partition_active = True
        duration = self._rng.randint(*PARTITION_DURATION_RANGE)
        self._heal_at = step_index + duration
        return f"partitioned {sorted(group_a)} | {sorted(group_b)} for {duration} steps"

    def _do_heal(self, step_index: int) -> str:
        if not self._partition_active:
            return "skipped (no active partition)"
        if step_index < self._heal_at:
            return f"skipped (partition persists until step {self._heal_at})"
        self.cluster.heal()
        self._partition_active = False
        return "healed"

    def _do_advance_clock(self) -> str:
        now = self.cluster.clock.now()
        next_event = self.cluster.next_event_time()

        if next_event is None:
            ms = self._rng.randint(*IDLE_ADVANCE_RANGE_MS)
        else:
            headroom = next_event - now
            if headroom <= 0:
                return "skipped (already at or past the next due event)"
            ms = self._rng.randint(1, headroom)

        self.cluster.clock.advance(ms)
        return f"advanced clock by {ms}ms (now={self.cluster.clock.now()})"

    def _check_invariants(self, step_index: int) -> None:
        try:
            check_election_safety(self.cluster)
            check_leader_append_only(self.cluster, self._append_only_history)
            check_log_matching(self.cluster)
        except SafetyViolation as violation:
            violation.step = step_index
            violation.trace = tuple(self.trace)
            raise
