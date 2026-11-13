"""Linearizability checking for `raft.net.loadgen` histories.

The model is a single register per key, initially empty (`None`): a put
sets it, a get returns it. Linearizability is compositional, so the
history is partitioned by key and each key is checked on its own.

Each key's history is checked with the Wing & Gong search plus Lowe's
memoization of (linearized set, state) pairs -- the approach Porcupine
uses. Operations become call and return events on a doubly linked list
ordered by time; the search repeatedly linearizes some pending call
whose effect is legal in the current state, and backtracks when it
reaches the return of an operation it hasn't linearized yet. A cache of
every (linearized set, state) already explored bounds the work.

Rules for the history:

- An operation is the interval [start, end]. One that ends at or after
  another starts is concurrent with it; only end < start orders two.
- A get that failed (``ok=false``) carries no information and is dropped.
- A put with an unknown outcome (``ok=false``) may have taken effect at
  any point after its start, or never. Every put value is unique, which
  makes this exact to model: a put nobody ever reads can always be placed
  after everything else -- indistinguishable from never happening -- so it
  is dropped; a put somebody reads must take effect before its first
  reader, so its interval is cut off at the latest reader's end.
- A get returning a value no put on that key ever wrote is a violation on
  its own.

Scaling. The search is exponential in the number of *concurrent*
operations, not in the length of the history -- but the linked list, the
bitsets and the cache all grow with the history. So each key's history
is split at quiescent points, where every earlier operation has ended
before any later one starts; real-time order forces everything before
such a point to be linearized before everything after it. A segment can
end in more than one state (when concurrent puts are never read), so the
segments are searched depth-first: the first state a segment can end in
is carried forward, and the segment is asked for another only if a later
segment fails from it. With the unknown-outcome rule above, every interval
is finite, so a history splits into segments of a few operations each.

Cost. Checking linearizability is NP-complete in general, and this
search is exponential in the number of operations concurrent on one key.
loadgen's clients are closed-loop -- one operation at a time each -- so
that is at most the number of clients, and in practice one or two. A
linearizable history costs about one first-success search per segment;
only a violation makes it explore every alternative, and only within the
segments leading up to it.

On a violation, the counterexample is minimized: from everything up to
and including the failing segment, operations are removed (chunks first,
then one at a time) for as long as the rest still fails. Removing a put
also removes the gets that read it, so the counterexample never fails
merely because a read lost its write.

    python -m raft.net.linearizability bench/results/<name>.history.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from raft.net.loadgen import OpResult

State = str | None


@dataclass(frozen=True)
class Op:
    """One operation, as the checker sees it."""

    client: str
    kind: str  # "put" or "get"
    key: str
    value: str | None
    start: float
    end: float  # math.inf for a put with an unknown outcome
    ok: bool
    node: str | None = None

    @classmethod
    def of(cls, op: OpResult | Mapping[str, Any]) -> Op:
        if isinstance(op, OpResult):
            return cls(op.client, op.kind, op.key, op.value, op.start, op.end, op.ok, op.node)
        return cls(
            client=op["client"],
            kind=op["kind"],
            key=op["key"],
            value=op["value"],
            start=op["start"],
            end=op["end"],
            ok=op["ok"],
            node=op.get("node"),
        )


@dataclass(frozen=True)
class Violation:
    key: str
    reason: str
    ops: tuple[Op, ...]  # the minimal counterexample, sorted by start


@dataclass(frozen=True)
class Result:
    operations: int  # in the history, before anything was dropped
    checked: int  # operations that went into the search
    keys: int
    segments: int
    seconds: float
    violation: Violation | None

    @property
    def ok(self) -> bool:
        return self.violation is None

    def summary(self) -> str:
        if self.violation is None:
            return f"{self.operations} operations, linearizable"
        return f"{self.operations} operations, NOT linearizable\n{format_violation(self.violation)}"


def load_history(path: str | Path) -> list[Op]:
    with Path(path).open() as f:
        return [Op.of(json.loads(line)) for line in f if line.strip()]


def check(history: Iterable[OpResult | Mapping[str, Any] | Op]) -> Result:
    """Check a whole history; see the module docstring."""
    started = time.perf_counter()
    ops = [op if isinstance(op, Op) else Op.of(op) for op in history]
    by_key: dict[str, list[Op]] = defaultdict(list)
    for op in ops:
        by_key[op.key].append(op)
    checked = segments = 0
    violation = None
    for key in sorted(by_key):
        outcome = _check_key(by_key[key])
        checked += outcome.checked
        segments += outcome.segments
        if outcome.failure is not None:
            violation = _minimize(key, outcome.failure)
            break
    elapsed = time.perf_counter() - started
    return Result(len(ops), checked, len(by_key), segments, elapsed, violation)


def format_violation(violation: Violation) -> str:
    t0 = min(op.start for op in violation.ops)

    def ms(t: float) -> str:
        return "?" if math.isinf(t) else f"{(t - t0) * 1000:.1f}"

    lines = [f"key {violation.key!r}: {violation.reason}"]
    for op in violation.ops:
        arrow = "<-" if op.kind == "get" else ":="
        outcome = "ok" if op.ok else f"unknown outcome (client gave up at {ms(op.end)} ms)"
        end = op.end if op.ok else math.inf  # it may take effect any time after its start
        node = f" via {op.node}" if op.node else ""
        lines.append(
            f"  {op.client:>6} {op.kind} {arrow} {op.value!r:<14} "
            f"[{ms(op.start)}, {ms(end)}] ms  {outcome}{node}"
        )
    return "\n".join(lines)


# -- One key --


@dataclass(frozen=True)
class _KeyOutcome:
    checked: int
    segments: int
    failure: tuple[str, list[Op]] | None  # reason, and the ops to minimize from


def _prepare(ops: Sequence[Op]) -> tuple[list[tuple[Op, Op]], Op | None]:
    """Apply the history rules: drop failed gets and unread unknown puts,
    cut a read unknown put off at its last reader. Returns (prepared,
    original) pairs sorted by start, and the first get of a value never
    written, if any."""
    written = {op.value for op in ops if op.kind == "put"}
    last_read: dict[str | None, float] = {}
    for op in ops:
        if op.kind == "get" and op.ok:
            if op.value is not None and op.value not in written:
                return [], op
            last_read[op.value] = max(last_read.get(op.value, -math.inf), op.end)
    prepared = []
    for op in ops:
        if op.ok:
            prepared.append((op, op))
        elif op.kind == "put" and op.value in last_read:
            prepared.append((replace(op, end=last_read[op.value]), op))
    prepared.sort(key=lambda pair: (pair[0].start, pair[0].end))
    return prepared, None


def _segments(ops: list[tuple[Op, Op]]) -> list[list[tuple[Op, Op]]]:
    """Split at quiescent points: every operation before ended strictly
    before any operation after starts."""
    out: list[list[tuple[Op, Op]]] = []
    latest_end = -math.inf
    for pair in ops:
        if pair[0].start > latest_end:
            out.append([])
        out[-1].append(pair)
        latest_end = max(latest_end, pair[0].end)
    return out


def _check_key(ops: Sequence[Op]) -> _KeyOutcome:
    """Depth-first over segments: take the first state a segment can end
    in and go on; ask the segment for another only if a later segment
    fails from it. A linearizable history costs one first-success search
    per segment; only a real failure explores every alternative.
    (segment, starting state) pairs already known to fail are skipped."""
    prepared, unwritten = _prepare(ops)
    if unwritten is not None:
        return _KeyOutcome(0, 0, ("read a value that no put ever wrote", [unwritten]))
    segments = [[op for op, _ in segment] for segment in _segments(prepared)]
    if not segments:
        return _KeyOutcome(0, 0, None)
    failed: set[tuple[int, State]] = set()
    stack: list[tuple[int, State, Iterator[State]]] = [(0, None, _successors(segments[0], None))]
    deepest = 0
    while stack:
        i, start, finals = stack[-1]
        deepest = max(deepest, i)
        final = next(finals, _EXHAUSTED)
        if final is _EXHAUSTED:
            failed.add((i, start))
            stack.pop()
        elif i + 1 == len(segments):
            return _KeyOutcome(len(prepared), len(segments), None)
        elif (i + 1, final) not in failed:
            stack.append((i + 1, final, _successors(segments[i + 1], final)))
    # No way through; minimize from everything up to the segment it got stuck on.
    upto = {id(op) for segment in segments[: deepest + 1] for op in segment}
    done = [original for op, original in prepared if id(op) in upto]
    reason = "no order of these operations is consistent with a register"
    return _KeyOutcome(len(done), deepest + 1, (reason, done))


_EXHAUSTED: Any = object()


def _successors(segment: Sequence[Op], state: State) -> Iterator[State]:
    """The states `segment` can end in from `state`. Most segments are a
    single operation, with exactly one outcome or none: answer those
    directly instead of setting up a search."""
    if len(segment) == 1:
        (op,) = segment
        if op.kind == "put":
            return iter((op.value,))
        return iter((state,) if op.value == state else ())
    return _final_states(segment, state)


def _final_states(ops: Sequence[Op], initial: State) -> Iterator[State]:
    """Yield, lazily and without repeats, each state the register can be
    in after some linearization of `ops` starting from `initial`.

    The Wing & Gong search with Lowe's memoization. Each time it has
    linearized everything it yields the state, then backtracks for more
    only if asked. Events live in a doubly linked list -- index `head` is
    the sentinel, -1 the end -- so linearizing an operation unlinks its
    call and return in O(1), and backtracking relinks them.
    """
    # At equal times calls sort first: touching intervals are concurrent.
    calls = [(op.start, 0, i) for i, op in enumerate(ops)]
    events = sorted(calls + [(op.end, 1, i) for i, op in enumerate(ops)])
    m = len(events)
    head = m
    nxt = [*range(1, m), -1, 0]
    prv = [head, *range(m - 1)]
    is_call = [kind == 0 for _, kind, _ in events]
    op_of = [i for _, _, i in events]
    return_of = [0] * len(ops)
    for position, (_, kind, i) in enumerate(events):
        if kind == 1:
            return_of[i] = position

    def unlink(e: int) -> None:
        nxt[prv[e]] = nxt[e]
        if nxt[e] != -1:
            prv[nxt[e]] = prv[e]

    def relink(e: int) -> None:
        nxt[prv[e]] = e
        if nxt[e] != -1:
            prv[nxt[e]] = e

    yielded: set[State] = set()
    seen: set[tuple[int, State]] = set()
    state = initial
    linearized = 0
    stack: list[tuple[int, State]] = []
    entry = nxt[head]
    while True:
        if nxt[head] == -1 or not is_call[entry]:
            if nxt[head] == -1 and state not in yielded:  # everything linearized
                yielded.add(state)
                yield state
            if not stack:  # a return we can't get past, or nothing left to try
                return
            entry, state = stack.pop()
            i = op_of[entry]
            linearized &= ~(1 << i)
            relink(return_of[i])
            relink(entry)
            entry = nxt[entry]
            continue
        i = op_of[entry]
        op = ops[i]
        if op.kind == "put":
            legal, new_state = True, op.value
        else:
            legal, new_state = op.value == state, state
        if legal:
            marked = linearized | (1 << i)
            if (marked, new_state) not in seen:
                seen.add((marked, new_state))
                stack.append((entry, state))
                state, linearized = new_state, marked
                unlink(entry)
                unlink(return_of[i])
                entry = nxt[head]
                continue
        entry = nxt[entry]


# -- Counterexamples --


def _fails(ops: Sequence[Op]) -> bool:
    return _check_key(ops).failure is not None


def _without(ops: list[Op], removed: Sequence[Op]) -> list[Op]:
    """`ops` minus `removed`, minus any get reading a removed put's value."""
    gone = set(map(id, removed))
    lost = {op.value for op in removed if op.kind == "put"}
    return [op for op in ops if id(op) not in gone and not (op.kind == "get" and op.value in lost)]


def _minimize(key: str, failure: tuple[str, list[Op]]) -> Violation:
    reason, ops = failure
    chunk = max(1, len(ops) // 2)
    while True:
        i = 0
        while i < len(ops):
            candidate = _without(ops, ops[i : i + chunk])
            if candidate and _fails(candidate):
                ops = candidate
            else:
                i += chunk
        if chunk == 1:
            break
        chunk = max(1, chunk // 2)
    return Violation(key, reason, tuple(sorted(ops, key=lambda op: op.start)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m raft.net.linearizability")
    parser.add_argument("history", type=Path, help="a loadgen .history.jsonl file")
    args = parser.parse_args(argv)
    result = check(load_history(args.history))
    print(f"{result.summary()}  ({result.seconds:.2f} s)")
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
