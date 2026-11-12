"""Closed-loop load generation against a `raft.net` cluster.

`run_load` runs N workers for a fixed duration. Each worker owns one
client and issues one operation at a time -- the next starts only when
the previous one finishes, which is both what a closed-loop benchmark
means and what `RaftClient` requires (it must not pipeline). Operations
are a seeded mix of puts and gets over a small key space. Every put
value is ``"<client_id>:<serial>"``, unique across the whole run, so a
verifier can tell exactly which write a log entry or a read came from.

Every operation is recorded as an `OpResult`, successful or not, with
start and end on `time.monotonic()` -- the same clock the benchmark
script uses for fault times, so the two line up without conversion.

`summarize` turns a history into the numbers a benchmark reports, and
`write_history` saves it as JSON Lines, one operation per line.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Literal, Protocol

OpKind = Literal["put", "get"]


class LoadClient(Protocol):
    """What a worker needs from a client; `RaftClient` satisfies it."""

    client_id: str
    leader: str | None

    async def put(self, key: str, value: str) -> None: ...

    async def get(self, key: str) -> str | None: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class OpResult:
    client: str
    kind: OpKind
    key: str
    value: str | None  # written (put) or read (get); None if absent or failed
    start: float  # time.monotonic()
    end: float
    ok: bool
    node: str | None = None  # who answered, when known
    error: str | None = None


async def run_load(
    make_client: Callable[[int], LoadClient],
    *,
    clients: int,
    duration: float,
    put_ratio: float = 0.5,
    keys: int = 100,
    seed: int = 0,
) -> list[OpResult]:
    """Run `clients` closed-loop workers for `duration` seconds.

    A worker starts no new operation once `duration` has passed, but
    lets the one in flight finish (or fail), so every acknowledged write
    is in the history. Results are sorted by start time.
    """
    if clients < 1:
        raise ValueError("clients must be at least 1")
    if not 0.0 <= put_ratio <= 1.0:
        raise ValueError("put_ratio must be between 0 and 1")
    deadline = time.monotonic() + duration
    histories = await asyncio.gather(
        *(
            _worker(make_client(n), deadline, put_ratio, keys, random.Random(seed * 100_003 + n))
            for n in range(clients)
        )
    )
    return sorted((op for history in histories for op in history), key=lambda op: op.start)


async def _worker(
    client: LoadClient, deadline: float, put_ratio: float, keys: int, rng: random.Random
) -> list[OpResult]:
    history: list[OpResult] = []
    serial = 0  # mirrors the client's own serial: one per operation
    try:
        while time.monotonic() < deadline:
            serial += 1
            key = f"key{rng.randrange(keys)}"
            kind: OpKind = "put" if rng.random() < put_ratio else "get"
            value: str | None = f"{client.client_id}:{serial}" if kind == "put" else None
            start = time.monotonic()
            error = None
            try:
                if kind == "put":
                    assert value is not None
                    await client.put(key, value)
                else:
                    value = await client.get(key)
            except Exception as exc:  # recorded, not raised: the run goes on
                error = f"{type(exc).__name__}: {exc}"
                if kind == "get":
                    value = None
            history.append(
                OpResult(
                    client=client.client_id,
                    kind=kind,
                    key=key,
                    value=value,
                    start=start,
                    end=time.monotonic(),
                    ok=error is None,
                    node=client.leader if error is None else None,
                    error=error,
                )
            )
    finally:
        await client.close()
    return history


# -- Summaries --


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile: the smallest value with at least `p`
    percent of `values` at or below it, i.e. the ``ceil(p/100 * n)``-th
    smallest. The rank is computed in decimal, not binary floating point:
    ``99.9 / 100 * 1000`` is 999.0000000000001 in floats, which would
    round p99.9 of 1000 values up to the maximum."""
    if not values:
        raise ValueError("percentile of no values")
    if not 0 < p <= 100:
        raise ValueError("p must be in (0, 100]")
    rank = math.ceil(Decimal(str(p)) * len(values) / 100)
    return sorted(values)[rank - 1]


def longest_gap(times: Iterable[float], start: float, end: float) -> tuple[float, float]:
    """The longest interval within `[start, end]` containing none of
    `times`, as `(length, gap_start)`."""
    points = [start, *sorted(t for t in times if start <= t <= end), end]
    length, at = max((b - a, a) for a, b in zip(points, points[1:], strict=False))
    return length, at


def first_write_after(
    ops: Iterable[OpResult], at: float, not_from: str | None = None
) -> float | None:
    """Seconds from `at` to the first successful put that completed
    after it, or None if none did. `not_from` excludes puts answered by
    that node -- for a killed leader, a reply already on the wire when it
    died would otherwise count as a write the new leader committed."""
    ends = [
        op.end
        for op in ops
        if op.ok and op.kind == "put" and op.end > at and (not_from is None or op.node != not_from)
    ]
    return min(ends) - at if ends else None


@dataclass(frozen=True)
class Fault:
    label: str
    at: float  # time.monotonic()
    node: str | None = None  # the node it hit, if any


@dataclass
class Summary:
    window: float
    ops: int
    ok: int
    failed: int
    puts_ok: int
    gets_ok: int
    ops_per_sec: float
    latency_ms: dict[str, float | None]
    longest_gap_s: float
    longest_gap_at_s: float  # seconds after the window start
    first_write_after_fault_s: dict[str, float | None] = field(default_factory=dict)


def summarize(
    ops: Sequence[OpResult],
    *,
    start: float,
    end: float,
    faults: Sequence[Fault] = (),
) -> Summary:
    """Summarize a history over the window `[start, end]` (monotonic)."""
    window = end - start
    done = [op for op in ops if op.ok]
    latencies = [(op.end - op.start) * 1000 for op in done]
    gap, gap_at = longest_gap((op.end for op in done), start, end)
    return Summary(
        window=window,
        ops=len(ops),
        ok=len(done),
        failed=len(ops) - len(done),
        puts_ok=sum(op.kind == "put" for op in done),
        gets_ok=sum(op.kind == "get" for op in done),
        ops_per_sec=len(done) / window if window > 0 else 0.0,
        latency_ms={
            name: (percentile(latencies, p) if latencies else None)
            for name, p in (("p50", 50), ("p99", 99), ("p99.9", 99.9))
        },
        longest_gap_s=gap,
        longest_gap_at_s=gap_at - start,
        first_write_after_fault_s={
            fault.label: first_write_after(ops, fault.at, not_from=fault.node) for fault in faults
        },
    )


def write_history(ops: Iterable[OpResult], path: str | Path) -> None:
    with Path(path).open("w") as f:
        for op in ops:
            f.write(json.dumps(asdict(op)) + "\n")
