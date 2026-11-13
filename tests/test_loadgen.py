"""Tests for `raft.net.loadgen`: percentiles, gaps, summaries, and the
closed-loop runner against fake clients."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from raft.net.client import OperationTimeout
from raft.net.loadgen import (
    Fault,
    OpResult,
    first_write_after,
    longest_gap,
    percentile,
    run_load,
    summarize,
    write_history,
)

# -- percentile --


def test_percentile_nearest_rank_textbook_example() -> None:
    values = [15.0, 20.0, 35.0, 40.0, 50.0]  # shuffled order must not matter
    shuffled = [40.0, 15.0, 50.0, 35.0, 20.0]
    assert [percentile(shuffled, p) for p in (5, 30, 40, 50, 100)] == [15, 20, 20, 35, 50]
    assert percentile(values, 50) == 35


def test_percentile_p999_of_1000_is_not_the_maximum() -> None:
    values = [float(v) for v in range(1, 1001)]
    assert percentile(values, 99.9) == 999.0  # float rank arithmetic gives 1000
    assert percentile(values, 99) == 990.0
    assert percentile(values, 50) == 500.0
    assert percentile([7.0], 99.9) == 7.0


@pytest.mark.parametrize("p", [0, -1, 100.1])
def test_percentile_rejects_bad_p(p: float) -> None:
    with pytest.raises(ValueError, match="p must"):
        percentile([1.0], p)


def test_percentile_of_nothing() -> None:
    with pytest.raises(ValueError, match="no values"):
        percentile([], 50)


# -- gaps and faults --


def test_longest_gap_includes_the_window_edges() -> None:
    assert longest_gap([2.0, 3.0], 0.0, 10.0) == (7.0, 3.0)  # trailing
    assert longest_gap([5.0, 6.0], 0.0, 10.0) == (5.0, 0.0)  # leading
    assert longest_gap([1.0, 4.0, 9.0, 99.0], 0.0, 10.0) == (5.0, 4.0)  # 99 is outside
    assert longest_gap([], 0.0, 10.0) == (10.0, 0.0)


def op(kind: str, start: float, end: float, ok: bool = True, node: str | None = "n0") -> OpResult:
    return OpResult("c", kind, "k", "c:1", start, end, ok, node)  # type: ignore[arg-type]


def test_first_write_after_ignores_gets_failures_and_the_dead_node() -> None:
    ops = [
        op("put", 9.0, 10.001, node="n0"),  # answered by the killed leader
        op("get", 9.5, 10.5, node="n1"),
        op("put", 9.0, 11.0, ok=False, node=None),
        op("put", 9.0, 12.0, node="n1"),
        op("put", 12.5, 13.0, node="n1"),
    ]
    assert first_write_after(ops, 10.0, not_from="n0") == pytest.approx(2.0)
    assert first_write_after(ops, 10.0) == pytest.approx(0.001)
    assert first_write_after(ops, 13.0) is None


def test_summarize() -> None:
    ops = [
        op("put", 0.0, 0.010),
        op("get", 0.010, 0.030),
        op("put", 0.030, 5.0, ok=False, node=None),
        op("put", 5.0, 6.0, node="n1"),
    ]
    summary = summarize(ops, start=0.0, end=10.0, faults=[Fault("kill", 4.0, "n0")])
    assert (summary.ops, summary.ok, summary.failed) == (4, 3, 1)
    assert (summary.puts_ok, summary.gets_ok) == (2, 1)
    assert summary.ops_per_sec == pytest.approx(0.3)
    assert summary.latency_ms == {
        "p50": pytest.approx(20.0),
        "p99": pytest.approx(1000.0),
        "p99.9": pytest.approx(1000.0),
    }
    assert (summary.longest_gap_s, summary.longest_gap_at_s) == (
        pytest.approx(5.97),
        pytest.approx(0.03),
    )
    assert summary.first_write_after_fault_s == {"kill": pytest.approx(2.0)}


def test_summarize_with_no_successes() -> None:
    summary = summarize([op("put", 0.0, 1.0, ok=False)], start=0.0, end=0.0)
    assert summary.ops_per_sec == 0.0
    assert summary.latency_ms == {"p50": None, "p99": None, "p99.9": None}


def test_write_history_is_one_json_object_per_line(tmp_path: Path) -> None:
    ops = [op("put", 0.0, 1.0), op("get", 1.0, 2.0, ok=False, node=None)]
    path = tmp_path / "history.jsonl"
    write_history(ops, path)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0] == {
        "client": "c",
        "kind": "put",
        "key": "k",
        "value": "c:1",
        "start": 0.0,
        "end": 1.0,
        "ok": True,
        "node": "n0",
        "error": None,
    }
    assert lines[1]["ok"] is False


# -- run_load --


class FakeClient:
    """An in-memory store shared by all fakes; every `fail_every`-th
    operation raises like a client that gave up."""

    def __init__(self, n: int, store: dict[str, str], fail_every: int = 0) -> None:
        self.client_id = f"client{n}"
        self.leader: str | None = "n0"
        self.store = store
        self.fail_every = fail_every
        self.calls = 0
        self.closed = False

    async def _step(self) -> None:
        self.calls += 1
        await asyncio.sleep(0.002)
        if self.fail_every and self.calls % self.fail_every == 0:
            raise OperationTimeout("gave up")

    async def put(self, key: str, value: str) -> None:
        await self._step()
        self.store[key] = value

    async def get(self, key: str) -> str | None:
        await self._step()
        return self.store.get(key)

    async def close(self) -> None:
        self.closed = True


def test_run_load_closed_loop_with_unique_put_values() -> None:
    store: dict[str, str] = {}
    fakes: list[FakeClient] = []

    def make(n: int) -> FakeClient:
        fakes.append(FakeClient(n, store, fail_every=5))
        return fakes[-1]

    ops = asyncio.run(run_load(make, clients=4, duration=0.2, put_ratio=0.5, keys=5, seed=1))

    assert all(f.closed for f in fakes)
    assert {o.client for o in ops} == {f"client{n}" for n in range(4)}
    assert [o.start for o in ops] == sorted(o.start for o in ops)
    puts = [o for o in ops if o.kind == "put"]
    assert len({o.value for o in puts}) == len(puts)  # every put value unique
    for o in puts:  # "<client_id>:<serial>", serials counting up per client
        assert o.value is not None and o.value.startswith(f"{o.client}:")
    serials = [int(o.value.split(":")[1]) for o in puts if o.client == "client0" and o.value]
    assert serials == sorted(serials)
    assert any(o.kind == "get" for o in ops)
    failed = [o for o in ops if not o.ok]
    assert failed and all(o.error == "OperationTimeout: gave up" and o.node is None for o in failed)
    assert all(o.value is None for o in failed if o.kind == "get")
    for client in {o.client for o in ops}:  # one operation at a time per client
        mine = [o for o in ops if o.client == client]
        assert all(a.end <= b.start for a, b in zip(mine, mine[1:], strict=False))


@pytest.mark.parametrize(
    ("kwargs", "match"), [({"clients": 0}, "clients"), ({"put_ratio": 2}, "put_ratio")]
)
def test_run_load_validates(kwargs: dict[str, float], match: str) -> None:
    args = {"clients": 1, "duration": 0.0, **kwargs}
    with pytest.raises(ValueError, match=match):
        asyncio.run(run_load(lambda n: FakeClient(n, {}), **args))  # type: ignore[arg-type]
