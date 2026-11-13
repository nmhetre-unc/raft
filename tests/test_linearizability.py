"""Tests for `raft.net.linearizability`: hand-written histories, generated
histories of a correct register (always linearizable), and the same with a
stale read injected (never linearizable)."""

from __future__ import annotations

import math
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from raft.net.linearizability import (
    Op,
    Violation,
    _final_states,
    check,
    format_violation,
    load_history,
    main,
)
from raft.net.loadgen import OpResult, write_history


def put(value: str, start: float, end: float, ok: bool = True, key: str = "k") -> Op:
    return Op(value.split(":")[0], "put", key, value, start, end, ok, "n0")


def get(value: str | None, start: float, end: float, ok: bool = True, key: str = "k") -> Op:
    return Op("r", "get", key, value, start, end, ok, "n0")


def violation(*ops: Op) -> Violation:
    result = check(ops)
    assert result.violation is not None, "expected a violation"
    return result.violation


def linearizable(*ops: Op) -> bool:
    return check(ops).ok


# -- Hand-written --


def test_sequential_history() -> None:
    assert linearizable(get(None, 0, 1), put("a:1", 2, 3), get("a:1", 4, 5), put("b:1", 6, 7))


def test_concurrent_read_may_see_either_value() -> None:
    for seen in ("a:1", "b:1"):
        assert linearizable(put("a:1", 0, 1), put("b:1", 2, 6), get(seen, 3, 4))


def test_touching_intervals_are_concurrent() -> None:
    assert linearizable(put("a:1", 0, 1), put("b:1", 2, 4), get("a:1", 4, 5))


def test_stale_read_after_a_newer_write_completed() -> None:
    v = violation(put("a:1", 0, 1), put("b:1", 2, 3), get("a:1", 4, 5))
    assert v.key == "k"
    assert [op.value for op in v.ops] == ["a:1", "b:1", "a:1"]  # all three, nothing else
    assert "no order" in v.reason


def test_counterexample_is_minimal_within_a_long_history() -> None:
    noise = [put(f"x:{i}", 10 * i, 10 * i + 1) for i in range(1, 30)]
    reads = [get(f"x:{i}", 10 * i + 2, 10 * i + 3) for i in range(1, 30)]
    stale = [put("a:1", 400, 401), put("b:1", 402, 403), get("a:1", 404, 405)]
    v = violation(*noise, *reads, *stale, put("z:1", 500, 501))
    assert [(op.kind, op.value) for op in v.ops] == [
        ("put", "a:1"),
        ("put", "b:1"),
        ("get", "a:1"),
    ]


def test_read_of_a_value_never_written() -> None:
    v = violation(put("a:1", 0, 1), get("ghost:1", 2, 3))
    assert [op.value for op in v.ops] == ["ghost:1"]
    assert v.reason == "read a value that no put ever wrote"


def test_empty_read_after_a_completed_write() -> None:
    v = violation(put("a:1", 0, 1), get(None, 2, 3))
    assert [op.value for op in v.ops] == ["a:1", None]


def test_unknown_put_observed_later_took_effect() -> None:
    assert linearizable(put("a:1", 0, 1), put("b:1", 2, 3, ok=False), get("b:1", 10, 11))


def test_unknown_put_never_observed_may_never_have_happened() -> None:
    assert linearizable(put("a:1", 0, 1), put("b:1", 2, 3, ok=False), get("a:1", 10, 11))


def test_unknown_put_cannot_be_read_before_it_started() -> None:
    v = violation(get("b:1", 0, 1), put("b:1", 2, 3, ok=False))
    assert {op.value for op in v.ops} == {"b:1"}


def test_unknown_put_takes_effect_once() -> None:
    # Observed at 4-5, so it took effect by then; the older value can't come back.
    v = violation(put("a:1", 0, 1), put("b:1", 2, 3, ok=False), get("b:1", 4, 5), get("a:1", 6, 7))
    assert len(v.ops) == 4
    text = format_violation(v)
    assert "[2000.0, ?] ms  unknown outcome (client gave up at 3000.0 ms)" in text


def test_failed_gets_carry_no_information() -> None:
    assert linearizable(put("a:1", 0, 1), get(None, 2, 3, ok=False), get("a:1", 4, 5))


def test_unobserved_concurrent_puts_leave_either_value() -> None:
    # Two concurrent puts nobody reads, then quiescence: either may be last.
    for seen in ("a:1", "b:1"):
        assert linearizable(put("a:1", 0, 2), put("b:1", 1, 3), get(seen, 10, 11))
    assert set(_final_states([put("a:1", 0, 2), put("b:1", 1, 3)], None)) == {"a:1", "b:1"}


def test_keys_are_checked_independently() -> None:
    v = violation(
        put("a:1", 0, 1, key="x"),
        get("a:1", 2, 3, key="x"),
        put("c:1", 0, 1, key="y"),
        put("d:1", 2, 3, key="y"),
        get("c:1", 4, 5, key="y"),
    )
    assert v.key == "y"


def test_summary_and_formatting() -> None:
    ok = check([put("a:1", 0, 1), get("a:1", 2, 3)])
    assert ok.summary() == "2 operations, linearizable"
    assert (ok.keys, ok.checked) == (1, 2)
    text = check([put("a:1", 0, 1), put("b:1", 2, 3), get("a:1", 4, 5)]).summary()
    lines = text.splitlines()
    assert lines[0] == "3 operations, NOT linearizable"
    assert lines[1].startswith("key 'k': no order")
    assert lines[2].split() == ["a", "put", ":=", "'a:1'", "[0.0,", "1000.0]", "ms", "ok", "via",
                                "n0"]  # fmt: skip
    assert lines[4].split()[1:4] == ["get", "<-", "'a:1'"]


def test_reads_loadgen_histories_and_cli(tmp_path: Path) -> None:
    good = [
        OpResult("c", "put", "k", "c:1", 0.0, 1.0, True, "n0"),
        OpResult("c", "get", "k", "c:1", 2.0, 3.0, True, "n0"),
    ]
    path = tmp_path / "h.jsonl"
    write_history(good, path)
    assert load_history(path) == [Op.of(op) for op in good]
    assert main([str(path)]) == 0
    stale = [
        *good,
        OpResult("c", "put", "k", "c:2", 4.0, 5.0, True, "n0"),
        OpResult("c", "get", "k", "c:1", 6.0, 7.0, True, "n0"),
    ]
    write_history(stale, path)
    assert main([str(path)]) == 1


# -- Generated histories of a correct register --

KEYS = ["k0", "k1", "k2"]


@st.composite
def register_histories(draw: st.DrawFn) -> list[Op]:
    """What loadgen produces, from a register known to be correct: a few
    closed-loop clients, each running one operation at a time, so their
    intervals overlap only across clients. Each operation gets a
    linearization point inside its interval and is applied to the register
    in that order. An unknown-outcome put takes effect at some point after
    its start -- possibly after the client gave up -- or never; a failed
    get reads nothing."""
    clients = draw(st.integers(1, 4))
    raw = []
    for c in range(clients):
        t: float = draw(st.integers(0, 5))
        for n in range(draw(st.integers(0, 12))):
            start = t + draw(st.integers(0, 3))
            point: float = start + draw(st.integers(0, 6))
            end = point + draw(st.integers(0, 6))
            t = end
            kind = draw(st.sampled_from(["put", "get"]))
            outcome = draw(st.sampled_from(["ok", "ok", "ok", "unknown"]))
            key = draw(st.sampled_from(KEYS))
            if kind == "put" and outcome == "unknown":
                took_effect = draw(st.booleans())
                point = start + draw(st.integers(0, 40)) if took_effect else math.inf
            raw.append((point, f"c{c}", n, kind, outcome, key, start, end))
    register: dict[str, str | None] = dict.fromkeys(KEYS)
    ops = []
    for point, client, n, kind, outcome, key, start, end in sorted(raw):
        if kind == "put":
            value = f"{client}:{n}"
            if not math.isinf(point):
                register[key] = value
            ops.append(Op(client, "put", key, value, start, end, outcome == "ok"))
        else:
            seen = register[key] if outcome == "ok" else None
            ops.append(Op(client, "get", key, seen, start, end, outcome == "ok"))
    return ops


@settings(max_examples=400, deadline=None)
@given(register_histories())
def test_correct_register_histories_are_linearizable(ops: list[Op]) -> None:
    result = check(ops)
    assert result.ok, format_violation(result.violation) if result.violation else ""


@settings(max_examples=300, deadline=None)
@given(register_histories(), st.data())
def test_injected_stale_read_is_caught(ops: list[Op], data: st.DataObject) -> None:
    """A read of A, after B was written strictly after A and strictly
    before the read, can't be linearized. Inject one: into an existing
    get when the history has such a pair, else as three new operations."""
    key = data.draw(st.sampled_from(KEYS))
    puts = [op for op in ops if op.kind == "put" and op.ok and op.key == key]
    candidates = [
        (a, b, g)
        for g in ops
        if g.kind == "get" and g.ok and g.key == key
        for b in puts
        if b.end < g.start
        for a in puts
        if a.end < b.start
    ]
    if candidates:
        a, b, g = data.draw(st.sampled_from(candidates))
        stale = Op(g.client, "get", key, a.value, g.start, g.end, True)
        ops = [stale if op is g else op for op in ops]
    else:
        t = max((op.end for op in ops), default=0) + 1
        ops = [*ops, put("sa:1", t, t + 1, key=key), put("sb:1", t + 2, t + 3, key=key),
               get("sa:1", t + 4, t + 5, key=key)]  # fmt: skip
    v = check(ops).violation
    assert v is not None and v.key == key
