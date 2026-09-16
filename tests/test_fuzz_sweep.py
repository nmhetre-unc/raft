"""Sweeps the Fuzzer across many seeds, looking for a genuine safety violation.

Two sweeps share the same logic, differing only in how many seeds they
cover:

- `test_fuzz_sweep_fixed_seeds`: 50 fixed seeds, runs every time (this is
  what CI's main job exercises).
- `test_fuzz_sweep_large`: 1000 seeds, marked `slow` and deselected by
  default (pyproject.toml's `addopts` excludes it); CI's separate "fuzz"
  job runs it explicitly with `-m slow`.

`pytest tests/test_fuzz_sweep.py --seed 8471` collapses whichever sweep
you point it at down to that one seed, to reproduce a specific failure in
isolation rather than re-running the whole sweep.

On a violation, the shrinker (`Fuzzer.shrink`) runs automatically before
the test fails, and the failure message is a self-contained block naming
the seed, the step it happened at, the property that broke, and the
shrunk trace -- meant to be pasted straight into BUGS.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from raft.invariants import SafetyViolation
from raft.sim.fuzz import Fuzzer, TraceEntry

if TYPE_CHECKING:
    import _pytest.python

N_NODES = 5
STEPS_PER_SEED = 300
FIXED_SWEEP_SEED_COUNT = 50
LARGE_SWEEP_SEED_COUNT = 1000


def pytest_generate_tests(metafunc: _pytest.python.Metafunc) -> None:
    if "seed" not in metafunc.fixturenames:
        return

    single_seed = metafunc.config.getoption("--seed")
    if single_seed is not None:
        metafunc.parametrize("seed", [single_seed])
        return

    if metafunc.definition.get_closest_marker("slow"):
        metafunc.parametrize("seed", range(LARGE_SWEEP_SEED_COUNT))
    else:
        metafunc.parametrize("seed", range(FIXED_SWEEP_SEED_COUNT))


def _format_violation_report(
    seed: int, violation: SafetyViolation, shrunk_trace: list[TraceEntry]
) -> str:
    lines = [
        "",
        "=" * 72,
        "Raft safety violation -- paste straight into BUGS.md",
        "=" * 72,
        f"Seed: {seed}",
        f"Step: {violation.step}",
        f"Property violated: {violation.reason}",
        "",
        f"Minimal reproducing trace ({len(shrunk_trace)} action(s)):",
    ]
    for index, entry in enumerate(shrunk_trace):
        lines.append(f"  {index}: {entry.action.value:<13} {entry.detail}")
    lines.append("=" * 72)
    return "\n".join(lines)


def _run_and_report(seed: int) -> None:
    fuzzer = Fuzzer(n=N_NODES, seed=seed, steps=STEPS_PER_SEED)

    violation: SafetyViolation | None = None
    try:
        fuzzer.run()
    except SafetyViolation as caught:
        violation = caught

    if violation is None:
        return  # no violation: this seed passes

    # Shrink before reporting, per policy -- nobody should have to
    # manually re-run shrink() on a failure this test already saw.
    shrunk_trace = fuzzer.shrink(violation.trace)
    # Raised outside the `except` block so the failure message stands
    # alone, without Python chaining the original SafetyViolation onto it
    # as "the above exception was the direct cause of the following one".
    pytest.fail(_format_violation_report(seed, violation, shrunk_trace), pytrace=False)


def test_fuzz_sweep_fixed_seeds(seed: int) -> None:
    _run_and_report(seed)


@pytest.mark.slow
def test_fuzz_sweep_large(seed: int) -> None:
    _run_and_report(seed)
