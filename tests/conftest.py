"""Shared pytest configuration: the --seed option and fuzz-sweep timing.

`--seed` exists so a single failing seed found by the fuzz sweep can be
reproduced in isolation:

    pytest tests/test_fuzz_sweep.py --seed 8471

`tests/test_fuzz_sweep.py` reads it back via `pytest_generate_tests` to
decide which seed(s) to parametrize its sweep tests over.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import _pytest.config
    import _pytest.terminal


def pytest_addoption(parser: _pytest.config.Parser) -> None:
    parser.addoption(
        "--seed",
        action="store",
        default=None,
        type=int,
        help=(
            "Run tests/test_fuzz_sweep.py's sweep(s) for only this seed, "
            "to reproduce one specific failure in isolation."
        ),
    )


# Sweep test node IDs (from tests/test_fuzz_sweep.py) to report aggregate
# wall-clock time for, keyed by the human-readable label to print.
_SWEEP_LABELS = {
    "test_fuzz_sweep_fixed_seeds": "50-seed sweep",
    "test_fuzz_sweep_large": "1000-seed sweep",
}


def pytest_terminal_summary(
    terminalreporter: _pytest.terminal.TerminalReporter,
    exitstatus: int,
    config: _pytest.config.Config,
) -> None:
    """Report total wall-clock time per fuzz sweep, for judging CI budget.

    Each seed runs as its own parametrized test case (so a failure names
    exactly which seed to reproduce with --seed), which means no single
    test's own duration is the sweep's total -- this sums every
    matching case's reported duration instead.
    """
    del exitstatus, config
    totals: dict[str, float] = {}
    for reports in terminalreporter.stats.values():
        for report in reports:
            if getattr(report, "when", None) != "call":
                continue
            for function_name, label in _SWEEP_LABELS.items():
                if f"::{function_name}[" in report.nodeid or report.nodeid.endswith(
                    f"::{function_name}"
                ):
                    totals[label] = totals.get(label, 0.0) + report.duration

    if not totals:
        return
    terminalreporter.write_sep("=", "fuzz sweep wall-clock time")
    for label, total_seconds in totals.items():
        terminalreporter.write_line(f"{label}: {total_seconds:.2f}s")
