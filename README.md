# raft

A Raft consensus implementation in Python, built around a deterministic
fault-injection harness. Every run reproduces exactly from a seed.

The test harness is the point. A randomized scheduler injects network
partitions, message loss and reordering, and node crashes mid-write, then
checks Raft's safety properties after every step. A failing 160-step trace
shrinks by delta debugging to the 1 or 2 actions that actually reproduce the
violation. 1000 randomized schedules run in 28 seconds in CI.

Currently implements leader election. Log replication, commitment, and
snapshotting are in progress.

## Bugs found

See [BUGS.md](BUGS.md).

## Setup

Requires Python 3.12+.

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

Run the tests:

```bash
pytest                                    # default suite, 50 fuzz seeds
pytest -m slow                            # 1000-seed sweep
pytest tests/test_fuzz_sweep.py --seed 8471   # reproduce one seed
```

If mypy fails on Windows with an Application Control policy error, reinstall
it from source: `pip install --no-binary mypy mypy==2.3.1`