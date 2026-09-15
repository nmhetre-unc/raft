# Bugs

Safety violations found by the fuzzer. Each entry records the seed, the
minimal reproducing trace, the property that broke, and the root cause.

Reproduce any of these with `pytest tests/test_fuzz.py --seed <N>`.