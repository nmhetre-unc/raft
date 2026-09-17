"""Tests for the STALE_REDELIVER fuzzer action's own mechanics: the no-op
case, replay determinism, and the recall()-returns-None-for-a-forced-
entry failure mode. Mutation-1-specific end-to-end scenarios live in
tests/test_mutation_detection.py, next to the mutation helper they share;
this file is about the action itself, independent of any mutation.
"""

import pytest

from raft.sim.fuzz import Action, Fuzzer, TraceEntry

STEPS = 300


def _effectful(fuzzer: Fuzzer, action: Action) -> list[TraceEntry]:
    return [
        entry
        for entry in fuzzer.trace
        if entry.action is action and not entry.detail.startswith("skipped")
    ]


def test_no_op_when_nothing_is_eligible_is_recorded_not_raised_or_silent() -> None:
    """A fresh cluster with nothing delivered yet has no candidates at
    all -- drawing STALE_REDELIVER here must be a recorded, drawn-but-void
    TraceEntry (matching every other action's existing no-op convention),
    not an exception and not simply absent from the trace.
    """
    fuzzer = Fuzzer(n=3, seed=1, steps=1)

    entry = fuzzer._do_stale_redeliver(0, forced=None)

    assert entry.action is Action.STALE_REDELIVER
    assert entry.detail == "skipped (no stale-and-superseded message to redeliver)"
    assert entry.msg_id is None
    assert entry.step == 0


def test_stale_redeliver_fires_and_replays_identically_across_two_runs() -> None:
    """Running the same seed's live run() twice from scratch, with
    STALE_REDELIVER in the default action mix, must produce byte-for-byte
    identical traces -- and the run must actually have drawn
    STALE_REDELIVER effectfully at least once, or this wouldn't be
    testing anything about this action specifically.
    """

    def run_once() -> list[TraceEntry]:
        fuzzer = Fuzzer(n=5, seed=1, steps=STEPS)
        fuzzer.run()
        return fuzzer.trace

    trace_a = run_once()
    trace_b = run_once()

    assert trace_a == trace_b
    assert len(trace_a) == len(trace_b)
    for index, (a, b) in enumerate(zip(trace_a, trace_b, strict=True)):
        assert a == b, f"first divergence at step {index}: {a!r} != {b!r}"

    fuzzer = Fuzzer(n=5, seed=1, steps=STEPS)
    fuzzer.run()
    redeliveries = _effectful(fuzzer, Action.STALE_REDELIVER)
    assert redeliveries, "seed 1 never actually redelivered anything -- test proves nothing"
    assert all(entry.msg_id is not None for entry in redeliveries)


def test_stale_redeliver_replay_matches_a_live_runs_own_trace() -> None:
    """replay() of a run's own recorded trace, including its
    STALE_REDELIVER entries, must reproduce the identical trace -- the
    same guarantee replay() already gives every other action, now
    checked for this one specifically.
    """
    fuzzer = Fuzzer(n=5, seed=1, steps=STEPS)
    fuzzer.run()
    original_trace = list(fuzzer.trace)
    assert _effectful(fuzzer, Action.STALE_REDELIVER)  # sanity: exercised

    fuzzer.replay(original_trace)

    assert fuzzer.trace == original_trace


def test_replay_raises_when_a_forced_msg_id_is_not_in_history() -> None:
    """A shrunk-style trace: a STALE_REDELIVER entry recorded as having
    actually redelivered something, replayed against a fresh cluster
    where that msg_id was never assigned at all (the simplest case of
    'no longer in Network's retained history' -- an evicted id looks
    identical to this from recall()'s point of view). Must raise, not
    silently degrade to a skip: a silent skip would replay a DIFFERENT
    event than the one recorded, breaking replay()'s "identical" contract
    -- exactly the class of shrink-induced structural inconsistency
    _signature() already has a blanket `except Exception` for.
    """
    fuzzer = Fuzzer(n=2, seed=1, steps=1)
    forced_trace = [
        TraceEntry(
            step=0,
            action=Action.STALE_REDELIVER,
            detail="redelivered message 0 (0 -> 1), now stale and superseded on that link",
            msg_id=0,
        ),
    ]

    with pytest.raises(ValueError, match="no longer in Network's retained history"):
        fuzzer.replay(forced_trace)


def test_shrinks_signature_treats_missing_msg_id_like_other_structural_inconsistencies() -> None:
    """_signature() (which shrink() relies on to tell "still reproduces"
    from "no longer valid") already swallows any non-SafetyViolation
    exception from replay() as "not a valid reproduction". Confirms the
    ValueError from a missing msg_id is handled exactly the same way as
    every other shrink-induced inconsistency, not specially -- so a
    shrink candidate that happens to drop the delivery that created this
    msg_id fails closed (treated as invalid) rather than raising out of
    shrink() itself.
    """
    fuzzer = Fuzzer(n=2, seed=1, steps=1)
    forced_trace = [
        TraceEntry(
            step=0,
            action=Action.STALE_REDELIVER,
            detail="redelivered message 0 (0 -> 1), now stale and superseded on that link",
            msg_id=0,
        ),
    ]

    assert fuzzer._signature(forced_trace) is None


def test_forced_skip_replay_does_not_touch_network_at_all() -> None:
    """A recorded no-op (msg_id=None) must replay as a no-op too, without
    attempting any recall() -- there is nothing to look up, and this must
    not raise even against a cluster with an empty history.
    """
    fuzzer = Fuzzer(n=2, seed=1, steps=1)
    forced_trace = [
        TraceEntry(
            step=0,
            action=Action.STALE_REDELIVER,
            detail="skipped (no stale-and-superseded message to redeliver)",
            msg_id=None,
        ),
    ]

    fuzzer.replay(forced_trace)  # must not raise

    assert fuzzer.trace[0].detail == "skipped (no stale-and-superseded message to redeliver)"
    assert fuzzer.trace[0].msg_id is None
