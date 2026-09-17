"""Tests for the standalone KVStateMachine -- no Raft involved here at all."""

import pytest

from raft.statemachine import NOT_FOUND, Delete, Get, KVStateMachine, Put


def test_put_then_get_returns_the_stored_value() -> None:
    sm = KVStateMachine()

    assert sm.apply(Put(key="x", value=1)) is None
    assert sm.apply(Get(key="x")) == 1


def test_put_overwrites_an_existing_key() -> None:
    sm = KVStateMachine()
    sm.apply(Put(key="x", value=1))

    sm.apply(Put(key="x", value=2))

    assert sm.apply(Get(key="x")) == 2


def test_get_on_a_missing_key_returns_not_found() -> None:
    sm = KVStateMachine()

    assert sm.apply(Get(key="missing")) is NOT_FOUND


def test_stored_none_is_distinguishable_from_missing() -> None:
    # None is a legitimate value to Put -- must not be confused with "the
    # key was never set", which is exactly what the NOT_FOUND sentinel
    # (distinct from None) exists to keep apart.
    sm = KVStateMachine()
    sm.apply(Put(key="x", value=None))

    assert sm.apply(Get(key="x")) is None
    assert sm.apply(Get(key="never-set")) is NOT_FOUND


def test_delete_removes_the_key_and_returns_its_prior_value() -> None:
    sm = KVStateMachine()
    sm.apply(Put(key="x", value="hello"))

    result = sm.apply(Delete(key="x"))

    assert result == "hello"
    assert sm.apply(Get(key="x")) is NOT_FOUND


def test_delete_on_a_missing_key_is_a_no_op_returning_not_found() -> None:
    sm = KVStateMachine()

    assert sm.apply(Delete(key="missing")) is NOT_FOUND


def test_not_found_has_a_readable_repr() -> None:
    assert repr(NOT_FOUND) == "NOT_FOUND"


def test_apply_rejects_an_unrecognized_command() -> None:
    sm = KVStateMachine()

    with pytest.raises(TypeError, match="unknown command"):
        sm.apply("not a real command")  # type: ignore[arg-type]


def test_read_is_a_direct_local_lookup_matching_get() -> None:
    sm = KVStateMachine()
    sm.apply(Put(key="x", value="hello"))

    assert sm.read("x") == "hello"
    assert sm.read("missing") is NOT_FOUND


def test_read_distinguishes_a_stored_none_from_a_missing_key() -> None:
    # test_stored_none_is_distinguishable_from_missing already proves this
    # for apply(Get(...)); read() is a separate code path (a plain dict
    # lookup, not routed through apply() at all -- see KVStateMachine.read's
    # own docstring) and needed its own, direct confirmation before
    # anything else relies on read()'s NOT_FOUND-vs-None contract.
    sm = KVStateMachine()
    sm.apply(Put(key="x", value=None))

    assert sm.read("x") is None
    assert sm.read("never-set") is NOT_FOUND


def test_read_never_mutates_state() -> None:
    sm = KVStateMachine()

    sm.read("missing")

    assert sm.snapshot() == {}


def test_snapshot_reflects_current_state() -> None:
    sm = KVStateMachine()
    sm.apply(Put(key="a", value=1))
    sm.apply(Put(key="b", value=2))
    sm.apply(Delete(key="a"))

    assert sm.snapshot() == {"b": 2}


def test_snapshot_is_a_copy_not_a_live_view() -> None:
    sm = KVStateMachine()
    sm.apply(Put(key="x", value=1))

    snapshot = sm.snapshot()
    snapshot["x"] = 999
    snapshot["y"] = "injected"

    assert sm.snapshot() == {"x": 1}  # internal state untouched by mutating the copy


def test_two_instances_given_the_identical_command_sequence_converge() -> None:
    """The determinism guarantee this whole module exists for: same
    sequence of applied commands, in the same order, on two entirely
    separate instances -- identical final state."""
    commands = [
        Put(key="a", value=1),
        Put(key="b", value=2),
        Delete(key="a"),
        Put(key="a", value=3),
        Put(key="c", value=None),
    ]

    sm_a = KVStateMachine()
    sm_b = KVStateMachine()
    for command in commands:
        sm_a.apply(command)
    for command in commands:
        sm_b.apply(command)

    assert sm_a.snapshot() == sm_b.snapshot() == {"a": 3, "b": 2, "c": None}
