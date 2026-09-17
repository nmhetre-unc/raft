"""Tests for client-request session tracking: KVStateMachine.
apply_client_request's own dedup rule, the positive control proving a
naive leader-only dedup design fails across a leader change while the
real, state-machine-side design doesn't, and fuzz-level confirmation
that no committed client request is ever genuinely double-applied.
"""

from __future__ import annotations

import itertools
import random

import pytest

from raft.node import RaftNode, Role
from raft.sim.fuzz import Fuzzer
from raft.statemachine import ClientRequest, KVStateMachine, Put
from raft.storage import MemoryStorage

# -- apply_client_request's own dedup rule, direct and unit-level --


def test_dedup_applies_exactly_once_for_a_repeated_serial_number() -> None:
    """The exact scenario the requirement names: the same command
    submitted twice with the same serial_number. Detected via a call
    counter on the underlying apply(), not final key/value equality --
    a Put retried with the identical value would look the same either
    way even if it ran twice.
    """
    sm = KVStateMachine()
    apply_calls: list[object] = []
    original_apply = sm.apply

    def counting_apply(command: object) -> object:
        apply_calls.append(command)
        return original_apply(command)

    sm.apply = counting_apply  # type: ignore[method-assign]

    request = ClientRequest(client_id="c1", serial_number=1, command=Put(key="x", value=1))

    first_result = sm.apply_client_request(request)
    second_result = sm.apply_client_request(request)  # identical retry

    assert len(apply_calls) == 1  # the underlying apply() ran exactly once
    assert first_result is None  # Put's own result
    assert second_result == first_result  # the recorded result, not a fresh one
    assert sm.snapshot() == {"x": 1}
    assert sm.sessions_snapshot() == {"c1": (1, None)}


def test_dedup_returns_recorded_result_even_if_a_different_command_reuses_the_serial() -> None:
    """A real client never legitimately reuses a serial number for a
    different logical request, but this confirms the mechanism keys
    purely on (client_id, serial_number) and never re-inspects command
    content to decide whether to skip.
    """
    sm = KVStateMachine()
    sm.apply_client_request(
        ClientRequest(client_id="c1", serial_number=1, command=Put(key="x", value=1))
    )

    result = sm.apply_client_request(
        ClientRequest(client_id="c1", serial_number=1, command=Put(key="x", value=999))
    )

    assert result is None  # the ORIGINAL result, not a re-application
    assert sm.snapshot() == {"x": 1}  # untouched by the "different" retry


def test_dedup_also_skips_a_serial_number_older_than_the_recorded_one() -> None:
    sm = KVStateMachine()
    sm.apply_client_request(
        ClientRequest(client_id="c1", serial_number=5, command=Put(key="x", value=1))
    )

    result = sm.apply_client_request(
        ClientRequest(client_id="c1", serial_number=3, command=Put(key="x", value=999))
    )

    assert result is None
    assert sm.snapshot() == {"x": 1}


def test_a_new_serial_number_for_the_same_client_applies_normally() -> None:
    sm = KVStateMachine()
    sm.apply_client_request(
        ClientRequest(client_id="c1", serial_number=1, command=Put(key="x", value=1))
    )

    sm.apply_client_request(
        ClientRequest(client_id="c1", serial_number=2, command=Put(key="x", value=2))
    )

    assert sm.snapshot() == {"x": 2}
    assert sm.sessions_snapshot() == {"c1": (2, None)}


def test_different_clients_are_tracked_independently() -> None:
    sm = KVStateMachine()
    sm.apply_client_request(
        ClientRequest(client_id="c1", serial_number=1, command=Put(key="x", value=1))
    )
    sm.apply_client_request(
        ClientRequest(client_id="c2", serial_number=1, command=Put(key="x", value=2))
    )

    assert sm.snapshot() == {"x": 2}  # c2's Put ran too, independently
    assert sm.sessions_snapshot() == {"c1": (1, None), "c2": (1, None)}


# -- Positive control: naive leader-only dedup vs. the real, state-machine-side design --


def _force_election_timeout(node: RaftNode) -> None:
    node.election_deadline = 0
    node._deadline_initialized = True


def _drain(nodes: dict[str, RaftNode], queue: list[tuple[str, str, object]], now: int) -> None:
    """Process every (dst, src, msg) by hand -- feeding whatever gets
    produced back into the queue -- until nothing's left. No Cluster or
    Network involved: just RaftNode objects passed messages directly,
    exactly like tests/test_replication.py's own style, extended here to
    drain a full cascade (votes, replies, backoff retries) rather than
    one message at a time. `dst` not in `nodes` means that node has
    "crashed" -- the message is simply dropped, never delivered.
    """
    while queue:
        dst, src, msg = queue.pop(0)
        if dst not in nodes:
            continue
        produced = nodes[dst].handle(msg, src=src, now=now)
        for out_dst, out_msg in produced:
            queue.append((out_dst, dst, out_msg))


def _build_three_node_cluster() -> dict[str, RaftNode]:
    ids = ("0", "1", "2")
    return {
        node_id: RaftNode(
            node_id,
            [peer for peer in ids if peer != node_id],
            MemoryStorage(),
            rng=random.Random(1),
        )
        for node_id in ids
    }


def _count_real_applies(node: RaftNode) -> list[object]:
    """Wrap `node.state_machine.apply` -- the low-level, state-mutating
    method -- to record every command it genuinely executes. Independent
    of whatever `apply_client_request` decides internally: this only
    observes whether the underlying mutation actually ran.
    """
    calls: list[object] = []
    original_apply = node.state_machine.apply

    def counting_apply(command: object) -> object:
        calls.append(command)
        return original_apply(command)

    node.state_machine.apply = counting_apply  # type: ignore[method-assign]
    return calls


def _run_leader_crash_and_retry_scenario(
    use_client_request_wrapper: bool,
) -> tuple[dict[str, RaftNode], list[object]]:
    """Leader A (node "0") commits a client request and crashes before
    it could ever reply; a new leader B (node "1") is elected from the
    remaining majority; the client, having no way to know whether its
    first attempt ever went through, retries the IDENTICAL request
    against B. Returns the surviving nodes ("1" and "2") plus every
    command node "1" has genuinely applied by the end.

    `use_client_request_wrapper=True` is the real design: a
    `ClientRequest` applied through `KVStateMachine.apply_client_request`.
    `use_client_request_wrapper=False` is a hand-built naive stand-in:
    dedup tracked ONLY in each leader's own private, in-memory dict,
    checked before deciding whether to even append -- and a bare
    `Command` (no client wrapper at all) actually placed in the log.
    That private dict is exactly what node "0"'s crash discards, and
    node "1"'s own dict starts out empty -- it has no way to know
    serial_number 1 was ever seen, because nothing about it was ever
    written to the log.
    """
    nodes = _build_three_node_cluster()
    applies_on_new_leader = _count_real_applies(nodes["1"])  # installed before anything happens

    _force_election_timeout(nodes["0"])
    produced = nodes["0"].tick(now=0)
    _drain(nodes, [(dst, "0", msg) for dst, msg in produced], now=0)
    assert nodes["0"].role is Role.LEADER

    request = ClientRequest(client_id="c1", serial_number=1, command=Put(key="x", value=1))
    # One dict PER LEADER, not one shared dict -- modeling each leader's
    # own private, in-memory bookkeeping, genuinely separate from any
    # other node's (and from whatever the crashed former leader once
    # knew). A single shared dict here would accidentally make the
    # "naive" design behave correctly by giving it memory a real crash
    # would never let it keep.
    seen_by_leader: dict[str, dict[str, int]] = {"0": {}, "1": {}}

    def submit(leader_id: str, now: int) -> None:
        if use_client_request_wrapper:
            produced = nodes[leader_id].append_command(request, now)
        else:
            seen_on_this_leader = seen_by_leader[leader_id]
            if seen_on_this_leader.get(request.client_id, 0) >= request.serial_number:
                return  # this leader's own memory thinks it already saw this
            seen_on_this_leader[request.client_id] = request.serial_number
            produced = nodes[leader_id].append_command(request.command, now)  # bare, unwrapped
        _drain(nodes, [(dst, leader_id, msg) for dst, msg in produced], now)

    submit("0", now=1)
    assert nodes["0"].commit_index >= 1  # committed on a majority...
    assert nodes["0"].last_applied >= 1  # ...and applied -- "after commit, before reply"

    del nodes["0"]  # leader A crashes: discarded, never consulted again

    _force_election_timeout(nodes["1"])
    produced = nodes["1"].tick(now=2)
    _drain(nodes, [(dst, "1", msg) for dst, msg in produced], now=2)
    assert nodes["1"].role is Role.LEADER  # a new leader, elected from the surviving majority

    submit("1", now=3)  # the client retries the identical request against the new leader

    # A follower only learns the leader's commit_index (and so only
    # applies) via a LATER AppendEntries carrying it -- the retry's own
    # round only tells node "2" it holds the entry, not that it's safe to
    # apply yet. One more heartbeat closes that gap, exactly as it would
    # in a real running cluster a moment later.
    next_heartbeat = nodes["1"]._next_heartbeat
    produced = nodes["1"].tick(now=next_heartbeat)
    _drain(nodes, [(dst, "1", msg) for dst, msg in produced], now=next_heartbeat)

    return nodes, applies_on_new_leader


def test_correct_design_applies_the_retried_request_exactly_once() -> None:
    nodes, applies = _run_leader_crash_and_retry_scenario(use_client_request_wrapper=True)

    assert len(applies) == 1, f"expected exactly one real apply(), got {applies!r}"
    assert nodes["1"].state_machine.snapshot() == {"x": 1}
    assert nodes["2"].state_machine.snapshot() == {"x": 1}
    assert nodes["1"].state_machine.sessions_snapshot() == {"c1": (1, None)}


def test_naive_leader_only_dedup_applies_the_retried_request_twice() -> None:
    """The positive control: the SAME scenario, but with dedup tracked
    only in each leader's own memory instead of on the state machine.
    Node "0"'s crash discards its private record of having seen serial 1;
    node "1" starts with none of its own, so it has no way to recognize
    the retry as one -- and, because the log itself carries no client
    identity at all in this design, neither does anything downstream.
    Detected via the same apply() counter as the correct case above --
    NOT by comparing final state, which would look identical either way
    since the retried Put reuses the same value.
    """
    nodes, applies = _run_leader_crash_and_retry_scenario(use_client_request_wrapper=False)

    assert len(applies) == 2, (
        f"expected the naive design to double-apply (proving the real design's "
        f"fix matters), got {applies!r}"
    )
    # Final state still looks "fine" -- exactly why a value-only check would
    # have missed this bug entirely.
    assert nodes["1"].state_machine.snapshot() == {"x": 1}


# -- Fuzz integration: no genuine double-application across the fixed seed sweep --


def test_no_double_application_across_the_fixed_seed_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    """With CLIENT_REQUEST now sending real, retryable requests --
    including retries deliberately coinciding with crashes and elections
    (see raft.sim.fuzz's own module docstring) -- confirm no *single,
    continuously-live* state machine ever genuinely re-executes a
    command for a (client_id, serial_number) pair it has already
    applied.

    "Continuously-live" matters: `last_applied`/`state_machine` are
    volatile (see `raft.node`'s module docstring) -- a crashed-then-
    restarted node gets a brand new `KVStateMachine` and correctly
    replays the whole committed log into it from scratch, which is by
    design, not a violation. Tracking by raw `id()` alone would conflate
    that fresh instance with an unrelated, already-garbage-collected one
    that happened to get the same address (a real hazard measured while
    building this test -- seed 9 produced exactly that false positive on
    the first attempt), silently reporting a "double application" that
    never happened on any actual single instance. Each `KVStateMachine`
    is tagged with a monotonic id at construction instead, re-tagged
    every time `__init__` runs, so a reused address is never mistaken for
    the instance that previously held it.

    Detection is otherwise independent of `apply_client_request`'s own
    internal dedup decision: a spy wraps each state machine's `apply` --
    the low-level, state-mutating method -- for the duration of exactly
    one `apply_client_request` call, and only records that call as a
    "real" application if the underlying `apply()` actually ran. A bug in
    `apply_client_request`'s own dedup condition couldn't hide from this
    by consistently agreeing with itself, because this never asks that
    condition anything -- it only observes the one method that actually
    mutates state.
    """
    instance_ids: dict[int, int] = {}
    next_instance_id = itertools.count()
    original_init = KVStateMachine.__init__

    def tagging_init(self: KVStateMachine) -> None:
        original_init(self)
        instance_ids[id(self)] = next(next_instance_id)

    monkeypatch.setattr(KVStateMachine, "__init__", tagging_init)

    applied_log: list[tuple[int, str, int]] = []
    original_apply_client_request = KVStateMachine.apply_client_request

    def spy_apply_client_request(self: KVStateMachine, request: ClientRequest) -> object:
        mutated = {"flag": False}
        original_apply = self.apply

        def tracking_apply(command: object) -> object:
            mutated["flag"] = True
            return original_apply(command)

        self.apply = tracking_apply  # type: ignore[method-assign]
        try:
            result = original_apply_client_request(self, request)
        finally:
            del self.apply  # restores the class method for the next call
        if mutated["flag"]:
            applied_log.append((instance_ids[id(self)], request.client_id, request.serial_number))
        return result

    monkeypatch.setattr(KVStateMachine, "apply_client_request", spy_apply_client_request)

    seeds_with_any_application = 0
    for seed in range(50):
        applied_log.clear()
        fuzzer = Fuzzer(n=5, seed=seed, steps=300)
        fuzzer.run()  # must not raise -- also covered by test_fuzz_sweep.py

        by_machine_client: dict[tuple[int, str], list[int]] = {}
        for machine_id, client_id, serial in applied_log:
            by_machine_client.setdefault((machine_id, client_id), []).append(serial)

        if by_machine_client:
            seeds_with_any_application += 1

        for (machine_id, client_id), serials in by_machine_client.items():
            assert len(serials) == len(set(serials)), (
                f"seed={seed} state_machine={machine_id} client={client_id!r}: a serial "
                f"number was genuinely applied more than once: {serials}"
            )
            assert serials == sorted(serials), (
                f"seed={seed} state_machine={machine_id} client={client_id!r}: applied "
                f"out of order: {serials}"
            )

    assert seeds_with_any_application > 0, "no seed ever applied a real client request at all"

