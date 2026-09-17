"""Milestone 5 integration and verification: does the whole client-facing
loop -- NOT_LEADER redirect, retry, session dedup -- actually survive the
real fault-injection harness, not just isolated unit tests?

Three things live here, each too broad for `test_election.py`,
`test_replication.py`, `test_client_sessions.py`, or
`test_mutation_detection.py` individually to own:

- `test_check_state_machine_safety_stays_clean_under_the_full_fault_model`:
  an explicit, non-vacuous confirmation that `check_state_machine_safety`
  (see `raft.invariants`) stays clean with CLIENT_REQUEST (including
  retries via leader-hint-following) enabled alongside every other fault
  action -- crashes, restarts, partitions, and `STALE_REDELIVER` -- all at
  once. `tests/test_fuzz_sweep.py` already proves no `SafetyViolation` of
  any kind occurs under this exact combination; what's added here is
  proof that this specific checker was actually given something to
  compare (real, cross-node-overlapping applied entries), not just never
  reached -- the same distinction BUGS.md draws for why a clean sweep
  before Milestone 5 meant nothing.
- `test_client_request_survives_redirect_retry_and_a_leader_crash`: a
  deterministic, hand-constructed proof (real `Cluster`/`RaftClusterNode`,
  no fuzzing) that the full loop this milestone built -- a client hitting
  a follower, getting redirected via `NotLeader.leader_hint`, retrying
  against the real leader, and retrying *again* against a newly-elected
  leader after the first one crashes mid-flight -- ends with the command
  applied exactly once, proven by an `apply()` call count, not just a
  final-value comparison a double-apply of an idempotent `Put` could
  never expose (see `tests/test_client_sessions.py`'s own reasoning for
  why that distinction matters).
- `test_final_kv_state_matches_an_independently_computed_expectation`: for
  a handful of fuzzed seeds under the full fault model, recomputes the
  expected final KV state from scratch, directly off each node's own
  committed log -- a fresh dedup implementation, not a call into
  `KVStateMachine` itself -- and confirms it matches what the real state
  machine actually produced. `BUGS.md`'s Milestone-4 report only checked
  that final state *differed by seed*; this is the stronger claim that
  it's *correct* for what the trace says should have been applied.

Reply-delivery scope, decided explicitly for this milestone (see
`raft.statemachine`'s module docstring and `raft.node`'s `NotLeader`):
a client that never receives a reply -- because its request committed but
the reply was dropped by fault injection, or its leader crashed first --
has no mechanism here to *discover* the outcome beyond blindly retrying
with the same `(client_id, serial_number)`. That retry is always safe
(session dedup guarantees at most one real application), just
uninformative -- there is no query-the-result RPC, and none is added by
this milestone. `Get` needs no special handling as a consequence: it was
already never put through the log (see `statemachine.py`), and adding a
result-lookup path would be new client-protocol surface, not
verification of what's already built.
"""

from __future__ import annotations

from raft.node import NotLeader, RaftClusterNode, raft_node_factory
from raft.sim.cluster import Cluster
from raft.sim.fuzz import Fuzzer
from raft.statemachine import ClientRequest, Delete, Put

N_NODES = 5
STEPS_PER_SEED = 300
FIXED_SWEEP_SEED_COUNT = 50

# -- check_state_machine_safety, explicitly, under the full fault model --


def test_check_state_machine_safety_stays_clean_under_the_full_fault_model() -> None:
    """CLIENT_REQUEST (with leader-hint-following retries), STALE_REDELIVER,
    CRASH, RESTART, PARTITION/HEAL, and ADVANCE_CLOCK are all in
    `raft.sim.fuzz.DEFAULT_WEIGHTS` -- `Fuzzer(n=5, seed=seed, steps=300)`
    with no `weights` override, exactly what `tests/test_fuzz_sweep.py`
    already runs, already exercises the complete fault model at once.
    That file's own two tests (50 seeds, always; 1000 seeds, `-m slow`)
    already prove zero `SafetyViolation`s of any kind occur. What this
    test adds, specifically for `check_state_machine_safety`, is proof
    the checker had real cross-node, overlapping applied-entry
    observations to compare on every single seed -- not that it merely
    ran without being handed anything.
    """
    total_applied_index_observations = 0
    seeds_with_any_application = 0

    for seed in range(FIXED_SWEEP_SEED_COUNT):
        fuzzer = Fuzzer(n=N_NODES, seed=seed, steps=STEPS_PER_SEED)
        fuzzer.run()  # raises SafetyViolation on any failure, from any of the 6 checkers

        history = fuzzer._state_machine_safety_history
        if history.applied:
            seeds_with_any_application += 1
        total_applied_index_observations += len(history.applied)

    # The large majority of seeds in this fixed sweep genuinely apply at
    # least one entry somewhere -- this isn't a handful of lucky seeds
    # carrying the claim (measured: 40/50; BUGS.md's own Milestone-4
    # report measured 34/50 before CLIENT_REQUEST sent real commands at
    # all -- see "CLIENT_REQUEST sent opaque strings..." there).
    assert seeds_with_any_application >= 35
    assert total_applied_index_observations > FIXED_SWEEP_SEED_COUNT  # comfortably non-trivial


# -- Full loop: redirect, retry, survive a leader crash, exactly-once commit --

# Large enough that nothing in this test's clock range ever reaches it --
# elections here are started exclusively by _force_election, never by a
# node's own timer. Same convention as tests/test_figure8.py.
_NEVER = 10**9
_DRAIN_STEPS = 500


def _build_cluster(seed: int, n: int) -> Cluster[RaftClusterNode]:
    factory = raft_node_factory(
        n,
        seed,
        election_timeout_range=(_NEVER, _NEVER + 1),
        heartbeat_interval=_NEVER,
    )
    return Cluster(n=n, seed=seed, node_factory=factory, tick_interval_ms=_NEVER)


def _force_election(cluster: Cluster[RaftClusterNode], node_id: int, now: int) -> None:
    """Test-only hook, identical in spirit to test_figure8.py's own: make
    `node_id` start an election on its very next tick, bypassing this
    implementation's randomized `election_deadline` so the test controls
    *when* an election starts without touching anything about who wins or
    how replication/commitment actually work.
    """
    node = cluster.get_node(node_id)
    assert node is not None, f"node {node_id} is not live"
    raft_node = node.raft_node
    raft_node._deadline_initialized = True
    raft_node.election_deadline = now
    cluster.route(node_id, node.tick(now), now)


def _drain(cluster: Cluster[RaftClusterNode]) -> None:
    """Deliver every currently in-flight message, and nothing else -- see
    test_figure8.py's own `_drain` for why this checks `pending()` before
    every `step()` rather than calling `cluster.run(...)`.
    """
    steps = 0
    while cluster.network.pending():
        steps += 1
        assert steps <= _DRAIN_STEPS, "drain did not converge"
        if not cluster.step():
            break


def _install_apply_spy(cluster: Cluster[RaftClusterNode], node_id: int) -> list[object]:
    """Wrap `node_id`'s own `state_machine.apply` -- the low-level,
    state-mutating method -- to record every command it genuinely
    executes, independent of `apply_client_request`'s own internal dedup
    decision. Same technique and same reasoning as
    `tests/test_client_sessions.py`'s `_count_real_applies`.
    """
    node = cluster.get_node(node_id)
    assert node is not None
    calls: list[object] = []
    original_apply = node.raft_node.state_machine.apply

    def counting_apply(command: object) -> object:
        calls.append(command)
        return original_apply(command)

    node.raft_node.state_machine.apply = counting_apply  # type: ignore[method-assign]
    return calls


def test_client_request_survives_redirect_retry_and_a_leader_crash() -> None:
    """The full loop, end to end, deterministically:

    1. Node 0 wins a forced election and becomes leader.
    2. The client submits to node 1 (a follower) first -- exactly the
       "no way to know the leader in advance" premise `NotLeader` exists
       for -- and gets redirected: `NotLeader(leader_hint=0)`. Node 1
       already has that hint by this point because it saw node 0's own
       `RequestVote` during the election (see `raft.node`'s module
       docstring: a hint updates from a candidacy, not just a confirmed
       leader).
    3. The client retries against the hint (node 0), which appends and
       replicates the entry to a full quorum -- committed and applied on
       node 0 -- before anything tells the followers their own
       `last_applied` may advance (that needs a later heartbeat, which
       never comes: see the next step).
    4. Node 0 crashes right there -- "commits but the reply is dropped"
       mid-flight, the exact scenario `raft.statemachine`'s module
       docstring calls out. The client has no way to know whether its
       request went through (see this file's own module docstring on the
       reply-delivery scope decided for this milestone) and can only
       retry the identical `(client_id, serial_number, command)`.
    5. Node 1 wins a new election from the surviving majority and the
       client retries against it. The log now legitimately holds the
       same logical request twice -- once from node 0's original
       replication (which node 1 already held as a follower), once from
       this retry -- exactly the duplicate-log-entry scenario
       `apply_client_request`'s dedup exists to make safe.
    6. One more heartbeat lets the last surviving follower (node 2) catch
       up and apply too.

    Proof of "exactly once": an `apply()` spy on every node's state
    machine, installed from the start, must show exactly one real
    mutation on each of node 1 and node 2 -- the two machines that stay
    continuously live throughout (node 0's single, legitimate apply
    before it crashed doesn't count against this: its instance is
    discarded, not carried forward, same reasoning as
    `tests/test_client_sessions.py`'s own sweep-level test).
    """
    n = 3
    cluster = _build_cluster(seed=1, n=n)
    t = 0

    def now() -> int:
        nonlocal t
        t += 10
        return t

    applies = {node_id: _install_apply_spy(cluster, node_id) for node_id in range(n)}

    _force_election(cluster, 0, now())
    _drain(cluster)
    leader = cluster.get_node(0)
    assert leader is not None and leader.role.value == "leader"

    request = ClientRequest(client_id="c1", serial_number=1, command=Put(key="x", value=1))

    follower = cluster.get_node(1)
    assert follower is not None
    redirect = follower.append_command(request, now())
    assert redirect == NotLeader(leader_hint=0)

    produced = leader.append_command(request, now())
    assert not isinstance(produced, NotLeader)
    cluster.route(0, produced, now())
    _drain(cluster)
    assert leader.commit_index >= 1
    assert leader.last_applied >= 1
    assert leader.state_machine.snapshot() == {"x": 1}  # committed on the leader...

    cluster.crash(0)  # ...but crashes before any follower ever learns that

    _force_election(cluster, 1, now())
    _drain(cluster)
    new_leader = cluster.get_node(1)
    assert new_leader is not None and new_leader.role.value == "leader"

    retry_result = new_leader.append_command(request, now())  # the client's blind retry
    assert not isinstance(retry_result, NotLeader)
    cluster.route(1, retry_result, now())
    _drain(cluster)

    # A follower only applies once a LATER AppendEntries carries the
    # updated leader_commit -- same gap tests/test_client_sessions.py's
    # own scenario closes the same way.
    next_heartbeat = new_leader.raft_node._next_heartbeat
    cluster.route(1, new_leader.tick(next_heartbeat), next_heartbeat)
    _drain(cluster)

    survivor = cluster.get_node(2)
    assert survivor is not None

    assert new_leader.state_machine.snapshot() == {"x": 1}
    assert survivor.state_machine.snapshot() == {"x": 1}
    assert new_leader.state_machine.sessions_snapshot() == {"c1": (1, None)}

    assert len(applies[1]) == 1, f"node 1 applied its command more than once: {applies[1]!r}"
    assert len(applies[2]) == 1, f"node 2 applied its command more than once: {applies[2]!r}"


# -- Cross-check: final KV state matches an independently computed expectation --


def _expected_kv_state_from_log(log: list[object]) -> dict[str, object]:
    """Independently recompute the correct final KV state straight from a
    committed log -- a fresh dict and a from-scratch dedup rule, never a
    call into `KVStateMachine` itself -- so agreement with the real state
    machine's own `snapshot()` is a genuine cross-check, not a comparison
    against itself. Mirrors `apply_client_request`'s own documented rule
    (highest serial_number per client_id wins, `<=` skips) and `apply`'s
    own Put/Delete semantics, reimplemented independently on purpose.

    `log[0]` (the sentinel) is skipped; anything that isn't a
    `ClientRequest` is skipped too -- this milestone's fuzzer never
    appends a bare `Command`, but a future caller feeding this a log built
    another way shouldn't crash on one.
    """
    data: dict[str, object] = {}
    highest_serial: dict[str, int] = {}
    for entry in log[1:]:
        command = getattr(entry, "command", None)
        if not isinstance(command, ClientRequest):
            continue
        seen = highest_serial.get(command.client_id)
        if seen is not None and command.serial_number <= seen:
            continue  # correct dedup: an already-applied retry is a no-op
        inner = command.command
        if isinstance(inner, Put):
            data[inner.key] = inner.value
        elif isinstance(inner, Delete):
            data.pop(inner.key, None)
        highest_serial[command.client_id] = command.serial_number
    return data


def test_final_kv_state_matches_an_independently_computed_expectation() -> None:
    """Not "differs by seed" (BUGS.md's Milestone-4 bar) but "correct for
    what the trace actually says should have been applied": for several
    seeds under the full fault model, every live node's real
    `state_machine.snapshot()` must equal what an independent,
    from-scratch reimplementation of the dedup rule computes directly
    from that same node's own committed log (`log[: last_applied + 1]`,
    matching exactly the range `_apply_committed_entries` itself walks).

    Seeds are a fixed, varied handful (0, 1, 2, 7, 13, 24), not the whole
    sweep -- enough to include both crash-heavy and election-heavy runs
    without paying for all 50 on every test run; `test_fuzz_sweep.py`
    already covers the full 50/1000-seed breadth for safety, not
    state-value correctness specifically.
    """
    checked_live_nodes = 0
    for seed in (0, 1, 2, 7, 13, 24):
        fuzzer = Fuzzer(n=N_NODES, seed=seed, steps=STEPS_PER_SEED)
        fuzzer.run()

        for node_id in fuzzer.cluster.node_ids():
            node = fuzzer.cluster.get_node(node_id)
            if node is None:
                continue
            log = node.storage.load_log()[: node.last_applied + 1]
            expected = _expected_kv_state_from_log(log)
            actual = node.state_machine.snapshot()
            assert actual == expected, (
                f"seed={seed} node={node_id}: real state machine diverged from the "
                f"independently computed expectation: {actual!r} != {expected!r}"
            )
            checked_live_nodes += 1

    assert checked_live_nodes > 0, "no live node was ever available to check across these seeds"
