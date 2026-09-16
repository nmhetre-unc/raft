"""Figure 8 of the Raft paper (S5.4.2), reconstructed exactly, deterministically,
through the real `Cluster` -- no fuzzing.

The scenario in full, servers numbered S1..S5 (indices 0..4 below):

    a) S1 leads term 2, appends an entry at index 2, replicates it to S2
       only. Not committed -- two of five. S1 crashes.
    b) S5 wins term 3, voted for by S3, S4, and itself, none of which hold
       index 2. S5 appends its own entry at index 2, term 3. Crashes
       before replicating.
    c) S1 restarts, wins term 4, replicates its OLD term-2 entry at index 2
       to S2 and S3. Three of five now hold it.
    d) The trap: a naive leader commits index 2 here, on the count alone.
    e) S1 crashes. S5 wins term 5 -- its term-3 entry at index 2 beats the
       term-2 entries on S3 and S4 in the up-to-date comparison. S5
       overwrites index 2 everywhere.

Two things need setting up before (a) can even start, and both are done
through the real election/replication code paths rather than assumed:

- A common ancestor entry at index 1, term 1, held by all five servers --
  Figure 8's diagram starts from this state without showing how it got
  there. It's produced here by a first, throwaway election (S2 leads term
  1, replicates one entry, fully converges) that (a) never otherwise
  touches.
- S1 must reach term 2 as a *fresh* candidacy, not as a continuation of
  some earlier leadership of its own -- a sitting RaftNode leader has no
  election timer to force in this implementation (Figure 2: only
  followers and candidates run one). Using S2, not S1, as the throwaway
  term-1 leader sidesteps that for free: S1 spends term 1 as an ordinary
  follower, so forcing its election is just forcing a follower's timeout,
  and its RequestVote(term=2) deposes S2 the same way any higher-term
  message would.

Test-only hook -- `_force_election`: nothing in `RaftNode` lets a caller
pick *which* node wins a given election; that's decided by whichever
randomized `election_deadline` expires first. `_force_election` reaches
past that by writing directly to `election_deadline` and
`_deadline_initialized` -- the same private seam `tests/test_election.py`
and `tests/test_replication.py` already poke on bare `RaftNode`s, applied
here through `RaftClusterNode.raft_node` so it works on a `Cluster`-driven
node instead. Every node's own `election_timeout_range` and
`heartbeat_interval` are set far beyond this test's clock range so nothing
times out *except* where `_force_election` says so. Only *when* an
election starts is being dictated this way -- who wins, and everything
about replication and commitment, still runs through the real RequestVote
/ AppendEntries / majority / log-currency code.

S1's two-term climb (attempt at term 3 falls short, retry at term 4 wins)
falls out of this for free rather than needing to be engineered: S3 and S4
already gave term 3 to S5 in step (b), so S1's first post-restart attempt
-- which can only ever claim term 3, one past its own last known term --
is rejected by both (already voted, this term) and never reaches a
majority. Forcing a *second* election on the same (still-a-candidate)
node is exactly what a real election-timeout retry does, and *that*
attempt claims term 4, which nobody has given away yet.

Reachability check the task asked for explicitly: step (b) requires S5 to
win with a log that is behind S1's and S2's. It is reachable, with no
workaround needed -- S3 and S4 never received index 2 (they were never
sent it; no partition was even required to keep it from them), so their
own last log term (1) is behind S5's candidacy exactly as much as it's
behind S1's, and Figure 2's up-to-date check only ever compares a
candidate against *the voter's own* log, never against some third node's.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from raft.node import RaftClusterNode, RaftNode, Role, raft_node_factory
from raft.sim.cluster import Cluster
from raft.storage import LogEntry

N = 5
S1, S2, S3, S4, S5 = range(5)

# Large enough that nothing in this test's clock range (a few hundred
# milliseconds) ever reaches it -- elections in this test are started
# exclusively by _force_election, never by a node's own timer.
_NEVER = 10**9

# cluster.run() drains pending network traffic only (see _build_cluster:
# tick_interval_ms is _NEVER, so the periodic-tick branch of Cluster.step()
# never fires here) -- generous relative to the handful of in-flight
# messages a 5-node cluster ever produces at once.
_DRAIN_STEPS = 500


def _build_cluster(seed: int) -> Cluster[RaftClusterNode]:
    factory = raft_node_factory(
        N,
        seed,
        election_timeout_range=(_NEVER, _NEVER + 1),
        heartbeat_interval=_NEVER,
    )
    return Cluster(n=N, seed=seed, node_factory=factory, tick_interval_ms=_NEVER)


def _force_election(cluster: Cluster[RaftClusterNode], node_id: int, now: int) -> None:
    """Test-only hook: make `node_id` start an election on its very next tick.

    See the module docstring for why this is necessary and what it does
    and doesn't control. `now` must be >= cluster.clock.now() -- exactly
    the same constraint any other caller driving this cluster's `now`
    values has to respect.
    """
    node = cluster.get_node(node_id)
    assert node is not None, f"node {node_id} is not live"
    raft_node = node.raft_node
    raft_node._deadline_initialized = True
    raft_node.election_deadline = now
    cluster.route(node_id, node.tick(now), now)


def _drain(cluster: Cluster[RaftClusterNode]) -> None:
    """Deliver every currently in-flight message, and nothing else.

    Deliberately not `cluster.run(_DRAIN_STEPS)`: `Cluster.step()` always
    processes *something* -- the next due message, or, once none is
    pending, the next due tick, no matter how far off it is. With
    `tick_interval_ms` set to `_NEVER` (see `_build_cluster`), the instant
    this cluster's message queue runs dry, `step()` would jump `clock`
    all the way to `_NEVER` and tick every live node there -- silently
    running a huge number of elections this test never asked for. Checking
    `pending()` before every `step()` call stops the moment there's
    nothing left in flight, so the tick branch of `step()` is never
    reached at all.
    """
    steps = 0
    while cluster.network.pending():
        steps += 1
        assert steps <= _DRAIN_STEPS, "drain did not converge"
        if not cluster.step():
            break


def _entry_at(cluster: Cluster[RaftClusterNode], node_id: int, index: int) -> LogEntry | None:
    node = cluster.get_node(node_id)
    if node is None:
        return None
    log = node.storage.load_log()
    return log[index] if index < len(log) else None


def _live(cluster: Cluster[RaftClusterNode], node_id: int) -> RaftClusterNode:
    node = cluster.get_node(node_id)
    assert node is not None, f"node {node_id} is not live"
    return node


def _naive_advance_commit_index(self: RaftNode) -> None:
    """The rule Figure 8 exists to rule out: commit on majority match_index
    alone, no matter which term wrote the entry.

    A copy of `RaftNode._advance_commit_index` with exactly one thing
    removed -- the `log[candidate].term != self.current_term` guard --
    so the counterfactual half of this test exercises identical
    majority-counting logic and differs only in the one restriction under
    test.
    """
    match_counts = [*self.match_index.values(), self.storage.last_log_index()]
    majority_needed = len(match_counts) // 2 + 1
    for candidate in range(self.storage.last_log_index(), self.commit_index, -1):
        confirmations = sum(1 for match_index in match_counts if match_index >= candidate)
        if confirmations >= majority_needed:
            self.commit_index = candidate
            return


@dataclass
class Figure8Outcome:
    # S1's own commit_index right after step (c) -- the trap point (d).
    commit_index_at_trap: int
    # What every live node's log holds at index 2 at that same instant.
    index2_at_trap: dict[int, LogEntry | None]
    # What every live node's log holds at index 2 after step (e) settles.
    index2_final: dict[int, LogEntry | None]
    # index 1 (the term-1 common ancestor) on every live node, at the end --
    # nothing in this scenario should ever be able to touch it.
    index1_final: dict[int, LogEntry | None]


def _run_figure8(cluster: Cluster[RaftClusterNode]) -> Figure8Outcome:
    t = 0

    def now() -> int:
        nonlocal t
        t += 10
        return t

    # -- Common ancestor: S2 leads term 1, replicates one entry everywhere. --
    _force_election(cluster, S2, now())
    _drain(cluster)
    bootstrap = _live(cluster, S2)
    assert bootstrap.role is Role.LEADER and bootstrap.current_term == 1

    cluster.route(S2, bootstrap.append_command("base", now()), t)
    _drain(cluster)
    for node_id in range(N):
        assert _entry_at(cluster, node_id, 1) == LogEntry(term=1, command="base")

    # -- (a): S1 leads term 2 (deposing S2 the same way any higher-term
    # message would), appends index 2, replicates to S2 only, crashes. --
    _force_election(cluster, S1, now())
    _drain(cluster)
    s1 = _live(cluster, S1)
    assert s1.role is Role.LEADER and s1.current_term == 2

    cluster.partition({S1, S2}, {S3, S4, S5})
    cluster.route(S1, s1.append_command("only-s2", now()), t)
    _drain(cluster)
    assert _entry_at(cluster, S2, 2) == LogEntry(term=2, command="only-s2")
    assert _entry_at(cluster, S3, 2) is None
    assert _entry_at(cluster, S4, 2) is None
    assert _entry_at(cluster, S5, 2) is None

    cluster.crash(S1)
    cluster.heal()

    # -- (b): S5 wins term 3 on S3 + S4's votes (S2's log is too far ahead
    # to grant it; S1 is down). Appends its own index 2, then crashes
    # before that append_command's output is ever routed -- "replicates
    # to no one" is enforced by simply never sending it, not by a
    # partition. --
    _force_election(cluster, S5, now())
    _drain(cluster)
    s5 = _live(cluster, S5)
    assert s5.role is Role.LEADER and s5.current_term == 3

    s5.append_command("only-s5", now())  # produced output deliberately discarded
    cluster.crash(S5)

    assert _entry_at(cluster, S3, 2) is None
    assert _entry_at(cluster, S4, 2) is None

    # -- (c): S1 restarts. Partitioned off from S4 (and crashed S5) for
    # this whole step, so S4 categorically cannot receive anything from
    # it -- matching "replicates to S2 and S3" without relying on timing
    # to stop short. S1's first re-election attempt can only claim term 3,
    # which S3 already gave to S5; it retries and wins term 4. --
    cluster.restart(S1)
    cluster.partition({S1, S2, S3}, {S4, S5})

    _force_election(cluster, S1, now())
    _drain(cluster)
    s1 = _live(cluster, S1)
    assert s1.role is Role.CANDIDATE, "term-3 attempt should fall short: S3 already voted for S5"

    _force_election(cluster, S1, now())
    _drain(cluster)
    s1 = _live(cluster, S1)
    assert s1.role is Role.LEADER and s1.current_term == 4

    assert _entry_at(cluster, S2, 2) == LogEntry(term=2, command="only-s2")
    assert _entry_at(cluster, S3, 2) == LogEntry(term=2, command="only-s2")
    assert _entry_at(cluster, S4, 2) is None  # partitioned off: never even reached

    # -- (d): the trap. Three of five (S1, S2, S3) now hold index 2. --
    commit_index_at_trap = s1.commit_index
    index2_at_trap = {node_id: _entry_at(cluster, node_id, 2) for node_id in range(N)}

    # -- (e): S1 crashes again. S5 restarts; its first attempt can only
    # claim term 4, which S2 and S3 already gave to S1, so it retries and
    # wins term 5 -- on S2, S3, and S4's votes, its term-3 entry at index 2
    # beating S2/S3's term 2 and S4's term 1. It then overwrites index 2
    # everywhere. --
    cluster.crash(S1)
    cluster.heal()
    cluster.restart(S5)

    _force_election(cluster, S5, now())
    _drain(cluster)
    s5 = _live(cluster, S5)
    assert s5.role is Role.CANDIDATE, "term-4 attempt should fall short: S1 already holds it"

    _force_election(cluster, S5, now())
    _drain(cluster)
    s5 = _live(cluster, S5)
    assert s5.role is Role.LEADER and s5.current_term == 5

    _drain(cluster)  # let the backtracking retries fully converge

    index2_final = {node_id: _entry_at(cluster, node_id, 2) for node_id in range(N)}
    index1_final = {node_id: _entry_at(cluster, node_id, 1) for node_id in range(N)}

    return Figure8Outcome(
        commit_index_at_trap=commit_index_at_trap,
        index2_at_trap=index2_at_trap,
        index2_final=index2_final,
        index1_final=index1_final,
    )


def test_figure8_guarded_leader_never_commits_the_old_term_entry() -> None:
    cluster = _build_cluster(seed=1)
    result = _run_figure8(cluster)

    # (d): 3 of 5 hold the term-2 entry at index 2 when S1 wins term 4 -- a
    # majority -- but it was written in term 2, not S1's current term (4),
    # so the guard must refuse to commit it.
    assert result.commit_index_at_trap < 2
    assert result.index2_at_trap[S1] == LogEntry(term=2, command="only-s2")

    # (e): S5 legitimately overwrites index 2 everywhere it was held.
    # Legitimate, specifically, *because* nothing ever reported it
    # committed -- there is nothing here for the overwrite to violate.
    for node_id in (S2, S3, S4, S5):
        assert result.index2_final[node_id] == LogEntry(term=3, command="only-s5")

    # Whatever the leader's guard *did* let through committed (the term-1
    # common ancestor, at minimum) is untouched by any of this, on every
    # live node.
    for node_id in (S2, S3, S4, S5):
        assert result.index1_final[node_id] == LogEntry(term=1, command="base")


def test_figure8_without_the_guard_commits_and_then_overwrites_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A test that only ran the guarded version proves nothing about the
    guard itself -- this is its other half. Same sequence, same seed,
    with `_advance_commit_index`'s current-term restriction deleted:
    the trap in (d) is sprung for real, and the entry it wrongly commits
    is the exact one (e) then overwrites.
    """
    monkeypatch.setattr(RaftNode, "_advance_commit_index", _naive_advance_commit_index)

    cluster = _build_cluster(seed=1)
    result = _run_figure8(cluster)

    # (d): the naive, count-only rule commits index 2 on 3-of-5 alone.
    assert result.commit_index_at_trap >= 2
    committed_entry = result.index2_at_trap[S1]
    assert committed_entry == LogEntry(term=2, command="only-s2")

    # (e): the same election result as the guarded run -- S5 still
    # legitimately wins term 5 and still overwrites index 2 -- but this
    # time that entry had been reported committed. Overwriting it is
    # exactly the State Machine Safety violation Figure 8 demonstrates.
    for node_id in (S2, S3, S4, S5):
        assert result.index2_final[node_id] == LogEntry(term=3, command="only-s5")
        assert result.index2_final[node_id] != committed_entry
