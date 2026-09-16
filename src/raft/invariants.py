"""Checkers for Raft's safety properties (Figure 3 of the paper).

Each checker takes a `Cluster` (and, where the property needs history
across time, a caller-owned accumulator) and either returns cleanly or
raises `SafetyViolation`. Nothing here mutates the cluster -- these are
read-only observers meant to be called after every step, or periodically,
by a test or a future fuzzer.

Implemented now -- the two properties expressible from a cluster's
current, observable state alone:

- Election Safety (`check_election_safety`): at most one leader per term.
- Log Matching (`check_log_matching`): two entries at the same index and
  term imply identical logs through that index.
- Leader Append-Only (`check_leader_append_only`): a leader's log only
  ever grows, never shrinks or rewrites an entry it already has. This one
  genuinely needs memory -- a single snapshot can't show whether a log
  *changed* -- so it takes a `history` accumulator the caller keeps
  across calls.
- No Spurious Truncation (`check_no_spurious_truncation`): no *node's*
  log -- leader or follower -- ever tears down and rebuilds an entry that
  already matched what was being sent. Also needs a `history`
  accumulator, for the same reason, and catches a class of bug
  `check_leader_append_only` structurally cannot: that one only ever
  looks at a node while it currently holds the leader role, but a
  blind-truncate bug lives in the *follower*-side receiver.

Not yet implemented -- Leader Completeness and State Machine Safety both
depend on knowing which entries are *committed*, and there is no
commitIndex anywhere in this codebase yet (Milestone 3 is election-only).
Both are stubbed to raise `NotImplementedError` naming Milestone 4 (log
replication), rather than silently omitted, so a caller that reaches for
them gets a clear signal instead of an ImportError or a typo'd name.

A crashed node has no live state to inspect -- `Cluster.get_node` already
returns `None` for one -- so every checker here simply skips it rather
than treating a crash as a violation or an error.
"""

from __future__ import annotations

from raft.node import RaftClusterNode, Role
from raft.sim.cluster import Cluster
from raft.storage import LogEntry


class SafetyViolation(AssertionError):
    """A Raft safety property was violated. Carries the seed and step.

    Subclasses `AssertionError` so a violation reads naturally as a
    failed assertion under pytest, while still being a distinct,
    catchable type for anything (a fuzzer's shrinker, say) that wants to
    handle a safety violation specifically rather than any assertion.

    `trace` defaults to empty: the three checkers below never populate it,
    since they see only a `Cluster`, not whatever drove it there. A
    caller with a history of its own -- `raft.sim.fuzz.Fuzzer`, for
    instance -- can catch a violation, attach its trace and the step at
    which it actually noticed, and re-raise the same exception.

    `reason` holds `message` on its own, before the `(seed=..., step=...)`
    suffix gets appended for the exception's displayed text. Comparing
    `reason` (rather than parsing the formatted message back apart) is
    how a shrinker recognizes "the same violation" after `seed` stays
    fixed but `step` legitimately shifts once the trace producing it has
    been edited down.
    """

    def __init__(
        self, message: str, *, seed: int, step: int, trace: tuple[object, ...] = ()
    ) -> None:
        super().__init__(f"{message} (seed={seed}, step={step})")
        self.reason = message
        self.seed = seed
        self.step = step
        self.trace = trace


def check_election_safety(cluster: Cluster[RaftClusterNode]) -> None:
    """At most one leader per term.

    Groups every live node by (current_term, role) and asserts no term
    has more than one node reporting itself as LEADER.
    """
    leaders_by_term: dict[int, list[str]] = {}

    for node_id in cluster.node_ids():
        node = cluster.get_node(node_id)
        if node is None:
            continue  # crashed: no live state to inspect
        if node.role is Role.LEADER:
            leaders_by_term.setdefault(node.current_term, []).append(node.node_id)

    for term, leader_ids in leaders_by_term.items():
        if len(leader_ids) > 1:
            raise SafetyViolation(
                f"term {term} has {len(leader_ids)} leaders: {sorted(leader_ids)}",
                seed=cluster.seed,
                step=cluster.step_count,
            )


def check_leader_append_only(
    cluster: Cluster[RaftClusterNode], history: dict[str, list[LogEntry]]
) -> None:
    """A leader never overwrites or deletes its own log entries.

    A single snapshot of a log can't show whether it *changed* -- that
    needs comparing against an earlier snapshot -- so `history` is an
    accumulator the caller creates once (`history: dict[str,
    list[LogEntry]] = {}`) and passes to every call over the course of a
    run; this function both reads and updates it. It only means anything
    across repeated calls: calling it once tells you nothing, since the
    first observation of any node just becomes its baseline.

    The baseline for a node is cleared whenever that node is not
    *currently* leader (crashed, or simply not the one in charge right
    now), not just tracked forever. That's deliberate: a follower's log
    can legitimately be truncated by a new leader's conflicting entries,
    and if this function kept comparing against a stale pre-step-down
    snapshot, a node that steps down, gets legitimately truncated as a
    follower, and later wins a *later* election would be flagged as
    having violated append-only -- which it never did, since the
    truncation happened while it held no leadership to violate. Each
    continuous stint as leader gets its own, fresh baseline.
    """
    for node_id in cluster.node_ids():
        node = cluster.get_node(node_id)

        if node is None or node.role is not Role.LEADER:
            if node is not None:
                history.pop(node.node_id, None)
            continue

        current_log = node.storage.load_log()
        previous_log = history.get(node.node_id)

        if previous_log is not None:
            if len(current_log) < len(previous_log):
                raise SafetyViolation(
                    f"leader {node.node_id!r} log shrank from "
                    f"{len(previous_log)} to {len(current_log)} entries",
                    seed=cluster.seed,
                    step=cluster.step_count,
                )
            if current_log[: len(previous_log)] != previous_log:
                raise SafetyViolation(
                    f"leader {node.node_id!r} rewrote a log entry it already had",
                    seed=cluster.seed,
                    step=cluster.step_count,
                )

        history[node.node_id] = current_log


def check_no_spurious_truncation(
    cluster: Cluster[RaftClusterNode], history: dict[str, list[LogEntry]]
) -> None:
    """No node's log entry is ever replaced by a *different* object holding
    an *equal* value -- a sign the log was torn down and rebuilt instead
    of correctly recognized as already matching and left alone.

    `check_leader_append_only` catches a log shrinking or an entry's
    *value* changing, but it only watches leaders, and it compares by
    `==`. Neither is enough for the bug this checker exists for: a
    follower's receiver that always truncates from `prev_log_index + 1`
    and reappends, even when the incoming entries are identical to what
    it already has. A resend from the *same* leader's unchanged storage
    would reconstruct the same object references and slip past this --
    but a resend after a leadership change, from a *different* node's
    storage, produces a distinct Python object for the same (term,
    command), which is exactly what a real duplicate-message scenario
    looks like and exactly what an unnecessary truncate-and-rebuild would
    silently replace a perfectly good entry with.

    Unlike `check_leader_append_only`, this tracks *every* live node,
    every step, regardless of role -- the bug it's watching for lives on
    the follower side. The baseline is never cleared for a role change
    (there's no "legitimate while not leader" exception here the way
    there is for shrinking: a *value* change at an index is always fine,
    Figure 2 expects it, but replacing an already-*matching* entry with a
    different object never is, in any role). Crashed nodes are simply
    skipped, exactly as elsewhere -- their `Storage`, and every object
    already in it, is untouched by a crash, so there is nothing to
    invalidate about a baseline recorded before one.
    """
    for node_id in cluster.node_ids():
        node = cluster.get_node(node_id)
        if node is None:
            continue  # crashed: Storage (and every entry already in it) is untouched

        current_log = node.storage.load_log()
        previous_log = history.get(node.node_id)

        if previous_log is not None:
            for index in range(min(len(current_log), len(previous_log))):
                old_entry = previous_log[index]
                new_entry = current_log[index]
                if new_entry is not old_entry and new_entry == old_entry:
                    raise SafetyViolation(
                        f"node {node.node_id!r} log entry at index {index} was "
                        f"replaced by a different (but equal) object -- torn down "
                        f"and rebuilt rather than left alone",
                        seed=cluster.seed,
                        step=cluster.step_count,
                    )

        history[node.node_id] = current_log


def check_log_matching(cluster: Cluster[RaftClusterNode]) -> None:
    """If two logs share an entry at the same index and term, they are
    identical through that index.

    Checked pairwise across every live node's log, including the index-0
    sentinel (which is always `term=0` everywhere by construction, so it
    trivially satisfies this on its own).
    """
    live: list[RaftClusterNode] = []
    for node_id in cluster.node_ids():
        node = cluster.get_node(node_id)
        if node is not None:
            live.append(node)

    for i in range(len(live)):
        for j in range(i + 1, len(live)):
            _check_log_matching_pair(live[i], live[j], cluster)


def _check_log_matching_pair(
    node_a: RaftClusterNode, node_b: RaftClusterNode, cluster: Cluster[RaftClusterNode]
) -> None:
    log_a = node_a.storage.load_log()
    log_b = node_b.storage.load_log()

    for index in range(min(len(log_a), len(log_b))):
        if log_a[index].term != log_b[index].term:
            continue  # different terms at this index: nothing to check here
        if log_a[: index + 1] != log_b[: index + 1]:
            raise SafetyViolation(
                f"log matching violated between {node_a.node_id!r} and "
                f"{node_b.node_id!r}: both have term {log_a[index].term} at "
                f"index {index}, but their logs differ before it",
                seed=cluster.seed,
                step=cluster.step_count,
            )


def check_leader_completeness(cluster: Cluster[RaftClusterNode]) -> None:
    """If a log entry is committed, every future leader has it.

    Not implemented: this needs a commitIndex, which doesn't exist yet --
    Milestone 3 is election-only. Filled in alongside commit tracking in
    Milestone 4 (log replication).
    """
    raise NotImplementedError(
        "check_leader_completeness needs commit tracking; implemented in Milestone 4"
    )


def check_state_machine_safety(cluster: Cluster[RaftClusterNode]) -> None:
    """If a server has applied a log entry at a given index, no other
    server ever applies a different entry for that index.

    Not implemented: this needs a commitIndex (and an apply cursor),
    neither of which exist yet -- Milestone 3 is election-only. Filled in
    alongside commit tracking in Milestone 4 (log replication).
    """
    raise NotImplementedError(
        "check_state_machine_safety needs commit tracking; implemented in Milestone 4"
    )
