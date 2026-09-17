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

- Leader Completeness (`check_leader_completeness`): once an entry at some
  term has been reported committed by *any* node, every leader of every
  *later* term must have that exact entry. A single snapshot can't answer
  this either -- the node that committed it may since have crashed, and
  nothing about a cluster's current state says what used to be true -- so
  this also takes a `history` accumulator: a `dict[int, CommittedEntry]`
  mapping each index ever observed committed to both the term it was
  written at and the term whose commit *decision* actually made it safe
  (not always the same term -- see `check_leader_completeness` for why).
  Recording is a one-way ratchet (an index, once recorded, is never
  revisited or overwritten) and only scans each node's *newly* committed
  range past a shared high-water mark, so accumulating it costs O(entries
  newly committed since the last call), not O(log length), every time.
- State Machine Safety (`check_state_machine_safety`): if any node has
  *applied* an entry at some index (`last_applied` has advanced past it --
  distinct from that index merely being committed), no node ever applies
  a different entry at that index. Nothing in this codebase advances
  `last_applied` yet (there is no state machine to apply *to* -- see
  `raft.node`'s module docstring), so this checker currently has nothing
  to ever observe and stays vacuously satisfied; it's implemented now, in
  full, so it's ready the moment something starts advancing it. Its
  accumulator is an `AppliedEntryHistory`: one dict recording the first
  value ever observed applied at each index, and a second tracking how
  far *each node's own* `last_applied` has already been scanned into it --
  two dicts, not one, because unlike commit_index for Leader Completeness,
  different nodes apply at genuinely different paces, so a single shared
  high-water mark would silently skip re-checking a slower node's own
  catch-up range against what a faster node already recorded.

A crashed node has no live state to inspect -- `Cluster.get_node` already
returns `None` for one -- so every checker here simply skips it rather
than treating a crash as a violation or an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    """No node's log ever loses or overwrites an entry except as Figure 2's
    own conflict-resolution rule permits: a genuine term mismatch at the
    first point of difference, immediately followed by the new entries
    that mismatch introduced.

    This used to be an object-identity check ("was an entry replaced by a
    *different* object holding an *equal* value"), looking for a follower
    that tears an entry down and rebuilds it identically instead of
    recognizing it already matches and leaving it alone. That signature
    turned out to be structurally unreachable through this simulator: no
    `LogEntry` is ever cloned, so a resend built from a leader's own
    unchanged storage always carries the exact same object references a
    follower already stored from an earlier delivery -- true regardless of
    how many times, or in what order, that resend is delivered. Meanwhile
    the real bug this checker was meant to catch (a follower that always
    truncates+reappends on `AppendEntries`, even when nothing conflicts)
    turned out to have a much more damaging failure mode the identity
    check couldn't see at all: when a *stale, shorter* `AppendEntries` --
    built from an earlier, smaller `next_index`, sent before the leader's
    log grew further -- is delivered by the (reordering) `Network` *after*
    a later, longer one already extended the follower past it, blindly
    truncating from that stale message's `prev_log_index + 1` discards
    every entry the follower held *beyond* that message's own range,
    including ones a leader may have already committed. Nothing then puts
    those entries back, because as far as anything else in the system can
    tell, this follower simply never had them.

    Two independent facts, not one, made the old identity check miss this:
    (a) `next_index` genuinely never walks backward past a point of real
    agreement (still true, and still the reason a *rejection*-driven retry
    can never resend something already known-matching); and (b)
    `LogEntry` objects are never cloned, so any resend from a given
    leader's own storage -- whatever `next_index` it was built from, and
    regardless of delivery order -- is always identity-equal to what that
    same leader already sent before. (a) says nothing about (b), and (b)
    is what actually keeps the identity signature unreachable: the harmful
    case above is a live counterexample to ever treating those two facts
    as one combined argument again. See `BUGS.md` for the full seed-3 /
    seed-35 diagnosis this rewrite is based on.

    The rule, compared by *value* (`LogEntry` is a `NamedTuple`; `==`
    compares `term` and `command`, never identity), not object identity:
    find the first index `k` (scanning only through `previous_log`'s own
    length -- entries appended past it are ordinary growth, not
    divergence) where `current_log` no longer matches `previous_log`,
    including `k` falling off the end of a now-shorter `current_log`
    entirely. If there is no such `k`, nothing was lost or changed.
    Otherwise: if `current_log` still has *some* entry at `k` and its
    `term` differs from `previous_log`'s entry there, this is Figure 2
    rule 3 (delete the conflicting entry onward) immediately followed by
    rule 4 (append what's new) -- a legitimate, expected truncation, and
    passes. Anything else -- the entry at `k` is simply gone with nothing
    at `k` at all, or it's still there but *unchanged in term* (so no
    conflict justified touching it), or it's there with the same term but
    a different `command` (a Log Matching Property violation on its own)
    -- means an entry was lost or rewritten without the one justification
    Figure 2 allows, and raises.

    Unlike `check_leader_append_only`, this tracks *every* live node,
    every step, regardless of role -- the bug it watches for lives on the
    follower side. The baseline is never cleared for a role change (a
    *value* change at the first point of divergence is fine exactly when
    Figure 2 says so, in any role). Crashed nodes are skipped, exactly as
    elsewhere -- `Cluster.crash()` only discards volatile state, so a
    crashed node's `Storage` is untouched and there's nothing to
    invalidate about a baseline recorded before the crash.

    Design note on a case this function deliberately does *not*
    special-case: `MemoryStorage.fail_after` can raise `StorageError`
    between `truncate_from` and the `append_entries` that Figure 2's rules
    3 and 4 expect to follow it in the same call, leaving a real node with
    a genuinely shorter log and no term conflict to justify it -- exactly
    the shape this function raises on. Today, nothing in `Cluster` or
    `Fuzzer` ever calls `fail_after`, so no fuzzer run can reach this
    state; this function does not add an exemption for it anyway, for two
    reasons. First, there is nothing here to test: an exemption with no
    reachable path to exercise it is dead code by construction. Second,
    and more importantly, if this ever *did* happen, flagging it would be
    correct, not a false positive -- `Cluster._deliver_one` has no
    `try/except` around `node.handle(...)`, so a raised `StorageError`
    would propagate out of `Fuzzer.run()` rather than being caught and
    translated into a `Cluster.crash()` call the way a real crash is
    modeled everywhere else in this harness. The node would stay in
    `Cluster`'s live set, indistinguishable from any other live node, with
    a log that silently lost entries a majority may already have
    committed -- the identical hazard this whole function exists to
    catch. Treating that case as exempt would mean quietly trusting a live
    node's log for exactly the reason it can't be trusted.
    """
    for node_id in cluster.node_ids():
        node = cluster.get_node(node_id)
        if node is None:
            continue  # crashed: Storage (and every entry already in it) is untouched

        current_log = node.storage.load_log()
        previous_log = history.get(node.node_id)

        if previous_log is not None:
            divergence = _first_divergence(previous_log, current_log)
            if divergence is not None:
                still_present = divergence < len(current_log)
                conflicting_term = (
                    still_present and current_log[divergence].term != previous_log[divergence].term
                )
                if not conflicting_term:
                    now = repr(current_log[divergence]) if still_present else "<removed>"
                    raise SafetyViolation(
                        f"node {node.node_id!r} log entry at index {divergence} was lost "
                        f"or changed without a term conflict to justify it: had "
                        f"{previous_log[divergence]!r}, now {now}",
                        seed=cluster.seed,
                        step=cluster.step_count,
                    )

        history[node.node_id] = current_log


def _first_divergence(
    previous_log: list[LogEntry], current_log: list[LogEntry]
) -> int | None:
    """The first index where `current_log` no longer matches `previous_log`.

    Compares by value (`LogEntry` equality: `term` and `command`), never
    by object identity. Only scans through `previous_log`'s own length --
    an index `current_log` has grown into beyond that is ordinary
    replication, not divergence. An index `previous_log` had that
    `current_log` no longer reaches at all (the log got shorter) counts
    as a divergence there, the same as a value that changed in place.
    """
    for index, old_entry in enumerate(previous_log):
        if index >= len(current_log) or current_log[index] != old_entry:
            return index
    return None


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


@dataclass(frozen=True)
class CommittedEntry:
    """One index's committed history, as recorded by `check_leader_completeness`.

    `term` is the entry's own term -- what a future leader's log at this
    index must match, per the Log Matching Property. `established_at_term`
    is the term of whichever leader's commit *decision* first covered
    this index -- not necessarily the same term, and it's the one that
    matters for deciding which future leaders are even bound by this at
    all. See `check_leader_completeness` for why these two can differ.
    """

    term: int
    established_at_term: int


def check_leader_completeness(
    cluster: Cluster[RaftClusterNode], history: dict[int, CommittedEntry]
) -> None:
    """If an entry was committed at some term, every leader of every later
    term has that exact entry.

    `history` maps each index ever observed committed to a
    `CommittedEntry` -- see the module docstring for why this needs
    memory a single snapshot can't provide (the node that committed it
    may since have crashed) and why recording is safe to do with a single
    shared high-water mark across every node (unlike
    `check_state_machine_safety`'s two-dict accumulator): this function
    never needs to compare one node's committed range against another's,
    only to remember the first observation at each index, so it's fine if
    a fast node's commit_index is what ends up recording an index a
    slower node would also, eventually, reach.

    Why `term` and `established_at_term` can differ: Figure 8's
    "current-term-only" restriction lets a leader commit a whole prefix
    in one step -- committing its own current-term entry at index N also
    commits every earlier-term entry below N, riding along via the Log
    Matching Property (see `RaftNode._advance_commit_index`). Those
    earlier entries are only actually *guaranteed* from that leader's
    OWN current term onward -- the safety proof runs on "a majority held
    the leader's entire log through N at the moment of the term-C
    decision", not on the older entry's own original term. A leader
    elected in some term between the entry's own term and C was never
    part of that guarantee and legitimately might not have it yet
    (nothing wrong happened; the entry just wasn't safe yet when that
    leader won). Gating on the entry's own term instead of C is exactly
    the bug this two-field design replaced: it flagged a real, honestly
    elected lower-term leader as "missing" an entry that hadn't actually
    been protected yet at the time it was elected.

    A node's own `commit_index` is always <= its own `last_log_index`
    (`_advance_commit_index` never picks a candidate above it, and a
    follower's own adoption of `leader_commit` is clamped by the same
    bound -- see `RaftNode._handle_append_entries`, step 5), so every
    index this function reads off a node's log, up through its
    commit_index, is guaranteed to actually be there.
    """
    high_water = max(history) if history else 0
    for node_id in cluster.node_ids():
        node = cluster.get_node(node_id)
        if node is None or node.commit_index <= high_water:
            continue  # crashed, or nothing newly committed on this node

        log = node.storage.load_log()
        for index in range(high_water + 1, node.commit_index + 1):
            if index not in history:
                history[index] = CommittedEntry(
                    term=log[index].term, established_at_term=node.current_term
                )
        high_water = max(high_water, node.commit_index)

    for node_id in cluster.node_ids():
        node = cluster.get_node(node_id)
        if node is None or node.role is not Role.LEADER:
            continue

        log = node.storage.load_log()
        for index, committed in history.items():
            if committed.established_at_term >= node.current_term:
                continue  # this leader predates (or ties) the guarantee
            if index >= len(log) or log[index].term != committed.term:
                raise SafetyViolation(
                    f"leader {node.node_id!r} (term {node.current_term}) is "
                    f"missing entry (index={index}, term={committed.term}), "
                    f"safe since term {committed.established_at_term}",
                    seed=cluster.seed,
                    step=cluster.step_count,
                )


@dataclass
class AppliedEntryHistory:
    """Accumulator for `check_state_machine_safety`.

    Two dicts, not one -- see the module docstring for why a single
    shared high-water mark (sufficient for `check_leader_completeness`)
    isn't here: different nodes apply at different paces, and this
    function, unlike that one, must compare every node's own applied
    range against what's recorded, not just extend the record.
    """

    applied: dict[int, LogEntry] = field(default_factory=dict)
    """The first value ever observed applied at each index."""

    checked_through: dict[str, int] = field(default_factory=dict)
    """How far each node's own `last_applied` has already been scanned."""


def check_state_machine_safety(
    cluster: Cluster[RaftClusterNode], history: AppliedEntryHistory
) -> None:
    """If any node has applied an entry at some index, no node -- including
    that same one, later, after some corruption -- ever applies a
    different entry at that index.

    "Applied" means `last_applied` has advanced past the index, not
    merely `commit_index` -- see the module docstring for why nothing in
    this codebase can trigger that yet, and why this is still implemented
    now regardless.
    """
    for node_id in cluster.node_ids():
        node = cluster.get_node(node_id)
        if node is None:
            continue  # crashed: no live state to inspect

        checked_through = history.checked_through.get(node.node_id, 0)
        if node.last_applied <= checked_through:
            continue  # nothing newly applied on this node since last checked

        log = node.storage.load_log()
        for index in range(checked_through + 1, node.last_applied + 1):
            entry = log[index]
            previously_applied = history.applied.get(index)
            if previously_applied is not None and previously_applied != entry:
                raise SafetyViolation(
                    f"node {node.node_id!r} applied {entry!r} at index {index}, "
                    f"but {previously_applied!r} was already applied there, "
                    f"elsewhere",
                    seed=cluster.seed,
                    step=cluster.step_count,
                )
            history.applied[index] = entry

        history.checked_through[node.node_id] = node.last_applied
