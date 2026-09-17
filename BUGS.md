# Bugs

Safety violations found by the fuzzer. Each entry records the seed, the
minimal reproducing trace, the property that broke, and the root cause.

Reproduce any of these with `pytest tests/test_fuzz_sweep.py --seed <N>`.

## Restart steals leadership from a healthy leader

Found by: `test_restart_does_not_disrupt_cluster`

`election_deadline` was drawn once at construction, relative to `now=0`. At
cluster boot that is correct. On `Cluster.restart()`, which happens thousands
of simulated milliseconds in, the restarted node's deadline was already long
expired, so it fired an election on its first tick and could depose a healthy
leader. Every node restart was a cluster disruption.

Fix: defer the first deadline draw to the node's first `tick(now)`, so
construction and restart behave identically.

## Messages silently dropped on node id type mismatch

Found by: no convergence across 20 seeds over 50,000 simulated ms

`RaftNode` addresses peers as `str` while `Cluster` keys nodes by `int`.
Every routed message looked addressed to a nonexistent node and was dropped
with no error. Nodes never heard each other's votes and livelocked in
perpetual re-election — a symptom indistinguishable from a split-vote bug in
the election logic.

Fix: a `RaftClusterNode` adapter translating `int` and `str` at the
`tick`/`handle` boundary.

## A reply could be attributed to the wrong, overlapping send, inflating match_index

Found by: `check_leader_completeness` on the plain, unmutated 50-seed sweep —
seeds 7, 18, 39 (not a mutation test; this was live, unmodified code).

Before this fix, `AppendEntriesReply` didn't echo back what it was replying
to, so the leader recorded, per peer, the `(prev_log_index, entry_count)` it
had most recently *sent* — overwritten on every send — and interpreted
whatever reply came back against that record. The reasoning (still visible
in git history) was that a second send before the first's reply arrives only
ever carries a superset of what the first one sent, so misattributing a
reply to the latest record would still land on the correct final
`match_index`. That's false: if the reply to the *first, smaller* send is
delayed and arrives *after* a second, larger send has already overwritten
the record, it gets interpreted against the larger one — inflating
`match_index` past what the peer had actually confirmed. In seed 7, this let
a leader advance `commit_index` to an entry held by only two of five nodes
(itself and one honest peer), not the three needed for an actual majority.

`check_leader_completeness` caught the resulting gap directly: a later,
honestly elected leader was missing an entry an earlier leader believed was
already committed.

Fix: `AppendEntriesReply` now carries the follower's own `match_index` —
computed by the follower from the specific request it's answering, so there
is no leader-side guess left to misattribute. The leader's only job became
never letting a reply move `match_index` backward (`if msg.match_index >
self.match_index[src]`, not an overwrite) — a delayed reply reporting a
smaller confirmed index than one already recorded is simply stale, not
corrupt. This also let `_outstanding` (the record that enabled the bug) be
deleted outright.

## check_leader_completeness gated on an entry's own term, not the term that made it safe

Found by: `test_fuzz_sweep.py::test_fuzz_sweep_large[711]` (the 1000-seed
sweep), on the fixed, unmutated code, immediately after the fix above.

A false positive in the *checker*, not the algorithm: Figure 8's
current-term-only restriction lets a leader commit a whole prefix in one
step — committing its own current-term entry at index N also commits every
earlier-term entry below N, riding along via the Log Matching Property. The
original implementation recorded each such index's committing term as *the
entry's own term*, then required every leader with a later term to have it.
But an earlier-term entry riding along is only actually guaranteed from the
*triggering* (current) term onward, not from its own, older term — a leader
elected in between (later than the entry's own term, but before the
triggering commit ever happened) was never part of the majority that made
it safe, and legitimately might not have it yet. Gating on the entry's own
term flagged exactly that honest leader as having "lost" something.

Fix: `CommittedEntry` now separately records `term` (the entry's own, for
matching *which* value must be present) and `established_at_term` (the
committing leader's current term at the moment this index was first covered,
for deciding *which future leaders are bound at all*).

## check_no_spurious_truncation compared log entries by identity, missing real entry loss

Found by: diagnosing why Mutation 1 (`test_mutation_detection.py`'s blind
truncate-and-reappend on every `AppendEntries`, never committed to
`src/raft/node.py`) was reported caught in only 2/50 seeds, and only via
`check_leader_completeness` — never via `check_no_spurious_truncation`,
the checker that mutation was specifically added for. Shrinking those two
seeds (3 and 35) and instrumenting every truncating `AppendEntries`
delivery pinned the mechanism down exactly (see the diagnosis this fix is
based on): a *stale, shorter* `AppendEntries` — built from an earlier,
smaller `next_index`, before the leader's own log grew further — arrives,
via the `Network`'s own reordering, *after* a longer one already extended
the follower past it. Mutation 1's blind truncate-from-`prev_log_index+1`
then discards everything the follower held beyond that stale message's
range, including entries already covered by a leader's commit decision —
in seed 3, an entry a majority had already made safe under the
current-term rule, gone from the one follower that mattered by the time
it was later, honestly elected leader.

Root cause, in the checker, not the mutation: `check_no_spurious_
truncation` compared entries pairwise over
`range(min(len(current_log), len(previous_log)))` and raised only on
"different object, same value" — an object-identity check specifically
aimed at a follower tearing an already-matching entry down and rebuilding
it. Two structural properties of this codebase, independent of each
other, kept that signature from ever appearing here: (a) `next_index`
genuinely never walks backward past a point of real agreement, so a
*rejection*-driven retry can never resend something already
known-matching; and (b) `LogEntry` is never cloned anywhere, so a resend
built from a given leader's own storage is always identity-equal to
whatever that leader sent before, regardless of `next_index`, delivery
order, or how many times it's resent. (a) is a true claim about which
index ranges a leader will ever construct into a message; it says nothing
about (b), and it was (b) alone that made the identity signature
unreachable — a fact the original reasoning stated as one combined
argument rather than two independent ones. Meanwhile the `min(...)`-bounded
loop meant that even value-level loss was invisible by construction: once
`current_log` is shorter than `previous_log`, the discarded indices are
never in range to compare at all, identity or otherwise.

Fix: `check_no_spurious_truncation` now compares by *value* (`LogEntry`
equality: `term` and `command`), not identity, and its scan is anchored
to `previous_log`'s own length rather than the shorter of the two, so an
index that disappears entirely is itself a difference, not something the
loop bound quietly excuses. It finds the first index where the two logs
diverge — including an index `previous_log` had that `current_log` no
longer reaches — and passes only if `current_log` still has an entry
there whose `term` differs (Figure 2 rule 3's conflict, immediately
followed by rule 4's append: the one truncation Figure 2 actually
permits). Anything else — gone entirely, unchanged in term but different
in value, or unchanged in both but still removed and never restored —
raises. Detection on Mutation 1 went from 2/50 (all via
`check_leader_completeness`, only once a much later election happened to
expose the gap) to 43/50, all via `check_no_spurious_truncation` itself,
catching seeds 3 and 35 at the moment of loss — 140 and 37 steps earlier,
respectively, than the leader-completeness violations that used to be the
only signal. (Since superseded again by `STALE_REDELIVER` below: 45/50.)

## STALE_REDELIVER: deliberately constructing Mutation 1's precondition instead of waiting on it

The value-based `check_no_spurious_truncation` fix above still left 7/50
seeds undetected for Mutation 1 — not a checker gap, but a *fuzzer*-
coverage one: the harmful precondition (a stale, shorter `AppendEntries`
delivered after a longer one already extended a follower past it) only
ever arose from `Network`'s own random reordering, and 300 steps wasn't
always enough for a given seed to produce it by chance.

`Fuzzer` gained a new action, `STALE_REDELIVER`, that constructs this
precondition on purpose: it scans `Network`'s own bounded, per-link
delivery history for an `AppendEntries` a *later* delivery to the same
link has since covered a larger final index than, and redelivers it.
Verified by hand first, independent of the fuzzer, that this is a
harmless no-op on real (unmutated) code — the real handler's "already
matches" logic recognizes it and leaves the log untouched, and the
resulting stale reply is correctly ignored by the leader's own
never-move-`match_index`-backward guard — before ever wiring it in.

Three things needed solving to make replay of this action exact, the
same standard every other action already meets:

- **Identity**: `Network` now assigns each `send()` call a monotonic
  `msg_id`, kept entirely internal to `Network` (a new `_Pending` field)
  — nothing about `messages.py`, `RaftNode`, or any of the ~76 existing
  hand-constructed message call sites across the codebase changed.
- **History**: `Network` retains delivered messages in a bounded,
  per-`(src, dst)` ring buffer (`history_depth`, a constructor parameter,
  not a hard-coded constant) — payload-agnostic, exactly like the rest of
  `Network`; the `AppendEntries`-specific "is this stale" interpretation
  lives entirely in `Fuzzer`, not `Network`. The depth was *measured*, not
  guessed: an initial estimate of 16 was checked against an unbounded
  ground truth over both sweeps and found to lose 9–6% of genuinely
  eligible candidates to eviction before a draw could use them (11 out of
  19,623 draws over 1000 seeds found *zero* candidates despite some truly
  existing). 32 recovers 99.9–99.97% of them, with zero fully-evicted
  draws in either sweep, and 64+ showed no further measurable gain — see
  `raft/sim/network.py`'s `DEFAULT_HISTORY_DEPTH`.
- **Replay**: `TraceEntry` gained one field, `msg_id`, following the
  exact resolved-value pattern every other action already uses — recorded
  on a live draw, applied directly via a new, deterministic
  `Network.recall(msg_id)` on replay, no RNG of any kind involved. A
  forced replay whose `msg_id` `recall()` can't find (a shrink candidate
  that dropped the delivery which created it) raises rather than
  silently degrading to a skip — replaying a *different* event than what
  was recorded would break `replay()`'s "identical" contract, and
  `shrink()`'s own `_signature()` already has a blanket `except Exception`
  that treats this exactly like any other shrink-induced structural
  inconsistency.

With `STALE_REDELIVER` weighted at 7.0 (the same class as `CRASH`/
`RESTART` — a targeted, occasional fault, not a constant presence)
wired into `DEFAULT_WEIGHTS`, Mutation 1 detection rose from 43/50 to
**45/50**. The *specific set* of caught seeds shifted, not just grew:
weaving a new action into the shared, weighted `self._rng` stream
perturbs every seed's entire subsequent random walk, not only the ones
that need the new action, so some seeds that used to trigger this via
lucky native reordering no longer do, while more that never did now do
via deliberate redelivery. Seeds 3 and 35 — the two this diagnosis was
built around — are still both caught.

## Invariant coverage gaps, measured

A mutation test against the 50-seed sweep, re-run after `check_leader_
completeness` and `check_state_machine_safety` (Milestone 4's
commit-index-dependent checks) were wired in, and again after
`check_no_spurious_truncation`'s value-based rewrite and `STALE_REDELIVER`
(see the two entries above):

- Blind truncate-and-reappend on a duplicate AppendEntries: **45/50
  seeds, all via `check_no_spurious_truncation`** (0 via
  `check_leader_completeness`, which runs later in the fixed check order
  and never gets the chance — a `SafetyViolation` stops the run at the
  first property that catches it). History: 2/50 (identity-based checker,
  `check_leader_completeness` only) → 43/50 (value-based checker,
  native reordering only) → 45/50 (with `STALE_REDELIVER` deliberately
  constructing the precondition too).
- Advancing match_index/next_index on a rejected reply, trusting it
  regardless of the reply's success flag: **closed**. 4/50 seeds, always via
  `check_leader_completeness` — some later, honestly elected leader ends up
  missing an entry an earlier leader believed it had committed, purely
  because of the corrupted match_index. This is exactly the class of real
  bug described above, and exactly the gap Leader Completeness was added to
  close.
- Skipping the AppendEntries consistency check entirely: 31/50 seeds (was
  33/50 before `STALE_REDELIVER` joined `DEFAULT_WEIGHTS` — see above for
  why a new action shifts counts for mutations it has nothing to do with,
  by perturbing the shared RNG stream, not a regression) — 30 via log
  matching, 1 via leader completeness (a `SafetyViolation` stops the run
  at the first property that catches it, so which one fires first for a
  given seed depends on exactly when each condition becomes checkable).
- next_index initialized to 1 rather than last_log_index + 1: still 0/50,
  and still not obviously a safety bug — prev_log_index 0 always passes the
  sentinel check, so peers get a redundant resend that correct follower
  logic treats as a no-op.

A clean sweep means the implemented checks found nothing, not that the
implementation is correct.

## check_state_machine_safety is a guaranteed-pass no-op until Milestone 5

`last_applied` is initialized to 0 in RaftNode.__init__ and never advanced
anywhere in the codebase — nothing reads commit_index and moves last_applied
toward it, and no state machine exists to apply an entry's command to.

check_state_machine_safety's guard is `if node.last_applied <= checked_through:
continue`. With last_applied permanently 0, this is true for every node on
every call, so the function returns before reaching its comparison-and-raise
logic at all. This differs from a checker like check_log_matching, which
executes its real comparison every step and simply hasn't found a mismatch:
check_state_machine_safety never gets that far. It is correctly implemented
and wired into every fuzzer step, but until something advances last_applied
it verifies nothing. A clean sweep does not count as evidence for this
property specifically.

Scope: applying committed entries to a state machine is Raft's separate
"apply" step (Figure 2 treats commit and apply as distinct), and belongs to
the KV store milestone, not commitment. This was documented in RaftNode's
own module docstring before this session and is not a new gap — it is
flagged here so the mutation table and sweep results are read correctly:
of the five implemented invariants, four are live and one is dormant.