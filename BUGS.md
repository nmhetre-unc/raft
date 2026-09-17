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
(see the two entries above), and again after Milestone 5's `CLIENT_REQUEST`
stopped omnisciently targeting the actual leader and started following
`raft.node.NotLeader`'s own `leader_hint` instead (see the "leader-hint
redirect" entry below) — that traffic-pattern change shifts sweep counts
for mutations it has nothing to do with, the same way `STALE_REDELIVER`
joining `DEFAULT_WEIGHTS` did before it, without touching either the
mutation or the checker:

- Blind truncate-and-reappend on a duplicate AppendEntries: **41/50
  seeds, all via `check_no_spurious_truncation`** (0 via
  `check_leader_completeness`, which runs later in the fixed check order
  and never gets the chance — a `SafetyViolation` stops the run at the
  first property that catches it). History: 2/50 (identity-based checker,
  `check_leader_completeness` only) → 43/50 (value-based checker,
  native reordering only) → 45/50 (with `STALE_REDELIVER` deliberately
  constructing the precondition too) → 41/50 (leader-hint-following
  `CLIENT_REQUEST`; see `tests/test_mutation_detection.py` for which of
  the two seeds this diagnosis was built around, 3 and 35, is still in
  the caught set and which no longer is).
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
- Applying past commit_index — ignoring Figure 2's "apply" boundary
  entirely and replaying straight to the end of the log, committed or
  not (Milestone 5's replicated state machine and `last_applied` wiring,
  see the entry below): 8/50 seeds (was 11/50 before the leader-hint
  traffic-pattern shift described above), always via
  `check_state_machine_safety` — this checker's first real detection
  in this project.
- Skipping `apply_client_request`'s own serial_number dedup check
  entirely — always re-applying a retried/redelivered `ClientRequest`
  instead of recognizing it as already applied (Milestone 5's session
  dedup, see "CLIENT_REQUEST sent opaque strings..." below): **0/50 via
  any of the six `check_*` invariants — a structural gap, not a coverage
  shortfall**. None of them ever inspect `KVStateMachine._data`/
  `_sessions`; this mutation touches nothing else. The actually
  appropriate detector — an `apply()`-call-counting spy, independent of
  the mutation's own broken dedup decision, reused from
  `tests/test_client_sessions.py`'s own positive-behavior test —
  catches it on **38/50 seeds**. See
  `tests/test_mutation_detection.py`'s two Mutation 6 tests: the sweep
  numbers above, and a hand-built, fuzzer-independent proof that this is
  a genuine final-value correctness hazard (a duplicate redelivery of an
  already-superseded request clobbers a newer value with an older one)
  that no existing checker could ever have a chance to see.

A clean sweep means the implemented checks found nothing, not that the
implementation is correct.

## check_state_machine_safety was permanently un-triggerable before Milestone 5; now live

Prior to Milestone 5, `last_applied` was initialized to 0 in
`RaftNode.__init__` and never advanced anywhere in the codebase — nothing
read `commit_index` and moved `last_applied` toward it, and no state
machine existed to apply an entry's command to.

`check_state_machine_safety`'s guard is `if node.last_applied <=
checked_through: continue`. With `last_applied` permanently 0, this was
true for every node on every call, so the function returned before ever
reaching its comparison-and-raise logic. This differed from a checker
like `check_log_matching`, which executes its real comparison every step
and simply hadn't found a mismatch: `check_state_machine_safety` never
got that far. It was correctly implemented and wired into every fuzzer
step, but with nothing ever advancing `last_applied`, it was not merely
quiet — it was structurally incapable of ever firing, on any run, no
matter how long. The hand-constructed unit tests in
`test_invariants.py` (`test_check_state_machine_safety_raises_when_two_
nodes_apply_different_entries`, etc.) proved the checker's own logic was
sound, but proved nothing about whether real `RaftNode` behavior could
ever reach it — and, until this milestone, nothing could.

Milestone 5 (`src/raft/statemachine.py`'s `KVStateMachine`, and
`RaftNode._apply_committed_entries`, called at the end of `tick`,
`handle`, and `append_command` — see `raft.node`'s module docstring)
gives `last_applied` something to actually advance toward. First real
results:

- **Unmutated code, 50-seed sweep**: `last_applied` advances on at least
  one live node in 34/50 seeds, reaching as high as 22 across 199 total
  live-node observations, and never once exceeds `commit_index` —
  `check_state_machine_safety` is now genuinely exercised on ordinary
  fuzzing, not just vacuously satisfied (see
  `tests/test_apply.py::test_last_applied_is_bounded_and_nonzero_across_the_fixed_seed_sweep`).
- **Mutation 5** (apply past `commit_index` instead of stopping there —
  see `tests/test_mutation_detection.py`): **11/50 seeds, always via
  `check_state_machine_safety`** — the first time this checker has ever
  caught anything through real `RaftNode` execution rather than a
  hand-set field. Mechanism: an entry applied before it was actually
  committed can still be truncated and overwritten by a later, honestly
  elected leader that never saw it as committed (exactly the hazard
  Figure 8 exists to rule out at the *commit* level; this is its
  *apply*-level analogue). When that happens, whatever next applies that
  same index sees genuinely different log content than an earlier,
  premature application already recorded there.

Scope, unchanged from before: applying committed entries to a state
machine is Raft's separate "apply" step (Figure 2 treats commit and
apply as distinct). This was documented in `RaftNode`'s own module
docstring before this milestone and is not a new gap — it is corrected
here now that the gap has actually been closed: of the five implemented
invariants, all five are live.

## CLIENT_REQUEST sent opaque strings, so apply() was never really exercised by fuzzing; client sessions added

The apply-wiring milestone above made `last_applied` advance correctly,
but the entries it advanced past were never real: `Fuzzer.
_do_client_request` built `command = f"cmd-{step_index}"` — a plain
string `RaftNode._apply_committed_entries` correctly treats as inert
(see that milestone's design choice for why), meaning every committed
entry the fuzzer ever produced was a no-op from `KVStateMachine`'s point
of view. `check_state_machine_safety`'s 50-seed sweep numbers above are
real (last_applied genuinely advances, never exceeds commit_index), but
`KVStateMachine.apply()` itself had never once actually run through
ordinary fuzzer-driven traffic — only through hand-built tests.

Fixed: `CLIENT_REQUEST` now sends a real `raft.statemachine.ClientRequest`
(a `Put` or `Delete`, keyed off a small, reused set of keys so the two
operations actually interact) instead of a placeholder string. Confirmed
directly, not by absence of exceptions: across a 6-seed sample, final KV
state differs by seed (`{'key-1': 61}`, `{'key-1': 16}`, `{'key-1': 46}`,
...), and across the full 50-seed sweep, 17 seeds end with non-empty KV
state and produce **17 distinct final states** — `apply()` is now
genuinely doing varied, seed-dependent work, not returning the same
answer regardless of input.

Alongside this, per-client session tracking was added — `client_id`,
`serial_number`, wrapped as `ClientRequest`, deduplicated by
`KVStateMachine.apply_client_request`. This lives on `KVStateMachine`
itself (a `_sessions` dict, right beside the KV data), not anywhere
leader-specific — deliberately, since a naive design that instead tracks
"have I seen this serial number" only in whichever node happens to be
leader breaks the moment that leader crashes and a different node takes
over: the new leader's own memory starts empty, has no way to know the
old leader already applied it, and re-applies a retried request. Proven
directly (`tests/test_client_sessions.py`
`test_naive_leader_only_dedup_applies_the_retried_request_twice` vs.
`test_correct_design_applies_the_retried_request_exactly_once`, both
constructing the identical leader-crash-then-retry scenario by hand):
the naive design double-applies (2 real `apply()` calls, caught by a
call counter, since the retried `Put` reuses the same value and a
final-state comparison alone would have missed it entirely), the
state-machine-side design applies exactly once. Confirmed clean of this
failure mode across the full 50-seed sweep too
(`test_no_double_application_across_the_fixed_seed_sweep`), including
retries that deliberately coincide with crashes and elections (see
`raft.sim.fuzz`'s own module docstring for how `CLIENT_REQUEST` now
decides to retry vs. start fresh).

**Known limitation, decided explicitly, not defaulted into**: `_sessions`
grows by one entry per distinct `client_id` ever seen and is never
pruned. The safe way to bound this is folding old sessions into a
snapshot and discarding the log (and the sessions map) before it —
Raft's own compaction mechanism — but that milestone doesn't exist in
this project and, per its own stated priorities, may never be built at
all. A bound implemented without compaction would not actually be safe:
evicting a session a client might still legitimately retry against
would silently reopen the exact double-apply hazard this mechanism
exists to prevent, and making eviction safe would need real
client-lifecycle machinery (explicit open/close, or a lease tied to some
liveness signal) this project has no model of anywhere. Decided to
document this as a permanent, accepted characteristic of an in-memory,
non-snapshotting implementation rather than build a bound that would
either be unsafe or be substantially more scope than "minimal" — see
`raft/statemachine.py`'s own module docstring for the same reasoning in
place.

## Milestone 5 complete: state machine, client sessions, and leader-hint redirect, verified under the full fault model

Milestone 5 shipped across four prompts: the replicated `KVStateMachine`
and `last_applied` wiring (above), per-client session dedup (above), a
`NOT_LEADER`/`leader_hint` redirect so a client can find the leader
without knowing it in advance, and this final pass — running the whole
client-facing loop through the complete fault-injection harness at once,
not each piece in isolation.

**Leader-hint redirect** (`raft.node.NotLeader`, `RaftNode.leader_hint`):
a node that isn't leader answers a client's `append_command` with
`NotLeader(leader_hint=...)` instead of silently doing nothing —
`leader_hint` is that node's own best current guess (the last node it saw
a `RequestVote`/`AppendEntries` from at a term at least as new as its
own), refreshed on every incoming message, read by nothing else in this
file. `Fuzzer`'s `CLIENT_REQUEST` action follows it the way a real client
has to (`_choose_client_request_target`), never omnisciently picking the
actual leader — proven, not assumed, by two positive controls run
directly against `_maybe_update_leader_hint`: a follower that hints
itself is caught directly by
`tests/test_replication.py::test_append_command_on_a_follower_hints_the_
leader_it_last_saw`; a follower that forgets to check a message's term
before trusting it — hinting whoever it saw N terms ago, stale or not —
was, on first attempt, caught only *incidentally* by an unrelated
mutation-detection sweep count shifting by one, not by any test actually
asserting the property. That gap was closed by adding
`test_append_command_hint_ignores_a_stale_message_from_an_earlier_term`,
a direct assertion that a stale, lower-term message never overwrites a
fresher hint. `leader_hint` itself is a plain node-id (`str` on
`RaftNode`, `int` once `RaftClusterNode` translates it) used only as a
dict-lookup key, never a message or node object reference — checked
directly, not assumed safe by analogy to the id()-reuse hazard the
client-sessions test above measured: there is no channel here for a
reused Python object address to leak into a decision.

**Reply-delivery scope, decided explicitly**: a client whose request
commits but whose reply is dropped by fault injection (or whose leader
crashes before it can reply at all) has no way, in this project, to
*discover* the outcome beyond blindly retrying with the identical
`(client_id, serial_number, command)`. That retry is always safe — session
dedup guarantees at most one real application regardless — just
uninformative. Building a way to answer "what happened to serial N?"
would mean `append_command` consulting the leader's own session state
before appending and a new return shape distinct from both `NotLeader`
and "appended, in flight" — genuine new client-protocol surface, and one
that still wouldn't fully solve the problem it targets (an *in-flight*,
not-yet-committed request still has no result to report). Decided to
document this as a permanent, accepted limitation instead, the same way
`_sessions`'s unbounded growth and `read()`'s non-linearizability are
documented above — not a gap to close later, a property of this
project's scope. `Get` needs no special handling as a consequence: it was
already never put through the log (see `raft/statemachine.py`'s own
module docstring), and this decision doesn't change that.

**Verified under the full combined fault model** — `CLIENT_REQUEST`
(including retries that follow `leader_hint`), `STALE_REDELIVER`,
`CRASH`, `RESTART`, `PARTITION`/`HEAL`, and `ADVANCE_CLOCK` all active at
once, exactly `raft.sim.fuzz.DEFAULT_WEIGHTS`, both at 50 seeds (always)
and 1000 seeds (`-m slow`):

- Zero `SafetyViolation`s of any kind, at either sweep size.
- `check_state_machine_safety` specifically, not bundled into that
  generic result: measured non-vacuous over the 50-seed sweep — 40/50
  seeds genuinely apply at least one entry somewhere, 292 distinct
  applied-index observations total (6050 over the 1000-seed sweep) — so
  a clean run means this checker was actually exercised, the same
  distinction drawn above for why a clean sweep meant nothing before
  Milestone 4's apply-wiring landed. See
  `tests/test_client_request_integration.py::test_check_state_machine_
  safety_stays_clean_under_the_full_fault_model`.
- A deterministic, hand-built proof
  (`test_client_request_survives_redirect_retry_and_a_leader_crash`) that
  the whole loop holds together end to end: a client hits a follower,
  gets redirected via `leader_hint`, retries against the real leader,
  that leader commits and crashes before any follower learns of it, the
  client retries again — blind to whether its first attempt ever
  committed — against a newly-elected leader, and the command ends up
  applied exactly once (an `apply()` call count, not a final-value
  comparison, which a duplicate `Put` could never expose).
- Final KV state cross-checked, for a handful of fuzzed seeds, against an
  independently recomputed expectation derived straight from each node's
  own committed log (a from-scratch dedup reimplementation, never a call
  into `KVStateMachine` itself) —
  `test_final_kv_state_matches_an_independently_computed_expectation`.
  Stronger than Milestone 4's own bar ("differs by seed"): this confirms
  *correct*, not just *varied*.
- Mutation 6, this milestone's own target bug (skip session dedup
  entirely) — see the coverage-gaps table above for the full report:
  0/50 via the six existing `check_*` invariants (a structural gap, not a
  shortfall), 38/50 via the appropriate detector.

Of the five implemented invariants plus the client-session dedup
mechanism, all are now exercised — not just wired in — by the complete
fault model at once. Milestone 5 is complete; snapshotting and cluster
membership changes remain out of scope, tracked as Milestone 6 (or
later), not started here.