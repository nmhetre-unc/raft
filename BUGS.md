# Bugs

Every bug found, in the implementation and in the harness itself, in the
order they were found.

The entries up to "Milestone 5 complete" are from the simulator: safety
violations found by the fuzzer, each with its seed, the minimal reproducing
trace, the property that broke and the root cause. Reproduce any of them
with `pytest tests/test_fuzz_sweep.py --seed <N>`.

The entries from "A restarted node could never catch up" on were found by
running the same `RaftNode` as real processes (`raft.net`: gRPC, durable
`FileStorage`) under load and faults -- bugs the simulator could not see.
Each records how it was found, the symptom, the root cause, why the
simulator missed it, the fix, and before/after numbers with the result
files they came from (`bench/results/`).

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
  regardless of the reply's success flag: **closed**. 2/50 seeds (was
  4/50 before the leader-hint traffic-pattern shift described above —
  same cause, same "shifts counts for mutations it has nothing to do
  with" effect, not a regression), always via `check_leader_completeness`
  — some later, honestly elected leader ends up missing an entry an
  earlier leader believed it had committed, purely because of the
  corrupted match_index. This is exactly the class of real bug described
  above, and exactly the gap Leader Completeness was added to close.
- Skipping the AppendEntries consistency check entirely: 25/50 seeds now
  — history: 33/50 (native) → 31/50 (once `STALE_REDELIVER` joined
  `DEFAULT_WEIGHTS`; 30 via log matching, 1 via leader completeness) →
  25/50, all via log matching, 0 via leader completeness (once the
  leader-hint traffic-pattern shift described above moved the one seed
  that used to surface via leader completeness onto the log-matching
  path instead — its own traffic changed enough that log matching now
  catches it first). A `SafetyViolation` stops the run at the first
  property that catches it, so which checker fires first for a given
  seed depends on exactly when each condition becomes checkable, not
  which bug is "more real."
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
## A restarted node could never catch up: one-entry log backtracking

Found by: `bench/cluster_bench.py` leader-kill runs (step 6 baseline), not the
fuzzer. Regression test: `tests/test_catchup.py`.

**Symptom.** In 2 of 3 leader-kill benchmark runs (16 clients, kill the leader
at 10 s, restart it at 20 s), the restarted node still had commit index 0
sixty seconds after the load stopped, while the other four agreed. Re-measured
on the pre-fix code with catch-up timing added: 0 of 3 runs caught up. Both
safety checks passed every time; this is liveness, not safety.

**Root cause.** On a rejected AppendEntries, the leader backed `next_index`
up by exactly one entry and retried (the code said "no conflict-term
optimization yet"). Three things compounded it:

- the restarted node came back about 4,400 entries behind;
- every retry carried the whole log tail from `next_index` on, so each of
  those thousands of round trips encoded and decoded thousands of entries;
- every leader elected while the node was down (and every election in the
  restart storm) initializes `next_index` optimistically to its own log end,
  restarting the walk from the top.

**Why the simulator missed it.** Two reasons, both structural:

1. Fuzzed logs are tiny: across the 1000-seed sweep the longest log at the
   end of a run has 37 entries (median 10), so even one-entry backtracking
   needs at most a few dozen round trips.
2. The simulated network delivers with zero delay, so a round trip costs no
   simulated time. Thousands of them would still look instantaneous.

The fuzzer now has a liveness phase (`Fuzzer.check_liveness`, run by both
sweeps): after the last fault, heal, restart everything, give every link
5 ms of delay, and require that a leader commits its whole log and every
node reports that commit index within 3,000 simulated ms. The bound was
fixed before measuring. The pre-fix code **passes** it on all 1000 seeds
(worst case 1,605 ms) -- the logs are simply too short to expose this. It
stays as a guard against liveness regressions the fuzzer *can* reach. The
deterministic test is what reproduces this bug: crash a follower, commit
5,000 entries, crash the leader so a new one is elected while the follower
is down, restart the follower with 5 ms links, and require catch-up within
1,000 simulated ms and 200 messages. Pre-fix it fails: after 500 ms and 201
messages the follower holds 0 of 5,002 entries. The leader change is
essential: a leader that was leading when the follower crashed keeps
`next_index` at the crash point and needs no backtracking at all.

**Fix (1 of 2: fast backtracking).** `AppendEntriesReply` carries
`conflict_index` and `conflict_term` on failure (the Raft paper's section
5.3 optimization). The follower reports either "too short: my last index + 1"
or "I have term T there, starting at index i". The leader probes just past
its own last entry of T if it has one, else at i. One round trip per
conflicting term instead of per entry. It stays safe because `next_index`
only picks which `prev_log_index` to probe; the follower's consistency check
still gates every append. It always makes progress because the leader's term
at the rejected index must be newer than T, so all its term-T entries lie
before it. A delayed failure reply can no longer push `next_index` below
`match_index + 1`. After the fix the restarted node caught up in all 3 runs,
in 4.9-6.2 s. The remaining seconds are the restart election storm, a
separate bug (next entry).

(Correction: this entry first reported 4.3-6.1 s, finishing 0.1-0.16 s after
the storm. Those numbers came from a bench whose own Status poller used gRPC's
default reconnect backoff, so it couldn't see the restarted node for seconds
after it came up. The poller now uses short backoff, and the figures above are
re-measured with it on the same Fix-1 code.)

## Restarting a node set off an election storm against a healthy leader

Found by: `bench/cluster_bench.py` leader-kill runs, column (c), once Fix 1
let the restarted node catch up at all. Regression tests:
`tests/test_prevote.py`, `tests/test_net.py::test_server_keeps_reconnect_backoff_under_the_election_timeout`.

**Symptom.** After the killed leader was restarted, the cluster went through
20-26 elections in the next 5-6 seconds, and the restarted node took that long
to catch up. Every one of those elections made the healthy leader step down
and win again. Writes kept flowing (stalls of 56-166 ms), but the restarted
node didn't catch up until the storm ended.

**Root cause.** Two things together:

1. *Transport.* gRPC's default reconnect backoff starts at 1 s and grows.
   While the old leader was dead for 10 s, every other node's channel to it
   backed off further and further, so after it restarted the current leader
   could not reach it for seconds.
2. *Protocol.* A node that hears no AppendEntries for one election timeout
   increments its term and campaigns. The restarted node did exactly that,
   every 150-300 ms. It could never win (its log was behind), but its
   RequestVote carried a higher term, and every node that received one,
   leader included, adopted that term and stepped down.

**Why the simulator missed it.** The simulator has no transport: a message to
a node that has just restarted is delivered as soon as it is sent. Nothing
models a sender that can't reach a peer that is up. The protocol half (a node
cut off from the leader disrupts it on return) *is* reproducible in the
simulator by partitioning one node, and now is:
`test_partitioned_node_rejoining_does_not_disrupt_a_stable_leader` fails its
control case (PreVote off) exactly this way. The fuzzer never flagged it
because it is not a safety violation, and the cluster does converge
eventually.

**Fix (a): transport.** `GrpcTransport` sets reconnect backoff to 20 ms
initial, 50 ms max, 20 ms minimum connect timeout. The server caps them at a
third of the minimum election timeout. The leader now reaches a restarted
peer within a heartbeat or two. On its own this ended the storm: 0 elections
after restart in 3 of 3 runs, catch-up 229-276 ms, mostly the server process
starting.

**Fix (b): PreVote** (Ongaro's thesis, section 9.6). On by default in the
server, off by default in `RaftNode`, so the simulator's pinned mutation counts
keep measuring the same protocol. Both fuzz sweeps run both modes. On an
election timeout a node first sends `PreVote` for its term + 1 without
changing any state. A voter grants only if that term is newer than its own,
it is not the leader, it has not accepted an AppendEntries from a current
leader within the minimum election timeout, and the candidate's log is at
least as up-to-date as its own. Only a majority of grants starts a real
election. This covers the cases backoff can't, such as a genuine partition:
in the simulator, a node cut off for 3 s rejoins with its term unchanged and
the leader keeps its term. In the leader-kill bench, where backoff had
already removed the storm, PreVote changed nothing measurable (0 elections,
catch-up 246-317 ms). It adds one round trip before each real election, which
is within noise in kill-to-leader time.

## A follower far enough behind could never catch up: one AppendEntries carried the whole gap

Found by: `bench/large_gap_repro.py`, written to test whether this could
happen, after reading the code rather than from a failing run. Regression
tests: `tests/test_chunking.py`.

**Symptom.** Kill a follower, commit 1,500 puts of 4 KiB through the leader,
restart the follower. It never caught up: it stayed at commit index 0 for the
full 30 s timeout, while every other node was at 1,501. With 500 such puts
(2.06 MB to send) it caught up in 0.2 s. While the follower was down, the
1,500 writes also took 24.7 s, against 2.6 s for 500.

**Root cause.** The leader built every AppendEntries with *everything* from
the peer's `next_index` to the end of its log. For a follower 1,500 entries
behind, that is one 6,186,343-byte `Envelope`. Neither our server nor
`GrpcTransport` sets a message size limit, so gRPC's defaults apply: no limit
on sending, 4 MiB (4,194,304 bytes) on receiving. The follower's server
rejected the RPC with `RESOURCE_EXHAUSTED: Received message larger than max`,
which was confirmed directly against a server process: 4,121,885 bytes
accepted, 4,534,085 rejected. The transport counts a failed RPC as a lost
packet, so the leader got no reply and sent the same message, never smaller,
on every heartbeat and every write. The slow writes have the same cause: every
client write built and encoded the whole multi-megabyte tail again for the
dead peer.

**Why the simulator missed it.** The simulator has no message size limit and
no encoding: a message carrying 5,000 entries costs the same as one carrying
none. Its logs also stay short (at most a few dozen entries in the fuzzer). The
leader-kill bench missed it too: its values are about 20 bytes, so a 10 s
outage left roughly 2,300 entries, about 100 KB to send.

**Fix.** An AppendEntries now carries at most `max_append_entries` entries
(default 512) and at most `max_append_bytes` (default 1 MiB), by
`entry_size`. It always carries at least one entry when the peer is behind,
so a single oversized entry still goes out. The server measures size with the
real protobuf encoding; `RaftNode` on its own uses an overestimate based on
`repr`. Both caps are configurable in the cluster config. When a success reply
confirms everything last sent to a peer that is still behind, the leader sends
the next chunk immediately instead of waiting for the next heartbeat. When
newer entries are already in flight, as in normal pipelined replication under
load, it sends nothing extra, so no entry goes out twice. In the simulator this
rule never fires unless a chunk was capped, so the fuzz sweeps and pinned
mutation counts are unchanged.

Results:

- **Large-gap reproduction:** the follower now catches up in 0.26-0.37 s
  (3 runs), in chunks of about 254 entries, the 1 MiB cap.
- **Leader-kill bench:** catch-up is 258-401 ms, against 246-317 ms before. A
  gap of about 2,300 small entries now takes about 5 chunk round trips under
  load instead of one message.
- **Still open:** while a peer is down, every write still builds and encodes
  a chunk of up to 1 MiB for it. The 1,500 writes took 13-14 s, against
  24.7 s before. Capping bounds this cost, but it doesn't remove it. *(Resolved by
  per-peer flow control, below: the same 1,500 writes now take 1.3-1.6 s.)*

## Every write re-sent every unacknowledged entry to every peer

Found by: profiling the leader with py-spy during the no-fault bench (16
clients, deferred fsync), plus send counters added to `GrpcTransport`
(`stats()`) and the bench (`resources` in the result JSON). Regression tests:
`tests/test_flow_control.py`.

**Symptom.** Throughput stayed at 230-310 ops/s from 1 to 32 clients and fell
at 32. The leader's event-loop thread was at 99-100% CPU from 8 clients up.
Every client operation is one log entry, and for each one the leader sent 4
AppendEntries (one per peer) carrying 60-200 entries in total, about
1.8-5.7 KB encoded. The minimum is 4 entries.

**Root cause.** `append_command` sent each peer everything from its
`next_index` on. `next_index` only moves when a reply arrives, so with N
writes arriving within one round trip, each write re-sent all the earlier
ones still unacknowledged. The number of entries sent per write grew with
load, and so did the cost of encoding them. That cost came twice: once to
send and once in Fix 2's `_wire_entry_size`, which encodes each entry to
measure it. On the leader's main thread, encoding took 31% of busy time and
per-RPC gRPC overhead 48%. `load_log()` copies took 1.4%.

**Why the simulator missed it.** It isn't a safety or liveness problem:
duplicate entries are idempotent on the follower, and the simulator charges
nothing for encoding or sending. The fuzzer's zero-delay network also never
has more than a write or two in flight.

**Fix: per-peer flow control** in `RaftNode`. At most one AppendEntries
carrying entries is in flight to each peer. While it is unacknowledged and
younger than one heartbeat interval, new entries wait. When the success reply
arrives, everything that waited goes out in one message, up to Fix 2's caps.
Heartbeats still go out on schedule; a peer with something in flight gets an
empty one, so it keeps hearing from its leader and learns the commit index.

*Why it's safe:* it only changes *when* an AppendEntries is sent, never what
it contains. Every message is still built from `next_index` and
consistency-checked by the follower, and commitment still waits for
`match_index`, which only replies move. Holding a message back looks to the
protocol exactly like a slow network.

*Lost messages and replies:* "in flight" expires one heartbeat interval after
the send. The next heartbeat then re-sends a full chunk from `next_index`, so
a loss costs at most about two heartbeat intervals. A dead peer gets at most
one chunk per heartbeat instead of one per write, which also fixes Fix 2's
open item.

Results (20 s no-fault runs, unprofiled):

- **Load:** at 8-32 clients, entries sent per write are 4.0, the minimum.
  AppendEntries per write fell from 4.0 to 0.95-1.2, because writes are now
  batched.
- **Throughput:** at 16 clients, 891 ops/s (deferred fsync) and 871
  (per-write), against 295 and 274. p99 fell from 89-103 ms to 29-31 ms. A
  single client is unchanged (about 250-290 ops/s), since nothing was
  duplicated there.
- **Restart:** leader-kill restart catch-up is 519-637 ms, against 258-401 ms.
  The gap is now 3-6× larger because of the higher write rate, and it closes
  at 30-38k entries/s against 5-10k.
- **Large-gap reproduction:** still catches up (0.26-0.32 s).

The leader's main thread is still at 100%. What dominates it now is per-RPC
gRPC overhead (about 66%, including asyncio scheduling) and `load_log()`
copies (7.8%, growing with log length).

Five mutation-detection tests had their pinned counts shift (all mutations still detected), because
fewer resends change which seeds reach each precondition; each test records
the old and new count. `test_every_send_happens_after_sync` now expects 2
messages for a burst of 5 instead of 10.

## Throughput fell with uptime: every message copied the whole log

Found by: the 300 s no-fault bench (16 clients, deferred fsync) with the
leader's stats sampled every 20 s (`--stats-every`). FileStorage now counts
`load_log()` calls, entries copied and time spent, and reports them in the
SIGUSR1 stats. Regression tests: `tests/test_log_reads.py`, the new reads in
`tests/test_storage.py`, and the read operations in `tests/test_filestorage.py`'s
model test.

**Symptom.** Throughput fell steadily as the log grew: 842 committed
entries/s in the first 20 s (log 17k), 354 at 52k, and 165 at 94k. The
leader's event loop stayed at 100%. By the end, `load_log()` was 38% of wall
time and copied about 400,000 entries per committed entry. That 38% counts
only the copying itself, not freeing the copy or the callers' work on it. The
short benches understated all this, because each started with an empty log:
the 891 ops/s reported for flow control was the average of a rate already
falling.

**Root cause.** `Storage` had one way to read the log, `load_log()`, which
returns a copy of the whole log. `RaftNode` called it on every per-message
path:

- building each AppendEntries;
- the follower's consistency check;
- every apply;
- every commit-index check;
- every reply that triggered a send.

The driver called it too, to check each resolved client request's term. So
each message cost time proportional to everything ever committed. Two loops
were also linear in a run of one term, which can be most of the log:

- the follower's walk back to the start of the conflicting term's run, in
  `_conflict_reply`;
- the leader's backward scan for its last entry of the conflict term, in
  `_next_index_after_conflict`.

**Why the simulator missed it.** Its logs stay short (at most a few dozen
entries in the fuzzer, 5,000 in the catch-up test), and a copy costs no
simulated time. Correctness never depended on it.

**Fix.** `Storage` gains bounded reads that cost the same whatever the log's
length:

- `entry_at(index)` and `term_at(index)`, which raise IndexError out of
  range, including for negative indexes, which a Python list would silently
  count from the end;
- `entries_from(start, max_count)`, a copy of only the requested slice.

Both `MemoryStorage` and `FileStorage` implement them. Every per-message path
in `RaftNode`, and the driver's waiter check, now uses them. Both conflict
searches are binary searches; that is valid because terms never decrease
along a Raft log, which the old scan already relied on. `load_log()` stays
for callers that need a full copy: tests, the fuzzer's checks, and Status
with `include_log`. Nothing in the node's behaviour changed: every existing
test, including the pinned mutation-detection counts, passed unchanged.

Results (same 300 s bench):

- **Throughput:** flat at 1,000-1,130 entries/s from a 22k-entry log to a
  286k-entry one, with `load_log()` at 0 calls on the hot path. Overall
  1,024 ops/s, against 325; p99 25 ms, against 130 ms.
- **Not explained yet:** a dip to 869-895 entries/s at 200-240 s, which then
  recovered to 1,082 by 280 s and so is not tied to log length. Also one
  229 ms write gap at t=294 s (p99.9 is 47 ms).
- **Leader-kill bench:** 0 elections after restart, catch-up at the same
  per-entry rate as before, verify passed.
- **Large-gap reproduction:** still catches up (0.26-0.31 s).

## Every local benchmark before step 8 ran on tmpfs: fsync was free

Found by: the first Docker cluster runs, whose volumes are on ext4. A
benchmark-methodology bug, not a Raft bug.

**Symptom.** The Docker cluster ran 3.5x slower than the local process
bench. With its data moved to ext4, the local bench matched Docker (286
against 279-314 ops/s).

**Root cause.** `bench/cluster_bench.py` put its data directories in
`tempfile.mkdtemp()`, which here is `/tmp`, and `/tmp` is tmpfs. `os.fsync`
on tmpfs returns at once, so no earlier local number paid for durability.
That covers every throughput and latency figure in the entries above,
including the 3x from per-peer flow control and the ~1,000 ops/s after
bounded reads. The correctness results (verify, linearizability, catch-up)
don't depend on it.

**Why the simulator missed it.** The simulator has no disk; `MemoryStorage`
syncs for free by design. This was in the benchmark harness.

**Fix.** Data directories now default to `~/.cache/raft-bench` (ext4), with
`--tmpfs` (`/dev/shm`) for comparison, and every result records the data
directory and its filesystem type (`environment.storage`). The leader's
stats now count log fsyncs, their time, and the entries they made durable.

**Results.** Same 20 s no-fault runs, ops/s (`bench/results/disk/`):

| clients | ext4 per_write | ext4 deferred | tmpfs per_write | tmpfs deferred |
|--------:|---------------:|--------------:|----------------:|---------------:|
| 1       | 127            | 124           | 237             | 192            |
| 8       | 304            | 309           | 721             | 653            |
| 16      | 329            | 334           | 616             | 751            |
| 32      | 336            | 318           | 646             | 782            |

ext4 runs repeat within ~3%; tmpfs runs, which are CPU-bound, varied by up
to ~30% between repeats on this WSL host. A 300 s ext4 deferred run held
274-321 entries/s throughout (298 ops/s overall, p99 81 ms).

**Deferred fsync did not batch** (found here; fixed in the next entry).
In every run, in both modes, the leader averaged 1.00-1.01 entries per
log fsync, at 1 client or 32, and on ext4 those fsyncs took ~60% of its
wall time. So on a real disk `deferred` was no faster than `per_write`.

## Group commit never happened: the fsync blocked the event loop

Found by: the fsync counters added in the previous entry; then the driver's
per-batch counters and `--trace-batches`.

**Symptom.** The driver is built for group commit -- take every queued
event as one batch, sync once, then send -- but `deferred` mode never
batched: each log fsync on the leader covered 1.00-1.01 client writes at 1
client or 32, and on ext4 throughput was flat at ~330 ops/s whatever the
client count.

**Root cause, measured.** The driver now counts, per batch, its events, its
client submits and its sync time, and can trace every batch
(`--trace-batches`); the bench lines the leader's sync intervals up with
the clients' send times (both `CLOCK_MONOTONIC`). ext4, 16 clients,
deferred, 20 s, sync still on the loop:

- batches held 1.01 submits on average (4,445 batches with 1, 36 with 2,
  none with more);
- yet at the start of each sync, 7.8 operations on average (mostly 5-10)
  had already been sent by clients and not yet taken in by the driver;
- only 0.26 of them, on average, were sent *during* the sync (83% of
  syncs had none): the queue wasn't new arrivals, it was a standing
  backlog in front of the loop;
- between two syncs the loop admitted 0.74 submits.

`storage.sync()` ran on the event loop thread, so for the ~2 ms of each
fsync the loop admitted nothing. Afterwards, the first submit admitted
woke the driver, which processed it alone and blocked the loop in the
next fsync before the gRPC server had admitted another call. About 8
writes waited, and each fsync covered one. The leader's loop was saturated
(~60% of wall in fsync plus ~43% CPU) at ~330 ops/s whatever the client
count.

**Why the simulator missed it.** The simulator has no disk and no event
loop: `MemoryStorage.sync()` costs nothing, and a node handles a step's
messages with no notion of concurrent arrivals. Batching is a property of
the networked driver only.

**Fix: classic group commit.** The fsync runs on a worker thread while
the loop carries on admitting RPCs:

- `FileStorage.sync()` is split. `begin_sync()`, on the loop, snapshots
  everything the sync must write -- the encoded term and vote, the log
  bytes, their offset, any truncation -- and returns a job that touches
  only the files and that snapshot. `sync()` is `begin_sync()` plus
  running the job, so `per_write` and every other caller is unchanged.
- The driver runs the job with `run_in_executor` and awaits it. Its
  messages and replies still leave only after the job completes, so the
  persist-before-send invariant is exactly as before; the existing
  sync-before-send test passes unchanged.
- At most one sync is in flight: the driver's loop awaits it before
  taking the next batch. The node is never touched while a sync runs:
  the driver's loop is the only caller of the node, and it is suspended
  awaiting the job; inputs arriving meanwhile only enqueue. The job reads
  nothing the loop could change anyway: `begin_sync()` copied it out.
- Fail-stop is kept. A failed job fails the driver exactly as a failed
  `sync()` did; events that queued up during it are discarded
  unprocessed, and their submits are never answered. A cancelled driver
  (shutdown) waits for a job in flight to finish before returning, so the
  storage is never closed under a write.

Regression tests (`tests/test_driver.py`): submits accumulating while a sync is held in the worker
thread (not in the log, nothing sent, then one batch of all five); a
sync failure with submits, ticks and messages queued behind it; and
cancellation during a sync, successful or failing.

**Results** (ext4, 20 s, no faults; "before" is the same code with the
one call back on the loop, run interleaved with "after"; ops/s, p50 ms,
entries per leader fsync; `bench/results/groupcommit/`):

| mode, clients | before            | after              |
|---------------|-------------------|--------------------|
| deferred, 1   | 116, 8.3, 1.00    | 95, 9.9, 1.00      |
| deferred, 8   | 260, 28.9, 1.00   | 384, 20.5, 2.01    |
| deferred, 16  | 333, 47.7, 1.01   | 668, 23.5, 3.79    |
| deferred, 32  | 337, 95.2, 1.01   | 979, 31.0, 5.75    |
| per_write, 1  | 141, 6.8, 1.00    | 114, 8.1, 1.00     |
| per_write, 8  | 283, 27.4, 1.00   | 246, 31.0, 1.00    |
| per_write, 16 | 319, 49.8, 1.00   | 335, 47.9, 1.00    |
| per_write, 32 | 336, 95.7, 1.00   | 344, 93.4, 1.00    |

- `deferred` now scales with load: 2.0x at 16 clients and 2.9x at 32,
  with p50 down by two thirds. The leader uses more CPU (123-134% against
  ~70%) because it is no longer idle-blocked in fsync for 60% of its
  wall time.
- `per_write` is unchanged by design: it fsyncs inside each write call.
  Its run-to-run spread at 1 and 8 clients was ~10-15% (a repeat with
  the order swapped gave after 127 / before 119 ops/s at 1 client).
- At 1 client `deferred` is ~10% slower (repeat: 109 against 121 ops/s,
  p50 +1.1 ms): with nothing to batch, each operation pays a thread
  hand-off for the leader's sync and again for the follower's.

## A partitioned follower rejoined seconds late: the leader's TCP connection was in retransmission backoff

Found by: Docker partition runs (`scripts/chaos_docker.sh partition`),
diagnosed with `ss -tin` inside the containers. Regression tests:
`tests/test_net.py` (keepalive options; the server accepting keepalive
pings).

**Symptom.** On the Docker cluster, a follower cut off for 10 s (`docker
network disconnect`, PreVote on) took about 5 s after its network came
back to catch up. A follower restarted after a crash catches up in well
under a second.

**Cause, confirmed with `ss -tin` inside the containers** (one separate
diagnostic run per setting, sampling every container's sockets every
0.25 s; not part of the measured runs below, because the sampling cut
throughput to about a third). A `docker network disconnect` drops packets
silently: no RST, no ICMP. So the leader's gRPC connection to the victim
stayed `ESTAB` the whole time, with HTTP/2 frames piling up behind one
unacknowledged segment. Times are seconds since load start, with the
partition at 10.87 s:

| time | leader → victim socket |
|---|---|
| 10.60 | rto 204, first retransmit |
| 11.04 / 11.45 / 12.26 | rto 408 / 816 / 1632 |
| 13.84 | rto 3264, backoff 4 |
| 17.35 | rto 6528, backoff 5, send_q ~9 KB and growing |
| ~21.0 | network back; nothing happens |
| 24.12 | retransmit timer fires, rto 204, backoff 0: recovered |

- The victim's side of the same connection was also backing off its
  unacknowledged replies (rto 6912, backoff 5), and recovered at the same
  moment.
- TCP's exponential backoff is deterministic from the moment of the
  partition. With a 10 s partition the next retransmit always falls about
  13 s after the partition started, so about 3 s after the network
  returns. That is why the three measured runs without keepalive agree so
  closely.
- On the node's own clock, the first commit progress after rejoin came
  3.1-3.4 s after its interface reappeared. A second stall of about 1.1 s
  followed, then the backlog streamed in about 0.3 s. That second stall
  is not decomposed. A plausible cause is draining the stale queued
  frames with the congestion window reset to one segment after the RTO,
  but the capture does not show it directly.

Raft was not at fault: the leader kept sending heartbeats and chunks into
a socket that would not deliver them until the kernel's next retransmit.

**Why the simulator missed it.** The simulator has no transport: a healed
partition delivers the next message at once. Silent packet loss on a live
TCP connection, and the kernel's retransmission backoff, exist only below
gRPC. The process bench missed it too: it kills processes (the socket
closes, with an RST), it never black-holes a link.

**Fix: gRPC keepalive, behind `ClusterConfig.keepalive_ms`** (cluster
JSON `"keepalive_ms"`, default 1000; 0 turns it off).
- Client channel options: `grpc.keepalive_time_ms` = `keepalive_timeout_ms`
  = keepalive_ms, `grpc.keepalive_permit_without_calls` 1, and
  `grpc.http2.max_pings_without_data` 0.
- The server must accept those pings: `keepalive_permit_without_calls` 1,
  `http2.min_recv_ping_interval_without_data_ms` keepalive_ms/2, and
  `http2.max_ping_strikes` 0. A default server answers a ping without
  data more often than every 5 minutes with GOAWAY "too_many_pings"
  after two strikes.
- gRPC clamps the client interval to at least 1 s, so 1000 is the
  floor.

With keepalive on, the diagnostic run shows the leader declaring the
connection dead about 1.4 s into the partition (a ping unanswered
for the 1 s keepalive timeout). It then retries the connection (`SYN-SENT`, a new attempt about
every 0.5 s) instead of waiting on a backed-off socket. The first
attempt after the network returned succeeded at once (`ESTAB`, rto 208,
backoff 0).

**Results** on Docker Desktop: 5 nodes, PreVote on, 16 clients, deferred
fsync, a 10 s partition of a follower, 3 runs per setting, no sampler
(`bench/results/keepalive/ka-{on,off}-{1,2,3}.json`; the two diagnostic
runs and their `ss` captures are `diag2-*` there).
Catch-up is timed on the victim's own container clock, from its network
interface reappearing to its commit index reaching the highest index any
node had reported:

| keepalive | catch-up (ms) | elections | verify | linearizable |
|---|---|---|---|---|
| off (`docker/cluster-nokeepalive.json`) | 5004 / 5092 / 4926 | 0 | PASS | yes |
| on (`docker/cluster.json`) | 425 / 319 / 427 | 0 | PASS | yes |

In every run the victim came back in term 1 and the leader kept its
term.

**Measuring it honestly on Docker Desktop.** The first figures (about
3.3 s even with keepalive) were not Raft's. After `docker network
connect`, Docker Desktop holds back both published-port forwarding and
`docker exec` output streams for about 3.3-3.8 s. A host-side poll, and
even a poll inside the container whose output is read on the host, sees
everything arrive at once, late.
- `scripts/chaos_docker.sh` now stamps each sample inside the container
  with its CLOCK_MONOTONIC, whether it has a non-loopback interface, and
  PID 1's start time on that same clock (`/proc/1/stat` field 22).
- `cluster_bench.py` computes catch-up from those stamps
  (`restart.in_network`): from the 0→1 network transition (partition) or
  the server's start (restart).
- The host-side figure is still reported, but on Docker Desktop it
  includes that forwarding delay.

**Tests.**
- `test_server_accepts_keepalive_pings_without_goaway`: a channel with
  1 s keepalive idles 6.5 s against the server and stays READY. Without
  the server-side allowance it receives GOAWAY and goes IDLE, which was
  checked by stubbing the allowance out.
- Tests that keepalive off adds no ping options and that a negative
  value is rejected.

The 9 Docker fault runs in `bench/results/docker/` (leader kill, partition
with and without PreVote, 3 each) were rerun with keepalive on: all
verified and linearizable; partition catch-up 262-401 ms with PreVote,
374-647 ms without.

## A follower's commit index went backwards: Figure 2's rule 5 taken literally

Found by: the bench's catch-up trace on the Docker cluster, while
investigating the partition stall (previous entry). Regression tests:
`tests/test_commit_index.py`; invariant `check_commit_index_monotonic`.

**Symptom.** On the Docker cluster, after a 10 s partition, the rejoined
follower's reported commit index flipped between 713 and 1225 for ~7 s
while it caught up (the bench's catch-up trace polls it every 50 ms). A
node's commit index must never decrease.

**Root cause.** `RaftNode._handle_append_entries` implemented AppendEntries
rule 5 word for word: "if leaderCommit > commitIndex, set commitIndex =
min(leaderCommit, index of last new entry)". When a message's last
covered index (`prev_log_index + len(entries)`) is *below* the follower's
commit index while its `leader_commit` is above it, that assignment lowers
the commit index. The leader produces such messages routinely since
per-peer flow control: to a peer with entries in flight, its heartbeat is
empty and anchored at `next_index - 1` -- the last index that peer has
*acknowledged* -- but carries the leader's *current* commit index. A
follower catching up chunk by chunk already holds the chunk in flight and
has committed through it (the chunk carried the commit index too); the
heartbeat then drops it back one chunk. On Docker the chunks were 512
entries (713 -> 1225), and the leader's view of the follower stayed stale
for seconds because its TCP connection to it was in retransmission backoff
(previous entry), so heartbeats kept arriving anchored at 713.

Reproduced directly on a `RaftNode`: entries 1-10 with leader_commit 10,
then a heartbeat with prev_log_index 5 and leader_commit 12: commit_index
10 -> 5.

**Why safety held.** Nothing is ever uncommitted by this: entries up to
the old commit index stay committed everywhere (a majority holds them,
and every later leader has them), `last_applied` never moves back so no
entry is un-applied or re-applied, and clients are answered from applied
state. What broke: the follower *reported* a stale, non-monotonic commit
index (Status, and the benchmark's catch-up measurement), and it stopped
applying until its commit index passed `last_applied` again. Had it won
an election meanwhile it would have started from a lower commit index,
which is safe -- a new leader recomputes it from match indexes.

**Why the simulator missed it.** No invariant looked at commit_index over
time, and the precondition is nearly unreachable in the fuzz phase:

- it needs a heartbeat to land *inside* a chunk's flight -- after the
  follower processed the chunk, before the leader processed the reply.
  The fuzz phase runs every link at zero delay, so a reply returns
  without simulated time passing; only an ADVANCE_CLOCK landing in exactly
  that gap opens the window;
- the other source of stale-anchored messages, STALE_REDELIVER and
  duplicates, always carries an *old* `leader_commit`, which the
  `leader_commit > commit_index` guard rejects.

Counted across both 1000-seed sweeps (2,000 runs, 66,214 AppendEntries
handled): 3,330 had a stale anchor and 17,721 a fresh leader_commit, but
none had both. With the new invariant in place and the fix reverted,
both sweeps still pass.

**Fix.** Only ever raise it:
`commit_index = max(commit_index, min(leader_commit, last_new_index))`.

- New invariant `check_commit_index_monotonic`, run by the fuzzer after
  every step: within one incarnation (a restart legitimately starts again
  from 0; incarnations are told apart by node object identity), a node's
  commit_index never decreases. Both sweeps check it now.
- `tests/test_commit_index.py`: the `RaftNode` reproduction; a
  deterministic simulator test -- a follower 20 entries behind catching up
  in 5-entry chunks, with a 33 ms delay on its replies only and a 10 ms
  tick, so each reply sends the next chunk mid-interval and the next
  heartbeat finds it in flight -- which fails without the fix ("node 0
  commit_index decreased from 10 to 5") and passes with it; and the
  checker's own restart/crash handling.
- No mutation-detection count moved.

**Results.** Earlier catch-up figures from the process bench were not affected: none of
their catch-up traces (leader-kill and large-gap, before and after the
fix) shows a decrease -- on localhost a chunk's reply always beats the
next heartbeat. After the fix, same benches, ext4, deferred, 16 clients:
leader-kill catch-up 597 / 718 / 517 ms (before: 774 / 580 / 591 ms),
large-gap 0.257 s (before: 0.365 s), within run-to-run spread; all
verified and linearizable (`bench/results/commitfix/`).
