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

## Invariant coverage gaps, measured

A mutation test against the 50-seed sweep, re-run after `check_leader_
completeness` and `check_state_machine_safety` (Milestone 4's
commit-index-dependent checks) were wired in:

- Blind truncate-and-reappend on a duplicate AppendEntries:
  `check_no_spurious_truncation`'s own specific signature ("different
  object, same value") still never arises through the sweep — confirmed
  exactly as before, by instrumenting `Storage.truncate_from` directly and
  by constructing that triggering condition by hand
  (`test_mutation_1_triggering_condition_is_caught_when_constructed_directly`),
  which proves the checker fires the moment that state exists. The reason
  is structural: `next_index` only ever walks backward one rejection at a
  time, and a rejection means the follower's entry at `prev_log_index`
  doesn't match — so by construction the walk lands exactly on the point of
  genuine agreement before any entries are ever sent. Whatever a leader
  sends past that point is therefore always either new or a real conflict,
  never "the follower already has this, from someone else." But
  `check_leader_completeness` now catches 2/50 seeds anyway, via a
  *different* consequence of the same mutation: always truncating from
  `prev_log_index + 1` also discards anything a follower holds *beyond* the
  resent range, and on those 2 seeds that collateral damage later mattered
  to an election. Narrower than "the checker can't see this bug": the
  specific signature remains a fuzzer-reachability gap; it's just no longer
  true that nothing catches this mutation.
- Advancing match_index/next_index on a rejected reply, trusting it
  regardless of the reply's success flag: **closed**. 4/50 seeds, always via
  `check_leader_completeness` — some later, honestly elected leader ends up
  missing an entry an earlier leader believed it had committed, purely
  because of the corrupted match_index. This is exactly the class of real
  bug described above, and exactly the gap Leader Completeness was added to
  close.
- Skipping the AppendEntries consistency check entirely: 33/50 seeds — 32
  via log matching, 1 via leader completeness (a `SafetyViolation` stops the
  run at the first property that catches it, so which one fires first for a
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