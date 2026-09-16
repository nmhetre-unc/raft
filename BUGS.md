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

## Invariant coverage gaps, measured

A mutation test against the 50-seed sweep found that 3 of 4 deliberately
injected replication bugs survive undetected:

- Blind truncate-and-reappend on a duplicate AppendEntries: invisible to
  value-comparing checks, since the log looks identical afterward.
  check_no_spurious_truncation was added specifically for this (it compares
  entry object identity, not value, and is proven correct on a
  hand-constructed violation) — but wiring it into the sweep did **not**
  close the gap. Instrumenting `Storage.truncate_from` directly across
  thousands of fuzzer-driven truncations (250 seeds total, up to n=7 with
  chaos weights skewed hard toward crash/restart/partition/client-request)
  never once produced the "different object, same value" condition the
  checker watches for. The reason is structural: `next_index` only ever
  walks backward one rejection at a time, and a rejection means the
  follower's entry at prev_log_index doesn't match — so by construction the
  walk lands exactly on the point of genuine agreement before any entries
  are ever sent. Whatever a leader sends past that point is therefore
  always either new or a real conflict, never "the follower already has
  this, from someone else." Still open; would need either a fuzzer action
  built specifically to construct that condition, or a different signal.
- match_index advancing on a failed reply: unobservable while nothing reads
  match_index for a safety decision. Expected to close in Milestone 4, when
  commitment reads it.
- next_index initialized to 1 rather than last_log_index + 1: not a safety
  bug. prev_log_index 0 always passes the sentinel check, so peers get a
  redundant resend that correct follower logic treats as a no-op.

Only skipping the consistency check entirely is caught by the current suite,
42 of 50 seeds, via log matching.

A clean sweep means the implemented checks found nothing, not that the
implementation is correct.