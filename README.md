# raft

A Raft consensus implementation in Python, built the other way around: the
fault-injection harness first, the replication code it exists to catch bugs
in second.

## Premise

The actual deliverable here is the harness, not the consensus protocol —
a deterministic simulator, a seeded fuzzer with replay and shrink, six
invariant checkers, and [BUGS.md](BUGS.md) as the running record of what
each one has caught. That ordering is deliberate, not incidental: Milestone
2 (the fuzzer and its invariant checks) was built and hardened *before*
Milestone 3 (log replication) — the code most likely to contain real bugs —
specifically so that when replication landed, a wrong `AppendEntries`
handler or a miscounted majority would get caught the day it was written,
against a harness whose own correctness had already been exercised on
simpler code (leader election) first.

Every run reproduces exactly from `(n, seed, steps)`. A failing run shrinks
by delta debugging (`Fuzzer.shrink`) to the smallest trace that still
reproduces the identical violation before it ever reaches a bug report —
BUGS.md's entries are written from shrunk traces, not raw multi-hundred-step
runs.

## What was found and proven

Leading with results, not features — everything below is backed by a
passing test that asserts the specific number cited, not a general "tests
pass."

**Figure 8's commit-safety guard, proven in both directions.** Raft's
Figure 8 describes a scenario where a leader can be tricked into committing
an entry that a later, honestly elected leader then legitimately overwrites
— unless commitment is restricted to entries written in the leader's own
current term. `tests/test_figure8.py` reconstructs the exact scenario
through real election and replication code (no hand-built log state) and
proves the guard both ways: with it, `commit_index` never advances past the
trap entry and the overwrite that follows is never a violation of
anything; with it removed
(`test_figure8_without_the_guard_commits_and_then_overwrites_it`, the
restriction deleted via `monkeypatch`, nothing else changed), the same
sequence commits the entry and then overwrites it — the exact State
Machine Safety violation the guard exists to prevent. A test that only ran
the guarded version would have proven nothing about the guard itself;
this is deliberately both halves.

**Two bugs found in the harness's own code**, not the code under test —
worth stating explicitly, since a fault-injection harness that's never
wrong about its own logic would be a first:

- `check_no_spurious_truncation` compared log entries by Python object
  identity, not value. On this codebase specifically, that signature was
  *structurally unreachable* — no `LogEntry` is ever cloned, so a resend
  built from a leader's own storage is always identity-equal to what it
  sent before — while genuine value-level loss (a stale, reordered
  `AppendEntries` discarding entries a follower already held) sailed
  through undetected. Diagnosed by chasing why a truncation-on-resend
  mutation was only caught in 2/50 seeds, and only by an unrelated
  checker. Fixed by rewriting the comparison to be value-based; see
  [BUGS.md](BUGS.md#check_no_spurious_truncation-compared-log-entries-by-identity-missing-real-entry-loss).
- A sweep-level test verifying no client request is ever double-applied
  tagged each `KVStateMachine` instance by raw `id()` to tell "genuinely
  reused" apart from "crashed and correctly rebuilt." On the first
  attempt this produced a false positive on seed 9: an address freed by
  garbage collection got reused by an unrelated instance, and the test
  read that as the same machine double-applying a command it never
  touched. Fixed by tagging each instance with a monotonic counter at
  construction instead of trusting its address — see
  `tests/test_client_sessions.py::test_no_double_application_across_the_fixed_seed_sweep`.

**Mutation testing's honest zeros.** Six deliberate bugs, injected via
`monkeypatch` (never committed to `src/`), run through the same 50-seed
sweep the invariant checkers are graded by. Two are worth calling out by
name for what they show about the checkers themselves, not just the bugs:

- **Mutation 4** — a leader initializing `next_index` to 1 instead of
  `last_log_index + 1` for every peer — is **correctly undetected: 0/50**.
  Every legitimate log index is 1-based with a sentinel at index 0 (see
  Architecture below), so `prev_log_index = 0` always passes the
  consistency check trivially, and a follower's own idempotent handling of
  a redundant resend absorbs the rest. This isn't a coverage gap being
  chased down; it's a genuinely harmless mutation, reported as one instead
  of dressed up as a false success.
- **Mutation 1** — blind truncate-and-reappend on every `AppendEntries`,
  even an exact resend that should be a no-op — has the project's most
  instructive reachability story: 2/50 (only via an unrelated checker,
  before the identity-vs-value fix above) → 43/50 (once that fix landed) →
  45/50 (once `STALE_REDELIVER` started deliberately constructing the
  precondition instead of waiting for lucky network reordering) → 41/50
  (after `CLIENT_REQUEST` stopped omnisciently targeting the actual leader
  and started following a `NOT_LEADER` redirect hint the way a real client
  has to — a traffic-pattern change that shifts *which* seeds happen to
  hit the precondition, not a regression in the checker or the mutation).
  Four different numbers for the identical bug, each one explained, not
  just recorded.

**The structural gap Milestone 5 found in its own checkers.** All six
invariant checkers inspect *replicated log content* or replication-level
bookkeeping (`match_index`, `commit_index`, `last_applied`'s bound) — none
of them ever look at what the state machine actually applied
(`KVStateMachine._data`/`_sessions`). Mutation 6 proves it: skipping
client-session dedup entirely (always re-applying a retried or redelivered
request instead of recognizing it as already handled) scores **0/50** across
all six checkers — not because the bug is subtle, but because nothing in
the checker suite was ever looking at the state machine's own applied
output. It's still a real, provable bug: a hand-built test
(`test_mutation_skip_session_dedup_check_is_a_structural_gap_for_existing_checkers`)
shows it clobbering a newer value with an older one under the right
interleaving, caught by a dedicated `apply()`-call-counting spy (38/50 on
the fuzz sweep) — just not by anything wired into `Fuzzer._check_invariants`.
Reported as an open structural gap, not silently patched around.

## Architecture, briefly

Three decisions carry disproportionate weight; `src/raft/node.py` and
`src/raft/invariants.py`'s own module docstrings go into full depth, not
repeated here:

- **The log is 1-based, with a permanent sentinel at index 0** (`term=0`,
  no command). Every `prev_log_index = 0` check passes unconditionally —
  no special-casing an empty log — and it's why Mutation 4 above is
  genuinely harmless rather than merely undetected.
- **`next_index` is a guess; `match_index` is known truth**, and the two
  are never derived from each other. `next_index[peer]` is this leader's
  optimistic belief about where to send next, adjusted on rejection.
  `match_index[peer]` only ever advances on an actual, self-reported
  confirmation — enforced with an assertion, not a `max()`, because a
  would-be regression means a reply got misattributed somewhere, and
  that's a bug to surface, not paper over.
- **`AppendEntriesReply` is self-describing.** It carries the index the
  *follower itself* just verified, computed from the specific request it's
  answering — not the leader's record of what it last sent. The earlier
  design (leader tracks "what I most recently sent per peer," reinterprets
  whichever reply arrives against that) is exactly what let an overlapping
  in-flight send misattribute a reply and inflate `match_index` past what
  was actually confirmed; see BUGS.md's "reply attributed to the wrong,
  overlapping send" entry.
- **Handlers are idempotent by construction.** A follower only ever
  truncates and rewrites storage from the first genuine point of
  divergence — Raft's Log Matching Property guarantees agreement wherever
  terms already match, so a resent, already-matching `AppendEntries` never
  touches storage at all. This is precisely the property Mutation 1 (above)
  breaks and the harness eventually learned to catch reliably.

## Limitations

Stated specifically, not softened:

- **Milestone 6 (snapshotting / cluster membership changes) was not
  built.** Deliberately, not left unfinished: it was already lowest
  priority per this project's own plan, and its main would-be benefit —
  bounding the client-session map — was weighed against the correctness
  risk of implementing session eviction without any real client-lifecycle
  machinery (no open/close, no lease). Evicting a session a client might
  still legitimately retry against would silently reopen the exact
  double-apply hazard session dedup exists to prevent. Skipped on purpose;
  see BUGS.md's Milestone 5 entries for the full reasoning.
- **The client-session map grows unbounded.** One entry per distinct
  `client_id` ever seen, never pruned — documented as a permanent,
  accepted characteristic of an in-memory, non-snapshotting implementation
  since Milestone 5, not an oversight.
- **No invariant checker inspects applied state-machine content.** All six
  operate on the replicated log or replication bookkeeping; none read
  `KVStateMachine._data`/`_sessions`. Mutation 6 (above) proved this is a
  real, open gap — a genuine correctness bug in that area would currently
  need a dedicated test like the `apply()`-spy to be noticed at all, not
  the standard fuzz sweep. Not fixed here; recorded as a known limitation.
- **A client whose reply is dropped has no way to learn the outcome.** If
  a request commits but its reply is lost to fault injection (or its
  leader crashes before replying), the client's only recourse is a blind
  retry with the identical `(client_id, serial_number, command)`. That
  retry is always safe — session dedup guarantees at most one real
  application — just uninformative. This is a documented scope decision,
  not a bug: building a way to answer "what happened to serial N?" would
  mean new client-protocol surface, and it still wouldn't cover a request
  that's genuinely still in flight.
- **`get()` is a local, non-linearizable read**, documented since
  Milestone 5.1. It answers directly from whatever a node has applied so
  far, with no round trip through the log — a client reading a
  soon-to-be-deposed leader, or a lagging follower, can see stale or
  behind-committed data. Making reads linearizable needs a leader lease or
  Raft's own read-index protocol, neither implemented here.

## Where to look for more

- **[BUGS.md](BUGS.md)** — every bug found, in the implementation and in
  the harness itself, with root cause and fix, in the order they were
  found.
- **`tests/test_mutation_detection.py`** — the full mutation table: all
  six injected bugs, their detection rates, which checker (if any) caught
  each one, and why, including the reachability history above in full.
- **`src/raft/sim/fuzz.py`** (`Fuzzer`, `replay`, `shrink`) — the tooling
  itself. To reproduce any specific seed's failure:
  `pytest tests/test_fuzz_sweep.py --seed <N>`.

## Setup

Requires Python 3.12+.

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

```bash
pytest                                    # default suite, 50 fuzz seeds
pytest -m slow                            # 1000-seed sweep
pytest tests/test_fuzz_sweep.py --seed 8471   # reproduce one seed
```

If mypy fails on Windows with an Application Control policy error,
reinstall it from source: `pip install --no-binary mypy mypy==2.3.1`.
