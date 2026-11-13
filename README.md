# raft

[![CI](https://github.com/nmhetre-unc/raft/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nmhetre-unc/raft/actions/workflows/ci.yml)

A Raft consensus implementation in Python, built the other way around: the
fault-injection harness first, the replication code it exists to catch bugs
in second. Then the same protocol code was run as a real cluster -- separate
processes, gRPC, fsync'd logs, killed leaders, partitioned containers --
which found eight bugs the simulator could not (see
[Running it for real](#running-it-for-real)).

## At a glance

- **528 tests** in the default suite (50 fuzz seeds, plus a real
  multi-process cluster test), **100% line coverage**, enforced in CI
  (`--cov-fail-under=100`).
- **Zero safety violations** on the unmutated code across the 50-seed default
  sweep and the 1000-seed sweep (`pytest -m slow`), which runs every seed
  with and without PreVote (2,000 runs).
- **A real 5-process cluster** that survives leader kills and network
  partitions under load, with every run checked for agreement, lost
  writes and linearizability; CI runs a leader-kill cluster bench on
  every push.
- **Figure 8's commit-safety guard proven in both directions**: with the guard
  the exact scenario is safe; delete only the guard and the same run commits
  and then overwrites an entry.
- **Six deliberately injected bugs**, each graded by the 50-seed sweep: four
  caught by the invariant checkers (8/50 to 29/50 seeds), one correctly
  undetected because it is harmless (0/50), and one that every checker misses
  (0/50) but a dedicated apply-counting spy catches (36/50).
- **Three bugs found in the harness's own code and tests**, one of which hid
  real log-entry loss.

## Running it for real

The simulator proves the protocol logic; it can't see transports, disks,
message sizes, or time spent encoding. So the same `RaftNode` runs a second
way: as separate server processes over gRPC with fsync'd logs, under a
closed-loop load generator, while leaders are killed and followers are
partitioned. Each run is then checked for agreement, lost writes and
linearizability.

| Scenario (5 nodes, 16 clients, ext4) | Result | Before the fixes |
|---|---|---|
| Throughput, fsync before every ack | **668 ops/s** at 16 clients, **979** at 32 (p50 23.5 / 31.0 ms) | 333 / 337 ops/s |
| Leader SIGKILLed under load | new leader in **204-253 ms**; restarted node caught up in **517-718 ms**, 0 extra elections | restarted node **never** caught up (0 of 3 runs) |
| Follower 1,500 × 4 KiB writes behind (6.2 MB gap) | caught up in **0.26 s** | never (30 s timeout) |
| Follower partitioned 10 s (Docker) | caught up **262-401 ms** after rejoin, leader kept its term | 4.9-5.1 s |
| Every run above | all nodes agree on the committed prefix, **0 acknowledged writes lost**, client history **linearizable** | -- |

Sources: `bench/results/groupcommit/{before,after}-ext4-deferred-c{16,32}.json`,
`bench/results/commitfix/kill-ext4-deferred-{1,2,3}.json` and
`large-gap-ext4.json`, `bench/results/{before-fix1-run*,large-gap-before}.json`,
`bench/results/docker/partition-prevote-*.json`,
`bench/results/keepalive/ka-off-*.json`.

### Architecture

- **One protocol core, two drivers.** `RaftNode` is a pure state machine:
  messages and ticks in, messages out. The simulator drives it in
  deterministic steps; `raft.net.NodeDriver` drives the same object from an
  asyncio event loop in a real server process. No protocol code differs
  between the two.
- **Transport:** gRPC (`GrpcTransport`), one `Envelope` protobuf per Raft
  message, short reconnect backoff, and HTTP/2 keepalive so a dead link is
  detected in about a second.
- **Storage:** `FileStorage`, an append-only, checksummed log plus a term
  and vote file, with torn-write recovery on startup.
- **Durability with group commit:** the driver takes every queued event as
  one batch, appends to the log, and fsyncs once on a worker thread while
  the event loop keeps accepting requests. No message or client reply
  leaves before the fsync covering it completes.
- **Clients:** a retrying client with sessions (`client_id`, serial), so a
  retried write is applied at most once. Reads go through the log too.

### How correctness is checked on real processes

After every bench run, `bench/cluster_bench.py`:

1. **Committed-prefix agreement.** It reads every node's log and checks
   that all nodes agree on every entry up to the cluster's commit index.
2. **No lost acknowledged writes.** Every put a client was told succeeded
   must be in that committed prefix.
3. **Linearizability.** The full client history (every call and return,
   including timeouts with unknown outcome) is checked against a
   per-key register model by `raft.net.linearizability`, a Wing & Gong
   search with Lowe's memoization (Porcupine's approach), that minimizes
   any counterexample.

The checker was tested against a real bug. With the server patched to
answer `Get` from its local state machine instead of through the log, a
leader-kill run was flagged as not linearizable, minimized to 3 operations:
a get served by one node returned a value that a put acknowledged *before
the get started* had already overwritten
(`bench/results/mutation-local-reads.json`). Every run since the checker
was added passes all three checks.

### Bugs real deployment exposed that the simulator missed

Each is written up in [BUGS.md](BUGS.md) with the root cause and why the
simulator couldn't see it.

Correctness and liveness:

- **Restarted nodes never caught up**: the leader backed up one entry per
  round trip. 0 of 3 runs caught up → all caught up after conflict-term
  backtracking.
  [→](BUGS.md#a-restarted-node-could-never-catch-up-one-entry-log-backtracking)
- **Restart election storm**: gRPC's 1 s+ reconnect backoff hid the
  restarted node from the leader, so it kept calling elections. 20-26
  elections and 4.9-6.2 s catch-up → 0 elections and 229-276 ms (short
  backoff, plus PreVote).
  [→](BUGS.md#restarting-a-node-set-off-an-election-storm-against-a-healthy-leader)
- **Oversized catch-up messages**: one AppendEntries carried the whole
  gap (6.2 MB), over gRPC's 4 MiB limit, and was resent forever. Never
  caught up → 0.26-0.37 s, in 1 MiB chunks.
  [→](BUGS.md#a-follower-far-enough-behind-could-never-catch-up-one-appendentries-carried-the-whole-gap)
- **Partition stall**: a silently partitioned TCP connection sat in
  retransmission backoff (RTO 6.5 s) after the network returned. 4.9-5.1 s
  → 0.32-0.43 s with gRPC keepalive.
  [→](BUGS.md#a-partitioned-follower-rejoined-seconds-late-the-leaders-tcp-connection-was-in-retransmission-backoff)
- **Commit-index regression**: Figure 2's rule 5, taken literally, lowered
  a follower's commit index when a heartbeat was anchored below it (1225 →
  713, repeatedly for ~7 s) → never decreases; a new invariant checks it in
  every fuzz run.
  [→](BUGS.md#a-followers-commit-index-went-backwards-figure-2s-rule-5-taken-literally)

Performance:

- **Duplicate sends**: every write resent every unacknowledged entry to
  every peer. 295 → 891 ops/s at 16 clients with per-peer flow control
  (tmpfs).
  [→](BUGS.md#every-write-re-sent-every-unacknowledged-entry-to-every-peer)
- **Full-log copies**: every message copied the whole log, so throughput
  fell with uptime (842 → 165 entries/s over 300 s). Bounded reads: flat
  at 1,000-1,130 entries/s to a 286k-entry log (tmpfs).
  [→](BUGS.md#throughput-fell-with-uptime-every-message-copied-the-whole-log)
- **No group commit**: the fsync blocked the event loop, so each fsync
  covered one write. 333 → 668 ops/s at 16 clients and 337 → 979 at 32
  (ext4), at 3.8 and 5.8 writes per fsync.
  [→](BUGS.md#group-commit-never-happened-the-fsync-blocked-the-event-loop)

### Benchmark methodology, and its limits

- **Machine:** Intel Core 7 150U laptop (12 logical CPUs), 7.6 GiB RAM, Linux 6.18
  under WSL2 on Windows, Python 3.12. All 5 nodes and the clients share
  this one machine; nothing crosses a real network.
- **ext4 vs tmpfs.** Until step 8 the bench wrote to `/tmp`, which is
  tmpfs, where fsync is free; that was itself a bug
  ([BUGS.md](BUGS.md#every-local-benchmark-before-step-8-ran-on-tmpfs-fsync-was-free)).
  The headline throughput numbers above are process benches on ext4 (WSL2's
  virtual disk), with an fsync before every acknowledgement. Numbers
  measured on tmpfs (flow control, bounded reads) are labeled as such and
  aren't comparable to ext4 numbers. Every result records its filesystem
  (`environment.storage`).
- **Docker is for fault behavior only.** The Docker Desktop cluster
  (`bench/results/docker/`, `bench/results/keepalive/`) is used for
  elections, terms, catch-up timing and correctness under real network
  partitions, not throughput: its volumes live in Docker Desktop's VM.
  Catch-up there is timed on each container's own clock, because Docker
  Desktop delays port forwarding by seconds after a network reconnect.
- **Group commit costs a single client about 10%**: with nothing to batch,
  each write pays a thread hand-off (95-109 vs 116-121 ops/s at 1 client).
- **The remaining ceiling is the Python event loop.** On tmpfs, where
  fsync is free, the leader's event-loop thread is at 100% at about 1,000
  ops/s, mostly per-RPC gRPC and asyncio overhead. On ext4 at 32 clients
  it is already at 81%, with an fsync in flight 73% of the time
  (`groupcommit/after-ext4-deferred-c32.json`). Going much further would
  take batching several client requests per RPC or moving work off that
  single Python thread.
- Each fault run was repeated 3 times; ranges are min-max across those runs.

### How to run

```bash
pip install -e ".[dev,net]"

# 5 local server processes, 16 clients for 30 s; kill the leader at 10 s,
# restart it at 20 s. Writes bench/results/<name>.json and the client
# history; exits 1 unless verification and linearizability pass.
python bench/cluster_bench.py --clients 16 --duration 30 --kill-at 10 --restart-at 20

# A follower 1,500 × 4 KiB writes behind, then restarted.
python bench/large_gap_repro.py --entries 1500 --value-bytes 4096

# The same against a 5-node Docker cluster, with a 10 s partition.
docker compose up -d --build
python bench/cluster_bench.py --external docker/hosts.json --clients 16 --duration 30 \
  --chaos "scripts/chaos_docker.sh partition 10" --chaos-at 10
# Or kill the leader for 10 s: --chaos "scripts/chaos_docker.sh kill-leader 10"
docker compose down -v
```

CI runs the first command at a smaller scale on every push (3 nodes, 10 s,
leader killed at 3 s and restarted at 6 s) and fails on any verification or
linearizability failure.

## Premise

The actual deliverable here is the harness, not the consensus protocol —
a deterministic simulator, a seeded fuzzer with replay and shrink, seven
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

**Three bugs found in the harness's own code and tests**, not the code under
test — worth stating explicitly, since a fault-injection harness that's never
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
- `check_leader_completeness` gated on an entry's own term instead of the term
  that made the entry safe. Figure 8's current-term rule lets a leader commit a
  whole earlier-term prefix in one step, but the checker required every leader
  elected after the entry's *own* older term to hold it, so it falsely flagged
  an honestly elected leader (seed 711 of the 1000-seed sweep). Fixed by
  recording the entry's own term and the term at which it became safe
  (`established_at_term`) separately; see [BUGS.md](BUGS.md).
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
  hit the precondition, not a regression in the checker or the mutation)
  → 29/50 (after per-peer flow control stopped resending unacknowledged
  entries on every write, so far fewer resends land in the window this
  mutation needs). Five different numbers for the identical bug, each one
  explained, not just recorded.

**The structural gap Milestone 5 found in its own checkers.** All the
invariant checkers inspect *replicated log content* or replication-level
bookkeeping (`match_index`, `commit_index`, `last_applied`'s bound) — none
of them ever look at what the state machine actually applied
(`KVStateMachine._data`/`_sessions`). Mutation 6 proves it: skipping
client-session dedup entirely (always re-applying a retried or redelivered
request instead of recognizing it as already handled) scores **0/50** across
every checker — not because the bug is subtle, but because nothing in
the checker suite was ever looking at the state machine's own applied
output. It's still a real, provable bug: a hand-built test
(`test_mutation_skip_session_dedup_check_is_a_structural_gap_for_existing_checkers`)
shows it clobbering a newer value with an older one under the right
interleaving, caught by a dedicated `apply()`-call-counting spy (36/50 on
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
- **No invariant checker inspects applied state-machine content.** All seven
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
- **In the simulator, `get()` is a local, non-linearizable read**,
  documented since Milestone 5.1. It answers directly from whatever a node
  has applied so far — a client reading a soon-to-be-deposed leader, or a
  lagging follower, can see stale data. The networked server does not do
  this: a `Get` goes through the log like a write, which is linearizable
  but costs a full replication round trip. A leader lease or Raft's
  read-index protocol would make reads cheaper; neither is implemented.
- **The networked cluster has a fixed membership and no snapshots**, so
  its log, and a restarting node's replay of it, grows without bound.

## Where to look for more

- **[BUGS.md](BUGS.md)** — every bug found, in the implementation and in
  the harness itself, with root cause and fix, in the order they were
  found; the real-cluster bugs with before/after numbers.
- **`src/raft/net/`** — the networked driver, gRPC transport, durable
  storage, load generator and linearizability checker.
- **`bench/results/`** — every benchmark result cited here and in
  BUGS.md, each with its full client history.
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
pip install -e ".[dev,net]"     # net: gRPC and protobuf, for raft.net
```

```bash
pytest                                    # default suite, 50 fuzz seeds
pytest -m "not slow and not integration"  # the same, minus the multi-process test
pytest -m slow                            # 1000-seed sweep, both election modes
pytest tests/test_fuzz_sweep.py --seed 8471   # reproduce one seed
```

If mypy fails on Windows with an Application Control policy error,
reinstall it from source: `pip install --no-binary mypy mypy==2.3.1`.