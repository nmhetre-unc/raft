"""`NodeDriver`: runs one `RaftNode` against a real clock and transport.

`raft.sim.cluster.Cluster` is the first driver for the sans-IO
`RaftNode`; this is the second. It does no I/O itself. It is handed a
clock function and a `Transport`, and something outside it (the gRPC
server, or a test) feeds it three kinds of event:

- `tick()`, called periodically;
- `deliver(src, message)`, for every message that arrives;
- `submit(request)`, for every client request.

All three only enqueue. `run()` is the single consumer: it is the only
code that ever calls into the `RaftNode`, so the node never sees two
calls at once, and nothing else needs a lock.

**Persist before anything leaves.** `run()` takes every event already
queued as one batch, feeds each to the node, then makes the batch's
writes durable exactly once, and only after that completes sends the
batch's messages and resolves its client replies. One fsync covers the
whole batch, and no peer or client ever sees the effect of a write that
isn't durable. That includes a `NotLeader` reply: it carries a hint
derived from whatever the node just learned, so it waits for the sync
like everything else.

**Group commit.** The fsync runs on a worker thread, not the event loop:
`storage.begin_sync()` snapshots what must be written, on the loop, and
the job it returns writes and fsyncs that snapshot in the thread. While
it runs, the loop stays free to admit RPCs, so submits and messages pile
up in the inbox; when the sync completes and the batch's output has
gone, the next batch takes all of them, and one fsync covers them all.
(With the fsync on the loop, the loop admitted about one submit between
syncs, and each fsync covered one write however many were waiting.)

At most one sync is ever in flight, and the node is never touched while
it is: the driver's loop is the only code that calls into the node, and
it is suspended, awaiting the job, until the job completes -- inputs
arriving meanwhile only enqueue. The job itself reads nothing the loop
can change: everything it writes was copied out by `begin_sync()`.

**Fail-stop.** If the sync -- or anything else in the loop -- raises,
the driver stops for good: the batch's messages are never sent, its
replies and every other pending client reply are never resolved, events
queued during the failed sync are discarded unprocessed, later
`deliver`/`tick` calls are dropped, and `submit` raises
`DriverStopped`. `run()` re-raises the original error, so whatever
awaits it (the server's main) exits. A `FileStorage` whose `sync()`
failed refuses all later writes anyway; this makes sure no message
built from state that may not be on disk ever reaches anyone.

**Answering clients.** A submit is always appended to the log, `Get`
included, so a read is linearizable: it is answered from the state
machine at the point its own entry is applied, never by a local read.
When the entry is appended the driver records its `(index, term)`. Once
`last_applied >= index`:

- a different term at that index means a newer leader overwrote the
  entry before it committed: `Lost`;
- otherwise the result is the client's session in the state machine,
  provided the session's serial is this request's serial: `Ok`. If it
  isn't (a newer request from the same client was applied in the same
  batch, or this was a stale retry), the result is gone: `Timeout`.

A request not resolved within `request_timeout` (in the clock's units)
gets `Timeout`. `Lost` and `Timeout` both mean "retry with the same
serial number"; the state machine's dedup makes the retry safe.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from random import Random
from typing import Protocol

from raft.messages import Message
from raft.node import (
    DEFAULT_MAX_APPEND_BYTES,
    DEFAULT_MAX_APPEND_ENTRIES,
    NotLeader,
    RaftNode,
    Role,
    approximate_entry_size,
)
from raft.statemachine import ClientRequest
from raft.storage import LogEntry, Storage


class Transport(Protocol):
    """Where a driver's outgoing messages go. Must not block or raise:
    queue the message, and drop it if the peer is unreachable -- Raft
    already tolerates lost messages."""

    def send(self, dst: str, message: Message) -> None: ...


class SyncStorage(Storage, Protocol):
    """A `Storage` whose writes become durable only when synced, like
    `raft.net.filestorage.FileStorage` in ``deferred`` mode."""

    def begin_sync(self) -> Callable[[], None] | None:
        """Snapshot every write so far; return a job, safe to run on
        another thread, that makes them durable (None: nothing to do).
        See `FileStorage.begin_sync`."""
        ...


@dataclass(frozen=True)
class Ok:
    """The request was applied; `result` is what applying it returned."""

    result: object


@dataclass(frozen=True)
class Timeout:
    """The request's fate is unknown; retry with the same serial number."""


@dataclass(frozen=True)
class Lost:
    """A newer leader overwrote the entry before it committed; retry with
    the same serial number."""


SubmitOutcome = Ok | NotLeader | Timeout | Lost


@dataclass(frozen=True)
class Status:
    node_id: str
    role: Role
    current_term: int
    commit_index: int
    last_applied: int
    leader_hint: str | None


class DriverStopped(RuntimeError):
    """The driver hit a fatal error and has stopped; see `__cause__`."""


@dataclass
class DriverStats:
    """How the loop batches, for diagnosing group commit. Times are
    `time.monotonic()` seconds."""

    batches: int = 0
    events: int = 0
    submits: int = 0
    sync_seconds: float = 0.0
    # Submits enqueued while a sync was in progress: they can only be
    # covered by the next sync.
    submits_during_sync: int = 0
    # batch size (in client submits) -> how many batches had it
    submits_per_batch: Counter[int] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, object]:
        return {
            "batches": self.batches,
            "events": self.events,
            "submits": self.submits,
            "sync_seconds": self.sync_seconds,
            "submits_during_sync": self.submits_during_sync,
            "submits_per_batch": {str(k): v for k, v in sorted(self.submits_per_batch.items())},
        }


@dataclass(frozen=True)
class BatchRecord:
    """One trip round the loop (`trace_batches`)."""

    start: float  # batch taken from the inbox
    sync_start: float
    sync_end: float
    events: int
    submits: int


@dataclass(frozen=True)
class _Tick:
    pass


@dataclass(frozen=True)
class _Deliver:
    src: str
    message: Message


@dataclass(frozen=True)
class _Submit:
    request: ClientRequest
    future: asyncio.Future[SubmitOutcome]


_Event = _Tick | _Deliver | _Submit


@dataclass(frozen=True)
class _Waiter:
    index: int
    term: int
    client_id: str
    serial_number: int
    deadline: int
    future: asyncio.Future[SubmitOutcome]


class NodeDriver:
    def __init__(
        self,
        node_id: str,
        peers: list[str],
        storage: SyncStorage,
        transport: Transport,
        clock: Callable[[], int],
        *,
        request_timeout: int,
        election_timeout_range: tuple[int, int] = (150, 300),
        heartbeat_interval: int = 50,
        rng: Random | None = None,
        pre_vote: bool = True,
        max_append_entries: int = DEFAULT_MAX_APPEND_ENTRIES,
        max_append_bytes: int = DEFAULT_MAX_APPEND_BYTES,
        entry_size: Callable[[LogEntry], int] = approximate_entry_size,
        trace_batches: int = 0,
    ) -> None:
        """`trace_batches`: keep a `BatchRecord` for each of the last that
        many batches, in `batch_trace` (0: none)."""
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        self._node = RaftNode(
            node_id,
            peers,
            storage,
            election_timeout_range=election_timeout_range,
            heartbeat_interval=heartbeat_interval,
            rng=rng,
            append_noop_on_election=True,
            pre_vote=pre_vote,
            max_append_entries=max_append_entries,
            max_append_bytes=max_append_bytes,
            entry_size=entry_size,
        )
        self._storage = storage
        self._transport = transport
        self._clock = clock
        self._request_timeout = request_timeout
        self._inbox: asyncio.Queue[_Event] = asyncio.Queue()
        self._waiters: list[_Waiter] = []
        self._failure: BaseException | None = None
        self._syncing = False
        self.stats = DriverStats()
        self.batch_trace: deque[BatchRecord] = deque(maxlen=trace_batches)

    # -- Inputs (enqueue only) --

    def tick(self) -> None:
        if self._failure is None:
            self._inbox.put_nowait(_Tick())

    def deliver(self, src: str, message: Message) -> None:
        if self._failure is None:
            self._inbox.put_nowait(_Deliver(src, message))

    async def submit(self, request: ClientRequest) -> SubmitOutcome:
        """Replicate and apply `request`; wait for its outcome.

        Never returns after the driver has stopped: a submit pending at
        that moment waits forever (the process is exiting), and one made
        afterwards raises `DriverStopped`.
        """
        if self._failure is not None:
            raise DriverStopped("driver has stopped") from self._failure
        future: asyncio.Future[SubmitOutcome] = asyncio.get_running_loop().create_future()
        if self._syncing:
            self.stats.submits_during_sync += 1
        self._inbox.put_nowait(_Submit(request, future))
        return await future

    # -- Inspection --

    def status(self) -> Status:
        node = self._node
        return Status(
            node_id=node.node_id,
            role=node.role,
            current_term=node.current_term,
            commit_index=node.commit_index,
            last_applied=node.last_applied,
            leader_hint=node.leader_hint,
        )

    async def wait_idle(self) -> None:
        """Wait until every event queued so far has been processed."""
        await self._inbox.join()

    # -- The loop --

    async def run(self) -> None:
        """Process events until cancelled or a fatal error; see the module
        docstring. Re-raises the error that stopped it."""
        if self._failure is not None:
            raise DriverStopped("driver has stopped") from self._failure
        try:
            while True:
                batch = [await self._inbox.get()]
                while not self._inbox.empty():
                    batch.append(self._inbox.get_nowait())
                try:
                    await self._process(batch)
                finally:
                    for _ in batch:
                        self._inbox.task_done()
        except BaseException as exc:
            self._failure = exc  # from now on the inputs queue nothing
            # Whatever queued up during the failed sync is never processed
            # (its submits are never answered); unblock wait_idle().
            while not self._inbox.empty():
                self._inbox.get_nowait()
                self._inbox.task_done()
            raise

    async def _process(self, batch: list[_Event]) -> None:
        start = time.monotonic()
        out: list[tuple[str, Message]] = []
        replies: list[tuple[asyncio.Future[SubmitOutcome], SubmitOutcome]] = []
        for event in batch:
            now = self._clock()
            if isinstance(event, _Tick):
                out.extend(self._node.tick(now))
            elif isinstance(event, _Deliver):
                out.extend(self._node.handle(event.message, event.src, now))
            else:
                out.extend(self._append(event, now, replies))
            replies.extend(self._resolve_waiters(now))

        sync_start = time.monotonic()
        job = self._storage.begin_sync()  # raises -> nothing below happens, ever
        if job is not None:
            await self._sync(job)  # likewise
        sync_end = time.monotonic()
        submits = sum(isinstance(event, _Submit) for event in batch)
        stats = self.stats
        stats.batches += 1
        stats.events += len(batch)
        stats.submits += submits
        stats.sync_seconds += sync_end - sync_start
        stats.submits_per_batch[submits] += 1
        if self.batch_trace.maxlen:
            self.batch_trace.append(BatchRecord(start, sync_start, sync_end, len(batch), submits))

        for dst, message in out:
            self._transport.send(dst, message)
        for future, outcome in replies:
            if not future.done():  # the caller may have given up
                future.set_result(outcome)

    async def _sync(self, job: Callable[[], None]) -> None:
        """Run `job` on a worker thread; the loop runs on meanwhile."""
        self._syncing = True
        try:
            done = asyncio.get_running_loop().run_in_executor(None, job)
            try:
                await asyncio.shield(done)
            except asyncio.CancelledError:
                # Cancelled mid-sync (shutdown): the write can't be stopped,
                # so don't return -- and let the storage be closed --
                # until it has finished.
                while not done.done():
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.wait([done])
                if not done.cancelled():
                    done.exception()  # retrieved: the storage is broken anyway
                raise
        finally:
            self._syncing = False

    def _append(
        self,
        event: _Submit,
        now: int,
        replies: list[tuple[asyncio.Future[SubmitOutcome], SubmitOutcome]],
    ) -> list[tuple[str, Message]]:
        request = event.request
        result = self._node.append_command(request, now)
        if isinstance(result, NotLeader):
            replies.append((event.future, result))
            return []
        self._waiters.append(
            _Waiter(
                index=self._storage.last_log_index(),
                term=self._node.current_term,
                client_id=request.client_id,
                serial_number=request.serial_number,
                deadline=now + self._request_timeout,
                future=event.future,
            )
        )
        return result

    def _resolve_waiters(
        self, now: int
    ) -> list[tuple[asyncio.Future[SubmitOutcome], SubmitOutcome]]:
        ready: list[tuple[asyncio.Future[SubmitOutcome], SubmitOutcome]] = []
        pending: list[_Waiter] = []
        for waiter in self._waiters:
            outcome: SubmitOutcome
            if waiter.future.done():
                continue  # cancelled by the caller; forget it
            if self._node.last_applied >= waiter.index:
                if self._storage.term_at(waiter.index) != waiter.term:
                    outcome = Lost()
                else:
                    session = self._node.state_machine.session(waiter.client_id)
                    if session is not None and session[0] == waiter.serial_number:
                        outcome = Ok(session[1])
                    else:
                        outcome = Timeout()
            elif now >= waiter.deadline:
                outcome = Timeout()
            else:
                pending.append(waiter)
                continue
            ready.append((waiter.future, outcome))
        self._waiters = pending
        return ready
