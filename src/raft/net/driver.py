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
queued as one batch, feeds each to the node, then calls
`storage.sync()` exactly once, and only after it returns sends the
batch's messages and resolves its client replies. One fsync covers the
whole batch, and no peer or client ever sees the effect of a write that
isn't durable. That includes a `NotLeader` reply: it carries a hint
derived from whatever the node just learned, so it waits for the sync
like everything else.

**Fail-stop.** If `sync()` -- or anything else in the loop -- raises,
the driver stops for good: the batch's messages are never sent, its
replies and every other pending client reply are never resolved,
later `deliver`/`tick` calls are dropped, and `submit` raises
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
from collections.abc import Callable
from dataclasses import dataclass
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
    """A `Storage` whose writes become durable only at `sync()`, like
    `raft.net.filestorage.FileStorage` in ``deferred`` mode."""

    def sync(self) -> None: ...


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
    ) -> None:
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
                    self._process(batch)
                finally:
                    for _ in batch:
                        self._inbox.task_done()
        except BaseException as exc:
            # The inbox is empty here: _process is synchronous, so nothing
            # can be queued while a batch runs, and from now on the inputs
            # queue nothing.
            self._failure = exc
            raise

    def _process(self, batch: list[_Event]) -> None:
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

        self._storage.sync()  # raises -> nothing below happens, ever

        for dst, message in out:
            self._transport.send(dst, message)
        for future, outcome in replies:
            if not future.done():  # the caller may have given up
                future.set_result(outcome)

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
