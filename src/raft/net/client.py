"""`RaftClient`: an async key/value client for a `raft.net.server` cluster.

**One operation at a time -- never pipeline.** Each operation takes the
next serial number, and the cluster keeps only one session per client:
the last serial applied and its result. If two operations from the same
client commit together, the earlier one's result is overwritten before
the server can read it, and it is answered `timeout` even though it was
applied (see ``test_superseded_result_is_a_timeout`` in
``tests/test_driver.py``). So operations are serialized by a lock:
concurrent callers on one `RaftClient` simply queue. For parallelism,
use several clients, each with its own `client_id`.

**One serial per operation, for its whole life.** Every retry of an
operation resends the identical `(client_id, serial_number, command)`,
so the state machine's dedup applies it at most once however many times
it is retried:

- ``not_leader`` with a hint: go to the hinted node (a *redirect*);
- ``not_leader`` without one, ``lost``, ``timeout``, or a connection
  error / RPC deadline: forget the leader, back off briefly, and try the
  next member in turn.

A node that answered ``ok`` is remembered as the leader and tried first
next time. If an operation can't finish within `op_timeout`, it raises
`OperationTimeout`; it may still have been applied. Its serial is never
reused -- the next operation takes a new one.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

import grpc

from raft.net import raft_pb2 as pb
from raft.net.codec import decode_result, encode_client_request
from raft.net.raft_pb2_grpc import RaftStub
from raft.statemachine import NOT_FOUND, ClientRequest, Command, Delete, Get, Put


class OperationTimeout(TimeoutError):
    """An operation got no definite answer in time. It may or may not
    have been applied."""


@dataclass(frozen=True)
class OpRecord:
    """How one completed operation went."""

    serial_number: int
    command: Command
    attempts: int
    redirects: int
    elapsed: float  # seconds, first attempt to answer
    node: str  # who answered


class RaftClient:
    def __init__(
        self,
        members: dict[str, str],
        *,
        client_id: str | None = None,
        rpc_timeout: float = 3.0,
        op_timeout: float = 15.0,
        backoff: float = 0.05,
    ) -> None:
        """`members` maps each node id to its ``host:port``, in the order
        to try them when no leader is known."""
        if not members:
            raise ValueError("members must not be empty")
        self.client_id = client_id if client_id is not None else uuid.uuid4().hex
        self.leader: str | None = None
        self.history: list[OpRecord] = []
        self._members = dict(members)
        self._order = list(members)
        self._next = 0
        self._serial = 0
        self._rpc_timeout = rpc_timeout
        self._op_timeout = op_timeout
        self._backoff = backoff
        self._lock = asyncio.Lock()
        self._channels: dict[str, grpc.aio.Channel] = {}

    # -- Operations --

    async def put(self, key: str, value: str) -> None:
        await self.execute(Put(key, value))

    async def get(self, key: str) -> str | None:
        """The value under `key`, or None if there is none. Linearizable:
        it goes through the log like a write."""
        return self._value_or_none(await self.execute(Get(key)))

    async def delete(self, key: str) -> str | None:
        """Remove `key`; return the value it had, or None."""
        return self._value_or_none(await self.execute(Delete(key)))

    async def execute(self, command: Command) -> object:
        """Apply `command` exactly once; return `KVStateMachine.apply`'s result."""
        async with self._lock:
            self._serial += 1
            request = ClientRequest(self.client_id, self._serial, command)
            return await self._submit(encode_client_request(request), request)

    async def close(self) -> None:
        for channel in self._channels.values():
            await channel.close()
        self._channels.clear()

    # -- Internals --

    async def _submit(self, wire: pb.ClientRequest, request: ClientRequest) -> object:
        start = time.monotonic()
        deadline = start + self._op_timeout
        attempts = redirects = 0
        while True:
            if time.monotonic() >= deadline:
                raise OperationTimeout(
                    f"serial {request.serial_number} of {self.client_id}: no answer "
                    f"after {attempts} attempts"
                )
            node = self.leader if self.leader is not None else self._order[self._next]
            attempts += 1
            try:
                reply = await self._stub(node).Submit(wire, timeout=self._rpc_timeout)
            except grpc.aio.AioRpcError:
                reply = None

            if reply is not None and reply.WhichOneof("outcome") == "ok":
                self.leader = node
                self.history.append(
                    OpRecord(
                        request.serial_number,
                        request.command,
                        attempts,
                        redirects,
                        time.monotonic() - start,
                        node,
                    )
                )
                return decode_result(reply.ok)

            if reply is not None and reply.WhichOneof("outcome") == "not_leader":
                hint = reply.not_leader.leader_hint  # "" when unset
                if hint in self._members and hint != node:
                    self.leader = hint
                    redirects += 1
                    continue

            # No usable hint, lost, timeout, or unreachable: next member.
            self.leader = None
            self._next = (self._order.index(node) + 1) % len(self._order)
            await asyncio.sleep(self._backoff)

    def _stub(self, node: str) -> RaftStub:
        channel = self._channels.get(node)
        if channel is None:
            channel = self._channels[node] = grpc.aio.insecure_channel(self._members[node])
        return RaftStub(channel)

    @staticmethod
    def _value_or_none(result: object) -> str | None:
        if result is NOT_FOUND:
            return None
        assert isinstance(result, str)
        return result
