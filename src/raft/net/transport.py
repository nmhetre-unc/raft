"""`GrpcTransport`: the driver's `Transport` over gRPC.

Every message goes to its peer as a one-way unary ``Deliver`` RPC whose
response is an empty ``Unit``. A reply (a ``RequestVoteReply``, an
``AppendEntriesReply``) is just another message the receiving node
produces, so it travels back as its own ``Deliver`` -- never as the
response to the one that prompted it. That keeps the wire shape exactly
the shape `RaftNode` already has: `(dst, message)` pairs, nothing more.

`send()` never blocks and never raises, as `Transport` requires. It
encodes the message and drops it on the peer's bounded queue; one sender
task per peer drains that queue, one RPC at a time. When a queue is full
the new message is dropped (and counted in `dropped`): a peer that is
down or slow then costs a fixed amount of memory, and Raft re-sends
everything that matters anyway -- heartbeats repeat, and AppendEntries
always carries everything from the peer's next_index onward. An RPC that
fails (peer down, deadline exceeded) is likewise a lost packet, counted
in `failed` and otherwise ignored.
"""

from __future__ import annotations

import asyncio
import logging

import grpc

from raft.messages import Message
from raft.net import raft_pb2 as pb
from raft.net.codec import encode_message
from raft.net.raft_pb2_grpc import RaftStub

log = logging.getLogger(__name__)


class GrpcTransport:
    def __init__(
        self,
        node_id: str,
        peers: dict[str, str],
        *,
        queue_size: int = 256,
        rpc_timeout: float = 0.5,
    ) -> None:
        """`peers` maps each peer's node id to its ``host:port``."""
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.node_id = node_id
        self.dropped = 0
        self.failed = 0
        self._rpc_timeout = rpc_timeout
        self._queues: dict[str, asyncio.Queue[pb.Envelope]] = {
            peer: asyncio.Queue(maxsize=queue_size) for peer in peers
        }
        self._channels = {peer: grpc.aio.insecure_channel(addr) for peer, addr in peers.items()}
        self._tasks = [
            asyncio.create_task(self._sender(peer), name=f"send-{node_id}-to-{peer}")
            for peer in peers
        ]

    def send(self, dst: str, message: Message) -> None:
        try:
            self._queues[dst].put_nowait(encode_message(message, self.node_id))
        except asyncio.QueueFull:
            self.dropped += 1

    async def _sender(self, peer: str) -> None:
        stub = RaftStub(self._channels[peer])
        queue = self._queues[peer]
        while True:
            envelope = await queue.get()
            try:
                await stub.Deliver(envelope, timeout=self._rpc_timeout)
            except grpc.aio.AioRpcError as exc:
                self.failed += 1
                log.debug("%s -> %s: %s", self.node_id, peer, exc.code())

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for channel in self._channels.values():
            await channel.close()
