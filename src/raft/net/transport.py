"""`GrpcTransport`: the driver's `Transport` over gRPC.

Every message goes to its peer as a one-way unary ``Deliver`` RPC whose
response is an empty ``Unit``. A reply (a ``RequestVoteReply``, an
``AppendEntriesReply``) is just another message the receiving node
produces, so it travels back as its own ``Deliver`` -- never as the
response to the one that prompted it. That keeps the wire shape exactly
the shape `RaftNode` already has: `(dst, message)` pairs, nothing more.

Reconnect backoff is kept far below the election timeout. gRPC's
defaults (1 s initial, growing to 120 s) mean that once a peer has been
down for a few seconds, the channel to it may sit in backoff for seconds
after the peer comes back -- and a node that hears nothing from the
leader for longer than its election timeout starts an election. That
was the restart election storm the benchmark measured (see BUGS.md):
the restarted node kept timing out and bumping the term while the
leader's channel to it was still waiting to retry. With backoff capped
well under the minimum election timeout, the leader reaches a restarted
peer within one heartbeat or two of it listening again.

`send()` never blocks and never raises, as `Transport` requires. It
encodes the message and drops it on the peer's bounded queue; one sender
task per peer drains that queue, one RPC at a time. When a queue is full
the new message is dropped (and counted in `dropped`): a peer that is
down or slow then costs a fixed amount of memory, and Raft re-sends
everything that matters anyway -- heartbeats repeat, and AppendEntries
always carries the next chunk from the peer's next_index. An RPC that
fails (peer down, deadline exceeded) is likewise a lost packet, counted
in `failed` and otherwise ignored.

`stats()` reports what this node has put on the wire: messages by type,
how many AppendEntries carried entries and how many entries in all, and
the encoded bytes -- counted when a message is encoded, so a message then
dropped on a full queue still counts.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from typing import Any

import grpc

from raft.messages import AppendEntries, Message
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
        reconnect_backoff_ms: tuple[int, int] = (20, 50),
        min_connect_timeout_ms: int = 20,
    ) -> None:
        """`peers` maps each peer's node id to its ``host:port``.

        `reconnect_backoff_ms` is gRPC's (initial, max) reconnect backoff
        and `min_connect_timeout_ms` its minimum connect timeout; keep all
        of them well under the minimum election timeout.
        """
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        initial_backoff, max_backoff = reconnect_backoff_ms
        if not 0 < initial_backoff <= max_backoff or min_connect_timeout_ms <= 0:
            raise ValueError("need 0 < initial backoff <= max backoff, connect timeout > 0")
        self.channel_options = [
            ("grpc.initial_reconnect_backoff_ms", initial_backoff),
            ("grpc.max_reconnect_backoff_ms", max_backoff),
            ("grpc.min_reconnect_backoff_ms", min_connect_timeout_ms),
        ]
        self.node_id = node_id
        self.dropped = 0
        self.failed = 0
        self.sent: Counter[str] = Counter()
        self.append_entries_with_entries = 0
        self.entries_sent = 0
        self.bytes_encoded = 0
        self._rpc_timeout = rpc_timeout
        self._queues: dict[str, asyncio.Queue[pb.Envelope]] = {
            peer: asyncio.Queue(maxsize=queue_size) for peer in peers
        }
        self._channels = {
            peer: grpc.aio.insecure_channel(addr, options=self.channel_options)
            for peer, addr in peers.items()
        }
        self._tasks = [
            asyncio.create_task(self._sender(peer), name=f"send-{node_id}-to-{peer}")
            for peer in peers
        ]

    def send(self, dst: str, message: Message) -> None:
        envelope = encode_message(message, self.node_id)
        self.sent[type(message).__name__] += 1
        self.bytes_encoded += envelope.ByteSize()
        if isinstance(message, AppendEntries) and message.entries:
            self.append_entries_with_entries += 1
            self.entries_sent += len(message.entries)
        try:
            self._queues[dst].put_nowait(envelope)
        except asyncio.QueueFull:
            self.dropped += 1

    def stats(self) -> dict[str, Any]:
        return {
            "sent": dict(self.sent),
            "append_entries_with_entries": self.append_entries_with_entries,
            "entries_sent": self.entries_sent,
            "bytes_encoded": self.bytes_encoded,
            "dropped": self.dropped,
            "failed": self.failed,
        }

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
