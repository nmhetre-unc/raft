"""End-to-end tests for `raft.net`: real gRPC servers on localhost.

Three `RaftServer`s run in-process, in one event loop per test, on free
localhost ports, with real `FileStorage` under `tmp_path` and the real
monotonic clock. Nothing here sleeps for a fixed time and hopes: every
wait polls for the condition it needs, with a generous deadline.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import threading
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import grpc
import pytest

from raft.messages import AppendEntriesReply
from raft.net import raft_pb2 as pb
from raft.net.client import OperationTimeout, RaftClient
from raft.net.codec import decode_log_entry, encode_log_entry
from raft.net.driver import Lost, Ok, Timeout
from raft.net.filestorage import FileStorage
from raft.net.raft_pb2_grpc import RaftStub
from raft.net.server import ClusterConfig, RaftServer, ServerConfig, main, run, submit_reply
from raft.net.transport import GrpcTransport
from raft.node import NotLeader
from raft.statemachine import NOT_FOUND
from raft.storage import StorageError

WAIT = 15.0  # seconds; generous so a loaded machine doesn't flake


def open_storage(directory: Path) -> FileStorage:
    return FileStorage(
        directory,
        encode=lambda e: encode_log_entry(e).SerializeToString(),
        decode=lambda b: decode_log_entry(pb.LogEntry.FromString(b)),
    )


def free_ports(n: int) -> list[int]:
    sockets = [socket.socket() for _ in range(n)]
    try:
        for s in sockets:
            s.bind(("127.0.0.1", 0))
        return [s.getsockname()[1] for s in sockets]
    finally:
        for s in sockets:
            s.close()


async def eventually(condition: Callable[[], Coroutine[Any, Any, bool]], what: str) -> None:
    """Poll `condition` until true. An RPC error counts as "not yet": a
    channel to a node that just restarted stays in reconnect backoff for
    a while and fails calls fast until it reconnects."""
    deadline = time.monotonic() + WAIT
    while True:
        try:
            if await condition():
                return
        except grpc.aio.AioRpcError:
            pass
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.02)


class Node:
    """One server as a test controls it: start, stop, inspect."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.server: RaftServer | None = None
        self.serving: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        self._stop = asyncio.Event()
        self.server = RaftServer(self.config)
        await self.server.start()
        self.serving = asyncio.create_task(self.server.serve(self._stop))

    async def stop(self) -> None:
        assert self.server is not None and self.serving is not None
        self._stop.set()
        await asyncio.gather(self.serving, return_exceptions=True)
        await self.server.stop()
        self.server = self.serving = None


class LiveCluster:
    def __init__(self, tmp_path: Path) -> None:
        ports = free_ports(3)
        self.cluster = ClusterConfig(
            nodes={f"n{i}": f"127.0.0.1:{port}" for i, port in enumerate(ports)},
            request_timeout_ms=1000,
        )
        self.tmp_path = tmp_path
        self.nodes = {i: Node(self.config(i)) for i in self.cluster.nodes}
        self._channels: dict[str, grpc.aio.Channel] = {}
        self.clients: list[RaftClient] = []

    def config(self, node_id: str, data_dir: Path | None = None) -> ServerConfig:
        return ServerConfig(node_id, self.cluster, data_dir or self.tmp_path / node_id)

    def stub(self, node_id: str) -> RaftStub:
        if node_id not in self._channels:
            self._channels[node_id] = grpc.aio.insecure_channel(self.cluster.nodes[node_id])
        return RaftStub(self._channels[node_id])

    def client(self, order: list[str] | None = None) -> RaftClient:
        order = order or list(self.cluster.nodes)
        client = RaftClient({i: self.cluster.nodes[i] for i in order})
        self.clients.append(client)
        return client

    async def status(self, node_id: str, include_log: bool = False) -> pb.StatusReply:
        request = pb.StatusRequest(include_log=include_log)
        reply: pb.StatusReply = await self.stub(node_id).Status(request, timeout=2)
        return reply

    def running(self) -> list[str]:
        return [i for i, n in self.nodes.items() if n.server is not None]

    async def leader(self) -> str:
        """Wait for one leader among the running nodes that every other
        running node follows, then return it."""
        found: list[str] = []

        async def settled() -> bool:
            statuses = [await self.status(i) for i in self.running()]
            leaders = [s for s in statuses if s.role == "leader"]
            if len(leaders) != 1:
                return False
            leader = leaders[0]
            found[:] = [leader.node_id]
            return all(
                s.current_term == leader.current_term
                and (s.node_id == leader.node_id or s.leader_hint == leader.node_id)
                for s in statuses
            )

        await eventually(settled, "a settled leader")
        return found[0]

    async def start_all(self) -> None:
        for node in self.nodes.values():
            await node.start()

    async def close(self) -> None:
        for client in self.clients:
            await client.close()
        for channel in self._channels.values():
            await channel.close()
        for node in self.nodes.values():
            if node.server is not None:
                await node.stop()


def live(body: Callable[[LiveCluster], Coroutine[Any, Any, None]]) -> Callable[[Path], None]:
    """Run `body` against a started 3-node cluster in a fresh event loop."""

    def test(tmp_path: Path) -> None:
        async def main() -> None:
            cluster = LiveCluster(tmp_path)
            try:
                await cluster.start_all()
                await body(cluster)
            finally:
                await cluster.close()

        asyncio.run(main())

    test.__name__ = body.__name__
    test.__doc__ = body.__doc__
    return test


# -- Client operations --


@live
async def test_put_get_delete(cluster: LiveCluster) -> None:
    client = cluster.client()
    await client.put("k", "v")
    assert await client.get("k") == "v"
    assert await client.delete("k") == "v"
    assert await client.get("k") is None
    assert await client.delete("k") is None
    assert [r.serial_number for r in client.history] == [1, 2, 3, 4, 5]
    assert all(r.elapsed > 0 for r in client.history)


@live
async def test_starting_at_a_follower_costs_one_redirect(cluster: LiveCluster) -> None:
    leader = await cluster.leader()
    follower = next(i for i in cluster.nodes if i != leader)
    client = cluster.client([follower, *(i for i in cluster.nodes if i != follower)])

    await client.put("k", "v")

    (record,) = client.history
    assert (record.redirects, record.attempts, record.node) == (1, 2, leader)
    assert client.leader == leader
    await client.put("k", "w")  # remembered: straight to the leader
    assert client.history[-1].attempts == 1


@live
async def test_writes_continue_after_stopping_the_leader(cluster: LiveCluster) -> None:
    client = cluster.client()
    await client.put("a", "1")
    old = await cluster.leader()

    await cluster.nodes[old].stop()
    await client.put("b", "2")

    new = await cluster.leader()
    assert new != old
    assert (await client.get("a"), await client.get("b")) == ("1", "2")


@live
async def test_restarted_node_catches_up_to_identical_state(cluster: LiveCluster) -> None:
    client = cluster.client()
    await client.put("a", "1")
    leader = await cluster.leader()
    lagging = next(i for i in cluster.nodes if i != leader)

    await cluster.nodes[lagging].stop()
    for n in range(10):
        await client.put(f"k{n}", str(n))
    await client.delete("a")
    await cluster.nodes[lagging].start()  # same data dir

    async def identical() -> bool:
        statuses = [await cluster.status(i, include_log=True) for i in cluster.nodes]
        return all(
            (s.last_applied, list(s.log)) == (statuses[0].last_applied, list(statuses[0].log))
            for s in statuses
        )

    await eventually(identical, "every node to hold the same applied log")
    snapshots = []
    for node in cluster.nodes.values():
        assert node.server is not None
        snapshots.append(node.server.driver._node.state_machine.snapshot())
    assert snapshots[0] == {f"k{n}": str(n) for n in range(10)}
    assert snapshots[0] == snapshots[1] == snapshots[2]


# -- Failure --


@live
async def test_storage_failure_stops_the_server(cluster: LiveCluster) -> None:
    client = cluster.client()
    await client.put("k", "v")
    leader = await cluster.leader()
    node = cluster.nodes[leader]
    assert node.server is not None and node.serving is not None
    storage = node.server.storage

    def failing_sync() -> None:
        storage._broken = True  # as a real failed fsync leaves it
        raise StorageError("simulated fsync failure")

    storage.sync = failing_sync  # type: ignore[method-assign]

    with pytest.raises(StorageError, match="simulated"):
        await node.serving
    # Between the failure and teardown, the node refuses rather than answers.
    with pytest.raises(grpc.aio.AioRpcError) as refused:
        await cluster.stub(leader).Submit(
            pb.ClientRequest(client_id="x", serial_number=1, command=pb.Command(get=pb.Get())),
            timeout=2,
        )
    assert refused.value.code() is grpc.StatusCode.UNAVAILABLE

    await node.server.stop()
    node.server = node.serving = None
    with pytest.raises(grpc.aio.AioRpcError) as gone:
        await cluster.status(leader)
    assert gone.value.code() is grpc.StatusCode.UNAVAILABLE

    await client.put("k", "w")  # the other two carry on
    assert await client.get("k") == "w"


@live
async def test_second_server_on_the_same_data_dir_is_refused(cluster: LiveCluster) -> None:
    with pytest.raises(StorageError, match="already open"):
        await run(cluster.config("n0"))
    await cluster.status("n0")  # the first is unaffected


@live
async def test_failed_startup_releases_the_data_dir(cluster: LiveCluster) -> None:
    # n1's address is taken (by n1), so binding fails after storage opened.
    other_dir = cluster.tmp_path / "other"
    config = ServerConfig("n1", cluster.cluster, other_dir)
    with pytest.raises(RuntimeError):
        await run(config)
    open_storage(other_dir).close()


# -- Server details --


@live
async def test_malformed_requests_are_rejected(cluster: LiveCluster) -> None:
    stub = cluster.stub("n0")
    with pytest.raises(grpc.aio.AioRpcError) as bad_envelope:
        await stub.Deliver(pb.Envelope(src="n1"), timeout=2)
    assert bad_envelope.value.code() is grpc.StatusCode.INVALID_ARGUMENT
    with pytest.raises(grpc.aio.AioRpcError) as bad_request:
        await stub.Submit(pb.ClientRequest(client_id="c", serial_number=1), timeout=2)
    assert bad_request.value.code() is grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (Ok("v"), pb.SubmitReply(ok=pb.Result(value="v"))),
        (Ok(NOT_FOUND), pb.SubmitReply(ok=pb.Result(not_found=pb.Unit()))),
        (NotLeader("n2"), pb.SubmitReply(not_leader=pb.NotLeader(leader_hint="n2"))),
        (NotLeader(None), pb.SubmitReply(not_leader=pb.NotLeader())),
        (Lost(), pb.SubmitReply(lost=pb.Unit())),
        (Timeout(), pb.SubmitReply(timeout=pb.Unit())),
    ],
)
def test_submit_reply_mapping(outcome: object, expected: pb.SubmitReply) -> None:
    assert submit_reply(outcome) == expected  # type: ignore[arg-type]


# -- Client edge cases --


def test_client_gives_up_after_op_timeout() -> None:
    async def body() -> None:
        (port,) = free_ports(1)  # nothing listening
        client = RaftClient(
            {"n0": f"127.0.0.1:{port}"}, rpc_timeout=0.1, op_timeout=0.3, backoff=0.01
        )
        with pytest.raises(OperationTimeout):
            await client.put("k", "v")
        assert client.leader is None
        await client.close()

    asyncio.run(body())


def test_client_needs_members() -> None:
    with pytest.raises(ValueError, match="members"):
        RaftClient({})


# -- Transport --


def test_transport_counts_drops_and_failures() -> None:
    async def body() -> None:
        (port,) = free_ports(1)  # nothing listening
        transport = GrpcTransport("n0", {"n1": f"127.0.0.1:{port}"}, queue_size=1, rpc_timeout=0.2)
        reply = AppendEntriesReply(term=1, success=True, match_index=0)
        for _ in range(3):  # no await between: the sender can't drain yet
            transport.send("n1", reply)
        assert transport.dropped == 2

        deadline = time.monotonic() + WAIT
        while transport.failed < 1:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.02)
        await transport.close()

    asyncio.run(body())


def test_transport_rejects_empty_queue() -> None:
    with pytest.raises(ValueError, match="queue_size"):
        GrpcTransport("n0", {}, queue_size=0)


# -- main() --


def write_config(tmp_path: Path, nodes: dict[str, str]) -> Path:
    path = tmp_path / "cluster.json"
    path.write_text(
        json.dumps(
            {
                "nodes": nodes,
                "election_timeout_ms": [150, 300],
                "heartbeat_ms": 50,
                "request_timeout_ms": 1000,
            }
        )
    )
    return path


def test_main_serves_until_sigterm_and_releases_the_lock(tmp_path: Path) -> None:
    (port,) = free_ports(1)
    address = f"127.0.0.1:{port}"
    config = write_config(tmp_path, {"solo": address})
    data_dir = tmp_path / "solo"
    loaded = ClusterConfig.load(config)
    assert loaded.election_timeout_ms == (150, 300) and loaded.request_timeout_ms == 1000

    def terminate_once_leader() -> None:
        async def poll() -> None:
            async with grpc.aio.insecure_channel(address) as channel:
                stub = RaftStub(channel)
                deadline = time.monotonic() + WAIT
                while time.monotonic() < deadline:
                    try:
                        status = await stub.Status(pb.StatusRequest(), timeout=1)
                        if status.role == "leader":
                            break
                    except grpc.aio.AioRpcError:
                        pass
                    await asyncio.sleep(0.05)
            os.kill(os.getpid(), signal.SIGTERM)

        asyncio.run(poll())

    thread = threading.Thread(target=terminate_once_leader)
    thread.start()
    code = main(["--id", "solo", "--config", str(config), "--data-dir", str(data_dir)])
    thread.join()

    assert code == 0
    open_storage(data_dir).close()


def test_main_exits_nonzero_when_the_data_dir_is_locked(tmp_path: Path) -> None:
    (port,) = free_ports(1)
    config = write_config(tmp_path, {"solo": f"127.0.0.1:{port}"})
    data_dir = tmp_path / "solo"
    with open_storage(data_dir):
        code = main(
            [
                "--id",
                "solo",
                "--config",
                str(config),
                "--data-dir",
                str(data_dir),
                "--fsync",
                "per_write",
            ]
        )
    assert code == 1


def test_main_rejects_an_unknown_id(tmp_path: Path) -> None:
    config = write_config(tmp_path, {"solo": "127.0.0.1:1"})
    with pytest.raises(SystemExit) as exit_info:
        main(["--id", "nope", "--config", str(config), "--data-dir", str(tmp_path / "d")])
    assert exit_info.value.code == 2
