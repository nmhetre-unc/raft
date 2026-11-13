"""A real three-process cluster: start, write, SIGKILL the leader, recover.

Each node is its own ``python -m raft.net.server`` subprocess with its
own data directory and ``--fsync per_write``. The client and the Status
polling run in this process over real gRPC. Timeouts are deliberately
generous (a loaded CI runner can stall a process for seconds); every
wait polls for the condition it needs rather than sleeping a fixed time.

On any failure each node's stderr (its log: role and term transitions,
startup, fatal errors) is printed, so a CI failure can be read without
rerunning. Every process is killed in ``finally``, pass or fail.

Marked ``integration``; skip it locally with
``-m "not slow and not integration"`` (a bare ``-m`` replaces the
``not slow`` default from pyproject.toml rather than adding to it).
"""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import IO

import grpc
import pytest

from raft.net import raft_pb2 as pb
from raft.net.client import RaftClient
from raft.net.codec import decode_log_entry
from raft.net.raft_pb2_grpc import RaftStub
from raft.statemachine import ClientRequest, Put

pytestmark = pytest.mark.integration

WAIT = 60.0  # seconds, per condition


def free_ports(n: int) -> list[int]:
    sockets = [socket.socket() for _ in range(n)]
    try:
        for s in sockets:
            s.bind(("127.0.0.1", 0))
        return [s.getsockname()[1] for s in sockets]
    finally:
        for s in sockets:
            s.close()


class ProcessCluster:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        ports = free_ports(3)
        self.nodes = {f"n{i}": f"127.0.0.1:{port}" for i, port in enumerate(ports)}
        self.config_path = tmp_path / "cluster.json"
        self.config_path.write_text(json.dumps({"nodes": self.nodes}))
        self.procs: dict[str, subprocess.Popen[bytes]] = {}
        self._logs: list[IO[bytes]] = []
        self._channels: dict[str, grpc.aio.Channel] = {}

    def log_path(self, node_id: str) -> Path:
        return self.tmp_path / f"{node_id}.log"

    def start(self, node_id: str) -> None:
        log = self.log_path(node_id).open("ab")  # a restart appends
        self._logs.append(log)
        self.procs[node_id] = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "raft.net.server",
                "--id",
                node_id,
                "--config",
                str(self.config_path),
                "--data-dir",
                str(self.tmp_path / node_id),
                "--fsync",
                "per_write",
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
        )

    def kill(self, node_id: str) -> None:
        proc = self.procs[node_id]
        proc.kill()  # SIGKILL: no handler runs, nothing is flushed or closed
        proc.wait(timeout=WAIT)

    def cleanup(self) -> None:
        for proc in self.procs.values():
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=WAIT)
        for log in self._logs:
            log.close()

    def dump_logs(self) -> None:
        for node_id in self.nodes:
            path = self.log_path(node_id)
            text = path.read_text(errors="replace") if path.exists() else "(no log)"
            proc = self.procs.get(node_id)
            code = proc.poll() if proc is not None else None
            print(f"\n===== {node_id} ({self.nodes[node_id]}), exit code {code} =====\n{text}")

    # -- Status over gRPC --

    def stub(self, node_id: str) -> RaftStub:
        if node_id not in self._channels:
            self._channels[node_id] = grpc.aio.insecure_channel(self.nodes[node_id])
        return RaftStub(self._channels[node_id])

    async def status(self, node_id: str, include_log: bool = False) -> pb.StatusReply:
        reply: pb.StatusReply = await self.stub(node_id).Status(
            pb.StatusRequest(include_log=include_log), timeout=5
        )
        return reply

    async def statuses(self, nodes: list[str], include_log: bool = False) -> list[pb.StatusReply]:
        return [await self.status(i, include_log) for i in nodes]

    async def close_channels(self) -> None:
        for channel in self._channels.values():
            await channel.close()


async def eventually(condition: Callable[[], Awaitable[bool]], what: str) -> None:
    """Poll until `condition` holds; an RPC error (node down, starting,
    or a channel in reconnect backoff) counts as "not yet"."""
    deadline = time.monotonic() + WAIT
    while True:
        try:
            if await condition():
                return
        except grpc.aio.AioRpcError:
            pass
        assert time.monotonic() < deadline, f"timed out after {WAIT}s waiting for {what}"
        await asyncio.sleep(0.1)


async def one_leader(cluster: ProcessCluster, nodes: list[str]) -> str:
    """Wait until exactly one of `nodes` leads and the rest follow it in
    the same term; return its id."""
    found: list[str] = []

    async def settled() -> bool:
        statuses = await cluster.statuses(nodes)
        leaders = [s.node_id for s in statuses if s.role == "leader"]
        if len(leaders) != 1:
            return False
        (leader,) = leaders
        term = next(s.current_term for s in statuses if s.node_id == leader)
        found[:] = [leader]
        return all(
            s.current_term == term and (s.node_id == leader or s.leader_hint == leader)
            for s in statuses
        )

    await eventually(settled, f"exactly one leader among {nodes}")
    return found[0]


def acknowledged_puts(entries: list[pb.LogEntry]) -> dict[str, str]:
    """Final value of every key Put through a client request, in log order."""
    values: dict[str, str] = {}
    for wire in entries:
        command = decode_log_entry(wire).command
        if isinstance(command, ClientRequest) and isinstance(command.command, Put):
            assert isinstance(command.command.value, str)
            values[command.command.key] = command.command.value
    return values


async def scenario(cluster: ProcessCluster) -> None:
    all_nodes = list(cluster.nodes)
    for node_id in all_nodes:
        cluster.start(node_id)
    leader = await one_leader(cluster, all_nodes)

    client = RaftClient(cluster.nodes, rpc_timeout=5.0, op_timeout=WAIT)
    try:
        acked: dict[str, str] = {}
        for n in range(20):
            await client.put(f"k{n}", f"v{n}")
            acked[f"k{n}"] = f"v{n}"

        cluster.kill(leader)
        survivors = [i for i in all_nodes if i != leader]
        new_leader = await one_leader(cluster, survivors)
        assert new_leader != leader
        await client.put("after-kill", "yes")
        acked["after-kill"] = "yes"
        assert client.history[-1].node == new_leader

        cluster.start(leader)  # same data dir

        async def converged() -> bool:
            statuses = await cluster.statuses(all_nodes, include_log=True)
            commits = {s.commit_index for s in statuses}
            if len(commits) != 1:
                return False
            (commit,) = commits
            committed = [list(s.log)[:commit] for s in statuses]  # log[0] is entry 1
            return all(log == committed[0] for log in committed)

        await eventually(converged, "identical committed logs on all three nodes")

        statuses = await cluster.statuses(all_nodes, include_log=True)
        for status in statuses:
            committed = list(status.log)[: status.commit_index]
            assert acknowledged_puts(committed) == acked, status.node_id
        for key, value in acked.items():  # and readable through the cluster
            assert await client.get(key) == value
    finally:
        await client.close()


def test_three_processes_survive_a_leader_sigkill(tmp_path: Path) -> None:
    cluster = ProcessCluster(tmp_path)

    async def main() -> None:
        try:
            await scenario(cluster)
        finally:
            await cluster.close_channels()

    try:
        asyncio.run(main())
    except BaseException:
        cluster.cleanup()  # first, so every log is complete
        cluster.dump_logs()
        raise
    finally:
        cluster.cleanup()
