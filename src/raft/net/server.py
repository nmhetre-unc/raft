"""One Raft node as a process: ``python -m raft.net.server``.

::

    python -m raft.net.server --id n0 --config cluster.json --data-dir data/n0

``cluster.json`` lists every member and, optionally, timing in
milliseconds::

    {
      "nodes": {"n0": "127.0.0.1:7000", "n1": "127.0.0.1:7001", "n2": "127.0.0.1:7002"},
      "election_timeout_ms": [150, 300],
      "heartbeat_ms": 50,
      "request_timeout_ms": 2000,
      "pre_vote": true,
      "max_append_entries": 512,
      "max_append_bytes": 1048576
    }

`max_append_entries` and `max_append_bytes` cap one AppendEntries; the
byte cap is measured with the real protobuf encoding and must stay well
under gRPC's 4 MiB default receive limit, which nothing here raises.

The process owns three things: a `NodeDriver`'s `run()` task, a ticker
task that calls `tick()` every 10 ms (and is therefore also what makes
request timeouts fire on time -- the driver only checks deadlines when
it processes an event), and a gRPC server exposing ``Deliver`` (from
peers), ``Submit`` and ``Status`` (from clients).

Startup opens the storage first, so a second process on the same data
directory fails on the storage lock before it ever binds a port.

**Fail-stop.** If `run()` raises -- a failed fsync, typically -- the
gRPC server is stopped without answering any pending request, the error
is logged and the process exits with status 1. A node whose durable
state is in doubt must not keep participating; a supervisor restarts it
and recovery runs from whatever actually reached disk.

**Clean stop.** SIGTERM or SIGINT stops the ticker, the gRPC server and
the driver, then closes the storage, which syncs it and releases the
lock. Exit status 0.

`main()` only parses arguments, sets up logging and signals, and calls
`run(config)`, which is what the tests drive.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import grpc

from raft.net import raft_pb2 as pb
from raft.net.codec import (
    CodecError,
    decode_client_request,
    decode_envelope,
    decode_log_entry,
    encode_log_entry,
    encode_result,
)
from raft.net.driver import DriverStopped, Lost, NodeDriver, Ok, SubmitOutcome, Timeout
from raft.net.filestorage import FileStorage, FsyncMode
from raft.net.raft_pb2_grpc import RaftServicer, add_RaftServicer_to_server
from raft.net.transport import GrpcTransport
from raft.node import DEFAULT_MAX_APPEND_BYTES, DEFAULT_MAX_APPEND_ENTRIES, NotLeader
from raft.storage import LogEntry

log = logging.getLogger(__name__)

TICK_INTERVAL = 0.010  # seconds


@dataclass(frozen=True)
class ClusterConfig:
    nodes: dict[str, str]
    election_timeout_ms: tuple[int, int] = (150, 300)
    heartbeat_ms: int = 50
    request_timeout_ms: int = 2000
    pre_vote: bool = True
    max_append_entries: int = DEFAULT_MAX_APPEND_ENTRIES
    max_append_bytes: int = DEFAULT_MAX_APPEND_BYTES

    @classmethod
    def load(cls, path: str | Path) -> ClusterConfig:
        raw = json.loads(Path(path).read_text())
        defaults = cls(nodes={})
        lo, hi = raw.get("election_timeout_ms", defaults.election_timeout_ms)
        return cls(
            nodes={str(k): str(v) for k, v in raw["nodes"].items()},
            election_timeout_ms=(int(lo), int(hi)),
            heartbeat_ms=int(raw.get("heartbeat_ms", defaults.heartbeat_ms)),
            request_timeout_ms=int(raw.get("request_timeout_ms", defaults.request_timeout_ms)),
            pre_vote=bool(raw.get("pre_vote", defaults.pre_vote)),
            max_append_entries=int(raw.get("max_append_entries", defaults.max_append_entries)),
            max_append_bytes=int(raw.get("max_append_bytes", defaults.max_append_bytes)),
        )


@dataclass(frozen=True)
class ServerConfig:
    node_id: str
    cluster: ClusterConfig
    data_dir: Path
    fsync: FsyncMode = "deferred"
    # Where to listen, when not the address peers reach this node by --
    # e.g. 0.0.0.0:7000 in a container whose peers know it as n0:7000.
    listen: str | None = None
    # Keep a record of the driver's last N batches; written to
    # batches.json alongside stats.json (0: off).
    trace_batches: int = 0

    @property
    def address(self) -> str:
        return self.cluster.nodes[self.node_id]


def _encode(entry: LogEntry) -> bytes:
    return encode_log_entry(entry).SerializeToString()


def _wire_entry_size(entry: LogEntry) -> int:
    """`entry`'s size inside an AppendEntries: its encoding plus at most
    6 bytes of field tag and length prefix."""
    return encode_log_entry(entry).ByteSize() + 6


def _decode(data: bytes) -> LogEntry:
    return decode_log_entry(pb.LogEntry.FromString(data))


def _clock() -> int:
    return int(time.monotonic() * 1000)


def submit_reply(outcome: SubmitOutcome) -> pb.SubmitReply:
    if isinstance(outcome, Ok):
        return pb.SubmitReply(ok=encode_result(outcome.result))
    if isinstance(outcome, NotLeader):
        hint = outcome.leader_hint
        return pb.SubmitReply(
            not_leader=pb.NotLeader(leader_hint=None if hint is None else str(hint))
        )
    if isinstance(outcome, Lost):
        return pb.SubmitReply(lost=pb.Unit())
    assert isinstance(outcome, Timeout)
    return pb.SubmitReply(timeout=pb.Unit())


class _Servicer(RaftServicer):
    def __init__(self, driver: NodeDriver, storage: FileStorage) -> None:
        self._driver = driver
        self._storage = storage

    async def Deliver(
        self, request: pb.Envelope, context: grpc.aio.ServicerContext[pb.Envelope, pb.Unit]
    ) -> pb.Unit:
        try:
            src, message = decode_envelope(request)
        except CodecError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        self._driver.deliver(src, message)
        return pb.Unit()

    async def Submit(
        self,
        request: pb.ClientRequest,
        context: grpc.aio.ServicerContext[pb.ClientRequest, pb.SubmitReply],
    ) -> pb.SubmitReply:
        try:
            client_request = decode_client_request(request)
        except CodecError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        try:
            outcome = await self._driver.submit(client_request)
        except DriverStopped:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "node has stopped")
        return submit_reply(outcome)

    async def Status(
        self,
        request: pb.StatusRequest,
        context: grpc.aio.ServicerContext[pb.StatusRequest, pb.StatusReply],
    ) -> pb.StatusReply:
        status = self._driver.status()
        reply = pb.StatusReply(
            node_id=status.node_id,
            role=status.role.value,
            current_term=status.current_term,
            commit_index=status.commit_index,
            last_applied=status.last_applied,
            leader_hint=status.leader_hint,
        )
        if request.include_log:
            reply.log.extend(encode_log_entry(e) for e in self._storage.load_log()[1:])
        return reply


class RaftServer:
    """Everything one node process runs, with explicit start and stop."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._resources = contextlib.AsyncExitStack()

    async def start(self) -> None:
        """Bring the node up; on failure, release whatever was acquired."""
        async with contextlib.AsyncExitStack() as stack:
            await self._start(stack)
            self._resources = stack.pop_all()
        log.info(
            "%s: serving on %s (as %s)",
            self.config.node_id,
            self.config.listen or self.config.address,
            self.config.address,
        )

    async def _start(self, stack: contextlib.AsyncExitStack) -> None:
        config = self.config
        cluster = config.cluster
        # Storage first: its lock is what keeps two processes off one
        # data directory, and that must fail before anything is bound.
        # Released last, after everything that could still write to it.
        self.storage = FileStorage(
            config.data_dir, encode=_encode, decode=_decode, fsync=config.fsync
        )
        stack.callback(self.storage.close)  # syncs unless broken; releases the lock

        peers = {i: addr for i, addr in cluster.nodes.items() if i != config.node_id}
        # Reconnect backoff a third of the minimum election timeout, and
        # never above the transport's own 50 ms default (see transport.py).
        backoff_cap = max(1, min(50, cluster.election_timeout_ms[0] // 3))
        self.transport = GrpcTransport(
            config.node_id,
            peers,
            reconnect_backoff_ms=(min(20, backoff_cap), backoff_cap),
            min_connect_timeout_ms=min(20, backoff_cap),
        )
        stack.push_async_callback(self.transport.close)

        self.driver = NodeDriver(
            config.node_id,
            list(peers),
            self.storage,
            self.transport,
            _clock,
            request_timeout=cluster.request_timeout_ms,
            election_timeout_range=cluster.election_timeout_ms,
            heartbeat_interval=cluster.heartbeat_ms,
            pre_vote=cluster.pre_vote,
            max_append_entries=cluster.max_append_entries,
            max_append_bytes=cluster.max_append_bytes,
            entry_size=_wire_entry_size,
            trace_batches=config.trace_batches,
        )
        # SO_REUSEPORT off: a second process on this address must fail to
        # bind, not silently share the port.
        self.grpc_server = grpc.aio.server(options=[("grpc.so_reuseport", 0)])
        add_RaftServicer_to_server(_Servicer(self.driver, self.storage), self.grpc_server)
        self.grpc_server.add_insecure_port(config.listen or config.address)
        await self.grpc_server.start()
        # No grace: a pending Submit is cancelled, never given a made-up answer.
        stack.push_async_callback(self.grpc_server.stop, None)

        self.run_task = asyncio.create_task(self.driver.run(), name=f"driver-{config.node_id}")
        self.ticker_task = asyncio.create_task(self._ticker(), name=f"ticker-{config.node_id}")
        stack.push_async_callback(self._cancel_tasks)

    async def _cancel_tasks(self) -> None:
        self.ticker_task.cancel()
        self.run_task.cancel()
        await asyncio.gather(self.ticker_task, self.run_task, return_exceptions=True)

    async def _ticker(self) -> None:
        last = None
        while True:
            self.driver.tick()
            status = self.driver.status()
            current = (status.role, status.current_term)
            if current != last:
                log.info(
                    "%s: %s in term %d", status.node_id, status.role.value, status.current_term
                )
                last = current
            await asyncio.sleep(TICK_INTERVAL)

    async def serve(self, stop: asyncio.Event) -> None:
        """Run until `stop` is set, or re-raise whatever stopped the driver."""
        stop_wait = asyncio.create_task(stop.wait())
        try:
            await asyncio.wait({self.run_task, stop_wait}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stop_wait.cancel()
        if self.run_task.done():
            self.run_task.result()  # raises the fail-stop error

    def write_stats(self) -> None:
        """Write this process's counters to ``<data-dir>/stats.json``: what
        the transport has sent (see `GrpcTransport.stats`), the commit
        index, and CPU time. For benchmarks: ``kill -USR1`` before and
        after a load and take the difference."""
        times = os.times()
        status = self.driver.status()
        stats = {
            "node_id": self.config.node_id,
            "role": status.role.value,
            "commit_index": status.commit_index,
            "monotonic": time.monotonic(),
            "cpu_user_s": times.user,
            "cpu_system_s": times.system,
            "transport": self.transport.stats(),
            "driver": self.driver.stats.as_dict(),
            "storage": {
                "log_length": self.storage.last_log_index(),
                "load_log_calls": self.storage.load_log_calls,
                "load_log_entries": self.storage.load_log_entries,
                "load_log_seconds": self.storage.load_log_seconds,
                "log_fsyncs": self.storage.log_fsyncs,
                "log_fsync_seconds": self.storage.log_fsync_seconds,
                "entries_synced": self.storage.entries_synced,
            },
        }
        files: dict[str, object] = {"stats.json": stats}
        if self.driver.batch_trace.maxlen:
            # Written first: a reader waiting for stats.json then finds both.
            files = {
                "batches.json": [dataclasses.astuple(b) for b in self.driver.batch_trace],
                **files,
            }
        for name, content in files.items():
            path = self.config.data_dir / name
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(content))
            tmp.replace(path)

    async def stop(self) -> None:
        """Stop the tasks and the gRPC server, then close the transport and
        the storage. Safe whether the driver is alive or has failed."""
        await self._resources.aclose()
        log.info("%s: stopped", self.config.node_id)


async def run(
    config: ServerConfig, stop: asyncio.Event | None = None, *, stats_on_sigusr1: bool = False
) -> None:
    """Serve until `stop` is set (cleanly) or the driver fails (raises).
    With `stats_on_sigusr1`, SIGUSR1 calls `RaftServer.write_stats`."""
    server = RaftServer(config)
    await server.start()
    if stats_on_sigusr1:
        asyncio.get_running_loop().add_signal_handler(signal.SIGUSR1, server.write_stats)
    try:
        await server.serve(stop if stop is not None else asyncio.Event())
    finally:
        await server.stop()


async def _run_until_signalled(config: ServerConfig) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    await run(config, stop, stats_on_sigusr1=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m raft.net.server")
    parser.add_argument("--id", required=True, help="this node's id in the config")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--fsync", choices=["per_write", "deferred"], default="deferred")
    parser.add_argument(
        "--listen", help="host:port to listen on (default: this node's address in the config)"
    )
    parser.add_argument(
        "--trace-batches",
        type=int,
        default=0,
        metavar="N",
        help="record the driver's last N batches; SIGUSR1 writes them to batches.json",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cluster = ClusterConfig.load(args.config)
    if args.id not in cluster.nodes:
        parser.error(f"--id {args.id!r} is not in {args.config}")
    config = ServerConfig(
        args.id, cluster, args.data_dir, args.fsync, args.listen, args.trace_batches
    )
    try:
        asyncio.run(_run_until_signalled(config))
    except Exception:
        log.exception("%s: fatal error, exiting", args.id)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
