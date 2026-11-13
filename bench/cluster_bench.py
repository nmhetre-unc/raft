"""Benchmark a real multi-process `raft.net` cluster, optionally with a
leader SIGKILL and restart.

    python bench/cluster_bench.py --clients 16 --duration 30 --fsync deferred \\
        --kill-at 10 --restart-at 20

Starts `--nodes` (default 5) ``python -m raft.net.server`` processes on
free localhost ports, each with its own data directory under a temporary
directory, waits for a leader, then runs `raft.net.loadgen` closed-loop
clients for `--duration` seconds. Meanwhile:

- every node's Status is polled every 50 ms; every term change (per
  node) and every change of cluster leader -- the reachable node
  claiming leadership in the highest term, including the same node
  winning a new term -- is recorded with a timestamp;
- at `--kill-at` the current leader is SIGKILLed;
- at `--restart-at` it is restarted on the same data directory.

Afterwards it waits for all live nodes to report the same commit index,
fetches every log, runs both `raft.net.verify` checks, and checks the
history for linearizability (`raft.net.linearizability`); any failure
makes the exit status 1. It prints a
report and writes ``<name>.json`` (environment, options, summary,
failover and restart metrics, events, verification) and
``<name>.history.jsonl`` (every operation) to `--out-dir`.

With `--external HOSTS.json` it starts nothing: it benchmarks a cluster
that is already running -- e.g. ``docker compose up -d`` -- reaching node
``<id>`` at ``HOSTS[id]`` (``docker/hosts.json``). Faults then come from
`--chaos`, a command run at `--chaos-at` (``scripts/chaos_docker.sh
kill-leader 10``) that prints ``down <node>`` when the fault takes effect
and ``up <node>`` when the node is back (and optionally ``leader <node>
<term>`` first, and ``begin down|up <node>`` just before each docker
command, which then times the event: the upper bound);
`--kill-at`/`--restart-at` and the resource figures need processes of our
own and are not available. A ``down`` of the node the
chaos command named as leader counts as a kill (failover metrics);
otherwise as a partition, for which it also reports the leader and its
term when the fault began and at the end, the term the isolated node
reported when it came back, and the leader changes after it did. ``up``
is treated like `--restart-at`: catch-up, elections and stall are measured
from it.

Data directories go on a real disk, under ``~/.cache/raft-bench``, so
fsync costs what it costs; `--tmpfs` puts them in /dev/shm instead, for
comparison. Every result records the filesystem type the data was on.
(Before this option existed, every run used the system temporary
directory, which here is tmpfs: fsync was free.)

All times in the output are seconds since the load started.

Failover is reported as (a) kill -> first observation of a new leader
(50 ms polling resolution) and (b) kill -> first successful client write
answered by a node other than the killed one. Restart disruption is
(c) elections triggered after the restart -- the rise in the highest
term any node reports -- and (d) the longest interval after the restart
with no write completing. Catch-up is the time from the restart to the
first poll at which the restarted node's commit index reaches the
highest commit index any node reported at the moment of the restart:
everything committed while it was down. It is watched until the run's
final convergence wait ends, so it can land after the load stops.

Resources, over the load window: every server's CPU time (from /proc),
and the leader's transport counters (`GrpcTransport.stats`, read with
SIGUSR1) per committed client operation -- AppendEntries sent, entries
they carried, bytes encoded. Every client operation, get or put, is one
log entry. With `--profile`, each server runs under ``py-spy record
--format raw --idle`` (``--native`` too with `--profile-native`) and the
profiles land in the result directory as ``<name>.<node>.raw``; py-spy
pauses the process for each sample, so profiled throughput is lower.

Not part of the test suite or coverage; ruff and mypy clean.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import contextlib
import dataclasses
import json
import os
import platform
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import IO, Any

import grpc

from raft.net import raft_pb2 as pb
from raft.net.client import RaftClient
from raft.net.codec import decode_log_entry
from raft.net.linearizability import check as check_linearizability
from raft.net.loadgen import (
    Fault,
    OpResult,
    first_write_after,
    longest_gap,
    run_load,
    summarize,
    write_history,
)
from raft.net.raft_pb2_grpc import RaftStub
from raft.net.verify import NodeLog, acknowledged_puts_committed, committed_prefixes_agree

POLL_INTERVAL = 0.05
POLL_CHANNEL_OPTIONS = [
    ("grpc.max_receive_message_length", -1),  # the final Status carries whole logs
    ("grpc.initial_reconnect_backoff_ms", 20),
    ("grpc.max_reconnect_backoff_ms", 50),
    ("grpc.min_reconnect_backoff_ms", 20),
]
STARTUP_TIMEOUT = 60.0
CONVERGE_TIMEOUT = 60.0


def free_ports(n: int) -> list[int]:
    sockets = [socket.socket() for _ in range(n)]
    try:
        for s in sockets:
            s.bind(("127.0.0.1", 0))
        return [s.getsockname()[1] for s in sockets]
    finally:
        for s in sockets:
            s.close()


DISK_ROOT = Path.home() / ".cache" / "raft-bench"
TMPFS_ROOT = Path("/dev/shm")


def filesystem_of(path: Path) -> dict[str, str]:
    """The mount `path` is on: its type, device and mount point, from the
    longest matching mount point in /proc/self/mounts."""
    target = str(path.resolve())
    best = {"fs_type": "unknown", "device": "unknown", "mount_point": "/"}
    best_len = -1
    for line in Path("/proc/self/mounts").read_text().splitlines():
        device, mount_point, fs_type = line.split()[:3]
        mount_point = mount_point.replace("\\040", " ")
        inside = target == mount_point or target.startswith(mount_point.rstrip("/") + "/")
        if inside and len(mount_point) > best_len:
            best = {"fs_type": fs_type, "device": device, "mount_point": mount_point}
            best_len = len(mount_point)
    return best


def docker_data_filesystem(container: str = "raft-n0") -> dict[str, str]:
    """Where an --external docker-compose cluster keeps its data: the
    engine (with Docker Desktop, volumes live in its VM, not on this
    host's disks) and, from inside `container`, the filesystem behind its
    /data volume."""

    def docker(*argv: str) -> str:
        try:
            return subprocess.run(
                ["docker", *argv], capture_output=True, text=True, check=True, timeout=30
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return ""

    engine = docker("info", "--format", "{{.OperatingSystem}} {{.ServerVersion}}")
    root = docker("info", "--format", "{{.DockerRootDir}}")
    # "<fstype> <source>" of the volume, as the container sees it.
    df = docker("exec", container, "df", "--output=fstype,source", "/data").splitlines()
    fs_type, _, device = df[-1].partition(" ") if len(df) > 1 else ("", "", "")
    where = "Docker Desktop's VM" if engine.startswith("Docker Desktop") else "the engine"
    return {
        "data_dir": f"docker volume /data of {container}, under {root or '?'} in {where}"
        + (", not on this host's disks" if where != "the engine" else ""),
        "engine": engine or "unknown",
        "fs_type": fs_type or "unknown",
        "device": device.strip() or "unknown",
    }


def server_cpu_s(pid: int) -> float:
    """User + system CPU seconds of `pid` so far (/proc/<pid>/stat)."""
    return server_cpu_s_of(Path(f"/proc/{pid}/stat"))


def thread_cpu_s(pid: int) -> dict[str, float]:
    """CPU seconds so far per thread name (several threads may share one)."""
    out: dict[str, float] = {}
    for task in Path(f"/proc/{pid}/task").iterdir():
        try:
            comm = (task / "comm").read_text().strip()
            if task.name == str(pid):
                comm = "main (event loop)"
            out[comm] = out.get(comm, 0.0) + server_cpu_s_of(task / "stat")
        except FileNotFoundError:
            continue  # thread exited
    return out


def server_cpu_s_of(stat: Path) -> float:
    fields = stat.read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


class Cluster:
    def __init__(
        self,
        nodes: int,
        fsync: str,
        workdir: Path,
        profile: list[str] | None = None,
        trace_batches: int = 0,
    ) -> None:
        self.trace_batches = trace_batches
        self.profile = profile
        self.fsync = fsync
        self.workdir = workdir
        self.nodes = {f"n{i}": f"127.0.0.1:{port}" for i, port in enumerate(free_ports(nodes))}
        self.config = workdir / "cluster.json"
        self.config.write_text(json.dumps({"nodes": self.nodes}))
        self.procs: dict[str, subprocess.Popen[bytes]] = {}
        self._logs: list[IO[bytes]] = []
        # Short reconnect backoff on the measuring side too: with gRPC's
        # defaults, after a node has been dead for seconds the poller's own
        # channel sits in backoff for seconds after it restarts, and every
        # restart measurement would include that delay.
        self._channels = {
            i: grpc.aio.insecure_channel(a, options=POLL_CHANNEL_OPTIONS)
            for i, a in self.nodes.items()
        }

    def start(self, node: str) -> None:
        log = (self.workdir / f"{node}.log").open("ab")
        self._logs.append(log)
        wrapper = []
        if self.profile is not None:
            out = self.workdir / f"{node}.raw"
            wrapper = ["py-spy", "record", "--format", "raw", "--idle", "--threads", "-o", str(out)]
            wrapper += [*self.profile, "--"]
        self.procs[node] = subprocess.Popen(
            [
                *wrapper,
                sys.executable,
                "-m",
                "raft.net.server",
                "--id",
                node,
                "--config",
                str(self.config),
                "--data-dir",
                str(self.workdir / node),
                "--fsync",
                self.fsync,
                "--trace-batches",
                str(self.trace_batches),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
        )

    def server_pid(self, node: str) -> int:
        """The server process's pid: the Popen's, or under py-spy its child's."""
        pid = self.procs[node].pid
        if self.profile is None:
            return pid
        for _ in range(100):
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            if children:
                return int(children[0])
            time.sleep(0.05)
        raise RuntimeError(f"no server process under py-spy for {node}")

    def kill(self, node: str) -> None:
        if self.profile is not None:
            os.kill(self.server_pid(node), signal.SIGKILL)
        self.procs[node].kill()
        self.procs[node].wait()

    async def leader_stats(self, node: str) -> dict[str, Any]:
        """Ask `node` to write its stats.json (SIGUSR1) and read it."""
        path = self.workdir / node / "stats.json"
        path.unlink(missing_ok=True)
        os.kill(self.server_pid(node), signal.SIGUSR1)
        for _ in range(200):
            if path.exists():
                stats: dict[str, Any] = json.loads(path.read_text())
                return stats
            await asyncio.sleep(0.01)
        raise TimeoutError(f"{node} wrote no stats")

    def alive(self) -> list[str]:
        return [n for n, p in self.procs.items() if p.poll() is None]

    async def status(
        self, node: str, include_log: bool = False, timeout: float = 1.0
    ) -> pb.StatusReply | None:
        try:
            reply: pb.StatusReply = await RaftStub(self._channels[node]).Status(
                pb.StatusRequest(include_log=include_log), timeout=timeout
            )
        except grpc.aio.AioRpcError:
            return None
        return reply

    async def statuses(
        self, include_log: bool = False, timeout: float = 1.0
    ) -> dict[str, pb.StatusReply]:
        replies = await asyncio.gather(*(self.status(n, include_log, timeout) for n in self.nodes))
        return {n: r for n, r in zip(self.nodes, replies, strict=True) if r is not None}

    async def shutdown(self) -> None:
        for channel in self._channels.values():
            await channel.close()
        for node, proc in self.procs.items():
            if proc.poll() is None:
                # Under py-spy, stop the server itself: py-spy writes its
                # profile when its child exits.
                os.kill(self.server_pid(node), signal.SIGTERM)
        for proc in self.procs.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for log in self._logs:
            log.close()


class ExternalCluster(Cluster):
    """A cluster someone else runs: `hosts` maps node id to ``host:port``.
    Nodes reported down by the chaos command are not expected to answer."""

    def __init__(self, hosts: dict[str, str]) -> None:
        self.profile = None
        self.nodes = dict(hosts)
        self.procs = {}
        self._logs = []
        self.down: set[str] = set()
        self._channels = {
            i: grpc.aio.insecure_channel(a, options=POLL_CHANNEL_OPTIONS)
            for i, a in self.nodes.items()
        }

    def alive(self) -> list[str]:
        return [n for n in self.nodes if n not in self.down]


def current_leader(statuses: dict[str, pb.StatusReply]) -> tuple[str | None, int]:
    """The node claiming leadership in the highest term, and that term."""
    leaders = [s for s in statuses.values() if s.role == "leader"]
    if not leaders:
        return None, max((s.current_term for s in statuses.values()), default=0)
    best = max(leaders, key=lambda s: s.current_term)
    return best.node_id, best.current_term


class Poller:
    """Polls every node's Status and records term and leader changes."""

    def __init__(self, cluster: Cluster, t0: float) -> None:
        self.cluster = cluster
        self.t0 = t0
        self.events: list[dict[str, Any]] = []
        self.terms: dict[str, int] = {}
        self.leader: str | None = None
        self.leader_term = 0
        self.max_term = 0
        self.max_commit = 0
        # (node, commit index it must reach) once a restart is being watched
        self.catch_up_target: tuple[str, int] | None = None
        self.caught_up_at: float | None = None
        # (seconds since t0, commit index) of the watched node, every poll
        # until it catches up: shows whether catch-up is steady or stalled.
        self.catch_up_trace: list[tuple[float, int]] = []

    def _record(self, **event: Any) -> None:
        self.events.append({"t": round(time.monotonic() - self.t0, 4), **event})

    async def poll_once(self) -> None:
        statuses = await self.cluster.statuses(timeout=POLL_INTERVAL * 4)
        for node, status in statuses.items():
            if self.terms.get(node) != status.current_term:
                self.terms[node] = status.current_term
                self._record(kind="term", node=node, term=status.current_term)
        self.max_term = max([self.max_term, *self.terms.values()])
        self.max_commit = max([self.max_commit, *(s.commit_index for s in statuses.values())])
        if self.catch_up_target is not None and self.caught_up_at is None:
            node, target = self.catch_up_target
            if node in statuses:
                self.catch_up_trace.append(
                    (round(time.monotonic() - self.t0, 3), statuses[node].commit_index)
                )
            if node in statuses and statuses[node].commit_index >= target:
                self.caught_up_at = time.monotonic()
                self._record(kind="caught_up", node=node, commit_index=target)
        leader, term = current_leader(statuses)
        # A re-election of the same node in a new term is a leadership
        # change too: it stepped down in between.
        if leader is not None and (leader, term) != (self.leader, self.leader_term):
            self._record(kind="leader", leader=leader, term=term, same_node=leader == self.leader)
            self.leader, self.leader_term = leader, term

    async def run(self) -> None:
        while True:
            started = time.monotonic()
            await self.poll_once()
            await asyncio.sleep(max(0.0, POLL_INTERVAL - (time.monotonic() - started)))


async def wait_for_leader(cluster: Cluster) -> str:
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        statuses = await cluster.statuses()
        leader, term = current_leader(statuses)
        if (
            leader is not None
            and len(statuses) == len(cluster.alive())
            and all(
                s.current_term == term and (s.node_id == leader or s.leader_hint == leader)
                for s in statuses.values()
            )
        ):
            return leader
        await asyncio.sleep(0.1)
    raise TimeoutError("no settled leader")


async def wait_for_convergence(cluster: Cluster) -> tuple[bool, dict[str, Any]]:
    """All live nodes report the same commit index on two polls in a row.
    Also returns the last per-node view, for diagnosing a failure."""
    deadline = time.monotonic() + CONVERGE_TIMEOUT
    previous = None
    view: dict[str, Any] = {}
    while time.monotonic() < deadline:
        statuses = await cluster.statuses()
        view = {
            n: {
                "role": s.role,
                "term": s.current_term,
                "commit": s.commit_index,
                "applied": s.last_applied,
                "hint": s.leader_hint,
            }
            for n, s in sorted(statuses.items())
        }
        view["alive"] = cluster.alive()
        commits = {s.commit_index for s in statuses.values()}
        if len(statuses) == len(cluster.alive()) and len(commits) == 1:
            if commits == previous:
                return True, view
            previous = commits
        else:
            previous = None
        await asyncio.sleep(0.2)
    return False, view


def git_revision() -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return rev + ("-dirty" if dirty else "")


async def bench(args: argparse.Namespace, workdir: Path) -> dict[str, Any]:
    profile = None
    if args.profile:
        profile = ["--native"] if args.profile_native else []
    external = args.external is not None
    cluster: Cluster
    if external:
        cluster = ExternalCluster(json.loads(args.external.read_text()))
    else:
        cluster = Cluster(args.nodes, args.fsync, workdir, profile, args.trace_batches)
    try:
        if not external:
            for node in cluster.nodes:
                cluster.start(node)
        load_leader = await wait_for_leader(cluster)
        if not external:
            pids = {n: cluster.server_pid(n) for n in cluster.nodes}
            cpu_start = {n: server_cpu_s(pid) for n, pid in pids.items()}
            stats_start = await cluster.leader_stats(load_leader)
            threads_start = thread_cpu_s(pids[load_leader])

        t0 = time.monotonic()
        t0_unix = time.time()  # to line events up with outside captures
        poller = Poller(cluster, t0)
        await poller.poll_once()
        poll_task = asyncio.create_task(poller.run())

        timeline: list[dict[str, Any]] = []

        async def sample_leader() -> None:
            """Every --stats-every seconds, one bucket of leader stats."""
            previous = stats_start
            while True:
                await asyncio.sleep(args.stats_every)
                current = await cluster.leader_stats(load_leader)
                dt = current["monotonic"] - previous["monotonic"]
                ps, cs = previous["storage"], current["storage"]
                committed = current["commit_index"] - previous["commit_index"]
                cpu = (current["cpu_user_s"] + current["cpu_system_s"]) - (
                    previous["cpu_user_s"] + previous["cpu_system_s"]
                )
                timeline.append(
                    {
                        "t_end_s": round(current["monotonic"] - t0, 2),
                        "log_length": cs["log_length"],
                        "committed_per_s": committed / dt,
                        "leader_cpu_percent": 100 * cpu / dt,
                        "load_log_percent_of_wall": 100
                        * (cs["load_log_seconds"] - ps["load_log_seconds"])
                        / dt,
                        "load_log_calls_per_entry": (cs["load_log_calls"] - ps["load_log_calls"])
                        / max(committed, 1),
                        "entries_copied_per_entry": (
                            cs["load_log_entries"] - ps["load_log_entries"]
                        )
                        / max(committed, 1),
                    }
                )
                previous = current

        sampler = (
            asyncio.create_task(sample_leader())
            if args.stats_every and not args.kill_at and not external
            else None
        )

        faults: list[Fault] = []
        restart_info: dict[str, Any] = {}

        async def kill_leader() -> None:
            await asyncio.sleep(args.kill_at)
            await poller.poll_once()
            leader = poller.leader
            assert leader is not None, "no leader to kill"
            at = time.monotonic()
            cluster.kill(leader)
            faults.append(Fault("kill", at, leader))
            poller._record(kind="kill", node=leader)

        async def restart_killed() -> None:
            await asyncio.sleep(args.restart_at)
            (fault,) = [f for f in faults if f.label == "kill"]
            assert fault.node is not None
            await poller.poll_once()
            restart_info["max_term_before"] = poller.max_term
            restart_info["at"] = time.monotonic()
            poller.catch_up_target = (fault.node, poller.max_commit)
            cluster.start(fault.node)
            poller._record(kind="restart", node=fault.node)

        partition: dict[str, Any] = {}
        # From --chaos "commit" lines -- the node's own Status, polled from
        # inside its container: (node, commit, term, container clock,
        # has a network, container's server start on that clock).
        in_network: list[tuple[str, int, int, float, int, float]] = []

        async def chaos() -> None:
            """Run --chaos at --chaos-at; act on its down/up lines."""
            assert isinstance(cluster, ExternalCluster)
            await asyncio.sleep(args.chaos_at)
            proc = await asyncio.create_subprocess_exec(
                *shlex.split(args.chaos), stdout=asyncio.subprocess.PIPE
            )
            assert proc.stdout is not None
            named_leader = None
            begun: dict[tuple[str, str], float] = {}
            async for raw in proc.stdout:
                at = time.monotonic()
                word, node, *rest = raw.decode().split()
                if word == "begin":
                    # "begin down|up <node>": the docker command is starting.
                    # Its own latency (~300 ms for docker kill) is not the
                    # cluster's, so faults are timed from here.
                    begun[(node, rest[0])] = at
                    continue
                at = begun.pop((word, node), at)
                if word == "leader":
                    named_leader = node
                    poller._record(kind="chaos_leader", node=node, term=int(rest[0]))
                elif word == "down":
                    cluster.down.add(node)
                    kind = "kill" if node == named_leader else "partition"
                    faults.append(Fault(kind, at, node))
                    poller._record(kind=kind, node=node)
                    if kind == "partition":
                        partition.update(
                            node=node,
                            leader_at_down=poller.leader,
                            leader_term_at_down=poller.leader_term,
                        )
                elif word == "commit":
                    commit, term, clock, net, started = rest[:5]
                    in_network.append(
                        (node, int(commit), int(term), float(clock), int(net), float(started))
                    )
                elif word == "up":
                    restart_info["max_term_before"] = poller.max_term
                    restart_info["at"] = at
                    poller.catch_up_target = (node, poller.max_commit)
                    cluster.down.discard(node)
                    poller._record(kind="up", node=node)
                    if partition:
                        # The term it brings back: with PreVote off, every
                        # election timeout while isolated raised it by one.
                        deadline = time.monotonic() + 5
                        while time.monotonic() < deadline:
                            s = await cluster.status(node, timeout=0.2)
                            if s is not None:
                                partition["node_term_on_rejoin"] = s.current_term
                                partition["node_term_seen_at"] = round(time.monotonic() - at, 3)
                                break
                            await asyncio.sleep(0.01)
            if await proc.wait() != 0:
                raise RuntimeError(f"--chaos exited with {proc.returncode}")

        fault_tasks = []
        if args.chaos is not None:
            fault_tasks.append(asyncio.create_task(chaos()))
        if args.kill_at is not None:
            fault_tasks.append(asyncio.create_task(kill_leader()))
        if args.restart_at is not None:
            fault_tasks.append(asyncio.create_task(restart_killed()))

        def make_client(n: int) -> RaftClient:
            # The client checks its budget only between attempts, so one RPC
            # must not outlive it: cap the RPC timeout at the op timeout.
            return RaftClient(
                cluster.nodes,
                client_id=f"c{n:02d}",
                rpc_timeout=min(2.0, args.op_timeout),
                op_timeout=args.op_timeout,
            )

        ops = await run_load(
            make_client,
            clients=args.clients,
            duration=args.duration,
            put_ratio=args.put_ratio,
            keys=args.keys,
            seed=args.seed,
        )
        load_end = t0 + args.duration
        if sampler is not None:
            sampler.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sampler
        resources: dict[str, Any] = {"timeline": timeline}
        if not args.kill_at and not external:
            stats_end = await cluster.leader_stats(load_leader)
            wall = stats_end["monotonic"] - stats_start["monotonic"]
            cpu = {n: server_cpu_s(pid) - cpu_start[n] for n, pid in pids.items()}
            threads_end = thread_cpu_s(pids[load_leader])
            leader_threads = {
                t: round(100 * (c - threads_start.get(t, 0.0)) / wall, 1)
                for t, c in sorted(threads_end.items(), key=lambda kv: -kv[1])
            }
            committed = stats_end["commit_index"] - stats_start["commit_index"]
            t_start, t_end = stats_start["transport"], stats_end["transport"]
            sent_ae = t_end["sent"].get("AppendEntries", 0) - t_start["sent"].get(
                "AppendEntries", 0
            )
            per_entry = max(committed, 1)
            s_start, s_end = stats_start["storage"], stats_end["storage"]
            fsyncs = s_end["log_fsyncs"] - s_start["log_fsyncs"]
            synced = s_end["entries_synced"] - s_start["entries_synced"]
            fsync_s = s_end["log_fsync_seconds"] - s_start["log_fsync_seconds"]
            d_start, d_end = stats_start["driver"], stats_end["driver"]
            batches = d_end["batches"] - d_start["batches"]
            per_batch = {
                int(k): v - d_start["submits_per_batch"].get(k, 0)
                for k, v in d_end["submits_per_batch"].items()
            }
            with_submits = sum(v for k, v in per_batch.items() if k > 0)
            resources |= {
                "leader_driver": {
                    "batches": batches,
                    "events_per_batch": (d_end["events"] - d_start["events"]) / max(batches, 1),
                    "submits_per_batch_with_any": (d_end["submits"] - d_start["submits"])
                    / max(with_submits, 1),
                    "submits_per_batch_histogram": dict(sorted(per_batch.items())),
                    "submits_during_sync": d_end["submits_during_sync"]
                    - d_start["submits_during_sync"],
                    "sync_percent_of_wall": 100
                    * (d_end["sync_seconds"] - d_start["sync_seconds"])
                    / wall,
                },
                "leader_log_fsyncs": fsyncs,
                "leader_log_fsyncs_per_s": fsyncs / wall,
                "leader_entries_per_fsync": synced / max(fsyncs, 1),
                "leader_fsync_ms_mean": 1000 * fsync_s / max(fsyncs, 1),
                "leader_fsync_percent_of_wall": 100 * fsync_s / wall,
                "leader": load_leader,
                "wall_s": wall,
                "cpu_percent": {n: 100 * c / wall for n, c in cpu.items()},
                "leader_thread_cpu_percent": leader_threads,
                "committed_entries": committed,
                "append_entries_per_entry": sent_ae / per_entry,
                "append_entries_with_entries_per_entry": (
                    t_end["append_entries_with_entries"] - t_start["append_entries_with_entries"]
                )
                / per_entry,
                "entries_sent_per_entry": (t_end["entries_sent"] - t_start["entries_sent"])
                / per_entry,
                "bytes_encoded_per_entry": (t_end["bytes_encoded"] - t_start["bytes_encoded"])
                / per_entry,
                "leader_stats_start": stats_start,
                "leader_stats_end": stats_end,
            }
        if args.trace_batches and not args.kill_at and not external:
            trace = json.loads((workdir / load_leader / "batches.json").read_text())
            shutil.copy(workdir / load_leader / "batches.json", workdir / "batches.json")
            resources["sync_arrivals"] = sync_arrivals(trace, ops, t0, load_end)
        await asyncio.gather(*fault_tasks)

        converged, final_view = await wait_for_convergence(cluster)
        poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poll_task
        await poller.poll_once()

        statuses = await cluster.statuses(include_log=True, timeout=30.0)
        logs = {
            node: NodeLog(s.commit_index, [decode_log_entry(e) for e in s.log])
            for node, s in statuses.items()
        }
        longest = max(logs.values(), key=lambda log: log.commit_index)
        acked = [(op.key, op.value) for op in ops if op.ok and op.kind == "put" and op.value]
        verification = {
            "converged": converged,
            "final_view": final_view,
            "nodes_checked": sorted(logs),
            "commit_index": longest.commit_index,
            "acknowledged_puts": len(acked),
            "prefix_problems": committed_prefixes_agree(logs),
            "ack_problems": acknowledged_puts_committed(longest.committed(), acked),
        }

        linear = check_linearizability(ops)
        verification["linearizability"] = {
            "linearizable": linear.ok,
            "operations": linear.operations,
            "checked": linear.checked,
            "keys": linear.keys,
            "segments": linear.segments,
            "seconds": round(linear.seconds, 3),
            "summary": linear.summary(),
            "unknown_outcome_puts": unknown_outcome_puts(ops),
        }
        summary = summarize(ops, start=t0, end=load_end, faults=faults)
        result: dict[str, Any] = {
            "t0_unix": t0_unix,
            "summary": dataclasses.asdict(summary),
            "resources": resources,
            "verification": verification,
            "events": poller.events,
        }
        kill = next((f for f in faults if f.label == "kill"), None)
        if kill is not None:
            new_leader = next(
                (
                    e
                    for e in poller.events
                    if e["kind"] == "leader" and e["t"] >= kill.at - t0 and e["leader"] != kill.node
                ),
                None,
            )
            result["failover"] = {
                "killed": kill.node,
                "kill_at": kill.at - t0,
                "new_leader": new_leader["leader"] if new_leader else None,
                "a_kill_to_new_leader_s": (
                    new_leader["t"] - (kill.at - t0) if new_leader else None
                ),
                "b_kill_to_first_write_s": first_write_after(ops, kill.at, not_from=kill.node),
            }
        if partition:
            up = restart_info["at"] - t0
            partition |= {
                "leader_at_end": poller.leader,
                "leader_term_at_end": poller.leader_term,
                "leader_changes_after_rejoin": sum(
                    1 for e in poller.events if e["kind"] == "leader" and e["t"] >= up
                ),
            }
            result["partition"] = partition
        if restart_info:
            restart_at = restart_info["at"]
            put_ends = [op.end for op in ops if op.ok and op.kind == "put"]
            stall, stall_at = longest_gap(put_ends, restart_at, load_end)
            caught_up_at = poller.caught_up_at
            result["restart"] = {
                "restart_at": restart_at - t0,
                "catch_up_target_commit": poller.catch_up_target[1]
                if poller.catch_up_target
                else None,
                "caught_up": caught_up_at is not None,
                "catch_up_trace": poller.catch_up_trace,
                "e_restart_to_catch_up_s": (
                    caught_up_at - restart_at if caught_up_at is not None else None
                ),
                "c_elections_after_restart": poller.max_term - restart_info["max_term_before"],
                "d_longest_write_stall_s": stall,
                "d_stall_starts_at": stall_at - t0,
            }
            if in_network and poller.catch_up_target is not None:
                result["restart"]["in_network"] = in_network_catch_up(
                    in_network, *poller.catch_up_target, partitioned=bool(partition)
                )
        result["_ops"] = ops
        return result
    finally:
        await cluster.shutdown()


def in_network_catch_up(
    samples: list[tuple[str, int, int, float, int, float]],
    node: str,
    target: int,
    *,
    partitioned: bool,
) -> dict[str, Any]:
    """Catch-up timed on the node's own container clock (see
    scripts/chaos_docker.sh): from the moment it was back -- its network
    interface reappearing (partition) or its server starting (restart) --
    to the first sample at or past `target`, the highest commit index any
    node reported when it came back."""
    mine = [x for x in samples if x[0] == node]
    back = None
    if partitioned:
        back = next((b[3] for a, b in zip(mine, mine[1:], strict=False) if not a[4] and b[4]), None)
    elif mine:
        back = mine[0][5]
    after = [x for x in mine if back is not None and x[3] >= back]
    done = next((x for x in after if x[1] >= target), None)
    return {
        "samples": len(mine),
        "back_at_container_clock": back,
        "e_restart_to_catch_up_s": round(done[3] - back, 3)
        if done is not None and back is not None
        else None,
        "first_sample_after_back_s": round(after[0][3] - back, 3)
        if after and back is not None
        else None,
        "term_while_out_max": max(
            (x[2] for x in mine if back is None or x[3] < back), default=None
        ),
        "term_first_after_back": after[0][2] if after else None,
        "trace": [
            (x[3], x[1], x[2], x[4])
            for i, x in enumerate(mine)
            if i == 0 or x[1:3] != mine[i - 1][1:3] or x[4] != mine[i - 1][4]
        ],
    }


def sync_arrivals(
    trace: list[list[float]], ops: list[OpResult], t0: float, end: float
) -> dict[str, Any]:
    """Line the leader's batches up against the clients' send times (both
    CLOCK_MONOTONIC, shared by every process on the host): for each sync
    that wrote client entries, how many operations clients *sent* while it
    ran, and how many submits the next batch then held."""
    sends = sorted(op.start for op in ops)
    rows = [r for r in trace if t0 <= r[0] <= end]  # (start, sync_start, sync_end, events, submits)
    per_sync: list[int] = []
    next_batch: list[int] = []
    for row, following in zip(rows, rows[1:], strict=False):
        _, sync_start, sync_end, _, submits = row
        if submits == 0:
            continue
        per_sync.append(bisect.bisect_left(sends, sync_end) - bisect.bisect_left(sends, sync_start))
        next_batch.append(int(following[4]))
    # Sent but not yet taken in by the driver, when each batch with a sync
    # of client entries starts its sync: what one sync *could* have covered.
    admitted = 0
    backlog: list[int] = []
    for row in rows:
        admitted += int(row[4])
        if row[4] > 0:
            backlog.append(bisect.bisect_left(sends, row[1]) - admitted)
    n = max(len(per_sync), 1)
    sync_ms = [1000 * (r[2] - r[1]) for r in rows if r[4] > 0]
    gaps = [1000 * (b[0] - a[2]) for a, b in zip(rows, rows[1:], strict=False)]
    return {
        "syncs_with_submits": len(per_sync),
        "sync_ms_mean": sum(sync_ms) / max(len(sync_ms), 1),
        "sends_during_sync_mean": sum(per_sync) / n,
        "sends_during_sync_histogram": {k: per_sync.count(k) for k in sorted(set(per_sync))},
        "syncs_with_a_send_during_them_percent": 100 * sum(x > 0 for x in per_sync) / n,
        "next_batch_submits_mean": sum(next_batch) / n,
        "next_batch_submits_when_sends_during_sync": (
            sum(b for a, b in zip(per_sync, next_batch, strict=True) if a > 0)
            / max(sum(a > 0 for a in per_sync), 1)
        ),
        "gap_between_batches_ms_mean": sum(gaps) / max(len(gaps), 1),
        "backlog_at_sync_start_mean": sum(backlog) / max(len(backlog), 1),
        "backlog_at_sync_start_histogram": {k: backlog.count(k) for k in sorted(set(backlog))},
    }


def fmt(value: float | None, scale: float = 1.0, digits: int = 1) -> str:
    return "-" if value is None else f"{value * scale:.{digits}f}"


def unknown_outcome_puts(ops: list[OpResult]) -> dict[str, int]:
    """Failed puts, split by whether a successful get later returned their
    value (unique per put): if one did, the write took effect."""
    read = {op.value for op in ops if op.kind == "get" and op.ok}
    failed = [op for op in ops if op.kind == "put" and not op.ok]
    observed = sum(1 for op in failed if op.value in read)
    return {"failed": len(failed), "observed": observed, "unobserved": len(failed) - observed}


def verified(v: dict[str, Any]) -> bool:
    return (
        v["converged"]
        and not v["prefix_problems"]
        and not v["ack_problems"]
        and v["linearizability"]["linearizable"]
    )


def report(name: str, result: dict[str, Any]) -> None:
    s = result["summary"]
    lat = s["latency_ms"]
    v = result["verification"]
    ok = verified(v)
    print(f"== {name}")
    disk = result.get("environment", {}).get("storage")
    if disk:
        print(f"   data on {disk['fs_type']} ({disk['data_dir']})")
    print(
        f"   {s['ops_per_sec']:.0f} ops/s ({s['ok']} ok, {s['failed']} failed)  "
        f"p50 {fmt(lat['p50'])} ms  p99 {fmt(lat['p99'])} ms  p99.9 {fmt(lat['p99.9'])} ms  "
        f"longest gap {fmt(s['longest_gap_s'], 1000, 0)} ms"
    )
    if "failover" in result:
        f = result["failover"]
        print(
            f"   failover: killed {f['killed']}, new leader {f['new_leader']}  "
            f"(a) {fmt(f['a_kill_to_new_leader_s'], 1000, 0)} ms  "
            f"(b) {fmt(f['b_kill_to_first_write_s'], 1000, 0)} ms"
        )
    if "partition" in result:
        p = result["partition"]
        print(
            f"   partition: {p['node']} isolated; leader {p['leader_at_down']} term "
            f"{p['leader_term_at_down']} at down -> {p['leader_at_end']} term "
            f"{p['leader_term_at_end']} at end; {p['node']} came back in term "
            f"{p.get('node_term_on_rejoin', '?')}; {p['leader_changes_after_rejoin']} leader "
            f"changes after rejoin"
        )
    if "restart" in result:
        r = result["restart"]
        catch_up = (
            f"caught up in {fmt(r['e_restart_to_catch_up_s'], 1000, 0)} ms"
            if r["caught_up"]
            else "NEVER caught up"
        )
        print(
            f"   restart: {catch_up}  (c) {r['c_elections_after_restart']} elections  "
            f"(d) longest write stall {fmt(r['d_longest_write_stall_s'], 1000, 0)} ms"
        )
        if "in_network" in r:
            n = r["in_network"]
            print(
                f"   on the node's own clock: caught up in "
                f"{fmt(n['e_restart_to_catch_up_s'], 1000, 0)} ms (first sample "
                f"{fmt(n['first_sample_after_back_s'], 1000, 0)} ms after it was back); "
                f"its term while out {n['term_while_out_max']}, once back "
                f"{n['term_first_after_back']}"
            )
    res = result.get("resources")
    if res and "cpu_percent" in res:
        cpu = "  ".join(f"{n} {c:.0f}%" for n, c in sorted(res["cpu_percent"].items()))
        print(f"   cpu: {cpu}  (leader {res['leader']})")
        threads = "  ".join(
            f"{t} {c:.0f}%" for t, c in res["leader_thread_cpu_percent"].items() if c >= 1
        )
        print(f"   leader threads: {threads}")
        d = res["leader_driver"]
        print(
            f"   leader batches: {d['events_per_batch']:.2f} events each; "
            f"{d['submits_per_batch_with_any']:.2f} submits per batch holding any; "
            f"{d['submits_during_sync']} submits queued during a sync; "
            f"sync {d['sync_percent_of_wall']:.0f}% of wall"
        )
        if "sync_arrivals" in res:
            a = res["sync_arrivals"]
            print(
                f"   per sync ({a['syncs_with_submits']}, {a['sync_ms_mean']:.2f} ms mean): "
                f"{a['sends_during_sync_mean']:.2f} client sends during it "
                f"({a['syncs_with_a_send_during_them_percent']:.0f}% had any); next batch held "
                f"{a['next_batch_submits_mean']:.2f} submits "
                f"({a['next_batch_submits_when_sends_during_sync']:.2f} after a sync with sends); "
                f"sent but not yet admitted at sync start: {a['backlog_at_sync_start_mean']:.2f}"
            )
        print(
            f"   leader log fsyncs: {res['leader_log_fsyncs_per_s']:.0f}/s, "
            f"{res['leader_entries_per_fsync']:.1f} entries each, "
            f"{res['leader_fsync_ms_mean']:.2f} ms mean, "
            f"{res['leader_fsync_percent_of_wall']:.0f}% of wall"
        )
        print(
            f"   leader per committed entry: {res['append_entries_per_entry']:.2f} AppendEntries "
            f"({res['append_entries_with_entries_per_entry']:.2f} carrying entries), "
            f"{res['entries_sent_per_entry']:.2f} entries sent, "
            f"{res['bytes_encoded_per_entry']:.0f} bytes encoded"
        )
    for b in (res or {}).get("timeline", []):
        print(
            f"   t={b['t_end_s']:6.1f}s  log {b['log_length']:7d}  "
            f"{b['committed_per_s']:6.0f} entries/s  leader cpu {b['leader_cpu_percent']:4.0f}%  "
            f"load_log {b['load_log_percent_of_wall']:4.1f}% of wall, "
            f"{b['entries_copied_per_entry']:8.0f} entries copied per entry"
        )
    print(
        f"   verify: {'PASS' if ok else 'FAIL'} (converged={v['converged']}, "
        f"{len(v['prefix_problems'])} prefix problems, {len(v['ack_problems'])} missing of "
        f"{v['acknowledged_puts']} acknowledged puts)"
    )
    lin = v["linearizability"]
    first, *rest = lin["summary"].splitlines()
    unknown = lin["unknown_outcome_puts"]
    print(
        f"   {first}  (checked in {lin['seconds']:.2f} s; {unknown['failed']} failed puts: "
        f"{unknown['observed']} observed by a read, {unknown['unobserved']} not)"
    )
    for line in rest:
        print(f"   {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--nodes", type=int, default=5)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fsync", choices=["per_write", "deferred"], default="deferred")
    parser.add_argument("--kill-at", type=float, help="seconds into the load")
    parser.add_argument("--restart-at", type=float, help="seconds into the load")
    parser.add_argument("--put-ratio", type=float, default=0.5)
    parser.add_argument("--keys", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--op-timeout",
        type=float,
        default=30.0,
        help="each client's per-operation budget, seconds; shorter makes ops fail in failover",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--name", help="result file stem (default: derived from options)")
    parser.add_argument("--keep", action="store_true", help="keep data dirs and server logs")
    parser.add_argument(
        "--trace-batches",
        type=int,
        default=0,
        metavar="N",
        help="servers record their last N driver batches; analysed against client sends",
    )
    parser.add_argument(
        "--tmpfs",
        action="store_true",
        help=f"data dirs in {TMPFS_ROOT} (fsync is free) instead of {DISK_ROOT}",
    )
    parser.add_argument("--profile", action="store_true", help="run each server under py-spy")
    parser.add_argument(
        "--stats-every", type=float, default=0.0, help="leader stats bucket, seconds (0: off)"
    )
    parser.add_argument("--profile-native", action="store_true", help="py-spy --native too")
    parser.add_argument(
        "--external", type=Path, help="JSON {node: host:port} of a running cluster; start nothing"
    )
    parser.add_argument("--chaos", help="with --external: fault command (down/up lines)")
    parser.add_argument("--chaos-at", type=float, default=10.0, help="seconds into the load")
    args = parser.parse_args()
    if args.external is not None and (args.kill_at is not None or args.profile):
        parser.error("--external runs no processes of its own: use --chaos, not --kill-at")
    if args.chaos is not None and args.external is None:
        parser.error("--chaos needs --external")
    if args.restart_at is not None and (args.kill_at is None or args.restart_at <= args.kill_at):
        parser.error("--restart-at needs a --kill-at before it")

    name = args.name or (
        f"{time.strftime('%Y%m%d-%H%M%S')}-c{args.clients}-{args.fsync}"
        + (f"-kill{args.kill_at:g}" if args.kill_at is not None else "")
        + (f"-restart{args.restart_at:g}" if args.restart_at is not None else "")
        + ("-external" if args.external is not None else "")
        + ("-tmpfs" if args.tmpfs else "")
    )
    root = TMPFS_ROOT if args.tmpfs else DISK_ROOT
    root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="raft-bench-", dir=root))
    if args.external is not None:
        storage = docker_data_filesystem()
    else:
        storage = {"data_dir": str(workdir), **filesystem_of(workdir)}
    if args.tmpfs and storage.get("fs_type") != "tmpfs":
        parser.error(f"--tmpfs: {root} is {storage.get('fs_type')}, not tmpfs")
    try:
        result = asyncio.run(bench(args, workdir))
    except BaseException:
        print(f"bench failed; server logs and data are in {workdir}", file=sys.stderr)
        raise
    ops: list[OpResult] = result.pop("_ops")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history = args.out_dir / f"{name}.history.jsonl"
    write_history(ops, history)
    output = {
        "name": name,
        "environment": {
            "cpu_count": os.cpu_count(),
            "os": platform.platform(),
            "python": sys.version,
            "git": git_revision(),
            "storage": storage,
        },
        "options": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        **result,
        "history": history.name,
    }
    (args.out_dir / f"{name}.json").write_text(json.dumps(output, indent=2) + "\n")
    report(name, output)
    if (workdir / "batches.json").exists():
        shutil.copy(workdir / "batches.json", args.out_dir / f"{name}.batches.json")
    for raw in sorted(workdir.glob("*.raw")):
        shutil.copy(raw, args.out_dir / f"{name}.{raw.name}")

    if args.keep:
        print(f"   kept {workdir}")
    else:
        shutil.rmtree(workdir)
    v = output["verification"]
    return 0 if verified(v) else 1


if __name__ == "__main__":
    sys.exit(main())
