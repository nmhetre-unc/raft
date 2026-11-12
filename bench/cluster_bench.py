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
fetches every log, and runs both `raft.net.verify` checks. It prints a
report and writes ``<name>.json`` (environment, options, summary,
failover and restart metrics, events, verification) and
``<name>.history.jsonl`` (every operation) to `--out-dir`.

All times in the output are seconds since the load started.

Failover is reported as (a) kill -> first observation of a new leader
(50 ms polling resolution) and (b) kill -> first successful client write
answered by a node other than the killed one. Restart disruption is
(c) elections triggered after the restart -- the rise in the highest
term any node reports -- and (d) the longest interval after the restart
with no write completing.

Not part of the test suite or coverage; ruff and mypy clean.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import os
import platform
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


class Cluster:
    def __init__(self, nodes: int, fsync: str, workdir: Path) -> None:
        self.fsync = fsync
        self.workdir = workdir
        self.nodes = {f"n{i}": f"127.0.0.1:{port}" for i, port in enumerate(free_ports(nodes))}
        self.config = workdir / "cluster.json"
        self.config.write_text(json.dumps({"nodes": self.nodes}))
        self.procs: dict[str, subprocess.Popen[bytes]] = {}
        self._logs: list[IO[bytes]] = []
        self._channels = {i: grpc.aio.insecure_channel(a) for i, a in self.nodes.items()}

    def start(self, node: str) -> None:
        log = (self.workdir / f"{node}.log").open("ab")
        self._logs.append(log)
        self.procs[node] = subprocess.Popen(
            [
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
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
        )

    def kill(self, node: str) -> None:
        self.procs[node].kill()
        self.procs[node].wait()

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
        for proc in self.procs.values():
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in self.procs.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for log in self._logs:
            log.close()


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

    def _record(self, **event: Any) -> None:
        self.events.append({"t": round(time.monotonic() - self.t0, 4), **event})

    async def poll_once(self) -> None:
        statuses = await self.cluster.statuses(timeout=POLL_INTERVAL * 4)
        for node, status in statuses.items():
            if self.terms.get(node) != status.current_term:
                self.terms[node] = status.current_term
                self._record(kind="term", node=node, term=status.current_term)
        self.max_term = max([self.max_term, *self.terms.values()])
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
    cluster = Cluster(args.nodes, args.fsync, workdir)
    try:
        for node in cluster.nodes:
            cluster.start(node)
        await wait_for_leader(cluster)

        t0 = time.monotonic()
        poller = Poller(cluster, t0)
        await poller.poll_once()
        poll_task = asyncio.create_task(poller.run())

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
            cluster.start(fault.node)
            poller._record(kind="restart", node=fault.node)

        fault_tasks = []
        if args.kill_at is not None:
            fault_tasks.append(asyncio.create_task(kill_leader()))
        if args.restart_at is not None:
            fault_tasks.append(asyncio.create_task(restart_killed()))

        def make_client(n: int) -> RaftClient:
            return RaftClient(
                cluster.nodes, client_id=f"c{n:02d}", rpc_timeout=2.0, op_timeout=30.0
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

        summary = summarize(ops, start=t0, end=load_end, faults=faults)
        result: dict[str, Any] = {
            "summary": dataclasses.asdict(summary),
            "verification": verification,
            "events": poller.events,
        }
        if faults:
            (kill,) = faults
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
        if restart_info:
            restart_at = restart_info["at"]
            put_ends = [op.end for op in ops if op.ok and op.kind == "put"]
            stall, stall_at = longest_gap(put_ends, restart_at, load_end)
            result["restart"] = {
                "restart_at": restart_at - t0,
                "c_elections_after_restart": poller.max_term - restart_info["max_term_before"],
                "d_longest_write_stall_s": stall,
                "d_stall_starts_at": stall_at - t0,
            }
        result["_ops"] = ops
        return result
    finally:
        await cluster.shutdown()


def fmt(value: float | None, scale: float = 1.0, digits: int = 1) -> str:
    return "-" if value is None else f"{value * scale:.{digits}f}"


def report(name: str, result: dict[str, Any]) -> None:
    s = result["summary"]
    lat = s["latency_ms"]
    v = result["verification"]
    ok = not v["prefix_problems"] and not v["ack_problems"] and v["converged"]
    print(f"== {name}")
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
    if "restart" in result:
        r = result["restart"]
        print(
            f"   restart: (c) {r['c_elections_after_restart']} elections  "
            f"(d) longest write stall {fmt(r['d_longest_write_stall_s'], 1000, 0)} ms"
        )
    print(
        f"   verify: {'PASS' if ok else 'FAIL'} (converged={v['converged']}, "
        f"{len(v['prefix_problems'])} prefix problems, {len(v['ack_problems'])} missing of "
        f"{v['acknowledged_puts']} acknowledged puts)"
    )


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
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--name", help="result file stem (default: derived from options)")
    parser.add_argument("--keep", action="store_true", help="keep data dirs and server logs")
    args = parser.parse_args()
    if args.restart_at is not None and (args.kill_at is None or args.restart_at <= args.kill_at):
        parser.error("--restart-at needs a --kill-at before it")

    name = args.name or (
        f"{time.strftime('%Y%m%d-%H%M%S')}-c{args.clients}-{args.fsync}"
        + (f"-kill{args.kill_at:g}" if args.kill_at is not None else "")
        + (f"-restart{args.restart_at:g}" if args.restart_at is not None else "")
    )
    workdir = Path(tempfile.mkdtemp(prefix="raft-bench-"))
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
        },
        "options": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        **result,
        "history": history.name,
    }
    (args.out_dir / f"{name}.json").write_text(json.dumps(output, indent=2) + "\n")
    report(name, output)

    if args.keep:
        print(f"   kept {workdir}")
    else:
        shutil.rmtree(workdir)
    v = output["verification"]
    return 0 if v["converged"] and not v["prefix_problems"] and not v["ack_problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
