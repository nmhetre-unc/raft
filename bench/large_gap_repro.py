"""Reproduce (or rule out) a follower that can never catch up because the
catch-up AppendEntries is larger than gRPC will accept.

    python bench/large_gap_repro.py --entries 1500 --value-bytes 4096

Starts `--nodes` server processes, waits for a leader, SIGKILLs one
follower, commits `--entries` puts of `--value-bytes` each through the
leader, then restarts the follower on its data directory and polls its
commit index every 50 ms until it reaches the leader's commit index at
the moment of the restart, or `--timeout` seconds pass.

It also reports the exact encoded size of the AppendEntries the leader
must send to close the gap -- every entry the follower is missing, as an
`Envelope`, which is what `GrpcTransport` puts on the wire -- against
gRPC's default 4 MiB receive limit (neither our server nor our transport
overrides it).

Every put and, after catch-up, one read of every key are recorded and the
history is checked for linearizability; a violation makes the exit status
1. Writes ``<name>.json`` and ``<name>.history.jsonl`` to `--out-dir`.
Not part of the test suite.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import grpc

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cluster_bench import (  # noqa: E402
    DISK_ROOT,
    POLL_CHANNEL_OPTIONS,
    Cluster,
    filesystem_of,
    git_revision,
    wait_for_leader,
)

from raft.messages import AppendEntries  # noqa: E402
from raft.net import raft_pb2 as pb  # noqa: E402
from raft.net.client import RaftClient  # noqa: E402
from raft.net.codec import decode_log_entry, encode_message  # noqa: E402
from raft.net.linearizability import check as check_linearizability  # noqa: E402
from raft.net.loadgen import OpKind, OpResult, write_history  # noqa: E402
from raft.net.raft_pb2_grpc import RaftStub  # noqa: E402

GRPC_DEFAULT_MAX_RECEIVE = 4 * 1024 * 1024
POLL_INTERVAL = 0.05


async def fetch_log(addr: str) -> pb.StatusReply:
    """Status with the whole log, on a channel allowed to receive it."""
    options = [*POLL_CHANNEL_OPTIONS, ("grpc.max_receive_message_length", -1)]
    async with grpc.aio.insecure_channel(addr, options=options) as channel:
        reply: pb.StatusReply = await RaftStub(channel).Status(
            pb.StatusRequest(include_log=True), timeout=30.0
        )
        return reply


async def timed(
    history: list[OpResult], client: RaftClient, kind: OpKind, key: str, value: str | None
) -> None:
    """Run one put or get and record it, loadgen-style, for the checker."""
    start = time.monotonic()
    error = None
    try:
        if kind == "put":
            assert value is not None
            await client.put(key, value)
        else:
            value = await client.get(key)
    except Exception as exc:  # recorded, not raised
        error = f"{type(exc).__name__}: {exc}"
        value = value if kind == "put" else None
    history.append(
        OpResult(
            client.client_id, kind, key, value, start, time.monotonic(), error is None,
            client.leader, error,
        )
    )  # fmt: skip


async def write(
    members: dict[str, str], entries: int, value_bytes: int, clients: int, history: list[OpResult]
) -> float:
    remaining = iter(range(entries))
    start = time.monotonic()

    async def worker(n: int) -> None:
        client = RaftClient(members, client_id=f"w{n}")
        try:
            for i in remaining:
                tag = f"w{n}:{i}:"  # every value unique, as the checker requires
                value = tag + "x" * max(0, value_bytes - len(tag))
                await timed(history, client, "put", f"k{i % 64}", value)
        finally:
            await client.close()

    await asyncio.gather(*(worker(n) for n in range(clients)))
    return time.monotonic() - start


async def read_back(members: dict[str, str], history: list[OpResult]) -> None:
    """One read of every key, so the history has reads to check."""
    client = RaftClient(members, client_id="reader")
    try:
        for k in range(64):
            await timed(history, client, "get", f"k{k}", None)
    finally:
        await client.close()


async def run(args: argparse.Namespace, workdir: Path) -> dict[str, Any]:
    cluster = Cluster(args.nodes, "deferred", workdir)
    try:
        for node in cluster.nodes:
            cluster.start(node)
        leader = await wait_for_leader(cluster)
        follower = next(n for n in cluster.nodes if n != leader)
        before = await cluster.status(follower)
        assert before is not None
        follower_last = before.commit_index  # all it can be sure to hold
        cluster.kill(follower)

        live = {n: a for n, a in cluster.nodes.items() if n != follower}
        history: list[OpResult] = []
        write_s = await write(live, args.entries, args.value_bytes, args.clients, history)

        reply = await fetch_log(cluster.nodes[leader])
        assert reply.role == "leader", f"{leader} lost leadership during the writes"
        log = [decode_log_entry(e) for e in reply.log]
        target = reply.commit_index
        missing = tuple(log[follower_last:])
        catch_up = AppendEntries(
            term=reply.current_term,
            leader_id=leader,
            prev_log_index=follower_last,
            prev_log_term=log[follower_last - 1].term if follower_last else 0,
            entries=missing,
            leader_commit=target,
        )
        size = encode_message(catch_up, leader).ByteSize()

        restarted = time.monotonic()
        cluster.start(follower)
        trace: list[tuple[float, int | None, int | None]] = []
        caught_up_s = None
        while time.monotonic() - restarted < args.timeout:
            s = await cluster.status(follower, timeout=0.5)
            t = round(time.monotonic() - restarted, 3)
            point = (t, s.commit_index if s else None, s.current_term if s else None)
            if not trace or trace[-1][1:] != point[1:]:
                trace.append(point)
            if s is not None and s.commit_index >= target:
                caught_up_s = t
                break
            await asyncio.sleep(POLL_INTERVAL)
        await read_back(cluster.nodes, history)
        linear = check_linearizability(history)
        final = await cluster.statuses()
        return {
            "linearizability": {
                "linearizable": linear.ok,
                "seconds": round(linear.seconds, 3),
                "summary": linear.summary(),
            },
            "_history": history,
            "args": vars(args),
            "git": git_revision(),
            "storage": {"data_dir": str(workdir), **filesystem_of(workdir)},
            "leader": leader,
            "follower": follower,
            "follower_commit_at_kill": follower_last,
            "entries_missing": len(missing),
            "catch_up_append_entries_bytes": size,
            "grpc_max_receive_bytes": GRPC_DEFAULT_MAX_RECEIVE,
            "exceeds_limit": size > GRPC_DEFAULT_MAX_RECEIVE,
            "write_s": round(write_s, 2),
            "target_commit": target,
            "caught_up": caught_up_s is not None,
            "restart_to_catch_up_s": caught_up_s,
            "follower_trace": trace,
            "final": {
                n: {"role": s.role, "term": s.current_term, "commit": s.commit_index}
                for n, s in sorted(final.items())
            },
        }
    finally:
        await cluster.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--nodes", type=int, default=5)
    parser.add_argument("--entries", type=int, default=1500)
    parser.add_argument("--value-bytes", type=int, default=4096)
    parser.add_argument("--clients", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--name", default="large-gap")
    parser.add_argument("--out-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--keep", action="store_true", help="keep the data directory")
    args = parser.parse_args()

    DISK_ROOT.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="raft-gap-", dir=DISK_ROOT))  # a real disk
    result = asyncio.run(run(args, workdir))
    history = result.pop("_history")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_history(history, args.out_dir / f"{args.name}.history.jsonl")
    (args.out_dir / f"{args.name}.json").write_text(json.dumps(result, indent=2, default=str))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "follower_trace"}, indent=2, default=str
        )
    )
    print(
        "trace (t_s, commit, term):",
        result["follower_trace"][:10],
        "...",
        result["follower_trace"][-3:],
    )
    if not args.keep:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print("data in", workdir)
    lin = result["linearizability"]
    first, *rest = lin["summary"].splitlines()
    print("\n".join([f"{first}  (checked in {lin['seconds']:.2f} s)", *rest]))
    return 0 if lin["linearizable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
