#!/usr/bin/env bash
# Fault injection against the docker-compose cluster (docker-compose.yml).
#
#   scripts/chaos_docker.sh kill-leader SECONDS
#       Find the leader via Status, `docker kill` it (SIGKILL), wait
#       SECONDS, `docker start` it again.
#   scripts/chaos_docker.sh partition SECONDS
#       Pick a follower, `docker network disconnect` it from the cluster
#       network (no traffic in or out, published port included), wait
#       SECONDS, reconnect it at the same IP and name.
#
# Prints one line per event on stdout, for cluster_bench.py --chaos:
#   leader <node> <term>     who led when the fault was chosen
#   begin down <node>        about to run the docker command that injects it
#   down <node>              that command returned: the fault is in effect
#   begin up <node>          about to restart or reconnect the node
#   up <node>                that command returned: the node is back
# `docker kill` takes ~300 ms to return, and the signal lands somewhere in
# that window, so measure from the "begin" line: an upper bound.
# Diagnostics go to stderr. Needs the project venv (PYTHON, default
# .venv/bin/python) to call Status; the host ports are docker/hosts.json.
set -euo pipefail

usage() { echo "usage: $0 {kill-leader|partition} SECONDS" >&2; exit 2; }
[[ $# -eq 2 ]] || usage
mode=$1 seconds=$2
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-.venv/bin/python}
NETWORK=${RAFT_NETWORK:-raft}

# "<leader> <term> <follower> <follower> ..." from the nodes that answer
# Status; the leader is the one with the highest term claiming the role.
# Retries until there is exactly one.
roles() {
    "$PYTHON" - <<'PY'
import asyncio, json, sys, time
import grpc
from raft.net import raft_pb2 as pb
from raft.net.raft_pb2_grpc import RaftStub

hosts = json.load(open("docker/hosts.json"))

async def status(addr):
    try:
        async with grpc.aio.insecure_channel(addr) as ch:
            return await RaftStub(ch).Status(pb.StatusRequest(), timeout=0.5)
    except grpc.aio.AioRpcError:
        return None

async def main():
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        replies = await asyncio.gather(*(status(a) for a in hosts.values()))
        seen = {n: s for n, s in zip(hosts, replies) if s is not None}
        leaders = [s for s in seen.values() if s.role == "leader"]
        if leaders:
            top = max(leaders, key=lambda s: s.current_term)
            if sum(s.current_term == top.current_term for s in leaders) == 1:
                followers = [n for n in seen if n != top.node_id]
                print(top.node_id, top.current_term, *followers)
                return 0
        await asyncio.sleep(0.05)
    print("no leader within 30 s", file=sys.stderr)
    return 1

sys.exit(asyncio.run(main()))
PY
}

ip_of() {  # the node's address on the cluster network, from compose
    docker inspect -f "{{(index .NetworkSettings.Networks \"$NETWORK\").IPAddress}}" "raft-$1"
}

read -r leader term followers <<<"$(roles)"
echo "leader $leader $term"

case $mode in
kill-leader)
    echo "begin down $leader"
    docker kill "raft-$leader" >/dev/null
    echo "down $leader"
    sleep "$seconds"
    echo "begin up $leader"
    docker start "raft-$leader" >/dev/null
    echo "up $leader"
    ;;
partition)
    read -ra candidates <<<"$followers"
    victim=${candidates[RANDOM % ${#candidates[@]}]}
    ip=$(ip_of "$victim")
    echo "begin down $victim"
    docker network disconnect "$NETWORK" "raft-$victim"
    echo "down $victim"
    sleep "$seconds"
    echo "begin up $victim"
    docker network connect --ip "$ip" --alias "$victim" "$NETWORK" "raft-$victim"
    echo "up $victim"
    ;;
*) usage ;;
esac
