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
#   commit <node> <commit> <term> <t> <net> <started>
#                            the node's Status, polled every 50 ms from
#                            *inside its container* while it is out and for
#                            WATCH seconds (default 8) after it is back.
#                            <t> is the container's CLOCK_MONOTONIC, <net>
#                            1 if it has a network interface besides lo, and
#                            <started> when its server (PID 1) started, on
#                            that same clock. Time catch-up with these, not
#                            with when the lines arrive: Docker Desktop holds
#                            back exec output (and published ports) for
#                            seconds after a network connect.
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

WATCH=${WATCH:-8}

# watch_node NODE SECONDS: stream "commit <node> <commit> <term>" from
# inside the node's container for that long.
watch_node() {
    docker exec "raft-$1" python -u -c "
import os, time, grpc
from raft.net import raft_pb2 as pb
from raft.net.raft_pb2_grpc import RaftStub
started = int(open('/proc/1/stat').read().rsplit(')', 1)[1].split()[19]) / os.sysconf('SC_CLK_TCK')
stub = RaftStub(grpc.insecure_channel('127.0.0.1:7000'))
end = time.monotonic() + $2
while time.monotonic() < end:
    net = int(any(i != 'lo' for i in os.listdir('/sys/class/net')))
    try:
        r = stub.Status(pb.StatusRequest(), timeout=0.2)
        print('commit', '$1', r.commit_index, r.current_term, round(time.monotonic(), 3), net,
              round(started, 2), flush=True)
    except grpc.RpcError:
        pass
    time.sleep(0.05)
" || true
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
    watch_node "$leader" "$WATCH"
    ;;
partition)
    read -ra candidates <<<"$followers"
    victim=${candidates[RANDOM % ${#candidates[@]}]}
    ip=$(ip_of "$victim")
    echo "begin down $victim"
    docker network disconnect "$NETWORK" "raft-$victim"
    echo "down $victim"
    # Started now (the container is still up), so it is already polling
    # when the partition heals.
    watch_node "$victim" "$(( seconds + WATCH ))" &
    watcher=$!
    sleep "$seconds"
    echo "begin up $victim"
    docker network connect --ip "$ip" --alias "$victim" "$NETWORK" "raft-$victim"
    echo "up $victim"
    wait "$watcher"
    ;;
*) usage ;;
esac
