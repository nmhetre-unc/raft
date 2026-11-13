#!/usr/bin/env sh
# Regenerate src/raft/net/raft_pb2{.py,.pyi,_grpc.py} from proto/raft.proto.
#
# grpc_tools emits `import raft_pb2 as raft__pb2` in the _grpc module,
# which only resolves if src/raft/net is itself on sys.path. Rewriting it
# to a package-relative import makes it resolve as raft.net.raft_pb2 like
# everything else. The sed writes to a temp file rather than using -i,
# whose syntax differs between GNU and BSD sed.
set -eu

cd "$(dirname "$0")/.."

out=src/raft/net
python -m grpc_tools.protoc \
    -I proto \
    --python_out="$out" \
    --pyi_out="$out" \
    --grpc_python_out="$out" \
    proto/raft.proto

grpc_module="$out/raft_pb2_grpc.py"
sed 's/^import raft_pb2 as raft__pb2$/from . import raft_pb2 as raft__pb2/' \
    "$grpc_module" > "$grpc_module.tmp"
mv "$grpc_module.tmp" "$grpc_module"

grep -q '^from \. import raft_pb2 as raft__pb2$' "$grpc_module" || {
    echo "gen_proto.sh: import rewrite did not apply; check grpc_tools output" >&2
    exit 1
}
