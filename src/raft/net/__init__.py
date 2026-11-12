"""Running `raft.node.RaftNode` as real processes over gRPC.

`raft.sim.cluster.Cluster` is the first driver for the sans-IO
`RaftNode`; this package is the second. Nothing here changes Raft logic:
it translates between protobuf and `raft.messages` (`codec`), persists
durable state to disk (`filestorage`), and moves `(dst, message)` pairs
between processes.

Requires the `[net]` extra. Nothing outside this package imports it, so
the core package and the simulator stay dependency-free.
"""
