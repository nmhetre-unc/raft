"""Translation between the generated protobuf types and this project's own.

`RaftNode` speaks `raft.messages` dataclasses, `raft.storage.LogEntry`,
and `raft.statemachine` commands; gRPC speaks `raft.net.raft_pb2`. This
module is the only place the two meet, so nothing else in `raft.net`
ever touches a protobuf field directly and nothing in `RaftNode` ever
learns protobuf exists. Every function is a field-for-field copy --
`proto/raft.proto` uses the dataclasses' own field names for exactly
that reason.

Encoding is strict where the dataclasses are loose. `Put.value` and
`LogEntry.command` are typed `object`/`Any`, because the simulator
happily replicates ints and opaque placeholder strings; the wire format
can't, and silently stringifying an int would hand a client back
something other than what it wrote. So an unrepresentable value raises
`TypeError` at encode time, on the sending side, instead of arriving as
something subtly different on the other.

Decoding is strict too: a proto3 message with a required `oneof` left
unset (an `Envelope` with no message, a `Command` with no op) raises
`CodecError` rather than decoding to a default. proto3 can't express
"required", so this is the only place that rule can live.
"""

from __future__ import annotations

from raft.messages import (
    AppendEntries,
    AppendEntriesReply,
    Message,
    RequestVote,
    RequestVoteReply,
)
from raft.net import raft_pb2 as pb
from raft.statemachine import NOT_FOUND, ClientRequest, Command, Delete, Get, Put
from raft.storage import LogEntry


class CodecError(ValueError):
    """A protobuf message that doesn't decode to any valid value."""


# -- Commands and log entries --


def encode_command(command: Command) -> pb.Command:
    if isinstance(command, Put):
        if not isinstance(command.value, str):
            raise TypeError(f"Put.value must be a str on the wire, got {command.value!r}")
        return pb.Command(put=pb.Put(key=command.key, value=command.value))
    if isinstance(command, Get):
        return pb.Command(get=pb.Get(key=command.key))
    if isinstance(command, Delete):
        return pb.Command(delete=pb.Delete(key=command.key))
    raise TypeError(f"not a Command: {command!r}")


def decode_command(msg: pb.Command) -> Command:
    op = msg.WhichOneof("op")
    if op == "put":
        return Put(key=msg.put.key, value=msg.put.value)
    if op == "get":
        return Get(key=msg.get.key)
    if op == "delete":
        return Delete(key=msg.delete.key)
    raise CodecError("Command has no op set")


def encode_client_request(request: ClientRequest) -> pb.ClientRequest:
    return pb.ClientRequest(
        client_id=request.client_id,
        serial_number=request.serial_number,
        command=encode_command(request.command),
    )


def decode_client_request(msg: pb.ClientRequest) -> ClientRequest:
    if not msg.HasField("command"):
        raise CodecError("ClientRequest has no command")
    return ClientRequest(
        client_id=msg.client_id,
        serial_number=msg.serial_number,
        command=decode_command(msg.command),
    )


def encode_log_entry(entry: LogEntry) -> pb.LogEntry:
    """`command=None` (the sentinel, or an election no-op) encodes as an
    entry with no payload set; anything else must be a `ClientRequest`
    or a bare `Command` -- the only two shapes `RaftNode` applies."""
    command = entry.command
    if command is None:
        return pb.LogEntry(term=entry.term)
    if isinstance(command, ClientRequest):
        return pb.LogEntry(term=entry.term, client_request=encode_client_request(command))
    if isinstance(command, Put | Get | Delete):
        return pb.LogEntry(term=entry.term, command=encode_command(command))
    raise TypeError(f"log entry command has no wire form: {command!r}")


def decode_log_entry(msg: pb.LogEntry) -> LogEntry:
    payload = msg.WhichOneof("payload")
    if payload == "client_request":
        return LogEntry(term=msg.term, command=decode_client_request(msg.client_request))
    if payload == "command":
        return LogEntry(term=msg.term, command=decode_command(msg.command))
    return LogEntry(term=msg.term, command=None)


# -- Raft RPCs --


def encode_message(msg: Message, src: str) -> pb.Envelope:
    """Wrap `msg` for the wire, stamped with the sending node's id.

    `src` rides in the envelope, not the message, because `RaftNode.handle`
    takes it as a separate argument -- the dataclasses themselves don't
    carry a sender, and adding one would mean changing `raft.messages`.
    """
    if isinstance(msg, RequestVote):
        return pb.Envelope(
            src=src,
            request_vote=pb.RequestVote(
                term=msg.term,
                candidate_id=msg.candidate_id,
                last_log_index=msg.last_log_index,
                last_log_term=msg.last_log_term,
            ),
        )
    if isinstance(msg, RequestVoteReply):
        return pb.Envelope(
            src=src,
            request_vote_reply=pb.RequestVoteReply(term=msg.term, vote_granted=msg.vote_granted),
        )
    if isinstance(msg, AppendEntries):
        return pb.Envelope(
            src=src,
            append_entries=pb.AppendEntries(
                term=msg.term,
                leader_id=msg.leader_id,
                prev_log_index=msg.prev_log_index,
                prev_log_term=msg.prev_log_term,
                entries=[encode_log_entry(entry) for entry in msg.entries],
                leader_commit=msg.leader_commit,
            ),
        )
    if isinstance(msg, AppendEntriesReply):
        return pb.Envelope(
            src=src,
            append_entries_reply=pb.AppendEntriesReply(
                term=msg.term,
                success=msg.success,
                match_index=msg.match_index,
                conflict_index=msg.conflict_index,
                conflict_term=msg.conflict_term,
            ),
        )
    raise TypeError(f"not a Message: {msg!r}")


def decode_envelope(env: pb.Envelope) -> tuple[str, Message]:
    """Unwrap an `Envelope` into `(src, message)`, ready for `RaftNode.handle`."""
    kind = env.WhichOneof("message")
    msg: Message
    if kind == "request_vote":
        rv = env.request_vote
        msg = RequestVote(
            term=rv.term,
            candidate_id=rv.candidate_id,
            last_log_index=rv.last_log_index,
            last_log_term=rv.last_log_term,
        )
    elif kind == "request_vote_reply":
        rvr = env.request_vote_reply
        msg = RequestVoteReply(term=rvr.term, vote_granted=rvr.vote_granted)
    elif kind == "append_entries":
        ae = env.append_entries
        msg = AppendEntries(
            term=ae.term,
            leader_id=ae.leader_id,
            prev_log_index=ae.prev_log_index,
            prev_log_term=ae.prev_log_term,
            entries=tuple(decode_log_entry(entry) for entry in ae.entries),
            leader_commit=ae.leader_commit,
        )
    elif kind == "append_entries_reply":
        aer = env.append_entries_reply
        msg = AppendEntriesReply(
            term=aer.term,
            success=aer.success,
            match_index=aer.match_index,
            conflict_index=aer.conflict_index,
            conflict_term=aer.conflict_term,
        )
    else:
        raise CodecError("Envelope has no message set")
    return env.src, msg


# -- Client results --


def encode_result(value: object) -> pb.Result:
    """Encode what `KVStateMachine.apply` returned: `None` (a `Put`),
    `NOT_FOUND`, or a stored value -- always a `str` on this path, since
    `encode_command` only ever lets a `str` into the log."""
    if value is None:
        return pb.Result(none=pb.Unit())
    if value is NOT_FOUND:
        return pb.Result(not_found=pb.Unit())
    if isinstance(value, str):
        return pb.Result(value=value)
    raise TypeError(f"result has no wire form: {value!r}")


def decode_result(msg: pb.Result) -> object:
    kind = msg.WhichOneof("kind")
    if kind == "none":
        return None
    if kind == "not_found":
        return NOT_FOUND
    if kind == "value":
        return msg.value
    raise CodecError("Result has no value set")
