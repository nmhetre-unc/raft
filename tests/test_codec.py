"""Tests for `raft.net.codec`: protobuf <-> dataclass translation.

Round trips go all the way through bytes (`SerializeToString` then
`FromString`), not just dataclass -> protobuf object -> dataclass, so a
field the codec sets but proto3 doesn't actually put on the wire (a
default value, an unset `oneof`) would be caught here rather than first
showing up between two processes.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from raft.messages import (
    AppendEntries,
    AppendEntriesReply,
    Message,
    PreVote,
    PreVoteReply,
    RequestVote,
    RequestVoteReply,
)
from raft.net import raft_pb2 as pb
from raft.net.codec import (
    CodecError,
    decode_client_request,
    decode_command,
    decode_envelope,
    decode_log_entry,
    decode_result,
    encode_client_request,
    encode_command,
    encode_log_entry,
    encode_message,
    encode_result,
)
from raft.statemachine import NOT_FOUND, ClientRequest, Command, Delete, Get, Put
from raft.storage import SENTINEL, LogEntry

# protobuf strings must be valid UTF-8, which rules out lone surrogates.
texts = st.text(alphabet=st.characters(exclude_categories=["Cs"]), max_size=20)
ints = st.integers(min_value=0, max_value=2**63 - 1)

commands: st.SearchStrategy[Command] = st.one_of(
    st.builds(Put, key=texts, value=texts),
    st.builds(Get, key=texts),
    st.builds(Delete, key=texts),
)
client_requests = st.builds(ClientRequest, client_id=texts, serial_number=ints, command=commands)
log_entries = st.builds(
    LogEntry, term=ints, command=st.one_of(st.none(), commands, client_requests)
)
messages: st.SearchStrategy[Message] = st.one_of(
    st.builds(
        RequestVote,
        term=ints,
        candidate_id=texts,
        last_log_index=ints,
        last_log_term=ints,
    ),
    st.builds(RequestVoteReply, term=ints, vote_granted=st.booleans()),
    st.builds(PreVote, term=ints, candidate_id=texts, last_log_index=ints, last_log_term=ints),
    st.builds(PreVoteReply, term=ints, vote_granted=st.booleans()),
    st.builds(
        AppendEntries,
        term=ints,
        leader_id=texts,
        prev_log_index=ints,
        prev_log_term=ints,
        entries=st.lists(log_entries, max_size=5).map(tuple),
        leader_commit=ints,
    ),
    st.builds(
        AppendEntriesReply,
        term=ints,
        success=st.booleans(),
        match_index=ints,
        conflict_index=ints,
        conflict_term=ints,
    ),
)


# -- Round trips through bytes --


@given(msg=messages, src=texts)
def test_every_message_round_trips_through_bytes(msg: Message, src: str) -> None:
    wire = encode_message(msg, src).SerializeToString()
    assert decode_envelope(pb.Envelope.FromString(wire)) == (src, msg)


@given(entry=log_entries)
def test_every_log_entry_shape_round_trips_through_bytes(entry: LogEntry) -> None:
    wire = encode_log_entry(entry).SerializeToString()
    assert decode_log_entry(pb.LogEntry.FromString(wire)) == entry


@given(request=client_requests)
def test_client_requests_round_trip_through_bytes(request: ClientRequest) -> None:
    wire = encode_client_request(request).SerializeToString()
    assert decode_client_request(pb.ClientRequest.FromString(wire)) == request


@pytest.mark.parametrize("value", [None, NOT_FOUND, "", "v"])
def test_every_result_shape_round_trips_through_bytes(value: object) -> None:
    wire = encode_result(value).SerializeToString()
    assert decode_result(pb.Result.FromString(wire)) == value


def test_not_found_decodes_to_the_singleton_not_a_lookalike() -> None:
    # Callers compare with `is NOT_FOUND`, same as KVStateMachine's own.
    wire = encode_result(NOT_FOUND).SerializeToString()
    assert decode_result(pb.Result.FromString(wire)) is NOT_FOUND


def test_the_sentinel_round_trips_as_a_none_command() -> None:
    assert decode_log_entry(encode_log_entry(SENTINEL)) == SENTINEL


def test_empty_string_and_none_stay_distinct() -> None:
    # Both are proto3's default for a str field; only the oneof tells them apart.
    assert decode_result(encode_result("")) == ""
    assert decode_result(encode_result(None)) is None
    put_empty = LogEntry(term=1, command=Put(key="", value=""))
    assert decode_log_entry(encode_log_entry(put_empty)) == put_empty


# -- Values with no wire form are rejected at encode time --


@pytest.mark.parametrize("value", [1, 1.5, None, b"bytes", ["list"]])
def test_a_non_str_put_value_raises(value: object) -> None:
    with pytest.raises(TypeError, match="Put.value must be a str"):
        encode_command(Put(key="k", value=value))


def test_a_non_str_put_value_raises_from_inside_a_client_request() -> None:
    with pytest.raises(TypeError, match="Put.value must be a str"):
        encode_client_request(ClientRequest("c", 1, Put(key="k", value=1)))


def test_a_non_str_put_value_raises_from_inside_append_entries() -> None:
    entry = LogEntry(term=1, command=ClientRequest("c", 1, Put(key="k", value=1)))
    msg = AppendEntries(
        term=1, leader_id="0", prev_log_index=0, prev_log_term=0, entries=(entry,), leader_commit=0
    )
    with pytest.raises(TypeError, match="Put.value must be a str"):
        encode_message(msg, "0")


def test_a_non_command_raises() -> None:
    with pytest.raises(TypeError, match="not a Command"):
        encode_command("set x=1")  # type: ignore[arg-type]


def test_an_opaque_log_command_raises() -> None:
    # The simulator's tests replicate placeholder strings; the wire can't.
    with pytest.raises(TypeError, match="no wire form"):
        encode_log_entry(LogEntry(term=1, command="cmd-1"))


def test_a_non_message_raises() -> None:
    with pytest.raises(TypeError, match="not a Message"):
        encode_message("hello", "0")  # type: ignore[arg-type]


def test_a_result_with_no_wire_form_raises() -> None:
    with pytest.raises(TypeError, match="no wire form"):
        encode_result(42)


# -- An unset required oneof is rejected at decode time --


def test_an_envelope_with_no_message_raises() -> None:
    with pytest.raises(CodecError, match="Envelope"):
        decode_envelope(pb.Envelope(src="0"))


def test_a_command_with_no_op_raises() -> None:
    with pytest.raises(CodecError, match="Command"):
        decode_command(pb.Command())


def test_a_client_request_with_no_command_raises() -> None:
    with pytest.raises(CodecError, match="ClientRequest"):
        decode_client_request(pb.ClientRequest(client_id="c", serial_number=1))


def test_a_result_with_no_value_raises() -> None:
    with pytest.raises(CodecError, match="Result"):
        decode_result(pb.Result())
