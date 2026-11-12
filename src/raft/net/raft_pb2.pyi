from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Put(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class Get(_message.Message):
    __slots__ = ("key",)
    KEY_FIELD_NUMBER: _ClassVar[int]
    key: str
    def __init__(self, key: _Optional[str] = ...) -> None: ...

class Delete(_message.Message):
    __slots__ = ("key",)
    KEY_FIELD_NUMBER: _ClassVar[int]
    key: str
    def __init__(self, key: _Optional[str] = ...) -> None: ...

class Command(_message.Message):
    __slots__ = ("put", "get", "delete")
    PUT_FIELD_NUMBER: _ClassVar[int]
    GET_FIELD_NUMBER: _ClassVar[int]
    DELETE_FIELD_NUMBER: _ClassVar[int]
    put: Put
    get: Get
    delete: Delete
    def __init__(self, put: _Optional[_Union[Put, _Mapping]] = ..., get: _Optional[_Union[Get, _Mapping]] = ..., delete: _Optional[_Union[Delete, _Mapping]] = ...) -> None: ...

class ClientRequest(_message.Message):
    __slots__ = ("client_id", "serial_number", "command")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    SERIAL_NUMBER_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    serial_number: int
    command: Command
    def __init__(self, client_id: _Optional[str] = ..., serial_number: _Optional[int] = ..., command: _Optional[_Union[Command, _Mapping]] = ...) -> None: ...

class LogEntry(_message.Message):
    __slots__ = ("term", "client_request", "command")
    TERM_FIELD_NUMBER: _ClassVar[int]
    CLIENT_REQUEST_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    term: int
    client_request: ClientRequest
    command: Command
    def __init__(self, term: _Optional[int] = ..., client_request: _Optional[_Union[ClientRequest, _Mapping]] = ..., command: _Optional[_Union[Command, _Mapping]] = ...) -> None: ...

class RequestVote(_message.Message):
    __slots__ = ("term", "candidate_id", "last_log_index", "last_log_term")
    TERM_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_ID_FIELD_NUMBER: _ClassVar[int]
    LAST_LOG_INDEX_FIELD_NUMBER: _ClassVar[int]
    LAST_LOG_TERM_FIELD_NUMBER: _ClassVar[int]
    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int
    def __init__(self, term: _Optional[int] = ..., candidate_id: _Optional[str] = ..., last_log_index: _Optional[int] = ..., last_log_term: _Optional[int] = ...) -> None: ...

class RequestVoteReply(_message.Message):
    __slots__ = ("term", "vote_granted")
    TERM_FIELD_NUMBER: _ClassVar[int]
    VOTE_GRANTED_FIELD_NUMBER: _ClassVar[int]
    term: int
    vote_granted: bool
    def __init__(self, term: _Optional[int] = ..., vote_granted: _Optional[bool] = ...) -> None: ...

class AppendEntries(_message.Message):
    __slots__ = ("term", "leader_id", "prev_log_index", "prev_log_term", "entries", "leader_commit")
    TERM_FIELD_NUMBER: _ClassVar[int]
    LEADER_ID_FIELD_NUMBER: _ClassVar[int]
    PREV_LOG_INDEX_FIELD_NUMBER: _ClassVar[int]
    PREV_LOG_TERM_FIELD_NUMBER: _ClassVar[int]
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    LEADER_COMMIT_FIELD_NUMBER: _ClassVar[int]
    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: _containers.RepeatedCompositeFieldContainer[LogEntry]
    leader_commit: int
    def __init__(self, term: _Optional[int] = ..., leader_id: _Optional[str] = ..., prev_log_index: _Optional[int] = ..., prev_log_term: _Optional[int] = ..., entries: _Optional[_Iterable[_Union[LogEntry, _Mapping]]] = ..., leader_commit: _Optional[int] = ...) -> None: ...

class AppendEntriesReply(_message.Message):
    __slots__ = ("term", "success", "match_index", "conflict_index", "conflict_term")
    TERM_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MATCH_INDEX_FIELD_NUMBER: _ClassVar[int]
    CONFLICT_INDEX_FIELD_NUMBER: _ClassVar[int]
    CONFLICT_TERM_FIELD_NUMBER: _ClassVar[int]
    term: int
    success: bool
    match_index: int
    conflict_index: int
    conflict_term: int
    def __init__(self, term: _Optional[int] = ..., success: _Optional[bool] = ..., match_index: _Optional[int] = ..., conflict_index: _Optional[int] = ..., conflict_term: _Optional[int] = ...) -> None: ...

class Envelope(_message.Message):
    __slots__ = ("src", "request_vote", "request_vote_reply", "append_entries", "append_entries_reply")
    SRC_FIELD_NUMBER: _ClassVar[int]
    REQUEST_VOTE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_VOTE_REPLY_FIELD_NUMBER: _ClassVar[int]
    APPEND_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    APPEND_ENTRIES_REPLY_FIELD_NUMBER: _ClassVar[int]
    src: str
    request_vote: RequestVote
    request_vote_reply: RequestVoteReply
    append_entries: AppendEntries
    append_entries_reply: AppendEntriesReply
    def __init__(self, src: _Optional[str] = ..., request_vote: _Optional[_Union[RequestVote, _Mapping]] = ..., request_vote_reply: _Optional[_Union[RequestVoteReply, _Mapping]] = ..., append_entries: _Optional[_Union[AppendEntries, _Mapping]] = ..., append_entries_reply: _Optional[_Union[AppendEntriesReply, _Mapping]] = ...) -> None: ...

class Unit(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Result(_message.Message):
    __slots__ = ("none", "not_found", "value")
    NONE_FIELD_NUMBER: _ClassVar[int]
    NOT_FOUND_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    none: Unit
    not_found: Unit
    value: str
    def __init__(self, none: _Optional[_Union[Unit, _Mapping]] = ..., not_found: _Optional[_Union[Unit, _Mapping]] = ..., value: _Optional[str] = ...) -> None: ...

class NotLeader(_message.Message):
    __slots__ = ("leader_hint",)
    LEADER_HINT_FIELD_NUMBER: _ClassVar[int]
    leader_hint: str
    def __init__(self, leader_hint: _Optional[str] = ...) -> None: ...

class SubmitReply(_message.Message):
    __slots__ = ("ok", "not_leader", "timeout", "lost")
    OK_FIELD_NUMBER: _ClassVar[int]
    NOT_LEADER_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    LOST_FIELD_NUMBER: _ClassVar[int]
    ok: Result
    not_leader: NotLeader
    timeout: Unit
    lost: Unit
    def __init__(self, ok: _Optional[_Union[Result, _Mapping]] = ..., not_leader: _Optional[_Union[NotLeader, _Mapping]] = ..., timeout: _Optional[_Union[Unit, _Mapping]] = ..., lost: _Optional[_Union[Unit, _Mapping]] = ...) -> None: ...

class StatusRequest(_message.Message):
    __slots__ = ("include_log",)
    INCLUDE_LOG_FIELD_NUMBER: _ClassVar[int]
    include_log: bool
    def __init__(self, include_log: _Optional[bool] = ...) -> None: ...

class StatusReply(_message.Message):
    __slots__ = ("node_id", "role", "current_term", "commit_index", "last_applied", "leader_hint", "log")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_TERM_FIELD_NUMBER: _ClassVar[int]
    COMMIT_INDEX_FIELD_NUMBER: _ClassVar[int]
    LAST_APPLIED_FIELD_NUMBER: _ClassVar[int]
    LEADER_HINT_FIELD_NUMBER: _ClassVar[int]
    LOG_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    role: str
    current_term: int
    commit_index: int
    last_applied: int
    leader_hint: str
    log: _containers.RepeatedCompositeFieldContainer[LogEntry]
    def __init__(self, node_id: _Optional[str] = ..., role: _Optional[str] = ..., current_term: _Optional[int] = ..., commit_index: _Optional[int] = ..., last_applied: _Optional[int] = ..., leader_hint: _Optional[str] = ..., log: _Optional[_Iterable[_Union[LogEntry, _Mapping]]] = ...) -> None: ...
