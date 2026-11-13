"""Tests for `raft.net.verify`, including that each check actually fails."""

from __future__ import annotations

from raft.net.verify import NodeLog, acknowledged_puts_committed, committed_prefixes_agree
from raft.statemachine import ClientRequest, Get, Put
from raft.storage import LogEntry


def put(key: str, value: str, term: int = 1) -> LogEntry:
    return LogEntry(term, ClientRequest("c", 1, Put(key, value)))


NOOP = LogEntry(1, None)
A, B, C = put("a", "c:1"), put("b", "c:2"), put("c", "c:3")


def test_identical_committed_prefixes_agree() -> None:
    logs = {
        "n0": NodeLog(3, [NOOP, A, B]),
        "n1": NodeLog(3, [NOOP, A, B, C]),  # an uncommitted extra entry is fine
        "n2": NodeLog(2, [NOOP, A]),  # lagging is fine
    }
    assert committed_prefixes_agree(logs) == []


def test_uncommitted_divergence_is_allowed() -> None:
    logs = {
        "n0": NodeLog(2, [NOOP, A, B]),
        "n1": NodeLog(2, [NOOP, A, put("x", "other:1", term=2)]),
    }
    assert committed_prefixes_agree(logs) == []


def test_committed_divergence_is_reported() -> None:
    logs = {
        "n0": NodeLog(3, [NOOP, A, B]),
        "n1": NodeLog(3, [NOOP, A, put("b", "c:2", term=2)]),
        "n2": NodeLog(3, [NOOP, A, B]),
    }
    problems = committed_prefixes_agree(logs)
    assert len(problems) == 2  # n0-n1 and n1-n2, once each
    assert all("committed index 3" in p for p in problems)


def test_commit_index_past_the_log_is_reported() -> None:
    problems = committed_prefixes_agree({"n0": NodeLog(4, [NOOP]), "n1": NodeLog(1, [NOOP])})
    assert problems == ["n0: commit_index 4 is past its last entry 1"]


def test_every_acknowledged_put_is_present() -> None:
    committed = NodeLog(4, [NOOP, A, LogEntry(1, ClientRequest("c", 9, Get("a"))), B, C])
    assert acknowledged_puts_committed(committed.committed(), [("a", "c:1"), ("b", "c:2")]) == []


def test_missing_or_mismatched_acknowledged_puts_are_reported() -> None:
    committed = [NOOP, A, LogEntry(1, Put("raw", "x"))]  # a bare Command doesn't count
    problems = acknowledged_puts_committed(
        committed, [("a", "c:1"), ("a", "c:9"), ("b", "c:2"), ("raw", "x")]
    )
    assert problems == [
        "acknowledged put a='c:9' is not in the committed log",
        "acknowledged put b='c:2' is not in the committed log",
        "acknowledged put raw='x' is not in the committed log",
    ]
