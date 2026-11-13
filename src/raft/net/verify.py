"""Post-run safety checks over logs fetched from every node.

Both checks return a list of human-readable violations; an empty list
means the check passed. They take decoded `LogEntry` lists in Status
order -- entry 1 first, the index-0 sentinel omitted -- so
``entries[i]`` is log index ``i + 1``.

- `committed_prefixes_agree`: Raft's State Machine Safety, as far as an
  outside observer can check it. Any two nodes must hold identical
  entries at every index both consider committed.
- `acknowledged_puts_committed`: durability of acknowledged writes. Every
  put a client was told succeeded must appear in the committed log.
  Put values are unique per run (see `raft.net.loadgen`), so "appears"
  means an exact (key, value) match, not merely the key.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from raft.statemachine import ClientRequest, Put
from raft.storage import LogEntry


@dataclass(frozen=True)
class NodeLog:
    commit_index: int
    entries: Sequence[LogEntry]  # entries[i] is log index i + 1

    def committed(self) -> Sequence[LogEntry]:
        return self.entries[: self.commit_index]


def committed_prefixes_agree(logs: Mapping[str, NodeLog]) -> list[str]:
    problems = []
    for node, log in logs.items():
        if log.commit_index > len(log.entries):
            problems.append(
                f"{node}: commit_index {log.commit_index} is past its last entry {len(log.entries)}"
            )
    nodes = sorted(logs)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            shared = min(
                logs[a].commit_index,
                logs[b].commit_index,
                len(logs[a].entries),
                len(logs[b].entries),
            )
            for index in range(1, shared + 1):
                ea = logs[a].entries[index - 1]
                eb = logs[b].entries[index - 1]
                if ea != eb:
                    problems.append(
                        f"{a} and {b} differ at committed index {index}: {ea!r} != {eb!r}"
                    )
                    break  # the first divergence says it all for this pair
    return problems


def acknowledged_puts_committed(
    committed: Sequence[LogEntry], acknowledged: Iterable[tuple[str, str]]
) -> list[str]:
    """`acknowledged` is the `(key, value)` of every put a client was
    told succeeded; `committed` is the longest committed log."""
    present = {
        (entry.command.command.key, entry.command.command.value)
        for entry in committed
        if isinstance(entry.command, ClientRequest) and isinstance(entry.command.command, Put)
    }
    return [
        f"acknowledged put {key}={value!r} is not in the committed log"
        for key, value in acknowledged
        if (key, value) not in present
    ]
