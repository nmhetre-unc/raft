"""Tests for RaftNode's log replication: leader-side per-follower state
(`next_index`, `match_index`), constructing per-peer AppendEntries, the
`append_command` client entry point, the follower-side AppendEntries
receiver implementation (Figure 2's consistency check, conflict
resolution, and append), and commitment (leader-side majority advancement
under the current-term-only restriction, and follower-side adoption of
`leader_commit`).
"""

import random

from raft.messages import AppendEntries, AppendEntriesReply, RequestVoteReply
from raft.node import RaftNode, Role
from raft.storage import SENTINEL, LogEntry, MemoryStorage


def _force_election_timeout(node: RaftNode) -> None:
    """Make `node`'s very next tick() start an election.

    `election_deadline`'s real first value is only drawn on a node's
    first tick() call (see RaftNode's docstring for why), so setting
    `election_deadline` alone before that first call would just be
    overwritten. Marking it already-initialized keeps this override in
    place.
    """
    node.election_deadline = 0
    node._deadline_initialized = True


def _win_an_election(node: RaftNode, granting_peer: str, now: int) -> None:
    """Drive `node` through a full election it wins, via one peer's vote."""
    _force_election_timeout(node)
    node.tick(now=now)
    assert node.role is Role.CANDIDATE
    node.handle(
        RequestVoteReply(term=node.current_term, vote_granted=True), src=granting_peer, now=now + 1
    )
    assert node.role is Role.LEADER


# -- Leader replication state: next_index (a guess) and match_index (known truth) --


def test_next_index_and_match_index_initialized_on_election_win() -> None:
    storage = MemoryStorage()
    storage.append_entries([LogEntry(term=1, command="a"), LogEntry(term=1, command="b")])
    node = RaftNode("0", ["1", "2"], storage, rng=random.Random(1))

    _win_an_election(node, granting_peer="1", now=1000)

    # last_log_index is 2 -> next_index is optimistic, one past it;
    # match_index is 0 everywhere -- nothing confirmed yet.
    assert node.next_index == {"1": 3, "2": 3}
    assert node.match_index == {"1": 0, "2": 0}


def test_next_index_and_match_index_reset_on_a_new_election_win() -> None:
    storage = MemoryStorage()
    node = RaftNode("0", ["1"], storage, rng=random.Random(1))

    _win_an_election(node, granting_peer="1", now=1000)
    assert node.next_index == {"1": 1}  # empty log -> last_log_index 0

    # Simulate replication progress that must not survive to a new term.
    node.match_index["1"] = 5
    node.next_index["1"] = 10

    # Discover a higher term and step down.
    stale_heartbeat = AppendEntries(
        term=node.current_term + 5,
        leader_id="1",
        prev_log_index=0,
        prev_log_term=0,
        entries=(),
        leader_commit=0,
    )
    node.handle(stale_heartbeat, src="1", now=1002)
    assert node.role is Role.FOLLOWER

    # The log grows legitimately while this node is just a follower.
    storage.append_entries([LogEntry(term=node.current_term, command="x")])

    _win_an_election(node, granting_peer="1", now=2000)

    # Reflects the *current* log (last_log_index now 1), not the stale
    # progress from the previous leadership stint.
    assert node.next_index == {"1": 2}
    assert node.match_index == {"1": 0}


def test_each_peer_receives_append_entries_reflecting_its_own_next_index() -> None:
    storage = MemoryStorage()
    storage.append_entries(
        [
            LogEntry(term=1, command="a"),
            LogEntry(term=1, command="b"),
            LogEntry(term=2, command="c"),
        ]
    )
    node = RaftNode("0", ["caught-up", "behind"], storage, rng=random.Random(1))
    node.role = Role.LEADER
    node.leader_id = "0"
    node.next_index = {"caught-up": 4, "behind": 2}  # last_log_index is 3
    node.match_index = {"caught-up": 3, "behind": 0}

    produced = node._send_append_entries(now=100)
    by_peer = dict(produced)

    caught_up_msg = by_peer["caught-up"]
    assert isinstance(caught_up_msg, AppendEntries)
    assert caught_up_msg.entries == ()  # nothing left to send
    assert caught_up_msg.prev_log_index == 3
    assert caught_up_msg.prev_log_term == 2

    behind_msg = by_peer["behind"]
    assert isinstance(behind_msg, AppendEntries)
    assert behind_msg.entries == (LogEntry(term=1, command="b"), LogEntry(term=2, command="c"))
    assert behind_msg.prev_log_index == 1
    assert behind_msg.prev_log_term == 1


def test_peer_at_last_log_index_plus_one_receives_empty_entries() -> None:
    storage = MemoryStorage()
    storage.append_entries([LogEntry(term=1, command="a")])
    node = RaftNode("0", ["1"], storage, rng=random.Random(1))
    node.role = Role.LEADER
    node.next_index = {"1": storage.last_log_index() + 1}
    node.match_index = {"1": 0}

    produced = node._send_append_entries(now=0)

    assert len(produced) == 1
    _, msg = produced[0]
    assert isinstance(msg, AppendEntries)
    assert msg.entries == ()


def test_append_command_on_a_follower_returns_nothing_and_appends_nothing() -> None:
    storage = MemoryStorage()
    node = RaftNode("0", ["1"], storage, rng=random.Random(1))
    assert node.role is Role.FOLLOWER

    result = node.append_command("set x=1", now=0)

    assert result == []
    assert storage.last_log_index() == 0  # nothing was appended


def test_append_command_on_a_leader_appends_and_sends_to_every_peer() -> None:
    storage = MemoryStorage()
    node = RaftNode("0", ["1", "2"], storage, rng=random.Random(1))
    node.role = Role.LEADER
    node.leader_id = "0"
    node.next_index = {"1": 1, "2": 1}
    node.match_index = {"1": 0, "2": 0}

    produced = node.append_command("set x=1", now=100)

    expected_entry = LogEntry(term=node.current_term, command="set x=1")
    assert storage.load_log()[-1] == expected_entry
    assert {dst for dst, _msg in produced} == {"1", "2"}
    for _dst, msg in produced:
        assert isinstance(msg, AppendEntries)
        assert msg.entries == (expected_entry,)


# -- Follower-side AppendEntries: Figure 2's receiver implementation --


def test_follower_with_a_matching_prefix_accepts_and_appends() -> None:
    storage = MemoryStorage()
    storage.append_entries([LogEntry(term=1, command="a")])
    node = RaftNode("f", ["leader"], storage, rng=random.Random(1))

    new_entry = LogEntry(term=1, command="b")
    msg = AppendEntries(
        term=1,
        leader_id="leader",
        prev_log_index=1,
        prev_log_term=1,
        entries=(new_entry,),
        leader_commit=0,
    )

    reply = node.handle(msg, src="leader", now=0)

    assert reply == [("leader", AppendEntriesReply(term=1, success=True, match_index=2))]
    assert storage.load_log() == [SENTINEL, LogEntry(term=1, command="a"), new_entry]


def test_follower_missing_the_entry_at_prev_log_index_rejects() -> None:
    storage = MemoryStorage()  # empty log: only the index-0 sentinel
    node = RaftNode("f", ["leader"], storage, rng=random.Random(1))

    msg = AppendEntries(
        term=1, leader_id="leader", prev_log_index=5, prev_log_term=1, entries=(), leader_commit=0
    )

    reply = node.handle(msg, src="leader", now=0)

    assert reply == [("leader", AppendEntriesReply(term=1, success=False))]
    assert storage.load_log() == [SENTINEL]  # untouched


def test_follower_with_a_different_term_at_prev_log_index_rejects() -> None:
    storage = MemoryStorage()
    storage.append_entries([LogEntry(term=1, command="a")])
    node = RaftNode("f", ["leader"], storage, rng=random.Random(1))

    msg = AppendEntries(
        term=2, leader_id="leader", prev_log_index=1, prev_log_term=99, entries=(), leader_commit=0
    )

    reply = node.handle(msg, src="leader", now=0)

    assert reply == [("leader", AppendEntriesReply(term=2, success=False))]
    assert storage.load_log() == [SENTINEL, LogEntry(term=1, command="a")]  # untouched


def test_conflicting_entry_and_everything_after_it_is_deleted_before_appending() -> None:
    storage = MemoryStorage()
    storage.append_entries(
        [
            LogEntry(term=1, command="a"),
            LogEntry(term=1, command="stale-b"),
            LogEntry(term=1, command="stale-c"),
        ]
    )
    node = RaftNode("f", ["leader"], storage, rng=random.Random(1))

    new_b = LogEntry(term=2, command="b")
    new_c = LogEntry(term=2, command="c")
    msg = AppendEntries(
        term=2,
        leader_id="leader",
        prev_log_index=1,
        prev_log_term=1,
        entries=(new_b, new_c),
        leader_commit=0,
    )

    reply = node.handle(msg, src="leader", now=0)

    assert reply == [("leader", AppendEntriesReply(term=2, success=True, match_index=3))]
    assert storage.load_log() == [SENTINEL, LogEntry(term=1, command="a"), new_b, new_c]


def test_resending_entries_the_follower_already_has_is_a_true_noop() -> None:
    storage = MemoryStorage()
    storage.append_entries([LogEntry(term=1, command="a"), LogEntry(term=1, command="b")])
    node = RaftNode("f", ["leader"], storage, rng=random.Random(1))

    before = storage.load_log()

    # Freshly-constructed LogEntry objects, not the ones already stored --
    # only (index, term) needs to match for this to be a true no-op, per
    # Raft's Log Matching Property (matching term implies matching
    # command, so this is exactly "already have it", not an approximation).
    resent = (LogEntry(term=1, command="a"), LogEntry(term=1, command="b"))
    msg = AppendEntries(
        term=1,
        leader_id="leader",
        prev_log_index=0,
        prev_log_term=0,
        entries=resent,
        leader_commit=0,
    )

    reply = node.handle(msg, src="leader", now=0)
    after = storage.load_log()

    assert reply == [("leader", AppendEntriesReply(term=1, success=True, match_index=2))]
    # Equal is not the point -- these must be the *same objects*, proving
    # storage was never truncated and reappended, only left alone.
    assert len(before) == len(after)
    assert all(a is b for a, b in zip(before, after, strict=True))


def test_prev_log_index_0_always_passes_the_consistency_check() -> None:
    storage = MemoryStorage()
    # This follower's log has diverged from the leader well beyond the
    # sentinel -- but index 0 itself, the sentinel, is universal.
    storage.append_entries([LogEntry(term=7, command="unrelated")])
    node = RaftNode("f", ["leader"], storage, rng=random.Random(1))
    before = storage.load_log()

    msg = AppendEntries(
        term=7, leader_id="leader", prev_log_index=0, prev_log_term=0, entries=(), leader_commit=0
    )

    reply = node.handle(msg, src="leader", now=0)

    assert reply == [("leader", AppendEntriesReply(term=7, success=True))]
    assert storage.load_log() == before  # a bare heartbeat touches nothing


def test_longer_follower_log_is_preserved_unless_the_leader_actually_conflicts() -> None:
    storage = MemoryStorage()
    storage.append_entries(
        [
            LogEntry(term=1, command="a"),
            LogEntry(term=1, command="b"),
            LogEntry(term=1, command="c"),  # this follower has more than the leader sends
        ]
    )
    node = RaftNode("f", ["leader"], storage, rng=random.Random(1))
    before = storage.load_log()

    # The leader only knows about (and sends) up through "b" -- matching
    # what this follower already has there. It says nothing about "c".
    matching_msg = AppendEntries(
        term=1,
        leader_id="leader",
        prev_log_index=1,
        prev_log_term=1,
        entries=(LogEntry(term=1, command="b"),),
        leader_commit=0,
    )
    reply = node.handle(matching_msg, src="leader", now=0)
    after_matching = storage.load_log()

    assert reply == [("leader", AppendEntriesReply(term=1, success=True, match_index=2))]
    assert after_matching == before  # "c" survives: no conflict was ever sent
    assert all(a is b for a, b in zip(before, after_matching, strict=True))

    # Now the leader genuinely conflicts, at index 3.
    conflicting_msg = AppendEntries(
        term=1,
        leader_id="leader",
        prev_log_index=2,
        prev_log_term=1,
        entries=(LogEntry(term=2, command="new-c"),),
        leader_commit=0,
    )
    node.handle(conflicting_msg, src="leader", now=1)

    assert storage.load_log() == [
        SENTINEL,
        LogEntry(term=1, command="a"),
        LogEntry(term=1, command="b"),
        LogEntry(term=2, command="new-c"),
    ]


# -- Leader-side AppendEntriesReply handling --


def test_successful_reply_advances_next_index_and_match_index() -> None:
    storage = MemoryStorage()
    node = RaftNode("0", ["1"], storage, rng=random.Random(1))
    node.role = Role.LEADER
    node.leader_id = "0"
    node.next_index = {"1": 1}
    node.match_index = {"1": 0}

    # The reply is self-describing -- it names the index it confirms --
    # so, unlike the old _outstanding-based design, nothing needs to have
    # been sent first for this to mean anything.
    out = node.handle(
        AppendEntriesReply(term=node.current_term, success=True, match_index=2), src="1", now=1
    )

    assert out == []
    assert node.match_index["1"] == 2
    assert node.next_index["1"] == 3


def test_failed_reply_decrements_next_index_and_produces_a_retry() -> None:
    storage = MemoryStorage()
    storage.append_entries([LogEntry(term=1, command="a"), LogEntry(term=1, command="b")])
    node = RaftNode("0", ["1"], storage, rng=random.Random(1))
    node.role = Role.LEADER
    node.leader_id = "0"
    node.next_index = {"1": 3}
    node.match_index = {"1": 0}

    out = node.handle(AppendEntriesReply(term=node.current_term, success=False), src="1", now=1)

    assert node.next_index["1"] == 2  # decremented by exactly one
    assert node.match_index["1"] == 0  # untouched by a failure

    assert len(out) == 1
    retry_dst, retry_msg = out[0]
    assert retry_dst == "1"
    assert isinstance(retry_msg, AppendEntries)
    assert retry_msg.prev_log_index == 1  # the new, decremented next_index(2) - 1


def test_next_index_never_goes_below_1() -> None:
    storage = MemoryStorage()
    node = RaftNode("0", ["1"], storage, rng=random.Random(1))
    node.role = Role.LEADER
    node.leader_id = "0"
    node.next_index = {"1": 1}
    node.match_index = {"1": 0}

    node.handle(AppendEntriesReply(term=node.current_term, success=False), src="1", now=1)

    assert node.next_index["1"] == 1  # clamped at 1, not decremented to 0


def test_stale_reply_from_an_older_term_is_ignored() -> None:
    storage = MemoryStorage()
    node = RaftNode("0", ["1"], storage, rng=random.Random(1))
    node.current_term = 5
    node.role = Role.LEADER
    node.leader_id = "0"
    node.next_index = {"1": 1}
    node.match_index = {"1": 0}

    # A generous match_index, to prove even that doesn't get applied once
    # the term check alone should reject this reply outright.
    out = node.handle(AppendEntriesReply(term=3, success=True, match_index=5), src="1", now=1)

    assert out == []
    assert node.match_index["1"] == 0  # untouched
    assert node.next_index["1"] == 1  # untouched


def test_reply_arriving_after_stepping_down_is_ignored() -> None:
    storage = MemoryStorage()
    node = RaftNode("0", ["1"], storage, rng=random.Random(1))
    node.role = Role.LEADER
    node.leader_id = "0"
    node.next_index = {"1": 1}
    node.match_index = {"1": 0}

    node.role = Role.FOLLOWER  # stepped down (e.g. discovered a legitimate leader)

    out = node.handle(
        AppendEntriesReply(term=node.current_term, success=True, match_index=5), src="1", now=1
    )

    assert out == []
    assert node.match_index["1"] == 0  # untouched


def test_out_of_order_replies_never_move_match_index_backward() -> None:
    storage = MemoryStorage()
    node = RaftNode("0", ["1"], storage, rng=random.Random(1))
    node.role = Role.LEADER
    node.leader_id = "0"
    node.next_index = {"1": 1}
    node.match_index = {"1": 0}

    # The reply to a later, larger send arrives first.
    node.handle(
        AppendEntriesReply(term=node.current_term, success=True, match_index=5), src="1", now=1
    )
    assert node.match_index["1"] == 5
    assert node.next_index["1"] == 6

    # A reply to an earlier, smaller send -- delayed, arriving second --
    # is telling the truth about what THAT request confirmed; it's just
    # stale relative to what a newer reply already established. It must
    # not move match_index backward.
    node.handle(
        AppendEntriesReply(term=node.current_term, success=True, match_index=3), src="1", now=2
    )

    assert node.match_index["1"] == 5  # never moved backward
    assert node.next_index["1"] == 6


def test_full_convergence_walks_back_and_overwrites_a_divergent_tail() -> None:
    """A follower whose last 10 entries are stale gets walked back and
    overwritten to match the leader exactly, one rejection at a time."""
    leader_storage = MemoryStorage()
    leader_storage.append_entries([LogEntry(term=2, command=f"leader-{i}") for i in range(1, 13)])

    follower_storage = MemoryStorage()
    follower_storage.append_entries(
        [LogEntry(term=2, command="leader-1"), LogEntry(term=2, command="leader-2")]
    )
    # The follower's tail (indices 3..12) is from an old, overwritten term.
    follower_storage.append_entries(
        [LogEntry(term=1, command=f"stale-{i}") for i in range(3, 13)]
    )

    leader = RaftNode("leader", ["follower"], leader_storage, rng=random.Random(1))
    follower = RaftNode("follower", ["leader"], follower_storage, rng=random.Random(2))
    leader.current_term = 2
    leader.role = Role.LEADER
    leader.leader_id = "leader"
    last_index = leader_storage.last_log_index()
    leader.next_index = {"follower": last_index + 1}  # optimistic: 13
    leader.match_index = {"follower": 0}

    now = 0
    produced = leader._send_append_entries(now=now)

    rounds = 0
    while produced:
        rounds += 1
        assert rounds < 100  # guard against an infinite retry loop
        [(dst, msg)] = produced
        assert dst == "follower"
        [(reply_dst, reply_msg)] = follower.handle(msg, src="leader", now=now)
        assert reply_dst == "leader"
        now += 1
        produced = leader.handle(reply_msg, src="follower", now=now)

    assert rounds == 11  # 10 rejections walking back past the stale tail, then 1 acceptance
    assert leader.match_index["follower"] == last_index
    assert leader.next_index["follower"] == last_index + 1
    assert follower_storage.load_log() == leader_storage.load_log()


# -- Commitment: leader-side majority advancement, the current-term-only
# restriction (Figure 2's last paragraph; Figure 8), and follower-side
# adoption of leader_commit --


def test_leader_commits_an_index_a_majority_of_match_index_has_reached() -> None:
    storage = MemoryStorage()
    storage.append_entries([LogEntry(term=1, command=f"e{i}") for i in range(1, 6)])  # indices 1..5
    node = RaftNode("0", ["1", "2", "3", "4"], storage, rng=random.Random(1))
    node.current_term = 1
    node.role = Role.LEADER
    node.leader_id = "0"
    node.next_index = dict.fromkeys(node.peers, 6)
    node.match_index = {"1": 5, "2": 0, "3": 0, "4": 0}

    assert node.commit_index == 0

    # Leader (implicit) + "1" + "2" now = 3 of 5 -- a majority.
    node.handle(AppendEntriesReply(term=1, success=True, match_index=5), src="2", now=0)

    assert node.commit_index == 5


def test_leader_does_not_commit_an_earlier_term_entry_despite_a_full_majority() -> None:
    storage = MemoryStorage()
    storage.append_entries([LogEntry(term=1, command="old")])  # index 1, from an EARLIER term
    node = RaftNode("0", ["1", "2"], storage, rng=random.Random(1))
    node.current_term = 3  # this leader's current term has moved on
    node.role = Role.LEADER
    node.leader_id = "0"
    node.next_index = {"1": 2, "2": 2}
    node.match_index = {"1": 1, "2": 0}

    # Every node in the cluster -- a full, not just bare, majority -- ends
    # up holding index 1 once this reply lands.
    node.handle(AppendEntriesReply(term=3, success=True, match_index=1), src="2", now=0)

    assert node.commit_index == 0  # never committed: index 1's entry is term 1, not 3


def test_committing_a_current_term_entry_retroactively_commits_earlier_entries_below_it() -> None:
    storage = MemoryStorage()
    storage.append_entries(
        [
            LogEntry(term=1, command="old-1"),  # index 1, earlier term
            LogEntry(term=1, command="old-2"),  # index 2, earlier term
            LogEntry(term=3, command="new"),  # index 3, THIS leader's current term
        ]
    )
    node = RaftNode("0", ["1", "2"], storage, rng=random.Random(1))
    node.current_term = 3
    node.role = Role.LEADER
    node.leader_id = "0"
    node.next_index = {"1": 4, "2": 4}
    node.match_index = {"1": 3, "2": 0}

    assert node.commit_index == 0

    node.handle(AppendEntriesReply(term=3, success=True, match_index=3), src="2", now=0)

    # Reaching index 3 (this leader's own term) in one step also commits
    # indices 1 and 2 -- despite being from an earlier term, which on
    # their own (previous test) would never be committed directly.
    assert node.commit_index == 3


def test_follower_commit_index_never_exceeds_its_own_last_log_index() -> None:
    storage = MemoryStorage()
    storage.append_entries([LogEntry(term=1, command="a")])  # this follower only has index 1
    node = RaftNode("f", ["leader"], storage, rng=random.Random(1))

    # The leader claims a commit_index far beyond anything sent in (or
    # already held by) this message -- exactly the case min() exists for.
    msg = AppendEntries(
        term=1, leader_id="leader", prev_log_index=1, prev_log_term=1, entries=(), leader_commit=100
    )

    node.handle(msg, src="leader", now=0)

    assert node.commit_index == 1  # clamped to what this follower actually has
    assert node.commit_index <= node.storage.last_log_index()


def test_commit_index_is_0_after_restart_and_recovers_from_the_next_heartbeat() -> None:
    storage = MemoryStorage()
    storage.append_entries([LogEntry(term=1, command="a"), LogEntry(term=1, command="b")])
    node = RaftNode("f", ["leader"], storage, rng=random.Random(1))
    node.commit_index = 2  # this follower had learned entries 1 and 2 were committed

    # A crash + restart: a fresh RaftNode over the same (persisted, so
    # unaffected) storage. commit_index is volatile -- it does not survive.
    restarted = RaftNode("f", ["leader"], storage, rng=random.Random(2))
    assert restarted.commit_index == 0

    heartbeat = AppendEntries(
        term=1, leader_id="leader", prev_log_index=2, prev_log_term=1, entries=(), leader_commit=2
    )
    restarted.handle(heartbeat, src="leader", now=0)

    assert restarted.commit_index == 2  # recovered from the leader's heartbeat


def test_noop_on_election_lets_a_lone_leader_commit_immediately() -> None:
    storage = MemoryStorage()
    node = RaftNode("solo", [], storage, rng=random.Random(1), append_noop_on_election=True)
    _force_election_timeout(node)

    node.tick(now=10)

    assert node.role is Role.LEADER
    assert node.storage.last_log_index() == 1  # the no-op, at this leader's own term
    assert node.commit_index == 1  # a lone node is trivially its own majority


def test_without_noop_a_lone_leader_with_an_empty_log_commits_nothing() -> None:
    storage = MemoryStorage()
    node = RaftNode("solo", [], storage, rng=random.Random(1), append_noop_on_election=False)
    _force_election_timeout(node)

    node.tick(now=10)

    assert node.role is Role.LEADER
    assert node.storage.last_log_index() == 0  # nothing appended
    assert node.commit_index == 0  # the sentinel is term 0, never a real current_term
