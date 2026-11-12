"""Tests for `raft.net.driver.NodeDriver`.

Three drivers run in one event loop, wired together by `Net`, a fake
transport that delivers straight into the destination driver's inbox
unless the link is cut. Time is a plain integer the test advances by
hand, ticking every driver at each step, and `Net.settle()` runs the
loop until no driver has anything left to process -- so every test is
deterministic.

Every send goes through `Net.send`, which asserts the sender's storage
has nothing unsynced at that moment: persist-before-send is checked in
every test, not just the one named for it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from pathlib import Path
from random import Random
from typing import Any

import pytest

from raft.messages import Message
from raft.net import raft_pb2 as pb
from raft.net.codec import decode_log_entry, encode_log_entry
from raft.net.driver import (
    DriverStopped,
    Lost,
    NodeDriver,
    Ok,
    SubmitOutcome,
    Timeout,
)
from raft.net.filestorage import FileStorage
from raft.node import NotLeader, Role
from raft.statemachine import NOT_FOUND, ClientRequest, Command, Delete, Get, Put
from raft.storage import LogEntry, StorageError

STEP = 10


class TrackedStorage(FileStorage):
    """Deferred-mode `FileStorage` that knows whether it holds unsynced
    writes, counts syncs, and can be armed to fail its next sync."""

    def __init__(self, directory: Path) -> None:
        super().__init__(
            directory,
            encode=lambda e: encode_log_entry(e).SerializeToString(),
            decode=lambda b: decode_log_entry(pb.LogEntry.FromString(b)),
            fsync="deferred",
        )
        self.dirty = False
        self.syncs = 0
        self.fail_next_sync = False

    def save_term_and_vote(self, term: int, voted_for: str | None) -> None:
        super().save_term_and_vote(term, voted_for)
        self.dirty = True

    def append_entries(self, entries: list[LogEntry]) -> None:
        super().append_entries(entries)
        self.dirty = True

    def truncate_from(self, index: int) -> None:
        super().truncate_from(index)
        self.dirty = True

    def sync(self) -> None:
        if self.fail_next_sync:
            self._broken = True  # as FileStorage does after a real failed sync
            raise StorageError("simulated fsync failure")
        super().sync()
        self.dirty = False
        self.syncs += 1


class Link:
    def __init__(self, net: Net, src: str) -> None:
        self.net = net
        self.src = src

    def send(self, dst: str, message: Message) -> None:
        self.net.send(self.src, dst, message)


class Net:
    def __init__(self, tmp_path: Path, *, request_timeout: int = 1000) -> None:
        self.now = 0
        self.ids = ["n0", "n1", "n2"]
        self.cut: set[tuple[str, str]] = set()
        self.sent: list[tuple[str, str, Message]] = []
        self.storages = {i: TrackedStorage(tmp_path / i) for i in self.ids}
        self.drivers = {
            node_id: NodeDriver(
                node_id,
                [p for p in self.ids if p != node_id],
                self.storages[node_id],
                Link(self, node_id),
                lambda: self.now,
                request_timeout=request_timeout,
                rng=Random(seed),
            )
            for seed, node_id in enumerate(self.ids)
        }
        self.tasks = {i: asyncio.create_task(d.run()) for i, d in self.drivers.items()}

    def send(self, src: str, dst: str, message: Message) -> None:
        assert not self.storages[src].dirty, f"{src} sent {message!r} before syncing"
        self.sent.append((src, dst, message))
        if (src, dst) not in self.cut:
            self.drivers[dst].deliver(src, message)

    def isolate(self, node: str) -> None:
        for other in self.ids:
            if other != node:
                self.cut |= {(node, other), (other, node)}

    def heal(self) -> None:
        self.cut.clear()

    def sent_by(self, node: str) -> int:
        return sum(1 for src, _, _ in self.sent if src == node)

    async def settle(self) -> None:
        await asyncio.sleep(0)  # let freshly created submit tasks enqueue
        while True:
            before = len(self.sent)
            for driver in self.drivers.values():
                await driver.wait_idle()
            if len(self.sent) == before:
                return

    async def advance(self, ms: int) -> None:
        for _ in range(ms // STEP):
            self.now += STEP
            for driver in self.drivers.values():
                driver.tick()
            await self.settle()

    async def until(self, condition: Callable[[], bool], limit: int = 5000) -> None:
        deadline = self.now + limit
        await self.settle()
        while not condition():
            assert self.now < deadline, "condition not reached in time"
            await self.advance(STEP)

    def leaders(self, among: list[str] | None = None) -> list[str]:
        return [i for i in (among or self.ids) if self.drivers[i].status().role is Role.LEADER]

    async def elect(self, among: list[str] | None = None) -> str:
        """Run until exactly one of `among` leads and has committed its no-op."""

        def ready() -> bool:
            leaders = self.leaders(among)
            if len(leaders) != 1:
                return False
            status = self.drivers[leaders[0]].status()
            log = self.storages[leaders[0]].load_log()
            return log[status.commit_index].term == status.current_term

        await self.until(ready)
        (leader,) = self.leaders(among)
        return leader

    def start_submit(
        self, node: str, client: str, serial: int, command: Command
    ) -> asyncio.Task[SubmitOutcome]:
        request = ClientRequest(client_id=client, serial_number=serial, command=command)
        return asyncio.create_task(self.drivers[node].submit(request))

    async def submit(self, node: str, client: str, serial: int, command: Command) -> SubmitOutcome:
        task = self.start_submit(node, client, serial, command)
        await self.until(task.done)
        return task.result()

    async def shutdown(self) -> None:
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        for storage in self.storages.values():
            storage.close()


def async_test(
    body: Callable[[Net], Coroutine[Any, Any, None]], **net_kwargs: Any
) -> Callable[[Path], None]:
    """Run `body(net)` on a fresh `Net` in its own event loop."""

    def test(tmp_path: Path) -> None:
        async def main() -> None:
            net = Net(tmp_path, **net_kwargs)
            try:
                await body(net)
            finally:
                await net.shutdown()

        asyncio.run(main())

    # Not functools.wraps: pytest would follow __wrapped__ and look for a
    # fixture named `net`.
    test.__name__ = body.__name__
    test.__doc__ = body.__doc__
    return test


def with_net(**net_kwargs: Any) -> Callable[..., Callable[[Path], None]]:
    return lambda body: async_test(body, **net_kwargs)


# -- 1. Election and results --


@with_net()
async def test_leader_elected_and_submits_complete(net: Net) -> None:
    leader = await net.elect()

    assert await net.submit(leader, "c", 1, Put("k", "v")) == Ok(None)
    assert await net.submit(leader, "c", 2, Get("k")) == Ok("v")
    assert await net.submit(leader, "c", 3, Delete("k")) == Ok("v")
    assert await net.submit(leader, "c", 4, Get("k")) == Ok(NOT_FOUND)

    await net.advance(100)  # followers learn the final commit index
    statuses = [d.status() for d in net.drivers.values()]
    assert len({(s.current_term, s.commit_index, s.last_applied) for s in statuses}) == 1
    assert sum(s.role is Role.LEADER for s in statuses) == 1


# -- 2. Redirects --


@with_net()
async def test_not_leader_reply_carries_the_hint(net: Net) -> None:
    leader = await net.elect()
    follower = next(i for i in net.ids if i != leader)

    outcome = await net.submit(follower, "c", 1, Put("k", "v"))

    assert outcome == NotLeader(leader_hint=leader)
    assert await net.submit(leader, "c", 2, Get("k")) == Ok(NOT_FOUND)  # never appended


# -- 3. At-most-once --


@with_net()
async def test_duplicate_request_is_applied_once(net: Net) -> None:
    leader = await net.elect()
    assert await net.submit(leader, "c1", 1, Put("k", "v")) == Ok(None)

    # Two copies of the same delete in flight at once: two log entries,
    # one application.
    first = net.start_submit(leader, "c1", 2, Delete("k"))
    second = net.start_submit(leader, "c1", 2, Delete("k"))
    await net.until(lambda: first.done() and second.done())
    assert first.result() == second.result() == Ok("v")

    # Another client re-creates the key; a late retry of the delete must
    # not remove it again.
    assert await net.submit(leader, "c2", 1, Put("k", "w")) == Ok(None)
    assert await net.submit(leader, "c1", 2, Delete("k")) == Ok("v")
    assert await net.submit(leader, "c2", 2, Get("k")) == Ok("w")


@with_net()
async def test_superseded_result_is_a_timeout(net: Net) -> None:
    """Serials 1 and 2 from one client applied in the same batch: by the
    time serial 1 is checked, its session already holds serial 2's
    result, so serial 1 cannot be answered and gets Timeout."""
    leader = await net.elect()
    net.isolate(leader)  # so both entries reach the followers, and commit, together
    one = net.start_submit(leader, "c", 1, Put("k", "1"))
    two = net.start_submit(leader, "c", 2, Get("k"))
    await net.settle()
    net.heal()
    await net.until(lambda: one.done() and two.done())
    assert (one.result(), two.result()) == (Timeout(), Ok("1"))


# -- 4. Minority leader --


@with_net(request_timeout=500)
async def test_minority_leader_times_out(net: Net) -> None:
    leader = await net.elect()
    net.isolate(leader)
    start = net.now

    outcome = await net.submit(leader, "c", 1, Put("k", "v"))

    assert outcome == Timeout()
    assert 500 <= net.now - start < 500 + 2 * STEP
    assert net.drivers[leader].status().role is Role.LEADER  # still thinks so


# -- 5. Overwritten entry --


@pytest.mark.parametrize(
    "new_request",
    [
        ClientRequest("other", 1, Put("k", "new")),
        # Same client_id and serial, different command: the session for
        # ("c", 1) ends up holding *this* request's result, so only the
        # term check stops the old waiter from being told Ok.
        ClientRequest("c", 1, Put("k", "new")),
    ],
    ids=["other-client", "same-client-and-serial"],
)
def test_overwritten_entry_is_lost_never_ok(tmp_path: Path, new_request: ClientRequest) -> None:
    @with_net(request_timeout=100_000)
    async def body(net: Net) -> None:
        old = await net.elect()
        await net.advance(100)  # everyone holds the no-op
        net.isolate(old)
        waiting = net.start_submit(old, "c", 1, Put("k", "old"))
        await net.settle()
        old_index = net.storages[old].last_log_index()

        majority = [i for i in net.ids if i != old]
        new = await net.elect(among=majority)
        assert (
            net.storages[new].load_log()[old_index].term
            > net.storages[old].load_log()[old_index].term
        )  # the new leader's no-op sits at the old entry's index
        new_outcome = await net.submit(
            new, new_request.client_id, new_request.serial_number, new_request.command
        )
        assert new_outcome == Ok(None)
        assert not waiting.done()

        net.heal()
        await net.until(waiting.done)
        assert waiting.result() == Lost()
        # Whoever leads now (the old leader may win a later term), "old" was never applied.
        assert await net.submit(await net.elect(), "reader", 1, Get("k")) == Ok("new")

    body(tmp_path)


# -- 6. Persist before send --


@with_net()
async def test_every_send_happens_after_sync(net: Net) -> None:
    leader = await net.elect()
    for serial in range(1, 4):
        assert await net.submit(leader, "c", serial, Put("k", str(serial))) == Ok(None)
    assert len(net.sent) > 0  # every one of these passed Net.send's assertion

    # A burst queued together is one batch covered by one fsync. (RaftNode's
    # flow control sends each peer the first new entry and holds the rest
    # until that is acknowledged -- never, here: the leader is isolated --
    # so 5 submits send 2 messages, one per peer, both after that sync.)
    net.isolate(leader)
    syncs, sent = net.storages[leader].syncs, net.sent_by(leader)
    burst = [net.start_submit(leader, "c", s, Put("k", str(s))) for s in range(4, 9)]
    await net.settle()
    assert net.storages[leader].syncs == syncs + 1
    assert net.sent_by(leader) == sent + 2
    for task in burst:
        task.cancel()

    # The cancelled callers' entries still commit; their waiters are
    # dropped quietly rather than answered.
    net.heal()
    await net.until(lambda: net.drivers[leader].status().last_applied >= 9)
    assert await net.submit(leader, "c", 9, Get("k")) == Ok("8")


# -- 7. Fail-stop --


@with_net()
async def test_sync_failure_stops_the_driver(net: Net) -> None:
    leader = await net.elect()
    driver = net.drivers[leader]
    net.storages[leader].fail_next_sync = True
    sent = net.sent_by(leader)

    pending = net.start_submit(leader, "c", 1, Put("k", "v"))
    await net.settle()

    with pytest.raises(StorageError, match="simulated"):
        await net.tasks[leader]
    await net.advance(1000)  # ticks and deliveries are dropped from here on
    assert net.sent_by(leader) == sent
    assert not pending.done()
    with pytest.raises(DriverStopped):
        await driver.submit(ClientRequest("c", 2, Get("k")))
    with pytest.raises(DriverStopped):
        await driver.run()
    pending.cancel()

    # The other two carry on without it.
    new = await net.elect(among=[i for i in net.ids if i != leader])
    assert await net.submit(new, "c", 2, Put("k", "w")) == Ok(None)


# -- Construction --


def test_request_timeout_must_be_positive(tmp_path: Path) -> None:
    with TrackedStorage(tmp_path) as storage, pytest.raises(ValueError, match="request_timeout"):
        NodeDriver("n0", [], storage, Link(None, "n0"), lambda: 0, request_timeout=0)  # type: ignore[arg-type]
