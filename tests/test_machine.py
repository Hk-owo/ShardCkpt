"""快照状态机测试：主干转移、两条易错边、门闸（跳过 vs 强制等待）。"""

from __future__ import annotations

import pytest

from async_ import (
    SnapshotError,
    SnapshotManager,
    SnapshotState,
    TriggerDecision,
)


class FakePool:
    """最小 buffer 池：只记账，不代表真实内存。"""

    def __init__(self, *, free: int = 4, has_free: bool | None = None) -> None:
        self.free = free
        self.acquired: list[object] = []
        self.released: list[object] = []
        self._has_free = has_free

    def has_free(self) -> bool:
        return self._has_free if self._has_free is not None else self.free > 0

    def acquire(self, nbytes: int) -> object:
        assert self.has_free(), "池里没有可用块时不该调用 acquire"
        self.free -= 1
        token = object()
        self.acquired.append(token)
        return token

    def release(self, buffer: object) -> None:
        self.released.append(buffer)
        self.free += 1


def start(manager: SnapshotManager, step: int = 1):
    trigger = manager.trigger(step, nbytes=1024)
    assert trigger.decision is TriggerDecision.START
    return trigger.snapshot


def test_happy_path_releases_buffer_exactly_once() -> None:
    pool = FakePool()
    manager = SnapshotManager(pool)
    snap = start(manager, step=10)

    assert snap.state is SnapshotState.SNAPSHOTTING
    assert snap.holds_buffer
    assert pool.released == [], "D2H 完成前不该归还 buffer"

    snap.d2h_done()
    assert snap.state is SnapshotState.PERSISTING
    assert pool.released == [], "持久化还在读 host 数据，不能归还"

    snap.persist_ok()
    assert snap.state is SnapshotState.COMMITTING
    assert pool.released == []

    snap.commit_ok()
    assert snap.state is SnapshotState.COMMITTED
    assert pool.released == [snap.buffer], "提交后归还一次"
    assert snap.released
    assert manager.active == ()
    assert manager.stats.committed == 1
    assert manager.stats.started == 1


def test_d2h_failure_drops_and_never_needs_cleanup() -> None:
    """易错边 1：SNAPSHOTTING 失败必须回 DROPPED，而不是 ABORTED。"""
    pool = FakePool()
    manager = SnapshotManager(pool)
    snap = start(manager)

    snap.fail("cuda error: invalid device pointer")

    assert snap.state is SnapshotState.DROPPED
    assert snap.reason == "cuda error: invalid device pointer"
    assert pool.released == [snap.buffer], "失败也要归还 buffer"
    assert manager.stats.dropped == 1
    assert manager.stats.aborted == 0, "存储侧没有任何写入，不该产生待清理目录"
    assert manager.active == ()
    with pytest.raises(SnapshotError):
        snap.mark_cleaned()  # 没有目录可清理


def test_persist_failure_aborts_then_allows_cleanup() -> None:
    pool = FakePool()
    manager = SnapshotManager(pool)
    snap = start(manager)
    snap.d2h_done()

    snap.fail("timeout writing s3://ckpt/runs/step_1")

    assert snap.state is SnapshotState.ABORTED
    assert manager.stats.aborted == 1
    assert pool.released == [snap.buffer]

    snap.mark_cleaned()
    assert snap.cleaned
    with pytest.raises(SnapshotError):
        snap.mark_cleaned()  # 不能清理两次

    # 目录清理完之后可以正常开下一代
    assert manager.trigger(2).decision is TriggerDecision.START


def test_commit_failure_aborts() -> None:
    pool = FakePool()
    manager = SnapshotManager(pool)
    snap = start(manager)
    snap.d2h_done()
    snap.persist_ok()

    snap.fail("COMMIT rename failed")

    assert snap.state is SnapshotState.ABORTED
    assert manager.stats.aborted == 1
    assert pool.released == [snap.buffer]


def test_single_buffer_skips_while_previous_in_flight() -> None:
    pool = FakePool()
    manager = SnapshotManager(pool, max_inflight=1)
    first = start(manager, step=1)
    first.d2h_done()  # 还在 PERSISTING，buffer 未归还

    trigger = manager.trigger(2)

    assert trigger.decision is TriggerDecision.SKIP, "训练不该因为持久化慢而阻塞"
    assert trigger.snapshot is None
    assert "buffer" in trigger.reason
    assert manager.stats.skipped == 1
    assert manager.skipped_streak == 1

    # 上一代提交后又能开新一代
    first.persist_ok()
    first.commit_ok()
    assert manager.trigger(2).decision is TriggerDecision.START


def test_force_wait_after_repeated_skips() -> None:
    pool = FakePool()
    manager = SnapshotManager(pool, max_inflight=1, force_after_skips=3)
    first = start(manager, step=1)
    first.d2h_done()

    # 连续跳过 3 次之后才阻塞训练（对应文档「距上次成功超过阈值」）
    decisions = [manager.trigger(step).decision for step in (2, 3, 4, 5)]

    assert decisions == [
        TriggerDecision.SKIP,
        TriggerDecision.SKIP,
        TriggerDecision.SKIP,
        TriggerDecision.FORCE_WAIT,
    ]
    assert manager.stats.skipped == 3
    assert manager.stats.forced == 1
    assert manager.active == (first,), "强制等待不改变在途代"

    # 阻塞等到上一代结束之后，重新触发才能开始
    first.persist_ok()
    first.commit_ok()
    assert manager.trigger(6).decision is TriggerDecision.START


def test_commit_resets_skip_streak() -> None:
    pool = FakePool()
    manager = SnapshotManager(pool, max_inflight=1, force_after_skips=3)
    first = start(manager, step=1)
    first.d2h_done()
    manager.trigger(2)
    manager.trigger(3)
    assert manager.skipped_streak == 2

    first.persist_ok()
    first.commit_ok()
    assert manager.skipped_streak == 0, "距上次成功的计数要清零"

    assert manager.trigger(4).decision is TriggerDecision.START


def test_pool_without_free_block_blocks_new_generation() -> None:
    pool = FakePool(has_free=False)
    manager = SnapshotManager(pool)

    trigger = manager.trigger(1)

    assert trigger.decision is TriggerDecision.SKIP
    assert "没有可用块" in trigger.reason
    assert pool.acquired == [], "门闸不通过时不许真的分配"


def test_illegal_transitions_raise() -> None:
    pool = FakePool()
    manager = SnapshotManager(pool)
    snap = start(manager)

    with pytest.raises(SnapshotError):
        snap.persist_ok()  # 还没 D2H 完成
    with pytest.raises(SnapshotError):
        snap.commit_ok()
    with pytest.raises(SnapshotError):
        snap.mark_cleaned()

    snap.d2h_done()
    snap.persist_ok()
    snap.commit_ok()
    with pytest.raises(SnapshotError):
        snap.d2h_done()  # 终态不可再推进
    with pytest.raises(SnapshotError):
        snap.fail("too late")


def test_two_generations_in_flight() -> None:
    """max_inflight=2 即双缓冲：上一代 PERSISTING 时下一代可以开始快照。"""
    pool = FakePool()
    manager = SnapshotManager(pool, max_inflight=2)
    first = start(manager, step=1)
    first.d2h_done()

    second = start(manager, step=2)
    assert manager.active == (first, second)
    assert pool.released == []

    # 第三代越线：跳过
    assert manager.trigger(3).decision is TriggerDecision.SKIP

    first.persist_ok()
    first.commit_ok()
    second.d2h_done()
    second.persist_ok()
    second.commit_ok()

    assert pool.released == [first.buffer, second.buffer]
    assert manager.stats.committed == 2
    assert manager.active == ()
