"""save_local 的默认 pinned 池：跨 step 复用、扩容有界、可配置 / 可关闭。

默认池是进程级常驻的，所以这些用例都靠 conftest 的 autouse 夹具在前后各关一次；
每个用例内部只看「本次从零建起来的池」。
"""

from __future__ import annotations

from pathlib import Path

import pytest

import api as ckpt
from async_ import (
    DEFAULT_CHUNK_BYTES,
    PinnedBufferPool,
    SnapshotState,
    close_default_pool,
    configure_default_pool,
    default_pool,
)
from core import PoolError
from storage.local import LocalStorage


def test_default_pool_reuses_the_block_across_saves(dataset: Path, tmp_path: Path) -> None:
    data = ckpt.load(dataset, device="cuda:0")
    root = tmp_path / "local"

    first = ckpt.save_local(data.tensors, root, step=1)
    second = ckpt.save_local(data.tensors, root, step=2)
    third = ckpt.save_local(data.tensors, root, step=3)

    pool = ckpt.default_pool()
    assert pool.stats.allocations == 1, "第一次分配之后，同一份 checkpoint 不该再分配"
    assert pool.stats.reuses == 2
    assert pool.stats.chunks == 1, "稳态只留一块"
    assert first.pool_reused is False, "首版是新分配"
    assert second.pool_reused is True, "第二版命中同一块"
    assert third.pool_reused is True
    assert third.state is SnapshotState.COMMITTED


def test_default_pool_uses_chunk_granularity_not_whole_checkpoint(
    dataset: Path, tmp_path: Path
) -> None:
    """默认池的块粒度是 chunk_bytes，而不是「刚好装下这一份 checkpoint」。"""
    data = ckpt.load(dataset, device="cuda:0")
    root = tmp_path / "local"
    report = ckpt.save_local(data.tensors, root, step=1)

    pool = ckpt.default_pool()
    assert pool.chunk_bytes == DEFAULT_CHUNK_BYTES
    assert pool.max_bytes is None, "默认不设上限，容量不足时可以扩容而不是卡住"
    assert pool.stats.capacity_bytes >= report.size
    assert pool.stats.capacity_bytes % DEFAULT_CHUNK_BYTES == 0


def test_default_pool_capacity_stays_flat(dataset: Path, tmp_path: Path) -> None:
    """连续保存 6 代：池不长大（每代 trim 后只留一块复用候选）。"""
    data = ckpt.load(dataset, device="cuda:0")
    root = tmp_path / "local"

    ckpt.save_local(data.tensors, root, step=1)
    capacity = ckpt.default_pool().stats.capacity_bytes

    for step in range(2, 7):
        ckpt.save_local(data.tensors, root, step=step)

    pool = ckpt.default_pool()
    assert pool.stats.capacity_bytes == capacity
    assert pool.stats.chunks == 1
    assert pool.stats.allocations == 1


def test_close_default_pool_releases_memory(dataset: Path, tmp_path: Path) -> None:
    data = ckpt.load(dataset, device="cuda:0")
    root = tmp_path / "local"
    ckpt.save_local(data.tensors, root, step=1)

    old = ckpt.default_pool()
    assert old.stats.capacity_bytes > 0

    close_default_pool()

    assert old.closed
    assert old.stats.capacity_bytes == 0
    fresh = default_pool()
    assert fresh is not old
    assert fresh.stats.allocations == 0, "重新创建后从零开始"


def test_configure_default_pool_applies_to_the_next_creation(dataset: Path, tmp_path: Path) -> None:
    configure_default_pool(chunk_bytes=1 << 20, max_bytes=1 << 20)

    data = ckpt.load(dataset, device="cuda:0")
    ckpt.save_local(data.tensors, tmp_path / "local", step=1)

    pool = ckpt.default_pool()
    assert pool.chunk_bytes == 1 << 20
    assert pool.max_bytes == 1 << 20


def test_configure_after_creation_is_rejected() -> None:
    default_pool()
    with pytest.raises(PoolError):
        configure_default_pool(chunk_bytes=1 << 20)

    close_default_pool(keep_config=True)
    configure_default_pool(chunk_bytes=2 << 20)  # 关掉之后可以重新配
    assert default_pool().chunk_bytes == 2 << 20


def test_unknown_pool_option_is_rejected() -> None:
    with pytest.raises(TypeError):
        default_pool(chunk_byte=4096)  # noqa: F821 - 故意写错的键名


def test_failed_save_returns_the_block(dataset: Path, tmp_path: Path, monkeypatch) -> None:
    """保存失败也要把块还回默认池，否则下一代会一直等/重新分配。"""
    data = ckpt.load(dataset, device="cuda:0")

    def boom(self, key, buffer):  # noqa: ANN001
        raise OSError("disk full")

    monkeypatch.setattr(LocalStorage, "write_view", boom)
    with pytest.raises(OSError):
        ckpt.save_local(data.tensors, tmp_path / "local", step=1)

    pool = ckpt.default_pool()
    assert pool.stats.in_use_bytes == 0
    assert pool.has_free()
    assert pool.stats.releases == 1


def test_explicit_pool_still_wins(dataset: Path, tmp_path: Path) -> None:
    """显式传池时不用默认池，也不动默认池的统计。"""
    data = ckpt.load(dataset, device="cuda:0")
    pool = PinnedBufferPool(chunk_bytes=1 << 20, min_chunks=1, max_bytes=1 << 20, timeout=0)
    report = ckpt.save_local(data.tensors, tmp_path / "local", step=1, pool=pool)

    assert report.pool_reused is True, "预分配的块算复用"
    assert default_pool().stats.allocations == 0, "默认池没被碰到"


def test_larger_alignment_rebuilds_the_pool() -> None:
    """要求更大的地址对齐时重建池（旧块不满足 O_DIRECT 前提），不是悄悄复用。"""
    old = default_pool(alignment=4096)
    same = default_pool(alignment=4096)
    assert same is old, "同样的对齐要求复用同一个池"

    bigger = default_pool(alignment=8192)
    assert bigger is not old
    assert old.closed
    assert default_pool(alignment=4096) is bigger, "已经按 8 KiB 对齐，4 KiB 要求照样满足"


def test_alignment_change_is_refused_while_a_block_is_in_use() -> None:
    pool = default_pool(alignment=4096)
    held = pool.acquire(4096)
    try:
        with pytest.raises(PoolError):
            default_pool(alignment=8192)
    finally:
        held.release()


def test_pool_is_rebuilt_after_fork(monkeypatch) -> None:
    """fork 出来的子进程不能复用父进程的块（内容与锁状态都不可信）。"""
    import types

    from async_ import pinned as pinned_module

    parent = default_pool()
    monkeypatch.setattr(pinned_module, "os", types.SimpleNamespace(getpid=lambda: 424242))

    child = default_pool()
    assert child is not parent
    assert parent.stats.capacity_bytes == 0, "父进程那份在子进程里被丢掉了"
