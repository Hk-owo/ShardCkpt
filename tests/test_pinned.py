"""pinned buffer 池测试：预分配、复用、best-fit、容量上限与反压、归还语义。"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import api as ckpt
from async_ import (
    D2HStream,
    PinnedBuffer,
    PinnedBufferPool,
    SnapshotManager,
    SnapshotState,
    TriggerDecision,
    get_backend,
)
from core import PoolError, PoolExhaustedError


def test_preallocates_min_chunks() -> None:
    pool = PinnedBufferPool(chunk_bytes=4096, min_chunks=2)
    stats = pool.stats

    assert stats.chunks == 2
    assert stats.capacity_bytes == 8192
    assert stats.allocations == 2, "预分配发生在启动时，不该等到 acquire"
    assert stats.in_use_bytes == 0


def test_reuses_released_block_without_allocating() -> None:
    pool = PinnedBufferPool(chunk_bytes=4096, min_chunks=1)

    first = pool.acquire(1024)
    first.release()
    second = pool.acquire(1024)

    stats = pool.stats
    assert stats.allocations == 1, "两次 acquire 都没有再分配，说明块被复用了"
    assert stats.reuses == 2, "预分配的块与归还后的块都是复用"
    assert stats.acquisitions == 2
    assert stats.releases == 1
    assert stats.in_use_bytes == 4096
    assert stats.peak_in_use_bytes == 4096
    second.release()


def test_best_fit_picks_smallest_sufficient_block() -> None:
    pool = PinnedBufferPool(chunk_bytes=4096, min_chunks=2)
    big = pool.acquire(8192)  # 需要扩容；此时空闲块是两块 4 KiB
    big.release()  # 空闲：4K, 4K, 8K

    small = pool.acquire(1024)
    assert small.capacity == 4096, "best-fit 应该拿够用的最小块，而不是 8 KiB 那块"
    small.release()


def test_best_fit_works_when_the_match_is_not_first() -> None:
    """命中块不在空闲列表首位时也要能取到（numpy 的 == 不能拿来查列表）。"""
    pool = PinnedBufferPool(chunk_bytes=4096, min_chunks=0)
    big = pool.acquire(8192)
    small = pool.acquire(4096)
    big.release()  # 空闲: [8K]
    small.release()  # 空闲: [8K, 4K]

    got = pool.acquire(1024)
    assert got.capacity == 4096, "best-fit 取 4K，而它在列表第二位"
    got.release()


def test_request_larger_than_chunk_rounds_up() -> None:
    pool = PinnedBufferPool(chunk_bytes=4096, min_chunks=0)
    buffer = pool.acquire(9000)

    assert buffer.capacity == 12288, "按 chunk_bytes 向上取整"
    assert buffer.requested == 9000
    buffer.release()


def test_released_buffer_becomes_invalid() -> None:
    pool = PinnedBufferPool(chunk_bytes=4096)
    buffer = pool.acquire(64)
    assert buffer.valid and not buffer.released
    assert len(buffer) == 4096

    buffer.release()

    assert buffer.released and not buffer.valid
    with pytest.raises(RuntimeError):
        _ = buffer.view  # 池可能已把内存交给下一次快照，不能静默读到脏数据


def test_double_release_raises() -> None:
    pool = PinnedBufferPool(chunk_bytes=4096)
    buffer = pool.acquire(64)
    buffer.release()

    with pytest.raises(PoolError):
        buffer.release()
    with pytest.raises(PoolError):
        pool.release(buffer)


def test_exhausted_pool_fails_fast_with_zero_timeout() -> None:
    pool = PinnedBufferPool(chunk_bytes=4096, min_chunks=1, max_bytes=4096, timeout=0)
    first = pool.acquire(4096)

    with pytest.raises(PoolExhaustedError):
        pool.acquire(1)  # 没有空闲块，也不能扩容
    assert pool.stats.timeouts == 1

    first.release()
    assert pool.has_free()


def test_acquire_waits_for_pending_release() -> None:
    """反压点：没有可用块时 acquire 阻塞，等别人归还。"""
    pool = PinnedBufferPool(chunk_bytes=4096, min_chunks=1, max_bytes=4096, timeout=5.0)
    held = pool.acquire(4096)

    released_at: list[float] = []

    def release_after_delay() -> None:
        time.sleep(0.05)
        released_at.append(time.monotonic())
        held.release()

    worker = threading.Thread(target=release_after_delay)
    worker.start()
    try:
        buffer = pool.acquire(4096)  # 阻塞到 worker 归还
    finally:
        worker.join()

    assert released_at, "worker 应该已经归还"
    assert pool.stats.waits == 1
    buffer.release()


def test_has_free_reflects_capacity() -> None:
    pool = PinnedBufferPool(chunk_bytes=4096, min_chunks=1, max_bytes=4096, timeout=0)
    assert pool.has_free()

    buffer = pool.acquire(4096)
    assert not pool.has_free(), "既没有空闲块，也不能扩容"

    buffer.release()
    assert pool.has_free()


def test_close_drops_free_chunks() -> None:
    pool = PinnedBufferPool(chunk_bytes=4096, min_chunks=2)
    in_use = pool.acquire(1024)

    pool.close()

    assert pool.closed
    assert pool.stats.chunks == 1, "空闲块被丢掉，在用的那块归还时才释放"
    assert pool.stats.capacity_bytes == 4096
    assert not pool.has_free()
    with pytest.raises(PoolError):
        pool.acquire(1024)

    in_use.release()  # 关闭之后归还也要能正常走完
    assert pool.stats.chunks == 0
    assert pool.stats.capacity_bytes == 0


def test_tensor_copy_to_pinned_block(dataset: Path) -> None:
    """Tensor.copy_to 直接把数据落到池里的块上，不额外分配 host 副本。"""
    data = ckpt.load(dataset, device="cuda:0")
    backend = get_backend(data.device)
    pool = PinnedBufferPool(chunk_bytes=1 << 20, min_chunks=1)

    name = data.names()[0]
    tensor = data[name]
    expected = tensor.tobytes()
    backend.stats.reset()

    buffer = pool.acquire(tensor.nbytes)
    try:
        copied = tensor.copy_to(buffer)
        assert copied == tensor.nbytes
        assert buffer.view[:copied] == expected
    finally:
        buffer.release()

    assert backend.stats.device_to_host_copies == 1
    assert backend.stats.device_to_host_bytes == tensor.nbytes
    assert pool.stats.acquisitions == 1 and pool.stats.reuses == 1
    assert pool.stats.allocations == 1, "预分配的那块就够了，不该有新的分配"


def test_pool_drives_snapshot_machine(dataset: Path) -> None:
    """池 + 状态机 + 拷贝 stream 的组合：一次完整快照不阻塞、不重复分配。"""
    data = ckpt.load(dataset, device="cuda:0")
    tensors = list(data.items())[:4]
    total = sum(t.nbytes for _, t in tensors)

    pool = PinnedBufferPool(chunk_bytes=1 << 20, min_chunks=1, max_bytes=1 << 20, timeout=0)
    manager = SnapshotManager(pool)

    trigger = manager.trigger(step=12000, nbytes=total)
    assert trigger.decision is TriggerDecision.START
    snapshot = trigger.snapshot
    buffer = snapshot.buffer

    assert isinstance(buffer, PinnedBuffer)
    assert len(buffer) >= total
    assert not pool.has_free(), "块被这一代占着且不能扩容，门闸必须挡住"
    assert manager.trigger(step=13000, nbytes=total).decision is TriggerDecision.SKIP

    with D2HStream() as stream:
        wait_handles = []
        offset = 0
        for name, tensor in tensors:
            wait_handles.append(stream.submit_copy(tensor, buffer, offset=offset))
            offset += tensor.nbytes
        for handle in wait_handles:
            assert handle.wait(timeout=10) == handle.nbytes
        stream.synchronize()
        assert stream.stats.completed == len(tensors)
        assert stream.stats.bytes == total

    snapshot.d2h_done()
    assert snapshot.state is SnapshotState.PERSISTING

    # 后台"写完"之后提交，块归还，池回到可复用状态
    snapshot.persist_ok()
    snapshot.commit_ok()
    assert snapshot.state is SnapshotState.COMMITTED
    assert buffer.released
    assert pool.has_free()

    again = pool.acquire(total)
    assert again.capacity == buffer.capacity
    assert pool.stats.allocations == 1, "下一代替换时复用同一块，不再分配"
    assert pool.stats.reuses == 2
    again.release()
