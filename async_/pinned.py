"""pinned buffer 池：启动时预分配、反复复用，mock 下用普通 host 内存承载。

真机上 pinned（page-locked）内存是异步 DMA 的前提：申请很慢，而且占用不可换页的
物理内存，所以必须预分配 + 复用，不能每次快照现申请。

本机没有 GPU，默认分配器是 ``np.zeros``（普通内存，``PinnedBuffer.pinned == False``）；
接入真机时换掉 ``allocator``（``cudaHostAlloc``）即可，池的语义、统计与状态机的对接
方式都不用改。``aligned_allocator()`` 用 C++ 的 ``AlignedBuffer`` 分配页对齐内存，
让 O_DIRECT 零拷贝写盘成立（numpy 分配的地址不保证 4096 对齐）。

池实现 :class:`async_.machine.BufferPool` 协议：``acquire`` / ``release`` /
``has_free``。``acquire`` 在池空且无法扩容时等待别人归还（``timeout`` 到点抛
:class:`core.PoolExhaustedError`），这正是「下一次快照等上一批 buffer 归还」的那条
反压链。
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from async_.machine import BufferPool
from core import PoolError, PoolExhaustedError

DEFAULT_CHUNK_BYTES = 8 << 20  # 8 MiB


@dataclass
class PoolStats:
    """池的累计统计。``reuses`` 是「没有重新分配」的证据。"""

    chunks: int = 0
    capacity_bytes: int = 0
    in_use_bytes: int = 0
    peak_in_use_bytes: int = 0
    allocations: int = 0
    reuses: int = 0
    acquisitions: int = 0
    releases: int = 0
    waits: int = 0
    timeouts: int = 0


class PinnedBuffer:
    """池里的一块 buffer。有效期由 acquire/release 界定。

    ``capacity`` 是块的字节数（>= 申请值），``requested`` 是当时申请的字节数。
    归还之后 ``valid`` 变 False，再碰 ``view`` 会抛 ``RuntimeError`` —— 池随时可能把
    这块内存交给下一次快照，静默读到脏数据比直接报错危险得多。
    """

    __slots__ = ("_array", "_view", "_pool", "capacity", "requested", "pinned", "_released")

    def __init__(self, array: np.ndarray, requested: int, pool: "PinnedBufferPool") -> None:
        self._array = array
        self._view = memoryview(array)
        self._pool = pool
        self.capacity = int(array.nbytes)
        self.requested = requested
        # numpy 分配的必然是普通内存；真机的 CudaPinnedPool 会把它置为 True。
        self.pinned = False
        self._released = False

    # --- 状态 ---------------------------------------------------------------
    @property
    def released(self) -> bool:
        return self._released

    @property
    def valid(self) -> bool:
        return not self._released

    @property
    def view(self) -> memoryview:
        if self._released:
            raise RuntimeError("buffer 已归还给池，不能继续使用")
        return self._view

    # --- 归还 ---------------------------------------------------------------
    def release(self) -> None:
        self._pool.release(self)

    def __enter__(self) -> "PinnedBuffer":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.release()
        return False

    def __len__(self) -> int:
        return self.capacity

    def __repr__(self) -> str:
        state = "released" if self._released else "in-use"
        return f"PinnedBuffer(capacity={self.capacity}, requested={self.requested}, {state})"


class PinnedBufferPool(BufferPool):
    """固定块大小的 pinned buffer 池。

    ``chunk_bytes``:  每次分配的最小粒度；请求更大时按它的整数倍分配
    ``min_chunks``:   启动时预分配的块数（对应「启动时预分配、反复复用」）
    ``max_bytes``:    总容量上限；达到上限且无空闲块时 acquire 等待，不无限增长
    ``timeout``:      acquire 的等待上限（秒）；``None`` 等到有人归还，``0`` 立即失败
    """

    def __init__(
        self,
        *,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        min_chunks: int = 1,
        max_bytes: int | None = None,
        timeout: float | None = None,
        allocator: Callable[[int], np.ndarray] | None = None,
    ) -> None:
        if chunk_bytes <= 0:
            raise ValueError(f"chunk_bytes must be > 0, got {chunk_bytes}")
        if min_chunks < 0:
            raise ValueError(f"min_chunks must be >= 0, got {min_chunks}")
        if max_bytes is not None and max_bytes < chunk_bytes * min_chunks:
            raise ValueError(
                f"max_bytes={max_bytes} 小于预分配的 {chunk_bytes * min_chunks} 字节"
            )
        if timeout is not None and timeout < 0:
            raise ValueError(f"timeout must be >= 0 or None, got {timeout}")

        self.chunk_bytes = chunk_bytes
        self.min_chunks = min_chunks
        self._allocator = allocator or _default_allocator
        self.max_bytes = max_bytes
        self.timeout = timeout

        self._cond = threading.Condition()
        self._free: list[np.ndarray] = []
        self._in_use: dict[int, np.ndarray] = {}
        self._stats = PoolStats()
        self._closed = False

        for _ in range(min_chunks):
            self._free.append(self._new_chunk(chunk_bytes))

    # --- 查询 ---------------------------------------------------------------
    @property
    def stats(self) -> PoolStats:
        with self._cond:
            return replace(self._stats)

    @property
    def closed(self) -> bool:
        return self._closed

    def has_free(self) -> bool:
        """门闸判定：还有空闲块，或者还能扩容。"""
        with self._cond:
            if self._closed:
                return False
            return bool(self._free) or self._can_grow(self.chunk_bytes)

    # --- 借还 ---------------------------------------------------------------
    def acquire(self, nbytes: int) -> PinnedBuffer:
        """借一块至少 ``nbytes`` 的 buffer（best-fit 复用，不够就扩容）。"""
        if nbytes < 0:
            raise ValueError(f"nbytes must be >= 0, got {nbytes}")
        with self._cond:
            while True:
                if self._closed:
                    raise PoolError("池已关闭")
                chunk = self._best_fit(nbytes)
                if chunk is not None:
                    self._free.remove(chunk)
                    self._stats.reuses += 1
                    return self._take(chunk, nbytes)

                capacity = self._chunk_capacity(nbytes)
                if self._can_grow(capacity):
                    return self._take(self._new_chunk(capacity), nbytes)

                # 没空闲块也不能扩容：等别人归还（反压点）
                self._stats.waits += 1
                if not self._cond.wait(self.timeout):
                    self._stats.timeouts += 1
                    raise PoolExhaustedError(
                        f"池里没有 >= {nbytes} 字节的块，等待 {self.timeout} 秒仍未归还"
                        f"（capacity={self._stats.capacity_bytes}, max={self.max_bytes}）"
                    )

    def release(self, buffer: PinnedBuffer) -> None:
        with self._cond:
            if not isinstance(buffer, PinnedBuffer):
                raise TypeError(f"expected PinnedBuffer, got {type(buffer).__name__}")
            array = self._in_use.pop(id(buffer._array), None)
            if array is None:
                raise PoolError("这块 buffer 不属于本池，或已经归还过")
            buffer._released = True
            self._stats.in_use_bytes -= int(array.nbytes)
            self._stats.releases += 1
            if self._closed:
                self._stats.capacity_bytes -= int(array.nbytes)
                self._stats.chunks -= 1
            else:
                self._free.append(array)
            self._cond.notify()

    # --- 生命周期 -----------------------------------------------------------
    def close(self) -> None:
        """关闭池：丢掉所有空闲块；在用块归还时直接被丢弃。"""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            for array in self._free:
                self._stats.capacity_bytes -= int(array.nbytes)
                self._stats.chunks -= 1
            self._free.clear()
            self._cond.notify_all()

    def __enter__(self) -> "PinnedBufferPool":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        with self._cond:
            return (
                f"PinnedBufferPool(chunk_bytes={self.chunk_bytes}, "
                f"free={len(self._free)}, in_use={len(self._in_use)}, "
                f"capacity={self._stats.capacity_bytes})"
            )

    # --- 内部 ---------------------------------------------------------------
    def _new_chunk(self, capacity: int) -> np.ndarray:
        array = self._allocator(capacity)
        self._stats.allocations += 1
        self._stats.chunks += 1
        self._stats.capacity_bytes += capacity
        return array

    def _chunk_capacity(self, nbytes: int) -> int:
        if nbytes <= self.chunk_bytes:
            return self.chunk_bytes
        return -(-nbytes // self.chunk_bytes) * self.chunk_bytes

    def _can_grow(self, capacity: int) -> bool:
        if self.max_bytes is None:
            return True
        return self._stats.capacity_bytes + capacity <= self.max_bytes

    def _best_fit(self, nbytes: int) -> np.ndarray | None:
        best: np.ndarray | None = None
        for array in self._free:
            if array.nbytes >= nbytes and (best is None or array.nbytes < best.nbytes):
                best = array
        return best

    def _take(self, array: np.ndarray, nbytes: int) -> PinnedBuffer:
        self._in_use[id(array)] = array
        self._stats.in_use_bytes += int(array.nbytes)
        self._stats.peak_in_use_bytes = max(
            self._stats.peak_in_use_bytes, self._stats.in_use_bytes
        )
        self._stats.acquisitions += 1
        return PinnedBuffer(array, requested=nbytes, pool=self)


def _default_allocator(nbytes: int) -> np.ndarray:
    return np.zeros(nbytes, dtype=np.uint8)


# AlignedBuffer 的 owner（native.AlignedView）不能挂在 ndarray 上，用 weakref 表保命：
# 数组被回收时同步丢掉对应的 AlignedView，底层内存随之释放。
_ALIGNED_KEEPALIVE: dict[int, object] = {}


def aligned_allocator(alignment: int = 4096) -> Callable[[int], np.ndarray]:
    """用 C++ 的 ``AlignedBuffer`` 分配页对齐内存（O_DIRECT 零拷贝写盘的前提）。

    ``ckpt_native`` 没构建时退回 ``np.zeros``，行为不变、只是失去对齐保证。
    """
    try:
        import native
    except ImportError:  # pragma: no cover - native 包缺失
        return _default_allocator

    if not native.available():
        return _default_allocator

    def allocate(nbytes: int) -> np.ndarray:
        view = native.aligned_view(max(nbytes, alignment), alignment)
        array = np.frombuffer(view.view, dtype=np.uint8)[:nbytes]
        _ALIGNED_KEEPALIVE[id(array)] = view
        weakref.finalize(array, _ALIGNED_KEEPALIVE.pop, id(array), None)
        return array

    return allocate


__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "PinnedBuffer",
    "PinnedBufferPool",
    "PoolStats",
    "aligned_allocator",
]
