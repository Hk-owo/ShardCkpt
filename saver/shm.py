"""共享锁页内存数据面：``memfd_create`` + ``mmap`` + 区间分配池（§4）。

真机 ``cudaHostRegister(ptr, size, cudaHostRegisterMapped)`` 待验项说明：
- 当前无 GPU 环境（``MockCudaBackend``）下 ``pinned=False``；
- 若启用了 ``cuda_register=True``（或显式提供真机 ``host_registrar``）但设备注册失败，
  必须直接抛 :class:`core.UnsupportedDeviceError`，绝不静默降级。
"""

from __future__ import annotations

import ctypes
import mmap
import os
import threading
from dataclasses import replace
from typing import Any, Callable

from async_.machine import BufferPool
from async_.pinned import DEFAULT_CHUNK_BYTES, PoolStats
from core import PoolError, PoolExhaustedError, UnsupportedDeviceError
from core.metadata import Device
from storage.local import DEFAULT_ALIGNMENT, _address_of

HostRegisterFn = Callable[[int, int], bool]
HostUnregisterFn = Callable[[int], None]


def _align_up(nbytes: int, alignment: int) -> int:
    if nbytes <= 0:
        return alignment
    return -(-nbytes // alignment) * alignment


def _default_cuda_register(ptr: int, size: int) -> bool:
    """检查当前 CUDA 后端是否支持对外部 memfd 内存执行锁页注册。"""
    from async_ import get_backend

    backend = get_backend(Device.parse("cuda:0"))
    if getattr(backend, "is_mock", False):
        return False
    register_fn = getattr(backend, "register_host_memory", None)
    if not callable(register_fn):
        raise UnsupportedDeviceError(
            f"{type(backend).__name__}: 真机 CUDA 后端未提供 register_host_memory()，"
            "无法对 memfd 执行 cudaHostRegister，拒绝静默降级"
        )
    ok = register_fn(ptr, size)
    if not ok:
        raise UnsupportedDeviceError(
            f"cudaHostRegister(ptr=0x{ptr:x}, size={size}) 失败，拒绝静默降级"
        )
    return True


def _default_cuda_unregister(ptr: int) -> None:
    from async_ import get_backend

    try:
        backend = get_backend(Device.parse("cuda:0"))
    except UnsupportedDeviceError:
        return
    unregister_fn = getattr(backend, "unregister_host_memory", None)
    if callable(unregister_fn):
        unregister_fn(ptr)


class ShmSegment:
    """由 ``memfd_create`` + ``mmap(MAP_SHARED)`` 承载的一段跨进程共享内存。"""

    def __init__(
        self,
        capacity: int,
        *,
        alignment: int = DEFAULT_ALIGNMENT,
        name: str = "ckpt-pinned",
        cuda_register: bool = False,
        host_registrar: HostRegisterFn | None = None,
        host_unregistrar: HostUnregisterFn | None = None,
    ) -> None:
        if alignment < 1:
            raise ValueError(f"alignment must be >= 1, got {alignment}")
        if capacity < alignment:
            capacity = alignment
        self.alignment = alignment
        self.name = name
        self._cuda_register = bool(cuda_register)
        self._host_registrar = host_registrar
        self._host_unregistrar = host_unregistrar
        self._fd: int = -1
        self._mmap: mmap.mmap | None = None
        self._view: memoryview | None = None
        self._capacity = 0
        self._address: int = 0
        self._pinned = False
        self._segment_id = 0
        self._closed = False
        self._allocate(_align_up(capacity, alignment))

    @property
    def fd(self) -> int:
        if self._closed or self._fd < 0:
            raise PoolError("ShmSegment is closed")
        return self._fd

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def address(self) -> int:
        return self._address

    @property
    def pinned(self) -> bool:
        return self._pinned

    @property
    def segment_id(self) -> int:
        return self._segment_id

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def view(self) -> memoryview:
        if self._closed or self._view is None:
            raise PoolError("ShmSegment is closed")
        return self._view

    def ensure(self, nbytes: int) -> None:
        """若当前容量足够直接返回；否则原地重建（调用方需保证无在途块）。"""
        if nbytes <= self._capacity:
            return
        self.reallocate(nbytes)

    def reallocate(self, nbytes: int) -> None:
        """释放旧 memfd 并按新容量重新分配（仅允许在 ``in_use == 0`` 时调用）。"""
        if self._closed:
            raise PoolError("ShmSegment is closed")
        target = _align_up(max(nbytes, self.alignment), self.alignment)
        self._release_mapping()
        self._allocate(target)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._release_mapping()

    def _allocate(self, capacity: int) -> None:
        fd = os.memfd_create(self.name, os.MFD_CLOEXEC)
        mm: mmap.mmap | None = None
        mv: memoryview | None = None
        try:
            os.ftruncate(fd, capacity)
            mm = mmap.mmap(fd, capacity, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            mv = memoryview(mm)
            address = _address_of(mv)
            if address is None or address % self.alignment != 0:
                raise PoolError(
                    f"memfd mmap base address {address} is not {self.alignment}-aligned"
                )
            pinned = False
            if self._host_registrar is not None:
                try:
                    ok = bool(self._host_registrar(address, capacity))
                except UnsupportedDeviceError:
                    raise
                except Exception as exc:
                    raise UnsupportedDeviceError(
                        f"cudaHostRegister 失败：{exc}"
                    ) from exc
                if not ok:
                    raise UnsupportedDeviceError(
                        f"cudaHostRegister(ptr=0x{address:x}, size={capacity}) 返回失败，拒绝降级"
                    )
                pinned = True
            elif self._cuda_register:
                pinned = _default_cuda_register(address, capacity)

            self._fd = fd
            self._mmap = mm
            self._view = mv
            self._capacity = capacity
            self._address = address
            self._pinned = pinned
            self._segment_id += 1
        except BaseException:
            if mv is not None:
                mv.release()
            if mm is not None:
                mm.close()
            os.close(fd)
            raise

    def _release_mapping(self) -> None:
        if self._pinned and self._address:
            try:
                if self._host_unregistrar is not None:
                    self._host_unregistrar(self._address)
                elif self._cuda_register:
                    _default_cuda_unregister(self._address)
            except Exception:
                pass
        self._pinned = False
        self._address = 0
        if self._view is not None:
            try:
                self._view.release()
            except Exception:
                pass
            self._view = None
        if self._mmap is not None:
            try:
                self._mmap.close()
            except BufferError:
                pass
            self._mmap = None
        if self._fd >= 0:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = -1


class ShmBuffer:
    """``ShmSegment`` 内的一块对齐切片，对应一次快照占用的区间 ``[offset, offset + length)``。"""

    __slots__ = (
        "_pool",
        "_view",
        "segment_id",
        "offset",
        "capacity",
        "requested",
        "pinned",
        "_released",
    )

    def __init__(
        self,
        *,
        pool: "ShmPinnedPool",
        segment_id: int,
        offset: int,
        capacity: int,
        requested: int,
        view: memoryview,
        pinned: bool,
    ) -> None:
        self._pool = pool
        self.segment_id = segment_id
        self.offset = offset
        self.capacity = capacity
        self.requested = requested
        self._view: memoryview | None = view
        self.pinned = pinned
        self._released = False

    @property
    def nbytes(self) -> int:
        return self.capacity

    @property
    def released(self) -> bool:
        return self._released

    @property
    def valid(self) -> bool:
        return not self._released

    @property
    def view(self) -> memoryview:
        if self._released or self._view is None:
            raise RuntimeError("ShmBuffer 已归还，不能继续访问")
        return self._view

    def release(self) -> None:
        self._pool.release(self)

    def _invalidate(self) -> None:
        self._released = True
        if self._view is not None:
            try:
                self._view.release()
            except Exception:
                pass
            self._view = None

    def __len__(self) -> int:
        return self.capacity

    def __enter__(self) -> "ShmBuffer":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.release()
        return False

    def __repr__(self) -> str:
        state = "released" if self._released else "in-use"
        return (
            f"ShmBuffer(seg={self.segment_id}, offset={self.offset}, "
            f"capacity={self.capacity}, requested={self.requested}, {state})"
        )


class ShmPinnedPool(BufferPool):
    """基于单个可扩容 ``ShmSegment`` 的块区间池。

    异步路径绝不在有在途代时重建共享内存，也不做 ``trim()``；仅在 ``in_use == 0``
    且容量不足或跨过 1.5x 阈值时整体 ``reallocate``。
    """

    def __init__(
        self,
        *,
        capacity: int = DEFAULT_CHUNK_BYTES,
        max_bytes: int | None = None,
        alignment: int = DEFAULT_ALIGNMENT,
        cuda_register: bool = False,
        host_registrar: HostRegisterFn | None = None,
        host_unregistrar: HostUnregisterFn | None = None,
    ) -> None:
        if alignment < 1:
            raise ValueError(f"alignment must be >= 1, got {alignment}")
        aligned_cap = _align_up(max(capacity, alignment), alignment)
        if max_bytes is not None and max_bytes < aligned_cap:
            raise ValueError(
                f"max_bytes ({max_bytes}) must be >= initial capacity ({aligned_cap})"
            )
        self.alignment = alignment
        self.max_bytes = max_bytes
        self._cond = threading.Condition()
        self._segment = ShmSegment(
            aligned_cap,
            alignment=alignment,
            cuda_register=cuda_register,
            host_registrar=host_registrar,
            host_unregistrar=host_unregistrar,
        )
        self._free: list[tuple[int, int]] = [(0, self._segment.capacity)]
        self._in_use: dict[int, ShmBuffer] = {}
        self._last_block_bytes = alignment
        self._closed = False
        self._stats = PoolStats(
            chunks=1,
            capacity_bytes=self._segment.capacity,
            allocations=1,
        )

    @property
    def segment(self) -> ShmSegment:
        return self._segment

    @property
    def segment_id(self) -> int:
        return self._segment.segment_id

    @property
    def fd(self) -> int:
        return self._segment.fd

    @property
    def capacity(self) -> int:
        return self._segment.capacity

    @property
    def stats(self) -> PoolStats:
        with self._cond:
            return replace(self._stats)

    @property
    def closed(self) -> bool:
        return self._closed

    def has_free(self) -> bool:
        with self._cond:
            if self._closed:
                return False
            if self._largest_free() >= self._last_block_bytes:
                return True
            if not self._in_use:
                needed = self._last_block_bytes
                return self.max_bytes is None or needed <= self.max_bytes
            return False

    def ensure_capacity(self, block_bytes: int, *, slots: int = 1) -> bool:
        """在无在途块时确保段容量至少能容纳 ``slots`` 个 ``block_bytes``（必要时 ``reallocate``）。

        返回是否触发了 ``reallocate``（调用方据此决定是否向 worker 重发 memfd）。
        """
        needed_block = _align_up(max(block_bytes, self.alignment), self.alignment)
        with self._cond:
            if self._closed:
                raise PoolError("ShmPinnedPool is closed")
            self._last_block_bytes = needed_block
            desired = needed_block * max(1, slots)
            if self.max_bytes is not None:
                desired = min(desired, self.max_bytes)
            if self._in_use:
                return False
            if self._segment.capacity >= desired:
                return False
            # 只有 in_use == 0 且需要扩容时才 reallocate（1.5x 步进避免频繁重分配）
            target = max(desired, int(self._segment.capacity * 1.5))
            if self.max_bytes is not None:
                target = min(target, self.max_bytes)
            target = _align_up(target, self.alignment)
            if target <= self._segment.capacity:
                return False
            self._reallocate_locked(target)
            return True

    def acquire(self, nbytes: int) -> ShmBuffer:
        if nbytes < 0:
            raise ValueError(f"nbytes must be >= 0, got {nbytes}")
        block_bytes = _align_up(max(nbytes, self.alignment), self.alignment)
        with self._cond:
            if self._closed:
                raise PoolError("ShmPinnedPool is closed")
            self._last_block_bytes = block_bytes
            idx = self._find_best_fit(block_bytes)
            if idx is None:
                if not self._in_use:
                    if self.max_bytes is not None and block_bytes > self.max_bytes:
                        raise PoolExhaustedError(
                            f"请求 {block_bytes} 字节超过 max_bytes={self.max_bytes}"
                        )
                    target = max(block_bytes, int(self._segment.capacity * 1.5))
                    if self.max_bytes is not None:
                        target = min(target, self.max_bytes)
                    target = _align_up(target, self.alignment)
                    self._reallocate_locked(target)
                    idx = self._find_best_fit(block_bytes)
                if idx is None:
                    raise PoolExhaustedError(
                        f"共享内存池没有 >= {block_bytes} 字节的连续空闲块"
                        f"（capacity={self._segment.capacity}, in_use={self._stats.in_use_bytes}）"
                    )

            free_off, free_len = self._free.pop(idx)
            if free_len > block_bytes:
                self._free.append((free_off + block_bytes, free_len - block_bytes))
                self._free.sort()

            if self._stats.acquisitions > 0:
                self._stats.reuses += 1
            self._stats.acquisitions += 1
            self._stats.in_use_bytes += block_bytes
            self._stats.peak_in_use_bytes = max(
                self._stats.peak_in_use_bytes, self._stats.in_use_bytes
            )
            slice_view = self._segment.view[free_off : free_off + block_bytes]
            buf = ShmBuffer(
                pool=self,
                segment_id=self._segment.segment_id,
                offset=free_off,
                capacity=block_bytes,
                requested=nbytes,
                view=slice_view,
                pinned=self._segment.pinned,
            )
            self._in_use[id(buf)] = buf
            return buf

    def release(self, buffer: object) -> None:
        if not isinstance(buffer, ShmBuffer):
            raise TypeError(f"expected ShmBuffer, got {type(buffer).__name__}")
        with self._cond:
            tracked = self._in_use.pop(id(buffer), None)
            if tracked is None:
                raise PoolError("这块 ShmBuffer 不属于本池，或已经归还过")
            buffer._invalidate()
            self._stats.in_use_bytes -= buffer.capacity
            self._stats.releases += 1
            if not self._closed and buffer.segment_id == self._segment.segment_id:
                self._insert_and_coalesce(buffer.offset, buffer.capacity)
            if self._closed and not self._in_use:
                self._segment.close()
                self._stats.capacity_bytes = 0
                self._stats.chunks = 0
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._free.clear()
            if not self._in_use:
                self._segment.close()
                self._stats.capacity_bytes = 0
                self._stats.chunks = 0
            self._cond.notify_all()

    def _find_best_fit(self, nbytes: int) -> int | None:
        best: int | None = None
        for idx, (_off, length) in enumerate(self._free):
            if length >= nbytes and (best is None or length < self._free[best][1]):
                best = idx
        return best

    def _largest_free(self) -> int:
        return max((length for _off, length in self._free), default=0)

    def _insert_and_coalesce(self, offset: int, length: int) -> None:
        self._free.append((offset, length))
        self._free.sort()
        merged: list[tuple[int, int]] = []
        for off, size in self._free:
            if merged and merged[-1][0] + merged[-1][1] == off:
                prev_off, prev_size = merged[-1]
                merged[-1] = (prev_off, prev_size + size)
            else:
                merged.append((off, size))
        self._free = merged

    def _reallocate_locked(self, target_bytes: int) -> None:
        assert not self._in_use
        self._segment.reallocate(target_bytes)
        self._free = [(0, self._segment.capacity)]
        self._stats.allocations += 1
        self._stats.capacity_bytes = self._segment.capacity


__all__ = [
    "HostRegisterFn",
    "HostUnregisterFn",
    "ShmBuffer",
    "ShmPinnedPool",
    "ShmSegment",
]
