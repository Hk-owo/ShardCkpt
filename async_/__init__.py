"""设备内存与拷贝后端：pinned buffer 池、D2H stream 的对接面。

这是「接口与真实 GPU 读取一致」的关键：读取路径只依赖 DeviceBackend 的
allocate / copy_h2d / copy_d2h / synchronize，不关心底层是显存还是普通内存。

- CpuBackend      真实 host 内存
- MockCudaBackend 用 host 内存模拟显存，供无 GPU 环境跑通完整读取路径
- 将来接入真机：实现一个 CudaBackend 并 register_backend() 覆盖 "cuda" 即可，
  调用方代码与测试代码都不用改。

Tensor 的 buffer 由 backend 分配，调用方拿不到裸指针，只能通过 ``tobytes()`` /
``numpy()`` 取 host 副本，这样真实 GPU 实现与 mock 实现的调用方式完全一致。
"""

from __future__ import annotations

import abc
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from async_.machine import (
    BufferPool,
    GateBlock,
    GateCollective,
    MachineStats,
    Snapshot,
    SnapshotError,
    SnapshotManager,
    SnapshotState,
    Trigger,
    TriggerDecision,
    decide,
    pack_block,
    unpack_block,
)
from async_.pinned import (
    DEFAULT_CHUNK_BYTES,
    PinnedBuffer,
    PinnedBufferPool,
    PoolStats,
    aligned_allocator,
    close_default_pool,
    configure_default_pool,
    default_pool,
)
from core import UnsupportedDeviceError
from core.metadata import Device, numpy_dtype


@dataclass
class BackendStats:
    """后端累计统计，测试用它断言「数据确实经过了 H2D 拷贝」。"""

    allocations: int = 0
    frees: int = 0
    host_to_device_copies: int = 0
    host_to_device_bytes: int = 0
    device_to_host_copies: int = 0
    device_to_host_bytes: int = 0

    def reset(self) -> None:
        self.allocations = 0
        self.frees = 0
        self.host_to_device_copies = 0
        self.host_to_device_bytes = 0
        self.device_to_host_copies = 0
        self.device_to_host_bytes = 0


class DeviceBackend(abc.ABC):
    """一块设备内存的分配与拷贝。所有实现必须保持调用语义一致。"""

    device_type: str = ""
    label: str = ""

    def __init__(self) -> None:
        self.stats = BackendStats()

    @property
    def is_mock(self) -> bool:
        """是否为「无 GPU 时的模拟实现」。"""
        return False

    @abc.abstractmethod
    def allocate(self, nbytes: int) -> np.ndarray:
        """分配 nbytes 字节，返回 uint8 的 device buffer（mock 下是 host 内存）。"""

    @abc.abstractmethod
    def free(self, buffer: np.ndarray) -> None:
        """释放 allocate 返回的 buffer。"""

    @abc.abstractmethod
    def copy_h2d(self, dst: np.ndarray, src: memoryview) -> None:
        """host -> device。真实实现是 cudaMemcpy(HostToDevice)，mock 是 memcpy。"""

    @abc.abstractmethod
    def copy_d2h(self, dst: memoryview, src: np.ndarray) -> None:
        """device -> host。"""

    @abc.abstractmethod
    def synchronize(self) -> None:
        """等待设备侧拷贝完成。mock 是空操作。"""


class _NumpyBackend(DeviceBackend):
    """用 numpy 实现的后端骨架：CPU 与 mock CUDA 共用。"""

    def allocate(self, nbytes: int) -> np.ndarray:
        if nbytes < 0:
            raise ValueError(f"nbytes must be >= 0, got {nbytes}")
        self.stats.allocations += 1
        return np.zeros(nbytes, dtype=np.uint8)

    def free(self, buffer: np.ndarray) -> None:
        self.stats.frees += 1
        del buffer

    def copy_h2d(self, dst: np.ndarray, src: memoryview) -> None:
        if len(src) != dst.nbytes:
            raise ValueError(f"size mismatch: dst={dst.nbytes} bytes, src={len(src)} bytes")
        np.copyto(dst, np.frombuffer(src, dtype=np.uint8))
        self.stats.host_to_device_copies += 1
        self.stats.host_to_device_bytes += len(src)

    def copy_d2h(self, dst: memoryview, src: np.ndarray) -> None:
        if len(dst) != src.nbytes:
            raise ValueError(f"size mismatch: dst={len(dst)} bytes, src={src.nbytes} bytes")
        np.copyto(np.frombuffer(dst, dtype=np.uint8), src)
        self.stats.device_to_host_copies += 1
        self.stats.device_to_host_bytes += src.nbytes

    def synchronize(self) -> None:
        return None


class CpuBackend(_NumpyBackend):
    device_type = "cpu"
    label = "cpu"


class MockCudaBackend(_NumpyBackend):
    """无 GPU 环境下的 cuda 后端：内存仍是 host 内存，但走完整的设备拷贝路径。"""

    device_type = "cuda"
    label = "mock-cuda"

    # 模拟真实机器上的卡数量约束
    device_count = 8

    @property
    def is_mock(self) -> bool:
        return True


_registry: dict[str, DeviceBackend] = {}


def register_backend(backend: DeviceBackend) -> None:
    """注册/覆盖某类设备的 backend。接入真机 CUDA 时用它覆盖 "cuda"。"""
    _registry[backend.device_type] = backend


def get_backend(device: Device) -> DeviceBackend:
    try:
        backend = _registry[device.type]
    except KeyError:
        raise UnsupportedDeviceError(
            f"no backend registered for device type {device.type!r}; "
            f"available: {sorted(_registry)}"
        ) from None
    if backend.device_type == "cuda" and isinstance(backend, MockCudaBackend):
        if device.index >= backend.device_count:
            raise UnsupportedDeviceError(
                f"{device} out of range: backend exposes {backend.device_count} devices"
            )
    return backend


def available_device_types() -> list[str]:
    return sorted(_registry)


def reset_registry() -> None:
    """测试用：恢复默认（cpu + mock cuda）。"""
    _registry.clear()
    register_backend(CpuBackend())
    register_backend(MockCudaBackend())


reset_registry()


class Tensor:
    """一个位于 device 上的张量。

    与 torch.Tensor 的对应关系：
    - ``t.device`` / ``t.dtype`` / ``t.shape``
    - ``t.to("cpu")``      设备迁移
    - ``t.numpy()``        取 host 副本（真实 GPU 上等价于 D2H 拷贝）
    """

    __slots__ = ("_backend", "_buffer", "_device", "_dtype", "_name", "_shape")

    def __init__(
        self,
        *,
        name: str,
        buffer: np.ndarray,
        device: Device,
        dtype: str,
        shape: tuple[int, ...],
        backend: DeviceBackend,
    ) -> None:
        self._name = name
        self._buffer = buffer
        self._device = device
        self._dtype = dtype
        self._shape = tuple(shape)
        self._backend = backend

    @property
    def name(self) -> str:
        return self._name

    @property
    def device(self) -> Device:
        return self._device

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def numel(self) -> int:
        return int(np.prod(self._shape, dtype=np.int64)) if self._shape else 1

    @property
    def nbytes(self) -> int:
        return self._buffer.nbytes

    def tobytes(self) -> bytes:
        """device -> host 的原始字节拷贝（走 ``copy_to_host``，与搬运算子同一路径）。"""
        return copy_to_host(self._buffer, device=self._device, backend=self._backend)

    def copy_to(self, buffer: Any, *, offset: int = 0) -> int:
        """把整块数据 D2H 到调用方给的 host buffer（从 ``offset`` 起），返回字节数。

        与 ``tobytes()`` 走同一个 ``copy_into``，区别只是不额外分配：pinned 池的块
        直接当落点，省掉一次 host 侧拷贝。
        """
        return copy_into(
            self._buffer, buffer, offset=offset, device=self._device, backend=self._backend
        )

    def numpy(self) -> np.ndarray:
        """返回 host 上的数组副本（可写），不共享 device 内存。"""
        raw = np.frombuffer(self.tobytes(), dtype=numpy_dtype(self._dtype))
        return raw.reshape(self._shape).copy()

    def to(self, device: "str | Device") -> "Tensor":
        """迁移到另一个设备；同设备直接返回自身。"""
        target = Device.parse(device)
        if target == self._device:
            return self

        host = copy_to_host(self._buffer, device=self._device, backend=self._backend)

        dst_backend = get_backend(target)
        dst_buffer = dst_backend.allocate(self.nbytes)
        dst_backend.copy_h2d(dst_buffer, memoryview(host))
        dst_backend.synchronize()

        return Tensor(
            name=self._name,
            buffer=dst_buffer,
            device=target,
            dtype=self._dtype,
            shape=self._shape,
            backend=dst_backend,
        )

    def crc32(self) -> int:
        return zlib.crc32(self.tobytes()) & 0xFFFFFFFF

    def __len__(self) -> int:
        return self._shape[0] if self._shape else 0

    def __repr__(self) -> str:
        shape = "x".join(str(d) for d in self._shape) if self._shape else "scalar"
        return f"Tensor({self._name!r}, dtype={self._dtype}, shape={shape}, device={self._device})"


__all__ = [
    "BackendStats",
    "BufferPool",
    "CopyHandle",
    "CpuBackend",
    "D2HStream",
    "DEFAULT_CHUNK_BYTES",
    "DeviceBackend",
    "GateBlock",
    "GateCollective",
    "MachineStats",
    "MockCudaBackend",
    "PinnedBuffer",
    "PinnedBufferPool",
    "PoolStats",
    "Snapshot",
    "SnapshotError",
    "SnapshotManager",
    "SnapshotState",
    "StreamStats",
    "Tensor",
    "Trigger",
    "TriggerDecision",
    "aligned_allocator",
    "available_device_types",
    "close_default_pool",
    "configure_default_pool",
    "copy_into",
    "copy_to_host",
    "decide",
    "default_pool",
    "get_backend",
    "pack_block",
    "register_backend",
    "reset_registry",
    "unpack_block",
]

# 放在文件末尾导入：d2h 依赖本模块的 get_backend，提前导入会撞上未初始化的循环。
from async_.d2h import (  # noqa: E402
    CopyHandle,
    D2HStream,
    StreamStats,
    copy_into,
    copy_to_host,
)
