"""设备内存后端。

这是「接口与真实 GPU 读取一致」的关键：读取路径只依赖 DeviceBackend 的
allocate / copy_h2d / copy_d2h / synchronize，不关心底层是显存还是普通内存。

- CpuBackend      真实 host 内存
- MockCudaBackend 用 host 内存模拟显存，供无 GPU 环境跑通完整读取路径
- 将来接入真机：实现一个 CudaBackend 并 register_backend() 覆盖 "cuda" 即可，
  调用方代码与测试代码都不用改。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

from .device import Device
from .errors import UnsupportedDeviceError


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
