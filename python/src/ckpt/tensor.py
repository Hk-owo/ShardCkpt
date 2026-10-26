"""张量对象：数据在 device buffer 里，host 侧只能通过拷贝访问。

与 torch.Tensor 的对应关系：
- ``t.device`` / ``t.dtype`` / ``t.shape``
- ``t.to("cpu")``      设备迁移
- ``t.numpy()``        取 host 副本（真实 GPU 上等价于 D2H 拷贝）
"""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING

import numpy as np

from .backend import DeviceBackend
from .device import Device
from .dtype import numpy_dtype

if TYPE_CHECKING:  # pragma: no cover
    pass


class Tensor:
    """一个位于 device 上的张量。

    buffer 由 backend 分配，生命周期跟随本对象；调用方拿不到裸指针，
    只能通过 ``tobytes()`` / ``numpy()`` 取 host 副本，这样真实 GPU 实现
    与 mock 实现的调用方式完全一致。
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
        """device -> host 的原始字节拷贝。"""
        host = bytearray(self.nbytes)
        self._backend.copy_d2h(memoryview(host), self._buffer)
        self._backend.synchronize()
        return bytes(host)

    def numpy(self) -> np.ndarray:
        """返回 host 上的数组副本（可写），不共享 device 内存。"""
        raw = np.frombuffer(self.tobytes(), dtype=numpy_dtype(self._dtype))
        return raw.reshape(self._shape).copy()

    def to(self, device: "str | Device") -> "Tensor":
        """迁移到另一个设备；同设备直接返回自身。"""
        target = Device.parse(device)
        if target == self._device:
            return self

        from .backend import get_backend  # 延迟导入，避免循环依赖

        host = bytearray(self.nbytes)
        self._backend.copy_d2h(memoryview(host), self._buffer)
        self._backend.synchronize()

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
