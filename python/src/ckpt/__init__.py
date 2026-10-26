"""ckpt：ShardCkpt 数据读取接口。

当前机器没有 GPU，但接口与「从 GPU 读取」完全一致：
指定 device="cuda:0" 时，数据会经 staging -> H2D 拷贝进入 device buffer，
只是底层 backend 换成了 mock 实现。

示例：
    >>> import ckpt
    >>> data = ckpt.load("/tmp/data/step_0", device="cuda:0")
    >>> data["model.embed_tokens.weight"].device
    Device(type='cuda', index=0)
    >>> data.to("cpu")["model.embed_tokens.weight"].numpy().shape
    (1000, 128)
"""

from __future__ import annotations

from .backend import (
    BackendStats,
    CpuBackend,
    DeviceBackend,
    MockCudaBackend,
    available_device_types,
    get_backend,
    register_backend,
    reset_registry,
)
from .device import CPU, Device, cuda
from .errors import (
    CheckpointError,
    ChecksumError,
    FormatError,
    MissingTensorError,
    UnsupportedDeviceError,
)
from .reader import CheckpointData, CheckpointReader, LoadOptions, load
from .tensor import Tensor

__all__ = [
    "CPU",
    "BackendStats",
    "CheckpointData",
    "CheckpointError",
    "CheckpointReader",
    "ChecksumError",
    "CpuBackend",
    "Device",
    "DeviceBackend",
    "FormatError",
    "LoadOptions",
    "MissingTensorError",
    "MockCudaBackend",
    "Tensor",
    "UnsupportedDeviceError",
    "available_device_types",
    "cuda",
    "get_backend",
    "load",
    "register_backend",
    "reset_registry",
]
