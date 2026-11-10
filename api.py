"""save_async / load：唯一把 core / planner / storage / async_ 串起来的地方。

对外接口按「从 GPU 读取」设计：

1. 加载时显式指定目标设备（对应 ``torch.load(map_location=...)``）
2. 数据路径固定为  host 文件 -> staging buffer -> H2D 拷贝 -> device buffer
3. 迁移与取值都经过 backend，调用方从不直接触碰 device 内存

示例：
    >>> import api as ckpt
    >>> data = ckpt.load("/tmp/data/step_0", device="cuda:0")
    >>> data["model.embed_tokens.weight"].device
    Device(type='cuda', index=0)
    >>> data.to("cpu")["model.embed_tokens.weight"].numpy().shape
    (1000, 128)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from async_ import (
    BackendStats,
    CpuBackend,
    DeviceBackend,
    MockCudaBackend,
    Tensor,
    available_device_types,
    get_backend,
    register_backend,
    reset_registry,
)
from core import (
    CheckpointError,
    ChecksumError,
    FormatError,
    MissingTensorError,
    UnsupportedDeviceError,
)
from core.metadata import CPU, Device, cuda
from storage.fs import CheckpointData, CheckpointReader, LoadOptions


def load(
    path: str | Path,
    *,
    device: "str | Device" = "cpu",
    verify_crc: bool = True,
    staging: bool = True,
) -> CheckpointData:
    """便捷入口，对应 ``torch.load(path, map_location=device)``。"""
    reader = CheckpointReader.open(
        path, device=device, verify_crc=verify_crc, staging=staging
    )
    return reader.read_all()


def save_async(state_dict: Any, path: str | Path, **options: Any) -> Any:
    """异步保存入口（planner/storage/commit/async_ 尚未实现）。"""
    raise NotImplementedError("save_async is not implemented yet")


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
    "save_async",
]
