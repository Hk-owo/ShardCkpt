"""存储后端：本地盘 / 并行文件系统 / 对象存储，统一接口见 base.py。

对象存储后端依赖 boto3，用 ``__getattr__`` 延迟导入：不碰 S3 的调用方
（例如只读本地盘的 ``api.load``）不需要装 boto3。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from storage.base import ObjectNotFoundError, StorageBackend, StorageError

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型检查
    from storage.object import S3Config, S3Storage

_LAZY = {"S3Config", "S3Storage"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from storage import object as _object

        return getattr(_object, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ObjectNotFoundError",
    "S3Config",
    "S3Storage",
    "StorageBackend",
    "StorageError",
]
