"""存储后端统一接口：write / read / read_range / rename_or_put / list / delete。

所有后端（fs / object）都必须保持这里的调用语义一致；
planner 产出的读写计划只依赖这个接口，不关心底层是本地盘还是对象存储。

约定：

- ``key`` 是相对 checkpoint 根位置（本地目录或对象存储 prefix）的路径，
  形如 ``tensors/model_embed_tokens_weight.bin``，不以 ``/`` 开头。
- ``write`` 成功返回即表示数据对读者可见（对象存储 PUT 原子性 / fs 的 fsync+rename）。
- 失败统一收敛到 :class:`StorageError` 的子类，调用方不需要认识 boto3 / OSError。
"""

from __future__ import annotations

import abc

from core import CheckpointError


class StorageError(CheckpointError):
    """存储层失败：网络、鉴权、IO，或后端返回了不符合预期的响应。"""


class ObjectNotFoundError(StorageError):
    """请求的对象不存在。"""


class StorageBackend(abc.ABC):
    """读写一个 checkpoint 根位置所需的最小接口。"""

    @abc.abstractmethod
    def write(self, key: str, data: bytes) -> None:
        """整对象写入。"""

    @abc.abstractmethod
    def read(self, key: str) -> bytes:
        """整对象读取。"""

    @abc.abstractmethod
    def read_range(self, key: str, offset: int, length: int) -> bytes:
        """读取 ``[offset, offset + length)`` 区间，长度不符时报错而非静默截断。"""

    @abc.abstractmethod
    def rename_or_put(self, src: str, dst: str) -> None:
        """让 dst 持有 src 的内容：本地盘是 rename，对象存储是 copy + delete。"""

    @abc.abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """按字典序列出前缀下的 key。"""

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """删除对象；对象不存在视为成功（幂等）。"""

    def __enter__(self) -> "StorageBackend":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False

    def close(self) -> None:
        """释放连接等资源；默认无操作。"""


__all__ = [
    "ObjectNotFoundError",
    "StorageBackend",
    "StorageError",
]
