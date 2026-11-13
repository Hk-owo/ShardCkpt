"""存储后端统一接口：write / read / read_range / rename_or_put / list / delete。

所有后端（fs / object）都必须保持这里的调用语义一致；
planner 产出的读写计划只依赖这个接口，不关心底层是本地盘还是对象存储。

约定：

- ``key`` 是相对 checkpoint 根位置（本地目录或对象存储 prefix）的路径，
  形如 ``tensors/model_embed_tokens_weight.bin``，不以 ``/`` 开头。
- ``write`` 成功返回即表示数据对读者可见（对象存储 PUT 原子性 / fs 的 fsync+rename）。
- 失败统一收敛到 :class:`StorageError` 的子类，调用方不需要认识 boto3 / OSError。

条件写（:class:`ConditionalBackend`）是**写路径的硬前置**：对象存储端的代际与
fence（见 ``docs/generation.md``）依赖 ``If-None-Match`` / ``If-Match`` 的服务端
原子性。没有它的端点是只读的，不做任何降级写入。
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from core import CheckpointError


class StorageError(CheckpointError):
    """存储层失败：网络、鉴权、IO，或后端返回了不符合预期的响应。"""


class ObjectNotFoundError(StorageError):
    """请求的对象不存在。"""


class ConditionalWritesDisabled(StorageError):
    """端点被配置成 ``conditional_writes="off"``（只读）。

    写路径要求 S3 conditional writes；只读端点只允许 read / read_range /
    download / head / list。
    """


class UnsupportedConditionalWrites(StorageError):
    """端点声称支持条件写，探针却发现条件没有被真正执行。

    这是**启动即失败**而不是降级：条件被静默忽略意味着 fence 失效，
    比保存失败危险得多（见 ``docs/generation.md`` 的「端点能力探针」）。
    """


class WriteOutcome(str, enum.Enum):
    """一次条件写的结果。

    四个失败值对调用方是四种不同处置，不能合并成一个布尔：``EXISTS`` 要
    ``head`` 比对判断是不是自己上一轮写的，``CONFLICT`` 要重读重试，
    ``GONE`` 说明对象被删过。
    """

    CREATED = "created"
    """``If-None-Match: *`` 成功：这个 key 是我创建的。"""

    SWAPPED = "swapped"
    """``If-Match: <etag>`` 成功：内容已换掉我读到的那一版。"""

    EXISTS = "exists"
    """``If-None-Match`` 失败（412）：别人先创建了这个 key。"""

    CONFLICT = "conflict"
    """``If-Match`` 条件不满足（412）：ETag 已不是我读到的那个。"""

    GONE = "gone"
    """``If-Match`` 但对象不存在（404）：被删了，或从来没写成。"""

    @property
    def written(self) -> bool:
        """本次调用是否真的改变了远端内容。"""
        return self is WriteOutcome.CREATED or self is WriteOutcome.SWAPPED


@dataclass(frozen=True, slots=True)
class ObjectMeta:
    """对象的版本标识与大小。

    ``etag`` 只作 CAS token：**不作内容校验**。multipart 对象的 ETag 是
    ``<md5-of-part-md5s>-N``，加密时也不是 MD5，客户端一律不解释它。
    """

    key: str
    etag: str
    size: int


@runtime_checkable
class ConditionalBackend(Protocol):
    """条件写协议：对象存储端的代际与 fence 只依赖这四个动作。

    ``etag`` 一律按服务端原样回传（带不带引号都行）：它是不透明的 CAS token。
    """

    supports_conditional_writes: bool

    def put_if_absent(self, key: str, data: bytes) -> WriteOutcome:
        """``If-None-Match: *``：只有 key 不存在时才写入。"""

    def put_if_match(self, key: str, data: bytes, etag: str) -> WriteOutcome:
        """``If-Match: <etag>``：只有当前 ETag 仍是这个值时才替换。"""

    def head(self, key: str) -> "ObjectMeta | None":
        """取对象的 ETag 与大小；对象不存在返回 ``None``。"""

    def write_file(
        self,
        key: str,
        path: "str | Path",
        *,
        if_none_match: bool = False,
        if_match: str | None = None,
    ) -> WriteOutcome:
        """从本地文件写入（大文件流式分片），条件挂在可见化那一步上。"""


class StorageBackend(abc.ABC):
    """读写一个 checkpoint 根位置所需的最小接口。"""

    supports_conditional_writes: bool = False
    """本后端是否实现了 :class:`ConditionalBackend`；本地盘恒为 ``False``。"""

    @abc.abstractmethod
    def write(self, key: str, data: bytes) -> None:
        """整对象写入。"""

    @abc.abstractmethod
    def read(self, key: str) -> bytes:
        """整对象读取。"""

    @abc.abstractmethod
    def read_range(self, key: str, offset: int, length: int) -> bytes:
        """读取 ``[offset, offset + length)`` 区间，长度不符时报错而非静默截断。"""

    def download(self, key: str, path: "str | Path") -> int:
        """把整对象写到本地文件，返回字节数。

        默认实现是「整对象读进内存再落盘」，只适合小对象或测试；对象存储后端覆盖
        成流式写，避免把整个 checkpoint 数据文件读进内存。
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.read(key)
        target.write_bytes(payload)
        return len(payload)

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
    "ConditionalBackend",
    "ConditionalWritesDisabled",
    "ObjectMeta",
    "ObjectNotFoundError",
    "StorageBackend",
    "StorageError",
    "UnsupportedConditionalWrites",
    "WriteOutcome",
]
