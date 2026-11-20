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
import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core import CheckpointError, FormatError


class StorageError(CheckpointError):
    """存储层失败：网络、鉴权、IO，或后端返回了不符合预期的响应。"""

    def __init__(self, message: str = "", *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)


class TransientStorageError(StorageError):
    """可重试的瞬态存储错误（断流、超时、5xx、传输校验错）。"""

    def __init__(self, message: str = "") -> None:
        super().__init__(message, retryable=True)


def is_retryable_storage_error(exc: BaseException) -> bool:
    """判断异常是否属于可对单文件重试的瞬态传输错误。"""
    if isinstance(exc, (ObjectNotFoundError, ConditionalWritesDisabled, UnsupportedConditionalWrites, FormatError)):
        return False
    if isinstance(exc, StorageError):
        return bool(getattr(exc, "retryable", False))
    if isinstance(exc, (TimeoutError, ConnectionError, EOFError)):
        return True
    return False


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

    def read_with_meta(self, key: str) -> tuple[bytes, ObjectMeta]:
        """整对象读取，同时返回对象元数据（``ETag`` 与大小）。

        默认在 :meth:`read` 后按需 :meth:`head`；对象存储等后端可覆盖为单次
        读取直接从响应头取 ``ETag``，避免额外的 ``HEAD`` 往返与竞态窗口。
        """
        payload = self.read(key)
        head = getattr(self, "head", None)
        if callable(head):
            meta = head(key)
            if meta is not None:
                return payload, meta
        return payload, ObjectMeta(key=key, etag="", size=len(payload))

    @abc.abstractmethod
    def read_range(self, key: str, offset: int, length: int) -> bytes:
        """读取 ``[offset, offset + length)`` 区间，长度不符时报错而非静默截断。"""

    def download(self, key: str, path: "str | Path") -> int:
        """把整对象先写临时文件再原子替换到目标路径，返回字节数。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.read(key)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.tmp.", dir=str(target.parent)
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
                fh.flush()
            os.replace(tmp_path, target)
            return len(payload)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

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


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def validate_run_id(run_id: str) -> str:
    """校验训练轮次标识：非空单段目录名，不与 ``step_N`` 冲突。"""
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if run_id in (".", "..") or not _RUN_ID_RE.match(run_id) or run_id.startswith("step_"):
        raise ValueError(
            f"invalid run_id {run_id!r}: must be a single safe segment not starting with 'step_'"
        )
    return run_id


class PrefixedStorage(StorageBackend):
    """按轮次前缀隔离的存储包装器：所有相对 key 自动挂在 ``<prefix>/`` 下。"""

    def __init__(self, inner: StorageBackend, prefix: str) -> None:
        self._inner = inner
        self._prefix = validate_run_id(prefix)

    @property
    def inner(self) -> StorageBackend:
        return self._inner

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def supports_conditional_writes(self) -> bool:
        return bool(getattr(self._inner, "supports_conditional_writes", False))

    def _scoped(self, key: str) -> str:
        return f"{self._prefix}/{key}" if key else self._prefix

    def _strip(self, key: str) -> str:
        head = f"{self._prefix}/"
        return key[len(head) :] if key.startswith(head) else key

    def write(self, key: str, data: bytes) -> None:
        self._inner.write(self._scoped(key), data)

    def read(self, key: str) -> bytes:
        return self._inner.read(self._scoped(key))

    def read_with_meta(self, key: str) -> tuple[bytes, ObjectMeta]:
        payload, meta = self._inner.read_with_meta(self._scoped(key))
        return payload, ObjectMeta(key=key, etag=meta.etag, size=meta.size)

    def read_range(self, key: str, offset: int, length: int) -> bytes:
        return self._inner.read_range(self._scoped(key), offset, length)

    def download(self, key: str, path: "str | Path") -> int:
        return self._inner.download(self._scoped(key), path)

    def rename_or_put(self, src: str, dst: str) -> None:
        self._inner.rename_or_put(self._scoped(src), self._scoped(dst))

    def list(self, prefix: str = "") -> list[str]:
        scoped = self._scoped(prefix.strip("/")) if prefix else self._prefix
        return sorted(self._strip(k) for k in self._inner.list(scoped))

    def delete(self, key: str) -> None:
        self._inner.delete(self._scoped(key))

    def put_if_absent(self, key: str, data: bytes) -> WriteOutcome:
        return getattr(self._inner, "put_if_absent")(self._scoped(key), data)

    def put_if_match(self, key: str, data: bytes, etag: str) -> WriteOutcome:
        return getattr(self._inner, "put_if_match")(self._scoped(key), data, etag)

    def head(self, key: str) -> ObjectMeta | None:
        meta = getattr(self._inner, "head")(self._scoped(key))
        if meta is None:
            return None
        return ObjectMeta(key=key, etag=meta.etag, size=meta.size)

    def write_file(
        self,
        key: str,
        path: "str | Path",
        *,
        if_none_match: bool = False,
        if_match: str | None = None,
    ) -> WriteOutcome:
        return getattr(self._inner, "write_file")(
            self._scoped(key),
            path,
            if_none_match=if_none_match,
            if_match=if_match,
        )

    def ensure_conditional_writes(self) -> Any:
        fn = getattr(self._inner, "ensure_conditional_writes", None)
        return fn() if callable(fn) else None

    def list_multipart_uploads(self, prefix: str = "") -> list[Any]:
        fn = getattr(self._inner, "list_multipart_uploads", None)
        if not callable(fn):
            return []
        scoped = self._scoped(prefix.strip("/")) if prefix else self._prefix
        return [replace(u, key=self._strip(u.key)) for u in fn(scoped)]

    def abort_multipart_upload(self, key: str, upload_id: str) -> None:
        fn = getattr(self._inner, "abort_multipart_upload", None)
        if callable(fn):
            fn(self._scoped(key), upload_id)

    def path_for(self, key: str) -> Path:
        fn = getattr(self._inner, "path_for", None)
        if not callable(fn):
            raise AttributeError(f"{type(self._inner).__name__} has no path_for")
        return fn(self._scoped(key))

    def close(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"PrefixedStorage({self._inner!r}, prefix={self._prefix!r})"


def run_scope(storage: StorageBackend, run_id: str | None) -> StorageBackend:
    """若给定 ``run_id`` 则包装为轮次前缀作用域；为 ``None`` 时原样返回。"""
    if run_id is None:
        return storage
    if isinstance(storage, PrefixedStorage) and storage.prefix == run_id:
        return storage
    return PrefixedStorage(storage, run_id)


def unwrap_storage(storage: StorageBackend) -> StorageBackend:
    while isinstance(storage, PrefixedStorage):
        storage = storage.inner
    return storage


__all__ = [
    "ConditionalBackend",
    "ConditionalWritesDisabled",
    "ObjectMeta",
    "ObjectNotFoundError",
    "PrefixedStorage",
    "StorageBackend",
    "StorageError",
    "TransientStorageError",
    "UnsupportedConditionalWrites",
    "WriteOutcome",
    "is_retryable_storage_error",
    "run_scope",
    "unwrap_storage",
    "validate_run_id",
]
