"""本地层存储后端：native（io_uring + O_DIRECT + fsync + rename）或纯 Python 回退。

与 `storage/object.py` 是同一个 `StorageBackend` 接口的两个实现，key 一律相对
``root``：本地层是目录，对象存储是 prefix。后端只做「单文件读写 + 原子 rename」，
分片布局、校验和、提交顺序都留给 planner / commit 层。

两条写入引擎：

- ``engine="native"``：`ckpt_native.write_file`（临时文件 → 顺序 WRITE → fsync →
  rename → 目录 fsync），buffer 地址按 alignment 对齐时走 O_DIRECT，零拷贝；
- ``engine="python"``：`tempfile` + `write` + `fsync` + `os.replace` + 目录 fsync，
  功能等价，只是不绕 page cache。C++ 扩展没构建时自动回退到它。

``write_view()`` 是本地层特有的零拷贝入口：它接受任何实现了 buffer 协议的对象
（pinned 块、`native.AlignedView`、memoryview），不经过 `bytes` 复制。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from core import CheckpointError
from storage.base import ObjectNotFoundError, StorageBackend, StorageError

DEFAULT_ALIGNMENT = 4096
"""O_DIRECT 的块大小；planner 的对齐补齐也用它。"""


def _normalize_key(key: str) -> str:
    """本地 key 的安全校验：拒绝绝对路径、``..``、空段，与对象存储后端一致。"""
    if not isinstance(key, str) or not key:
        raise ValueError(f"invalid key {key!r}")
    if key.startswith("/"):
        raise ValueError(f"key must be relative, got {key!r}")
    parts = key.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"invalid key segment in {key!r}")
    return key


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _address_of(buffer: Any) -> int | None:
    """尽力拿到 buffer 的内存地址；只读 buffer 或非 buffer 对象返回 None。"""
    inner = getattr(buffer, "buffer", None)  # native.AlignedView 持有 C++ AlignedBuffer
    address = getattr(inner, "address", None)
    if isinstance(address, int):
        return address

    target = getattr(buffer, "view", None)
    if isinstance(target, memoryview):
        buffer = target
    try:
        import ctypes

        return ctypes.addressof(ctypes.c_char.from_buffer(buffer))
    except (TypeError, ValueError):
        return None


class LocalStorage(StorageBackend):
    """一个目录作为 checkpoint 根位置的存储后端。

    ``direct_io``: 大块顺序写是否走 O_DIRECT（需要 buffer 按 alignment 对齐）
    ``engine``:   ``"auto"``（默认，native 可用就用）/ ``"native"`` / ``"python"``
    """

    def __init__(
        self,
        root: str | Path,
        *,
        direct_io: bool = True,
        alignment: int = DEFAULT_ALIGNMENT,
        engine: str = "auto",
    ) -> None:
        if engine not in ("auto", "native", "python"):
            raise ValueError(f"unknown engine {engine!r}")
        self._root = Path(root)
        self.direct_io = direct_io
        self.alignment = alignment
        self._engine = self._resolve_engine(engine)

    # --- 基本信息 -----------------------------------------------------------
    @property
    def root(self) -> Path:
        return self._root

    @property
    def engine(self) -> str:
        """实际使用的写入引擎：``"native"`` 或 ``"python"``。"""
        return self._engine

    def path_for(self, key: str) -> Path:
        return self._root / _normalize_key(key)

    @staticmethod
    def _resolve_engine(engine: str) -> str:
        if engine != "auto":
            return engine
        try:
            import native

            return "native" if native.available() else "python"
        except ImportError:  # pragma: no cover - native 包本身缺失
            return "python"

    # --- 生命周期 -----------------------------------------------------------
    def close(self) -> None:
        return None

    def __repr__(self) -> str:
        return (
            f"LocalStorage(root={str(self._root)!r}, engine={self._engine!r}, "
            f"direct_io={self.direct_io})"
        )

    # --- 写 -----------------------------------------------------------------
    def write(self, key: str, data: bytes) -> None:
        """整文件写入。输入是 bytes 时无法保证地址对齐，大块会在对齐缓冲里中转一次。"""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(f"data must be bytes-like, got {type(data).__name__}")
        self.write_view(key, data)

    def write_view(self, key: str, buffer: Any) -> None:
        """零拷贝整文件写入：直接拿调用方 buffer 的字节。

        ``direct_io`` 打开时只有「地址 + 长度都对到 alignment」才走 O_DIRECT；
        否则退回 buffered 写，保证不说谎也不失败。
        """
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        view = _as_view(buffer)
        chunks: list[Any] = [view]
        direct = self.direct_io and self._is_aligned(buffer, view)

        if self._engine == "native":
            self._write_native(path, chunks, direct_io=direct)
        else:
            self._write_python(path, chunks, direct_io=False)

    def _is_aligned(self, buffer: Any, view: memoryview) -> bool:
        if len(view) % self.alignment != 0:
            return False
        address = _address_of(buffer)
        return address is not None and address % self.alignment == 0

    def _write_native(self, path: Path, chunks: list[Any], *, direct_io: bool) -> None:
        import native

        try:
            native.write_file(
                str(path),
                chunks,
                direct_io=direct_io,
                fsync_data=True,
                fsync_dir=True,
                alignment=self.alignment,
            )
        except CheckpointError:
            raise
        except OSError as exc:
            raise StorageError(f"{path}: {exc}") from exc

    def _write_python(self, path: Path, chunks: list[Any], *, direct_io: bool) -> None:
        # direct_io 在纯 Python 引擎下不可用（没有对齐 buffer 保证），忽略即可
        temp = path.with_name(path.name + ".tmp")
        try:
            with open(temp, "wb") as fh:
                for chunk in chunks:
                    fh.write(_as_view(chunk))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temp, path)
            _fsync_dir(path.parent)
        except OSError as exc:
            try:
                temp.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - 清理失败不掩盖原始错误
                pass
            raise StorageError(f"{path}: {exc}") from exc

    # --- 读 -----------------------------------------------------------------
    def read(self, key: str) -> bytes:
        path = self.path_for(key)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise ObjectNotFoundError(f"{key}: not found") from exc
        except OSError as exc:
            raise StorageError(f"{key}: {exc}") from exc

    def read_range(self, key: str, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0:
            raise ValueError(f"offset/length must be >= 0, got {offset}/{length}")
        path = self.path_for(key)
        try:
            fd = os.open(path, os.O_RDONLY)
        except FileNotFoundError as exc:
            raise ObjectNotFoundError(f"{key}: not found") from exc
        except OSError as exc:
            raise StorageError(f"{key}: {exc}") from exc
        try:
            payload = os.pread(fd, length, offset)
        except OSError as exc:
            raise StorageError(f"{key}: {exc}") from exc
        finally:
            os.close(fd)
        if len(payload) != length:
            raise StorageError(
                f"{key}: short read at [{offset}, {offset + length}), got {len(payload)} bytes"
            )
        return payload

    # --- 提交原语 -----------------------------------------------------------
    def rename_or_put(self, src: str, dst: str) -> None:
        """本地盘的 rename 是原子的：同目录直接 rename + 目录 fsync。"""
        if src == dst:
            return
        src_path = self.path_for(src)
        dst_path = self.path_for(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(src_path, dst_path)
            _fsync_dir(dst_path.parent)
        except FileNotFoundError as exc:
            raise ObjectNotFoundError(f"{src}: not found") from exc
        except OSError as exc:
            raise StorageError(f"{src} -> {dst}: {exc}") from exc

    def list(self, prefix: str = "") -> list[str]:
        if prefix:
            prefix = _normalize_key(prefix.rstrip("/"))
        if not self._root.is_dir():
            return []
        keys: list[str] = []
        for path in self._root.rglob("*"):
            if not path.is_file():
                continue
            key = path.relative_to(self._root).as_posix()
            if prefix and not (key == prefix or key.startswith(prefix + "/")):
                continue
            keys.append(key)
        return sorted(keys)

    def delete(self, key: str) -> None:
        path = self.path_for(key)
        try:
            path.unlink(missing_ok=True)  # 幂等，与对象存储后端一致
        except OSError as exc:
            raise StorageError(f"{key}: {exc}") from exc


def _as_view(buffer: Any) -> memoryview:
    if isinstance(buffer, memoryview):
        return buffer
    view = getattr(buffer, "view", None)  # native.AlignedView / PinnedBuffer
    if isinstance(view, memoryview):
        return view
    return memoryview(buffer)


__all__ = ["DEFAULT_ALIGNMENT", "LocalStorage"]
