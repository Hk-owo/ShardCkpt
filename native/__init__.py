"""ckpt_native（C++ 内核）的加载点。

`ckpt_native` 是 `ckpt_core` 的薄包装（`bindings/python.cpp`），构建产物在
`build/python/` 下，不入库。构建方式：

.. code-block:: bash

    uv pip install --python .venv/bin/python pybind11
    cmake -S . -B build -G Ninja \\
          -DPython3_EXECUTABLE=$PWD/.venv/bin/python \\
          -Dpybind11_DIR=$(uv run --python .venv/bin/python python -m pybind11 --cmakedir)
    cmake --build build

本模块负责把构建目录挂进 ``sys.path`` 再导入扩展，并在没构建时给出可读的报错。
Python 侧只从这里拿 C++ 能力，不要直接 ``import ckpt_native``。
"""

from __future__ import annotations

import ctypes
import importlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
"""仓库根目录；`build/python` 是 CMake 的 LIBRARY_OUTPUT_DIRECTORY。"""

_BUILD_DIRS = (ROOT / "build" / "python",)
"""依次尝试的扩展搜索路径（后续加了 wheel 安装可以再补）。"""

_BUILD_HINT = (
    "ckpt_native 未构建。先装构建期依赖再编译：\n"
    "  uv pip install --python .venv/bin/python pybind11\n"
    f"  cmake -S {ROOT} -B {ROOT / 'build'} -G Ninja "
    f"-DPython3_EXECUTABLE={ROOT / '.venv/bin/python'} "
    "-Dpybind11_DIR=$(uv run --python .venv/bin/python python -m pybind11 --cmakedir)\n"
    f"  cmake --build {ROOT / 'build'}"
)

_MODULE: Any = None
_ERROR: ImportError | None = None


def _ensure_on_path() -> None:
    for candidate in _BUILD_DIRS:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.append(str(candidate))


def _load() -> Any:
    _ensure_on_path()
    try:
        return importlib.import_module("ckpt_native")
    except ImportError as exc:  # 没构建，或 ABI 版本对不上
        raise ImportError(f"{_BUILD_HINT}\n\n原始错误：{exc}") from exc


try:
    _MODULE = _load()
except ImportError as exc:  # 允许纯 Python 环境继续工作（测试会 skip）
    _ERROR = exc


def available() -> bool:
    """扩展是否已构建并能导入。"""
    return _MODULE is not None


def require() -> Any:
    """返回扩展模块；没构建时抛带构建命令的 ImportError。"""
    if _MODULE is None:
        assert _ERROR is not None
        raise _ERROR
    return _MODULE


# --- 便捷封装 ---------------------------------------------------------------


class AlignedView:
    """`AlignedBuffer` 的可写 memoryview 视图（零拷贝）。

    O_DIRECT 写要求 buffer 地址按 alignment 对齐，而 numpy/bytearray 的地址不保证
    对齐；这个类把 C++ 分配的对齐内存包成 memoryview，可以直接当 D2H 落点或
    ``write_file`` 的 chunk 用。**本对象存活期间底层内存才有效**。
    """

    __slots__ = ("buffer", "view")

    def __init__(self, buffer: Any) -> None:
        self.buffer = buffer
        # ctypes 数组的 memoryview 格式是 '<B'，不支持切片赋值；cast 成 'B' 才能当
        # 普通可写字节缓冲区用（C 连续，cast 不需要复制）。
        self.view = memoryview(
            (ctypes.c_ubyte * buffer.size).from_address(buffer.address)
        ).cast("B")

    def __len__(self) -> int:
        return self.buffer.size

    def release(self) -> None:
        self.view.release()

    def __repr__(self) -> str:
        return f"AlignedView(size={self.buffer.size}, address=0x{self.buffer.address:x})"


def aligned_view(nbytes: int, alignment: int = 4096) -> AlignedView:
    """分配一块按 alignment 对齐的内存并包成可写 memoryview。"""
    return AlignedView(require().AlignedBuffer.allocate(nbytes, alignment))


_NAMES = (
    "AlignedBuffer",
    "DEFAULT_ALIGNMENT",
    "DEFAULT_CHUNK_BYTES",
    "DEFAULT_DEPTH",
    "LocalFile",
    "fsync_dir",
    "read_file_ranges",
    "rename_atomic",
    "write_file",
)

__all__ = [
    *sorted(_NAMES),
    "AlignedView",
    "available",
    "aligned_view",
    "require",
]


def __getattr__(name: str) -> Any:
    """把扩展里的符号透出去，这样 `from native import write_file` 也能用。"""
    if name in _NAMES:
        return getattr(require(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
