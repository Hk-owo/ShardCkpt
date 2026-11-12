"""CRC-32C（Castagnoli）校验和：本地层分片与小状态的字节级自证。

写路径把每个分片/小状态按 ``crc_chunk_max``（默认 1 MiB）切成块，每块一个
CRC-32C 写进索引的 ``crc32c_chunks``；读路径按同样的块边界切 IO，读一块校验一块。
选 CRC-32C 的理由是它有好用的加速实现——x86 的 SSE4.2 ``crc32`` 指令、ARMv8 的
``crc32c``——存储系统（iSCSI / Btrfs / Ceph / NVMe）普遍用它，``google-crc32c``
直接复用这些指令。

实现分两层：

- 首选 :mod:`google_crc32c`（C 扩展，走硬件指令，几十 GB/s）。它的 C 扩展只接受
  ``bytes`` 与 numpy 数组；``memoryview`` 会先转成底层内存的**只读 ndarray 视图**
  （``np.frombuffer``，不复制数据），只有非连续 buffer 才退回分块复制后累加。
- 缺依赖时退回纯 Python 查表。它只保证正确性，速度低一个量级，是「依赖没装好也
  要能跑通」的兜底，不是生产路径。

两个容易踩的坑，这里定死：

- **算法别混**：``zlib.crc32`` 是 IEEE CRC-32，与 CRC-32C 的多项式不同（前者
  ``0xEDB88320``，后者 ``0x82F63B78`` 的反射形式），两者结果**不通用**。
- **空数据**：CRC-32C 的全零初值 + 末尾取反使空输入的值为 0，与 S3 的 ``CRC32C``
  语义一致，不用特判。
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no branch - 缺依赖时整个模块走回退路径
    from google_crc32c import extend as _extend
    from google_crc32c import value as _value
except ImportError:  # pragma: no cover - 依赖缺失才走这里
    _value = None  # type: ignore[assignment]
    _extend = None  # type: ignore[assignment]

CRC32C_POLY = 0x82F63B78
"""CRC-32C 的反射多项式（Castagnoli），用于纯 Python 回退的查表生成。"""

_CHUNK_BYTES = 4 << 20
"""非连续 buffer 回退路径的分块大小：够大以摊薄复制开销，够小不显著抬高峰值内存。"""


def _build_table() -> tuple[int, ...]:
    table = []
    for index in range(256):
        crc = index
        for _ in range(8):
            crc = (crc >> 1) ^ CRC32C_POLY if crc & 1 else crc >> 1
        table.append(crc)
    return tuple(table)


_TABLE = _build_table()


def _pure_crc32c(data: Any) -> int:
    """纯 Python 查表实现（回退路径）。"""
    view = memoryview(data)
    if view.format != "B" or view.ndim != 1:
        view = view.cast("B")
    crc = 0xFFFFFFFF
    table = _TABLE
    for byte in view:
        crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def crc32c(data: Any) -> int:
    """算 ``data`` 的 CRC-32C，返回 32 位无符号整数。

    ``data`` 可以是 ``bytes``、numpy 数组（含切片视图）、``memoryview`` 或任意
    支持 buffer 协议的对象。``memoryview`` 与大块 ndarray 都是零拷贝路径。
    """
    if _value is None:  # pragma: no cover - 依赖缺失才走这里
        return _pure_crc32c(data)

    try:
        return _value(data) & 0xFFFFFFFF
    except TypeError:
        # google-crc32c 的 C 扩展只吃 bytes / numpy 数组，memoryview 走这里。
        # np.frombuffer 得到的是底层内存的**只读视图**，不复制数据。
        try:
            return _value(np.frombuffer(memoryview(data), dtype=np.uint8)) & 0xFFFFFFFF
        except (TypeError, ValueError):
            return _chunked_crc32c(data)


def _chunked_crc32c(data: Any) -> int:  # pragma: no cover - 非连续 buffer 才走
    """非连续 / 无法零拷贝的 buffer：分块复制后增量累加。"""
    view = memoryview(data)
    if view.format != "B" or view.ndim != 1:
        view = view.cast("B")
    crc = 0
    for start in range(0, len(view), _CHUNK_BYTES):
        crc = _extend(crc, bytes(view[start : start + _CHUNK_BYTES]))
    return crc & 0xFFFFFFFF


def crc32c_hex(data: Any) -> str:
    """``crc32c`` 的 8 位小写十六进制形式，即索引里 ``crc32c_chunks`` 的写法。"""
    return f"{crc32c(data):08x}"


# --- 分块 -------------------------------------------------------------------

CRC_CHUNK_MIN = 512 << 10
"""CRC 块的目标下限：512 KiB。

它只描述「大分片切块后的目标区间」；``length <= max_bytes`` 的分片整片一块，
所以小分片的实际块长会小于这个值。读路径不重新推导块大小，直接用索引里的值。
"""

CRC_CHUNK_MAX = 1 << 20
"""CRC 块的默认上限：1 MiB，可通过 ``save_local(crc_chunk_max=...)`` 覆盖。

块数由分片长度推导：``blocks = ceil(length / max_bytes)``，再令
``chunk = ceil(length / blocks)``。这样最后一块不会出现「只剩几 KB」的截断——
尾块长度总是接近块大小，而不是固定块长除不尽剩下的零头。
"""


def _byte_view(data: Any) -> memoryview:
    """把任意 buffer 变成一维字节视图；非连续视图退回一次复制。"""
    view = memoryview(data)
    if view.format == "B" and view.ndim == 1 and view.contiguous:
        return view
    try:
        return view.cast("B")
    except TypeError:  # pragma: no cover - 非连续 buffer 才走这里
        return memoryview(bytes(data))


def crc_chunk_bytes(
    length: int,
    *,
    itemsize: int = 1,
    max_bytes: int = CRC_CHUNK_MAX,
) -> int:
    """选一个 CRC 块大小：``<= max_bytes`` 整片一块，更大时均匀切块。

    ``itemsize`` 让块边界落在元素边界上（``chunk`` 是 ``itemsize`` 的整数倍），
    这样读路径按块切分时不会把元素劈成两半。``chunk`` 最多比 ``max_bytes``
    大 ``itemsize - 1`` 字节。
    """
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")
    if itemsize <= 0:
        raise ValueError(f"itemsize must be > 0, got {itemsize}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be > 0, got {max_bytes}")
    if length <= max_bytes:
        return length
    blocks = -(-length // max_bytes)
    elements = -(-length // itemsize)
    elems_per_block = -(-elements // blocks)
    return elems_per_block * itemsize


def crc32c_chunks(
    data: Any,
    *,
    itemsize: int = 1,
    max_bytes: int = CRC_CHUNK_MAX,
) -> tuple[int, tuple[str, ...]]:
    """按 :func:`crc_chunk_bytes` 切块，返回 ``(块大小, 每块 CRC-32C)``。

    算的是 ``data`` 的全部字节，块与块之间不重叠、不遗漏；块数 =
    ``ceil(len(data) / 块大小)``。读路径拿索引里的块大小与块 CRC 逐块比对。
    """
    view = _byte_view(data)
    length = len(view)
    chunk = crc_chunk_bytes(length, itemsize=itemsize, max_bytes=max_bytes)
    chunks = tuple(
        crc32c_hex(view[start : start + chunk]) for start in range(0, length, chunk)
    )
    return chunk, chunks


__all__ = [
    "CRC32C_POLY",
    "CRC_CHUNK_MAX",
    "CRC_CHUNK_MIN",
    "crc32c",
    "crc32c_chunks",
    "crc32c_hex",
    "crc_chunk_bytes",
]
