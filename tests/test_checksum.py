"""CRC-32C 校验和本身：算法对不对、各种 buffer 输入等价、回退实现一致。

这里守两类错误：

- **算法混用**：``crc32c`` 必须是 Castagnoli（``0xE3069283`` 检查值），不是
  ``zlib.crc32`` 的 IEEE CRC-32。历史上索引字段叫 ``crc32c`` 而存的是 zlib 的值，
  这种「名字对了、算法错了」的 bug 用检查值一行就能锁死。
- **输入类型**：写路径给的是 pinned 块（``memoryview``）、读路径给的是 staging 块，
  两者都必须走零拷贝或至少正确的路径，不能因为 buffer 类型不同算出不同结果。
"""

from __future__ import annotations

import base64
import zlib

import numpy as np
import pytest

import s3_stub
from storage import checksum
from storage.checksum import (
    CRC_CHUNK_MAX,
    crc32c,
    crc32c_chunks,
    crc32c_hex,
    crc_chunk_bytes,
)

CHECK_INPUT = b"123456789"
CHECK_VALUE = 0xE3069283
"""CRC-32C 的标准检查值（RFC 3720 / iSCSI 用它做算法自检）。"""


def test_known_check_value() -> None:
    assert crc32c(CHECK_INPUT) == CHECK_VALUE
    assert crc32c_hex(CHECK_INPUT) == "e3069283"


def test_empty_payload_is_zero() -> None:
    """S3 的 CRC32C 语义：空对象校验和为 0，不特判。"""
    assert crc32c(b"") == 0
    assert crc32c_hex(b"") == "00000000"


def test_crc32c_differs_from_zlib_ieee_crc32() -> None:
    """同名不同算法：zlib 的 CRC-32 与 CRC-32C 结果必须不同，防止再混用。"""
    payload = b"checkpoint" * 100
    assert crc32c(payload) != zlib.crc32(payload) & 0xFFFFFFFF


def test_all_buffer_flavours_agree() -> None:
    """bytes / 只读 memoryview / 可写 memoryview / ndarray 及其切片算出的值相同。"""
    payload = bytes(range(256)) * 7
    array = np.frombuffer(payload, dtype=np.uint8)
    writable = bytearray(payload)

    expected = crc32c(payload)
    assert crc32c(memoryview(payload)) == expected
    assert crc32c(memoryview(writable)) == expected
    assert crc32c(array) == expected
    # 切片视图：写路径就是这么用的（一个分片的字节区间）
    assert crc32c(array[100:900]) == crc32c(payload[100:900])


def test_slices_start_at_their_own_offset() -> None:
    """切片必须从自己的起点算：同内容同长度的两段得出同值，但不能等于整块的值。"""
    payload = bytes(range(256)) * 4
    assert crc32c(payload[256:512]) == crc32c(payload[:256])  # 同内容、不同偏移 -> 同值
    assert crc32c(payload[256:512]) != crc32c(payload)  # 但不是把整块算进去


def test_pure_python_fallback_matches_c_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    """依赖缺失时的纯 Python 回退必须与 C 扩展逐位一致（跨分块边界也要一致）。"""
    payloads = [b"", b"a", bytes(range(256)), bytes(range(256)) * 40]
    expected = [crc32c(payload) for payload in payloads]

    monkeypatch.setattr(checksum, "_value", None)
    for payload, want in zip(payloads, expected):
        assert checksum.crc32c(payload) == want, len(payload)


def test_s3_stub_uses_the_same_algorithm() -> None:
    """测试桩里的 CRC32C 是独立实现，这里交叉验证它没漂移。"""
    expected = base64.b64encode(CHECK_VALUE.to_bytes(4, "big")).decode("ascii")
    assert s3_stub.checksum_b64(CHECK_INPUT, "crc32c") == expected


# --- 分块规则 ---------------------------------------------------------------


MiB = 1 << 20


@pytest.mark.parametrize(
    ("length", "chunk", "blocks"),
    [
        (128 << 10, 128 << 10, 1),
        (1 * MiB, 1 * MiB, 1),
        (3 * MiB, 1 * MiB, 3),
        (1 * MiB + (1 << 19), (1 << 19) + (1 << 18), 2),
        (8 * MiB, 1 * MiB, 8),
        (8 * MiB + 1, -(-(8 * MiB + 1) // 9), 9),
        (10 * MiB, 1 * MiB, 10),
        (1 << 30, 1 * MiB, 1024),
    ],
)
def test_chunk_size_rule(length: int, chunk: int, blocks: int) -> None:
    """块数由长度推导，尾块均匀、不截断；<= max_bytes 整片一块。"""
    assert crc_chunk_bytes(length) == chunk
    assert -(-length // chunk) == blocks
    # 最后一块不会退化成几 KB 的零头
    tail = length - (blocks - 1) * chunk
    assert tail >= chunk - blocks
    assert chunk <= CRC_CHUNK_MAX + 1


def test_chunk_size_respects_item_size_alignment() -> None:
    """块边界必须落在元素边界上，否则读路径按字节切会劈开元素。"""
    for itemsize in (2, 4, 8):
        chunk = crc_chunk_bytes(9 * MiB + 3, itemsize=itemsize)
        assert chunk % itemsize == 0
        assert chunk <= CRC_CHUNK_MAX + itemsize - 1


def test_crc32c_chunks_matches_per_chunk_values() -> None:
    payload = bytes(range(256)) * ((12 * MiB) // 256)
    chunk, chunks = crc32c_chunks(payload)
    assert chunk == 1 * MiB
    assert len(chunks) == 12
    assert chunks == tuple(
        crc32c_hex(payload[start : start + chunk])
        for start in range(0, len(payload), chunk)
    )


def test_crc32c_chunks_rejects_empty() -> None:
    with pytest.raises(ValueError):
        crc_chunk_bytes(0)
    with pytest.raises(ValueError):
        crc_chunk_bytes(1, itemsize=0)
