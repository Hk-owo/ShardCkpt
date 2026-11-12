"""把逻辑读块切成「可校验的 IO 子区间」。

读取计划里的 ``ReadItem`` 是「一次逻辑读」；它的字节范围可能跨多个 CRC 块，
也可能只覆盖某个块的一部分。校验只能对**完整块**做：索引里每块一个 CRC，
没有「块内一段」的校验和。

这个模块只做几何切分，不碰 IO：

- ``verify=False``：一个子区间，保持原样；
- ``verify=True`` 且分片带块 CRC：按 ``crc_chunk_bytes`` 切；
  完整块带 ``expected_crc``，头/尾部分块 ``expected_crc=None``；
- 找不到分片 / 分片没有块 CRC：一个子区间，``expected_crc=None``。

``expected_crc=None`` 的区间照常读，但调用方要如实记进
``LoadReport.unverified``，不能假装校验过。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from core import ChecksumError
from core.index import CheckpointIndex, ShardMeta
from planner.load_plan import ReadItem
from storage.checksum import crc32c_hex


@dataclass(frozen=True, slots=True)
class CrcIoRange:
    """一次实际 IO：文件内的字节区间 + staging 落点 + 可选的期望 CRC。"""

    file: str
    byte_offset: int
    length: int
    staging_offset: int
    label: str
    """逻辑读块的标签（``file@offset+length``），用于报告未校验项。"""
    expected_crc: str | None = None
    """完整 CRC 块的期望值；``None`` 表示这段没有可比的整块校验和。"""

    @property
    def end(self) -> int:
        return self.byte_offset + self.length


def item_label(item: ReadItem) -> str:
    return f"{item.file}@{item.byte_offset}+{item.length}"


def covering_shard(index: CheckpointIndex, item: ReadItem) -> ShardMeta | None:
    """找到完整包住 ``item`` 的分片；跨分片的读块返回 ``None``。"""
    meta = index.tensors.get(item.fqn)
    if meta is None:
        return None
    for shard in meta.shards:
        if (
            shard.file == item.file
            and shard.byte_offset <= item.byte_offset
            and item.end <= shard.end
        ):
            return shard
    return None


def crc_io_ranges(
    index: CheckpointIndex,
    item: ReadItem,
    staging_offset: int = 0,
    *,
    verify: bool = True,
) -> tuple[CrcIoRange, ...]:
    """按分片块边界把 ``item`` 切成 IO 子区间。"""
    label = item_label(item)
    if not verify:
        return (
            CrcIoRange(
                file=item.file,
                byte_offset=item.byte_offset,
                length=item.length,
                staging_offset=staging_offset,
                label=label,
            ),
        )

    shard = covering_shard(index, item)
    chunk = shard.crc_chunk_bytes if shard is not None else None
    if shard is None or chunk is None or not shard.crc32c_chunks:
        return (
            CrcIoRange(
                file=item.file,
                byte_offset=item.byte_offset,
                length=item.length,
                staging_offset=staging_offset,
                label=label,
            ),
        )

    local_start = item.byte_offset - shard.byte_offset
    local_end = local_start + item.length
    out: list[CrcIoRange] = []
    pos = local_start
    while pos < local_end:
        block = pos // chunk
        block_start = block * chunk
        block_end = min((block + 1) * chunk, shard.length)
        seg_start = max(pos, block_start)
        seg_end = min(local_end, block_end)
        expected = None
        if seg_start == block_start and seg_end == block_end:
            expected = shard.crc32c_chunks[block]
        out.append(
            CrcIoRange(
                file=item.file,
                byte_offset=shard.byte_offset + seg_start,
                length=seg_end - seg_start,
                staging_offset=staging_offset + (seg_start - local_start),
                label=label,
                expected_crc=expected,
            )
        )
        pos = seg_end
    return tuple(out)


def verify_crc_range(view: memoryview, rng: CrcIoRange) -> bool:
    """校验一个 IO 子区间；返回 ``False`` 表示这段没有可比 CRC。"""
    if rng.expected_crc is None:
        return False
    actual = crc32c_hex(view[rng.staging_offset : rng.staging_offset + rng.length])
    if actual != rng.expected_crc:
        raise ChecksumError(
            f"{rng.label}: crc32c {actual} != index {rng.expected_crc} "
            f"({rng.file}@{rng.byte_offset}+{rng.length})"
        )
    return True


# --- 读取单元：扩读 + 同块合并 ----------------------------------------------


@dataclass(frozen=True, slots=True)
class Consumer:
    """一个读取单元服务的原读块，以及它在单元内的字节偏移。"""

    item: ReadItem
    offset_in_unit: int


@dataclass(frozen=True, slots=True)
class ReadUnit:
    """一次实际 IO：可能是一个完整 CRC 块，也可能是多个块合并后的区间。

    ``chunks`` 是单元内**完整覆盖**的 CRC 块：``(offset_in_unit, length, crc)``。
    部分块不在里面，校验时计入 ``unverified_bytes``。
    """

    file: str
    byte_offset: int
    length: int
    staging_offset: int
    label: str
    chunks: tuple[tuple[int, int, str], ...]
    consumers: tuple[Consumer, ...]


def crc_read_units(
    index: CheckpointIndex,
    items: Sequence[ReadItem],
    *,
    padding: str = "block",
    padding_ratio: float = 0.125,
) -> tuple[ReadUnit, ...]:
    """把逻辑读块编成实际 IO 单元。

    ``padding="block"``：未完整覆盖块的读块扩读到块边界，同块/相邻块合并；
    ``padding="auto"``：只有读块实际使用字节 / 扩展后字节 >= ``padding_ratio``
    才扩读；``padding="none"``：不扩读，只校验完整覆盖的块。
    """
    if padding not in ("none", "auto", "block"):
        raise ValueError(f"unknown crc padding {padding!r}")
    if not 0 <= padding_ratio <= 1:
        raise ValueError(f"padding_ratio must be in [0, 1], got {padding_ratio}")

    grouped: dict[tuple[str, int], list[tuple[int, int, ReadItem]]] = {}
    shards: dict[tuple[str, int], ShardMeta | None] = {}
    for item in items:
        shard = covering_shard(index, item)
        start, end = item.byte_offset, item.end
        if shard is not None and shard.crc_chunk_bytes and padding != "none":
            chunk = shard.crc_chunk_bytes
            first = (start - shard.byte_offset) // chunk
            last = (end - 1 - shard.byte_offset) // chunk
            expanded_start = shard.byte_offset + first * chunk
            expanded_end = shard.byte_offset + min(
                (last + 1) * chunk, shard.length
            )
            if (
                padding == "block"
                or item.length / (expanded_end - expanded_start) >= padding_ratio
            ):
                start, end = expanded_start, expanded_end
        key = (
            item.file,
            shard.byte_offset if shard is not None else item.byte_offset,
        )
        grouped.setdefault(key, []).append((start, end, item))
        shards[key] = shard

    units: list[ReadUnit] = []
    for key in sorted(grouped):
        entries = sorted(grouped[key], key=lambda entry: (entry[0], entry[1]))
        shard = shards[key]
        start, end, first_item = entries[0]
        current = [first_item]
        for next_start, next_end, item in entries[1:]:
            if next_start <= end:
                end = max(end, next_end)
                current.append(item)
            else:
                units.append(_make_unit(key[0], start, end, current, shard))
                start, end, current = next_start, next_end, [item]
        units.append(_make_unit(key[0], start, end, current, shard))

    offset = 0
    out: list[ReadUnit] = []
    for unit in units:
        out.append(replace(unit, staging_offset=offset))
        offset += unit.length
    return tuple(out)


def _make_unit(
    file: str,
    start: int,
    end: int,
    items: Sequence[ReadItem],
    shard: ShardMeta | None,
) -> ReadUnit:
    length = end - start
    consumers = tuple(Consumer(item, item.byte_offset - start) for item in items)
    chunks: list[tuple[int, int, str]] = []
    if shard is not None and shard.crc_chunk_bytes and shard.crc32c_chunks:
        chunk = shard.crc_chunk_bytes
        first = (start - shard.byte_offset) // chunk
        last = (end - 1 - shard.byte_offset) // chunk
        for block in range(first, last + 1):
            block_start = shard.byte_offset + block * chunk
            block_end = shard.byte_offset + min((block + 1) * chunk, shard.length)
            if start <= block_start and end >= block_end:
                chunks.append(
                    (
                        block_start - start,
                        block_end - block_start,
                        shard.crc32c_chunks[block],
                    )
                )
    return ReadUnit(
        file=file,
        byte_offset=start,
        length=length,
        staging_offset=0,
        label=f"{file}@{start}+{length}",
        chunks=tuple(chunks),
        consumers=consumers,
    )


__all__ = [
    "Consumer",
    "CrcIoRange",
    "ReadUnit",
    "covering_shard",
    "crc_io_ranges",
    "crc_read_units",
    "item_label",
    "verify_crc_range",
]
