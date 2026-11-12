"""把逻辑读块切成可校验 IO 子区间：完整块带 CRC，头/尾部分块如实标 None。"""

from __future__ import annotations

import pytest

from core.index import CheckpointIndex, ShardMeta, TensorMeta
from planner.load_plan import CopySpan, ReadItem
from storage.crc_io import crc_io_ranges, crc_read_units

MiB = 1 << 20


def make_index(
    *,
    chunk_bytes: int | None = 1 * MiB,
    chunks: tuple[str, ...] | None = None,
):
    if chunks is None:
        chunks = tuple(f"{index:08x}" for index in range(12))
    shard = ShardMeta.rectangular(
        file="f",
        byte_offset=0,
        length=12 * MiB,
        offsets=(0,),
        shape=(3 * MiB,),
        crc_chunk_bytes=chunk_bytes,
        crc32c_chunks=chunks if chunk_bytes is not None else (),
    )
    return CheckpointIndex(
        tensors={"t": TensorMeta(global_shape=(3 * MiB,), dtype="f32", shards=(shard,))}
    )


def make_index_4mib():
    """``crc_io_ranges`` 的旧用例按 4 MiB 块写；显式指定，避免跟默认值耦合。"""
    return make_index(chunk_bytes=4 * MiB, chunks=("a" * 8, "b" * 8, "c" * 8))


def make_item(byte_offset: int, length: int) -> ReadItem:
    elements = length // 4
    return ReadItem(
        fqn="t",
        file="f",
        byte_offset=byte_offset,
        length=length,
        read_offsets=(byte_offset // 4,),
        read_shape=(elements,),
        dtype="f32",
        spans=(CopySpan((0,), (0,), (elements,)),),
        strategy="contiguous",
    )


def test_partial_head_and_tail_are_marked_unverifiable() -> None:
    # 读 [1 MiB, 9 MiB)：中间第二块 [4, 8) 完整，头尾各跨半个块
    ranges = crc_io_ranges(make_index_4mib(), make_item(1 * MiB, 8 * MiB))
    assert [(rng.byte_offset, rng.length, rng.expected_crc) for rng in ranges] == [
        (1 * MiB, 3 * MiB, None),
        (4 * MiB, 4 * MiB, "b" * 8),
        (8 * MiB, 1 * MiB, None),
    ]
    assert [rng.staging_offset for rng in ranges] == [0, 3 * MiB, 7 * MiB]


def test_aligned_block_is_one_verifiable_range() -> None:
    ranges = crc_io_ranges(make_index_4mib(), make_item(4 * MiB, 4 * MiB))
    assert len(ranges) == 1
    assert ranges[0].expected_crc == "b" * 8
    assert ranges[0].staging_offset == 0


def test_index_without_chunks_reads_whole_item() -> None:
    ranges = crc_io_ranges(make_index(chunk_bytes=None), make_item(1 * MiB, 8 * MiB))
    assert len(ranges) == 1
    assert ranges[0].byte_offset == 1 * MiB
    assert ranges[0].length == 8 * MiB
    assert ranges[0].expected_crc is None


def test_verify_false_keeps_one_range() -> None:
    ranges = crc_io_ranges(make_index(), make_item(1 * MiB, 8 * MiB), verify=False)
    assert len(ranges) == 1
    assert ranges[0].expected_crc is None


def test_span_that_is_not_aligned_to_shard_start() -> None:
    """分片起点不是读块起点时，块边界仍从分片起点算。"""
    item = make_item(2 * MiB, 4 * MiB)  # [2, 6)：第一块部分、第二块部分
    ranges = crc_io_ranges(make_index_4mib(), item)
    assert all(rng.expected_crc is None for rng in ranges)
    assert sum(rng.length for rng in ranges) == 4 * MiB


def test_cross_shard_item_is_not_verifiable() -> None:
    """跨分片的读块找不到唯一分片，整个区间标 None。"""
    item = make_item(11 * MiB, 2 * MiB)  # [11, 13) 越过 12 MiB 分片边界
    ranges = crc_io_ranges(make_index_4mib(), item)
    assert len(ranges) == 1
    assert ranges[0].byte_offset == 11 * MiB
    assert ranges[0].length == 2 * MiB
    assert ranges[0].expected_crc is None


# --- crc_read_units：扩读 + 同块合并 ----------------------------------------


def test_block_padding_merges_items_in_same_chunk() -> None:
    items = [make_item(0, 4096), make_item(8192, 4096)]
    units = crc_read_units(make_index(), items, padding="block")

    assert len(units) == 1
    unit = units[0]
    assert unit.byte_offset == 0
    assert unit.length == 1 * MiB
    assert len(unit.chunks) == 1
    assert [consumer.offset_in_unit for consumer in unit.consumers] == [0, 8192]


def test_none_padding_keeps_partial_units() -> None:
    items = [make_item(0, 4096), make_item(8192, 4096)]
    units = crc_read_units(make_index(), items, padding="none")

    assert len(units) == 2
    assert all(unit.chunks == () for unit in units)
    assert sum(unit.length for unit in units) == 8192


def test_auto_padding_skips_small_fraction() -> None:
    units = crc_read_units(
        make_index(), [make_item(0, 4096)], padding="auto", padding_ratio=0.125
    )
    assert len(units) == 1
    assert units[0].length == 4096
    assert units[0].chunks == ()


def test_auto_padding_expands_when_fraction_is_enough() -> None:
    units = crc_read_units(
        make_index(),
        [make_item(0, 256 * 1024)],
        padding="auto",
        padding_ratio=0.125,
    )
    assert len(units) == 1
    assert units[0].length == 1 * MiB
    assert len(units[0].chunks) == 1


def test_cross_chunk_item_expands_to_all_intersecting_chunks() -> None:
    item = make_item(512 * 1024, 1 * MiB)  # [0.5 MiB, 1.5 MiB)
    units = crc_read_units(make_index(), [item], padding="block")

    assert len(units) == 1
    unit = units[0]
    assert unit.byte_offset == 0
    assert unit.length == 2 * MiB
    assert len(unit.chunks) == 2


def test_aligned_full_chunk_is_one_unit() -> None:
    units = crc_read_units(make_index(), [make_item(1 * MiB, 1 * MiB)])
    assert len(units) == 1
    assert units[0].byte_offset == 1 * MiB
    assert units[0].length == 1 * MiB
    assert len(units[0].chunks) == 1
    assert units[0].chunks[0][2] == "00000001"
