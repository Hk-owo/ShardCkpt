"""区域与展平换算的测试：求交、展平往返、切分。

这一层是 planner 的地基，用例只做纯计算，不碰任何 IO。
"""

from __future__ import annotations

import pytest
from core.region import (
    Interval,
    Region,
    contiguous_blocks,
    flatten_index,
    flatten_range,
    flatten_slice,
    is_flatten_contiguous,
    is_row_contiguous,
    numel,
    split,
)


# --- Interval ---------------------------------------------------------------


def test_interval_of_and_end() -> None:
    span = Interval.of(10, 20)
    assert (span.start, span.length, span.end) == (10, 10, 20)
    assert not span.is_empty
    assert Interval(5, 0).is_empty


def test_interval_rejects_negative_and_inverted() -> None:
    with pytest.raises(ValueError):
        Interval(-1, 4)
    with pytest.raises(ValueError):
        Interval(0, -4)
    with pytest.raises(ValueError):
        Interval.of(20, 10)


def test_interval_intersect_covers_edge_cases() -> None:
    assert Interval(0, 10).intersect(Interval(5, 10)) == Interval(5, 5)
    assert Interval(0, 10).intersect(Interval(10, 5)) is None  # 紧邻不算重叠
    assert Interval(0, 10).intersect(Interval(3, 2)) == Interval(3, 2)
    assert Interval(0, 10).contains(Interval(0, 10))
    assert not Interval(0, 10).contains(Interval(9, 2))


def test_interval_merge_requires_touching() -> None:
    assert Interval(0, 4).merge(Interval(4, 6)) == Interval(0, 10)
    assert Interval(0, 4).merge(Interval(2, 6)) == Interval(0, 8)
    assert Interval(0, 4).touches(Interval(4, 6))
    assert not Interval(0, 4).touches(Interval(5, 1))
    with pytest.raises(ValueError):
        Interval(0, 4).merge(Interval(5, 1))


# --- Region -----------------------------------------------------------------


def test_region_basics() -> None:
    region = Region((2, 3), (4, 5))
    assert region.rank == 2
    assert region.numel == 20
    assert region.end == (6, 8)
    assert region.slices == (slice(2, 6), slice(3, 8))
    assert not region.is_empty
    assert Region((0, 0), (4, 0)).is_empty


def test_region_intersect_is_per_dimension() -> None:
    a = Region((0, 0), (4, 4))
    b = Region((2, 1), (4, 4))
    assert a.intersect(b) == Region((2, 1), (2, 3))
    assert a.intersect(Region((4, 0), (2, 2))) is None  # 首维紧邻
    assert a.overlaps(Region((3, 3), (2, 2)))
    assert not a.overlaps(Region((4, 4), (1, 1)))


def test_region_contains_and_relative_to() -> None:
    outer = Region((2, 3), (4, 5))
    inner = Region((3, 4), (2, 2))
    assert outer.contains(inner)
    assert not inner.contains(outer)
    assert inner.relative_to(outer.offsets) == Region((1, 1), (2, 2))
    assert inner.relative_to(inner.offsets) == Region((0, 0), (2, 2))
    assert outer.translate((-2, -3)) == Region((0, 0), (4, 5))
    with pytest.raises(ValueError):
        Region((1, 1), (2, 2)).relative_to((2, 2))


def test_region_rejects_bad_rank_and_negative() -> None:
    with pytest.raises(ValueError):
        Region((0,), (1, 2))
    with pytest.raises(ValueError):
        Region((), ())
    with pytest.raises(ValueError):
        Region((-1,), (2,))


# --- 展平换算 ---------------------------------------------------------------


def test_numel_and_flatten_index() -> None:
    assert numel((4, 5)) == 20
    assert numel(()) == 1
    assert flatten_index((4, 5), (0, 0)) == 0
    assert flatten_index((4, 5), (1, 0)) == 5
    assert flatten_index((4, 5), (3, 4)) == 19
    with pytest.raises(ValueError):
        flatten_index((4, 5), (4, 0))


def test_row_contiguous_detection() -> None:
    shape = (4, 5, 6)
    assert is_row_contiguous(shape, Region((0, 0, 0), (4, 5, 6)))
    assert is_row_contiguous(shape, Region((1, 0, 0), (2, 5, 6)))
    assert not is_row_contiguous(shape, Region((0, 0, 1), (4, 5, 5)))  # 列子集
    assert not is_row_contiguous(shape, Region((0, 1, 0), (4, 4, 6)))  # 中间维子集
    assert is_row_contiguous(shape, Region((0, 0, 0), (0, 5, 6)))  # 空区域


def test_flatten_contiguous_is_looser_than_row_contiguous() -> None:
    """扁平连续性：整行块、以及「单行内的前缀子段」都算连续，列子集不算。"""
    shape = (4, 5, 6)
    assert is_flatten_contiguous(shape, Region((0, 0, 0), (4, 5, 6)))
    assert is_flatten_contiguous(shape, Region((1, 0, 0), (2, 5, 6)))
    assert is_flatten_contiguous(shape, Region((2, 1, 0), (1, 4, 6)))  # 单行内连续
    assert not is_flatten_contiguous(shape, Region((0, 0, 1), (4, 5, 5)))  # 列子集
    assert not is_flatten_contiguous(shape, Region((0, 1, 0), (3, 4, 6)))  # 跨行且不满
    assert is_flatten_contiguous((7,), Region((3,), (4,)))  # 一维恒连续


def test_contiguous_blocks_splits_column_subset_per_row() -> None:
    shape = (4, 5)
    blocks = contiguous_blocks(shape, Region((0, 2), (4, 3)))
    assert len(blocks) == 4  # 每行一段
    assert sum(block.numel for block in blocks) == 12
    assert all(is_flatten_contiguous(shape, block) for block in blocks)
    assert blocks[0] == Region((0, 2), (1, 3))
    assert blocks[-1] == Region((3, 2), (1, 3))


def test_contiguous_blocks_keeps_contiguous_regions_whole() -> None:
    shape = (4, 5, 6)
    whole = Region((1, 0, 0), (2, 5, 6))
    assert contiguous_blocks(shape, whole) == [whole]
    row = Region((2, 1, 0), (1, 4, 6))
    assert contiguous_blocks(shape, row) == [row]


def test_contiguous_blocks_handles_inner_dimension_gaps() -> None:
    """中间维不满时递归拆到单行内的一段，块数不会失控。"""
    shape = (3, 4, 5)
    region = Region((0, 1, 0), (3, 2, 5))
    blocks = contiguous_blocks(shape, region)
    assert len(blocks) == 3
    assert sum(block.numel for block in blocks) == 30
    assert all(is_flatten_contiguous(shape, block) for block in blocks)


def test_flatten_range_roundtrip() -> None:
    shape = (4, 5, 6)
    region = Region((1, 0, 0), (2, 5, 6))
    span = flatten_range(shape, region)
    assert span == Interval(1 * 30, 2 * 30)
    assert flatten_slice(shape, span) == [region]
    with pytest.raises(ValueError):
        flatten_range(shape, Region((0, 0, 1), (4, 5, 5)))


def test_flatten_slice_splits_non_rectangular_range() -> None:
    """中间的展平区间跨越行边界时，要拆成首段 + 整段 + 尾段。"""
    shape = (4, 3)
    pieces = flatten_slice(shape, Interval.of(2, 8))  # [2, 8) 共 6 个元素
    assert pieces == [
        Region((0, 2), (1, 1)),
        Region((1, 0), (1, 3)),
        Region((2, 0), (1, 2)),
    ]
    assert sum(piece.numel for piece in pieces) == 6


def _covered_indices(shape: tuple[int, ...], region: Region) -> set[int]:
    """区域覆盖的行主序下标集合（逐个坐标算，用来核对覆盖关系）。"""
    from itertools import product

    ranges = [range(start, start + size) for start, size in zip(region.offsets, region.shape)]
    return {flatten_index(shape, coords) for coords in product(*ranges)}


def test_flatten_slice_roundtrip_over_every_offset() -> None:
    """任意展平区间展开后必须逐元素精确覆盖，且不重叠。"""
    for shape in [(7,), (4, 5), (3, 4, 5), (2, 3, 4, 5)]:
        total = numel(shape)
        for start in range(total):
            for length in range(1, total - start + 1):
                span = Interval(start, length)
                pieces = flatten_slice(shape, span)
                covered: set[int] = set()
                for piece in pieces:
                    indices = _covered_indices(shape, piece)
                    assert not (indices & covered), f"{shape} {span} {piece}"
                    covered |= indices
                assert covered == set(range(span.start, span.end)), f"{shape} {span}"


def test_flatten_slice_piece_count_is_bounded() -> None:
    """段数上界 2 * rank：不会被展平切分放大成碎片。"""
    shape = (8, 7, 6, 5)
    pieces = flatten_slice(shape, Interval.of(3, 1000))
    assert len(pieces) <= 2 * len(shape)


def test_flatten_slice_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        flatten_slice((4, 5), Interval(0, 21))
    assert flatten_slice((4, 5), Interval(0, 0)) == []


# --- 切分 -------------------------------------------------------------------


def test_split_even_and_uneven() -> None:
    assert split((8, 4), 0, 0, 4) == Region((0, 0), (2, 4))
    assert split((8, 4), 0, 3, 4) == Region((6, 0), (2, 4))
    assert split((10, 4), 0, 0, 3) == Region((0, 0), (4, 4))
    assert split((10, 4), 0, 2, 3) == Region((7, 0), (3, 4))


def test_split_covers_whole_tensor_without_overlap() -> None:
    shape = (10, 7)
    produced = [split(shape, 0, index, 3) for index in range(3)]
    assert sum(region.numel for region in produced) == numel(shape)
    for i, left in enumerate(produced):
        for right in produced[i + 1 :]:
            assert left.intersect(right) is None


def test_split_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        split((4, 4), 2, 0, 1)
    with pytest.raises(ValueError):
        split((4, 4), 0, 4, 1)
    with pytest.raises(ValueError):
        split((4, 4), 0, 0, 0)
