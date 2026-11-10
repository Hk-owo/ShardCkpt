"""读取计划的测试：求交、读块几何、合并、覆盖校验、窗口化。

文档 §3 的两个表格例子（TP 8→4、TP 8→16）原样做成了断言。
"""

from __future__ import annotations

import pytest
from core import PlanError
from core.index import CheckpointIndex, Placement, ShardMeta, TensorMeta
from core.region import Interval, Region, split
from planner.load_plan import CopySpan, ReadItem, ReadPlan, TargetTensor, build_load_plan

SHAPE = (4096, 4096)
F32 = 4


def sharded_index(
    *,
    shape: tuple[int, ...] = SHAPE,
    dtype: str = "f32",
    dim: int = 1,
    count: int = 8,
    fqn: str = "w",
) -> CheckpointIndex:
    """沿 ``dim`` 切成 ``count`` 份，每份一个文件、从文件头开始。"""
    itemsize = 4 if dtype in ("f32", "i32") else 2
    shards = []
    for index in range(count):
        region = split(shape, dim, index, count)
        shards.append(
            ShardMeta.rectangular(
                file=f"__{index}_0.distcp",
                byte_offset=0,
                length=region.numel * itemsize,
                offsets=region.offsets,
                shape=region.shape,
            )
        )
    return CheckpointIndex(
        tensors={fqn: TensorMeta(global_shape=shape, dtype=dtype, shards=tuple(shards))}
    )


# --- 布局不变：退化成整读自己的旧文件 ---------------------------------------


def test_unchanged_layout_reads_its_own_shard_directly() -> None:
    index = sharded_index(count=8)
    for index_rank in range(8):
        target = TargetTensor.for_shard("w", SHAPE, "f32", dim=1, index=index_rank, count=8)
        plan = build_load_plan(index, [target])
        items = plan.for_target("w")
        assert len(items) == 1
        item = items[0]
        assert item.file == f"__{index_rank}_0.distcp"
        assert item.byte_offset == 0
        assert item.length == 4096 * 512 * F32
        assert item.strategy == "contiguous"
        assert item.spans == (CopySpan((0, 0), (0, 0), (4096, 512)),)
        assert plan.amplification == 1.0


# --- 文档 §3 的两个例子 -----------------------------------------------------


def test_tp8_to_tp4_reads_two_whole_shards() -> None:
    """新 TP=4：rank j 需要列 [1024j, 1024j+1024)，正好覆盖旧分片 2j、2j+1。"""
    index = sharded_index(count=8)
    target = TargetTensor.for_shard("w", SHAPE, "f32", dim=1, index=3, count=4)
    plan = build_load_plan(index, [target])

    assert plan.files() == ["__6_0.distcp", "__7_0.distcp"]
    assert [item.length for item in plan.items] == [4096 * 512 * F32] * 2
    assert all(item.strategy == "contiguous" for item in plan.items)
    assert [span.dst_offsets for item in plan.items for span in item.spans] == [
        (0, 0),
        (0, 512),
    ]
    assert plan.amplification == 1.0


def test_tp8_to_tp16_reads_half_a_shard_with_2x_amplification() -> None:
    """新 TP=16：rank 5 只要旧分片 2 的一半列，整读 8 MB 再切，读放大 2 倍。"""
    index = sharded_index(count=8)
    target = TargetTensor.for_shard("w", SHAPE, "f32", dim=1, index=5, count=16)
    plan = build_load_plan(index, [target])

    assert len(plan.items) == 1
    item = plan.items[0]
    assert item.file == "__2_0.distcp"
    assert item.strategy == "whole"
    assert item.read_region() == Region((0, 1024), (4096, 512))  # 整读旧分片
    assert item.length == 4096 * 512 * F32
    # 源列在该旧分片里是后半段（列 1280 落在分片 2 的 [1024, 1536) 内）
    assert item.spans == (CopySpan((0, 256), (0, 0), (4096, 256)),)
    assert item.amplification == 2.0
    assert plan.total_bytes == 4096 * 512 * F32
    assert plan.useful_bytes == 4096 * 256 * F32


def test_tp8_to_tp16_with_strided_strategy_reads_one_row_per_request() -> None:
    index = sharded_index(count=8)
    target = TargetTensor.for_shard("w", SHAPE, "f32", dim=1, index=5, count=16)
    plan = build_load_plan(index, [target], column_strategy="strided")

    items = plan.for_target("w")
    assert len(items) == 4096  # 每行一次 read_range
    assert all(item.strategy == "strided" for item in items)
    assert all(item.read_numel == item.copy_numel for item in items)
    assert all(item.length == 256 * F32 for item in items)
    assert len({item.byte_offset for item in items}) == 4096
    assert plan.amplification == 1.0
    assert plan.total_bytes == 4096 * 256 * F32


# --- 放置变化 ---------------------------------------------------------------


def test_replicate_target_reads_every_shard() -> None:
    index = sharded_index(count=8)
    target = TargetTensor.whole("w", SHAPE, "f32")
    plan = build_load_plan(index, [target])

    assert len(plan.items) == 8
    assert plan.amplification == 1.0
    assert plan.useful_bytes == 4096 * 4096 * F32
    assert [span.dst_offsets for item in plan.items for span in item.spans] == [
        (0, 512 * index) for index in range(8)
    ]


def test_uneven_split_target_covers_padded_tail() -> None:
    """不整除时前几份各多一行，末份少一行：读取计划必须只覆盖真实元素。"""
    shape = (10, 4)
    index = sharded_index(shape=shape, dim=0, count=3)
    for index_rank in range(3):
        target = TargetTensor.for_shard("w", shape, "f32", dim=0, index=index_rank, count=3)
        plan = build_load_plan(index, [target])
        assert sum(span.numel for item in plan.items for span in item.spans) == target.numel
        assert plan.amplification == 1.0


def test_reading_a_shard_into_a_smaller_target_region() -> None:
    """目标只覆盖旧分片的一部分（例如新布局把 dim0 也切开了）。"""
    index = sharded_index(count=4, dim=1)
    target = TargetTensor(
        fqn="w",
        global_shape=SHAPE,
        dtype="f32",
        region=Region((1024, 0), (1024, 2048)),
    )
    plan = build_load_plan(index, [target])
    assert len(plan.items) == 2
    assert all(item.strategy == "contiguous" for item in plan.items)
    assert [span.dst_offsets for item in plan.items for span in item.spans] == [
        (0, 0),
        (0, 1024),
    ]


# --- 展平分片（ZeRO 索引） --------------------------------------------------


def test_flattened_index_shards_are_read_as_contiguous_blocks() -> None:
    shape = (32, 32)
    index = CheckpointIndex(
        tensors={
            "w": TensorMeta(
                global_shape=shape,
                dtype="f32",
                shards=(
                    ShardMeta.flattened(
                        file="__0_0", byte_offset=0, length=512 * F32, start=0, elements=512
                    ),
                    ShardMeta.flattened(
                        file="__1_0", byte_offset=0, length=512 * F32, start=512, elements=512
                    ),
                ),
            )
        }
    )
    plan = build_load_plan(index, [TargetTensor.whole("w", shape, "f32")])
    assert len(plan.items) == 2
    assert all(item.strategy == "contiguous" for item in plan.items)
    assert plan.amplification == 1.0
    assert sum(span.numel for item in plan.items for span in item.spans) == 1024
    assert plan.items[0].read_region() == Region((0, 0), (16, 32))
    assert plan.items[1].read_region() == Region((16, 0), (16, 32))


def test_flattened_index_of_odd_shape_reads_offset_range() -> None:
    """非整行对齐的展平区间：读取计划仍然逐块准确（读放大 1）。"""
    shape = (4, 5)
    index = CheckpointIndex(
        tensors={
            "w": TensorMeta(
                global_shape=shape,
                dtype="f32",
                shards=(
                    ShardMeta.flattened(
                        file="__0_0", byte_offset=0, length=10 * F32, start=3, elements=10
                    ),
                ),
            )
        }
    )
    target = TargetTensor(
        fqn="w", global_shape=shape, dtype="f32", region=Region((0, 0), (4, 5))
    )
    plan = build_load_plan(index, [target], strict=False)
    assert plan.incomplete == ("w",)  # 索引只覆盖 [3, 13)，缺两端
    assert plan.amplification == 1.0
    assert sum(span.numel for item in plan.items for span in item.spans) == 10


# --- 合并 -------------------------------------------------------------------


def test_adjacent_requests_in_one_file_are_merged() -> None:
    """同一文件里字节相邻的两个分片，目标整读时合成一次 read_range。"""
    shape = (8, 4)
    shards = tuple(
        ShardMeta.rectangular(
            file="__0_0",
            byte_offset=index * 4 * 4 * F32,
            length=4 * 4 * F32,
            offsets=(index * 4, 0),
            shape=(4, 4),
        )
        for index in range(2)
    )
    index = CheckpointIndex(
        tensors={"w": TensorMeta(global_shape=shape, dtype="f32", shards=shards)}
    )
    plan = build_load_plan(index, [TargetTensor.whole("w", shape, "f32")])
    assert len(plan.items) == 1
    item = plan.items[0]
    assert item.length == 8 * 4 * F32
    assert item.read_region() == Region((0, 0), (8, 4))
    assert item.spans == (
        CopySpan((0, 0), (0, 0), (4, 4)),
        CopySpan((4, 0), (4, 0), (4, 4)),
    )
    assert plan.amplification == 1.0


def test_strided_requests_are_not_merged_across_gaps() -> None:
    index = sharded_index(count=8)
    target = TargetTensor.for_shard("w", SHAPE, "f32", dim=1, index=5, count=16)
    plan = build_load_plan(index, [target], column_strategy="strided")
    offsets = [item.byte_offset for item in plan.items]
    assert offsets == sorted(offsets)
    assert len(set(offsets)) == len(offsets)


# --- 覆盖校验与 strict ------------------------------------------------------


def test_missing_tensor_is_reported_and_raised() -> None:
    index = sharded_index(count=2)
    target = TargetTensor.whole("not-there", SHAPE, "f32")
    report = build_load_plan(index, [target], strict=False)
    assert report.missing == ("not-there",)
    assert report.items == ()
    with pytest.raises(PlanError):
        build_load_plan(index, [target])


def test_incomplete_coverage_is_reported_and_raised() -> None:
    shape = (8, 8)
    index = CheckpointIndex(
        tensors={
            "w": TensorMeta(
                global_shape=shape,
                dtype="f32",
                shards=(
                    ShardMeta.rectangular(
                        file="__0_0", byte_offset=0, length=4 * 8 * F32,
                        offsets=(0, 0), shape=(4, 8),
                    ),
                ),
            )
        }
    )
    target = TargetTensor.whole("w", shape, "f32")
    report = build_load_plan(index, [target], strict=False)
    assert report.incomplete == ("w",)
    with pytest.raises(PlanError):
        build_load_plan(index, [target])


def test_extra_tensors_are_reported_and_can_be_an_error() -> None:
    index = sharded_index(count=2)
    extra = TensorMeta(
        global_shape=(2, 2),
        dtype="f32",
        shards=(
            ShardMeta.rectangular(
                file="__9_0", byte_offset=0, length=4 * F32, offsets=(0, 0), shape=(2, 2)
            ),
        ),
    )
    index = CheckpointIndex(tensors={**index.tensors, "optim.step": extra})
    target = TargetTensor.whole("w", SHAPE, "f32")
    plan = build_load_plan(index, [target])
    assert plan.extra == ("optim.step",)
    with pytest.raises(PlanError):
        build_load_plan(index, [target], allow_extra=False)


def test_index_metadata_mismatch_is_rejected() -> None:
    index = sharded_index(count=2)
    with pytest.raises(PlanError):
        build_load_plan(index, [TargetTensor.whole("w", (4096, 2048), "f32")])
    with pytest.raises(PlanError):
        build_load_plan(index, [TargetTensor.whole("w", SHAPE, "f16")])


# --- 窗口化 -----------------------------------------------------------------


def test_windows_split_by_budget() -> None:
    index = sharded_index(count=8)
    plan = build_load_plan(index, [TargetTensor.whole("w", SHAPE, "f32")])
    shard_bytes = 4096 * 512 * F32

    windows = list(plan.windows(shard_bytes))
    assert len(windows) == 8
    assert all(len(window) == 1 for window in windows)

    windows = list(plan.windows(2 * shard_bytes))
    assert len(windows) == 4
    assert all(sum(item.length for item in window) <= 2 * shard_bytes for window in windows)

    # 单个请求超过预算时自己占一个窗口，不会被切开
    windows = list(plan.windows(1))
    assert len(windows) == 8
    with pytest.raises(ValueError):
        list(plan.windows(0))


def test_windows_cover_all_items_once() -> None:
    index = sharded_index(count=8)
    plan = build_load_plan(index, [TargetTensor.whole("w", SHAPE, "f32")])
    flattened = [item for window in plan.windows(3 * 4096 * 512 * F32) for item in window]
    assert flattened == list(plan.items)


# --- 输入校验 ---------------------------------------------------------------


def test_target_tensor_rejects_bad_regions() -> None:
    with pytest.raises(ValueError):
        TargetTensor(fqn="w", global_shape=SHAPE, dtype="f32", region=Region((0, 0), (4096, 4097)))
    with pytest.raises(ValueError):
        TargetTensor(fqn="w", global_shape=SHAPE, dtype="f32", region=Region((0, 0), (0, 4)))
    with pytest.raises(ValueError):
        TargetTensor(fqn="w", global_shape=SHAPE, dtype="f32", region=Region((0,), (4,)))
    with pytest.raises(ValueError):
        TargetTensor(fqn="w", global_shape=SHAPE, dtype="fp32", region=Region((0, 0), (1, 1)))
    with pytest.raises(ValueError):
        TargetTensor.whole("w", SHAPE, "f32", placement=Placement.shard(-1))


def test_build_load_plan_rejects_bad_targets_and_strategy() -> None:
    index = sharded_index(count=2)
    target = TargetTensor.whole("w", SHAPE, "f32")
    with pytest.raises(ValueError):
        build_load_plan(index, [target], column_strategy="smart")
    with pytest.raises(TypeError):
        build_load_plan(index, ["w"])
    with pytest.raises(ValueError):
        build_load_plan(index, [target, target])
    with pytest.raises(ValueError):
        build_load_plan(index, {"other": target})


def test_read_plan_helpers() -> None:
    index = sharded_index(count=2)
    plan = build_load_plan(index, [TargetTensor.whole("w", SHAPE, "f32")])
    assert isinstance(plan, ReadPlan)
    assert plan.files() == ["__0_0.distcp", "__1_0.distcp"]
    assert len(plan.for_target("w")) == 2
    assert plan.for_target("nope") == ()
    assert plan.useful_bytes == SHAPE[0] * SHAPE[1] * F32
    assert plan.amplification == 1.0
    assert all(isinstance(item, ReadItem) for item in plan.items)
    assert plan.items[0].end == 4096 * 2048 * F32


def test_flat_index_reads_rectangular_target_region() -> None:
    """目标也是展平区间时（不是本轮实现的分支），整读张量仍然覆盖完整。"""
    shape = (4, 4)
    index = CheckpointIndex(
        tensors={
            "w": TensorMeta(
                global_shape=shape,
                dtype="f32",
                shards=(
                    ShardMeta.flattened(
                        file="__0_0", byte_offset=0, length=16 * F32, start=0, elements=16
                    ),
                ),
            )
        }
    )
    target = TargetTensor(
        fqn="w", global_shape=shape, dtype="f32", region=Region((1, 0), (2, 4))
    )
    plan = build_load_plan(index, [target])
    assert plan.amplification == 1.0
    assert plan.items[0].spans == (CopySpan((0, 0), (0, 0), (2, 4)),)
    assert plan.items[0].read_region() == Region((1, 0), (2, 4))


def test_copy_span_regions_are_rectangular() -> None:
    """CopySpan 的几何一定是矩形（读取端按矩形拷入），Interval 只用于展平索引。"""
    span = CopySpan((0, 0), (0, 0), (2, 2))
    assert span.src_region() == Region((0, 0), (2, 2))
    assert span.dst_region() == Region((0, 0), (2, 2))
    assert span.numel == 4
    assert Interval(0, 4).length == 4
