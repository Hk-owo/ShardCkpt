"""保存计划的测试：去重、写者分配、偏移与对齐、覆盖校验。"""

from __future__ import annotations

import pytest
from core import PlanError
from core.index import CheckpointIndex, TensorMeta
from core.region import Interval, Region
from planner.save_plan import (
    DEFAULT_ALIGN,
    BlobItem,
    SavePlan,
    ShardSource,
    build_save_plan,
    default_writer,
)

HFN = (64, 32)
F32 = 4


def source(fqn: str, region: Region, **kwargs) -> ShardSource:
    return ShardSource(fqn=fqn, global_shape=HFN, dtype="f32", region=region, **kwargs)


def merge_plans(plans: list[SavePlan]) -> tuple[dict[str, TensorMeta], dict]:
    tensors: dict[str, TensorMeta] = {}
    blobs: dict = {}
    for plan in plans:
        for fqn, meta in plan.partial_index().items():
            if fqn in tensors:
                merged = TensorMeta(
                    global_shape=meta.global_shape,
                    dtype=meta.dtype,
                    shards=tensors[fqn].shards + meta.shards,
                )
                tensors[fqn] = merged
            else:
                tensors[fqn] = meta
        blobs.update(plan.partial_blobs())
    return tensors, blobs


# --- 偏移与对齐 -------------------------------------------------------------


def test_single_rank_writes_whole_tensor_with_aligned_offsets() -> None:
    plan = build_save_plan(
        [source("w", Region((0, 0), HFN)), source("b", Region((0, 0), HFN))],
        rank=0,
        blobs={"lr_scheduler": 456, "rng/rank_0": 64},
    )
    assert plan.file == "__0_0.distcp"
    assert [item.fqn for item in plan.items] == ["b", "w"]
    assert [blob.name for blob in plan.blobs] == ["lr_scheduler", "rng/rank_0"]
    assert plan.items[0].byte_offset == 0
    assert plan.items[1].byte_offset == HFN[0] * HFN[1] * F32
    assert plan.items[1].byte_offset % DEFAULT_ALIGN == 0
    assert plan.blobs[0].byte_offset % DEFAULT_ALIGN == 0
    assert plan.size == plan.blobs[-1].end  # 末项不补齐
    assert plan.size == plan.data_bytes + plan.padding_bytes
    plan.validate()


def test_align_one_disables_padding() -> None:
    plan = build_save_plan(
        [source("a", Region((0, 0), (16, 32))), source("b", Region((0, 0), (16, 32)))],
        rank=0,
        align=1,
    )
    assert plan.padding_bytes == 0
    assert [item.byte_offset for item in plan.items] == [0, 16 * 32 * F32]


def test_start_offset_shifts_whole_file() -> None:
    plan = build_save_plan([source("a", Region((0, 0), (16, 32)))], rank=0, start_offset=8192)
    assert plan.items[0].byte_offset == 8192
    assert plan.size == 8192 + 16 * 32 * F32


def test_blob_only_plan_has_no_tensor_items() -> None:
    plan = build_save_plan([], rank=3, blobs={"step": 8})
    assert plan.items == ()
    assert [blob.name for blob in plan.blobs] == ["step"]
    assert plan.file == "__3_0.distcp"
    assert plan.size == 8


def test_empty_plan() -> None:
    plan = build_save_plan([], rank=1)
    assert plan.items == () and plan.blobs == () and plan.size == 0


# --- 去重与写者分配 ---------------------------------------------------------


def test_replicated_shards_have_exactly_one_writer_each() -> None:
    """DP 复制：每个 rank 持有一份相同数据，只有 hash 选中的那份落盘。"""
    fqns = [f"model.layers.{i}.weight" for i in range(8)]
    sources_per_rank = {
        replicas: [
            source(fqn, Region((0, 0), HFN), replica_index=replicas, n_replicas=2)
            for fqn in fqns
        ]
        for replicas in (0, 1)
    }
    plans = [
        build_save_plan(sources_per_rank[replicas], rank=replicas) for replicas in (0, 1)
    ]
    written = [item.fqn for plan in plans for item in plan.items]
    assert sorted(written) == sorted(fqns)  # 每个 FQN 恰好一个写者
    assert len(written) == len(set(written))

    # 写入量按 FQN 均摊：两个副本都分到活，而不是全部压给 rank 0
    assert all(plan.items for plan in plans)
    assert sum(len(plan.items) for plan in plans) == len(fqns)

    tensors, _ = merge_plans(plans)
    CheckpointIndex(tensors=tensors).validate()


def test_skipped_keeps_other_replicas() -> None:
    """每个 rank 只看到自己那份副本：恰好一个 rank 写，其余进 skipped。"""
    owner = default_writer("w", 3)
    plans = [
        build_save_plan(
            [source("w", Region((0, 0), HFN), replica_index=index, n_replicas=3)],
            rank=index,
        )
        for index in range(3)
    ]
    assert sum(len(plan.items) for plan in plans) == 1
    assert [index for index, plan in enumerate(plans) if plan.items] == [owner]
    for index, plan in enumerate(plans):
        assert (len(plan.skipped) == 1) is (index != owner)


def test_custom_writer_is_used() -> None:
    sources = [
        source("w", Region((0, 0), HFN), replica_index=index, n_replicas=2)
        for index in range(2)
    ]
    plan = build_save_plan(sources, rank=1, writer=lambda fqn, n: 1)
    assert len(plan.items) == 1
    assert [item.fqn for item in plan.skipped] == ["w"]


def test_writer_out_of_range_is_plan_error() -> None:
    with pytest.raises(PlanError):
        build_save_plan([source("w", Region((0, 0), HFN))], rank=0, writer=lambda fqn, n: 3)


# --- 展平切分（方案 A） -----------------------------------------------------


def test_flattened_shards_from_zero_style_split() -> None:
    """ZeRO-1：两个 rank 各拿参数展平后的一半，索引里留一维区间。"""
    total = HFN[0] * HFN[1]
    sources = [
        ShardSource(
            fqn="w",
            global_shape=HFN,
            dtype="f32",
            flat=Interval(index * total // 2, total // 2),
        )
        for index in range(2)
    ]
    plans = [build_save_plan([src], rank=index) for index, src in enumerate(sources)]
    item = plans[0].items[0]
    assert item.flat == Interval(0, total // 2)
    assert item.length == total // 2 * F32
    assert item.to_shard_meta().is_flat

    tensors, _ = merge_plans(plans)
    CheckpointIndex(tensors=tensors).validate()


def test_overlapping_shards_are_rejected_locally() -> None:
    sources = [
        source("w", Region((0, 0), (64, 32))),
        source("w", Region((0, 16), (64, 16))),
    ]
    with pytest.raises(PlanError):
        build_save_plan(sources, rank=0)


def test_incomplete_shards_are_allowed_locally() -> None:
    """单个 rank 只见自己那一片，局部校验不要求覆盖完整。"""
    plan = build_save_plan([source("w", Region((0, 0), (32, 32)))], rank=0)
    plan.validate()
    assert plan.partial_index()["w"].covered_numel == 32 * 32


def test_inconsistent_metadata_for_same_fqn_is_rejected() -> None:
    sources = [
        ShardSource(fqn="w", global_shape=HFN, dtype="f32", region=Region((0, 0), (32, 32))),
        ShardSource(fqn="w", global_shape=HFN, dtype="f16", region=Region((32, 0), (32, 32))),
    ]
    with pytest.raises(PlanError):
        build_save_plan(sources, rank=0)


# --- 输入校验 ---------------------------------------------------------------


def test_shard_source_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        ShardSource(fqn="w", global_shape=HFN, dtype="f32")
    with pytest.raises(ValueError):
        ShardSource(
            fqn="w",
            global_shape=HFN,
            dtype="f32",
            region=Region((0, 0), (8, 8)),
            flat=Interval(0, 8),
        )
    with pytest.raises(ValueError):  # 越界
        ShardSource(fqn="w", global_shape=HFN, dtype="f32", region=Region((0, 32), (64, 32)))
    with pytest.raises(ValueError):  # 展平区间越界
        ShardSource(
            fqn="w", global_shape=HFN, dtype="f32", flat=Interval(HFN[0] * HFN[1] - 1, 2)
        )
    with pytest.raises(ValueError):
        ShardSource(fqn="w", global_shape=HFN, dtype="f32", flat=Interval(0, 8), n_replicas=0)
    with pytest.raises(ValueError):
        ShardSource(
            fqn="w",
            global_shape=HFN,
            dtype="f32",
            flat=Interval(0, 8),
            replica_index=2,
            n_replicas=2,
        )
    with pytest.raises(ValueError):
        ShardSource(fqn="w", global_shape=HFN, dtype="fp32", flat=Interval(0, 8))


def test_build_save_plan_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        build_save_plan([], rank=-1)
    with pytest.raises(ValueError):
        build_save_plan([], rank=0, align=0)
    with pytest.raises(ValueError):
        build_save_plan([], rank=0, start_offset=-1)
    with pytest.raises(ValueError):
        build_save_plan([], rank=0, blobs={"step": 0})
    with pytest.raises(TypeError):
        build_save_plan([("w", Interval(0, 4))], rank=0)


def test_segments_expands_flat_range() -> None:
    flat = ShardSource(fqn="w", global_shape=HFN, dtype="f32", flat=Interval(1, 40))
    assert sum(segment.numel for segment in flat.segments()) == 40
    assert len(flat.segments()) >= 2
    assert flat.start_element == 1
    assert flat.elements == 40
    assert flat.nbytes == 40 * F32


def test_blob_item_orders_and_metadata() -> None:
    plan = build_save_plan([], rank=0, blobs={"b": 16, "a": 8})
    assert [blob.name for blob in plan.blobs] == ["a", "b"]
    assert plan.blobs[0].to_blob_meta().length == 8
    assert isinstance(plan.blobs[0], BlobItem)
