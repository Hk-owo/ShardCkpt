"""``plan_ranks``：rank 0 汇总全部 rank 的分片，给出「每 rank 写什么」+ 几何骨架。

锁住的不变量：

- **布局零变化**：``per_rank[r]`` 与 rank r 自己调 ``build_save_plan`` 逐字段相同
  —— 汇总不引入新的偏移策略，所以打开它不会让任何一份 checkpoint 的物理布局变样；
- **跨 rank 标注校验**（单 rank 查不出来的那一半）：``n_replicas`` 必须一致、
  ``replica_index`` 必须恰好铺满 ``{0..n_replicas-1}``、同一副本号不能被两个 rank
  同时声明；
- **全局覆盖校验**在落盘前做（``complete=True``），漏分片直接报错；
- 骨架**不带 CRC**：块 CRC 只能在 D2H 之后对本 rank 的 buffer 算。
"""

from __future__ import annotations

import pytest

from core import PlanError
from core.region import Region
from planner.save_plan import (
    ShardSource,
    build_save_plan,
    plan_ranks,
)

SHAPE = (8, 8)


def S(fqn: str, off: int, length: int, *, ri: int = 0, nr: int = 1) -> ShardSource:
    """沿 dim 1 切一刀的矩形分片：``[off, off+length)`` 列。"""
    return ShardSource(
        fqn=fqn,
        global_shape=SHAPE,
        dtype="f32",
        region=Region((0, off), (8, length)),
        replica_index=ri,
        n_replicas=nr,
    )


def dp2_tp2() -> dict[int, list[ShardSource]]:
    """DP=2 副本 × TP=2 片：每个 rank 持有同一个张量的两片。"""
    return {
        dp: [S("w", 0, 4, ri=dp, nr=2), S("w", 4, 4, ri=dp, nr=2)]
        for dp in (0, 1)
    }


# --- 对拍：汇总 == 各自算 --------------------------------------------------


def test_per_rank_plans_match_build_save_plan() -> None:
    """核心性质：rank 0 汇总出的计划与各 rank 自算完全一致（布局零变化）。"""
    per_rank = dp2_tp2()
    global_plan = plan_ranks(per_rank)

    for rank in (0, 1):
        assert global_plan.plan_for(rank) == build_save_plan(per_rank[rank], rank=rank)


def test_per_rank_plans_match_with_blobs_and_align() -> None:
    """小状态与对齐参数也要一起对拍：偏移随 blobs 变化，容易只在一边算对。"""
    per_rank = {r: [S("w", 0, 8, ri=r, nr=2)] for r in (0, 1)}
    blobs = {r: {"step": 8, "rng/rank_%d" % r: 16} for r in (0, 1)}
    global_plan = plan_ranks(per_rank, blobs=blobs, align=64)

    for rank in (0, 1):
        assert global_plan.plan_for(rank) == build_save_plan(
            per_rank[rank], rank=rank, blobs=blobs[rank], align=64
        )


def test_total_bytes_sums_per_rank_sizes() -> None:
    per_rank = dp2_tp2()
    global_plan = plan_ranks(per_rank)
    # owner 只写一份副本，所以另一个 rank 的 plan 是空的
    assert global_plan.total_bytes == sum(
        plan.size for plan in global_plan.per_rank.values()
    )
    assert global_plan.ranks == (0, 1)


# --- 骨架 ------------------------------------------------------------------


def test_skeleton_covers_everything_and_has_no_crc() -> None:
    global_plan = plan_ranks(dp2_tp2())
    skeleton = global_plan.skeleton

    assert skeleton.validate() is None  # complete=True 不抛
    meta = skeleton.tensors["w"]
    assert meta.covered_numel == meta.numel
    for shard in meta.shards:
        assert shard.crc_chunk_bytes is None
        assert shard.crc32c_chunks == ()


def test_skeleton_merges_blobs_once() -> None:
    """同名小状态每个 rank 都写一份（内容相同），骨架里只留第一份。"""
    per_rank = {r: [S("w", r * 4, 4)] for r in (0, 1)}
    blobs = {r: {"step": 8} for r in (0, 1)}
    skeleton = plan_ranks(per_rank, blobs=blobs).skeleton
    assert set(skeleton.blobs) == {"step"}


# --- 跨 rank 标注校验 ------------------------------------------------------


def test_inconsistent_n_replicas_is_rejected() -> None:
    """同一 fqn 的两个分片对副本数说法不同 —— 会让 owner 算出两个人，必须拒绝。"""
    per_rank = {
        0: [S("w", 0, 4, ri=0, nr=2)],
        1: [S("w", 4, 4, ri=0, nr=1)],  # 说自己是「唯一那份」
    }
    with pytest.raises(PlanError, match="n_replicas 标注不一致"):
        plan_ranks(per_rank)


def test_duplicate_replica_index_is_rejected() -> None:
    """两个 rank 都声称自己是同一块覆盖范围的第 0 份副本。"""
    per_rank = {
        0: [S("w", 0, 8, ri=0, nr=2)],
        1: [S("w", 0, 8, ri=0, nr=2)],
    }
    with pytest.raises(PlanError, match="同时声明"):
        plan_ranks(per_rank)


def test_missing_replica_index_is_rejected() -> None:
    """声称 3 副本却只有 2 份声明：必然漏写一份。"""
    per_rank = {
        0: [S("w", 0, 8, ri=0, nr=3)],
        1: [S("w", 0, 8, ri=1, nr=3)],
    }
    with pytest.raises(PlanError, match="副本声明不完整"):
        plan_ranks(per_rank)


def test_rank_numbering_must_be_contiguous() -> None:
    per_rank = {0: [S("w", 0, 4)], 2: [S("w", 4, 4)]}
    with pytest.raises(ValueError, match="连续编号"):
        plan_ranks(per_rank)


def test_non_shard_source_is_rejected() -> None:
    with pytest.raises(TypeError, match="expected ShardSource"):
        plan_ranks({0: [object()]})  # type: ignore[list-item]


# --- 全局覆盖校验 ----------------------------------------------------------


def test_missing_shard_fails_before_writing() -> None:
    """真漏一片：单 rank 视角完全自洽，只有全局骨架能发现。"""
    per_rank = {
        0: [S("w", 0, 4)],
        1: [S("w", 4, 4)],
    }
    # 先证明局部视角查不出来：rank 0 只看到自己那一片
    assert build_save_plan(per_rank[0], rank=0).validate() is None

    # 再让第二片消失：单 rank 依旧自洽，只有全局骨架报覆盖不全
    per_rank[1] = []
    assert build_save_plan(per_rank[0], rank=0).validate() is None
    with pytest.raises(PlanError, match="cover"):
        plan_ranks(per_rank)


def test_plan_for_rejects_unknown_rank() -> None:
    global_plan = plan_ranks({0: [S("w", 0, 8)]})
    with pytest.raises(PlanError, match="不在全局计划里"):
        global_plan.plan_for(5)


# --- 纯 TP：每片只有一份 ---------------------------------------------------


def test_pure_tp_shards_have_no_replicas() -> None:
    """TP=4 每 rank 一片不同区域：每块覆盖范围只有 replica_index=0 一份。"""
    per_rank = {r: [S("w", r * 2, 2)] for r in range(4)}
    global_plan = plan_ranks(per_rank)
    assert global_plan.skeleton.tensors["w"].covered_numel == 64
    assert all(plan.items for plan in global_plan.per_rank.values())
