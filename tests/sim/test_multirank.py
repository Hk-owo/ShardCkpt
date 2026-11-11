"""L2 多 rank 分片仿真：多进程各写一份 → rank0 汇总 → 任意布局恢复。

``dist/`` 的 gather 还没实现，所以 rank0 的汇总走 ``sim.merge_rank_versions`` 这条
test-only 旁路（读回各 rank 的 ``.metadata`` 再拼）。真 gather 落地后，这里换成
collective，断言一条都不用改 —— 这正是这层仿真的价值。

进程用 ``spawn``：每个 rank 拿到干净的 backend 注册表，行为与「独立训练进程」一致。
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

import sim
from commit.local import committed_steps, latest_step
from core import PlanError
from core.index import META_FILE, CheckpointIndex
from planner.save_plan import default_writer
from storage.local import LocalStorage
from storage.local_load import load_local

DEVICE = "cuda:0"
STEP = 1


def blobs_for(rank: int) -> dict[str, bytes]:
    """每个 rank 自己的小状态（RNG 位置不同，正好验证按 rank 落盘）。"""
    return {f"rng/rank_{rank}": bytes([rank]) * 16, "lr_scheduler": b"L" * 32}


def save_all_ranks(
    state: sim.TrainingState, *, tp: int, root: Path, dims: dict[str, int]
) -> list[dict]:
    """spawn ``tp`` 个进程，每个 rank 写自己那份本地版本，返回各自的摘要。"""
    jobs = [
        {
            "root": str(root / f"rank_{rank}"),
            "step": STEP,
            "rank": rank,
            "tp": tp,
            "dims": dims,
            "values": state.values,
            "dtypes": state.dtypes,
            "blobs": blobs_for(rank),
            "device": DEVICE,
        }
        for rank in range(tp)
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=tp, mp_context=context) as pool:
        return list(pool.map(sim.save_rank, jobs))


def rank_root(root: Path, rank: int) -> Path:
    return root / f"rank_{rank}"


def restore_parts(root: Path, expected: sim.TrainingState, *, tp: int) -> list[dict]:
    """按 ``tp`` 布局逐 rank 恢复，返回每个 rank 的 host 数组。"""
    dims = sim.column_dims(expected)
    parts = []
    for rank in range(tp):
        report = load_local(
            root, step=STEP, device=DEVICE, tp=tp, tp_rank=rank, shard_dims=dims
        )
        parts.append({name: tensor.numpy() for name, tensor in report.tensors.items()})
    return parts


# --- 汇总后的索引 ----------------------------------------------------------


def test_merged_version_is_globally_consistent(state, tmp_path) -> None:
    tp = 2
    dims = sim.column_dims(state)
    root = tmp_path / "ckpt"

    summaries = save_all_ranks(state, tp=tp, root=root, dims=dims)
    assert [item["rank"] for item in summaries] == list(range(tp))
    assert all(item["state"] == "COMMITTED" for item in summaries)

    merged = tmp_path / "merged"
    index = sim.merge_rank_versions(
        [rank_root(root, rank) for rank in range(tp)], merged, step=STEP, layout={"tp": tp}
    )
    index.validate()  # 全局覆盖 + 同一文件内区间不重叠

    storage = LocalStorage(merged)
    assert committed_steps(storage) == [STEP]
    assert latest_step(storage) == STEP
    assert index.saved_layout == {"tp": tp}

    for name, meta in index.tensors.items():
        if name in dims:
            assert len(meta.shards) == tp, f"{name}: 每个 rank 一个分片"
        else:
            assert len(meta.shards) == 1, f"{name}: Replicate 只落一份"


def test_every_tp_rank_rebuilds_the_global_state(state, tmp_path) -> None:
    tp = 4
    saved_state = state.step_once()
    dims = sim.column_dims(saved_state)
    root = tmp_path / "ckpt"
    save_all_ranks(saved_state, tp=tp, root=root, dims=dims)

    merged = tmp_path / "merged"
    sim.merge_rank_versions(
        [rank_root(root, rank) for rank in range(tp)], merged, step=STEP, layout={"tp": tp}
    )

    parts = restore_parts(merged, saved_state, tp=tp)
    sim.assert_state_matches(sim.reassemble(parts, saved_state, dims=dims), saved_state)

    # 布局不变（tp=1）的整读路径也必须给出同一份数据
    whole = load_local(merged, step=STEP, device=DEVICE)
    sim.assert_state_matches(
        {name: tensor.numpy() for name, tensor in whole.tensors.items()}, saved_state
    )


# --- 写者分配 --------------------------------------------------------------


def test_replicate_shards_have_exactly_one_writer(state, tmp_path) -> None:
    """Replicate 张量只由一个 rank 落盘，且每个字节只被写一次。"""
    tp = 4
    dims = sim.column_dims(state)
    root = tmp_path / "ckpt"
    summaries = save_all_ranks(state, tp=tp, root=root, dims=dims)

    owners: dict[str, list[int]] = {}
    for rank in range(tp):
        index = CheckpointIndex.loads(
            LocalStorage(rank_root(root, rank))
            .read(f"step_{STEP}/{META_FILE}")
            .decode("utf-8")
        )
        for name in index.tensors:
            owners.setdefault(name, []).append(rank)

    for name in state.names():
        if name in dims:
            assert owners[name] == list(range(tp)), f"{name}: 分片张量每个 rank 都写"
        else:
            assert owners[name] == [default_writer(name, tp)], f"{name}: Replicate 写者错"

    # 各 rank 的 D2H 总量加起来正好等于数据量（没有整份复制给每个 rank）
    global_bytes = state.step_once().total_bytes()
    total_d2h = sum(item["d2h_bytes"] for item in summaries)
    blobs_total = sum(
        sum(len(payload) for payload in blobs_for(rank).values()) for rank in range(tp)
    )
    assert total_d2h == global_bytes + blobs_total


def test_blobs_are_written_by_their_own_rank(state, tmp_path) -> None:
    tp = 3
    dims = sim.column_dims(state)
    root = tmp_path / "ckpt"
    save_all_ranks(state, tp=tp, root=root, dims=dims)

    merged = tmp_path / "merged"
    index = sim.merge_rank_versions(
        [rank_root(root, rank) for rank in range(tp)], merged, step=STEP, layout={"tp": tp}
    )
    assert sorted(index.blobs) == sorted(
        [f"rng/rank_{rank}" for rank in range(tp)] + ["lr_scheduler"]
    )

    report = load_local(merged, step=STEP, device=DEVICE)
    for rank in range(tp):
        assert report.blobs[f"rng/rank_{rank}"] == bytes([rank]) * 16
    assert report.blobs["lr_scheduler"] == b"L" * 32


# --- 汇总的守卫 ------------------------------------------------------------


def test_merge_rejects_incomplete_coverage(state, tmp_path) -> None:
    """少一个 rank 的分片就是覆盖不全，汇总必须直接拒绝而不是写半份索引。"""
    tp = 2
    dims = sim.column_dims(state)
    root = tmp_path / "ckpt"
    save_all_ranks(state, tp=tp, root=root, dims=dims)

    with pytest.raises(PlanError):
        sim.merge_rank_versions(
            [rank_root(root, 0)], tmp_path / "merged", step=STEP, layout={"tp": tp}
        )
    assert not (tmp_path / "merged" / f"step_{STEP}" / META_FILE).exists()


def test_merge_is_byte_identical_across_runs(state, tmp_path) -> None:
    """同样的 rank 输入，汇总两次得到同一份 ``.metadata``。"""
    tp = 2
    dims = sim.column_dims(state)
    root = tmp_path / "ckpt"
    save_all_ranks(state, tp=tp, root=root, dims=dims)

    first = sim.merge_rank_versions(
        [rank_root(root, rank) for rank in range(tp)], tmp_path / "m1", step=STEP
    )
    second = sim.merge_rank_versions(
        [rank_root(root, rank) for rank in range(tp)], tmp_path / "m2", step=STEP
    )
    assert first.dumps() == second.dumps()
