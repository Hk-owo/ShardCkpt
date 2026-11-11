"""L1 单进程训练 loop 仿真：周期性保存 → 版本隔离 → 恢复 → 继续训练。

这里验证的是**写闭环和读闭环拼在一起**才暴露的性质：单文件用例各自都是绿的，
但它们对「版本怎么隔离」「恢复后能不能接着训练」的假设没人校过。

伪训练是确定性的（见 ``sim.TrainingState``），所以「恢复后继续跑」可以和不间断
跑的结果逐字节比对，不需要容差。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import api as ckpt
import sim
from async_ import PinnedBufferPool, SnapshotManager, SnapshotState, aligned_allocator
from commit.local import committed_steps, latest_step
from storage.local import LocalStorage
from storage.local_load import load_local

SAVE_EVERY = 3
TOTAL_STEPS = 12
DEVICE = "cuda:0"


def run_training(
    state: sim.TrainingState,
    *,
    steps: int,
    root: Path,
    save_every: int = SAVE_EVERY,
    pool: PinnedBufferPool | None = None,
    manager: SnapshotManager | None = None,
) -> tuple[sim.TrainingState, dict[int, sim.TrainingState]]:
    """跑 ``steps`` 步，每 ``save_every`` 步存一次；返回 (最终状态, {step: 当时的快照})。"""
    saved: dict[int, sim.TrainingState] = {}
    current = state
    for _ in range(steps):
        current = current.step_once()
        if current.step % save_every:
            continue
        tensors = sim.to_device(current, device=DEVICE)
        report = ckpt.save_local(
            tensors,
            root,
            step=current.step,
            blobs={"rng/rank_0": bytes([current.step]) * 16},
            layout={"tp": 1, "dp": 1},
            pool=pool,
            manager=manager,
        )
        assert report.state is SnapshotState.COMMITTED
        saved[current.step] = current.snapshot()
    return current, saved


def host_arrays(tensors):
    """device 张量 → host 数组（走 ``Tensor.numpy()``，即 D2H 路径）。"""
    return {name: tensor.numpy() for name, tensor in tensors.items()}


def fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


# --- 主路径 ----------------------------------------------------------------


def test_periodic_saves_are_individually_recoverable(state, tmp_path) -> None:
    root = tmp_path / "ckpt"
    final, saved = run_training(state, steps=TOTAL_STEPS, root=root)

    assert sorted(saved) == [3, 6, 9, 12]
    storage = LocalStorage(root)
    assert committed_steps(storage) == [3, 6, 9, 12]
    assert latest_step(storage) == 12

    for step, expected in saved.items():
        report = load_local(root, step=step, device=DEVICE)
        sim.assert_state_matches(host_arrays(report.tensors), expected)
        assert report.blobs[f"rng/rank_0"] == bytes([step]) * 16

    # 伪训练本身要有效：12 步之后每个张量都和初始状态不同
    assert final.diff(state) == final.names()


def test_each_version_keeps_its_own_bytes(state, tmp_path) -> None:
    """step 3 落盘之后又写了 6/9/12，读回来仍然必须是 step 3 的内容。"""
    root = tmp_path / "ckpt"
    _, saved = run_training(state, steps=TOTAL_STEPS, root=root)

    early = load_local(root, step=3, device=DEVICE)
    late = load_local(root, step=12, device=DEVICE)

    sim.assert_state_matches(host_arrays(early.tensors), saved[3])
    sim.assert_state_matches(host_arrays(late.tensors), saved[12])
    assert saved[3].diff(saved[12]) == saved[3].names(), "版本之间必须真有差异"

    # 两个版本的数据文件互不覆盖
    assert early.index.tensors["model.embed_tokens.weight"].shards[0].file == "__0_0.distcp"
    assert (root / "step_3" / "__0_0.distcp").read_bytes() != (
        root / "step_12" / "__0_0.distcp"
    ).read_bytes()


def test_resume_from_checkpoint_continues_the_same_trajectory(state, tmp_path) -> None:
    """从 step 6 恢复再跑 6 步，与一口气跑 12 步逐字节相同。"""
    root = tmp_path / "ckpt"
    uninterrupted = state.step_once(TOTAL_STEPS)
    _, saved = run_training(state, steps=6, root=root)

    report = load_local(root, step=6, device=DEVICE)
    resumed = sim.state_from_tensors(report.tensors, step=report.step).step_once(6)

    assert resumed.step == uninterrupted.step == 12
    sim.assert_state_matches(resumed.values, uninterrupted)
    sim.assert_state_matches(resumed.values, saved[6].step_once(6))


# --- 布局变更 --------------------------------------------------------------


def test_tp_change_after_save_rebuilds_the_global_state(state, tmp_path) -> None:
    """TP=1 写、TP=4 读：每个 rank 拿自己的列，拼起来等于全局。"""
    root = tmp_path / "ckpt"
    _, saved = run_training(state, steps=3, root=root)
    expected = saved[3]
    dims = sim.column_dims(expected)

    parts = []
    for rank in range(4):
        report = load_local(
            root, step=3, device=DEVICE, tp=4, tp_rank=rank, shard_dims=dims
        )
        parts.append(host_arrays(report.tensors))
        assert report.amplification > 1.0, "column_strategy=whole 下应读出放大"

    merged = sim.reassemble(parts, expected, dims=dims)
    sim.assert_state_matches(merged, expected)

    # 整读路径（tp=1）与换布局路径必须给出同一份数据
    whole = load_local(root, step=3, device=DEVICE)
    sim.assert_state_matches(host_arrays(whole.tensors), expected)


# --- 长跑与资源回收 --------------------------------------------------------


def test_many_steps_reuse_one_pool_without_leaking(state, tmp_path) -> None:
    """18 步共用一块 pinned 池：块被复用、没有残留占用、fd 不涨。"""
    root = tmp_path / "ckpt"
    pool = PinnedBufferPool(
        chunk_bytes=1 << 20,
        min_chunks=1,
        max_bytes=4 << 20,
        timeout=0,
        allocator=aligned_allocator(),
    )
    manager = SnapshotManager(pool)

    # 先跑一轮热身，让文件和 backend 的首次分配都发生，再记录 fd 基线
    run_training(
        state,
        steps=1,
        root=root / "warmup",
        save_every=1,
        pool=pool,
        manager=manager,
    )
    load_local(root / "warmup", device=DEVICE, budget_bytes=64 << 10, pool=pool)
    fds_before = fd_count()

    _, saved = run_training(state, steps=18, root=root, pool=pool, manager=manager)
    for step in saved:
        load_local(root, step=step, device=DEVICE, budget_bytes=64 << 10, pool=pool)

    stats = pool.stats
    assert stats.in_use_bytes == 0, "所有块都必须归还"
    assert stats.reuses > 0, "跨步复用才说明池在工作，而不是每次重新分配"
    assert stats.capacity_bytes <= 4 << 20, "复用不应把池撑大"
    assert manager.active == (), "每代快照都必须走到终态并归还 buffer"
    assert fd_count() <= fds_before + 2, "文件描述符不能随步数增长"


@pytest.mark.parametrize("dtype", ["f32", "f16", "bf16"])
def test_training_loop_works_for_other_dtypes(dtype: str, tmp_path) -> None:
    """f16 / bf16 的 state 也要能走完整闭环（bf16 用 uint16 承载）。"""
    state = sim.make_state(step=0, dtype=dtype)
    root = tmp_path / f"ckpt-{dtype}"
    _, saved = run_training(state, steps=3, root=root)

    report = load_local(root, step=3, device=DEVICE)
    sim.assert_state_matches(host_arrays(report.tensors), saved[3])
    assert {tensor.dtype for tensor in report.tensors.values()} == {dtype}
