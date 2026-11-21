"""L1/L2 异步训练 loop 仿真：每 3 步异步保存跑 12 步，最终恢复内容与同步版逐字节一致。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import api as ckpt
import s3_stub
import sim
from saver.comm import NodeComm, ROLE_TRAINER
from saver.config import SaverConfig
from storage.local_load import load_local
from storage.object import S3Config, S3Storage

pytestmark = pytest.mark.sim

SAVE_EVERY = 3
TOTAL_STEPS = 12
DEVICE = "cuda:0"


def test_async_train_loop_matches_sync(state, tmp_path: Path) -> None:
    """单卡：每 3 步 save_async 跑 12 步，与 save_local 同步版每代内容逐字节一致。"""
    sync_root = tmp_path / "sync"
    async_root = tmp_path / "async"

    # 同步基线
    cur_sync = state
    for _ in range(TOTAL_STEPS):
        cur_sync = cur_sync.step_once()
        if cur_sync.step % SAVE_EVERY == 0:
            ckpt.save_local(
                sim.to_device(cur_sync, device=DEVICE),
                sync_root,
                step=cur_sync.step,
                blobs={"rng/rank_0": bytes([cur_sync.step]) * 16},
                layout={"tp": 1, "dp": 1},
            )

    # 异步版（双缓冲）
    cfg = SaverConfig(local_root=async_root, max_inflight=2)
    ckpt.start_saver(cfg)
    snapshots: dict[int, sim.TrainingState] = {}
    try:
        cur_async = state
        handles = []
        for _ in range(TOTAL_STEPS):
            cur_async = cur_async.step_once()
            if cur_async.step % SAVE_EVERY == 0:
                snapshots[cur_async.step] = cur_async.snapshot()
                h = ckpt.save_async(
                    sim.to_device(cur_async, device=DEVICE),
                    step=cur_async.step,
                    blobs={"rng/rank_0": bytes([cur_async.step]) * 16},
                    layout={"tp": 1, "dp": 1},
                    upload=False,
                )
                handles.append((cur_async.step, h))
        for _step, h in handles:
            res = h.wait(timeout=15.0)
            assert res is not None
    finally:
        ckpt.close_saver(timeout=15.0)

    for step in (3, 6, 9, 12):
        sync_rep = load_local(sync_root, step=step, device=DEVICE)
        async_rep = load_local(async_root, step=step, device=DEVICE)
        sync_arr = {k: v.numpy() for k, v in sync_rep.tensors.items()}
        async_arr = {k: v.numpy() for k, v in async_rep.tensors.items()}
        sim.assert_state_matches(sync_arr, snapshots[step])
        sim.assert_state_matches(async_arr, snapshots[step])
        assert async_rep.blobs == sync_rep.blobs


def test_async_multirank_train_loop_with_s3_and_node_comm(state, tmp_path: Path) -> None:
    """多卡：NodeComm + S3 后台上传 + 有界保留 + 对称退出收敛后可完整恢复 step 12。"""
    tp = 2
    states: list[sim.TrainingState] = []
    cur = state
    for _ in range(TOTAL_STEPS):
        cur = cur.step_once()
        if cur.step % SAVE_EVERY == 0:
            states.append(cur.snapshot())

    with s3_stub.start() as server, NodeComm(world_size=tp, base_dir=tmp_path / "comm") as comm:
        s3_cfg = S3Config(
            bucket="ckpt",
            prefix="runs",
            endpoint_url=server.endpoint_url,
            access_key_id="k",
            secret_access_key="s",
        )
        savers = [
            ckpt.AsyncSaver(
                SaverConfig(
                    local_root=tmp_path / f"r{r}",
                    remote_config=s3_cfg,
                    run_id="run_0001",
                    rank=r,
                    ranks=(0, 1),
                    max_inflight=2,
                    keep_versions=3,
                    keep_local=1,
                    shutdown_timeout=20.0,
                ),
                comm=comm.client(r, role=ROLE_TRAINER),
                world_size=tp,
            )
            for r in range(tp)
        ]

        def run_rank(r: int):
            saver = savers[r]
            try:
                dims = sim.column_dims(state)
                handles = []
                for snap in states:
                    t = sim.tensors_for_rank(snap, rank=r, tp=tp, dims=dims, device="cpu")
                    srcs = sim.sources_for_rank(snap, rank=r, tp=tp, dims=dims)
                    h = saver.save(
                        t,
                        step=snap.step,
                        sources=srcs,
                        blobs={f"rng/rank_{r}": bytes([snap.step + r]) * 16},
                        layout={"tp": tp},
                        planner="gathered",
                    )
                    handles.append(h)
                for h in handles:
                    assert h.wait(timeout=15.0) is not None
            finally:
                saver.close(timeout=20.0)

        with ThreadPoolExecutor(max_workers=tp) as pool:
            for f in [pool.submit(run_rank, r) for r in range(tp)]:
                f.result(timeout=45.0)

        storage = S3Storage(s3_cfg)
        try:
            restored = ckpt.load_object(storage, step=12, device="cpu", run_id="run_0001")
            sim.assert_state_matches(
                {k: v.numpy() for k, v in restored.tensors.items()}, states[-1]
            )
            assert set(restored.blobs) == {"rng/rank_0", "rng/rank_1"}
        finally:
            storage.close()
