"""M3 测试：节点独立通信转发层（树形归约/审计/广播/作废）+ 存储侧上传协调 + GC 三类保护集。"""

from __future__ import annotations

import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import api as ckpt
import s3_stub
import sim
from commit.gc import LOCAL, PROTECTED, REMOTE, GcWorker, collect, plan
from commit.generation import acquire_generation
from commit.local import write_local_generation
from core import PlanError
from saver.comm import (
    CommError,
    NodeComm,
    NodeCommClient,
    NodeCommProcess,
    NodeCommServer,
    ROLE_STORAGE,
    ROLE_TRAINER,
    make_tree_configs,
)
from saver.config import SaverConfig, SaverFaults
from storage.base import run_scope
from storage.object import S3Config, S3Storage
from storage.upload import committed_remote_steps, upload_rank

DEVICE = "cpu"


@pytest.fixture
def s3_server():
    with s3_stub.start() as server:
        yield server


@pytest.fixture
def s3_config(s3_server) -> S3Config:
    return S3Config(
        bucket="ckpt",
        prefix="runs",
        endpoint_url=s3_server.endpoint_url,
        access_key_id="test-key",
        secret_access_key="test-secret",
    )


def _wait_until(predicate, *, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _close_all(savers: list[ckpt.AsyncSaver], *, timeout: float = 5.0) -> None:
    with ThreadPoolExecutor(max_workers=max(1, len(savers))) as pool:
        futs = [pool.submit(s.close, timeout=timeout) for s in savers]
        for f in futs:
            f.result(timeout=timeout + 3.0)


def _tensors_and_sources(state: sim.TrainingState, *, rank: int, tp: int):
    dims = sim.column_dims(state)
    tensors = sim.tensors_for_rank(state, rank=rank, tp=tp, dims=dims, device=DEVICE)
    sources = sim.sources_for_rank(state, rank=rank, tp=tp, dims=dims)
    return tensors, sources


def test_async_gathered_tree_audit_and_run_id(
    state, tmp_path: Path, s3_config: S3Config
) -> None:
    """多节点树形预去重覆盖审计 + 指纹确认 + run_id 轮次隔离 + step 单调校验 + 对称收敛。"""
    s1 = state.step_once()
    s2 = s1.step_once()
    num_nodes = 2
    ranks_per_node = 2
    tp = num_nodes * ranks_per_node  # 4 ranks 跨越 2 个节点通信树
    tree_cfgs = make_tree_configs(
        num_nodes=num_nodes,
        ranks_per_node=ranks_per_node,
        base_dir=tmp_path / "comm",
        fanout=2,
    )
    servers = [NodeCommServer(cfg).bind() for cfg in tree_cfgs]
    for srv in servers:
        srv.start()

    try:
        # 1. 先验证树形布局审计能捕获分片缺失/覆盖不全（PlanError）
        bad_clients = [
            NodeCommClient(servers[r // ranks_per_node].endpoint, rank=r, role=ROLE_TRAINER)
            for r in range(tp)
        ]
        try:
            def audit_with_missing(r: int):
                _t, srcs = _tensors_and_sources(s1, rank=r, tp=tp)
                if r == tp - 1:
                    srcs = srcs[:-1]  # 最后一个 rank 少报一个分片
                return bad_clients[r].audit_save_plan(srcs, timeout=5.0)

            with ThreadPoolExecutor(max_workers=tp) as pool:
                futs = [pool.submit(audit_with_missing, r) for r in range(tp)]
                for f in futs:
                    with pytest.raises(PlanError):
                        f.result(timeout=10.0)
        finally:
            for c in bad_clients:
                c.close()

        # 2. 完整 4-rank gathered 异步保存 + 远端上传 + 退出对称收敛
        savers = [
            ckpt.AsyncSaver(
                SaverConfig(
                    local_root=tmp_path / f"local-r{r}",
                    remote_config=s3_config,
                    run_id="run_0001",
                    rank=r,
                    ranks=tuple(range(tp)),
                    comm_endpoint=servers[r // ranks_per_node].endpoint,
                    upload_timeout=15.0,
                    shutdown_timeout=15.0,
                ),
                world_size=tp,
            )
            for r in range(tp)
        ]

        def run_rank(r: int):
            saver = savers[r]
            try:
                t1, src1 = _tensors_and_sources(s1, rank=r, tp=tp)
                h1 = saver.save(
                    t1,
                    step=1,
                    sources=src1,
                    blobs={f"rng/rank_{r}": bytes([r + 1]) * 16},
                    layout={"tp": tp},
                    planner="gathered",
                )
                assert h1.wait(timeout=10.0) is not None

                # 同 run_id 内 step 必须严格单调递增
                with pytest.raises(PlanError, match="必须严格大于"):
                    saver.save(t1, step=1, sources=src1, planner="gathered")

                t2, src2 = _tensors_and_sources(s2, rank=r, tp=tp)
                h2 = saver.save(
                    t2,
                    step=2,
                    sources=src2,
                    blobs={f"rng/rank_{r}": bytes([r + 2]) * 16},
                    layout={"tp": tp},
                    planner="gathered",
                )
                assert h2.wait(timeout=10.0) is not None
            finally:
                saver.close(timeout=15.0)

        with ThreadPoolExecutor(max_workers=tp) as pool:
            for f in [pool.submit(run_rank, r) for r in range(tp)]:
                f.result(timeout=30.0)

        # 3. 远端恢复 run_0001 step 2，逐字节一致
        storage = S3Storage(s3_config)
        try:
            restored = ckpt.load_object(storage, step=2, device=DEVICE, run_id="run_0001")
            sim.assert_state_matches(
                {k: v.numpy() for k, v in restored.tensors.items()}, s2
            )
            assert set(restored.blobs) == {f"rng/rank_{r}" for r in range(tp)}
        finally:
            storage.close()
    finally:
        for srv in servers:
            srv.close()


def test_node_comm_tree_disconnect_aborts_strong_sync(tmp_path: Path) -> None:
    """强同步原语下任一子节点断连，全树一致放弃并抛 CommError，绝不降级。"""
    cfgs = make_tree_configs(
        num_nodes=2, ranks_per_node=1, base_dir=tmp_path / "comm", fanout=2
    )
    servers = [NodeCommServer(cfg).bind() for cfg in cfgs]
    for s in servers:
        s.start()

    c0 = NodeCommClient(servers[0].endpoint, rank=0, role=ROLE_TRAINER)
    c1 = NodeCommClient(servers[1].endpoint, rank=1, role=ROLE_TRAINER)
    try:
        # 只有 rank 0 进入门闸等待时，杀掉子节点 server 1（模拟物理机失联）
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(c0.all_reduce_max_int64, [0, 1], timeout=5.0)
            time.sleep(0.05)
            c1.close()
            servers[1].close()
            with pytest.raises(CommError):
                fut.result(timeout=5.0)
    finally:
        c0.close()
        c1.close()
        for s in servers:
            s.close()


def test_gc_protects_persisting_uploading_and_recovering_steps(
    state, tmp_path: Path, s3_config: S3Config
) -> None:
    """I12 / Q5：落盘中、上传中、恢复读取中的 step 均不被 GC 删除，且严格按 run_id 隔离。"""
    # 1. 纯函数 plan() 验证三类保护集
    gc_plan = plan(
        versions=[1, 2, 5],
        rank_commits=[(1, 0, "gen_0001-aaaa"), (2, 0, "gen_0001-bbbb"), (5, 0, "gen_0001-cccc")],
        prefixes=[1, 2, 3, 4, 5],
        local_steps=[1, 2, 3, 4, 5],
        rank=0,
        generations={
            1: ["gen_0001-aaaa"],
            2: ["gen_0001-bbbb"],
            3: ["gen_0001-dddd"],
            4: ["gen_0001-eeee"],
            5: ["gen_0001-cccc"],
        },
        current_generations={1: "gen_0001-aaaa", 2: "gen_0001-bbbb", 5: "gen_0001-cccc"},
        pointer_generations={1: "gen_0001-aaaa", 2: "gen_0001-bbbb", 5: "gen_0001-cccc"},
        keep_local=1,
        protected_steps=[1, 3, 4],  # 1: 恢复读取中, 3: 上传中, 4: 落盘中
    )
    assert gc_plan.steps_for(LOCAL) == (2,)
    assert gc_plan.steps_for(REMOTE) == ()
    for protected_step in (1, 3, 4):
        assert (LOCAL, protected_step, PROTECTED) in gc_plan.kept
        assert (REMOTE, protected_step, PROTECTED) in gc_plan.kept

    # 2. 真实目录 + S3 + GcWorker + run_id 跨轮次隔离验证
    s1 = state.step_once()
    storage = S3Storage(s3_config)
    root = tmp_path / "local"
    try:
        # run_0001 存 step 1 和 2；run_0002 存 step 1 和 2
        for rid in ("run_0001", "run_0002"):
            for step in (1, 2):
                gen = acquire_generation(run_scope(storage, rid), step, ranks=[0])
                ckpt.save_local(
                    sim.to_device(s1, device=DEVICE),
                    root,
                    step=step,
                    generation=gen.id,
                    run_id=rid,
                )
                ckpt.upload_local(root, storage, step=step, rank=0, generation=gen, run_id=rid)
                ckpt.finalize_upload(storage, step=step, ranks=[0], generation=gen, run_id=rid)

        # 为 run_0001 的 step 1 加恢复保护，对 run_0001 跑 GC：step 1 不被删，且 run_0002 完全不受影响
        gc = GcWorker(root, storage, rank=0, keep_local=1, run_id="run_0001")
        gc.protect([1], tag="recovering")
        rep = gc.run_once()
        assert rep.deleted == ()
        assert (root / "run_0001" / "step_1" / "COMMIT").is_file()
        assert (root / "run_0002" / "step_1" / "COMMIT").is_file()
        assert (root / "run_0002" / "step_2" / "COMMIT").is_file()

        # 解除保护后，run_0001 的 step 1 被收掉，但 run_0002 仍毫发无损
        gc.unprotect([1], tag="recovering")
        rep2 = gc.run_once()
        assert rep2.steps_for(LOCAL) == (1,)
        assert not (root / "run_0001" / "step_1").exists()
        assert (root / "run_0002" / "step_1" / "COMMIT").is_file()
    finally:
        storage.close()


def test_async_instant_invalidation_broadcast(
    state, tmp_path: Path, s3_config: S3Config
) -> None:
    """I9：某卡上传失败，存储侧立即经节点通信层广播 INVALIDATE_EVENT，同伴当场取消上传止损。"""
    s1 = state.step_once()
    tp = 2
    with NodeComm(world_size=tp, base_dir=tmp_path / "comm") as comm:
        s0 = ckpt.AsyncSaver(
            SaverConfig(
                local_root=tmp_path / "r0",
                remote_config=s3_config,
                run_id="run_0001",
                rank=0,
                ranks=(0, 1),
                faults=SaverFaults(slow_upload_ms=5000.0),
            ),
            comm=comm.client(0, role=ROLE_TRAINER),
            world_size=tp,
        )
        s1_saver = ckpt.AsyncSaver(
            SaverConfig(
                local_root=tmp_path / "r1",
                remote_config=s3_config,
                run_id="run_0001",
                rank=1,
                ranks=(0, 1),
                faults=SaverFaults(fail_upload_step=1, fail_upload_rank=1),
            ),
            comm=comm.client(1, role=ROLE_TRAINER),
            world_size=tp,
        )

        def save_one(saver: ckpt.AsyncSaver, rank: int):
            t, srcs = _tensors_and_sources(s1, rank=rank, tp=tp)
            return saver.save(t, step=1, sources=srcs).wait(timeout=10.0)

        t0 = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=tp) as pool:
                f0 = pool.submit(save_one, s0, 0)
                f1 = pool.submit(save_one, s1_saver, 1)
                assert f0.result(timeout=10.0) is not None
                assert f1.result(timeout=10.0) is not None

            # 不调用下一次 save_async，等同伴因即时广播作废而快速结束（<< 5s）
            assert _wait_until(
                lambda: len(s1_saver.upload_failures) >= 1 and s0.upload_done_steps == (),
                timeout=3.0,
            )
        finally:
            _close_all([s0, s1_saver], timeout=3.0)
        elapsed = time.monotonic() - t0
        assert elapsed < 3.0, f"rank 0 未被即时作废信号打断，耗时 {elapsed:.2f}s"

        storage = S3Storage(s3_config)
        try:
            assert committed_remote_steps(run_scope(storage, "run_0001")) == []
        finally:
            storage.close()


def test_invalidation_before_local_commit_skips_upload(
    state, tmp_path: Path, s3_config: S3Config
) -> None:
    """作废事件先于本地 L1 落盘到达：落盘后查作废表直接跳过远端上传。"""
    s1 = state.step_once()
    tp = 2
    with NodeComm(world_size=tp, base_dir=tmp_path / "comm") as comm:
        # rank 0 本地落盘慢（400ms），rank 1 立即上传失败并先广播作废
        s0 = ckpt.AsyncSaver(
            SaverConfig(
                local_root=tmp_path / "r0",
                remote_config=s3_config,
                run_id="run_0001",
                rank=0,
                ranks=(0, 1),
                faults=SaverFaults(slow_write_ms=400.0),
            ),
            comm=comm.client(0, role=ROLE_TRAINER),
            world_size=tp,
        )
        s1_saver = ckpt.AsyncSaver(
            SaverConfig(
                local_root=tmp_path / "r1",
                remote_config=s3_config,
                run_id="run_0001",
                rank=1,
                ranks=(0, 1),
                faults=SaverFaults(fail_upload_step=1, fail_upload_rank=1),
            ),
            comm=comm.client(1, role=ROLE_TRAINER),
            world_size=tp,
        )

        def save_one(saver: ckpt.AsyncSaver, rank: int):
            t, srcs = _tensors_and_sources(s1, rank=rank, tp=tp)
            return saver.save(t, step=1, sources=srcs).wait(timeout=10.0)

        try:
            with ThreadPoolExecutor(max_workers=tp) as pool:
                f0 = pool.submit(save_one, s0, 0)
                f1 = pool.submit(save_one, s1_saver, 1)
                assert f0.result(timeout=10.0) is not None
                assert f1.result(timeout=10.0) is not None
        finally:
            _close_all([s0, s1_saver], timeout=5.0)

        storage = S3Storage(s3_config)
        try:
            keys = run_scope(storage, "run_0001").list("")
            # rank 0 在 L1 后查作废表跳过上传，未向远端写任何对象
            assert not any("rank_0" in k for k in keys), f"rank 0 不应上传：{keys}"
        finally:
            storage.close()


def test_worker_crash_watchdog_compensates_invalidation(
    state, tmp_path: Path, s3_config: S3Config
) -> None:
    """某卡存储进程崩死来不及广播时，训练侧看门狗补偿广播 INVALIDATE_EVENT 令同伴止损。"""
    s1 = state.step_once()
    tp = 2
    with NodeComm(world_size=tp, base_dir=tmp_path / "comm") as comm:
        s0 = ckpt.AsyncSaver(
            SaverConfig(
                local_root=tmp_path / "r0",
                remote_config=s3_config,
                run_id="run_0001",
                rank=0,
                ranks=(0, 1),
                faults=SaverFaults(slow_upload_ms=5000.0),
            ),
            comm=comm.client(0, role=ROLE_TRAINER),
            world_size=tp,
        )
        s1_saver = ckpt.AsyncSaver(
            SaverConfig(
                local_root=tmp_path / "r1",
                remote_config=s3_config,
                run_id="run_0001",
                rank=1,
                ranks=(0, 1),
                heartbeat_interval=0.05,
                heartbeat_timeout=0.4,
                restart_backoff=0.01,
                faults=SaverFaults(slow_write_ms=2000.0),
            ),
            comm=comm.client(1, role=ROLE_TRAINER),
            world_size=tp,
        )

        t0 = time.monotonic()
        try:
            def run_rank(saver: ckpt.AsyncSaver, rank: int):
                t, srcs = _tensors_and_sources(s1, rank=rank, tp=tp)
                h = saver.save(t, step=1, sources=srcs)
                if rank == 1:
                    h.wait_d2h(timeout=5.0)
                    time.sleep(0.05)
                    # 直接 SIGKILL rank 1 的存储子进程
                    assert saver.worker_pid is not None
                    os.kill(saver.worker_pid, signal.SIGKILL)
                return h

            with ThreadPoolExecutor(max_workers=tp) as pool:
                f0 = pool.submit(run_rank, s0, 0)
                f1 = pool.submit(run_rank, s1_saver, 1)
                h0 = f0.result(timeout=10.0)
                h1 = f1.result(timeout=10.0)

            assert h0.wait(timeout=10.0) is not None
            with pytest.raises(ckpt.SaverDeadError):
                h1.wait(timeout=10.0)
        finally:
            _close_all([s0, s1_saver], timeout=3.0)

        elapsed = time.monotonic() - t0
        assert elapsed < 3.5, f"看门狗补偿作废未在窗口内令同伴退出：{elapsed:.2f}s"


def test_recovery_holds_gc_until_cluster_confirmed(
    state, tmp_path: Path, s3_config: S3Config
) -> None:
    """启动恢复期间暂停/锁定 GC，完成尾巴提升与全员加载复核后才放行 GC 并同步水位线。"""
    s1 = state.step_once()
    s2 = s1.step_once()
    tp = 2
    storage = S3Storage(s3_config)
    scoped = run_scope(storage, "run_0001")

    try:
        # 预先写入 step 1（已提交）和 step 2（L1+L2 齐全、缺 L3 的尾巴版本）
        for step, st in ((1, s1), (2, s2)):
            gen = acquire_generation(scoped, step, ranks=(0, 1))
            for r in (0, 1):
                t, srcs = _tensors_and_sources(st, rank=r, tp=tp)
                ckpt.save_local(
                    t,
                    tmp_path / f"r{r}",
                    step=step,
                    sources=srcs,
                    generation=gen.id,
                    run_id="run_0001",
                )
                upload_rank(
                    tmp_path / f"r{r}",
                    storage,
                    step=step,
                    rank=r,
                    generation=gen,
                    run_id="run_0001",
                )
            if step == 1:
                ckpt.finalize_upload(
                    storage, step=1, ranks=(0, 1), generation=gen, run_id="run_0001"
                )

        # 启动异步保存会话与后台定时 GC，执行集群恢复（应先提升 step 2 尾巴并全程保护）
        with NodeComm(world_size=tp, base_dir=tmp_path / "comm") as comm:
            savers = [
                ckpt.AsyncSaver(
                    SaverConfig(
                        local_root=tmp_path / f"r{r}",
                        remote_config=s3_config,
                        run_id="run_0001",
                        rank=r,
                        ranks=(0, 1),
                        gc_interval=0.05,
                        keep_local=1,
                    ),
                    comm=comm.client(r, role=ROLE_TRAINER),
                    world_size=tp,
                )
                for r in range(tp)
            ]
            try:
                def do_recover(r: int):
                    return savers[r].recover(
                        storage,
                        device=DEVICE,
                        tp=tp,
                        tp_rank=r,
                        shard_dims=sim.column_dims(s2),
                        promote_tails=True,
                    )

                with ThreadPoolExecutor(max_workers=tp) as pool:
                    reports = [
                        f.result(timeout=15.0)
                        for f in [pool.submit(do_recover, r) for r in range(tp)]
                    ]
                assert all(rep.step == 2 for rep in reports)
                # 同轮次恢复后水位线已推进到 2
                assert all(s.max_step == 2 for s in savers)
                # 恢复完成后步数 2 仍留在恢复层窗口，step 1 可被后台 GC 正常回收
                assert _wait_until(
                    lambda: not (tmp_path / "r0" / "run_0001" / "step_1").exists(),
                    timeout=3.0,
                )
                assert (tmp_path / "r0" / "run_0001" / "step_2" / "COMMIT").is_file()
            finally:
                _close_all(savers, timeout=5.0)
    finally:
        storage.close()


def test_remote_outage_does_not_block_training(state, tmp_path: Path) -> None:
    """I1 / I3：远端对象存储不可达时，训练进程 save_async 同步段与本地 L1 落盘不受阻塞。"""
    s1 = state.step_once()
    dead_s3 = S3Config(
        bucket="ckpt",
        prefix="runs",
        endpoint_url="http://127.0.0.1:1",
        access_key_id="k",
        secret_access_key="s",
    )
    cfg = SaverConfig(
        local_root=tmp_path / "async",
        remote_config=dead_s3,
        upload_timeout=0.5,
        shutdown_timeout=1.0,
    )
    saver = ckpt.start_saver(cfg)
    try:
        t0 = time.monotonic()
        h = ckpt.save_async(sim.to_device(s1, device=DEVICE), step=1, upload=True)
        sync_elapsed = time.monotonic() - t0
        assert sync_elapsed < 0.15

        res = h.wait(timeout=5.0)
        assert res is not None and res.step == 1
        assert (tmp_path / "async" / "step_1" / "COMMIT").is_file()
    finally:
        # 远端断网时快速关闭，不无限阻塞
        try:
            ckpt.close_saver(timeout=1.0)
        except Exception:
            pass
