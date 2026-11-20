"""阶段三测试：嵌入式提交协调监控闭环（C1 / C2 / C3 / C4）。"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import api as ckpt
import dist
import s3_stub
import sim
from async_ import PinnedBufferPool, TriggerDecision, aligned_allocator
from commit.coordinator import CommitCoordinator
from commit.generation import read_status
from commit.local import read_local_generation
from core import CheckpointError, PlanError
from storage.base import StorageError, run_scope
from storage.local import LocalStorage
from storage.object import S3Config, S3Storage
from storage.upload import upload_rank

DEVICE = "cpu"


@pytest.fixture
def s3_server():
    with s3_stub.start() as server:
        yield server


@pytest.fixture
def s3_config(s3_server) -> dict:
    return {
        "bucket": "ckpt",
        "prefix": "runs",
        "endpoint_url": s3_server.endpoint_url,
        "access_key_id": "test-key",
        "secret_access_key": "test-secret",
    }


def _wait_until(predicate, *, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _tensors_and_sources(state: sim.TrainingState, *, rank: int, tp: int):
    dims = sim.column_dims(state)
    tensors = sim.tensors_for_rank(state, rank=rank, tp=tp, dims=dims, device=DEVICE)
    sources = sim.sources_for_rank(state, rank=rank, tp=tp, dims=dims)
    return tensors, sources


# --- C1 + C2 + C4.2：统一开代、后台见齐即提交 L3、退出收敛 ----------------


def test_c1_c2_c4_coordinated_multirank_background_commit_and_exit_convergence(
    state, tmp_path, s3_config
) -> None:
    """C1+C2+C4.2：rank 0 开代写本地代际标记；后台见齐即写 L3（不等训练线程）；退出收敛。"""
    s1 = state.step_once()
    s2 = s1.step_once()
    tp = 2
    links = dist.ThreadedCollective.linked(tp)
    clients = [S3Storage(S3Config(**s3_config)) for _ in range(tp)]
    pools = [
        PinnedBufferPool(
            chunk_bytes=1 << 20,
            min_chunks=1,
            max_bytes=4 << 20,
            timeout=0,
            allocator=aligned_allocator(),
        )
        for _ in range(tp)
    ]

    coordinators = [
        CommitCoordinator(
            tmp_path / f"local-r{r}",
            clients[r],
            run_id="run_0001",
            collective=links[r],
            pool=pools[r],
            poll_interval=0.01,
        )
        for r in range(tp)
    ]

    def run_rank(rank: int):
        coord = coordinators[rank]
        with coord:
            t1, src1 = _tensors_and_sources(s1, rank=rank, tp=tp)
            r1 = coord.save_step(
                t1,
                step=1,
                sources=src1,
                blobs={f"rng/rank_{rank}": bytes([rank + 1]) * 16},
                layout={"tp": tp},
            )
            assert r1.decision is TriggerDecision.START
            assert r1.generation is not None

            # 验证 C2：本地落盘同步写入了代际标记文件
            marker = read_local_generation(
                LocalStorage(tmp_path / f"local-r{rank}" / "run_0001"), 1
            )
            assert marker is not None and marker.generation == r1.generation.id
            assert marker.run_id == "run_0001"

            # 验证「正常完成：rank 0 后台见齐 L2 即写 L3，不等训练线程下次触发」
            scoped = run_scope(clients[rank], "run_0001")
            assert _wait_until(
                lambda: (st := read_status(scoped, 1)) is not None
                and st.committed is not None
                and coord._inflight[0].done_event.is_set()
            ), "rank 0 后台线程应在训练线程继续前就完成 L3 总提交"

            t2, src2 = _tensors_and_sources(s2, rank=rank, tp=tp)
            r2 = coord.save_step(
                t2,
                step=2,
                sources=src2,
                blobs={f"rng/rank_{rank}": bytes([rank + 2]) * 16},
                layout={"tp": tp},
            )
            assert r2.decision is TriggerDecision.START
            assert 1 in r2.converged_steps
            return r1.generation.id, r2.generation.id

    try:
        with ThreadPoolExecutor(max_workers=tp) as pool:
            gens = [f.result(timeout=30) for f in [pool.submit(run_rank, r) for r in range(tp)]]
        assert gens[0] == gens[1], "所有 rank 必须使用 rank 0 统一分配的同一代际"
        for coord in coordinators:
            assert coord.converged_steps == (1, 2)
            assert coord.invalidated_generations == ()

        # 退出后远端 step 2 可完整恢复
        restored = ckpt.load_object(
            clients[0], step=2, device=DEVICE, run_id="run_0001"
        )
        sim.assert_state_matches(
            {name: t.numpy() for name, t in restored.tensors.items()}, s2
        )
    finally:
        for c in clients:
            c.close()
        for p in pools:
            p.close()


# --- C3.1：同轮次内 step 严格递增 -----------------------------------------


def test_c3_rejects_duplicate_or_non_increasing_step_in_same_run(
    state, tmp_path, s3_config
) -> None:
    """C3.1：同轮次内新提交 step <= 已提交/在途最大 step 直接拒绝报错。"""
    s1 = state.step_once()
    client = S3Storage(S3Config(**s3_config))
    pool = PinnedBufferPool(
        chunk_bytes=1 << 20, min_chunks=1, max_bytes=4 << 20, timeout=0,
        allocator=aligned_allocator(),
    )
    try:
        with CommitCoordinator(
            tmp_path / "local", client, run_id="run_0001", pool=pool
        ) as coord:
            coord.save_step(sim.to_device(s1, device=DEVICE), step=5, wait_upload=True)
            with pytest.raises(PlanError, match="必须严格大于"):
                coord.save_step(sim.to_device(s1, device=DEVICE), step=5)
            with pytest.raises(PlanError, match="必须严格大于"):
                coord.save_step(sim.to_device(s1, device=DEVICE), step=4)
    finally:
        client.close()
        pool.close()


# --- C1 + C3.2：搭便车全归约发现任一 rank 失败/超时 → 全员作废该代 ----------


def test_c3_piggyback_invalidates_failed_generation_across_all_ranks(
    state, tmp_path, s3_config
) -> None:
    """C3.2：rank 1 在 step 1 后台上传失败，step 2 触发点搭便车全归约让全员作废 step 1 且不重跑旧 step。"""
    s1 = state.step_once()
    s2 = s1.step_once()
    tp = 2
    links = dist.ThreadedCollective.linked(tp)
    clients = [S3Storage(S3Config(**s3_config)) for _ in range(tp)]
    pools = [
        PinnedBufferPool(
            chunk_bytes=1 << 20, min_chunks=1, max_bytes=4 << 20, timeout=0,
            allocator=aligned_allocator(),
        )
        for _ in range(tp)
    ]
    coordinators = [
        CommitCoordinator(
            tmp_path / f"local-r{r}",
            clients[r],
            run_id="run_0001",
            collective=links[r],
            pool=pools[r],
            upload_timeout=5.0,
            poll_interval=0.01,
        )
        for r in range(tp)
    ]

    rank1_failed_event = threading.Event()
    rank0_started_step1 = threading.Event()

    def upload_with_fault(root, storage, *, step, rank, generation, run_id, cancel_event):
        if step == 1 and rank == 1:
            rank1_failed_event.set()
            raise StorageError("injected rank 1 upload failure")
        if step == 1 and rank == 0:
            # 让 rank 0 等 rank 1 确认失败后再跑，验证 cancel_event 与全员作废
            rank0_started_step1.set()
        return upload_rank(
            root,
            storage,
            step=step,
            rank=rank,
            generation=generation,
            run_id=run_id,
            cancel_event=cancel_event,
        )

    def run_rank(rank: int):
        coord = coordinators[rank]
        t1, src1 = _tensors_and_sources(s1, rank=rank, tp=tp)
        res1 = coord.save_step(
            t1, step=1, sources=src1, _upload_fn=upload_with_fault
        )
        assert res1.decision is TriggerDecision.START

        # 等 rank 1 后台失败落定后，训练线程继续推进到 step 2 触发点
        assert rank1_failed_event.wait(timeout=5.0)
        assert _wait_until(lambda: coordinators[1]._inflight[0].error is not None)

        t2, src2 = _tensors_and_sources(s2, rank=rank, tp=tp)
        res2 = coord.save_step(t2, step=2, sources=src2)
        assert res2.decision is TriggerDecision.START
        assert len(res2.invalidated) == 1
        assert res2.invalidated[0].step == 1
        coord.finish()
        coord.close(wait=False)

    try:
        with ThreadPoolExecutor(max_workers=tp) as pool:
            futs = [pool.submit(run_rank, r) for r in range(tp)]
            for f in futs:
                f.result(timeout=30)

        for coord in coordinators:
            assert [inv.step for inv in coord.invalidated_generations] == [1]
            assert coord.converged_steps == (2,)

        # step 1 绝没有被提交到远端，最新已提交版本就是 step 2（没有重跑 step 1）
        scoped = run_scope(clients[0], "run_0001")
        st1 = read_status(scoped, 1)
        assert st1 is not None and st1.committed is None
        restored = ckpt.load_object(clients[0], device=DEVICE, run_id="run_0001")
        assert restored.step == 2
        sim.assert_state_matches(
            {name: t.numpy() for name, t in restored.tensors.items()}, s2
        )
    finally:
        for c in clients:
            c.close()
        for p in pools:
            p.close()


def test_c3_upload_timeout_triggers_global_invalidation(
    state, tmp_path, s3_config
) -> None:
    """C3.2：任一 rank 上传超时，搭便车全归约检测超时并全员作废该代。"""
    s1 = state.step_once()
    s2 = s1.step_once()
    tp = 2
    links = dist.ThreadedCollective.linked(tp)
    clients = [S3Storage(S3Config(**s3_config)) for _ in range(tp)]
    pools = [
        PinnedBufferPool(
            chunk_bytes=1 << 20, min_chunks=1, max_bytes=4 << 20, timeout=0,
            allocator=aligned_allocator(),
        )
        for _ in range(tp)
    ]
    coordinators = [
        CommitCoordinator(
            tmp_path / f"local-r{r}",
            clients[r],
            run_id="run_0001",
            collective=links[r],
            pool=pools[r],
            upload_timeout=0.35,
            poll_interval=0.01,
        )
        for r in range(tp)
    ]

    def stalled_upload(root, storage, *, step, rank, generation, run_id, cancel_event):
        if step == 1 and rank == 1:
            cancel_event.wait(timeout=2.0)
            return 0
        return upload_rank(
            root,
            storage,
            step=step,
            rank=rank,
            generation=generation,
            run_id=run_id,
            cancel_event=cancel_event,
        )

    def run_rank(rank: int):
        coord = coordinators[rank]
        t1, src1 = _tensors_and_sources(s1, rank=rank, tp=tp)
        coord.save_step(t1, step=1, sources=src1, _upload_fn=stalled_upload)
        time.sleep(0.45)  # 超过 upload_timeout=0.35s
        t2, src2 = _tensors_and_sources(s2, rank=rank, tp=tp)
        res2 = coord.save_step(t2, step=2, sources=src2)
        assert [inv.step for inv in res2.invalidated] == [1]
        coord.finish()
        coord.close(wait=False)

    try:
        with ThreadPoolExecutor(max_workers=tp) as pool:
            for f in [pool.submit(run_rank, r) for r in range(tp)]:
                f.result(timeout=30)
        assert [inv.step for inv in coordinators[0].invalidated_generations] == [1]
        assert [inv.step for inv in coordinators[1].invalidated_generations] == [1]
    finally:
        for c in clients:
            c.close()
        for p in pools:
            p.close()


# --- C4.1 (R16b)：重启时仅提升「远端分片已齐、缺总提交点」的尾巴 ------------


def test_c4_startup_promotes_complete_tail_and_ignores_incomplete_residue(
    state, tmp_path, s3_config
) -> None:
    """C4.1：step 2 全 rank L2 已齐只缺 L3，step 3 仅上传 rank 0；重启恢复时提升 step 2、放弃 step 3。"""
    s1 = state.step_once()
    s2 = s1.step_once()
    s3 = s2.step_once()
    tp = 2
    writer = S3Storage(S3Config(**s3_config))
    try:
        # step 1 正常提交
        g1 = ckpt.begin_upload(writer, step=1, ranks=[0, 1], run_id="run_0001")
        for r in range(tp):
            t, src = _tensors_and_sources(s1, rank=r, tp=tp)
            ckpt.save_local(
                t, tmp_path / f"r{r}", step=1, rank=r, sources=src, generation=g1, run_id="run_0001"
            )
            ckpt.upload_local(
                tmp_path / f"r{r}", writer, step=1, rank=r, generation=g1, run_id="run_0001"
            )
        ckpt.finalize_upload(writer, step=1, ranks=[0, 1], generation=g1, run_id="run_0001")

        # step 2 全 rank L2 已齐，但 rank 0 崩在写 L3 之前
        g2 = ckpt.begin_upload(writer, step=2, ranks=[0, 1], run_id="run_0001")
        for r in range(tp):
            t, src = _tensors_and_sources(s2, rank=r, tp=tp)
            ckpt.save_local(
                t, tmp_path / f"r{r}", step=2, rank=r, sources=src, generation=g2, run_id="run_0001"
            )
            ckpt.upload_local(
                tmp_path / f"r{r}", writer, step=2, rank=r, generation=g2, run_id="run_0001"
            )

        # step 3 只有 rank 0 上传了，rank 1 没传完（不可提升）
        g3 = ckpt.begin_upload(writer, step=3, ranks=[0, 1], run_id="run_0001")
        t3, src3 = _tensors_and_sources(s3, rank=0, tp=tp)
        ckpt.save_local(
            t3, tmp_path / "r0", step=3, rank=0, sources=src3, generation=g3, run_id="run_0001"
        )
        ckpt.upload_local(
            tmp_path / "r0", writer, step=3, rank=0, generation=g3, run_id="run_0001"
        )
    finally:
        writer.close()

    # 重启协调层并恢复：rank 0 先提升 step 2，不提升 step 3，全员一致恢复到 step 2
    links = dist.ThreadedCollective.linked(tp)
    clients = [S3Storage(S3Config(**s3_config)) for _ in range(tp)]
    pools = [
        PinnedBufferPool(
            chunk_bytes=1 << 20, min_chunks=1, max_bytes=4 << 20, timeout=0,
            allocator=aligned_allocator(),
        )
        for _ in range(tp)
    ]
    try:
        with ThreadPoolExecutor(max_workers=tp) as pool:
            def recover_rank(rank: int):
                with CommitCoordinator(
                    tmp_path / f"r{rank}",
                    clients[rank],
                    run_id="run_0001",
                    collective=links[rank],
                    pool=pools[rank],
                ) as coord:
                    rep = coord.recover(device=DEVICE)
                    return rep, coord.max_step

            results = [f.result(timeout=30) for f in [pool.submit(recover_rank, r) for r in range(tp)]]

        for rep, max_step in results:
            assert rep.step == 2
            assert max_step == 2
            sim.assert_state_matches(
                {name: t.numpy() for name, t in rep.tensors.items()}, s2
            )
        scoped = run_scope(clients[0], "run_0001")
        assert read_status(scoped, 3).committed is None
    finally:
        for c in clients:
            c.close()
        for p in pools:
            p.close()


# --- C4.2：正常退出等待点遇到最后一代失败时全员明确报错 --------------------


def test_c4_finish_raises_consistently_on_all_ranks_when_last_upload_fails(
    state, tmp_path, s3_config
) -> None:
    """C4.2：最后一代某 rank 后台上传失败，finish() 等待收敛时全员一致报错退出。"""
    s1 = state.step_once()
    tp = 2
    links = dist.ThreadedCollective.linked(tp)
    clients = [S3Storage(S3Config(**s3_config)) for _ in range(tp)]
    pools = [
        PinnedBufferPool(
            chunk_bytes=1 << 20, min_chunks=1, max_bytes=4 << 20, timeout=0,
            allocator=aligned_allocator(),
        )
        for _ in range(tp)
    ]
    coordinators = [
        CommitCoordinator(
            tmp_path / f"local-r{r}",
            clients[r],
            run_id="run_0001",
            collective=links[r],
            pool=pools[r],
            poll_interval=0.01,
        )
        for r in range(tp)
    ]

    def fail_rank1(root, storage, *, step, rank, generation, run_id, cancel_event):
        if rank == 1:
            raise StorageError("last generation rank 1 failure")
        return upload_rank(
            root, storage, step=step, rank=rank, generation=generation, run_id=run_id, cancel_event=cancel_event
        )

    def run_rank(rank: int):
        coord = coordinators[rank]
        t1, src1 = _tensors_and_sources(s1, rank=rank, tp=tp)
        coord.save_step(t1, step=1, sources=src1, _upload_fn=fail_rank1)
        try:
            coord.finish()
        finally:
            coord.close(wait=False)

    try:
        with ThreadPoolExecutor(max_workers=tp) as pool:
            futs = [pool.submit(run_rank, r) for r in range(tp)]
            with pytest.raises(CheckpointError):
                futs[0].result(timeout=30)
            with pytest.raises(StorageError, match="last generation rank 1 failure"):
                futs[1].result(timeout=30)
    finally:
        for c in clients:
            c.close()
        for p in pools:
            p.close()
