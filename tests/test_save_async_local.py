"""M1 测试：异步保存本地层（memfd 零拷贝、增量 PLAN、I6 逐字节一致、单/双缓冲门闸、run_id 单调性）。"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

import api as ckpt
import sim
from async_ import SnapshotState, Tensor, TriggerDecision, get_backend
from core import CheckpointError, PlanError, UnsupportedDeviceError
from core.metadata import Device
from saver.config import SaverConfig, SaverFaults
from saver.shm import ShmSegment

DEVICE = "cuda:0"


def _make_tensors(state: sim.TrainingState) -> dict[str, Tensor]:
    return sim.to_device(state, device=DEVICE)


def _tree_files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_async_returns_before_write(state, tmp_path: Path) -> None:
    """I1：slow_write_ms=250 时 save_async 立即返回且 handle.done 为 False。"""
    s1 = state.step_once()
    tensors = _make_tensors(s1)
    blobs = {"rng/rank_0": b"rng-state-payload-1234"}

    cfg = SaverConfig(
        local_root=tmp_path / "async",
        faults=SaverFaults(slow_write_ms=250.0),
    )
    saver = ckpt.start_saver(cfg)
    try:
        t0 = time.monotonic()
        handle = ckpt.save_async(tensors, step=3, blobs=blobs, upload=False)
        elapsed = time.monotonic() - t0

        assert handle.skipped is False
        assert elapsed < 0.15, f"save_async 同步段耗时过长: {elapsed:.3f}s"
        assert handle.done is False

        res = handle.wait(timeout=10.0)
        assert res is not None
        assert handle.done is True
        assert res.step == 3
        assert res.state is SnapshotState.COMMITTED

        loaded = ckpt.load(tmp_path / "async" / "step_3", device="cpu")
        sim.assert_state_matches(
            {k: v.numpy() for k, v in loaded.tensors.items()}, s1
        )
    finally:
        ckpt.close_saver()


def test_second_snap_frame_is_small(state, tmp_path: Path) -> None:
    """I4 / D8：连续两代布局不变时，第 2 代只发送增量 SNAP 而跳过 PLAN 下发。"""
    s1 = state.step_once()
    s2 = s1.step_once()

    cfg = SaverConfig(local_root=tmp_path / "async")
    saver = ckpt.start_saver(cfg)
    try:
        h1 = ckpt.save_async(_make_tensors(s1), step=1, upload=False)
        r1 = h1.wait(timeout=10.0)
        assert r1 is not None
        assert len(saver._plan_sent) == 1
        sent_after_first = set(saver._plan_sent)

        h2 = ckpt.save_async(_make_tensors(s2), step=2, upload=False)
        r2 = h2.wait(timeout=10.0)
        assert r2 is not None
        # plan_id 未变，没有新增 PLAN 发送
        assert saver._plan_sent == sent_after_first
    finally:
        ckpt.close_saver()


def test_sync_and_async_bytes_identical(state, tmp_path: Path) -> None:
    """I6：数据文件（含存储侧写入的 blobs 与清零空隙）、.metadata、GENERATION、COMMIT 逐字节一致。"""
    s1 = state.step_once()
    blobs = {
        "rng/rank_0": b"deterministic-rng-bytes-0123456789",
        "opt/step": b'{"step": 5}',
    }
    layout = {"tp": 1, "world_size": 1}
    gen_id = "gen_0001-deadbeef"
    run_id = "run_0001"

    sync_root = tmp_path / "sync"
    async_root = tmp_path / "async"

    sync_report = ckpt.save_local(
        _make_tensors(s1),
        sync_root,
        step=5,
        blobs=blobs,
        layout=layout,
        generation=gen_id,
        run_id=run_id,
    )

    cfg = SaverConfig(local_root=async_root, run_id=run_id)
    ckpt.start_saver(cfg)
    try:
        handle = ckpt.save_async(
            _make_tensors(s1),
            step=5,
            blobs=blobs,
            layout=layout,
            generation=gen_id,
            upload=False,
        )
        async_result = handle.wait(timeout=10.0)
        assert async_result is not None
    finally:
        ckpt.close_saver()

    assert async_result.engine == sync_report.engine
    assert async_result.size == sync_report.size
    assert async_result.commit == sync_report.commit

    sync_files = _tree_files(sync_root)
    async_files = _tree_files(async_root)
    assert set(sync_files) == set(async_files)
    for rel_path in sorted(sync_files):
        assert async_files[rel_path] == sync_files[rel_path], (
            f"{rel_path}: 异步与同步落盘字节不一致"
        )


def test_single_buffer_skips_then_forces(state, tmp_path: Path) -> None:
    """I2：单缓冲 max_inflight=1 下，在途期间前 3 次 SKIP，第 4 次触发 FORCE_WAIT 阻塞至落盘完成。"""
    s1 = state.step_once()
    s2 = s1.step_once()

    cfg = SaverConfig(
        local_root=tmp_path / "async",
        max_inflight=1,
        force_after_skips=3,
        faults=SaverFaults(slow_write_ms=350.0),
    )
    saver = ckpt.start_saver(cfg)
    try:
        h1 = ckpt.save_async(_make_tensors(s1), step=1, upload=False)
        assert h1.decision is TriggerDecision.START

        for skipped_step in (2, 3, 4):
            hs = ckpt.save_async(_make_tensors(s1), step=skipped_step, upload=False)
            assert hs.skipped is True
            assert hs.decision is TriggerDecision.SKIP

        # 连续跳过 3 次达到阈值，第 5 步必须 FORCE_WAIT 等第 1 步归还 buffer 后再启动
        t0 = time.monotonic()
        h5 = ckpt.save_async(_make_tensors(s2), step=5, upload=False)
        waited = time.monotonic() - t0
        assert h5.decision is TriggerDecision.START
        assert h1.done is True
        assert saver.manager.stats.skipped == 3
        assert saver.manager.stats.forced == 1
        assert waited >= 0.10
        r5 = h5.wait(timeout=10.0)
        assert r5 is not None and r5.step == 5
    finally:
        ckpt.close_saver()


def test_double_buffer_two_inflight(state, tmp_path: Path) -> None:
    """I2：双缓冲 max_inflight=2 允许两代同时在途，第 3 代 SKIP。"""
    s1 = state.step_once()
    s2 = s1.step_once()
    s3 = s2.step_once()

    cfg = SaverConfig(
        local_root=tmp_path / "async",
        max_inflight=2,
        force_after_skips=3,
        faults=SaverFaults(slow_write_ms=220.0),
    )
    saver = ckpt.start_saver(cfg)
    try:
        h1 = ckpt.save_async(_make_tensors(s1), step=1, upload=False)
        h2 = ckpt.save_async(_make_tensors(s2), step=2, upload=False)
        assert h1.decision is TriggerDecision.START
        assert h2.decision is TriggerDecision.START
        assert len(saver.manager.active) == 2

        h3 = ckpt.save_async(_make_tensors(s3), step=3, upload=False)
        assert h3.skipped is True
        assert h3.decision is TriggerDecision.SKIP

        r1 = h1.wait(timeout=10.0)
        r2 = h2.wait(timeout=10.0)
        assert r1 is not None and r2 is not None
        assert len(saver.manager.active) == 0
    finally:
        ckpt.close_saver()


def test_run_id_monotonic_step_validation(state, tmp_path: Path) -> None:
    """Q2 / I12：同一 run_id 内 step 必须严格单调递增，重复或倒退直接报 PlanError。"""
    s1 = state.step_once()
    cfg = SaverConfig(local_root=tmp_path / "async", run_id="run_0001")
    ckpt.start_saver(cfg)
    try:
        h1 = ckpt.save_async(_make_tensors(s1), step=10, upload=False)
        assert h1.wait(timeout=10.0) is not None

        with pytest.raises(PlanError, match="必须严格大于"):
            ckpt.save_async(_make_tensors(s1), step=10, upload=False)
        with pytest.raises(PlanError, match="必须严格大于"):
            ckpt.save_async(_make_tensors(s1), step=9, upload=False)
    finally:
        ckpt.close_saver()


def test_shm_reallocate_grows_and_cuda_register_fails_closed(tmp_path: Path) -> None:
    """§4：无在途块时跨越容量阈值可扩容重发 memfd；真机锁页注册失败绝不静默降级。"""
    # 1. host_registrar 返回 False 时必须抛 UnsupportedDeviceError
    with pytest.raises(UnsupportedDeviceError, match="拒绝降级"):
        ShmSegment(4096, host_registrar=lambda _ptr, _size: False)

    # 2. 小初始 shm_bytes 随更大模型自动 reallocate 并重发新 fd 给 worker
    cfg = SaverConfig(
        local_root=tmp_path / "async",
        shm_bytes=4096,
    )
    saver = ckpt.start_saver(cfg)
    try:
        initial_seg_id = saver.pool.segment_id
        arr = np.arange(8192, dtype=np.float32)  # 32 KiB > 4096
        backend = get_backend(Device.parse("cpu"))
        buf = backend.allocate(arr.nbytes)
        backend.copy_h2d(buf, memoryview(arr.tobytes()))
        big = Tensor(
            name="big.weight",
            buffer=buf,
            device=Device.parse("cpu"),
            dtype="f32",
            shape=arr.shape,
            backend=backend,
        )
        res = ckpt.save_async({"big.weight": big}, step=1, upload=False).wait(timeout=10.0)
        assert res is not None
        assert saver.pool.segment_id > initial_seg_id
        loaded = ckpt.load(tmp_path / "async" / "step_1", device="cpu")
        assert np.array_equal(loaded.tensors["big.weight"].numpy(), arr)
    finally:
        ckpt.close_saver()


def test_on_error_raise_next_surfaces_background_persist_failure(state, tmp_path: Path) -> None:
    """后台落盘失败且未调用 wait() 时，下一次 save_async(on_error='raise_next') 抛出异常。"""
    s1 = state.step_once()
    cfg = SaverConfig(
        local_root=tmp_path / "async",
        faults=SaverFaults(fail_persist_step=1),
    )
    ckpt.start_saver(cfg)
    try:
        h1 = ckpt.save_async(_make_tensors(s1), step=1, upload=False)
        deadline = time.monotonic() + 5.0
        while not h1.done and time.monotonic() < deadline:
            time.sleep(0.02)
        assert h1.done is True
        assert h1.error is not None

        with pytest.raises(CheckpointError, match="injected local persist failure"):
            ckpt.save_async(_make_tensors(s1), step=2, upload=False, on_error="raise_next")
    finally:
        ckpt.close_saver()
