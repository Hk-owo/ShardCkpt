"""M2 测试：心跳看门狗、无训练调用自动重启、SIGSTOP 卡死检测、熔断与孤儿清理。"""

from __future__ import annotations

import multiprocessing
import os
import signal
import time
from pathlib import Path

import pytest

import api as ckpt
import sim
from saver.config import SaverConfig, SaverDeadError, SaverFaults

DEVICE = "cuda:0"


def _wait_until(predicate, *, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_worker_killed_is_restarted_without_training_call(state, tmp_path: Path) -> None:
    """I7：杀掉存储子进程后不调用 save_async，ckpt-io 后台线程在超时内自动重启。"""
    s1 = state.step_once()
    cfg = SaverConfig(
        local_root=tmp_path / "async",
        heartbeat_interval=0.05,
        heartbeat_timeout=0.4,
        restart_backoff=0.01,
    )
    saver = ckpt.start_saver(cfg)
    try:
        old_pid = saver.worker_pid
        assert old_pid is not None
        os.kill(old_pid, signal.SIGKILL)

        # 全程不调 save_async，等待 ckpt-io 自主发现死亡并重启
        assert _wait_until(
            lambda: saver.worker_restarts == 1 and saver.worker_pid not in (None, old_pid),
            timeout=5.0,
        ), f"worker 未在看门狗超时内自动重启 (restarts={saver.worker_restarts})"

        res = ckpt.save_async(sim.to_device(s1, device=DEVICE), step=1, upload=False).wait(
            timeout=10.0
        )
        assert res is not None and res.step == 1
        loaded = ckpt.load(tmp_path / "async" / "step_1", device="cpu")
        sim.assert_state_matches(
            {k: v.numpy() for k, v in loaded.tensors.items()}, s1
        )
    finally:
        ckpt.close_saver()


def test_stuck_worker_detected_by_heartbeat(state, tmp_path: Path) -> None:
    """SIGSTOP 卡死存储子进程 → 心跳超时触发 SIGKILL + 重启 + 在途代 FAIL + STARTUP 清理残目录。"""
    s1 = state.step_once()
    s2 = s1.step_once()
    root = tmp_path / "async"
    # 预先制造一个残留无 COMMIT 目录，模拟卡死代在盘上的残迹
    stale_dir = root / "run_0001" / "step_1"
    stale_dir.mkdir(parents=True)
    (stale_dir / "__0_0.distcp").write_bytes(b"partial-data")

    cfg = SaverConfig(
        local_root=root,
        run_id="run_0001",
        heartbeat_interval=0.05,
        heartbeat_timeout=0.35,
        restart_backoff=0.01,
        faults=SaverFaults(slow_write_ms=2000.0),
    )
    saver = ckpt.start_saver(cfg)
    try:
        old_pid = saver.worker_pid
        assert old_pid is not None
        h1 = ckpt.save_async(sim.to_device(s1, device=DEVICE), step=1, upload=False)
        h1.wait_d2h(timeout=5.0)
        # 给 worker 一点时间进入 SNAP 并发送一次 PROGRESS 后 SIGSTOP
        time.sleep(0.08)
        os.kill(old_pid, signal.SIGSTOP)

        with pytest.raises(SaverDeadError):
            h1.wait(timeout=5.0)

        assert saver.worker_restarts >= 1
        assert saver.worker_pid not in (None, old_pid)
        # STARTUP.cleanup 应在重启握手时收掉作废的 step_1 残目录
        assert _wait_until(lambda: not stale_dir.exists(), timeout=3.0)

        # 后续更高 step 仍可正常落盘
        res2 = ckpt.save_async(sim.to_device(s2, device=DEVICE), step=2, upload=False).wait(
            timeout=10.0
        )
        assert res2 is not None and res2.step == 2
    finally:
        ckpt.close_saver()


def test_crash_loop_opens_circuit(state, tmp_path: Path) -> None:
    """每次启动即死 → 超过 max_restarts 熔断，save_async 抛 SaverDeadError。"""
    s1 = state.step_once()
    cfg = SaverConfig(
        local_root=tmp_path / "async",
        max_restarts=2,
        restart_backoff=0.01,
        faults=SaverFaults(exit_on_start=True),
    )
    saver = ckpt.start_saver(cfg)
    try:
        assert saver.dead is True
        assert saver.worker_restarts == 2
        with pytest.raises(SaverDeadError, match="熔断"):
            ckpt.save_async(sim.to_device(s1, device=DEVICE), step=1, upload=False)
    finally:
        ckpt.close_saver()


def test_crash_loop_fallback_sync(state, tmp_path: Path) -> None:
    """on_worker_dead='fallback_sync' 在熔断后退回 save_local 完成写盘。"""
    s1 = state.step_once()
    cfg = SaverConfig(
        local_root=tmp_path / "async",
        max_restarts=1,
        restart_backoff=0.01,
        on_worker_dead="fallback_sync",
        faults=SaverFaults(exit_on_start=True),
    )
    saver = ckpt.start_saver(cfg)
    try:
        assert saver.dead is True
        res = ckpt.save_async(sim.to_device(s1, device=DEVICE), step=4, upload=False).wait(
            timeout=5.0
        )
        assert res is not None and res.step == 4
        loaded = ckpt.load(tmp_path / "async" / "step_4", device="cpu")
        sim.assert_state_matches(
            {k: v.numpy() for k, v in loaded.tensors.items()}, s1
        )
    finally:
        ckpt.close_saver()


def _trainer_process_main(root: str, conn) -> None:
    """子进程模拟训练进程：提交慢写盘后上报 worker_pid，等父进程将其 SIGKILL。"""
    import api as ckpt_api
    import sim as sim_mod
    from saver.config import SaverConfig, SaverFaults

    state = sim_mod.make_state(step=1)
    cfg = SaverConfig(
        local_root=root,
        faults=SaverFaults(slow_write_ms=3000.0),
    )
    saver = ckpt_api.start_saver(cfg)
    h = ckpt_api.save_async(sim_mod.to_device(state, device="cpu"), step=1, upload=False)
    h.wait_d2h(timeout=5.0)
    conn.send(saver.worker_pid)
    conn.close()
    while True:
        time.sleep(1.0)


def test_training_killed_leaves_no_orphan(tmp_path: Path) -> None:
    """I10：训练进程被 SIGKILL → 存储子进程 1s 内凭 PR_SET_PDEATHSIG 退出且无 COMMIT。"""
    root = tmp_path / "async"
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    trainer = ctx.Process(target=_trainer_process_main, args=(str(root), child_conn))
    trainer.start()
    child_conn.close()

    try:
        assert parent_conn.poll(timeout=10.0)
        worker_pid = int(parent_conn.recv())
        assert _pid_exists(worker_pid)

        # 给 worker 进入写盘停顿的一点点时间后再直接 SIGKILL 训练进程
        time.sleep(0.05)
        os.kill(trainer.pid, signal.SIGKILL)
        trainer.join(timeout=5.0)

        # worker 应在 1s 内随 PDEATHSIG 退出
        assert _wait_until(lambda: not _pid_exists(worker_pid), timeout=1.5), (
            f"孤儿存储进程 {worker_pid} 未随训练进程退出"
        )
        assert not (root / "step_1" / "COMMIT").exists()
    finally:
        parent_conn.close()
        if trainer.is_alive():
            trainer.kill()
            trainer.join(timeout=2.0)
