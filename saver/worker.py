"""存储子进程（§7）：纯 CPU 消费者 + 本地落盘 + 存储侧上传协调 + GcWorker。

严禁在本模块内执行任何 CUDA 操作或触碰训练梯度通信。
"""

from __future__ import annotations

import ctypes
import mmap
import os
import queue
import signal
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from commit.coordinator import acquire_generation_for_step, wait_and_finalize_generation
from commit.gc import GcWorker, delete_local_version
from commit.generation import FencedError, Generation, as_generation
from commit.local import write_local_generation
from core.region import Interval, Region
from planner.save_plan import SavePlan
from saver.comm import (
    CommEndpoint,
    CommError,
    NodeCommClient,
    OP_COMMIT_BROADCAST,
    OP_GEN_BROADCAST,
    OP_INVALIDATE_EVENT,
    OP_PROTECT_BROADCAST,
    ROLE_STORAGE,
)
from saver.config import SaverConfig, SaverFaults
from saver.persist import persist_host_buffer
from saver.protocol import (
    OP_BYE,
    OP_CLOSE,
    OP_DONE,
    OP_FAIL,
    OP_MEMFD,
    OP_NEED_PLAN,
    OP_PLAN,
    OP_PROGRESS,
    OP_PROTECT,
    OP_READY,
    OP_SNAP,
    OP_STARTUP,
    OP_UPLOAD_DONE,
    OP_UPLOAD_FAIL,
    decode_blobs,
    plan_from_dict,
    recv_msg,
    send_msg,
)
from storage.base import StorageBackend, run_scope
from storage.local import LocalStorage
from storage.object import S3Storage
from storage.upload import local_run_root, upload_rank

_PR_SET_PDEATHSIG = 1


def _install_parent_death_signal() -> None:
    """父进程意外死亡（如 SIGKILL）时内核自动向本进程发送 SIGTERM（I10）。"""
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
        except OSError:
            pass
    if os.getppid() == 1:
        os._exit(0)


@dataclass(frozen=True)
class _CachedPlan:
    plan: SavePlan
    layout: dict[str, int]
    crc_chunk_max: int
    alignment: int


@dataclass
class _UploadJob:
    step: int
    run_id: str | None
    layout: dict[str, int]
    generation: Generation | None
    enqueued_at: float
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)
    error: str | None = None
    committed: bool = False


class _StorageWorker:
    def __init__(self, sock: socket.socket, config: SaverConfig) -> None:
        self.sock = sock
        self.config = config
        self.faults = config.faults or SaverFaults()
        self._parent_pid = os.getppid()
        self._send_lock = threading.Lock()
        self._stop = threading.Event()

        self._memfd: int = -1
        self._mmap: mmap.mmap | None = None
        self._view: memoryview | None = None
        self._capacity: int = 0

        self._plans: dict[str, _CachedPlan] = {}
        self._deferred_frames: deque[tuple[dict[str, Any], list[int]]] = deque()

        # 进度心跳状态
        self._hb_lock = threading.Lock()
        self._active_gen: int | None = None
        self._active_written: int = 0
        self._active_started: float = 0.0
        self._hb_thread: threading.Thread | None = None

        # 远端与 GC（纯 CPU / HTTP）
        self.remote: StorageBackend | None = (
            S3Storage(config.remote_config)
            if config.remote_config is not None
            else None
        )
        self.scoped_remote: StorageBackend | None = (
            run_scope(self.remote, config.run_id) if self.remote is not None else None
        )
        self.gc = GcWorker(
            config.local_root,
            self.remote,
            rank=config.rank,
            interval=config.gc_interval if config.gc_interval is not None else 60.0,
            dry_run=config.gc_dry_run,
            keep_local=config.keep_local,
            run_id=config.run_id,
            staging_dir=config.staging_dir,
            abort_stale_mpu=config.abort_stale_mpu,
            mpu_timeout_seconds=config.mpu_timeout_seconds,
        )
        if config.gc_interval is not None and self.remote is not None:
            self.gc.start()

        # 节点独立通信转发层（M3）
        self.comm: NodeCommClient | None = None
        self._state_lock = threading.Lock()
        self._cond = threading.Condition(self._state_lock)
        self._known_gens: dict[tuple[str | None, int], Generation] = {}
        self._committed_steps: dict[str | None, int] = {}
        self._committed_step_list: list[int] = []
        self._invalidated: dict[tuple[str | None, int], str] = {}
        self._active_uploads: dict[tuple[str | None, int], _UploadJob] = {}
        self._upload_queue: queue.Queue[_UploadJob | None] = queue.Queue()
        self._upload_errors: list[str] = []
        self._upload_thread: threading.Thread | None = None

        if config.comm_endpoint is not None:
            self.comm = NodeCommClient(
                config.comm_endpoint,
                rank=config.rank,
                role=ROLE_STORAGE,
                on_event=self._on_comm_event,
            )

        if self.remote is not None:
            self._upload_thread = threading.Thread(
                target=self._upload_loop,
                name=f"ckpt-upload-r{config.rank}",
                daemon=True,
            )
            self._upload_thread.start()

    # --- 通信发送与心跳 -----------------------------------------------------

    def _send(self, msg: dict[str, Any]) -> None:
        with self._send_lock:
            send_msg(self.sock, msg)

    def _start_heartbeat_thread(self) -> None:
        if self._hb_thread is not None:
            return
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"ckpt-hb-r{self.config.rank}",
            daemon=True,
        )
        self._hb_thread.start()

    def _heartbeat_loop(self) -> None:
        interval = min(float(self.config.heartbeat_interval), 0.2)
        while not self._stop.wait(interval):
            if os.getppid() != self._parent_pid:
                os._exit(0)
            if self.faults.skip_heartbeat or self.faults.hang_on_snap:
                continue
            with self._hb_lock:
                gen = self._active_gen
                written = self._active_written
                started = self._active_started
            if gen is None:
                continue
            try:
                self._send(
                    {
                        "op": OP_PROGRESS,
                        "gen": gen,
                        "written": written,
                        "elapsed": max(0.0, time.monotonic() - started),
                    }
                )
            except OSError:
                return

    # --- memfd 管理 ---------------------------------------------------------

    def _set_memfd(self, fd: int, capacity: int) -> None:
        if self._view is not None:
            try:
                self._view.release()
            except Exception:
                pass
            self._view = None
        if self._mmap is not None:
            try:
                self._mmap.close()
            except BufferError:
                pass
            self._mmap = None
        if self._memfd >= 0:
            try:
                os.close(self._memfd)
            except OSError:
                pass
            self._memfd = -1

        self._memfd = fd
        self._capacity = int(capacity)
        self._mmap = mmap.mmap(
            fd, self._capacity, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
        )
        self._view = memoryview(self._mmap)

    def _close_memfd(self) -> None:
        if self._view is not None:
            try:
                self._view.release()
            except Exception:
                pass
            self._view = None
        if self._mmap is not None:
            try:
                self._mmap.close()
            except BufferError:
                pass
            self._mmap = None
        if self._memfd >= 0:
            try:
                os.close(self._memfd)
            except OSError:
                pass
            self._memfd = -1

    # --- 主循环 -------------------------------------------------------------

    def run(self) -> None:
        if self.faults.exit_on_start:
            os._exit(2)

        self._start_heartbeat_thread()

        # 等待初始 memfd
        while not self._stop.is_set():
            res = recv_msg(self.sock)
            if res is None:
                return
            msg, fds = res
            if msg.get("op") == OP_MEMFD and fds:
                self._set_memfd(fds[0], int(msg["capacity"]))
                self._send(
                    {
                        "op": OP_READY,
                        "pid": os.getpid(),
                        "capacity": self._capacity,
                    }
                )
                break
            self._close_fds(fds)
            self._deferred_frames.append((msg, []))

        while not self._stop.is_set():
            if self._deferred_frames:
                msg, fds = self._deferred_frames.popleft()
            else:
                res = recv_msg(self.sock)
                if res is None:
                    break
                msg, fds = res

            op = msg.get("op")
            if op == OP_MEMFD:
                if fds:
                    self._set_memfd(fds[0], int(msg["capacity"]))
                continue
            self._close_fds(fds)

            if op == OP_STARTUP:
                self._handle_startup(msg)
            elif op == OP_PLAN:
                self._handle_plan(msg)
            elif op == OP_PROTECT:
                self._handle_protect(msg)
            elif op == OP_SNAP:
                self._handle_snap(msg)
            elif op == OP_CLOSE:
                self._handle_close(msg)
                break

    def _close_fds(self, fds: list[int]) -> None:
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass

    def _handle_startup(self, msg: dict[str, Any]) -> None:
        for run_id_item, step_item in msg.get("cleanup", ()):
            try:
                scoped = local_run_root(self.config.local_root, run_id_item)
                delete_local_version(scoped, int(step_item))
            except Exception:
                pass
        protected = msg.get("protected_steps", ())
        if protected:
            self.gc.protect(protected, tag="recovering")

    def _handle_plan(self, msg: dict[str, Any]) -> None:
        plan_id = str(msg["plan_id"])
        plan = plan_from_dict(msg["plan"])
        layout = {str(k): int(v) for k, v in dict(msg.get("layout", {})).items()}
        crc_chunk_max = int(msg["crc_chunk_max"])
        alignment = int(msg["alignment"])
        self._plans[plan_id] = _CachedPlan(
            plan=plan,
            layout=layout,
            crc_chunk_max=crc_chunk_max,
            alignment=alignment,
        )

    def _handle_protect(self, msg: dict[str, Any]) -> None:
        add_steps = msg.get("add_steps", ())
        release_steps = msg.get("release_steps", ())
        pause_gc = msg.get("pause_gc")
        if pause_gc is True:
            self.gc.pause()
        elif pause_gc is False:
            self.gc.resume()
        if add_steps:
            self.gc.protect(add_steps, tag="recovering")
        if release_steps:
            self.gc.unprotect(release_steps, tag="recovering")

    def _fetch_plan_if_missing(self, plan_id: str) -> _CachedPlan:
        cached = self._plans.get(plan_id)
        if cached is not None:
            return cached
        self._send({"op": OP_NEED_PLAN, "plan_id": plan_id})
        while not self._stop.is_set():
            res = recv_msg(self.sock)
            if res is None:
                raise EOFError("connection closed while waiting for PLAN")
            msg, fds = res
            if msg.get("op") == OP_PLAN:
                self._close_fds(fds)
                self._handle_plan(msg)
                cached = self._plans.get(plan_id)
                if cached is not None:
                    return cached
            else:
                self._deferred_frames.append((msg, fds))
        raise EOFError("worker stopping")

    def _handle_snap(self, msg: dict[str, Any]) -> None:
        gen = int(msg["gen"])
        step = int(msg["step"])
        snap_run_id = msg.get("run_id")
        plan_id = str(msg["plan_id"])
        buf_info = msg["buffer"]
        offset = int(buf_info["offset"])
        length = int(buf_info["length"])
        aligned_size = int(buf_info["aligned_size"])
        blobs = decode_blobs(msg.get("blobs"))
        engine = str(msg.get("engine", self.config.engine))
        direct_io = bool(msg.get("direct_io", self.config.direct_io))
        want_upload = bool(msg.get("upload", True))
        explicit_gen = msg.get("generation")

        if self.faults.hang_on_snap:
            with self._hb_lock:
                self._active_gen = gen
                self._active_started = time.monotonic()
            while not self._stop.wait(0.2):
                pass
            return

        cached = self._fetch_plan_if_missing(plan_id)
        self.gc.protect([step], tag="persisting")
        with self._hb_lock:
            self._active_gen = gen
            self._active_written = 0
            self._active_started = time.monotonic()

        persisted_ok = False
        try:
            if self.faults.slow_write_ms > 0:
                end_sleep = time.monotonic() + (self.faults.slow_write_ms / 1000.0)
                while not self._stop.is_set():
                    rem = end_sleep - time.monotonic()
                    if rem <= 0:
                        break
                    time.sleep(min(rem, 0.05))

            if (
                self.faults.fail_persist_step is not None
                and self.faults.fail_persist_step == step
            ):
                raise OSError(f"injected local persist failure at step {step}")

            assert self._view is not None
            buf_slice = self._view[offset : offset + length]
            try:
                result = persist_host_buffer(
                    buf_slice,
                    self.config.local_root,
                    step=step,
                    plan=cached.plan,
                    aligned_size=aligned_size,
                    blobs=blobs,
                    layout=cached.layout,
                    alignment=cached.alignment,
                    engine=engine,
                    direct_io=direct_io,
                    crc_chunk_max=cached.crc_chunk_max,
                    generation=explicit_gen,
                    run_id=snap_run_id,
                )
            finally:
                buf_slice.release()

            with self._hb_lock:
                self._active_written = result.size

            persisted_ok = True
            will_upload = want_upload and self.remote is not None
            if will_upload:
                # 在释放 persisting 前先把 step 纳入 uploading 保护集，零空窗
                self.gc.protect([step], tag="uploading")
            self.gc.unprotect([step], tag="persisting")

            self._send(
                {
                    "op": OP_DONE,
                    "gen": gen,
                    "step": step,
                    "size": result.size,
                    "data_bytes": result.data_bytes,
                    "engine": result.engine,
                    "commit": {
                        "metadata_bytes": result.metadata_bytes,
                        "metadata_crc32c": result.metadata_crc32c,
                        "files": dict(result.commit.files),
                    },
                    "generation": result.generation,
                }
            )

            if will_upload:
                gen_obj = (
                    as_generation(explicit_gen, step=step)
                    if explicit_gen is not None
                    else None
                )
                job = _UploadJob(
                    step=step,
                    run_id=snap_run_id,
                    layout=dict(cached.layout),
                    generation=gen_obj,
                    enqueued_at=time.monotonic(),
                )
                with self._state_lock:
                    self._active_uploads[(snap_run_id, step)] = job
                self._upload_queue.put(job)
        except BaseException as exc:
            if not persisted_ok:
                try:
                    delete_local_version(
                        local_run_root(self.config.local_root, snap_run_id), step
                    )
                except Exception:
                    pass
                self.gc.unprotect([step], tag="persisting")
            try:
                self._send(
                    {
                        "op": OP_FAIL,
                        "gen": gen,
                        "step": step,
                        "kind": type(exc).__name__,
                        "message": str(exc),
                        "state": "aborted",
                    }
                )
            except OSError:
                pass
            if self.comm is not None and want_upload and self.remote is not None:
                try:
                    self.comm.broadcast_invalidation(
                        run_id=snap_run_id,
                        step=step,
                        generation=explicit_gen,
                        reason=f"local persist failed: {type(exc).__name__}: {exc}",
                    )
                except Exception:
                    pass
        finally:
            with self._hb_lock:
                self._active_gen = None
                self._active_written = 0

    # --- 节点通信事件处理 & 存储侧上传协调（§7.3） -------------------------

    def _on_comm_event(self, msg: dict[str, Any]) -> None:
        op = msg.get("op")
        if op == OP_GEN_BROADCAST:
            run_id = msg.get("run_id")
            step = int(msg["step"])
            gen = Generation(
                step=step, seq=int(msg["seq"]), writer=str(msg["writer"])
            )
            with self._cond:
                self._known_gens[(run_id, step)] = gen
                self._cond.notify_all()
            return

        if op == OP_COMMIT_BROADCAST:
            run_id = msg.get("run_id")
            step = int(msg["step"])
            with self._cond:
                prev = self._committed_steps.get(run_id, -1)
                if step > prev:
                    self._committed_steps[run_id] = step
                if step not in self._committed_step_list:
                    self._committed_step_list.append(step)
                # 竞态规则 4：更新步数已 L3 全局提交 → 立即取消更早旧步数上传并解除保护
                for (jrun, jstep), job in list(self._active_uploads.items()):
                    if jrun == run_id and jstep < step:
                        job.cancel_event.set()
                        self.gc.unprotect([jstep], tag="uploading")
                # 标记本步已提交（供非 0 rank 感知）
                cur_job = self._active_uploads.get((run_id, step))
                if cur_job is not None:
                    cur_job.committed = True
                self._cond.notify_all()
            return

        if op == OP_INVALIDATE_EVENT:
            run_id = msg.get("run_id")
            step = int(msg["step"])
            reason = str(msg.get("reason", "peer invalidated"))
            with self._cond:
                self._invalidated[(run_id, step)] = reason
                # 淘汰过旧记录防止无限增长
                if len(self._invalidated) > 1024:
                    oldest = next(iter(self._invalidated))
                    self._invalidated.pop(oldest, None)
                job = self._active_uploads.get((run_id, step))
                if job is not None:
                    job.cancel_event.set()
                    self.gc.unprotect([step], tag="uploading")
                self._cond.notify_all()
            return

        if op == OP_PROTECT_BROADCAST:
            pause_gc = msg.get("pause_gc")
            if pause_gc is True:
                self.gc.pause()
            elif pause_gc is False:
                self.gc.resume()
            add_steps = msg.get("add_steps", ())
            release_steps = msg.get("release_steps", ())
            if add_steps:
                self.gc.protect(add_steps, tag="recovering")
            if release_steps:
                self.gc.unprotect(release_steps, tag="recovering")

    def _upload_loop(self) -> None:
        while True:
            job = self._upload_queue.get()
            if job is None:
                return
            try:
                self._process_upload_job(job)
            finally:
                with self._cond:
                    self._active_uploads.pop((job.run_id, job.step), None)
                    job.done_event.set()
                    self._cond.notify_all()
                self.gc.unprotect([job.step], tag="uploading")

    def _is_aborted_or_superseded(self, job: _UploadJob) -> str | None:
        with self._state_lock:
            inv = self._invalidated.get((job.run_id, job.step))
            if inv is not None:
                return inv
            committed = self._committed_steps.get(job.run_id)
            if committed is not None and job.step < committed:
                return f"superseded by committed step {committed}"
        if job.cancel_event.is_set():
            return "cancelled"
        return None

    def _resolve_upload_generation(self, job: _UploadJob) -> Generation | None:
        if job.generation is not None:
            return job.generation
        assert self.remote is not None
        scoped = run_scope(self.remote, job.run_id)
        world_size = self.comm.world_size if self.comm is not None else 1
        ranks = self.config.ranks or tuple(range(world_size))

        if self.config.rank == 0 or world_size == 1:
            if self.faults.fail_acquire_generation:
                raise FencedError("injected acquire_generation failure")
            gen = acquire_generation_for_step(scoped, job.step, ranks=ranks)
            if self.comm is not None and world_size > 1:
                self.comm.broadcast_generation(
                    run_id=job.run_id,
                    step=job.step,
                    generation=gen.id,
                    seq=gen.seq,
                    writer=gen.writer,
                    ranks=ranks,
                )
            return gen

        # 非 0 rank 等待主卡存储进程经节点通信层广播 GEN_BROADCAST
        deadline = time.monotonic() + self.config.upload_timeout
        with self._cond:
            while not self._stop.is_set():
                if (job.run_id, job.step) in self._invalidated or job.cancel_event.is_set():
                    return None
                committed = self._committed_steps.get(job.run_id)
                if committed is not None and job.step < committed:
                    return None
                known = self._known_gens.get((job.run_id, job.step))
                if known is not None:
                    return known
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"step {job.step}: 等待主卡远端代际广播超时"
                    )
                self._cond.wait(min(remaining, 0.2))
        return None

    def _process_upload_job(self, job: _UploadJob) -> None:
        assert self.remote is not None
        # 竞态规则 1 & 4：作废先到或更新 step 已提交 → 直接跳过上传
        abort_reason = self._is_aborted_or_superseded(job)
        if abort_reason is not None:
            return

        gen: Generation | None = None
        try:
            gen = self._resolve_upload_generation(job)
            if gen is None:
                return

            abort_reason = self._is_aborted_or_superseded(job)
            if abort_reason is not None:
                return

            # 补写本地 GENERATION 标记（若异步开代滞后于本地 L1 落盘）
            local_storage = LocalStorage(
                local_run_root(self.config.local_root, job.run_id)
            )
            write_local_generation(
                local_storage, job.step, gen.id, run_id=job.run_id
            )

            if self.faults.slow_upload_ms > 0:
                if job.cancel_event.wait(self.faults.slow_upload_ms / 1000.0):
                    return

            if (
                self.faults.fail_upload_step is not None
                and self.faults.fail_upload_step == job.step
                and (
                    self.faults.fail_upload_rank is None
                    or self.faults.fail_upload_rank == self.config.rank
                )
            ):
                raise OSError(
                    f"injected upload failure at step {job.step} rank {self.config.rank}"
                )

            if self._is_aborted_or_superseded(job) is not None:
                return

            upload_rank(
                self.config.local_root,
                self.remote,
                step=job.step,
                rank=self.config.rank,
                generation=gen,
                run_id=job.run_id,
                cancel_event=job.cancel_event,
            )

            # 本卡写完 L2 即移出上传保护集
            self.gc.unprotect([job.step], tag="uploading")

            world_size = self.comm.world_size if self.comm is not None else 1
            ranks = self.config.ranks or tuple(range(world_size))
            if self.config.rank == 0 or world_size == 1:
                scoped = run_scope(self.remote, job.run_id)
                commit = wait_and_finalize_generation(
                    scoped,
                    step=job.step,
                    generation=gen,
                    ranks=ranks,
                    layout=job.layout,
                    keep_versions=self.config.keep_versions,
                    cancel_event=job.cancel_event,
                    timeout=self.config.upload_timeout,
                )
                if commit is None:
                    return
                with self._cond:
                    prev = self._committed_steps.get(job.run_id, -1)
                    if job.step > prev:
                        self._committed_steps[job.run_id] = job.step
                    if job.step not in self._committed_step_list:
                        self._committed_step_list.append(job.step)
                    job.committed = True
                    self._cond.notify_all()
                if self.comm is not None and world_size > 1:
                    self.comm.broadcast_commit(
                        run_id=job.run_id, step=job.step, generation=gen.id
                    )

            try:
                self._send(
                    {
                        "op": OP_UPLOAD_DONE,
                        "step": job.step,
                        "run_id": job.run_id,
                        "generation": gen.id,
                    }
                )
            except OSError:
                pass
        except BaseException as exc:
            # 若是被对端作废事件打断（例如 upload_rank 内 cancel_event 触发 FencedError），不重复广播
            externally_invalidated = self._is_aborted_or_superseded(job) is not None
            msg = f"{type(exc).__name__}: {exc}"
            job.error = msg
            if not externally_invalidated:
                with self._state_lock:
                    self._upload_errors.append(msg)
                if self.comm is not None:
                    try:
                        self.comm.broadcast_invalidation(
                            run_id=job.run_id,
                            step=job.step,
                            generation=gen.id if gen is not None else None,
                            reason=msg,
                        )
                    except Exception:
                        pass
            try:
                self._send(
                    {
                        "op": OP_UPLOAD_FAIL,
                        "step": job.step,
                        "run_id": job.run_id,
                        "generation": gen.id if gen is not None else None,
                        "error": msg,
                    }
                )
            except OSError:
                pass

    def _handle_close(self, msg: dict[str, Any]) -> None:
        drain = bool(msg.get("drain_uploads", True))
        timeout = float(msg.get("timeout", self.config.shutdown_timeout))
        deadline = time.monotonic() + timeout
        errors: list[str] = []

        if self._upload_thread is not None:
            if drain:
                with self._cond:
                    while self._active_uploads or not self._upload_queue.empty():
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            errors.append(f"upload drain timeout ({timeout}s)")
                            for job in self._active_uploads.values():
                                job.cancel_event.set()
                            break
                        self._cond.wait(min(remaining, 0.1))
                # 多卡经节点通信层等待全员上传/L3 收敛（§8.3）
                if self.comm is not None and self.comm.world_size > 1 and not errors:
                    remaining = max(0.5, deadline - time.monotonic())
                    try:
                        with self._state_lock:
                            local_errs = list(self._upload_errors)
                        cluster_errs = self.comm.storage_barrier(
                            errors=local_errs, timeout=remaining
                        )
                        errors.extend(cluster_errs)
                    except CommError as exc:
                        errors.append(str(exc))
            else:
                with self._cond:
                    for job in self._active_uploads.values():
                        job.cancel_event.set()
            self._upload_queue.put(None)
            self._upload_thread.join(timeout=max(1.0, deadline - time.monotonic()))

        with self._state_lock:
            committed = sorted(self._committed_step_list)

        try:
            self._send(
                {
                    "op": OP_BYE,
                    "converged_steps": committed,
                    "error": "; ".join(errors) if errors else None,
                }
            )
        except OSError:
            pass

    def shutdown(self) -> None:
        self._stop.set()
        if self._upload_thread is not None and self._upload_thread.is_alive():
            with self._cond:
                for job in self._active_uploads.values():
                    job.cancel_event.set()
                self._cond.notify_all()
            self._upload_queue.put(None)
            self._upload_thread.join(timeout=2.0)
        self.gc.stop(timeout=2.0)
        if self.comm is not None:
            self.comm.close()
        if self.remote is not None:
            try:
                self.remote.close()
            except Exception:
                pass
        self._close_memfd()
        try:
            self.sock.close()
        except OSError:
            pass


def worker_main(sock: socket.socket, config: SaverConfig) -> None:
    """存储子进程入口（通过 ``multiprocessing.get_context("spawn")`` 启动）。"""
    _install_parent_death_signal()
    worker = _StorageWorker(sock, config)

    def _on_sigterm(_signum: int, _frame: Any) -> None:
        # 信号处理函数内严禁触碰 threading.Lock / Event（会自死锁 GIL）
        if not config.drain_on_parent_death:
            os._exit(0)
        try:
            worker.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        worker.run()
    finally:
        worker.shutdown()


__all__ = ["worker_main"]
