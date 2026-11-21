"""训练进程侧异步保存客户端（§6）：``AsyncSaver`` / ``SaveHandle`` / ``ckpt-io`` 线程。

关键约束：
- ``_reconcile()`` 仅在训练线程（``save_async`` 入口与 ``SaveHandle.wait``）执行，
  保证 ``SnapshotManager`` 与 ``ShmPinnedPool`` 全程无锁；
- 训练进程不做 ``_clear_padding``，也不把 ``blobs`` 写入共享内存（Q4）；
- ``ckpt-io`` 线程独占 worker 控制套接字，负责发送、接收、心跳看门狗、退避重启、
  熔断与看门狗补偿作废广播。
"""

from __future__ import annotations

import multiprocessing
import os
import queue
import selectors
import signal
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from async_ import (
    CopyHandle,
    D2HStream,
    Snapshot,
    SnapshotManager,
    SnapshotState,
    Tensor,
    TriggerDecision,
)
from commit.local import CommitRecord
from core import CheckpointError, PlanError
from core.region import Region
from planner.save_plan import PLANNER_MODES, SavePlan, ShardSource, build_save_plan
from saver.comm import (
    CommEndpoint,
    NodeComm,
    NodeCommClient,
    ROLE_TRAINER,
    build_local_audit_summary,
    finalize_audit_summary,
    source_to_audit_dict,
)
from saver.config import SaverConfig, SaverDeadError, SaverFaults
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
    ProtocolError,
    compute_plan_id,
    encode_blobs,
    plan_to_dict,
    recv_msg,
    send_msg,
)
from saver.shm import ShmBuffer, ShmPinnedPool
from saver.worker import worker_main
from storage.base import StorageError
from storage.checksum import CRC_CHUNK_MAX
from storage.local import DEFAULT_ALIGNMENT


@dataclass(frozen=True)
class SaveResult:
    """异步保存单次提交结果（不含索引，以盘上 ``.metadata`` 为权威，I11）。"""

    step: int
    file: str
    size: int
    data_bytes: int
    d2h_bytes: int
    engine: str
    state: SnapshotState
    commit: CommitRecord
    generation: str | None = None


class SaveHandle:
    """``save_async()`` 返回的异步句柄。"""

    def __init__(
        self,
        *,
        step: int,
        decision: TriggerDecision,
        reason: str = "",
        snapshot: Snapshot | None = None,
        saver: "AsyncSaver | None" = None,
        gen: int | None = None,
    ) -> None:
        self.step = step
        self.decision = decision
        self.reason = reason
        self.snapshot = snapshot
        self._saver = saver
        self._gen = gen

        self._d2h_done_event = threading.Event()
        self._io_arrived = threading.Event()
        self._done_event = threading.Event()
        self._d2h_error: BaseException | None = None
        self._error: BaseException | None = None
        self._result: SaveResult | None = None
        self._error_reported = False
        self._tensors_ref: Any = None
        self._d2h_bytes: int = 0

        if decision is not TriggerDecision.START:
            self._d2h_done_event.set()
            self._io_arrived.set()
            self._done_event.set()

    @classmethod
    def completed_sync(cls, report: Any) -> "SaveHandle":
        """``on_worker_dead="fallback_sync"`` 降级同步写盘后构造已完成句柄。"""
        handle = cls(
            step=report.step,
            decision=TriggerDecision.START,
            reason="fallback_sync",
            snapshot=None,
        )
        handle._d2h_bytes = report.d2h_bytes
        handle._result = SaveResult(
            step=report.step,
            file=report.file,
            size=report.size,
            data_bytes=report.data_bytes,
            d2h_bytes=report.d2h_bytes,
            engine=report.engine,
            state=report.state,
            commit=report.commit,
            generation=None,
        )
        handle._d2h_done_event.set()
        handle._io_arrived.set()
        handle._done_event.set()
        return handle

    @property
    def skipped(self) -> bool:
        return self.decision is not TriggerDecision.START

    @property
    def done(self) -> bool:
        if (
            not self._done_event.is_set()
            and self._io_arrived.is_set()
            and self._saver is not None
            and threading.current_thread() is not self._saver._io_thread
        ):
            self._saver._reconcile()
        return self._done_event.is_set()

    @property
    def error(self) -> BaseException | None:
        if (
            not self._done_event.is_set()
            and self._io_arrived.is_set()
            and self._saver is not None
            and threading.current_thread() is not self._saver._io_thread
        ):
            self._saver._reconcile()
        return self._error

    @property
    def state(self) -> SnapshotState | None:
        if self.snapshot is not None:
            return self.snapshot.state
        if self._result is not None:
            return self._result.state
        return None

    def wait_d2h(self, timeout: float | None = None) -> None:
        if not self._d2h_done_event.wait(timeout=timeout):
            raise TimeoutError(f"step {self.step}: 等待 D2H 拷贝超时")
        if self._d2h_error is not None:
            raise self._d2h_error

    def wait(self, timeout: float | None = None) -> SaveResult | None:
        if self.skipped:
            return None
        if self._done_event.is_set():
            return self.result()

        deadline = None if timeout is None else (time.monotonic() + timeout)
        while not self._done_event.is_set():
            if self._saver is not None and threading.current_thread() is not self._saver._io_thread:
                self._saver._reconcile()
                if self._done_event.is_set():
                    break
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining is not None and remaining <= 0:
                raise TimeoutError(f"step {self.step}: 等待异步落盘超时（{timeout}s）")
            wait_step = 0.05 if remaining is None else min(remaining, 0.05)
            self._io_arrived.wait(timeout=wait_step)
            if self._saver is not None and threading.current_thread() is not self._saver._io_thread:
                self._saver._reconcile()
                if self._done_event.is_set():
                    break
            if self._done_event.wait(timeout=0.0):
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"step {self.step}: 等待异步落盘超时（{timeout}s）")
        return self.result()

    def result(self) -> SaveResult | None:
        if self.skipped:
            return None
        if not self.done:
            raise CheckpointError(f"step {self.step}: 异步保存尚未完成")
        if self._error is not None:
            self._error_reported = True
            raise self._error
        return self._result


def _sources_from_tensors(tensors: Mapping[str, Tensor]) -> list[ShardSource]:
    sources: list[ShardSource] = []
    for name, tensor in tensors.items():
        shape = tuple(int(dim) for dim in tensor.shape)
        if not shape:
            raise ValueError(f"{name}: 0 维张量请走 blobs 路径，不要当分片")
        sources.append(
            ShardSource(
                fqn=name,
                global_shape=shape,
                dtype=tensor.dtype,
                region=Region(tuple(0 for _ in shape), shape),
            )
        )
    return sources


@dataclass
class _PendingSnapshot:
    gen: int
    step: int
    run_id: str | None
    snapshot: Snapshot
    handle: SaveHandle
    on_error: str
    want_upload: bool
    generation: str | None


class AsyncSaver:
    """训练进程侧异步保存管理器（持有 ``ShmPinnedPool`` / ``SnapshotManager`` / ``ckpt-io``）。"""

    def __init__(
        self,
        config: SaverConfig,
        *,
        comm: NodeComm | NodeCommClient | CommEndpoint | None = None,
        world_size: int | None = None,
    ) -> None:
        self._owner_thread = threading.current_thread()
        self._reconcile_lock = threading.Lock()
        self._closed = False
        self.dead = False
        self._dead_error: BaseException | None = None
        self.worker_restarts = 0

        # 解析节点独立通信转发层（M3）
        self._owned_node_comm: NodeComm | None = None
        self._owned_comm_client: NodeCommClient | None = None
        self.comm: NodeCommClient | None = None
        effective_endpoint = config.comm_endpoint

        if comm is not None:
            if isinstance(comm, NodeCommClient):
                self.comm = comm
                effective_endpoint = comm.endpoint
            elif isinstance(comm, NodeComm):
                self.comm = comm.client(config.rank, role=ROLE_TRAINER)
                effective_endpoint = comm.endpoint
            elif isinstance(comm, CommEndpoint):
                self._owned_comm_client = NodeCommClient(
                    comm, rank=config.rank, role=ROLE_TRAINER
                )
                self.comm = self._owned_comm_client
                effective_endpoint = comm
            else:
                raise TypeError(
                    f"comm must be NodeComm, NodeCommClient, CommEndpoint or None, "
                    f"got {type(comm).__name__}"
                )
        elif effective_endpoint is not None:
            self._owned_comm_client = NodeCommClient(
                effective_endpoint, rank=config.rank, role=ROLE_TRAINER
            )
            self.comm = self._owned_comm_client
        else:
            ws = 1 if world_size is None else int(world_size)
            if ws > 1:
                raise ValueError(
                    f"world_size={ws} 需要提供 comm（节点独立通信转发层）"
                )
            if config.remote_config is not None or ws >= 1:
                self._owned_node_comm = NodeComm(
                    world_size=ws, timeout=config.collective_timeout
                )
                self.comm = self._owned_node_comm.client(config.rank, role=ROLE_TRAINER)
                effective_endpoint = self._owned_node_comm.endpoint

        if world_size is not None and self.comm is not None and self.comm.world_size != world_size:
            raise ValueError(
                f"声明的 world_size={world_size} 与 comm.world_size={self.comm.world_size} 不一致"
            )

        if effective_endpoint is not None and effective_endpoint != config.comm_endpoint:
            from dataclasses import replace

            config = replace(config, comm_endpoint=effective_endpoint)

        self.config = config
        self._faults = config.faults or SaverFaults()

        # 数据面：memfd + ShmPinnedPool
        self._pool = ShmPinnedPool(
            capacity=config.shm_bytes,
            max_bytes=config.shm_max_bytes,
            alignment=config.alignment,
            cuda_register=config.cuda_register,
        )
        self._manager = SnapshotManager(
            self._pool,
            max_inflight=config.max_inflight,
            force_after_skips=config.force_after_skips,
            collective=self.comm,
            world_size=self.comm.world_size if self.comm is not None else 1,
            collective_timeout=config.collective_timeout,
        )
        # 确保 D2HStream 在 max_inflight >= 2 时绝不因上一代拷贝任务阻塞主线程
        self._d2h = D2HStream(max_inflight=4096, name=f"ckpt-d2h-r{config.rank}")

        self._max_step: int | None = None
        self._gen_seq = 0
        self._pending: dict[int, _PendingSnapshot] = {}
        self._all_handles: list[SaveHandle] = []
        self._known_plans: dict[str, dict[str, Any]] = {}
        self._plan_sent: set[str] = set()
        self._discarded: list[tuple[str | None, int]] = []
        self._protected_steps: set[int] = set()
        self._upload_done_steps: list[int] = []
        self._upload_failures: list[dict[str, Any]] = []
        self._converged_steps: list[int] = []

        # ckpt-io 通信队列与事件
        self._outbox: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self._inbox: queue.Queue[tuple[str, int, dict[str, Any]]] = queue.Queue()
        self._wakeup_r, self._wakeup_w = socket.socketpair()
        self._wakeup_r.setblocking(False)
        self._wakeup_w.setblocking(False)

        self._io_stop = threading.Event()
        self._io_ready = threading.Event()
        self._io_init_error: BaseException | None = None
        self._io_closed_event = threading.Event()
        self._shutdown_error: str | None = None
        self._restart_times: deque[float] = deque()

        self._worker_proc: Any = None
        self._worker_sock: socket.socket | None = None

        self._io_thread = threading.Thread(
            target=self._io_loop,
            name=f"ckpt-io-r{config.rank}",
            daemon=True,
        )
        self._io_thread.start()
        self._io_ready.wait(timeout=15.0)
        if self._io_init_error is not None:
            err = self._io_init_error
            self.close(timeout=2.0)
            raise err

    @property
    def pool(self) -> ShmPinnedPool:
        return self._pool

    @property
    def manager(self) -> SnapshotManager:
        return self._manager

    @property
    def max_step(self) -> int | None:
        return self._max_step

    @property
    def worker_pid(self) -> int | None:
        proc = self._worker_proc
        return proc.pid if proc is not None else None

    @property
    def upload_done_steps(self) -> tuple[int, ...]:
        return tuple(self._upload_done_steps)

    @property
    def upload_failures(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._upload_failures)

    @property
    def converged_steps(self) -> tuple[int, ...]:
        return tuple(self._converged_steps)

    def set_max_step(self, step: int | None) -> None:
        """同轮次恢复后同步水位线（§8.3）。"""
        self._max_step = step

    def recover(
        self,
        remote: Any,
        *,
        step: int | None = None,
        source_run_id: str | None = None,
        device: str = "cpu",
        tp: int = 1,
        tp_rank: int = 0,
        shard_dims: Mapping[str, int] | None = None,
        prefer_local: bool = True,
        promote_tails: bool = True,
        **kwargs: Any,
    ) -> Any:
        """经节点独立通信转发层执行集群恢复，并在恢复期间暂停/锁定后台 GC（§8.3）。"""
        from storage.upload import load_checkpoint_version

        self.protect_steps(
            add_steps=[step] if step is not None else (),
            pause_gc=True,
        )
        protected_step = step
        try:
            report = load_checkpoint_version(
                remote,
                local_root=self.config.local_root,
                step=step,
                device=device,
                tp=tp,
                tp_rank=tp_rank,
                shard_dims=shard_dims,
                prefer_local=prefer_local,
                collective=self.comm,
                run_id=self.config.run_id,
                source_run_id=source_run_id,
                promote_tails=promote_tails,
                on_target_step=lambda target_step: self.protect_steps(
                    add_steps=[target_step]
                ),
                **kwargs,
            )
            protected_step = report.step
            if source_run_id is None and (
                self._max_step is None or report.step > self._max_step
            ):
                self._max_step = report.step
            return report
        finally:
            release = [protected_step] if protected_step is not None else ()
            self.protect_steps(release_steps=release, pause_gc=False)

    def protect_steps(
        self,
        *,
        add_steps: Sequence[int] = (),
        release_steps: Sequence[int] = (),
        pause_gc: bool | None = None,
    ) -> None:
        """通知存储子进程保护或释放正在恢复读取的步数（与 GC 联动）。"""
        for s in add_steps:
            self._protected_steps.add(int(s))
        for s in release_steps:
            self._protected_steps.discard(int(s))
        if self.comm is not None and self.comm.world_size > 1:
            self.comm.broadcast_protect(
                add_steps=add_steps,
                release_steps=release_steps,
                pause_gc=pause_gc,
            )
        else:
            self._enqueue_outbox(
                (
                    OP_PROTECT,
                    {
                        "op": OP_PROTECT,
                        "add_steps": [int(s) for s in add_steps],
                        "release_steps": [int(s) for s in release_steps],
                        "pause_gc": pause_gc,
                    },
                )
            )

    def save(
        self,
        tensors: Mapping[str, Tensor],
        *,
        step: int,
        blobs: Mapping[str, bytes] | None = None,
        layout: Mapping[str, int] | None = None,
        rank: int | None = None,
        sources: list[ShardSource] | None = None,
        planner: str = "local",
        alignment: int | None = None,
        engine: str | None = None,
        direct_io: bool | None = None,
        crc_chunk_max: int = CRC_CHUNK_MAX,
        upload: bool = True,
        generation: str | None = None,
        on_error: str = "raise_next",
    ) -> SaveHandle:
        """同步段：reconcile → step 单调校验 → plan → 门闸 → D2H 提交 → 立即返回。"""
        if self._closed:
            raise CheckpointError("AsyncSaver is already closed")
        if on_error not in ("raise_next", "ignore"):
            raise ValueError(f"on_error must be 'raise_next' or 'ignore', got {on_error!r}")

        self._reconcile()

        # on_error="raise_next"：若先前异步保存失败且调用方未在 wait() 上消费该错误，在此抛出
        if on_error == "raise_next":
            for h in list(self._all_handles):
                if h.done and h._error is not None and not h._error_reported:
                    h._error_reported = True
                    raise h._error

        eff_rank = self.config.rank if rank is None else int(rank)
        eff_alignment = self.config.alignment if alignment is None else int(alignment)
        eff_engine = self.config.engine if engine is None else str(engine)
        eff_direct_io = self.config.direct_io if direct_io is None else bool(direct_io)

        if self.dead:
            if self.config.on_worker_dead == "fallback_sync":
                import api as ckpt_api

                if self._max_step is not None and step <= self._max_step:
                    raise PlanError(
                        f"run {self.config.run_id or '<default>'}: step {step} 必须严格大于"
                        f"当前轮次水位 {self._max_step}"
                    )
                report = ckpt_api.save_local(
                    tensors,
                    self.config.local_root,
                    step=step,
                    blobs=blobs,
                    layout=layout,
                    rank=eff_rank,
                    sources=sources,
                    planner=planner,
                    alignment=eff_alignment,
                    engine=eff_engine,
                    direct_io=eff_direct_io,
                    crc_chunk_max=crc_chunk_max,
                    generation=generation,
                    run_id=self.config.run_id,
                )
                self._max_step = step
                return SaveHandle.completed_sync(report)
            assert self._dead_error is not None
            raise self._dead_error

        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        # Q2 / I12：同轮次内 step 严格单调递增
        if self._max_step is not None and step <= self._max_step:
            raise PlanError(
                f"run {self.config.run_id or '<default>'}: step {step} 必须严格大于"
                f"当前轮次水位 {self._max_step}"
            )

        if planner not in PLANNER_MODES:
            raise ValueError(f"planner must be one of {PLANNER_MODES}, got {planner!r}")

        blob_payloads = dict(blobs or {})
        resolved_sources = (
            _sources_from_tensors(tensors) if sources is None else list(sources)
        )
        blob_sizes = {name: len(payload) for name, payload in blob_payloads.items()}

        if planner == "gathered":
            if self.comm is not None and self.comm.world_size > 1:
                self.comm.audit_save_plan(
                    resolved_sources,
                    blob_names=sorted(blob_sizes),
                    timeout=self.config.collective_timeout,
                )
            else:
                summary = build_local_audit_summary(
                    [source_to_audit_dict(s) for s in resolved_sources],
                    blob_names=sorted(blob_sizes),
                )
                finalize_audit_summary(summary)

        plan = build_save_plan(
            resolved_sources,
            rank=eff_rank,
            blobs=blob_sizes,
            align=eff_alignment,
        )
        plan.validate()
        if plan.size == 0 and planner != "gathered":
            raise PlanError("保存计划是空的：没有分片也没有小状态")
        aligned_size = -(-plan.size // eff_alignment) * eff_alignment

        # 仅在途代数为 0 时允许扩容 memfd，不调用 trim()（§4）
        if not self._manager.active:
            reallocated = self._pool.ensure_capacity(
                aligned_size, slots=self.config.max_inflight
            )
            if reallocated:
                self._enqueue_outbox(
                    (
                        OP_MEMFD,
                        self._pool.segment_id,
                        self._pool.fd,
                        self._pool.capacity,
                    )
                )

        trigger = self._manager.trigger(step, nbytes=aligned_size)
        if trigger.decision is TriggerDecision.FORCE_WAIT:
            self._wait_oldest_inflight()
            self._reconcile()
            trigger = self._manager.trigger(step, nbytes=aligned_size)

        if trigger.decision is not TriggerDecision.START or trigger.snapshot is None:
            return SaveHandle(
                step=step,
                decision=trigger.decision,
                reason=trigger.reason,
                snapshot=None,
                saver=self,
            )

        snapshot = trigger.snapshot
        # 门闸裁决 START：立刻推进当前轮次水位线（Q2）
        self._max_step = step
        self._gen_seq += 1
        gen = self._gen_seq

        handle = SaveHandle(
            step=step,
            decision=TriggerDecision.START,
            snapshot=snapshot,
            saver=self,
            gen=gen,
        )
        # 强引用 tensors 直至 D2H 完成（§6）
        handle._tensors_ref = dict(tensors)

        pending = _PendingSnapshot(
            gen=gen,
            step=step,
            run_id=self.config.run_id,
            snapshot=snapshot,
            handle=handle,
            on_error=on_error,
            want_upload=bool(upload),
            generation=generation,
        )
        self._pending[gen] = pending
        self._all_handles.append(handle)
        if len(self._all_handles) > 128:
            del self._all_handles[:-64]

        plan_id = compute_plan_id(
            plan,
            layout=layout,
            alignment=eff_alignment,
            crc_chunk_max=crc_chunk_max,
        )
        plan_payload = {
            "op": OP_PLAN,
            "plan_id": plan_id,
            "plan": plan_to_dict(plan),
            "layout": {str(k): int(v) for k, v in sorted((layout or {}).items())},
            "crc_chunk_max": int(crc_chunk_max),
            "alignment": int(eff_alignment),
        }
        self._known_plans[plan_id] = plan_payload

        buf = snapshot.buffer
        assert isinstance(buf, ShmBuffer)
        snap_payload = {
            "op": OP_SNAP,
            "gen": gen,
            "step": step,
            "run_id": self.config.run_id,
            "plan_id": plan_id,
            "buffer": {
                "offset": buf.offset,
                "length": buf.capacity,
                "aligned_size": aligned_size,
            },
            "blobs": encode_blobs(blob_payloads),
            "engine": eff_engine,
            "direct_io": eff_direct_io,
            "upload": bool(upload),
            "generation": generation,
        }

        # 校验张量并提交 D2H 拷贝（不碰 blobs、不清零空隙）
        copy_handles: list[CopyHandle[int]] = []
        total_tensor_bytes = 0
        try:
            for item in plan.items:
                tensor = tensors.get(item.fqn)
                if tensor is None:
                    raise PlanError(f"{item.fqn}: 计划里有，但没有对应的张量")
                if tensor.nbytes != item.length:
                    raise PlanError(
                        f"{item.fqn}: 本地张量 {tensor.nbytes} 字节与计划里的 "
                        f"{item.length} 不一致"
                    )
                total_tensor_bytes += item.length
                copy_handles.append(
                    self._d2h.submit_copy(tensor, buf, offset=item.byte_offset)
                )
        except BaseException as exc:
            self._pending.pop(gen, None)
            snapshot.fail(f"{type(exc).__name__}: {exc}")
            handle._tensors_ref = None
            handle._d2h_error = exc
            handle._d2h_done_event.set()
            handle._io_arrived.set()
            handle._done_event.set()
            raise

        handle._d2h_bytes = total_tensor_bytes + sum(len(v) for v in blob_payloads.values())

        def on_all_d2h_done(_last_copy: Any = None) -> None:
            handle._tensors_ref = None
            first_err: BaseException | None = None
            for ch in copy_handles:
                exc = ch.exception()
                if exc is not None:
                    first_err = exc
                    break
            if first_err is not None:
                handle._d2h_error = first_err
                handle._d2h_done_event.set()
                self._push_inbox(
                    "FAIL",
                    gen,
                    {
                        "op": OP_FAIL,
                        "gen": gen,
                        "step": step,
                        "kind": "d2h_error",
                        "message": f"{type(first_err).__name__}: {first_err}",
                        "state": "dropped",
                    },
                    handle=handle,
                )
                return

            handle._d2h_done_event.set()
            self._enqueue_outbox(
                (OP_SNAP, gen, plan_id, snap_payload, handle, bool(upload), generation)
            )

        if copy_handles:
            copy_handles[-1].add_done_callback(on_all_d2h_done)
        else:
            on_all_d2h_done()

        return handle

    def _wait_oldest_inflight(self) -> None:
        while self._manager.active:
            oldest = self._manager.active[0]
            matching = next(
                (p.handle for p in self._pending.values() if p.snapshot is oldest),
                None,
            )
            if matching is None:
                break
            matching.wait()
            self._reconcile()

    def _reconcile(self) -> None:
        """训练线程唯一入口：排空 inbox，推进 Snapshot 状态机并归还 ShmBuffer。"""
        with self._reconcile_lock:
            self._reconcile_locked()

    def _reconcile_locked(self) -> None:
        while True:
            try:
                kind, gen, msg = self._inbox.get_nowait()
            except queue.Empty:
                break

            pending = self._pending.pop(gen, None)
            if pending is None:
                continue
            snap = pending.snapshot
            handle = pending.handle

            if kind == "DONE":
                if snap.state is SnapshotState.SNAPSHOTTING:
                    snap.d2h_done()
                if snap.state is SnapshotState.PERSISTING:
                    snap.persist_ok()
                if snap.state is SnapshotState.COMMITTING:
                    snap.commit_ok()

                commit_info = msg["commit"]
                commit = CommitRecord(
                    step=int(msg["step"]),
                    metadata_bytes=int(commit_info["metadata_bytes"]),
                    metadata_crc32c=str(commit_info["metadata_crc32c"]),
                    files={
                        str(k): int(v) for k, v in dict(commit_info["files"]).items()
                    },
                )
                file_name = (
                    next(iter(commit.files))
                    if commit.files
                    else f"__{self.config.rank}_0.distcp"
                )
                handle._result = SaveResult(
                    step=int(msg["step"]),
                    file=file_name,
                    size=int(msg["size"]),
                    data_bytes=int(msg["data_bytes"]),
                    d2h_bytes=handle._d2h_bytes,
                    engine=str(msg["engine"]),
                    state=snap.state,
                    commit=commit,
                    generation=msg.get("generation"),
                )
                handle._done_event.set()
            else:
                err_kind = str(msg.get("kind", "CheckpointError"))
                err_msg = str(msg.get("message", "async save failed"))
                if err_kind != "d2h_error" and snap.state is SnapshotState.SNAPSHOTTING:
                    snap.d2h_done()
                if not snap.is_terminal:
                    snap.fail(f"{err_kind}: {err_msg}")
                    if snap.state is SnapshotState.ABORTED:
                        try:
                            snap.mark_cleaned()
                        except Exception:
                            pass
                if err_kind == "worker_died":
                    handle._error = SaverDeadError(err_msg)
                elif "Storage" in err_kind or "OS" in err_kind:
                    handle._error = StorageError(err_msg)
                else:
                    handle._error = CheckpointError(err_msg)
                handle._done_event.set()

    def _push_inbox(
        self,
        kind: str,
        gen: int,
        msg: dict[str, Any],
        *,
        handle: SaveHandle | None = None,
    ) -> None:
        self._inbox.put((kind, gen, msg))
        if handle is not None:
            handle._io_arrived.set()

    def _enqueue_outbox(self, item: tuple[Any, ...]) -> None:
        self._outbox.put(item)
        try:
            self._wakeup_w.send(b"\x01")
        except OSError:
            pass

    # --- ckpt-io 后台线程（发送 / 接收 / 心跳 / 重启 / 熔断） ----------------

    def _io_loop(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        io_inflight: dict[int, dict[str, Any]] = {}
        closing = False
        close_timeout = self.config.shutdown_timeout

        def spawn_worker() -> bool:
            while not self._io_stop.is_set():
                now = time.monotonic()
                while (
                    self._restart_times
                    and (now - self._restart_times[0]) > self.config.restart_window
                ):
                    self._restart_times.popleft()

                parent_sock, child_sock = socket.socketpair(
                    socket.AF_UNIX, socket.SOCK_STREAM
                )
                proc = ctx.Process(
                    target=worker_main,
                    args=(child_sock, self.config),
                    name=f"shardckpt-worker-r{self.config.rank}",
                )
                proc.start()
                child_sock.close()

                try:
                    # 握手：发 memfd → 等 READY → 发 STARTUP
                    send_msg(
                        parent_sock,
                        {
                            "op": OP_MEMFD,
                            "capacity": self._pool.capacity,
                            "segment_id": self._pool.segment_id,
                        },
                        fds=[self._pool.fd],
                    )
                    parent_sock.settimeout(10.0)
                    ready = recv_msg(parent_sock)
                    parent_sock.settimeout(None)
                    if ready is None or ready[0].get("op") != OP_READY:
                        raise EOFError("worker did not send READY")
                    for fd in ready[1]:
                        os.close(fd)

                    send_msg(
                        parent_sock,
                        {
                            "op": OP_STARTUP,
                            "cleanup": [
                                [r_id, st] for r_id, st in self._discarded
                            ],
                            "protected_steps": sorted(self._protected_steps),
                        },
                    )
                    self._worker_proc = proc
                    self._worker_sock = parent_sock
                    self._plan_sent.clear()
                    return True
                except BaseException as exc:
                    try:
                        parent_sock.close()
                    except OSError:
                        pass
                    if proc.is_alive():
                        proc.kill()
                    proc.join(timeout=2.0)
                    if not self._record_restart_or_trip_circuit():
                        if not self._io_ready.is_set():
                            self._io_ready.set()
                        return False
                    if self.config.restart_backoff > 0:
                        time.sleep(self.config.restart_backoff)
                    self.worker_restarts += 1
            return False

        def handle_worker_death(reason: str) -> bool:
            proc = self._worker_proc
            sock = self._worker_sock
            self._worker_proc = None
            self._worker_sock = None
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
            if proc is not None:
                if proc.is_alive():
                    try:
                        os.kill(proc.pid, signal.SIGKILL)
                    except OSError:
                        pass
                proc.join(timeout=2.0)

            failed_items = list(io_inflight.items())
            io_inflight.clear()
            self._plan_sent.clear()

            for _gen, info in failed_items:
                step = int(info["step"])
                run_id = info["run_id"]
                pair = (run_id, step)
                if pair not in self._discarded:
                    self._discarded.append(pair)
                # 看门狗补偿广播作废事件，避免同伴干等（§7.3 规则 2）
                if (
                    self.comm is not None
                    and info.get("want_upload")
                    and self.config.remote_config is not None
                ):
                    try:
                        self.comm.broadcast_invalidation(
                            run_id=run_id,
                            step=step,
                            generation=info.get("generation"),
                            reason=f"worker died: {reason}",
                        )
                    except Exception:
                        pass

            restarted = False
            if not closing and not self._io_stop.is_set():
                if self._record_restart_or_trip_circuit():
                    if self.config.restart_backoff > 0:
                        time.sleep(self.config.restart_backoff)
                    if spawn_worker():
                        self.worker_restarts += 1
                        restarted = True

            for gen, info in failed_items:
                self._push_inbox(
                    "FAIL",
                    gen,
                    {
                        "op": OP_FAIL,
                        "gen": gen,
                        "step": int(info["step"]),
                        "kind": "worker_died",
                        "message": reason,
                        "state": "aborted",
                    },
                    handle=info.get("handle"),
                )

            return restarted

        if not spawn_worker():
            self._io_closed_event.set()
            return
        self._io_ready.set()

        sel_timeout = min(1.0, max(0.05, float(self.config.heartbeat_timeout) / 4.0))

        while not self._io_stop.is_set():
            sock = self._worker_sock
            proc = self._worker_proc
            if sock is None or proc is None:
                break

            # 1. 发送 outbox
            while True:
                try:
                    item = self._outbox.get_nowait()
                except queue.Empty:
                    break
                op = item[0]
                try:
                    if op == OP_MEMFD:
                        _op, _seg_id, fd, cap = item
                        send_msg(sock, {"op": OP_MEMFD, "capacity": cap}, fds=[fd])
                    elif op == OP_PROTECT:
                        _op, payload = item
                        send_msg(sock, payload)
                    elif op == OP_SNAP:
                        _op, gen, plan_id, snap_payload, handle, want_upload, explicit_gen = item
                        if plan_id not in self._plan_sent:
                            send_msg(sock, self._known_plans[plan_id])
                            self._plan_sent.add(plan_id)
                        now = time.monotonic()
                        io_inflight[gen] = {
                            "step": snap_payload["step"],
                            "run_id": snap_payload.get("run_id"),
                            "generation": explicit_gen,
                            "want_upload": want_upload,
                            "last_hb": now,
                            "started_at": now,
                            "handle": handle,
                        }
                        send_msg(sock, snap_payload)
                    elif op == OP_CLOSE:
                        _op, drain_uploads, close_timeout = item
                        closing = True
                        send_msg(
                            sock,
                            {
                                "op": OP_CLOSE,
                                "drain_uploads": drain_uploads,
                                "timeout": close_timeout,
                            },
                        )
                except (OSError, ProtocolError) as exc:
                    if not handle_worker_death(f"send failed: {exc}"):
                        self._io_closed_event.set()
                        return
                    break

            sock = self._worker_sock
            proc = self._worker_proc
            if sock is None or proc is None:
                break

            # 2. select 等待控制帧或超时检查心跳
            sel = selectors.DefaultSelector()
            try:
                sel.register(self._wakeup_r, selectors.EVENT_READ, data="wakeup")
                sel.register(sock, selectors.EVENT_READ, data="worker")
                wait_t = sel_timeout if not closing else min(0.2, close_timeout)
                events = sel.select(timeout=wait_t)
            finally:
                sel.close()

            worker_dead_reason: str | None = None
            got_bye = False
            for key, _mask in events:
                if key.data == "wakeup":
                    try:
                        self._wakeup_r.recv(1024)
                    except OSError:
                        pass
                elif key.data == "worker":
                    try:
                        res = recv_msg(sock)
                    except (OSError, EOFError, ProtocolError) as exc:
                        res = None
                        worker_dead_reason = f"recv failed: {exc}"
                    if res is None:
                        worker_dead_reason = worker_dead_reason or "worker closed socket"
                        break
                    msg, fds = res
                    for fd in fds:
                        os.close(fd)
                    mop = msg.get("op")
                    if mop == OP_PROGRESS:
                        g = int(msg["gen"])
                        if g in io_inflight:
                            io_inflight[g]["last_hb"] = time.monotonic()
                    elif mop == OP_NEED_PLAN:
                        pid = str(msg["plan_id"])
                        plan_msg = self._known_plans.get(pid)
                        if plan_msg is not None:
                            try:
                                send_msg(sock, plan_msg)
                                self._plan_sent.add(pid)
                            except OSError as exc:
                                worker_dead_reason = f"resend plan failed: {exc}"
                                break
                    elif mop in (OP_DONE, OP_FAIL):
                        g = int(msg["gen"])
                        info = io_inflight.pop(g, None)
                        if mop == OP_FAIL and info is not None:
                            pair = (info["run_id"], int(info["step"]))
                            if pair not in self._discarded:
                                self._discarded.append(pair)
                        self._push_inbox(
                            str(mop),
                            g,
                            msg,
                            handle=info.get("handle") if info else None,
                        )
                    elif mop == OP_UPLOAD_DONE:
                        self._upload_done_steps.append(int(msg["step"]))
                    elif mop == OP_UPLOAD_FAIL:
                        self._upload_failures.append(msg)
                    elif mop == OP_BYE:
                        self._converged_steps = [
                            int(s) for s in msg.get("converged_steps", ())
                        ]
                        self._shutdown_error = msg.get("error")
                        got_bye = True
                        break

            if got_bye:
                break

            if worker_dead_reason is None and not proc.is_alive():
                worker_dead_reason = f"worker exited with code {proc.exitcode}"

            if worker_dead_reason is None and io_inflight and not closing:
                now = time.monotonic()
                oldest_hb = min(info["last_hb"] for info in io_inflight.values())
                if now - oldest_hb > self.config.heartbeat_timeout:
                    worker_dead_reason = (
                        f"heartbeat timeout ({self.config.heartbeat_timeout}s)"
                    )

            if worker_dead_reason is not None:
                if not handle_worker_death(worker_dead_reason):
                    break

        self._io_closed_event.set()

    def _record_restart_or_trip_circuit(self) -> bool:
        now = time.monotonic()
        while (
            self._restart_times
            and (now - self._restart_times[0]) > self.config.restart_window
        ):
            self._restart_times.popleft()
        if len(self._restart_times) >= self.config.max_restarts:
            self.dead = True
            self._dead_error = SaverDeadError(
                f"存储子进程在 {self.config.restart_window}s 窗口内重启超过 "
                f"max_restarts={self.config.max_restarts} 次，已熔断"
            )
            return False
        self._restart_times.append(now)
        return True

    def close(self, *, timeout: float | None = None) -> None:
        if self._closed:
            return
        wait_timeout = (
            self.config.shutdown_timeout if timeout is None else float(timeout)
        )
        deadline = time.monotonic() + wait_timeout
        try:
            # 1. 训练线程等最后一代显存拷贝与本地落盘（L1）完成（§8.3）
            for pending in list(self._pending.values()):
                rem = max(0.1, deadline - time.monotonic())
                try:
                    pending.handle.wait(timeout=rem)
                except Exception:
                    pass
            self._reconcile()

            # 2. 通知 worker 排空后台上传 + 经节点通信层对称收敛（§8.3）
            if not self.dead and self._worker_sock is not None:
                rem = max(0.5, deadline - time.monotonic())
                self._enqueue_outbox((OP_CLOSE, True, rem))
                self._io_closed_event.wait(timeout=rem + 1.0)
        finally:
            self._closed = True
            self._io_stop.set()
            try:
                self._wakeup_w.send(b"\x01")
            except OSError:
                pass
            if self._io_thread.is_alive() and self._io_thread is not threading.current_thread():
                self._io_thread.join(timeout=2.0)

            if self._worker_sock is not None:
                try:
                    self._worker_sock.close()
                except OSError:
                    pass
                self._worker_sock = None
            if self._worker_proc is not None:
                self._worker_proc.join(timeout=2.0)
                if self._worker_proc.is_alive():
                    self._worker_proc.kill()
                    self._worker_proc.join(timeout=1.0)
                self._worker_proc = None

            self._d2h.close()
            self._pool.close()
            if self._owned_comm_client is not None:
                self._owned_comm_client.close()
            if self._owned_node_comm is not None:
                self._owned_node_comm.close()
            for s in (self._wakeup_r, self._wakeup_w):
                try:
                    s.close()
                except OSError:
                    pass


# --- 全局默认 AsyncSaver 会话管理 -------------------------------------------

_DEFAULT_SAVER_LOCK = threading.Lock()
_DEFAULT_SAVER: AsyncSaver | None = None
_DEFAULT_SAVER_PID = 0


def start_saver(
    config: SaverConfig,
    *,
    comm: NodeComm | NodeCommClient | CommEndpoint | None = None,
    world_size: int | None = None,
) -> AsyncSaver:
    """启动全局默认异步保存会话（若已存在则先报错，需先 ``close_saver()``）。"""
    global _DEFAULT_SAVER, _DEFAULT_SAVER_PID
    with _DEFAULT_SAVER_LOCK:
        if _DEFAULT_SAVER is not None and _DEFAULT_SAVER_PID == os.getpid() and not _DEFAULT_SAVER._closed:
            raise CheckpointError("AsyncSaver 已启动，请先调用 close_saver()")
        saver = AsyncSaver(config, comm=comm, world_size=world_size)
        _DEFAULT_SAVER = saver
        _DEFAULT_SAVER_PID = os.getpid()
        return saver


def close_saver(timeout: float = 30.0) -> None:
    """优雅关闭全局默认异步保存会话（等待 L1 + 远端上传收敛）。"""
    global _DEFAULT_SAVER, _DEFAULT_SAVER_PID
    with _DEFAULT_SAVER_LOCK:
        saver, _DEFAULT_SAVER = _DEFAULT_SAVER, None
        _DEFAULT_SAVER_PID = 0
    if saver is not None:
        saver.close(timeout=timeout)
        if saver._shutdown_error:
            raise CheckpointError(f"close_saver 远端收敛失败: {saver._shutdown_error}")


def default_saver() -> AsyncSaver:
    with _DEFAULT_SAVER_LOCK:
        if _DEFAULT_SAVER is None or _DEFAULT_SAVER_PID != os.getpid() or _DEFAULT_SAVER._closed:
            raise CheckpointError("尚未启动 AsyncSaver，请先调用 start_saver(config) 或传入 saver=")
        return _DEFAULT_SAVER


def save_async(
    tensors: Mapping[str, Tensor],
    *,
    step: int,
    blobs: Mapping[str, bytes] | None = None,
    layout: Mapping[str, int] | None = None,
    rank: int = 0,
    sources: list[ShardSource] | None = None,
    planner: str = "local",
    alignment: int = DEFAULT_ALIGNMENT,
    engine: str = "auto",
    direct_io: bool = True,
    crc_chunk_max: int = CRC_CHUNK_MAX,
    saver: AsyncSaver | None = None,
    upload: bool = True,
    generation: str | None = None,
    on_error: str = "raise_next",
) -> SaveHandle:
    """异步保存入口（§9）：提交至当前 ``AsyncSaver`` 并立即返回 ``SaveHandle``。"""
    target = saver if saver is not None else default_saver()
    return target.save(
        tensors,
        step=step,
        blobs=blobs,
        layout=layout,
        rank=rank,
        sources=sources,
        planner=planner,
        alignment=alignment,
        engine=engine,
        direct_io=direct_io,
        crc_chunk_max=crc_chunk_max,
        upload=upload,
        generation=generation,
        on_error=on_error,
    )


__all__ = [
    "AsyncSaver",
    "SaveHandle",
    "SaveResult",
    "close_saver",
    "default_saver",
    "save_async",
    "start_saver",
]
