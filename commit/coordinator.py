"""嵌入式提交协调监控层（阶段三：C1 / C2 / C3 / C4）。

形态 A（无独立常驻协调服务）：每个 rank 进程内挂一个 :class:`CommitCoordinator`，
复用 :mod:`dist` 的集群通信抽象（单 rank 自动退化为 :class:`dist.SingleRankCollective`）。

分工边界：
- **正常完成（远端见齐即提交）**：各 rank 后台上传完毕写各自的远端分片提交点
  （``rank_<r>/COMMIT``）；rank 0 后台见所有 rank 分片提交点齐备即直接写远端
  总提交点（``STATUS.committed``），不等训练线程下一次触发。
- **异常与状态同步（搭便车全归约）**：利用保存触发点（门闸全归约）与退出等待点
  的 ``all_reduce(MAX)`` 交换各 rank 的在途上传状态，统一得出「上一代是否作废/收敛」
  与「本步是否开新代」两个全局一致结论。
"""

from __future__ import annotations

import enum
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from async_ import (
    BufferPool,
    GateBlock,
    PinnedBufferPool,
    Snapshot,
    SnapshotManager,
    SnapshotState,
    Tensor,
    Trigger,
    TriggerDecision,
    default_pool,
)
from commit.gc import (
    DEFAULT_KEEP_LOCAL_VERSIONS,
    DEFAULT_KEEP_VERSIONS,
)
from commit.generation import (
    Generation,
    acquire_generation,
    new_writer,
)
from commit.local import delete_local_generation, write_local_generation
from core import CheckpointError, FormatError, PlanError
from core.metadata import Device
from dist import Collective, DEFAULT_TIMEOUT, SingleRankCollective
from planner.save_plan import ShardSource
from storage.base import StorageBackend, StorageError, run_scope, validate_run_id
from storage.checksum import CRC_CHUNK_MAX
from storage.local import DEFAULT_ALIGNMENT
from storage.upload import (
    DEFAULT_BUDGET_BYTES,
    DEFAULT_DOWNLOAD_BACKOFF,
    DEFAULT_DOWNLOAD_RETRIES,
    ObjectLoadReport,
    UploadCommit,
    committed_remote_steps,
    finalize_upload,
    generation_prefix,
    index_key,
    load_checkpoint_version,
    local_run_root,
    rank_commit_key,
    upload_rank,
)


class InflightStatus(enum.IntEnum):
    """在途后台上传任务的本地状态码（数值越大越优先通过 MAX 归约传播）。"""

    NONE = 0
    DONE = 1
    UPLOADING = 2
    FAILED = 3


@dataclass(frozen=True, slots=True)
class InvalidatedGeneration:
    """因失败或超时被全员一致宣告作废的代际记录（C3）。"""

    step: int
    generation: str
    reason: str


@dataclass(frozen=True)
class CoordinatedSaveResult:
    """协调层单步保存结果（可能跳过、或已提交本地并进入后台上传）。"""

    step: int
    decision: TriggerDecision
    generation: Generation | None = None
    reason: str = ""
    local_report: Any = None
    invalidated: tuple[InvalidatedGeneration, ...] = ()
    converged_steps: tuple[int, ...] = ()

    @property
    def skipped(self) -> bool:
        return self.decision is not TriggerDecision.START


@dataclass
class _InflightTask:
    step: int
    generation: Generation
    ranks: tuple[int, ...]
    layout: dict[str, int]
    started_at: float
    cancel_event: threading.Event = field(default_factory=threading.Event)
    local_upload_done: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)
    future: Future[Any] | None = None
    error: BaseException | None = None
    committed_remote: bool = False
    converged: bool = False
    invalidated: bool = False

    def current_status(self, *, now: float, upload_timeout: float) -> InflightStatus:
        if self.invalidated or self.converged:
            return InflightStatus.NONE
        if self.error is not None:
            return InflightStatus.FAILED
        if self.done_event.is_set():
            return InflightStatus.DONE
        if (now - self.started_at) >= upload_timeout:
            return InflightStatus.FAILED
        return InflightStatus.UPLOADING


def acquire_generation_for_step(
    scoped_remote: StorageBackend,
    step: int,
    *,
    ranks: Sequence[int],
    writer: str | None = None,
) -> Generation:
    """在远端为指定 ``step`` 执行端点条件写探针并 CAS 抢占新代际（可被存储侧复用）。"""
    ensure = getattr(scoped_remote, "ensure_conditional_writes", None)
    if callable(ensure):
        ensure()
    return acquire_generation(
        scoped_remote, step, ranks=ranks, writer=writer or new_writer()
    )


def wait_and_finalize_generation(
    scoped_remote: StorageBackend,
    *,
    step: int,
    generation: Generation,
    ranks: Sequence[int],
    layout: Mapping[str, int] | None = None,
    keep_versions: int | None = DEFAULT_KEEP_VERSIONS,
    cancel_event: threading.Event | None = None,
    timeout: float = 300.0,
    poll_interval: float = 0.02,
    deadline: float | None = None,
) -> UploadCommit | None:
    """主节点等待所有 rank 的 L2（``.metadata`` + ``rank_r/COMMIT``）齐备后原子落 L3。

    若 ``cancel_event`` 在等待期间置位则返回 ``None``；超时则抛 ``TimeoutError``。
    """
    prefix = generation_prefix(step, generation)
    expected_keys = {
        key
        for rank in ranks
        for key in (
            index_key(step, rank, generation),
            rank_commit_key(step, rank, generation),
        )
    }
    end_at = deadline if deadline is not None else (time.monotonic() + timeout)
    while True:
        if cancel_event is not None and cancel_event.is_set():
            return None
        keys = set(scoped_remote.list(prefix))
        if expected_keys.issubset(keys):
            return finalize_upload(
                scoped_remote,
                step,
                ranks=ranks,
                generation=generation,
                layout=layout,
                keep=keep_versions,
            )
        if time.monotonic() >= end_at:
            raise TimeoutError(
                f"step {step} ({generation.id}): 等待全 rank L2 齐备超时"
            )
        if cancel_event is not None:
            cancel_event.wait(poll_interval)
        else:
            time.sleep(poll_interval)


class CommitCoordinator:
    """嵌入式提交协调监控模块：串联开代、在途监控、异常全员作废与收尾收敛。"""

    def __init__(
        self,
        local_root: str | Path,
        remote: StorageBackend | None = None,
        *,
        run_id: str | None = None,
        collective: Collective | None = None,
        world_size: int | None = None,
        ranks: Sequence[int] | None = None,
        rank: int | None = None,
        pool: BufferPool | None = None,
        manager: SnapshotManager | None = None,
        max_inflight: int = 1,
        force_after_skips: int = 3,
        upload_timeout: float = 300.0,
        keep_versions: int | None = DEFAULT_KEEP_VERSIONS,
        keep_local: int = DEFAULT_KEEP_LOCAL_VERSIONS,
        collective_timeout: float = DEFAULT_TIMEOUT,
        poll_interval: float = 0.02,
    ) -> None:
        if upload_timeout <= 0:
            raise ValueError(f"upload_timeout must be > 0, got {upload_timeout}")
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if run_id is not None:
            validate_run_id(run_id)

        if collective is None:
            if world_size is not None and world_size > 1:
                raise ValueError(
                    f"world_size={world_size} 但没注入 collective：协调层需要集体通信"
                )
            collective = SingleRankCollective()
        elif world_size is not None and world_size != collective.world_size:
            raise ValueError(
                f"声明的 world_size={world_size} 与 collective.world_size={collective.world_size} 不一致"
            )

        self.collective: Collective = collective
        self.world_size = collective.world_size
        self.comm_rank = collective.rank
        self.rank = collective.rank if rank is None else int(rank)
        self.ranks: tuple[int, ...] = (
            tuple(int(r) for r in ranks)
            if ranks is not None
            else tuple(range(self.world_size))
        )
        self.run_id = run_id
        self.base_local_root = Path(local_root)
        self.local_root = local_run_root(local_root, run_id)
        self.remote = remote
        self.scoped_remote = (
            run_scope(remote, run_id) if remote is not None else None
        )
        self.upload_timeout = float(upload_timeout)
        self.keep_versions = keep_versions
        self.keep_local = int(keep_local)
        self.collective_timeout = float(collective_timeout)
        self.poll_interval = float(poll_interval)

        if pool is not None and manager is not None:
            raise ValueError("pool 与 manager 只能给一个")
        if manager is not None:
            self._manager = manager
            self._pool = manager._pool  # noqa: SLF001
        else:
            self._pool = pool if pool is not None else default_pool()
            self._manager = SnapshotManager(
                self._pool,
                max_inflight=max_inflight,
                force_after_skips=force_after_skips,
                collective=self.collective,
                collective_timeout=self.collective_timeout,
            )

        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"ckpt-coord-r{self.rank}"
        )
        self._inflight: list[_InflightTask] = []
        self._max_step: int | None = None
        self._invalidated: list[InvalidatedGeneration] = []
        self._converged_steps: list[int] = []
        self._closed = False

        # 同轮次启动时读取已有最高远端已提交 step，防止重启后回退/重复提交（C3.1）
        self._sync_initial_max_step()

    def _sync_initial_max_step(self) -> None:
        if self.scoped_remote is None:
            return
        try:
            steps = committed_remote_steps(self.scoped_remote)
        except (StorageError, FormatError):
            return
        if steps:
            self._max_step = max(steps)

    @property
    def manager(self) -> SnapshotManager:
        return self._manager

    @property
    def max_step(self) -> int | None:
        return self._max_step

    @property
    def invalidated_generations(self) -> tuple[InvalidatedGeneration, ...]:
        return tuple(self._invalidated)

    @property
    def converged_steps(self) -> tuple[int, ...]:
        return tuple(self._converged_steps)

    @property
    def inflight_steps(self) -> tuple[int, ...]:
        return tuple(
            task.step
            for task in self._inflight
            if not task.converged and not task.invalidated
        )

    # --- 恢复入口（C4 尾巴提升 + 集群恢复闭环） -----------------------------

    def recover(
        self,
        *,
        step: int | None = None,
        source_run_id: str | None = None,
        device: "str | Device" = "cpu",
        tp: int = 1,
        tp_rank: int = 0,
        shard_dims: Mapping[str, int] | None = None,
        budget_bytes: int = DEFAULT_BUDGET_BYTES,
        column_strategy: str = "whole",
        verify_crc: bool = True,
        crc_padding: str = "block",
        crc_padding_ratio: float = 0.125,
        io_concurrency: int = 1,
        local_staging_dir: str | Path | None = None,
        prefer_local: bool = True,
        fallback_on_local_error: bool = True,
        skip_corrupt_status: bool = False,
        allow_fallback: bool = False,
        download_retries: int = DEFAULT_DOWNLOAD_RETRIES,
        download_backoff: float = DEFAULT_DOWNLOAD_BACKOFF,
        promote_tails: bool = True,
    ) -> ObjectLoadReport:
        """集群启动恢复入口：rank 0 先提升尾巴（C4）→ 选版广播 → 热副本/下载 → 全归约复核。"""
        if self.remote is None:
            raise ValueError("recover requires remote storage")
        report = load_checkpoint_version(
            self.remote,
            local_root=self.base_local_root,
            step=step,
            device=device,
            tp=tp,
            tp_rank=tp_rank,
            shard_dims=shard_dims,
            budget_bytes=budget_bytes,
            column_strategy=column_strategy,
            verify_crc=verify_crc,
            crc_padding=crc_padding,
            crc_padding_ratio=crc_padding_ratio,
            io_concurrency=io_concurrency,
            local_staging_dir=local_staging_dir,
            prefer_local=prefer_local,
            fallback_on_local_error=fallback_on_local_error,
            skip_corrupt_status=skip_corrupt_status,
            allow_fallback=allow_fallback,
            collective=self.collective,
            download_retries=download_retries,
            download_backoff=download_backoff,
            run_id=self.run_id,
            source_run_id=source_run_id,
            promote_tails=promote_tails,
        )
        # 同一轮次恢复后刷新最大水位；跨轮次回滚（source_run_id != None）新轮次从头计水位
        if source_run_id is None:
            self._sync_initial_max_step()
            if self._max_step is None or report.step > self._max_step:
                self._max_step = report.step
        return report

    # --- 触发保存 + 搭便车监控 + 后台上传 -----------------------------------

    def save_step(
        self,
        tensors: Mapping[str, Tensor],
        *,
        step: int,
        blobs: Mapping[str, bytes] | None = None,
        layout: Mapping[str, int] | None = None,
        sources: list[ShardSource] | None = None,
        planner: str = "local",
        alignment: int = DEFAULT_ALIGNMENT,
        engine: str = "auto",
        direct_io: bool = True,
        crc_chunk_max: int = CRC_CHUNK_MAX,
        wait_upload: bool = False,
        _upload_fn: Callable[..., int] | None = None,
    ) -> CoordinatedSaveResult:
        """在保存触发点做搭便车全归约（C1）→ 统一开代（C2）→ 本地落盘 → 后台上传（C3）。"""
        if self._closed:
            raise CheckpointError("CommitCoordinator is already closed")
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        # C3.1：同轮次内 step 必须严格递增
        if self._max_step is not None and step <= self._max_step:
            raise PlanError(
                f"run {self.run_id or '<default>'}: step {step} 必须严格大于"
                f"当前轮次已提交/在途最大 step {self._max_step}"
            )

        import api as ckpt_api

        # 先准备本地保存计划与对齐大小，用于向 SnapshotManager 租块
        plan, aligned_size = ckpt_api.prepare_save_plan(
            tensors,
            blobs=blobs,
            rank=self.rank,
            sources=sources,
            planner=planner,
            alignment=alignment,
            collective=self.collective if planner == "gathered" else None,
        )
        trim = getattr(self._pool, "trim", None)
        if trim is not None:
            trim(keep_bytes=aligned_size)

        # C1：扩宽保存触发点全归约，搭便车携带上一代/在途代上传状态与异常
        local_status, failed_step_code = self._local_piggyback_state()
        trigger: Trigger = self._manager.trigger(
            step,
            nbytes=aligned_size,
            extra_reduce=[int(local_status), int(failed_step_code)],
        )
        newly_invalidated, newly_converged = self._reconcile_piggyback(
            trigger.reduced_extra
        )

        if trigger.decision is not TriggerDecision.START or trigger.snapshot is None:
            return CoordinatedSaveResult(
                step=step,
                decision=trigger.decision,
                reason=trigger.reason,
                invalidated=newly_invalidated,
                converged_steps=newly_converged,
            )

        snapshot: Snapshot = trigger.snapshot
        # 全体一致决定 START：推进当前轮次水位（即使本次后续上传失败也不重跑该旧 step）
        self._max_step = step

        try:
            generation = self._allocate_generation(step)
            local_report = ckpt_api.persist_snapshot(
                tensors,
                self.base_local_root,
                step=step,
                plan=plan,
                aligned_size=aligned_size,
                snapshot=snapshot,
                pool=self._pool,
                blobs=blobs,
                layout=layout,
                alignment=alignment,
                engine=engine,
                direct_io=direct_io,
                crc_chunk_max=crc_chunk_max,
                generation=generation.id,
                run_id=self.run_id,
            )
        except BaseException:
            # 异常若发生在 allocate_generation（persist_snapshot 内部自己会 fail snapshot）
            if not snapshot.is_terminal:
                snapshot.fail("generation allocation failed")
            raise

        if self.remote is None:
            return CoordinatedSaveResult(
                step=step,
                decision=TriggerDecision.START,
                generation=generation,
                local_report=local_report,
                invalidated=newly_invalidated,
                converged_steps=newly_converged,
            )

        task = _InflightTask(
            step=step,
            generation=generation,
            ranks=self.ranks,
            layout=dict(layout or {}),
            started_at=time.monotonic(),
        )
        self._inflight.append(task)
        task.future = self._executor.submit(
            self._run_upload_task, task, _upload_fn or upload_rank
        )

        if wait_upload:
            task.done_event.wait(timeout=self.upload_timeout)
            if task.error is not None:
                raise task.error
            if not task.done_event.is_set():
                raise TimeoutError(
                    f"step {step} ({generation.id}): 后台上传超时（{self.upload_timeout}s）"
                )

        return CoordinatedSaveResult(
            step=step,
            decision=TriggerDecision.START,
            generation=generation,
            local_report=local_report,
            invalidated=newly_invalidated,
            converged_steps=newly_converged,
        )

    def _local_piggyback_state(self) -> tuple[InflightStatus, int]:
        """汇总本 rank 当前在途上传状态及失败 step（供 MAX 归约）。"""
        now = time.monotonic()
        worst = InflightStatus.NONE
        failed_step_code = 0
        for task in self._inflight:
            if task.converged or task.invalidated:
                continue
            status = task.current_status(now=now, upload_timeout=self.upload_timeout)
            if status > worst:
                worst = status
            if status is InflightStatus.FAILED:
                if task.error is None:
                    task.error = TimeoutError(
                        f"step {task.step} ({task.generation.id}): 上传超时（{self.upload_timeout}s）"
                    )
                failed_step_code = max(failed_step_code, task.step + 1)
        return worst, failed_step_code

    def _reconcile_piggyback(
        self, reduced_extra: Sequence[int]
    ) -> tuple[tuple[InvalidatedGeneration, ...], tuple[int, ...]]:
        """根据归约结果在全体 rank 上同步宣告作废（C3.2）或收敛。"""
        if len(reduced_extra) < 2:
            return (), ()
        global_status = InflightStatus(int(reduced_extra[0]))
        global_failed_step = int(reduced_extra[1]) - 1
        invalidated: list[InvalidatedGeneration] = []
        converged: list[int] = []

        if global_status is InflightStatus.FAILED:
            # 任一 rank 上报某在途代上传失败或超时 → 全员统一宣告该代作废
            for task in list(self._inflight):
                if task.converged or task.invalidated:
                    continue
                if global_failed_step < 0 or task.step <= global_failed_step:
                    reason = (
                        f"{type(task.error).__name__}: {task.error}"
                        if task.error is not None
                        else "peer rank failed or timed out during background upload"
                    )
                    task.invalidated = True
                    task.cancel_event.set()
                    inv = InvalidatedGeneration(
                        step=task.step, generation=task.generation.id, reason=reason
                    )
                    self._invalidated.append(inv)
                    invalidated.append(inv)
                    self._inflight.remove(task)
            return tuple(invalidated), ()

        if global_status is InflightStatus.DONE:
            for task in list(self._inflight):
                if task.converged or task.invalidated:
                    continue
                if task.done_event.is_set() and task.error is None:
                    task.converged = True
                    self._converged_steps.append(task.step)
                    converged.append(task.step)
                    self._inflight.remove(task)

        return tuple(invalidated), tuple(converged)

    def _allocate_generation(self, step: int) -> Generation:
        """C2：裁决允许开新代后，由 rank 0 统一生成/抢占并广播 ``(step, seq, writer)``。"""
        if self.scoped_remote is not None:
            ensure = getattr(self.scoped_remote, "ensure_conditional_writes", None)
            if callable(ensure):
                ensure()
            if self.comm_rank == 0:
                mine = acquire_generation_for_step(
                    self.scoped_remote, step, ranks=self.ranks
                )
                seq, token = mine.seq, int(mine.writer, 16)
            else:
                mine, seq, token = None, 0, 0
        else:
            if self.comm_rank == 0:
                mine = Generation.new(step, 1, writer=new_writer())
                seq, token = mine.seq, int(mine.writer, 16)
            else:
                mine, seq, token = None, 0, 0

        seq, token = self.collective.all_reduce_max_int64(
            [seq, token], timeout=self.collective_timeout
        )
        if mine is not None:
            return mine
        return Generation(step=step, seq=int(seq), writer=format(int(token), "08x"))

    def _run_upload_task(
        self, task: _InflightTask, upload_fn: Callable[..., int]
    ) -> None:
        """后台上传线程：上传本 rank 分片（L2）；rank 0 见齐即写总提交点（L3）。"""
        assert self.remote is not None and self.scoped_remote is not None
        try:
            if task.cancel_event.is_set():
                return
            upload_fn(
                self.base_local_root,
                self.remote,
                step=task.step,
                rank=self.rank,
                generation=task.generation,
                run_id=self.run_id,
                cancel_event=task.cancel_event,
            )
            task.local_upload_done.set()
            if self.comm_rank == 0:
                self._wait_and_finalize_rank0(task)
        except BaseException as exc:
            task.error = exc
        finally:
            task.local_upload_done.set()
            task.done_event.set()

    def _wait_and_finalize_rank0(self, task: _InflightTask) -> UploadCommit | None:
        """rank 0 后台见所有 rank 分片提交点齐备即直接写远端总提交点，不等训练线程。"""
        assert self.scoped_remote is not None
        commit = wait_and_finalize_generation(
            self.scoped_remote,
            step=task.step,
            generation=task.generation,
            ranks=task.ranks,
            layout=task.layout,
            keep_versions=self.keep_versions,
            cancel_event=task.cancel_event,
            poll_interval=self.poll_interval,
            deadline=task.started_at + self.upload_timeout,
        )
        if commit is not None:
            task.committed_remote = True
        return commit

    # --- C4.2：退出等待点与末尾收敛 ----------------------------------------

    def finish(self, *, timeout: float | None = None) -> tuple[int, ...]:
        """训练退出/主动等待点：等待最后一代远端总提交收敛并做全员归约复核（C4.2）。

        各 rank 先等自身分片上传完成（``local_upload_done``），再通过 ``all_reduce(MAX)``
        轮询收敛：任一 rank 上传失败或超时，rank 0 立即中止等待其他 rank 的 L2，全员
        统一宣告该代作废并抛错退出。
        """
        wait_limit = self.upload_timeout if timeout is None else float(timeout)
        deadline = time.monotonic() + wait_limit

        # 1. 先等本 rank 自身上传完成（或超时），避免 rank 0 在 L2 等待里卡死全员
        for task in list(self._inflight):
            if task.converged or task.invalidated:
                continue
            remaining = max(0.0, deadline - time.monotonic())
            if not task.local_upload_done.wait(timeout=remaining):
                task.error = TimeoutError(
                    f"step {task.step} ({task.generation.id}): 退出收敛等待超时（{wait_limit}s）"
                )
                task.cancel_event.set()

        # 2. 轮询全归约：若任一 rank 失败立即全员作废退出；否则等 rank 0 完成 L3 提交
        all_converged: list[int] = []
        while True:
            local_status, failed_step_code = self._local_piggyback_state()
            if (
                local_status is InflightStatus.UPLOADING
                and time.monotonic() >= deadline
            ):
                for task in self._inflight:
                    if not task.converged and not task.invalidated and task.error is None:
                        task.error = TimeoutError(
                            f"step {task.step} ({task.generation.id}): 退出收敛等待超时（{wait_limit}s）"
                        )
                        task.cancel_event.set()
                local_status, failed_step_code = self._local_piggyback_state()

            reduced = self.collective.all_reduce_max_int64(
                [int(local_status), int(failed_step_code)],
                timeout=self.collective_timeout,
            )
            global_status = InflightStatus(int(reduced[0]))
            if global_status is InflightStatus.FAILED:
                errors = [
                    task.error
                    for task in self._inflight
                    if task.error is not None
                    and not task.converged
                    and not task.invalidated
                ]
                newly_invalidated, _ = self._reconcile_piggyback(reduced)
                if errors:
                    raise errors[0]
                detail = ", ".join(
                    f"step_{inv.step}({inv.generation}): {inv.reason}"
                    for inv in newly_invalidated
                )
                raise CheckpointError(
                    f"退出收敛失败，其他 rank 上传或提交异常，在途代际已作废：{detail}"
                )

            if global_status in (InflightStatus.NONE, InflightStatus.DONE):
                _, newly_converged = self._reconcile_piggyback(reduced)
                all_converged.extend(newly_converged)
                return tuple(all_converged)

            # 仍有 rank（通常是 rank 0 正在做 L3 finalize）处于 UPLOADING：短暂等待后再复核
            for task in self._inflight:
                if not task.converged and not task.invalidated and not task.done_event.is_set():
                    remaining = max(0.0, deadline - time.monotonic())
                    task.done_event.wait(timeout=min(self.poll_interval, remaining))
                    break

    def close(self, *, timeout: float | None = None, wait: bool = True) -> None:
        if self._closed:
            return
        try:
            if wait and self._inflight:
                self.finish(timeout=timeout)
        finally:
            self._closed = True
            for task in self._inflight:
                task.cancel_event.set()
            self._executor.shutdown(wait=True)

    def __enter__(self) -> "CommitCoordinator":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        # 异常退出时不强求 finish 掩盖原异常，仅安全停后台线程
        if exc_info[0] is not None:
            self.close(wait=False)
        else:
            self.close(wait=True)
        return False


__all__ = [
    "CommitCoordinator",
    "CoordinatedSaveResult",
    "InflightStatus",
    "InvalidatedGeneration",
    "acquire_generation_for_step",
    "wait_and_finalize_generation",
]
