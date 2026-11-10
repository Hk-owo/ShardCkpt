"""快照状态机：一个 step 一代（per-generation），非法转移直接抛错。

设计文档 §4 只画了主干，实现还需要三件它没写的东西：

1. ``SNAPSHOTTING`` 失败不进 ``ABORTED``：此时存储侧一个字节都没写，终态是
   ``DROPPED``（等价于 IDLE 的无副作用终态），清理逻辑不得处理它。否则清理会去删
   一个不存在的目录，还会让"跳过 / 强制写"的判定看到错误的在途状态。
2. buffer 的归还点是「数据写完或放弃」，不是「D2H 完成」：``COMMITTED`` /
   ``ABORTED`` / ``DROPPED`` 三个终态各自归还一次，且只归还一次。
3. 门闸看的是「上一代是否还占着 buffer」，不是「D2H 是否完成」：``max_inflight=1``
   是单缓冲，``2`` 即双缓冲；越线时先 ``SKIP``（训练不等），连续跳过达到阈值才
   ``FORCE_WAIT``（训练阻塞到上一代结束）。

池只被用到 ``acquire`` / ``release`` / ``has_free`` 三个动作，buffer 对状态机而言是
不透明句柄，所以本模块不依赖 GPU，在 mock 环境下可以完整测试。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Protocol

from core import CheckpointError


class SnapshotError(CheckpointError):
    """状态机上的非法操作：不可转移、重复清理、门闸不满足等。"""


class SnapshotState(enum.Enum):
    """一代 checkpoint 的生命周期，与设计文档 §4 的状态机一一对应。"""

    SNAPSHOTTING = "snapshotting"
    PERSISTING = "persisting"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ABORTED = "aborted"
    DROPPED = "dropped"


# 合法转移表：写死在这里，所有事件方法都过一遍校验。
_TRANSITIONS: dict[SnapshotState, frozenset[SnapshotState]] = {
    SnapshotState.SNAPSHOTTING: frozenset(
        {SnapshotState.PERSISTING, SnapshotState.DROPPED}
    ),
    SnapshotState.PERSISTING: frozenset({SnapshotState.COMMITTING, SnapshotState.ABORTED}),
    SnapshotState.COMMITTING: frozenset({SnapshotState.COMMITTED, SnapshotState.ABORTED}),
    SnapshotState.COMMITTED: frozenset(),
    SnapshotState.ABORTED: frozenset(),
    SnapshotState.DROPPED: frozenset(),
}

TERMINAL_STATES = frozenset(
    {SnapshotState.COMMITTED, SnapshotState.ABORTED, SnapshotState.DROPPED}
)


class BufferPool(Protocol):
    """状态机需要的池接口。真实实现是 pinned buffer 池。"""

    def acquire(self, nbytes: int) -> object:
        """借一块至少 nbytes 的 buffer；没有可用块时抛错，不阻塞。

        实现需要保证 ``has_free()`` 到 ``acquire()`` 之间不被别的线程抢占，
        否则门闸判定会与实际分配不一致。
        """

    def release(self, buffer: object) -> None:
        """归还 acquire 得到的 buffer。"""

    def has_free(self) -> bool:
        """是否还有可用块（门闸判定用，不需要真的分配）。"""


@dataclass(eq=False)
class Snapshot:
    """一代快照。``state`` 只能通过下面的方法推进，非法转移抛 ``SnapshotError``。

    ``pool`` / ``on_change`` 由 :class:`SnapshotManager` 注入，调用方不用管。
    """

    step: int
    nbytes: int
    buffer: object
    pool: BufferPool = field(repr=False)
    on_change: Callable[["Snapshot"], None] = field(repr=False)
    state: SnapshotState = SnapshotState.SNAPSHOTTING
    reason: str = ""
    cleaned: bool = False
    released: bool = field(default=False, repr=False)

    # --- 状态查询 -----------------------------------------------------------
    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def holds_buffer(self) -> bool:
        """buffer 是否还被这一代占着（归还后为 False）。"""
        return not self.released

    # --- 事件 ---------------------------------------------------------------
    def d2h_done(self) -> None:
        """D2H 完成（``copy_done`` event 已完成）：host 数据从此不可变。"""
        self._move(SnapshotState.PERSISTING)

    def persist_ok(self) -> None:
        """数据写完 + fsync + 校验和算出，可以进入提交阶段。"""
        self._move(SnapshotState.COMMITTING)

    def commit_ok(self) -> None:
        """``.metadata`` → ``COMMIT`` → ``latest`` 都已落盘。"""
        self._move(SnapshotState.COMMITTED)

    def fail(self, reason: str) -> None:
        """失败或超时。

        ``SNAPSHOTTING`` 阶段失败没有任何存储侧副作用，进 ``DROPPED``；
        ``PERSISTING`` / ``COMMITTING`` 阶段失败说明目录里有残缺数据，进 ``ABORTED``
        等清理。两种终态都会归还 buffer。
        """
        target = (
            SnapshotState.DROPPED
            if self.state is SnapshotState.SNAPSHOTTING
            else SnapshotState.ABORTED
        )
        self._move(target, reason=reason)

    def mark_cleaned(self) -> None:
        """存储层清理完 ``ABORTED`` 的目录后调用；不改变状态，只记一笔。"""
        if self.state is not SnapshotState.ABORTED:
            raise SnapshotError(
                f"step {self.step}: 只有 ABORTED 的目录需要清理，当前 {self.state.value}"
            )
        if self.cleaned:
            raise SnapshotError(f"step {self.step}: 已经清理过")
        self.cleaned = True

    # --- 内部 ---------------------------------------------------------------
    def _move(self, target: SnapshotState, *, reason: str = "") -> None:
        if target not in _TRANSITIONS[self.state]:
            raise SnapshotError(
                f"step {self.step}: {self.state.value} -> {target.value} 不是合法转移"
            )
        self.state = target
        if reason:
            self.reason = reason
        if target in TERMINAL_STATES:
            self._release()
        self.on_change(self)

    def _release(self) -> None:
        if self.released:
            return
        self.pool.release(self.buffer)
        self.released = True


class TriggerDecision(enum.Enum):
    """``trigger()`` 的三种结果，对应转移表里 IDLE 上的三条边。"""

    START = "start"
    SKIP = "skip"
    FORCE_WAIT = "force_wait"


@dataclass(frozen=True)
class Trigger:
    """一次触发的结果。``snapshot`` 只在 ``START`` 时非空。"""

    decision: TriggerDecision
    snapshot: Snapshot | None = None
    reason: str = ""


@dataclass
class MachineStats:
    started: int = 0
    committed: int = 0
    aborted: int = 0
    dropped: int = 0
    skipped: int = 0
    forced: int = 0

    def reset(self) -> None:
        self.started = 0
        self.committed = 0
        self.aborted = 0
        self.dropped = 0
        self.skipped = 0
        self.forced = 0


class SnapshotManager:
    """多代门闸 + 状态机入口。

    ``max_inflight`` 是在途代数上限：1 为单缓冲（只有上一代归还后下一代才能开始），
    2 即双缓冲。连续跳过 ``force_after_skips`` 次之后，触发返回 ``FORCE_WAIT``，
    由调用方阻塞到上一代结束后重新触发（对应设计文档「距上次成功超过阈值就阻塞训练」）。
    """

    def __init__(
        self,
        pool: BufferPool,
        *,
        max_inflight: int = 1,
        force_after_skips: int = 3,
    ) -> None:
        if max_inflight < 1:
            raise ValueError(f"max_inflight must be >= 1, got {max_inflight}")
        if force_after_skips < 1:
            raise ValueError(f"force_after_skips must be >= 1, got {force_after_skips}")
        self._pool = pool
        self.max_inflight = max_inflight
        self.force_after_skips = force_after_skips
        self._active: list[Snapshot] = []
        self._skipped_streak = 0
        self.stats = MachineStats()

    # --- 查询 ---------------------------------------------------------------
    @property
    def active(self) -> tuple[Snapshot, ...]:
        return tuple(self._active)

    @property
    def skipped_streak(self) -> int:
        """连续跳过的次数；成功提交后归零。"""
        return self._skipped_streak

    def current(self) -> Snapshot | None:
        return self._active[0] if self._active else None

    # --- 触发 ---------------------------------------------------------------
    def trigger(self, step: int, nbytes: int = 0) -> Trigger:
        blocked = self._blocked_reason()
        if blocked:
            if self._skipped_streak >= self.force_after_skips:
                self.stats.forced += 1
                return Trigger(TriggerDecision.FORCE_WAIT, reason=blocked)
            self._skipped_streak += 1
            self.stats.skipped += 1
            return Trigger(TriggerDecision.SKIP, reason=blocked)

        snapshot = Snapshot(
            step=step,
            nbytes=nbytes,
            buffer=self._pool.acquire(nbytes),
            pool=self._pool,
            on_change=self._on_change,
        )
        self._active.append(snapshot)
        self.stats.started += 1
        return Trigger(TriggerDecision.START, snapshot=snapshot)

    # --- 内部 ---------------------------------------------------------------
    def _blocked_reason(self) -> str:
        if len(self._active) >= self.max_inflight:
            return (
                f"{len(self._active)} 代仍在途（上限 {self.max_inflight}），"
                "上一代还没归还 buffer"
            )
        if not self._pool.has_free():
            return "池里没有可用块"
        return ""

    def _on_change(self, snapshot: Snapshot) -> None:
        if not snapshot.is_terminal:
            return
        if snapshot in self._active:
            self._active.remove(snapshot)
        if snapshot.state is SnapshotState.COMMITTED:
            self.stats.committed += 1
            self._skipped_streak = 0
        elif snapshot.state is SnapshotState.ABORTED:
            self.stats.aborted += 1
        else:  # DROPPED
            self.stats.dropped += 1


__all__ = [
    "BufferPool",
    "MachineStats",
    "Snapshot",
    "SnapshotError",
    "SnapshotManager",
    "SnapshotState",
    "TERMINAL_STATES",
    "Trigger",
    "TriggerDecision",
]
