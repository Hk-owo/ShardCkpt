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
4. **裁决必须全局唯一**：本 rank 的在途代数与池空闲块只是**输入**，裁决由一次
   ``all_reduce(MAX)`` 得出 —— 任一 rank 阻塞则全体 ``SKIP`` / ``FORCE_WAIT``。
   同一个 step 上「rank 0 跳过、rank 1 开写」会让版本残缺（各 rank 分片凑不齐），
   而一旦保存路径里接入 collective，一边进一边不进就是死锁。规格见 ``docs/gate.md``。

``_skipped_streak`` 的写点收敛在 ``trigger()`` 一处（提交成功只记一个待清零标志），
所有写都发生在集体通信的同侧，各 rank 的 streak 才会同步演进。

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


class GateBlock(enum.IntEnum):
    """门闸的**本地**阻塞原因。

    值就是优先级：归约取 ``MAX`` 天然得到「最严重的原因」，不需要额外的比较逻辑。
    """

    NONE = 0
    NO_FREE_BLOCK = 1
    INFLIGHT = 2

    def describe(self, *, max_inflight: int) -> str:
        """不依赖本地状态的通用描述（跨 rank 报原因时用）。"""
        if self is GateBlock.INFLIGHT:
            return f"在途代数已达上限 {max_inflight}，上一代还没归还 buffer"
        if self is GateBlock.NO_FREE_BLOCK:
            return "池里没有可用块"
        return "门闸通过"


_RANK_MASK = (1 << 32) - 1


def pack_block(block: GateBlock, *, rank: int) -> int:
    """打包成 ``(kind << 32) | rank``：高位给原因，低位给最严重的 rank（排障用）。"""
    if not 0 <= rank <= _RANK_MASK:
        raise ValueError(f"rank {rank} 超出打包范围 [0, {_RANK_MASK}]")
    return (int(block) << 32) | rank


def unpack_block(packed: int) -> tuple[GateBlock, int]:
    """解包 :func:`pack_block`；归约结果不是已知原因说明 collective 被写坏了。"""
    kind, rank = packed >> 32, packed & _RANK_MASK
    try:
        return GateBlock(kind), rank
    except ValueError as exc:
        raise SnapshotError(f"门闸归约返回未知的阻塞原因 {kind}") from exc


class GateCollective(Protocol):
    """门闸需要的通信面：一个 int64 的 ``all_reduce(MAX)``，带超时。

    实现**不许**在超时时降级成本地判据 —— 只能抛错让全体一致地放弃本次保存。
    真机后端见 ``dist/``（``dist.Collective`` 即满足本协议）。
    """

    def all_reduce_max_int64(
        self, values: list[int], *, timeout: float
    ) -> list[int]:
        """逐元素求全局最大值；入参与返回值等长同序。"""


class TriggerDecision(enum.Enum):
    """``trigger()`` 的三种结果，对应转移表里 IDLE 上的三条边。"""

    START = "start"
    SKIP = "skip"
    FORCE_WAIT = "force_wait"


def decide(
    block: GateBlock, *, streak: int, force_after_skips: int
) -> TriggerDecision:
    """门闸裁决（纯函数）：先跳过，连续跳过达到阈值才强制等待。

    入参必须是**全局**原因 —— 本函数不认识 rank，也不做任何本地判定。
    """
    if block is GateBlock.NONE:
        return TriggerDecision.START
    if streak >= force_after_skips:
        return TriggerDecision.FORCE_WAIT
    return TriggerDecision.SKIP


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

    多 rank 时必须注入 ``collective``：每次 ``trigger()`` 把本 rank 的阻塞原因打成一个
    ``int64`` 做 ``all_reduce(MAX)``，得到**全局唯一**的裁决 —— 任一 rank 阻塞则全体
    ``SKIP`` / ``FORCE_WAIT``。本地判据只是输入，所以「别的 rank 忙、我空闲」也不会
    单独开写。

    ``rank`` / ``world_size`` 取自 ``collective``（通信世界），与「本 rank 写哪些分片」
    无关 —— 后者是 plan 的事。``world_size`` 参数只是**显式声明**，用来在没给
    collective 的情况下拦住「多 rank 各判各的」：声明 > 1 却没人可通信，直接报错。
    """

    def __init__(
        self,
        pool: BufferPool,
        *,
        max_inflight: int = 1,
        force_after_skips: int = 3,
        collective: GateCollective | None = None,
        world_size: int | None = None,
        collective_timeout: float = 30.0,
    ) -> None:
        if max_inflight < 1:
            raise ValueError(f"max_inflight must be >= 1, got {max_inflight}")
        if force_after_skips < 1:
            raise ValueError(f"force_after_skips must be >= 1, got {force_after_skips}")
        if collective is not None:
            actual = getattr(collective, "world_size", 1)
            if world_size is not None and world_size != actual:
                raise ValueError(
                    f"声明的 world_size={world_size} 与 collective 的 {actual} 不一致"
                )
            world_size = actual
        elif world_size is None:
            world_size = 1
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if world_size > 1 and collective is None:
            raise ValueError(
                f"world_size={world_size} 但没注入 collective：门闸裁决会各 rank 各判各的，"
                "同一个 step 上必然出现有的 rank 写、有的跳过（版本残缺；有 collective 时"
                "还会死锁）"
            )
        self._pool = pool
        self._collective = collective
        self.max_inflight = max_inflight
        self.force_after_skips = force_after_skips
        self.rank = int(getattr(collective, "rank", 0))
        self.world_size = world_size
        self.collective_timeout = collective_timeout
        self._active: list[Snapshot] = []
        self._skipped_streak = 0
        self._pending_clear = False
        self.stats = MachineStats()

    # --- 查询 ---------------------------------------------------------------
    @property
    def active(self) -> tuple[Snapshot, ...]:
        return tuple(self._active)

    @property
    def skipped_streak(self) -> int:
        """连续跳过的次数；有代提交成功后归零。

        清零在**下一次 ``trigger()`` 入口**兑现（见 ``_consume_pending_clear``），
        这样 streak 的写点只有 ``trigger()`` 一处，各 rank 才同步演变。
        """
        return 0 if self._pending_clear else self._skipped_streak

    def current(self) -> Snapshot | None:
        return self._active[0] if self._active else None

    # --- 触发 ---------------------------------------------------------------
    def trigger(self, step: int, nbytes: int = 0) -> Trigger:
        self._consume_pending_clear()
        block, reported_by = self._resolve_block()
        decision = decide(
            block, streak=self.skipped_streak, force_after_skips=self.force_after_skips
        )
        if decision is not TriggerDecision.START:
            if decision is TriggerDecision.FORCE_WAIT:
                self.stats.forced += 1
            else:
                self._skipped_streak += 1
                self.stats.skipped += 1
            return Trigger(decision, reason=self._reason(block, reported_by))

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
    def _consume_pending_clear(self) -> None:
        """把「有代提交成功」记下的清零在 ``trigger()`` 入口兑现。"""
        if self._pending_clear:
            self._skipped_streak = 0
            self._pending_clear = False

    def _resolve_block(self) -> tuple[GateBlock, int]:
        """本地原因 → 全局原因（AND 语义：任一 rank 阻塞即全体阻塞）。

        返回 ``(全局原因, 报出最严重原因的那个 rank)``。归约失败直接抛错让全体放弃，
        不降级成本地判据 —— 降级就是各 rank 重新走上不同分支。
        """
        local = self._local_block()
        if self._collective is None:
            return local, self.rank
        packed = pack_block(local, rank=self.rank)
        result = self._collective.all_reduce_max_int64(
            [packed], timeout=self.collective_timeout
        )
        if len(result) != 1:
            raise SnapshotError(
                f"门闸归约返回 {len(result)} 个值，期望 1 个（检查 collective 实现）"
            )
        return unpack_block(int(result[0]))

    def _local_block(self) -> GateBlock:
        if len(self._active) >= self.max_inflight:
            return GateBlock.INFLIGHT
        if not self._pool.has_free():
            return GateBlock.NO_FREE_BLOCK
        return GateBlock.NONE

    def _reason(self, block: GateBlock, reported_by: int) -> str:
        if reported_by == self.rank:
            if block is GateBlock.INFLIGHT:
                return (
                    f"{len(self._active)} 代仍在途（上限 {self.max_inflight}），"
                    "上一代还没归还 buffer"
                )
            return block.describe(max_inflight=self.max_inflight)
        return (
            f"{block.describe(max_inflight=self.max_inflight)}"
            f"（最严重的 rank 是 {reported_by}）"
        )

    def _on_change(self, snapshot: Snapshot) -> None:
        if not snapshot.is_terminal:
            return
        if snapshot in self._active:
            self._active.remove(snapshot)
        if snapshot.state is SnapshotState.COMMITTED:
            self.stats.committed += 1
            # 只记标志，不在回调里改 streak：清零统一在 trigger() 入口兑现
            self._pending_clear = True
        elif snapshot.state is SnapshotState.ABORTED:
            self.stats.aborted += 1
        else:  # DROPPED
            self.stats.dropped += 1


__all__ = [
    "BufferPool",
    "GateBlock",
    "GateCollective",
    "MachineStats",
    "Snapshot",
    "SnapshotError",
    "SnapshotManager",
    "SnapshotState",
    "TERMINAL_STATES",
    "Trigger",
    "TriggerDecision",
    "decide",
    "pack_block",
    "unpack_block",
]
