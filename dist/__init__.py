"""通信抽象：gather / broadcast / all-reduce，带超时。

第一个落地用例来自快照门闸（规格见 `docs/gate.md`）：每 step 一次
``all_reduce(MAX)``，归约值把「本地 blocked 原因 + rank」打包成 ``int64`` ——
任一 rank 阻塞则全体阻塞，裁决因此全局唯一。约定两条：

- 归约**不允许**在超时时降级成本地判据（那会让各 rank 重新走上不同的分支），
  只能抛 :class:`CollectiveError` 让全体一致地放弃本次保存；
- 集体通信不允许分支，调用点必须在所有 rank 上以相同顺序无条件执行。

本模块只有 ``all_reduce_max_int64``：门闸是唯一需要通信的判据，先把这一条做扎实。
``ThreadedCollective`` 是**同进程多 rank** 的实现，给仿真与单元测试用；
真机后端（torch.distributed / NCCL）落地时实现同一个 ``Collective`` 协议即可。
"""

from __future__ import annotations

import threading
import time
from typing import Protocol

from core import CheckpointError

DEFAULT_TIMEOUT = 30.0


class CollectiveError(CheckpointError):
    """集体通信失败：超时、参与方数量不符、各 rank 入参形状不一致。"""


class Collective(Protocol):
    """一次 all-reduce 的最小协议。调用方只依赖这一个方法，方便注入替身。"""

    rank: int
    world_size: int

    def all_reduce_max_int64(
        self, values: list[int], *, timeout: float = DEFAULT_TIMEOUT
    ) -> list[int]:
        """逐元素求全局最大值，返回值与入参等长、顺序相同。"""


class SingleRankCollective:
    """单 rank：all-reduce 就是恒等。给 ``world_size=1`` 保留同一条代码路径。"""

    rank = 0
    world_size = 1

    def all_reduce_max_int64(
        self, values: list[int], *, timeout: float = DEFAULT_TIMEOUT
    ) -> list[int]:
        return list(values)


class _SharedLink:
    """同进程多 rank 共享的「全连接」。

    每一轮 all-reduce 按**调用序号**分代（每个 rank 的第 k 次调用互相配对）：先读完
    的 rank 可以直接进第 k+1 次，不会撞进第 k 代还没回收的槽位。最后离开的 rank
    负责回收那一代，所以槽位不会无限增长。
    """

    def __init__(self, world_size: int) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        self.world_size = world_size
        self.cond = threading.Condition()
        self._slots: dict[int, list[list[int] | None]] = {}
        self._arrived: dict[int, int] = {}
        self._departed: dict[int, int] = {}


class ThreadedCollective:
    """同进程多 rank 的 ``all_reduce(MAX)``：每个 rank 一个实例，共享一条 ``_SharedLink``。

    **只用于测试与同进程仿真**：真机跨进程要用通信库后端。语义与真 collective 对齐：
    所有参与方到齐才返回，超时抛 :class:`CollectiveError`（不做任何降级）。
    超时之后这一组 rank 的调用序号已经错位，只能整体放弃 —— 与真机一致。
    """

    def __init__(self, rank: int, world_size: int, link: _SharedLink) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} out of range for world_size {world_size}")
        self.rank = rank
        self.world_size = world_size
        self._link = link
        self._call = 0

    @classmethod
    def linked(cls, world_size: int) -> tuple["ThreadedCollective", ...]:
        """一次造出同一组的全部 rank 句柄（顺序即 rank 顺序）。"""
        link = _SharedLink(world_size)
        return tuple(cls(rank, world_size, link) for rank in range(world_size))

    def all_reduce_max_int64(
        self, values: list[int], *, timeout: float = DEFAULT_TIMEOUT
    ) -> list[int]:
        width = len(values)
        link = self._link
        key = self._call
        with link.cond:
            if key not in link._slots:
                link._slots[key] = [None] * self.world_size
                link._arrived[key] = 0
                link._departed[key] = 0
            slots = link._slots[key]
            slots[self.rank] = list(values)
            link._arrived[key] += 1
            link.cond.notify_all()

            deadline = time.monotonic() + timeout
            while link._arrived[key] < self.world_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CollectiveError(
                        f"all-reduce 超时：{link._arrived[key]}/{self.world_size} 个 rank "
                        f"到达（{timeout}s）"
                    )
                link.cond.wait(remaining)

            widths = {len(slot) for slot in slots if slot is not None}
            if widths != {width}:
                raise CollectiveError(
                    f"各 rank 入参宽度不一致：本 rank {width}，全局 {sorted(widths)}"
                )
            out = [max(slot[i] for slot in slots) for i in range(width)]

            link._departed[key] += 1
            if link._departed[key] == self.world_size:
                del link._slots[key]
                del link._arrived[key]
                del link._departed[key]
                link.cond.notify_all()

        self._call += 1
        return out


__all__ = [
    "Collective",
    "CollectiveError",
    "DEFAULT_TIMEOUT",
    "SingleRankCollective",
    "ThreadedCollective",
]
