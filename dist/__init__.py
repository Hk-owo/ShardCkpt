"""通信抽象：gather / broadcast / all-reduce，带超时。

第一个落地用例来自快照门闸（规格见 `docs/gate.md`）：每 step 一次
``all_reduce(MAX)``，归约值把「本地 blocked 原因 + rank」打包成 ``int64`` ——
任一 rank 阻塞则全体阻塞，裁决因此全局唯一。约定两条：

- 归约**不允许**在超时时降级成本地判据（那会让各 rank 重新走上不同的分支），
  只能抛 :class:`CollectiveError` 让全体一致地放弃本次保存；
- 集体通信不允许分支，调用点必须在所有 rank 上以相同顺序无条件执行。

本模块提供四类原语，语义各不同，别混用：

- ``all_reduce_max_int64``：门闸裁决。每 rank 提等长向量，全体拿到逐元素最大值。
- ``all_gather_objects``：**每 rank 各交一个对象，全员拿到按 rank 索引的全量列表**。
- ``scatter_objects``：**root 交一个长度 N 的列表，rank i 只拿到第 i 项**。用于
  「每个 rank 拿到的都不一样」的分发；用 ``broadcast_object`` 代替它会让每 rank
  收 N 份、丢掉 N-1 份，带宽放大 N 倍。
- ``broadcast_object``：root 交一个对象，**全员拿到同一份**。只用于「每人都要的
  全局视图」。

``ThreadedCollective`` 是**同进程多 rank** 的实现，给仿真与单元测试用；
真机后端（torch.distributed / NCCL）落地时实现同一个 ``Collective`` 协议即可。
对象交换与 all-reduce 用**独立的调用计数器**，所以两种原语交替调用不会错配；
但同一组 rank 的调用顺序仍然必须一致（真机同理）。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Protocol, Sequence

from core import CheckpointError

DEFAULT_TIMEOUT = 30.0


class CollectiveError(CheckpointError):
    """集体通信失败：超时、参与方数量不符、各 rank 入参形状不一致。"""


class Collective(Protocol):
    """门闸裁决 + 对象交换的完整协议；调用方只依赖这几个方法，方便注入替身。"""

    rank: int
    world_size: int

    def all_reduce_max_int64(
        self, values: list[int], *, timeout: float = DEFAULT_TIMEOUT
    ) -> list[int]:
        """逐元素求全局最大值，返回值与入参等长、顺序相同。"""

    def all_gather_objects(
        self, obj: Any, *, timeout: float = DEFAULT_TIMEOUT
    ) -> list[Any]:
        """本 rank 交一个对象，返回长度 ``world_size`` 的列表（索引即 rank）。"""

    def scatter_objects(
        self, objs: Sequence[Any] | None, *, timeout: float = DEFAULT_TIMEOUT
    ) -> Any:
        """root 交长度 ``world_size`` 的列表，返回本 rank 那**一项**。

        非 root 必须传 ``None``（分支在调用前就固定，不依赖通信结果）。
        """

    def broadcast_object(
        self, obj: Any, *, root: int = 0, timeout: float = DEFAULT_TIMEOUT
    ) -> Any:
        """root 交一个对象，全体拿到同一个；非 root 传 ``None``。"""


class SingleRankCollective:
    """单 rank：三类通信全部退化成恒等，给 ``world_size=1`` 保留同一条代码路径。"""

    rank = 0
    world_size = 1

    def all_reduce_max_int64(
        self, values: list[int], *, timeout: float = DEFAULT_TIMEOUT
    ) -> list[int]:
        return list(values)

    def all_gather_objects(
        self, obj: Any, *, timeout: float = DEFAULT_TIMEOUT
    ) -> list[Any]:
        return [obj]

    def scatter_objects(
        self, objs: Sequence[Any] | None, *, timeout: float = DEFAULT_TIMEOUT
    ) -> Any:
        if objs is None:
            raise CollectiveError("scatter_objects: root 必须提供 objs 列表")
        items = list(objs)
        if len(items) != 1:
            raise CollectiveError(
                f"scatter_objects: 单 rank 需要长度 1 的列表，得到 {len(items)}"
            )
        return items[0]

    def broadcast_object(
        self, obj: Any, *, root: int = 0, timeout: float = DEFAULT_TIMEOUT
    ) -> Any:
        if root != 0:
            raise CollectiveError(f"broadcast_object: 单 rank 的 root 只能是 0，得到 {root}")
        return obj


class _SharedLink:
    """同进程多 rank 共享的「全连接」。

    每一轮通信按**调用序号**分代（每个 rank 的第 k 次调用互相配对）：先读完
    的 rank 可以直接进第 k+1 次，不会撞进第 k 代还没回收的槽位。最后离开的 rank
    负责回收那一代，所以槽位不会无限增长。

    all-reduce 与对象交换各有**独立的计数器与槽位表**：两者交替调用不会错配，
    但也意味着「第 k 代」这个序号在两张表里互相独立。
    """

    def __init__(self, world_size: int) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        self.world_size = world_size
        self.cond = threading.Condition()
        self._slots: dict[int, list[list[int] | None]] = {}
        self._arrived: dict[int, int] = {}
        self._departed: dict[int, int] = {}
        self._obj_slots: dict[int, list[object]] = {}
        self._obj_arrived: dict[int, int] = {}
        self._obj_departed: dict[int, int] = {}


class ThreadedCollective:
    """同进程多 rank 的 collective：每个 rank 一个实例，共享一条 ``_SharedLink``。

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
        self._obj_call = 0

    @classmethod
    def linked(cls, world_size: int) -> tuple["ThreadedCollective", ...]:
        """一次造出同一组的全部 rank 句柄（顺序即 rank 顺序）。"""
        link = _SharedLink(world_size)
        return tuple(cls(rank, world_size, link) for rank in range(world_size))

    def _object_exchange(self, payload: object, *, timeout: float) -> list[object]:
        """把 ``payload`` 放进本代的槽位，等全员到齐后返回按 rank 索引的列表。

        三个对象原语共用这一个栅栏，差别只在「谁交什么、取哪一项」。
        """
        link = self._link
        key = self._obj_call
        with link.cond:
            if key not in link._obj_slots:
                link._obj_slots[key] = [None] * self.world_size
                link._obj_arrived[key] = 0
                link._obj_departed[key] = 0
            slots = link._obj_slots[key]
            slots[self.rank] = payload
            link._obj_arrived[key] += 1
            link.cond.notify_all()

            deadline = time.monotonic() + timeout
            while link._obj_arrived[key] < self.world_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CollectiveError(
                        f"对象交换超时：{link._obj_arrived[key]}/{self.world_size} 个 rank "
                        f"到达（{timeout}s）"
                    )
                link.cond.wait(remaining)

            out = list(slots)

            link._obj_departed[key] += 1
            if link._obj_departed[key] == self.world_size:
                del link._obj_slots[key]
                del link._obj_arrived[key]
                del link._obj_departed[key]
                link.cond.notify_all()

        self._obj_call += 1
        return out

    def all_gather_objects(
        self, obj: Any, *, timeout: float = DEFAULT_TIMEOUT
    ) -> list[Any]:
        return self._object_exchange(obj, timeout=timeout)

    def scatter_objects(
        self, objs: Sequence[Any] | None, *, timeout: float = DEFAULT_TIMEOUT
    ) -> Any:
        if self.rank == 0:
            if objs is None:
                raise CollectiveError("scatter_objects: root 必须提供 objs 列表")
            payload: object = list(objs)
            if len(payload) != self.world_size:
                raise CollectiveError(
                    f"scatter_objects: root 提供 {len(payload)} 项，"
                    f"期望 world_size={self.world_size}"
                )
        else:
            if objs is not None:
                raise CollectiveError(
                    f"scatter_objects: 只有 root 提供 objs，rank {self.rank} 不该传"
                )
            payload = None

        items = self._object_exchange(payload, timeout=timeout)[0]
        if not isinstance(items, (list, tuple)) or len(items) != self.world_size:
            raise CollectiveError(
                f"scatter_objects: root 的分发列表不合法（{type(items).__name__}）"
            )
        return items[self.rank]

    def broadcast_object(
        self, obj: Any, *, root: int = 0, timeout: float = DEFAULT_TIMEOUT
    ) -> Any:
        if not 0 <= root < self.world_size:
            raise CollectiveError(
                f"broadcast_object: root {root} 越界（world_size {self.world_size}）"
            )
        payload = obj if self.rank == root else None
        return self._object_exchange(payload, timeout=timeout)[root]

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
