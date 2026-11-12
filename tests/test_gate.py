"""门闸的跨 rank 一致化：裁决由一次 ``all_reduce(MAX)`` 统一得出。

这里锁住的不变量（用户可观察的那一层）：

- **任一 rank 阻塞 ⇒ 全体不写**：本地空闲的 rank 也不许单独开新一代，否则版本残缺；
- **各 rank 的裁决序列逐 step 相等**，``MachineStats.skipped/forced`` 也相等；
- **``FORCE_WAIT`` 之后的重试仍在 collective 内**：每轮重试都发生一次归约；
- **归约失败不降级**：超时抛错、本代不产出任何提交点，而不是各自猜一个裁决。

并发形状与真机一致：每个 rank 一个线程，``ThreadedCollective`` 自己就是栅栏 ——
不到齐不返回，所以两个线程的 ``trigger()`` 天然成对。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

import api as ckpt
import dist
import sim
from async_ import (
    GateBlock,
    SnapshotManager,
    TriggerDecision,
    decide,
    pack_block,
    unpack_block,
)
from commit.local import committed_steps
from storage.local import LocalStorage

START = TriggerDecision.START
SKIP = TriggerDecision.SKIP
FORCE_WAIT = TriggerDecision.FORCE_WAIT

DEVICE = "cuda:0"


class GatePool:
    """最小 buffer 池：``free`` 可以中途改，用来模拟「上一代归还了」。"""

    def __init__(self, *, free: int = 4) -> None:
        self.free = free
        self.acquired: list[object] = []
        self.released: list[object] = []

    def has_free(self) -> bool:
        return self.free > 0

    def acquire(self, nbytes: int) -> object:
        assert self.has_free(), "门闸判过 has_free 才该 acquire"
        self.free -= 1
        token = object()
        self.acquired.append(token)
        return token

    def release(self, buffer: object) -> None:
        self.released.append(buffer)
        self.free += 1


def make_ranks(
    world_size: int,
    *,
    max_inflight: int = 1,
    force_after_skips: int = 3,
    free: list[int] | None = None,
    collective_timeout: float = 10.0,
) -> tuple[list[SnapshotManager], list[GatePool], tuple[dist.ThreadedCollective, ...]]:
    """一套同组 rank：各自一个池、一个 manager，共享一条 ``ThreadedCollective``。"""
    links = dist.ThreadedCollective.linked(world_size)
    pools = [GatePool(free=(free[r] if free else 4)) for r in range(world_size)]
    managers = [
        SnapshotManager(
            pools[r],
            max_inflight=max_inflight,
            force_after_skips=force_after_skips,
            collective=links[r],
            collective_timeout=collective_timeout,
        )
        for r in range(world_size)
    ]
    return managers, pools, links


def run_ranks(managers, steps, *, before_trigger=None):
    """并发跑多 rank：每个 rank 对每个 step 调一次 ``trigger()``，返回各自的 Trigger。"""
    traced: dict[int, list] = {}

    def worker(rank: int) -> None:
        seen = []
        for step in steps:
            if before_trigger is not None:
                before_trigger(rank, step)
            seen.append(managers[rank].trigger(step))
        traced[rank] = seen

    with ThreadPoolExecutor(max_workers=len(managers)) as pool:
        futures = [pool.submit(worker, rank) for rank in range(len(managers))]
        for future in futures:
            future.result(timeout=30)
    return traced


def commit(snapshot) -> None:
    snapshot.d2h_done()
    snapshot.persist_ok()
    snapshot.commit_ok()


class _Counted:
    """给 collective 套一层计数，用来证明每轮重试都真的归约过。"""

    def __init__(self, link, calls: list[int]) -> None:
        self._link = link
        self._calls = calls
        self.rank = link.rank
        self.world_size = link.world_size

    def all_reduce_max_int64(self, values: list[int], *, timeout: float = 10.0) -> list[int]:
        self._calls.append(self.rank)
        return self._link.all_reduce_max_int64(values, timeout=timeout)


# --- 纯函数层 --------------------------------------------------------------


def test_pack_keeps_kind_and_rank_apart() -> None:
    packed = pack_block(GateBlock.INFLIGHT, rank=3)
    assert unpack_block(packed) == (GateBlock.INFLIGHT, 3)
    # 优先级：INFLIGHT 的原因压过 NO_FREE_BLOCK，同原因时最大 rank 胜出
    assert max(packed, pack_block(GateBlock.NO_FREE_BLOCK, rank=3)) == packed
    assert max(packed, pack_block(GateBlock.INFLIGHT, rank=1)) == packed


def test_unpack_rejects_unknown_kind() -> None:
    with pytest.raises(Exception, match="未知的阻塞原因"):
        unpack_block(99 << 32)


def test_decide_is_the_whole_gate_rule() -> None:
    assert decide(GateBlock.NONE, streak=9, force_after_skips=3) is START
    assert decide(GateBlock.INFLIGHT, streak=0, force_after_skips=3) is SKIP
    assert decide(GateBlock.INFLIGHT, streak=2, force_after_skips=3) is SKIP
    assert decide(GateBlock.NO_FREE_BLOCK, streak=3, force_after_skips=3) is FORCE_WAIT


def test_multi_rank_without_collective_is_rejected() -> None:
    """多 rank 不给 collective 就等于各判各的：构造期直接拒绝，不留「看起来能跑」。"""
    with pytest.raises(ValueError, match="collective"):
        SnapshotManager(GatePool(), world_size=2)
    # collective 与显式声明对不上：拒绝，不猜哪个是真的
    with pytest.raises(ValueError, match="不一致"):
        SnapshotManager(
            GatePool(), collective=dist.SingleRankCollective(), world_size=2
        )


# --- 跨 rank 一致 ----------------------------------------------------------


def test_every_rank_starts_when_all_are_free() -> None:
    managers, pools, _ = make_ranks(2)
    traced = run_ranks(managers, [1])

    assert [t.decision for t in traced[0]] == [START]
    assert [t.decision for t in traced[1]] == [START]
    # 每个 rank 用自己的块，不是共享一块
    assert managers[0].current().buffer is not managers[1].current().buffer
    assert pools[0].acquired == [managers[0].current().buffer]
    assert pools[1].acquired == [managers[1].current().buffer]
    assert managers[0].stats.started == managers[1].stats.started == 1


def test_busy_rank_blocks_everyone() -> None:
    """rank 1 池里没块：rank 0 明明能开，也必须一起跳过，且一个字节都不分配。"""
    managers, pools, _ = make_ranks(2, free=[4, 0])
    traced = run_ranks(managers, [1])

    assert [t.decision for t in traced[0]] == [SKIP]
    assert [t.decision for t in traced[1]] == [SKIP]
    assert pools[0].acquired == [], "空闲 rank 不许单独开新一代"
    assert "没有可用块" in traced[0][0].reason
    assert "最严重的 rank 是 1" in traced[0][0].reason, "原因要指到最严重的那个 rank"
    assert managers[0].stats.skipped == managers[1].stats.skipped == 1


def test_rank_with_inflight_generation_holds_back_the_others() -> None:
    """rank 0 上一代还没归还 buffer，rank 1 本地空闲 —— 裁决仍然是全体跳过。"""
    managers, _pools, _ = make_ranks(2)
    run_ranks(managers, [1])
    commit(managers[1].current())  # 只有 rank 1 推进到终态，rank 0 还占着 buffer

    traced = run_ranks(managers, [2])

    assert [t.decision for t in traced[0]] == [SKIP]
    assert [t.decision for t in traced[1]] == [SKIP]
    assert managers[1].current() is None, "rank 1 自己没开新一代"
    assert managers[0].skipped_streak == managers[1].skipped_streak == 1
    assert "buffer" in traced[1][0].reason


def test_decision_sequence_identical_across_ranks() -> None:
    """连续阻塞：SKIP 累到阈值后一起 FORCE_WAIT，各 rank 序列逐 step 相等。"""
    managers, pools, _ = make_ranks(2, free=[4, 0], force_after_skips=2)
    traced = run_ranks(managers, [1, 2, 3, 4])

    expected = [SKIP, SKIP, FORCE_WAIT, FORCE_WAIT]
    assert [t.decision for t in traced[0]] == expected
    assert [t.decision for t in traced[1]] == expected
    for manager in managers:
        assert (manager.stats.skipped, manager.stats.forced) == (2, 2)
    assert pools[0].acquired == []


def test_force_wait_retry_still_goes_through_the_collective() -> None:
    """FORCE_WAIT 之后的每一轮重试都要重新归约，否则各 rank 会走出不同的步数。"""
    calls: list[int] = []
    links = [_Counted(link, calls) for link in dist.ThreadedCollective.linked(2)]
    pools = [GatePool(free=4), GatePool(free=0)]
    managers = [
        SnapshotManager(pools[r], force_after_skips=2, collective=links[r])
        for r in range(2)
    ]

    def heal(rank: int, step: int) -> None:
        # 第 4 轮之前「上一代结束」：rank 1 的块回来了
        if step == 4:
            pools[1].free = 4

    traced = run_ranks(managers, [1, 2, 3, 4], before_trigger=heal)

    assert [t.decision for t in traced[0]] == [SKIP, SKIP, FORCE_WAIT, START]
    assert [t.decision for t in traced[1]] == [SKIP, SKIP, FORCE_WAIT, START]
    assert calls.count(0) == calls.count(1) == 4, "4 轮都应该归约过（含 FORCE_WAIT 那轮）"


def test_collective_timeout_abandons_the_save() -> None:
    """归约超时：抛错、不降级、不分配 buffer，这一代什么都没发生。"""
    link = dist.ThreadedCollective.linked(2)[0]
    pool = GatePool()
    manager = SnapshotManager(
        pool, collective=link, collective_timeout=0.05
    )

    with pytest.raises(dist.CollectiveError, match="超时"):
        manager.trigger(1)

    assert manager.active == ()
    assert manager.stats.started == 0
    assert pool.acquired == []


def test_single_rank_collective_matches_no_collective() -> None:
    """单 rank 注入 collective 后行为不变：同一条代码路径，不是两套门闸。"""

    def sequence(manager: SnapshotManager) -> list[TriggerDecision]:
        first = manager.trigger(1)
        first.snapshot.d2h_done()
        decisions = [first.decision, manager.trigger(2).decision]
        first.snapshot.persist_ok()
        first.snapshot.commit_ok()
        decisions.append(manager.trigger(3).decision)
        return decisions

    plain = sequence(SnapshotManager(GatePool()))
    with_collective = sequence(
        SnapshotManager(GatePool(), collective=dist.SingleRankCollective())
    )
    assert plain == with_collective == [START, SKIP, START]


# --- api 层：多 rank 保存必须全家一致 -------------------------------------


def two_rank_dims(state) -> dict[str, int]:
    return sim.column_dims(state)


def test_api_rejects_multi_rank_without_collective(tmp_path) -> None:
    state = sim.make_state(step=0)
    tensors = sim.tensors_for_rank(state, rank=0, tp=2, dims=two_rank_dims(state))

    with pytest.raises(ValueError, match="collective"):
        ckpt.save_local(tensors, tmp_path / "ckpt", step=3, world_size=2)


def test_api_rejects_manager_together_with_collective(tmp_path) -> None:
    state = sim.make_state(step=0)
    tensors = sim.tensors_for_rank(state, rank=0, tp=2, dims=two_rank_dims(state))
    manager = SnapshotManager(GatePool())

    with pytest.raises(ValueError, match="只能给一个"):
        ckpt.save_local(
            tensors,
            tmp_path / "ckpt",
            step=3,
            manager=manager,
            collective=dist.SingleRankCollective(),
        )


def test_two_ranks_save_concurrently_over_one_collective(tmp_path) -> None:
    """两个 rank 各写自己的分片、共用一条 collective：两边都必须提交成功。"""
    state = sim.make_state(step=0)
    dims = two_rank_dims(state)
    links = dist.ThreadedCollective.linked(2)
    reports: dict[int, ckpt.SaveReport] = {}

    def save(rank: int) -> None:
        root = tmp_path / f"rank_{rank}"
        tensors = sim.tensors_for_rank(state, rank=rank, tp=2, dims=dims, device=DEVICE)
        sources = sim.sources_for_rank(state, rank=rank, tp=2, dims=dims)
        reports[rank] = ckpt.save_local(
            tensors,
            root,
            step=3,
            rank=rank,
            sources=sources,
            collective=links[rank],
            world_size=2,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(save, rank) for rank in range(2)]
        for future in futures:
            future.result(timeout=60)

    assert {rank: report.state.value for rank, report in reports.items()} == {
        0: "committed",
        1: "committed",
    }
    for rank in range(2):
        assert committed_steps(LocalStorage(tmp_path / f"rank_{rank}")) == [3]


def test_one_busy_rank_leaves_no_half_written_version(tmp_path) -> None:
    """rank 1 拿不到 buffer：全体放弃，两个 rank 都不留 ``step_3/``。"""
    state = sim.make_state(step=0)
    dims = two_rank_dims(state)
    links = dist.ThreadedCollective.linked(2)
    errors: dict[int, BaseException] = {}

    def save(rank: int) -> None:
        root = tmp_path / f"rank_{rank}"
        tensors = sim.tensors_for_rank(state, rank=rank, tp=2, dims=dims, device=DEVICE)
        sources = sim.sources_for_rank(state, rank=rank, tp=2, dims=dims)
        try:
            ckpt.save_local(
                tensors,
                root,
                step=3,
                rank=rank,
                sources=sources,
                pool=GatePool(free=4 if rank == 0 else 0),
                collective=links[rank],
                world_size=2,
            )
        except BaseException as exc:  # noqa: BLE001 - 记录到断言里
            errors[rank] = exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(save, rank) for rank in range(2)]
        for future in futures:
            future.result(timeout=60)

    assert set(errors) == {0, 1}, "空闲的 rank 0 也必须一起放弃"
    for error in errors.values():
        assert "pinned buffer" in str(error)
    for rank in range(2):
        assert committed_steps(LocalStorage(tmp_path / f"rank_{rank}")) == []
