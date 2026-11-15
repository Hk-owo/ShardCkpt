"""dist 的对象原语：gather / scatter / broadcast 的语义，以及「不降级」。

这里锁住的不变量（用户可观察的那一层）：

- ``all_gather_objects`` 全员拿到按 rank 索引的全量列表；
- ``scatter_objects`` 每 rank **只**拿到自己那一项（不是每人一份全量）；
- ``broadcast_object`` 全员拿到同一份，``root`` 可指定；
- 形状错、越界、缺席都抛 :class:`dist.CollectiveError`，**不返回部分视图**；
- 对象原语与 ``all_reduce_max_int64`` 用独立计数器，交替调用不会错配。

``scatter`` 与 ``broadcast`` 必须分得清：把按 rank 不同的数据用 broadcast 发，
每 rank 会收 N 份、丢掉 N-1 份，带宽放大 N 倍 —— 所以两者的语义在这里分开测。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

import dist


def run_ranks(count: int, body):
    """并发跑 ``count`` 个 rank，返回每个 rank 的返回值（列表，索引即 rank）。"""
    links = dist.ThreadedCollective.linked(count)
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(body, links, rank) for rank in range(count)]
        return [future.result(timeout=30) for future in futures]


# --- SingleRankCollective：同一条代码路径的恒等退化 -------------------------


def test_single_rank_identity() -> None:
    link = dist.SingleRankCollective()
    assert link.all_gather_objects("x") == ["x"]
    assert link.scatter_objects(["only"]) == "only"
    assert link.broadcast_object("whole") == "whole"


def test_single_rank_rejects_bad_shapes() -> None:
    link = dist.SingleRankCollective()
    with pytest.raises(dist.CollectiveError):
        link.scatter_objects(None)
    with pytest.raises(dist.CollectiveError):
        link.scatter_objects(["a", "b"])
    with pytest.raises(dist.CollectiveError):
        link.broadcast_object("x", root=1)


# --- all_gather_objects ----------------------------------------------------


def test_all_gather_returns_full_list_by_rank() -> None:
    def body(links, rank):
        return links[rank].all_gather_objects({"rank": rank})

    got = run_ranks(3, body)
    expected = [{"rank": 0}, {"rank": 1}, {"rank": 2}]
    assert got == [expected, expected, expected]


def test_all_gather_accepts_none_payload() -> None:
    def body(links, rank):
        return links[rank].all_gather_objects(None if rank == 1 else rank)

    got = run_ranks(3, body)
    assert got[0] == [0, None, 2]


# --- scatter_objects -------------------------------------------------------


def test_scatter_gives_each_rank_only_its_own_item() -> None:
    payload = ["a", "b", "c"]

    def body(links, rank):
        return links[rank].scatter_objects(payload if rank == 0 else None)

    assert run_ranks(3, body) == ["a", "b", "c"]


def test_scatter_rejects_wrong_list_length() -> None:
    """root 的列表长度不等于 world_size 时全员一致失败，不静默截断。"""
    links = dist.ThreadedCollective.linked(3)
    with pytest.raises(dist.CollectiveError, match="期望 world_size=3"):
        links[0].scatter_objects(["a", "b"])


def test_scatter_rejects_payload_from_non_root() -> None:
    links = dist.ThreadedCollective.linked(2)
    with pytest.raises(dist.CollectiveError, match="只有 root"):
        links[1].scatter_objects(["a", "b"])


# --- broadcast_object ------------------------------------------------------


def test_broadcast_from_rank_zero() -> None:
    def body(links, rank):
        return links[rank].broadcast_object({"shared": True} if rank == 0 else None)

    assert run_ranks(3, body) == [{"shared": True}] * 3


def test_broadcast_from_non_zero_root() -> None:
    def body(links, rank):
        return links[rank].broadcast_object("from-1" if rank == 1 else None, root=1)

    assert run_ranks(3, body) == ["from-1"] * 3


def test_broadcast_rejects_out_of_range_root() -> None:
    links = dist.ThreadedCollective.linked(2)
    with pytest.raises(dist.CollectiveError, match="越界"):
        links[0].broadcast_object("x", root=2)


# --- 不降级 ----------------------------------------------------------------


def test_missing_rank_raises_instead_of_returning_partial() -> None:
    """只有一个 rank 参加时，超时抛错；不返回「半份全量」。"""
    links = dist.ThreadedCollective.linked(2)
    with pytest.raises(dist.CollectiveError, match="对象交换超时"):
        links[0].all_gather_objects("lonely", timeout=0.05)


def test_object_timeout_does_not_advance_all_reduce_generation() -> None:
    """对象交换超时不影响 all-reduce 的调用序号：两张表互相独立。

    超时之后这一组 rank 已经不可用（真机同样），但 all-reduce 那一侧不该被
    「对象交换失败」额外推进一步。
    """
    links = dist.ThreadedCollective.linked(2)
    assert links[0]._call == 0
    assert links[0]._obj_call == 0
    with pytest.raises(dist.CollectiveError):
        links[0].all_gather_objects("x", timeout=0.05)
    assert links[0]._obj_call == 0  # 失败不推进对象代
    assert links[0]._call == 0  # 也不推进 all-reduce 代


# --- 交替调用不错配 --------------------------------------------------------


def test_alternating_primitives_do_not_cross_pairs() -> None:
    """同一个 rank 交替调 all-reduce 与对象原语，配对必须各在各的通道里。"""

    def body(links, rank):
        link = links[rank]
        reduce_1 = link.all_reduce_max_int64([rank])
        gathered = link.all_gather_objects(f"obj{rank}")
        reduce_2 = link.all_reduce_max_int64([rank * 10])
        scattered = link.scatter_objects(
            [f"slot{i}" for i in range(3)] if rank == 0 else None
        )
        broadcast = link.broadcast_object("whole" if rank == 1 else None, root=1)
        return reduce_1, gathered, reduce_2, scattered, broadcast

    got = run_ranks(3, body)
    for rank, (reduce_1, gathered, reduce_2, scattered, broadcast) in enumerate(got):
        assert reduce_1 == [2]  # 第一次归约：max(0,1,2)
        assert gathered == ["obj0", "obj1", "obj2"]  # 第二次是对象全量
        assert reduce_2 == [20]  # 第三次归约没被对象交换挤到别的代：max(0,10,20)
        assert scattered == f"slot{rank}"
        assert broadcast == "whole"
