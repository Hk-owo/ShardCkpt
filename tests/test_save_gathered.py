"""gathered 模式的写闭环：全员上报 → rank 0 统一分配 → scatter → 各 rank 落盘。

锁住的不变量：

- **布局零变化**：gathered 与 local 两条路径产出的 ``.metadata`` **逐字节相同**
  （偏移、文件、CRC 全都一致），所以打开 gathered 不会让任何一份 checkpoint 变样；
- **去重仍然全局唯一**：Replicate 副本里只有一个 rank 真的写；
- **空计划是合法的**：被去重跳过的 rank 也提交一份空版本（空索引 + COMMIT），
  「每个 rank 在 ``step_N`` 都有一个已提交版本」这条不变量对所有 rank 一致成立；
- **scatter 的结果要能被本地校验**：rank 0 若把本 rank 没有的张量分下来，D2H
  之前就报错，不留半份。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

import api as ckpt
import dist
from async_ import PinnedBufferPool, SnapshotState, aligned_allocator
from commit.local import COMMIT_FILE, read_commit, version_key
from core import PlanError
from core.index import META_FILE, CheckpointIndex, TensorMeta
from core.region import Region
from planner.save_plan import ShardSource, build_save_plan, default_writer

STEP = 42
ALIGN = 4096


def make_pool() -> PinnedBufferPool:
    return PinnedBufferPool(
        chunk_bytes=1 << 20,
        min_chunks=1,
        max_bytes=1 << 20,
        timeout=0,
        allocator=aligned_allocator(),
    )


def replicated_sources(tensors, *, replica_index: int, n_replicas: int):
    """把每个张量整份标成 Replicate 副本 ``replica_index``。"""
    return [
        replace(source, replica_index=replica_index, n_replicas=n_replicas)
        for source in ckpt._sources_from_tensors(tensors)
    ]


def run_ranks(tensors_of, sources_of, roots, *, world: int):
    """并发跑 ``world`` 个 rank；返回 ``{rank: SaveReport | BaseException}``。"""
    links = dist.ThreadedCollective.linked(world)
    out: dict[int, object] = {}

    def worker(rank: int) -> None:
        try:
            out[rank] = ckpt.save_local(
                tensors_of(rank),
                roots[rank],
                step=STEP,
                rank=rank,
                sources=sources_of(rank),
                planner="gathered",
                alignment=ALIGN,
                direct_io=False,
                pool=make_pool(),
                collective=links[rank],
            )
        except BaseException as exc:  # 让断言看得到异常类型与消息
            out[rank] = exc

    with ThreadPoolExecutor(max_workers=world) as executor:
        futures = [executor.submit(worker, rank) for rank in range(world)]
        for future in futures:
            future.result(timeout=30)
    return out


def merge_index(reports) -> CheckpointIndex:
    """把几个 rank 的局部索引拼成全局的（与 upload.py 读侧合并同一套约定）。"""
    shards: dict[str, list] = {}
    metas: dict[str, TensorMeta] = {}
    for report in reports:
        for fqn, meta in report.index.tensors.items():
            shards.setdefault(fqn, []).extend(meta.shards)
            metas.setdefault(fqn, meta)
    return CheckpointIndex(
        tensors={
            fqn: TensorMeta(
                global_shape=metas[fqn].global_shape,
                dtype=metas[fqn].dtype,
                shards=tuple(entries),
            )
            for fqn, entries in shards.items()
        }
    )


# --- 端到端 ----------------------------------------------------------------


def test_gathered_writes_each_replica_once(dataset: Path, tmp_path: Path) -> None:
    """DP=2 副本：全局恰好一份，两个 rank 都提交，合并后覆盖完整。"""
    roots = {rank: tmp_path / f"rank{rank}" for rank in range(2)}
    out = run_ranks(
        lambda rank: ckpt.load(dataset, device="cuda:0").tensors,
        lambda rank: replicated_sources(
            ckpt.load(dataset, device="cuda:0").tensors,
            replica_index=rank,
            n_replicas=2,
        ),
        roots,
        world=2,
    )
    for rank in (0, 1):
        assert not isinstance(out[rank], BaseException), out[rank]
        assert out[rank].state is SnapshotState.COMMITTED
        assert (roots[rank] / version_key(STEP) / COMMIT_FILE).is_file()

    merged = merge_index([out[0], out[1]])
    assert merged.validate() is None  # complete=True：全局覆盖不缺
    for name, meta in merged.tensors.items():
        assert meta.covered_numel == meta.numel, name

    # 每个张量恰好一份分片：去重在全局是唯一的
    assert all(len(meta.shards) == 1 for meta in merged.tensors.values())


def test_gathered_matches_local_layout_byte_for_byte(
    dataset: Path, tmp_path: Path
) -> None:
    """核心性质：gathered 与 local 产出的索引**逐字节相同**。

    比较 ``COMMIT`` 里记下的索引 CRC-32C —— 它覆盖索引的全部字节（偏移、文件、
    覆盖范围、块 CRC 一起），所以相等就意味着布局完全没变。
    """
    sources = replicated_sources(
        ckpt.load(dataset, device="cuda:0").tensors, replica_index=0, n_replicas=2
    )
    local_root = tmp_path / "local"
    local_report = ckpt.save_local(
        ckpt.load(dataset, device="cuda:0").tensors,
        local_root,
        step=STEP,
        rank=0,
        sources=sources,
        alignment=ALIGN,
        direct_io=False,
        pool=make_pool(),
    )

    gathered_root = tmp_path / "gathered"
    out = run_ranks(
        lambda rank: ckpt.load(dataset, device="cuda:0").tensors,
        lambda rank: replicated_sources(
            ckpt.load(dataset, device="cuda:0").tensors,
            replica_index=rank,
            n_replicas=2,
        ),
        {rank: gathered_root / f"rank{rank}" for rank in range(2)},
        world=2,
    )
    gathered_report = out[0]
    assert not isinstance(gathered_report, BaseException), gathered_report

    assert gathered_report.file == local_report.file
    assert gathered_report.size == local_report.size
    assert gathered_report.data_bytes == local_report.data_bytes
    assert gathered_report.commit.metadata_bytes == local_report.commit.metadata_bytes
    assert gathered_report.commit.metadata_crc32c == local_report.commit.metadata_crc32c


# --- 空计划 ----------------------------------------------------------------


def _single_tensor_state(dataset: Path):
    """只留一个张量，并把名字改成 ``w``（``crc32("w") % 2 == 0``，owner 恒为 rank 0）。"""
    tensors = ckpt.load(dataset, device="cuda:0").tensors
    _, tensor = next(iter(tensors.items()))
    return {"w": tensor}


def test_skipped_rank_commits_an_empty_version(dataset: Path, tmp_path: Path) -> None:
    """全被去重跳过的 rank：空计划合法，照样留下 COMMIT（空索引、无数据文件）。"""
    state = _single_tensor_state(dataset)
    assert default_writer("w", 2) == 0  # 前提：owner 落在 rank 0

    roots = {rank: tmp_path / f"rank{rank}" for rank in range(2)}
    out = run_ranks(
        lambda rank: state,
        lambda rank: replicated_sources(state, replica_index=rank, n_replicas=2),
        roots,
        world=2,
    )
    winner, skipped = out[0], out[1]
    assert not isinstance(skipped, BaseException), skipped
    assert winner.size > 0
    assert skipped.size == 0
    assert skipped.data_bytes == 0
    assert skipped.index.tensors == {}
    assert dict(skipped.commit.files) == {}

    # 两层提交点都在，恢复端看得到这一版
    assert (roots[1] / version_key(STEP) / COMMIT_FILE).is_file()
    assert (roots[1] / version_key(STEP) / META_FILE).is_file()
    assert read_commit(ckpt.LocalStorage(roots[1]), STEP).metadata_bytes > 0


# --- 拒绝：配置错误在碰数据之前失败 ----------------------------------------


def test_missing_collective_is_rejected(dataset: Path, tmp_path: Path) -> None:
    tensors = ckpt.load(dataset, device="cuda:0").tensors
    with pytest.raises(ValueError, match="需要 collective"):
        ckpt.save_local(
            tensors, tmp_path, step=STEP, planner="gathered", rank=0, direct_io=False
        )


def test_rank_must_match_collective_rank(dataset: Path, tmp_path: Path) -> None:
    tensors = ckpt.load(dataset, device="cuda:0").tensors
    links = dist.ThreadedCollective.linked(2)
    with pytest.raises(ValueError, match="不一致"):
        ckpt.save_local(
            tensors,
            tmp_path,
            step=STEP,
            rank=1,  # 与 links[0].rank 不符
            planner="gathered",
            collective=links[0],
            direct_io=False,
        )


def test_collective_without_object_primitives_is_rejected(
    dataset: Path, tmp_path: Path
) -> None:
    class GateOnly:
        """只有门闸所需的那一个方法：gathered 必须拒绝，不能静默走 local。"""

        rank = 0
        world_size = 1

        def all_reduce_max_int64(self, values, *, timeout=None):  # noqa: ANN001
            return list(values)

    tensors = ckpt.load(dataset, device="cuda:0").tensors
    with pytest.raises(TypeError, match="all_gather_objects"):
        ckpt.save_local(
            tensors,
            tmp_path,
            step=STEP,
            rank=0,
            planner="gathered",
            collective=GateOnly(),
            direct_io=False,
        )


def test_assignment_for_tensor_missing_locally_is_rejected() -> None:
    """rank 0 把本 rank 没有的张量分下来：在 D2H 之前报错，不写半份。

    这一条用单元测试而不是端到端：集体通信里任一方在 scatter 之后抛错，其余 rank
    会卡在下一步的门闸上（真机同样如此）。校验函数本身是纯的，直接测它更准。
    """
    tensors = {"w": object()}
    source = ShardSource(
        fqn="ghost",
        global_shape=(8, 8),
        dtype="f32",
        region=Region((0, 0), (8, 8)),
    )
    plan = build_save_plan([source], rank=0)
    with pytest.raises(PlanError, match="本地没有"):
        ckpt._check_assignment(
            plan, rank=0, tensors=tensors, blob_names=set()
        )


def test_assignment_with_wrong_rank_is_rejected() -> None:
    source = ShardSource(
        fqn="w",
        global_shape=(8, 8),
        dtype="f32",
        region=Region((0, 0), (8, 8)),
    )
    plan = build_save_plan([source], rank=0)
    with pytest.raises(PlanError, match="本 rank 是 1"):
        ckpt._check_assignment(
            plan, rank=1, tensors={"w": object()}, blob_names=set()
        )


def test_assignment_for_blob_missing_locally_is_rejected() -> None:
    source = ShardSource(
        fqn="w",
        global_shape=(8, 8),
        dtype="f32",
        region=Region((0, 0), (8, 8)),
    )
    plan = build_save_plan([source], rank=0, blobs={"step": 8})
    with pytest.raises(PlanError, match="小状态"):
        ckpt._check_assignment(
            plan, rank=0, tensors={"w": object()}, blob_names=set()
        )
