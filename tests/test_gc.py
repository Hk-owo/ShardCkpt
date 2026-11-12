"""GC 判据的纯函数测试：不碰 IO、不看时间、不看进程状态。

判据只有一条不变量（step 单调推进）推导出的两条规则：

1. 本 rank 的分片提交点（L2）在远端 ⇒ 本地 ``step_N/`` 是冗余副本；
2. 存在更新的版本提交点（L3）⇒ 这一代是残片。

反过来「最新且未完成」一律保留，任何一条都不需要「写入者是否活着」的外部判断。
"""

from __future__ import annotations

import time

import pytest

import api as ckpt
import sim
from commit.gc import (
    COMMITTED,
    LOCAL,
    NOT_UPLOADED,
    NO_NEWER_VERSION,
    REMOTE,
    SUPERSEDED,
    UPLOADED,
    GcWorker,
    collect,
)
from commit.gc import plan
from commit.local import version_key
from storage.base import StorageBackend, StorageError
from storage.local import LocalStorage

DEVICE = "cuda:0"


def test_plan_is_a_pure_function() -> None:
    """同样的输入给出同样的决策与理由，且不需要任何外部状态。"""
    kwargs = dict(
        versions=[1, 2],
        rank_commits=[(2, 0)],
        prefixes=[1, 2, 3],
        local_steps=[2, 3],
        rank=0,
    )
    first = plan(**kwargs)
    second = plan(**kwargs)

    assert first == second
    assert first.reasons() == {
        (LOCAL, 2): UPLOADED,
    }
    assert first.kept == (
        (LOCAL, 3, NOT_UPLOADED),
        (REMOTE, 1, COMMITTED),
        (REMOTE, 2, COMMITTED),
        (REMOTE, 3, NO_NEWER_VERSION),
    )


def test_uploaded_version_is_reclaimed() -> None:
    """本 rank 的 L2 一出现，本地这一版就是冗余副本。"""
    result = plan(
        versions=[7],
        rank_commits=[(7, 3)],
        prefixes=[7],
        local_steps=[7],
        rank=3,
    )

    assert result.steps_for(LOCAL) == (7,)
    assert result.reasons()[(LOCAL, 7)] == UPLOADED


def test_other_ranks_commit_does_not_free_local() -> None:
    """别的 rank 传完了不代表我这份传完了：本地判据只看自己的分片提交点。"""
    result = plan(
        versions=[7],
        rank_commits=[(7, 0), (7, 1)],
        prefixes=[7],
        local_steps=[7],
        rank=3,
    )

    assert result.steps_for(LOCAL) == ()
    assert result.kept == ((LOCAL, 7, NOT_UPLOADED), (REMOTE, 7, COMMITTED))


def test_newest_uncommitted_step_is_kept() -> None:
    """最新一代且没有分片提交点：可能是正在写，一条判据都不命中。"""
    result = plan(
        versions=[6],
        rank_commits=[(6, 0)],
        prefixes=[6, 7],
        local_steps=[7],
        rank=0,
    )

    assert result.steps_for(LOCAL) == ()
    assert result.steps_for(REMOTE) == ()
    assert result.kept == (
        (LOCAL, 7, NOT_UPLOADED),
        (REMOTE, 6, COMMITTED),
        (REMOTE, 7, NO_NEWER_VERSION),
    )


def test_superseded_version_is_reclaimed() -> None:
    """存在更新的已提交版本：本地这一代（哪怕没传成功）也要收掉。"""
    result = plan(
        versions=[9],
        rank_commits=[(9, 0)],
        prefixes=[8, 9],
        local_steps=[8, 9],
        rank=0,
    )

    # 远端 8 也没有版本提交点、又被 9 越过：本地与远端两侧都收掉
    assert result.steps_for(LOCAL) == (8, 9)
    assert result.reasons() == {
        (LOCAL, 8): SUPERSEDED,
        (LOCAL, 9): UPLOADED,
        (REMOTE, 8): SUPERSEDED,
    }
    assert result.kept == ((REMOTE, 9, COMMITTED),)


def test_stale_remote_prefix_is_reclaimed() -> None:
    """远端残片：没有版本提交点，且已经有更新的一代提交过。"""
    result = plan(
        versions=[12],
        rank_commits=[(12, 0)],
        prefixes=[11, 12],
        local_steps=[],
        rank=0,
    )

    assert result.steps_for(REMOTE) == (11,)
    assert result.reasons()[(REMOTE, 11)] == SUPERSEDED
    assert result.kept == ((REMOTE, 12, COMMITTED),)


def test_remote_newest_prefix_is_kept() -> None:
    """远端最新且没有版本提交点：可能正在传，也可能 rank 0 崩在最后一步。"""
    result = plan(
        versions=[12],
        rank_commits=[(12, 0)],
        prefixes=[12, 13],
        local_steps=[],
        rank=0,
    )

    assert result.steps_for(REMOTE) == ()
    assert result.kept == ((REMOTE, 12, COMMITTED), (REMOTE, 13, NO_NEWER_VERSION))


def test_committed_remote_version_is_never_removed() -> None:
    """残片判据永不删已提交版本：有界保留是另一条判据（``plan_retention``）。"""
    result = plan(
        versions=[1, 2, 3, 4],
        rank_commits=[(4, 0)],
        prefixes=[1, 2, 3, 4],
        local_steps=[],
        rank=0,
    )

    assert result.candidates == ()
    assert result.steps_for(REMOTE) == ()


def test_nothing_committed_keeps_everything() -> None:
    """远端一个版本提交点都没有（首次保存还没提交）：什么都别删。"""
    result = plan(
        versions=[],
        rank_commits=[],
        prefixes=[1],
        local_steps=[1],
        rank=0,
    )

    assert result.candidates == ()
    assert result.kept == ((LOCAL, 1, NOT_UPLOADED), (REMOTE, 1, NO_NEWER_VERSION))
    assert result.versions == frozenset()


# --- 执行器 ---------------------------------------------------------------


class _BrokenRemote(StorageBackend):
    """远端整体不可达：任何操作都抛 ``StorageError``。"""

    def write(self, key: str, data: bytes) -> None:
        raise StorageError("remote is down")

    def read(self, key: str) -> bytes:
        raise StorageError("remote is down")

    def read_range(self, key: str, offset: int, length: int) -> bytes:
        raise StorageError("remote is down")

    def rename_or_put(self, src: str, dst: str) -> None:
        raise StorageError("remote is down")

    def list(self, prefix: str = "") -> list[str]:
        raise StorageError("remote is down")

    def delete(self, key: str) -> None:
        raise StorageError("remote is down")


def _save(state, local, step: int) -> None:
    ckpt.save_local(sim.to_device(state.step_once(), device=DEVICE), local, step=step)


def _upload(local, remote, step: int, rank: int = 0) -> None:
    ckpt.upload_local(local, remote, step=step, rank=rank)


def _finalize(remote, step: int, ranks=(0,)) -> None:
    ckpt.finalize_upload(remote, step=step, ranks=list(ranks), layout={"tp": 1})


def test_dry_run_only_plans_then_real_run_deletes(state, tmp_path) -> None:
    """``dry_run`` 只给计划；真跑之后本地目录消失，远端分片提交点不动。"""
    local = tmp_path / "local"
    remote_root = tmp_path / "remote"
    remote = LocalStorage(remote_root)

    _save(state, local, 1)
    _upload(local, remote, 1)
    _finalize(remote, 1)

    plan_only = collect(local, remote, rank=0, dry_run=True)
    assert plan_only.steps_for(LOCAL) == ()
    assert plan_only.planned[0].location == LOCAL
    assert plan_only.planned[0].reason == UPLOADED
    assert (local / version_key(1)).is_dir(), "dry_run 不能动文件"

    report = collect(local, remote, rank=0, dry_run=False)
    assert report.steps_for(LOCAL) == (1,)
    assert report.reclaimed_bytes > 0
    assert not (local / version_key(1)).exists()
    assert (remote_root / version_key(1) / "COMMIT").is_file(), "远端已提交版本永不删"


def test_upload_incomplete_keeps_local(state, tmp_path) -> None:
    """没上传完（远端没有本 rank 的分片提交点）：本地是唯一副本，必须留着。"""
    local = tmp_path / "local"
    remote = LocalStorage(tmp_path / "remote")

    _save(state, local, 1)

    report = collect(local, remote, rank=0, dry_run=False)

    assert report.deleted == ()
    assert report.skipped == ((LOCAL, 1, NOT_UPLOADED),)
    assert (local / version_key(1) / "COMMIT").is_file()


def test_other_rank_upload_does_not_release_this_one(state, tmp_path) -> None:
    """远端有 rank 1 的分片提交点，不代表本 rank（0）那份传完了。"""
    local = tmp_path / "local"
    remote = LocalStorage(tmp_path / "remote")

    _save(state, local, 1)
    _upload(local, remote, 1, rank=1)

    report = collect(local, remote, rank=0, dry_run=False)

    assert report.deleted == ()
    assert (local / version_key(1)).is_dir()


def test_superseded_version_is_reclaimed_on_disk(state, tmp_path) -> None:
    """第 1 代没传成功、第 2 代已经提交：本地 1 是确定的残片。"""
    local = tmp_path / "local"
    remote = LocalStorage(tmp_path / "remote")

    _save(state, local, 1)          # 假设上传失败 / 被放弃
    _save(state, local, 2)
    _upload(local, remote, 2)
    _finalize(remote, 2)

    report = collect(local, remote, rank=0, dry_run=False)

    assert sorted(report.steps_for(LOCAL)) == [1, 2]
    assert {item.step: item.reason for item in report.deleted} == {
        1: SUPERSEDED,
        2: UPLOADED,
    }
    assert list(local.glob("step_*")) == []


def test_remote_unreachable_deletes_nothing(state, tmp_path) -> None:
    """远端不可达：「查不到」不能当成「可以删」，本地一个都不动。"""
    local = tmp_path / "local"

    _save(state, local, 1)

    report = collect(local, _BrokenRemote(), rank=0, dry_run=False)

    assert report.aborted is not None and "unreachable" in report.aborted
    assert report.deleted == ()
    assert (local / version_key(1)).is_dir()


def test_crash_residue_is_cleaned_once_superseded(state, tmp_path) -> None:
    """崩溃留下的无 COMMIT 残目录：等有更新的一代提交后自动回收。"""
    local = tmp_path / "local"
    remote = LocalStorage(tmp_path / "remote")

    residue = local / version_key(1)
    residue.mkdir(parents=True)
    (residue / "__0_0.distcp").write_bytes(b"x" * 4096)

    _save(state, local, 2)
    _upload(local, remote, 2)
    _finalize(remote, 2)

    report = collect(local, remote, rank=0, dry_run=False)

    assert 1 in report.steps_for(LOCAL)
    assert not residue.exists()


def test_gc_is_idempotent_after_interrupt(state, tmp_path) -> None:
    """删到一半被打断（只剩数据文件、提交点已没）：重跑收敛且不抛错。"""
    local = tmp_path / "local"
    remote_root = tmp_path / "remote"
    remote = LocalStorage(remote_root)

    _save(state, local, 1)
    _upload(local, remote, 1)
    _finalize(remote, 1)
    _save(state, local, 2)
    _upload(local, remote, 2)
    _finalize(remote, 2)

    # 打断的现场：本地 step 1 的 COMMIT 已删，数据还在
    (local / version_key(1) / "COMMIT").unlink()
    assert (local / version_key(1) / "__0_0.distcp").is_file()

    first = collect(local, remote, rank=0, dry_run=False)
    second = collect(local, remote, rank=0, dry_run=False)

    assert sorted(first.steps_for(LOCAL)) == [1, 2]
    assert second.deleted == (), "第二次没有东西可删"
    assert [p.name for p in local.glob("step_*")] == []


def test_stale_remote_prefix_is_reclaimed_on_disk(state, tmp_path) -> None:
    """远端半途的上传前缀（没有版本提交点）+ 有更新的一代 ⇒ 整个前缀删掉。"""
    local = tmp_path / "local"
    remote_root = tmp_path / "remote"
    remote = LocalStorage(remote_root)

    stale = remote_root / version_key(1) / "rank_0"
    stale.mkdir(parents=True)
    (stale / "__0_0.distcp").write_bytes(b"x" * 4096)

    _save(state, local, 2)
    _upload(local, remote, 2)
    _finalize(remote, 2)

    report = collect(local, remote, rank=0, dry_run=False)

    assert report.steps_for(REMOTE) == (1,)
    assert not (remote_root / version_key(1)).exists()
    assert (remote_root / version_key(2) / "COMMIT").is_file()


# --- 周期线程 -------------------------------------------------------------


def _wait(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_worker_cleans_on_start_and_stops(state, tmp_path) -> None:
    """启动时立刻跑一次（回收上次运行崩溃的残目录），之后按周期跑。"""
    local = tmp_path / "local"
    remote = LocalStorage(tmp_path / "remote")

    _save(state, local, 1)
    _upload(local, remote, 1)
    _finalize(remote, 1)

    worker = GcWorker(local, remote, rank=0, interval=0.05, dry_run=False)
    with worker:
        assert _wait(lambda: not (local / version_key(1)).exists()), "启动那一次就该清掉"

    assert worker.reports and worker.reports[0].steps_for(LOCAL) == (1,)
    assert not worker.running


def test_worker_keeps_running_when_remote_is_down(state, tmp_path) -> None:
    """远端不可达：每次都是空操作（``aborted``），线程不退出、不误删。"""
    local = tmp_path / "local"
    _save(state, local, 1)

    worker = GcWorker(local, _BrokenRemote(), rank=0, interval=0.02, dry_run=False)
    with worker:
        assert _wait(lambda: len(worker.reports) >= 2), "不可达也要继续按周期跑"

    assert all(report.aborted for report in worker.reports)
    assert all(report.deleted == () for report in worker.reports)
    assert (local / version_key(1)).is_dir(), "远端不可达时本地一个都不删"
