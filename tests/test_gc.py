"""GC 判据的纯函数测试：不碰 IO、不看时间、不看进程状态。

判据只有一条不变量（step 单调推进）推导出的三条规则：

1. 本 rank 的分片提交点（L2）在远端 ⇒ 本地 ``step_N/`` 是冗余副本；
2. 存在更新的版本提交点（L3）⇒ 这一代是残片；
3. 同一个 step 下不在当前代际里的 ``step_N/<gen>/`` ⇒ 被取代的尝试，整代删。

反过来「最新且未完成」一律保留，任何一条都不需要「写入者是否活着」的外部判断。
只读端点（``conditional_writes="off"``）能列不能删：远端候选一律记
``remote_read_only`` 跳过。
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
    RECOVERY_COPY,
    REMOTE,
    REMOTE_READ_ONLY,
    STALE_GENERATION,
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
        keep_local=0,
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
    """本 rank 的 L2 一出现，本地这一版就是冗余副本（恢复层窗口为 0 时立刻删）。"""
    result = plan(
        versions=[7],
        rank_commits=[(7, 3)],
        prefixes=[7],
        local_steps=[7],
        rank=3,
        keep_local=0,
    )

    assert result.steps_for(LOCAL) == (7,)
    assert result.reasons()[(LOCAL, 7)] == UPLOADED


# --- 本地恢复层（keep_local） ----------------------------------------------


def test_newest_committed_version_stays_as_recovery_copy() -> None:
    """远端已提交 + 本 rank 已上传：最近一个本地版本留作热副本，不删。"""
    result = plan(
        versions=[7],
        rank_commits=[(7, 3)],
        prefixes=[7],
        local_steps=[7],
        rank=3,
    )

    assert result.steps_for(LOCAL) == ()
    assert result.reasons() == {}
    assert result.kept == ((LOCAL, 7, RECOVERY_COPY), (REMOTE, 7, COMMITTED))
    assert result.recovery_steps() == (7,)


def test_recovery_copy_window_keeps_only_the_newest() -> None:
    """窗口内的最新一代留下，更旧的已提交版本照旧回收。"""
    result = plan(
        versions=[1, 2],
        rank_commits=[(1, 0), (2, 0)],
        prefixes=[1, 2],
        local_steps=[1, 2],
        rank=0,
    )

    assert result.recovery_steps() == (2,)
    assert result.steps_for(LOCAL) == (1,)
    assert result.reasons() == {(LOCAL, 1): UPLOADED}


def test_recovery_copy_does_not_keep_unreadable_versions() -> None:
    """L3 读不出来（坏 JSON / 被删）的版本不当热副本：远端这份不保证可恢复。"""
    result = plan(
        versions=[3],
        rank_commits=[(3, 0)],
        prefixes=[3],
        local_steps=[3],
        rank=0,
        unknown_versions=[3],
    )

    assert result.recovery_steps() == ()
    assert result.steps_for(LOCAL) == (3,)


def test_recovery_copy_does_not_keep_other_ranks_uploads() -> None:
    """本 rank 还没传完的版本不进恢复层（本地仍是唯一副本，走 ``not_uploaded``）。"""
    result = plan(
        versions=[4],
        rank_commits=[(4, 1)],
        prefixes=[4],
        local_steps=[4],
        rank=0,
    )

    assert result.recovery_steps() == ()
    assert result.kept == ((LOCAL, 4, NOT_UPLOADED), (REMOTE, 4, COMMITTED))


def test_keep_local_zero_is_the_old_eager_reclaim() -> None:
    """``keep_local=0`` 回到「上传完即删」，且负数是配置错误。"""
    result = plan(
        versions=[5],
        rank_commits=[(5, 0)],
        prefixes=[5],
        local_steps=[5],
        rank=0,
        keep_local=0,
    )

    assert result.steps_for(LOCAL) == (5,)

    with pytest.raises(ValueError):
        plan(
            versions=[5],
            rank_commits=[(5, 0)],
            prefixes=[5],
            local_steps=[5],
            rank=0,
            keep_local=-1,
        )


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
        keep_local=0,
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
    remote = LocalStorage(remote_root, conditional_writes=True)

    _save(state, local, 1)
    _upload(local, remote, 1)
    _finalize(remote, 1)

    plan_only = collect(local, remote, rank=0, dry_run=True, keep_local=0)
    assert plan_only.steps_for(LOCAL) == ()
    assert plan_only.planned[0].location == LOCAL
    assert plan_only.planned[0].reason == UPLOADED
    assert (local / version_key(1)).is_dir(), "dry_run 不能动文件"

    report = collect(local, remote, rank=0, dry_run=False, keep_local=0)
    assert report.steps_for(LOCAL) == (1,)
    assert report.reclaimed_bytes > 0
    assert not (local / version_key(1)).exists()
    assert (remote_root / version_key(1) / "COMMIT").is_file(), "远端已提交版本永不删"


def test_upload_incomplete_keeps_local(state, tmp_path) -> None:
    """没上传完（远端没有本 rank 的分片提交点）：本地是唯一副本，必须留着。"""
    local = tmp_path / "local"
    remote = LocalStorage(tmp_path / "remote", conditional_writes=True)

    _save(state, local, 1)

    report = collect(local, remote, rank=0, dry_run=False)

    assert report.deleted == ()
    assert report.skipped == ((LOCAL, 1, NOT_UPLOADED),)
    assert (local / version_key(1) / "COMMIT").is_file()


def test_other_rank_upload_does_not_release_this_one(state, tmp_path) -> None:
    """远端有 rank 1 的分片提交点，不代表本 rank（0）那份传完了。"""
    local = tmp_path / "local"
    remote = LocalStorage(tmp_path / "remote", conditional_writes=True)

    _save(state, local, 1)
    _upload(local, remote, 1, rank=1)

    report = collect(local, remote, rank=0, dry_run=False)

    assert report.deleted == ()
    assert (local / version_key(1)).is_dir()


def test_superseded_version_is_reclaimed_on_disk(state, tmp_path) -> None:
    """第 1 代没传成功、第 2 代已经提交：本地 1 是确定的残片。"""
    local = tmp_path / "local"
    remote = LocalStorage(tmp_path / "remote", conditional_writes=True)

    _save(state, local, 1)          # 假设上传失败 / 被放弃
    _save(state, local, 2)
    _upload(local, remote, 2)
    _finalize(remote, 2)

    report = collect(local, remote, rank=0, dry_run=False, keep_local=0)

    assert sorted(report.steps_for(LOCAL)) == [1, 2]
    assert {item.step: item.reason for item in report.deleted} == {
        1: SUPERSEDED,
        2: UPLOADED,
    }
    assert list(local.glob("step_*")) == []


def test_recovery_copy_stays_local_and_is_reused(state, tmp_path) -> None:
    """默认窗口下：最新的已提交版本留在本地，回退时零下载直接读。"""
    local = tmp_path / "local"
    remote = LocalStorage(tmp_path / "remote", conditional_writes=True)

    _save(state, local, 1)
    _upload(local, remote, 1)
    _finalize(remote, 1)
    _save(state, local, 2)
    _upload(local, remote, 2)
    _finalize(remote, 2)

    report = collect(local, remote, rank=0, dry_run=False)

    assert report.recovery_steps() == (2,)
    assert report.steps_for(LOCAL) == (1,)
    assert (local / version_key(2)).is_dir(), "最新一代留在本地当热副本"
    assert not (local / version_key(1)).exists(), "更旧的一代照旧回收"

    reused = ckpt.load_checkpoint(remote, local_root=local, step=2, device=DEVICE)
    assert reused.source == "local"
    assert reused.download_bytes == 0


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
    remote = LocalStorage(tmp_path / "remote", conditional_writes=True)

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
    remote = LocalStorage(remote_root, conditional_writes=True)

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

    assert first.steps_for(LOCAL) == (1,), "被打断的 1 收敛掉，2 留在恢复层"
    assert first.recovery_steps() == (2,)
    assert second.deleted == (), "第二次没有东西可删"
    assert [p.name for p in local.glob("step_*")] == [version_key(2)]


def test_stale_remote_prefix_is_reclaimed_on_disk(state, tmp_path) -> None:
    """远端半途的上传前缀（没有版本提交点）+ 有更新的一代 ⇒ 整个前缀删掉。"""
    local = tmp_path / "local"
    remote_root = tmp_path / "remote"
    remote = LocalStorage(remote_root, conditional_writes=True)

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
    remote = LocalStorage(tmp_path / "remote", conditional_writes=True)

    _save(state, local, 1)
    _upload(local, remote, 1)
    _finalize(remote, 1)

    worker = GcWorker(local, remote, rank=0, interval=0.05, dry_run=False, keep_local=0)
    with worker:
        assert _wait(lambda: not (local / version_key(1)).exists()), "启动那一次就该清掉"

    assert worker.reports and worker.reports[0].steps_for(LOCAL) == (1,)
    assert not worker.running


def test_worker_keeps_the_recovery_copy_by_default(state, tmp_path) -> None:
    """周期线程默认按恢复层窗口跑：最新已提交版本留在本地。"""
    local = tmp_path / "local"
    remote = LocalStorage(tmp_path / "remote", conditional_writes=True)

    _save(state, local, 1)
    _upload(local, remote, 1)
    _finalize(remote, 1)

    worker = GcWorker(local, remote, rank=0, interval=60.0, dry_run=False)
    report = worker.run_once()

    assert report.recovery_steps() == (1,)
    assert (local / version_key(1)).is_dir()


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


# --- 代际残片（stale generation） ------------------------------------------


def _upload_generation(local, remote, step: int, generation) -> None:
    ckpt.upload_local(local, remote, step=step, rank=0, generation=generation)


def test_plan_marks_generations_outside_the_current_one() -> None:
    """同一 step 下不在当前代际的前缀是残片；当前代永远保留。"""
    result = plan(
        versions={5},
        rank_commits={(5, 0, "gen_0002-bbbbbbbb")},
        prefixes={5},
        local_steps=(),
        rank=0,
        generations={5: {"gen_0001-aaaaaaaa", "gen_0002-bbbbbbbb"}},
        current_generations={5: "gen_0002-bbbbbbbb"},
    )

    assert result.steps_for(REMOTE) == (5,)
    assert result.generations_for(5) == ("gen_0001-aaaaaaaa",)
    assert result.reasons()[(REMOTE, 5)] == STALE_GENERATION


def test_plan_keeps_the_newest_generation_when_there_is_no_commit() -> None:
    """还没有 L3：保留序号最大的一代（可能正在传），其余按残片收掉。"""
    result = plan(
        versions=(),
        rank_commits=(),
        prefixes={7},
        local_steps=(),
        rank=0,
        generations={7: {"gen_0001-aaaaaaaa", "gen_0003-cccccccc"}},
    )

    assert result.generations_for(7) == ("gen_0001-aaaaaaaa",)
    assert (REMOTE, 7, NO_NEWER_VERSION) in result.kept


def test_plan_keeps_everything_when_the_commit_is_unreadable() -> None:
    """L3 读不出来（坏 JSON / 被删）：不猜当前代，一个代际都不删。"""
    result = plan(
        versions={5},
        rank_commits={(5, 0, "gen_0001-aaaaaaaa")},
        prefixes={5},
        local_steps=(),
        rank=0,
        generations={5: {"gen_0001-aaaaaaaa", "gen_0002-bbbbbbbb"}},
        current_generations={},
        unknown_versions={5},
    )

    assert result.steps_for(REMOTE) == ()
    assert (REMOTE, 5, COMMITTED) in result.kept


def test_collect_deletes_stale_generation_and_keeps_current(state, tmp_path) -> None:
    local = tmp_path / "local"
    remote = LocalStorage(tmp_path / "remote", conditional_writes=True)
    _save(state, local, 5)

    stale = ckpt.begin_upload(remote, step=5, ranks=[0])
    _upload_generation(local, remote, 5, stale)
    current = ckpt.begin_upload(remote, step=5, ranks=[0])
    _upload_generation(local, remote, 5, current)
    _finalize(remote, 5)

    report = collect(local, remote, rank=0, dry_run=False)

    assert [
        item.generation for item in report.deleted if item.location == REMOTE
    ] == [stale.id]
    assert remote.list(f"step_5/{stale.id}") == [], "旧代整段删掉"
    assert remote.list(f"step_5/{current.id}"), "当前代不动"
    assert report.recovery_steps() == (5,), "本 rank 已上传，本地留作恢复层"
    assert (local / version_key(5)).is_dir()


def test_gen_pointer_alone_never_makes_a_version_visible(tmp_path) -> None:
    """只写了 GEN、没写数据：既不是版本，也不会被当成「正在传」删掉。"""
    remote = LocalStorage(tmp_path / "remote", conditional_writes=True)
    generation = ckpt.begin_upload(remote, step=7, ranks=[0])

    report = collect(tmp_path / "local", remote, rank=0, dry_run=False)

    assert 7 not in report.versions
    assert report.steps_for(REMOTE) == ()
    assert remote.list(f"step_7/{generation.id}") == []
    assert remote.list("step_7/GEN"), "GEN 本身不是数据，留着让下次重试知道序号"


def test_stale_generation_delete_order_is_rank_commit_then_data(state, tmp_path) -> None:
    """单代删除：先删 L2，再删数据 —— 不会留下「看着像传完了」的前缀。"""
    local = tmp_path / "local"
    remote = LocalStorage(tmp_path / "remote", conditional_writes=True)
    _save(state, local, 5)
    stale = ckpt.begin_upload(remote, step=5, ranks=[0])
    _upload_generation(local, remote, 5, stale)
    current = ckpt.begin_upload(remote, step=5, ranks=[0])
    _upload_generation(local, remote, 5, current)
    _finalize(remote, 5)

    deleted: list[str] = []
    real_delete = remote.delete
    remote.delete = lambda key: (deleted.append(key), real_delete(key))[1]  # type: ignore[method-assign]

    collect(local, remote, rank=0, dry_run=False)

    old = [key for key in deleted if key.startswith(f"step_5/{stale.id}/")]
    rank_commit = f"step_5/{stale.id}/rank_0/COMMIT"
    data = next(key for key in old if key.endswith(".distcp"))
    assert old[0] == rank_commit
    assert old.index(rank_commit) < old.index(data)


def test_read_only_remote_is_listed_but_never_deleted(state, tmp_path) -> None:
    """只读端点：远端候选一律跳过（``remote_read_only``），一个对象都不删。"""
    from storage.object import S3Config, S3Storage
    import s3_stub

    with s3_stub.start() as server:
        config = dict(
            bucket="ckpt",
            prefix="runs",
            endpoint_url=server.endpoint_url,
            access_key_id="test-key",
            secret_access_key="test-secret",
        )
        writer = S3Storage(S3Config(**config))
        try:
            # 已提交的 step 2 + 一段被越过的 step 1 残片（没有 L3）
            root = tmp_path / "local-2"
            _save(state.step_once(), root, 2)
            generation = ckpt.begin_upload(writer, step=2, ranks=[0])
            ckpt.upload_local(root, writer, step=2, rank=0, generation=generation)
            ckpt.finalize_upload(
                writer, step=2, ranks=[0], generation=generation, keep=None
            )
            writer.write("step_1/rank_0/junk.bin", b"stale residue")
        finally:
            writer.close()

        reader = S3Storage(S3Config(**config, conditional_writes="off"))
        try:
            report = collect(tmp_path / "local-2", reader, rank=0, dry_run=False)
        finally:
            reader.close()

    assert not [item for item in report.deleted if item.location == REMOTE]
    assert (REMOTE, 1, REMOTE_READ_ONLY) in report.skipped
    assert (REMOTE, 2, COMMITTED) in report.skipped
