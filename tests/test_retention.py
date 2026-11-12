"""对象存储侧的有界保留：rank 0 在版本提交完成后只留最近 ``keep`` 代。

和 ``commit/gc.py`` 的另外两条判据（本地上传完即删、远端残片）分开测：这里的输入
只有「带版本 ``COMMIT`` 的 step」，输出是「最旧的那几代整段删掉」，删除顺序与写入
镜像 —— **先删这一代的版本 COMMIT，再删分片提交点与数据**。
"""

from __future__ import annotations

from pathlib import Path

import pytest

import api as ckpt
import s3_stub
import sim
from commit.gc import (
    DEFAULT_KEEP_VERSIONS,
    REMOTE,
    RETENTION,
    plan_retention,
    retain_remote_versions,
)
from storage.base import StorageBackend, StorageError
from storage.object import S3Config, S3Storage
from storage.upload import commit_key, version_prefix

DEVICE = "cuda:0"


@pytest.fixture
def s3_server():
    with s3_stub.start() as server:
        yield server


@pytest.fixture
def s3_config(s3_server) -> dict:
    return {
        "bucket": "ckpt",
        "prefix": "runs",
        "endpoint_url": s3_server.endpoint_url,
        "access_key_id": "test-key",
        "secret_access_key": "test-secret",
    }


@pytest.fixture
def storage(s3_config) -> S3Storage:
    client = S3Storage(S3Config(**s3_config))
    yield client
    client.close()


def _commit(
    state: sim.TrainingState,
    tmp_path: Path,
    storage: StorageBackend,
    step: int,
    *,
    keep: int | None = DEFAULT_KEEP_VERSIONS,
):
    """走完整提交链路：本地保存 → 本 rank 上传 → rank 0 写 COMMIT（并裁剪）。"""
    root = tmp_path / f"local-{step}"
    ckpt.save_local(sim.to_device(state, device=DEVICE), root, step=step)
    ckpt.upload_local(root, storage, step=step, rank=0)
    return ckpt.finalize_upload(storage, step=step, ranks=[0], layout={"tp": 1}, keep=keep)


def _commit_steps(state, tmp_path, storage, steps, *, keep=DEFAULT_KEEP_VERSIONS):
    """连续提交若干个 step，返回最后一个 step 的状态。"""
    snapshot = state
    for step in steps:
        snapshot = snapshot.step_once()
        _commit(snapshot, tmp_path, storage, step, keep=keep)
    return snapshot


def _versions(storage: StorageBackend) -> list[int]:
    """远端带版本 ``COMMIT`` 的 step（升序）。"""
    return sorted(
        int(key.split("/")[0][len("step_") :])
        for key in storage.list("")
        if key.count("/") == 1 and key.endswith("/COMMIT")
    )


class _RecordingRemote(StorageBackend):
    """透传后端，只额外记录 ``delete`` 的顺序。"""

    def __init__(self, inner: StorageBackend) -> None:
        self.inner = inner
        self.deleted: list[str] = []

    def write(self, key: str, data: bytes) -> None:
        self.inner.write(key, data)

    def read(self, key: str) -> bytes:
        return self.inner.read(key)

    def read_range(self, key: str, offset: int, length: int) -> bytes:
        return self.inner.read_range(key, offset, length)

    def rename_or_put(self, src: str, dst: str) -> None:
        self.inner.rename_or_put(src, dst)

    def list(self, prefix: str = "") -> list[str]:
        return self.inner.list(prefix)

    def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.inner.delete(key)


class _DeleteFails(_RecordingRemote):
    """``delete`` 一律失败：模拟裁剪途中远端抖动。"""

    def delete(self, key: str) -> None:
        self.deleted.append(key)
        raise StorageError(f"{key}: delete failed")


# --- 判据（纯函数） --------------------------------------------------------


def test_plan_retention_keeps_the_newest_window() -> None:
    """只留最新 ``keep`` 代，返回要删的 step（最旧在前）；顺序无关、自动去重。"""
    assert plan_retention(versions=[4, 1, 3, 2], keep=3) == (1,)
    assert plan_retention(versions=[1, 2, 3], keep=3) == ()
    assert plan_retention(versions=[1, 2, 3], keep=1) == (1, 2)
    assert plan_retention(versions=[], keep=3) == ()
    assert plan_retention(versions=[2, 4], keep=1) == plan_retention(versions=[4, 2], keep=1)


def test_plan_retention_rejects_an_empty_window() -> None:
    """``keep`` 必须 >= 1：最新那一代永远留着。"""
    with pytest.raises(ValueError):
        plan_retention(versions=[1, 2], keep=0)
    with pytest.raises(ValueError):
        retain_remote_versions(None, keep=0)


# --- 提交即裁剪 ------------------------------------------------------------


def test_each_commit_keeps_the_latest_three(state, tmp_path, storage) -> None:
    """第 4 次提交写完 COMMIT 之后，最旧的那一代整段消失。"""
    _commit_steps(state, tmp_path, storage, (1, 2, 3, 4))

    assert _versions(storage) == [2, 3, 4]
    assert storage.list(version_prefix(1)) == [], "最旧的一代连数据一起删"
    assert commit_key(2) in storage.list(version_prefix(2))

    with pytest.raises(ckpt.FormatError):
        ckpt.load_object(storage, step=1, device=DEVICE)
    assert ckpt.load_object(storage, step=2, device=DEVICE).step == 2, "窗口最旧的一代仍可恢复"
    assert ckpt.load_object(storage, device=DEVICE).step == 4


def test_oldest_version_commit_goes_before_its_data(state, tmp_path, storage) -> None:
    """删除顺序与写入镜像：版本 COMMIT → 分片 COMMIT → 数据。"""
    _commit_steps(state, tmp_path, storage, (1, 2, 3, 4), keep=None)

    recording = _RecordingRemote(storage)
    report = retain_remote_versions(recording, keep=3)

    assert report.steps() == (1,)
    assert report.deleted[0].reason == RETENTION
    assert report.deleted[0].location == REMOTE

    old = [key for key in recording.deleted if key.startswith(f"{version_prefix(1)}/")]
    assert old[0] == commit_key(1), "先删这一版的版本 COMMIT"
    rank_commit = f"{version_prefix(1)}/rank_0/COMMIT"
    data = f"{version_prefix(1)}/rank_0/__0_0.distcp"
    assert old.index(rank_commit) < old.index(data), "分片提交点排在数据之前"


def test_retention_dry_run_only_plans(state, tmp_path, storage) -> None:
    """``dry_run`` 只给计划：窗口外的 step 列出来，但一个对象都不删。"""
    _commit_steps(state, tmp_path, storage, (1, 2, 3, 4), keep=None)

    report = retain_remote_versions(storage, keep=2, dry_run=True)

    assert report.planned == (1, 2)
    assert report.kept == (3, 4)
    assert report.deleted == ()
    assert _versions(storage) == [1, 2, 3, 4]


def test_keep_none_disables_retention(state, tmp_path, storage) -> None:
    """``keep=None``：提交照做，但不动任何旧版本（交给运维单独裁剪）。"""
    _commit_steps(state, tmp_path, storage, (1, 2, 3, 4), keep=None)

    assert _versions(storage) == [1, 2, 3, 4]


def test_retention_is_idempotent(state, tmp_path, storage) -> None:
    """收敛：窗口外的代删完之后，再跑一次什么都不删。"""
    _commit_steps(state, tmp_path, storage, (1, 2, 3, 4))
    first = retain_remote_versions(storage, keep=3)
    second = retain_remote_versions(storage, keep=3)

    assert first.deleted == ()
    assert second.deleted == ()
    assert second.kept == (2, 3, 4)


def test_retention_ignores_prefixes_without_a_commit(state, tmp_path, storage) -> None:
    """没有版本 COMMIT 的旧前缀不归保留判据管（那是 ``collect`` 的残片判据）。"""
    stale = f"{version_prefix(0)}/rank_0/__0_0.distcp"
    storage.write(stale, b"x" * 64)

    _commit_steps(state, tmp_path, storage, (1, 2, 3, 4))

    report = retain_remote_versions(storage, keep=3)

    assert report.deleted == ()
    assert storage.read(stale) == b"x" * 64, "半途上传的前缀不能被保留判据碰掉"


# --- 降级 ------------------------------------------------------------------


def test_prune_failure_is_reported_and_keeps_the_newest(state, tmp_path, storage) -> None:
    """删到一半失败：停在那里并记原因，最新那一代照样可恢复。"""
    _commit_steps(state, tmp_path, storage, (1, 2, 3), keep=None)

    failing = _DeleteFails(storage)
    report = retain_remote_versions(failing, keep=2)

    assert report.aborted is not None
    assert report.deleted == ()
    assert report.planned == (1,)
    assert ckpt.load_object(storage, device=DEVICE).step == 3


def test_commit_succeeds_even_when_pruning_fails(state, tmp_path, storage) -> None:
    """裁剪是收尾动作：失败不能让已经写完 COMMIT 的这次提交被判成失败。"""
    snapshot = _commit_steps(state, tmp_path, storage, (1, 2, 3), keep=None)

    failing = _DeleteFails(storage)
    commit = _commit(snapshot.step_once(), tmp_path, failing, 4)

    assert commit.step == 4
    assert ckpt.load_object(storage, step=4, device=DEVICE).step == 4
    # 旧代没删掉不是错误，下一次提交成功后同一条判据会把它收掉
    assert _versions(storage) == [1, 2, 3, 4]
    assert retain_remote_versions(storage, keep=3).steps() == (1,)
