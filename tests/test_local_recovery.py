"""本地恢复层：远端已提交的版本在本地留一份热副本，回退时零下载直接读。

两条判据：

- 本地恢复层由 :func:`commit.gc.collect` 的 ``keep_local`` 决定（默认 1）：只保留
  「远端已有版本 ``COMMIT`` + 本 rank 已上传」的最近一代；
- 恢复端 :func:`api.load_checkpoint` 先看本地这份能不能用（:func:`api.local_copy_available`），
  能用就直接读、不下载；不能用（没留、缺文件、覆盖不全）就回落整份下载。

两件事合起来才是「回退时直接用本地文件」：只有远端同一代也有版本 ``COMMIT`` 时本地这份
才是热副本，不是唯一副本。
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

import api as ckpt
import sim
from commit.gc import LOCAL
from commit.local import version_key
from storage.local import LocalStorage
from storage.upload import (
    NO_LOCAL_COPY,
    PARTIAL_LOCAL_COPY,
    REUSABLE,
    REMOTE_NOT_COMMITTED,
    SOURCE_LOCAL,
    SOURCE_OBJECT,
)

DEVICE = "cuda:0"
STEP = 1


@pytest.fixture
def published(state, tmp_path):
    """本地保存 → 上传 → rank 0 提交版本 → 跑一次默认窗口的回收。"""
    local = tmp_path / "local"
    remote = LocalStorage(tmp_path / "remote", conditional_writes=True)
    ckpt.save_local(sim.to_device(state.step_once(), device=DEVICE), local, step=STEP)
    ckpt.upload_local(local, remote, step=STEP, rank=0)
    ckpt.finalize_upload(remote, step=STEP, ranks=[0], layout={"tp": 1})
    report = ckpt.gc_local(local, remote, rank=0, dry_run=False)
    return local, remote, report


def test_committed_version_is_reused_from_local(published) -> None:
    """回收之后本地还留着这一代，恢复时零下载直接读，数据与下载路径逐张量一致。"""
    local, remote, report = published

    assert report.recovery_steps() == (STEP,)
    assert (local / version_key(STEP)).is_dir()

    check = ckpt.local_copy_available(local, remote, step=STEP)
    assert check.usable and check.reason == REUSABLE
    assert check.step == STEP

    reused = ckpt.load_checkpoint(remote, local_root=local, step=STEP, device=DEVICE)
    assert reused.source == SOURCE_LOCAL
    assert reused.download_bytes == 0
    assert reused.local_staging is None

    downloaded = ckpt.load_object(remote, step=STEP, device=DEVICE)
    assert reused.tensors.keys() == downloaded.tensors.keys()
    for name, tensor in reused.tensors.items():
        np.testing.assert_array_equal(tensor.numpy(), downloaded.tensors[name].numpy())


def test_missing_local_copy_falls_back_to_download(published) -> None:
    """本地没留这一代：判定为不可用，恢复自动回落到整份下载。"""
    local, remote, _ = published
    shutil.rmtree(local / version_key(STEP))

    check = ckpt.local_copy_available(local, remote, step=STEP)
    assert not check.usable and check.reason == NO_LOCAL_COPY

    report = ckpt.load_checkpoint(remote, local_root=local, step=STEP, device=DEVICE)
    assert report.source == SOURCE_OBJECT
    assert report.download_bytes > 0
    assert report.local_staging is not None


def test_keep_local_zero_leaves_no_recovery_copy(state, tmp_path) -> None:
    """``keep_local=0`` 回到「上传完即删」：本地清空，恢复只能下载。"""
    local = tmp_path / "local"
    remote = LocalStorage(tmp_path / "remote", conditional_writes=True)
    ckpt.save_local(sim.to_device(state.step_once(), device=DEVICE), local, step=STEP)
    ckpt.upload_local(local, remote, step=STEP, rank=0)
    ckpt.finalize_upload(remote, step=STEP, ranks=[0], layout={"tp": 1})

    report = ckpt.gc_local(local, remote, rank=0, dry_run=False, keep_local=0)

    assert report.steps_for(LOCAL) == (STEP,)
    assert report.recovery_steps() == ()
    assert not (local / version_key(STEP)).exists()

    loaded = ckpt.load_checkpoint(remote, local_root=local, step=STEP, device=DEVICE)
    assert loaded.source == SOURCE_OBJECT


def test_prefer_local_false_forces_download(published) -> None:
    """强制走下载：热副本还在，但这次不用它。"""
    local, remote, _ = published

    report = ckpt.load_checkpoint(
        remote, local_root=local, step=STEP, device=DEVICE, prefer_local=False
    )

    assert report.source == SOURCE_OBJECT
    assert report.download_bytes > 0


def test_local_copy_without_remote_commit_is_never_used(state, tmp_path) -> None:
    """远端还没写版本 ``COMMIT``：本地这份可能是唯一副本，不能当热副本读。"""
    local = tmp_path / "local"
    remote = LocalStorage(tmp_path / "remote", conditional_writes=True)
    ckpt.save_local(sim.to_device(state.step_once(), device=DEVICE), local, step=STEP)

    check = ckpt.local_copy_available(local, remote, step=STEP)

    assert not check.usable and check.reason == REMOTE_NOT_COMMITTED
    assert (local / version_key(STEP) / "COMMIT").is_file(), "本地版本本身没被动过"


def test_partial_local_copy_falls_back_to_download(state, tmp_path) -> None:
    """本地只有本 rank 的分片：全局覆盖校验不过，不能当热副本，必须回落下载。"""
    local = tmp_path / "local"
    remote = LocalStorage(tmp_path / "remote", conditional_writes=True)
    dims = sim.column_dims(state)
    tensors = sim.tensors_for_rank(state, rank=0, tp=2, dims=dims, device=DEVICE)
    sources = sim.sources_for_rank(state, rank=0, tp=2, dims=dims)
    ckpt.save_local(
        tensors,
        local,
        step=STEP,
        rank=0,
        sources=sources,
        layout={"tp": 2, "tp_rank": 0},
    )
    ckpt.upload_local(local, remote, step=STEP, rank=0)
    ckpt.finalize_upload(remote, step=STEP, ranks=[0], layout={"tp": 2})

    check = ckpt.local_copy_available(
        local, remote, step=STEP, tp=2, tp_rank=0, shard_dims=dims
    )

    assert not check.usable and check.reason == PARTIAL_LOCAL_COPY
    assert (local / version_key(STEP) / "COMMIT").is_file(), "本地版本本身没被动过"


def test_corrupted_local_index_falls_back_to_download(published) -> None:
    """本地索引坏了（这里是把文件截断）：回落到下载，而不是拿坏索引去读。"""
    local, remote, _ = published
    metadata = local / version_key(STEP) / ".metadata"
    metadata.write_bytes(metadata.read_bytes()[:16])

    check = ckpt.local_copy_available(local, remote, step=STEP)
    assert not check.usable and check.reason == PARTIAL_LOCAL_COPY

    report = ckpt.load_checkpoint(remote, local_root=local, step=STEP, device=DEVICE)
    assert report.source == SOURCE_OBJECT
