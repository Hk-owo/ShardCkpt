"""L3 故障注入：写到一半崩溃、提交点缺失、索引/数据文件被破坏。

守的是恢复端的承诺：**要么读到完整的旧版本，要么在读第一个张量之前报错**。
没有任何一条用例允许「返回半份数据」。

本地层不做字节级校验和（见 README「为什么本地层不做字节级校验」），所以这里
也不假装能抓到「长度不变、内容被改」—— 那条用 ``test_local_layer_does_not_
checksum_bytes`` 明确锁住，避免以后有人误以为它是 bug。
"""

from __future__ import annotations

import json
import multiprocessing
import signal
from pathlib import Path

import pytest

import api as ckpt
import sim
from async_ import PinnedBufferPool, aligned_allocator
from commit.local import COMMIT_FILE, latest_step
from core import CheckpointError, FormatError, PlanError
from core.index import META_FILE
from storage.local import LocalStorage
from storage.local_load import load_local

DEVICE = "cuda:0"


def save_step(state: sim.TrainingState, root: Path, step: int) -> None:
    report = ckpt.save_local(
        sim.to_device(state, device=DEVICE),
        root,
        step=step,
        blobs={"rng/rank_0": bytes([step]) * 16},
        layout={"tp": 1},
    )
    assert report.step == step


def version_dir(root: Path, step: int) -> Path:
    return root / f"step_{step}"


def data_file(root: Path, step: int) -> Path:
    return version_dir(root, step) / "__0_0.distcp"


@pytest.fixture
def saved(state, tmp_path) -> tuple[Path, sim.TrainingState]:
    """一个已经提交的 step 1，作为「旧版本可读」的基准。"""
    root = tmp_path / "ckpt"
    committed = state.step_once()
    save_step(committed, root, 1)
    return root, committed.snapshot()


# --- 写到一半崩溃 ----------------------------------------------------------


def test_crash_before_commit_leaves_version_invisible(saved) -> None:
    """数据与索引都写完了、COMMIT 没写，进程被 SIGKILL：这一版对恢复端不存在。"""
    root, committed = saved
    crashed = committed.step_once()

    job = {
        "root": str(root),
        "step": 2,
        "rank": 0,
        "tp": 1,
        "dims": {},
        "values": crashed.values,
        "dtypes": crashed.dtypes,
        "blobs": {"rng/rank_0": b"C" * 16},
        "device": DEVICE,
    }
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=sim.save_rank_then_die, args=(job,))
    process.start()
    process.join(timeout=120)
    assert process.exitcode == -signal.SIGKILL, "子进程必须在写 COMMIT 之前被杀死"

    # 半成品确实落在了盘上：数据文件 + 索引都在，只缺提交点
    partial = version_dir(root, 2)
    assert (partial / META_FILE).is_file()
    assert (partial / "__0_0.distcp").is_file()
    assert not (partial / COMMIT_FILE).exists()

    # 恢复端完全看不到这一版
    storage = LocalStorage(root)
    assert latest_step(storage) == 1
    with pytest.raises(FormatError, match="COMMIT missing"):
        load_local(root, step=2, device=DEVICE)

    # 读最新版本仍然拿到完整的旧版本，而不是半份新数据
    report = load_local(root, device=DEVICE)
    assert report.step == 1
    sim.assert_state_matches(
        {name: tensor.numpy() for name, tensor in report.tensors.items()}, committed
    )


# --- 提交点 / 索引 / 数据文件的破坏矩阵 ------------------------------------


def remove_commit(root: Path) -> None:
    (version_dir(root, 1) / COMMIT_FILE).unlink()


def truncate_metadata(root: Path) -> None:
    path = version_dir(root, 1) / META_FILE
    path.write_bytes(path.read_bytes()[:-1])


def remove_data_file(root: Path) -> None:
    data_file(root, 1).unlink()


def truncate_data_file(root: Path) -> None:
    path = data_file(root, 1)
    path.write_bytes(path.read_bytes()[:-4096])


def unlist_data_file(root: Path) -> None:
    """COMMIT 的清单少了索引引用的数据文件。"""
    storage = LocalStorage(root)
    key = f"step_1/{COMMIT_FILE}"
    payload = json.loads(storage.read(key).decode("utf-8"))
    payload["files"] = {}
    storage.write(key, json.dumps(payload).encode("utf-8"))


DAMAGE = {
    "commit_missing": (remove_commit, "COMMIT missing"),
    "metadata_truncated": (truncate_metadata, "bytes, COMMIT records"),
    "data_file_missing": (remove_data_file, "data file listed in COMMIT is missing"),
    "data_file_truncated": (truncate_data_file, "bytes on disk, COMMIT records"),
    "data_file_unlisted": (unlist_data_file, "COMMIT does not list data file"),
}


@pytest.mark.parametrize("damage", sorted(DAMAGE))
def test_commit_gate_rejects_damaged_version(saved, damage: str) -> None:
    """每一种破坏都必须在读第一个张量之前失败，且报错指向具体原因。"""
    root, _ = saved
    apply_damage, message = DAMAGE[damage]
    apply_damage(root)

    with pytest.raises(FormatError, match=message):
        load_local(root, step=1, device=DEVICE)
    with pytest.raises(CheckpointError):
        load_local(root, device=DEVICE)  # 不指定 step：取最大已提交版本


# --- 写失败不回退可见状态 --------------------------------------------------


def test_write_failure_leaves_no_newer_commit(saved) -> None:
    """拿不到 pinned 块导致保存失败：不能留下 step 2 目录，也不能出现更新的提交。"""
    root, committed = saved
    blocked = PinnedBufferPool(
        chunk_bytes=1 << 20,
        min_chunks=1,
        max_bytes=1 << 20,
        timeout=0,
        allocator=aligned_allocator(),
    )
    held = blocked.acquire(1 << 20)  # 占满唯一一块，save_local 必然拿不到
    try:
        with pytest.raises(PlanError):
            ckpt.save_local(
                sim.to_device(committed.step_once(), device=DEVICE),
                root,
                step=2,
                pool=blocked,
            )
    finally:
        held.release()
        blocked.close()

    assert not version_dir(root, 2).exists()
    assert latest_step(LocalStorage(root)) == 1
    report = load_local(root, device=DEVICE)
    assert report.step == 1
    sim.assert_state_matches(
        {name: tensor.numpy() for name, tensor in report.tensors.items()}, committed
    )


# --- 设计边界：本地层不校验字节 --------------------------------------------


def test_local_layer_does_not_checksum_bytes(saved) -> None:
    """改一个字节不会被发现 —— 这是设计决定，不是缺陷。

    本地版本目录只是上传对象存储之前的暂存：完整性由「COMMIT 清单 + 原子 rename」
    和上传边界的对象级校验和负责（见 README「为什么本地层不做字节级校验」）。
    新写的索引里没有 ``crc32c``，所以 ``verify_crc=True`` 也只会如实报告「没校验成」。
    """
    root, committed = saved
    path = data_file(root, 1)
    payload = bytearray(path.read_bytes())
    payload[0] ^= 0xFF
    path.write_bytes(bytes(payload))

    report = load_local(root, step=1, device=DEVICE)
    assert report.step == 1

    report = load_local(root, step=1, device=DEVICE, verify_crc=True)
    assert report.crc_verified is True
    assert report.verified_blocks == 0, "新索引不写 crc32c，没有可校验的整分片"
    assert report.unverified, "开了校验却校验不了时，必须如实报告"

    # 差异确实没被发现：内容已经被改过了
    assert any(
        tensor.numpy().tobytes() != committed.values[name].tobytes()
        for name, tensor in report.tensors.items()
    )
