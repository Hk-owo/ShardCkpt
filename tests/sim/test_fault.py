"""L3 故障注入：写到一半崩溃、提交点缺失、索引/数据文件被破坏。

守的是恢复端的承诺：**要么读到完整的旧版本，要么在读第一个张量之前报错**。
没有任何一条用例允许「返回半份数据」。

本地层的字节级校验**默认开启**（见 README「本地分片 CRC 与对象存储校验的分工」）：
``save_local`` 把每个分片按默认 1 MiB 切块写 CRC-32C，读路径按块比对，
「长度不变、内容被改」会在返回数据前抛 ``ChecksumError``。显式 ``verify_crc=False``
才会放行——那条也锁住，避免以后有人把「关掉校验」当成默认。
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
from core import CheckpointError, ChecksumError, FormatError, PlanError
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


def test_local_corruption_detected_by_default(saved) -> None:
    """改一个字节：默认校验必须报 ChecksumError；显式关掉才放行。

    本地版本目录的完整性由「COMMIT 清单 + 原子 rename」兜结构性问题，字节级
    正确性由分片块 CRC-32C 兜；默认开启，需要跳过时可以显式 ``verify_crc=False``。
    """
    root, committed = saved
    path = data_file(root, 1)
    payload = bytearray(path.read_bytes())
    payload[0] ^= 0xFF
    path.write_bytes(bytes(payload))

    # 默认校验：坏字节在读第一个张量之前就被抓住，不会返回坏数据
    with pytest.raises(ChecksumError, match="crc32c"):
        load_local(root, step=1, device=DEVICE)

    # 显式关掉校验：读得出来，而且差异确实在（这是选择，不是默认）
    report = load_local(root, step=1, device=DEVICE, verify_crc=False)
    assert report.step == 1
    assert report.crc_verified is False
    assert any(
        tensor.numpy().tobytes() != committed.values[name].tobytes()
        for name, tensor in report.tensors.items()
    )
