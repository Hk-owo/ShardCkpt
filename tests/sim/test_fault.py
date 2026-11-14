"""L3 故障注入：写到一半崩溃、提交点缺失、索引/数据文件被破坏。

守的是恢复端的承诺：**要么读到完整的旧版本，要么在读第一个张量之前报错**。
没有任何一条用例允许「返回半份数据」。

本地层的字节级校验**默认开启**（见 README「本地分片 CRC 与对象存储校验的分工」）：
``save_local`` 把每个分片按默认 1 MiB 切块写 CRC-32C，读路径按块比对，
「长度不变、内容被改」会在返回数据前抛 ``ChecksumError``。显式 ``verify_crc=False``
才会放行——那条也锁住，避免以后有人把「关掉校验」当成默认。
"""

from __future__ import annotations

import hashlib
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


# --- 重存同一个 step：本地层没有代际隔离 ------------------------------------


def spawn_dying_save(
    root: Path, state: sim.TrainingState, step: int, before: str
) -> multiprocessing.process.BaseProcess:
    """起一个「写到 ``before`` 那一步之前就被 SIGKILL」的子进程。"""
    job = {
        "root": str(root),
        "step": step,
        "rank": 0,
        "tp": 1,
        "dims": {},
        "values": state.values,
        "dtypes": state.dtypes,
        "blobs": {"rng/rank_0": b"C" * 16},
        "device": DEVICE,
    }
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=sim.save_rank_then_die, args=(job, before))
    process.start()
    return process


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("before", "message"),
    [
        # 崩在写 COMMIT 之前：索引已换成新的，COMMIT 记的还是上一份索引的摘要
        ("commit", "索引内容与提交点不符"),
        # 崩在写 .metadata 之前：索引与 COMMIT 都是旧的，数据已被新内容覆盖
        ("metadata", "!= index"),
    ],
)
def test_resaving_a_committed_step_crashes_fail_closed(
    saved: tuple[Path, sim.TrainingState], before: str, message: str
) -> None:
    """重存同一个 step 崩在两步之间：这一版报废，但恢复端报错而不是读半份数据。

    本地层 ``step_N/`` 是「单写者 + 原子 rename」，**没有代际隔离**，所以重存会把
    上一次已提交的版本就地覆盖。重存总是先写数据文件，所以两种死法下数据都已经是
    新的；差别在索引写了没有，两道门分别兜住这两个窗口：

    - 崩在 ``write_commit`` 之前 ⇒ 盘上是「新数据 + 新 ``.metadata`` + 旧 ``COMMIT``」，
      ``COMMIT`` 记的索引 CRC-32C 对不上：在读第一个字节之前失败（元数据门，
      ``storage/recover.py``）；
    - 崩在 ``write_metadata`` 之前 ⇒ 盘上是「新数据 + 旧 ``.metadata`` + 旧
      ``COMMIT``」，索引摘要对得上，但块级 CRC-32C 对不上：在读第一块数据时失败
      （数据门，``storage/local_load.py``）。

    两种情况下 ``COMMIT`` 都还在，所以 ``latest_step()`` 仍指向 1 —— 「选版本」这一层
    看不出异常，是校验把它挡下的。远端不受这一条影响：``gen_*`` 前缀隔离，重存永远
    落在新代下（``docs/generation.md`` §8）。
    """
    root, committed = saved
    version = version_dir(root, 1)
    before_digests = {
        name: digest(version / name) for name in (META_FILE, "__0_0.distcp")
    }
    commit_before = digest(version / COMMIT_FILE)

    process = spawn_dying_save(root, committed.step_once(), 1, before)
    process.join(timeout=120)
    assert process.exitcode == -signal.SIGKILL, f"子进程必须死在写 {before} 之前"

    # 重存总是先写数据文件：两种死法下数据都已经是新的
    assert digest(version / "__0_0.distcp") != before_digests["__0_0.distcp"]
    # COMMIT 始终没被改写；.metadata 只在崩得更晚（写 COMMIT 之前）时才换成新的
    assert digest(version / COMMIT_FILE) == commit_before
    metadata_replaced = digest(version / META_FILE) != before_digests[META_FILE]
    assert metadata_replaced is (before == "commit"), (
        f"before={before}: .metadata 期望{'已' if before == 'commit' else '未'}被覆盖"
    )

    # COMMIT 还在 ⇒ 选版本仍指向 1，但读它就失败：这一版不可恢复
    assert latest_step(LocalStorage(root)) == 1
    with pytest.raises(ChecksumError, match=message):
        load_local(root, step=1, device=DEVICE)
    with pytest.raises(CheckpointError):
        load_local(root, device=DEVICE)  # 不指定 step 也一样：latest 就是它


def test_failed_resave_deletes_the_previously_committed_version(
    saved: tuple[Path, sim.TrainingState], monkeypatch: pytest.MonkeyPatch
) -> None:
    """重存同一个 step 在 PERSISTING 阶段失败：整个 ``step_1/`` 被删，旧版本也没了。

    ``save_local`` 的 ``ABORTED`` 清理是 ``delete_local_version(root, step)`` —— 删
    **整个版本目录**，不是只删本次写了一半的文件。所以非 SIGKILL 的写失败比 SIGKILL
    更彻底：SIGKILL 让这一版不可恢复但文件原样留着，``ABORTED`` 直接让它消失。

    与 ``DROPPED`` 对照：``SNAPSHOTTING`` 阶段（存储侧一个字节都没写）失败不做任何
    清理，旧版本原样保留。两个终态的清理语义相反，都是「本地层没有代际」的直接
    后果；可恢复性由远端保证（``gen_*`` 隔离），本地这份只是中转与热副本。
    """
    root, committed = saved
    assert latest_step(LocalStorage(root)) == 1

    def out_of_space(*args: object, **kwargs: object) -> None:
        raise OSError(28, "No space left on device (模拟磁盘满)")

    # 数据文件已经写完，状态机在 PERSISTING：失败 ⇒ ABORTED ⇒ 就地清理整个目录
    monkeypatch.setattr(ckpt, "write_metadata", out_of_space)
    with pytest.raises(OSError, match="No space left"):
        save_step(committed.step_once(), root, 1)

    assert not version_dir(root, 1).exists()
    assert latest_step(LocalStorage(root)) is None
    with pytest.raises(FormatError, match="no committed checkpoint"):
        load_local(root, device=DEVICE)


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
