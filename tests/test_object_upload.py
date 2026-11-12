"""对象存储上传流程：每个 rank 独立上传，rank 0 只 ``list`` 前缀检查。

设计目标（demo）：

- 对象存储用进程内的 S3 兼容服务（``tests/s3_stub.py``）模拟，上传路径与真实
  对象存储一致（PUT / Range GET / list / multipart / 对象级校验和）；
- 每个 rank 各自 ``save_local`` 再 ``upload_local``，只写 ``step_N/rank_<i>/``；
- rank 0 用 ``finalize_upload`` 只 ``list`` 一次版本前缀，确认每个 rank 的
  ``.metadata`` 与 ``rank_<i>/COMMIT`` 都在，然后写版本 ``COMMIT`` —— **不读任何人的索引**；
- 恢复端自己 ``list`` + 读各 rank 的 per-rank 索引合并，再做全局覆盖校验。
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import api as ckpt
import s3_stub
import sim
from async_ import Tensor, get_backend
from core import CheckpointError, ChecksumError, FormatError
from core.metadata import CPU
from storage.base import ObjectNotFoundError, StorageError
from storage.object import S3Config, S3Storage
from storage.upload import commit_key, data_key, index_key, rank_commit_key, version_prefix

DEVICE = "cuda:0"
STEP = 1
PREFIX = "runs"


@pytest.fixture
def s3_server():
    with s3_stub.start() as server:
        yield server


@pytest.fixture
def s3_config(s3_server) -> dict:
    return {
        "bucket": "ckpt",
        "prefix": PREFIX,
        "endpoint_url": s3_server.endpoint_url,
        "access_key_id": "test-key",
        "secret_access_key": "test-secret",
    }


@pytest.fixture
def storage(s3_config) -> S3Storage:
    client = S3Storage(S3Config(**s3_config))
    yield client
    client.close()


def blobs_for(rank: int) -> dict[str, bytes]:
    """每个 rank 自己的小状态：RNG 位置不同，另有一个所有人相同的小状态。"""
    return {f"rng/rank_{rank}": bytes([rank]) * 16, "lr_scheduler": b"L" * 32}


def upload_ranks(
    state: sim.TrainingState,
    *,
    tp: int,
    local_root: Path,
    s3_config: dict,
    step: int,
    ranks: list[int] | None = None,
) -> list[dict]:
    """spawn 每个 rank 的进程：本地提交 + 独立上传。"""
    jobs = [
        {
            "root": str(local_root / f"rank_{rank}"),
            "step": step,
            "rank": rank,
            "tp": tp,
            "dims": sim.column_dims(state),
            "values": state.values,
            "dtypes": state.dtypes,
            "blobs": blobs_for(rank),
            "device": DEVICE,
            "s3": s3_config,
        }
        for rank in (ranks if ranks is not None else list(range(tp)))
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=context) as pool:
        return list(pool.map(sim.save_and_upload_rank, jobs))


def keys_of(storage: S3Storage, prefix: str) -> set[str]:
    return set(storage.list(prefix))


# --- 上传与 rank 0 的检查 ---------------------------------------------------


def test_ranks_upload_independently_and_rank0_only_lists(
    state, tmp_path, storage, s3_config, s3_server
) -> None:
    """每个 rank 只写自己的前缀；rank 0 提交时只有 list + PUT，没有 GET 索引。"""
    tp = 2
    saved = state.step_once()
    summaries = upload_ranks(
        saved, tp=tp, local_root=tmp_path / "local", s3_config=s3_config, step=STEP
    )
    assert [item["rank"] for item in summaries] == list(range(tp))
    assert all(item["uploaded"] > 0 for item in summaries)

    keys = keys_of(storage, version_prefix(STEP))
    for rank in range(tp):
        assert index_key(STEP, rank) in keys
        assert rank_commit_key(STEP, rank) in keys
        assert f"{version_prefix(STEP)}/rank_{rank}/__{rank}_0.distcp" in keys
    assert commit_key(STEP) not in keys, "commit 之前这一版对恢复端不可见"

    # rank 0 的提交动作：只允许 list（前缀查询）与 PUT，不允许读任何 rank 的索引
    s3_server.store.requests.clear()
    commit = ckpt.finalize_upload(storage, step=STEP, ranks=list(range(tp)), layout={"tp": tp})

    reads = [
        request
        for request in s3_server.store.requests
        if request[0] == "GET" and ".metadata" in request[1]
    ]
    assert reads == [], f"rank 0 不能汇总索引，却读了：{reads}"

    assert commit.ranks == tuple(range(tp))
    assert commit.layout == {"tp": tp}
    assert commit_key(STEP) in keys_of(storage, version_prefix(STEP))
    assert "latest" not in keys_of(storage, ""), "不再写 latest 指针"


def test_restore_roundtrips_bytes_and_blobs(state, tmp_path, storage, s3_config) -> None:
    saved = state.step_once()
    upload_ranks(saved, tp=2, local_root=tmp_path / "local", s3_config=s3_config, step=STEP)
    ckpt.finalize_upload(storage, step=STEP, ranks=[0, 1], layout={"tp": 2})

    report = ckpt.load_object(storage, step=STEP, device=DEVICE)

    assert report.step == STEP
    assert report.amplification == pytest.approx(1.0)
    assert report.h2d_bytes == saved.total_bytes()
    sim.assert_state_matches(
        {name: tensor.numpy() for name, tensor in report.tensors.items()}, saved
    )
    assert report.blobs["rng/rank_0"] == bytes([0]) * 16
    assert report.blobs["rng/rank_1"] == bytes([1]) * 16
    assert report.blobs["lr_scheduler"] == b"L" * 32

    # 恢复端从索引合并结果里能看到两个 rank 的分片
    for name, meta in report.index.tensors.items():
        if name in sim.column_dims(saved):
            assert len(meta.shards) == 2, name
            assert {shard.file.split("/")[0] for shard in meta.shards} == {"rank_0", "rank_1"}


def test_tp_change_reassembles_from_object_storage(state, tmp_path, storage, s3_config) -> None:
    """TP=4 存、按 TP=4 逐 rank 恢复再拼回；TP=1 整读也要给出同一份数据。"""
    tp = 4
    saved = state.step_once()
    dims = sim.column_dims(saved)
    upload_ranks(saved, tp=tp, local_root=tmp_path / "local", s3_config=s3_config, step=STEP)
    ckpt.finalize_upload(storage, step=STEP, ranks=list(range(tp)), layout={"tp": tp})

    parts = []
    for rank in range(tp):
        report = ckpt.load_object(
            storage, step=STEP, device=DEVICE, tp=tp, tp_rank=rank, shard_dims=dims
        )
        parts.append({name: tensor.numpy() for name, tensor in report.tensors.items()})
    sim.assert_state_matches(sim.reassemble(parts, saved, dims=dims), saved)

    whole = ckpt.load_object(storage, step=STEP, device=DEVICE)
    sim.assert_state_matches(
        {name: tensor.numpy() for name, tensor in whole.tensors.items()}, saved
    )


# --- rank 0 的门禁 ---------------------------------------------------------


def test_finalize_upload_is_rerunnable(state, tmp_path, storage, s3_config) -> None:
    """rank 0 崩在写完版本 COMMIT 之前：重启补调一次就能把这一版补成可恢复。

    判据只看每个 rank 的分片提交点在不在，所以这一步是幂等的 —— 不需要任何
    attempt/epoch 标记来区分「第一次」和「补跑」。
    """
    saved = state.step_once()
    upload_ranks(saved, tp=2, local_root=tmp_path / "local", s3_config=s3_config, step=STEP)

    # 第一次「崩了」：rank 0 还没来得及写版本 COMMIT
    assert commit_key(STEP) not in keys_of(storage, version_prefix(STEP))

    first = ckpt.finalize_upload(storage, step=STEP, ranks=[0, 1], layout={"tp": 2})
    second = ckpt.finalize_upload(storage, step=STEP, ranks=[0, 1], layout={"tp": 2})

    assert first == second
    report = ckpt.load_object(storage, step=STEP, device=DEVICE)
    assert report.step == STEP
    sim.assert_state_matches(
        {name: tensor.numpy() for name, tensor in report.tensors.items()}, saved
    )


def test_finalize_rejects_missing_rank(state, tmp_path, storage, s3_config) -> None:
    """少一个 rank 上传：rank 0 的 list 检查直接失败，且不写 COMMIT。"""
    saved = state.step_once()
    upload_ranks(
        saved, tp=2, local_root=tmp_path / "local", s3_config=s3_config, step=STEP, ranks=[0]
    )

    with pytest.raises(FormatError, match="not all uploaded"):
        ckpt.finalize_upload(storage, step=STEP, ranks=[0, 1], layout={"tp": 2})

    assert commit_key(STEP) not in keys_of(storage, version_prefix(STEP))
    with pytest.raises(FormatError, match="COMMIT missing"):
        ckpt.load_object(storage, step=STEP, device=DEVICE)


def test_finalize_rejects_half_uploaded_rank(state, tmp_path, storage, s3_config) -> None:
    """数据传了、分片提交点还没写（进程死在中间）：同样不算完成。"""
    saved = state.step_once()
    upload_ranks(
        saved, tp=2, local_root=tmp_path / "local", s3_config=s3_config, step=STEP, ranks=[0, 1]
    )
    storage.delete(rank_commit_key(STEP, 1))

    with pytest.raises(FormatError, match="not all uploaded"):
        ckpt.finalize_upload(storage, step=STEP, ranks=[0, 1], layout={"tp": 2})

    assert commit_key(STEP) not in keys_of(storage, version_prefix(STEP))


def test_finalize_rejects_stray_rank_prefix(state, tmp_path, storage, s3_config) -> None:
    """COMMIT 之外的 rank 前缀是上一次重试的残片，必须报出来而不是悄悄忽略。"""
    saved = state.step_once()
    upload_ranks(
        saved, tp=2, local_root=tmp_path / "local", s3_config=s3_config, step=STEP, ranks=[0]
    )
    storage.write(index_key(STEP, 9), b"{}")
    storage.write(rank_commit_key(STEP, 9), b"{}")

    with pytest.raises(FormatError, match="unexpected rank prefix"):
        ckpt.finalize_upload(storage, step=STEP, ranks=[0], layout={"tp": 2})


# --- 恢复端自己的门禁 ------------------------------------------------------


def test_restore_rejects_missing_rank_index(state, tmp_path, storage, s3_config) -> None:
    """COMMIT 声明了 rank 1，但它的索引对象不在了：直接拒绝，不吐半份数据。"""
    saved = state.step_once()
    upload_ranks(saved, tp=2, local_root=tmp_path / "local", s3_config=s3_config, step=STEP)
    ckpt.finalize_upload(storage, step=STEP, ranks=[0, 1], layout={"tp": 2})

    storage.delete(index_key(STEP, 1))

    with pytest.raises(FormatError, match="index objects are missing"):
        ckpt.load_object(storage, step=STEP, device=DEVICE)


def test_restore_reports_missing_rank_commit(state, tmp_path, storage, s3_config) -> None:
    """L3 声明了 rank 1、索引也在，但它的分片提交点没了：按 rank 报出来。

    这一条是「诊断」而不是「放行依据」：完整性始终由版本 COMMIT 决定，分片提交点
    只让报错能精确指向哪个 rank 的对象被外部删过。
    """
    saved = state.step_once()
    upload_ranks(saved, tp=2, local_root=tmp_path / "local", s3_config=s3_config, step=STEP)
    ckpt.finalize_upload(storage, step=STEP, ranks=[0, 1], layout={"tp": 2})

    storage.delete(rank_commit_key(STEP, 1))

    with pytest.raises(FormatError, match=r"rank\(s\) \[1\]"):
        ckpt.load_object(storage, step=STEP, device=DEVICE)


def test_missing_data_object_is_caught_at_restore(state, tmp_path, storage, s3_config) -> None:
    """rank 0 只信分片提交点、不读数据；真缺了对象由恢复端的 Range GET 抓。

    这是「提交者只证明齐全、读的人负责一致性」的边界：rank 0 不逐个对象去 HEAD，
    那是 O(对象数) 的额外往返，而且它本来就不知道每个 rank 写了哪些文件。
    """
    saved = state.step_once()
    upload_ranks(saved, tp=2, local_root=tmp_path / "local", s3_config=s3_config, step=STEP)
    ckpt.finalize_upload(storage, step=STEP, ranks=[0, 1], layout={"tp": 2})

    storage.delete(data_key(STEP, 1, "__1_0.distcp"))

    with pytest.raises((ObjectNotFoundError, StorageError)):
        ckpt.load_object(storage, step=STEP, device=DEVICE)


# --- 版本推进 --------------------------------------------------------------


def test_default_step_is_the_largest_commit(state, tmp_path, storage, s3_config) -> None:
    first = state.step_once()
    second = first.step_once()

    for step, snapshot in ((1, first), (2, second)):
        upload_ranks(
            snapshot, tp=2, local_root=tmp_path / f"local-{step}", s3_config=s3_config, step=step
        )
        ckpt.finalize_upload(storage, step=step, ranks=[0, 1], layout={"tp": 2})

    newest = ckpt.load_object(storage, device=DEVICE)
    assert newest.step == 2
    sim.assert_state_matches(
        {name: tensor.numpy() for name, tensor in newest.tensors.items()}, second
    )

    old = ckpt.load_object(storage, step=1, device=DEVICE)
    assert old.step == 1
    sim.assert_state_matches(
        {name: tensor.numpy() for name, tensor in old.tensors.items()}, first
    )


def test_unfinalized_version_is_invisible_until_commit(
    state, tmp_path, storage, s3_config
) -> None:
    """全部 rank 都传完了但 rank 0 还没提交：恢复端必须看不到这一版。"""
    saved = state.step_once()
    upload_ranks(saved, tp=2, local_root=tmp_path / "local", s3_config=s3_config, step=STEP)

    with pytest.raises(CheckpointError):
        ckpt.load_object(storage, step=STEP, device=DEVICE)

    ckpt.finalize_upload(storage, step=STEP, ranks=[0, 1], layout={"tp": 2})
    assert ckpt.load_object(storage, step=STEP, device=DEVICE).step == STEP


# --- 方案 B：整文件下载 + 本地块校验 ----------------------------------------


MiB = 1 << 20


def make_large_tensor(name: str, array: np.ndarray) -> Tensor:
    backend = get_backend(CPU)
    payload = np.ascontiguousarray(array).tobytes()
    buffer = backend.allocate(len(payload))
    backend.copy_h2d(buffer, memoryview(payload))
    return Tensor(
        name=name,
        buffer=buffer,
        device=CPU,
        dtype="f32",
        shape=array.shape,
        backend=backend,
    )


def test_large_object_restore_downloads_and_verifies_chunks(tmp_path, storage) -> None:
    array = np.arange(3 * MiB, dtype=np.float32)  # 12 MiB -> 3 个 4 MiB 块
    local = tmp_path / "local"
    ckpt.save_local({"big": make_large_tensor("big", array)}, local, step=1)
    ckpt.upload_local(local, storage, step=1, rank=0)
    ckpt.finalize_upload(storage, step=1, ranks=[0], layout={})

    report = ckpt.load_object(storage, step=1, device="cpu")

    assert report.download_bytes > 0
    assert report.crc_verified is True
    assert report.verified_blocks == 12
    assert report.io_requests == 1
    assert report.unverified == ()
    assert report.tensors["big"].numpy().tobytes() == array.tobytes()
    assert report.local_staging is not None
    assert not Path(report.local_staging).exists(), "临时目录应在返回前清理"


def test_object_restore_detects_middle_chunk_corruption(tmp_path, storage) -> None:
    array = np.arange(3 * MiB, dtype=np.float32)
    local = tmp_path / "local"
    ckpt.save_local({"big": make_large_tensor("big", array)}, local, step=1)
    ckpt.upload_local(local, storage, step=1, rank=0)
    ckpt.finalize_upload(storage, step=1, ranks=[0], layout={})

    key = data_key(1, 0, "__0_0.distcp")
    payload = bytearray(storage.read(key))
    payload[4 * MiB] ^= 0xFF  # 第二块开头，长度不变
    storage.write(key, bytes(payload))

    with pytest.raises(ChecksumError, match="crc32c"):
        ckpt.load_object(storage, step=1, device="cpu")

    # 显式关掉校验仍能下载、能读
    unchecked = ckpt.load_object(storage, step=1, device="cpu", verify_crc=False)
    assert unchecked.step == 1
