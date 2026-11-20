"""阶段一测试：零冲突恢复底座（R1 / R2 / R3 / R4 / R8 / R10 / R11 / R12）。"""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import time
from pathlib import Path

import numpy as np
import pytest

import api as ckpt
import s3_stub
import sim
from async_ import Tensor, get_backend
from commit.local import COMMIT_FILE, CommitRecord, version_key, write_commit, write_metadata
from core import ChecksumError, FormatError
from core.index import META_FILE, BlobMeta, CheckpointIndex
from core.metadata import CPU
from storage.base import ObjectNotFoundError, StorageError
from storage.local import LocalStorage
from storage.object import S3Config, S3Storage
from storage.upload import (
    RankCommit,
    STAGING_DIR_PREFIX,
    STAGING_LOCK_FILE,
    _acquire_staging_dir,
    _release_staging_dir,
    cleanup_stale_staging_dirs,
    data_key,
    index_key,
    open_object_version,
    rank_commit_key,
    status_key,
)

DEVICE = "cpu"


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
        "checksum_algorithm": None,
    }


@pytest.fixture
def storage(s3_config) -> S3Storage:
    client = S3Storage(S3Config(**s3_config))
    yield client
    client.close()


def make_tensor(name: str, array: np.ndarray) -> Tensor:
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


def save_upload_finalize(
    state: sim.TrainingState,
    tmp_path: Path,
    storage: S3Storage,
    *,
    step: int = 1,
    tp: int = 1,
    blobs_fn=None,
) -> tuple[str, Path]:
    dims = sim.column_dims(state) if tp > 1 else {}
    gen = ckpt.begin_upload(storage, step=step, ranks=list(range(tp)))
    for rank in range(tp):
        rank_root = tmp_path / f"local-r{rank}"
        tensors = (
            sim.tensors_for_rank(state, rank=rank, tp=tp, dims=dims, device=DEVICE)
            if tp > 1
            else sim.to_device(state, device=DEVICE)
        )
        sources = sim.sources_for_rank(state, rank=rank, tp=tp, dims=dims) if tp > 1 else None
        blobs = (
            blobs_fn(rank)
            if blobs_fn is not None
            else ({f"rng/rank_{rank}": bytes([rank + 1]) * 32} if rank == 0 else {})
        )
        ckpt.save_local(
            tensors,
            rank_root,
            step=step,
            rank=rank,
            sources=sources,
            blobs=blobs,
            generation=gen,
        )
        ckpt.upload_local(rank_root, storage, step=step, rank=rank, generation=gen)
    ckpt.finalize_upload(
        storage, step=step, ranks=list(range(tp)), generation=gen, layout={"tp": tp}
    )
    return gen.id, tmp_path / "local-r0"


# --- R1：远端下载后的文件大小核对 --------------------------------------------


def test_r1_truncated_remote_object_rejected_even_with_verify_crc_off(
    state, tmp_path, storage, s3_server
) -> None:
    """远端对象被截断但 STATUS/RankCommit 完好：关掉服务端校验和本地 CRC 也必须拦截。"""
    saved = state.step_once()
    gen_id, _ = save_upload_finalize(saved, tmp_path, storage, step=1)
    obj_key = storage.object_key(data_key(1, 0, "__0_0.distcp", gen_id))
    with s3_server.store.lock:
        s3_server.store.objects[obj_key] = s3_server.store.objects[obj_key][:-128]

    with pytest.raises(FormatError, match="downloaded .* expects"):
        ckpt.load_object(storage, step=1, device=DEVICE, verify_crc=False)


def test_r1_missing_file_in_rank_commit_manifest_rejected(
    state, tmp_path, storage
) -> None:
    """合并索引引用的文件若不在 RankCommit.files 清单内，open_object_version 直接拒。"""
    saved = state.step_once()
    gen_id, _ = save_upload_finalize(saved, tmp_path, storage, step=1)
    rc_key = rank_commit_key(1, 0, gen_id)
    rc = RankCommit.loads(storage.read(rc_key).decode("utf-8"))
    forged = RankCommit(
        step=rc.step,
        rank=rc.rank,
        metadata_bytes=rc.metadata_bytes,
        metadata_crc32c=rc.metadata_crc32c,
        files={},
        generation=rc.generation,
        seq=rc.seq,
    )
    storage.write(rc_key, forged.dumps().encode("utf-8"))

    with pytest.raises(FormatError, match="files 清单缺少"):
        open_object_version(storage, step=1)


def test_r1_temp_commit_uses_manifest_expected_sizes(
    state, tmp_path, storage, monkeypatch
) -> None:
    """写临时本地 COMMIT 时使用 RankCommit 声明的预期大小。"""
    import storage.upload as upload_mod

    saved = state.step_once()
    gen_id, _ = save_upload_finalize(saved, tmp_path, storage, step=1)
    expected_rc = RankCommit.loads(
        storage.read(rank_commit_key(1, 0, gen_id)).decode("utf-8")
    )

    recorded: list[CommitRecord] = []
    real_write_commit = upload_mod.write_commit

    def spy_write_commit(local_storage, record: CommitRecord) -> None:
        recorded.append(record)
        real_write_commit(local_storage, record)

    monkeypatch.setattr(upload_mod, "write_commit", spy_write_commit)
    ckpt.load_object(storage, step=1, device=DEVICE)

    assert len(recorded) == 1
    assert recorded[0].files == {
        f"rank_0/{name}": size for name, size in expected_rc.files.items()
    }


# --- R10：小状态读取强制分块 CRC 校验 ---------------------------------------


def test_r10_blob_corruption_detected_by_crc_and_bypassed_when_off(
    state, tmp_path
) -> None:
    """仅篡改数据文件中某小状态区间的 1 字节，开启校验必报错，关闭才放行。"""
    saved = state.step_once()
    local = tmp_path / "local"
    blob_payload = b"S" * 64
    report = ckpt.save_local(
        sim.to_device(saved, device=DEVICE),
        local,
        step=1,
        blobs={"lr_scheduler": blob_payload},
    )
    blob = report.index.blobs["lr_scheduler"]
    data_path = local / version_key(1) / blob.file
    raw = bytearray(data_path.read_bytes())
    raw[blob.byte_offset + 5] ^= 0x01
    data_path.write_bytes(bytes(raw))

    with pytest.raises(ChecksumError, match="blob 'lr_scheduler'"):
        ckpt.load_local(local, step=1, device=DEVICE, verify_crc=True)

    unchecked = ckpt.load_local(local, step=1, device=DEVICE, verify_crc=False)
    assert unchecked.blobs["lr_scheduler"] != blob_payload


def test_r10_missing_blob_crc_is_rejected_without_legacy_compat(
    state, tmp_path
) -> None:
    """不做旧版无 CRC 数据兼容：小状态缺失分块 CRC 时开启校验直接报 FormatError。"""
    saved = state.step_once()
    local = tmp_path / "local"
    report = ckpt.save_local(
        sim.to_device(saved, device=DEVICE),
        local,
        step=1,
        blobs={"lr_scheduler": b"L" * 32},
    )
    old_blob = report.index.blobs["lr_scheduler"]
    stripped_index = CheckpointIndex(
        tensors=report.index.tensors,
        blobs={
            "lr_scheduler": BlobMeta(
                file=old_blob.file,
                byte_offset=old_blob.byte_offset,
                length=old_blob.length,
            )
        },
        step=1,
    )
    storage = LocalStorage(local)
    metadata = write_metadata(storage, 1, stripped_index)
    write_commit(
        storage,
        CommitRecord.of_metadata(step=1, metadata=metadata, files=report.commit.files),
    )

    with pytest.raises(FormatError, match="missing crc_chunk_bytes"):
        ckpt.load_local(local, step=1, device=DEVICE, verify_crc=True)

    ok = ckpt.load_local(local, step=1, device=DEVICE, verify_crc=False)
    assert ok.blobs["lr_scheduler"] == b"L" * 32


# --- R11：远端分片三坐标断言与同名小状态冲突 --------------------------------


def test_r11_rejects_cross_step_transplanted_rank_metadata(
    state, tmp_path, storage
) -> None:
    """将 step_1 的 rank 索引/提交点拷到 step_2 同名代际下，必须在读数据前拦截。"""
    s1 = state.step_once()
    s2 = s1.step_once()
    gen_id = "gen_0001-aaaa1111"
    # 手动固定同一代际名存入 step 1 与 step 2
    for step, st in ((1, s1), (2, s2)):
        gen = ckpt.begin_upload(storage, step=step, ranks=[0], writer="aaaa1111")
        assert gen.id == gen_id
        rank_root = tmp_path / f"local-{step}"
        ckpt.save_local(sim.to_device(st, device=DEVICE), rank_root, step=step, generation=gen)
        ckpt.upload_local(rank_root, storage, step=step, rank=0, generation=gen)
        ckpt.finalize_upload(storage, step=step, ranks=[0], generation=gen)

    # 把 step 1 的 rank_0/COMMIT 与 .metadata 拷到 step 2 同名代际下
    storage.write(
        rank_commit_key(2, 0, gen_id), storage.read(rank_commit_key(1, 0, gen_id))
    )
    storage.write(index_key(2, 0, gen_id), storage.read(index_key(1, 0, gen_id)))

    with pytest.raises(FormatError, match="声明 step 1"):
        open_object_version(storage, step=2)


def test_r11_rejects_duplicate_blob_names_across_ranks(
    state, tmp_path, storage
) -> None:
    """两个 rank 声明同名小状态必须在合并索引时显式报错，不再静默取最小 rank。"""
    saved = state.step_once()
    save_upload_finalize(
        saved,
        tmp_path,
        storage,
        step=1,
        tp=2,
        blobs_fn=lambda rank: {"lr_scheduler": bytes([rank]) * 16},
    )

    with pytest.raises(FormatError, match="同名小状态 'lr_scheduler'"):
        open_object_version(storage, step=1)


# --- R12：直接加载未提交的 step_N 目录 --------------------------------------


def test_r12_direct_load_of_uncommitted_step_dir_reports_commit_missing(
    state, tmp_path
) -> None:
    """直接加载缺 COMMIT 的 step_N 目录时明确报 step_N: COMMIT missing。"""
    saved = state.step_once()
    local = tmp_path / "local"
    ckpt.save_local(sim.to_device(saved, device=DEVICE), local, step=5)
    step_dir = local / version_key(5)
    (step_dir / COMMIT_FILE).unlink()

    with pytest.raises(FormatError, match=r"step_5: COMMIT missing"):
        ckpt.load(step_dir)

    # 即使连 .metadata 也没有的空 step_5 目录同样明确报错
    (step_dir / META_FILE).unlink()
    with pytest.raises(FormatError, match=r"step_5: COMMIT missing"):
        ckpt.load(step_dir)


# --- R2：单文件下载原子落盘与退避重试 --------------------------------------


def test_r2_single_file_retries_503_and_stream_drop_without_redownloading_healthy_peer(
    state, tmp_path, storage, s3_server
) -> None:
    """注入前 2 次 503/断流：单文件退避重试恢复成功，且不重下已成功的并发文件。"""
    saved = state.step_once()
    gen_id, _ = save_upload_finalize(saved, tmp_path, storage, step=1, tp=2)

    key0 = storage.object_key(data_key(1, 0, "__0_0.distcp", gen_id))
    key1 = storage.object_key(data_key(1, 1, "__1_0.distcp", gen_id))

    with s3_server.store.lock:
        s3_server.store.fail_get_queue[key0] = [503]
        s3_server.store.drop_get_stream[key0] = 1
        s3_server.store.requests.clear()

    report = ckpt.load_object(
        storage,
        step=1,
        device=DEVICE,
        io_concurrency=2,
        download_retries=3,
        download_backoff=0.001,
    )
    assert report.step == 1
    gets_key0 = [r for r in s3_server.store.requests if r[0] == "GET" and r[1].endswith(key0)]
    gets_key1 = [r for r in s3_server.store.requests if r[0] == "GET" and r[1].endswith(key1)]
    assert len(gets_key0) == 3  # 503 + 断流 + 成功
    assert len(gets_key1) == 1, "并发的其他健康文件绝不重复下载"


def test_r2_non_retryable_error_fails_fast(
    state, tmp_path, storage, s3_server
) -> None:
    """不可重试错误（404/403）整批快速失败，不做重试。"""
    saved = state.step_once()
    gen_id, _ = save_upload_finalize(saved, tmp_path, storage, step=1)
    key0 = storage.object_key(data_key(1, 0, "__0_0.distcp", gen_id))
    with s3_server.store.lock:
        s3_server.store.fail_get_queue[key0] = [403, 200]
        s3_server.store.requests.clear()

    with pytest.raises(StorageError, match="403|Forbidden|InjectedError"):
        ckpt.load_object(
            storage, step=1, device=DEVICE, download_retries=3, download_backoff=0.001
        )
    gets = [r for r in s3_server.store.requests if r[0] == "GET" and r[1].endswith(key0)]
    assert len(gets) == 1


# --- R3：下载落盘后本地校验损坏的定向重下 ----------------------------------


def test_r3_transient_wire_corruption_triggers_single_targeted_redownload(
    state, tmp_path, storage, s3_server
) -> None:
    """首次下载传输层改 1 字节、二次正常 → 定向重下出错文件 1 次并自动恢复。"""
    saved = state.step_once()
    gen_id, _ = save_upload_finalize(saved, tmp_path, storage, step=1, tp=2)
    key0 = storage.object_key(data_key(1, 0, "__0_0.distcp", gen_id))
    key1 = storage.object_key(data_key(1, 1, "__1_0.distcp", gen_id))

    with s3_server.store.lock:
        s3_server.store.corrupt_get_once[key0] = 1
        s3_server.store.requests.clear()

    report = ckpt.load_object(storage, step=1, device=DEVICE, verify_crc=True)

    assert report.redownloaded == ("rank_0/__0_0.distcp",)
    gets_key0 = [r for r in s3_server.store.requests if r[0] == "GET" and r[1].endswith(key0)]
    gets_key1 = [r for r in s3_server.store.requests if r[0] == "GET" and r[1].endswith(key1)]
    assert len(gets_key0) == 2, "出错文件只重下 1 次"
    assert len(gets_key1) == 1, "未出错文件不重下"


def test_r3_permanent_remote_corruption_stops_after_one_redownload(
    state, tmp_path, storage, s3_server
) -> None:
    """远端本身损坏 → 定向重下 1 次仍失败后停止重试并报 ChecksumError。"""
    saved = state.step_once()
    gen_id, _ = save_upload_finalize(saved, tmp_path, storage, step=1)
    key0 = storage.object_key(data_key(1, 0, "__0_0.distcp", gen_id))

    with s3_server.store.lock:
        mutated = bytearray(s3_server.store.objects[key0])
        mutated[0] ^= 0xFF
        s3_server.store.objects[key0] = bytes(mutated)
        s3_server.store.requests.clear()

    with pytest.raises(ChecksumError):
        ckpt.load_object(storage, step=1, device=DEVICE, verify_crc=True)

    gets = [r for r in s3_server.store.requests if r[0] == "GET" and r[1].endswith(key0)]
    assert len(gets) == 2, "初下 1 次 + 定向重下 1 次后即停止"


# --- R4：SIGKILL/OOM 残留临时目录的安全清理 ---------------------------------


def _hold_staging_and_block(staging_parent: str, ready_q) -> None:
    staging_dir, fd = _acquire_staging_dir(staging_parent)
    ready_q.put(str(staging_dir))
    while True:
        time.sleep(1.0)


def test_r4_cleanup_removes_sigkilled_staging_dir_and_keeps_live_sibling(
    tmp_path
) -> None:
    """子进程被 SIGKILL 后残留目录被清理，同时不误删存活兄弟进程的目录与无锁目录。"""
    staging_parent = tmp_path / "staging"
    staging_parent.mkdir()

    ctx = multiprocessing.get_context("spawn")
    q_dead = ctx.Queue()
    q_live = ctx.Queue()
    dead_proc = ctx.Process(target=_hold_staging_and_block, args=(str(staging_parent), q_dead))
    live_proc = ctx.Process(target=_hold_staging_and_block, args=(str(staging_parent), q_live))
    dead_proc.start()
    live_proc.start()
    try:
        dead_dir = Path(q_dead.get(timeout=10))
        live_dir = Path(q_live.get(timeout=10))

        # 没有锁文件的同名前缀目录不应被清理
        foreign_dir = staging_parent / f"{STAGING_DIR_PREFIX}foreign"
        foreign_dir.mkdir()

        os.kill(dead_proc.pid, signal.SIGKILL)
        dead_proc.join(timeout=10)
        assert dead_proc.exitcode == -signal.SIGKILL

        cleaned = cleanup_stale_staging_dirs(staging_parent)
        assert dead_dir in cleaned
        assert not dead_dir.exists()
        assert live_dir.is_dir() and (live_dir / STAGING_LOCK_FILE).is_file()
        assert foreign_dir.is_dir()
    finally:
        if live_proc.is_alive():
            live_proc.kill()
            live_proc.join(timeout=10)


# --- R8：坏 STATUS 跳过开关 ------------------------------------------------


def test_r8_skip_corrupt_status_only_when_enabled_and_step_omitted(
    state, tmp_path, storage
) -> None:
    """默认严格报 FormatError；开容错且未指定 step 时告警跳过，精确指定 step 绝不跳过。"""
    s1 = state.step_once()
    s2 = s1.step_once()
    save_upload_finalize(s1, tmp_path / "s1", storage, step=1)
    save_upload_finalize(s2, tmp_path / "s2", storage, step=2)

    storage.write(status_key(2), b"{corrupt-status")

    # 默认严格模式
    with pytest.raises(FormatError, match="STATUS"):
        ckpt.load_object(storage, device=DEVICE)

    # 开启容错 + step=None：告警并跳过 step 2，恢复 step 1
    with pytest.warns(RuntimeWarning, match="跳过损坏的 STATUS"):
        report = ckpt.load_object(storage, device=DEVICE, skip_corrupt_status=True)
    assert report.step == 1
    assert len(report.skipped_versions) == 1
    assert report.skipped_versions[0][0] == 2

    # 精确指定 step=2 时绝不跳过
    with pytest.raises(FormatError, match="STATUS"):
        ckpt.load_object(storage, step=2, device=DEVICE, skip_corrupt_status=True)
