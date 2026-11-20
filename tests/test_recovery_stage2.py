"""阶段二测试：集群恢复闭环、本地热副本、轮次隔离与残留 MPU 清理（R5/R6/R7/R9/R13/R14/R16a）。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

import api as ckpt
import dist
import s3_stub
import sim
from commit.gc import collect, cleanup_stale_multipart_uploads
from commit.local import GENERATION_FILE, read_local_generation, version_key, write_local_generation
from core import ChecksumError, FormatError, InvalidTargetError
from storage.local import LocalStorage
from storage.object import S3Config, S3Storage
from storage.upload import (
    GENERATION_MISMATCH,
    SOURCE_LOCAL,
    SOURCE_OBJECT,
    data_key,
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
    }


@pytest.fixture
def storage(s3_config) -> S3Storage:
    client = S3Storage(S3Config(**s3_config))
    yield client
    client.close()


def _save_upload_commit(
    state: sim.TrainingState,
    local_root: Path,
    storage: S3Storage,
    *,
    step: int,
    tp: int = 1,
    run_id: str | None = None,
    keep: int | None = 3,
) -> str:
    dims = sim.column_dims(state) if tp > 1 else {}
    gen = ckpt.begin_upload(
        storage, step=step, ranks=list(range(tp)), run_id=run_id
    )
    for rank in range(tp):
        r_root = local_root / f"rank_{rank}" if tp > 1 else local_root
        tensors = (
            sim.tensors_for_rank(state, rank=rank, tp=tp, dims=dims, device=DEVICE)
            if tp > 1
            else sim.to_device(state, device=DEVICE)
        )
        sources = (
            sim.sources_for_rank(state, rank=rank, tp=tp, dims=dims) if tp > 1 else None
        )
        ckpt.save_local(
            tensors,
            r_root,
            step=step,
            rank=rank,
            sources=sources,
            blobs={f"rng/rank_{rank}": bytes([rank + 1]) * 16} if rank == 0 else {},
            generation=gen,
            run_id=run_id,
        )
        ckpt.upload_local(
            r_root, storage, step=step, rank=rank, generation=gen, run_id=run_id
        )
    ckpt.finalize_upload(
        storage,
        step=step,
        ranks=list(range(tp)),
        generation=gen,
        layout={"tp": tp},
        keep=keep,
        run_id=run_id,
    )
    return gen.id


# --- R5 + R6：本地代际标记与热副本损坏自动回退 -------------------------------


def test_r6_generation_marker_mismatch_forces_remote_download(
    state, tmp_path, storage
) -> None:
    """R6：本地代际标记与集群锁定代际不一致时判过期，自动走远端下载。"""
    s1 = state.step_once()
    s2 = s1.step_once()
    local = tmp_path / "local"
    gen1 = _save_upload_commit(s1, local, storage, step=1)
    assert (local / version_key(1) / GENERATION_FILE).is_file()
    marker = read_local_generation(LocalStorage(local), 1)
    assert marker is not None and marker.generation == gen1

    # 本地重新写了同 step（例如新尝试覆盖了本地目录），标记变成另一代但远端仍提交 gen1
    ckpt.save_local(sim.to_device(s2, device=DEVICE), local, step=1)
    write_local_generation(LocalStorage(local), 1, "gen_0099-deadbeef")

    check = ckpt.local_copy_available(local, storage, step=1)
    assert not check.usable and check.reason == GENERATION_MISMATCH

    report = ckpt.load_checkpoint(storage, local_root=local, step=1, device=DEVICE)
    assert report.source == SOURCE_OBJECT
    assert report.local_fallback_reason == GENERATION_MISMATCH
    assert report.generation == gen1
    sim.assert_state_matches(
        {name: t.numpy() for name, t in report.tensors.items()}, s1
    )


def test_r5_corrupted_local_hot_copy_warns_cleans_and_falls_back_to_remote(
    state, tmp_path, storage
) -> None:
    """R5：本地数据文件损坏时默认告警、清理坏目录并自动回退远端下载；可显式关闭。"""
    s1 = state.step_once()
    local = tmp_path / "local"
    _save_upload_commit(s1, local, storage, step=1)

    data_path = local / version_key(1) / "__0_0.distcp"
    raw = bytearray(data_path.read_bytes())
    raw[0] ^= 0xFF
    data_path.write_bytes(bytes(raw))

    # 关闭回退用于硬件排查：直接抛 ChecksumError，不静默下载
    with pytest.warns(RuntimeWarning, match="读取本地热副本失败"):
        with pytest.raises(ChecksumError):
            ckpt.load_checkpoint(
                storage,
                local_root=local,
                step=1,
                device=DEVICE,
                fallback_on_local_error=False,
            )

    # 默认开启回退：告警 + 清理坏目录 + 自动改走远端下载 + 记录回退原因
    with pytest.warns(RuntimeWarning, match="读取本地热副本失败"):
        report = ckpt.load_checkpoint(storage, local_root=local, step=1, device=DEVICE)

    assert report.source == SOURCE_OBJECT
    assert report.download_bytes > 0
    assert report.local_fallback_reason is not None
    assert "ChecksumError" in report.local_fallback_reason
    assert not (local / version_key(1)).exists(), "损坏的本地热副本目录应被清理"
    sim.assert_state_matches(
        {name: t.numpy() for name, t in report.tensors.items()}, s1
    )


# --- R7 + R13：集群选版广播 + 本地/远端混合加载 + 全归约复核 ----------------


class _ListCountingBackend:
    """记录每个 rank 在恢复选版期间是否主动 list 远端（R7 断言非 0 rank 不探查）。"""

    def __init__(self, inner: S3Storage) -> None:
        self._inner = inner
        self.list_calls: list[str] = []

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def list(self, prefix: str = "") -> list[str]:
        self.list_calls.append(prefix)
        return self._inner.list(prefix)


def test_r7_r13_cluster_recovery_only_rank0_probes_and_consensus_passes(
    state, tmp_path, s3_config
) -> None:
    """R7+R13：仅 rank 0 探查远端选版并广播；各 rank 各自加载后 0/1/2 全归约放行。"""
    s1 = state.step_once()
    writer = S3Storage(S3Config(**s3_config))
    try:
        gen_id = _save_upload_commit(s1, tmp_path / "src", writer, step=1, tp=2)
    finally:
        writer.close()

    # 准备一个完整的单机热副本给 rank 0（使其本地命中），rank 1 不给本地目录（走远端下载）
    local_r0 = tmp_path / "hot-r0"
    writer2 = S3Storage(S3Config(**s3_config))
    try:
        # 把 step 1 也存一份完整副本在 local_r0 下并打上 gen_id 标记
        ckpt.save_local(
            sim.to_device(s1, device=DEVICE), local_r0, step=1, generation=gen_id
        )
    finally:
        writer2.close()

    links = dist.ThreadedCollective.linked(2)
    clients = [_ListCountingBackend(S3Storage(S3Config(**s3_config))) for _ in range(2)]
    local_roots = [local_r0, tmp_path / "empty-r1"]

    def worker(rank: int):
        return ckpt.load_checkpoint(
            clients[rank],
            local_root=local_roots[rank],
            device=DEVICE,
            collective=links[rank],
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            reports = [f.result(timeout=30) for f in [pool.submit(worker, r) for r in range(2)]]
    finally:
        for c in clients:
            c._inner.close()

    assert reports[0].source == SOURCE_LOCAL
    assert reports[1].source == SOURCE_OBJECT
    assert reports[0].step == reports[1].step == 1
    assert reports[0].generation == reports[1].generation == gen_id
    # rank 1 没有自己调用 list("") 扫远端选版（只在大文件 open_object_version 时 list 指定 generation 前缀）
    assert "" not in clients[1].list_calls
    assert "" in clients[0].list_calls


def test_r9_r13_cluster_unified_fallback_only_when_enabled_and_step_omitted(
    state, tmp_path, s3_config
) -> None:
    """R9：版本损坏时默认不回退；开启且未指定 step 时全员统一回退到上一个好版本。"""
    s1 = state.step_once()
    s2 = s1.step_once()
    writer = S3Storage(S3Config(**s3_config))
    try:
        _save_upload_commit(s1, tmp_path / "s1", writer, step=1, tp=2)
        gen2 = _save_upload_commit(s2, tmp_path / "s2", writer, step=2, tp=2)
        # 破坏 step 2 中 rank 1 的数据文件（导致只有 rank 1 报错，rank 0 自己能读）
        bad_key = data_key(2, 1, "__1_0.distcp", gen2)
        raw = bytearray(writer.read(bad_key))
        raw[0] ^= 0xFF
        writer.write(bad_key, bytes(raw))
    finally:
        writer.close()

    def run_cluster(*, allow_fallback: bool, step: int | None):
        links = dist.ThreadedCollective.linked(2)
        clients = [S3Storage(S3Config(**s3_config)) for _ in range(2)]
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futs = [
                    pool.submit(
                        ckpt.load_checkpoint,
                        clients[r],
                        step=step,
                        device=DEVICE,
                        allow_fallback=allow_fallback,
                        collective=links[r],
                    )
                    for r in range(2)
                ]
                return [f.result(timeout=30) for f in futs]
        finally:
            for c in clients:
                c.close()

    # 1. 默认关闭回退：哪怕 rank 0 自己没读到坏文件，全员也必须一致报错退出
    with pytest.raises((ChecksumError, FormatError)):
        run_cluster(allow_fallback=False, step=None)

    # 2. 精确指定 step=2：即使开启 allow_fallback=True 也绝不回退
    with pytest.raises((ChecksumError, FormatError)):
        run_cluster(allow_fallback=True, step=2)

    # 3. 显式开启 allow_fallback=True 且 step=None：全员统一放弃 step 2，回退到 step 1
    reports = run_cluster(allow_fallback=True, step=None)
    assert [r.step for r in reports] == [1, 1]
    assert reports[0].skipped_versions[0][0] == 2
    assert reports[1].skipped_versions[0][0] == 2
    sim.assert_state_matches(
        {name: t.numpy() for name, t in reports[0].tensors.items()}, s1
    )


def test_r13_fatal_error_never_triggers_fallback(
    state, tmp_path, s3_config
) -> None:
    """R13：不可回退致命错误（如目标布局非法）状态码 2，开启回退也必须全员一致报错。"""
    s1 = state.step_once()
    s2 = s1.step_once()
    writer = S3Storage(S3Config(**s3_config))
    try:
        _save_upload_commit(s1, tmp_path / "s1", writer, step=1)
        _save_upload_commit(s2, tmp_path / "s2", writer, step=2)
    finally:
        writer.close()

    links = dist.ThreadedCollective.linked(2)
    clients = [S3Storage(S3Config(**s3_config)) for _ in range(2)]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            # rank 1 传入非法 shard_dims -> InvalidTargetError (code 2)
            f0 = pool.submit(
                ckpt.load_checkpoint,
                clients[0],
                device=DEVICE,
                allow_fallback=True,
                collective=links[0],
            )
            f1 = pool.submit(
                ckpt.load_checkpoint,
                clients[1],
                device=DEVICE,
                shard_dims={"nonexistent.tensor": 0},
                allow_fallback=True,
                collective=links[1],
            )
            with pytest.raises(InvalidTargetError):
                f1.result(timeout=30)
            with pytest.raises(Exception):
                f0.result(timeout=30)
    finally:
        for c in clients:
            c.close()


# --- R14：轮次隔离 -----------------------------------------------------------


def test_r14_run_isolation_prevents_reading_old_run_higher_step(
    state, tmp_path, storage
) -> None:
    """轮次 1 存 step 1..3 → 开轮次 2 从轮次 1 的 step 1 恢复并存 step 2 → 轮次 2 重启必须落到 step 2。"""
    s1 = state.step_once()
    s2 = s1.step_once()
    s3 = s2.step_once()
    local_root = tmp_path / "local"

    for step, st in ((1, s1), (2, s2), (3, s3)):
        _save_upload_commit(st, local_root, storage, step=step, run_id="run_0001")

    # 开轮次 2，显式从 run_0001 的 step 1 恢复并重训得到新分叉状态
    fork_base = ckpt.load_checkpoint(
        storage,
        local_root=local_root,
        step=1,
        device=DEVICE,
        run_id="run_0002",
        source_run_id="run_0001",
    )
    assert fork_base.step == 1
    assert fork_base.run_id == "run_0001"
    sim.assert_state_matches(
        {name: t.numpy() for name, t in fork_base.tensors.items()}, s1
    )

    # 轮次 2 的新状态与轮次 1 的 step 2 故意不同，保存为 run_0002/step_2
    s2_branch = sim.state_from_tensors(fork_base.tensors, step=1).step_once().step_once()
    _save_upload_commit(s2_branch, local_root, storage, step=2, run_id="run_0002")

    # 轮次 2 自动恢复：绝不能读到 run_0001 的 step 3，也绝不能读到 run_0001 的 step 2
    resumed = ckpt.load_checkpoint(
        storage, local_root=local_root, device=DEVICE, run_id="run_0002"
    )
    assert resumed.step == 2
    assert resumed.run_id == "run_0002"
    sim.assert_state_matches(
        {name: t.numpy() for name, t in resumed.tensors.items()}, s2_branch
    )

    # 历史轮次 run_0001 的 step 1..3 完好保留，未被硬删
    assert ckpt.load_object(storage, step=3, device=DEVICE, run_id="run_0001").step == 3


# --- R16a：远端残留分片上传（MPU）超时自动清理 ------------------------------


def test_r16a_gc_aborts_timed_out_and_stale_multipart_uploads(
    state, tmp_path, storage, s3_server
) -> None:
    """R16a：超时或已失效代际的未完成 MPU 被自动中止，当前活跃的未超时 MPU 保留。"""
    s1 = state.step_once()
    local = tmp_path / "local"
    _save_upload_commit(s1, local, storage, step=2)

    # 造 3 个未完成 MPU：
    # 1) step_1 (已被 step_2 superseded)
    stale_key = storage.object_key("step_1/gen_0001-aaaaaaaa/rank_0/__0_0.distcp")
    # 2) step_3 当前代但超时（Initiated 设为 2 小时前）
    expired_key = storage.object_key("step_3/gen_0001-bbbbbbbb/rank_0/__0_0.distcp")
    # 3) step_3 当前代刚发起（未超时）
    fresh_key = storage.object_key("step_3/gen_0001-bbbbbbbb/rank_1/__1_0.distcp")

    uid_stale = storage.client.create_multipart_upload(
        Bucket="ckpt", Key=stale_key
    )["UploadId"]
    uid_expired = storage.client.create_multipart_upload(
        Bucket="ckpt", Key=expired_key
    )["UploadId"]
    uid_fresh = storage.client.create_multipart_upload(
        Bucket="ckpt", Key=fresh_key
    )["UploadId"]

    with s3_server.store.lock:
        s3_server.store.upload_meta[uid_expired]["Initiated"] = (
            datetime.now(timezone.utc) - timedelta(hours=2)
        )
        s3_server.store.upload_meta[uid_expired]["initiated"] = (
            datetime.now(timezone.utc) - timedelta(hours=2)
        )

    report = collect(
        local,
        storage,
        rank=0,
        dry_run=False,
        mpu_timeout_seconds=1800.0,
    )

    aborted_ids = {item.upload_id for item in report.aborted_uploads}
    assert aborted_ids == {uid_stale, uid_expired}
    remaining = {u.upload_id for u in storage.list_multipart_uploads("")}
    assert remaining == {uid_fresh}, "未超时的活跃 MPU 必须保留"

    # dry_run 模式只给计划不中止
    dry = cleanup_stale_multipart_uploads(
        storage, max_age_seconds=0.0, dry_run=True
    )
    assert {u.upload_id for u in dry} == {uid_fresh}
    assert {u.upload_id for u in storage.list_multipart_uploads("")} == {uid_fresh}
