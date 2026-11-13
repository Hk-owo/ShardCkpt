"""对象存储端的代际（generation）与 fence。

这一层要守住两件事，都是「只有 step 一个维度」时做不到的：

1. **不混代**：同一 step 的两次尝试各传一半，永远凑不出一份「看着完整」的版本；
2. **不越权**：落后的写者写不了自己的 L2、提交不了 L3，已提交版本也不会被重试
   破坏（新代完整之前，恢复端读到的仍是旧代）。

协议规格见 ``docs/generation.md``；CAS 原语见 ``tests/test_conditional.py``。
"""

from __future__ import annotations

import numpy as np
import pytest

import api as ckpt
import s3_stub
from async_ import Tensor, get_backend
from commit.generation import (
    FencedError,
    Generation,
    GenerationPointer,
    acquire_generation,
    next_seq,
    pointer_key,
    verify_holder,
)
from core import ChecksumError, FormatError
from core.index import META_FILE
from core.metadata import CPU
from storage.base import WriteOutcome
from storage.checksum import crc32c_hex
from storage.object import S3Config, S3Storage
from storage.upload import (
    COMMIT_FILE,
    RankCommit,
    UploadCommit,
    commit_key,
    data_key,
    generation_prefix,
    index_key,
    rank_commit_key,
    version_prefix,
)

STEP = 1
DEVICE = "cpu"


@pytest.fixture
def s3_server():
    with s3_stub.start() as server:
        yield server


@pytest.fixture
def storage(s3_server) -> S3Storage:
    client = S3Storage(
        S3Config(
            bucket="ckpt",
            prefix="runs",
            endpoint_url=s3_server.endpoint_url,
            access_key_id="test-key",
            secret_access_key="test-secret",
        )
    )
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


def save_one(local, *, step: int = STEP, value: float = 1.0, rank: int = 0) -> None:
    """存一份本 rank 的产物；每个 rank 都带自己的小状态，计划不会为空。"""
    array = np.full((4, 4), value, dtype=np.float32)
    ckpt.save_local(
        {"w": make_tensor("w", array)},
        local,
        step=step,
        rank=rank,
        blobs={f"rng/rank_{rank}": bytes([rank % 256]) * 8},
    )


def commit_value(storage: S3Storage, *, step: int = STEP) -> float:
    report = ckpt.load_object(storage, step=step, device=DEVICE)
    return float(report.tensors["w"].numpy()[0, 0])


def read_commit(storage: S3Storage, *, step: int = STEP) -> UploadCommit:
    return UploadCommit.loads(storage.read(commit_key(step)).decode("utf-8"))


# --- 纯函数 -----------------------------------------------------------------


def test_generation_id_roundtrips() -> None:
    gen = Generation(step=3, seq=7, writer="1f3a9c2b")

    assert gen.id == "gen_0007-1f3a9c2b"
    assert Generation.parse(gen.id, step=3) == gen
    assert Generation(step=3, seq=7, writer="1f3a9c2b").next().seq == 8


def test_generation_rejects_bad_ids() -> None:
    for bad in ("gen_", "gen_x-1f3a9c2b", "1f3a9c2b", "gen_0001-", "gen_0001-ABC"):
        with pytest.raises(FormatError):
            Generation.parse(bad, step=1)
    with pytest.raises(FormatError):
        Generation(step=1, seq=0, writer="1f3a9c2b")


def test_next_seq_never_goes_backwards() -> None:
    """GEN 被删掉时，已有代目录仍把序号顶住 —— seq 回退会让 L3 的 fence 失效。"""
    assert next_seq([], current=0) == 1
    assert next_seq(["gen_0007-abcd1234/rank_0/.metadata"], current=0) == 8
    assert next_seq(["gen_0007-abcd1234/rank_0/.metadata"], current=11) == 12
    assert next_seq(["step_9/COMMIT", "step_9/GEN"], current=0) == 1


def test_pointer_roundtrip_keeps_state_and_ranks() -> None:
    pointer = GenerationPointer(
        step=2,
        generation=Generation(step=2, seq=4, writer="deadbeef"),
        ranks=(0, 1, 2),
        state="sealed",
        etag='"etag"',
    )

    loaded = GenerationPointer.loads(pointer.dumps(), etag='"etag"')

    assert loaded.generation == pointer.generation
    assert loaded.ranks == (0, 1, 2)
    assert loaded.state == "sealed"
    assert loaded.etag == '"etag"'
    assert loaded.dumps() == pointer.dumps()


# --- acquire：CAS 线性化 ----------------------------------------------------


def test_acquire_creates_first_generation(storage: S3Storage) -> None:
    gen = acquire_generation(storage, STEP, ranks=(0,), writer="aaaaaaaa")

    assert gen.seq == 1
    assert storage.read(pointer_key(STEP)) != b""
    assert verify_holder(storage, gen).state == "open"


def test_acquire_advances_seq_and_fences_the_old_one(storage: S3Storage) -> None:
    first = acquire_generation(storage, STEP, ranks=(0,), writer="aaaaaaaa")
    second = acquire_generation(storage, STEP, ranks=(0,), writer="bbbbbbbb")

    assert second.seq == first.seq + 1
    with pytest.raises(FencedError, match="已被 fence"):
        verify_holder(storage, first)
    assert verify_holder(storage, second).generation == second


def test_acquire_retries_when_someone_else_wins(
    storage: S3Storage, monkeypatch
) -> None:
    """第一次 CAS 输给别人（EXISTS）：重读重试，第二次拿到。"""
    real = storage.put_if_absent
    calls = {"n": 0}

    def flaky(key: str, data: bytes) -> WriteOutcome:
        calls["n"] += 1
        if calls["n"] == 1:
            return WriteOutcome.EXISTS
        return real(key, data)

    monkeypatch.setattr(storage, "put_if_absent", flaky)

    gen = acquire_generation(storage, STEP, ranks=(0,), writer="aaaaaaaa")

    assert gen.seq == 1
    assert calls["n"] == 2


def test_acquire_gives_up_after_bounded_retries(
    storage: S3Storage, monkeypatch
) -> None:
    monkeypatch.setattr(
        storage, "put_if_absent", lambda key, data: WriteOutcome.EXISTS
    )

    with pytest.raises(FencedError, match="CAS 内没拿到写权"):
        acquire_generation(storage, STEP, ranks=(0,), retries=2)


def test_acquire_generation_is_epoch_stable(storage: S3Storage) -> None:
    """同一个 step 连续开代，seq 单调；不同 step 各自从 1 开始。"""
    a = acquire_generation(storage, 1, ranks=(0,))
    b = acquire_generation(storage, 1, ranks=(0,))
    c = acquire_generation(storage, 2, ranks=(0,))

    assert (a.seq, b.seq, c.seq) == (1, 2, 1)


# --- fence：落后写者 --------------------------------------------------------


def test_stale_writer_cannot_write_rank_commit(storage: S3Storage, tmp_path) -> None:
    """被取代的写者拿不到 L2：这一代因此永远凑不齐。"""
    local = tmp_path / "local"
    save_one(local)
    stale = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    ckpt.begin_upload(storage, step=STEP, ranks=[0])  # 新尝试接管

    with pytest.raises(FencedError, match="已被 fence"):
        ckpt.upload_local(local, storage, step=STEP, rank=0, generation=stale)

    assert storage.list(generation_prefix(STEP, stale)) == []


def test_finalize_with_stale_generation_is_rejected(
    storage: S3Storage, tmp_path
) -> None:
    local = tmp_path / "local"
    save_one(local)
    stale = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    ckpt.upload_local(local, storage, step=STEP, rank=0, generation=stale)
    ckpt.begin_upload(storage, step=STEP, ranks=[0])  # 新尝试接管 GEN

    with pytest.raises(FencedError, match="已被 fence"):
        ckpt.finalize_upload(storage, step=STEP, ranks=[0], generation=stale)

    assert commit_key(STEP) not in set(storage.list(""))


def test_finalize_rejects_rank_mismatch(storage: S3Storage, tmp_path) -> None:
    """GEN 声明 ranks [0]，finalize 却传 [0, 1]：配置漂移，直接拒绝。"""
    local = tmp_path / "local"
    save_one(local)
    generation = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    ckpt.upload_local(local, storage, step=STEP, rank=0, generation=generation)

    with pytest.raises(FormatError, match="GEN 声明的 ranks"):
        ckpt.finalize_upload(storage, step=STEP, ranks=[0, 1])


# --- 隔离：同一 step 的两次尝试 --------------------------------------------


def test_retry_does_not_mix_generations(storage: S3Storage, tmp_path) -> None:
    """★ 两次尝试各传一半，绝不能拼成一份「看着完整」的版本。

    改前（只有 step 一个维度）这里会成功，恢复端静默拿到「rank 0 的旧数据 +
    rank 1 的新数据」。代际把两半放进不同前缀，任何一半都过不了齐全性检查。
    """
    first = tmp_path / "local-first"
    second = tmp_path / "local-second"
    save_one(first, value=1.0, rank=0)
    save_one(second, value=2.0, rank=1)

    gen1 = ckpt.begin_upload(storage, step=STEP, ranks=[0, 1])
    ckpt.upload_local(first, storage, step=STEP, rank=0, generation=gen1)
    gen2 = ckpt.begin_upload(storage, step=STEP, ranks=[0, 1])
    ckpt.upload_local(second, storage, step=STEP, rank=1, generation=gen2)

    with pytest.raises(FormatError, match="not all uploaded"):
        ckpt.finalize_upload(storage, step=STEP, ranks=[0, 1], generation=gen2)
    with pytest.raises(FencedError):
        # 旧代申请提交同样不行：它已经被 gen2 fence 掉了
        ckpt.finalize_upload(storage, step=STEP, ranks=[0, 1], generation=gen1)
    assert commit_key(STEP) not in set(storage.list(""))


def test_committed_version_survives_failed_retry(
    storage: S3Storage, tmp_path
) -> None:
    """已提交的 gen1 在 gen2 半途而废之后仍然可恢复、字节不变。"""
    local = tmp_path / "local"
    save_one(local, value=1.0)
    gen1 = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    ckpt.upload_local(local, storage, step=STEP, rank=0, generation=gen1)
    ckpt.finalize_upload(storage, step=STEP, ranks=[0], generation=gen1)
    assert commit_value(storage) == pytest.approx(1.0)

    gen2 = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    # 新尝试只落了一个数据对象就崩了（没有 .metadata、没有 L2、没有 L3）
    storage.put_if_absent(data_key(STEP, 0, "__0_0.distcp", gen2), b"half-written")

    assert commit_value(storage) == pytest.approx(1.0), "已提交版本不受新尝试影响"
    assert read_commit(storage).generation == gen1.id


def test_second_generation_takes_over_the_commit(storage: S3Storage, tmp_path) -> None:
    """重存同一 step：新一代完整之后 L3 才指向它。"""
    first = tmp_path / "local-first"
    second = tmp_path / "local-second"
    save_one(first, value=1.0)
    gen1 = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    ckpt.upload_local(first, storage, step=STEP, rank=0, generation=gen1)
    ckpt.finalize_upload(storage, step=STEP, ranks=[0], generation=gen1)

    save_one(second, value=2.0)
    gen2 = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    ckpt.upload_local(second, storage, step=STEP, rank=0, generation=gen2)
    commit = ckpt.finalize_upload(storage, step=STEP, ranks=[0], generation=gen2)

    assert commit.generation == gen2.id
    assert read_commit(storage).generation == gen2.id
    assert commit_value(storage) == pytest.approx(2.0)
    # 旧代的对象还在：它没有被覆盖，只是不再被 L3 引用（归 GC 收）
    assert storage.list(generation_prefix(STEP, gen1))


def test_data_object_is_never_overwritten(storage: S3Storage, tmp_path) -> None:
    """同代重复上传是幂等成功；内容不同则报错而不是覆盖。"""
    local = tmp_path / "local"
    save_one(local, value=1.0)
    gen = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    ckpt.upload_local(local, storage, step=STEP, rank=0, generation=gen)

    before = storage.read(index_key(STEP, 0, gen))
    ckpt.upload_local(local, storage, step=STEP, rank=0, generation=gen)  # 幂等重跑
    assert storage.read(index_key(STEP, 0, gen)) == before


# --- 恢复端：L3 是唯一权威 ------------------------------------------------


def test_recovery_ignores_gen_pointer(storage: S3Storage, tmp_path) -> None:
    local = tmp_path / "local"
    save_one(local, value=1.5)
    gen = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    ckpt.upload_local(local, storage, step=STEP, rank=0, generation=gen)
    ckpt.finalize_upload(storage, step=STEP, ranks=[0], generation=gen)

    storage.delete(pointer_key(STEP))

    assert commit_value(storage) == pytest.approx(1.5), "GEN 只是写路径的仲裁点"


def test_recovery_rejects_mismatched_rank_commit_generation(
    storage: S3Storage, tmp_path
) -> None:
    """L3 声明 gen A、某 rank 的 L2 却声明 gen B：拒绝恢复，不吐半份数据。"""
    local = tmp_path / "local"
    save_one(local, value=1.0)
    gen = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    ckpt.upload_local(local, storage, step=STEP, rank=0, generation=gen)
    ckpt.finalize_upload(storage, step=STEP, ranks=[0], generation=gen)

    index_payload = storage.read(index_key(STEP, 0, gen))
    forged = RankCommit(
        step=STEP,
        rank=0,
        metadata_bytes=len(index_payload),
        metadata_crc32c=crc32c_hex(index_payload),
        files={},
        generation=Generation(step=STEP, seq=gen.seq + 5, writer="deadbeef").id,
        seq=gen.seq + 5,
    )
    storage.write(rank_commit_key(STEP, 0, gen), forged.dumps().encode("utf-8"))

    with pytest.raises(FormatError, match="混了两次尝试"):
        ckpt.load_object(storage, step=STEP, device=DEVICE)


def test_recovery_rejects_rank_commit_from_old_layout(
    storage: S3Storage, tmp_path
) -> None:
    """v1 的 L2（没有 generation）混进 v2 版本：同样拒绝。"""
    local = tmp_path / "local"
    save_one(local, value=1.0)
    gen = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    ckpt.upload_local(local, storage, step=STEP, rank=0, generation=gen)
    ckpt.finalize_upload(storage, step=STEP, ranks=[0], generation=gen)

    legacy = RankCommit(
        step=STEP,
        rank=0,
        metadata_bytes=storage.head(index_key(STEP, 0, gen)).size,
        format_version=1,
    )
    storage.write(rank_commit_key(STEP, 0, gen), legacy.dumps().encode("utf-8"))

    with pytest.raises(FormatError, match="混了两次尝试"):
        ckpt.load_object(storage, step=STEP, device=DEVICE)


def test_legacy_v1_layout_still_restores(storage: S3Storage, tmp_path) -> None:
    """没有代际层的老对象照常能读：向后兼容不是靠迁移，而是读侧认两套布局。"""
    from commit.local import read_commit as read_local_commit
    from storage.local import LocalStorage

    local = tmp_path / "local"
    save_one(local, value=3.0)
    version = version_prefix(STEP)
    local_backend = LocalStorage(local, direct_io=False)
    committed = read_local_commit(local_backend, STEP)
    assert committed is not None

    metadata = local_backend.read(f"{version}/{META_FILE}")
    storage.write(f"{version}/rank_0/{META_FILE}", metadata)
    for name in committed.files:
        storage.write(f"{version}/rank_0/{name}", local_backend.read(f"{version}/{name}"))
    storage.write(
        f"{version}/rank_0/{COMMIT_FILE}",
        RankCommit(
            step=STEP,
            rank=0,
            metadata_bytes=len(metadata),
            files=dict(committed.files),
            format_version=1,
        ).dumps().encode("utf-8"),
    )
    storage.write(
        f"{version}/{COMMIT_FILE}",
        UploadCommit(step=STEP, ranks=(0,), format_version=1).dumps().encode("utf-8"),
    )

    assert commit_value(storage) == pytest.approx(3.0)


# --- L2 的索引摘要 ----------------------------------------------------------


def test_rank_commit_records_metadata_crc(storage: S3Storage, tmp_path) -> None:
    """L2 记下本 rank ``.metadata`` 的字节数与 CRC-32C，与 L1 的 COMMIT 同一算法。"""
    local = tmp_path / "local"
    save_one(local)
    gen = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    ckpt.upload_local(local, storage, step=STEP, rank=0, generation=gen)

    rank_commit = RankCommit.loads(
        storage.read(rank_commit_key(STEP, 0, gen)).decode("utf-8")
    )
    payload = storage.read(index_key(STEP, 0, gen))

    assert rank_commit.metadata_bytes == len(payload)
    assert rank_commit.metadata_crc32c == crc32c_hex(payload)


def test_rank_commit_v2_requires_metadata_crc() -> None:
    """写侧必须给摘要：v2 的 L2 少了它就不是提交点。"""
    with pytest.raises(FormatError, match="crc32c"):
        RankCommit(step=STEP, rank=0, metadata_bytes=8)
    with pytest.raises(FormatError, match="crc32c"):
        RankCommit(step=STEP, rank=0, metadata_bytes=8, metadata_crc32c="deadbee")


def test_corrupted_rank_metadata_is_rejected(storage: S3Storage, tmp_path) -> None:
    """某个 rank 的 ``.metadata`` 被改坏（长度不变）：合并索引之前就报出来。

    没有 L2 摘要时这类篡改会被合并进索引 —— 改的是 step 这种「结构上完全合法」的
    字段，``CheckpointIndex.validate()`` 也无从判断。
    """
    local = tmp_path / "local"
    save_one(local, value=1.0)
    gen = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    ckpt.upload_local(local, storage, step=STEP, rank=0, generation=gen)
    ckpt.finalize_upload(storage, step=STEP, ranks=[0], generation=gen)

    key = index_key(STEP, 0, gen)
    text = storage.read(key).decode("utf-8")
    before = f'"step": {STEP}'
    after = f'"step": {STEP + 1}'
    assert text.count(before) == 1
    tampered = text.replace(before, after).encode("utf-8")
    assert len(tampered) == len(text.encode("utf-8")), "篡改必须保持字节数不变"
    storage.write(key, tampered)

    with pytest.raises(ChecksumError, match="crc32c"):
        ckpt.load_object(storage, step=STEP, device=DEVICE)


def test_finalize_is_idempotent_without_extra_writes(
    storage: S3Storage, tmp_path, s3_server
) -> None:
    local = tmp_path / "local"
    save_one(local)
    gen = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    ckpt.upload_local(local, storage, step=STEP, rank=0, generation=gen)
    first = ckpt.finalize_upload(storage, step=STEP, ranks=[0], generation=gen)

    s3_server.store.requests.clear()
    second = ckpt.finalize_upload(storage, step=STEP, ranks=[0], generation=gen)

    assert second == first
    writes = [
        request
        for request in s3_server.store.requests
        if request[0] in ("PUT", "POST", "DELETE")
    ]
    assert writes == [], f"幂等重跑不应产生写入，却有：{writes}"


def test_begin_upload_uses_collective_to_broadcast() -> None:
    """``begin_upload`` 用一次 all-reduce 把 rank 0 的代际广播给全体。"""
    from dist import ThreadedCollective

    with s3_stub.start() as server:
        config = dict(
            bucket="ckpt",
            prefix="runs",
            endpoint_url=server.endpoint_url,
            access_key_id="test-key",
            secret_access_key="test-secret",
        )
        peers = ThreadedCollective.linked(2)
        clients = [S3Storage(S3Config(**config)) for _ in peers]
        results: list[Generation] = []
        errors: list[BaseException] = []

        import threading

        def run(peer, client) -> None:
            try:
                results.append(
                    ckpt.begin_upload(client, step=STEP, ranks=[0, 1], collective=peer)
                )
            except BaseException as exc:  # pragma: no cover - 断言失败时带出来
                errors.append(exc)

        threads = [
            threading.Thread(target=run, args=(peer, client))
            for peer, client in zip(peers, clients)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        for client in clients:
            client.close()

        assert not errors, errors
        assert len(results) == 2
        assert {gen.id for gen in results} == {results[0].id}
