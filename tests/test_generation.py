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
    CommittedState,
    FencedError,
    Generation,
    StepStatus,
    acquire_generation,
    next_seq,
    read_status,
    status_key,
    verify_holder,
)
from core import ChecksumError, FormatError
from core.index import META_FILE
from core.metadata import CPU
from storage.base import WriteOutcome
from storage.checksum import crc32c_hex
from storage.object import S3Config, S3Storage
from storage.upload import (
    RankCommit,
    UploadCommit,
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
    status = read_status(storage, step)
    assert status is not None and status.committed is not None
    return UploadCommit.from_committed(status.committed)


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


def test_status_roundtrip_keeps_active_and_committed() -> None:
    active = Generation(step=2, seq=4, writer="deadbeef")
    committed = CommittedState(
        generation=Generation(step=2, seq=3, writer="cafe1234"),
        ranks=(0, 1, 2),
        layout={"tp": 3},
    )
    status = StepStatus(
        step=2,
        generation=active,
        ranks=(0, 1, 2),
        committed=committed,
        etag='"etag"',
    )

    loaded = StepStatus.loads(status.dumps(), etag='"etag"')

    assert loaded.generation == active
    assert loaded.ranks == (0, 1, 2)
    assert loaded.committed == committed
    assert loaded.state == "open"
    assert loaded.etag == '"etag"'
    assert loaded.dumps() == status.dumps()


# --- acquire：CAS 线性化 ----------------------------------------------------


def test_acquire_creates_first_generation(storage: S3Storage) -> None:
    gen = acquire_generation(storage, STEP, ranks=(0,), writer="aaaaaaaa")

    assert gen.seq == 1
    assert storage.read(status_key(STEP)) != b""
    status = verify_holder(storage, gen)
    assert status.state == "open"
    assert status.committed is None


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
    ckpt.begin_upload(storage, step=STEP, ranks=[0])  # 新尝试接管 STATUS

    with pytest.raises(FencedError, match="已被 fence"):
        ckpt.finalize_upload(storage, step=STEP, ranks=[0], generation=stale)

    status = read_status(storage, STEP)
    assert status is not None and status.committed is None


def test_finalize_rejects_rank_mismatch(storage: S3Storage, tmp_path) -> None:
    """STATUS 声明 ranks [0]，finalize 却传 [0, 1]：配置漂移，直接拒绝。"""
    local = tmp_path / "local"
    save_one(local)
    generation = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    ckpt.upload_local(local, storage, step=STEP, rank=0, generation=generation)

    with pytest.raises(FormatError, match="STATUS 声明的 ranks"):
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
    status = read_status(storage, STEP)
    assert status is not None and status.committed is None


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
    """重存同一 step：新一代完整之后 L3 才指向它，旧代随即被回收。"""
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
    # 旧代从来没被覆盖（CAS + 独立前缀保证），L3 一旦指向 gen2 它就是残片：
    # finalize_upload 顺手收掉，不必等周期 collect()。
    assert storage.list(generation_prefix(STEP, gen1)) == []


def test_old_generation_survives_while_it_is_the_committed_one(
    storage: S3Storage, tmp_path
) -> None:
    """新一代只传了一半就崩：已提交的 gen1 一个对象都不能少。"""
    local = tmp_path / "local"
    save_one(local, value=1.0)
    gen1 = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    ckpt.upload_local(local, storage, step=STEP, rank=0, generation=gen1)
    ckpt.finalize_upload(storage, step=STEP, ranks=[0], generation=gen1)

    gen2 = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    storage.put_if_absent(data_key(STEP, 0, "__0_0.distcp", gen2), b"half-written")

    before = storage.list(generation_prefix(STEP, gen1))
    # 半途而废的 gen2 不会触发任何回收：提交没成，L3 仍指 gen1
    assert commit_value(storage) == pytest.approx(1.0)
    assert storage.list(generation_prefix(STEP, gen1)) == before
    assert storage.list(generation_prefix(STEP, gen2)), "在传的那一代不能被当成残片"


def test_data_object_is_never_overwritten(storage: S3Storage, tmp_path) -> None:
    """同代重复上传是幂等成功；内容不同则报错而不是覆盖。"""
    local = tmp_path / "local"
    save_one(local, value=1.0)
    gen = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    ckpt.upload_local(local, storage, step=STEP, rank=0, generation=gen)

    before = storage.read(index_key(STEP, 0, gen))
    ckpt.upload_local(local, storage, step=STEP, rank=0, generation=gen)  # 幂等重跑
    assert storage.read(index_key(STEP, 0, gen)) == before


# --- 恢复端：STATUS.committed 是唯一权威 -----------------------------------


def test_deleting_status_makes_step_unrecoverable(storage: S3Storage, tmp_path) -> None:
    """单点 STATUS 一旦被删，这一版立刻不可恢复（不再单独存在 L3 COMMIT）。"""
    local = tmp_path / "local"
    save_one(local, value=1.5)
    gen = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    ckpt.upload_local(local, storage, step=STEP, rank=0, generation=gen)
    ckpt.finalize_upload(storage, step=STEP, ranks=[0], generation=gen)

    storage.delete(status_key(STEP))

    with pytest.raises(FormatError, match="STATUS"):
        ckpt.load_object(storage, step=STEP, device=DEVICE)


def test_recovery_rejects_mismatched_rank_commit_generation(
    storage: S3Storage, tmp_path
) -> None:
    """STATUS.committed 声明 gen A、某 rank 的 L2 却声明 gen B：拒绝恢复，不吐半份数据。"""
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


def test_rank_commit_rejects_legacy_v1_format(storage: S3Storage, tmp_path) -> None:
    """放弃远端 v1 兼容：缺 generation / format_version=1 的 L2 直接拒绝。"""
    local = tmp_path / "local"
    save_one(local, value=1.0)
    gen = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    ckpt.upload_local(local, storage, step=STEP, rank=0, generation=gen)
    ckpt.finalize_upload(storage, step=STEP, ranks=[0], generation=gen)

    storage.write(
        rank_commit_key(STEP, 0, gen),
        '{"format_version": 1, "step": 1, "rank": 0, "metadata": {"bytes": 8}, "files": {}}'.encode(
            "utf-8"
        ),
    )

    with pytest.raises(FormatError, match="format_version"):
        ckpt.load_object(storage, step=STEP, device=DEVICE)


# --- 单点 CAS：消除跨 key 并发窗口 ------------------------------------------


def test_stale_generation_cannot_commit_after_new_generation_acquires_status(
    storage: S3Storage, tmp_path, monkeypatch
) -> None:
    """★ gen1 在 finalize 的 list 阶段停顿，期间 gen2 抢走 STATUS；gen1 提交必然 412。

    在旧的双对象（GEN + COMMIT）模型下，这一步 gen1 仍能把 L3 写出来；单点 STATUS
    下 acquire 已经改掉了 ETag，gen1 的 If-Match 在服务端原子层直接失败。
    """
    local = tmp_path / "local"
    save_one(local, value=1.0)
    gen1 = ckpt.begin_upload(storage, step=STEP, ranks=[0])
    ckpt.upload_local(local, storage, step=STEP, rank=0, generation=gen1)

    real_list = storage.list
    interleaved = {"done": False}

    def list_then_preempt(prefix: str = "") -> list[str]:
        result = real_list(prefix)
        if not interleaved["done"] and prefix == generation_prefix(STEP, gen1):
            interleaved["done"] = True
            # gen1 刚检查完前缀准备提交，gen2 此时抢走 STATUS 写权
            ckpt.begin_upload(storage, step=STEP, ranks=[0])
        return result

    monkeypatch.setattr(storage, "list", list_then_preempt)

    with pytest.raises(FencedError, match="已被接管"):
        ckpt.finalize_upload(storage, step=STEP, ranks=[0], generation=gen1)

    status = read_status(storage, STEP)
    assert status is not None
    assert status.committed is None
    assert status.seq == 2


def test_latest_remote_step_skips_uncommitted_status_and_fails_on_corrupt_status(
    storage: S3Storage, tmp_path
) -> None:
    """只靠 STATUS 存在不算已提交：未提交的 step 2 被跳过，坏 JSON 则 fail-closed。"""
    local1 = tmp_path / "local-1"
    save_one(local1, step=1, value=1.0)
    gen1 = ckpt.begin_upload(storage, step=1, ranks=[0])
    ckpt.upload_local(local1, storage, step=1, rank=0, generation=gen1)
    ckpt.finalize_upload(storage, step=1, ranks=[0], generation=gen1)

    # step 2 开了代（写了 STATUS，committed=null），还没提交
    ckpt.begin_upload(storage, step=2, ranks=[0])

    loaded = ckpt.load_object(storage, device=DEVICE)
    assert loaded.step == 1
    assert float(loaded.tensors["w"].numpy()[0, 0]) == pytest.approx(1.0)

    # 最新 step 的 STATUS 损坏：fail-closed，不能静默回退
    storage.write(status_key(2), b"{broken-json")
    with pytest.raises(FormatError, match="STATUS"):
        ckpt.load_object(storage, device=DEVICE)


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
