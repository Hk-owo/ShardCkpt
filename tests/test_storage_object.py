"""S3 对象存储后端测试：用进程内 S3 兼容服务端到端验证 boto3 客户端。"""

from __future__ import annotations

import pytest

import s3_stub
import storage.object as storage_object
from storage.base import ObjectNotFoundError, StorageError
from storage.object import MIN_PART_SIZE, S3Config, S3Storage


@pytest.fixture
def s3_server():
    with s3_stub.start() as server:
        yield server


@pytest.fixture
def storage(s3_server) -> S3Storage:
    config = S3Config(
        bucket="ckpt",
        prefix="runs/step_0",
        endpoint_url=s3_server.endpoint_url,
        access_key_id="test-key",
        secret_access_key="test-secret",
    )
    return S3Storage(config)


def stored_keys(s3_server) -> list[str]:
    return [key for key, _ in s3_stub.iter_objects(s3_server)]


def test_write_then_read_roundtrip(storage: S3Storage, s3_server) -> None:
    payload = bytes(range(256)) * 4
    storage.write("tensors/model_embed.bin", payload)

    assert storage.read("tensors/model_embed.bin") == payload
    assert stored_keys(s3_server) == ["runs/step_0/tensors/model_embed.bin"]


def test_write_accepts_empty_object(storage: S3Storage) -> None:
    storage.write("empty.bin", b"")
    assert storage.read("empty.bin") == b""


def test_read_range_matches_slice(storage: S3Storage, s3_server) -> None:
    payload = bytes(range(200))
    storage.write("tensors/a.bin", payload)

    assert storage.read_range("tensors/a.bin", 10, 20) == payload[10:30]
    assert storage.read_range("tensors/a.bin", 0, 1) == payload[:1]
    assert storage.read_range("tensors/a.bin", 199, 1) == payload[-1:]
    # 区间读取不应该拉整个对象：服务端看到的就是 Range 请求
    assert stored_keys(s3_server) == ["runs/step_0/tensors/a.bin"]


def test_read_range_out_of_bounds_is_storage_error(storage: S3Storage) -> None:
    storage.write("tensors/a.bin", b"0123456789")
    with pytest.raises(StorageError):
        storage.read_range("tensors/a.bin", 10, 1)


def test_read_range_rejects_bad_arguments(storage: S3Storage) -> None:
    storage.write("tensors/a.bin", b"0123456789")
    with pytest.raises(ValueError):
        storage.read_range("tensors/a.bin", -1, 1)
    with pytest.raises(ValueError):
        storage.read_range("tensors/a.bin", 0, 0)


def test_missing_object_raises_not_found(storage: S3Storage) -> None:
    with pytest.raises(ObjectNotFoundError):
        storage.read("tensors/missing.bin")


def test_list_returns_relative_keys(storage: S3Storage, s3_server) -> None:
    for key in ("manifest.json", "tensors/b.bin", "tensors/a.bin"):
        storage.write(key, key.encode())

    assert storage.list() == ["manifest.json", "tensors/a.bin", "tensors/b.bin"]
    assert storage.list("tensors") == ["tensors/a.bin", "tensors/b.bin"]


def test_list_prefix_does_not_cross_sibling_prefix(storage: S3Storage, s3_server) -> None:
    """prefix 末尾补 '/'：runs/step_0 不能顺带列出 runs/step_0_extra 的对象。"""
    storage.write("manifest.json", b"{}")
    other = S3Storage(
        S3Config(
            bucket="ckpt",
            prefix="runs/step_0_extra",
            endpoint_url=s3_server.endpoint_url,
            access_key_id="test-key",
            secret_access_key="test-secret",
        )
    )
    other.write("manifest.json", b"{}")

    assert storage.list() == ["manifest.json"]


def test_rename_or_put_moves_object(storage: S3Storage, s3_server) -> None:
    storage.write("tensors/tmp.bin", b"payload")
    storage.rename_or_put("tensors/tmp.bin", "tensors/final.bin")

    assert storage.read("tensors/final.bin") == b"payload"
    with pytest.raises(ObjectNotFoundError):
        storage.read("tensors/tmp.bin")
    assert stored_keys(s3_server) == ["runs/step_0/tensors/final.bin"]


def test_rename_or_put_same_key_is_noop(storage: S3Storage) -> None:
    storage.write("tensors/a.bin", b"payload")
    storage.rename_or_put("tensors/a.bin", "tensors/a.bin")
    assert storage.read("tensors/a.bin") == b"payload"


def test_delete_is_idempotent(storage: S3Storage) -> None:
    storage.write("tensors/a.bin", b"payload")
    storage.delete("tensors/a.bin")
    storage.delete("tensors/a.bin")  # 不存在也算成功
    with pytest.raises(ObjectNotFoundError):
        storage.read("tensors/a.bin")


@pytest.mark.parametrize("key", ["", "/abs.bin", "../escape.bin", "a/../b.bin", "a//b.bin"])
def test_object_key_rejects_invalid_paths(storage: S3Storage, key: str) -> None:
    with pytest.raises(ValueError):
        storage.write(key, b"x")


def test_config_from_url() -> None:
    config = S3Config.from_url("s3://ckpt/runs/step_0")
    assert config.bucket == "ckpt"
    assert config.prefix == "runs/step_0"
    assert config.url == "s3://ckpt/runs/step_0"

    assert S3Config.from_url("s3://ckpt").prefix == ""
    with pytest.raises(ValueError):
        S3Config.from_url("/tmp/ckpt")
    with pytest.raises(ValueError):
        S3Config.from_url("s3://")


def test_config_requires_secret_when_key_given() -> None:
    with pytest.raises(ValueError):
        S3Storage(S3Config(bucket="ckpt", access_key_id="only-key"))


# --- multipart 分片上传 -----------------------------------------------------


def payload_of(nbytes: int) -> bytes:
    pattern = bytes(range(256))
    return (pattern * (nbytes // len(pattern) + 1))[:nbytes]


@pytest.fixture
def multipart_storage(s3_server) -> S3Storage:
    """超过 5 MiB 就走分片，方便用小数据验证 multipart 路径。"""
    return S3Storage(
        S3Config(
            bucket="ckpt",
            prefix="runs/step_0",
            endpoint_url=s3_server.endpoint_url,
            access_key_id="test-key",
            secret_access_key="test-secret",
            multipart_threshold=MIN_PART_SIZE,
        )
    )


def test_large_write_is_split_into_parts(multipart_storage: S3Storage, s3_server) -> None:
    payload = payload_of(11 * 1024 * 1024)  # 8 MiB + 3 MiB 两片

    multipart_storage.write("tensors/big.bin", payload)

    assert multipart_storage.read("tensors/big.bin") == payload
    # 分片上传在服务端没有残留
    assert s3_server.store.uploads == {}

    requests = s3_server.store.requests
    creates = [r for r in requests if r[0] == "POST" and "uploads" in r[2]]
    parts = [r for r in requests if "partNumber" in r[2]]
    completes = [r for r in requests if r[0] == "POST" and "uploadId" in r[2]]
    single_puts = [r for r in requests if r[0] == "PUT" and "partNumber" not in r[2]]

    assert len(creates) == 1
    assert sorted(int(r[2]["partNumber"]) for r in parts) == [1, 2]
    assert len(completes) == 1
    assert single_puts == [], "大对象不应该走单次 PUT"


def test_small_write_stays_single_put(storage: S3Storage, s3_server) -> None:
    storage.write("tensors/small.bin", payload_of(1024))

    requests = s3_server.store.requests
    assert not any("uploads" in r[2] for r in requests)
    assert len([r for r in requests if r[0] == "PUT"]) == 1


def test_multipart_threshold_boundary(storage: S3Storage, s3_server) -> None:
    """刚好等于阈值时仍走单次 PUT（阈值语义是「超过」）。"""
    threshold = storage.config.multipart_threshold
    storage.write("tensors/exact.bin", payload_of(threshold))

    assert not any("uploads" in r[2] for r in s3_server.store.requests)


def test_failed_part_aborts_upload(multipart_storage: S3Storage, s3_server) -> None:
    s3_server.store.fail_upload_part = True
    try:
        with pytest.raises(StorageError):
            multipart_storage.write("tensors/big.bin", payload_of(11 * 1024 * 1024))
    finally:
        s3_server.store.fail_upload_part = False

    # 失败的分片上传被 abort 掉，对象也不存在
    assert s3_server.store.uploads == {}
    assert any(r[0] == "DELETE" and "uploadId" in r[2] for r in s3_server.store.requests)
    with pytest.raises(ObjectNotFoundError):
        multipart_storage.read("tensors/big.bin")


def test_config_rejects_invalid_multipart_sizes() -> None:
    with pytest.raises(ValueError):
        S3Config(bucket="ckpt", part_size=MIN_PART_SIZE - 1)
    with pytest.raises(ValueError):
        S3Config(bucket="ckpt", part_size=storage_object.MAX_PART_SIZE + 1)
    with pytest.raises(ValueError):
        S3Config(bucket="ckpt", multipart_threshold=0)


# --- 大对象分片 copy（UploadPartCopy） --------------------------------------


@pytest.fixture
def small_copy_limit(monkeypatch) -> int:
    """把 copy_object 的 5 GiB 上限调小，用小对象验证 UploadPartCopy 分支。"""
    limit = 1024 * 1024
    monkeypatch.setattr(storage_object, "MAX_SINGLE_COPY_BYTES", limit)
    return limit


def test_large_rename_uses_multipart_copy(
    storage: S3Storage, s3_server, small_copy_limit: int
) -> None:
    payload = payload_of(11 * 1024 * 1024)  # 超过调小后的上限，也跨两个 8 MiB 分片
    storage.write("tensors/src.bin", payload)
    s3_server.store.requests.clear()

    storage.rename_or_put("tensors/src.bin", "tensors/dst.bin")

    assert storage.read("tensors/dst.bin") == payload
    with pytest.raises(ObjectNotFoundError):
        storage.read("tensors/src.bin")

    requests = s3_server.store.requests
    creates = [r for r in requests if r[0] == "POST" and "uploads" in r[2]]
    parts = [r for r in requests if "partNumber" in r[2]]
    completes = [r for r in requests if r[0] == "POST" and "uploadId" in r[2]]
    plain_copies = [r for r in requests if r[0] == "PUT" and "partNumber" not in r[2]]

    assert len(creates) == 1
    assert sorted(int(r[2]["partNumber"]) for r in parts) == [1, 2]
    assert len(completes) == 1
    assert plain_copies == [], "超过上限的源对象不应该走 copy_object"
    assert s3_server.store.uploads == {}


def test_small_rename_uses_single_copy(storage: S3Storage, s3_server) -> None:
    storage.write("tensors/src.bin", b"payload")
    s3_server.store.requests.clear()

    storage.rename_or_put("tensors/src.bin", "tensors/dst.bin")

    requests = s3_server.store.requests
    assert any(r[0] == "PUT" and "partNumber" not in r[2] for r in requests)
    assert not any("partNumber" in r[2] for r in requests)


def test_multipart_copy_failure_keeps_source(
    storage: S3Storage, s3_server, small_copy_limit: int
) -> None:
    payload = payload_of(3 * 1024 * 1024)
    storage.write("tensors/src.bin", payload)
    s3_server.store.fail_upload_part = True
    try:
        with pytest.raises(StorageError):
            storage.rename_or_put("tensors/src.bin", "tensors/dst.bin")
    finally:
        s3_server.store.fail_upload_part = False

    # 分片 copy 失败：上传被 abort，源对象保持完整，目标不存在
    assert s3_server.store.uploads == {}
    assert any(r[0] == "DELETE" and "uploadId" in r[2] for r in s3_server.store.requests)
    assert storage.read("tensors/src.bin") == payload
    with pytest.raises(ObjectNotFoundError):
        storage.read("tensors/dst.bin")


def test_rename_missing_source_raises(storage: S3Storage) -> None:
    with pytest.raises(ObjectNotFoundError):
        storage.rename_or_put("tensors/missing.bin", "tensors/dst.bin")


def test_copy_part_size_scales_for_huge_objects(storage: S3Storage) -> None:
    # 小对象沿用配置的分片大小
    assert storage._copy_part_size(11 * 1024 * 1024) == storage.config.part_size

    # 100 GiB 若仍用 8 MiB 分片会超过 10000 片上限，分片必须放大
    huge = 100 * 1024**3
    part_size = storage._copy_part_size(huge)
    assert part_size >= -(-huge // storage_object.MAX_MULTIPART_PARTS)
    assert -(-huge // part_size) <= storage_object.MAX_MULTIPART_PARTS
    assert storage_object.MIN_PART_SIZE <= part_size <= storage_object.MAX_PART_SIZE




# --- 对象级校验和：本地层不做字节校验后的唯一把关点 ---------------------------


def test_put_carries_object_checksum(storage: S3Storage, s3_server) -> None:
    """默认配置下每个 PUT 都带对象级校验和头，服务端会独立重算并比对。"""
    s3_server.store.require_checksum = True  # 不带头的 PUT 直接 400

    storage.write("tensors/a.bin", b"payload")

    assert s3_server.store.objects["runs/step_0/tensors/a.bin"] == b"payload"
    assert s3_server.store.checksum_headers, "客户端没有发送 x-amz-checksum-* 头"


def test_multipart_parts_carry_checksum(multipart_storage: S3Storage, s3_server) -> None:
    """multipart 的每一片也要带校验和，坏片在服务端就被拒掉。"""
    multipart_storage.write("tensors/big.bin", payload_of(11 * 1024 * 1024))

    algorithms = {algorithm for _, algorithm, _ in s3_server.store.checksum_headers}
    assert algorithms == {"crc32"}
    # 两片 + create 时的一条记录（create 不带 body，不会进 checksum_headers）
    part_headers = [h for h in s3_server.store.checksum_headers if h[0].endswith("big.bin")]
    assert len(part_headers) == 2


def test_read_detects_corrupted_object(storage: S3Storage, s3_server) -> None:
    """存储侧静默损坏：内容被改而校验和还停在写入时的值，读取必须报错。

    这是本地层不再做 CRC 之后，唯一能抓住「存进去的东西坏了」的地方。
    """
    storage.write("tensors/a.bin", b"hello world" * 100)
    key = "runs/step_0/tensors/a.bin"
    s3_server.store.objects[key] = b"HELLO WORLD" * 100  # 同长度、不同内容

    with pytest.raises(StorageError, match="checksum"):
        storage.read("tensors/a.bin")


def test_checksum_can_be_disabled_for_legacy_endpoints(s3_server) -> None:
    """不认识 x-amz-checksum-* 的老实现：checksum_algorithm=None 退回旧行为。"""
    legacy = S3Storage(
        S3Config(
            bucket="ckpt",
            prefix="runs/step_0",
            endpoint_url=s3_server.endpoint_url,
            access_key_id="test-key",
            secret_access_key="test-secret",
            checksum_algorithm=None,
        )
    )

    legacy.write("tensors/a.bin", b"payload")

    assert legacy.read("tensors/a.bin") == b"payload"
    assert s3_server.store.checksum_headers == [], "关掉之后不该再发校验和头"


def test_config_rejects_unknown_checksum_algorithm() -> None:
    with pytest.raises(ValueError, match="unsupported checksum_algorithm"):
        S3Config(bucket="ckpt", checksum_algorithm="MD5")
