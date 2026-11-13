"""条件写（CAS）原语与端点能力探针。

对象存储端的代际与 fence 建立在两条服务端保证上：

1. ``If-None-Match: *`` 只有 key 不存在时才写入（create-exclusive）；
2. ``If-Match: <etag>`` 只有当前 ETag 仍是这个值时才替换（compare-and-swap）。

这里验的是**原语本身**（含分片上传的条件只挂在 complete 上、409 后必须重开 MPU），
以及「端点不执行条件时必须拒绝写入」这条 fail-closed 判据。代际协议见
``tests/test_generation.py``。
"""

from __future__ import annotations

import pytest

import s3_stub
from storage.base import (
    ConditionalWritesDisabled,
    UnsupportedConditionalWrites,
    WriteOutcome,
)
from storage.object import PROBE_PREFIX, S3Config, S3Storage


@pytest.fixture
def s3_server():
    with s3_stub.start() as server:
        yield server


def make_storage(s3_server, **overrides: object) -> S3Storage:
    kwargs: dict[str, object] = {
        "bucket": "ckpt",
        "prefix": "runs",
        "endpoint_url": s3_server.endpoint_url,
        "access_key_id": "test-key",
        "secret_access_key": "test-secret",
    }
    kwargs.update(overrides)
    return S3Storage(S3Config(**kwargs))  # type: ignore[arg-type]


@pytest.fixture
def storage(s3_server) -> S3Storage:
    return make_storage(s3_server)


@pytest.fixture
def multipart_storage(s3_server) -> S3Storage:
    """超过 1 KiB 就走分片：用小数据验证「条件只挂在 complete 上」。"""
    return make_storage(s3_server, multipart_threshold=1024)


def conditional_headers(s3_server) -> list[tuple[str, str, dict[str, str]]]:
    return list(s3_server.store.conditional_requests)


# --- create-exclusive / CAS -------------------------------------------------


def test_put_if_absent_creates_then_reports_exists(storage: S3Storage) -> None:
    assert storage.put_if_absent("step_1/GEN", b'{"seq": 1}') is WriteOutcome.CREATED
    assert storage.read("step_1/GEN") == b'{"seq": 1}'

    assert storage.put_if_absent("step_1/GEN", b'{"seq": 2}') is WriteOutcome.EXISTS
    assert storage.read("step_1/GEN") == b'{"seq": 1}', "If-None-Match 失败不能覆盖"


def test_put_if_match_swaps_only_on_current_etag(storage: S3Storage) -> None:
    storage.put_if_absent("step_1/GEN", b"v1")
    meta = storage.head("step_1/GEN")
    assert meta is not None

    assert storage.put_if_match("step_1/GEN", b"v2", meta.etag) is WriteOutcome.SWAPPED
    assert storage.read("step_1/GEN") == b"v2"

    # 拿旧 ETag 再 CAS 必须失败，且内容不动
    assert storage.put_if_match("step_1/GEN", b"v3", meta.etag) is WriteOutcome.CONFLICT
    assert storage.read("step_1/GEN") == b"v2"


def test_put_if_match_reports_gone_when_object_absent(storage: S3Storage) -> None:
    assert storage.put_if_match("step_1/missing", b"v", '"deadbeef"') is WriteOutcome.GONE
    assert storage.head("step_1/missing") is None


def test_head_returns_etag_and_size(storage: S3Storage) -> None:
    storage.write("step_1/blob", b"0123456789")
    meta = storage.head("step_1/blob")
    assert meta is not None
    assert meta.size == 10
    assert meta.etag.startswith('"') and meta.etag.endswith('"')


def test_put_if_match_rejects_empty_etag(storage: S3Storage) -> None:
    with pytest.raises(ValueError, match="ETag"):
        storage.put_if_match("step_1/GEN", b"v", "")


def test_condition_headers_are_mutually_exclusive(storage: S3Storage) -> None:
    with pytest.raises(ValueError, match="不能同时给"):
        storage._condition_kwargs(if_none_match=True, if_match='"x"')


# --- 分片上传 + 条件完成 ----------------------------------------------------


def test_write_file_streams_multipart_and_condition_lands_on_complete(
    multipart_storage: S3Storage, s3_server, tmp_path
) -> None:
    payload = bytes(range(256)) * 16  # 4 KiB > 1 KiB 阈值 → 走 multipart
    source = tmp_path / "data.bin"
    source.write_bytes(payload)

    outcome = multipart_storage.write_file("step_1/data.bin", source, if_none_match=True)

    assert outcome is WriteOutcome.CREATED
    assert multipart_storage.read("step_1/data.bin") == payload

    requests = s3_server.store.requests
    creates = [r for r in requests if r[0] == "POST" and "uploads" in r[2]]
    parts = [r for r in requests if "partNumber" in r[2]]
    completes = [r for r in requests if r[0] == "POST" and "uploadId" in r[2]]
    assert len(creates) == 1
    assert len(parts) >= 1
    assert len(completes) == 1, "条件只作用在 complete 上"

    # If-None-Match 只出现在 complete 那一次请求里：UploadPart 不接受条件头
    with_condition = [
        (method, path, headers)
        for method, path, headers in conditional_headers(s3_server)
        if "If-None-Match" in headers
    ]
    assert len(with_condition) == 1
    assert with_condition[0][0] == "POST"


def test_conditional_complete_reports_exists_when_object_appeared(
    multipart_storage: S3Storage, tmp_path
) -> None:
    """别人先落地了同 key 的普通 PUT：我的条件 complete 必须 412，且不改字节。

    这正是 AWS 文档里的「in-flight MPU 对其它条件写不可见」语义。
    """
    multipart_storage.write("step_1/data.bin", b"winner")
    source = tmp_path / "data.bin"
    source.write_bytes(b"loser" * 400)

    outcome = multipart_storage.write_file("step_1/data.bin", source, if_none_match=True)

    assert outcome is WriteOutcome.EXISTS
    assert multipart_storage.read("step_1/data.bin") == b"winner"
    assert multipart_storage.client is not None


def test_write_file_reports_conflict_on_stale_etag(multipart_storage: S3Storage, tmp_path) -> None:
    multipart_storage.write("step_1/GEN", b"v1")
    meta = multipart_storage.head("step_1/GEN")
    assert meta is not None
    source = tmp_path / "gen.json"
    source.write_bytes(b"v2" * 600)

    assert multipart_storage.write_file(
        "step_1/GEN", source, if_match='"00000000000000000000000000000000"'
    ) is WriteOutcome.CONFLICT
    assert multipart_storage.read("step_1/GEN") == b"v1"

    assert multipart_storage.write_file("step_1/GEN", source, if_match=meta.etag) is (
        WriteOutcome.SWAPPED
    )
    assert multipart_storage.read("step_1/GEN") == b"v2" * 600


def test_multipart_restarts_after_409(multipart_storage: S3Storage, s3_server, tmp_path) -> None:
    """409 ConditionalRequestConflict 说明这次 MPU 已被判死：必须重新 create。"""
    s3_server.store.fail_complete_times = 1
    source = tmp_path / "data.bin"
    payload = b"x" * 4096
    source.write_bytes(payload)

    assert multipart_storage.write_file("step_1/data.bin", source, if_none_match=True) is (
        WriteOutcome.CREATED
    )
    assert multipart_storage.read("step_1/data.bin") == payload
    assert s3_server.store.uploads == {}, "重开前的半截上传要 abort 干净"

    creates = [
        r for r in s3_server.store.requests if r[0] == "POST" and "uploads" in r[2]
    ]
    assert len(creates) == 2, "409 之后应重新 CreateMultipartUpload"


# --- 端点能力探针 -----------------------------------------------------------


def test_probe_accepts_conditional_endpoint(storage: S3Storage, s3_server) -> None:
    probe = storage.ensure_conditional_writes()

    assert probe.supported is True
    assert probe.checks, "探针要留下逐步流水，便于排查"
    names = [name for name, _ in probe.checks]
    assert names == [
        "put_if_absent/first",
        "put_if_absent/second",
        "head",
        "put_if_match/stale",
        "put_if_match/current",
    ]
    # 探针对象自己清掉，不污染 checkpoint 前缀
    assert not [k for k in s3_server.store.objects if PROBE_PREFIX in k]


def test_probe_rejects_endpoint_ignoring_conditions(storage: S3Storage, s3_server) -> None:
    s3_server.store.ignore_conditional_headers = True

    probe = storage.probe_conditional_writes()

    assert probe.supported is False
    assert "If-None-Match" in probe.detail
    with pytest.raises(UnsupportedConditionalWrites, match="不支持条件写"):
        storage.ensure_conditional_writes()
    assert not [k for k in s3_server.store.objects if PROBE_PREFIX in k]


def test_probe_is_cached_between_calls(storage: S3Storage, s3_server) -> None:
    storage.ensure_conditional_writes()
    before = len(s3_server.store.requests)

    storage.ensure_conditional_writes()

    assert len(s3_server.store.requests) == before, "探针结果应缓存，不重复发请求"


# --- conditional_writes="off" = 只读端点 ------------------------------------


def test_off_mode_rejects_writes_but_allows_reads(s3_server) -> None:
    writer = make_storage(s3_server)
    writer.write("step_1/blob", b"payload")

    reader = make_storage(s3_server, conditional_writes="off")
    try:
        assert reader.read("step_1/blob") == b"payload"
        assert reader.read_range("step_1/blob", 0, 3) == b"pay"
        assert reader.head("step_1/blob") is not None
        assert reader.list("step_1") == ["step_1/blob"]

        for call in (
            lambda: reader.write("step_1/x", b"v"),
            lambda: reader.put_if_absent("step_1/x", b"v"),
            lambda: reader.put_if_match("step_1/blob", b"v", '"e"'),
            lambda: reader.delete("step_1/blob"),
            lambda: reader.rename_or_put("step_1/blob", "step_1/copy"),
            lambda: reader.write_file("step_1/x", __file__),
        ):
            with pytest.raises(ConditionalWritesDisabled):
                call()
    finally:
        reader.close()


def test_off_mode_never_probes_and_never_writes(s3_server) -> None:
    reader = make_storage(s3_server, conditional_writes="off")
    try:
        probe = reader.probe_conditional_writes()
        assert probe.supported is False
        assert "只读" in probe.detail
        assert not s3_server.store.requests, "只读端点连探针都不该发请求"
        with pytest.raises(ConditionalWritesDisabled):
            reader.ensure_conditional_writes()
    finally:
        reader.close()


def test_config_rejects_unknown_conditional_mode() -> None:
    with pytest.raises(ValueError, match="conditional_writes"):
        S3Config(bucket="ckpt", conditional_writes="maybe")
