"""端到端链路测试：GPU（mock cuda）-> host -> S3 兼容对象存储 -> 读回校验。"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pytest

import api as ckpt
import s3_stub
from async_ import copy_to_host, get_backend
from storage.object import S3Config, S3Storage


@pytest.fixture
def s3_server():
    with s3_stub.start() as server:
        yield server


def make_storage(server, *, prefix: str = "runs/step_0", **overrides: object) -> S3Storage:
    config = S3Config(
        bucket="ckpt",
        prefix=prefix,
        endpoint_url=server.endpoint_url,
        access_key_id="test-key",
        secret_access_key="test-secret",
        **overrides,  # type: ignore[arg-type]
    )
    return S3Storage(config)


def manifest_of(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def test_gpu_tensors_roundtrip_through_object_storage(dataset: Path, s3_server) -> None:
    """GPU 上的全部张量搬回 host、PUT 到对象存储，读回后与磁盘上的 bin 逐字节一致。"""
    manifest = manifest_of(dataset)
    files = {entry["name"]: entry["file"] for entry in manifest["tensors"]}
    crcs = {entry["name"]: entry["crc32"] for entry in manifest["tensors"]}

    data = ckpt.load(dataset, device="cuda:0")
    backend = get_backend(data.device)
    assert backend.is_mock, "当前环境应使用 mock cuda backend"
    backend.stats.reset()

    s3 = make_storage(s3_server)

    # GPU -> host -> 对象存储：tensor.tobytes() 就是 copy_to_host 的调用方入口
    keys = {}
    for name, tensor in data.items():
        key = f"tensors/{name.replace('.', '_')}.bin"
        s3.write(key, tensor.tobytes())
        keys[name] = key

    # 每个张量恰好一次 D2H，字节数与张量总长一致
    assert backend.stats.device_to_host_copies == len(data)
    assert backend.stats.device_to_host_bytes == sum(t.nbytes for t in data.tensors.values())

    # 写入的对象集合与张量一一对应（key 相对 prefix）
    assert s3.list("tensors") == sorted(keys.values())

    # 读回：整对象逐字节等于源文件，CRC32 与 manifest 一致，Range 读同样可用
    for name, key in keys.items():
        payload = s3.read(key)
        assert payload == (dataset / files[name]).read_bytes(), name
        assert zlib.crc32(payload) & 0xFFFFFFFF == crcs[name], name
        head = s3.read_range(key, 0, min(64, len(payload)))
        assert head == payload[: len(head)], name

    # 索引最后写：读得到索引才说明这批对象可以恢复
    s3.write("manifest.json", json.dumps({**manifest, "keys": keys}).encode("utf-8"))
    index = json.loads(s3.read("manifest.json").decode("utf-8"))
    assert index["keys"] == keys
    assert s3.list("") == sorted([*keys.values(), "manifest.json"])


def test_large_gpu_tensor_streams_through_multipart(s3_server) -> None:
    """单张量超过 multipart_threshold 时走 create/upload_part/complete，读回一致。"""
    payload = np.arange(1 << 18, dtype=np.uint32).tobytes()  # 1 MiB
    device = ckpt.Device.parse("cuda:0")
    backend = get_backend(device)
    buffer = backend.allocate(len(payload))
    backend.copy_h2d(buffer, memoryview(payload))

    s3 = make_storage(s3_server, multipart_threshold=256 << 10, part_size=5 << 20)
    s3.write("tensors/big.bin", copy_to_host(buffer, device=device, backend=backend))

    assert any("uploads" in query for _, _, query in s3_server.store.requests), "未走 multipart"
    assert s3.read("tensors/big.bin") == payload
    assert s3.read_range("tensors/big.bin", 1024, 16) == payload[1024:1040]
