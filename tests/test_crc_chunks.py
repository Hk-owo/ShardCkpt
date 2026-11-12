"""大分片的块级 CRC：写侧切块、读侧按块 IO、中间块损坏能被抓住。"""

from __future__ import annotations

import numpy as np
import pytest

import api as ckpt
from async_ import Tensor, get_backend
from core import ChecksumError
from core.metadata import CPU
from storage.local import LocalStorage
from storage.local_load import load_local

MiB = 1 << 20


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


@pytest.fixture
def large_saved(tmp_path):
    root = tmp_path / "local"
    array = np.arange(3 * MiB, dtype=np.float32)  # 12 MiB
    tensor = make_tensor("big", array)
    report = ckpt.save_local({"big": tensor}, root, step=1)
    return root, array, report


def test_large_shard_is_split_into_crc_chunks(large_saved) -> None:
    root, array, report = large_saved
    shard = report.index.tensors["big"].shards[0]

    assert shard.crc_chunk_bytes == 1 * MiB
    assert len(shard.crc32c_chunks) == 12

    loaded = load_local(root, step=1, device="cpu")
    assert loaded.crc_verified is True
    assert loaded.verified_blocks == 12
    assert loaded.io_requests == 1
    assert loaded.unverified == ()
    assert loaded.unverified_bytes == 0
    assert loaded.tensors["big"].numpy().tobytes() == array.tobytes()


def test_corruption_in_a_middle_chunk_is_detected(large_saved) -> None:
    root, _, report = large_saved
    storage = LocalStorage(root)
    key = f"{report.key}/{report.file}"
    payload = bytearray(storage.read(key))
    payload[1 * MiB] ^= 0xFF  # 第二块开头，长度不变
    storage.write(key, bytes(payload))

    with pytest.raises(ChecksumError, match="crc32c"):
        load_local(root, step=1, device="cpu")

    # 显式关掉校验仍能读 —— 差异正是「有没有校验」
    unchecked = load_local(root, step=1, device="cpu", verify_crc=False)
    assert unchecked.step == 1


def test_verify_off_reads_one_io_range_per_item(large_saved) -> None:
    root, _, _ = large_saved
    report = load_local(root, step=1, device="cpu", verify_crc=False)
    assert report.io_requests == report.requests
    assert report.verified_blocks == 0
    assert report.unverified == ()


def test_small_shard_still_has_one_chunk(tmp_path) -> None:
    root = tmp_path / "local"
    array = np.arange(MiB // 4, dtype=np.float32)  # 1 MiB
    report = ckpt.save_local({"small": make_tensor("small", array)}, root, step=2)
    shard = report.index.tensors["small"].shards[0]
    assert shard.crc_chunk_bytes == 1 * MiB
    assert len(shard.crc32c_chunks) == 1

    loaded = load_local(root, step=2, device="cpu")
    assert loaded.verified_blocks == 1
    assert loaded.io_requests == 1
