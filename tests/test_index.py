"""索引结构的测试：序列化往返、dtype/字节数一致性、覆盖校验。"""

from __future__ import annotations

import json

import pytest
from core import FormatError, PlanError
from core.index import (
    FORMAT_VERSION,
    BlobMeta,
    CheckpointIndex,
    Placement,
    ShardMeta,
    TensorMeta,
)


def rect(file: str, offset: int, nbytes: int, offsets, shape) -> ShardMeta:
    return ShardMeta.rectangular(
        file=file, byte_offset=offset, length=nbytes, offsets=offsets, shape=shape
    )


# --- Placement --------------------------------------------------------------


def test_placement_roundtrip() -> None:
    assert Placement.replicate().to_dict() == "replicate"
    assert Placement.shard(1).to_dict() == {"shard": 1}
    assert Placement.from_dict("replicate") == Placement.replicate()
    assert Placement.from_dict({"shard": 1}) == Placement.shard(1)
    assert str(Placement.shard(0)) == "shard(0)"
    assert str(Placement.replicate()) == "replicate"


def test_placement_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        Placement("partial")
    with pytest.raises(ValueError):
        Placement.shard(-1)
    with pytest.raises(ValueError):
        Placement("replicate", 0)
    with pytest.raises(FormatError):
        Placement.from_dict({"shard": -1})
    with pytest.raises(FormatError):
        Placement.from_dict({"dim": 1})


# --- ShardMeta --------------------------------------------------------------


def test_shard_meta_roundtrip_rectangular() -> None:
    shard = rect("__3_0.distcp", 4096, 8 * 1024 * 1024, (0, 1536), (4096, 512))
    data = shard.to_dict()
    assert data == {
        "file": "__3_0.distcp",
        "byte_offset": 4096,
        "length": 8 * 1024 * 1024,
        "offsets": [0, 1536],
        "shape": [4096, 512],
    }
    assert ShardMeta.from_dict(data) == shard
    assert shard.elements == 4096 * 512
    assert shard.region is not None and shard.interval is None
    assert shard.end == 4096 + 8 * 1024 * 1024


def test_shard_meta_roundtrip_flat() -> None:
    shard = ShardMeta.flattened(
        file="__0_0.distcp", byte_offset=0, length=4096, start=1024, elements=1024
    )
    data = shard.to_dict()
    assert data["flat"] == [1024, 1024]
    assert "offsets" not in data
    assert ShardMeta.from_dict(data) == shard
    assert shard.is_flat
    assert shard.region is None and shard.interval is not None


def test_shard_meta_rejects_unsafe_or_inconsistent() -> None:
    with pytest.raises(FormatError):
        rect("/abs/path.bin", 0, 4, (0,), (1,))
    with pytest.raises(FormatError):
        rect("../escape.bin", 0, 4, (0,), (1,))
    with pytest.raises(FormatError):
        rect("a//b.bin", 0, 4, (0,), (1,))
    with pytest.raises(FormatError):
        rect("ok.bin", -1, 4, (0,), (1,))
    with pytest.raises(FormatError):
        rect("ok.bin", 0, 0, (0,), (1,))
    with pytest.raises(FormatError):
        rect("ok.bin", 0, 4, (0,), (1, 2))  # rank 不一致
    with pytest.raises(FormatError):
        ShardMeta(file="ok.bin", byte_offset=0, length=4, offsets=(0,), shape=(1,), flat=(0, 1))
    with pytest.raises(FormatError):
        ShardMeta.from_dict({"file": "ok.bin", "byte_offset": 0, "length": 4})


# --- TensorMeta -------------------------------------------------------------


def test_tensor_meta_validates_complete_coverage() -> None:
    meta = TensorMeta(
        global_shape=(4096, 4096),
        dtype="f32",
        shards=tuple(
            rect("__%d_0" % i, i * 8 * 1024 * 1024, 8 * 1024 * 1024, (0, 512 * i), (4096, 512))
            for i in range(8)
        ),
    )
    meta.validate()
    assert meta.numel == 4096 * 4096
    assert meta.nbytes == 4096 * 4096 * 4
    assert meta.covered_numel == meta.numel


def test_tensor_meta_detects_missing_and_overlap() -> None:
    partial = TensorMeta(
        global_shape=(4096, 4096),
        dtype="f32",
        shards=(rect("__0_0", 0, 8 * 1024 * 1024, (0, 0), (4096, 512)),),
    )
    partial.validate(complete=False)  # 单个 rank 只见一部分，允许不完整
    with pytest.raises(PlanError):
        partial.validate(complete=True)

    overlapping = TensorMeta(
        global_shape=(8, 8),
        dtype="f32",
        shards=(
            rect("a.bin", 0, 4 * 32, (0, 0), (4, 8)),
            rect("b.bin", 0, 4 * 32, (2, 0), (4, 8)),
        ),
    )
    with pytest.raises(PlanError):
        overlapping.validate(complete=False)


def test_tensor_meta_detects_bad_length_and_out_of_bounds() -> None:
    with pytest.raises(PlanError):
        TensorMeta(
            global_shape=(4, 4), dtype="f32", shards=(rect("a.bin", 0, 12, (0, 0), (4, 4)),)
        ).validate()
    with pytest.raises(PlanError):
        TensorMeta(
            global_shape=(4, 4), dtype="f32", shards=(rect("a.bin", 0, 64, (2, 0), (4, 4)),)
        ).validate()
    with pytest.raises(PlanError):  # rank 与全局形状不一致
        TensorMeta(
            global_shape=(4, 4), dtype="f32", shards=(rect("a.bin", 0, 16, (0,), (4,)),)
        ).validate(complete=False)


def test_tensor_meta_accepts_flat_shards() -> None:
    """ZeRO 式展平切分：2 个 rank 各拿 512 个元素的一维区间。"""
    meta = TensorMeta(
        global_shape=(32, 32),
        dtype="f16",
        shards=(
            ShardMeta.flattened(file="__0_0", byte_offset=0, length=1024, start=0, elements=512),
            ShardMeta.flattened(
                file="__1_0", byte_offset=0, length=1024, start=512, elements=512
            ),
        ),
    )
    meta.validate()
    assert meta.covered_numel == 1024
    assert len(meta.segments()) >= 2
    assert sum(segment.numel for segment in meta.segments()) == 1024


def test_tensor_meta_rejects_bad_dtype_and_shape() -> None:
    with pytest.raises(FormatError):
        TensorMeta(global_shape=(4, 4), dtype="fp32", shards=())
    with pytest.raises(FormatError):
        TensorMeta(global_shape=(), dtype="f32", shards=())
    with pytest.raises(FormatError):
        TensorMeta(global_shape=(0, 4), dtype="f32", shards=())


def test_tensor_meta_roundtrip_with_placement() -> None:
    meta = TensorMeta(
        global_shape=(4096, 4096),
        dtype="f32",
        shards=(rect("__0_0", 0, 8 * 1024 * 1024, (0, 0), (4096, 512)),),
        placement=Placement.shard(1),
    )
    data = meta.to_dict()
    assert data["placement"] == {"shard": 1}
    assert TensorMeta.from_dict(data) == meta
    assert "placement" not in TensorMeta(
        global_shape=(2,), dtype="f32", shards=(rect("a", 0, 8, (0,), (2,)),)
    ).to_dict()


# --- CheckpointIndex --------------------------------------------------------


def sample_index() -> CheckpointIndex:
    return CheckpointIndex(
        step=12000,
        saved_layout={"dp": 4, "tp": 8, "pp": 16},
        tensors={
            "model.layers.5.attn.q_proj.weight": TensorMeta(
                global_shape=(8, 8),
                dtype="f32",
                shards=(
                    rect("__3_0.distcp", 0, 4 * 32, (0, 0), (4, 8)),
                    rect("__3_0.distcp", 4 * 32, 4 * 32, (4, 0), (4, 8)),
                ),
                placement=Placement.shard(0),
            ),
            "model.embed_tokens.weight": TensorMeta(
                global_shape=(16,),
                dtype="bf16",
                shards=(rect("__0_0.distcp", 0, 32, (0,), (16,)),),
                placement=Placement.replicate(),
            ),
        },
        blobs={
            "lr_scheduler": BlobMeta(file="__0_0.distcp", byte_offset=4096, length=456),
            "rng/rank_3": BlobMeta(file="__3_0.distcp", byte_offset=8192, length=64),
        },
    )


def test_index_roundtrip_and_json_keys() -> None:
    index = sample_index()
    index.validate()
    text = index.dumps()
    payload = json.loads(text)
    assert payload["format_version"] == FORMAT_VERSION
    assert payload["step"] == 12000
    assert payload["saved_layout"] == {"dp": 4, "pp": 16, "tp": 8}
    assert set(payload["bytes"]) == {"lr_scheduler", "rng/rank_3"}
    assert CheckpointIndex.loads(text) == index


def test_index_dumps_is_stable() -> None:
    """同样的索引两次序列化必须逐字节一致（键有序），便于校验和。"""
    index = sample_index()
    assert index.dumps() == index.dumps()
    assert index.dumps() == CheckpointIndex.loads(index.dumps()).dumps()


def test_index_total_bytes_counts_shards_and_blobs() -> None:
    index = sample_index()
    assert index.total_bytes == 4 * 32 * 2 + 32 + 456 + 64


def test_index_detects_byte_overlap_in_same_file() -> None:
    index = CheckpointIndex(
        tensors={
            "a": TensorMeta(global_shape=(4,), dtype="f32", shards=(rect("f.bin", 0, 16, (0,), (4,)),)),
            "b": TensorMeta(global_shape=(4,), dtype="f32", shards=(rect("f.bin", 8, 16, (0,), (4,)),)),
        }
    )
    with pytest.raises(PlanError):
        index.validate()


def test_index_detects_blob_overlapping_shard() -> None:
    index = CheckpointIndex(
        tensors={
            "a": TensorMeta(global_shape=(4,), dtype="f32", shards=(rect("f.bin", 0, 16, (0,), (4,)),))
        },
        blobs={"meta": BlobMeta(file="f.bin", byte_offset=8, length=8)},
    )
    with pytest.raises(PlanError):
        index.validate()


def test_index_rejects_unknown_format_version() -> None:
    with pytest.raises(FormatError):
        CheckpointIndex(format_version=FORMAT_VERSION + 1)
    with pytest.raises(FormatError):
        CheckpointIndex.loads(json.dumps({"format_version": 99, "tensors": {}}))
    with pytest.raises(FormatError):
        CheckpointIndex.loads("not json")


def test_index_from_dict_rejects_bad_types() -> None:
    with pytest.raises(FormatError):
        CheckpointIndex.from_dict([])
    with pytest.raises(FormatError):
        CheckpointIndex.from_dict({"tensors": []})
    with pytest.raises(FormatError):
        CheckpointIndex.from_dict({"tensors": {}, "bytes": []})
    with pytest.raises(FormatError):
        CheckpointIndex.from_dict({"tensors": {}, "saved_layout": []})
    with pytest.raises(FormatError):
        CheckpointIndex.from_dict({"tensors": {}, "step": -1})
    with pytest.raises(FormatError):
        CheckpointIndex.from_dict({"tensors": {}, "saved_layout": {"tp": "8"}})


def test_index_empty_is_valid() -> None:
    index = CheckpointIndex()
    index.validate()
    assert index.total_bytes == 0
    assert CheckpointIndex.loads(index.dumps()) == index
