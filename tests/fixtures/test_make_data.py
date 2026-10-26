"""make_data.py 的自测：确定性、文件与 manifest 一致性、分片规则。"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_data  # noqa: E402


def read_manifest(out_dir: Path) -> dict:
    return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))


def test_same_seed_is_byte_identical(tmp_path: Path) -> None:
    a = make_data.write_dataset(tmp_path / "a", seed=7, scale="tiny")
    b = make_data.write_dataset(tmp_path / "b", seed=7, scale="tiny")
    assert a.total_bytes == b.total_bytes
    for entry in read_manifest(a.out_dir)["tensors"]:
        fa = a.out_dir / entry["file"]
        fb = b.out_dir / entry["file"]
        assert fa.read_bytes() == fb.read_bytes(), entry["name"]


def test_different_seed_changes_payload(tmp_path: Path) -> None:
    make_data.write_dataset(tmp_path / "a", seed=1, scale="tiny")
    make_data.write_dataset(tmp_path / "b", seed=2, scale="tiny")
    name = "tensors/model_embed_tokens_weight.bin"
    assert (tmp_path / "a" / name).read_bytes() != (tmp_path / "b" / name).read_bytes()


def test_manifest_matches_files(tmp_path: Path) -> None:
    info = make_data.write_dataset(tmp_path, seed=42, scale="tiny")
    manifest = read_manifest(tmp_path)

    assert manifest["total_bytes"] == info.total_bytes
    assert len(manifest["tensors"]) == info.tensor_count
    assert info.tensor_count > 0

    total = 0
    for entry in manifest["tensors"]:
        payload = (tmp_path / entry["file"]).read_bytes()
        assert len(payload) == entry["nbytes"], entry["name"]
        assert zlib.crc32(payload) & 0xFFFFFFFF == entry["crc32"], entry["name"]
        assert int(np.prod(entry["shape"], dtype=np.int64)) * np.dtype(
            make_data.DTYPE_TO_NUMPY[entry["dtype"]]
        ).itemsize == entry["nbytes"]
        total += len(payload)
    assert total == manifest["total_bytes"]


def test_bf16_uses_two_bytes(tmp_path: Path) -> None:
    info = make_data.write_dataset(tmp_path, seed=3, scale="tiny", dtype="bf16")
    manifest = read_manifest(tmp_path)
    for entry in manifest["tensors"]:
        assert entry["dtype"] == "bf16"
        assert entry["nbytes"] == int(np.prod(entry["shape"], dtype=np.int64)) * 2
    assert info.total_bytes > 0


def test_no_optimizer_flag(tmp_path: Path) -> None:
    info = make_data.write_dataset(tmp_path, scale="tiny", with_optimizer=False)
    manifest = read_manifest(tmp_path)
    assert all(e["kind"] == "model" for e in manifest["tensors"])
    assert info.tensor_count == len([e for e in manifest["tensors"]])


def test_rank_sharding_partitions_optimizer_state(tmp_path: Path) -> None:
    world_size = 3
    seen: set[str] = set()
    model_names = None

    for rank in range(world_size):
        out = tmp_path / f"rank{rank}"
        make_data.write_dataset(out, scale="tiny", world_size=world_size, rank=rank)
        manifest = read_manifest(out)
        names = [e["name"] for e in manifest["tensors"]]

        model = {n for n in names if not n.startswith("optimizer.")}
        opt = {n for n in names if n.startswith("optimizer.")}

        # 每个 rank 都持有完整模型参数副本
        if model_names is None:
            model_names = model
        assert model == model_names

        # optimizer 状态不重不漏
        assert not (opt & seen), "optimizer state assigned to more than one rank"
        seen |= opt

    all_opt = {
        s.name
        for s in make_data.build_specs(**make_data.SCALES["tiny"])
        if s.name.startswith("optimizer.")
    }
    assert seen == all_opt


def test_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        make_data.write_dataset(tmp_path, scale="nope")
    with pytest.raises(ValueError):
        make_data.write_dataset(tmp_path, dtype="float128x")
    with pytest.raises(ValueError):
        make_data.write_dataset(tmp_path, world_size=2, rank=2)
