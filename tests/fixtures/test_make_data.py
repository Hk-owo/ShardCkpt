"""make_data.py 的自测：确定性、checkpoint 与数据一致、分片规则。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import make_data  # noqa: E402

import api as ckpt  # noqa: E402


def load(out_dir: Path) -> ckpt.CheckpointData:
    return ckpt.load(out_dir, device="cpu")


def test_same_seed_is_byte_identical(tmp_path: Path) -> None:
    a = make_data.write_dataset(tmp_path / "a", seed=7, scale="tiny")
    b = make_data.write_dataset(tmp_path / "b", seed=7, scale="tiny")
    assert a.total_bytes == b.total_bytes

    left = load(a.out_dir)
    right = load(b.out_dir)
    assert left.names() == right.names()
    for name in left.names():
        assert left[name].tobytes() == right[name].tobytes(), name


def test_different_seed_changes_payload(tmp_path: Path) -> None:
    a = make_data.write_dataset(tmp_path / "a", seed=1, scale="tiny")
    b = make_data.write_dataset(tmp_path / "b", seed=2, scale="tiny")
    name = "model.embed_tokens.weight"
    assert load(a.out_dir)[name].tobytes() != load(b.out_dir)[name].tobytes()


def test_checkpoint_matches_info(tmp_path: Path) -> None:
    info = make_data.write_dataset(tmp_path, seed=42, scale="tiny")
    data = load(tmp_path)

    assert info.tensor_count == len(data)
    assert info.total_bytes == sum(tensor.nbytes for tensor in data.tensors.values())
    assert info.tensor_count > 0

    for name, tensor in data.items():
        assert len(tensor.tobytes()) == tensor.nbytes, name

    assert load(tmp_path).meta["step"] == info.step


def test_bf16_uses_two_bytes(tmp_path: Path) -> None:
    info = make_data.write_dataset(tmp_path, seed=3, scale="tiny", dtype="bf16")
    data = load(tmp_path)
    for name, tensor in data.items():
        assert tensor.dtype == "bf16", name
        assert tensor.nbytes == int(np.prod(tensor.shape, dtype=np.int64)) * 2, name
    assert info.total_bytes > 0


def test_no_optimizer_flag(tmp_path: Path) -> None:
    info = make_data.write_dataset(tmp_path, scale="tiny", with_optimizer=False)
    data = load(tmp_path)
    assert all(not name.startswith("optimizer.") for name in data.names())
    assert info.tensor_count == len(data)


def test_rank_sharding_partitions_optimizer_state(tmp_path: Path) -> None:
    world_size = 3
    seen: set[str] = set()
    model_names = None

    for rank in range(world_size):
        out = tmp_path / f"rank{rank}"
        make_data.write_dataset(out, scale="tiny", world_size=world_size, rank=rank)
        names = load(out).names()

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
