#!/usr/bin/env python3
"""生成 ShardCkpt 测试数据。

只造数据，不做训练模拟：没有训练循环、没有参数演化、没有 loss/lr。
数据形态对齐 PyTorch 训练时的 state_dict（模型参数 + optimizer 状态），
输出为裸二进制 + manifest.json，C++ 与 Python 测试都能直接读。

用法：
    python tests/fixtures/make_data.py --out /tmp/ckpt-data --scale small
    python tests/fixtures/make_data.py --out /tmp/ckpt-data --dtype bf16 --world-size 2 --rank 0
"""

from __future__ import annotations

import argparse
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# dtype 名称与 C++ 侧 DType 枚举保持一致；bf16 没有 numpy 原生类型，用 uint16 承载原始位。
DTYPE_TO_NUMPY: dict[str, np.dtype] = {
    "f64": np.dtype(np.float64),
    "f32": np.dtype(np.float32),
    "f16": np.dtype(np.float16),
    "bf16": np.dtype(np.uint16),
    "i64": np.dtype(np.int64),
    "i32": np.dtype(np.int32),
    "i16": np.dtype(np.int16),
    "i8": np.dtype(np.int8),
    "u8": np.dtype(np.uint8),
}

SCALES: dict[str, dict[str, int]] = {
    "tiny": {"hidden": 32, "layers": 1, "vocab": 64, "intermediate": 64},
    "small": {"hidden": 128, "layers": 2, "vocab": 1000, "intermediate": 512},
    "medium": {"hidden": 512, "layers": 4, "vocab": 8000, "intermediate": 2048},
}


@dataclass(frozen=True)
class TensorSpec:
    """一个待生成 tensor 的描述。kind 用于区分模型参数与 optimizer 状态。"""

    name: str
    shape: tuple[int, ...]
    dtype: str = "f32"
    kind: str = "model"  # "model" | "optimizer"

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def nbytes(self) -> int:
        return self.numel * DTYPE_TO_NUMPY[self.dtype].itemsize

    @property
    def filename(self) -> str:
        return self.name.replace(".", "_") + ".bin"


def build_specs(
    *,
    hidden: int = 128,
    layers: int = 2,
    vocab: int = 1000,
    intermediate: int = 512,
    dtype: str = "f32",
    with_optimizer: bool = True,
) -> list[TensorSpec]:
    """一个典型 decoder-only transformer 的参数表，外加 Adam 的 exp_avg / exp_avg_sq。"""
    if dtype not in DTYPE_TO_NUMPY:
        raise ValueError(f"unsupported dtype: {dtype}")

    model: list[TensorSpec] = [
        TensorSpec("model.embed_tokens.weight", (vocab, hidden), dtype),
    ]
    for i in range(layers):
        p = f"model.blocks.{i}"
        model += [
            TensorSpec(f"{p}.attn.qkv.weight", (3 * hidden, hidden), dtype),
            TensorSpec(f"{p}.attn.qkv.bias", (3 * hidden,), dtype),
            TensorSpec(f"{p}.attn.proj.weight", (hidden, hidden), dtype),
            TensorSpec(f"{p}.attn.proj.bias", (hidden,), dtype),
            TensorSpec(f"{p}.mlp.fc1.weight", (intermediate, hidden), dtype),
            TensorSpec(f"{p}.mlp.fc1.bias", (intermediate,), dtype),
            TensorSpec(f"{p}.mlp.fc2.weight", (hidden, intermediate), dtype),
            TensorSpec(f"{p}.mlp.fc2.bias", (hidden,), dtype),
            TensorSpec(f"{p}.norm1.weight", (hidden,), dtype),
            TensorSpec(f"{p}.norm1.bias", (hidden,), dtype),
        ]
    model += [
        TensorSpec("model.norm.weight", (hidden,), dtype),
        TensorSpec("model.norm.bias", (hidden,), dtype),
        TensorSpec("model.lm_head.weight", (vocab, hidden), dtype),
    ]

    specs = list(model)
    if with_optimizer:
        for s in model:
            base = s.name.removeprefix("model.")
            specs.append(TensorSpec(f"optimizer.state.exp_avg.{base}", s.shape, dtype, "optimizer"))
            specs.append(
                TensorSpec(f"optimizer.state.exp_avg_sq.{base}", s.shape, dtype, "optimizer")
            )
    return specs


def make_tensor(spec: TensorSpec, seed: int) -> np.ndarray:
    """确定性生成一个 tensor。相同 (seed, name) 必然得到相同字节。"""
    rng = np.random.default_rng([seed & 0xFFFFFFFF, zlib.crc32(spec.name.encode())])
    n = spec.numel

    if spec.kind == "optimizer" and ".exp_avg_sq." in spec.name:
        values = rng.uniform(0.0, 1e-4, size=n)
    else:
        values = rng.normal(0.0, 0.02, size=n)

    if spec.dtype == "bf16":
        # bf16 = float32 的高 16 位（按位截断，不做舍入）。
        bits = values.astype(np.float32).view(np.uint32)
        raw = (bits >> np.uint32(16)).astype(np.uint16)
    else:
        raw = values.astype(DTYPE_TO_NUMPY[spec.dtype], copy=False)
    return raw.reshape(spec.shape)


def owned_by_rank(name: str, *, world_size: int, rank: int) -> bool:
    """模拟 DDP 的数据分布：模型参数每个 rank 都有，optimizer 状态按参数名分片。"""
    if world_size <= 1:
        return True
    if not name.startswith("optimizer."):
        return True
    return (zlib.crc32(name.encode()) % world_size) == rank


def iter_tensors(
    specs: list[TensorSpec], *, seed: int, world_size: int = 1, rank: int = 0
) -> Iterator[tuple[TensorSpec, np.ndarray]]:
    """按本 rank 的归属产出 (spec, ndarray)，不落盘。"""
    for spec in specs:
        if owned_by_rank(spec.name, world_size=world_size, rank=rank):
            yield spec, make_tensor(spec, seed)


@dataclass
class DatasetInfo:
    out_dir: Path
    step: int
    tensor_count: int
    total_bytes: int


def write_dataset(
    out_dir: str | Path,
    *,
    seed: int = 1234,
    step: int = 0,
    scale: str = "small",
    dtype: str = "f32",
    with_optimizer: bool = True,
    world_size: int = 1,
    rank: int = 0,
    hidden: int | None = None,
    layers: int | None = None,
    vocab: int | None = None,
    intermediate: int | None = None,
) -> DatasetInfo:
    """把一份数据集写到 out_dir：tensors/*.bin + manifest.json。"""
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}, expected one of {sorted(SCALES)}")
    if not 0 <= rank < max(1, world_size):
        raise ValueError(f"rank {rank} out of range for world_size {world_size}")

    cfg = dict(SCALES[scale])
    overrides = {"hidden": hidden, "layers": layers, "vocab": vocab, "intermediate": intermediate}
    cfg.update({k: v for k, v in overrides.items() if v is not None})

    specs = build_specs(dtype=dtype, with_optimizer=with_optimizer, **cfg)

    out_dir = Path(out_dir)
    tensor_dir = out_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    total_bytes = 0
    for spec, array in iter_tensors(specs, seed=seed, world_size=world_size, rank=rank):
        payload = np.ascontiguousarray(array).tobytes()
        if len(payload) != spec.nbytes:
            raise AssertionError(f"{spec.name}: expected {spec.nbytes} bytes, got {len(payload)}")
        (tensor_dir / spec.filename).write_bytes(payload)
        entries.append(
            {
                "name": spec.name,
                "file": f"tensors/{spec.filename}",
                "kind": spec.kind,
                "dtype": spec.dtype,
                "shape": list(spec.shape),
                "nbytes": spec.nbytes,
                "crc32": zlib.crc32(payload) & 0xFFFFFFFF,
            }
        )
        total_bytes += spec.nbytes

    manifest = {
        "format": "ckpt-fixture/1",
        "step": step,
        "seed": seed,
        "dtype": dtype,
        "world_size": world_size,
        "rank": rank,
        "model": cfg,
        "total_bytes": total_bytes,
        "tensors": entries,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return DatasetInfo(
        out_dir=out_dir, step=step, tensor_count=len(entries), total_bytes=total_bytes
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 ShardCkpt 测试数据（无训练过程）")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--scale", default="small", choices=sorted(SCALES))
    parser.add_argument("--dtype", default="f32", choices=sorted(DTYPE_TO_NUMPY))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--step", type=int, default=0, help="写进 manifest 的 step 编号，仅作标记")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--no-optimizer", action="store_true", help="只生成模型参数")
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--vocab", type=int)
    parser.add_argument("--intermediate", type=int)
    args = parser.parse_args(argv)

    info = write_dataset(
        args.out,
        seed=args.seed,
        step=args.step,
        scale=args.scale,
        dtype=args.dtype,
        with_optimizer=not args.no_optimizer,
        world_size=args.world_size,
        rank=args.rank,
        hidden=args.hidden,
        layers=args.layers,
        vocab=args.vocab,
        intermediate=args.intermediate,
    )
    print(
        f"wrote {info.tensor_count} tensors, {info.total_bytes / 1e6:.2f} MB "
        f"(step={info.step}) -> {info.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
