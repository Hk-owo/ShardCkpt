"""仿真的训练世界：确定性伪训练状态 + 走真实搬运路径的 device 张量。

这里**不模拟真实训练**：没有前向/反向/loss，只有一条「每个 step 必然变化、且可以
按 ``(seed, step, 张量名)`` 重放」的参数演化函数。仿真测试关心的是 checkpoint
生命周期的行为（版本可见性、布局变更、崩溃恢复、成本），不是数值正确性。

可复用的部件：

- :class:`TrainingState`：一步训练后的完整 state（模型参数 + Adam 状态，host 上）
- :func:`to_device` / :func:`tensors_for_rank`：host 数组 → ``Tensor``，走
  ``backend.allocate`` + ``copy_h2d``，与真机同一条搬运路径
- :func:`merge_rank_versions`：**test-only** 的 rank0 汇总。``dist/`` 的 gather 还没
  实现，这里用「每个 rank 各写一份本地版本、rank0 读回来合并」代替；真 gather 落地
  后换掉这个函数，用例不用改。

约定：张量名/形状取自 :func:`make_data.build_specs`，与 ``tests/fixtures`` 对齐。
"""

from __future__ import annotations

import os
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from async_ import Tensor, get_backend
from commit.local import (
    CommitRecord,
    version_key,
    write_commit,
    write_metadata,
)
from core.index import META_FILE, BlobMeta, CheckpointIndex, ShardMeta, TensorMeta
from core.metadata import Device, numpy_dtype
from core.region import Region, split
from make_data import build_specs, make_tensor
from planner.save_plan import ShardSource
from storage.local import LocalStorage

# 每个 step 的扰动幅度：按 dtype 的可表示精度挑。对 f16 用 1e-4 会一步就下溢成 0
# （f16 的最小正规数约 6e-5），那样「每个 step 都变」的断言会假通过。
_NOISE_SCALE: dict[str, float] = {"f16": 1e-2, "f32": 1e-4, "f64": 1e-6, "bf16": 1e-4}


# --- 伪训练状态 -------------------------------------------------------------


@dataclass(frozen=True)
class TrainingState:
    """一步训练后的完整状态：``FQN -> host 数组``，外加每个张量的 dtype。"""

    step: int
    dtypes: Mapping[str, str]
    values: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtypes", dict(self.dtypes))
        object.__setattr__(self, "values", dict(self.values))
        missing = sorted(set(self.values) - set(self.dtypes))
        if missing:
            raise ValueError(f"missing dtype for {missing}")

    def names(self) -> list[str]:
        """按名字排序，保证所有断言与索引顺序稳定。"""
        return sorted(self.values)

    def bytes_of(self, name: str) -> bytes:
        return np.ascontiguousarray(self.values[name]).tobytes()

    def total_bytes(self) -> int:
        return sum(len(self.bytes_of(name)) for name in self.names())

    def snapshot(self) -> "TrainingState":
        """深拷贝一份，用来记住「这一版落盘时应该长什么样」。"""
        return TrainingState(
            step=self.step,
            dtypes=self.dtypes,
            values={name: array.copy() for name, array in self.values.items()},
        )

    def step_once(self, count: int = 1) -> "TrainingState":
        """往前走 ``count`` 步；同样的起点与步数必然得到同样的字节。"""
        state = self
        for _ in range(count):
            state = state._evolve()
        return state

    def shard(self, name: str, *, dim: int | None = None, index: int = 0, count: int = 1) -> np.ndarray:
        """本 rank 该持有的那一块（``dim=None`` 或 ``count=1`` 时是完整副本）。"""
        array = self.values[name]
        if dim is None or count == 1:
            return np.ascontiguousarray(array)
        return np.ascontiguousarray(array[split(array.shape, dim, index, count).slices])

    def diff(self, other: "TrainingState") -> list[str]:
        """与另一份状态不同的张量名（断言「每一步都真的变了」）。"""
        return [name for name in self.names() if self.bytes_of(name) != other.bytes_of(name)]

    def _evolve(self) -> "TrainingState":
        step = self.step + 1
        values = {
            name: _evolve_array(
                self.values[name], self.dtypes[name], step=step, name=name
            )
            for name in self.names()
        }
        return TrainingState(step=step, dtypes=self.dtypes, values=values)


def _evolve_array(array: np.ndarray, dtype: str, *, step: int, name: str) -> np.ndarray:
    """``array -> array * 0.999 + noise(step, name)``，按 dtype 截回原精度。"""
    scale = _NOISE_SCALE.get(dtype)
    if scale is None:
        raise ValueError(f"{dtype}: 仿真只支持浮点 dtype，收到 {dtype!r}")

    if dtype == "bf16":
        # bf16 用 uint16 承载 float32 的高 16 位（与 make_data 一致）
        values = (np.asarray(array, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(
            np.float32
        )
    elif dtype == "f64":
        values = np.ascontiguousarray(array, dtype=np.float64)
    else:
        values = np.ascontiguousarray(array, dtype=np.float32)

    rng = np.random.default_rng([step & 0xFFFFFFFF, zlib.crc32(name.encode("utf-8"))])
    noise = rng.standard_normal(values.shape, dtype=np.float32) * np.float32(scale)
    updated = values * np.float32(0.999) + noise

    if dtype == "bf16":
        bits = updated.astype(np.float32, copy=False).view(np.uint32)
        return (bits >> np.uint32(16)).astype(np.uint16).reshape(array.shape)
    return updated.astype(numpy_dtype(dtype), copy=False).reshape(array.shape)


def make_state(
    *,
    seed: int = 1234,
    hidden: int = 32,
    layers: int = 1,
    vocab: int = 64,
    intermediate: int = 64,
    dtype: str = "f32",
    with_optimizer: bool = True,
    step: int = 0,
) -> TrainingState:
    """按 ``tests/fixtures`` 的参数表造一份初始 state（模型 + Adam 状态）。"""
    specs = build_specs(
        hidden=hidden,
        layers=layers,
        vocab=vocab,
        intermediate=intermediate,
        dtype=dtype,
        with_optimizer=with_optimizer,
    )
    return TrainingState(
        step=step,
        dtypes={spec.name: spec.dtype for spec in specs},
        values={spec.name: make_tensor(spec, seed) for spec in specs},
    )


def state_from_tensors(
    tensors: Mapping[str, Tensor], *, step: int
) -> TrainingState:
    """把恢复出来的 device 张量变回 host 状态（走 ``Tensor.numpy()``，即 D2H 路径）。"""
    return TrainingState(
        step=step,
        dtypes={name: tensor.dtype for name, tensor in tensors.items()},
        values={name: tensor.numpy() for name, tensor in tensors.items()},
    )


# --- host 数组 → device 张量 -------------------------------------------------


def as_tensor(
    array: np.ndarray,
    *,
    name: str,
    dtype: str,
    device: "str | Device" = "cuda:0",
    backend: Any = None,
) -> Tensor:
    """一块 host 数组 → ``Tensor``：全是 ``allocate`` + ``copy_h2d``，不直接塞内存。"""
    spec = Device.parse(device)
    backend = backend or get_backend(spec)
    payload = np.ascontiguousarray(array).tobytes()
    buffer = backend.allocate(len(payload))
    backend.copy_h2d(buffer, memoryview(payload))
    return Tensor(
        name=name,
        buffer=buffer,
        device=spec,
        dtype=dtype,
        shape=tuple(int(dim) for dim in array.shape),
        backend=backend,
    )


def to_device(
    state: TrainingState,
    *,
    device: "str | Device" = "cuda:0",
    names: Sequence[str] | None = None,
) -> dict[str, Tensor]:
    """整份 state → device 张量（每个张量一次 H2D）。"""
    spec = Device.parse(device)
    backend = get_backend(spec)
    return {
        name: as_tensor(
            state.values[name],
            name=name,
            dtype=state.dtypes[name],
            device=spec,
            backend=backend,
        )
        for name in (names or state.names())
    }


def tensors_for_rank(
    state: TrainingState,
    *,
    rank: int,
    tp: int = 1,
    dims: Mapping[str, int] | None = None,
    device: "str | Device" = "cuda:0",
) -> dict[str, Tensor]:
    """本 rank 在 ``tp`` 布局下持有的张量：列张量取自己那块，其余是完整副本。"""
    dims = dict(dims or {})
    spec = Device.parse(device)
    backend = get_backend(spec)
    tensors: dict[str, Tensor] = {}
    for name in state.names():
        dim = dims.get(name)
        chunk = state.shard(name, dim=dim, index=rank, count=tp)
        tensors[name] = as_tensor(
            chunk,
            name=name,
            dtype=state.dtypes[name],
            device=spec,
            backend=backend,
        )
    return tensors


def column_dims(state: TrainingState) -> dict[str, int]:
    """2 维张量沿 ``dim=1`` 切，1 维张量（norm/bias）保持 Replicate。"""
    return {name: 1 for name, array in state.values.items() if array.ndim >= 2}


def sources_for_rank(
    state: TrainingState, *, rank: int, tp: int = 1, dims: Mapping[str, int] | None = None
) -> list[ShardSource]:
    """本 rank 的分片描述，交给 ``save_local(sources=...)``。"""
    dims = dict(dims or {})
    sources: list[ShardSource] = []
    for name in state.names():
        array = state.values[name]
        dim = dims.get(name)
        if dim is None:
            sources.append(
                ShardSource(
                    fqn=name,
                    global_shape=array.shape,
                    dtype=state.dtypes[name],
                    region=Region((0,) * array.ndim, array.shape),
                    replica_index=rank,
                    n_replicas=tp,
                )
            )
        else:
            sources.append(
                ShardSource(
                    fqn=name,
                    global_shape=array.shape,
                    dtype=state.dtypes[name],
                    region=split(array.shape, dim, rank, tp),
                )
            )
    return sources


# --- 恢复结果的拼装与比对 ---------------------------------------------------


def reassemble(
    parts: Sequence[Mapping[str, np.ndarray]],
    state: TrainingState,
    *,
    dims: Mapping[str, int] | None = None,
) -> dict[str, np.ndarray]:
    """把各 rank 恢复出来的块沿 ``dims`` 拼回全局张量（Replicate 张量取任一份）。"""
    dims = dict(dims or {})
    merged: dict[str, np.ndarray] = {}
    for name in state.names():
        dim = dims.get(name)
        if dim is None:
            merged[name] = parts[0][name]
            continue
        merged[name] = np.concatenate([part[name] for part in parts], axis=dim)
    return merged


def assert_state_matches(
    merged: Mapping[str, np.ndarray],
    expected: TrainingState,
    *,
    names: Iterable[str] | None = None,
) -> None:
    """逐字节比对；失败时给出名字与首个差异偏移，便于定位。"""
    for name in names or expected.names():
        got = np.ascontiguousarray(merged[name])
        want = expected.values[name]
        assert got.shape == want.shape, f"{name}: shape {got.shape} != {want.shape}"
        got_bytes = got.tobytes()
        want_bytes = np.ascontiguousarray(want).tobytes()
        assert got_bytes == want_bytes, (
            f"{name}: {len(got_bytes)} 字节内容不一致（首个差异偏移 "
            f"{next((i for i, (a, b) in enumerate(zip(got_bytes, want_bytes)) if a != b), -1)}）"
        )


# --- rank0 汇总（test-only mailbox，取代未实现的 dist/gather） ----------------


def merge_rank_versions(
    rank_roots: Sequence[Path],
    merged_root: Path,
    *,
    step: int,
    layout: Mapping[str, int] | None = None,
) -> CheckpointIndex:
    """把各 rank 的本地版本合并成一个可恢复版本目录。

    ``dist/`` 的 gather 还没实现，所以 rank0 走「读回每个 rank 的 ``.metadata``」这
    条旁路：索引里的分片直接拼接，数据文件用硬链接搬进合并目录（名字形如
    ``__{rank}_0.distcp``，天然不冲突）。真 gather 落地后这个函数整体删掉。
    """
    storage = LocalStorage(merged_root)
    key = version_key(step)
    merged_dir = storage.path_for(f"{key}/{META_FILE}").parent
    merged_dir.mkdir(parents=True, exist_ok=True)

    metas: dict[str, TensorMeta] = {}
    shards: dict[str, list[ShardMeta]] = {}
    blobs: dict[str, BlobMeta] = {}
    files: dict[str, int] = {}
    for root in rank_roots:
        rank_storage = LocalStorage(root)
        index = CheckpointIndex.loads(
            rank_storage.read(f"{key}/{META_FILE}").decode("utf-8")
        )
        for name, meta in index.tensors.items():
            metas.setdefault(name, meta)
            shards.setdefault(name, []).extend(meta.shards)
        blobs.update(index.blobs)

        referenced = {shard.file for meta in index.tensors.values() for shard in meta.shards}
        referenced |= {blob.file for blob in index.blobs.values()}
        for file in sorted(referenced):
            if file in files:
                continue
            source = rank_storage.path_for(f"{key}/{file}")
            target = merged_dir / file
            os.link(source, target)
            files[file] = target.stat().st_size

    index = CheckpointIndex(
        tensors={
            name: TensorMeta(
                global_shape=metas[name].global_shape,
                dtype=metas[name].dtype,
                shards=tuple(entries),
            )
            for name, entries in shards.items()
        },
        blobs=blobs,
        step=step,
        saved_layout=dict(layout or {}),
    )
    index.validate()

    metadata_bytes = write_metadata(storage, step, index)
    write_commit(
        storage,
        CommitRecord(step=step, metadata_bytes=metadata_bytes, files=files),
    )
    return index


# --- 子进程入口（L2 多 rank / L3 崩溃注入共用） ------------------------------


def save_rank(job: Mapping[str, Any]) -> dict[str, Any]:
    """子进程：把本 rank 的分片写成一份本地版本，返回摘要。

    ``job`` 必须可 pickle（spawn），所以只传数组、dtype 与布局，不传 Tensor。
    """
    import api as ckpt  # 子进程里重新 import，才能拿到干净的 backend 注册表

    values: Mapping[str, np.ndarray] = job["values"]
    dtypes: Mapping[str, str] = job["dtypes"]
    dims: Mapping[str, int] = job["dims"]
    rank: int = job["rank"]
    tp: int = job["tp"]
    step: int = job["step"]

    state = TrainingState(step=step, dtypes=dtypes, values=values)
    tensors = tensors_for_rank(
        state, rank=rank, tp=tp, dims=dims, device=job.get("device", "cuda:0")
    )
    report = ckpt.save_local(
        tensors,
        job["root"],
        step=step,
        blobs=job.get("blobs") or {},
        layout={"tp": tp},
        rank=rank,
        sources=sources_for_rank(state, rank=rank, tp=tp, dims=dims),
        engine=job.get("engine", "auto"),
    )
    return {
        "rank": rank,
        "step": report.step,
        "file": report.file,
        "file_size": report.file_size,
        "state": report.state.name,
        "engine": report.engine,
        "d2h_bytes": report.d2h_bytes,
    }


def save_rank_then_die(job: Mapping[str, Any]) -> dict[str, Any]:
    """子进程：写完数据文件与 ``.metadata``，在写 ``COMMIT`` 之前被 SIGKILL。

    模拟「保存打到一半掉电/被 kill」：数据文件与索引都已落盘，提交点缺失。
    这里不随机 kill，而是精确地让写 COMMIT 的那一步直接终止进程，所以结果是
    确定的半成品目录，不会 flaky。
    """
    import os
    import signal

    import api as ckpt

    def kill_before_commit(*args: Any, **kwargs: Any) -> None:
        # 不执行 original：COMMIT 绝不能落盘，留下的正是「数据齐、提交点缺失」
        os.kill(os.getpid(), signal.SIGKILL)

    ckpt.write_commit = kill_before_commit  # type: ignore[assignment]
    save_rank(job)
    raise AssertionError("unreachable: 子进程应当已经被 SIGKILL")


def save_and_upload_rank(job: Mapping[str, Any]) -> dict[str, Any]:
    """子进程：先本地提交本 rank 的分片，再把它独立上传到对象存储。

    ``job["s3"]`` 是 ``S3Config`` 的关键字参数（endpoint/bucket/prefix/凭据），
    子进程自己建连接 —— 与「每个 rank 一条独立上传通道」的真实形态一致。
    """
    import api as ckpt
    from storage.object import S3Config, S3Storage

    info = save_rank(job)
    storage = S3Storage(S3Config(**job["s3"]))
    try:
        info["uploaded"] = ckpt.upload_local(
            job["root"], storage, step=job["step"], rank=job["rank"]
        )
    finally:
        storage.close()
    return info
