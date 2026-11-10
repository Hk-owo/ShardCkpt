"""save_local / load：唯一把 core / planner / storage / async_ / commit 串起来的地方。

对外接口按「从 GPU 读取」设计：

1. 加载时显式指定目标设备（对应 ``torch.load(map_location=...)``）
2. 数据路径固定为  host 文件 -> staging buffer -> H2D 拷贝 -> device buffer
3. 迁移与取值都经过 backend，调用方从不直接触碰 device 内存

示例：
    >>> import api as ckpt
    >>> data = ckpt.load("/tmp/data/step_0", device="cuda:0")
    >>> data["model.embed_tokens.weight"].device
    Device(type='cuda', index=0)
    >>> data.to("cpu")["model.embed_tokens.weight"].numpy().shape
    (1000, 128)
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from async_ import (
    BackendStats,
    CpuBackend,
    DeviceBackend,
    MockCudaBackend,
    PinnedBufferPool,
    Snapshot,
    SnapshotManager,
    SnapshotState,
    Tensor,
    TriggerDecision,
    aligned_allocator,
    available_device_types,
    get_backend,
    register_backend,
    reset_registry,
)
from commit.local import (
    CommitRecord,
    update_latest,
    version_key,
    write_commit,
    write_metadata,
)
from core.index import BlobMeta, CheckpointIndex, ShardMeta, TensorMeta
from core.region import Region
from planner.save_plan import SavePlan, ShardSource, build_save_plan
from storage.local import DEFAULT_ALIGNMENT, LocalStorage
from core import (
    CheckpointError,
    ChecksumError,
    FormatError,
    MissingTensorError,
    PlanError,
    UnsupportedDeviceError,
)
from core.metadata import CPU, Device, cuda
from storage.fs import CheckpointData, CheckpointReader, LoadOptions


def load(
    path: str | Path,
    *,
    device: "str | Device" = "cpu",
    verify_crc: bool = True,
    staging: bool = True,
) -> CheckpointData:
    """便捷入口，对应 ``torch.load(path, map_location=device)``。"""
    reader = CheckpointReader.open(
        path, device=device, verify_crc=verify_crc, staging=staging
    )
    return reader.read_all()


def save_async(state_dict: Any, path: str | Path, **options: Any) -> Any:
    """异步保存入口：后台进程 + 上传队列还没做，现在请用 :func:`save_local`。"""
    raise NotImplementedError("save_async is not implemented yet; use save_local()")


# --- 本地层的写闭环 ---------------------------------------------------------


@dataclass(frozen=True)
class SaveReport:
    """一次本地保存的结果。``state`` 成功时是 ``COMMITTED``。"""

    step: int
    key: str
    file: str
    size: int
    data_bytes: int
    engine: str
    index: CheckpointIndex
    commit: CommitRecord
    state: SnapshotState
    d2h_bytes: int

    @property
    def file_size(self) -> int:
        return self.commit.files.get(self.file, 0)


def _sources_from_tensors(tensors: Mapping[str, Tensor]) -> list[ShardSource]:
    """没有分片信息时，每个张量整体作为一个 Replicate 分片。"""
    sources: list[ShardSource] = []
    for name, tensor in tensors.items():
        shape = tuple(int(dim) for dim in tensor.shape)
        if not shape:
            raise ValueError(f"{name}: 0 维张量请走 blobs 路径，不要当分片")
        sources.append(
            ShardSource(
                fqn=name,
                global_shape=shape,
                dtype=tensor.dtype,
                region=Region(tuple(0 for _ in shape), shape),
            )
        )
    return sources


def _as_view(buffer: Any) -> memoryview:
    view = getattr(buffer, "view", None)
    return view if isinstance(view, memoryview) else memoryview(buffer)


def _crc32_hex(view: memoryview) -> str:
    """直接对 buffer 切片算 CRC（零拷贝，zlib 接受 buffer 协议）。"""
    return f"{zlib.crc32(view) & 0xFFFFFFFF:08x}"


def _build_index(
    plan: SavePlan,
    view: memoryview,
    *,
    step: int,
    layout: Mapping[str, int] | None,
) -> CheckpointIndex:
    """从「已经躺在 buffer 里的字节」直接生成索引（顺带算每个分片的 CRC）。"""
    grouped: dict[str, list[ShardMeta]] = {}
    shapes: dict[str, tuple[tuple[int, ...], str]] = {}
    for item in plan.items:
        # 字段名沿用索引格式里的 crc32c，值这里是 zlib.crc32（见 commit/local.py）
        crc = _crc32_hex(view[item.byte_offset : item.end])
        grouped.setdefault(item.fqn, []).append(item.to_shard_meta(crc32c=crc))
        shapes[item.fqn] = (item.global_shape, item.dtype)

    tensors = {
        fqn: TensorMeta(
            global_shape=shapes[fqn][0], dtype=shapes[fqn][1], shards=tuple(shards)
        )
        for fqn, shards in grouped.items()
    }
    blobs: dict[str, BlobMeta] = plan.partial_blobs()
    return CheckpointIndex(
        tensors=tensors, blobs=blobs, step=step, saved_layout=dict(layout or {})
    )


def _clear_padding(plan: SavePlan, view: memoryview, end: int) -> None:
    """对齐缓冲区没初始化，必须把「不属于任何项」的字节清零，否则写盘内容不确定。"""
    spans = [(item.byte_offset, item.end) for item in plan.items]
    spans += [(blob.byte_offset, blob.end) for blob in plan.blobs]
    cursor = 0
    for start, stop in sorted(spans):
        if start > cursor:
            view[cursor:start] = b"\x00" * (start - cursor)
        cursor = max(cursor, stop)
    if cursor < end:
        view[cursor:end] = b"\x00" * (end - cursor)


def save_local(
    tensors: Mapping[str, Tensor],
    root: str | Path,
    *,
    step: int,
    blobs: Mapping[str, bytes] | None = None,
    layout: Mapping[str, int] | None = None,
    rank: int = 0,
    sources: list[ShardSource] | None = None,
    alignment: int = DEFAULT_ALIGNMENT,
    engine: str = "auto",
    direct_io: bool = True,
    pool: PinnedBufferPool | None = None,
    manager: SnapshotManager | None = None,
) -> SaveReport:
    """把一组 device 上的张量写成一份可恢复的本地 checkpoint（同步版本）。

    顺序固定：plan → 取 pinned 块 → 按 ``item.byte_offset`` D2H → 写数据文件 →
    写 ``.metadata`` → 写 ``COMMIT`` → 更新 ``latest``，每一步都对应状态机的一条边。

    ``tensors``: FQN -> ``Tensor``（真机在 GPU 上，mock 下是 host 内存）
    ``blobs``:   小状态的**已序列化字节**（RNG / lr scheduler / dataloader 位置）
    ``layout``:  写进索引的 ``saved_layout``，只用于诊断与快路径
    """
    blob_payloads = dict(blobs or {})
    if sources is None:
        sources = _sources_from_tensors(tensors)

    plan = build_save_plan(
        sources,
        rank=rank,
        blobs={name: len(payload) for name, payload in blob_payloads.items()},
        align=alignment,
    )
    plan.validate()
    if plan.size == 0:
        raise PlanError("保存计划是空的：没有分片也没有小状态")

    # O_DIRECT 要求长度对齐：文件按块边界补齐，读取端按索引里的 length 截断
    aligned_size = -(-plan.size // alignment) * alignment

    if pool is None:
        chunk = max(aligned_size, alignment)
        pool = PinnedBufferPool(
            chunk_bytes=chunk,
            min_chunks=1,
            max_bytes=chunk,
            timeout=0,
            allocator=aligned_allocator(alignment),
        )
    snapshot_manager = manager or SnapshotManager(pool)

    trigger = snapshot_manager.trigger(step, nbytes=aligned_size)
    if trigger.decision is not TriggerDecision.START or trigger.snapshot is None:
        raise PlanError(f"step {step}: 拿不到 pinned buffer（{trigger.reason}）")
    snapshot: Snapshot = trigger.snapshot

    storage = LocalStorage(root, direct_io=direct_io, alignment=alignment, engine=engine)
    key = version_key(step)

    try:
        d2h_bytes = 0
        for item in plan.items:
            tensor = tensors.get(item.fqn)
            if tensor is None:
                raise PlanError(f"{item.fqn}: 计划里有，但没有对应的张量")
            if tensor.nbytes != item.length:
                raise PlanError(
                    f"{item.fqn}: 本地张量 {tensor.nbytes} 字节与计划里的 "
                    f"{item.length} 不一致"
                )
            d2h_bytes += tensor.copy_to(snapshot.buffer, offset=item.byte_offset)

        view = _as_view(snapshot.buffer)
        for blob in plan.blobs:
            payload = blob_payloads.get(blob.name)
            if payload is None:
                raise PlanError(f"{blob.name}: 缺少小状态字节")
            if len(payload) != blob.length:
                raise PlanError(
                    f"{blob.name}: {len(payload)} 字节与计划里的 {blob.length} 不一致"
                )
            view[blob.byte_offset : blob.end] = payload
            d2h_bytes += len(payload)

        _clear_padding(plan, view, aligned_size)
        snapshot.d2h_done()  # host 数据从这一刻起不可变

        index = _build_index(plan, view, step=step, layout=layout)
        index.validate()

        # 1) 数据文件：临时文件 -> fsync -> rename -> 目录 fsync
        storage.write_view(f"{key}/{plan.file}", view[:aligned_size])
        file_size = storage.path_for(f"{key}/{plan.file}").stat().st_size
        if file_size != aligned_size:
            raise PlanError(f"{key}/{plan.file}: 写入 {file_size} 字节，期望 {aligned_size}")

        # 2) 索引，也兼作提交点之一
        metadata_crc32, metadata_bytes = write_metadata(storage, step, index)
        snapshot.persist_ok()  # 本 rank 的数据已落地

        # 3) COMMIT + latest：这两步之后这一版才可恢复
        record = CommitRecord(
            step=step,
            metadata_crc32=metadata_crc32,
            metadata_bytes=metadata_bytes,
            files={plan.file: file_size},
        )
        write_commit(storage, record)
        update_latest(storage, step)
        snapshot.commit_ok()
    except BaseException as exc:
        # 状态机自己区分：还没写过文件 -> DROPPED；有残缺目录 -> ABORTED
        snapshot.fail(f"{type(exc).__name__}: {exc}")
        raise

    return SaveReport(
        step=step,
        key=key,
        file=plan.file,
        size=aligned_size,
        data_bytes=plan.data_bytes,
        engine=storage.engine,
        index=index,
        commit=record,
        state=snapshot.state,
        d2h_bytes=d2h_bytes,
    )


__all__ = [
    "CPU",
    "BackendStats",
    "CheckpointData",
    "CheckpointError",
    "CheckpointReader",
    "ChecksumError",
    "CpuBackend",
    "Device",
    "DeviceBackend",
    "FormatError",
    "LoadOptions",
    "MissingTensorError",
    "MockCudaBackend",
    "PinnedBufferPool",
    "SaveReport",
    "SnapshotManager",
    "SnapshotState",
    "Tensor",
    "TriggerDecision",
    "UnsupportedDeviceError",
    "available_device_types",
    "cuda",
    "get_backend",
    "load",
    "register_backend",
    "reset_registry",
    "save_async",
    "save_local",
]
