"""纯 host 落盘闭环（§7.1）：同步 ``persist_snapshot`` 与异步存储子进程共用。

把 ``blobs`` 写入、对齐空隙清零、分片 CRC-32C 索引构建、``LocalStorage.write_view``、
``.metadata``、``GENERATION`` 与 ``COMMIT`` 收敛到单一实现，从结构上保证 I6（逐字节一致）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from commit.generation import Generation
from commit.local import (
    CommitRecord,
    delete_local_generation,
    version_key,
    write_commit,
    write_local_generation,
    write_metadata,
)
from core import PlanError
from core.index import BlobMeta, CheckpointIndex, ShardMeta, TensorMeta
from core.metadata import numpy_dtype
from planner.save_plan import SavePlan
from storage.checksum import CRC_CHUNK_MAX, crc32c_chunks
from storage.local import DEFAULT_ALIGNMENT, LocalStorage
from storage.upload import local_run_root


@dataclass(frozen=True)
class HostPersistResult:
    """一次纯 host 落盘的结果（不含 D2H 细节，不依赖 GPU 状态机）。"""

    step: int
    key: str
    file: str
    size: int
    data_bytes: int
    blob_bytes: int
    engine: str
    index: CheckpointIndex
    metadata_bytes: int
    metadata_crc32c: str
    commit: CommitRecord
    generation: str | None = None


def as_view(buffer: Any) -> memoryview:
    """把 host buffer 规范成 ``memoryview``。"""
    if isinstance(buffer, memoryview):
        return buffer
    view = getattr(buffer, "view", None)
    return view if isinstance(view, memoryview) else memoryview(buffer)


def clear_padding(plan: SavePlan, view: memoryview, end: int) -> None:
    """对齐缓冲区未初始化，将不属于任何张量/小状态的空隙字节清零。"""
    spans = [(item.byte_offset, item.end) for item in plan.items]
    spans += [(blob.byte_offset, blob.end) for blob in plan.blobs]
    cursor = 0
    for start, stop in sorted(spans):
        if start > cursor:
            view[cursor:start] = b"\x00" * (start - cursor)
        cursor = max(cursor, stop)
    if cursor < end:
        view[cursor:end] = b"\x00" * (end - cursor)


def build_index(
    plan: SavePlan,
    *,
    buffer: Any,
    step: int,
    layout: Mapping[str, int] | None,
    crc_chunk_max: int,
) -> CheckpointIndex:
    """从已经躺在 host buffer 里的有效字节切片直接构造 ``CheckpointIndex``。"""
    view = as_view(buffer)
    grouped: dict[str, list[ShardMeta]] = {}
    shapes: dict[str, tuple[tuple[int, ...], str]] = {}
    for item in plan.items:
        chunk_bytes, chunks = crc32c_chunks(
            view[item.byte_offset : item.end],
            itemsize=numpy_dtype(item.dtype).itemsize,
            max_bytes=crc_chunk_max,
        )
        grouped.setdefault(item.fqn, []).append(
            item.to_shard_meta(crc_chunk_bytes=chunk_bytes, crc32c_chunks=chunks)
        )
        shapes[item.fqn] = (item.global_shape, item.dtype)

    tensors = {
        fqn: TensorMeta(
            global_shape=shapes[fqn][0], dtype=shapes[fqn][1], shards=tuple(shards)
        )
        for fqn, shards in grouped.items()
    }
    blobs: dict[str, BlobMeta] = {}
    for blob in plan.blobs:
        chunk_bytes, chunks = crc32c_chunks(
            view[blob.byte_offset : blob.end], max_bytes=crc_chunk_max
        )
        blobs[blob.name] = blob.to_blob_meta(
            crc_chunk_bytes=chunk_bytes, crc32c_chunks=chunks
        )
    return CheckpointIndex(
        tensors=tensors, blobs=blobs, step=step, saved_layout=dict(layout or {})
    )


def persist_host_buffer(
    buffer: Any,
    root: str | Path,
    *,
    step: int,
    plan: SavePlan,
    aligned_size: int,
    blobs: Mapping[str, bytes] | None = None,
    layout: Mapping[str, int] | None = None,
    alignment: int = DEFAULT_ALIGNMENT,
    engine: str = "auto",
    direct_io: bool = True,
    crc_chunk_max: int = CRC_CHUNK_MAX,
    generation: Generation | str | None = None,
    run_id: str | None = None,
    on_d2h_done: Callable[[], None] | None = None,
    on_persist_ok: Callable[[], None] | None = None,
    on_commit_ok: Callable[[], None] | None = None,
    write_metadata_fn: Callable[..., bytes] = write_metadata,
    write_commit_fn: Callable[..., None] = write_commit,
) -> HostPersistResult:
    """在已拷贝好显存权重的 host buffer 上完成落盘闭环（Q4 / §7.1）。

    步骤：
    1. 写入 ``blobs`` 至尾部偏移并执行 ``clear_padding``；
    2. 触发 ``on_d2h_done()``（若有）；
    3. 构建并校验 ``CheckpointIndex``（含分块 CRC-32C）；
    4. 若 ``aligned_size > 0``，通过 ``LocalStorage.write_view`` 零拷贝写数据大文件；
    5. 写 ``.metadata`` 并触发 ``on_persist_ok()``；
    6. 写或清理本地 ``GENERATION`` 代际标记；
    7. 原子写入 ``COMMIT`` 并触发 ``on_commit_ok()``。
    """
    view = as_view(buffer)
    if len(view) < aligned_size:
        raise PlanError(
            f"step {step}: host buffer {len(view)} 字节小于对齐计划大小 {aligned_size}"
        )

    blob_payloads = dict(blobs or {})
    blob_bytes = 0
    for blob in plan.blobs:
        payload = blob_payloads.get(blob.name)
        if payload is None:
            raise PlanError(f"{blob.name}: 缺少小状态字节")
        if len(payload) != blob.length:
            raise PlanError(
                f"{blob.name}: {len(payload)} 字节与计划里的 {blob.length} 不一致"
            )
        view[blob.byte_offset : blob.end] = payload
        blob_bytes += len(payload)

    clear_padding(plan, view, aligned_size)
    if on_d2h_done is not None:
        on_d2h_done()

    index = build_index(
        plan,
        buffer=view,
        step=step,
        layout=layout,
        crc_chunk_max=crc_chunk_max,
    )
    index.validate(complete=False)

    scoped_root = local_run_root(root, run_id)
    storage = LocalStorage(
        scoped_root, direct_io=direct_io, alignment=alignment, engine=engine
    )
    key = version_key(step)
    gen_id: str | None = None
    if generation is not None:
        gen_id = generation.id if isinstance(generation, Generation) else str(generation)

    files: dict[str, int] = {}
    if aligned_size:
        storage.write_view(f"{key}/{plan.file}", view[:aligned_size])
        file_size = storage.path_for(f"{key}/{plan.file}").stat().st_size
        if file_size != aligned_size:
            raise PlanError(
                f"{key}/{plan.file}: 写入 {file_size} 字节，期望 {aligned_size}"
            )
        files[plan.file] = file_size

    metadata = write_metadata_fn(storage, step, index)
    if on_persist_ok is not None:
        on_persist_ok()

    if gen_id is not None:
        write_local_generation(storage, step, gen_id, run_id=run_id)
    else:
        delete_local_generation(storage, step)

    record = CommitRecord.of_metadata(
        step=step,
        metadata=metadata,
        files=files,
    )
    write_commit_fn(storage, record)
    if on_commit_ok is not None:
        on_commit_ok()

    return HostPersistResult(
        step=step,
        key=key,
        file=plan.file,
        size=aligned_size,
        data_bytes=plan.data_bytes,
        blob_bytes=blob_bytes,
        engine=storage.engine,
        index=index,
        metadata_bytes=record.metadata_bytes,
        metadata_crc32c=record.metadata_crc32c,
        commit=record,
        generation=gen_id,
    )


__all__ = [
    "HostPersistResult",
    "as_view",
    "build_index",
    "clear_padding",
    "persist_host_buffer",
]
