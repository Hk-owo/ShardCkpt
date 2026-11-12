"""保存计划：去重、写者分配、覆盖校验。

纯函数：输入「本 rank 可见的分片」（来源是一段本地 buffer 或一个本地张量区域），
输出「写哪个文件、写到哪个字节偏移、索引里记什么」。不碰 IO、不碰 GPU、
不做集合通信 —— 真正的写入、fsync、gather 到 rank 0 都在 storage / commit 层。

两个关键约定来自设计文档：

- **去重按拓扑指定写者，不做内容比对**：同一张量在 Replicate 维度上有 N 份完全
  相同的数据，``hash(FQN) % N`` 决定哪一份写，其余副本进 ``skipped``。
  这样各副本的写入量按 FQN 均摊，也不会出现「全交给 DP rank 0」。
- **对齐责任在 planner**：顺序写默认走 O_DIRECT，buffer 地址、offset、length 都要
  按块对齐，所以每项之后补齐到 ``align``，padding 字节不落盘数据。
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from core import PlanError
from core.index import BlobMeta, CheckpointIndex, ShardMeta, TensorMeta
from core.metadata import numpy_dtype
from core.region import Interval, Region, Shape, flatten_index, flatten_slice, numel

DEFAULT_ALIGN = 4096
"""顺序写默认按 4 KiB 对齐（O_DIRECT 的块大小）。``align=1`` 可关闭补齐。"""

DEFAULT_FILE = "__{rank}_{shard}.distcp"
"""数据文件名：``__<rank>_<序号>.distcp``，与文档 §3 的示例一致。"""

WriterFn = Callable[[str, int], int]
"""``writer(fqn, n_replicas) -> 写者副本编号``；默认按 FQN 的 CRC32 取模。"""


def default_writer(fqn: str, n_replicas: int) -> int:
    """``hash(FQN) % 副本数``：让各副本的写入量按 FQN 均摊。"""
    if n_replicas < 1:
        raise ValueError(f"n_replicas must be >= 1, got {n_replicas}")
    return zlib.crc32(fqn.encode("utf-8")) % n_replicas


# --- 输入 -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShardSource:
    """本 rank 持有的一个分片，以及它在全局张量里覆盖的范围。

    ``region`` 与 ``flat`` 二选一：

    - ``region``：矩形块，FSDP2 / TP 列切分的自然形态；
    - ``flat``：参数展平后的一维区间，ZeRO / FSDP1 / Megatron distributed
      optimizer 的保存零通信方案（文档 §2 方案 A）。

    ``replica_index`` / ``n_replicas`` 描述该分片在 Replicate 维度上的副本身份：
    内容相同的 N 份里只有 :func:`default_writer` 选中的那份会落盘。
    """

    fqn: str
    global_shape: Shape
    dtype: str
    region: Region | None = None
    flat: Interval | None = None
    replica_index: int = 0
    n_replicas: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "global_shape", tuple(int(v) for v in self.global_shape))
        if not self.fqn:
            raise ValueError("fqn must not be empty")
        if not self.global_shape or any(v <= 0 for v in self.global_shape):
            raise ValueError(f"invalid global_shape {self.global_shape}")
        try:
            numpy_dtype(self.dtype)
        except ValueError as exc:
            raise ValueError(f"unsupported dtype {self.dtype!r}") from exc

        if (self.region is None) == (self.flat is None):
            raise ValueError("exactly one of region / flat must be set")
        if self.n_replicas < 1:
            raise ValueError(f"n_replicas must be >= 1, got {self.n_replicas}")
        if not 0 <= self.replica_index < self.n_replicas:
            raise ValueError(
                f"replica_index {self.replica_index} out of range for "
                f"n_replicas {self.n_replicas}"
            )

        extent = Region((0,) * len(self.global_shape), self.global_shape)
        if self.region is not None:
            if self.region.rank != len(self.global_shape):
                raise ValueError(
                    f"{self.fqn}: region rank {self.region.rank} != global rank "
                    f"{len(self.global_shape)}"
                )
            if self.region.is_empty or not extent.contains(self.region):
                raise ValueError(f"{self.fqn}: region {self.region} outside {self.global_shape}")
        else:
            assert self.flat is not None
            if self.flat.is_empty or self.flat.end > numel(self.global_shape):
                raise ValueError(
                    f"{self.fqn}: flat range {self.flat} outside numel "
                    f"{numel(self.global_shape)}"
                )

    @property
    def is_flat(self) -> bool:
        return self.flat is not None

    @property
    def elements(self) -> int:
        """覆盖的元素个数。"""
        if self.flat is not None:
            return self.flat.length
        assert self.region is not None
        return self.region.numel

    @property
    def nbytes(self) -> int:
        return self.elements * numpy_dtype(self.dtype).itemsize

    @property
    def start_element(self) -> int:
        """覆盖范围在全局张量里的起始展平下标（排序用）。"""
        if self.flat is not None:
            return self.flat.start
        assert self.region is not None
        return flatten_index(self.global_shape, self.region.offsets)

    def segments(self) -> tuple[Region, ...]:
        """覆盖范围的矩形分解：展平区间按行主序拆回 N 维。"""
        if self.flat is not None:
            return tuple(flatten_slice(self.global_shape, self.flat))
        assert self.region is not None
        return (self.region,)


# --- 输出 -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WriteItem:
    """一项待写数据：来源（某分片）→ 目标（文件 + 字节区间）。"""

    fqn: str
    file: str
    byte_offset: int
    length: int
    global_shape: Shape
    dtype: str
    region: Region | None = None
    flat: Interval | None = None
    replica_index: int = 0
    padding: int = 0
    """本项之后为对齐补齐的字节数（不落盘数据）。"""

    @property
    def end(self) -> int:
        """数据部分的右开字节边界（不含 padding）。"""
        return self.byte_offset + self.length

    @property
    def next_offset(self) -> int:
        """下一项的开始偏移（含 padding）。"""
        return self.end + self.padding

    def to_shard_meta(
        self,
        *,
        crc_chunk_bytes: int | None = None,
        crc32c_chunks: tuple[str, ...] = (),
    ) -> ShardMeta:
        if self.flat is not None:
            return ShardMeta.flattened(
                file=self.file,
                byte_offset=self.byte_offset,
                length=self.length,
                start=self.flat.start,
                elements=self.flat.length,
                crc_chunk_bytes=crc_chunk_bytes,
                crc32c_chunks=crc32c_chunks,
            )
        assert self.region is not None
        return ShardMeta.rectangular(
            file=self.file,
            byte_offset=self.byte_offset,
            length=self.length,
            offsets=self.region.offsets,
            shape=self.region.shape,
            crc_chunk_bytes=crc_chunk_bytes,
            crc32c_chunks=crc32c_chunks,
        )


@dataclass(frozen=True, slots=True)
class BlobItem:
    """一项小状态（step / lr scheduler / rng/rank_K）：已序列化好的一段字节。"""

    name: str
    file: str
    byte_offset: int
    length: int
    padding: int = 0

    @property
    def end(self) -> int:
        return self.byte_offset + self.length

    @property
    def next_offset(self) -> int:
        return self.end + self.padding

    def to_blob_meta(
        self,
        *,
        crc_chunk_bytes: int | None = None,
        crc32c_chunks: tuple[str, ...] = (),
    ) -> BlobMeta:
        return BlobMeta(
            file=self.file,
            byte_offset=self.byte_offset,
            length=self.length,
            crc_chunk_bytes=crc_chunk_bytes,
            crc32c_chunks=crc32c_chunks,
        )


@dataclass(frozen=True, slots=True)
class SavePlan:
    """本 rank 的保存计划。"""

    rank: int
    file: str
    items: tuple[WriteItem, ...] = ()
    blobs: tuple[BlobItem, ...] = ()
    skipped: tuple[ShardSource, ...] = ()

    @property
    def size(self) -> int:
        """本 rank 要写的文件大小（含对齐补齐）。"""
        if self.blobs:
            return self.blobs[-1].next_offset
        if self.items:
            return self.items[-1].next_offset
        return 0

    @property
    def data_bytes(self) -> int:
        """有效数据字节数（不含 padding）。"""
        return sum(item.length for item in self.items) + sum(blob.length for blob in self.blobs)

    @property
    def padding_bytes(self) -> int:
        return self.size - self.data_bytes

    def fqns(self) -> list[str]:
        seen: list[str] = []
        for item in self.items:
            if item.fqn not in seen:
                seen.append(item.fqn)
        return seen

    def partial_index(self) -> dict[str, TensorMeta]:
        """本 rank 写出的分片元数据，gather 到 rank 0 后合并成 ``.metadata``。"""
        grouped: dict[str, list[ShardMeta]] = {}
        shapes: dict[str, tuple[Shape, str]] = {}
        for item in self.items:
            grouped.setdefault(item.fqn, []).append(item.to_shard_meta())
            shapes[item.fqn] = (item.global_shape, item.dtype)
        return {
            fqn: TensorMeta(
                global_shape=shapes[fqn][0], dtype=shapes[fqn][1], shards=tuple(shards)
            )
            for fqn, shards in grouped.items()
        }

    def partial_blobs(self) -> dict[str, BlobMeta]:
        return {blob.name: blob.to_blob_meta() for blob in self.blobs}

    def validate(self) -> None:
        """自检：分片在界内、同张量内不重叠、同文件内字节区间不重叠。

        这里只做局部校验（本 rank 只看到一部分分片），全局覆盖校验在 rank 0
        汇总后用 ``CheckpointIndex.validate()`` 做。
        """
        CheckpointIndex(
            tensors=self.partial_index(), blobs=self.partial_blobs()
        ).validate(complete=False)


# --- 构建 -------------------------------------------------------------------


def build_save_plan(
    sources: Iterable[ShardSource],
    *,
    rank: int,
    blobs: Mapping[str, int] | None = None,
    file_name: str | None = None,
    align: int = DEFAULT_ALIGN,
    start_offset: int = 0,
    writer: WriterFn | None = None,
) -> SavePlan:
    """把本 rank 可见的分片编成写计划。

    ``blobs`` 是 ``名字 -> 字节数``，按名字顺序排在分片之后写进同一个文件；
    名字直接进索引，所以用 ``rng/rank_3`` 这类带斜杠的名字也没问题。
    """
    if rank < 0:
        raise ValueError(f"rank must be >= 0, got {rank}")
    if align < 1:
        raise ValueError(f"align must be >= 1, got {align}")
    if start_offset < 0:
        raise ValueError(f"start_offset must be >= 0, got {start_offset}")

    file = file_name or DEFAULT_FILE.format(rank=rank, shard=0)
    pick_writer = writer or default_writer

    candidates: list[ShardSource] = []
    skipped: list[ShardSource] = []
    for source in sources:
        if not isinstance(source, ShardSource):
            raise TypeError(f"expected ShardSource, got {type(source).__name__}")
        owner = pick_writer(source.fqn, source.n_replicas)
        if not 0 <= owner < source.n_replicas:
            raise PlanError(
                f"{source.fqn}: writer {owner} outside replica range "
                f"0..{source.n_replicas - 1}"
            )
        (candidates if owner == source.replica_index else skipped).append(source)

    if not candidates and not blobs:
        return SavePlan(rank=rank, file=file, skipped=tuple(sorted(skipped, key=_order)))

    candidates.sort(key=_order)

    # 同一 FQN 的多个分片必须 dtype / 全局形状一致，否则索引无法表达
    shapes: dict[str, tuple[Shape, str]] = {}
    for source in candidates:
        known = shapes.setdefault(source.fqn, (source.global_shape, source.dtype))
        if known != (source.global_shape, source.dtype):
            raise PlanError(
                f"{source.fqn}: inconsistent shard metadata "
                f"{known} vs {(source.global_shape, source.dtype)}"
            )

    ordered: list[WriteItem | BlobItem] = []
    cursor = start_offset
    for source in candidates:
        ordered.append(
            WriteItem(
                fqn=source.fqn,
                file=file,
                byte_offset=cursor,
                length=source.nbytes,
                global_shape=source.global_shape,
                dtype=source.dtype,
                region=source.region,
                flat=source.flat,
                replica_index=source.replica_index,
            )
        )
        cursor += source.nbytes

    sizes = blobs or {}
    for name in sorted(sizes):
        length = int(sizes[name])
        if length <= 0:
            raise ValueError(f"blob {name!r} must have a positive size, got {length}")
        ordered.append(BlobItem(name=name, file=file, byte_offset=cursor, length=length))
        cursor += length

    cursor = start_offset
    aligned: list[WriteItem | BlobItem] = []
    for index, entry in enumerate(ordered):
        padding = 0 if index == len(ordered) - 1 else (-(cursor + entry.length)) % align
        aligned.append(_with_offsets(entry, cursor, padding))
        cursor += entry.length + padding

    plan = SavePlan(
        rank=rank,
        file=file,
        items=tuple(entry for entry in aligned if isinstance(entry, WriteItem)),
        blobs=tuple(entry for entry in aligned if isinstance(entry, BlobItem)),
        skipped=tuple(sorted(skipped, key=_order)),
    )
    plan.validate()
    return plan


def _order(source: ShardSource) -> tuple[str, int, int]:
    return (source.fqn, source.start_element, source.replica_index)


def _with_offsets(
    entry: "WriteItem | BlobItem", byte_offset: int, padding: int
) -> "WriteItem | BlobItem":
    if isinstance(entry, WriteItem):
        return WriteItem(
            fqn=entry.fqn,
            file=entry.file,
            byte_offset=byte_offset,
            length=entry.length,
            global_shape=entry.global_shape,
            dtype=entry.dtype,
            region=entry.region,
            flat=entry.flat,
            replica_index=entry.replica_index,
            padding=padding,
        )
    return BlobItem(
        name=entry.name,
        file=entry.file,
        byte_offset=byte_offset,
        length=entry.length,
        padding=padding,
    )


__all__ = [
    "DEFAULT_ALIGN",
    "DEFAULT_FILE",
    "BlobItem",
    "SavePlan",
    "ShardSource",
    "WriteItem",
    "WriterFn",
    "build_save_plan",
    "default_writer",
]
