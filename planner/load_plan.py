"""读取计划：求交 → 读取计划，合并相邻读请求。

纯函数：输入「旧索引」+「本 rank 在新布局下需要的区域」，输出
「从哪个文件、哪个字节区间读多少、拷到本地张量的哪里」。不碰 IO、不碰 GPU。

对应设计文档 §3 的第 2~6 步：

1. 按新放置构造目标区域（``TargetTensor``）
2. 与索引里每个旧分片逐维求交
3. 交集 → 一次 ``read_range`` + 拷贝几何（``CopySpan``）
4. 同一文件内相邻请求合并（``Interval`` 的相邻判定）
5. 覆盖校验：拷入元素数必须等于目标区域元素数
6. ``windows()`` 按预算切窗口，控制 CPU staging 内存峰值

布局没变时计划自然退化成「整读自己的旧文件」，不需要单独的快路径代码。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping

from core import PlanError
from core.index import CheckpointIndex, Placement, ShardMeta
from core.metadata import numpy_dtype
from core.region import (
    Region,
    Shape,
    contiguous_blocks,
    flatten_index,
    flatten_slice,
    is_flatten_contiguous,
    numel,
    split,
)

COLUMN_STRATEGIES = ("whole", "strided")
"""列子集（TP 换卡数）的两种读法：

``whole``   整读覆盖该行的整行块再在内存里切，读放大 ≤ 分片倍数，I/O 次数最少；
``strided`` 每行一次 ``read_range``，读放大 1，但碎片 I/O 多（文档 §3 的对比）。
"""


# --- 目标 -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TargetTensor:
    """本 rank 在新布局下需要的一个张量区域。"""

    fqn: str
    global_shape: Shape
    dtype: str
    region: Region
    placement: Placement | None = None

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
        if self.region.rank != len(self.global_shape):
            raise ValueError(
                f"{self.fqn}: region rank {self.region.rank} != global rank "
                f"{len(self.global_shape)}"
            )
        if self.region.is_empty:
            raise ValueError(f"{self.fqn}: target region must not be empty")
        if not Region((0,) * self.region.rank, self.global_shape).contains(self.region):
            raise ValueError(f"{self.fqn}: region {self.region} outside {self.global_shape}")

    @classmethod
    def whole(
        cls,
        fqn: str,
        global_shape: Shape,
        dtype: str,
        *,
        placement: Placement | None = None,
    ) -> "TargetTensor":
        """整个张量都要（Replicate 放置，或布局不变时的退化解）。"""
        shape = tuple(int(v) for v in global_shape)
        return cls(
            fqn=fqn,
            global_shape=shape,
            dtype=dtype,
            region=Region((0,) * len(shape), shape),
            placement=placement or Placement.replicate(),
        )

    @classmethod
    def for_shard(
        cls,
        fqn: str,
        global_shape: Shape,
        dtype: str,
        *,
        dim: int,
        index: int,
        count: int,
        placement: Placement | None = None,
    ) -> "TargetTensor":
        """按新布局沿 ``dim`` 切分后第 ``index`` 份（不整除时前几份各多一个元素）。"""
        return cls(
            fqn=fqn,
            global_shape=global_shape,
            dtype=dtype,
            region=split(global_shape, dim, index, count),
            placement=placement or Placement.shard(dim),
        )

    @property
    def numel(self) -> int:
        return self.region.numel

    @property
    def nbytes(self) -> int:
        return self.numel * numpy_dtype(self.dtype).itemsize


# --- 输出 -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CopySpan:
    """一次拷贝：读块里的一块矩形 → 目标张量里的对应位置。"""

    src_offsets: Shape
    dst_offsets: Shape
    shape: Shape

    def __post_init__(self) -> None:
        object.__setattr__(self, "src_offsets", tuple(int(v) for v in self.src_offsets))
        object.__setattr__(self, "dst_offsets", tuple(int(v) for v in self.dst_offsets))
        object.__setattr__(self, "shape", tuple(int(v) for v in self.shape))
        if not self.shape or any(v <= 0 for v in self.shape):
            raise ValueError(f"copy span must be non-empty, got shape {self.shape}")
        if len(self.src_offsets) != len(self.shape) or len(self.dst_offsets) != len(self.shape):
            raise ValueError(
                f"copy span rank mismatch: src={self.src_offsets} dst={self.dst_offsets} "
                f"shape={self.shape}"
            )

    @property
    def numel(self) -> int:
        return numel(self.shape)

    def src_region(self) -> Region:
        return Region(self.src_offsets, self.shape)

    def dst_region(self) -> Region:
        return Region(self.dst_offsets, self.shape)


@dataclass(frozen=True, slots=True)
class ReadItem:
    """一次 ``read_range``：读块 + 若干拷贝段。

    ``read_shape`` 是读块覆盖的全局矩形（一定在行主序下连续），
    ``byte_offset`` 是它在文件里的位置。``whole`` 策略下读块可能大于实际需要的
    区域，拷贝段只取其中一部分 —— 这就是读放大的来源。
    """

    fqn: str
    file: str
    byte_offset: int
    length: int
    read_offsets: Shape
    read_shape: Shape
    dtype: str
    spans: tuple[CopySpan, ...]
    strategy: str = "contiguous"

    def __post_init__(self) -> None:
        object.__setattr__(self, "read_offsets", tuple(int(v) for v in self.read_offsets))
        object.__setattr__(self, "read_shape", tuple(int(v) for v in self.read_shape))
        object.__setattr__(self, "spans", tuple(self.spans))
        if self.byte_offset < 0:
            raise ValueError(f"byte_offset must be >= 0, got {self.byte_offset}")
        if not self.spans:
            raise ValueError("read item must carry at least one copy span")
        expected = self.read_numel * numpy_dtype(self.dtype).itemsize
        if self.length != expected:
            raise ValueError(
                f"read item length {self.length} inconsistent with read block "
                f"{self.read_numel} elements of {self.dtype} ({expected} bytes)"
            )

    @property
    def end(self) -> int:
        return self.byte_offset + self.length

    @property
    def read_numel(self) -> int:
        return numel(self.read_shape)

    @property
    def copy_numel(self) -> int:
        return sum(span.numel for span in self.spans)

    @property
    def amplification(self) -> float:
        """读放大倍数（读进来的字节 / 真正用到的字节）。"""
        return self.read_numel / self.copy_numel if self.copy_numel else 0.0

    def read_region(self) -> Region:
        return Region(self.read_offsets, self.read_shape)


@dataclass(frozen=True, slots=True)
class ReadPlan:
    """本 rank 的读取计划。"""

    items: tuple[ReadItem, ...] = ()
    missing: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()
    incomplete: tuple[str, ...] = ()

    @property
    def total_bytes(self) -> int:
        """要从存储读进来的总字节（含读放大）。"""
        return sum(item.length for item in self.items)

    @property
    def useful_bytes(self) -> int:
        """真正拷进目标的字节。"""
        return sum(
            span.numel * numpy_dtype(item.dtype).itemsize
            for item in self.items
            for span in item.spans
        )

    @property
    def amplification(self) -> float:
        return self.total_bytes / self.useful_bytes if self.useful_bytes else 0.0

    def files(self) -> list[str]:
        seen: list[str] = []
        for item in self.items:
            if item.file not in seen:
                seen.append(item.file)
        return seen

    def for_target(self, fqn: str) -> tuple[ReadItem, ...]:
        return tuple(item for item in self.items if item.fqn == fqn)

    def windows(self, budget_bytes: int) -> Iterator[tuple[ReadItem, ...]]:
        """按字节预算把读请求切成窗口，控制 staging buffer 的峰值。

        单个请求大于预算时自己占一个窗口（请求本身不能再切，切了就是另一次
        ``read_range``）。
        """
        if budget_bytes <= 0:
            raise ValueError(f"budget_bytes must be > 0, got {budget_bytes}")
        window: list[ReadItem] = []
        used = 0
        for item in self.items:
            if window and used + item.length > budget_bytes:
                yield tuple(window)
                window, used = [], 0
            window.append(item)
            used += item.length
        if window:
            yield tuple(window)


# --- 构建 -------------------------------------------------------------------


def build_load_plan(
    index: CheckpointIndex,
    targets: "Iterable[TargetTensor] | Mapping[str, TargetTensor]",
    *,
    strict: bool = True,
    allow_extra: bool = True,
    column_strategy: str = "whole",
) -> ReadPlan:
    """把索引和新布局的目标区域编成读取计划。

    ``strict`` 下缺失或覆盖不全直接抛 ``PlanError``；``strict=False`` 时把它们
    记在 ``missing`` / ``incomplete`` 里返回，让调用方决定怎么处理。
    ``allow_extra`` 控制「索引里有、目标不需要」的张量是否算错误。
    """
    if column_strategy not in COLUMN_STRATEGIES:
        raise ValueError(
            f"unknown column_strategy {column_strategy!r}, expected one of "
            f"{COLUMN_STRATEGIES}"
        )
    target_map = _target_map(targets)

    collected: list[ReadItem] = []
    missing: list[str] = []
    incomplete: list[str] = []

    for fqn, target in target_map.items():
        meta = index.tensors.get(fqn)
        if meta is None:
            missing.append(fqn)
            continue
        if tuple(meta.global_shape) != target.global_shape:
            raise PlanError(
                f"{fqn}: index global_shape {meta.global_shape} != target "
                f"{target.global_shape}"
            )
        if meta.dtype != target.dtype:
            raise PlanError(f"{fqn}: index dtype {meta.dtype!r} != target {target.dtype!r}")

        itemsize = numpy_dtype(meta.dtype).itemsize
        items: list[ReadItem] = []
        for shard in meta.shards:
            for segment in _segments(meta.global_shape, shard):
                inter = target.region.intersect(segment)
                if inter is None:
                    continue
                blocks, strategy = _plan_blocks(
                    shard, meta.global_shape, segment, inter, column_strategy
                )
                for block in blocks:
                    items.append(
                        _read_item(
                            fqn=fqn,
                            meta_shape=meta.global_shape,
                            dtype=meta.dtype,
                            shard=shard,
                            segment=segment,
                            block=block,
                            inter=inter,
                            target=target,
                            itemsize=itemsize,
                            strategy=strategy,
                        )
                    )

        merged = _merge(items)
        if sum(span.numel for item in merged for span in item.spans) != target.numel:
            incomplete.append(fqn)
        collected.extend(merged)

    collected.sort(key=lambda item: (item.file, item.byte_offset))

    extra = sorted(set(index.tensors) - set(target_map))
    plan = ReadPlan(
        items=tuple(collected),
        missing=tuple(sorted(missing)),
        extra=tuple(extra),
        incomplete=tuple(sorted(incomplete)),
    )

    if strict:
        if plan.missing:
            raise PlanError(f"tensors missing from index: {list(plan.missing)}")
        if plan.incomplete:
            raise PlanError(f"targets not fully covered by index: {list(plan.incomplete)}")
        if plan.extra and not allow_extra:
            raise PlanError(f"index has tensors not requested: {list(plan.extra)}")
    return plan


def _target_map(
    targets: "Iterable[TargetTensor] | Mapping[str, TargetTensor]",
) -> dict[str, TargetTensor]:
    if isinstance(targets, Mapping):
        pairs = list(targets.items())
    else:
        pairs = []
        for target in targets:
            if not isinstance(target, TargetTensor):
                raise TypeError(f"expected TargetTensor, got {type(target).__name__}")
            pairs.append((target.fqn, target))

    out: dict[str, TargetTensor] = {}
    for fqn, target in pairs:
        if not isinstance(target, TargetTensor):
            raise TypeError(f"expected TargetTensor, got {type(target).__name__}")
        if fqn != target.fqn:
            raise ValueError(f"target key {fqn!r} != tensor name {target.fqn!r}")
        if fqn in out:
            raise ValueError(f"duplicate target {fqn!r}")
        out[fqn] = target
    return out


def _segments(global_shape: Shape, shard: ShardMeta) -> tuple[Region, ...]:
    """分片覆盖的矩形块：展平分片按行主序拆回 N 维。"""
    interval = shard.interval
    if interval is None:
        region = shard.region
        assert region is not None
        return (region,)
    return tuple(flatten_slice(global_shape, interval))


def _plan_blocks(
    shard: ShardMeta,
    global_shape: Shape,
    segment: Region,
    inter: Region,
    column_strategy: str,
) -> tuple[list[Region], str]:
    """把交集切成「一次 read_range 能读的连续块」。

    连续性按**分片自己的文件内布局**判断，而不是全局行主序：

    - 矩形分片在文件里按本地 shape 的紧凑布局存放，所以整个分片本身就是连续的，
      即使它在全局行主序下是碎片（TP 列切分）；
    - 展平分片按 ``flat`` 展开顺序存放，连续性等价于全局展平。

    三种结果：

    - 交集本身连续 → 一次读，``contiguous``（读放大 1）；
    - 交集是列子集 → ``whole``：读分片内的整行块再在内存里切（文档 §3 的做法）；
    - 展平分片里的列子集 → ``strided``：拆成最小的连续块，读放大 1、I/O 次数多。
    """
    if shard.is_flat:
        if is_flatten_contiguous(global_shape, inter):
            return [inter], "contiguous"
        # 展平分片里的列子集：整行块不一定落在分片的连续区间内，只能按最小连续块拆
        return contiguous_blocks(global_shape, inter), "strided"

    local = inter.relative_to(segment.offsets)
    if is_flatten_contiguous(tuple(segment.shape), local):
        return [inter], "contiguous"
    if column_strategy == "whole":
        block = Region(
            (local.offsets[0],) + (0,) * (local.rank - 1),
            (local.shape[0],) + tuple(segment.shape[1:]),
        )
        return [block.translate(segment.offsets)], "whole"
    return [
        block.translate(segment.offsets)
        for block in contiguous_blocks(tuple(segment.shape), local)
    ], "strided"


def _read_item(
    *,
    fqn: str,
    meta_shape: Shape,
    dtype: str,
    shard: ShardMeta,
    segment: Region,
    block: Region,
    inter: Region,
    target: TargetTensor,
    itemsize: int,
    strategy: str,
) -> ReadItem:
    copy_region = inter.intersect(block)
    assert copy_region is not None  # inter 与 block 必然相交

    interval = shard.interval
    if interval is not None:
        # 展平分片：文件内按 flat 展开顺序紧密排列
        element = flatten_index(meta_shape, block.offsets) - interval.start
    else:
        # 矩形分片：文件内按分片自身的行主序紧密排列（本地 shape 的布局）
        local = block.relative_to(segment.offsets)
        element = flatten_index(tuple(segment.shape), local.offsets)

    if element < 0:
        raise PlanError(f"{fqn}: read block {block} starts before shard {shard.file}")

    return ReadItem(
        fqn=fqn,
        file=shard.file,
        byte_offset=shard.byte_offset + element * itemsize,
        length=block.numel * itemsize,
        read_offsets=block.offsets,
        read_shape=block.shape,
        dtype=dtype,
        spans=(
            CopySpan(
                src_offsets=copy_region.relative_to(block.offsets).offsets,
                dst_offsets=copy_region.relative_to(target.region.offsets).offsets,
                shape=copy_region.shape,
            ),
        ),
        strategy=strategy,
    )


def _merge(items: list[ReadItem]) -> list[ReadItem]:
    """把同一文件里相邻的读块合成一次 ``read_range``。"""
    if not items:
        return []
    ordered = sorted(items, key=lambda item: (item.file, item.byte_offset))
    merged: list[ReadItem] = [ordered[0]]
    for item in ordered[1:]:
        previous = merged[-1]
        if _can_merge(previous, item):
            merged[-1] = _combine(previous, item)
        else:
            merged.append(item)
    return merged


def _can_merge(left: ReadItem, right: ReadItem) -> bool:
    if left.fqn != right.fqn or left.file != right.file:
        return False
    if left.end != right.byte_offset:
        return False
    if left.read_offsets[1:] != right.read_offsets[1:]:
        return False
    if left.read_shape[1:] != right.read_shape[1:]:
        return False
    return left.read_offsets[0] + left.read_shape[0] == right.read_offsets[0]


def _combine(left: ReadItem, right: ReadItem) -> ReadItem:
    shift = left.read_shape[0]
    spans = list(left.spans)
    for span in right.spans:
        spans.append(
            CopySpan(
                src_offsets=(span.src_offsets[0] + shift,) + span.src_offsets[1:],
                dst_offsets=span.dst_offsets,
                shape=span.shape,
            )
        )
    return ReadItem(
        fqn=left.fqn,
        file=left.file,
        byte_offset=left.byte_offset,
        length=left.length + right.length,
        read_offsets=left.read_offsets,
        read_shape=(left.read_shape[0] + right.read_shape[0],) + left.read_shape[1:],
        dtype=left.dtype,
        spans=tuple(spans),
        strategy=left.strategy,
    )


__all__ = [
    "COLUMN_STRATEGIES",
    "CopySpan",
    "ReadItem",
    "ReadPlan",
    "TargetTensor",
    "build_load_plan",
]
