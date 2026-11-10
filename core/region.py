"""N 维区域、求交、一维区间（展平切分）。

纯函数、零依赖：后续规划器（planner/）的求交与切分都建立在这里。

术语
----
``Interval``
    一维半开区间 ``[start, start + length)``，用于「参数展平后的偏移」。
    相邻但不重叠的两段能用 :meth:`Interval.touches` / :meth:`Interval.merge`
    合成一次读写，planner 合并相邻请求依赖这一点。
``Region``
    N 维矩形块（``offsets`` + ``shape``），坐标相对全局张量的原点。

行主序（row-major）与 numpy / PyTorch 一致：最后一维变化最快。
所以「除首维外每维都满覆盖」的矩形块在展平后仍是连续区间，
而列子集是「每行一段」的碎片 —— 两边的换算由 :func:`flatten_range` 与
:func:`flatten_slice` 负责。
"""

from __future__ import annotations

from dataclasses import dataclass

Shape = tuple[int, ...]


def numel(shape: Shape) -> int:
    """``shape`` 的元素个数；``()`` 视为标量，返回 1。"""
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


# --- 一维区间 ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Interval:
    """半开区间 ``[start, start + length)``。"""

    start: int
    length: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "length", int(self.length))
        if self.start < 0:
            raise ValueError(f"interval start must be >= 0, got {self.start}")
        if self.length < 0:
            raise ValueError(f"interval length must be >= 0, got {self.length}")

    @classmethod
    def of(cls, start: int, end: int) -> "Interval":
        """用 ``[start, end)`` 构造。"""
        if end < start:
            raise ValueError(f"interval end {end} < start {start}")
        return cls(start, int(end) - int(start))

    @property
    def end(self) -> int:
        return self.start + self.length

    @property
    def is_empty(self) -> bool:
        return self.length == 0

    def overlaps(self, other: "Interval") -> bool:
        """是否有交集（共享至少一个元素）。"""
        return self.start < other.end and other.start < self.end

    def intersect(self, other: "Interval") -> "Interval | None":
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return Interval.of(start, end) if start < end else None

    def contains(self, other: "Interval") -> bool:
        return self.start <= other.start and other.end <= self.end

    def touches(self, other: "Interval") -> bool:
        """重叠或紧邻；合并后仍是一段连续区间。"""
        return self.start <= other.end and other.start <= self.end

    def merge(self, other: "Interval") -> "Interval":
        if not self.touches(other):
            raise ValueError(f"cannot merge disjoint intervals {self} and {other}")
        return Interval.of(min(self.start, other.start), max(self.end, other.end))

    def to_slice(self) -> slice:
        return slice(self.start, self.end)

    def __repr__(self) -> str:
        return f"Interval({self.start}, {self.length})"


# --- N 维区域 ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Region:
    """N 维矩形块：从 ``offsets`` 起，每维覆盖 ``shape`` 个元素。"""

    offsets: Shape
    shape: Shape

    def __post_init__(self) -> None:
        object.__setattr__(self, "offsets", tuple(int(v) for v in self.offsets))
        object.__setattr__(self, "shape", tuple(int(v) for v in self.shape))
        if len(self.offsets) != len(self.shape):
            raise ValueError(
                f"offsets rank {len(self.offsets)} != shape rank {len(self.shape)}"
            )
        if not self.shape:
            raise ValueError("region requires at least one dim; use Interval for flat ranges")
        if any(v < 0 for v in self.offsets):
            raise ValueError(f"region offsets must be >= 0, got {self.offsets}")
        if any(v < 0 for v in self.shape):
            raise ValueError(f"region shape must be >= 0, got {self.shape}")

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return numel(self.shape)

    @property
    def end(self) -> Shape:
        """每维的右开边界。"""
        return tuple(o + s for o, s in zip(self.offsets, self.shape))

    @property
    def is_empty(self) -> bool:
        return any(dim == 0 for dim in self.shape)

    @property
    def slices(self) -> tuple[slice, ...]:
        return tuple(slice(o, o + s) for o, s in zip(self.offsets, self.shape))

    def overlaps(self, other: "Region") -> bool:
        return self.intersect(other) is not None

    def intersect(self, other: "Region") -> "Region | None":
        """逐维求交：``start = max``、``end = min``，每维都满足 ``start < end`` 才算重叠。"""
        if self.rank != other.rank:
            raise ValueError(f"cannot intersect rank {self.rank} with rank {other.rank}")
        offsets: list[int] = []
        shape: list[int] = []
        for a_start, a_len, b_start, b_len in zip(
            self.offsets, self.shape, other.offsets, other.shape
        ):
            start = max(a_start, b_start)
            end = min(a_start + a_len, b_start + b_len)
            if start >= end:
                return None
            offsets.append(start)
            shape.append(end - start)
        return Region(tuple(offsets), tuple(shape))

    def contains(self, other: "Region") -> bool:
        """``other`` 是否完整落在本区域里（等价于 ``intersect == other``）。"""
        if self.rank != other.rank:
            raise ValueError(f"cannot compare rank {self.rank} with rank {other.rank}")
        return all(
            a_start <= b_start and b_start + b_len <= a_start + a_len
            for a_start, a_len, b_start, b_len in zip(
                self.offsets, self.shape, other.offsets, other.shape
            )
        )

    def translate(self, by: Shape) -> "Region":
        """整体平移（``by`` 可以为负，结果坐标仍需非负）。"""
        if len(by) != self.rank:
            raise ValueError(f"translate rank {len(by)} != region rank {self.rank}")
        return Region(tuple(o + int(d) for o, d in zip(self.offsets, by)), self.shape)

    def relative_to(self, base: Shape) -> "Region":
        """把坐标换算成相对 ``base`` 原点的坐标，用于「分片内 / 目标内」定位。"""
        if len(base) != self.rank:
            raise ValueError(f"base rank {len(base)} != region rank {self.rank}")
        offsets: list[int] = []
        for start, origin in zip(self.offsets, base):
            if start < origin:
                raise ValueError(
                    f"region offsets {self.offsets} not inside block origin {base}"
                )
            offsets.append(start - origin)
        return Region(tuple(offsets), self.shape)

    def __repr__(self) -> str:
        return f"Region({self.offsets}, {self.shape})"


# --- 展平换算 ---------------------------------------------------------------


def flatten_index(shape: Shape, coords: Shape) -> int:
    """行主序下 ``coords`` 对应的一维下标。"""
    if len(shape) != len(coords):
        raise ValueError(f"coords rank {len(coords)} != shape rank {len(shape)}")
    index = 0
    for dim, coord in zip(shape, coords):
        if not 0 <= int(coord) < int(dim):
            raise ValueError(f"coord {coord} out of range for dim {dim}")
        index = index * int(dim) + int(coord)
    return index


def is_row_contiguous(shape: Shape, region: Region) -> bool:
    """``region`` 在行主序展平后是否仍是一段连续区间。

    充要条件是「除首维外每维都满覆盖」。空区域视为连续。
    """
    if len(shape) != region.rank:
        raise ValueError(f"shape rank {len(shape)} != region rank {region.rank}")
    if region.is_empty:
        return True
    tail_offsets = (0,) * (region.rank - 1)
    tail_shape = tuple(int(d) for d in shape[1:])
    return region.offsets[1:] == tail_offsets and region.shape[1:] == tail_shape


def is_flatten_contiguous(shape: Shape, region: Region) -> bool:
    """``region`` 在行主序展平后是否为一段连续区间（比 :func:`is_row_contiguous` 宽）。

    :func:`flatten_slice` 拆出来的尾段是「某一行内的前缀子段」（首维为 1、
    其余维不满），它同样连续，但 :func:`is_row_contiguous` 会判成不连续。
    判断读取计划能不能合并成一次 ``read_range`` 时要用这个宽松版本。
    """
    if len(shape) != region.rank:
        raise ValueError(f"shape rank {len(shape)} != region rank {region.rank}")
    if region.is_empty or region.rank == 1:
        return True
    tail_offsets = (0,) * (region.rank - 1)
    if region.offsets[1:] == tail_offsets and region.shape[1:] == tuple(
        int(d) for d in shape[1:]
    ):
        return True
    if region.shape[0] == 1:
        return is_flatten_contiguous(
            tuple(int(d) for d in shape[1:]), Region(region.offsets[1:], region.shape[1:])
        )
    return False


def contiguous_blocks(shape: Shape, region: Region) -> list[Region]:
    """把任意矩形区域拆成若干「展平后连续」的块，用最少的分段数覆盖它。

    列子集这类区域在展平后是每行一段，只能拆；首维满覆盖的块原样返回。
    """
    if is_flatten_contiguous(shape, region):
        return [region]
    if region.rank == 1:
        return [region]
    blocks: list[Region] = []
    if region.shape[0] > 1:
        for index in range(region.shape[0]):
            row = Region(
                (region.offsets[0] + index,) + region.offsets[1:],
                (1,) + region.shape[1:],
            )
            blocks.extend(contiguous_blocks(shape, row))
        return blocks
    tail = Region(region.offsets[1:], region.shape[1:])
    return [
        Region((region.offsets[0],) + block.offsets, (1,) + block.shape)
        for block in contiguous_blocks(tuple(int(d) for d in shape[1:]), tail)
    ]


def flatten_range(shape: Shape, region: Region) -> Interval:
    """连续区域 → 展平后的一维区间；区域不连续时抛 ``ValueError``。"""
    if not is_row_contiguous(shape, region):
        raise ValueError(f"region {region} is not contiguous in row-major order")
    inner = numel(tuple(shape[1:]))
    return Interval(region.offsets[0] * inner, region.numel)


def flatten_slice(shape: Shape, interval: Interval) -> list[Region]:
    """展平区间 → 覆盖同样元素的一组矩形块，按行主序排列。

    这是「方案 A（保存零通信）」的核心换算：ZeRO / FSDP1 的分片在展平后
    是一维区间 ``[s, e)``，落回 N 维全局张量时通常不是矩形块，
    要拆成「首段 + 中间整段 + 尾段」，段数最多是 ``2 * rank``。
    """
    shape = tuple(int(d) for d in shape)
    if not shape:
        raise ValueError("flatten_slice requires at least one dim")
    if interval.is_empty:
        return []
    total = numel(shape)
    if interval.end > total:
        raise ValueError(f"interval {interval} exceeds numel {total} of shape {shape}")
    return _decompose(shape, interval.start, interval.length)


def _decompose(shape: Shape, start: int, length: int) -> list[Region]:
    """把 ``[start, start + length)`` 拆成覆盖它的矩形块列表。"""
    if length == 0:
        return []
    if len(shape) == 1:
        return [Region((start,), (length,))]

    inner = numel(shape[1:])
    pieces: list[Region] = []
    row = start // inner
    offset_in_row = start % inner
    remaining = length

    if offset_in_row:
        take = min(remaining, inner - offset_in_row)
        pieces.extend(_prepend_row(row, _decompose(shape[1:], offset_in_row, take)))
        remaining -= take
        row += 1

    full_rows, remaining = divmod(remaining, inner)
    if full_rows:
        pieces.append(
            Region((row,) + (0,) * (len(shape) - 1), (full_rows,) + shape[1:])
        )
        row += full_rows

    if remaining:
        pieces.extend(_prepend_row(row, _decompose(shape[1:], 0, remaining)))

    return pieces


def _prepend_row(row: int, pieces: list[Region]) -> list[Region]:
    return [Region((row,) + piece.offsets, (1,) + piece.shape) for piece in pieces]


# --- 切分 -------------------------------------------------------------------


def split(shape: Shape, dim: int, index: int, count: int) -> Region:
    """沿 ``dim`` 把全局张量切成 ``count`` 份后，第 ``index`` 份覆盖的区域。

    不整除时前 ``rem`` 份各多一个元素（FSDP2 的切分规则）。
    """
    shape = tuple(int(d) for d in shape)
    if not shape:
        raise ValueError("split requires at least one dim")
    if not 0 <= dim < len(shape):
        raise ValueError(f"dim {dim} out of range for rank {len(shape)}")
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if not 0 <= index < count:
        raise ValueError(f"index {index} out of range for count {count}")

    base, rem = divmod(shape[dim], count)
    start = index * base + min(index, rem)
    length = base + (1 if index < rem else 0)

    offsets = [0] * len(shape)
    offsets[dim] = start
    sizes = list(shape)
    sizes[dim] = length
    return Region(tuple(offsets), tuple(sizes))


__all__ = [
    "Interval",
    "Region",
    "Shape",
    "contiguous_blocks",
    "flatten_index",
    "flatten_range",
    "flatten_slice",
    "is_flatten_contiguous",
    "is_row_contiguous",
    "numel",
    "split",
]
