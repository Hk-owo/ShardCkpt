"""分片元数据与 checkpoint 索引：文档 §3 那份 ``.metadata`` 的数据结构与序列化。

零依赖（除 numpy 的 dtype 表），planner / storage / commit 共用：

- ``Placement``   DTensor 语义的放置方式：``Replicate`` 或 ``Shard(dim)``
- ``ShardMeta``   一个分片在存储中的位置（文件 + 字节区间）与它覆盖的全局坐标
- ``TensorMeta``  一个全局张量的全部分片，负责覆盖校验
- ``BlobMeta``    小状态的字节块（step / lr scheduler / rng/rank_K）
- ``CheckpointIndex``  整份索引：``.metadata`` 的读写与整体校验

一致性约定：所有张量坐标都是**全局**坐标（PP 的层号、EP 的专家号、
融合参数的拆分都在 adapters 里解决），所以这份索引与并行布局无关，
换卡数后仍能靠它重建读取计划。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from core import FormatError, PlanError
from core.metadata import numpy_dtype
from core.region import Interval, Region, Shape, flatten_slice, numel

FORMAT_VERSION = 1
"""索引格式版本；读到更高的版本直接报错，避免按旧规则误读新数据。"""

META_FILE = ".metadata"
"""索引在版本目录里的文件名；它同时也是提交点（见文档 §5）。"""


# --- 公共校验 ---------------------------------------------------------------


def _check_key(key: str) -> str:
    """相对 key 的安全校验：拒绝绝对路径、``..``、空段。"""
    if not isinstance(key, str) or not key:
        raise FormatError(f"invalid storage key {key!r}")
    if key.startswith("/"):
        raise FormatError(f"storage key must be relative: {key!r}")
    if any(part in ("", ".", "..") for part in key.split("/")):
        raise FormatError(f"storage key has an empty or unsafe segment: {key!r}")
    return key


def _ints(value: Any, field_name: str) -> Shape:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise FormatError(f"{field_name} must be a list of ints, got {value!r}")
    dims = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise FormatError(f"{field_name} must contain ints, got {item!r}")
        dims.append(int(item))
    return tuple(dims)


def _int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FormatError(f"{field_name} must be an int, got {value!r}")
    return int(value)


# --- 放置方式 ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Placement:
    """DTensor 的放置方式：``Replicate`` 或 ``Shard(dim)``。

    ``Partial`` 不落盘（checkpoint 不涉及中间输出），所以这里只有两种。
    """

    kind: str
    dim: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("replicate", "shard"):
            raise ValueError(f"unknown placement kind {self.kind!r}")
        if self.kind == "shard":
            if self.dim is None or int(self.dim) < 0:
                raise ValueError(f"Shard placement needs a dim >= 0, got {self.dim!r}")
            object.__setattr__(self, "dim", int(self.dim))
        elif self.dim is not None:
            raise ValueError("Replicate placement takes no dim")

    @classmethod
    def replicate(cls) -> "Placement":
        return cls("replicate")

    @classmethod
    def shard(cls, dim: int) -> "Placement":
        return cls("shard", dim)

    def to_dict(self) -> Any:
        return "replicate" if self.kind == "replicate" else {"shard": self.dim}

    @classmethod
    def from_dict(cls, data: Any) -> "Placement":
        if data == "replicate":
            return cls.replicate()
        if isinstance(data, dict) and set(data) == {"shard"}:
            try:
                return cls.shard(_int(data["shard"], "placement.shard"))
            except ValueError as exc:
                raise FormatError(f"invalid placement {data!r}: {exc}") from exc
        raise FormatError(f"invalid placement {data!r}")

    def __str__(self) -> str:
        return "replicate" if self.kind == "replicate" else f"shard({self.dim})"


# --- 分片 -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShardMeta:
    """一个分片：数据在哪（``file`` + 字节区间）、覆盖全局张量的哪里。

    两种覆盖方式：

    - 矩形分片：``offsets`` / ``shape``，与 ``Region`` 一一对应；
    - 展平分片：``flat = (start, length)``，即「参数展平后的一维区间」，
      用于 ZeRO / FSDP1 / Megatron distributed optimizer 的保存零通信方案。
    """

    file: str
    byte_offset: int
    length: int
    offsets: Shape = ()
    shape: Shape = ()
    flat: tuple[int, int] | None = None
    crc32c: str | None = None
    """旧版本写下的分片校验和；现在不再生成，只在显式 ``verify_crc=True`` 时使用。"""

    def __post_init__(self) -> None:
        object.__setattr__(self, "offsets", tuple(int(v) for v in self.offsets))
        object.__setattr__(self, "shape", tuple(int(v) for v in self.shape))
        object.__setattr__(self, "byte_offset", int(self.byte_offset))
        object.__setattr__(self, "length", int(self.length))
        _check_key(self.file)
        if self.byte_offset < 0:
            raise FormatError(f"byte_offset must be >= 0, got {self.byte_offset}")
        if self.length <= 0:
            raise FormatError(f"shard length must be > 0, got {self.length}")
        if self.flat is not None:
            if self.offsets or self.shape:
                raise FormatError("shard cannot be both flat and rectangular")
            start, size = (int(v) for v in self.flat)
            interval = Interval(start, size)  # 顺带校验非负
            if interval.is_empty:
                raise FormatError("flat shard must cover at least one element")
            object.__setattr__(self, "flat", (interval.start, interval.length))
        else:
            if not self.offsets or len(self.offsets) != len(self.shape):
                raise FormatError(
                    f"rectangular shard needs offsets/shape of equal rank, "
                    f"got offsets={self.offsets} shape={self.shape}"
                )
            if any(v < 0 for v in self.offsets):
                raise FormatError(f"shard offsets must be >= 0, got {self.offsets}")
            if any(v <= 0 for v in self.shape):
                raise FormatError(f"shard shape must be positive, got {self.shape}")

    @classmethod
    def rectangular(
        cls,
        *,
        file: str,
        byte_offset: int,
        length: int,
        offsets: Shape,
        shape: Shape,
        crc32c: str | None = None,
    ) -> "ShardMeta":
        return cls(
            file=file,
            byte_offset=byte_offset,
            length=length,
            offsets=offsets,
            shape=shape,
            crc32c=crc32c,
        )

    @classmethod
    def flattened(
        cls,
        *,
        file: str,
        byte_offset: int,
        length: int,
        start: int,
        elements: int,
        crc32c: str | None = None,
    ) -> "ShardMeta":
        return cls(
            file=file,
            byte_offset=byte_offset,
            length=length,
            flat=(start, elements),
            crc32c=crc32c,
        )

    @property
    def is_flat(self) -> bool:
        return self.flat is not None

    @property
    def region(self) -> "Region | None":
        """矩形分片覆盖的区域；展平分片返回 ``None``。"""
        if self.flat is not None:
            return None
        return Region(self.offsets, self.shape)

    @property
    def interval(self) -> "Interval | None":
        """展平分片覆盖的一维区间；矩形分片返回 ``None``。"""
        if self.flat is None:
            return None
        return Interval(*self.flat)

    @property
    def elements(self) -> int:
        """覆盖的元素个数（与 ``shape`` / ``flat`` 一致，与 dtype 无关）。"""
        if self.flat is not None:
            return self.flat[1]
        return numel(self.shape)

    @property
    def end(self) -> int:
        """文件内的右开字节边界。"""
        return self.byte_offset + self.length

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "file": self.file,
            "byte_offset": self.byte_offset,
            "length": self.length,
        }
        if self.flat is not None:
            data["flat"] = list(self.flat)
        else:
            data["offsets"] = list(self.offsets)
            data["shape"] = list(self.shape)
        if self.crc32c is not None:
            data["crc32c"] = self.crc32c
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "ShardMeta":
        if not isinstance(data, dict):
            raise FormatError(f"shard entry must be an object, got {data!r}")
        for required in ("file", "byte_offset", "length"):
            if required not in data:
                raise FormatError(f"shard entry missing {required!r}: {data!r}")
        crc = data.get("crc32c")
        if crc is not None and not isinstance(crc, str):
            raise FormatError(f"shard crc32c must be a hex string, got {crc!r}")

        if "flat" in data:
            flat = data["flat"]
            if isinstance(flat, dict):
                raise FormatError(f"flat shard must be [start, length], got {flat!r}")
            values = _ints(flat, "flat")
            if len(values) != 2:
                raise FormatError(f"flat shard must have 2 values, got {values!r}")
            return cls.flattened(
                file=data["file"],
                byte_offset=_int(data["byte_offset"], "byte_offset"),
                length=_int(data["length"], "length"),
                start=values[0],
                elements=values[1],
                crc32c=crc,
            )

        return cls.rectangular(
            file=data["file"],
            byte_offset=_int(data["byte_offset"], "byte_offset"),
            length=_int(data["length"], "length"),
            offsets=_ints(data.get("offsets", ()), "offsets"),
            shape=_ints(data.get("shape", ()), "shape"),
            crc32c=crc,
        )


# --- 张量 -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TensorMeta:
    """一个全局张量：全局形状 + dtype + 覆盖它的全部分片。"""

    global_shape: Shape
    dtype: str
    shards: tuple[ShardMeta, ...] = ()
    placement: Placement | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "global_shape", tuple(int(v) for v in self.global_shape))
        object.__setattr__(self, "shards", tuple(self.shards))
        if not all(isinstance(shard, ShardMeta) for shard in self.shards):
            raise FormatError("shards must be ShardMeta objects")
        if not self.global_shape:
            raise FormatError("global_shape must have at least one dim")
        if any(v <= 0 for v in self.global_shape):
            raise FormatError(f"global_shape must be positive, got {self.global_shape}")
        try:
            numpy_dtype(self.dtype)
        except ValueError as exc:
            raise FormatError(f"unsupported dtype {self.dtype!r}") from exc

    @property
    def numel(self) -> int:
        return numel(self.global_shape)

    @property
    def nbytes(self) -> int:
        return self.numel * numpy_dtype(self.dtype).itemsize

    @property
    def covered_numel(self) -> int:
        return sum(shard.elements for shard in self.shards)

    def segments(self) -> tuple[Region, ...]:
        """把分片展开成矩形块：展平分片按行主序拆回 N 维，矩形分片原样返回。"""
        pieces: list[Region] = []
        for shard in self.shards:
            interval = shard.interval
            if interval is None:
                assert shard.region is not None
                pieces.append(shard.region)
            else:
                pieces.extend(flatten_slice(self.global_shape, interval))
        return tuple(pieces)

    def validate(self, *, complete: bool = True) -> None:
        """覆盖校验：分片在界内、字节数与 dtype/shape 一致、两两不重叠、覆盖完整。

        ``complete=False`` 时只查局部一致性（单个 rank 自检用，此时它只看到一部分分片）。
        """
        itemsize = numpy_dtype(self.dtype).itemsize
        for shard in self.shards:
            expected = shard.elements * itemsize
            if shard.length != expected:
                raise PlanError(
                    f"{self._label(shard)}: length={shard.length} bytes inconsistent "
                    f"with {shard.elements} elements of {self.dtype} ({expected} bytes)"
                )
            interval = shard.interval
            if interval is not None:
                if interval.end > self.numel:
                    raise PlanError(
                        f"{self._label(shard)}: flat range {interval} exceeds numel {self.numel}"
                    )
            else:
                region = shard.region
                assert region is not None
                if region.rank != len(self.global_shape):
                    raise PlanError(
                        f"{self._label(shard)}: rank {region.rank} != global rank "
                        f"{len(self.global_shape)}"
                    )
                if not Region((0,) * region.rank, self.global_shape).contains(region):
                    raise PlanError(
                        f"{self._label(shard)}: region {region} exceeds global shape "
                        f"{self.global_shape}"
                    )

        pieces = self.segments()
        for index, left in enumerate(pieces):
            for right in pieces[index + 1 :]:
                if left.intersect(right) is not None:
                    raise PlanError(
                        f"shards overlap on global shape {self.global_shape}: {left} vs {right}"
                    )

        if complete and self.covered_numel != self.numel:
            raise PlanError(
                f"shards cover {self.covered_numel} of {self.numel} elements on "
                f"global shape {self.global_shape}"
            )

    def _label(self, shard: ShardMeta) -> str:
        return f"{shard.file}@{shard.byte_offset}"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "global_shape": list(self.global_shape),
            "dtype": self.dtype,
            "shards": [shard.to_dict() for shard in self.shards],
        }
        if self.placement is not None:
            data["placement"] = self.placement.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "TensorMeta":
        if not isinstance(data, dict):
            raise FormatError(f"tensor entry must be an object, got {data!r}")
        for required in ("global_shape", "dtype"):
            if required not in data:
                raise FormatError(f"tensor entry missing {required!r}")
        if not isinstance(data["dtype"], str):
            raise FormatError(f"tensor dtype must be a string, got {data['dtype']!r}")
        shards = data.get("shards", [])
        if not isinstance(shards, (list, tuple)):
            raise FormatError(f"tensor shards must be a list, got {shards!r}")
        placement = data.get("placement")
        return cls(
            global_shape=_ints(data["global_shape"], "global_shape"),
            dtype=data["dtype"],
            shards=tuple(ShardMeta.from_dict(item) for item in shards),
            placement=None if placement is None else Placement.from_dict(placement),
        )


# --- 小状态 -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BlobMeta:
    """小状态（step / lr scheduler / rng/rank_K ...）在存储里的一段连续字节。"""

    file: str
    byte_offset: int
    length: int
    crc32c: str | None = None
    """旧版本写下的校验和；现在不再生成。"""

    def __post_init__(self) -> None:
        object.__setattr__(self, "byte_offset", int(self.byte_offset))
        object.__setattr__(self, "length", int(self.length))
        _check_key(self.file)
        if self.byte_offset < 0:
            raise FormatError(f"byte_offset must be >= 0, got {self.byte_offset}")
        if self.length <= 0:
            raise FormatError(f"blob length must be > 0, got {self.length}")

    @property
    def end(self) -> int:
        return self.byte_offset + self.length

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "file": self.file,
            "byte_offset": self.byte_offset,
            "length": self.length,
        }
        if self.crc32c is not None:
            data["crc32c"] = self.crc32c
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "BlobMeta":
        if not isinstance(data, dict):
            raise FormatError(f"byte entry must be an object, got {data!r}")
        for required in ("file", "byte_offset", "length"):
            if required not in data:
                raise FormatError(f"byte entry missing {required!r}: {data!r}")
        crc = data.get("crc32c")
        if crc is not None and not isinstance(crc, str):
            raise FormatError(f"byte crc32c must be a hex string, got {crc!r}")
        return cls(
            file=data["file"],
            byte_offset=_int(data["byte_offset"], "byte_offset"),
            length=_int(data["length"], "length"),
            crc32c=crc,
        )


# --- 索引 -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckpointIndex:
    """一份 checkpoint 的索引，对应 ``.metadata``。

    字段与文档 §3 的 JSON 一一对应：``saved_layout`` 只用于诊断和走快路径，
    恢复逻辑不依赖它；``blobs`` 在 JSON 里叫 ``bytes``（避开 Python 内置名）。
    """

    tensors: Mapping[str, TensorMeta] = field(default_factory=dict)
    blobs: Mapping[str, BlobMeta] = field(default_factory=dict)
    step: int | None = None
    saved_layout: Mapping[str, int] = field(default_factory=dict)
    format_version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", dict(self.tensors))
        object.__setattr__(self, "blobs", dict(self.blobs))
        object.__setattr__(self, "saved_layout", dict(self.saved_layout))
        if self.format_version != FORMAT_VERSION:
            raise FormatError(
                f"unsupported index format_version {self.format_version}, "
                f"this build understands {FORMAT_VERSION}"
            )
        if self.step is not None and (
            isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0
        ):
            raise FormatError(f"step must be a non-negative int, got {self.step!r}")

    @property
    def total_bytes(self) -> int:
        return sum(shard.length for meta in self.tensors.values() for shard in meta.shards) + sum(
            blob.length for blob in self.blobs.values()
        )

    def validate(self, *, complete: bool = True) -> None:
        """整体校验：每个张量的覆盖，加上同一文件内字节区间不重叠。"""
        for name, meta in self.tensors.items():
            if not name:
                raise PlanError("tensor name must not be empty")
            meta.validate(complete=complete)

        spans: dict[str, list[tuple[int, int, str]]] = {}
        for name, meta in self.tensors.items():
            for shard in meta.shards:
                spans.setdefault(shard.file, []).append(
                    (shard.byte_offset, shard.end, f"tensor {name}")
                )
        for name, blob in self.blobs.items():
            spans.setdefault(blob.file, []).append(
                (blob.byte_offset, blob.end, f"byte {name}")
            )

        for file, entries in spans.items():
            entries.sort()
            for (start, end, what), (next_start, next_end, next_what) in zip(
                entries, entries[1:]
            ):
                if next_start < end:
                    raise PlanError(
                        f"{file}: byte range overlap between {what} "
                        f"[{start}, {end}) and {next_what} [{next_start}, {next_end})"
                    )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "format_version": self.format_version,
            "tensors": {
                name: self.tensors[name].to_dict() for name in sorted(self.tensors)
            },
        }
        if self.step is not None:
            data["step"] = self.step
        if self.saved_layout:
            data["saved_layout"] = {k: int(v) for k, v in sorted(self.saved_layout.items())}
        if self.blobs:
            data["bytes"] = {name: self.blobs[name].to_dict() for name in sorted(self.blobs)}
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "CheckpointIndex":
        if not isinstance(data, dict):
            raise FormatError(f"index must be a JSON object, got {type(data).__name__}")
        version = data.get("format_version", FORMAT_VERSION)
        if isinstance(version, bool) or not isinstance(version, int):
            raise FormatError(f"format_version must be an int, got {version!r}")

        tensors_raw = data.get("tensors", {})
        if not isinstance(tensors_raw, dict):
            raise FormatError(f"index 'tensors' must be an object, got {tensors_raw!r}")
        blobs_raw = data.get("bytes", {})
        if not isinstance(blobs_raw, dict):
            raise FormatError(f"index 'bytes' must be an object, got {blobs_raw!r}")
        layout_raw = data.get("saved_layout", {})
        if not isinstance(layout_raw, dict):
            raise FormatError(f"saved_layout must be an object, got {layout_raw!r}")

        step = data.get("step")
        if step is not None:
            step = _int(step, "step")

        layout: dict[str, int] = {}
        for key, value in layout_raw.items():
            if not isinstance(key, str):
                raise FormatError(f"saved_layout keys must be strings, got {key!r}")
            layout[key] = _int(value, f"saved_layout.{key}")

        return cls(
            tensors={
                name: TensorMeta.from_dict(entry) for name, entry in tensors_raw.items()
            },
            blobs={name: BlobMeta.from_dict(entry) for name, entry in blobs_raw.items()},
            step=step,
            saved_layout=layout,
            format_version=version,
        )

    def dumps(self) -> str:
        """稳定的 JSON 文本：键有序，便于 diff 与校验和。"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)

    @classmethod
    def loads(cls, text: str) -> "CheckpointIndex":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise FormatError(f"index is not valid JSON: {exc}") from exc
        return cls.from_dict(data)


__all__ = [
    "FORMAT_VERSION",
    "META_FILE",
    "BlobMeta",
    "CheckpointIndex",
    "Placement",
    "ShardMeta",
    "TensorMeta",
]
