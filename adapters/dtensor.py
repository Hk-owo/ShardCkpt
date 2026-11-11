"""FSDP2 / DTensor 适配：把 DTensor 的本地分片翻译成全局坐标（``ShardSource``）。

**不依赖 torch**：所有信息都从鸭子类型属性读，真 ``torch.distributed.tensor.DTensor``
和测试里的轻量 stub 走同一条代码路径 —— 这也是它能在没有 GPU 的机器上被单测覆盖的原因。

翻译规则与 ``core.region.split``（FSDP2 的切分规则）严格一致：

- ``Shard(dim)``：沿全局形状的第 ``dim`` 维切成 mesh 该维的份数，取本 rank 那一份；
  不整除时前 ``shape[dim] % n`` 份各多一个元素（与 ``torch.chunk`` 相同）
- ``Replicate()``：该维上所有副本内容相同，只影响 ``n_replicas`` / ``replica_index``；
  planner 按 ``hash(FQN) % n_replicas`` 挑一个副本落盘，不做内容比对
- ``Partial`` / ``_StridedShard``：存不了 —— 前者是部分和（先 all-reduce），后者是
  交错切分（TP 融合参数），落成矩形区间都会读出一份错的数据，所以直接报错

用法：

.. code-block:: python

    from adapters.dtensor import describe_state

    shards = describe_state(model.state_dict())        # DTensor -> 分片描述
    ckpt.save_local(
        {s.fqn: to_device(s.local) for s in shards},   # 本地分片（数据）
        root,
        step=step,
        sources=[s.source for s in shards],            # 全局坐标（metadata）
    )

普通张量（没有 ``placements``）也能一起吃进来：当成 1 副本的整体分片，等价于
``api._sources_from_tensors`` 的默认行为。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core import CheckpointError
from core.metadata import DTYPE_TO_NUMPY
from core.region import Region, Shape, split
from planner.save_plan import ShardSource


class DTensorAdapterError(CheckpointError):
    """分片描述翻译不出来：placement 不支持、拿不到本 rank 坐标、本地形状对不上。"""


@dataclass(frozen=True, slots=True)
class DTensorShard:
    """一个 DTensor 的分片：``source`` 是全局坐标，``local`` 是本地那份数据。

    ``local`` 原样返回（torch tensor 或 numpy 数组），设备搬运交给调用方：
    ``Tensor`` 只认 device buffer，适配器不该替它决定拷到哪张卡。
    """

    fqn: str
    source: ShardSource
    local: Any

    @property
    def region(self) -> Region:
        region = self.source.region
        if region is None:  # pragma: no cover - 适配器只构造 region 形态
            raise DTensorAdapterError(f"{self.fqn}: 适配器不会有 flat 形态的分片")
        return region


# --- dtype ------------------------------------------------------------------

_DTYPE_ALIASES: dict[str, str] = {
    "float64": "f64",
    "float32": "f32",
    "float16": "f16",
    "bfloat16": "bf16",
    "int64": "i64",
    "int32": "i32",
    "int16": "i16",
    "int8": "i8",
    "uint8": "u8",
    "bool": "bool",
}
"""torch / numpy 的 dtype 名字 → 索引里用的短名（与 manifest、``core.metadata`` 一致）。"""


def _dtype_name(dtype: Any) -> str:
    if dtype is None:
        raise DTensorAdapterError("张量没有 dtype，无法写进索引")
    if isinstance(dtype, str):
        raw = dtype
    else:
        # numpy 的 dtype 实例有 .name；torch dtype / numpy 标量类型靠 str 兜底
        name = getattr(dtype, "name", None)
        raw = name if isinstance(name, str) else str(dtype)
    # "torch.float32" / "<class 'numpy.float32'>" / "float32" 都归一成最后一段
    raw = raw.rsplit(".", 1)[-1].rstrip("'>")
    name = _DTYPE_ALIASES.get(raw, raw)
    if name not in DTYPE_TO_NUMPY:
        raise DTensorAdapterError(
            f"不支持的 dtype {raw!r}；索引只认 {sorted(DTYPE_TO_NUMPY)}"
        )
    return name


# --- placement 分类 ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Placement:
    kind: str  # "shard" / "replicate"
    dim: int | None = None


def _classify(placement: Any, fqn: str) -> _Placement:
    """按类名 + ``dim`` 属性识别 placement，不 import torch。"""
    name = type(placement).__name__
    if name == "Replicate":
        return _Placement("replicate")
    if name == "Shard":
        dim = getattr(placement, "dim", None)
        if not isinstance(dim, int):
            raise DTensorAdapterError(f"{fqn}: Shard 没有整数 dim 属性（got {dim!r}）")
        return _Placement("shard", dim)
    if name == "_StridedShard":
        raise DTensorAdapterError(
            f"{fqn}: _StridedShard（交错切分，TP 融合参数用）不支持；"
            "它不落成矩形区间，按 Shard 解释会读出错位的数据"
        )
    if name == "Partial":
        raise DTensorAdapterError(
            f"{fqn}: Partial（部分和）不能直接存；先 all-reduce / reduce_scatter 拿到完整值"
        )
    raise DTensorAdapterError(
        f"{fqn}: 不支持的 placement {name!r}；只支持 Shard(dim) 与 Replicate()"
    )


# --- mesh 与坐标 ------------------------------------------------------------


def _mesh_shape(mesh: Any, fqn: str) -> tuple[int, ...]:
    if mesh is None:
        raise DTensorAdapterError(f"{fqn}: 有 placements 但没有 device_mesh，无法确定 mesh 形状")
    shape = getattr(mesh, "shape", None)
    if shape is None:
        raise DTensorAdapterError(f"{fqn}: device_mesh 没有 shape 属性")
    return tuple(int(dim) for dim in shape)


def _coordinate(mesh: Any, coord: Any, fqn: str, ndim: int) -> tuple[int, ...]:
    """本 rank 在 mesh 各维上的下标。优先读 mesh，其次用调用方显式给的 ``coord``。"""
    if coord is None:
        getter = getattr(mesh, "get_coordinate", None)
        if callable(getter):
            value = getter()
            if value is None:
                raise DTensorAdapterError(
                    f"{fqn}: 本 rank 不在这个 device_mesh 里（get_coordinate() 返回 None）"
                )
            coord = value
        else:
            local_rank = getattr(mesh, "get_local_rank", None)
            if not callable(local_rank):
                raise DTensorAdapterError(
                    f"{fqn}: device_mesh 既不提供 get_coordinate() 也没有 get_local_rank()；"
                    "请显式传 coord=(...)"
                )
            if ndim != 1:
                raise DTensorAdapterError(
                    f"{fqn}: 只有 1 维 mesh 能用 get_local_rank() 反推坐标（这个 mesh 是 "
                    f"{ndim} 维）；请显式传 coord=(...)"
                )
            coord = (local_rank(),)

    values = tuple(int(value) for value in coord)
    if len(values) != ndim:
        raise DTensorAdapterError(
            f"{fqn}: coord {values} 的维数与 mesh shape 的 {ndim} 维不一致"
        )
    return values


# --- 翻译 -------------------------------------------------------------------


def describe(
    fqn: str,
    tensor: Any,
    *,
    coord: tuple[int, ...] | None = None,
    mesh: Any = None,
) -> DTensorShard:
    """把一个 DTensor（或普通张量）翻译成 ``DTensorShard``。

    ``coord``  本 rank 在 mesh 各维的下标；默认从 ``tensor.device_mesh`` 自己读，
               只有在 stub / 没有分布式环境时才需要显式给
    ``mesh``   覆盖 ``tensor.device_mesh``（测试或跨 mesh 搬运时用）
    """
    if not fqn:
        raise ValueError("fqn must not be empty")

    shape = tuple(int(dim) for dim in _shape_of(tensor, fqn))
    if not shape:
        raise DTensorAdapterError(f"{fqn}: 0 维张量没有分片坐标，走 blobs 路径")
    dtype = _dtype_name(getattr(tensor, "dtype", None))
    local = _local_of(tensor)
    placements = _placements_of(tensor)

    if not placements:
        # 普通张量 / 1 副本：整体一块，等价于 api._sources_from_tensors 的默认行为
        region = Region(tuple(0 for _ in shape), shape)
        n_replicas, replica_index = 1, 0
    else:
        mesh_obj = mesh if mesh is not None else getattr(tensor, "device_mesh", None)
        mesh_shape = _mesh_shape(mesh_obj, fqn)
        if len(placements) != len(mesh_shape):
            raise DTensorAdapterError(
                f"{fqn}: placements 有 {len(placements)} 项，但 mesh 是 "
                f"{len(mesh_shape)} 维（{mesh_shape}）—— 两者必须一一对应"
            )
        coordinate = _coordinate(mesh_obj, coord, fqn, len(mesh_shape))
        region, n_replicas, replica_index = _locate(
            fqn, shape, placements, mesh_shape, coordinate
        )

    _check_local_shape(fqn, local, region.shape)
    source = ShardSource(
        fqn=fqn,
        global_shape=shape,
        dtype=dtype,
        region=region,
        replica_index=replica_index,
        n_replicas=n_replicas,
    )
    return DTensorShard(fqn=fqn, source=source, local=local)


def describe_state(
    state: Mapping[str, Any],
    *,
    coord: tuple[int, ...] | None = None,
    mesh: Any = None,
) -> list[DTensorShard]:
    """一次翻译整个 ``state_dict``（保持输入顺序）。

    ``coord`` / ``mesh`` 会应用到所有张量：同一个 rank 在同一 mesh 上的坐标相同，
    所以通常不用传；真的跨多个 mesh（PP 的不同 stage）时，逐个调 :func:`describe`。
    """
    return [
        describe(fqn, tensor, coord=coord, mesh=mesh) for fqn, tensor in state.items()
    ]


def _locate(
    fqn: str,
    shape: Shape,
    placements: tuple[Any, ...],
    mesh_shape: tuple[int, ...],
    coordinate: tuple[int, ...],
) -> tuple[Region, int, int]:
    """算出矩形区域与副本身份：``(region, n_replicas, replica_index)``。"""
    offsets = [0] * len(shape)
    sizes = list(shape)
    sharded: set[int] = set()
    n_replicas = 1
    replica_index = 0

    for axis, placement in enumerate(placements):
        count = mesh_shape[axis]
        index = coordinate[axis]
        if not 0 <= index < count:
            raise DTensorAdapterError(
                f"{fqn}: coord[{axis}] = {index} 越界（mesh 该维大小 {count}）"
            )

        kind = _classify(placement, fqn)
        if kind.kind == "replicate":
            # 行主序把 Replicate 各维的下标压成一个副本编号
            n_replicas *= count
            replica_index = replica_index * count + index
            continue

        assert kind.dim is not None
        dim = kind.dim % len(shape)
        if dim in sharded:
            raise DTensorAdapterError(
                f"{fqn}: 第 {dim} 维被 Shard 了两次；DTensor 不会产生这种 placements"
            )
        sharded.add(dim)
        part = split(shape, dim, index, count)
        offsets[dim] = part.offsets[dim]
        sizes[dim] = part.shape[dim]

    return Region(tuple(offsets), tuple(sizes)), n_replicas, replica_index


# --- 鸭子类型取值 -----------------------------------------------------------


def _shape_of(tensor: Any, fqn: str) -> Any:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise DTensorAdapterError(f"{fqn}: 张量没有 shape 属性")
    return shape


def _local_of(tensor: Any) -> Any:
    """DTensor 的本地分片：``to_local()``；普通张量就是它自己。"""
    to_local = getattr(tensor, "to_local", None)
    return to_local() if callable(to_local) else tensor


def _placements_of(tensor: Any) -> tuple[Any, ...]:
    placements = getattr(tensor, "placements", None)
    return () if placements is None else tuple(placements)


def _check_local_shape(fqn: str, local: Any, expected: Shape) -> None:
    local_shape = getattr(local, "shape", None)
    if local_shape is None:
        raise DTensorAdapterError(f"{fqn}: 本地分片没有 shape（got {type(local).__name__}）")
    actual = tuple(int(dim) for dim in local_shape)
    if actual != tuple(expected):
        raise DTensorAdapterError(
            f"{fqn}: 本地分片形状 {actual} 与 placements 推出的 {tuple(expected)} 不一致；"
            "to_local() 应返回矩形分块（Partial / 交错切分会破坏这个前提）"
        )


__all__ = [
    "DTensorAdapterError",
    "DTensorShard",
    "describe",
    "describe_state",
]
