"""`adapters/dtensor.py` 的测试：placement → 全局坐标的翻译，以及接到保存/恢复管道。

不装 torch：用与 `torch.distributed.tensor` 同名同形的 stub（`Shard` / `Replicate` /
`_StridedShard` / `Partial`、带 `shape` 与 `get_coordinate()` 的 mesh），走的是和真
DTensor 完全相同的鸭子类型代码路径。
"""

from __future__ import annotations

import numpy as np
import pytest

import api as ckpt
import s3_stub
from adapters.dtensor import DTensorAdapterError, describe, describe_state
from async_ import Tensor, get_backend
from core.metadata import Device
from core.region import Region, split
from storage.object import S3Config, S3Storage

# --- 轻量 stub：对齐 torch 的接口面 -----------------------------------------


class Replicate:
    """`torch.distributed.tensor.Replicate()` 的替身。"""


class Shard:
    """`torch.distributed.tensor.Shard(dim)` 的替身。"""

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)


class Partial:
    """`Partial()`：部分和，存不了。"""


class _StridedShard(Shard):
    """`_StridedShard`：交错切分（TP 融合参数），存不了。"""


class FakeMesh:
    def __init__(self, shape: tuple[int, ...], coord: tuple[int, ...] | None) -> None:
        self.shape = tuple(shape)
        self._coord = coord

    def get_coordinate(self) -> tuple[int, ...] | None:
        return self._coord


class RankOnlyMesh:
    """只有 `get_local_rank()` 的 mesh（1 维场景的常见写法）。"""

    def __init__(self, shape: int | tuple[int, ...], rank: int) -> None:
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self._rank = rank

    def get_local_rank(self) -> int:
        return self._rank


class FakeDTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: object,
        local: object,
        placements: tuple[object, ...] = (),
        mesh: object = None,
    ) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self._local = local
        self.placements = placements
        self.device_mesh = mesh

    def to_local(self) -> object:
        return self._local


def make_array(shape: tuple[int, ...], dtype: object = np.float32) -> np.ndarray:
    return np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)


# --- Shard ------------------------------------------------------------------


def test_shard_on_1d_mesh_maps_to_split_region() -> None:
    array = make_array((10, 5))
    region = split(array.shape, 0, 1, 4)

    shard = describe(
        "w",
        FakeDTensor(
            array.shape, array.dtype, array[region.slices], (Shard(0),), FakeMesh((4,), (1,))
        ),
    )

    assert shard.region == region == Region((3, 0), (3, 5))
    assert shard.source.n_replicas == 1
    assert shard.source.replica_index == 0
    assert shard.source.dtype == "f32"
    assert shard.source.global_shape == (10, 5)


def test_uneven_split_follows_the_fsdp_rule() -> None:
    """10 切 4 份 → 前两份各多一个（与 torch.chunk / FSDP2 一致）。"""
    array = make_array((10, 3))
    seen = []
    for index in range(4):
        region = split(array.shape, 0, index, 4)
        shard = describe(
            "w",
            FakeDTensor(
                array.shape,
                array.dtype,
                array[region.slices],
                (Shard(0),),
                FakeMesh((4,), (index,)),
            ),
        )
        assert shard.region == region
        seen.append(shard.region.offsets[0])

    assert seen == [0, 3, 6, 8]
    assert [region.shape[0] for region in (split(array.shape, 0, i, 4) for i in range(4))] == [
        3,
        3,
        2,
        2,
    ]


def test_negative_shard_dim_is_normalized() -> None:
    array = make_array((4, 6))
    region = split(array.shape, 1, 1, 2)

    shard = describe(
        "w",
        FakeDTensor(
            array.shape, array.dtype, array[region.slices], (Shard(-1),), FakeMesh((2,), (1,))
        ),
    )

    assert shard.region == region == Region((0, 3), (4, 3))


# --- Replicate --------------------------------------------------------------


def test_replicate_becomes_replica_index() -> None:
    array = make_array((10, 5))

    shard = describe(
        "w",
        FakeDTensor(array.shape, array.dtype, array, (Replicate(),), FakeMesh((4,), (2,))),
    )

    assert shard.region == Region((0, 0), (10, 5)), "Replicate 拿到的是完整张量"
    assert shard.source.n_replicas == 4
    assert shard.source.replica_index == 2


def test_two_dimensional_mesh_mixes_replicate_and_shard() -> None:
    """mesh (dp=2, tp=2)：qkv 沿列切（TP），副本身份由 dp 维决定。"""
    array = make_array((8, 6))
    region = split(array.shape, 1, 1, 2)

    shard = describe(
        "qkv",
        FakeDTensor(
            array.shape,
            array.dtype,
            array[region.slices],
            (Replicate(), Shard(1)),
            FakeMesh((2, 2), (1, 1)),
        ),
    )

    assert shard.region == region == Region((0, 3), (8, 3))
    assert shard.source.n_replicas == 2, "只有 Replicate 的 mesh 维算副本"
    assert shard.source.replica_index == 1, "副本编号按 Replicate 维的下标"


def test_multiple_replicate_dims_multiply_replicas() -> None:
    array = make_array((4, 4))

    shard = describe(
        "w",
        FakeDTensor(
            array.shape, array.dtype, array, (Replicate(), Replicate()), FakeMesh((2, 3), (1, 2))
        ),
    )

    assert shard.source.n_replicas == 6
    assert shard.source.replica_index == 1 * 3 + 2, "行主序压平"


def test_fully_sharded_two_dimensional_block() -> None:
    array = make_array((5, 7))

    shard = describe(
        "w",
        FakeDTensor(
            array.shape,
            array.dtype,
            array[:2, 5:7],
            (Shard(0), Shard(1)),
            FakeMesh((2, 3), (1, 2)),
        ),
    )

    # dim0 切 2 → 3/2，取第 1 份；dim1 切 3 → 3/2/2，取第 2 份
    assert shard.region == Region((3, 5), (2, 2))
    assert shard.source.n_replicas == 1


# --- 普通张量与 dtype -------------------------------------------------------


def test_plain_tensor_is_a_single_replica() -> None:
    array = make_array((3, 4))

    shard = describe("w", FakeDTensor(array.shape, array.dtype, array))

    assert shard.region == Region((0, 0), (3, 4))
    assert shard.source.n_replicas == 1
    assert shard.source.replica_index == 0


@pytest.mark.parametrize(
    ("dtype", "want"),
    [
        (np.dtype(np.float32), "f32"),
        (np.float16, "f16"),
        ("torch.bfloat16", "bf16"),
        ("float64", "f64"),
        ("torch.bool", "bool"),
        ("f32", "f32"),
    ],
)
def test_dtype_names_are_normalized(dtype: object, want: str) -> None:
    array = make_array((2, 2))

    shard = describe("w", FakeDTensor(array.shape, dtype, array))

    assert shard.source.dtype == want


def test_unknown_dtype_is_reported() -> None:
    array = make_array((2, 2))

    with pytest.raises(DTensorAdapterError, match="dtype"):
        describe("w", FakeDTensor(array.shape, "torch.float8_e4m3fn", array))


# --- 拒绝的输入 -------------------------------------------------------------


def test_partial_is_rejected() -> None:
    array = make_array((4, 4))

    with pytest.raises(DTensorAdapterError, match="Partial"):
        describe(
            "w",
            FakeDTensor(array.shape, array.dtype, array, (Partial(),), FakeMesh((2,), (0,))),
        )


def test_strided_shard_is_rejected() -> None:
    array = make_array((4, 4))

    with pytest.raises(DTensorAdapterError, match="_StridedShard"):
        describe(
            "w",
            FakeDTensor(
                array.shape, array.dtype, array, (_StridedShard(0),), FakeMesh((2,), (0,))
            ),
        )


def test_unknown_placement_is_rejected() -> None:
    array = make_array((4, 4))

    class Interleaved: ...

    with pytest.raises(DTensorAdapterError, match="不支持的 placement"):
        describe(
            "w",
            FakeDTensor(array.shape, array.dtype, array, (Interleaved(),), FakeMesh((2,), (0,))),
        )


def test_placements_must_match_mesh_rank() -> None:
    array = make_array((4, 4))

    with pytest.raises(DTensorAdapterError, match="一一对应"):
        describe(
            "w",
            FakeDTensor(array.shape, array.dtype, array, (Shard(0),), FakeMesh((2, 2), (0, 0))),
        )


def test_missing_coordinate_is_reported() -> None:
    array = make_array((4, 4))

    class NoCoordinate:
        shape = (2,)

    with pytest.raises(DTensorAdapterError, match="coord"):
        describe("w", FakeDTensor(array.shape, array.dtype, array, (Shard(0),), NoCoordinate()))

    with pytest.raises(DTensorAdapterError, match="不在这个 device_mesh"):
        describe(
            "w",
            FakeDTensor(array.shape, array.dtype, array, (Shard(0),), FakeMesh((2,), None)),
        )


def test_local_rank_fallback_only_works_for_1d_mesh() -> None:
    array = make_array((4, 4))
    region = split(array.shape, 0, 1, 2)

    shard = describe(
        "w",
        FakeDTensor(
            array.shape,
            array.dtype,
            array[region.slices],
            (Shard(0),),
            RankOnlyMesh(2, 1),
        ),
    )
    assert shard.region == region

    with pytest.raises(DTensorAdapterError, match="1 维 mesh"):
        describe(
            "w",
            FakeDTensor(
                array.shape,
                array.dtype,
                array,
                (Replicate(), Shard(0)),
                RankOnlyMesh((2, 2), 1),
            ),
            coord=None,
        )


def test_coord_must_match_mesh_rank() -> None:
    array = make_array((4, 4))

    with pytest.raises(DTensorAdapterError, match="维数"):
        describe(
            "w",
            FakeDTensor(
                array.shape,
                array.dtype,
                array,
                (Replicate(), Shard(0)),
                FakeMesh((2, 2), (0, 0)),
            ),
            coord=(0,),
        )


def test_out_of_range_coordinate_is_reported() -> None:
    array = make_array((4, 4))

    with pytest.raises(DTensorAdapterError, match="越界"):
        describe(
            "w",
            FakeDTensor(array.shape, array.dtype, array, (Shard(0),), FakeMesh((2,), (0,))),
            coord=(5,),
        )


def test_local_shape_mismatch_is_reported() -> None:
    array = make_array((4, 4))
    region = split(array.shape, 0, 0, 2)

    with pytest.raises(DTensorAdapterError, match="本地分片形状"):
        describe(
            "w",
            FakeDTensor(
                array.shape, array.dtype, array[region.slices], (Shard(1),), FakeMesh((2,), (0,))
            ),
        )


def test_describe_state_keeps_input_order() -> None:
    a = make_array((2, 3))
    b = make_array((4,))

    shards = describe_state(
        {
            "z.weight": FakeDTensor(b.shape, b.dtype, b, (Replicate(),), FakeMesh((2,), (0,))),
            "a.weight": FakeDTensor(a.shape, a.dtype, a),
        }
    )

    assert [shard.fqn for shard in shards] == ["z.weight", "a.weight"]
    assert all(shard.source.n_replicas >= 1 for shard in shards)


# --- 端到端：适配器 + 保存 + 恢复 -------------------------------------------


def to_device(array: np.ndarray, *, name: str, dtype: str, device: str = "cpu") -> Tensor:
    """host 数组 → device 张量（与 examples/e2e_sim.py 的 `to_device` 同款三行）。"""
    spec = Device.parse(device)
    backend = get_backend(spec)
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


def test_multi_rank_regions_tile_the_global_tensor() -> None:
    """2×2 全切分：4 个 rank 的矩形块两两不重叠，并起来正好铺满全局张量。"""
    array = make_array((8, 6))
    regions = []

    for dp_index in range(2):
        for tp_index in range(2):
            rows = split(array.shape, 0, dp_index, 2)
            cols = split(array.shape, 1, tp_index, 2)
            region = Region(
                (rows.offsets[0], cols.offsets[1]), (rows.shape[0], cols.shape[1])
            )
            shard = describe(
                "w",
                FakeDTensor(
                    array.shape,
                    array.dtype,
                    np.ascontiguousarray(array[region.slices]),
                    (Shard(0), Shard(1)),
                    FakeMesh((2, 2), (dp_index, tp_index)),
                ),
            )
            assert shard.region == region
            regions.append(shard.region)

    assert sum(region.numel for region in regions) == array.size
    for index, left in enumerate(regions):
        for right in regions[index + 1 :]:
            assert not left.overlaps(right), f"{left} 与 {right} 重叠"


def test_dtensor_shards_round_trip_through_object_storage(tmp_path) -> None:
    """适配器 → 每 rank 本地暂存 → 独立上传 → rank 0 提交 → 从对象存储恢复。

    这是 e2e_sim 第 5 阶段的缩小版：每个 rank 只存自己那片，恢复端把各 rank 的
    分片拼回全局张量，逐字节一致。本地目录不能直接读（单 rank 索引覆盖不全，
    ``recover_version`` 会拒），所以必须走对象存储这条正式恢复路径。
    """
    array = np.random.default_rng(7).standard_normal((8, 6)).astype(np.float32)
    tp = 2

    with s3_stub.start() as server:
        storage = S3Storage(
            S3Config(
                bucket="ckpt",
                prefix="runs",
                endpoint_url=server.endpoint_url,
                access_key_id="test-key",
                secret_access_key="test-secret",
            )
        )
        try:
            # 代际由 rank 0 分配一次，所有 rank 用同一个：各自开代会落到不同的
            # gen_* 前缀下，这一版永远凑不齐。
            generation = ckpt.begin_upload(storage, step=1, ranks=list(range(tp)))
            for tp_index in range(tp):
                region = split(array.shape, 1, tp_index, tp)
                local = np.ascontiguousarray(array[region.slices])
                shard = describe(
                    "model.qkv.weight",
                    FakeDTensor(
                        array.shape,
                        array.dtype,
                        local,
                        (Shard(1),),
                        FakeMesh((tp,), (tp_index,)),
                    ),
                )
                assert shard.region == region

                # 每个 rank 有自己的本地暂存目录，索引只描述本 rank 那一片
                root = tmp_path / f"rank_{tp_index}"
                ckpt.save_local(
                    {shard.fqn: to_device(local, name=shard.fqn, dtype=shard.source.dtype)},
                    root,
                    step=1,
                    rank=tp_index,
                    sources=[shard.source],
                )
                ckpt.upload_local(
                    root, storage, step=1, rank=tp_index, generation=generation
                )

            ckpt.finalize_upload(
                storage,
                step=1,
                ranks=list(range(tp)),
                generation=generation,
                layout={"tp": tp},
            )
            report = ckpt.load_object(storage, step=1, device="cpu")
        finally:
            storage.close()

    got = report.tensors["model.qkv.weight"].numpy()
    assert got.shape == array.shape
    assert got.dtype == array.dtype
    assert got.tobytes() == array.tobytes()
