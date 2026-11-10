"""计划往返：保存计划 → 假文件 → 读取计划 → 逐元素比对。

前面的用例只校验计划的几何数字，这里真的把字节写进内存文件再读回来，
用 numpy 校验「新布局下拿到的数据 == 原张量的对应区域」。覆盖三条主路径：

- TP 换卡数（整分片读 / 整行块读放大 2 倍）
- ZeRO 展平切分（方案 A 的 flat 分片）
- 布局不变时的退化解
"""

from __future__ import annotations

import numpy as np
from core.index import CheckpointIndex, TensorMeta
from core.region import Interval, Region, split
from planner import ShardSource, TargetTensor, build_load_plan, build_save_plan

SHAPE = (32, 16)
FQN = "model.layers.0.attn.q_proj.weight"


def make_array() -> np.ndarray:
    """可辨认的浮点数据，错了会立刻出现在 diff 上。"""
    return np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE) * 0.5


def write_plans(plans, array: np.ndarray) -> dict[str, bytearray]:
    """把每个 plan 的 item 按计划写进内存文件（本地紧凑布局）。"""
    files: dict[str, bytearray] = {}
    flat = array.reshape(-1)
    for plan in plans:
        for item in plan.items:
            if item.flat is not None:
                payload = flat[item.flat.start : item.flat.end].tobytes()
            else:
                assert item.region is not None
                payload = array[item.region.slices].tobytes()
            assert len(payload) == item.length, (item.fqn, len(payload), item.length)
            buffer = files.setdefault(item.file, bytearray())
            need = item.byte_offset + item.length
            if len(buffer) < need:
                buffer.extend(b"\0" * (need - len(buffer)))
            buffer[item.byte_offset : item.byte_offset + item.length] = payload
    return files


def merge_index(plans) -> CheckpointIndex:
    tensors: dict[str, TensorMeta] = {}
    for plan in plans:
        for fqn, meta in plan.partial_index().items():
            if fqn in tensors:
                previous = tensors[fqn]
                meta = TensorMeta(
                    global_shape=meta.global_shape,
                    dtype=meta.dtype,
                    shards=previous.shards + meta.shards,
                )
            tensors[fqn] = meta
    index = CheckpointIndex(tensors=tensors)
    index.validate()
    return index


def read_target(files, plan, target: TargetTensor) -> np.ndarray:
    """按读取计划把字节拷进目标区域的局部数组。"""
    local = np.zeros(target.region.shape, dtype=np.float32)
    for item in plan.items:
        raw = bytes(files[item.file][item.byte_offset : item.byte_offset + item.length])
        block = np.frombuffer(raw, dtype=np.float32).reshape(item.read_shape)
        for span in item.spans:
            dst = tuple(
                slice(offset, offset + size) for offset, size in zip(span.dst_offsets, span.shape)
            )
            local[dst] = block[span.src_region().slices]
    return local


def tp_shards(array: np.ndarray, count: int, dim: int = 1):
    sources = [
        ShardSource(
            fqn=FQN,
            global_shape=SHAPE,
            dtype="f32",
            region=split(SHAPE, dim, index, count),
        )
        for index in range(count)
    ]
    plans = [build_save_plan([source], rank=index) for index, source in enumerate(sources)]
    return plans, write_plans(plans, array), merge_index(plans)


def test_tp8_to_tp4_roundtrip() -> None:
    """TP 8 → 4：新 rank 拿到的列必须与原张量逐元素相同。"""
    array = make_array()
    plans, files, index = tp_shards(array, 8)
    assert len(plans) == 8  # 每个 rank 一个分片文件

    for column_rank in range(4):
        target = TargetTensor.for_shard(FQN, SHAPE, "f32", dim=1, index=column_rank, count=4)
        plan = build_load_plan(index, [target])
        assert plan.amplification == 1.0
        got = read_target(files, plan, target)
        np.testing.assert_array_equal(got, array[:, column_rank * 4 : (column_rank + 1) * 4])


def test_tp8_to_tp16_roundtrip_with_read_amplification() -> None:
    """TP 8 → 16：整行块读 2 倍放大，拷入的列仍然正确。"""
    array = make_array()
    _, files, index = tp_shards(array, 8)

    for column_rank in range(16):
        target = TargetTensor.for_shard(FQN, SHAPE, "f32", dim=1, index=column_rank, count=16)
        plan = build_load_plan(index, [target])
        assert plan.amplification == 2.0
        got = read_target(files, plan, target)
        np.testing.assert_array_equal(got, array[:, column_rank : column_rank + 1])

        strided = build_load_plan(index, [target], column_strategy="strided")
        assert strided.amplification == 1.0
        np.testing.assert_array_equal(read_target(files, strided, target), array[:, column_rank : column_rank + 1])


def test_tp8_to_fsdp_dim0_roundtrip() -> None:
    """换成沿 dim 0 切分（FSDP2），读取计划同样正确。"""
    array = make_array()
    _, files, index = tp_shards(array, 4, dim=0)

    for row_rank in range(8):
        target = TargetTensor.for_shard(FQN, SHAPE, "f32", dim=0, index=row_rank, count=8)
        plan = build_load_plan(index, [target])
        assert plan.amplification == 1.0
        got = read_target(files, plan, target)
        np.testing.assert_array_equal(got, array[row_rank * 4 : (row_rank + 1) * 4, :])


def test_zero_flat_shards_roundtrip_to_whole_tensor() -> None:
    """ZeRO 展平切分（方案 A）→ 新布局整读：逐元素一致。"""
    array = make_array()
    total = int(np.prod(SHAPE))
    per_shard = total // 4

    sources = [
        ShardSource(
            fqn=FQN,
            global_shape=SHAPE,
            dtype="f32",
            flat=Interval(index * per_shard, per_shard),
        )
        for index in range(4)
    ]
    plans = [build_save_plan([source], rank=index) for index, source in enumerate(sources)]
    files = write_plans(plans, array)
    index = merge_index(plans)

    target = TargetTensor.whole(FQN, SHAPE, "f32")
    plan = build_load_plan(index, [target])
    assert plan.amplification == 1.0
    np.testing.assert_array_equal(read_target(files, plan, target), array)


def test_zero_flat_shards_roundtrip_to_column_subset() -> None:
    """ZeRO 展平切分 → TP 列子集：走 strided，读放大 1，数据仍然正确。"""
    array = make_array()
    total = int(np.prod(SHAPE))
    per_shard = total // 4
    plans = [
        build_save_plan(
            [
                ShardSource(
                    fqn=FQN,
                    global_shape=SHAPE,
                    dtype="f32",
                    flat=Interval(index * per_shard, per_shard),
                )
            ],
            rank=index,
        )
        for index in range(4)
    ]
    files = write_plans(plans, array)
    index = merge_index(plans)

    target = TargetTensor.for_shard(FQN, SHAPE, "f32", dim=1, index=3, count=4)
    plan = build_load_plan(index, [target])
    assert plan.amplification == 1.0
    np.testing.assert_array_equal(read_target(files, plan, target), array[:, 12:16])


def test_unchanged_layout_roundtrip_reads_own_file() -> None:
    """布局不变：每个 rank 只读自己那个文件，一次 read_range。"""
    array = make_array()
    _, files, index = tp_shards(array, 2)

    for column_rank in range(2):
        target = TargetTensor.for_shard(FQN, SHAPE, "f32", dim=1, index=column_rank, count=2)
        plan = build_load_plan(index, [target])
        assert len(plan.items) == 1
        assert plan.files() == [plan.items[0].file]
        np.testing.assert_array_equal(
            read_target(files, plan, target), array[:, column_rank * 8 : (column_rank + 1) * 8]
        )


def test_replicated_source_written_once_and_readable() -> None:
    """同一份 Replicate 数据只写一次；任意 rank 都能按索引读到全量。"""
    array = make_array()
    # 4 个副本各自的 replica_index 不同，只有 hash 选中的那个会写
    plans = [
        build_save_plan(
            [
                ShardSource(
                    fqn=FQN,
                    global_shape=SHAPE,
                    dtype="f32",
                    region=Region((0, 0), SHAPE),
                    replica_index=index,
                    n_replicas=4,
                )
            ],
            rank=index,
        )
        for index in range(4)
    ]
    assert sum(len(plan.items) for plan in plans) == 1

    files = write_plans(plans, array)
    index = merge_index(plans)
    target = TargetTensor.whole(FQN, SHAPE, "f32")
    np.testing.assert_array_equal(read_target(files, build_load_plan(index, [target]), target), array)
