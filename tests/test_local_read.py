"""本地层读闭环：COMMIT 门禁 → 读取计划 → io_uring 批量读 → staging → H2D。

覆盖三条主路径，全部以「写出来的字节」为准，不信任内存里的中间态：

- 布局不变：``save_local`` 写、``load_local`` 读，逐字节往返（含小状态 blobs）
- 换布局：TP=2 的版本按 TP=4 恢复，每个 rank 拿到自己那组列
- 校验：数据文件被改一个字节 -> ``ChecksumError``；索引/COMMIT 不一致 -> 读不到任何张量
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pytest

import api as ckpt
import native
from async_ import PinnedBufferPool, aligned_allocator
from commit.local import (
    COMMIT_FILE,
    CommitRecord,
    version_key,
    write_commit,
    write_metadata,
)
from core import ChecksumError, FormatError, PlanError
from core.index import META_FILE, CheckpointIndex, ShardMeta, TensorMeta
from core.region import Region, split
from planner.save_plan import ShardSource, build_save_plan
from planner.targets import targets_for_layout
from storage.local import LocalStorage
from storage.local_load import load_local

BLOBS = {"lr_scheduler": b"L" * 32, "rng/rank_0": bytes(range(64))}

#: 显式要求 `engine="native"` 的用例：扩展没构建时跳过，而不是 ImportError。
requires_native = pytest.mark.skipif(
    not native.available(), reason="ckpt_native 未构建（构建命令见 native/__init__.py）"
)


# --- 夹具 -------------------------------------------------------------------


@pytest.fixture
def saved(dataset: Path, tmp_path: Path):
    """一份已经保存好的本地 checkpoint：(root, gpu tensors, report)。"""
    data = ckpt.load(dataset, device="cuda:0")
    root = tmp_path / "local"
    report = ckpt.save_local(data.tensors, root, step=42, blobs=BLOBS)
    return root, data, report


def array_of(tensor) -> np.ndarray:
    """取张量的 host 副本，用于逐元素比较。"""
    return tensor.numpy()


def read_commit_json(storage: LocalStorage, step: int) -> dict:
    """直接改 COMMIT 的测试需要原始 JSON（CommitRecord 是不可变的）。"""
    return json.loads(storage.read(f"{version_key(step)}/{COMMIT_FILE}").decode())


def column_dims(tensors) -> dict[str, int]:
    """2 维张量沿 dim=1 切；1 维张量（norm/bias）保持 Replicate。"""
    return {name: 1 for name, tensor in tensors.items() if len(tensor.shape) == 2}


def write_tp_version(root: Path, tensors, *, step: int, tp: int, dims) -> CheckpointIndex:
    """按 TP=tp 切分写一个「多 rank」版本目录（每个 rank 一个数据文件）。

    ``save_local`` 只写单 rank，换布局恢复需要多个分片写在同一个版本里，所以这里
    直接用 planner + commit 组装：每个分片的 crc32 与 ``save_local`` 保持同一种算法。
    """
    storage = LocalStorage(root)
    shards: dict[str, list[ShardMeta]] = {}
    files: dict[str, int] = {}

    for rank in range(tp):
        sources = []
        for fqn, tensor in tensors.items():
            shape = tuple(int(dim) for dim in tensor.shape)
            dim = dims.get(fqn)
            if dim is None:
                # Replicate 张量在 tp 个副本里只由一个写者落盘（与 save_local 同一套
                # default_writer 去重），否则索引里的分片会互相重叠
                region = Region((0,) * len(shape), shape)
                replica_index, n_replicas = rank, tp
            else:
                region = split(shape, dim, rank, tp)
                replica_index, n_replicas = 0, 1
            sources.append(
                ShardSource(
                    fqn=fqn,
                    global_shape=shape,
                    dtype=tensor.dtype,
                    region=region,
                    replica_index=replica_index,
                    n_replicas=n_replicas,
                )
            )
        plan = build_save_plan(sources, rank=rank)
        if plan.size == 0:
            continue

        payload = bytearray(plan.size)
        for item in plan.items:
            array = array_of(tensors[item.fqn])
            chunk = array[item.region.slices].tobytes()
            assert len(chunk) == item.length, (item.fqn, len(chunk), item.length)
            payload[item.byte_offset : item.end] = chunk
            shards.setdefault(item.fqn, []).append(
                item.to_shard_meta(crc32c=f"{zlib.crc32(chunk) & 0xFFFFFFFF:08x}")
            )

        storage.write(f"{version_key(step)}/{plan.file}", bytes(payload))
        files[plan.file] = len(payload)

    index = CheckpointIndex(
        tensors={
            fqn: TensorMeta(
                global_shape=tuple(int(d) for d in tensors[fqn].shape),
                dtype=tensors[fqn].dtype,
                shards=tuple(entries),
            )
            for fqn, entries in shards.items()
        },
        step=step,
    )
    index.validate()

    metadata_bytes = write_metadata(storage, step, index)
    write_commit(
        storage,
        CommitRecord(step=step, metadata_bytes=metadata_bytes, files=files),
    )
    return index


# --- 布局不变：整读往返 -----------------------------------------------------


def test_roundtrip_bytes_and_blobs(saved) -> None:
    root, data, _ = saved

    report = load_local(root, step=42, device="cuda:0")

    assert report.step == 42
    assert report.engine in ("native", "python")
    assert set(report.tensors) == set(data.names())
    for name in data.names():
        got = array_of(report.tensors[name])
        expected = array_of(data[name])
        assert got.shape == expected.shape, name
        assert got.tobytes() == expected.tobytes(), name

    assert report.blobs == BLOBS
    assert report.unverified == (), "布局不变时每个读块都是完整分片，必须全部校验过"
    assert report.amplification == pytest.approx(1.0)


def test_load_goes_through_h2d_for_every_tensor(saved) -> None:
    root, data, _ = saved
    report = load_local(root, step=42, device="cuda:0")

    expected = sum(data[name].nbytes for name in data.names())
    assert report.h2d_bytes == expected
    assert report.copied_bytes == expected
    for tensor in report.tensors.values():
        assert tensor.device.type == "cuda"


def test_python_engine_reads_the_same_bytes(saved) -> None:
    """C++ 扩展没参与时走 preadv 回退，结果必须逐字节一致。"""
    root, data, _ = saved
    report = load_local(root, step=42, engine="python", device="cpu")

    assert report.engine == "python"
    for name in data.names():
        assert array_of(report.tensors[name]).tobytes() == array_of(data[name]).tobytes(), name


@requires_native
def test_native_and_python_engines_agree(saved) -> None:
    root, data, _ = saved
    # 局部名不叫 native：会把同名的 native 包遮住，读代码时容易误解
    via_native = load_local(root, step=42, engine="native", device="cpu")
    via_python = load_local(root, step=42, engine="python", device="cpu")

    for name in data.names():
        assert (
            via_native.tensors[name].numpy().tobytes()
            == via_python.tensors[name].numpy().tobytes()
        )


def test_newest_committed_step_is_used_when_step_is_omitted(
    dataset: Path, tmp_path: Path
) -> None:
    data = ckpt.load(dataset, device="cuda:0")
    root = tmp_path / "local"
    ckpt.save_local(data.tensors, root, step=10, blobs=BLOBS)
    ckpt.save_local(data.tensors, root, step=20, blobs={"lr_scheduler": b"M" * 16})

    report = load_local(root, device="cuda:0")

    assert report.step == 20
    assert report.blobs == {"lr_scheduler": b"M" * 16}


# --- 换布局：TP 2 -> 4 ------------------------------------------------------


def test_tp_change_recovers_the_right_columns(dataset: Path, tmp_path: Path) -> None:
    data = ckpt.load(dataset, device="cuda:0")
    root = tmp_path / "local"
    dims = column_dims(data.tensors)
    assert dims, "tiny 数据集里应当有 2 维张量"

    write_tp_version(root, data.tensors, step=7, tp=2, dims=dims)

    for tp_rank in range(4):
        report = load_local(
            root, step=7, tp=4, tp_rank=tp_rank, shard_dims=dims, device="cuda:0"
        )
        assert report.step == 7
        for target in targets_for_layout(report.index, tp=4, tp_rank=tp_rank, shard_dims=dims):
            got = array_of(report.tensors[target.fqn])
            expected = array_of(data[target.fqn])[target.region.slices]
            assert got.shape == expected.shape, (target.fqn, tp_rank)
            assert got.tobytes() == expected.tobytes(), (target.fqn, tp_rank)


def test_whole_strategy_reads_whole_shards_and_still_verifies_crc(
    dataset: Path, tmp_path: Path
) -> None:
    """TP 2->4 的整块读法读放大 2 倍，但读块正好覆盖旧分片，所以 CRC 能校验。"""
    data = ckpt.load(dataset, device="cuda:0")
    root = tmp_path / "local"
    dims = column_dims(data.tensors)
    write_tp_version(root, data.tensors, step=8, tp=2, dims=dims)

    report = load_local(
        root, step=8, tp=4, tp_rank=1, shard_dims=dims, column_strategy="whole", device="cpu"
    )

    assert report.amplification > 1.0
    assert report.unverified == ()


def test_strided_strategy_reports_unverified_blocks(dataset: Path, tmp_path: Path) -> None:
    """逐行读的块只是分片的一部分，没有可用的整分片 crc，必须如实记进 unverified。

    校验默认关闭，所以这里显式打开：测的是「要求校验时做不到」的诚实标注。
    """
    data = ckpt.load(dataset, device="cuda:0")
    root = tmp_path / "local"
    dims = column_dims(data.tensors)
    write_tp_version(root, data.tensors, step=9, tp=2, dims=dims)

    report = load_local(
        root,
        step=9,
        tp=4,
        tp_rank=3,
        shard_dims=dims,
        column_strategy="strided",
        device="cpu",
        verify_crc=True,
    )

    assert report.crc_verified is True
    assert report.unverified, "strided 读到的是分片切片，不能静默当成校验过"
    assert report.amplification == pytest.approx(1.0)
    for target in targets_for_layout(report.index, tp=4, tp_rank=3, shard_dims=dims):
        expected = array_of(data[target.fqn])[target.region.slices]
        assert report.tensors[target.fqn].numpy().tobytes() == expected.tobytes(), target.fqn


def test_unchanged_tp_degrades_to_one_read_per_shard(dataset: Path, tmp_path: Path) -> None:
    data = ckpt.load(dataset, device="cuda:0")
    root = tmp_path / "local"
    dims = column_dims(data.tensors)
    index = write_tp_version(root, data.tensors, step=11, tp=4, dims=dims)

    report = load_local(
        root, step=11, tp=4, tp_rank=2, shard_dims=dims, device="cpu", budget_bytes=1 << 20
    )

    # 布局不变：每个张量只读自己那一份分片，一次整读、无读放大
    assert report.requests == len(index.tensors)
    assert report.read_bytes == report.copied_bytes
    assert report.unverified == (), "整分片读必须全部通过 crc32 校验"


# --- 窗口与 staging 复用 ----------------------------------------------------


@requires_native
def test_small_budget_splits_windows_and_reuses_one_block(saved) -> None:
    root, data, _ = saved
    pool = PinnedBufferPool(
        chunk_bytes=8 << 10,
        min_chunks=1,
        max_bytes=None,
        timeout=0,
        allocator=aligned_allocator(),
    )

    report = load_local(
        root, step=42, budget_bytes=8 << 10, engine="native", pool=pool, device="cpu"
    )

    assert report.windows > 1
    assert pool.stats.reuses > 0, "窗口之间必须复用同一批 pinned 块"
    assert pool.stats.in_use_bytes == 0, "读完必须把块全部归还"
    for name in data.names():
        assert report.tensors[name].numpy().tobytes() == array_of(data[name]).tobytes(), name


# --- 校验与门禁 -------------------------------------------------------------


def test_corrupted_data_byte_raises_checksum_error(dataset: Path, tmp_path: Path) -> None:
    """索引里带分片 crc32c（旧格式）且显式开校验时，坏字节会被抓住。

    新写的 checkpoint 不再生成分片 crc32c（字节级把关移到上传边界），所以这条
    能力只对旧索引有效；它是本地层保留的唯一「自证」入口。
    """
    data = ckpt.load(dataset, device="cuda:0")
    root = tmp_path / "local"
    dims = column_dims(data.tensors)
    index = write_tp_version(root, data.tensors, step=13, tp=2, dims=dims)

    shard = next(
        entry
        for meta in index.tensors.values()
        for entry in meta.shards
        if entry.crc32c is not None and entry.file == "__0_0.distcp"
    )
    storage = LocalStorage(root)
    key = f"{version_key(13)}/{shard.file}"
    payload = bytearray(storage.read(key))
    payload[shard.byte_offset] ^= 0xFF  # 大小不变，只有内容坏了
    storage.write(key, bytes(payload))

    with pytest.raises(ChecksumError):
        load_local(
            root, step=13, tp=2, tp_rank=0, shard_dims=dims, verify_crc=True, device="cpu"
        )


def test_corrupted_data_byte_is_not_detected_by_default(saved) -> None:
    """默认不校验：坏字节会被静默接受——这是显式选择，代价由上传侧 checksum 兜。

    记在这里是为了让「默认关掉校验」这个决定有一个可执行的证据，而不是口头承诺。
    """
    root, _, report = saved
    storage = LocalStorage(root)
    key = f"{version_key(42)}/{report.file}"
    payload = bytearray(storage.read(key))
    payload[0] ^= 0xFF
    storage.write(key, bytes(payload))

    report = load_local(root, step=42, device="cpu")
    assert report.crc_verified is False
    assert report.verified_blocks == 0
    assert report.unverified == ()


def test_corrupted_metadata_raises_before_any_tensor_is_read(saved) -> None:
    """索引被改坏：不再靠 crc32，靠 JSON 解析 + ``CheckpointIndex.validate()``。

    两者都能在读任何张量之前失败，代价为零，所以索引侧的校验和是多余的。
    """
    root, _, _ = saved
    storage = LocalStorage(root)
    key = f"{version_key(42)}/{META_FILE}"
    payload = bytearray(storage.read(key))
    payload[10] ^= 0x01
    storage.write(key, bytes(payload))

    with pytest.raises((FormatError, ChecksumError)):
        load_local(root, step=42, device="cpu")


def test_missing_commit_is_not_recoverable(saved) -> None:
    root, _, _ = saved
    (root / version_key(42) / COMMIT_FILE).unlink()

    with pytest.raises(FormatError):
        load_local(root, step=42, device="cpu")


def test_unlisted_data_file_is_rejected(saved) -> None:
    """COMMIT 的清单必须覆盖索引引用到的文件。"""
    root, _, _ = saved
    storage = LocalStorage(root)
    commit = read_commit_json(storage, 42)
    commit["files"] = {}
    storage.write(f"{version_key(42)}/{COMMIT_FILE}", json.dumps(commit).encode())

    with pytest.raises(FormatError):
        load_local(root, step=42, device="cpu")


def test_missing_data_file_is_rejected(saved) -> None:
    root, _, report = saved
    (root / version_key(42) / report.file).unlink()

    with pytest.raises(FormatError):
        load_local(root, step=42, device="cpu")


def test_truncated_data_file_is_rejected(saved) -> None:
    """长度不符由 COMMIT 的文件清单抓住，与校验和无关（所以是 FormatError）。"""
    root, _, report = saved
    path = root / version_key(42) / report.file
    path.write_bytes(path.read_bytes()[: -4096])

    with pytest.raises(FormatError):
        load_local(root, step=42, device="cpu")


def test_verify_crc_can_be_turned_off(saved) -> None:
    root, _, report = saved
    storage = LocalStorage(root)
    key = f"{version_key(42)}/{report.file}"
    payload = bytearray(storage.read(key))
    payload[0] ^= 0xFF
    storage.write(key, bytes(payload))

    # 关掉校验就读得出来 —— 差别正是「有没有校验」，不是「能不能读」
    unchecked = load_local(root, step=42, verify_crc=False, device="cpu")
    assert unchecked.step == 42


# --- api.load 自动探测 ------------------------------------------------------


def test_api_load_autodetects_local_versions(saved) -> None:
    root, data, _ = saved

    loaded = ckpt.load(root, device="cuda:0")
    assert loaded.meta["format"] == "local"
    assert loaded.meta["step"] == 42
    for name in data.names():
        assert loaded[name].numpy().tobytes() == array_of(data[name]).tobytes(), name

    # 直接给 step 目录也可以
    direct = ckpt.load(root / version_key(42), device="cpu")
    assert direct.meta["step"] == 42


def test_api_load_rejects_unknown_layout(tmp_path: Path) -> None:
    empty = tmp_path / "nothing"
    empty.mkdir()

    with pytest.raises(FormatError):
        ckpt.load(empty)


# --- 参数校验 ---------------------------------------------------------------


def test_invalid_tp_rank_is_rejected(saved) -> None:
    root, _, _ = saved

    with pytest.raises(ValueError):
        load_local(root, step=42, tp=2, tp_rank=2, device="cpu")


def test_unknown_shard_dim_tensor_is_rejected(saved) -> None:
    root, _, _ = saved

    with pytest.raises(PlanError):
        load_local(root, step=42, shard_dims={"not.a.tensor": 1}, device="cpu")
