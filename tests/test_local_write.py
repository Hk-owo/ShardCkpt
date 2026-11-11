"""本地层写闭环的测试：plan → D2H → 数据 + .metadata + COMMIT → 可恢复。"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

import api as ckpt
from async_ import PinnedBufferPool, SnapshotManager, SnapshotState, aligned_allocator
from commit.local import (
    COMMIT_FILE,
    committed_steps,
    delete_version,
    latest_step,
    read_commit,
    version_key,
)
from core import PlanError
from core.index import META_FILE, CheckpointIndex
from core.region import Region
from planner.save_plan import ShardSource
from storage.local import LocalStorage

BLOBS = {"lr_scheduler": b"L" * 32, "rng/rank_0": bytes(range(64))}


def read_shard(storage: LocalStorage, report, shard) -> bytes:
    """索引里的 ``file`` 是版本目录内的文件名，读取时要拼上 step 目录。"""
    return storage.read_range(
        f"{report.key}/{shard.file}", shard.byte_offset, shard.length
    )


@pytest.fixture
def saved(dataset: Path, tmp_path: Path):
    """一份已经保存好的本地 checkpoint：(root, gpu tensors, report)。"""
    data = ckpt.load(dataset, device="cuda:0")
    root = tmp_path / "local"
    report = ckpt.save_local(data.tensors, root, step=42, blobs=BLOBS)
    return root, data, report


def test_save_local_produces_recoverable_version(saved) -> None:
    root, _, report = saved

    assert report.state is SnapshotState.COMMITTED
    assert report.key == version_key(42)
    assert report.file == "__0_0.distcp"
    assert report.size % 4096 == 0
    assert (root / "step_42" / COMMIT_FILE).is_file()
    assert (root / "step_42" / META_FILE).is_file()

    record = read_commit(LocalStorage(root), 42)
    assert record is not None
    assert record.step == 42
    assert record.files == {report.file: report.file_size}
    assert latest_step(LocalStorage(root)) == 42
    assert committed_steps(LocalStorage(root)) == [42]


def test_index_roundtrips_from_metadata_file(saved) -> None:
    root, _, _ = saved
    storage = LocalStorage(root)

    text = storage.read(f"{version_key(42)}/{META_FILE}").decode("utf-8")
    index = CheckpointIndex.loads(text)

    assert index.step == 42
    assert set(index.tensors) == set(saved[1].names())
    assert index.blobs.keys() == BLOBS.keys()
    index.validate()  # 覆盖完整、同文件内区间不重叠


def test_data_file_matches_tensor_bytes(saved) -> None:
    """按索引从数据文件里取每个分片，必须与 GPU 侧张量的字节完全一致。"""
    root, data, report = saved
    storage = LocalStorage(root)

    for fqn, meta in report.index.tensors.items():
        expected = data[fqn].tobytes()
        for shard in meta.shards:
            payload = read_shard(storage, report, shard)
            assert len(payload) == len(expected)
            assert payload == expected, fqn


def test_index_carries_no_checksums(saved) -> None:
    """索引不再存分片校验和，COMMIT 也不再记 ``.metadata`` 的 crc32。

    字节级完整性由上传边界负责（``storage/object.py`` 的对象级校验和），本地层
    只留「分片坐标 + 字节数」。字段仍在格式里，所以旧 checkpoint 照样能读。
    """
    root, _, report = saved
    storage = LocalStorage(root)

    for meta in report.index.tensors.values():
        for shard in meta.shards:
            assert shard.crc32c is None
    for blob in report.index.blobs.values():
        assert blob.crc32c is None

    assert report.commit.metadata_crc32 is None

    # 长度照旧要对得上：COMMIT 里的字节数仍是索引完整性的唯一凭据
    metadata = storage.read(f"{version_key(42)}/{META_FILE}")
    assert report.commit.metadata_bytes == len(metadata)


def test_blobs_land_at_planned_offsets(saved) -> None:
    root, _, report = saved
    storage = LocalStorage(root)

    for name, expected in BLOBS.items():
        blob = report.index.blobs[name]
        payload = read_shard(storage, report, blob)
        assert payload == expected, name


def test_padding_bytes_are_zero(saved) -> None:
    """对齐补齐区必须清零：否则写盘内容是缓冲区里的残留数据。"""
    root, _, report = saved
    storage = LocalStorage(root)
    payload = storage.read(f"{version_key(42)}/{report.file}")

    covered: list[tuple[int, int]] = []
    for meta in report.index.tensors.values():
        covered += [(shard.byte_offset, shard.end) for shard in meta.shards]
    for blob in report.index.blobs.values():
        covered += [(blob.byte_offset, blob.end)]
    covered.sort()

    cursor = 0
    for start, end in covered:
        assert payload[cursor:start] == b"\x00" * (start - cursor)
        cursor = end
    assert payload[cursor:] == b"\x00" * (len(payload) - cursor)


def test_newest_committed_step_wins(dataset: Path, tmp_path: Path) -> None:
    data = ckpt.load(dataset, device="cuda:0")
    root = tmp_path / "local"
    storage = LocalStorage(root)

    ckpt.save_local(data.tensors, root, step=10, blobs=BLOBS)
    ckpt.save_local(data.tensors, root, step=20, blobs=BLOBS)

    assert committed_steps(storage) == [10, 20]
    assert latest_step(storage) == 20
    assert not (root / "latest").exists(), "没有 latest 指针：最新版本靠扫描得出"


def test_python_engine_closes_the_loop(dataset: Path, tmp_path: Path) -> None:
    """C++ 扩展没构建时走纯 Python 引擎，闭环语义不变。"""
    data = ckpt.load(dataset, device="cuda:0")
    root = tmp_path / "local"
    report = ckpt.save_local(data.tensors, root, step=3, engine="python")

    assert report.engine == "python"
    storage = LocalStorage(root, engine="python")
    assert read_commit(storage, 3) is not None
    shard = next(iter(report.index.tensors.values())).shards[0]
    assert read_shard(storage, report, shard)


def test_write_failure_leaves_no_commit(dataset: Path, tmp_path: Path, monkeypatch) -> None:
    """写数据失败：状态机进 ABORTED，恢复端看不到这一版（没有 COMMIT）。"""
    data = ckpt.load(dataset, device="cuda:0")
    root = tmp_path / "local"

    pool = PinnedBufferPool(
        chunk_bytes=1 << 20, min_chunks=1, max_bytes=1 << 20, timeout=0,
        allocator=aligned_allocator(),
    )
    manager = SnapshotManager(pool)

    def boom(self, key, buffer):  # noqa: ANN001
        raise OSError("disk full")

    monkeypatch.setattr(LocalStorage, "write_view", boom)

    with pytest.raises(OSError):
        ckpt.save_local(
            data.tensors, root, step=5, blobs=BLOBS, pool=pool, manager=manager
        )

    assert manager.stats.aborted == 1
    storage = LocalStorage(root)
    assert read_commit(storage, 5) is None
    assert committed_steps(storage) == []


def test_missing_tensor_is_rejected_before_any_write(dataset: Path, tmp_path: Path) -> None:
    """计划里有两个张量、实际只给一个：写完之前就要报错，且不留目录。"""
    data = ckpt.load(dataset, device="cuda:0")
    names = data.names()[:2]
    root = tmp_path / "local"
    sources = [
        ShardSource(
            fqn=name,
            global_shape=tuple(int(d) for d in data[name].shape),
            dtype=data[name].dtype,
            region=Region(tuple(0 for _ in data[name].shape), tuple(int(d) for d in data[name].shape)),
        )
        for name in names
    ]

    with pytest.raises(PlanError):
        ckpt.save_local(
            {names[0]: data[names[0]]}, root, step=6, sources=sources
        )

    assert not (root / version_key(6) / COMMIT_FILE).exists()


def test_delete_version_makes_it_unrecoverable(dataset: Path, tmp_path: Path) -> None:
    data = ckpt.load(dataset, device="cuda:0")
    root = tmp_path / "local"
    storage = LocalStorage(root)
    ckpt.save_local(data.tensors, root, step=1, blobs=BLOBS)
    ckpt.save_local(data.tensors, root, step=2, blobs=BLOBS)

    delete_version(storage, 2)

    assert read_commit(storage, 2) is None
    assert committed_steps(storage) == [1]
    assert latest_step(storage) == 1, "删掉较新版本后，最新的已提交版本退到 1"
    assert (root / version_key(1) / COMMIT_FILE).is_file()


def test_second_save_reuses_the_pinned_block(dataset: Path, tmp_path: Path) -> None:
    data = ckpt.load(dataset, device="cuda:0")
    root = tmp_path / "local"
    pool = PinnedBufferPool(
        chunk_bytes=1 << 20, min_chunks=1, max_bytes=1 << 20, timeout=0,
        allocator=aligned_allocator(),
    )
    manager = SnapshotManager(pool)

    ckpt.save_local(data.tensors, root, step=1, pool=pool, manager=manager)
    ckpt.save_local(data.tensors, root, step=2, pool=pool, manager=manager)

    assert manager.stats.committed == 2
    assert pool.stats.allocations == 1, "第二版应该复用同一块 pinned buffer"
    assert pool.stats.reuses == 2


def test_commit_record_is_stable_json(saved) -> None:
    _, _, report = saved
    text = report.commit.dumps()
    payload = json.loads(text)

    assert payload["step"] == 42
    assert payload["metadata"]["file"] == META_FILE
    assert payload["format_version"] == 1
    assert list(payload["files"]) == [report.file]


# --- 两条引擎的临时文件命名必须同构 -------------------------------------------


def test_temp_path_matches_native_layout(tmp_path: Path) -> None:
    """``<final>.tmp.<pid>.<序号>``：与 C++ ``MakeTempPath`` 逐字一致。

    同目录是 rename 原子的前提；pid + 自增序号是两个写者不共用临时文件的保证。
    """
    import storage.local as local_backend  # noqa: PLC0415

    target = tmp_path / "step_1" / "__0_0.distcp"
    first = local_backend._temp_path(target)
    second = local_backend._temp_path(target)

    assert first != second, "同一进程内两次写入不能共用同一个临时文件"
    assert first.parent == target.parent, "临时文件必须与目标文件同目录"
    stem, pid, counter = first.name.rsplit(".", 2)
    assert stem == "__0_0.distcp.tmp"
    assert pid == str(os.getpid())
    assert counter.isdigit()


def test_concurrent_writers_get_distinct_temp_files(tmp_path: Path, monkeypatch) -> None:
    """两个写者同时写同一个 key：各写各的临时文件，最终内容是完整的一份。

    固定临时文件名（``<final>.tmp``）会让两个写者交错写进同一个文件，rename
    之后可能留下混合内容；这条测试锁住的是「原子性 ≠ 互斥性」那一格。
    """
    import storage.local as local_backend  # noqa: PLC0415

    seen: list[Path] = []
    real = local_backend._temp_path

    def spy(path: Path) -> Path:
        temp = real(path)
        seen.append(temp)
        return temp

    monkeypatch.setattr(local_backend, "_temp_path", spy)

    storage = LocalStorage(tmp_path, engine="python")
    payloads = [bytes([0xAA]) * 4096, bytes([0xBB]) * 4096]
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def writer(payload: bytes) -> None:
        barrier.wait()
        try:
            storage.write("step_1/data.bin", payload)
        except BaseException as exc:  # pragma: no cover - 失败即测试失败
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(payload,)) for payload in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"并发写失败: {errors}"
    assert len(seen) == 2 and seen[0] != seen[1], "两个写者必须使用不同的临时文件"

    final = (tmp_path / "step_1" / "data.bin").read_bytes()
    assert final in payloads, "最终内容必须是某个写者的完整内容，不能是交错的混合"
    assert not [p.name for p in (tmp_path / "step_1").iterdir() if ".tmp." in p.name]
