"""L4 规模与性能仿真：把数据放大到百 MB 级，量成本、抓定量不变量。

**只 gate 定量不变量，不 gate 墙钟**：CI 机器的抖动会让时间断言变成 flaky 测试，
所以吞吐只打印（``pytest -s`` 可见），真正断言的是「读放大 = 1.0」「请求数 =
张量数」「D2H 字节 = 数据量」这类与机器无关的性质。

默认不跑（``-m workload`` 才跑）：默认套件要保持秒级。
"""

from __future__ import annotations

import resource
import time
from pathlib import Path

import numpy as np
import pytest

import api as ckpt
import sim
from storage.local_load import load_local

pytestmark = pytest.mark.workload

DEVICE = "cuda:0"
CHUNKS = 8
CHUNK_SHAPE = (2048, 1024)  # 8 MiB / f32
TOTAL_BYTES = CHUNKS * int(np.prod(CHUNK_SHAPE)) * 4


def big_state(*, step: int = 0) -> sim.TrainingState:
    """``CHUNKS`` 个 8 MiB 张量，总共 64 MiB；内容按块号区分，便于逐块比对。"""
    values = {
        f"model.blocks.{index}.mlp.fc1.weight": np.full(
            CHUNK_SHAPE, index + 1, dtype=np.float32
        )
        for index in range(CHUNKS)
    }
    return sim.TrainingState(
        step=step, dtypes={name: "f32" for name in values}, values=values
    )


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def throughput_mb_s(total_bytes: int, seconds: float) -> float:
    return total_bytes / (1 << 20) / seconds if seconds > 0 else float("inf")


@pytest.fixture
def saved_big(tmp_path) -> tuple[Path, sim.TrainingState]:
    state = big_state()
    assert state.total_bytes() == TOTAL_BYTES
    root = tmp_path / "ckpt"

    started = time.perf_counter()
    report = ckpt.save_local(
        sim.to_device(state, device=DEVICE), root, step=1, layout={"tp": 1}
    )
    elapsed = time.perf_counter() - started

    print(
        f"\n[workload] save: {TOTAL_BYTES >> 20} MiB / {elapsed:.3f}s "
        f"= {throughput_mb_s(TOTAL_BYTES, elapsed):.1f} MiB/s, "
        f"file={report.file} size={report.size >> 20} MiB, "
        f"peak_rss={peak_rss_mb():.0f} MiB"
    )
    assert report.d2h_bytes == TOTAL_BYTES
    return root, state


def test_workload_save_load_invariants(saved_big) -> None:
    """整读的定量不变量：读放大 1.0、请求数 = 张量数、H2D 字节 = 数据量。"""
    root, state = saved_big

    started = time.perf_counter()
    report = load_local(root, step=1, device=DEVICE)
    elapsed = time.perf_counter() - started

    print(
        f"[workload] load: {TOTAL_BYTES >> 20} MiB / {elapsed:.3f}s "
        f"= {throughput_mb_s(TOTAL_BYTES, elapsed):.1f} MiB/s, "
        f"requests={report.requests} windows={report.windows}, "
        f"peak_rss={peak_rss_mb():.0f} MiB"
    )

    assert report.requests == len(state.names())
    assert report.windows >= 1
    assert report.amplification == pytest.approx(1.0)
    assert report.read_bytes == TOTAL_BYTES
    assert report.copied_bytes == TOTAL_BYTES
    assert report.h2d_bytes == TOTAL_BYTES

    # 抽查首尾两块：整读路径必须给出原始字节
    for name in ("model.blocks.0.mlp.fc1.weight", f"model.blocks.{CHUNKS - 1}.mlp.fc1.weight"):
        assert report.tensors[name].numpy().tobytes() == state.bytes_of(name)


def test_workload_tp_change_amplifies_reads_exactly(saved_big) -> None:
    """TP 1→8 时整读策略的读放大恰好是 8 倍：可以断言，不依赖机器。"""
    root, state = saved_big

    started = time.perf_counter()
    report = load_local(root, step=1, device=DEVICE, tp=8, tp_rank=3, shard_dims={
        name: 1 for name in state.names()
    })
    elapsed = time.perf_counter() - started

    print(
        f"[workload] load tp=8: read={report.read_bytes >> 20} MiB "
        f"copied={report.copied_bytes >> 20} MiB "
        f"amplification={report.amplification:.2f} "
        f"({elapsed:.3f}s), peak_rss={peak_rss_mb():.0f} MiB"
    )

    assert report.amplification == pytest.approx(8.0)
    assert report.copied_bytes == TOTAL_BYTES // 8
    assert report.read_bytes == TOTAL_BYTES

    # 本 rank 只拿到第 3 列：与全局张量的对应列逐字节一致
    name = "model.blocks.2.mlp.fc1.weight"
    expected = state.shard(name, dim=1, index=3, count=8)
    assert report.tensors[name].numpy().tobytes() == expected.tobytes()
