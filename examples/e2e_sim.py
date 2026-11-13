"""端到端模拟：伪训练 → 分片保存 → 崩溃 → 上传 → 从对象存储恢复续训 → 回收。

**不需要 GPU，也不需要外部对象存储**：显存由 `MockCudaBackend` 用 host 内存承载，
S3 是进程内的 `tests/s3_stub.py`（要连真的 SeaweedFS / MinIO 就传 `--endpoint-url`）。
读取路径、提交点语义、Range GET 与 H2D 都是正式实现，只有 device 和对象存储是模拟的。

剧本（每个阶段都断言，不通过就非零退出）：

    1  伪训练 + 周期保存：step 3 / 6 / 9，每个 rank 存自己那一片（沿 dim=1 列切分）
    2  本地门禁：只有带 COMMIT 的版本可恢复；逐字节比对回读内容
    3  崩溃注入：step 12 的最后一个 rank 在写 COMMIT 之前被 SIGKILL
    4  提交点语义：finalize(step=12) 失败、load_object(step=12) 报错，最新仍是 9
    5  从对象存储恢复：每个 rank 取自己那片、拼回全局，再整读一份交叉验证
    6  续训：从 step 9 恢复后跑 3 步 == 一口气跑到 step 12（逐字节）
    7  下一轮 step 15 提交成功 → 有界保留只留最近 3 代（step 3 被删）+ 回收本地暂存

用法::

    .venv/bin/python examples/e2e_sim.py            # 2 个 rank，内置 S3（默认）
    .venv/bin/python examples/e2e_sim.py --ranks 1  # 单 rank
    .venv/bin/python examples/e2e_sim.py --ranks 4  # 4 个 rank
    .venv/bin/python examples/e2e_sim.py --endpoint-url http://127.0.0.1:8333 \\
        --access-key ... --secret-key ... --bucket ckpt --prefix runs/e2e
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import os
import shutil
import signal
import sys
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# 让 demo 能 import 仓库内的模块（子进程会重新执行本文件的顶层，所以放模块级）
_REPO = Path(__file__).resolve().parents[1]
for _candidate in (str(_REPO), str(_REPO / "tests")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import api as ckpt  # noqa: E402
import s3_stub  # noqa: E402
from async_ import Tensor, get_backend  # noqa: E402
from commit.local import committed_steps  # noqa: E402
from core import FormatError, PlanError  # noqa: E402
from core.metadata import Device, numpy_dtype  # noqa: E402
from core.region import split  # noqa: E402
from planner.save_plan import ShardSource  # noqa: E402
from storage.object import S3Config, S3Storage  # noqa: E402

DEVICE = "cuda:0"

#: 伪模型的参数表：2 维张量沿 dim=1 列切（TP），1 维张量是 Replicate。
SHAPES: dict[str, tuple[int, ...]] = {
    "model.embed_tokens.weight": (64, 90),  # 90 不能被 4 整除，顺带验证 split 的余数规则
    "model.blocks.0.attn.qkv.weight": (96, 96),
    "model.norm.weight": (96,),
    "model.blocks.0.attn.qkv.bias": (96,),
}
DTYPES: dict[str, str] = {
    "model.embed_tokens.weight": "f32",
    "model.blocks.0.attn.qkv.weight": "f32",
    "model.norm.weight": "f32",
    "model.blocks.0.attn.qkv.bias": "f16",
}
COLUMN_DIM: dict[str, int] = {name: 1 for name, shape in SHAPES.items() if len(shape) == 2}

STEPS = (3, 6, 9, 15)
SAVED = STEPS[:3]
"""崩溃之前正常提交的三个版本：3 / 6 / 9。"""
LAST_SAVED = SAVED[-1]
CRASH_STEP = 12


# --- 伪训练 -----------------------------------------------------------------


def initial_state(seed: int = 20240611) -> dict[str, np.ndarray]:
    """一份确定性的初始权重（同样的 seed 每次都是同样的字节）。"""
    rng = np.random.default_rng(seed)
    return {
        name: rng.random(shape, dtype=np.float64).astype(numpy_dtype(DTYPES[name]))
        for name, shape in SHAPES.items()
    }


def evolve(state: Mapping[str, np.ndarray], step: int) -> dict[str, np.ndarray]:
    """伪训练一步：``w <- w * 0.999 + noise(step, name)``。

    噪声只由 ``(step, 张量名)`` 决定，所以「从头训到第 N 步」与「从第 M 步恢复后
    再训到第 N 步」必然得到同一份字节 —— 续训因此可以逐字节断言。
    """
    out: dict[str, np.ndarray] = {}
    for name, array in state.items():
        noise = np.random.default_rng([int(step), zlib.crc32(name.encode())]).random(
            array.shape
        )
        out[name] = (array.astype(np.float64) * 0.999 + noise * 0.01).astype(array.dtype)
    return out


def state_at(step: int) -> dict[str, np.ndarray]:
    """一口气训练到第 ``step`` 步的状态（续训比的终点）。"""
    state = initial_state()
    for current in range(1, step + 1):
        state = evolve(state, current)
    return state


def assert_same_state(
    got: Mapping[str, np.ndarray], want: Mapping[str, np.ndarray], *, what: str
) -> None:
    """逐字节比对，失败时给出张量名与首个差异偏移。"""
    assert set(got) == set(want), f"{what}: 张量名不一致 {sorted(set(got) ^ set(want))}"
    for name in want:
        want_bytes = np.ascontiguousarray(want[name]).tobytes()
        got_bytes = np.ascontiguousarray(got[name]).tobytes()
        if got_bytes == want_bytes:
            continue
        offset = next(i for i, (a, b) in enumerate(zip(got_bytes, want_bytes)) if a != b)
        raise AssertionError(
            f"{what}: {name} 第 {offset} 字节起不一致（{len(got_bytes)} 字节）"
        )


# --- host 数组 → device 张量（外部调用方要自己做的三行） ---------------------


def to_device(
    array: np.ndarray,
    *,
    name: str,
    dtype: str,
    device: str = DEVICE,
) -> Tensor:
    """一块 host 数组 → device 张量：``allocate`` + ``copy_h2d``。

    真机上这三行就是 ``buffer.to("cuda")``；这里显式写出来是因为 ``Tensor`` 只认
    device buffer，不认 numpy 数组。
    """
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


def shard_arrays(
    state: Mapping[str, np.ndarray], *, rank: int, tp: int
) -> dict[str, np.ndarray]:
    """本 rank 在 TP=``tp`` 下持有的那一片（1 维张量是整份副本）。"""
    out: dict[str, np.ndarray] = {}
    for name, array in state.items():
        dim = COLUMN_DIM.get(name)
        piece = array if dim is None else array[split(array.shape, dim, rank, tp).slices]
        out[name] = np.ascontiguousarray(piece)
    return out


def sources_for_rank(state: Mapping[str, np.ndarray], *, rank: int, tp: int) -> list[ShardSource]:
    """本 rank 的分片描述：``region`` 说清自己在全局张量里覆盖哪一块。"""
    sources: list[ShardSource] = []
    for name, array in state.items():
        dim = COLUMN_DIM.get(name)
        if dim is None:
            sources.append(
                ShardSource(
                    fqn=name,
                    global_shape=array.shape,
                    dtype=DTYPES[name],
                    region=split(array.shape, 0, 0, 1),
                    replica_index=rank,
                    n_replicas=tp,
                )
            )
        else:
            sources.append(
                ShardSource(
                    fqn=name,
                    global_shape=array.shape,
                    dtype=DTYPES[name],
                    region=split(array.shape, dim, rank, tp),
                )
            )
    return sources


def blobs_for_rank(rank: int) -> dict[str, bytes]:
    """小状态（RNG / lr scheduler 位置）作为已序列化字节一起存。"""
    return {
        f"rng/rank_{rank}": json.dumps({"rank": rank, "seed": 1234, "counter": rank + 7}).encode(
            "utf-8"
        )
    }


# --- 子进程 worker（每个 rank 一条独立通道） ---------------------------------


def _die_before_commit(*_args: Any, **_kwargs: Any) -> None:
    """把写 COMMIT 的那一步换成 SIGKILL：留下的正是「数据齐、提交点缺失」。"""
    os.kill(os.getpid(), signal.SIGKILL)


def save_rank(job: Mapping[str, Any]) -> dict[str, Any]:
    """一个 rank 的完整动作：分片 → ``save_local`` → （可选）上传到对象存储。"""
    state: Mapping[str, np.ndarray] = job["arrays"]
    rank: int = job["rank"]
    tp: int = job["tp"]
    step: int = job["step"]

    pieces = shard_arrays(state, rank=rank, tp=tp)
    tensors = {
        name: to_device(piece, name=name, dtype=DTYPES[name], device=job.get("device", DEVICE))
        for name, piece in pieces.items()
    }
    report = ckpt.save_local(
        tensors,
        job["root"],
        step=step,
        blobs=blobs_for_rank(rank),
        layout={"tp": tp, "world_size": job["world_size"]},
        rank=rank,
        sources=sources_for_rank(state, rank=rank, tp=tp),
    )
    info: dict[str, Any] = {
        "rank": rank,
        "step": report.step,
        "state": report.state.name,
        "engine": report.engine,
        "file_size": report.file_size,
        "d2h_bytes": report.d2h_bytes,
    }
    if job.get("s3"):
        storage = S3Storage(S3Config(**job["s3"]))
        try:
            info["uploaded"] = ckpt.upload_local(
                job["root"],
                storage,
                step=step,
                rank=rank,
                generation=job.get("generation"),
            )
        finally:
            storage.close()
    return info


def save_rank_then_die(job: Mapping[str, Any]) -> None:
    """一个 rank 在写 COMMIT 之前被 SIGKILL（确定性崩溃，不按时间随机 kill）。"""
    ckpt.write_commit = _die_before_commit  # type: ignore[assignment]
    save_rank(job)
    raise AssertionError("unreachable: 子进程应当已被 SIGKILL")


def _worker(job: Mapping[str, Any]) -> None:
    info = save_rank(job)
    uploaded = f"，上传 {info['uploaded']} 字节" if "uploaded" in info else ""
    print(
        f"    [rank {info['rank']}] step {info['step']} {info['state']} "
        f"engine={info['engine']} {info['file_size']} 字节{uploaded}"
    )


def _worker_then_die(job: Mapping[str, Any]) -> None:  # pragma: no cover - 子进程里被 SIGKILL
    save_rank_then_die(job)


# --- 场景 -------------------------------------------------------------------


@dataclass
class Scenario:
    workdir: Path
    ranks: int = 2
    tp: int = 2
    device: str = DEVICE
    s3: dict[str, Any] = field(default_factory=dict)
    states: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)

    def root(self, rank: int) -> Path:
        return self.workdir / f"rank_{rank}"

    def job(self, step: int, rank: int, *, fault: bool = False) -> dict[str, Any]:
        root = self.root(rank)
        root.mkdir(parents=True, exist_ok=True)
        return {
            "rank": rank,
            "tp": self.tp,
            "world_size": self.ranks,
            "step": step,
            "root": str(root),
            "arrays": self.states[step],
            "device": self.device,
            # 故障那一版不上传：先死在写 COMMIT 上，模拟训练进程崩溃
            "s3": None if fault else dict(self.s3),
            "generation": None,
        }

    def storage(self) -> S3Storage:
        return S3Storage(S3Config(**self.s3))


def _spawn(targets: Sequence[Any], jobs: Sequence[Mapping[str, Any]]) -> list[int]:
    """每个 rank 起一个独立进程（与真实「每 rank 一条上传通道」一致）。"""
    ctx = multiprocessing.get_context("spawn")
    procs = [
        (job, ctx.Process(target=target, args=(job,)))
        for target, job in zip(targets, jobs)
    ]
    for _job, proc in procs:
        proc.start()
    codes: list[int] = []
    for job, proc in procs:
        proc.join()
        codes.append(int(proc.exitcode if proc.exitcode is not None else -1))
        if proc.exitcode not in (0, None):
            print(f"    [rank {job['rank']}] 退出码 {proc.exitcode}")
    return codes


def _run_ranks(sc: Scenario, step: int, *, fault_rank: int | None = None) -> list[int]:
    """把这一版写到每个 rank 的本地目录（并上传），返回各进程退出码。

    代际在这里统一分配（等价于 rank 0 的 ``begin_upload``）再下发：多 rank 各自
    开代会落到不同的 ``gen_*`` 前缀下，那一版永远凑不齐。
    """
    jobs = [
        sc.job(step, rank, fault=rank == fault_rank) for rank in range(sc.ranks)
    ]
    uploading = [job for job in jobs if job["s3"]]
    if uploading:
        # 只有真的要上传时才开一代（崩溃阶段一个字节都不传）
        storage = sc.storage()
        try:
            generation = ckpt.begin_upload(
                storage, step=step, ranks=list(range(sc.ranks))
            ).id
        finally:
            storage.close()
        for job in uploading:
            job["generation"] = generation
    if sc.ranks == 1 and fault_rank is None:
        _worker(jobs[0])  # 单 rank 直接在主进程跑，省一次 spawn
        return [0]
    targets = [
        _worker_then_die if rank == fault_rank else _worker for rank in range(sc.ranks)
    ]
    return _spawn(targets, jobs)


# --- 阶段 -------------------------------------------------------------------


def phase_train(sc: Scenario) -> None:
    print(f"\n[1/7] 伪训练 + 周期保存（TP={sc.tp}，{sc.ranks} 个 rank，{len(SHAPES)} 个张量）")
    sc.states = {step: state_at(step) for step in (*STEPS, CRASH_STEP)}
    total = sum(array.nbytes for array in sc.states[STEPS[0]].values())
    print(f"    step {STEPS[0]} 的 state_dict 大小：{total} 字节（{model_summary()}）")
    print(f"    保存点：{', '.join(str(s) for s in STEPS)}；崩溃注入点：{CRASH_STEP}")


def phase_save(sc: Scenario, step: int) -> None:
    codes = _run_ranks(sc, step)
    assert all(code == 0 for code in codes), f"step {step}: 保存失败 {codes}"

    storage = sc.storage()
    try:
        ckpt.finalize_upload(storage, step=step, ranks=range(sc.ranks), layout={"tp": sc.tp})
    finally:
        storage.close()
    print(f"    step {step}: 本地 COMMIT + 各 rank 上传完成，rank 0 写了版本 COMMIT")


def phase_local_gate(sc: Scenario) -> None:
    print("\n[2/7] 本地读取门禁：只有带 COMMIT 的版本可恢复")
    root = sc.root(0)
    versions = sorted(p.name for p in root.iterdir() if (p / "COMMIT").is_file())
    print(f"    rank 0 本地可恢复版本：{versions}")
    assert versions, "本地没有任何带 COMMIT 的版本"

    if sc.ranks == 1:
        data = ckpt.load(root, device=sc.device, tp=sc.tp, tp_rank=0)
        print(f"    load({root.name}) → step={data.meta.get('step')}，{len(data.tensors)} 个张量")
        pieces = {name: tensor.numpy() for name, tensor in data.tensors.items()}
        assert_same_state(pieces, state_at(LAST_SAVED), what=f"本地 step {LAST_SAVED}")
        print(f"    逐字节与 step {LAST_SAVED} 的训练状态一致")
        return

    # 多 rank：每个 rank 的本地目录只覆盖全局的一部分，**单独不可恢复**。
    # 跨 rank 的汇总发生在恢复端（合并各 rank 的 .metadata），不在本地层。
    for rank in range(sc.ranks):
        files = sorted(p.name for p in (sc.root(rank) / f"step_{LAST_SAVED}").iterdir())
        assert files, f"rank {rank} 的 step_{LAST_SAVED} 是空的"
        print(f"    rank {rank} 本地 step_{LAST_SAVED}：{files}")
    try:
        ckpt.load(root, device=sc.device, tp=sc.tp, tp_rank=0)
    except PlanError as exc:  # 预期就是覆盖不全
        print(f"    单 rank 目录直接 load 如期被拒：{type(exc).__name__}: {exc}")
    else:
        raise AssertionError("单 rank 的分片目录不该能被当成完整版本读出来")
    print("    局部目录只是上传前的中转；完整性由各 rank 的 .metadata 在恢复端合并校验")


def phase_crash(sc: Scenario) -> None:
    print(f"\n[3/7] 崩溃注入：step {CRASH_STEP} 的 rank {sc.ranks - 1} 在写 COMMIT 之前被 SIGKILL")
    codes = _run_ranks(sc, CRASH_STEP, fault_rank=sc.ranks - 1)
    victim = codes[-1]
    print(f"    崩溃 rank 退出码：{victim}（SIGKILL = {-signal.SIGKILL}）")
    assert victim == -signal.SIGKILL, f"预期被 SIGKILL，实际退出码 {victim}"

    crashed = sc.root(sc.ranks - 1) / f"step_{CRASH_STEP}"
    assert crashed.is_dir(), f"崩溃的 rank 应该留下残缺目录 {crashed}"
    files = sorted(p.name for p in crashed.iterdir())
    print(f"    崩溃 rank 的 step_{CRASH_STEP} 目录内容：{files}")
    assert files, f"{crashed} 是空的：崩溃发生在写数据之前，这一阶段就没意义了"
    assert "COMMIT" not in files, "崩溃版本不应该有 COMMIT"
    print("    数据文件与 .metadata 都在，只有 COMMIT 缺失 —— 这一版对恢复端不可见")


def phase_commit_gate(sc: Scenario) -> None:
    print(f"\n[4/7] 提交点语义：step {CRASH_STEP} 不完整 ⇒ 不可提交、不可恢复")
    storage = sc.storage()
    try:
        try:
            ckpt.finalize_upload(
                storage, step=CRASH_STEP, ranks=range(sc.ranks), layout={"tp": sc.tp}
            )
        except FormatError as exc:
            print(f"    finalize_upload(step={CRASH_STEP}) 如期失败：{exc}")
        else:
            raise AssertionError("finalize_upload 不该成功：有 rank 没传完")

        try:
            ckpt.load_object(storage, step=CRASH_STEP, device=sc.device)
        except FormatError as exc:
            print(f"    load_object(step={CRASH_STEP}) 如期报错：{exc}")
        else:
            raise AssertionError("load_object 不该成功：版本 COMMIT 不存在")

        latest = ckpt.load_object(storage, device=sc.device)
        print(f"    load_object(step=None) 仍然恢复到 step {latest.step}（崩溃那一版没顶掉它）")
        assert latest.step == LAST_SAVED, f"最新可恢复版本应为 {LAST_SAVED}，实际 {latest.step}"
    finally:
        storage.close()


def phase_restore(sc: Scenario) -> dict[str, np.ndarray]:
    print("\n[5/7] 从对象存储恢复：每个 rank 取自己那片，再整读一份交叉验证")
    storage = sc.storage()
    try:
        whole = ckpt.load_object(storage, device=sc.device, tp=1)
        print(
            f"    tp=1 整读：{len(whole.tensors)} 个张量，请求 {whole.requests} 次，"
            f"读 {whole.read_bytes} 字节，读放大 {whole.amplification:.2f}"
        )
        assert_same_state(
            {k: v.numpy() for k, v in whole.tensors.items()},
            state_at(LAST_SAVED),
            what=f"S3 step {LAST_SAVED} 全局状态",
        )
        assert set(whole.blobs) == {f"rng/rank_{r}" for r in range(sc.ranks)}, (
            f"小状态不齐：{sorted(whole.blobs)}"
        )

        parts: dict[str, list[np.ndarray]] = {}
        for rank in range(sc.ranks):
            part = ckpt.load_object(
                storage,
                device=sc.device,
                tp=sc.tp,
                tp_rank=rank,
                shard_dims=COLUMN_DIM,  # 明确说清哪些张量沿哪个维切
            )
            assert part.step == LAST_SAVED
            for name, tensor in part.tensors.items():
                parts.setdefault(name, []).append(tensor.numpy())
            print(
                f"    tp_rank={rank}：{len(part.tensors)} 个张量，"
                f"{part.h2d_bytes} 字节 H2D，读放大 {part.amplification:.2f}"
            )
        merged = {
            name: (
                np.concatenate(chunks, axis=COLUMN_DIM[name])
                if name in COLUMN_DIM
                else chunks[0]
            )
            for name, chunks in parts.items()
        }
        assert_same_state(merged, state_at(LAST_SAVED), what="各 rank 分片拼回的全局状态")
        print(f"    {sc.ranks} 个 rank 的分片拼回全局，与整读结果逐字节一致")
        return merged
    finally:
        storage.close()


def phase_resume(sc: Scenario, restored: Mapping[str, np.ndarray]) -> None:
    print(f"\n[6/7] 续训：从 step {LAST_SAVED} 恢复，再跑 {CRASH_STEP - LAST_SAVED} 步")
    state = dict(restored)
    for current in range(LAST_SAVED + 1, CRASH_STEP + 1):
        state = evolve(state, current)
    assert_same_state(state, state_at(CRASH_STEP), what=f"续训到 step {CRASH_STEP}")
    print(f"    续训结果与「一口气训到 step {CRASH_STEP}」逐字节一致（无容差）")


def phase_gc(sc: Scenario) -> None:
    print(
        "\n[7/7] 提交完成后的有界保留（只留最近 "
        f"{ckpt.DEFAULT_KEEP_VERSIONS} 代）+ 回收本地暂存（最新 1 代留作恢复层）"
    )
    storage = sc.storage()
    try:
        # rank 0 在 finalize(step=15) 里已经裁过一次；这里只复核结果，不动手删。
        versions = sorted(committed_steps(storage))
        window = sorted(STEPS)[-ckpt.DEFAULT_KEEP_VERSIONS :]
        print(f"    远端保留的已提交版本：{[f'step_{v}' for v in versions]}；保留窗口：{window}")
        assert versions == window, f"远端版本清单异常：{versions}，期望 {window}"

        oldest = sorted(STEPS)[0]
        try:
            ckpt.load_object(storage, step=oldest, device=sc.device)
        except FormatError as exc:
            print(f"    窗口外的 step {oldest} 如期不可恢复：{exc}")
        else:
            raise AssertionError(f"step {oldest} 超出保留窗口，应该已经被删掉")

        print("    回收计划（dry_run=True，只看不删）：")
        for rank in range(sc.ranks):
            plan = ckpt.gc_local(sc.root(rank), storage, rank=rank, dry_run=True)
            for item in plan.planned:
                print(f"      rank {rank}: 删 {item.location}:#{item.step}（{item.reason}）")
            for location, step, why in plan.skipped:
                print(f"      rank {rank}: 留 {location}:#{step}（{why}）")

        print("    执行（幂等：判据只看提交点在不在，删一半被打断重跑收敛）：")
        for rank in range(sc.ranks):
            report = ckpt.gc_local(sc.root(rank), storage, rank=rank, dry_run=False)
            freed = sum(item.bytes for item in report.deleted)
            print(f"      rank {rank}: 删掉 {len(report.deleted)} 个版本，回收 {freed} 字节")
            again = ckpt.gc_local(sc.root(rank), storage, rank=rank, dry_run=False)
            assert not again.deleted, f"回收不幂等：{again.deleted}"

        remaining = {
            f"rank_{rank}": sorted(p.name for p in sc.root(rank).iterdir() if p.is_dir())
            for rank in range(sc.ranks)
            if sc.root(rank).is_dir()
        }
        print(f"    幂等重跑无新增删除；本地剩余目录：{remaining}")
        assert sorted(committed_steps(storage)) == window, "本地回收不该碰远端已提交版本"

        # 回退：本地恢复了什么就直接用什么。单 rank 的本地版本是完整的一份，远端同一代
        # 又有版本 COMMIT ⇒ 零下载；多 rank 时每个本地 root 只有自己的分片，全局覆盖
        # 校验不过 ⇒ 回落下载（两份数据一样，只是省不下带宽）。
        newest = sorted(STEPS)[-1]
        print(f"    回退到 step_{newest}：本地有完整热副本就零下载，否则整份下载")
        for rank in range(sc.ranks):
            root = sc.root(rank)
            check = ckpt.local_copy_available(
                root, storage, step=newest, tp=sc.tp, tp_rank=rank, shard_dims=COLUMN_DIM
            )
            loaded = ckpt.load_checkpoint(
                storage,
                local_root=root,
                step=newest,
                device=sc.device,
                tp=sc.tp,
                tp_rank=rank,
                shard_dims=COLUMN_DIM,
            )
            print(
                f"      rank {rank}: 本地判定={check.reason}；"
                f"恢复来源={loaded.source}（下载 {loaded.download_bytes} 字节）"
            )
            if sc.ranks == 1:
                assert check.usable, f"单 rank 的本地版本应当可直接读：{check.reason}"
                assert loaded.source == "local" and loaded.download_bytes == 0
            else:
                assert not check.usable, "多 rank 的单 root 只有本 rank 分片，不能全局读"
                assert loaded.source == "object", "覆盖不全就该回落到下载"
    finally:
        storage.close()


# --- 入口 -------------------------------------------------------------------


def model_summary() -> str:
    return ", ".join(f"{name}{SHAPES[name]}:{DTYPES[name]}" for name in SHAPES)


def run(*, ranks: int, tp: int | None, workdir: Path | None, keep: bool, s3: dict[str, Any],
        use_stub: bool) -> int:
    tp = tp or ranks
    created = workdir is None
    workdir = Path(tempfile.mkdtemp(prefix="shardckpt-e2e-")) if created else workdir
    workdir.mkdir(parents=True, exist_ok=True)
    sc = Scenario(workdir=workdir, ranks=ranks, tp=tp, s3=dict(s3))

    print("=" * 78)
    print("ShardCkpt 端到端模拟：伪训练 → 保存 → 崩溃 → 上传 → 恢复续训 → 回收")
    print("=" * 78)
    print(f"device={DEVICE}（MockCudaBackend：host 内存冒充显存，读取路径与真机一致）")
    print(f"对象存储={'进程内 s3_stub' if use_stub else sc.s3.get('endpoint_url')}")
    print(f"工作目录={workdir}")

    with contextlib.ExitStack() as stack:
        if use_stub:
            server = stack.enter_context(s3_stub.start())
            sc.s3["endpoint_url"] = server.endpoint_url
            # 内置 stub 不校验签名，但 boto3 仍要一对非空凭据
            if sc.s3.get("access_key_id") is None:
                sc.s3["access_key_id"] = "demo-key"
            if sc.s3.get("secret_access_key") is None:
                sc.s3["secret_access_key"] = "demo-secret"

        try:
            phase_train(sc)
            for step in SAVED:
                print(f"\n    · 保存 step {step}：每个 rank 独立保存自己那片，然后独立上传")
                phase_save(sc, step)
            phase_local_gate(sc)
            phase_crash(sc)
            phase_commit_gate(sc)
            restored = phase_restore(sc)
            phase_resume(sc, restored)
            print(f"\n    · 训练继续：step {CRASH_STEP} 崩溃后由下一轮 step {STEPS[-1]} 补上")
            phase_save(sc, STEPS[-1])
            phase_gc(sc)
        except BaseException:
            # 失败时把工作目录留下来，方便看残缺目录 / 提交点
            print(f"\n失败：工作目录保留在 {workdir}")
            raise

    print("\n" + "=" * 78)
    print("全部阶段通过：保存、崩溃、提交点门禁、跨 rank 恢复、续训、回收都符合预期")
    print("=" * 78)
    if created and not keep:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"产物保留在 {workdir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ShardCkpt 端到端模拟（mock cuda + 进程内 S3）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ranks", type=int, default=2, help="world size（伪训练的分片数）")
    parser.add_argument("--tp", type=int, default=None, help="保存布局的 tp，默认等于 --ranks")
    parser.add_argument("--workdir", type=Path, default=None, help="工作目录，默认用临时目录")
    parser.add_argument("--keep", action="store_true", help="跑完保留工作目录")
    parser.add_argument("--endpoint-url", default=None, help="真对象存储端点（留空用内置 stub）")
    parser.add_argument("--bucket", default="ckpt")
    parser.add_argument("--prefix", default="runs/e2e")
    parser.add_argument("--access-key", default=None)
    parser.add_argument("--secret-key", default=None)
    parser.add_argument("--region", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ranks < 1:
        raise SystemExit("--ranks 必须 >= 1")
    if args.tp is not None and args.tp != args.ranks:
        # demo 里每个 rank 就是 TP 组的一份，所以两者必须一致；
        # 真机上「tp 组数 < world size」要用 dp 复制，属于 adapters/ 的范围。
        raise SystemExit("--tp 必须等于 --ranks（每个 rank 对应一个 TP 分片）")
    use_stub = not args.endpoint_url
    s3 = {
        "bucket": args.bucket,
        "prefix": args.prefix,
        "region": args.region,
        "access_key_id": args.access_key,
        "secret_access_key": args.secret_key,
    }
    return run(
        ranks=args.ranks,
        tp=args.tp,
        workdir=args.workdir,
        keep=args.keep,
        s3=s3,
        use_stub=use_stub,
    )


if __name__ == "__main__":
    sys.exit(main())
