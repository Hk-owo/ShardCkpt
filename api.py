"""save_local / load：唯一把 core / planner / storage / async_ / commit 串起来的地方。

对外接口按「从 GPU 读取」设计：

1. 加载时显式指定目标设备（对应 ``torch.load(map_location=...)``）
2. 数据路径固定为  host 文件 -> staging buffer -> H2D 拷贝 -> device buffer
3. 迁移与取值都经过 backend，调用方从不直接触碰 device 内存

示例：
    >>> import api as ckpt
    >>> data = ckpt.load("/tmp/data/step_0", device="cuda:0")
    >>> data["model.embed_tokens.weight"].device
    Device(type='cuda', index=0)
    >>> data.to("cpu")["model.embed_tokens.weight"].numpy().shape
    (1000, 128)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from async_ import (
    BackendStats,
    CpuBackend,
    DeviceBackend,
    GateBlock,
    GateCollective,
    MockCudaBackend,
    PinnedBufferPool,
    Snapshot,
    SnapshotManager,
    SnapshotState,
    Tensor,
    TriggerDecision,
    available_device_types,
    close_default_pool,
    configure_default_pool,
    default_pool,
    get_backend,
    register_backend,
    reset_registry,
)
from commit.generation import FencedError, Generation, acquire_generation, new_writer
from commit.gc import (
    DEFAULT_KEEP_LOCAL_VERSIONS,
    DEFAULT_KEEP_VERSIONS,
    GcReport,
    GcWorker,
    RetentionReport,
    collect,
    delete_local_version,
    retain_remote_versions,
)
from commit.local import (
    CommitRecord,
    version_key,
    write_commit,
    write_metadata,
)
from core.index import BlobMeta, CheckpointIndex, ShardMeta, TensorMeta
from core.region import Region
from planner.save_plan import SavePlan, ShardSource, build_save_plan
from storage.local import DEFAULT_ALIGNMENT, LocalStorage
from core import (
    CheckpointError,
    ChecksumError,
    FormatError,
    MissingTensorError,
    PlanError,
    UnsupportedDeviceError,
)
from core.metadata import CPU, Device, cuda, numpy_dtype
from storage.checksum import CRC_CHUNK_MAX, crc32c_chunks
from storage.data import CheckpointData
from storage.local_load import LoadReport, load_local
from storage.upload import (
    DEFAULT_BUDGET_BYTES as DEFAULT_UPLOAD_BUDGET_BYTES,
    LocalCopyCheck,
    ObjectLoadReport,
    ObjectVersion,
    UploadCommit,
    check_local_copy,
    finalize_upload as _finalize_upload,
    load_checkpoint_version,
    load_object_version,
    upload_rank as _upload_rank,
)


def load(
    path: str | Path,
    *,
    device: "str | Device" = "cpu",
    verify_crc: bool = True,
    crc_padding: str = "block",
    crc_padding_ratio: float = 0.125,
    step: int | None = None,
    tp: int = 1,
    tp_rank: int = 0,
    shard_dims: Mapping[str, int] | None = None,
) -> CheckpointData:
    """便捷入口，对应 ``torch.load(path, map_location=device)``。

    只接受本地版本目录（``COMMIT`` / ``.metadata``）：读路径统一走
    :func:`storage.local_load.load_local`，校验按索引里的 CRC 块做。
    """
    root = Path(path)
    if _is_version_dir(root):
        base, version = _resolve_version_root(root)
        report = load_local(
            base,
            step=step if step is not None else version,
            device=device,
            tp=tp,
            tp_rank=tp_rank,
            shard_dims=shard_dims,
            verify_crc=verify_crc,
            crc_padding=crc_padding,
            crc_padding_ratio=crc_padding_ratio,
        )
        return CheckpointData(
            tensors=dict(report.tensors),
            meta={
                "format": "local",
                "step": report.step,
                "saved_layout": dict(report.index.saved_layout),
                "files": report.files,
                "blobs": sorted(report.blobs),
            },
            device=report.device,
        )

    raise FormatError(
        f"{root}: no committed version directory (COMMIT / .metadata) found"
    )


def _is_version_dir(root: Path) -> bool:
    """本地版本目录：根目录本身就是 ``step_N``，或者根下有带 COMMIT 的 step 目录。"""
    if (root / "COMMIT").is_file() or (root / ".metadata").is_file():
        return True
    if not root.is_dir():
        return False
    return any(
        child.is_dir() and (child / "COMMIT").is_file()
        for child in root.iterdir()
    )


def _resolve_version_root(root: Path) -> tuple[Path, int | None]:
    """``step_42`` 目录本身也接受：返回 (父目录, 42)，让 recover 能拼出 key。"""
    if not (root / "COMMIT").is_file():
        return root, None
    name = root.name
    if not name.startswith("step_"):
        raise FormatError(
            f"{root}: version directory must be named 'step_<N>' to be recovered by step"
        )
    try:
        return root.parent, int(name[len("step_") :])
    except ValueError:
        raise FormatError(f"{root}: cannot parse step number from {name!r}") from None


def save_async(state_dict: Any, path: str | Path, **options: Any) -> Any:
    """异步保存入口：后台进程 + 上传队列还没做，现在请用 :func:`save_local`。"""
    raise NotImplementedError("save_async is not implemented yet; use save_local()")


# --- 本地层的写闭环 ---------------------------------------------------------


@dataclass(frozen=True)
class SaveReport:
    """一次本地保存的结果。``state`` 成功时是 ``COMMITTED``。"""

    step: int
    key: str
    file: str
    size: int
    data_bytes: int
    engine: str
    index: CheckpointIndex
    commit: CommitRecord
    state: SnapshotState
    d2h_bytes: int
    # 这次租块是否命中了池里已有的块；池不提供统计时为 None
    pool_reused: bool | None = None

    @property
    def file_size(self) -> int:
        return self.commit.files.get(self.file, 0)


def _sources_from_tensors(tensors: Mapping[str, Tensor]) -> list[ShardSource]:
    """没有分片信息时，每个张量整体作为一个 Replicate 分片。"""
    sources: list[ShardSource] = []
    for name, tensor in tensors.items():
        shape = tuple(int(dim) for dim in tensor.shape)
        if not shape:
            raise ValueError(f"{name}: 0 维张量请走 blobs 路径，不要当分片")
        sources.append(
            ShardSource(
                fqn=name,
                global_shape=shape,
                dtype=tensor.dtype,
                region=Region(tuple(0 for _ in shape), shape),
            )
        )
    return sources


def _as_view(buffer: Any) -> memoryview:
    view = getattr(buffer, "view", None)
    return view if isinstance(view, memoryview) else memoryview(buffer)


def _pool_reuses(pool: Any) -> int | None:
    """池的累计复用次数；自定义池没有 ``stats.reuses`` 时返回 ``None``。"""
    reuses = getattr(getattr(pool, "stats", None), "reuses", None)
    return reuses if isinstance(reuses, int) else None


def _build_index(
    plan: SavePlan,
    *,
    buffer: Any,
    step: int,
    layout: Mapping[str, int] | None,
    crc_chunk_max: int,
) -> CheckpointIndex:
    """从「已经躺在 buffer 里的字节」直接生成索引。

    ``buffer``（对齐后的 pinned 块）里每个分片/小状态按 ``crc_chunk_max`` 切块，
    每块算一次 CRC-32C 写进索引：这是本地读路径做字节级自证（``verify_crc=True``）
    的依据。算的是 ``[byte_offset, end)`` 这段**有效字节**，不含对齐补齐（详见
    :mod:`storage.local_load`）。
    """
    grouped: dict[str, list[ShardMeta]] = {}
    shapes: dict[str, tuple[tuple[int, ...], str]] = {}
    for item in plan.items:
        chunk_bytes, chunks = crc32c_chunks(
            buffer[item.byte_offset : item.end],
            itemsize=numpy_dtype(item.dtype).itemsize,
            max_bytes=crc_chunk_max,
        )
        grouped.setdefault(item.fqn, []).append(
            item.to_shard_meta(crc_chunk_bytes=chunk_bytes, crc32c_chunks=chunks)
        )
        shapes[item.fqn] = (item.global_shape, item.dtype)

    tensors = {
        fqn: TensorMeta(
            global_shape=shapes[fqn][0], dtype=shapes[fqn][1], shards=tuple(shards)
        )
        for fqn, shards in grouped.items()
    }
    blobs: dict[str, BlobMeta] = {}
    for blob in plan.blobs:
        chunk_bytes, chunks = crc32c_chunks(
            buffer[blob.byte_offset : blob.end], max_bytes=crc_chunk_max
        )
        blobs[blob.name] = blob.to_blob_meta(
            crc_chunk_bytes=chunk_bytes, crc32c_chunks=chunks
        )
    return CheckpointIndex(
        tensors=tensors, blobs=blobs, step=step, saved_layout=dict(layout or {})
    )


def _clear_padding(plan: SavePlan, view: memoryview, end: int) -> None:
    """对齐缓冲区没初始化，必须把「不属于任何项」的字节清零，否则写盘内容不确定。"""
    spans = [(item.byte_offset, item.end) for item in plan.items]
    spans += [(blob.byte_offset, blob.end) for blob in plan.blobs]
    cursor = 0
    for start, stop in sorted(spans):
        if start > cursor:
            view[cursor:start] = b"\x00" * (start - cursor)
        cursor = max(cursor, stop)
    if cursor < end:
        view[cursor:end] = b"\x00" * (end - cursor)


def save_local(
    tensors: Mapping[str, Tensor],
    root: str | Path,
    *,
    step: int,
    blobs: Mapping[str, bytes] | None = None,
    layout: Mapping[str, int] | None = None,
    rank: int = 0,
    sources: list[ShardSource] | None = None,
    alignment: int = DEFAULT_ALIGNMENT,
    engine: str = "auto",
    direct_io: bool = True,
    crc_chunk_max: int = CRC_CHUNK_MAX,
    pool: PinnedBufferPool | None = None,
    manager: SnapshotManager | None = None,
    collective: GateCollective | None = None,
    world_size: int | None = None,
) -> SaveReport:
    """把一组 device 上的张量写成一份可恢复的本地 checkpoint（同步版本）。

    顺序固定：plan → 取 pinned 块 → 按 ``item.byte_offset`` D2H → 写数据文件 →
    写 ``.metadata`` → 写 ``COMMIT``，每一步都对应状态机的一条边。

    ``tensors``: FQN -> ``Tensor``（真机在 GPU 上，mock 下是 host 内存）
    ``blobs``:   小状态的**已序列化字节**（RNG / lr scheduler / dataloader 位置）
    ``layout``:  写进索引的 ``saved_layout``，只用于诊断与快路径
    ``crc_chunk_max``: CRC 块上限（默认 1 MiB）；块数由分片长度推导，不会截断
    ``pool``:    pinned 块池；``None`` 时用进程级常驻池（``api.default_pool()``），
                 同一份 checkpoint 反复保存时复用同一块，不再每次现造现弃。
                 常驻池在每次租块前会 ``trim()`` 掉多余空闲块，稳态只占一块。
    ``collective``: 多 rank 训练必须给（``dist.Collective`` 的任意实现）。门闸裁决由
                 一次 ``all_reduce(MAX)`` 统一得出：任一 rank 拿不到 buffer，**全体**
                 这一 step 都不写，不会出现「有的 rank 写、有的跳过」的残缺版本。
                 通信世界的 rank / world_size 由 collective 自己给出。
    ``world_size``: 只用于**显式声明**参与通信的 rank 数；不填则按 collective 推导
                 （没有 collective 即 1）。声明 > 1 却没给 collective 直接报错。
    """
    if manager is not None and collective is not None:
        raise ValueError(
            "manager 与 collective 只能给一个：collective 必须经 SnapshotManager 注入，"
            "两者同时给会有一份被静默忽略"
        )
    blob_payloads = dict(blobs or {})
    if sources is None:
        sources = _sources_from_tensors(tensors)

    plan = build_save_plan(
        sources,
        rank=rank,
        blobs={name: len(payload) for name, payload in blob_payloads.items()},
        align=alignment,
    )
    plan.validate()
    if plan.size == 0:
        raise PlanError("保存计划是空的：没有分片也没有小状态")

    # O_DIRECT 要求长度对齐：文件按块边界补齐，读取端按索引里的 length 截断
    aligned_size = -(-plan.size // alignment) * alignment

    if pool is None:
        # 默认走进程级常驻池：同一份 checkpoint 反复保存时复用同一块（不再每次
        # 「按整份大小现造、用完即弃」）。先把多出来的空闲块削掉，稳态只有一块。
        pool = default_pool(alignment=alignment)
        trim = getattr(pool, "trim", None)
        if trim is not None:
            trim(keep_bytes=aligned_size)
    snapshot_manager = manager or SnapshotManager(
        pool, collective=collective, world_size=world_size
    )

    reuses_before = _pool_reuses(pool)
    trigger = snapshot_manager.trigger(step, nbytes=aligned_size)
    if trigger.decision is not TriggerDecision.START or trigger.snapshot is None:
        # 门闸裁决是全局统一的：这里抛错时其他 rank 也拿到同一个裁决，
        # 于是这一 step 全体都不写，不会留下凑不齐分片的残缺版本。
        raise PlanError(f"step {step}: 拿不到 pinned buffer（{trigger.reason}）")
    snapshot: Snapshot = trigger.snapshot
    reuses_after = _pool_reuses(pool)
    pool_reused = (
        None
        if reuses_before is None or reuses_after is None
        else reuses_after > reuses_before
    )

    storage = LocalStorage(root, direct_io=direct_io, alignment=alignment, engine=engine)
    key = version_key(step)

    try:
        d2h_bytes = 0
        for item in plan.items:
            tensor = tensors.get(item.fqn)
            if tensor is None:
                raise PlanError(f"{item.fqn}: 计划里有，但没有对应的张量")
            if tensor.nbytes != item.length:
                raise PlanError(
                    f"{item.fqn}: 本地张量 {tensor.nbytes} 字节与计划里的 "
                    f"{item.length} 不一致"
                )
            d2h_bytes += tensor.copy_to(snapshot.buffer, offset=item.byte_offset)

        view = _as_view(snapshot.buffer)
        for blob in plan.blobs:
            payload = blob_payloads.get(blob.name)
            if payload is None:
                raise PlanError(f"{blob.name}: 缺少小状态字节")
            if len(payload) != blob.length:
                raise PlanError(
                    f"{blob.name}: {len(payload)} 字节与计划里的 {blob.length} 不一致"
                )
            view[blob.byte_offset : blob.end] = payload
            d2h_bytes += len(payload)

        _clear_padding(plan, view, aligned_size)
        snapshot.d2h_done()  # host 数据从这一刻起不可变

        index = _build_index(
            plan,
            buffer=snapshot.buffer.view,
            step=step,
            layout=layout,
            crc_chunk_max=crc_chunk_max,
        )
        # 单 rank 只看得到自己那片：`complete=False` 只做局部自检。全局覆盖由
        # 「汇总/恢复的一方」负责——本地多 rank 汇总后，或对象存储恢复端合并索引后。
        # 用 ``complete=True`` 会让任何带分片的保存（TP/ZeRO）直接 PlanError。
        index.validate(complete=False)

        # 1) 数据文件：临时文件 -> fsync -> rename -> 目录 fsync
        storage.write_view(f"{key}/{plan.file}", view[:aligned_size])
        file_size = storage.path_for(f"{key}/{plan.file}").stat().st_size
        if file_size != aligned_size:
            raise PlanError(f"{key}/{plan.file}: 写入 {file_size} 字节，期望 {aligned_size}")

        # 2) 索引，也兼作提交点之一
        metadata_bytes = write_metadata(storage, step, index)
        snapshot.persist_ok()  # 本 rank 的数据已落地

        # 3) COMMIT：这一步之后这一版才可恢复
        record = CommitRecord(
            step=step,
            metadata_bytes=metadata_bytes,
            files={plan.file: file_size},
        )
        write_commit(storage, record)
        snapshot.commit_ok()
    except BaseException as exc:
        # 状态机自己区分：还没写过文件 -> DROPPED；有残缺目录 -> ABORTED
        snapshot.fail(f"{type(exc).__name__}: {exc}")
        if snapshot.state is SnapshotState.ABORTED:
            # 就地清掉残缺目录（先 COMMIT 再数据），然后把状态机的待清理标记结掉。
            # 这样「本地无 COMMIT 的残目录」几乎只剩 SIGKILL 一种来源，定时 GC 只兜底。
            # 清理失败不能掩盖原始异常。
            try:
                delete_local_version(root, step)
                snapshot.mark_cleaned()
            except Exception:  # pragma: no cover - 清理失败不掩盖原始异常
                pass
        raise

    return SaveReport(
        step=step,
        key=key,
        file=plan.file,
        size=aligned_size,
        data_bytes=plan.data_bytes,
        engine=storage.engine,
        index=index,
        commit=record,
        state=snapshot.state,
        d2h_bytes=d2h_bytes,
        pool_reused=pool_reused,
    )


# --- 对象存储上传（每 rank 独立上传，rank 0 只 list 检查） -------------------


def begin_upload(
    storage: Any,
    *,
    step: int,
    ranks: "list[int] | tuple[int, ...]",
    collective: "GateCollective | None" = None,
    writer: str | None = None,
) -> Generation:
    """开一代（一次保存尝试）并把它广播给所有 rank。

    **多 rank 必须先调这个函数**，然后把返回的 ``Generation`` 传给每个 rank 的
    :func:`upload_local` 与 rank 0 的 :func:`finalize_upload`。原因：代际是「这次
    尝试」的身份，两个 rank 各自 :func:`upload_local` 自动开代会落到不同的
    ``gen_*`` 前缀下，那一版永远凑不齐（并且掩盖了真正的问题）。

    ``collective`` 用一次 ``all_reduce(MAX)`` 把 rank 0 抢到的 ``(seq, writer)``
    广播出去（rank 0 提真实值、其余提 0）：不需要新的通信原语，也不允许分支。
    单 rank 直接走恒等路径。

    端点探针在这里跑一次（``S3Storage.ensure_conditional_writes``）：没有真正执行
    条件写的端点直接失败，不降级。
    """
    ensure = getattr(storage, "ensure_conditional_writes", None)
    if callable(ensure):
        ensure()
    expected = tuple(int(rank) for rank in ranks)
    if collective is None or collective.world_size == 1:
        return acquire_generation(storage, step, ranks=expected, writer=writer)

    if writer is not None and not _is_hex_writer(writer):
        raise ValueError(
            f"writer must be 8 hex chars when broadcasting, got {writer!r}"
        )
    if collective.rank == 0:
        mine = acquire_generation(
            storage, step, ranks=expected, writer=writer or new_writer()
        )
        seq, token = mine.seq, int(mine.writer, 16)
    else:
        mine, seq, token = None, 0, 0
    seq, token = collective.all_reduce_max_int64([seq, token])
    if mine is not None:
        return mine
    return Generation(step=step, seq=seq, writer=format(token, "08x"))


def _is_hex_writer(writer: str) -> bool:
    try:
        return len(writer) == 8 and int(writer, 16) >= 0
    except ValueError:
        return False


def upload_local(
    root: str | Path,
    storage: Any,
    *,
    step: int,
    rank: int,
    generation: "Generation | str | None" = None,
) -> int:
    """把本地 ``step_<step>`` 里**本 rank** 的产物上传到对象存储，返回上传字节数。

    每个 rank 只写自己那一代的前缀（``step_42/gen_x/rank_<i>/``），先数据文件、
    再索引、最后 ``rank_<i>/COMMIT``；三个都落盘才算这个 rank 完成。这个提交点
    同时也是「本地 ``step_<step>/`` 可以删了」的证据。

    ``generation`` 省略时走单 rank 便利路径（复用现有 GEN 或开第一代）。**多 rank
    必须显式传入** :func:`begin_upload` 的结果，否则两次尝试的分片会混在同一代里。
    """
    return _upload_rank(root, storage, step=step, rank=rank, generation=generation)


def finalize_upload(
    storage: Any,
    *,
    step: int,
    ranks: "list[int] | tuple[int, ...]",
    generation: "Generation | str | None" = None,
    layout: Mapping[str, int] | None = None,
    keep: int | None = DEFAULT_KEEP_VERSIONS,
) -> UploadCommit:
    """rank 0 的提交动作：``list`` 本代前缀确认全部 rank 都传完了，再 CAS 写 ``COMMIT``。

    它**不读**任何 rank 的索引，也不汇总分片：齐全性由每个 rank 的 ``rank_<i>/COMMIT``
    证明，分片怎么拼由各 rank 的 ``.metadata`` 在恢复端合并时证明。判据只看提交点在
    不在，所以这一步可以幂等重跑（rank 0 崩了，重启补一次即可）。

    **fence**：先用 CAS 把 GEN 推到 ``sealed``，再用 CAS 写版本 ``COMMIT``。已经提交过
    更高 ``seq`` 的版本会让本次提交直接 :class:`FencedError` —— 旧代接管不了 L3。

    **提交完成即做有界保留**：版本 ``COMMIT`` 落盘之后，rank 0 接着只留最新 ``keep``
    代（默认 3 代），从最旧的一代开始删 —— 先删版本 ``COMMIT``，再删分片提交点与数据。
    ``keep=None`` 关掉自动裁剪；要看/单跑这一步用 :func:`retain_remote_versions`。
    """
    return _finalize_upload(
        storage,
        step,
        ranks=ranks,
        generation=generation,
        layout=layout,
        keep=keep,
    )


def load_object(
    storage: Any,
    *,
    step: int | None = None,
    device: "str | Device" = "cpu",
    tp: int = 1,
    tp_rank: int = 0,
    shard_dims: Mapping[str, int] | None = None,
    budget_bytes: int = DEFAULT_UPLOAD_BUDGET_BYTES,
    column_strategy: str = "whole",
    verify_crc: bool = True,
    crc_padding: str = "block",
    crc_padding_ratio: float = 0.125,
    io_concurrency: int = 1,
    local_staging_dir: str | Path | None = None,
) -> ObjectLoadReport:
    """从对象存储恢复：整文件下载到本地临时目录 → 本地读闭环按块校验 → H2D。"""
    return load_object_version(
        storage,
        step=step,
        device=device,
        tp=tp,
        tp_rank=tp_rank,
        shard_dims=shard_dims,
        budget_bytes=budget_bytes,
        column_strategy=column_strategy,
        verify_crc=verify_crc,
        crc_padding=crc_padding,
        crc_padding_ratio=crc_padding_ratio,
        io_concurrency=io_concurrency,
        local_staging_dir=local_staging_dir,
    )


def local_copy_available(
    local_root: str | Path | None,
    storage: Any,
    *,
    step: int | None = None,
    tp: int = 1,
    tp_rank: int = 0,
    shard_dims: Mapping[str, int] | None = None,
    column_strategy: str = "whole",
) -> LocalCopyCheck:
    """本地热副本能不能顶替一次下载（判定，不读数据）。

    条件是两条同时成立：远端有这一代的版本 ``COMMIT``，本地 ``step_N/`` 带 ``COMMIT``
    且索引覆盖请求的布局。
    """
    return check_local_copy(
        local_root,
        storage,
        step=step,
        tp=tp,
        tp_rank=tp_rank,
        shard_dims=shard_dims,
        column_strategy=column_strategy,
    )


def load_checkpoint(
    storage: Any,
    *,
    local_root: str | Path | None = None,
    step: int | None = None,
    device: "str | Device" = "cpu",
    tp: int = 1,
    tp_rank: int = 0,
    shard_dims: Mapping[str, int] | None = None,
    budget_bytes: int = DEFAULT_UPLOAD_BUDGET_BYTES,
    column_strategy: str = "whole",
    verify_crc: bool = True,
    crc_padding: str = "block",
    crc_padding_ratio: float = 0.125,
    io_concurrency: int = 1,
    local_staging_dir: str | Path | None = None,
    prefer_local: bool = True,
) -> ObjectLoadReport:
    """恢复一个版本：**本地热副本优先**，本地不满足条件再下载（回退用这个入口）。

    本地 ``local_root`` 下留着这一代（``gc_local`` 的恢复层窗口）、且远端同一代有版本
    ``COMMIT`` 时，直接读本地（``report.source == "local"``，零下载）；否则与
    :func:`load_object` 一样下载整份到临时目录再读。

    ``prefer_local=False`` 强制下载；``local_root=None`` 等价于 :func:`load_object`。
    """
    return load_checkpoint_version(
        storage,
        local_root=local_root,
        step=step,
        device=device,
        tp=tp,
        tp_rank=tp_rank,
        shard_dims=shard_dims,
        budget_bytes=budget_bytes,
        column_strategy=column_strategy,
        verify_crc=verify_crc,
        crc_padding=crc_padding,
        crc_padding_ratio=crc_padding_ratio,
        io_concurrency=io_concurrency,
        local_staging_dir=local_staging_dir,
        prefer_local=prefer_local,
    )


# --- 本地暂存回收（最近 1 代留作恢复层） ------------------------------------


def gc_local(
    local_root: str | Path,
    storage: Any,
    *,
    rank: int,
    dry_run: bool = True,
    keep_local: int = DEFAULT_KEEP_LOCAL_VERSIONS,
) -> GcReport:
    """回收本地 ``step_N/``：已上传的版本、以及被更新版本越过的残片。

    判据只依赖 step 单调性 —— **没有租约，没有时间阈值**：

    - 远端的 ``step_N/rank_<rank>/COMMIT`` 在 ⇒ 本 rank 的分片已经到远端 ⇒ 本地这份
      是冗余（本 rank 传完就能删，不等最慢的 rank，也不等 rank 0 写版本 COMMIT）；
    - 存在比 N 更新的版本 ``COMMIT`` ⇒ 训练已经越过第 N 代 ⇒ 本地 ``step_N/``
      （哪怕它从没上传成功）与远端整个 ``step_N/`` 前缀都是残片；
    - 其余一律保留：最新且未完成的可能就是**正在写**（本地）或**正在传**（远端）。

    ``keep_local`` 是**本地恢复层**：最近 ``keep_local`` 个「远端已有版本 ``COMMIT``
    且本 rank 已上传」的本地版本不删，回退到这一代时直接读本地、零下载
    （:func:`load_checkpoint`）；``0`` 回到「上传完即删」。窗口按「远端可恢复」定，
    所以留下的这份永远不是唯一副本。

    对象存储侧的**有界保留**是另一条判据（只留最近 ``keep`` 代已提交版本），由 rank 0
    在 :func:`finalize_upload` 里执行，不在这里：这个函数要能安全地周期跑，而删「完整
    但过旧」的版本只在「刚提交完一代」这个时刻做才有明确的锚点。

    ``dry_run=True``（默认）只返回计划。远端不可达时什么都不删（``GcReport.aborted``
    里记原因）。周期调用用 :class:`commit.gc.GcWorker`。
    """
    return collect(
        local_root, storage, rank=rank, dry_run=dry_run, keep_local=keep_local
    )


__all__ = [
    "CPU",
    "BackendStats",
    "CheckpointData",
    "CheckpointError",
    "ChecksumError",
    "CpuBackend",
    "DEFAULT_KEEP_LOCAL_VERSIONS",
    "DEFAULT_KEEP_VERSIONS",
    "Device",
    "DeviceBackend",
    "FormatError",
    "GcReport",
    "GcWorker",
    "GateBlock",
    "GateCollective",
    "LoadReport",
    "LocalCopyCheck",
    "MissingTensorError",
    "MockCudaBackend",
    "PinnedBufferPool",
    "RetentionReport",
    "SaveReport",
    "SnapshotManager",
    "SnapshotState",
    "Tensor",
    "TriggerDecision",
    "UnsupportedDeviceError",
    "FencedError",
    "Generation",
    "available_device_types",
    "begin_upload",
    "close_default_pool",
    "configure_default_pool",
    "cuda",
    "default_pool",
    "gc_local",
    "get_backend",
    "ObjectLoadReport",
    "ObjectVersion",
    "UploadCommit",
    "finalize_upload",
    "load",
    "load_checkpoint",
    "load_local",
    "load_object",
    "local_copy_available",
    "register_backend",
    "reset_registry",
    "retain_remote_versions",
    "save_async",
    "save_local",
    "upload_local",
]
