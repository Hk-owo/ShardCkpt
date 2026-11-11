"""对象存储（S3 兼容）上的 checkpoint 上传与恢复。

设计目标（demo）：

- 对象存储用本地 S3 兼容服务（SeaweedFS / MinIO / ``tests/s3_stub.py``）模拟，
  上传路径与真机一致；
- **每个 rank 独立上传**自己的数据文件与索引，互相不等；
- **rank 0 不汇总、不读别人的索引**：它只在所有 rank 都传完之后 ``list`` 一次
  版本前缀，确认每个 rank 的 ``.metadata`` 与分片 ``COMMIT`` 都在，然后写版本
  ``COMMIT``；
- 全局覆盖校验属于**读的人**：恢复端自己 ``list`` + 读各 rank 的 per-rank 索引，
  合并后再 ``CheckpointIndex.validate()``。提交路径上不做 O(rank 数) 的索引汇总。

key 布局（相对存储配置里的 prefix）::

    step_42/
      rank_0/
        __0_0.distcp        数据（planner 布局，按 4096 补齐到块边界）
        .metadata           本 rank 的分片索引
        COMMIT              本 rank 的提交点（最后写；它一出现本地就能回收）
      rank_1/
        ...
      COMMIT                rank 0 在全部 rank 提交点齐了之后写（版本提交点）

三层提交点（语义严格递增）：本地的 ``step_N/COMMIT`` 只管「本 rank 写完了，可以传」；
``rank_<r>/COMMIT`` 管「本 rank 传完了，本地可以删」；版本 ``COMMIT`` 管「这一版齐了，
可以恢复」。

顺序上的保证：

1. 每个 rank 先本地提交（本地 ``COMMIT``），再上传；上传内部按
   「数据文件 → ``.metadata`` → ``rank_<r>/COMMIT``」写，提交点最后一步。
2. rank 0 只用 ``list`` 判断齐全，所以它不需要知道别的 rank 写了什么。
3. 恢复端先读版本 ``COMMIT``（没有就拒绝），再按 ``COMMIT`` 里的 rank 清单读索引；
   缺任何一个 rank 的索引都直接报错，不会交出半份数据。

待补（见 ``tests/sim/README.md`` 的待补全清单）：

- **attempt/epoch**：同一 step 重试时旧残片会和新的混在同一前缀下。现在
  ``finalize_upload`` 会把「COMMIT 里没声明的 rank 前缀」当错误报出来，但不隔离；
  正式版应在 key 里带尝试号，``COMMIT`` 记住这次尝试的 id。
- **流式上传**：``upload_rank`` 现在整块读数据文件（demo 规模够用）。大文件应改成
  分块 ``upload_part``，``S3Storage`` 的 multipart 分支已经具备能力。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from async_ import DeviceBackend, Tensor, get_backend
from commit.local import COMMIT_FILE, latest_step, read_commit
from core import FormatError
from core.index import META_FILE, BlobMeta, CheckpointIndex, ShardMeta, TensorMeta
from core.metadata import Device, numpy_dtype
from planner.load_plan import ReadItem, TargetTensor, build_load_plan
from planner.targets import targets_for_layout
from storage.base import ObjectNotFoundError, StorageBackend
from storage.local import LocalStorage

COMMIT_FORMAT_VERSION = 1
DEFAULT_BUDGET_BYTES = 8 << 20


# --- key 布局 ---------------------------------------------------------------


def version_prefix(step: int) -> str:
    return f"step_{step}"


def rank_prefix(step: int, rank: int) -> str:
    return f"{version_prefix(step)}/rank_{rank}"


def data_key(step: int, rank: int, file: str) -> str:
    return f"{rank_prefix(step, rank)}/{file}"


def index_key(step: int, rank: int) -> str:
    return f"{rank_prefix(step, rank)}/{META_FILE}"


def rank_commit_key(step: int, rank: int) -> str:
    return f"{rank_prefix(step, rank)}/{COMMIT_FILE}"


def commit_key(step: int) -> str:
    return f"{version_prefix(step)}/{COMMIT_FILE}"


# --- 提交记录 ---------------------------------------------------------------


@dataclass(frozen=True)
class UploadCommit:
    """rank 0 写的版本提交点（L3）。

    只有 ``step`` / ``ranks`` / ``layout``：它不记录文件清单，因为 rank 0 没有
    gather 任何人的索引 —— 「齐不齐」由每个 rank 的 :class:`RankCommit` 证明，
    「分片怎么拼」由各 rank 自己的 ``.metadata`` 证明。
    """

    step: int
    ranks: tuple[int, ...]
    layout: Mapping[str, int] = field(default_factory=dict)
    format_version: int = COMMIT_FORMAT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranks", tuple(int(rank) for rank in self.ranks))
        object.__setattr__(self, "layout", dict(self.layout or {}))
        if self.format_version != COMMIT_FORMAT_VERSION:
            raise FormatError(
                f"unsupported upload COMMIT format_version {self.format_version}, "
                f"this build understands {COMMIT_FORMAT_VERSION}"
            )
        if self.step < 0:
            raise FormatError(f"step must be >= 0, got {self.step}")
        if not self.ranks:
            raise FormatError("COMMIT must declare at least one rank")
        if len(set(self.ranks)) != len(self.ranks):
            raise FormatError(f"duplicate ranks in COMMIT: {self.ranks}")
        if any(rank < 0 for rank in self.ranks):
            raise FormatError(f"ranks must be >= 0, got {self.ranks}")

    def dumps(self) -> str:
        return json.dumps(
            {
                "format_version": self.format_version,
                "step": self.step,
                "ranks": list(self.ranks),
                "layout": {key: int(value) for key, value in sorted(self.layout.items())},
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

    @classmethod
    def loads(cls, text: str) -> "UploadCommit":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise FormatError(f"COMMIT is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise FormatError(f"COMMIT must be a JSON object, got {type(data).__name__}")
        ranks = data.get("ranks")
        if not isinstance(ranks, list):
            raise FormatError("COMMIT 'ranks' must be a list")
        layout = data.get("layout", {})
        if not isinstance(layout, dict):
            raise FormatError("COMMIT 'layout' must be an object")
        try:
            return cls(
                step=int(data["step"]),
                ranks=tuple(int(rank) for rank in ranks),
                layout={str(key): int(value) for key, value in layout.items()},
                format_version=int(data.get("format_version", COMMIT_FORMAT_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FormatError(f"COMMIT is malformed: {exc}") from exc


@dataclass(frozen=True)
class RankCommit:
    """一个 rank 的分片提交点（L2）：``step_42/rank_3/COMMIT``。

    它只证明「这个 rank 的数据对象与 ``.metadata`` 都已落盘」，**不声明完整性** ——
    该读哪些 rank 由 :class:`UploadCommit` 决定，所以这里不记 ``ranks`` / ``layout``，
    避免两份数据漂移。

    两个用途：

    1. 取代原来只写 ``{"step", "rank"}`` 的 ``DONE``：远端第一次有了每个 rank 自己的
       文件清单，``cat step_42/rank_3/COMMIT`` 就能知道这个分片传了什么，排查残片/
       半传时不用再逐个 head 对象；
    2. **驱动本地回收**：本 rank 的 L2 一出现，本地的 ``step_N/`` 就可以删了，
       不必等最慢的 rank，也不必等 rank 0 写完 L3。
    """

    step: int
    rank: int
    metadata_bytes: int = 0
    files: Mapping[str, int] = field(default_factory=dict)
    format_version: int = COMMIT_FORMAT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", {str(k): int(v) for k, v in self.files.items()})
        if self.format_version != COMMIT_FORMAT_VERSION:
            raise FormatError(
                f"unsupported rank COMMIT format_version {self.format_version}, "
                f"this build understands {COMMIT_FORMAT_VERSION}"
            )
        if self.step < 0:
            raise FormatError(f"step must be >= 0, got {self.step}")
        if self.rank < 0:
            raise FormatError(f"rank must be >= 0, got {self.rank}")
        if self.metadata_bytes < 0:
            raise FormatError(f"metadata bytes must be >= 0, got {self.metadata_bytes}")

    def dumps(self) -> str:
        return json.dumps(
            {
                "format_version": self.format_version,
                "step": self.step,
                "rank": self.rank,
                "metadata": {"bytes": self.metadata_bytes},
                "files": {name: int(size) for name, size in sorted(self.files.items())},
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

    @classmethod
    def loads(cls, text: str) -> "RankCommit":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise FormatError(f"rank COMMIT is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise FormatError(
                f"rank COMMIT must be a JSON object, got {type(data).__name__}"
            )
        metadata = data.get("metadata", {})
        files = data.get("files", {})
        if not isinstance(metadata, dict):
            raise FormatError("rank COMMIT 'metadata' must be an object")
        if not isinstance(files, dict):
            raise FormatError("rank COMMIT 'files' must be an object")
        try:
            return cls(
                step=int(data["step"]),
                rank=int(data["rank"]),
                metadata_bytes=int(metadata.get("bytes", 0)),
                files={str(name): int(size) for name, size in files.items()},
                format_version=int(data.get("format_version", COMMIT_FORMAT_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FormatError(f"rank COMMIT is malformed: {exc}") from exc


# --- 上传：每个 rank 只碰自己的前缀 ----------------------------------------


def upload_rank(
    root: str | Path,
    storage: StorageBackend,
    step: int,
    *,
    rank: int,
) -> int:
    """把本地 ``step_<step>`` 版本里属于本 rank 的产物上传到对象存储。

    返回上传的字节数（数据 + 索引 + 提交记录）。顺序：数据文件 → ``.metadata`` →
    ``rank_<rank>/COMMIT``；提交记录落盘之前，rank 0 的 ``finalize_upload`` 不会认为
    这个 rank 完成，本地回收也不会动这一版。
    """
    local = LocalStorage(root, direct_io=False)
    key = version_prefix(step)
    if read_commit(local, step) is None:
        raise FormatError(
            f"{key}: 本地版本没有 COMMIT，先 save_local() 提交再上传"
        )

    index = CheckpointIndex.loads(local.read(f"{key}/{META_FILE}").decode("utf-8"))
    index.validate(complete=False)  # 单 rank 只做局部自检，全局覆盖在恢复端

    files = sorted({shard.file for meta in index.tensors.values() for shard in meta.shards})
    files += sorted({blob.file for blob in index.blobs.values()})
    uploaded = 0
    sizes: dict[str, int] = {}
    for file in files:
        payload = local.read(f"{key}/{file}")
        # 本地索引里的 ``file`` 是「版本目录内的文件名」，对象 key 里加上 rank 前缀，
        # 这样每个 rank 的产物互不覆盖，恢复端也能从名字直接定位到 rank。
        object_name = data_key(step, rank, file)
        storage.write(object_name, payload)
        sizes[file] = len(payload)
        uploaded += len(payload)

    metadata = local.read(f"{key}/{META_FILE}")
    storage.write(index_key(step, rank), metadata)
    uploaded += len(metadata)

    record = RankCommit(
        step=step,
        rank=rank,
        metadata_bytes=len(metadata),
        files=sizes,
    )
    payload = record.dumps().encode("utf-8")
    storage.write(rank_commit_key(step, rank), payload)
    uploaded += len(payload)
    return uploaded


# --- rank 0：只 list 前缀，不读别人的索引 ----------------------------------


def finalize_upload(
    storage: StorageBackend,
    step: int,
    *,
    ranks: Sequence[int],
    layout: Mapping[str, int] | None = None,
) -> UploadCommit:
    """rank 0 的提交动作：``list`` 一次前缀，确认每个 rank 都传完了，再写 ``COMMIT``。

    ``ranks`` 是**期望的写者清单**（训练配置里的 world size），由 rank 0 自己知道，
    不需要从别处汇总。任何缺 ``.metadata`` / ``rank_<rank>/COMMIT`` 的 rank 都会让
    这一步失败，失败时不写版本 ``COMMIT``，因此这一版对恢复端不可见。

    **幂等可重跑**：判据只看每个 rank 的分片提交点在不在，所以 rank 0 崩在写完
    ``COMMIT`` 之前时，重启后再调一次（或运维当 CLI 调）就能把这一版补成可恢复的。
    """
    expected = tuple(int(rank) for rank in ranks)
    commit = UploadCommit(step=step, ranks=expected, layout=dict(layout or {}))

    keys = set(storage.list(version_prefix(step)))
    missing = [
        key
        for rank in expected
        for key in (index_key(step, rank), rank_commit_key(step, rank))
        if key not in keys
    ]
    if missing:
        raise FormatError(
            f"step {step}: {len(missing)} object(s) missing, "
            f"ranks {expected} not all uploaded: {sorted(missing)}"
        )

    strays = sorted(
        key
        for key in keys
        if key.startswith(f"{version_prefix(step)}/rank_")
        and _rank_of(key, step) not in expected
    )
    if strays:
        raise FormatError(
            f"step {step}: unexpected rank prefix(es) {strays}; "
            f"COMMIT declares ranks {expected}（可能是上一次重试的残片）"
        )

    storage.write(commit_key(step), commit.dumps().encode("utf-8"))
    return commit


def _rank_of(key: str, step: int) -> int:
    """``step_42/rank_3/.metadata`` -> 3。"""
    head = f"{version_prefix(step)}/rank_"
    if not key.startswith(head):
        raise FormatError(f"{key}: not under {head}")
    segment = key[len(head) :].split("/", 1)[0]
    try:
        return int(segment)
    except ValueError:
        raise FormatError(f"{key}: cannot parse rank from {segment!r}") from None


# --- 恢复：list + 读各 rank 索引，合并后做全局校验 -------------------------


@dataclass(frozen=True)
class ObjectVersion:
    """一个已确认可恢复的对象存储版本。"""

    storage: StorageBackend
    step: int
    key: str
    index: CheckpointIndex
    commit: UploadCommit

    def file_key(self, name: str) -> str:
        """索引里的 ``file`` 是「版本目录内、带 rank 前缀」的名字。"""
        return f"{self.key}/{name}"

    def read_blobs(self) -> dict[str, bytes]:
        return {
            name: self.storage.read_range(self.file_key(blob.file), blob.byte_offset, blob.length)
            for name, blob in sorted(self.index.blobs.items())
        }


def open_object_version(
    storage: StorageBackend, step: int | None = None
) -> ObjectVersion:
    """恢复端的第一道门：读 ``COMMIT`` → 核对 rank 前缀 → 合并 per-rank 索引。

    ``step=None`` 取「所有带 ``COMMIT`` 的版本里最大的 step」，与本地层的
    :func:`commit.local.latest_step` 是同一套语义、同一份实现。
    """
    target = step if step is not None else latest_step(storage)
    if target is None:
        raise FormatError(
            f"no committed checkpoint found under {storage!r}: "
            "只有带 COMMIT 的版本才算可恢复"
        )

    key = version_prefix(target)
    try:
        payload = storage.read(commit_key(target))
    except ObjectNotFoundError:
        raise FormatError(f"{key}: COMMIT missing, this version is not recoverable") from None
    commit = UploadCommit.loads(payload.decode("utf-8"))
    if commit.step != target:
        raise FormatError(
            f"{key}: COMMIT declares step {commit.step}, but it was read as step {target}"
        )

    keys = set(storage.list(key))
    missing = sorted(
        index_key(target, rank)
        for rank in commit.ranks
        if index_key(target, rank) not in keys
    )
    if missing:
        raise FormatError(
            f"{key}: COMMIT declares ranks {commit.ranks}, "
            f"but these index objects are missing: {missing}"
        )

    # L2 的分片提交点在这里只作「诊断证据」：它在，说明这个 rank 当初确实传完了；
    # 它不在而 L3 声明了这个 rank，说明对象被外部删过 —— 在读数据之前就报出来，
    # 而不是等 Range GET 短读。直接用上面那次 list 的结果，不额外付 GET。
    not_committed = sorted(
        rank for rank in commit.ranks if rank_commit_key(target, rank) not in keys
    )
    if not_committed:
        raise FormatError(
            f"{key}: COMMIT declares ranks {commit.ranks}, but rank(s) {not_committed} "
            f"have no per-rank COMMIT (分片提交点缺失，远端对象可能被删过)"
        )

    shards: dict[str, list[ShardMeta]] = {}
    metas: dict[str, TensorMeta] = {}
    blobs: dict[str, BlobMeta] = {}
    for rank in commit.ranks:
        rank_index = CheckpointIndex.loads(
            storage.read(index_key(target, rank)).decode("utf-8")
        )
        for name, meta in rank_index.tensors.items():
            metas.setdefault(name, meta)
            shards.setdefault(name, []).extend(
                replace(shard, file=f"rank_{rank}/{shard.file}") for shard in meta.shards
            )
        for name, blob in rank_index.blobs.items():
            # 同名小状态每个 rank 都会写一份（内容相同），只留第一份
            blobs.setdefault(name, replace(blob, file=f"rank_{rank}/{blob.file}"))

    index = CheckpointIndex(
        tensors={
            name: TensorMeta(
                global_shape=metas[name].global_shape,
                dtype=metas[name].dtype,
                shards=tuple(entries),
            )
            for name, entries in shards.items()
        },
        blobs=blobs,
        step=target,
        saved_layout=dict(commit.layout),
    )
    # 全局覆盖校验在这里做：此时才看得到所有 rank 的分片
    index.validate()
    return ObjectVersion(
        storage=storage, step=target, key=key, index=index, commit=commit
    )


# --- 读取：Range GET + scatter + H2D ---------------------------------------


@dataclass(frozen=True)
class ObjectLoadReport:
    """一次从对象存储恢复的结果与成本。"""

    step: int
    device: Device
    tensors: Mapping[str, Tensor]
    blobs: Mapping[str, bytes]
    index: CheckpointIndex
    requests: int
    windows: int
    read_bytes: int
    copied_bytes: int
    h2d_bytes: int

    @property
    def amplification(self) -> float:
        return self.read_bytes / self.copied_bytes if self.copied_bytes else 0.0


def load_object_version(
    storage: StorageBackend,
    *,
    step: int | None = None,
    device: "str | Device" = "cpu",
    tp: int = 1,
    tp_rank: int = 0,
    shard_dims: Mapping[str, int] | None = None,
    budget_bytes: int = DEFAULT_BUDGET_BYTES,
    column_strategy: str = "whole",
) -> ObjectLoadReport:
    """从对象存储恢复张量：``COMMIT`` 门禁 → 读取计划 → Range GET → staging → H2D。"""
    version = open_object_version(storage, step)
    targets = targets_for_layout(
        version.index, tp=tp, tp_rank=tp_rank, shard_dims=shard_dims
    )
    plan = build_load_plan(
        version.index, targets, strict=True, column_strategy=column_strategy
    )

    device_spec = Device.parse(device)
    backend = get_backend(device_spec)
    host_arrays = {
        target.fqn: np.zeros(target.region.shape, dtype=numpy_dtype(target.dtype))
        for target in targets
    }

    read_bytes = 0
    copied_bytes = 0
    windows = 0
    for window in plan.windows(budget_bytes):
        for item in window:
            block = storage.read_range(
                version.file_key(item.file), item.byte_offset, item.length
            )
            _scatter(item, memoryview(block), host_arrays)
            read_bytes += item.length
            copied_bytes += sum(
                span.numel * numpy_dtype(item.dtype).itemsize for span in item.spans
            )
        windows += 1

    tensors: dict[str, Tensor] = {}
    h2d_bytes = 0
    for target in targets:
        host = host_arrays[target.fqn]
        tensors[target.fqn] = _to_device(target, host, device_spec, backend)
        h2d_bytes += host.nbytes

    return ObjectLoadReport(
        step=version.step,
        device=device_spec,
        tensors=tensors,
        blobs=version.read_blobs(),
        index=version.index,
        requests=len(plan.items),
        windows=windows,
        read_bytes=read_bytes,
        copied_bytes=copied_bytes,
        h2d_bytes=h2d_bytes,
    )


def _scatter(item: ReadItem, block: memoryview, host_arrays: Mapping[str, np.ndarray]) -> None:
    """把读块里的矩形按 ``CopySpan`` 拷进目标数组（与 ``storage/local_load.py`` 同算法）。"""
    source = np.frombuffer(block, dtype=numpy_dtype(item.dtype)).reshape(item.read_shape)
    target = host_arrays[item.fqn]
    for span in item.spans:
        dst = tuple(
            slice(offset, offset + size) for offset, size in zip(span.dst_offsets, span.shape)
        )
        target[dst] = source[span.src_region().slices]


def _to_device(
    target: TargetTensor, host: np.ndarray, device: Device, backend: DeviceBackend
) -> Tensor:
    payload = np.ascontiguousarray(host)
    buffer = backend.allocate(payload.nbytes)
    # memoryview(np 数组) 的长度是元素数，copy_h2d 要的是字节视图
    backend.copy_h2d(buffer, memoryview(payload).cast("B"))
    return Tensor(
        name=target.fqn,
        buffer=buffer,
        device=device,
        dtype=target.dtype,
        shape=tuple(int(dim) for dim in host.shape),
        backend=backend,
    )


__all__ = [
    "DEFAULT_BUDGET_BYTES",
    "ObjectLoadReport",
    "ObjectVersion",
    "RankCommit",
    "UploadCommit",
    "commit_key",
    "data_key",
    "finalize_upload",
    "index_key",
    "load_object_version",
    "open_object_version",
    "rank_commit_key",
    "rank_prefix",
    "upload_rank",
    "version_prefix",
]
