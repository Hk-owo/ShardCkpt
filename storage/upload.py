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
      gen_0003-1f3a9c2b/      一代 = 一次保存尝试（``docs/generation.md``）
        rank_0/
          __0_0.distcp        数据（planner 布局，按 4096 补齐到块边界）
          .metadata           本 rank 的分片索引
          COMMIT              本 rank 的提交点（最后写；它一出现本地就能回收）
        rank_1/
          ...
      GEN                     rank 0 抢到的当前代际（CAS 寄存器，写路径的 fence）
      COMMIT                  rank 0 在全部 rank 提交点齐了之后写（版本提交点）

**代际解决的是「重试残片混代」**：同一 step 的两次尝试落在不同 ``gen_*`` 前缀下，
所以旧尝试的 ``rank_0`` 永远不会和新尝试的 ``rank_1`` 拼成一份「看着完整」的版本。
恢复端要求每个 ``rank_r/COMMIT`` 声明的 generation 与版本 ``COMMIT`` 一致，不一致
直接 ``FormatError`` —— 混代在读第一个字节之前就被挡住。

**fence 解决的是「落后的写者」**：写者在三个检查点读/GEN 或 CAS 推进它（见
``commit/generation.py``），落后就抛 :class:`commit.generation.FencedError`，
不写自己的 L2，因此这一代永远凑不齐；已提交版本的 L3 也被 CAS 保护，旧代提交不了。

三层提交点（语义严格递增）：本地的 ``step_N/COMMIT`` 只管「本 rank 写完了，可以传」；
``rank_<r>/COMMIT`` 管「本 rank 传完了，本地可以删」；版本 ``COMMIT`` 管「这一版齐了，
可以恢复」。

顺序上的保证：

1. 每个 rank 先本地提交（本地 ``COMMIT``），再上传；上传内部按
   「数据文件 → ``.metadata`` → ``rank_<r>/COMMIT``」写，提交点最后一步。
2. rank 0 只用 ``list`` 判断齐全，所以它不需要知道别的 rank 写了什么。
3. 恢复端先读版本 ``COMMIT``（没有就拒绝），再按 ``COMMIT`` 里的 rank 清单读索引；
   缺任何一个 rank 的索引都直接报错，不会交出半份数据。

两层提交点都记着自己那份 ``.metadata`` 的**字节数 + CRC-32C**（本地 ``COMMIT`` 记本地
索引，``rank_<r>/COMMIT`` 记本 rank 索引），算法统一走 :mod:`storage.checksum`。
恢复端逐份比对后才合并 —— 合并后的 ``validate()`` 只看结构，改一个 ``step`` 或挪一个
``byte_offset`` 的索引在结构上完全合法。

对象存储侧**有界保留**：rank 0 写完版本 ``COMMIT`` 之后立刻只留最新 ``keep`` 代
（默认 3 代），从最旧的一代开始删 —— 先删它的版本 ``COMMIT``，再删分片提交点与数据
（:func:`commit.gc.retain_remote_versions`）。``keep=None`` 关掉自动裁剪。

待补（见 ``tests/sim/README.md`` 的待补全清单）：

- **每 M 步永久保留一份**：现在的保留窗口只有「最近 ``keep`` 代」，长跑时 rollback
  距离被限制在 ``keep`` 个 checkpoint 以内；硬链接/归档到另一个前缀还没做。
- **失败重试与后台上传队列**（``save_async``）：单次上传内部的 409 重开已经做了，
  跨「整代重传」的自动重试还没有。

已落地：代际隔离与 fence（本模块 + ``commit/generation.py``）、流式分片上传
（``S3Storage.write_file`` 的条件挂在 complete 上）、对象级条件写与端点探针
（``storage/object.py``）。
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
import warnings
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from async_ import Tensor
from commit.gc import (
    DEFAULT_KEEP_VERSIONS,
    reclaim_stale_generations,
    retain_remote_versions,
)
from commit.generation import (
    STATUS_FILE,
    STATUS_FORMAT_VERSION,
    CommittedState,
    FencedError,
    Generation,
    as_generation,
    commit_status,
    read_status,
    resolve_generation,
    status_key,
    verify_holder,
)
from commit.local import (
    COMMIT_FILE,
    CommitRecord,
    check_crc32c,
    delete_local_generation,
    read_commit,
    read_local_generation,
    version_key,
    write_commit,
    write_local_generation,
    write_metadata,
)
from core import CheckpointError, ChecksumError, FormatError, InvalidTargetError, PlanError
from core.index import META_FILE, BlobMeta, CheckpointIndex, ShardMeta, TensorMeta
from core.metadata import Device
from dist import Collective, CollectiveError, SingleRankCollective
from planner.load_plan import build_load_plan
from planner.targets import targets_for_layout
from storage.base import (
    ConditionalBackend,
    ConditionalWritesDisabled,
    ObjectNotFoundError,
    StorageBackend,
    StorageError,
    UnsupportedConditionalWrites,
    is_retryable_storage_error,
    run_scope,
    validate_run_id,
)
from storage.checksum import crc32c_hex
from storage.local import LocalStorage
from storage.local_load import LoadReport, load_local
from storage.recover import recover_version, verify_blob_crc32c

logger = logging.getLogger(__name__)

RANK_COMMIT_FORMAT_VERSION = 2
"""L2 ``rank_r/COMMIT`` 的格式版本（必须带 ``generation`` / ``seq`` 与 ``metadata.crc32c``）。"""

DEFAULT_BUDGET_BYTES = 8 << 20
DEFAULT_DOWNLOAD_RETRIES = 3
DEFAULT_DOWNLOAD_BACKOFF = 0.05

STAGING_DIR_PREFIX = "shardckpt-"
STAGING_LOCK_FILE = ".shardckpt.lock"

LOAD_OK = 0
LOAD_VERSION_CORRUPT = 1
LOAD_FATAL = 2


def local_run_root(root: str | Path, run_id: str | None) -> Path:
    """把本地根路径按轮次隔离到 ``<root>/<run_id>``（``run_id=None`` 时不变）。"""
    base = Path(root)
    if run_id is None:
        return base
    return base / validate_run_id(run_id)


# --- key 布局 ---------------------------------------------------------------


def version_prefix(step: int) -> str:
    return f"step_{step}"


def generation_prefix(step: int, generation: "Generation | str") -> str:
    """``step_42/gen_0003-1f3a9c2b``。"""
    gid = generation.id if isinstance(generation, Generation) else str(generation)
    if not gid:
        raise FormatError("generation prefix must not be empty")
    return f"{version_prefix(step)}/{gid}"


def rank_prefix(step: int, rank: int, generation: "Generation | str") -> str:
    return f"{generation_prefix(step, generation)}/rank_{rank}"


def data_key(step: int, rank: int, file: str, generation: "Generation | str") -> str:
    return f"{rank_prefix(step, rank, generation)}/{file}"


def index_key(step: int, rank: int, generation: "Generation | str") -> str:
    return f"{rank_prefix(step, rank, generation)}/{META_FILE}"


def rank_commit_key(step: int, rank: int, generation: "Generation | str") -> str:
    return f"{rank_prefix(step, rank, generation)}/{COMMIT_FILE}"


# --- 提交记录 ---------------------------------------------------------------


@dataclass(frozen=True)
class UploadCommit:
    """rank 0 提交后的版本视图（来自 ``step_N/STATUS`` 的 ``committed`` 字段）。

    它不记录文件清单：rank 0 没有 gather 任何人的索引 —— 「齐不齐」由每个 rank 的
    :class:`RankCommit` 证明，「分片怎么拼」由各 rank 自己的 ``.metadata`` 证明。
    """

    step: int
    ranks: tuple[int, ...]
    layout: Mapping[str, int] = field(default_factory=dict)
    generation: str = ""
    seq: int = 0
    format_version: int = STATUS_FORMAT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranks", tuple(int(rank) for rank in self.ranks))
        object.__setattr__(self, "layout", dict(self.layout or {}))
        if self.format_version != STATUS_FORMAT_VERSION:
            raise FormatError(
                f"unsupported upload COMMIT format_version {self.format_version}, "
                f"this build understands {STATUS_FORMAT_VERSION}"
            )
        if self.step < 0:
            raise FormatError(f"step must be >= 0, got {self.step}")
        if not self.ranks:
            raise FormatError("COMMIT must declare at least one rank")
        if len(set(self.ranks)) != len(self.ranks):
            raise FormatError(f"duplicate ranks in COMMIT: {self.ranks}")
        if any(rank < 0 for rank in self.ranks):
            raise FormatError(f"ranks must be >= 0, got {self.ranks}")
        parsed = Generation.parse(self.generation, step=self.step)
        if parsed.seq != self.seq:
            raise FormatError(
                f"COMMIT 的 generation {self.generation} 与 seq {self.seq} 不一致"
            )

    @classmethod
    def from_committed(cls, committed: CommittedState) -> "UploadCommit":
        return cls(
            step=committed.step,
            ranks=committed.ranks,
            layout=dict(committed.layout),
            generation=committed.generation.id,
            seq=committed.seq,
            format_version=STATUS_FORMAT_VERSION,
        )

    def dumps(self) -> str:
        return json.dumps(
            {
                "format_version": self.format_version,
                "step": self.step,
                "ranks": list(self.ranks),
                "layout": {key: int(value) for key, value in sorted(self.layout.items())},
                "generation": self.generation,
                "seq": self.seq,
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
                generation=str(data["generation"]),
                seq=int(data["seq"]),
                format_version=int(data["format_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FormatError(f"COMMIT is malformed: {exc}") from exc


@dataclass(frozen=True)
class RankCommit:
    """一个 rank 的分片提交点（L2）：``step_42/<gen>/rank_3/COMMIT``。

    它只证明「这个 rank 的数据对象与 ``.metadata`` 都已落盘」，**不声明完整性** ——
    该读哪些 rank 由 ``STATUS.committed`` 决定，所以这里不记 ``ranks`` / ``layout``，
    避免两份数据漂移。

    ``metadata_crc32c`` 是**本 rank** 那份 ``.metadata`` 的字节级摘要，算法与 L1 的
    ``COMMIT`` 一致（CRC-32C，8 位小写十六进制）。
    """

    step: int
    rank: int
    metadata_bytes: int = 0
    metadata_crc32c: str | None = None
    files: Mapping[str, int] = field(default_factory=dict)
    generation: str | None = None
    seq: int = 0
    format_version: int = RANK_COMMIT_FORMAT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", {str(k): int(v) for k, v in self.files.items()})
        if self.format_version != RANK_COMMIT_FORMAT_VERSION:
            raise FormatError(
                f"unsupported rank COMMIT format_version {self.format_version}, "
                f"this build understands {RANK_COMMIT_FORMAT_VERSION}"
            )
        if self.step < 0:
            raise FormatError(f"step must be >= 0, got {self.step}")
        if self.rank < 0:
            raise FormatError(f"rank must be >= 0, got {self.rank}")
        if self.metadata_bytes < 0:
            raise FormatError(f"metadata bytes must be >= 0, got {self.metadata_bytes}")
        if self.metadata_crc32c is None:
            raise FormatError(
                "rank COMMIT must record 'metadata.crc32c' (index checksum)"
            )
        check_crc32c(self.metadata_crc32c, "rank COMMIT metadata.crc32c")
        if self.generation is None:
            raise FormatError("rank COMMIT must record 'generation'")
        parsed = Generation.parse(self.generation, step=self.step)
        if parsed.seq != self.seq:
            raise FormatError(
                f"rank COMMIT 的 generation {self.generation} 与 seq {self.seq} 不一致"
            )

    def dumps(self) -> str:
        data: dict[str, object] = {
            "format_version": self.format_version,
            "step": self.step,
            "rank": self.rank,
            "metadata": {
                "bytes": self.metadata_bytes,
                "crc32c": self.metadata_crc32c,
            },
            "files": {name: int(size) for name, size in sorted(self.files.items())},
            "generation": self.generation,
            "seq": self.seq,
        }
        return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)

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
        crc32c = metadata.get("crc32c")
        if crc32c is not None and not isinstance(crc32c, str):
            raise FormatError("rank COMMIT 'metadata.crc32c' must be a string")
        generation = data.get("generation")
        try:
            return cls(
                step=int(data["step"]),
                rank=int(data["rank"]),
                metadata_bytes=int(metadata.get("bytes", 0)),
                metadata_crc32c=crc32c,
                files={str(name): int(size) for name, size in files.items()},
                generation=None if generation is None else str(generation),
                seq=int(data.get("seq", 0)),
                format_version=int(data.get("format_version", RANK_COMMIT_FORMAT_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FormatError(f"rank COMMIT is malformed: {exc}") from exc


# --- 上传：每个 rank 只碰自己的前缀 ----------------------------------------


_CONDITIONAL_METHODS = ("put_if_absent", "put_if_match", "head", "write_file")


def _as_conditional(storage: StorageBackend, *, what: str) -> ConditionalBackend:
    """写路径只接受条件后端：没有 CAS 的写入会让 fence 静默失效。

    用**鸭子检查**而不是 ``isinstance(ConditionalBackend)``：``runtime_checkable``
    的 Protocol 只认类上静态定义的属性，而测试里的透传包装（``__getattr__`` 代理）
    是运行期才有的。判据是「``supports_conditional_writes`` 为真 且 四个方法可调用」。
    """
    missing = [
        name for name in _CONDITIONAL_METHODS if not callable(getattr(storage, name, None))
    ]
    if missing or not getattr(storage, "supports_conditional_writes", False):
        raise FencedError(
            f"{what}: 存储后端没有条件写能力（缺 {missing or 'supports_conditional_writes'}）；"
            "写路径要求 S3Storage 的 conditional_writes=require"
        )
    return storage  # type: ignore[return-value]


def _reuse_uploaded_object(
    storage: ConditionalBackend, object_name: str, expected_size: int
) -> None:
    """``If-None-Match`` 返回已存在：是幂等重传，还是同代出现了第二个写者？

    大小相同视为「我上一轮已经传过这个对象」（重试幂等）；不同则同代同 key 出现了
    两份不同内容，必须报错而不是覆盖。ETag 不作内容比对（multipart 的 ETag 不是
    MD5），字节级一致性由对象级校验和与本地块级 CRC-32C 负责。
    """
    meta = storage.head(object_name)
    if meta is None:
        raise StorageError(
            f"{object_name}: If-None-Match 返回已存在，但 head 拿不到对象（并发删除？）"
        )
    if meta.size != expected_size:
        raise FormatError(
            f"{object_name}: 同一代际里已有一个 {meta.size} 字节的对象，本 rank 这份是 "
            f"{expected_size} 字节 —— 同代同 key 出现第二个写者，拒绝覆盖"
        )


def upload_rank(
    root: str | Path,
    storage: StorageBackend,
    step: int,
    *,
    rank: int,
    generation: "Generation | str | None" = None,
    run_id: str | None = None,
    cancel_event: threading.Event | None = None,
) -> int:
    """把本地 ``step_<step>`` 里**本 rank** 的产物上传到对象存储，返回上传字节数。

    每个 rank 只写自己那一代的前缀（``step_42/gen_x/rank_<i>/``），先数据文件、
    再索引、最后 ``rank_<i>/COMMIT``；三个都落盘才算这个 rank 完成。这个提交点
    同时也是「本地 ``step_<step>/`` 可以删了」的证据。

    所有 key 都落在 ``step_<step>/<generation>/rank_<rank>/`` 下，写入用
    ``If-None-Match: *``（同代同 key 不会被静默覆盖）。**写 L2 之前**做一次 fence
    检查：若自己已不是当前代际，直接 ``FencedError`` 且不写 L2 —— 这一代因此永远
    凑不齐，rank 0 不会把它提升为版本。
    """
    scoped_storage = run_scope(storage, run_id)
    local = LocalStorage(local_run_root(root, run_id), direct_io=False)
    key = version_prefix(step)
    if read_commit(local, step) is None:
        raise FormatError(
            f"{key}: 本地版本没有 COMMIT，先 save_local() 提交再上传"
        )

    conditional = _as_conditional(
        scoped_storage, what=f"upload_rank(step={step}, rank={rank})"
    )
    gen = as_generation(generation, step=step)
    if gen is None:
        # 单 rank 便利路径：GEN 已有就复用它，否则开第一代。多 rank 必须先
        # begin_upload() 分配，否则两次尝试的分片会混在同一代里。
        gen = resolve_generation(scoped_storage, step, ranks=(rank,))

    index = CheckpointIndex.loads(local.read(f"{key}/{META_FILE}").decode("utf-8"))
    index.validate(complete=False)  # 单 rank 只做局部自检，全局覆盖在恢复端

    # L2-pre：花带宽之前先证明我还持有这一代
    verify_holder(scoped_storage, gen)
    # R6：同步确保本地版本目录记下当前代际标记，热副本恢复时做纯本地比对
    existing_marker = read_local_generation(local, step)
    if (
        existing_marker is None
        or existing_marker.generation != gen.id
        or existing_marker.run_id != run_id
    ):
        write_local_generation(local, step, gen.id, run_id=run_id)

    files = sorted(
        {shard.file for meta in index.tensors.values() for shard in meta.shards}
        | {blob.file for blob in index.blobs.values()}
    )
    uploaded = 0
    sizes: dict[str, int] = {}
    for file in files:
        if cancel_event is not None and cancel_event.is_set():
            raise FencedError(f"step {step} ({gen.id}): 上传已被协调层作废取消")
        # 本地索引里的 ``file`` 是「版本目录内的文件名」，对象 key 里加上代际与 rank
        # 前缀，这样同 step 的不同尝试、以及各 rank 的产物互不覆盖。
        object_name = data_key(step, rank, file, gen)
        source = local.path_for(f"{key}/{file}")
        expected = source.stat().st_size
        outcome = conditional.write_file(object_name, source, if_none_match=True)
        if not outcome.written:
            _reuse_uploaded_object(conditional, object_name, expected)
        sizes[file] = expected
        uploaded += expected

    if cancel_event is not None and cancel_event.is_set():
        raise FencedError(f"step {step} ({gen.id}): 上传已被协调层作废取消")

    metadata = local.read(f"{key}/{META_FILE}")
    object_name = index_key(step, rank, gen)
    outcome = conditional.put_if_absent(object_name, metadata)
    if not outcome.written:
        _reuse_uploaded_object(conditional, object_name, len(metadata))
    uploaded += len(metadata)

    if cancel_event is not None and cancel_event.is_set():
        raise FencedError(f"step {step} ({gen.id}): 上传已被协调层作废取消")

    record = RankCommit(
        step=step,
        rank=rank,
        metadata_bytes=len(metadata),
        metadata_crc32c=crc32c_hex(metadata),
        files=sizes,
        generation=gen.id,
        seq=gen.seq,
    )
    payload = record.dumps().encode("utf-8")
    object_name = rank_commit_key(step, rank, gen)
    outcome = conditional.put_if_absent(object_name, payload)
    if not outcome.written:
        _reuse_uploaded_object(conditional, object_name, len(payload))
    uploaded += len(payload)
    return uploaded


# --- rank 0：只 list 前缀，不读别人的索引 ----------------------------------


def _status_steps(storage: StorageBackend) -> list[int]:
    steps: set[int] = set()
    for key in storage.list(""):
        parts = key.split("/")
        if len(parts) != 2 or parts[1] != STATUS_FILE:
            continue
        name = parts[0]
        if not name.startswith("step_"):
            continue
        try:
            steps.add(int(name[len("step_") :]))
        except ValueError:
            continue
    return sorted(steps)


def list_remote_candidates(
    storage: StorageBackend,
    *,
    skip_corrupt_status: bool = False,
) -> tuple[list[tuple[int, str]], tuple[tuple[int, str], ...]]:
    """按 step 从大到小列举远端已提交的 ``(step, generation_id)`` 候选列表。

    ``skip_corrupt_status=False``（默认严格模式）：遇到无法解析的 ``STATUS`` 直接抛错；
    ``skip_corrupt_status=True``：打告警并跳过损坏的 ``STATUS``，把跳过原因记进第二个返回值。
    """
    candidates: list[tuple[int, str]] = []
    skipped: list[tuple[int, str]] = []
    for step in reversed(_status_steps(storage)):
        try:
            status = read_status(storage, step)
        except (FormatError, UnicodeDecodeError) as exc:
            if not skip_corrupt_status:
                raise
            msg = f"step_{step}: 跳过损坏的 STATUS（{exc}）"
            logger.warning(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            skipped.append((step, f"corrupt STATUS: {exc}"))
            continue
        if status is not None and status.committed is not None:
            candidates.append((step, status.committed.generation.id))
    return candidates, tuple(skipped)


def committed_remote_steps(
    storage: StorageBackend, *, skip_corrupt_status: bool = False
) -> list[int]:
    """扫描远端所有 ``STATUS.committed`` 非空的 step，按升序返回。"""
    candidates, _ = list_remote_candidates(
        storage, skip_corrupt_status=skip_corrupt_status
    )
    return sorted(step for step, _ in candidates)


def latest_remote_step(
    storage: StorageBackend, *, skip_corrupt_status: bool = False
) -> int | None:
    """远端最新的已提交版本：扫描 ``step_N/STATUS``，从大到小找首个 ``committed`` 非空者。

    与本地层 :func:`commit.local.latest_step` 不同，远端正在传或半途失败的 step
    也会有 ``STATUS``（只是 ``committed == null``），不能只靠文件名判别。
    """
    candidates, _ = list_remote_candidates(
        storage, skip_corrupt_status=skip_corrupt_status
    )
    return candidates[0][0] if candidates else None


def promote_remote_tails(
    storage: StorageBackend,
    *,
    keep: int | None = DEFAULT_KEEP_VERSIONS,
    reclaim_stale: bool = True,
    run_id: str | None = None,
) -> list[UploadCommit]:
    """C4 / R16b：重启时仅提升「远端最新一代分片已齐、仅缺总提交点」的尾巴。

    只由 rank 0 在恢复选版前调用；只检查当前轮次下不早于最新已提交 step 的未提交
    ``STATUS.active`` 代际。分片不齐或无 ``STATUS`` 的残片一律不救，交由 GC 清理。
    """
    scoped = run_scope(storage, run_id)
    if not getattr(scoped, "supports_conditional_writes", False):
        return []

    steps = _status_steps(scoped)
    if not steps:
        return []

    latest_committed: int | None = None
    statuses: dict[int, Any] = {}
    for step in reversed(steps):
        try:
            status = read_status(scoped, step)
        except (FormatError, UnicodeDecodeError):
            continue
        if status is None:
            continue
        statuses[step] = status
        if latest_committed is None and status.committed is not None:
            latest_committed = step

    promoted: list[UploadCommit] = []
    for step in sorted(statuses):
        if latest_committed is not None and step < latest_committed:
            continue
        status = statuses[step]
        if (
            status.committed is not None
            and status.committed.generation.id == status.generation.id
        ):
            continue
        gen = status.generation
        prefix = generation_prefix(step, gen)
        keys = set(scoped.list(prefix))
        missing = [
            key
            for rank in status.ranks
            for key in (index_key(step, rank, gen), rank_commit_key(step, rank, gen))
            if key not in keys
        ]
        if missing:
            continue
        strays = [
            key
            for key in keys
            if key.startswith(f"{prefix}/rank_")
            and _rank_of(key, prefix) not in status.ranks
        ]
        if strays:
            continue
        try:
            promoted.append(
                finalize_upload(
                    scoped,
                    step,
                    ranks=status.ranks,
                    generation=gen,
                    keep=keep,
                    reclaim_stale=reclaim_stale,
                )
            )
        except (FormatError, FencedError):
            continue
    return promoted


def finalize_upload(
    storage: StorageBackend,
    step: int,
    *,
    ranks: Sequence[int],
    generation: "Generation | str | None" = None,
    layout: Mapping[str, int] | None = None,
    keep: int | None = DEFAULT_KEEP_VERSIONS,
    reclaim_stale: bool = True,
    run_id: str | None = None,
) -> UploadCommit:
    """rank 0 的提交动作：``list`` 本代前缀确认每个 rank 都传完了，再单点 CAS 更新 ``STATUS.committed``。

    ``ranks`` 是**期望的写者清单**（训练配置里的 world size），由 rank 0 自己知道，
    不需要从别处汇总。任何缺 ``.metadata`` / ``rank_<rank>/COMMIT`` 的 rank 都会让
    这一步失败，失败时不写入 ``STATUS.committed``，因此这一版对恢复端不可见。

    **代际与 fence**：``acquire_generation`` 与 ``commit_status`` 共享同一个
    ``step_N/STATUS`` 对象与同一份 ``ETag``。先读 ``STATUS`` 确认本代仍是
    ``active`` 且没被更高 ``seq`` 覆盖，检查本代前缀齐全后，用 ``If-Match`` 单次
    CAS 写入 ``committed`` —— 新代一旦在期间抢走写权，``ETag`` 就已变化，服务端
    原子返回 412，不存在跨 key 并发窗口。

    **幂等可重跑**：``STATUS.committed`` 已指向本代时直接返回，**不产生额外写入**。

    **有界保留 & 同 step 旧代**：提交成功后立刻保留最新 ``keep`` 代已提交版本，并顺手
    收掉本 step 下序号严格更早的代际前缀；收尾失败不影响本次提交。
    """
    storage = run_scope(storage, run_id)
    _as_conditional(storage, what=f"finalize_upload(step={step})")
    expected = tuple(int(rank) for rank in ranks)
    gen = as_generation(generation, step=step)
    if gen is None:
        gen = resolve_generation(storage, step, ranks=expected)

    status = read_status(storage, step)
    if status is None:
        raise FencedError(
            f"{status_key(step)}: STATUS 不存在，{gen.id} 不再持有写权"
        )
    if status.committed is not None:
        if status.committed.seq > gen.seq:
            raise FencedError(
                f"step {step}: STATUS 已被 seq={status.committed.seq}"
                f"（{status.committed.generation.id}）提交，{gen.id} 已被 fence"
            )
        if status.committed.seq == gen.seq:
            if status.committed.generation.id != gen.id:
                raise FencedError(
                    f"step {step}: STATUS 声明已提交 seq={status.committed.seq} 属于 "
                    f"{status.committed.generation.id}，不是 {gen.id}（同序号不同写者）"
                )
            if status.committed.ranks != expected:
                raise FormatError(
                    f"step {step}: 已提交的 STATUS 声明 ranks {status.committed.ranks}，"
                    f"本次传入 {expected}"
                )
            return UploadCommit.from_committed(status.committed)

    if status.generation.id != gen.id:
        raise FencedError(
            f"step {step}: 当前代际是 {status.generation.id}（{status.state}），"
            f"{gen.id} 已被 fence"
        )
    if status.ranks != expected:
        raise FormatError(
            f"step {step}: STATUS 声明 ranks {status.ranks}，finalize 传入 {expected}"
        )

    prefix = generation_prefix(step, gen)
    keys = set(storage.list(prefix))
    missing = [
        key
        for rank in expected
        for key in (index_key(step, rank, gen), rank_commit_key(step, rank, gen))
        if key not in keys
    ]
    if missing:
        raise FormatError(
            f"step {step}: {len(missing)} object(s) missing, "
            f"ranks {expected} not all uploaded under {prefix}: {sorted(missing)}"
        )

    strays = sorted(
        key
        for key in keys
        if key.startswith(f"{prefix}/rank_")
        and _rank_of(key, prefix) not in expected
    )
    if strays:
        raise FormatError(
            f"step {step}: unexpected rank prefix(es) {strays} under {prefix}; "
            f"STATUS declares ranks {expected}"
        )

    committed = commit_status(
        storage,
        status,
        CommittedState(generation=gen, ranks=expected, layout=dict(layout or {})),
    )
    commit = UploadCommit.from_committed(committed)
    if reclaim_stale:
        # STATUS.committed 已经指向本代：同一个 step 下序号更早的前缀只可能是被 fence 的残片，
        # 顺手收掉，不必等周期 collect()。只删 seq 严格更小的代 —— 更新的一代
        # 可能已经在传。失败不能反过来把提交判成失败。
        try:
            reclaim_stale_generations(storage, step, keep=gen.id)
        except Exception:  # pragma: no cover - 收尾动作，不能回滚提交
            pass
    if keep is not None:
        # 提交已经成立，裁剪失败不能反过来把这次提交判成失败：retain_remote_versions
        # 自己吞掉 StorageError（记进 aborted），这里再兜住其它意外。
        try:
            retain_remote_versions(storage, keep=keep)
        except Exception:  # pragma: no cover - 保留是收尾动作，不能回滚提交
            pass
    return commit


def _rank_of(key: str, prefix: str) -> int:
    """``step_42/gen_x/rank_3/.metadata`` -> 3（``prefix`` 是 ``step_42/gen_x``）。"""
    head = f"{prefix}/rank_"
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
    generation: str
    """这一版属于哪一代。"""
    file_sizes: Mapping[str, int] = field(default_factory=dict)
    """各 rank 分片提交点声明的 ``rank_<r>/<file> -> 预期字节数``（R1）。"""

    def file_key(self, name: str) -> str:
        """索引里的 ``file`` 是「版本目录内、带 rank 前缀」的名字。"""
        return f"{self.key}/{self.generation}/{name}"

    def read_blobs(self, *, verify_crc: bool = True) -> dict[str, bytes]:
        blobs: dict[str, bytes] = {}
        for name, blob in sorted(self.index.blobs.items()):
            payload = self.storage.read_range(
                self.file_key(blob.file), blob.byte_offset, blob.length
            )
            if verify_crc:
                verify_blob_crc32c(name, payload, blob)
            blobs[name] = payload
        return blobs


def open_object_version(
    storage: StorageBackend,
    step: int | None = None,
    *,
    expected_generation: str | None = None,
    skip_corrupt_status: bool = False,
    run_id: str | None = None,
) -> ObjectVersion:
    """恢复端的第一道门：读 ``STATUS.committed`` → 核对 rank 前缀 → 合并 per-rank 索引。

    ``step=None`` 取「所有 ``STATUS.committed`` 非空的版本里最大的 step」。
    """
    scoped = run_scope(storage, run_id)
    # 精确指定 step 时绝不跳过坏 STATUS（R8）
    target = (
        step
        if step is not None
        else latest_remote_step(scoped, skip_corrupt_status=skip_corrupt_status)
    )
    if target is None:
        raise FormatError(
            f"no committed checkpoint found under {scoped!r}: "
            "只有 STATUS.committed 非空的版本才算可恢复"
        )

    key = version_prefix(target)
    status = read_status(scoped, target)
    if status is None or status.committed is None:
        raise FormatError(
            f"{key}: STATUS missing or uncommitted, this version is not recoverable"
        )
    commit = UploadCommit.from_committed(status.committed)
    if commit.step != target:
        raise FormatError(
            f"{key}: STATUS declares step {commit.step}, but it was read as step {target}"
        )
    if expected_generation is not None and commit.generation != expected_generation:
        raise FormatError(
            f"{key}: STATUS committed generation is {commit.generation!r}, "
            f"expected locked generation {expected_generation!r}"
        )

    base = generation_prefix(target, commit.generation)
    keys = set(scoped.list(base))
    missing = sorted(
        index_key(target, rank, commit.generation)
        for rank in commit.ranks
        if index_key(target, rank, commit.generation) not in keys
    )
    if missing:
        raise FormatError(
            f"{key}: STATUS declares ranks {commit.ranks}, "
            f"but these index objects are missing: {missing}"
        )

    # L2 的分片提交点在这里既是「诊断证据」也是**混代检测点**：它不在而 STATUS 声明了
    # 这个 rank，说明对象被外部删过；它在但声明的 generation 与 STATUS.committed 不一致，
    # 说明这一版是两次尝试各传一半拼出来的 —— 两种都在读第一个字节之前报出来。
    not_committed = sorted(
        rank
        for rank in commit.ranks
        if rank_commit_key(target, rank, commit.generation) not in keys
    )
    if not_committed:
        raise FormatError(
            f"{key}: STATUS declares ranks {commit.ranks}, but rank(s) {not_committed} "
            f"have no per-rank COMMIT (分片提交点缺失，远端对象可能被删过)"
        )
    index_payloads: dict[int, bytes] = {}
    expected_file_sizes: dict[str, int] = {}
    for rank in commit.ranks:
        rank_commit = RankCommit.loads(
            scoped.read(rank_commit_key(target, rank, commit.generation)).decode("utf-8")
        )
        # R11：补齐 step 断言，形成 (step, rank, generation) 三坐标完整门禁
        if rank_commit.step != target:
            raise FormatError(
                f"{base}/rank_{rank}/COMMIT 声明 step {rank_commit.step}，期望 {target}"
            )
        if rank_commit.rank != rank:
            raise FormatError(
                f"{base}/rank_{rank}/COMMIT 声明 rank {rank_commit.rank}，路径与内容不一致"
            )
        if rank_commit.generation != commit.generation:
            raise FormatError(
                f"{base}: rank {rank} 的分片提交点声明 generation "
                f"{rank_commit.generation!r}，STATUS.committed 声明 {commit.generation!r} —— "
                "这一版混了两次尝试的分片，拒绝恢复"
            )

        # 合并索引是 N 份 ``.metadata`` 拼出来的：任何一份被改坏都会让「分片怎么拼」
        # 出错，而合并后的 ``validate()`` 看不出结构自洽的篡改 —— 长度 + CRC-32C 才是
        # 逐 rank 的字节级凭据。
        rank_index_payload = scoped.read(index_key(target, rank, commit.generation))
        if len(rank_index_payload) != rank_commit.metadata_bytes:
            raise FormatError(
                f"{base}/rank_{rank}/{META_FILE}: {len(rank_index_payload)} bytes, "
                f"rank COMMIT records {rank_commit.metadata_bytes}"
            )
        assert rank_commit.metadata_crc32c is not None
        actual = crc32c_hex(rank_index_payload)
        if actual != rank_commit.metadata_crc32c:
            raise ChecksumError(
                f"{base}/rank_{rank}/{META_FILE}: crc32c {actual}, rank COMMIT "
                f"records {rank_commit.metadata_crc32c}（分片索引与提交点不符）",
                file=f"rank_{rank}/{META_FILE}",
            )
        index_payloads[rank] = rank_index_payload
        for fname, fsize in rank_commit.files.items():
            expected_file_sizes[f"rank_{rank}/{fname}"] = int(fsize)

    shards: dict[str, list[ShardMeta]] = {}
    metas: dict[str, TensorMeta] = {}
    blobs: dict[str, BlobMeta] = {}
    for rank in commit.ranks:
        rank_index = CheckpointIndex.loads(index_payloads[rank].decode("utf-8"))
        # R11：分片索引声明的 step 也必须严格等于目标 step
        if rank_index.step != target:
            raise FormatError(
                f"{base}/rank_{rank}/{META_FILE} 声明 step {rank_index.step}，期望 {target}"
            )
        for name, meta in rank_index.tensors.items():
            metas.setdefault(name, meta)
            shards.setdefault(name, []).extend(
                replace(shard, file=f"rank_{rank}/{shard.file}") for shard in meta.shards
            )
        for name, blob in rank_index.blobs.items():
            # R11：不同 rank 声明同名小状态属于归属冲突，显式报错而不是静默覆盖
            if name in blobs:
                raise FormatError(
                    f"{key}: 同名小状态 {name!r} 被多个 rank 声明（rank {rank} 冲突）"
                )
            blobs[name] = replace(blob, file=f"rank_{rank}/{blob.file}")

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
    # R1：合并索引引用的每个文件都必须在对应 rank 的 COMMIT.files 清单内
    referenced = _version_files(index)
    missing_from_manifest = sorted(set(referenced) - set(expected_file_sizes))
    if missing_from_manifest:
        raise FormatError(
            f"{key}: rank COMMIT files 清单缺少索引引用的文件：{missing_from_manifest}"
        )
    # 全局覆盖校验在这里做：此时才看得到所有 rank 的分片
    index.validate()
    return ObjectVersion(
        storage=scoped,
        step=target,
        key=key,
        index=index,
        commit=commit,
        generation=commit.generation,
        file_sizes={name: expected_file_sizes[name] for name in referenced},
    )


# --- 读取：本地热副本优先，缺了再下载 --------------------------------------

SOURCE_LOCAL = "local"
"""恢复来源：本地保留的热副本（零下载）。"""

SOURCE_OBJECT = "object"
"""恢复来源：从对象存储整份下载到临时目录。"""

GENERATION_MISMATCH = "generation_mismatch"
"""本地热副本的代际标记与集群锁定的目标代际不一致。"""


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
    download_bytes: int = 0
    """本次从对象存储下载的字节数（整个版本的数据文件）。"""
    local_staging: str | None = None
    """下载用的临时目录路径；报告返回时已经清理，只留作诊断。"""
    crc_verified: bool = True
    verified_blocks: int = 0
    io_requests: int = 0
    unverified: tuple[str, ...] = ()
    unverified_bytes: int = 0
    source: str = SOURCE_OBJECT
    """数据从哪来：``"local"``（本地热副本，零下载）或 ``"object"``（下载）。"""
    redownloaded: tuple[str, ...] = ()
    """下载后本地按块校验失败、触发定向重下 1 次恢复的文件清单（R3）。"""
    skipped_versions: tuple[tuple[int, str], ...] = ()
    """因损坏/坏 STATUS 而被跳过的候选版本 ``(step, reason)``（R8/R9）。"""
    local_fallback_reason: str | None = None
    """本地热副本未命中或损坏而回退远端下载的原因（R5/R6）。"""
    generation: str | None = None
    """本次恢复实际使用的代际标识。"""
    run_id: str | None = None
    """本次恢复实际读取的轮次标识（R14）。"""

    @property
    def amplification(self) -> float:
        return self.read_bytes / self.copied_bytes if self.copied_bytes else 0.0


# --- R4：暂存目录文件锁与残留清理 -------------------------------------------


def _acquire_staging_dir(local_staging_dir: str | Path | None) -> tuple[Path, int]:
    """创建暂存目录并持有 OS 级排他文件锁（R4）。

    先在同父目录以 ``.creating-`` 前缀建目录并抢占 ``flock(LOCK_EX | LOCK_NB)``，
    再原子重命名为 ``shardckpt-*`` —— 任何时候都不会出现无锁的 ``shardckpt-*``。
    """
    parent = (
        Path(local_staging_dir)
        if local_staging_dir is not None
        else Path(tempfile.gettempdir())
    )
    parent.mkdir(parents=True, exist_ok=True)
    cleanup_stale_staging_dirs(parent)

    creating = Path(
        tempfile.mkdtemp(prefix=f".creating-{STAGING_DIR_PREFIX}", dir=str(parent))
    )
    lock_fd = -1
    try:
        lock_path = creating / STAGING_LOCK_FILE
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(lock_fd, f"{os.getpid()}\n".encode("ascii"))
        final_dir = creating.with_name(creating.name[len(".creating-") :])
        os.replace(creating, final_dir)
        return final_dir, lock_fd
    except BaseException:
        if lock_fd >= 0:
            try:
                os.close(lock_fd)
            except OSError:
                pass
        shutil.rmtree(creating, ignore_errors=True)
        raise


def _release_staging_dir(path: Path, lock_fd: int) -> None:
    """删除暂存目录并释放 OS 文件锁。"""
    try:
        shutil.rmtree(path, ignore_errors=True)
    finally:
        try:
            os.close(lock_fd)
        except OSError:
            pass


def cleanup_stale_staging_dirs(*roots: str | Path | None) -> tuple[Path, ...]:
    """扫描历史 ``shardckpt-*`` 暂存目录，非阻塞抢锁清理已死进程的残留（R4）。

    只清理同时满足「带 ``shardckpt-`` 前缀 + 存在 ``.shardckpt.lock`` + 非阻塞
    ``flock`` 成功」的目录；兄弟 rank 仍在使用的目录抢不到锁会自动跳过。
    """
    cleaned: list[Path] = []
    seen_roots: set[Path] = set()
    for root in roots:
        if root is None:
            continue
        parent = Path(root)
        try:
            resolved = parent.resolve()
        except OSError:
            resolved = parent
        if resolved in seen_roots or not parent.is_dir():
            continue
        seen_roots.add(resolved)
        try:
            children = list(parent.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir() or not child.name.startswith(STAGING_DIR_PREFIX):
                continue
            lock_path = child / STAGING_LOCK_FILE
            try:
                fd = os.open(lock_path, os.O_RDONLY)
            except OSError:
                # 系统临时目录里若没有专属锁文件，绝不越界删除
                continue
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    # 兄弟 rank 正在持有排他锁，跳过
                    continue
                shutil.rmtree(child, ignore_errors=True)
                cleaned.append(child)
            finally:
                try:
                    os.close(fd)
                except OSError:
                    pass
    return tuple(cleaned)


def _delete_corrupt_local_dir(local_root: Path, step: int) -> None:
    """清理损坏的本地热副本目录（R5）：先删 COMMIT 再清数据。"""
    from commit.gc import delete_local_version

    try:
        delete_local_version(local_root, step)
    except OSError:
        shutil.rmtree(local_root / version_key(step), ignore_errors=True)


# --- 单版本远程下载（含 R1 大小核对 + R2 重试 + R3 定向重下 + R4 锁目录） ----


def _load_remote_version(
    storage: StorageBackend,
    *,
    step: int | None = None,
    expected_generation: str | None = None,
    device: "str | Device" = "cpu",
    tp: int = 1,
    tp_rank: int = 0,
    shard_dims: Mapping[str, int] | None = None,
    budget_bytes: int = DEFAULT_BUDGET_BYTES,
    column_strategy: str = "whole",
    verify_crc: bool = True,
    crc_padding: str = "block",
    crc_padding_ratio: float = 0.125,
    io_concurrency: int = 1,
    local_staging_dir: str | Path | None = None,
    skip_corrupt_status: bool = False,
    download_retries: int = DEFAULT_DOWNLOAD_RETRIES,
    download_backoff: float = DEFAULT_DOWNLOAD_BACKOFF,
    run_id: str | None = None,
) -> ObjectLoadReport:
    if io_concurrency < 1:
        raise ValueError(f"io_concurrency must be >= 1, got {io_concurrency}")
    if download_retries < 0:
        raise ValueError(f"download_retries must be >= 0, got {download_retries}")

    version = open_object_version(
        storage,
        step,
        expected_generation=expected_generation,
        skip_corrupt_status=skip_corrupt_status,
        run_id=run_id,
    )

    staging_root, lock_fd = _acquire_staging_dir(local_staging_dir)
    try:
        version_dir = staging_root / version_prefix(version.step)
        version_dir.mkdir(parents=True, exist_ok=True)
        files = _version_files(version.index)
        sizes = _download_files(
            version.storage,
            version,
            files,
            version_dir,
            io_concurrency=io_concurrency,
            retries=download_retries,
            backoff=download_backoff,
        )

        # R1：临时本地 COMMIT 使用各 rank 分片提交点声明的预期大小
        expected_files = {
            name: version.file_sizes.get(name, sizes[name]) for name in files
        }
        local = LocalStorage(staging_root)
        metadata = write_metadata(local, version.step, version.index)
        write_commit(
            local,
            CommitRecord.of_metadata(
                step=version.step,
                metadata=metadata,
                files=expected_files,
            ),
        )

        redownloaded: list[str] = []
        while True:
            try:
                report: LoadReport = load_local(
                    staging_root,
                    step=version.step,
                    device=device,
                    tp=tp,
                    tp_rank=tp_rank,
                    shard_dims=shard_dims,
                    verify_crc=verify_crc,
                    crc_padding=crc_padding,
                    crc_padding_ratio=crc_padding_ratio,
                    budget_bytes=budget_bytes,
                    column_strategy=column_strategy,
                )
                break
            except ChecksumError as exc:
                # R3：本地按块校验失败时定向重下出错文件 1 次；仍坏则定性远端版本损坏
                corrupt_file = _identify_corrupt_file(exc, files)
                if corrupt_file is None or corrupt_file in redownloaded:
                    raise
                redownloaded.append(corrupt_file)
                _download_single_file(
                    version.storage,
                    version,
                    corrupt_file,
                    version_dir / corrupt_file,
                    expected_size=version.file_sizes.get(corrupt_file),
                    retries=download_retries,
                    backoff=download_backoff,
                )

        return ObjectLoadReport(
            step=report.step,
            device=report.device,
            tensors=report.tensors,
            blobs=report.blobs,
            index=report.index,
            requests=report.requests,
            windows=report.windows,
            read_bytes=report.read_bytes,
            copied_bytes=report.copied_bytes,
            h2d_bytes=report.h2d_bytes,
            download_bytes=sum(expected_files.values()),
            local_staging=str(staging_root),
            crc_verified=report.crc_verified,
            verified_blocks=report.verified_blocks,
            io_requests=report.io_requests,
            unverified=report.unverified,
            unverified_bytes=report.unverified_bytes,
            source=SOURCE_OBJECT,
            redownloaded=tuple(redownloaded),
            generation=version.generation,
            run_id=run_id,
        )
    finally:
        _release_staging_dir(staging_root, lock_fd)


def _identify_corrupt_file(exc: ChecksumError, files: Sequence[str]) -> str | None:
    candidate = getattr(exc, "file", None)
    if isinstance(candidate, str) and candidate in files:
        return candidate
    message = str(exc)
    for name in files:
        if name in message:
            return name
    return None


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
    verify_crc: bool = True,
    crc_padding: str = "block",
    crc_padding_ratio: float = 0.125,
    io_concurrency: int = 1,
    local_staging_dir: str | Path | None = None,
    skip_corrupt_status: bool = False,
    allow_fallback: bool = False,
    collective: Collective | None = None,
    world_size: int | None = None,
    download_retries: int = DEFAULT_DOWNLOAD_RETRIES,
    download_backoff: float = DEFAULT_DOWNLOAD_BACKOFF,
    run_id: str | None = None,
    source_run_id: str | None = None,
    promote_tails: bool = False,
    on_target_step: Callable[[int], None] | None = None,
) -> ObjectLoadReport:
    """从对象存储恢复：整文件下载到本地临时目录，再走本地读闭环。

    下载用 :meth:`StorageBackend.download` 流式写盘，不把整个数据文件读进内存；
    下载完核对各 rank 分片提交点预期大小并写合并后的 ``.metadata`` 与本地 ``COMMIT``，
    然后调用 :func:`storage.local_load.load_local` 按索引里的 CRC 块校验。
    """
    return load_checkpoint_version(
        storage,
        local_root=None,
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
        prefer_local=False,
        skip_corrupt_status=skip_corrupt_status,
        allow_fallback=allow_fallback,
        collective=collective,
        world_size=world_size,
        download_retries=download_retries,
        download_backoff=download_backoff,
        run_id=run_id,
        source_run_id=source_run_id,
        promote_tails=promote_tails,
        on_target_step=on_target_step,
    )


# --- 恢复：本地热副本优先，缺了再下载 --------------------------------------

REUSABLE = "reusable"
"""本地热副本可用：远端有版本 ``COMMIT``，本地索引也覆盖请求的布局。"""

NO_LOCAL_COPY = "no_local_copy"
"""本地没有这个版本的 ``step_N/COMMIT``：只能下载。"""

REMOTE_NOT_COMMITTED = "remote_not_committed"
"""远端没有这一代的版本 ``COMMIT``：本地这份可能是唯一副本，不能当热副本用。"""

PARTIAL_LOCAL_COPY = "partial_local_copy"
"""本地索引覆盖不全（多 rank 时每个 rank 只留自己的分片）或缺文件：回落到下载。"""


@dataclass(frozen=True, slots=True)
class LocalCopyCheck:
    """本地热副本能不能顶替一次下载。``bool(check)`` 等于 ``usable``。"""

    step: int | None
    usable: bool
    reason: str
    corrupted: bool = False

    def __bool__(self) -> bool:
        return self.usable


def check_local_copy(
    local_root: str | Path | None,
    remote: StorageBackend | None,
    *,
    step: int | None = None,
    tp: int = 1,
    tp_rank: int = 0,
    shard_dims: Mapping[str, int] | None = None,
    column_strategy: str = "whole",
    target_generation: str | None = None,
    skip_corrupt_status: bool = False,
    run_id: str | None = None,
) -> LocalCopyCheck:
    """判断本地是否留着一份「可以直接读」的热副本。

    条件：
    1. 远端有这一代（同一 step）的 ``STATUS.committed``；
    2. 本地 ``step_N/`` 带 ``COMMIT`` 与 ``GENERATION`` 代际标记文件，且与目标代际一致（R6）；
    3. 本地索引完整且覆盖请求的布局。
    """
    if local_root is None or remote is None:
        return LocalCopyCheck(step, False, NO_LOCAL_COPY)

    scoped_remote = run_scope(remote, run_id)
    local_dir = local_run_root(local_root, run_id)

    if target_generation is None:
        # 兼容单 rank 直接调 local_copy_available 而不传 target_generation 的情况
        if step is None:
            target = latest_remote_step(
                scoped_remote, skip_corrupt_status=skip_corrupt_status
            )
            if target is None:
                return LocalCopyCheck(None, False, REMOTE_NOT_COMMITTED)
        else:
            target = step
        status = read_status(scoped_remote, target)
        if status is None or status.committed is None:
            return LocalCopyCheck(target, False, REMOTE_NOT_COMMITTED)
        target_generation = status.committed.generation.id
    else:
        if step is None:
            raise ValueError("target_generation requires explicit step")
        target = step

    local = LocalStorage(local_dir, direct_io=False)
    try:
        commit_record = read_commit(local, target)
    except FormatError:
        return LocalCopyCheck(target, False, PARTIAL_LOCAL_COPY, corrupted=True)
    if commit_record is None:
        return LocalCopyCheck(target, False, NO_LOCAL_COPY)

    # R6：纯本地比对代际标记文件
    try:
        marker = read_local_generation(local, target)
    except FormatError:
        return LocalCopyCheck(target, False, PARTIAL_LOCAL_COPY, corrupted=True)
    if marker is None or marker.generation != target_generation:
        return LocalCopyCheck(target, False, GENERATION_MISMATCH)
    if run_id is not None and marker.run_id not in (None, run_id):
        return LocalCopyCheck(target, False, GENERATION_MISMATCH)

    try:
        version = recover_version(local, target)
    except (FormatError, ChecksumError, StorageError, OSError):
        return LocalCopyCheck(target, False, PARTIAL_LOCAL_COPY, corrupted=True)
    except PlanError:
        return LocalCopyCheck(target, False, PARTIAL_LOCAL_COPY, corrupted=False)

    targets = targets_for_layout(
        version.index, tp=tp, tp_rank=tp_rank, shard_dims=shard_dims
    )
    cover = build_load_plan(
        version.index, targets, strict=False, column_strategy=column_strategy
    )
    if cover.missing or cover.incomplete:
        return LocalCopyCheck(target, False, PARTIAL_LOCAL_COPY, corrupted=False)
    return LocalCopyCheck(target, True, REUSABLE)


def _normalize_collective(
    collective: Collective | None, world_size: int | None
) -> Collective:
    if collective is None:
        if world_size is not None and world_size > 1:
            raise ValueError(
                f"world_size={world_size} 但没注入 collective：多 rank 恢复必须走集群通信选版与复核"
            )
        return SingleRankCollective()
    if world_size is not None and world_size != collective.world_size:
        raise ValueError(
            f"声明的 world_size={world_size} 与 collective.world_size={collective.world_size} 不一致"
        )
    return collective


def _classify_load_error(exc: BaseException) -> int:
    """将恢复失败编码为 R13 全归约状态码：1=版本自身损坏可回退，2=不可回退致命错误。"""
    if isinstance(
        exc,
        (
            InvalidTargetError,
            ValueError,
            TypeError,
            ConditionalWritesDisabled,
            UnsupportedConditionalWrites,
        ),
    ):
        return LOAD_FATAL
    if isinstance(exc, (FormatError, ChecksumError, PlanError, ObjectNotFoundError)):
        return LOAD_VERSION_CORRUPT
    return LOAD_FATAL


def _discover_candidates_rank0(
    remote: StorageBackend,
    *,
    step: int | None,
    skip_corrupt_status: bool,
    promote_tails: bool,
    run_id: str | None,
) -> dict[str, Any]:
    """rank 0 唯一探查远端：可选提升尾巴 → 生成候选 ``(step, gen)`` 倒序列表。"""
    try:
        scoped = run_scope(remote, run_id)
        if promote_tails:
            promote_remote_tails(scoped)
        if step is not None:
            # 精确指定 step：严格模式，不跳过坏 STATUS，不回退
            status = read_status(scoped, step)
            if status is None or status.committed is None:
                raise FormatError(
                    f"{version_prefix(step)}: STATUS missing or uncommitted, "
                    "this version is not recoverable"
                )
            commit = UploadCommit.from_committed(status.committed)
            return {
                "candidates": [(step, commit.generation)],
                "skipped": [],
                "error": None,
            }

        candidates, skipped = list_remote_candidates(
            scoped, skip_corrupt_status=skip_corrupt_status
        )
        if not candidates:
            raise FormatError(
                f"no committed checkpoint found under {scoped!r}: "
                "只有 STATUS.committed 非空的版本才算可恢复"
            )
        return {
            "candidates": candidates,
            "skipped": list(skipped),
            "error": None,
        }
    except BaseException as exc:
        return {"candidates": [], "skipped": [], "error": exc}


def _try_local_hot_copy(
    local_root: str | Path | None,
    remote: StorageBackend,
    *,
    target_step: int,
    target_gen: str,
    run_id: str | None,
    device: "str | Device",
    tp: int,
    tp_rank: int,
    shard_dims: Mapping[str, int] | None,
    budget_bytes: int,
    column_strategy: str,
    verify_crc: bool,
    crc_padding: str,
    crc_padding_ratio: float,
    fallback_on_local_error: bool,
) -> tuple[ObjectLoadReport | None, str | None]:
    """纯本地比对热副本并尝试加载；返回 ``(report, local_fallback_reason)``。"""
    if local_root is None:
        return None, None

    local_dir = local_run_root(local_root, run_id)
    check = check_local_copy(
        local_root,
        remote,
        step=target_step,
        tp=tp,
        tp_rank=tp_rank,
        shard_dims=shard_dims,
        column_strategy=column_strategy,
        target_generation=target_gen,
        run_id=run_id,
    )
    if not check.usable:
        if check.corrupted:
            msg = (
                f"step_{target_step}: 本地热副本损坏（{check.reason}），"
                + ("清理并回退远端下载" if fallback_on_local_error else "禁止回退")
            )
            logger.warning(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
            if not fallback_on_local_error:
                raise FormatError(msg)
            _delete_corrupt_local_dir(local_dir, target_step)
            return None, f"local_corrupted: {check.reason}"
        return None, check.reason

    try:
        report = load_local(
            local_dir,
            step=target_step,
            device=device,
            tp=tp,
            tp_rank=tp_rank,
            shard_dims=shard_dims,
            verify_crc=verify_crc,
            crc_padding=crc_padding,
            crc_padding_ratio=crc_padding_ratio,
            budget_bytes=budget_bytes,
            column_strategy=column_strategy,
        )
    except (InvalidTargetError, ValueError):
        raise
    except (ChecksumError, FormatError, StorageError, OSError) as exc:
        msg = (
            f"step_{target_step}: 读取本地热副本失败（{type(exc).__name__}: {exc}），"
            + ("清理并回退远端下载" if fallback_on_local_error else "禁止回退")
        )
        logger.warning(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        if not fallback_on_local_error:
            raise
        _delete_corrupt_local_dir(local_dir, target_step)
        return None, f"local_corrupted: {type(exc).__name__}: {exc}"

    return (
        ObjectLoadReport(
            step=report.step,
            device=report.device,
            tensors=report.tensors,
            blobs=report.blobs,
            index=report.index,
            requests=report.requests,
            windows=report.windows,
            read_bytes=report.read_bytes,
            copied_bytes=report.copied_bytes,
            h2d_bytes=report.h2d_bytes,
            download_bytes=0,
            local_staging=None,
            crc_verified=report.crc_verified,
            verified_blocks=report.verified_blocks,
            io_requests=report.io_requests,
            unverified=report.unverified,
            unverified_bytes=report.unverified_bytes,
            source=SOURCE_LOCAL,
            generation=target_gen,
            run_id=run_id,
        ),
        None,
    )


def load_checkpoint_version(
    storage: StorageBackend,
    *,
    local_root: str | Path | None = None,
    step: int | None = None,
    device: "str | Device" = "cpu",
    tp: int = 1,
    tp_rank: int = 0,
    shard_dims: Mapping[str, int] | None = None,
    budget_bytes: int = DEFAULT_BUDGET_BYTES,
    column_strategy: str = "whole",
    verify_crc: bool = True,
    crc_padding: str = "block",
    crc_padding_ratio: float = 0.125,
    io_concurrency: int = 1,
    local_staging_dir: str | Path | None = None,
    prefer_local: bool = True,
    fallback_on_local_error: bool = True,
    skip_corrupt_status: bool = False,
    allow_fallback: bool = False,
    collective: Collective | None = None,
    world_size: int | None = None,
    download_retries: int = DEFAULT_DOWNLOAD_RETRIES,
    download_backoff: float = DEFAULT_DOWNLOAD_BACKOFF,
    run_id: str | None = None,
    source_run_id: str | None = None,
    promote_tails: bool = False,
    on_target_step: Callable[[int], None] | None = None,
) -> ObjectLoadReport:
    """统一的集群/单机恢复闭环（R5 + R6 + R7 + R8 + R9 + R13 + R14 + C4）。

    三步闭环（单 rank 经 ``SingleRankCollective`` 走同一条代码路径）：

    1. **选版对齐**：仅 rank 0 探查远端（可选尾巴提升），生成候选 ``(step, gen)``
       倒序列表并 ``broadcast_object`` 给全体；非 0 rank 不主动探查远端。
    2. **热副本比对与加载**：各 rank 用锁定代际做纯本地 ``GENERATION`` 比对；本地命中
       且健康直接零下载读取，本地损坏默认告警清理并回退远端下载。
    3. **全归约复核与统一回退**：全员编码 ``0(OK) / 1(版本坏) / 2(致命错)`` 并做一次
       ``all_reduce(MAX)`` —— 全员放行、全员报错、或（显式开启且未指定 step 时）
       全员统一回退到下一个候选版本。
    """
    peer = _normalize_collective(collective, world_size)
    effective_run_id = source_run_id if source_run_id is not None else run_id
    if effective_run_id is not None:
        validate_run_id(effective_run_id)
    if run_id is not None:
        validate_run_id(run_id)

    # 第 1 步：选版广播（只有 rank 0 探查远端）
    discovery = (
        _discover_candidates_rank0(
            storage,
            step=step,
            skip_corrupt_status=skip_corrupt_status,
            promote_tails=promote_tails,
            run_id=effective_run_id,
        )
        if peer.rank == 0
        else None
    )
    discovery = peer.broadcast_object(discovery, root=0)
    if discovery["error"] is not None:
        raise discovery["error"]

    candidates: list[tuple[int, str]] = [
        (int(c[0]), str(c[1])) for c in discovery["candidates"]
    ]
    skipped_versions: list[tuple[int, str]] = [
        (int(s[0]), str(s[1])) for s in discovery["skipped"]
    ]
    broadcast_protect = getattr(peer, "broadcast_protect", None)

    # 第 2 步 & 第 3 步：尝试候选版本 → 加载后 0/1/2 全归约复核 → 统一回退或退出
    for idx, (cand_step, cand_gen) in enumerate(candidates):
        if on_target_step is not None:
            on_target_step(cand_step)
        if callable(broadcast_protect) and peer.rank == 0:
            broadcast_protect(add_steps=[cand_step], pause_gc=True)

        report: ObjectLoadReport | None = None
        local_exc: BaseException | None = None
        local_code = LOAD_OK
        try:
            try:
                fallback_reason: str | None = None
                if prefer_local and local_root is not None:
                    report, fallback_reason = _try_local_hot_copy(
                        local_root,
                        storage,
                        target_step=cand_step,
                        target_gen=cand_gen,
                        run_id=effective_run_id,
                        device=device,
                        tp=tp,
                        tp_rank=tp_rank,
                        shard_dims=shard_dims,
                        budget_bytes=budget_bytes,
                        column_strategy=column_strategy,
                        verify_crc=verify_crc,
                        crc_padding=crc_padding,
                        crc_padding_ratio=crc_padding_ratio,
                        fallback_on_local_error=fallback_on_local_error,
                    )
                if report is None:
                    report = _load_remote_version(
                        storage,
                        step=cand_step,
                        expected_generation=cand_gen,
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
                        download_retries=download_retries,
                        download_backoff=download_backoff,
                        run_id=effective_run_id,
                    )
                    if fallback_reason is not None:
                        report = replace(report, local_fallback_reason=fallback_reason)
            except BaseException as exc:
                local_exc = exc
                if not fallback_on_local_error and "本地热副本" in str(exc):
                    local_code = LOAD_FATAL
                else:
                    local_code = _classify_load_error(exc)

            global_code = int(peer.all_reduce_max_int64([local_code])[0])
        finally:
            if callable(broadcast_protect) and peer.rank == 0:
                broadcast_protect(release_steps=[cand_step], pause_gc=False)

        if global_code == LOAD_OK:
            assert report is not None
            return replace(report, skipped_versions=tuple(skipped_versions))

        if global_code == LOAD_FATAL:
            if local_exc is not None:
                raise local_exc
            raise StorageError(
                f"step_{cand_step} ({cand_gen}): 其他 rank 遇到不可回退的致命恢复错误"
            )

        # global_code == LOAD_VERSION_CORRUPT
        has_next = idx + 1 < len(candidates)
        if allow_fallback and step is None and has_next:
            reason = (
                f"{type(local_exc).__name__}: {local_exc}"
                if local_exc is not None
                else "peer rank failed version integrity check"
            )
            skipped_versions.append((cand_step, reason))
            # 释放成功 rank 已加载的张量视图引用后再切到下一个候选版本
            report = None
            continue

        if local_exc is not None:
            raise local_exc
        raise FormatError(
            f"step_{cand_step} ({cand_gen}): 其他 rank 检测到版本损坏，全员一致放弃恢复"
        )

    raise FormatError("no recoverable checkpoint candidate found")


def _version_files(index: CheckpointIndex) -> list[str]:
    """版本里需要下载的全部数据文件（分片与小状态可能共用一个文件）。"""
    files = {
        shard.file for meta in index.tensors.values() for shard in meta.shards
    }
    files.update(blob.file for blob in index.blobs.values())
    return sorted(files)


def _download_single_file(
    storage: StorageBackend,
    version: ObjectVersion,
    name: str,
    target: Path,
    *,
    expected_size: int | None,
    retries: int,
    backoff: float,
) -> int:
    """下载单个远端数据文件：临时文件落盘 + R1 大小核对 + R2 有限次指数退避重试。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    object_name = version.file_key(name)
    size_attempts = 0
    attempt = 0
    while True:
        tmp_path = target.parent / f".{target.name}.dl.{uuid.uuid4().hex}"
        try:
            measured = storage.download(object_name, tmp_path)
            actual = tmp_path.stat().st_size if tmp_path.exists() else measured
            if expected_size is not None and actual != expected_size:
                tmp_path.unlink(missing_ok=True)
                if size_attempts < 1:
                    size_attempts += 1
                    continue
                raise FormatError(
                    f"{object_name}: downloaded {actual} bytes, "
                    f"rank COMMIT expects {expected_size}"
                )
            os.replace(tmp_path, target)
            return actual
        except BaseException as exc:
            tmp_path.unlink(missing_ok=True)
            if attempt < retries and is_retryable_storage_error(exc):
                if backoff > 0:
                    time.sleep(backoff * (2**attempt))
                attempt += 1
                continue
            raise


def _download_files(
    storage: StorageBackend,
    version: ObjectVersion,
    files: Sequence[str],
    version_dir: Path,
    *,
    io_concurrency: int,
    retries: int = DEFAULT_DOWNLOAD_RETRIES,
    backoff: float = DEFAULT_DOWNLOAD_BACKOFF,
) -> dict[str, int]:
    """把版本数据文件下载到本地目录，返回「文件 -> 字节数」。

    R2：单文件重试不连累已成功的并发文件；遇不可重试错误整批快速失败。
    """

    def fetch(name: str) -> tuple[str, int]:
        target = version_dir / name
        size = _download_single_file(
            storage,
            version,
            name,
            target,
            expected_size=version.file_sizes.get(name),
            retries=retries,
            backoff=backoff,
        )
        return name, size

    if io_concurrency <= 1 or len(files) <= 1:
        return dict(fetch(name) for name in files)

    results: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=io_concurrency) as pool:
        future_to_name = {pool.submit(fetch, name): name for name in files}
        pending = set(future_to_name)
        while pending:
            done, pending = wait(pending, return_when=FIRST_EXCEPTION)
            for fut in done:
                exc = fut.exception()
                if exc is not None:
                    for p in pending:
                        p.cancel()
                    raise exc
                name, size = fut.result()
                results[name] = size
    return results


__all__ = [
    "DEFAULT_BUDGET_BYTES",
    "DEFAULT_DOWNLOAD_BACKOFF",
    "DEFAULT_DOWNLOAD_RETRIES",
    "DEFAULT_KEEP_VERSIONS",
    "GENERATION_MISMATCH",
    "LOAD_FATAL",
    "LOAD_OK",
    "LOAD_VERSION_CORRUPT",
    "LocalCopyCheck",
    "NO_LOCAL_COPY",
    "ObjectLoadReport",
    "ObjectVersion",
    "PARTIAL_LOCAL_COPY",
    "RANK_COMMIT_FORMAT_VERSION",
    "RankCommit",
    "REMOTE_NOT_COMMITTED",
    "REUSABLE",
    "SOURCE_LOCAL",
    "SOURCE_OBJECT",
    "STAGING_DIR_PREFIX",
    "STAGING_LOCK_FILE",
    "UploadCommit",
    "check_local_copy",
    "cleanup_stale_staging_dirs",
    "committed_remote_steps",
    "data_key",
    "finalize_upload",
    "generation_prefix",
    "index_key",
    "latest_remote_step",
    "list_remote_candidates",
    "load_checkpoint_version",
    "load_object_version",
    "local_run_root",
    "open_object_version",
    "promote_remote_tails",
    "rank_commit_key",
    "rank_prefix",
    "status_key",
    "upload_rank",
    "version_prefix",
]
