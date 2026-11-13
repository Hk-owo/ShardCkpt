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

import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

from async_ import Tensor
from commit.gc import DEFAULT_KEEP_VERSIONS, retain_remote_versions
from commit.generation import (
    GENERATION_FORMAT_VERSION,
    SUPPORTED_FORMAT_VERSIONS,
    FencedError,
    Generation,
    as_generation,
    mark_committed,
    resolve_generation,
    seal_generation,
    verify_holder,
)
from commit.local import (
    COMMIT_FILE,
    CommitRecord,
    check_crc32c,
    latest_step,
    read_commit,
    write_commit,
    write_metadata,
)
from core import ChecksumError, FormatError, PlanError
from core.index import META_FILE, BlobMeta, CheckpointIndex, ShardMeta, TensorMeta
from core.metadata import Device
from planner.load_plan import build_load_plan
from planner.targets import targets_for_layout
from storage.base import (
    ConditionalBackend,
    ObjectNotFoundError,
    StorageBackend,
    StorageError,
)
from storage.checksum import crc32c_hex
from storage.local import LocalStorage
from storage.local_load import LoadReport, load_local
from storage.recover import recover_version

COMMIT_FORMAT_VERSION = 2
"""v2 起用 ``step_N/<gen>/`` 布局，且 L2 的 ``.metadata`` 记 CRC-32C。

读侧仍接受 v1（``generation=None``、没有 ``metadata.crc32c``，跳过校验）。
"""

DEFAULT_BUDGET_BYTES = 8 << 20


# --- key 布局 ---------------------------------------------------------------


def version_prefix(step: int) -> str:
    return f"step_{step}"


def generation_prefix(
    step: int, generation: "Generation | str | None" = None
) -> str:
    """``step_42/gen_0003-1f3a9c2b``；``generation=None`` 是 v1 老布局。"""
    base = version_prefix(step)
    if generation is None:
        return base
    gid = generation.id if isinstance(generation, Generation) else str(generation)
    return f"{base}/{gid}"


def rank_prefix(
    step: int, rank: int, generation: "Generation | str | None" = None
) -> str:
    return f"{generation_prefix(step, generation)}/rank_{rank}"


def data_key(
    step: int, rank: int, file: str, generation: "Generation | str | None" = None
) -> str:
    return f"{rank_prefix(step, rank, generation)}/{file}"


def index_key(
    step: int, rank: int, generation: "Generation | str | None" = None
) -> str:
    return f"{rank_prefix(step, rank, generation)}/{META_FILE}"


def rank_commit_key(
    step: int, rank: int, generation: "Generation | str | None" = None
) -> str:
    return f"{rank_prefix(step, rank, generation)}/{COMMIT_FILE}"


def commit_key(step: int) -> str:
    """版本提交点（L3）**不在代际前缀里**：它是唯一权威，重存时被 CAS 换掉。"""
    return f"{version_prefix(step)}/{COMMIT_FILE}"


# --- 提交记录 ---------------------------------------------------------------


@dataclass(frozen=True)
class UploadCommit:
    """rank 0 写的版本提交点（L3）。

    它不记录文件清单：rank 0 没有 gather 任何人的索引 —— 「齐不齐」由每个 rank 的
    :class:`RankCommit` 证明，「分片怎么拼」由各 rank 自己的 ``.metadata`` 证明。

    ``generation`` + ``seq`` 是这一版属于哪一代的**唯一权威**：恢复端只按它组装，
    不读 GEN（GEN 被删被改，已提交版本照样可恢复）。``etag`` 只在读出来时填，
    是下一次 CAS 的 ``If-Match`` 值，不参与序列化。
    """

    step: int
    ranks: tuple[int, ...]
    layout: Mapping[str, int] = field(default_factory=dict)
    generation: str | None = None
    seq: int = 0
    format_version: int = COMMIT_FORMAT_VERSION
    etag: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranks", tuple(int(rank) for rank in self.ranks))
        object.__setattr__(self, "layout", dict(self.layout or {}))
        if self.format_version not in SUPPORTED_FORMAT_VERSIONS:
            raise FormatError(
                f"unsupported upload COMMIT format_version {self.format_version}, "
                f"this build understands {sorted(SUPPORTED_FORMAT_VERSIONS)}"
            )
        if self.step < 0:
            raise FormatError(f"step must be >= 0, got {self.step}")
        if not self.ranks:
            raise FormatError("COMMIT must declare at least one rank")
        if len(set(self.ranks)) != len(self.ranks):
            raise FormatError(f"duplicate ranks in COMMIT: {self.ranks}")
        if any(rank < 0 for rank in self.ranks):
            raise FormatError(f"ranks must be >= 0, got {self.ranks}")
        if self.generation is not None:
            parsed = Generation.parse(self.generation, step=self.step)
            if parsed.seq != self.seq:
                raise FormatError(
                    f"COMMIT 的 generation {self.generation} 与 seq {self.seq} 不一致"
                )
        elif self.seq:
            raise FormatError(f"COMMIT 没有 generation 却带 seq={self.seq}（v1 布局没有代际）")

    def dumps(self) -> str:
        data: dict[str, object] = {
            "format_version": self.format_version,
            "step": self.step,
            "ranks": list(self.ranks),
            "layout": {key: int(value) for key, value in sorted(self.layout.items())},
        }
        if self.generation is not None:
            data["generation"] = self.generation
            data["seq"] = self.seq
        return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)

    @classmethod
    def loads(cls, text: str, *, etag: str = "") -> "UploadCommit":
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
        generation = data.get("generation")
        if generation is not None:
            generation = str(generation)
        try:
            return cls(
                step=int(data["step"]),
                ranks=tuple(int(rank) for rank in ranks),
                layout={str(key): int(value) for key, value in layout.items()},
                generation=generation,
                seq=int(data.get("seq", 0)),
                format_version=int(data.get("format_version", COMMIT_FORMAT_VERSION)),
                etag=etag,
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

    ``metadata_crc32c`` 是**本 rank** 那份 ``.metadata`` 的字节级摘要，算法与 L1 的
    ``COMMIT`` 一致（CRC-32C，8 位小写十六进制）。恢复端的合并索引是 N 份 ``.metadata``
    拼出来的，任何一份被改坏都会让「分片怎么拼」出错，而这在合并后的
    ``CheckpointIndex.validate()`` 里看不出来（结构自洽、指向别处的索引照样通过）。
    v1 老布局没有这个字段（``None``）：读侧跳过校验，写侧总是写。
    """

    step: int
    rank: int
    metadata_bytes: int = 0
    metadata_crc32c: str | None = None
    files: Mapping[str, int] = field(default_factory=dict)
    generation: str | None = None
    seq: int = 0
    format_version: int = COMMIT_FORMAT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", {str(k): int(v) for k, v in self.files.items()})
        if self.format_version not in SUPPORTED_FORMAT_VERSIONS:
            raise FormatError(
                f"unsupported rank COMMIT format_version {self.format_version}, "
                f"this build understands {sorted(SUPPORTED_FORMAT_VERSIONS)}"
            )
        if self.step < 0:
            raise FormatError(f"step must be >= 0, got {self.step}")
        if self.rank < 0:
            raise FormatError(f"rank must be >= 0, got {self.rank}")
        if self.metadata_bytes < 0:
            raise FormatError(f"metadata bytes must be >= 0, got {self.metadata_bytes}")
        if self.metadata_crc32c is None:
            if self.format_version >= GENERATION_FORMAT_VERSION:
                raise FormatError(
                    "rank COMMIT v2 must record 'metadata.crc32c' (index checksum)"
                )
        else:
            check_crc32c(self.metadata_crc32c, "rank COMMIT metadata.crc32c")
        if self.generation is not None:
            parsed = Generation.parse(self.generation, step=self.step)
            if parsed.seq != self.seq:
                raise FormatError(
                    f"rank COMMIT 的 generation {self.generation} 与 seq {self.seq} 不一致"
                )

    def dumps(self) -> str:
        metadata: dict[str, object] = {"bytes": self.metadata_bytes}
        if self.metadata_crc32c is not None:
            metadata["crc32c"] = self.metadata_crc32c
        data: dict[str, object] = {
            "format_version": self.format_version,
            "step": self.step,
            "rank": self.rank,
            "metadata": metadata,
            "files": {name: int(size) for name, size in sorted(self.files.items())},
        }
        if self.generation is not None:
            data["generation"] = self.generation
            data["seq"] = self.seq
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
                format_version=int(data.get("format_version", COMMIT_FORMAT_VERSION)),
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
) -> int:
    """把本地 ``step_<step>`` 版本里属于本 rank 的产物上传到对象存储。

    返回上传的字节数（数据 + 索引 + 提交记录）。顺序：数据文件 → ``.metadata`` →
    ``rank_<rank>/COMMIT``；提交记录落盘之前，rank 0 的 ``finalize_upload`` 不会认为
    这个 rank 完成，本地回收也不会动这一版。

    所有 key 都落在 ``step_<step>/<generation>/rank_<rank>/`` 下，写入用
    ``If-None-Match: *``（同代同 key 不会被静默覆盖）。**写 L2 之前**做一次 fence
    检查：若自己已不是当前代际，直接 ``FencedError`` 且不写 L2 —— 这一代因此永远
    凑不齐，rank 0 不会把它提升为版本。
    """
    local = LocalStorage(root, direct_io=False)
    key = version_prefix(step)
    if read_commit(local, step) is None:
        raise FormatError(
            f"{key}: 本地版本没有 COMMIT，先 save_local() 提交再上传"
        )

    conditional = _as_conditional(storage, what=f"upload_rank(step={step}, rank={rank})")
    gen = as_generation(generation, step=step)
    if gen is None:
        # 单 rank 便利路径：GEN 已有就复用它，否则开第一代。多 rank 必须先
        # begin_upload() 分配，否则两次尝试的分片会混在同一代里。
        gen = resolve_generation(storage, step, ranks=(rank,))

    index = CheckpointIndex.loads(local.read(f"{key}/{META_FILE}").decode("utf-8"))
    index.validate(complete=False)  # 单 rank 只做局部自检，全局覆盖在恢复端

    # L2-pre：花带宽之前先证明我还持有这一代
    verify_holder(storage, gen)

    files = sorted({shard.file for meta in index.tensors.values() for shard in meta.shards})
    files += sorted({blob.file for blob in index.blobs.values()})
    uploaded = 0
    sizes: dict[str, int] = {}
    for file in files:
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

    metadata = local.read(f"{key}/{META_FILE}")
    object_name = index_key(step, rank, gen)
    outcome = conditional.put_if_absent(object_name, metadata)
    if not outcome.written:
        _reuse_uploaded_object(conditional, object_name, len(metadata))
    uploaded += len(metadata)

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


def _read_upload_commit(storage: StorageBackend, step: int) -> UploadCommit | None:
    """读 L3（带 ETag）；不存在返回 ``None``。"""
    key = commit_key(step)
    try:
        payload = storage.read(key)
    except ObjectNotFoundError:
        return None
    etag = ""
    head = getattr(storage, "head", None)
    if callable(head):
        meta = head(key)
        if meta is not None:
            etag = meta.etag
    return UploadCommit.loads(payload.decode("utf-8"), etag=etag)


def _write_upload_commit(
    storage: StorageBackend,
    commit: UploadCommit,
    *,
    existing: UploadCommit | None,
) -> UploadCommit:
    """用 CAS 写 L3：首写 ``If-None-Match``，重存 ``If-Match``。

    这是 L3 的 fence：两个不同 ``seq`` 的 finalize 并发时，只有一个能换掉 L3；
    输的那个若发现 L3 的 ``seq`` 更大，立刻 ``FencedError``，不会把 COMMIT 指回
    自己那一代。
    """
    conditional = _as_conditional(storage, what=f"finalize_upload(step={commit.step})")
    key = commit_key(commit.step)
    payload = commit.dumps().encode("utf-8")
    current = existing
    for _ in range(2):
        if current is None:
            outcome = conditional.put_if_absent(key, payload)
        else:
            if not current.etag:
                raise StorageError(f"{key}: 读到的 L3 没有 ETag，无法用 If-Match 重存")
            outcome = conditional.put_if_match(key, payload, current.etag)
        if outcome.written:
            return commit
        reread = _read_upload_commit(storage, commit.step)
        if reread is not None and reread.seq > commit.seq:
            raise FencedError(
                f"step {commit.step}: L3 已被 seq={reread.seq}"
                f"（{reread.generation}）提交，{commit.generation} 已被 fence"
            )
        current = reread
    raise FencedError(
        f"step {commit.step}: L3 的 CAS 连续失败，{commit.generation} 放弃提交"
    )


def finalize_upload(
    storage: StorageBackend,
    step: int,
    *,
    ranks: Sequence[int],
    generation: "Generation | str | None" = None,
    layout: Mapping[str, int] | None = None,
    keep: int | None = DEFAULT_KEEP_VERSIONS,
) -> UploadCommit:
    """rank 0 的提交动作：``list`` 本代前缀，确认每个 rank 都传完了，再 CAS 写 ``COMMIT``。

    ``ranks`` 是**期望的写者清单**（训练配置里的 world size），由 rank 0 自己知道，
    不需要从别处汇总。任何缺 ``.metadata`` / ``rank_<rank>/COMMIT`` 的 rank 都会让
    这一步失败，失败时不写版本 ``COMMIT``，因此这一版对恢复端不可见。

    **代际与 fence**：先 ``seal_generation`` 用 CAS 把 GEN 推到 ``sealed``（这一代
    不再接受新写者），再用 CAS 写 L3。已经提交过的更高 ``seq`` 会让本次提交直接
    ``FencedError`` —— 旧代永远接管不了 L3。

    **幂等可重跑**：判据只看每个 rank 的分片提交点在不在。rank 0 崩在写完
    ``COMMIT`` 之前时，重启后再调一次（或运维当 CLI 调）就能把这一版补成可恢复的；
    L3 的 ``seq`` 与本次代际相同时直接返回，**不产生额外写入**。

    **有界保留**：``COMMIT`` 落盘之后，同一个 rank 0 立刻做一次对象存储侧的保留 ——
    只留最新 ``keep`` 代版本（默认 :data:`commit.gc.DEFAULT_KEEP_VERSIONS`，即 3 代），
    从最旧的一代开始删，先删版本 ``COMMIT`` 再删其余数据。``keep=None`` 关掉这一步
    （交给运维手工调 :func:`commit.gc.retain_remote_versions`）。

    保留失败（远端抖动、删到一半）**不影响这次提交**：版本 ``COMMIT`` 已经写完，这一版
    就是可恢复的；残留的旧代会在下一次提交成功后被同一条判据收掉。
    """
    expected = tuple(int(rank) for rank in ranks)
    gen = as_generation(generation, step=step)
    if gen is None:
        gen = resolve_generation(storage, step, ranks=expected)

    existing = _read_upload_commit(storage, step)
    if existing is not None:
        if existing.seq > gen.seq:
            raise FencedError(
                f"step {step}: L3 已被 seq={existing.seq}（{existing.generation}）提交，"
                f"{gen.id} 已被 fence"
            )
        if existing.seq == gen.seq:
            if existing.generation != gen.id:
                raise FencedError(
                    f"step {step}: L3 声明的 seq={existing.seq} 属于 "
                    f"{existing.generation}，不是 {gen.id}（同序号不同写者）"
                )
            if tuple(existing.ranks) != expected:
                raise FormatError(
                    f"step {step}: 已提交的 L3 声明 ranks {existing.ranks}，"
                    f"本次传入 {expected}"
                )
            return existing

    seal_generation(storage, gen, ranks=expected)

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
            f"COMMIT declares ranks {expected}"
        )

    commit = UploadCommit(
        step=step,
        ranks=expected,
        layout=dict(layout or {}),
        generation=gen.id,
        seq=gen.seq,
    )
    commit = _write_upload_commit(storage, commit, existing=existing)
    mark_committed(storage, gen)
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
    generation: str | None = None
    """这一版属于哪一代；``None`` 是 v1 老布局（``step_N/rank_r/`` 直下）。"""

    def file_key(self, name: str) -> str:
        """索引里的 ``file`` 是「版本目录内、带 rank 前缀」的名字。"""
        base = self.key if self.generation is None else f"{self.key}/{self.generation}"
        return f"{base}/{name}"

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

    base = generation_prefix(target, commit.generation)
    keys = set(storage.list(base))
    missing = sorted(
        index_key(target, rank, commit.generation)
        for rank in commit.ranks
        if index_key(target, rank, commit.generation) not in keys
    )
    if missing:
        raise FormatError(
            f"{key}: COMMIT declares ranks {commit.ranks}, "
            f"but these index objects are missing: {missing}"
        )

    # L2 的分片提交点在这里既是「诊断证据」也是**混代检测点**：它不在而 L3 声明了
    # 这个 rank，说明对象被外部删过；它在但声明的 generation 与 L3 不一致，说明这一版
    # 是两次尝试各传一半拼出来的 —— 两种都在读第一个字节之前报出来。
    not_committed = sorted(
        rank
        for rank in commit.ranks
        if rank_commit_key(target, rank, commit.generation) not in keys
    )
    if not_committed:
        raise FormatError(
            f"{key}: COMMIT declares ranks {commit.ranks}, but rank(s) {not_committed} "
            f"have no per-rank COMMIT (分片提交点缺失，远端对象可能被删过)"
        )
    index_payloads: dict[int, bytes] = {}
    for rank in commit.ranks:
        rank_commit = RankCommit.loads(
            storage.read(rank_commit_key(target, rank, commit.generation)).decode("utf-8")
        )
        if rank_commit.rank != rank:
            raise FormatError(
                f"{base}/rank_{rank}/COMMIT 声明 rank {rank_commit.rank}，路径与内容不一致"
            )
        if rank_commit.generation != commit.generation:
            raise FormatError(
                f"{base}: rank {rank} 的分片提交点声明 generation "
                f"{rank_commit.generation!r}，版本 COMMIT 声明 {commit.generation!r} —— "
                "这一版混了两次尝试的分片，拒绝恢复"
            )

        # 合并索引是 N 份 ``.metadata`` 拼出来的：任何一份被改坏都会让「分片怎么拼」
        # 出错，而合并后的 ``validate()`` 看不出结构自洽的篡改 —— 长度 + CRC-32C 才是
        # 逐 rank 的字节级凭据（v1 老布局没有摘要，只能靠结构校验）。
        rank_index_payload = storage.read(index_key(target, rank, commit.generation))
        # 长度只在 L2 真记了字节数时校验：v1 的 L2 可能根本没记（缺省 0）
        if rank_commit.metadata_bytes and len(rank_index_payload) != rank_commit.metadata_bytes:
            raise FormatError(
                f"{base}/rank_{rank}/{META_FILE}: {len(rank_index_payload)} bytes, "
                f"rank COMMIT records {rank_commit.metadata_bytes}"
            )
        if rank_commit.metadata_crc32c is not None:
            actual = crc32c_hex(rank_index_payload)
            if actual != rank_commit.metadata_crc32c:
                raise ChecksumError(
                    f"{base}/rank_{rank}/{META_FILE}: crc32c {actual}, rank COMMIT "
                    f"records {rank_commit.metadata_crc32c}（分片索引与提交点不符）"
                )
        index_payloads[rank] = rank_index_payload

    shards: dict[str, list[ShardMeta]] = {}
    metas: dict[str, TensorMeta] = {}
    blobs: dict[str, BlobMeta] = {}
    for rank in commit.ranks:
        rank_index = CheckpointIndex.loads(index_payloads[rank].decode("utf-8"))
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
        storage=storage,
        step=target,
        key=key,
        index=index,
        commit=commit,
        generation=commit.generation,
    )


# --- 读取：本地热副本优先，缺了再下载 --------------------------------------

SOURCE_LOCAL = "local"
"""恢复来源：本地保留的热副本（零下载）。"""

SOURCE_OBJECT = "object"
"""恢复来源：从对象存储整份下载到临时目录。"""


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
    verify_crc: bool = True,
    crc_padding: str = "block",
    crc_padding_ratio: float = 0.125,
    io_concurrency: int = 1,
    local_staging_dir: str | Path | None = None,
) -> ObjectLoadReport:
    """从对象存储恢复：整文件下载到本地临时目录，再走本地读闭环。

    下载用 :meth:`StorageBackend.download` 流式写盘，不把整个数据文件读进内存；
    下载完写合并后的 ``.metadata`` 与本地 ``COMMIT``，然后调用
    :func:`storage.local_load.load_local` 按索引里的 CRC 块校验。

    对象存储侧只负责「发回来的 = 存的」（整对象 GET 带 ``ChecksumMode=ENABLED``），
    本地 ``load_local`` 负责「存的 = 当初写的」。
    """
    version = open_object_version(storage, step)
    if io_concurrency < 1:
        raise ValueError(f"io_concurrency must be >= 1, got {io_concurrency}")

    staging_root = tempfile.mkdtemp(prefix="shardckpt-", dir=local_staging_dir)
    try:
        version_dir = Path(staging_root) / version_prefix(version.step)
        version_dir.mkdir(parents=True, exist_ok=True)
        files = _version_files(version.index)
        sizes = _download_files(
            storage,
            version,
            files,
            version_dir,
            io_concurrency=io_concurrency,
        )

        local = LocalStorage(staging_root)
        metadata = write_metadata(local, version.step, version.index)
        write_commit(
            local,
            CommitRecord.of_metadata(
                step=version.step,
                metadata=metadata,
                files=sizes,
            ),
        )

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
            download_bytes=sum(sizes.values()),
            local_staging=staging_root,
            crc_verified=report.crc_verified,
            verified_blocks=report.verified_blocks,
            io_requests=report.io_requests,
            unverified=report.unverified,
            unverified_bytes=report.unverified_bytes,
        )
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


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
) -> LocalCopyCheck:
    """判断本地是否留着一份「可以直接读」的热副本。

    两个条件缺一不可：

    1. **远端有这一代（同一 step）的版本 ``COMMIT``（L3）**：本地这份不是唯一副本，
       读它是安全的，删了也能重下；
    2. **本地 ``step_N/`` 带 ``COMMIT`` 且索引覆盖请求的布局**：多 rank 时每个 rank
       的本地目录只有自己的分片，覆盖不全就必须回落下载。

    这只是**判定**，不读数据、不校验 CRC。CRC 是 :func:`storage.local_load.load_local`
    在读的时候做的 —— 本地介质真坏了应该报 :class:`core.ChecksumError`，而不是悄悄
    重下掩盖问题；要强制走下载传 ``prefer_local=False``。

    ``step=None`` 取远端最新的已提交版本。读远端 L3 失败（``StorageError``）会往上抛：
    远端不可达时下载同样不可用，这里不把它伪装成「本地没有热副本」。
    """
    if local_root is None or remote is None:
        return LocalCopyCheck(step, False, NO_LOCAL_COPY)

    target = step if step is not None else latest_step(remote)
    if target is None:
        # 远端一个已提交版本都没有：调用方的下一步（下载）会给出同样的错误。
        return LocalCopyCheck(None, False, REMOTE_NOT_COMMITTED)

    local = LocalStorage(local_root, direct_io=False)
    if read_commit(local, target) is None:
        return LocalCopyCheck(target, False, NO_LOCAL_COPY)
    # 远端 L3 是 UploadCommit 格式（记 ranks），不用 CommitRecord 解析，只确认存在。
    try:
        remote.read(commit_key(target))
    except ObjectNotFoundError:
        return LocalCopyCheck(target, False, REMOTE_NOT_COMMITTED)

    try:
        version = recover_version(local, target)
    except (FormatError, PlanError):
        # 本地这个版本读不出来（索引缺/坏、COMMIT 对不上、只有本 rank 的分片所以
        # 全局覆盖校验不通过）：远端有完整版本，回落到下载。
        return LocalCopyCheck(target, False, PARTIAL_LOCAL_COPY)

    targets = targets_for_layout(
        version.index, tp=tp, tp_rank=tp_rank, shard_dims=shard_dims
    )
    cover = build_load_plan(
        version.index, targets, strict=False, column_strategy=column_strategy
    )
    if cover.missing or cover.incomplete:
        return LocalCopyCheck(target, False, PARTIAL_LOCAL_COPY)
    return LocalCopyCheck(target, True, REUSABLE)


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
) -> ObjectLoadReport:
    """恢复一个版本：**本地热副本优先**，本地不满足条件再走下载。

    ``local_root`` 是本地层 root（``step_N/`` 的父目录）。判定见
    :func:`check_local_copy`：远端有这一代的版本 ``COMMIT`` + 本地目录带 ``COMMIT``
    + 本地索引覆盖请求的布局，三者齐了就直接读本地（``report.source == "local"``、
    ``download_bytes == 0``）；否则与 :func:`load_object_version` 完全一样地下载。

    ``prefer_local=False`` 强制走下载。``local_root=None`` 等价于旧的
    ``load_object_version``：只从对象存储恢复。
    """
    if prefer_local:
        check = check_local_copy(
            local_root,
            storage,
            step=step,
            tp=tp,
            tp_rank=tp_rank,
            shard_dims=shard_dims,
            column_strategy=column_strategy,
        )
        if check.usable and check.step is not None and local_root is not None:
            report = load_local(
                local_root,
                step=check.step,
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
                download_bytes=0,
                local_staging=None,
                crc_verified=report.crc_verified,
                verified_blocks=report.verified_blocks,
                io_requests=report.io_requests,
                unverified=report.unverified,
                unverified_bytes=report.unverified_bytes,
                source=SOURCE_LOCAL,
            )

    return replace(
        load_object_version(
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
        ),
        source=SOURCE_OBJECT,
    )


def _version_files(index: CheckpointIndex) -> list[str]:
    """版本里需要下载的全部数据文件（分片与小状态可能共用一个文件）。"""
    files = {
        shard.file for meta in index.tensors.values() for shard in meta.shards
    }
    files.update(blob.file for blob in index.blobs.values())
    return sorted(files)


def _download_files(
    storage: StorageBackend,
    version: ObjectVersion,
    files: Sequence[str],
    version_dir: Path,
    *,
    io_concurrency: int,
) -> dict[str, int]:
    """把版本数据文件下载到本地目录，返回「文件 -> 字节数」。"""

    def fetch(name: str) -> tuple[str, int]:
        target = version_dir / name
        size = storage.download(version.file_key(name), target)
        return name, size

    if io_concurrency <= 1:
        return dict(fetch(name) for name in files)
    with ThreadPoolExecutor(max_workers=io_concurrency) as pool:
        return dict(pool.map(fetch, files))


__all__ = [
    "DEFAULT_BUDGET_BYTES",
    "DEFAULT_KEEP_VERSIONS",
    "LocalCopyCheck",
    "NO_LOCAL_COPY",
    "ObjectLoadReport",
    "ObjectVersion",
    "PARTIAL_LOCAL_COPY",
    "RankCommit",
    "REMOTE_NOT_COMMITTED",
    "REUSABLE",
    "SOURCE_LOCAL",
    "SOURCE_OBJECT",
    "UploadCommit",
    "check_local_copy",
    "commit_key",
    "data_key",
    "finalize_upload",
    "index_key",
    "load_checkpoint_version",
    "load_object_version",
    "open_object_version",
    "rank_commit_key",
    "rank_prefix",
    "upload_rank",
    "version_prefix",
]
