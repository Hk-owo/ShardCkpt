"""本地层垃圾回收：判据是纯函数，执行器在下面（:func:`collect`）。

回收规则只有一条不变量：**step 是单调推进的**。由此得到两条判据，都不需要租约、
时间戳或尝试号：

1. 远端的 ``step_N/rank_<r>/COMMIT`` 存在 ⇒ 本 rank 的分片已经在远端 ⇒ 本地
   ``step_N/`` 是纯冗余，可以删；
2. 存在比 N 更新的**版本** ``COMMIT`` ⇒ 训练已经越过第 N 代 ⇒ 本地 ``step_N/``
   （无论有没有本地 ``COMMIT``）与远端整个 ``step_N/`` 前缀都是残片，可以删。

判据 1 有一个**例外：本地恢复层**（:data:`RECOVERY_COPY`）。最新
``keep_local``（默认 1）个「远端已有版本 ``COMMIT`` 且本 rank 已上传」的本地版本不删：
它是一份**热副本** —— 回退到这一代不用从对象存储重下，直接读本地
（:func:`storage.upload.load_checkpoint_version`）。这份副本不承担唯一副本的责任，
远端同一代有版本 ``COMMIT``，删了也不丢数据；反过来，远端还没有版本 ``COMMIT`` 的
本地版本不进恢复层 —— 那一代在远端本来也还不可恢复。

反过来，「最新且未完成」一律保留 —— 它可能就是**正在写**（本地）或**正在传**
（远端）的那一代，所以不需要任何「写入者是否活着」的外部判断：

- 兄弟 rank 正在写的目录一定是最新 step，`rank_<r>/COMMIT` 不成立、也没有更新的
  版本提交，两条判据都不命中；
- rank 0 崩在写完版本 ``COMMIT`` 之前留下的完整前缀，会在下一次保存成功后立刻
  变成「非最新」，被判据 2 收掉。回收延迟上界是一个 checkpoint 间隔。

**降级**：本地判据依赖远端状态，所以远端不可达时本地一个都不删 —— 「查不到」
不能当成「可以删」。

删除顺序与写入镜像：先删提交点（本地 ``COMMIT`` / 远端版本 ``COMMIT`` /
``rank_<r>/COMMIT``），再删数据，最后删空目录。判据只看提交点在不在，所以删到一半
被打断，重跑一次收敛。

除此之外还有一条**代际残片**判据（:data:`STALE_GENERATION`）：同一个 step 下不在
当前代际里的 ``step_N/<gen>/`` 前缀只可能是被取代的尝试，直接删；没有版本 ``COMMIT``
的 step 保留序号最大的一代（它可能就是正在传的那一次），其余按残片收掉。

上面这些判据**只删残片**，远端已提交的版本一个都不碰。对象存储侧的**有界保留**是另一
条独立判据（:func:`plan_retention` / :func:`retain_remote_versions`）：版本 ``COMMIT``
一写完，rank 0 就只留最新的 ``keep`` 代（默认 3），从最旧的一代开始删，同样先删版本
``COMMIT`` 再删其余数据 —— 提交点一消失这一版立刻不可恢复，删到一半被打断也不会被
当成「完整版本」。
"""

from __future__ import annotations

import contextlib
import json
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping

from commit.generation import GENERATION_FILE, Generation
from commit.local import COMMIT_FILE, delete_version, version_key
from storage.base import StorageBackend, StorageError
from storage.local import LocalStorage

LOCAL = "local"
REMOTE = "remote"

UPLOADED = "uploaded"
"""本 rank 的分片已经传到远端：本地是冗余副本。"""

SUPERSEDED = "superseded"
"""存在更新的已提交版本：这一代是残片。"""

NOT_UPLOADED = "not_uploaded"
"""本地最新、但远端还没有它的分片提交点：可能正在写，也可能没传成功。"""

NO_NEWER_VERSION = "no_newer_version"
"""远端最新、且没有版本提交点：可能正在传，也可能 rank 0 崩在最后一步。"""

COMMITTED = "committed"
"""远端已提交、且落在保留窗口内的版本：:func:`plan` 永不删除它。"""

RETENTION = "retention"
"""远端已提交但超出保留窗口的版本：由 rank 0 在提交完成后删除。"""

REMOTE_READ_ONLY = "remote_read_only"
"""远端是只读端点（``conditional_writes="off"``）：能列、不能删，远端候选一律跳过。"""

STALE_GENERATION = "stale_generation"
"""同一 step 下不在当前代际里的 ``step_N/<gen>/``：被取代的尝试，整代删。"""

RECOVERY_COPY = "recovery_copy"
"""本地保留的热恢复副本：远端已有这一代的版本 ``COMMIT``，本地不删、回退时直接读。"""

DEFAULT_KEEP_VERSIONS = 3
"""对象存储侧有界保留的默认窗口：只留最新的这么多代已提交版本。"""

DEFAULT_KEEP_LOCAL_VERSIONS = 1
"""本地恢复层的默认窗口：保留最近这么多个「远端已提交且本 rank 已上传」的本地版本。"""


@dataclass(frozen=True, slots=True)
class GcCandidate:
    """一个待删的版本。``location`` 是 ``"local"`` 或 ``"remote"``。

    ``generation`` 非空表示只删这一代（``step_N/<gen>/``），空表示删整个 step 前缀。
    """

    location: str
    step: int
    reason: str
    generation: str | None = None


@dataclass(frozen=True, slots=True)
class GcPlan:
    """一次回收的决策结果：删哪些、留哪些、为什么。"""

    candidates: tuple[GcCandidate, ...]
    kept: tuple[tuple[str, int, str], ...]
    """``(location, step, reason)``：保留的版本与理由。"""

    versions: frozenset[int]
    """本次看到的、带版本 ``COMMIT`` 的 step。"""

    def steps_for(self, location: str) -> tuple[int, ...]:
        return tuple(
            dict.fromkeys(c.step for c in self.candidates if c.location == location)
        )

    def generations_for(self, step: int) -> tuple[str, ...]:
        return tuple(
            c.generation
            for c in self.candidates
            if c.step == step and c.generation is not None
        )

    def reasons(self) -> Mapping[tuple[str, int], str]:
        return {(c.location, c.step): c.reason for c in self.candidates}

    def recovery_steps(self) -> tuple[int, ...]:
        """留在本地当热副本的 step（``RECOVERY_COPY``），升序。"""
        return tuple(
            step
            for location, step, reason in self.kept
            if location == LOCAL and reason == RECOVERY_COPY
        )


def plan(
    *,
    versions: Iterable[int],
    rank_commits: Iterable[tuple[int, int, str | None]],
    prefixes: Iterable[int],
    local_steps: Iterable[int],
    rank: int,
    generations: Mapping[int, Iterable[str]] | None = None,
    current_generations: Mapping[int, str | None] | None = None,
    unknown_versions: Iterable[int] = (),
    keep_local: int = DEFAULT_KEEP_LOCAL_VERSIONS,
) -> GcPlan:
    """决定删哪些版本。**纯函数**：不碰 IO、不看时间、不看进程状态。

    ``versions``:          远端有版本 ``COMMIT`` 的 step（L3）
    ``rank_commits``:      远端有分片 ``COMMIT`` 的 ``(step, rank, generation)``（L2）
    ``prefixes``:          远端存在 ``step_N/`` 前缀的 step
    ``local_steps``:       本地 root 下存在 ``step_N/`` 目录的 step
    ``rank``:              本进程的 rank 号（本地判据只看自己的分片提交点）
    ``generations``:       ``step -> {generation}``：远端出现过的代际前缀
    ``current_generations``: ``step -> 当前代际``（从 L3 读出来；v1 是 ``None``）
    ``unknown_versions``:  读不出来的 L3 对应的 step：**一律保留**，不猜当前代
    ``keep_local``:        本地恢复层窗口：保留最近这么多个「远端已提交且本 rank 已上传」
                           的本地版本作为热副本（``0`` = 上传完即删，老行为）

    三条判据互不重叠：

    1. 本地已上传 / 被更新版本越过 → 删本地目录（恢复层窗口内的除外）；
    2. 远端无 L3 且被更新版本越过 → 删整个 step 前缀；
    3. 远端有 L3 但不是当前代际（或没有 L3 且不是最大代）→ 只删这一代前缀。
    """
    if keep_local < 0:
        raise ValueError(f"keep_local must be >= 0, got {keep_local}")
    known = frozenset(int(step) for step in versions)
    newest = max(known) if known else None
    current = {int(step): gen for step, gen in (current_generations or {}).items()}
    unknown = frozenset(int(step) for step in unknown_versions)
    gens_of = {
        int(step): {str(gen) for gen in values}
        for step, values in (generations or {}).items()
    }
    by_step: dict[int, set[int]] = {}
    for item in rank_commits:
        # 二元组是 v1 语义（没有代际），三元组带 generation。
        step, owner = int(item[0]), int(item[1])
        generation = item[2] if len(item) > 2 else None
        # 只认**当前代**的分片提交点：别的代的 L2 不能证明这一版传完了。
        if step in known and current.get(step) == generation:
            by_step.setdefault(step, set()).add(owner)
    all_prefixes = {int(step) for step in prefixes}
    local = {int(step) for step in local_steps}

    # 恢复层：远端可恢复（有 L3、且 L3 读得出来）+ 本 rank 已上传，取最近的几个。
    # 只有这两个条件同时成立时，本地这份才「不是唯一副本」，删了也不丢东西。
    reusable = sorted(
        step
        for step in local
        if int(rank) in by_step.get(step, set())
        and step in known
        and step not in unknown
    )
    recovery = frozenset(reusable[-keep_local:]) if keep_local else frozenset()

    def superseded(step: int) -> bool:
        return newest is not None and newest > step

    candidates: list[GcCandidate] = []
    kept: list[tuple[str, int, str]] = []

    for step in sorted(local):
        if step in recovery:
            kept.append((LOCAL, step, RECOVERY_COPY))
        elif int(rank) in by_step.get(step, set()):
            candidates.append(GcCandidate(LOCAL, step, UPLOADED))
        elif superseded(step):
            candidates.append(GcCandidate(LOCAL, step, SUPERSEDED))
        else:
            kept.append((LOCAL, step, NOT_UPLOADED))

    for step in sorted(all_prefixes):
        gens = gens_of.get(step, set())
        if step in unknown:
            kept.append((REMOTE, step, COMMITTED))
            continue
        if step in known:
            kept.append((REMOTE, step, COMMITTED))
            # 同一个 step 下不在当前代的代际前缀：被取代的尝试，整代删
            for generation in sorted(gens):
                if generation != current.get(step):
                    candidates.append(
                        GcCandidate(REMOTE, step, STALE_GENERATION, generation)
                    )
            continue
        if gens:
            # 还没有 L3：保留序号最大的一代（可能正在传），其余是残片
            newest_gen = max(gens, key=lambda gen: _generation_seq(gen, step=step))
            kept.append((REMOTE, step, NO_NEWER_VERSION))
            for generation in sorted(gens):
                if generation != newest_gen:
                    candidates.append(
                        GcCandidate(REMOTE, step, STALE_GENERATION, generation)
                    )
            continue
        if superseded(step):
            candidates.append(GcCandidate(REMOTE, step, SUPERSEDED))
        else:
            kept.append((REMOTE, step, NO_NEWER_VERSION))

    return GcPlan(
        candidates=tuple(candidates),
        kept=tuple(kept),
        versions=known,
    )


def plan_retention(
    *, versions: Iterable[int], keep: int = DEFAULT_KEEP_VERSIONS
) -> tuple[int, ...]:
    """对象存储侧的有界保留：只留最新 ``keep`` 代，返回要删的 step（最旧在前）。

    **纯函数**：不碰 IO。入参只取「带版本 ``COMMIT`` 的 step」—— 半途上传、没有提交点
    的前缀不归这条判据管（那是 :func:`plan` 的 ``superseded``）。

    ``keep`` 必须 >= 1：最新那一代永远留着，「删到没有可恢复版本」不是保留策略。
    """
    if keep < 1:
        raise ValueError(f"keep must be >= 1, got {keep}")
    ordered = sorted({int(step) for step in versions})
    return tuple(ordered[:-keep])


# --- 执行器 ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeletedVersion:
    """一个被回收的版本。``bytes`` 对远端记 0（``list`` 不带对象大小）。"""

    location: str
    step: int
    objects: int
    bytes: int
    reason: str
    generation: str | None = None


@dataclass(frozen=True, slots=True)
class GcReport:
    """一次回收的结果：删了什么、留了什么、为什么没做。"""

    dry_run: bool
    planned: tuple[GcCandidate, ...]
    """本次判据命中的候选（``dry_run`` 时也非空，便于先看计划）。"""

    deleted: tuple[DeletedVersion, ...]
    """实际删掉的版本；``dry_run=True`` 时为空。"""

    skipped: tuple[tuple[str, int, str], ...]
    """``(location, step, reason)``：保留的版本与理由。"""

    reclaimed_bytes: int
    versions: frozenset[int]
    aborted: str | None = None
    """整体放弃的原因（远端不可达）。非空时本次一个都没删。"""

    def steps_for(self, location: str) -> tuple[int, ...]:
        return tuple(item.step for item in self.deleted if item.location == location)

    def recovery_steps(self) -> tuple[int, ...]:
        """留在本地当热副本的 step（``RECOVERY_COPY``），升序。"""
        return tuple(
            step
            for location, step, reason in self.skipped
            if location == LOCAL and reason == RECOVERY_COPY
        )


def delete_local_version(root: str | Path, step: int) -> DeletedVersion:
    """删掉本地一个 ``step_N/`` 目录：先删 ``COMMIT``，再删数据，最后移空目录。

    先删提交点让这一版立刻「不可上传、不可恢复」；剩下的文件即使删到一半被打断，
    下次扫描依旧看到「没有提交点」，重跑收敛。``reason`` 由调用方填。
    """
    root_path = Path(root)
    directory = root_path / version_key(step)
    files = sorted(p for p in directory.rglob("*") if p.is_file()) if directory.is_dir() else []
    size = 0
    for path in files:
        try:
            size += path.stat().st_size
        except OSError:  # pragma: no cover - 统计失败不影响删除
            continue
    delete_version(LocalStorage(root_path), step)
    with contextlib.suppress(OSError):  # 目录非空（并发写入）就留着，下次再说
        directory.rmdir()
    return DeletedVersion(LOCAL, step, len(files), size, "")


def _scan_local(root: Path) -> set[int]:
    """本地 root 下的 ``step_N/`` 目录名对应的 step 号。"""
    found: set[int] = set()
    if not root.is_dir():
        return found
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("step_"):
            continue
        try:
            found.add(int(child.name[len("step_") :]))
        except ValueError:
            continue
    return found


@dataclass(frozen=True, slots=True)
class RemoteLayout:
    """一次 ``list("")`` 解析出的远端结构（纯数据，执行器与判据共用）。"""

    versions: set[int]
    """有版本 ``COMMIT``（L3）的 step。"""

    rank_commits: set[tuple[int, int, str | None]]
    """``(step, rank, generation)``：有分片 ``COMMIT``（L2）的组合；v1 布局的
    ``generation`` 是 ``None``。"""

    generations: dict[int, set[str]] = field(default_factory=dict)
    """``step -> {generation}``：远端出现过的代际前缀。"""

    prefixes: set[int] = field(default_factory=set)
    """存在 ``step_N/`` 前缀的 step。"""


def _scan_remote(keys: Iterable[str]) -> RemoteLayout:
    """从一次 ``list("")`` 的结果里解析出四层信息。

    - ``step_N/COMMIT``：版本提交点（L3）；
    - ``step_N/<gen>/rank_r/COMMIT``：某一代的分片提交点（L2）；
    - ``step_N/<gen>/...``：代际前缀；
    - 其余 ``step_N/...``：只用来确认前缀存在。

    v1 老布局（``step_N/rank_r/COMMIT``，没有 gen 层）仍然认，``generation=None``。
    """
    versions: set[int] = set()
    rank_commits: set[tuple[int, int, str | None]] = set()
    generations: dict[int, set[str]] = {}
    prefixes: set[int] = set()
    for key in keys:
        parts = key.split("/")
        if not parts or not parts[0].startswith("step_"):
            continue
        try:
            step = int(parts[0][len("step_") :])
        except ValueError:
            continue
        prefixes.add(step)
        if len(parts) == 2 and parts[1] == COMMIT_FILE:
            versions.add(step)
        elif len(parts) == 4 and parts[2].startswith("rank_") and parts[3] == COMMIT_FILE:
            generation = parts[1]
            try:
                rank = int(parts[2][len("rank_") :])
            except ValueError:
                continue
            rank_commits.add((step, rank, generation))
            generations.setdefault(step, set()).add(generation)
        elif len(parts) >= 3:
            if parts[1].startswith("gen_"):
                generations.setdefault(step, set()).add(parts[1])
            elif len(parts) == 3 and parts[1].startswith("rank_") and parts[2] == COMMIT_FILE:
                try:
                    rank_commits.add((step, int(parts[1][len("rank_") :]), None))
                except ValueError:
                    continue
    return RemoteLayout(
        versions=versions,
        rank_commits=rank_commits,
        generations=generations,
        prefixes=prefixes,
    )


def _generation_seq(generation: str, *, step: int) -> int:
    """代际前缀的序号；解析不了就排最前（宁可保留它也不误删）。"""
    try:
        return Generation.parse(generation, step=step).seq
    except Exception:
        return -1


def _delete_remote_prefix(
    remote: StorageBackend,
    step: int,
    keys: Iterable[str],
    *,
    generation: str | None = None,
) -> int:
    """删远端 ``step_N/``（或其中一代 ``step_N/<gen>/``），顺序与写入镜像。

    **整版**：先删**版本提交点** ``step_N/COMMIT``（L3）—— 它一消失这一版立刻不可恢复、
    也不会再被当成「已提交版本」；再删 ``GEN``（让残留写者下一轮检查立即被 fence）；
    再删各 ``rank_r/COMMIT``（L2）；最后才是数据对象。

    **单代**：这一代没有 L3 可删，先删它的 L2，再删数据 —— 删到一半被打断也不会留下
    「看着像这一代传完了」的前缀，重跑一次收敛。
    """
    root = version_key(step)
    prefix = f"{root}/{generation}/" if generation else f"{root}/"
    mine = sorted(key for key in keys if key.startswith(prefix))
    rank_commits = {
        key for key in mine if key.count("/") == 3 and key.endswith(f"/{COMMIT_FILE}")
    }
    if generation:
        ordered = sorted(rank_commits) + [key for key in mine if key not in rank_commits]
    else:
        version_commit = f"{root}/{COMMIT_FILE}"
        marker = f"{root}/{GENERATION_FILE}"
        markers = {marker} & set(mine)
        rest = [
            key
            for key in mine
            if key != version_commit and key not in rank_commits and key not in markers
        ]
        ordered = ([version_commit] if version_commit in mine else [])
        ordered += sorted(markers) + sorted(rank_commits) + rest
    for key in ordered:
        remote.delete(key)
    _prune_remote_dirs(remote, step, generation=generation)
    return len(mine)


def _prune_remote_dirs(
    remote: StorageBackend, step: int, *, generation: str | None = None
) -> None:
    """本地盘后端会把空目录留下来（对象存储没有目录概念），顺手收掉。"""
    if not isinstance(remote, LocalStorage):
        return
    directory = remote.path_for(version_key(step))
    if generation:
        with contextlib.suppress(OSError):  # 非空 = 有人还在写，留着
            (directory / generation).rmdir()
        return
    for child in sorted(directory.glob("*"), reverse=True):
        if child.is_dir():
            with contextlib.suppress(OSError):  # 非空 = 有人还在写，留着
                child.rmdir()
    with contextlib.suppress(OSError):
        directory.rmdir()


_UNKNOWN_VERSION = object()
"""L3 读不出来（被删 / 坏 JSON / 远端抖动）时的哨兵：这一 step 一律保留。"""


def _read_version_generation(remote: StorageBackend, step: int) -> object:
    """读 L3 里声明的代际；``None`` 表示 v1 老布局，哨兵表示读不出来。"""
    try:
        payload = remote.read(f"{version_key(step)}/{COMMIT_FILE}")
    except StorageError:
        return _UNKNOWN_VERSION
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _UNKNOWN_VERSION
    if not isinstance(data, dict):
        return _UNKNOWN_VERSION
    generation = data.get("generation")
    return None if generation is None else str(generation)


def collect(
    local_root: str | Path,
    remote: StorageBackend | None,
    *,
    rank: int,
    dry_run: bool = True,
    keep_local: int = DEFAULT_KEEP_LOCAL_VERSIONS,
) -> GcReport:
    """跑一次回收：一次 ``list`` + 一次目录扫描 + 按 :func:`plan` 的决策删除。

    ``dry_run=True``（默认）只给出计划，不动任何文件。

    ``keep_local`` 是本地恢复层窗口（默认 1）：最近这么多个「远端已提交且本 rank
    已上传」的本地版本留在本地当热副本，回退时直接读，其余照旧回收；``0`` 回到
    「上传完即删」。

    ``remote`` 不可达时**什么都不删**：本地判据依赖远端状态，「查不到」不能当成
    「可以删」。这种情况在 ``GcReport.aborted`` 里记原因，定时线程不会因此崩掉。
    """
    root = Path(local_root)
    if remote is None:
        return GcReport(
            dry_run=dry_run,
            planned=(),
            deleted=(),
            skipped=(),
            reclaimed_bytes=0,
            versions=frozenset(),
            aborted="no remote storage configured",
        )

    try:
        keys = remote.list("")
    except StorageError as exc:
        return GcReport(
            dry_run=dry_run,
            planned=(),
            deleted=(),
            skipped=(),
            reclaimed_bytes=0,
            versions=frozenset(),
            aborted=f"remote unreachable: {exc}",
        )

    layout = _scan_remote(keys)
    current: dict[int, str | None] = {}
    unknown: set[int] = set()
    for step in sorted(layout.versions):
        generation = _read_version_generation(remote, step)
        if generation is _UNKNOWN_VERSION:
            unknown.add(step)
        else:
            current[step] = generation  # type: ignore[assignment]
    result = plan(
        versions=layout.versions - unknown,
        rank_commits=layout.rank_commits,
        prefixes=layout.prefixes,
        local_steps=_scan_local(root),
        rank=rank,
        generations=layout.generations,
        current_generations=current,
        unknown_versions=unknown,
        keep_local=keep_local,
    )

    deleted: list[DeletedVersion] = []
    skipped: list[tuple[str, int, str]] = list(result.kept)
    writable = bool(getattr(remote, "supports_conditional_writes", True))
    if not dry_run:
        for candidate in result.candidates:
            if candidate.location == REMOTE:
                if not writable:
                    # 只读端点：list 能看、delete 不能做。远端一个对象都不动。
                    skipped.append((REMOTE, candidate.step, REMOTE_READ_ONLY))
                    continue
                count = _delete_remote_prefix(
                    remote, candidate.step, keys, generation=candidate.generation
                )
                deleted.append(
                    DeletedVersion(
                        REMOTE,
                        candidate.step,
                        count,
                        0,
                        candidate.reason,
                        candidate.generation,
                    )
                )
            else:
                record = delete_local_version(root, candidate.step)
                deleted.append(replace(record, reason=candidate.reason))

    return GcReport(
        dry_run=dry_run,
        planned=result.candidates,
        deleted=tuple(deleted),
        skipped=tuple(skipped),
        reclaimed_bytes=sum(item.bytes for item in deleted),
        versions=result.versions,
    )


# --- 对象存储侧的有界保留（rank 0 在版本提交完成后执行） --------------------


@dataclass(frozen=True, slots=True)
class RetentionReport:
    """一次有界保留的结果：删了哪些代、留下哪几代、为什么没做。"""

    keep: int
    dry_run: bool
    planned: tuple[int, ...]
    """超出保留窗口的 step，最旧在前（``dry_run`` 时也非空）。"""

    deleted: tuple[DeletedVersion, ...]
    """实际删掉的版本；``dry_run=True`` 时为空。"""

    kept: tuple[int, ...]
    """落在窗口内、保留的已提交 step。"""

    aborted: str | None = None
    """整体放弃（远端不可达）或删到一半失败的原因；非空时可能只删了一部分。"""

    def steps(self) -> tuple[int, ...]:
        return tuple(item.step for item in self.deleted)


def retain_remote_versions(
    remote: StorageBackend | None,
    *,
    keep: int = DEFAULT_KEEP_VERSIONS,
    dry_run: bool = False,
) -> RetentionReport:
    """有界保留：一次 ``list``，只留最新 ``keep`` 代已提交版本，从最旧的一代开始删。

    这是 :func:`collect` 之外的**另一条判据**，只在「版本 ``COMMIT`` 刚写完」这个
    时刻跑（rank 0 的提交动作，见 :func:`storage.upload.finalize_upload`）：

    - ``collect`` 删的是**残片**（没有提交点、或被更新版本越过）；
    - 这里删的是**完整但过旧**的版本：``keep`` 之外最旧的几代。

    删除顺序见 :func:`_delete_remote_prefix`：先版本 ``COMMIT``，再分片 ``COMMIT``，
    最后数据。所以删到一半被打断时，这一版已经「不可恢复」，下次 ``collect`` 会把它的
    剩余前缀当残片收掉，两条路径收敛到同一个结果。

    降级：远端不可达时什么都不删（``RetentionReport.aborted``）；删某一代中途失败就停在
    那里并记原因 —— 已经删掉 ``COMMIT`` 的那一代不会回到「可恢复」状态。

    已知限制：保留判据不看「谁正在读」。如果恢复端正好在读一代刚好被删的版本，
    会读失败（报错而不是读到半份数据）。``load_object(step=None)`` 永远落在窗口内。
    """
    if keep < 1:
        raise ValueError(f"keep must be >= 1, got {keep}")
    if remote is None:
        return RetentionReport(
            keep=keep,
            dry_run=dry_run,
            planned=(),
            deleted=(),
            kept=(),
            aborted="no remote storage configured",
        )

    try:
        keys = remote.list("")
    except StorageError as exc:
        return RetentionReport(
            keep=keep,
            dry_run=dry_run,
            planned=(),
            deleted=(),
            kept=(),
            aborted=f"remote unreachable: {exc}",
        )

    layout = _scan_remote(keys)
    doomed = plan_retention(versions=layout.versions, keep=keep)
    kept = tuple(sorted(layout.versions - set(doomed)))

    deleted: list[DeletedVersion] = []
    aborted: str | None = None
    if not dry_run:
        for step in doomed:
            try:
                count = _delete_remote_prefix(remote, step, keys)
            except StorageError as exc:  # 停在这一代，下次提交后再收敛
                aborted = f"step_{step}: delete failed: {exc}"
                break
            deleted.append(DeletedVersion(REMOTE, step, count, 0, RETENTION))

    return RetentionReport(
        keep=keep,
        dry_run=dry_run,
        planned=doomed,
        deleted=tuple(deleted),
        kept=kept,
        aborted=aborted,
    )


class GcWorker:
    """周期性回收的守护线程：**启动时先跑一次**，之后每 ``interval`` 秒一次。

    - 启动时那次是主力：它收掉上一次运行崩溃留下的残目录（SIGKILL 的产物）。
    - 运行期那次只处理「已经上传完」「已经被更新的版本越过」两类，正在写/正在传的
      目录天然不命中，所以不需要和保存路径做任何协调。
    - 单次失败（远端抖动、IO 错误）只记进 ``last_error``，线程继续跑；远端整体不可达
      时 :func:`collect` 自己是空操作，不会误删。

    它**不做有界保留**：裁剪「完整但过旧」的版本需要一个明确锚点（「刚提交完一代」），
    由 rank 0 在 :func:`storage.upload.finalize_upload` 里调
    :func:`retain_remote_versions`，不放进周期线程。**本地恢复层**（``keep_local``）
    不是这种保留：它随每次 :func:`collect` 生效，判据是纯函数。
    """

    def __init__(
        self,
        local_root: str | Path,
        remote: StorageBackend | None,
        *,
        rank: int,
        interval: float = 60.0,
        dry_run: bool = False,
        history: int = 8,
        keep_local: int = DEFAULT_KEEP_LOCAL_VERSIONS,
    ) -> None:
        if interval <= 0:
            raise ValueError(f"interval must be > 0, got {interval}")
        self.local_root = Path(local_root)
        self.remote = remote
        self.rank = int(rank)
        self.interval = float(interval)
        self.dry_run = dry_run
        self.keep_local = int(keep_local)
        self.history = max(1, int(history))
        self.reports: list[GcReport] = []
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- 生命周期 -----------------------------------------------------------
    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def run_once(self) -> GcReport:
        """跑一次回收（不启线程时也可以直接调）。"""
        report = collect(
            self.local_root,
            self.remote,
            rank=self.rank,
            dry_run=self.dry_run,
            keep_local=self.keep_local,
        )
        self.reports.append(report)
        del self.reports[: -self.history]
        return report

    def start(self) -> "GcWorker":
        if self._thread is not None:
            raise RuntimeError("GcWorker already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="ckpt-gc", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout)

    def __enter__(self) -> "GcWorker":
        return self.start()

    def __exit__(self, *exc_info: object) -> bool:
        self.stop()
        return False

    # --- 内部 ---------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as exc:  # 单次失败不能让线程退出
                self.last_error = f"{type(exc).__name__}: {exc}"
            if self._stop.wait(self.interval):
                return


__all__ = [
    "COMMITTED",
    "DEFAULT_KEEP_LOCAL_VERSIONS",
    "DEFAULT_KEEP_VERSIONS",
    "DeletedVersion",
    "GcCandidate",
    "GcPlan",
    "GcReport",
    "GcWorker",
    "LOCAL",
    "NOT_UPLOADED",
    "NO_NEWER_VERSION",
    "RECOVERY_COPY",
    "REMOTE",
    "REMOTE_READ_ONLY",
    "RETENTION",
    "RemoteLayout",
    "RetentionReport",
    "STALE_GENERATION",
    "SUPERSEDED",
    "UPLOADED",
    "collect",
    "delete_local_version",
    "plan",
    "plan_retention",
    "retain_remote_versions",
]
