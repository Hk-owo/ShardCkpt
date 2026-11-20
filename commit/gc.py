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

除此之外还有一条**代际残片**判据（:data:`STALE_GENERATION`）：同一个 step 下**序号
严格早于「最新一次尝试」**的 ``step_N/<gen>/`` 前缀只可能是被取代的尝试，删。

「最新一次尝试」不能只看版本 ``COMMIT``（L3）：L3 只在提交成功时才写，所以在
「新一代已 acquire、正在传」到「提交成功」这整段时间里 L3 仍指向**旧代** —— 拿 L3 当
当前代，就会把正在传的那一代当残片删掉，连带本地刚写好的数据（``uploaded`` 判据用的
也是同一个代）。所以判据取 ``max(step_N/GEN 声明的代, 目录里序号最大的代)``：

- GEN 指向新一代而它还只传了一半 ⇒ 那一代受保护，旧代（L3 声明的那个）照样受保护，
  一个都不删；
- 新一代提交成功后 L3 跟上，更早的代（包括被 fence 的中间代）才是残片；
- GEN 读不出来（被外部删 / 坏 JSON）⇒ 降级为「目录里序号最大的那一代」，
  宁可晚一代回收，也不误删正在写的那一代。

上面这些判据**只删残片**，远端已提交的版本一个都不碰。对象存储侧的**有界保留**是另一
条独立判据（:func:`plan_retention` / :func:`retain_remote_versions`）：版本 ``COMMIT``
一写完，rank 0 就只留最新的 ``keep`` 代（默认 3），从最旧的一代开始删，同样先删版本
``COMMIT`` 再删其余数据 —— 提交点一消失这一版立刻不可恢复，删到一半被打断也不会被
当成「完整版本」。
"""

from __future__ import annotations

import contextlib
import re
import tempfile
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from commit.generation import (
    STATUS_FILE,
    Generation,
    StepStatus,
    read_status,
    status_key,
)
from commit.local import COMMIT_FILE, delete_version, version_key
from core import FormatError
from storage.base import (
    StorageBackend,
    StorageError,
    run_scope,
    unwrap_storage,
    validate_run_id,
)
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

_UNRESOLVED = object()
"""判不出「最新一次尝试」的代际（坏目录名 / 既无 GEN 也无 L3）：一个代都不删。"""

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
    pointer_generations: Mapping[int, str] | None = None,
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
    ``current_generations``: ``step -> 已提交代际``（从 L3 读出来；v1 是 ``None``）
    ``pointer_generations``: ``step -> 最新一次尝试的代际``（从 ``step_N/GEN`` 读出来；
                           读不出来就不放这个 step，判据降级到目录名）
    ``unknown_versions``:  读不出来的 L3 对应的 step：**一律保留**，不猜当前代
    ``keep_local``:        本地恢复层窗口：保留最近这么多个「远端已提交且本 rank 已上传」
                           的本地版本作为热副本（``0`` = 上传完即删，老行为）

    三条判据互不重叠：

    1. 本地已在最新一次尝试里上传 / 被更新版本越过 → 删本地目录（恢复层窗口内的除外）；
    2. 远端无 L3 且被更新版本越过 → 删整个 step 前缀；
    3. 远端有 L3，但代际序号严格早于「最新一次尝试」且不是 L3 声明的代 → 只删这一代。

    「最新一次尝试」= ``max(GEN 声明的代, 目录里序号最大的代)``；两个都判不出来时
    一个代都不删（:data:`STALE_GENERATION` 的降级）。
    """
    if keep_local < 0:
        raise ValueError(f"keep_local must be >= 0, got {keep_local}")
    known = frozenset(int(step) for step in versions)
    newest = max(known) if known else None
    current = {int(step): gen for step, gen in (current_generations or {}).items()}
    pointer = {int(step): str(gen) for step, gen in (pointer_generations or {}).items()}
    unknown = frozenset(int(step) for step in unknown_versions)
    gens_of = {
        int(step): {str(gen) for gen in values}
        for step, values in (generations or {}).items()
    }
    all_prefixes = {int(step) for step in prefixes}
    local = {int(step) for step in local_steps}

    _attempts: dict[int, object] = {}

    def latest_attempt(step: int) -> object:
        """这个 step 最新一次尝试的代际；判不出来时返回 :data:`_UNRESOLVED`。

        取 ``max(GEN 声明的代, 目录里序号最大的代)``：GEN 是写权仲裁的线性化点，
        L3 落后时它才是「正在传的那一代」。GEN 读不出来就只看目录名（降级）。
        """
        if step in _attempts:
            return _attempts[step]
        attempt: object = _UNRESOLVED
        best: str | None = None
        best_seq = -1
        parseable = True
        for generation in sorted(gens_of.get(step, set())):
            seq = _generation_seq(generation, step=step)
            if seq < 0:  # 认不出的目录名：不猜，一个代都不删
                parseable = False
                break
            if seq > best_seq:
                best, best_seq = generation, seq
        if parseable:
            pointer_generation = pointer.get(step)
            if pointer_generation is not None:
                seq = _generation_seq(pointer_generation, step=step)
                if seq < 0:
                    parseable = False
                elif seq > best_seq:
                    best, best_seq = pointer_generation, seq
        if not parseable:
            attempt = _UNRESOLVED
        elif best is not None:
            attempt = best
        else:
            # 没有代际前缀也没有 GEN：退回 L3 里声明的那个（v1 布局就是 ``None``），
            # 这样 v1 的二元组 ``rank_commits`` 仍然能被认出来。
            attempt = current.get(step)
        _attempts[step] = attempt
        return attempt

    def stale_prefixes(step: int) -> list[str]:
        """同一 step 下可删的代际前缀：序号严格早于最新一次尝试，且不是已提交代。"""
        attempt = latest_attempt(step)
        if not isinstance(attempt, str):
            return []  # 判不出最新代（含 v1 布局、坏目录名）：一个都不删
        limit = _generation_seq(attempt, step=step)
        if limit < 0:
            return []
        committed = current.get(step)
        doomed: list[str] = []
        for generation in sorted(gens_of.get(step, set())):
            if generation == attempt or generation == committed:
                continue
            seq = _generation_seq(generation, step=step)
            if 0 <= seq < limit:
                doomed.append(generation)
        return doomed

    by_step: dict[int, set[int]] = {}
    for item in rank_commits:
        # 二元组是 v1 语义（没有代际），三元组带 generation。
        step, owner = int(item[0]), int(item[1])
        generation = item[2] if len(item) > 2 else None
        if step not in known:
            continue
        # 只认**最新一次尝试**的分片提交点：本地目录装的是那一次的产物，别的代的 L2
        # 不能证明它已经传完。
        attempt = latest_attempt(step)
        if attempt is _UNRESOLVED or attempt != generation:
            continue
        by_step.setdefault(step, set()).add(owner)

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
            # 同一个 step 下被取代的代际前缀：整代删。已提交的代（L3 声明的那个）
            # 与最新一次尝试（GEN / 目录里序号最大的那个）永远不动。
            for generation in stale_prefixes(step):
                candidates.append(
                    GcCandidate(REMOTE, step, STALE_GENERATION, generation)
                )
            continue
        if gens:
            if superseded(step):
                candidates.append(GcCandidate(REMOTE, step, SUPERSEDED))
                continue
            # 还没有 L3 且未被更新版本越过：保留最新一次尝试（可能正在传），其余是残片
            kept.append((REMOTE, step, NO_NEWER_VERSION))
            for generation in stale_prefixes(step):
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
class AbortedUpload:
    """R16a 被自动中止的孤儿/超时分片上传任务。"""

    key: str
    upload_id: str
    reason: str


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
    aborted_uploads: tuple[AbortedUpload, ...] = ()
    """R16a 中止的超时/失效分片上传任务（dry_run 时为空）。"""
    cleaned_staging: tuple[str, ...] = ()
    """R4 清理的已死进程暂存目录路径。"""

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

    status_steps: set[int]
    """存在 ``step_N/STATUS`` 的 step（是否已提交还要看 ``STATUS.committed``）。"""

    rank_commits: set[tuple[int, int, str]]
    """``(step, rank, generation)``：有分片 ``COMMIT``（L2）的组合。"""

    generations: dict[int, set[str]] = field(default_factory=dict)
    """``step -> {generation}``：远端出现过的代际前缀。"""

    prefixes: set[int] = field(default_factory=set)
    """存在 ``step_N/`` 前缀的 step。"""


def _scan_remote(keys: Iterable[str]) -> RemoteLayout:
    """从一次 ``list("")`` 的结果里解析出远端结构。

    - ``step_N/STATUS``：本 step 的单点状态对象（写权 + 提交指针）；
    - ``step_N/<gen>/rank_r/COMMIT``：某一代的分片提交点（L2）；
    - ``step_N/<gen>/...``：代际前缀；
    - 其余 ``step_N/...``：只用来确认前缀存在。
    """
    status_steps: set[int] = set()
    rank_commits: set[tuple[int, int, str]] = set()
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
        if len(parts) == 2 and parts[1] == STATUS_FILE:
            status_steps.add(step)
        elif len(parts) == 4 and parts[2].startswith("rank_") and parts[3] == COMMIT_FILE:
            generation = parts[1]
            try:
                rank = int(parts[2][len("rank_") :])
            except ValueError:
                continue
            rank_commits.add((step, rank, generation))
            generations.setdefault(step, set()).add(generation)
        elif len(parts) >= 3 and parts[1].startswith("gen_"):
            generations.setdefault(step, set()).add(parts[1])
    return RemoteLayout(
        status_steps=status_steps,
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

    **整版**：先删 ``step_N/STATUS`` —— 它一消失这一版立刻不可恢复、也立刻
    fence 掉残留写者；再删各 ``rank_r/COMMIT``（L2）；最后才是数据对象。

    **单代**：先删它的 L2，再删数据 —— 删到一半被打断也不会留下「看着像这一代
    传完了」的前缀，重跑一次收敛。
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
        status_obj = status_key(step)
        rest = [
            key
            for key in mine
            if key != status_obj and key not in rank_commits
        ]
        ordered = ([status_obj] if status_obj in mine else [])
        ordered += sorted(rank_commits) + rest
    for key in ordered:
        remote.delete(key)
    _prune_remote_dirs(remote, step, generation=generation)
    return len(mine)


def _prune_remote_dirs(
    remote: StorageBackend, step: int, *, generation: str | None = None
) -> None:
    """本地盘后端会把空目录留下来（对象存储没有目录概念），顺手收掉。"""
    if not isinstance(unwrap_storage(remote), LocalStorage):
        return
    path_for = getattr(remote, "path_for", None)
    if not callable(path_for):
        return
    directory = path_for(version_key(step))
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


_UNKNOWN_STATUS = object()
"""STATUS 读不出来（被删 / 坏 JSON / 远端抖动）时的哨兵：这一 step 一律保留。"""


def _read_step_status(remote: StorageBackend, step: int) -> object:
    """读 ``step_N/STATUS``；读不出来或格式坏返回哨兵（fail-closed，不误删）。"""
    try:
        status = read_status(remote, step)
    except (StorageError, FormatError, UnicodeDecodeError):
        return _UNKNOWN_STATUS
    return _UNKNOWN_STATUS if status is None else status


DEFAULT_MPU_TIMEOUT_SECONDS = 3600.0
"""R16a 远端残留分片上传（MPU）的默认超时阈值（1 小时）。"""

_STEP_GEN_KEY_RE = re.compile(r"^step_(\d+)/(gen_\d{4,}-[0-9a-z]{1,32})/")


def cleanup_stale_multipart_uploads(
    remote: StorageBackend | None,
    *,
    max_age_seconds: float = DEFAULT_MPU_TIMEOUT_SECONDS,
    now: datetime | None = None,
    dry_run: bool = False,
    run_id: str | None = None,
    stale_steps: Iterable[int] = (),
    stale_generations: Mapping[int, Iterable[str]] | None = None,
    max_committed_step: int | None = None,
    active_generations: Mapping[int, Iterable[str]] | None = None,
) -> tuple[AbortedUpload, ...]:
    """R16a：列举并中止超时或所属代际已失效的未完成分片上传（MPU）。"""
    if remote is None:
        return ()
    scoped = run_scope(remote, run_id)
    if not bool(getattr(scoped, "supports_conditional_writes", True)):
        return ()
    list_fn = getattr(scoped, "list_multipart_uploads", None)
    abort_fn = getattr(scoped, "abort_multipart_upload", None)
    if not callable(list_fn) or not callable(abort_fn):
        return ()

    current_time = now if now is not None else datetime.now(timezone.utc)
    doomed_steps = {int(s) for s in stale_steps}
    doomed_gens = {
        int(s): {str(g) for g in gens} for s, gens in (stale_generations or {}).items()
    }
    live_gens = {
        int(s): {str(g) for g in gens if g}
        for s, gens in (active_generations or {}).items()
    }
    aborted: list[AbortedUpload] = []
    for upload in list_fn(""):
        reason: str | None = None
        match = _STEP_GEN_KEY_RE.match(upload.key)
        if match:
            up_step = int(match.group(1))
            up_gen = match.group(2)
            if up_step in doomed_steps or (
                max_committed_step is not None and up_step < max_committed_step
            ):
                reason = SUPERSEDED
            elif up_gen in doomed_gens.get(up_step, set()):
                reason = STALE_GENERATION
            elif up_step in live_gens and up_gen not in live_gens[up_step]:
                reason = STALE_GENERATION
        if reason is None:
            initiated = upload.initiated
            if initiated.tzinfo is None:
                initiated = initiated.replace(tzinfo=timezone.utc)
            if (current_time - initiated).total_seconds() >= max_age_seconds:
                reason = "mpu_timeout"
        if reason is None:
            continue
        if not dry_run:
            abort_fn(upload.key, upload.upload_id)
        aborted.append(
            AbortedUpload(key=upload.key, upload_id=upload.upload_id, reason=reason)
        )
    return tuple(aborted)


def collect(
    local_root: str | Path,
    remote: StorageBackend | None,
    *,
    rank: int,
    dry_run: bool = True,
    keep_local: int = DEFAULT_KEEP_LOCAL_VERSIONS,
    run_id: str | None = None,
    staging_dir: str | Path | None = None,
    abort_stale_mpu: bool = True,
    mpu_timeout_seconds: float = DEFAULT_MPU_TIMEOUT_SECONDS,
) -> GcReport:
    """跑一次回收：一次 ``list`` + 一次目录扫描 + 按 :func:`plan` 的决策删除。

    ``dry_run=True``（默认）只给出计划，不动任何文件。

    ``keep_local`` 是本地恢复层窗口（默认 1）：最近这么多个「远端已提交且本 rank
    已上传」的本地版本留在本地当热副本，回退时直接读，其余照旧回收；``0`` 回到
    「上传完即删」。

    ``remote`` 不可达时**什么都不删**：本地判据依赖远端状态，「查不到」不能当成
    「可以删」。这种情况在 ``GcReport.aborted`` 里记原因，定时线程不会因此崩掉。
    """
    base_local = Path(local_root)
    root = base_local / validate_run_id(run_id) if run_id is not None else base_local
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

    scoped = run_scope(remote, run_id)
    try:
        keys = scoped.list("")
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
    versions: set[int] = set()
    current: dict[int, str | None] = {}
    pointer: dict[int, str] = {}
    unknown: set[int] = set()
    for step in sorted(layout.status_steps):
        status = _read_step_status(scoped, step)
        if status is _UNKNOWN_STATUS:
            unknown.add(step)
            continue
        assert isinstance(status, StepStatus)
        pointer[step] = status.generation.id
        if status.committed is not None:
            versions.add(step)
            current[step] = status.committed.generation.id
    result = plan(
        versions=versions,
        rank_commits=layout.rank_commits,
        prefixes=layout.prefixes,
        local_steps=_scan_local(root),
        rank=rank,
        generations=layout.generations,
        current_generations=current,
        pointer_generations=pointer,
        unknown_versions=unknown,
        keep_local=keep_local,
    )

    deleted: list[DeletedVersion] = []
    skipped: list[tuple[str, int, str]] = list(result.kept)
    aborted_uploads: tuple[AbortedUpload, ...] = ()
    cleaned_staging: tuple[str, ...] = ()
    writable = bool(getattr(scoped, "supports_conditional_writes", True))
    if not dry_run:
        from storage.upload import cleanup_stale_staging_dirs

        # R4：GC 时一并清理已死进程遗留的暂存目录（持锁已释放）
        cleaned_paths = cleanup_stale_staging_dirs(
            base_local, root, staging_dir, Path(tempfile.gettempdir())
        )
        cleaned_staging = tuple(str(p) for p in cleaned_paths)

        for candidate in result.candidates:
            if candidate.location == REMOTE:
                if not writable:
                    # 只读端点：list 能看、delete 不能做。远端一个对象都不动。
                    skipped.append((REMOTE, candidate.step, REMOTE_READ_ONLY))
                    continue
                count = _delete_remote_prefix(
                    scoped, candidate.step, keys, generation=candidate.generation
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

        if abort_stale_mpu and writable:
            stale_steps = {
                c.step
                for c in result.candidates
                if c.location == REMOTE and c.generation is None
            }
            stale_gens: dict[int, set[str]] = {}
            for c in result.candidates:
                if c.location == REMOTE and c.generation is not None:
                    stale_gens.setdefault(c.step, set()).add(c.generation)
            active_gens: dict[int, set[str]] = {}
            for s in set(current) | set(pointer):
                gens_set: set[str] = set()
                if current.get(s):
                    gens_set.add(str(current[s]))
                if pointer.get(s):
                    gens_set.add(str(pointer[s]))
                if gens_set:
                    active_gens[s] = gens_set
            try:
                aborted_uploads = cleanup_stale_multipart_uploads(
                    scoped,
                    max_age_seconds=mpu_timeout_seconds,
                    dry_run=False,
                    stale_steps=stale_steps,
                    stale_generations=stale_gens,
                    max_committed_step=max(versions) if versions else None,
                    active_generations=active_gens,
                )
            except StorageError:
                pass

    return GcReport(
        dry_run=dry_run,
        planned=result.candidates,
        deleted=tuple(deleted),
        skipped=tuple(skipped),
        reclaimed_bytes=sum(item.bytes for item in deleted),
        versions=result.versions,
        aborted_uploads=aborted_uploads,
        cleaned_staging=cleaned_staging,
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
    run_id: str | None = None,
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

    scoped = run_scope(remote, run_id)
    try:
        keys = scoped.list("")
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
    committed_versions: set[int] = set()
    for step in sorted(layout.status_steps):
        status = _read_step_status(scoped, step)
        if status is _UNKNOWN_STATUS:
            return RetentionReport(
                keep=keep,
                dry_run=dry_run,
                planned=(),
                deleted=(),
                kept=(),
                aborted=f"step_{step}: unreadable STATUS",
            )
        assert isinstance(status, StepStatus)
        if status.committed is not None:
            committed_versions.add(step)
    doomed = plan_retention(versions=committed_versions, keep=keep)
    kept = tuple(sorted(committed_versions - set(doomed)))

    deleted: list[DeletedVersion] = []
    aborted: str | None = None
    if not dry_run:
        for step in doomed:
            try:
                count = _delete_remote_prefix(scoped, step, keys)
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


def reclaim_stale_generations(
    remote: StorageBackend | None,
    step: int,
    *,
    keep: str | Iterable[str] = (),
) -> tuple[DeletedVersion, ...]:
    """删掉 ``step_N/`` 下**严格早于** ``keep`` 里这些代的代际前缀。

    这是 rank 0 提交成功之后的收尾（:func:`storage.upload.finalize_upload` 调）：L3
    已经指向 ``keep``，所以同一 step 的更早尝试只可能是被 fence 的残片，不必等下一次
    :func:`collect`。判据与 :func:`plan` 完全一致：

    - 只删序号**严格小于** ``keep`` 中最大序号的代 —— 已经有人在传更新的一代时，
      那一代不会被误删；
    - 序号读不出来的目录名一概留着（不猜）。

    返回每个被删代的记录（``reason="stale_generation"``）。远端不可达时抛
    :class:`StorageError`，由调用方决定是否吞掉 —— 提交本身已经成立，这不是失败。
    """
    if remote is None:
        return ()
    keepers = {keep} if isinstance(keep, str) else {str(item) for item in keep}
    seqs = [_generation_seq(generation, step=step) for generation in keepers]
    limits = [seq for seq in seqs if seq >= 0]
    if not limits:
        return ()
    limit = max(limits)
    keys = remote.list(version_key(step))
    layout = _scan_remote(keys)
    doomed = [
        generation
        for generation in sorted(layout.generations.get(step, set()))
        if generation not in keepers and 0 <= _generation_seq(generation, step=step) < limit
    ]
    deleted: list[DeletedVersion] = []
    for generation in doomed:
        count = _delete_remote_prefix(remote, step, keys, generation=generation)
        deleted.append(
            DeletedVersion(REMOTE, step, count, 0, STALE_GENERATION, generation)
        )
    return tuple(deleted)


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
        run_id: str | None = None,
        staging_dir: str | Path | None = None,
        abort_stale_mpu: bool = True,
        mpu_timeout_seconds: float = DEFAULT_MPU_TIMEOUT_SECONDS,
    ) -> None:
        if interval <= 0:
            raise ValueError(f"interval must be > 0, got {interval}")
        self.local_root = Path(local_root)
        self.remote = remote
        self.rank = int(rank)
        self.interval = float(interval)
        self.dry_run = dry_run
        self.keep_local = int(keep_local)
        self.run_id = run_id
        self.staging_dir = staging_dir
        self.abort_stale_mpu = abort_stale_mpu
        self.mpu_timeout_seconds = float(mpu_timeout_seconds)
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
            run_id=self.run_id,
            staging_dir=self.staging_dir,
            abort_stale_mpu=self.abort_stale_mpu,
            mpu_timeout_seconds=self.mpu_timeout_seconds,
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
    "AbortedUpload",
    "COMMITTED",
    "DEFAULT_KEEP_LOCAL_VERSIONS",
    "DEFAULT_KEEP_VERSIONS",
    "DEFAULT_MPU_TIMEOUT_SECONDS",
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
    "cleanup_stale_multipart_uploads",
    "collect",
    "delete_local_version",
    "plan",
    "plan_retention",
    "reclaim_stale_generations",
    "retain_remote_versions",
]
