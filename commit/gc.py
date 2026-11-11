"""本地层垃圾回收：判据是纯函数，执行器在下面（:func:`collect`）。

回收规则只有一条不变量：**step 是单调推进的**。由此得到两条判据，都不需要租约、
时间戳或尝试号：

1. 远端的 ``step_N/rank_<r>/COMMIT`` 存在 ⇒ 本 rank 的分片已经在远端 ⇒ 本地
   ``step_N/`` 是纯冗余，可以删；
2. 存在比 N 更新的**版本** ``COMMIT`` ⇒ 训练已经越过第 N 代 ⇒ 本地 ``step_N/``
   （无论有没有本地 ``COMMIT``）与远端整个 ``step_N/`` 前缀都是残片，可以删。

反过来，「最新且未完成」一律保留 —— 它可能就是**正在写**（本地）或**正在传**
（远端）的那一代，所以不需要任何「写入者是否活着」的外部判断：

- 兄弟 rank 正在写的目录一定是最新 step，`rank_<r>/COMMIT` 不成立、也没有更新的
  版本提交，两条判据都不命中；
- rank 0 崩在写完版本 ``COMMIT`` 之前留下的完整前缀，会在下一次保存成功后立刻
  变成「非最新」，被判据 2 收掉。回收延迟上界是一个 checkpoint 间隔。

**降级**：本地判据依赖远端状态，所以远端不可达时本地一个都不删 —— 「查不到」
不能当成「可以删」。

删除顺序与写入镜像：先删提交点（本地 ``COMMIT`` / 远端 ``rank_<r>/COMMIT``），
再删数据，最后删空目录。判据只看提交点在不在，所以删到一半被打断，重跑一次收敛。
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

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
"""远端已提交的版本：不做保留策略，永不删除。"""


@dataclass(frozen=True, slots=True)
class GcCandidate:
    """一个待删的版本。``location`` 是 ``"local"`` 或 ``"remote"``。"""

    location: str
    step: int
    reason: str


@dataclass(frozen=True, slots=True)
class GcPlan:
    """一次回收的决策结果：删哪些、留哪些、为什么。"""

    candidates: tuple[GcCandidate, ...]
    kept: tuple[tuple[str, int, str], ...]
    """``(location, step, reason)``：保留的版本与理由。"""

    versions: frozenset[int]
    """本次看到的、带版本 ``COMMIT`` 的 step。"""

    def steps_for(self, location: str) -> tuple[int, ...]:
        return tuple(c.step for c in self.candidates if c.location == location)

    def reasons(self) -> Mapping[tuple[str, int], str]:
        return {(c.location, c.step): c.reason for c in self.candidates}


def plan(
    *,
    versions: Iterable[int],
    rank_commits: Iterable[tuple[int, int]],
    prefixes: Iterable[int],
    local_steps: Iterable[int],
    rank: int,
) -> GcPlan:
    """决定删哪些版本。**纯函数**：不碰 IO、不看时间、不看进程状态。

    ``versions``:     远端有版本 ``COMMIT`` 的 step（L3）
    ``rank_commits``: 远端有分片 ``COMMIT`` 的 ``(step, rank)``（L2）
    ``prefixes``:     远端存在 ``step_N/`` 前缀的 step
    ``local_steps``:  本地 root 下存在 ``step_N/`` 目录的 step
    ``rank``:         本进程的 rank 号（本地判据只看自己的分片提交点）
    """
    known = frozenset(int(step) for step in versions)
    newest = max(known) if known else None
    uploaded = {int(step) for step, owner in rank_commits if int(owner) == int(rank)}
    all_prefixes = {int(step) for step in prefixes}
    local = {int(step) for step in local_steps}

    def superseded(step: int) -> bool:
        return newest is not None and newest > step

    candidates: list[GcCandidate] = []
    kept: list[tuple[str, int, str]] = []

    for step in sorted(local):
        if step in uploaded:
            candidates.append(GcCandidate(LOCAL, step, UPLOADED))
        elif superseded(step):
            candidates.append(GcCandidate(LOCAL, step, SUPERSEDED))
        else:
            kept.append((LOCAL, step, NOT_UPLOADED))

    for step in sorted(all_prefixes):
        if step in known:
            kept.append((REMOTE, step, COMMITTED))
        elif superseded(step):
            candidates.append(GcCandidate(REMOTE, step, SUPERSEDED))
        else:
            kept.append((REMOTE, step, NO_NEWER_VERSION))

    return GcPlan(
        candidates=tuple(candidates),
        kept=tuple(kept),
        versions=known,
    )


# --- 执行器 ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeletedVersion:
    """一个被回收的版本。``bytes`` 对远端记 0（``list`` 不带对象大小）。"""

    location: str
    step: int
    objects: int
    bytes: int
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

    def steps_for(self, location: str) -> tuple[int, ...]:
        return tuple(item.step for item in self.deleted if item.location == location)


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


def _scan_remote(
    keys: Iterable[str],
) -> tuple[set[int], set[tuple[int, int]], set[int]]:
    """从一次 ``list("")`` 的结果里解析出三层信息。

    ``step_N/COMMIT`` 是版本提交点，``step_N/rank_r/COMMIT`` 是分片提交点，
    其余 ``step_N/...`` 只用来确认前缀存在。
    """
    versions: set[int] = set()
    rank_commits: set[tuple[int, int]] = set()
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
        elif (
            len(parts) == 3
            and parts[2] == COMMIT_FILE
            and parts[1].startswith("rank_")
        ):
            try:
                rank_commits.add((step, int(parts[1][len("rank_") :])))
            except ValueError:
                continue
    return versions, rank_commits, prefixes


def _delete_remote_prefix(remote: StorageBackend, step: int, keys: Iterable[str]) -> int:
    """删远端整个 ``step_N/`` 前缀：先删分片提交点，再删其余对象。"""
    prefix = f"{version_key(step)}/"
    mine = sorted(key for key in keys if key.startswith(prefix))
    commits = [key for key in mine if key.count("/") == 2 and key.endswith(f"/{COMMIT_FILE}")]
    rest = [key for key in mine if key not in set(commits)]
    for key in commits + rest:
        remote.delete(key)
    _prune_remote_dirs(remote, step)
    return len(mine)


def _prune_remote_dirs(remote: StorageBackend, step: int) -> None:
    """本地盘后端会把空目录留下来（对象存储没有目录概念），顺手收掉。"""
    if not isinstance(remote, LocalStorage):
        return
    directory = remote.path_for(version_key(step))
    for child in sorted(directory.glob("*"), reverse=True):
        if child.is_dir():
            with contextlib.suppress(OSError):  # 非空 = 有人还在写，留着
                child.rmdir()
    with contextlib.suppress(OSError):
        directory.rmdir()


def collect(
    local_root: str | Path,
    remote: StorageBackend | None,
    *,
    rank: int,
    dry_run: bool = True,
) -> GcReport:
    """跑一次回收：一次 ``list`` + 一次目录扫描 + 按 :func:`plan` 的决策删除。

    ``dry_run=True``（默认）只给出计划，不动任何文件。

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

    versions, rank_commits, prefixes = _scan_remote(keys)
    result = plan(
        versions=versions,
        rank_commits=rank_commits,
        prefixes=prefixes,
        local_steps=_scan_local(root),
        rank=rank,
    )

    deleted: list[DeletedVersion] = []
    if not dry_run:
        for candidate in result.candidates:
            if candidate.location == REMOTE:
                count = _delete_remote_prefix(remote, candidate.step, keys)
                deleted.append(
                    DeletedVersion(REMOTE, candidate.step, count, 0, candidate.reason)
                )
            else:
                record = delete_local_version(root, candidate.step)
                deleted.append(replace(record, reason=candidate.reason))

    return GcReport(
        dry_run=dry_run,
        planned=result.candidates,
        deleted=tuple(deleted),
        skipped=result.kept,
        reclaimed_bytes=sum(item.bytes for item in deleted),
        versions=result.versions,
    )


class GcWorker:
    """周期性回收的守护线程：**启动时先跑一次**，之后每 ``interval`` 秒一次。

    - 启动时那次是主力：它收掉上一次运行崩溃留下的残目录（SIGKILL 的产物）。
    - 运行期那次只处理「已经上传完」「已经被更新的版本越过」两类，正在写/正在传的
      目录天然不命中，所以不需要和保存路径做任何协调。
    - 单次失败（远端抖动、IO 错误）只记进 ``last_error``，线程继续跑；远端整体不可达
      时 :func:`collect` 自己是空操作，不会误删。
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
    ) -> None:
        if interval <= 0:
            raise ValueError(f"interval must be > 0, got {interval}")
        self.local_root = Path(local_root)
        self.remote = remote
        self.rank = int(rank)
        self.interval = float(interval)
        self.dry_run = dry_run
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
        report = collect(self.local_root, self.remote, rank=self.rank, dry_run=self.dry_run)
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
    "DeletedVersion",
    "GcCandidate",
    "GcPlan",
    "GcReport",
    "GcWorker",
    "LOCAL",
    "NOT_UPLOADED",
    "NO_NEWER_VERSION",
    "REMOTE",
    "SUPERSEDED",
    "UPLOADED",
    "collect",
    "delete_local_version",
    "plan",
]
