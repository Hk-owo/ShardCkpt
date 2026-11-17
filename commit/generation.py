"""对象存储端的代际（generation）与 fence：单对象 ``step_N/STATUS`` CAS 状态机。

**代际**是「某个 step 的一次保存尝试」：``gen_0007-1f3a9c2b`` = 本 step 第 7 次
尝试 + 写者 token。**fence token** 是 ``(step, seq)``，字典序可比 —— ``seq`` 由
``step_N/STATUS`` 这个单点 CAS 寄存器的成功顺序线性化，不看时钟、不靠租约。

把写权（``active``）与版本提交指针（``committed``）合进同一个 ``step_N/STATUS``
对象后，``acquire`` 与 ``commit`` 共享同一份 ``ETag``：新代一抢走写权，旧代提交时的
``If-Match`` 在服务端原子层必然 412 失败，**不存在跨 key 并发窗口**。

四条不变量（``docs/generation.md`` 有完整规格与失败模式矩阵）：

1. **物理隔离**：不同代的产物落在不同 key（``step_N/<gen>/rank_r/...``），
   重试永不改写已提交版本的对象；
2. **``committed`` 唯一权威**：恢复端只认 ``step_N/STATUS`` 的 ``committed`` 字段
   及其声明的 generation，``committed == null`` 时哪怕 ``STATUS`` 存在也不可恢复；
   重试新代失败时 ``committed`` 保持指向上一次已提交代不变；
3. **fail-closed**：fence 检查「查不到 / 对不上 / 读失败」一律推定为落后，
   写者停止并抛 :class:`FencedError`，绝不降级为放行；
4. **无租约**：没有心跳、TTL、mtime 阈值。``acquire`` 只把 ``active.seq`` 往前推一格。

三个检查点（写路径）：

======  ========  =============================================  ============
点      谁        动作                                           落后时
======  ========  =============================================  ============
B       rank 0    ``acquire_generation``：CAS 推进 ``active``     不写任何对象
L2-pre  每个 rank ``verify_holder``：读 STATUS 比对 ``active``    不写 L2
L3      rank 0    ``commit_status``：单次 CAS 写入 ``committed``  不提交版本
======  ========  =============================================  ============
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

from commit.local import version_key
from core import CheckpointError, FormatError
from storage.base import ObjectNotFoundError, StorageBackend, WriteOutcome

STATUS_FILE = "STATUS"
"""本 step 唯一共享状态对象（同时持有写权 ``active`` 与提交指针 ``committed``）。"""

STATUS_FORMAT_VERSION = 3
"""``step_N/STATUS`` 的格式版本（单点 STATUS 状态机，不兼容旧双对象/v1 远端布局）。"""

DEFAULT_ACQUIRE_RETRIES = 4
"""``acquire_generation`` 的 CAS 重试上限。

超限即放弃（``FencedError``）：要么并发写者实在太多，要么端点根本没有原子执行
条件写 —— 两种都不该继续往远端写字节。
"""

_GENERATION_RE = re.compile(r"^gen_(\d{4,})-([0-9a-z]{1,32})$")


class FencedError(CheckpointError):
    """写者被 fence：它持有的代际已不是当前代际，或没能抢到写权。

    不继承 ``StorageError``：fence 是**写协议的裁决**，不是存储层故障。调用方要
    区分「远端挂了，可以重试同一代」与「我过时了，必须重新 begin」—— 后者重试
    同一代永远失败。
    """


@dataclass(frozen=True, slots=True)
class Generation:
    """一次保存尝试的身份：``gen_0007-1f3a9c2b``。"""

    step: int
    seq: int
    writer: str

    def __post_init__(self) -> None:
        if self.step < 0:
            raise FormatError(f"generation step must be >= 0, got {self.step}")
        if self.seq < 1:
            raise FormatError(f"generation seq must be >= 1, got {self.seq}")
        if not self.writer or not re.fullmatch(r"[0-9a-z]{1,32}", self.writer):
            raise FormatError(
                f"generation writer must be 1-32 chars of [0-9a-z], got {self.writer!r}"
            )

    @property
    def id(self) -> str:
        """key 里的一段：``gen_0007-1f3a9c2b``。"""
        return f"gen_{self.seq:04d}-{self.writer}"

    def __str__(self) -> str:  # pragma: no cover - 调试用
        return self.id

    def next(self, *, writer: str | None = None) -> "Generation":
        """同一 step 的下一代。"""
        return Generation(step=self.step, seq=self.seq + 1, writer=writer or new_writer())

    @classmethod
    def new(cls, step: int, seq: int, *, writer: str | None = None) -> "Generation":
        return cls(step=step, seq=seq, writer=writer or new_writer())

    @classmethod
    def parse(cls, text: str, *, step: int) -> "Generation":
        m = _GENERATION_RE.match(text.strip())
        if not m:
            raise FormatError(f"invalid generation id {text!r}; expected 'gen_0007-1f3a9c2b'")
        return cls(step=step, seq=int(m.group(1)), writer=m.group(2))


def new_writer() -> str:
    """一次进程内稳定的写者 token（调用方自己缓存，不要每个对象换一个）。"""
    return uuid.uuid4().hex[:8]


def _validate_ranks(ranks: Sequence[int], *, label: str) -> tuple[int, ...]:
    normalized = tuple(int(rank) for rank in ranks)
    if not normalized:
        raise FormatError(f"{label} must declare at least one rank")
    if len(set(normalized)) != len(normalized):
        raise FormatError(f"duplicate ranks in {label}: {normalized}")
    if any(rank < 0 for rank in normalized):
        raise FormatError(f"ranks in {label} must be >= 0, got {normalized}")
    return normalized


@dataclass(frozen=True, slots=True)
class CommittedState:
    """``step_N/STATUS`` 里已提交版本的权威指针。"""

    generation: Generation
    ranks: tuple[int, ...]
    layout: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ranks", _validate_ranks(self.ranks, label="STATUS.committed")
        )
        object.__setattr__(
            self,
            "layout",
            {str(k): int(v) for k, v in (self.layout or {}).items()},
        )

    @property
    def step(self) -> int:
        return self.generation.step

    @property
    def seq(self) -> int:
        return self.generation.seq

    def to_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation.id,
            "seq": self.generation.seq,
            "ranks": list(self.ranks),
            "layout": {key: int(value) for key, value in sorted(self.layout.items())},
        }

    @classmethod
    def from_dict(cls, data: object, *, step: int) -> "CommittedState":
        if not isinstance(data, dict):
            raise FormatError(
                f"STATUS.committed must be a JSON object, got {type(data).__name__}"
            )
        ranks = data.get("ranks")
        if not isinstance(ranks, list):
            raise FormatError("STATUS.committed 'ranks' must be a list")
        layout = data.get("layout", {})
        if not isinstance(layout, dict):
            raise FormatError("STATUS.committed 'layout' must be an object")
        try:
            generation = Generation.parse(str(data["generation"]), step=step)
            seq = int(data["seq"])
            if generation.seq != seq:
                raise FormatError(
                    f"STATUS.committed generation {generation.id} 与 seq {seq} 不一致"
                )
            return cls(
                generation=generation,
                ranks=tuple(int(rank) for rank in ranks),
                layout={str(k): int(v) for k, v in layout.items()},
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FormatError(f"STATUS.committed is malformed: {exc}") from exc


@dataclass(frozen=True, slots=True)
class StepStatus:
    """``step_N/STATUS`` 的内容（单对象 CAS 状态机）。

    ``etag`` 是**读出来时**服务端给的版本标识，不参与序列化：写回时要把它当作
    ``If-Match`` 的值，所以它必须与「我读到的那个 STATUS」严格对应。
    """

    step: int
    generation: Generation
    ranks: tuple[int, ...]
    committed: CommittedState | None = None
    format_version: int = STATUS_FORMAT_VERSION
    etag: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if self.format_version != STATUS_FORMAT_VERSION:
            raise FormatError(
                f"unsupported STATUS format_version {self.format_version}, "
                f"this build understands {STATUS_FORMAT_VERSION}"
            )
        if self.step < 0:
            raise FormatError(f"STATUS step must be >= 0, got {self.step}")
        if self.generation.step != self.step:
            raise FormatError(
                f"STATUS active generation {self.generation.id} belongs to step "
                f"{self.generation.step}, expected {self.step}"
            )
        object.__setattr__(
            self, "ranks", _validate_ranks(self.ranks, label="STATUS.active")
        )
        if self.committed is not None:
            if self.committed.step != self.step:
                raise FormatError(
                    f"STATUS committed generation belongs to step "
                    f"{self.committed.step}, expected {self.step}"
                )
            if self.committed.seq > self.generation.seq:
                raise FormatError(
                    f"STATUS committed seq {self.committed.seq} exceeds active seq "
                    f"{self.generation.seq}"
                )

    @property
    def seq(self) -> int:
        return self.generation.seq

    @property
    def is_committed(self) -> bool:
        return self.committed is not None

    @property
    def state(self) -> str:
        """只读诊断标签：当前代是否已提交（不用于写锁）。"""
        if self.committed is not None and self.committed.generation.id == self.generation.id:
            return "committed"
        return "open"

    def with_committed(self, committed: CommittedState) -> "StepStatus":
        return replace(self, committed=committed)

    def dumps(self) -> str:
        return json.dumps(
            {
                "format_version": self.format_version,
                "step": self.step,
                "active": {
                    "generation": self.generation.id,
                    "seq": self.generation.seq,
                    "writer": self.generation.writer,
                    "ranks": list(self.ranks),
                },
                "committed": None if self.committed is None else self.committed.to_dict(),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

    @classmethod
    def loads(cls, text: str, *, etag: str = "") -> "StepStatus":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise FormatError(f"STATUS is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise FormatError(f"STATUS must be a JSON object, got {type(data).__name__}")
        active = data.get("active")
        if not isinstance(active, dict):
            raise FormatError("STATUS missing 'active' object")
        ranks = active.get("ranks")
        if not isinstance(ranks, list):
            raise FormatError("STATUS.active 'ranks' must be a list")
        raw_committed = data.get("committed")
        try:
            step = int(data["step"])
            generation = Generation.parse(str(active["generation"]), step=step)
            seq = int(active["seq"])
            if generation.seq != seq:
                raise FormatError(
                    f"STATUS.active generation {generation.id} 与 seq {seq} 不一致"
                )
            committed = (
                None
                if raw_committed is None
                else CommittedState.from_dict(raw_committed, step=step)
            )
            return cls(
                step=step,
                generation=generation,
                ranks=tuple(int(rank) for rank in ranks),
                committed=committed,
                format_version=int(data["format_version"]),
                etag=etag,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FormatError(f"STATUS is malformed: {exc}") from exc


def status_key(step: int) -> str:
    """``step_42/STATUS``。"""
    return f"{version_key(step)}/{STATUS_FILE}"


def next_seq(keys: Sequence[str], *, current: int = 0) -> int:
    """下一个序号：``max(现有 gen 前缀的 seq, current) + 1``。

    **纯函数**。同时看 ``list`` 结果和 STATUS 里的 ``seq``：STATUS 被外部删掉时，
    已有代目录仍能把序号顶住，``seq`` 不会回退 —— 而 ``seq`` 一旦回退，fence
    比较就失效了。
    """
    best = int(current)
    for key in keys:
        head = key.split("/", 1)[0]
        m = _GENERATION_RE.match(head)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def read_status(storage: StorageBackend, step: int) -> StepStatus | None:
    """读 ``step_N/STATUS``；不存在返回 ``None``（不是「拿着旧代继续」，由调用方裁决）。"""
    key = status_key(step)
    try:
        payload = storage.read(key)
    except ObjectNotFoundError:
        return None
    etag = _etag_of(storage, key)
    return StepStatus.loads(payload.decode("utf-8"), etag=etag)


def _etag_of(storage: StorageBackend, key: str) -> str:
    head = getattr(storage, "head", None)
    if not callable(head):  # pragma: no cover - 只读兜底；写路径在 write_status 里拒绝无 ETag
        return ""
    meta = head(key)
    return "" if meta is None else meta.etag


def write_status(
    storage: StorageBackend,
    status: StepStatus,
    *,
    if_absent: bool = False,
) -> WriteOutcome:
    """写 ``step_N/STATUS``：``if_absent`` 走 create-exclusive，否则用 ``status.etag`` 做 CAS。"""
    put_if_absent = getattr(storage, "put_if_absent", None)
    put_if_match = getattr(storage, "put_if_match", None)
    if not callable(put_if_absent) or not callable(put_if_match):  # pragma: no cover
        raise FencedError(
            f"{status_key(status.step)}: 存储后端没有条件写能力；"
            "写路径要求 S3Storage 的 conditional_writes=require"
        )
    payload = status.dumps().encode("utf-8")
    key = status_key(status.step)
    if if_absent:
        return put_if_absent(key, payload)
    if not status.etag:
        raise FencedError(
            f"{key}: 没有读到 STATUS 的 ETag，无法做 CAS（并发写者可能刚创建/删除了它）"
        )
    return put_if_match(key, payload, status.etag)


def acquire_generation(
    storage: StorageBackend,
    step: int,
    *,
    ranks: Sequence[int],
    writer: str | None = None,
    retries: int = DEFAULT_ACQUIRE_RETRIES,
) -> Generation:
    """CAS 抢这个 step 的写权，返回**只属于我**的一代。

    ``seq = next_seq(list(step_N), current=现有 STATUS.seq)``：既看 STATUS 也看已有
    代目录，所以 STATUS 被删掉时序号也不会回退。抢写权只更新 ``active``，**原样
    保留**已有的 ``committed`` 指针，重试半途失败不会影响上一次已提交版本。
    """
    if retries < 1:
        raise ValueError(f"retries must be >= 1, got {retries}")
    token = writer or new_writer()
    expected = tuple(int(rank) for rank in ranks)
    if not expected:
        raise ValueError("acquire_generation 需要至少一个 rank")

    outcome: WriteOutcome | None = None
    for _ in range(retries):
        seen = read_status(storage, step)
        current_seq = (
            max(seen.seq, seen.committed.seq if seen.committed else 0) if seen else 0
        )
        seq = next_seq(storage.list(version_key(step)), current=current_seq)
        candidate = Generation(step=step, seq=seq, writer=token)
        status = StepStatus(
            step=step,
            generation=candidate,
            ranks=expected,
            committed=None if seen is None else seen.committed,
            etag="" if seen is None else seen.etag,
        )
        outcome = write_status(storage, status, if_absent=seen is None)
        if outcome.written:
            return candidate
    raise FencedError(
        f"step {step}: {retries} 次 CAS 内没拿到写权"
        f"（最后一次 {outcome.value if outcome else '未执行'}）；"
        "并发写者过多，或端点没有原子执行条件写（见 docs/generation.md 的端点探针）"
    )


def verify_holder(storage: StorageBackend, generation: Generation) -> StepStatus:
    """读证明：``generation`` 是否仍是这个 step 的 ``active`` 代际。

    落后就抛 :class:`FencedError` —— 调用方因此**不会**写自己的分片提交点，这一代
    永远凑不齐，也不会提交成功。
    """
    status = read_status(storage, generation.step)
    if status is None:
        raise FencedError(
            f"{status_key(generation.step)}: STATUS 不存在，{generation.id} 不再持有写权"
        )
    if status.generation.id != generation.id:
        raise FencedError(
            f"step {generation.step}: 当前代际是 {status.generation.id}"
            f"（{status.state}），{generation.id} 已被 fence"
        )
    return status


def commit_status(
    storage: StorageBackend,
    status: StepStatus,
    committed: CommittedState,
) -> CommittedState:
    """单点 CAS 提交：用 ``status.etag`` 原子更新 ``STATUS.committed``。

    因为 ``acquire_generation`` 与 ``commit_status`` 写的是同一个对象，只要新代在
    期间抢走了写权，``status.etag`` 就已过期，服务端必然返回 412，不可能出现
    「旧代在并发窗口后偷跑提交」的问题。
    """
    outcome = write_status(storage, status.with_committed(committed))
    if outcome is WriteOutcome.SWAPPED:
        return committed

    latest = read_status(storage, status.step)
    if (
        latest is not None
        and latest.committed is not None
        and latest.committed.generation.id == committed.generation.id
    ):
        if latest.committed.ranks != committed.ranks:
            raise FormatError(
                f"step {status.step}: 已提交的 STATUS 声明 ranks "
                f"{latest.committed.ranks}，本次传入 {committed.ranks}"
            )
        return latest.committed

    raise FencedError(
        f"step {status.step}: commit STATUS 的 CAS 失败（{outcome.value}），"
        f"{committed.generation.id} 已被接管"
    )


def resolve_generation(
    storage: StorageBackend,
    step: int,
    *,
    ranks: Sequence[int] | None = None,
    writer: str | None = None,
) -> Generation:
    """把「没显式给代际」的调用收敛到一个确定的代际。

    - STATUS 已存在：返回它的 active generation（**单 rank 便利路径**）。多 rank
      场景必须显式 ``begin_upload``：复用别人开的代会把两次尝试的分片混在一起。
    - STATUS 不存在：直接 :func:`acquire_generation` 开第一代。
    """
    status = read_status(storage, step)
    if status is not None:
        expected = tuple(int(rank) for rank in ranks) if ranks is not None else None
        if expected is not None and status.ranks != expected:
            raise FormatError(
                f"step {step}: STATUS 声明的 ranks {status.ranks} 与传入的 {expected} 不一致；"
                f"多 rank 场景请先调 begin_upload() 分配代际（当前代际 {status.generation.id}）"
            )
        return status.generation
    return acquire_generation(storage, step, ranks=ranks or (0,), writer=writer)


def as_generation(spec: "Generation | str | None", *, step: int) -> "Generation | None":
    """把外部传进来的「代际」统一成 :class:`Generation`。"""
    if spec is None:
        return None
    if isinstance(spec, Generation):
        if spec.step != step:
            raise FormatError(
                f"generation {spec.id} 属于 step {spec.step}，不能用于 step {step}"
            )
        return spec
    if not isinstance(spec, str):
        raise TypeError(f"generation must be Generation / str / None, got {type(spec).__name__}")
    return Generation.parse(spec, step=step)


__all__ = [
    "CommittedState",
    "DEFAULT_ACQUIRE_RETRIES",
    "FencedError",
    "Generation",
    "STATUS_FILE",
    "STATUS_FORMAT_VERSION",
    "StepStatus",
    "acquire_generation",
    "as_generation",
    "commit_status",
    "new_writer",
    "next_seq",
    "read_status",
    "resolve_generation",
    "status_key",
    "verify_holder",
    "write_status",
]
