"""对象存储端的代际（generation）与 fence：一个 CAS 寄存器 + 三个检查点。

**代际**是「某个 step 的一次保存尝试」：``gen_0007-1f3a9c2b`` = 本 step 第 7 次
尝试 + 写者 token。**fence token** 是 ``(step, seq)``，字典序可比 —— ``seq`` 由
``step_N/GEN`` 这个 CAS 寄存器的成功顺序线性化，不看时钟、不靠租约。

四条不变量（``docs/generation.md`` 有完整规格与失败模式矩阵）：

1. **物理隔离**：不同代的产物落在不同 key（``step_N/<gen>/rank_r/...``），
   重试永不改写已提交版本的对象；
2. **L3 唯一权威**：恢复端只读 ``step_N/COMMIT`` 及其声明的 generation，
   **不读 GEN** —— GEN 被删被改，已提交版本照常可恢复；
3. **fail-closed**：fence 检查「查不到 / 对不上 / 读失败」一律推定为落后，
   写者停止并抛 :class:`FencedError`，绝不降级为放行；
4. **无租约**：没有心跳、TTL、mtime 阈值。``acquire`` 只把 GEN 往前推一格。

三个检查点（写路径）：

======  ========  =============================================  ============
点      谁        动作                                           落后时
======  ========  =============================================  ============
B       rank 0    ``acquire_generation``：CAS 推进 ``seq``        不写任何对象
L2-pre  每个 rank ``verify_holder``：读 GEN 比对 generation      不写 L2
L3-pre  rank 0    ``seal_generation`` + CAS 写 L3                不写 L3
======  ========  =============================================  ============

``acquire`` 的 CAS 是**线性化点**：所有并发写者在这里排成一个全序，``seq`` 就是
这个全序的编号。所以不存在「两个写者都以为自己赢了」的窗口 —— 这是它相对
「覆盖写 + 回读自检」的关键差别。
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, replace
from typing import Sequence

from commit.local import version_key
from core import CheckpointError, FormatError
from storage.base import ObjectNotFoundError, StorageBackend, WriteOutcome

GENERATION_FILE = "GEN"
"""本 step 当前代际指针（唯一共享写点）。"""

GENERATION_FORMAT_VERSION = 2
"""GEN / L2 / L3 的格式版本。

v1 是「只有 step、没有代际」的老布局，读侧仍接受（``generation=None`` 走旧 key
布局）；写侧只写 v2。
"""

SUPPORTED_FORMAT_VERSIONS = frozenset({1, GENERATION_FORMAT_VERSION})

OPEN = "open"
SEALED = "sealed"
COMMITTED = "committed"
POINTER_STATES = frozenset({OPEN, SEALED, COMMITTED})

DEFAULT_ACQUIRE_RETRIES = 4
"""``acquire_generation`` 的 CAS 重试上限。

超限即放弃（``FencedError``）：要么并发写者实在太多，要么端点根本没有原子执行
条件写 —— 两种都不该继续往远端写字节。
"""

_GENERATION_RE = re.compile(r"^gen_(\d{4,})-([0-9a-z]{1,32})$")

_WRITER_ALPHABET = "0123456789abcdef"


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


@dataclass(frozen=True, slots=True)
class GenerationPointer:
    """``step_N/GEN`` 的内容。

    ``etag`` 是**读出来时**服务端给的版本标识，不参与序列化：写回时要把它当作
    ``If-Match`` 的值，所以它必须与「我读到的那个 GEN」严格对应。
    """

    step: int
    generation: Generation
    ranks: tuple[int, ...]
    state: str = OPEN
    format_version: int = GENERATION_FORMAT_VERSION
    etag: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranks", tuple(int(rank) for rank in self.ranks))
        if self.format_version not in SUPPORTED_FORMAT_VERSIONS:
            raise FormatError(
                f"unsupported GEN format_version {self.format_version}, "
                f"this build understands {sorted(SUPPORTED_FORMAT_VERSIONS)}"
            )
        if self.state not in POINTER_STATES:
            raise FormatError(
                f"unsupported GEN state {self.state!r}, expected one of "
                f"{sorted(POINTER_STATES)}"
            )
        if not self.ranks:
            raise FormatError("GEN must declare at least one rank")
        if len(set(self.ranks)) != len(self.ranks):
            raise FormatError(f"duplicate ranks in GEN: {self.ranks}")
        if any(rank < 0 for rank in self.ranks):
            raise FormatError(f"ranks must be >= 0, got {self.ranks}")

    @property
    def seq(self) -> int:
        return self.generation.seq

    def with_state(self, state: str) -> "GenerationPointer":
        return replace(self, state=state)

    def dumps(self) -> str:
        return json.dumps(
            {
                "format_version": self.format_version,
                "step": self.step,
                "generation": self.generation.id,
                "seq": self.generation.seq,
                "writer": self.generation.writer,
                "ranks": list(self.ranks),
                "state": self.state,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

    @classmethod
    def loads(cls, text: str, *, etag: str = "") -> "GenerationPointer":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise FormatError(f"GEN is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise FormatError(f"GEN must be a JSON object, got {type(data).__name__}")
        ranks = data.get("ranks")
        if not isinstance(ranks, list):
            raise FormatError("GEN 'ranks' must be a list")
        try:
            step = int(data["step"])
            generation = Generation.parse(str(data["generation"]), step=step)
            return cls(
                step=step,
                generation=generation,
                ranks=tuple(int(rank) for rank in ranks),
                state=str(data.get("state", OPEN)),
                format_version=int(data.get("format_version", GENERATION_FORMAT_VERSION)),
                etag=etag,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FormatError(f"GEN is malformed: {exc}") from exc


def pointer_key(step: int) -> str:
    """``step_42/GEN``。"""
    return f"{version_key(step)}/{GENERATION_FILE}"


def next_seq(keys: Sequence[str], *, current: int = 0) -> int:
    """下一个序号：``max(现有 gen 前缀的 seq, current) + 1``。

    **纯函数**。同时看 ``list`` 结果和 GEN 里的 ``seq``：GEN 被外部删掉时，已有
    代目录仍能把序号顶住，``seq`` 不会回退 —— 而 ``seq`` 一旦回退，L3 的 fence
    比较就失效了。
    """
    best = int(current)
    for key in keys:
        head = key.split("/", 1)[0]
        m = _GENERATION_RE.match(head)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def read_pointer(storage: StorageBackend, step: int) -> GenerationPointer | None:
    """读 GEN；不存在返回 ``None``（不是「拿着旧代继续」，调用方必须自己决定）。"""
    try:
        payload = storage.read(pointer_key(step))
    except ObjectNotFoundError:
        return None
    etag = _etag_of(storage, pointer_key(step))
    return GenerationPointer.loads(payload.decode("utf-8"), etag=etag)


def _etag_of(storage: StorageBackend, key: str) -> str:
    head = getattr(storage, "head", None)
    if not callable(head):  # pragma: no cover - 只有非条件后端才会走到
        raise FencedError(
            f"{key}: 存储后端不提供 head()，无法用 ETag 做 CAS（写路径要求 "
            "S3Storage 的 conditional_writes=require）"
        )
    meta = head(key)
    return "" if meta is None else meta.etag


def write_pointer(
    storage: StorageBackend,
    pointer: GenerationPointer,
    *,
    if_absent: bool = False,
) -> WriteOutcome:
    """写 GEN：``if_absent`` 走 create-exclusive，否则用 ``pointer.etag`` 做 CAS。"""
    put_if_absent = getattr(storage, "put_if_absent", None)
    put_if_match = getattr(storage, "put_if_match", None)
    if not callable(put_if_absent) or not callable(put_if_match):  # pragma: no cover
        raise FencedError(
            f"{pointer_key(pointer.step)}: 存储后端没有条件写能力；"
            "写路径要求 S3Storage 的 conditional_writes=require"
        )
    payload = pointer.dumps().encode("utf-8")
    key = pointer_key(pointer.step)
    if if_absent:
        return put_if_absent(key, payload)
    if not pointer.etag:
        raise FencedError(
            f"{key}: 没有读到 GEN 的 ETag，无法做 CAS（并发写者可能刚创建/删除了它）"
        )
    return put_if_match(key, payload, pointer.etag)


def acquire_generation(
    storage: StorageBackend,
    step: int,
    *,
    ranks: Sequence[int],
    writer: str | None = None,
    retries: int = DEFAULT_ACQUIRE_RETRIES,
) -> Generation:
    """CAS 抢这个 step 的写权，返回**只属于我**的一代。

    ``seq = next_seq(list(step_N), current=现有 GEN.seq)``：既看 GEN 也看已有代目录，
    所以 GEN 被删掉时序号也不会回退。之后 ``If-None-Match``（首代）或
    ``If-Match``（后继代）推进它 —— 失败就重读重试，超限抛 :class:`FencedError`。
    """
    if retries < 1:
        raise ValueError(f"retries must be >= 1, got {retries}")
    token = writer or new_writer()
    expected = tuple(int(rank) for rank in ranks)
    if not expected:
        raise ValueError("acquire_generation 需要至少一个 rank")

    outcome: WriteOutcome | None = None
    seen: GenerationPointer | None = None
    for _ in range(retries):
        seen = read_pointer(storage, step)
        seq = next_seq(storage.list(version_key(step)), current=seen.seq if seen else 0)
        candidate = Generation(step=step, seq=seq, writer=token)
        pointer = GenerationPointer(
            step=step,
            generation=candidate,
            ranks=expected,
            state=OPEN,
            etag="" if seen is None else seen.etag,
        )
        outcome = write_pointer(storage, pointer, if_absent=seen is None)
        if outcome.written:
            return candidate
    raise FencedError(
        f"step {step}: {retries} 次 CAS 内没拿到写权"
        f"（最后一次 {outcome.value if outcome else '未执行'}）；"
        "并发写者过多，或端点没有原子执行条件写（见 docs/generation.md 的端点探针）"
    )


def verify_holder(storage: StorageBackend, generation: Generation) -> GenerationPointer:
    """读证明：``generation`` 是否仍是这个 step 的当前代际。

    这是 L2-pre 与 L3-pre 共用的判据。落后就抛 :class:`FencedError` —— 调用方
    因此**不会**写自己的分片提交点，这一代永远凑不齐，也不会被 rank 0 提升。
    """
    pointer = read_pointer(storage, generation.step)
    if pointer is None:
        raise FencedError(
            f"{pointer_key(generation.step)}: GEN 不存在，{generation.id} 不再持有写权"
        )
    if pointer.generation.id != generation.id:
        raise FencedError(
            f"step {generation.step}: 当前代际是 {pointer.generation.id}"
            f"（state={pointer.state}），{generation.id} 已被 fence"
        )
    return pointer


def seal_generation(
    storage: StorageBackend, generation: Generation, *, ranks: Sequence[int]
) -> GenerationPointer:
    """CAS 把 GEN 推到 ``sealed``：这一代不再接受新写者，准备提交。

    幂等：已经 sealed / committed 且 generation 相同就直接返回（rank 0 崩在写 L3
    之前、重跑 finalize 时会走这条）。
    """
    pointer = verify_holder(storage, generation)
    if pointer.state in (SEALED, COMMITTED):
        return pointer
    expected = tuple(int(rank) for rank in ranks)
    if pointer.ranks != expected:
        raise FormatError(
            f"step {generation.step}: GEN 声明 ranks {pointer.ranks}，"
            f"finalize 传入 {expected}"
        )
    outcome = write_pointer(storage, pointer.with_state(SEALED))
    if outcome is not WriteOutcome.SWAPPED:
        raise FencedError(
            f"step {generation.step}: seal GEN 的 CAS 失败（{outcome.value}），"
            f"{generation.id} 在 seal 期间被接管"
        )
    return pointer.with_state(SEALED)


def mark_committed(
    storage: StorageBackend, generation: Generation
) -> GenerationPointer | None:
    """尽力把 GEN 推到 ``committed``（纯诊断，失败不影响提交成立）。"""
    try:
        pointer = read_pointer(storage, generation.step)
        if pointer is None or pointer.generation.id != generation.id:
            return None
        if pointer.state == COMMITTED:
            return pointer
        outcome = write_pointer(storage, pointer.with_state(COMMITTED))
        return pointer.with_state(COMMITTED) if outcome.written else None
    except Exception:  # pragma: no cover - 诊断状态不参与正确性
        return None


def resolve_generation(
    storage: StorageBackend,
    step: int,
    *,
    ranks: Sequence[int] | None = None,
    writer: str | None = None,
) -> Generation:
    """把「没显式给代际」的调用收敛到一个确定的代际。

    - GEN 已存在：返回它的 generation（**单 rank 便利路径**）。多 rank 场景必须
      显式 ``begin_upload``：复用别人开的代会把两次尝试的分片混在一起，这正是
      代际要防的那个 bug。
    - GEN 不存在：直接 :func:`acquire_generation` 开第一代。
    """
    pointer = read_pointer(storage, step)
    if pointer is not None:
        expected = tuple(int(rank) for rank in ranks) if ranks is not None else None
        if expected is not None and pointer.ranks != expected:
            raise FormatError(
                f"step {step}: GEN 声明的 ranks {pointer.ranks} 与传入的 {expected} 不一致；"
                f"多 rank 场景请先调 begin_upload() 分配代际（当前代际 {pointer.generation.id}）"
            )
        return pointer.generation
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
    "COMMITTED",
    "DEFAULT_ACQUIRE_RETRIES",
    "FencedError",
    "GENERATION_FILE",
    "GENERATION_FORMAT_VERSION",
    "Generation",
    "GenerationPointer",
    "OPEN",
    "POINTER_STATES",
    "SEALED",
    "SUPPORTED_FORMAT_VERSIONS",
    "acquire_generation",
    "as_generation",
    "mark_committed",
    "new_writer",
    "next_seq",
    "pointer_key",
    "read_pointer",
    "resolve_generation",
    "seal_generation",
    "verify_holder",
    "write_pointer",
]
