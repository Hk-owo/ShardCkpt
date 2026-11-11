"""本地层的提交协议：``.metadata`` → ``COMMIT`` → ``latest``。

一个 checkpoint 只有「完整可用」和「不存在」两种状态，所以顺序固定为：

1. 各 rank 写数据文件并 fsync（由 storage 层保证，`write()` 返回即已落盘）
2. 写 ``.metadata``（索引）并 fsync
3. 写 ``COMMIT``：临时文件 → rename（原子）→ 目录 fsync
4. 删旧版本时先删 ``COMMIT`` 再删数据（与写入顺序镜像）

恢复端只认带 ``COMMIT`` 的目录：没有 COMMIT 的版本一律视为未完成。

没有 ``latest`` 指针：``step=None`` 的答案恒等于「所有带 COMMIT 的版本里最大的
step」。指针省下的只是一次 ``list``（反正不读对象内容），却会带来「指针失效就
恢复不了」「删版本要维护指针」「谁能写指针」三类问题，所以直接扫描。

``COMMIT`` 只记 ``.metadata`` 的**字节数**，不记校验和：原子 rename/PUT 已经排除
「半写」，剩下的「长度不变、内容被改坏」由 JSON 解析 + ``CheckpointIndex.validate()``
兜住结构性问题，而字节级正确性交给上传时的对象级校验和（见 storage/object.py）。
旧格式里的 ``metadata.crc32`` 仍能被解析，只是不再校验，也写不回去。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping

from core import FormatError
from core.index import META_FILE, CheckpointIndex
from storage.base import ObjectNotFoundError, StorageBackend

COMMIT_FILE = "COMMIT"
COMMIT_TEMP = "COMMIT.tmp"
COMMIT_FORMAT_VERSION = 1


def version_key(step: int) -> str:
    """版本目录名：``step_12000``。"""
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    return f"step_{step}"


@dataclass(frozen=True, slots=True)
class CommitRecord:
    """``COMMIT`` 的内容：恢复端据此确认数据完整，不需要扫全部对象。"""

    step: int
    metadata_bytes: int
    files: Mapping[str, int] = field(default_factory=dict)
    """版本目录内的文件清单：文件名 -> 字节数（含 O_DIRECT 尾块补齐）。"""

    metadata_crc32: str | None = None
    """旧格式里的 ``.metadata`` 校验和；只解析、不再生成也不再校验。"""

    format_version: int = COMMIT_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != COMMIT_FORMAT_VERSION:
            raise FormatError(
                f"unsupported COMMIT format_version {self.format_version}, "
                f"this build understands {COMMIT_FORMAT_VERSION}"
            )
        if self.step < 0:
            raise FormatError(f"step must be >= 0, got {self.step}")
        object.__setattr__(self, "files", dict(self.files))

    def dumps(self) -> str:
        metadata: dict[str, object] = {"file": META_FILE, "bytes": self.metadata_bytes}
        if self.metadata_crc32 is not None:  # 旧记录原样透传，新记录不带这个字段
            metadata["crc32"] = self.metadata_crc32
        return json.dumps(
            {
                "format_version": self.format_version,
                "step": self.step,
                "metadata": metadata,
                "files": {name: int(size) for name, size in sorted(self.files.items())},
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

    @classmethod
    def loads(cls, text: str) -> "CommitRecord":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise FormatError(f"COMMIT is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise FormatError(f"COMMIT must be a JSON object, got {type(data).__name__}")
        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            raise FormatError("COMMIT missing 'metadata' object")
        files = data.get("files", {})
        if not isinstance(files, dict):
            raise FormatError("COMMIT 'files' must be an object")
        try:
            return cls(
                step=int(data["step"]),
                metadata_bytes=int(metadata["bytes"]),
                metadata_crc32=(
                    str(metadata["crc32"]) if metadata.get("crc32") is not None else None
                ),
                files={str(name): int(size) for name, size in files.items()},
                format_version=int(data.get("format_version", COMMIT_FORMAT_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FormatError(f"COMMIT is malformed: {exc}") from exc


def write_metadata(storage: StorageBackend, step: int, index: CheckpointIndex) -> int:
    """写 ``.metadata``（索引本身也是提交点之一），返回字节数。"""
    text = index.dumps()
    payload = text.encode("utf-8")
    storage.write(f"{version_key(step)}/{META_FILE}", payload)
    return len(payload)


def write_commit(storage: StorageBackend, record: CommitRecord) -> None:
    """写 ``COMMIT``：临时文件 → 原子 rename。返回即表示这一版可恢复。"""
    key = version_key(record.step)
    payload = record.dumps().encode("utf-8")
    storage.write(f"{key}/{COMMIT_TEMP}", payload)
    storage.rename_or_put(f"{key}/{COMMIT_TEMP}", f"{key}/{COMMIT_FILE}")


def read_commit(storage: StorageBackend, step: int) -> CommitRecord | None:
    """读某个版本的 COMMIT；不存在（未提交）返回 None。"""
    try:
        payload = storage.read(f"{version_key(step)}/{COMMIT_FILE}")
    except ObjectNotFoundError:
        return None
    return CommitRecord.loads(payload.decode("utf-8"))


def committed_steps(storage: StorageBackend) -> list[int]:
    """扫描所有带 COMMIT 的版本，按 step 升序返回。"""
    steps: list[int] = []
    for key in storage.list(""):
        parts = key.split("/")
        if len(parts) != 2 or parts[1] != COMMIT_FILE:
            continue
        name = parts[0]
        if not name.startswith("step_"):
            continue
        try:
            steps.append(int(name[len("step_") :]))
        except ValueError:
            continue
    return sorted(set(steps))


def latest_step(storage: StorageBackend) -> int | None:
    """最新的可恢复版本：所有带 ``COMMIT`` 的版本里最大的 step。

    没有 ``latest`` 指针需要校验，也没有「指针失效」这条分支：这一个函数就是
    ``step=None`` 的全部语义。
    """
    steps = committed_steps(storage)
    return steps[-1] if steps else None


def delete_version(storage: StorageBackend, step: int) -> None:
    """删一个版本：先删 COMMIT（立刻不可恢复），再删数据。

    顺序与写入镜像：``COMMIT`` 一删，这一版就既不可恢复也不可上传；剩下的数据
    文件即使删到一半被打断，重跑一次也能接着删（判据只看 COMMIT 在不在）。
    """
    key = version_key(step)
    storage.delete(f"{key}/{COMMIT_FILE}")
    for item in storage.list(key):
        storage.delete(item)


__all__ = [
    "COMMIT_FILE",
    "COMMIT_FORMAT_VERSION",
    "COMMIT_TEMP",
    "CommitRecord",
    "committed_steps",
    "delete_version",
    "latest_step",
    "read_commit",
    "version_key",
    "write_commit",
    "write_metadata",
]
