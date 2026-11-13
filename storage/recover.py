"""恢复端的第一道门：选版本 → 校验提交点 → 把 ``.metadata`` 读成索引。

读取只认带 ``COMMIT`` 的版本目录。``COMMIT`` 里记着 ``.metadata`` 的字节数、
CRC-32C 与数据文件清单（名字 -> 字节数），所以「索引写了一半、被截断、内容被改坏、
缺文件、版本错配」这些问题都在读第一个张量之前就能发现。

索引的 CRC-32C 是本层唯一的字节级校验：JSON 解析与 ``CheckpointIndex.validate()``
只看结构，一份被改过 ``byte_offset`` 的索引在结构上完全合法，只有摘要能识别。数据
文件本身仍由索引里的块级 CRC-32C 在读路径上把关（:mod:`storage.local_load`），
对象存储的对象级校验和在 :mod:`storage.object`。

这一步不做任何张量 IO，也因此不碰 device：选版本、验索引、查文件大小，全部是
元数据操作。真正的批量读在 :mod:`storage.local_load`。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from commit.local import CommitRecord, latest_step, read_commit, version_key
from core import ChecksumError, FormatError
from core.index import META_FILE, CheckpointIndex
from storage.base import ObjectNotFoundError, StorageBackend
from storage.checksum import crc32c_hex
from storage.local import DEFAULT_ALIGNMENT, LocalStorage


@dataclass(frozen=True)
class RecoveredVersion:
    """一个已确认可恢复的版本：存储后端 + 索引 + 提交记录。"""

    storage: StorageBackend
    step: int
    key: str
    index: CheckpointIndex
    commit: CommitRecord

    def file_key(self, name: str) -> str:
        """索引里的 ``file`` 是版本目录内的名字，读写存储时要拼上 step 目录。"""
        return f"{self.key}/{name}"

    def data_files(self) -> list[str]:
        """索引引用到的全部数据文件（版本目录内的相对名字）。"""
        names = {shard.file for meta in self.index.tensors.values() for shard in meta.shards}
        names |= {blob.file for blob in self.index.blobs.values()}
        return sorted(names)

    @property
    def blobs(self) -> dict[str, int]:
        """小状态：名字 -> 字节数（已序列化的原始字节）。"""
        return {name: blob.length for name, blob in sorted(self.index.blobs.items())}

    def read_blobs(self) -> dict[str, bytes]:
        """把全部小状态读回来（体积很小，走一次 ``read_range`` 一条）。"""
        return {
            name: self.storage.read_range(self.file_key(blob.file), blob.byte_offset, blob.length)
            for name, blob in sorted(self.index.blobs.items())
        }


def recover_version(storage: StorageBackend, step: int | None = None) -> RecoveredVersion:
    """从一个存储后端恢复某个版本；``step=None`` 时取最新的已提交版本。"""
    if step is not None and step < 0:
        raise ValueError(f"step must be >= 0, got {step}")

    target = step if step is not None else latest_step(storage)
    if target is None:
        raise FormatError(
            f"no committed checkpoint found under {storage!r}: "
            "只有带 COMMIT 的版本目录才算可恢复"
        )

    key = version_key(target)
    record = read_commit(storage, target)
    if record is None:
        raise FormatError(f"{key}: COMMIT missing, this version is not recoverable")

    payload = _read_metadata(storage, key, record)
    index = _parse_index(key, payload)
    _check_data_files(storage, key, index, record)
    return RecoveredVersion(
        storage=storage, step=target, key=key, index=index, commit=record
    )


def open_version(
    root: str | Path,
    step: int | None = None,
    *,
    alignment: int = DEFAULT_ALIGNMENT,
    engine: str = "auto",
) -> RecoveredVersion:
    """本地目录的便捷入口：``open_version("/ckpt")`` 读最新可恢复版本。"""
    storage = LocalStorage(root, direct_io=False, alignment=alignment, engine=engine)
    return recover_version(storage, step)


# --- 内部 -------------------------------------------------------------------


def _read_metadata(
    storage: StorageBackend, key: str, record: CommitRecord
) -> bytes:
    """读 ``.metadata``，先比长度再比 CRC-32C：内容坏了在解析之前就报错。"""
    meta_key = f"{key}/{META_FILE}"
    try:
        payload = storage.read(meta_key)
    except ObjectNotFoundError:
        raise FormatError(f"{meta_key}: index missing for committed version") from None

    if len(payload) != record.metadata_bytes:
        raise FormatError(
            f"{meta_key}: {len(payload)} bytes, COMMIT records {record.metadata_bytes}"
        )
    actual = crc32c_hex(payload)
    if actual != record.metadata_crc32c:
        raise ChecksumError(
            f"{meta_key}: crc32c {actual}, COMMIT records {record.metadata_crc32c}"
            "（索引内容与提交点不符）"
        )
    return payload


def _parse_index(key: str, payload: bytes) -> CheckpointIndex:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FormatError(f"{key}/{META_FILE}: not valid UTF-8: {exc}") from exc
    index = CheckpointIndex.loads(text)
    index.validate()
    return index


def _check_data_files(
    storage: StorageBackend,
    key: str,
    index: CheckpointIndex,
    record: CommitRecord,
) -> None:
    """COMMIT 的清单必须覆盖索引引用到的文件，且本地大小要对得上。"""
    names = {shard.file for meta in index.tensors.values() for shard in meta.shards}
    names |= {blob.file for blob in index.blobs.values()}
    unlisted = sorted(names - set(record.files))
    if unlisted:
        raise FormatError(
            f"{key}: COMMIT does not list data file(s) referenced by the index: {unlisted}"
        )

    if not isinstance(storage, LocalStorage):
        return  # 非本地后端在这里没有 stat；下载后会再落成本地文件并走同一套校验

    for name, expected in sorted(record.files.items()):
        path = storage.path_for(f"{key}/{name}")
        if not path.is_file():
            raise FormatError(f"{key}/{name}: data file listed in COMMIT is missing")
        actual = path.stat().st_size
        if actual != expected:
            raise FormatError(
                f"{key}/{name}: {actual} bytes on disk, COMMIT records {expected}"
            )


__all__ = ["RecoveredVersion", "open_version", "recover_version"]
