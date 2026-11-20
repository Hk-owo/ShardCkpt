"""提交协议、版本目录、垃圾回收。

本地层的实现见 :mod:`commit.local`：数据与索引先落盘，``COMMIT`` 最后写，
恢复端只认带 ``COMMIT`` 的版本目录。回收见 :mod:`commit.gc`。
"""

from __future__ import annotations

from commit.coordinator import (
    CommitCoordinator,
    CoordinatedSaveResult,
    InflightStatus,
    InvalidatedGeneration,
)
from commit.local import (
    COMMIT_FILE,
    COMMIT_FORMAT_VERSION,
    COMMIT_TEMP,
    CommitRecord,
    GENERATION_FILE,
    GENERATION_FORMAT_VERSION,
    GENERATION_TEMP,
    LocalGenerationMarker,
    committed_steps,
    delete_local_generation,
    delete_version,
    latest_step,
    read_commit,
    read_local_generation,
    version_key,
    write_commit,
    write_local_generation,
    write_metadata,
)

__all__ = [
    "COMMIT_FILE",
    "COMMIT_FORMAT_VERSION",
    "COMMIT_TEMP",
    "CommitCoordinator",
    "CommitRecord",
    "CoordinatedSaveResult",
    "GENERATION_FILE",
    "GENERATION_FORMAT_VERSION",
    "GENERATION_TEMP",
    "InflightStatus",
    "InvalidatedGeneration",
    "LocalGenerationMarker",
    "committed_steps",
    "delete_local_generation",
    "delete_version",
    "latest_step",
    "read_commit",
    "read_local_generation",
    "version_key",
    "write_commit",
    "write_local_generation",
    "write_metadata",
]
