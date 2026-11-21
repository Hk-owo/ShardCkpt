"""异步保存配置、故障注入定义与异常类型。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from async_.pinned import DEFAULT_CHUNK_BYTES
from commit.gc import (
    DEFAULT_KEEP_LOCAL_VERSIONS,
    DEFAULT_KEEP_VERSIONS,
    DEFAULT_MPU_TIMEOUT_SECONDS,
)
from core import CheckpointError
from storage.base import validate_run_id
from storage.local import DEFAULT_ALIGNMENT
from storage.object import S3Config

WORKER_DEAD_MODES = ("raise", "fallback_sync")


class SaverDeadError(CheckpointError):
    """存储子进程熔断或不可恢复死亡后抛出的异常。"""


@dataclass(frozen=True)
class CommEndpoint:
    """可跨 spawn 序列化的节点通信层接入描述符。"""

    local_address: str
    node_id: int = 0
    num_nodes: int = 1
    world_size: int = 1
    timeout: float = 30.0


@dataclass(frozen=True)
class SaverFaults:
    """测试注入钩子（全纯数据，可跨 spawn 传递）。"""

    slow_write_ms: float = 0.0
    exit_on_start: bool = False
    hang_on_snap: bool = False
    fail_persist_step: int | None = None
    fail_upload_step: int | None = None
    fail_upload_rank: int | None = None
    slow_upload_ms: float = 0.0
    fail_acquire_generation: bool = False
    skip_heartbeat: bool = False


@dataclass(frozen=True)
class SaverConfig:
    """启动期长期设定：整个异步保存会话期间保持不变（Q3 / Q5）。"""

    local_root: str | Path
    remote_config: S3Config | None = None
    run_id: str | None = None
    rank: int = 0
    ranks: tuple[int, ...] | None = None
    max_inflight: int = 1
    force_after_skips: int = 3
    max_restarts: int = 3
    restart_window: float = 60.0
    restart_backoff: float = 0.05
    heartbeat_interval: float = 1.0
    heartbeat_timeout: float = 5.0
    on_worker_dead: str = "raise"
    keep_versions: int | None = DEFAULT_KEEP_VERSIONS
    keep_local: int = DEFAULT_KEEP_LOCAL_VERSIONS
    gc_interval: float | None = None
    gc_dry_run: bool = False
    staging_dir: str | Path | None = None
    abort_stale_mpu: bool = True
    mpu_timeout_seconds: float = DEFAULT_MPU_TIMEOUT_SECONDS
    upload_timeout: float = 300.0
    collective_timeout: float = 30.0
    shutdown_timeout: float = 30.0
    shm_bytes: int = DEFAULT_CHUNK_BYTES
    shm_max_bytes: int | None = None
    alignment: int = DEFAULT_ALIGNMENT
    direct_io: bool = True
    engine: str = "auto"
    cuda_register: bool = False
    drain_on_parent_death: bool = False
    comm_endpoint: CommEndpoint | None = None
    faults: SaverFaults | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_root", Path(self.local_root))
        if self.staging_dir is not None:
            object.__setattr__(self, "staging_dir", Path(self.staging_dir))
        if self.run_id is not None:
            validate_run_id(self.run_id)
        if self.rank < 0:
            raise ValueError(f"rank must be >= 0, got {self.rank}")
        if self.ranks is not None:
            normalized = tuple(int(r) for r in self.ranks)
            if not normalized:
                raise ValueError("ranks must not be empty when specified")
            object.__setattr__(self, "ranks", normalized)
        if self.max_inflight < 1:
            raise ValueError(f"max_inflight must be >= 1, got {self.max_inflight}")
        if self.force_after_skips < 1:
            raise ValueError(
                f"force_after_skips must be >= 1, got {self.force_after_skips}"
            )
        if self.max_restarts < 0:
            raise ValueError(f"max_restarts must be >= 0, got {self.max_restarts}")
        if self.restart_window <= 0:
            raise ValueError(f"restart_window must be > 0, got {self.restart_window}")
        if self.restart_backoff < 0:
            raise ValueError(
                f"restart_backoff must be >= 0, got {self.restart_backoff}"
            )
        if self.heartbeat_interval <= 0:
            raise ValueError(
                f"heartbeat_interval must be > 0, got {self.heartbeat_interval}"
            )
        if self.heartbeat_timeout <= 0:
            raise ValueError(
                f"heartbeat_timeout must be > 0, got {self.heartbeat_timeout}"
            )
        if self.on_worker_dead not in WORKER_DEAD_MODES:
            raise ValueError(
                f"on_worker_dead must be one of {WORKER_DEAD_MODES}, "
                f"got {self.on_worker_dead!r}"
            )
        if self.keep_versions is not None and self.keep_versions < 1:
            raise ValueError(f"keep_versions must be >= 1, got {self.keep_versions}")
        if self.keep_local < 0:
            raise ValueError(f"keep_local must be >= 0, got {self.keep_local}")
        if self.gc_interval is not None and self.gc_interval <= 0:
            raise ValueError(f"gc_interval must be > 0, got {self.gc_interval}")
        if self.upload_timeout <= 0:
            raise ValueError(f"upload_timeout must be > 0, got {self.upload_timeout}")
        if self.collective_timeout <= 0:
            raise ValueError(
                f"collective_timeout must be > 0, got {self.collective_timeout}"
            )
        if self.alignment < 1:
            raise ValueError(f"alignment must be >= 1, got {self.alignment}")
        if self.shm_bytes < self.alignment:
            raise ValueError(
                f"shm_bytes ({self.shm_bytes}) must be >= alignment ({self.alignment})"
            )
        if self.shm_max_bytes is not None and self.shm_max_bytes < self.shm_bytes:
            raise ValueError(
                f"shm_max_bytes ({self.shm_max_bytes}) must be >= shm_bytes ({self.shm_bytes})"
            )


__all__ = [
    "CommEndpoint",
    "SaverConfig",
    "SaverDeadError",
    "SaverFaults",
    "WORKER_DEAD_MODES",
]
