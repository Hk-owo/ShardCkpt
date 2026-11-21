"""异步保存包：训练进程 / 存储进程 / 节点独立通信层三层分离实现。"""

from __future__ import annotations

from saver.client import (
    AsyncSaver,
    SaveHandle,
    SaveResult,
    close_saver,
    default_saver,
    save_async,
    start_saver,
)
from saver.comm import (
    CommError,
    NodeComm,
    NodeCommClient,
    NodeCommProcess,
    NodeCommServer,
    NodeTreeConfig,
    make_tree_configs,
)
from saver.config import CommEndpoint, SaverConfig, SaverDeadError, SaverFaults
from saver.persist import HostPersistResult, persist_host_buffer
from saver.shm import ShmBuffer, ShmPinnedPool, ShmSegment

__all__ = [
    "AsyncSaver",
    "CommEndpoint",
    "CommError",
    "HostPersistResult",
    "NodeComm",
    "NodeCommClient",
    "NodeCommProcess",
    "NodeCommServer",
    "NodeTreeConfig",
    "SaveHandle",
    "SaveResult",
    "SaverConfig",
    "SaverDeadError",
    "SaverFaults",
    "ShmBuffer",
    "ShmPinnedPool",
    "ShmSegment",
    "close_saver",
    "default_saver",
    "make_tree_configs",
    "persist_host_buffer",
    "save_async",
    "start_saver",
]
