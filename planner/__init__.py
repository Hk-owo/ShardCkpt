"""规划器：纯函数，不碰 IO 和通信。

两个入口，输入输出都是纯数据：

- :func:`~planner.save_plan.build_save_plan`：本 rank 的分片 → 「写哪个文件、
  哪个偏移」。含 Replicate 维度去重（``hash(FQN) % 副本数``）与对齐补齐。
- :func:`~planner.load_plan.build_load_plan`：旧索引 + 新布局的目标区域 →
  「从哪个文件读多少、拷到哪」。含逐维求交、读放大策略与相邻请求合并。

真正的读写、校验和计算、以及 gather 到 rank 0 都留给 storage / commit 层。
"""

from __future__ import annotations

from planner.load_plan import (
    COLUMN_STRATEGIES,
    CopySpan,
    ReadItem,
    ReadPlan,
    TargetTensor,
    build_load_plan,
)
from planner.save_plan import (
    DEFAULT_ALIGN,
    DEFAULT_FILE,
    BlobItem,
    SavePlan,
    ShardSource,
    WriteItem,
    WriterFn,
    build_save_plan,
    default_writer,
)

__all__ = [
    "COLUMN_STRATEGIES",
    "DEFAULT_ALIGN",
    "DEFAULT_FILE",
    "BlobItem",
    "CopySpan",
    "ReadItem",
    "ReadPlan",
    "SavePlan",
    "ShardSource",
    "TargetTensor",
    "WriteItem",
    "WriterFn",
    "build_load_plan",
    "build_save_plan",
    "default_writer",
]
