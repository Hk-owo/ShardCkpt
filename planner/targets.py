"""目标布局 → 本 rank 要读的区域（``TargetTensor``）。

恢复端不关心「存的时候是怎么切的」，只关心「新布局下本 rank 要哪一块」：

- ``shard_dims`` 为空或 ``tp == 1``：每个张量整读。布局不变时这就是自然退化，
  每个张量只有一次整分片读，没有单独的「快路径」代码。
- 给了 ``shard_dims``：列出的张量沿指定维切成 ``tp`` 份，取第 ``tp_rank`` 份；
  没列出的张量整读 —— Replicate 维度上的每个副本都需要完整数据。

切分规则与保存侧一致（:func:`core.region.split`：不整除时前几份各多一个元素），
所以 ``tp=8`` 存、``tp=16`` 读这种换布局恢复不需要额外的对齐约定。
"""

from __future__ import annotations

from typing import Mapping

from core import PlanError
from core.index import CheckpointIndex
from planner.load_plan import TargetTensor


def targets_for_layout(
    checkpoint: CheckpointIndex,
    *,
    tp: int = 1,
    tp_rank: int = 0,
    shard_dims: Mapping[str, int] | None = None,
) -> list[TargetTensor]:
    """按目标布局构造读取目标；返回按 FQN 排序的列表（顺序稳定，便于 diff）。

    ``tp``       目标 TP 大小（沿 ``shard_dims`` 里指定的维切几份）
    ``tp_rank``  本 rank 在 TP 组内的序号
    ``shard_dims`` ``{FQN: dim}``：哪些张量沿哪个维切；未列出的一律整读
    """
    if tp < 1:
        raise ValueError(f"tp must be >= 1, got {tp}")
    if not 0 <= tp_rank < tp:
        raise ValueError(f"tp_rank must be in [0, {tp}), got {tp_rank}")

    dims = dict(shard_dims or {})
    unknown = sorted(set(dims) - set(checkpoint.tensors))
    if unknown:
        raise PlanError(
            f"shard_dims references tensors that are not in the checkpoint: {unknown}"
        )

    targets: list[TargetTensor] = []
    for fqn in sorted(checkpoint.tensors):
        meta = checkpoint.tensors[fqn]
        dim = dims.get(fqn)
        if tp == 1 or dim is None:
            targets.append(TargetTensor.whole(fqn, meta.global_shape, meta.dtype))
            continue
        if dim < 0 or dim >= len(meta.global_shape):
            raise PlanError(
                f"{fqn}: shard dim {dim} out of range for global shape {meta.global_shape}"
            )
        targets.append(
            TargetTensor.for_shard(
                fqn,
                meta.global_shape,
                meta.dtype,
                dim=dim,
                index=tp_rank,
                count=tp,
            )
        )
    return targets


__all__ = ["targets_for_layout"]
