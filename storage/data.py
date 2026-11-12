"""一次加载得到的张量集合：``api.load`` 的返回类型。

它只做三件事：按名字取张量、迁移设备、暴露加载元信息。真正的读路径在
:mod:`storage.local_load`（本地版本目录）和 :mod:`storage.upload`（对象存储恢复）。
"""

from __future__ import annotations

from async_ import Tensor
from core import MissingTensorError
from core.metadata import Device


class CheckpointData:
    """一次加载得到的全部张量。"""

    def __init__(self, *, tensors: dict[str, Tensor], meta: dict, device: Device) -> None:
        self._tensors = tensors
        self.meta = meta
        self.device = device

    @property
    def tensors(self) -> dict[str, Tensor]:
        return self._tensors

    def names(self) -> list[str]:
        return list(self._tensors)

    def __getitem__(self, name: str) -> Tensor:
        try:
            return self._tensors[name]
        except KeyError:
            raise MissingTensorError(f"tensor {name!r} not found in ShardCkpt") from None

    def __contains__(self, name: object) -> bool:
        return name in self._tensors

    def __iter__(self):
        return iter(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)

    def items(self):
        return self._tensors.items()

    def to(self, device: "str | Device") -> "CheckpointData":
        """把所有张量迁移到目标设备。"""
        target = Device.parse(device)
        return CheckpointData(
            tensors={name: t.to(target) for name, t in self._tensors.items()},
            meta=self.meta,
            device=target,
        )

    def __repr__(self) -> str:
        return (
            f"CheckpointData({len(self)} tensors, device={self.device}, "
            f"step={self.meta.get('step')})"
        )


__all__ = ["CheckpointData"]
