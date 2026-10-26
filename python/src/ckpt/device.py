"""设备描述：与 PyTorch 的 "cpu" / "cuda:N" 写法保持一致。"""

from __future__ import annotations

import re
from dataclasses import dataclass

_DEVICE_RE = re.compile(r"^(cpu|cuda)(?::(\d+))?$")


@dataclass(frozen=True, slots=True)
class Device:
    """一个设备端点。

    与 torch.device 对齐："cpu" 的 index 恒为 0；"cuda:N" 表示第 N 张卡。
    当前机器没有 GPU，cuda 设备由 mock backend 承载，但类型和编号语义不变。
    """

    type: str
    index: int = 0

    def __post_init__(self) -> None:
        if self.type not in ("cpu", "cuda"):
            raise ValueError(f"unsupported device type {self.type!r}, expected 'cpu' or 'cuda'")
        if self.index < 0:
            raise ValueError(f"device index must be >= 0, got {self.index}")

    @classmethod
    def parse(cls, spec: "str | Device") -> "Device":
        if isinstance(spec, cls):
            return spec
        if not isinstance(spec, str):
            raise TypeError(f"device spec must be str or Device, got {type(spec).__name__}")
        m = _DEVICE_RE.match(spec.strip().lower())
        if not m:
            raise ValueError(f"unsupported device {spec!r}; expected 'cpu' or 'cuda[:N]'")
        return cls(m.group(1), int(m.group(2) or 0))

    @property
    def is_cuda(self) -> bool:
        return self.type == "cuda"

    def __str__(self) -> str:
        return "cpu" if self.type == "cpu" else f"cuda:{self.index}"


CPU = Device("cpu")


def cuda(index: int = 0) -> Device:
    return Device("cuda", index)
