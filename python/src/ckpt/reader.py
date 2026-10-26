"""ShardCkpt 数据读取接口。

接口形态按「从 GPU 读取」设计：

1. 加载时显式指定目标设备（对应 ``torch.load(map_location=...)``）
2. 数据路径固定为  host 文件 -> staging buffer -> H2D 拷贝 -> device buffer
3. 迁移与取值都经过 backend，调用方从不直接触碰 device 内存

当前机器没有 GPU，cuda 分支由 MockCudaBackend 承载，读取路径与真机一致；
接入真卡后只需注册一个真实的 cuda backend。
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

from .backend import DeviceBackend, get_backend
from .device import CPU, Device
from .dtype import itemsize
from .errors import ChecksumError, FormatError, MissingTensorError
from .tensor import Tensor

# 真实实现里 staging 是 page-locked（cudaHostAlloc）内存；这里的 bytearray 只保证
# 代码路径一致，不做 pinned 语义。
_STAGING_TYPE = bytearray


@dataclass(frozen=True, slots=True)
class LoadOptions:
    """加载选项。

    device:       目标设备，默认 CPU；可传 "cuda:0" 等。
    verify_crc:   是否校验 manifest 里的 CRC32。
    staging:      是否经 host staging buffer 中转（GPU 路径的默认行为）。
    """

    device: Device = CPU
    verify_crc: bool = True
    staging: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", Device.parse(self.device))


class CheckpointData:
    """一次加载得到的全部张量，按 manifest 顺序保存。"""

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
        return f"CheckpointData({len(self)} tensors, device={self.device}, step={self.meta.get('step')})"


def _resolve(root: Path, rel: str) -> Path:
    """把 manifest 里的相对路径解析成绝对路径，拒绝路径穿越。"""
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise FormatError(f"invalid tensor path in manifest: {rel!r}")
    return root / rel_path


class CheckpointReader:
    """读取一份 ShardCkpt 目录（manifest.json + tensors/*.bin）。"""

    def __init__(self, root: str | Path, options: LoadOptions | None = None) -> None:
        self._root = Path(root)
        self._options = options or LoadOptions()
        self._backend: DeviceBackend = get_backend(self._options.device)
        self._manifest = self._load_manifest()
        self._entries: dict[str, dict] = {e["name"]: e for e in self._manifest.get("tensors", [])}
        self._order: list[str] = [e["name"] for e in self._manifest.get("tensors", [])]

    # --- 构造 ---------------------------------------------------------------
    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        device: "str | Device" = "cpu",
        verify_crc: bool = True,
        staging: bool = True,
    ) -> "CheckpointReader":
        return cls(path, LoadOptions(device=Device.parse(device), verify_crc=verify_crc, staging=staging))

    # --- 元信息 -------------------------------------------------------------
    @property
    def root(self) -> Path:
        return self._root

    @property
    def options(self) -> LoadOptions:
        return self._options

    @property
    def device(self) -> Device:
        return self._options.device

    @property
    def backend(self) -> DeviceBackend:
        return self._backend

    @property
    def manifest(self) -> dict:
        return self._manifest

    @property
    def meta(self) -> dict:
        """去掉 tensors 列表后的元信息（step / seed / world_size 等）。"""
        return {k: v for k, v in self._manifest.items() if k != "tensors"}

    def names(self) -> list[str]:
        return list(self._order)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    # --- 读取 ---------------------------------------------------------------
    def read(self, name: str) -> Tensor:
        """读取单个张量到目标设备。"""
        entry = self._entries.get(name)
        if entry is None:
            raise MissingTensorError(f"tensor {name!r} not listed in manifest")

        nbytes = int(entry["nbytes"])
        expected = itemsize(entry["dtype"]) * _numel(tuple(entry["shape"]))
        if expected != nbytes:
            raise FormatError(
                f"{name}: manifest nbytes={nbytes} inconsistent with "
                f"dtype={entry['dtype']} shape={entry['shape']} (expected {expected})"
            )

        path = _resolve(self._root, entry["file"])
        if not path.is_file():
            raise MissingTensorError(f"{name}: data file not found: {path}")
        if path.stat().st_size != nbytes:
            raise FormatError(
                f"{name}: file size {path.stat().st_size} != manifest nbytes {nbytes}"
            )

        with path.open("rb") as fh:
            payload = fh.read(nbytes)
        if len(payload) != nbytes:
            raise FormatError(f"{name}: short read, got {len(payload)} of {nbytes} bytes")

        if self._options.verify_crc:
            actual = zlib.crc32(payload) & 0xFFFFFFFF
            if actual != int(entry["crc32"]):
                raise ChecksumError(
                    f"{name}: crc32 mismatch (manifest={entry['crc32']}, actual={actual})"
                )

        # host staging（真实实现为 pinned memory），随后 H2D 拷贝
        if self._options.staging:
            staging = _STAGING_TYPE(payload)
            source: memoryview | bytes = memoryview(staging)
        else:
            source = memoryview(payload)

        buffer = self._backend.allocate(nbytes)
        self._backend.copy_h2d(buffer, source)  # type: ignore[arg-type]
        self._backend.synchronize()

        return Tensor(
            name=name,
            buffer=buffer,
            device=self._options.device,
            dtype=entry["dtype"],
            shape=tuple(entry["shape"]),
            backend=self._backend,
        )

    def read_all(self) -> CheckpointData:
        """按 manifest 顺序加载全部张量。"""
        tensors = {name: self.read(name) for name in self._order}
        return CheckpointData(tensors=tensors, meta=self.meta, device=self._options.device)

    # --- 内部 ---------------------------------------------------------------
    def _load_manifest(self) -> dict:
        path = self._root / "manifest.json"
        if not path.is_file():
            raise FormatError(f"manifest not found: {path}")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FormatError(f"manifest is not valid JSON: {exc}") from exc
        if not isinstance(manifest, dict) or "tensors" not in manifest:
            raise FormatError("manifest missing 'tensors' list")
        if not isinstance(manifest["tensors"], list):
            raise FormatError("manifest 'tensors' must be a list")
        return manifest


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def load(
    path: str | Path,
    *,
    device: "str | Device" = "cpu",
    verify_crc: bool = True,
    staging: bool = True,
) -> CheckpointData:
    """便捷入口，对应 ``torch.load(path, map_location=device)``。"""
    reader = CheckpointReader.open(
        path, device=device, verify_crc=verify_crc, staging=staging
    )
    return reader.read_all()
