"""本地层的读闭环：COMMIT 门禁 → 读取计划 → 批量读 → staging → H2D → device。

数据路径与 :func:`api.save_local` 严格镜像：

```
open_version(选版本 + 校验索引) ──> targets_for_layout(新布局的目标区域)
   ──> build_load_plan(求交 → 读取计划) ──> 按窗口取 pinned staging 块
   ──> 按索引里的 CRC 块切/扩读 ──> io_uring 批量 read_ranges(零拷贝落进 staging)
   ──> 逐块 CRC-32C 校验 ──> 按 CopySpan 拷进目标 host 数组 ──> H2D ──> Tensor
```

**CRC 默认校验**（``verify_crc=True``）。写路径把每个分片/小状态按
``crc_chunk_max``（默认 1 MiB）切块，每块一个 CRC-32C 写进索引；读路径按同样的
块边界切 IO，默认还把不完整的读块扩读到完整块（``crc_padding="block"``），
读一块校验一块，不需要把整个分片/文件读进来。校验不通过就在**返回坏数据之前**
抛 :class:`ChecksumError`。

三个容易做错的地方，这里都定死了：

- **读块与分片的区别**：``ReadItem.length`` 是「逻辑读块」。默认扩读到完整 CRC
  块，``LoadReport.verified_blocks`` 是校验过的块数；只有在 ``crc_padding="none"``
  或 ``"auto"`` 未达阈值时，跨块边界的头/尾部分块才记进 ``unverified``，并计入
  ``unverified_bytes``，不静默假装校验过。
- **staging 生命周期**：块从 pinned 池借出，窗口内读完、拷完才归还；H2D 之后 device
  内存自己持有数据，与 staging 无关。
- **短读不静默截断**：C++ 侧 ``ReadRanges`` 与 Python 回退都报错，不会返回半份张量。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from async_ import (
    DEFAULT_CHUNK_BYTES,
    DeviceBackend,
    PinnedBufferPool,
    Tensor,
    aligned_allocator,
    get_backend,
)
from core import ChecksumError, PlanError
from core.index import CheckpointIndex
from core.metadata import Device, numpy_dtype
from planner.load_plan import ReadItem, TargetTensor, build_load_plan
from planner.targets import targets_for_layout
from storage.checksum import crc32c_hex
from storage.crc_io import ReadUnit, crc_read_units, item_label
from storage.local import DEFAULT_ALIGNMENT, LocalStorage
from storage.recover import RecoveredVersion, recover_version


@dataclass(frozen=True)
class LoadReport:
    """一次本地恢复的结果与成本。"""

    step: int
    root: Path
    device: Device
    engine: str
    tensors: Mapping[str, Tensor]
    blobs: Mapping[str, bytes]
    index: CheckpointIndex
    requests: int
    windows: int
    read_bytes: int
    copied_bytes: int
    h2d_bytes: int
    crc_verified: bool = True
    """本次是否执行了 CRC 校验（即 ``verify_crc`` 的生效值）。默认校验。"""
    verified_blocks: int = 0
    """通过 CRC-32C 校验的**块数**；``crc_verified=False`` 时恒为 0。"""
    io_requests: int = 0
    """实际 IO 单元数（扩读、同块合并之后的 ``read_range`` 次数）。"""
    unverified: tuple[str, ...] = ()
    """要求校验、但没有扩读到完整 CRC 块的读块标签。

    默认 ``crc_padding="block"`` 下为空；``crc_padding="none"`` 或 ``"auto"``
    未达阈值时非空。它**不**包含「按策略不校验」的读块——那种情况由
    ``crc_verified=False`` 统一表示。
    """
    unverified_bytes: int = 0
    """上述读块里没有可比 CRC 的字节数；``crc_verified=False`` 时恒为 0。"""

    @property
    def amplification(self) -> float:
        """读放大：从存储读进来的字节 / 真正拷进目标的字节。"""
        return self.read_bytes / self.copied_bytes if self.copied_bytes else 0.0

    @property
    def files(self) -> list[str]:
        return sorted({shard.file for meta in self.index.tensors.values() for shard in meta.shards})


def load_local(
    root: str | Path,
    *,
    step: int | None = None,
    device: "str | Device" = "cpu",
    tp: int = 1,
    tp_rank: int = 0,
    shard_dims: Mapping[str, int] | None = None,
    verify_crc: bool = True,
    crc_padding: str = "block",
    crc_padding_ratio: float = 0.125,
    budget_bytes: int = DEFAULT_CHUNK_BYTES,
    engine: str = "auto",
    alignment: int = DEFAULT_ALIGNMENT,
    column_strategy: str = "whole",
    pool: PinnedBufferPool | None = None,
) -> LoadReport:
    """从本地版本目录恢复张量（同步版本，与 ``save_local`` 对称）。

    ``step``          版本号；``None`` 取最新的已提交版本
    ``device``        目标设备（mock 下 cuda 仍走完整 H2D 路径）
    ``tp`` / ``tp_rank`` / ``shard_dims``  目标布局，见 :func:`planner.targets.targets_for_layout`
    ``budget_bytes``  单个 staging 窗口的字节预算
    ``engine``        ``"native"``（io_uring）/ ``"python"``（preadv）/ ``"auto"``
    ``column_strategy`` 列子集读法（``whole`` 整块读再切 / ``strided`` 逐行读）
    ``pool``          复用调用方的 pinned 池（窗口变大时会扩容）
    ``verify_crc``    逐块校验分片 CRC-32C；**默认开启**，见模块 docstring 的说明
    ``crc_padding``   ``block``（默认，扩读到完整 CRC 块）/ ``auto``（按比例）/
                      ``none``（不扩读，部分块记 unverified）
    ``crc_padding_ratio`` ``auto`` 的阈值：实际使用字节 / 扩展后字节
    """
    if budget_bytes <= 0:
        raise ValueError(f"budget_bytes must be > 0, got {budget_bytes}")

    resolved_engine = _resolve_engine(engine)
    storage = LocalStorage(root, direct_io=False, alignment=alignment, engine=resolved_engine)
    version = recover_version(storage, step)
    targets = targets_for_layout(version.index, tp=tp, tp_rank=tp_rank, shard_dims=shard_dims)
    plan = build_load_plan(
        version.index, targets, strict=True, column_strategy=column_strategy
    )

    device_spec = Device.parse(device)
    backend = get_backend(device_spec)
    host_arrays = {
        target.fqn: np.zeros(target.region.shape, dtype=numpy_dtype(target.dtype))
        for target in targets
    }

    owned_pool = pool is None
    if pool is None:
        pool = PinnedBufferPool(
            chunk_bytes=budget_bytes,
            min_chunks=1,
            max_bytes=None,
            timeout=0,
            allocator=aligned_allocator(alignment),
        )

    reader = _NativeReader() if resolved_engine == "native" else None
    read_bytes = 0
    copied_bytes = 0
    windows = 0
    requests = 0
    io_requests = 0
    verified_blocks = 0
    unverified: set[str] = set()
    unverified_bytes = 0
    try:
        for window in plan.windows(budget_bytes):
            units = crc_read_units(
                version.index,
                window,
                padding=crc_padding if verify_crc else "none",
                padding_ratio=crc_padding_ratio,
            )
            total = sum(unit.length for unit in units)
            buffer = pool.acquire(total)
            try:
                view = _as_view(buffer)
                if reader is not None:
                    reader.read_units(version, units, view)
                else:
                    _read_units_python(version, units, view)
                if verify_crc:
                    for unit in units:
                        verified_bytes = 0
                        for offset, length, expected in unit.chunks:
                            start = unit.staging_offset + offset
                            actual = crc32c_hex(
                                view[start : start + length]
                            )
                            if actual != expected:
                                raise ChecksumError(
                                    f"{unit.label}: crc32c {actual} != index "
                                    f"{expected} ({unit.file}@{unit.byte_offset + offset}"
                                    f"+{length})"
                                )
                            verified_blocks += 1
                            verified_bytes += length
                        missing = unit.length - verified_bytes
                        if missing:
                            unverified_bytes += missing
                            for consumer in unit.consumers:
                                unverified.add(item_label(consumer.item))
                for unit in units:
                    for consumer in unit.consumers:
                        start = unit.staging_offset + consumer.offset_in_unit
                        _scatter(
                            consumer.item,
                            view[start : start + consumer.item.length],
                            host_arrays,
                        )
                requests += len(window)
                io_requests += len(units)
                read_bytes += total
                copied_bytes += sum(
                    span.numel * numpy_dtype(item.dtype).itemsize
                    for item in window
                    for span in item.spans
                )
                windows += 1
            finally:
                buffer.release()
    finally:
        if reader is not None:
            reader.close()
        if owned_pool:
            pool.close()

    tensors: dict[str, Tensor] = {}
    h2d_bytes = 0
    for target in targets:
        host = host_arrays[target.fqn]
        tensors[target.fqn] = _to_device(target, host, device_spec, backend)
        h2d_bytes += host.nbytes

    return LoadReport(
        step=version.step,
        root=Path(storage.root),
        device=device_spec,
        engine=resolved_engine,
        tensors=tensors,
        blobs=version.read_blobs(),
        index=version.index,
        requests=requests,
        windows=windows,
        read_bytes=read_bytes,
        copied_bytes=copied_bytes,
        h2d_bytes=h2d_bytes,
        crc_verified=verify_crc,
        verified_blocks=verified_blocks,
        io_requests=io_requests,
        unverified=tuple(sorted(unverified)),
        unverified_bytes=unverified_bytes,
    )


# --- 读取 -------------------------------------------------------------------


def _resolve_engine(engine: str) -> str:
    if engine not in ("auto", "native", "python"):
        raise ValueError(f"unknown engine {engine!r}")
    if engine != "auto":
        return engine
    try:
        import native

        return "native" if native.available() else "python"
    except ImportError:  # pragma: no cover - native 包本身缺失
        return "python"


def _read_units_python(
    version: RecoveredVersion,
    units: tuple[ReadUnit, ...],
    view: memoryview,
) -> None:
    """纯 Python 回退：每个读取单元一次 ``preadv``，零拷贝落进 staging。"""
    storage = version.storage
    if not isinstance(storage, LocalStorage):  # pragma: no cover - load_local 只建 LocalStorage
        raise TypeError(f"python read engine requires LocalStorage, got {type(storage).__name__}")
    for unit in units:
        storage.read_range_into(
            version.file_key(unit.file),
            unit.byte_offset,
            view[unit.staging_offset : unit.staging_offset + unit.length],
        )


class _NativeReader:
    """按文件复用 io_uring 读句柄：一个版本读下来，同一文件只 open 一次。

    ``LocalFile`` 自带 fd 与 ring，且只能在创建它的线程上使用 —— 与
    ``load_local`` 的单线程执行模型一致。句柄在 :meth:`close` 里统一关闭。
    """

    def __init__(self) -> None:
        self._files: dict[str, Any] = {}

    def read_units(
        self,
        version: RecoveredVersion,
        units: tuple[ReadUnit, ...],
        view: memoryview,
    ) -> None:
        """按文件分组，一组一次 ``read_ranges``：一次提交读多段，零拷贝落进 staging。"""
        grouped: dict[str, list[ReadUnit]] = {}
        for unit in units:
            grouped.setdefault(unit.file, []).append(unit)
        storage = version.storage
        for name, entries in grouped.items():
            key = version.file_key(name)
            handle = self._files.get(key)
            if handle is None:
                handle = self._open(storage, key)
                self._files[key] = handle
            ranges_for_handle = [
                (unit.byte_offset, unit.length, unit.staging_offset) for unit in entries
            ]
            handle.read_ranges(ranges_for_handle, view)

    def close(self) -> None:
        while self._files:
            _, handle = self._files.popitem()
            handle.close()

    @staticmethod
    def _open(storage: Any, key: str) -> Any:
        import native

        if not isinstance(storage, LocalStorage):  # pragma: no cover
            raise TypeError(
                f"native read engine requires LocalStorage, got {type(storage).__name__}"
            )
        return native.LocalFile.open_read(str(storage.path_for(key)), direct_io=False)


def _as_view(buffer: Any) -> memoryview:
    view = getattr(buffer, "view", None)
    if isinstance(view, memoryview):
        return view
    return memoryview(buffer)


# --- 拷贝 -------------------------------------------------------------------


def _scatter(item: ReadItem, block: memoryview, host_arrays: Mapping[str, np.ndarray]) -> None:
    """把读块里的矩形按 ``CopySpan`` 拷进目标 host 数组的对应位置。"""
    source = np.frombuffer(block, dtype=numpy_dtype(item.dtype)).reshape(item.read_shape)
    target = host_arrays[item.fqn]
    for span in item.spans:
        dst = tuple(
            slice(offset, offset + size) for offset, size in zip(span.dst_offsets, span.shape)
        )
        target[dst] = source[span.src_region().slices]


def _to_device(
    target: TargetTensor,
    host: np.ndarray,
    device: Device,
    backend: DeviceBackend,
) -> Tensor:
    """目标 host 数组 -> device buffer：每个张量一次 H2D。"""
    if host.nbytes != target.nbytes:
        raise PlanError(
            f"{target.fqn}: host array {host.nbytes} bytes != target {target.nbytes}"
        )
    buffer = backend.allocate(host.nbytes)
    backend.copy_h2d(buffer, memoryview(host).cast("B"))
    backend.synchronize()
    return Tensor(
        name=target.fqn,
        buffer=buffer,
        device=device,
        dtype=target.dtype,
        shape=target.region.shape,
        backend=backend,
    )


__all__ = ["LoadReport", "load_local"]
