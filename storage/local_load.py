"""本地层的读闭环：COMMIT 门禁 → 读取计划 → 批量读 → staging → H2D → device。

数据路径与 :func:`api.save_local` 严格镜像：

```
open_version(选版本 + 校验索引) ──> targets_for_layout(新布局的目标区域)
   ──> build_load_plan(求交 → 读取计划) ──> 按窗口取 pinned staging 块
   ──> io_uring 批量 read_ranges(零拷贝落进 staging) ──> [可选] crc32 校验
   ──> 按 CopySpan 拷进目标 host 数组 ──> H2D ──> Tensor
```

**CRC 默认不校验**（``verify_crc=False``）。本地目录在这里只是「上传对象存储之前的
暂存」：完整性由 rename + ``COMMIT`` 的文件清单兜底，字节正确性由上传时的对象级
校验和（``storage/object.py``）兜底，读路径再逐块重算一遍纯属重复开销。需要自证
（离线排查、介质可疑、恢复演练）时显式传 ``verify_crc=True``。

三个容易做错的地方，这里都定死了：

- **读块与分片的区别**：``ReadItem.length`` 是「读进来的字节」（``whole`` 策略下有读
  放大），索引里的分片 crc32 只对**整分片读**成立，所以 ``verify_crc`` 只校验读块
  与分片字节区间完全一致的那些项，其余记进 ``LoadReport.unverified``（不静默假装校验过）。
- **staging 生命周期**：块从 pinned 池借出，窗口内读完、拷完才归还；H2D 之后 device
  内存自己持有数据，与 staging 无关。
- **短读不静默截断**：C++ 侧 ``ReadRanges`` 与 Python 回退都报错，不会返回半份张量。
"""

from __future__ import annotations

import zlib
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
from core.index import CheckpointIndex, ShardMeta
from core.metadata import Device, numpy_dtype
from planner.load_plan import ReadItem, TargetTensor, build_load_plan
from planner.targets import targets_for_layout
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
    crc_verified: bool = False
    """本次是否执行了分片 CRC 校验（即 ``verify_crc`` 的生效值）。默认不校验。"""
    verified_blocks: int = 0
    """通过整分片 CRC 校验的读块数；``crc_verified=False`` 时恒为 0。"""
    unverified: tuple[str, ...] = ()
    """要求校验、但读块覆盖不到一个完整分片（结构上无法校验）的项。

    ``column_strategy="strided"`` 下非空。它**不**包含「按策略不校验」的读块——
    那种情况由 ``crc_verified=False`` 统一表示，不逐项列举，否则默认配置下这个
    tuple 会和读块数一样大。
    """

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
    verify_crc: bool = False,
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
    ``verify_crc``    逐块校验分片 crc32；**默认关闭**，见模块 docstring 的说明
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
    verified_blocks = 0
    unverified: set[str] = set()
    try:
        for window in plan.windows(budget_bytes):
            total = sum(item.length for item in window)
            buffer = pool.acquire(total)
            try:
                view = _as_view(buffer)
                offsets = _window_offsets(window)
                if reader is not None:
                    reader.read_window(version, window, offsets, view)
                else:
                    _read_window_python(version, window, offsets, view)
                for item, offset in zip(window, offsets):
                    block = view[offset : offset + item.length]
                    if verify_crc:
                        if _verify_block(version.index, item, block):
                            verified_blocks += 1
                        else:
                            unverified.add(_block_label(item))
                    _scatter(item, block, host_arrays)
                requests += len(window)
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
        unverified=tuple(sorted(unverified)),
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


def _window_offsets(window: tuple[ReadItem, ...]) -> list[int]:
    """窗口内每个读块在 staging 里的偏移：按请求顺序紧凑排布。"""
    offsets: list[int] = []
    cursor = 0
    for item in window:
        offsets.append(cursor)
        cursor += item.length
    return offsets


def _read_window_python(
    version: RecoveredVersion,
    window: tuple[ReadItem, ...],
    offsets: list[int],
    view: memoryview,
) -> None:
    """纯 Python 回退：每个读块一次 ``preadv``，同样是零拷贝落进 staging。"""
    storage = version.storage
    if not isinstance(storage, LocalStorage):  # pragma: no cover - load_local 只建 LocalStorage
        raise TypeError(f"python read engine requires LocalStorage, got {type(storage).__name__}")
    for item, offset in zip(window, offsets):
        storage.read_range_into(
            version.file_key(item.file),
            item.byte_offset,
            view[offset : offset + item.length],
        )


class _NativeReader:
    """按文件复用 io_uring 读句柄：一个版本读下来，同一文件只 open 一次。

    ``LocalFile`` 自带 fd 与 ring，且只能在创建它的线程上使用 —— 与
    ``load_local`` 的单线程执行模型一致。句柄在 :meth:`close` 里统一关闭。
    """

    def __init__(self) -> None:
        self._files: dict[str, Any] = {}

    def read_window(
        self,
        version: RecoveredVersion,
        window: tuple[ReadItem, ...],
        offsets: list[int],
        view: memoryview,
    ) -> None:
        """按文件分组，一组一次 ``read_ranges``：一次提交读多段，零拷贝落进 staging。"""
        grouped: dict[str, list[tuple[ReadItem, int]]] = {}
        for item, offset in zip(window, offsets):
            grouped.setdefault(item.file, []).append((item, offset))
        storage = version.storage
        for name, entries in grouped.items():
            key = version.file_key(name)
            handle = self._files.get(key)
            if handle is None:
                handle = self._open(storage, key)
                self._files[key] = handle
            ranges = [(item.byte_offset, item.length, offset) for item, offset in entries]
            handle.read_ranges(ranges, view)

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


# --- 校验与拷贝 -------------------------------------------------------------


def _block_label(item: ReadItem) -> str:
    return f"{item.file}@{item.byte_offset}+{item.length}"


def _verify_block(index: CheckpointIndex, item: ReadItem, block: memoryview) -> bool:
    """读块正好是索引里的一个完整分片时校验 crc32；其余情况返回 False。

    返回 False 表示「这个读块没有可用的整分片校验和」，调用方据此记 unverified；
    校验和不匹配抛 :class:`ChecksumError`，不会返回坏数据。
    """
    shard = _matching_shard(index, item)
    if shard is None or shard.crc32c is None:
        return False
    actual = f"{zlib.crc32(block) & 0xFFFFFFFF:08x}"
    if actual != shard.crc32c:
        raise ChecksumError(
            f"{item.fqn}: crc32 {actual} != index {shard.crc32c} "
            f"({item.file}@{item.byte_offset}+{item.length})"
        )
    return True


def _matching_shard(index: CheckpointIndex, item: ReadItem) -> ShardMeta | None:
    meta = index.tensors.get(item.fqn)
    if meta is None:
        return None
    for shard in meta.shards:
        if (
            shard.file == item.file
            and shard.byte_offset == item.byte_offset
            and shard.length == item.length
        ):
            return shard
    return None


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
