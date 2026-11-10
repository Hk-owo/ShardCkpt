"""device -> host 搬运：唯一的底层拷贝、异步拷贝 stream。

三层的分工：

- ``copy_into``：全项目唯一的 D2H 落点，不分配内存，拷进调用方给的 host buffer
  （pinned 池里的块就是从这里拿的）。``Tensor.tobytes()`` / ``numpy()`` / ``to()``
  最终都走这里，所以 ``BackendStats.device_to_host_*`` 精确反映搬运次数与字节。
- ``copy_to_host``：在 ``copy_into`` 之上分配一份 host 副本，返回 ``bytes``。
- ``D2HStream``：独立「拷贝 stream」。``submit`` 提交即返回 ``CopyHandle``（等价于
  文档里的 ``copy_done`` event），``wait_stream`` / ``synchronize`` / ``max_inflight``
  对应设计文档 §4 的 stream 同步与限流。

mock 环境下没有真实 CUDA stream，``D2HStream`` 用单线程执行器承载：提交即返回、
拷贝在后台完成，调用语义与真机一致，只是不涉及 PCIe。接入真机时只需替换 backend
与这里的执行器，调用方不用改。
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

import numpy as np

from core import PoolError
from core.metadata import Device

if TYPE_CHECKING:  # 仅为类型标注：本模块由 async_/__init__ 末尾导入，避免循环导入
    from async_ import DeviceBackend, Tensor

T = TypeVar("T")


def _as_view(buffer: Any) -> memoryview:
    """把 host buffer 规范成 memoryview：bytearray / numpy / PinnedBuffer 都接受。"""
    if isinstance(buffer, memoryview):
        return buffer
    view = getattr(buffer, "view", None)  # PinnedBuffer 暴露的是 memoryview
    if isinstance(view, memoryview):
        return view
    return memoryview(buffer)


def _backend_of(device: "str | Device", backend: "DeviceBackend | None") -> "DeviceBackend":
    if backend is not None:
        return backend
    from async_ import get_backend  # 延迟导入：本模块由 async_/__init__ 末尾导入

    return get_backend(Device.parse(device))


def copy_into(
    src: np.ndarray,
    dst: Any,
    *,
    offset: int = 0,
    device: "str | Device" = "cuda:0",
    backend: "DeviceBackend | None" = None,
) -> int:
    """把 device 上的 ``src`` 整块拷进 ``dst`` 的 ``[offset, offset + src.nbytes)``。

    这是全项目唯一的 D2H 落点：不分配、不额外复制，函数返回时拷贝已完成。
    返回拷贝的字节数。
    """
    if not isinstance(src, np.ndarray):
        raise TypeError(f"src must be a numpy device buffer, got {type(src).__name__}")
    view = _as_view(dst)
    end = offset + src.nbytes
    if offset < 0 or end > len(view):
        raise ValueError(
            f"host buffer too small: need [{offset}, {end}), buffer has {len(view)} bytes"
        )
    resolved = _backend_of(device, backend)
    resolved.copy_d2h(view[offset:end], src)
    resolved.synchronize()
    return src.nbytes


def copy_to_host(
    src: np.ndarray,
    *,
    device: "str | Device" = "cuda:0",
    backend: "DeviceBackend | None" = None,
) -> bytes:
    """把 device 上的 buffer 整块搬到 host 内存，返回独立的 ``bytes`` 副本。"""
    if not isinstance(src, np.ndarray):
        raise TypeError(f"src must be a numpy device buffer, got {type(src).__name__}")
    host = bytearray(src.nbytes)
    copy_into(src, memoryview(host), device=device, backend=backend)
    return bytes(host)


@dataclass(frozen=True)
class StreamStats:
    """拷贝 stream 的累计统计。"""

    submitted: int = 0
    completed: int = 0
    failed: int = 0
    bytes: int = 0
    peak_inflight: int = 0


@dataclass(eq=False)
class CopyHandle(Generic[T]):
    """一次已提交的拷贝，等价于文档里的 ``copy_done`` event。

    ``wait()`` 阻塞到完成并返回结果；``done`` 是非阻塞自检 —— 对应文档里
    「拷贝后记录 event，下一次 optimizer step 前 wait」的用法。
    """

    name: str
    nbytes: int
    _future: Future = field(repr=False)

    @property
    def done(self) -> bool:
        return self._future.done()

    def wait(self, timeout: float | None = None) -> T:
        return self._future.result(timeout)

    def result(self) -> T:
        return self._future.result()

    def exception(self) -> BaseException | None:
        return self._future.exception()


class D2HStream:
    """独立拷贝 stream：提交即返回，``max_inflight`` 限流，``synchronize`` 全部等完。

    真机上一条 stream 内部顺序执行，``max_inflight=1`` 对应这种串行语义；调大等于
    允许多个拷贝同时在飞（分块搬运 / 限流）。
    """

    def __init__(self, *, max_inflight: int = 1, name: str = "d2h") -> None:
        if max_inflight < 1:
            raise ValueError(f"max_inflight must be >= 1, got {max_inflight}")
        self.name = name
        self.max_inflight = max_inflight
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._slots = threading.Semaphore(max_inflight)
        self._cond = threading.Condition()
        self._inflight = 0
        self._waits: list[Any] = []
        self._errors: list[BaseException] = []
        self._stats = StreamStats()
        self._closed = False

    # --- 查询 ---------------------------------------------------------------
    @property
    def stats(self) -> StreamStats:
        with self._cond:
            return self._stats

    @property
    def pending(self) -> int:
        """在途拷贝数量。"""
        with self._cond:
            return self._inflight

    # --- stream 同步 --------------------------------------------------------
    def wait_stream(self, other: Any) -> None:
        """对应 ``copy_stream.wait_stream(compute_stream)``。

        登记后，**下一个**提交的任务会先 ``other.synchronize()`` 再开始拷贝。只消费
        一次就够：同一条 stream 上的后续任务天然排在它后面。
        """
        with self._cond:
            self._waits.append(other)

    def synchronize(self) -> None:
        """等完所有在途拷贝；只要有一次拷贝失败，就在这里抛出它的异常。"""
        with self._cond:
            while self._inflight > 0:
                self._cond.wait()
            errors, self._errors = self._errors, []
        if errors:
            raise errors[0]

    # --- 提交 ---------------------------------------------------------------
    def submit(
        self,
        fn: Callable[[], T],
        *,
        name: str = "",
        nbytes: int = 0,
    ) -> CopyHandle[T]:
        """把 ``fn`` 提交到拷贝 stream；在途数量到 ``max_inflight`` 时在这里阻塞。"""
        self._slots.acquire()  # 限流：在途到达上限就等
        try:
            with self._cond:
                if self._closed:
                    raise PoolError(f"stream {self.name!r} 已关闭")
                waits, self._waits = self._waits, []
                self._inflight += 1
                self._stats = StreamStats(
                    submitted=self._stats.submitted + 1,
                    completed=self._stats.completed,
                    failed=self._stats.failed,
                    bytes=self._stats.bytes,
                    peak_inflight=max(self._stats.peak_inflight, self._inflight),
                )
        except BaseException:
            self._slots.release()
            raise

        future: Future = self._executor.submit(self._job(fn, waits, nbytes))
        return CopyHandle(name=name, nbytes=nbytes, _future=future)

    def submit_copy(
        self,
        tensor: "Tensor",
        buffer: Any,
        *,
        offset: int = 0,
    ) -> CopyHandle[int]:
        """把一个张量 D2H 到 host buffer 的 ``offset`` 处（pinned 池里的块）。"""
        nbytes = tensor.nbytes
        capacity = len(_as_view(buffer))
        if offset < 0 or offset + nbytes > capacity:
            raise ValueError(
                f"{tensor.name}: 需要 [{offset}, {offset + nbytes})，"
                f"但 buffer 只有 {capacity} 字节"
            )
        return self.submit(
            lambda: tensor.copy_to(buffer, offset=offset),
            name=tensor.name,
            nbytes=nbytes,
        )

    # --- 生命周期 -----------------------------------------------------------
    def close(self) -> None:
        with self._cond:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "D2HStream":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        return (
            f"D2HStream(name={self.name!r}, max_inflight={self.max_inflight}, "
            f"pending={self.pending})"
        )

    # --- 内部 ---------------------------------------------------------------
    def _job(self, fn: Callable[[], T], waits: list[Any], nbytes: int) -> Callable[[], T]:
        def job() -> T:
            try:
                for other in waits:
                    other.synchronize()
                result = fn()
            except BaseException as exc:
                with self._cond:
                    self._errors.append(exc)
                    self._stats = StreamStats(
                        submitted=self._stats.submitted,
                        completed=self._stats.completed,
                        failed=self._stats.failed + 1,
                        bytes=self._stats.bytes,
                        peak_inflight=self._stats.peak_inflight,
                    )
                raise
            else:
                with self._cond:
                    self._stats = StreamStats(
                        submitted=self._stats.submitted,
                        completed=self._stats.completed + 1,
                        failed=self._stats.failed,
                        bytes=self._stats.bytes + nbytes,
                        peak_inflight=self._stats.peak_inflight,
                    )
                return result
            finally:
                with self._cond:
                    self._inflight -= 1
                    self._cond.notify_all()
                self._slots.release()

        return job


__all__ = [
    "CopyHandle",
    "D2HStream",
    "StreamStats",
    "copy_into",
    "copy_to_host",
]
