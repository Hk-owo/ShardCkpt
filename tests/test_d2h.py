"""D2H 搬运测试：字节一致、backend 统计、错误分支、拷贝 stream 语义。"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

import api as ckpt
from async_ import D2HStream, DeviceBackend, PinnedBufferPool, copy_into, copy_to_host, get_backend
from core import PoolError, UnsupportedDeviceError


def make_device_buffer(backend: DeviceBackend, payload: bytes) -> np.ndarray:
    """按生产路径准备一块 device buffer：allocate + H2D。"""
    buffer = backend.allocate(len(payload))
    backend.copy_h2d(buffer, memoryview(payload))
    backend.synchronize()
    return buffer


def test_copy_to_host_returns_independent_bytes() -> None:
    payload = bytes(range(256)) * 4
    device = ckpt.Device.parse("cuda:0")
    backend = get_backend(device)
    buffer = make_device_buffer(backend, payload)

    host = copy_to_host(buffer, device="cuda:0", backend=backend)
    assert isinstance(host, bytes)
    assert host == payload

    # 不传 backend 时按 device 从注册表取同一个 backend，结果一致
    assert copy_to_host(buffer, device="cuda:0") == payload

    # host 副本与 device 内存不再共享：改写 device buffer 不影响已取回的字节
    buffer[:] = 0
    assert host == payload


def test_copy_to_host_counts_backend_d2h_once() -> None:
    payload = b"\x01\x02\x03" * 1000
    device = ckpt.Device.parse("cuda:1")
    backend = get_backend(device)
    buffer = make_device_buffer(backend, payload)
    backend.stats.reset()

    copy_to_host(buffer, device=device, backend=backend)

    assert backend.stats.device_to_host_copies == 1
    assert backend.stats.device_to_host_bytes == len(payload)


def test_copy_to_host_rejects_bad_arguments() -> None:
    with pytest.raises(TypeError):
        copy_to_host(bytearray(8), device="cuda:0")  # type: ignore[arg-type]

    with pytest.raises(UnsupportedDeviceError):
        copy_to_host(np.zeros(8, dtype=np.uint8), device="cuda:99")


def test_tensor_tobytes_uses_same_d2h_path(dataset: Path) -> None:
    """Tensor.tobytes() 与 copy_to_host 共用一条路径，统计口径一致。"""
    data = ckpt.load(dataset, device="cuda:0")
    backend = get_backend(data.device)
    backend.stats.reset()

    for name, tensor in data.items():
        assert len(tensor.tobytes()) == tensor.nbytes, name

    assert backend.stats.device_to_host_copies == len(data)
    assert backend.stats.device_to_host_bytes == sum(t.nbytes for t in data.tensors.values())


# --- 拷贝 stream -----------------------------------------------------------


class FakeComputeStream:
    """模拟 compute stream：只记录自己被 synchronize 过的顺序。"""

    def __init__(self, order: list[str]) -> None:
        self.order = order

    def synchronize(self) -> None:
        self.order.append("sync")


def test_stream_submit_returns_handle() -> None:
    with D2HStream() as stream:
        handle = stream.submit(lambda: 42, name="answer", nbytes=8)

        assert handle.name == "answer"
        assert handle.nbytes == 8
        assert handle.wait(timeout=5) == 42
        assert handle.done
        assert handle.exception() is None

        stream.synchronize()
        assert stream.pending == 0
        assert stream.stats.submitted == 1
        assert stream.stats.completed == 1
        assert stream.stats.bytes == 8


def test_stream_synchronize_waits_for_all() -> None:
    # max_inflight=3 才允许三次提交同时在飞；=1 时第二次 submit 就会阻塞
    with D2HStream(max_inflight=3) as stream:
        for _ in range(3):
            stream.submit(lambda: time.sleep(0.01), nbytes=0)
        assert stream.pending == 3

        stream.synchronize()

        assert stream.pending == 0
        assert stream.stats.completed == 3


def test_stream_max_inflight_limits_submissions() -> None:
    stream = D2HStream(max_inflight=1)
    started = threading.Event()
    gate = threading.Event()
    accepted: list[object] = []

    def blocking() -> str:
        started.set()
        gate.wait(5)
        return "first"

    try:
        first = stream.submit(blocking)
        assert started.wait(5), "第一个拷贝应该已经开始"

        def submit_second() -> None:
            accepted.append(stream.submit(lambda: "second"))

        worker = threading.Thread(target=submit_second)
        worker.start()
        time.sleep(0.1)

        assert stream.stats.submitted == 1, "在途到上限时第二个提交必须被挡住"
        assert accepted == []

        gate.set()
        worker.join(5)
        assert not worker.is_alive(), "gate 打开后第二个提交应该放行"
        assert stream.stats.submitted == 2
        assert first.wait(5) == "first"
        accepted[0].wait(5)
    finally:
        gate.set()
        stream.close()


def test_stream_wait_stream_orders_synchronize_before_copy() -> None:
    order: list[str] = []
    compute = FakeComputeStream(order)

    with D2HStream() as stream:
        stream.wait_stream(compute)
        handle = stream.submit(lambda: order.append("copy"))
        handle.wait(5)

    assert order == ["sync", "copy"], "拷贝前必须先等 compute stream"
    assert stream.stats.completed == 1


def test_stream_failure_propagates_on_synchronize() -> None:
    def boom() -> None:
        raise RuntimeError("cuda error: invalid device pointer")

    with D2HStream() as stream:
        handle = stream.submit(boom)

        with pytest.raises(RuntimeError):
            handle.wait(5)
        with pytest.raises(RuntimeError):
            stream.synchronize()

        assert stream.stats.failed == 1
        assert stream.stats.completed == 0


def test_stream_closed_submit_raises() -> None:
    stream = D2HStream()
    stream.close()

    with pytest.raises(PoolError):
        stream.submit(lambda: None)


def test_submit_copy_writes_into_pool_block(dataset: Path) -> None:
    data = ckpt.load(dataset, device="cuda:0")
    name = data.names()[0]
    tensor = data[name]

    pool = PinnedBufferPool(chunk_bytes=1 << 20)
    buffer = pool.acquire(tensor.nbytes)
    try:
        with D2HStream() as stream:
            handle = stream.submit_copy(tensor, buffer, offset=0)
            assert handle.wait(timeout=10) == tensor.nbytes
            assert handle.name == name
            stream.synchronize()
        assert buffer.view[: tensor.nbytes] == tensor.tobytes()
    finally:
        buffer.release()


def test_copy_into_respects_offset_and_capacity() -> None:
    payload = bytes(range(64))
    device = ckpt.Device.parse("cuda:0")
    backend = get_backend(device)
    src = backend.allocate(len(payload))
    backend.copy_h2d(src, memoryview(payload))

    host = bytearray(128)
    assert copy_into(src, host, offset=32, device=device, backend=backend) == len(payload)
    assert bytes(host[32:96]) == payload
    assert bytes(host[:32]) == b"\x00" * 32

    with pytest.raises(ValueError):
        copy_into(src, host, offset=96, device=device, backend=backend)
