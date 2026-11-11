"""C++ 内核绑定的测试：写读回、原子 rename、异常映射、线程归属、GIL 释放。

扩展没构建时整个文件 skip；构建方式见 `native/__init__.py`。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import native

pytestmark = pytest.mark.skipif(
    not native.available(), reason="ckpt_native 未构建（构建命令见 native/__init__.py）"
)

# 没构建时保持 None：模块级不抛错，交给上面的 skipif 跳过整组用例
native_mod = native.require() if native.available() else None
# 扩展已构建时才有这些快捷名（未构建会被 skipif 整组跳过）
rename_atomic = native_mod.rename_atomic if native_mod else None
fsync_dir = native_mod.fsync_dir if native_mod else None


def write_aligned(path: Path, payload: bytes, *, alignment: int = 4096) -> None:
    """把 payload 拷进一块对齐内存再整块写入（模拟 pinned 块直落盘）。"""
    view = native.aligned_view(max(len(payload), alignment), alignment)
    view.view[: len(payload)] = payload
    try:
        native_mod.write_file(str(path), [view.view[: len(payload)]], direct_io=True)
    finally:
        view.release()


# --- 基本写入 ---------------------------------------------------------------


def test_write_file_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "step_1.distcp"
    payload = bytes(range(256)) * 16

    write_aligned(target, payload)

    assert target.read_bytes()[: len(payload)] == payload


def test_write_file_multiple_chunks(tmp_path: Path) -> None:
    target = tmp_path / "multi.distcp"
    payloads = [b"A" * 4096, b"B" * 4096, b"C" * 4096]
    # O_DIRECT 要求 buffer 地址对齐：非末段不能直接用 bytes（地址由 CPython 给）。
    views = [native.aligned_view(len(payload), 4096) for payload in payloads]
    try:
        for view, payload in zip(views, payloads):
            view.view[:] = payload
        native_mod.write_file(str(target), [view.view for view in views], direct_io=True)
    finally:
        for view in views:
            view.release()

    written = target.read_bytes()
    assert written[: len(b"".join(payloads))] == b"".join(payloads)


def test_non_final_chunk_buffer_address_must_be_aligned(tmp_path: Path) -> None:
    """非末段的 buffer 地址不对齐必须被显式拒绝。

    不能把这条判断留给文件系统：不同内核对非对齐 O_DIRECT 的宽容度不同（有的
    EINVAL，有的静默接受），那样同一个写入序列会在不同机器上表现不一致。
    """
    target = tmp_path / "mid.bin"
    view = native.aligned_view(8192, 4096)
    try:
        view.view[:] = b"x" * 8192
        with pytest.raises(ValueError):
            native_mod.write_file(
                str(target), [view.view[1:4097], view.view[4096:8192]], direct_io=True
            )
    finally:
        view.release()


def test_final_chunk_buffer_address_may_be_unaligned(tmp_path: Path) -> None:
    """末段例外：内部转一次对齐缓冲并补零，调用方不用关心自己的 buffer 地址。"""
    target = tmp_path / "tail-unaligned.bin"
    payload = b"tail-unaligned" * 3
    view = native.aligned_view(4096, 4096)
    try:
        start = 1  # 故意让 buffer 地址不对齐
        view.view[start : start + len(payload)] = payload
        native_mod.write_file(
            str(target), [view.view[start : start + len(payload)]], direct_io=True
        )
    finally:
        view.release()

    written = target.read_bytes()
    assert written[: len(payload)] == payload
    assert len(written) == 4096


def test_write_file_without_direct_io_accepts_unaligned(tmp_path: Path) -> None:
    target = tmp_path / "plain.bin"
    payload = b"unaligned-bytes" * 7

    native_mod.write_file(str(target), [payload], direct_io=False)

    assert target.read_bytes()[: len(payload)] == payload


def test_final_chunk_is_zero_padded_to_block_boundary(tmp_path: Path) -> None:
    """O_DIRECT 的最后一段不足一块时自动补零（读取端按真实长度截断）。"""
    target = tmp_path / "tail.bin"
    payload = b"short"

    native_mod.write_file(str(target), [payload], direct_io=True)

    written = target.read_bytes()
    assert written[: len(payload)] == payload
    assert len(written) % 4096 == 0
    assert written[len(payload) :] == b"\x00" * (len(written) - len(payload))


def test_non_final_chunk_must_be_aligned(tmp_path: Path) -> None:
    """非末段必须对齐；这条由 C++ 校验并映射成 ValueError。"""
    target = tmp_path / "mid.bin"

    with pytest.raises(ValueError):
        native_mod.write_file(str(target), [b"x" * 100, b"y" * 4096], direct_io=True)


def test_local_file_write_at_and_sync(tmp_path: Path) -> None:
    target = tmp_path / "incremental.distcp"
    view = native.aligned_view(8192, 4096)
    try:
        file = native_mod.LocalFile.open_write(str(target), direct_io=True)
        file.write_at(0, view.view[:4096])
        file.write_at(4096, view.view[4096:])
        file.sync()
        file.close()
    finally:
        view.release()

    assert target.stat().st_size >= 8192


def test_local_file_zero_copy_keeps_python_buffer_alive(tmp_path: Path) -> None:
    """write_at 是零拷贝的：Sync() 之前不能丢引用，否则 C++ 会读到已回收内存。"""
    target = tmp_path / "keepalive.bin"
    view = native.aligned_view(4096, 4096)
    payload = b"Z" * 4096
    view.view[:] = payload

    file = native_mod.LocalFile.open_write(str(target), direct_io=True)
    file.write_at(0, view.view)
    file.sync()
    # 弱引用语义：write_at 之后即使调用方丢了 view，数据也应已经写出
    del view
    file.close()

    assert target.read_bytes()[:4096] == payload


# --- 提交原语 ---------------------------------------------------------------


def test_rename_atomic_and_fsync_dir(tmp_path: Path) -> None:
    src = tmp_path / "COMMIT.tmp"
    dst = tmp_path / "COMMIT"
    src.write_bytes(b"step=42")

    rename_atomic(str(src), str(dst))
    fsync_dir(str(tmp_path))

    assert dst.read_bytes() == b"step=42"
    assert not src.exists()


def test_rename_atomic_missing_source_is_not_found(tmp_path: Path) -> None:
    from storage.base import ObjectNotFoundError

    with pytest.raises(ObjectNotFoundError):
        rename_atomic(str(tmp_path / "nope"), str(tmp_path / "dst"))


def test_aligned_buffer_is_page_aligned() -> None:
    buffer = native_mod.AlignedBuffer.allocate(8192, 4096)

    assert buffer.size == 8192
    assert buffer.address % 4096 == 0


def test_alignment_must_be_power_of_two() -> None:
    with pytest.raises(ValueError):
        native_mod.AlignedBuffer.allocate(4096, 3)


# --- 绑定层必须做对的两件事 -------------------------------------------------


def test_local_file_rejects_cross_thread_use(tmp_path: Path) -> None:
    """io_uring 的 SQ/CQ 不支持多线程共享，误用要立刻报错。"""
    file = native_mod.LocalFile.open_write(str(tmp_path / "thread.bin"), direct_io=False)
    errors: list[BaseException] = []

    def use_from_other_thread() -> None:
        try:
            file.sync()
        except BaseException as exc:  # noqa: BLE001 - 这里就是要抓任意异常
            errors.append(exc)

    worker = threading.Thread(target=use_from_other_thread)
    worker.start()
    worker.join(5)

    assert errors and isinstance(errors[0], RuntimeError), errors
    file.close()


def test_write_file_releases_gil(tmp_path: Path) -> None:
    """阻塞 IO 期间必须放掉 GIL，否则后台写会和训练线程抢 GIL。"""
    payload_size = 64 << 20  # 64 MiB
    view = native.aligned_view(payload_size, 4096)
    ticks = 0
    stop = threading.Event()

    def counter() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            time.sleep(0.001)

    spinner = threading.Thread(target=counter)
    spinner.start()
    try:
        time.sleep(0.01)  # 让计数线程先跑起来
        baseline = ticks
        native_mod.write_file(str(tmp_path / "gil.bin"), [view.view], direct_io=True)
        during = ticks - baseline
    finally:
        stop.set()
        spinner.join(5)
        view.release()

    # 若持有 GIL，计数线程在整段写入期间一次都跑不到
    assert during > 3, f"写入期间另一个 Python 线程只推进了 {during} 次，GIL 可能没释放"


# --- 与 Python 侧链路闭环 ---------------------------------------------------


def test_pinned_block_flows_into_native_write(tmp_path: Path, dataset: Path) -> None:
    """pinned 池的块 -> Tensor.copy_to -> write_file：全程零拷贝落到本地文件。"""
    import api as ckpt

    data = ckpt.load(dataset, device="cuda:0")
    name = data.names()[0]
    tensor = data[name]

    block = native.aligned_view(max(tensor.nbytes, 4096), 4096)
    try:
        copied = tensor.copy_to(block.view)
        assert copied == tensor.nbytes

        target = tmp_path / (name.replace(".", "_") + ".bin")
        native_mod.write_file(str(target), [block.view[: tensor.nbytes]], direct_io=True)

        assert target.read_bytes()[: tensor.nbytes] == tensor.tobytes()
    finally:
        block.release()


# --- 批量读：read_file_ranges / LocalFile.read_ranges -----------------------


def test_read_file_ranges_reads_multiple_regions(tmp_path: Path) -> None:
    """多段随机读：每段的偏移与落点各自独立，一次调用全部填好。"""
    target = tmp_path / "ranges.bin"
    payload = bytes(range(256)) * 64
    native_mod.write_file(str(target), [payload], direct_io=False)

    buffer = bytearray(3 * 1024)
    native_mod.read_file_ranges(
        str(target),
        [(0, 1024, 0), (2048, 512, 1024), (len(payload) - 1024, 1024, 1536)],
        buffer,
    )

    assert bytes(buffer[:1024]) == payload[:1024]
    assert bytes(buffer[1024:1536]) == payload[2048:2560]
    assert bytes(buffer[1536:2560]) == payload[-1024:]


def test_read_file_ranges_short_read_is_an_error(tmp_path: Path) -> None:
    target = tmp_path / "short.bin"
    native_mod.write_file(str(target), [b"x" * 4096], direct_io=False)

    with pytest.raises(Exception) as info:
        native_mod.read_file_ranges(str(target), [(4096 - 4, 16, 0)], bytearray(16))
    # 短读走 kIoError -> 项目内的 StorageError
    assert type(info.value).__name__ == "StorageError"


def test_read_file_ranges_validates_destination(tmp_path: Path) -> None:
    target = tmp_path / "bounds.bin"
    native_mod.write_file(str(target), [b"y" * 4096], direct_io=False)

    with pytest.raises(ValueError, match="exceeds the destination buffer"):
        native_mod.read_file_ranges(str(target), [(0, 32, 24)], bytearray(32))
    with pytest.raises(ValueError, match="length must be > 0"):
        native_mod.read_file_ranges(str(target), [(0, 0, 0)], bytearray(32))


def test_local_file_read_ranges_reuses_the_fd(tmp_path: Path) -> None:
    """读模式句柄自带 ring/fd：多次批量读不必重复 open。"""
    target = tmp_path / "reuse.bin"
    payload = bytes(range(256)) * 16
    native_mod.write_file(str(target), [payload], direct_io=False)

    file = native_mod.LocalFile.open_read(str(target), direct_io=False)
    try:
        first = bytearray(64)
        second = bytearray(64)
        file.read_ranges([(0, 64, 0)], first)
        file.read_ranges([(128, 64, 0)], second)
    finally:
        file.close()

    assert bytes(first) == payload[:64]
    assert bytes(second) == payload[128:192]


def test_read_ranges_requires_read_mode(tmp_path: Path) -> None:
    file = native_mod.LocalFile.open_write(str(tmp_path / "write.bin"), direct_io=False)
    try:
        with pytest.raises(ValueError):
            file.read_ranges([(0, 16, 0)], bytearray(16))
    finally:
        file.close()
