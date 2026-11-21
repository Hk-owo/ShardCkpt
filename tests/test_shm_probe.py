"""M0 探针：验证 memfd_create + mmap + SCM_RIGHTS 跨 spawn 进程可见性与 O_DIRECT 写盘。

真机待验项记录（无 GPU 环境无法实测，见 docs/planned/save_async.md §4 & §13）：
1. ``cudaHostRegister(ptr, size, cudaHostRegisterMapped)`` 对 ``memfd_create + mmap``
   共享内存页的注册可用性（含 fork/spawn 后的映射稳定性）；
2. 已 ``cudaHostRegister`` 锁页的 ``memfd`` 内存切片直接作为 ``O_DIRECT``（``native.write_file``）
   源缓冲区时的内核 DMA 兼容性与吞吐表现。
"""

from __future__ import annotations

import mmap
import multiprocessing
import os
import socket
from pathlib import Path

import native
from storage.local import DEFAULT_ALIGNMENT, LocalStorage, _address_of


def _probe_child(sock: socket.socket, size: int, offset: int, length: int) -> None:
    """spawn 子进程：通过 recv_fds 接收 memfd 并 mmap，验证读写与 O_DIRECT 落盘。"""
    try:
        msg, fds, _flags, _addr = socket.recv_fds(sock, 64, 1)
        assert msg == b"MEMFD"
        assert len(fds) == 1
        fd = fds[0]
        try:
            mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            mv = memoryview(mm)
            try:
                base_addr = _address_of(mv)
                assert base_addr is not None and base_addr % DEFAULT_ALIGNMENT == 0
                block = mv[offset : offset + length]
                block_addr = _address_of(block)
                assert block_addr is not None and block_addr % DEFAULT_ALIGNMENT == 0

                parent_bytes = bytes(block)
                if parent_bytes != (b"P" * length):
                    raise AssertionError("child observed mismatched parent bytes")

                # 回写子进程数据供父进程校验
                block[:] = b"C" * length
                sock.sendall(b"OK")
            finally:
                if "block" in locals():
                    block.release()
                mv.release()
                mm.close()
        finally:
            os.close(fd)
    finally:
        sock.close()


def test_memfd_cross_process_visible_and_odirect_aligned(tmp_path: Path) -> None:
    """M0：父写子读、子写父读字节一致；基址与块偏移 4096 对齐且走 native O_DIRECT 写盘。"""
    size = 16 * 1024
    offset = 4096
    length = 8192

    fd = os.memfd_create("ckpt-probe", os.MFD_CLOEXEC)
    try:
        os.ftruncate(fd, size)
        mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        mv = memoryview(mm)
        try:
            base_addr = _address_of(mv)
            assert base_addr is not None and base_addr % DEFAULT_ALIGNMENT == 0

            block = mv[offset : offset + length]
            block_addr = _address_of(block)
            assert block_addr is not None and block_addr % DEFAULT_ALIGNMENT == 0
            block[:] = b"P" * length

            parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            ctx = multiprocessing.get_context("spawn")
            proc = ctx.Process(
                target=_probe_child, args=(child_sock, size, offset, length)
            )
            proc.start()
            child_sock.close()
            try:
                socket.send_fds(parent_sock, [b"MEMFD"], [fd])
                ack = parent_sock.recv(16)
                assert ack == b"OK"
            finally:
                parent_sock.close()
                proc.join(timeout=10)
            assert proc.exitcode == 0

            # 父进程能看到子进程回写的字节
            assert bytes(block) == b"C" * length

            # 验证 LocalStorage.write_view 对 memfd 切片判定对齐，并走 native O_DIRECT 写盘
            storage = LocalStorage(
                tmp_path, direct_io=True, alignment=DEFAULT_ALIGNMENT, engine="auto"
            )
            assert storage._is_aligned(block, block) is True
            if native.available():
                assert storage.engine == "native"
            storage.write_view("step_0/probe.distcp", block)
            assert (tmp_path / "step_0/probe.distcp").read_bytes() == b"C" * length
        finally:
            if "block" in locals():
                block.release()
            mv.release()
            mm.close()
    finally:
        os.close(fd)
