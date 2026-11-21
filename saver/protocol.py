"""训练进程 ↔ 存储进程的控制面协议（§5.1）。

传输：``AF_UNIX SOCK_STREAM``，4 字节大端长度前缀 + UTF-8 JSON；
``memfd`` 通过 ``SCM_RIGHTS``（``socket.send_fds`` / ``socket.recv_fds``）随帧附带发送。
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
from typing import Any, Mapping, Sequence

from core.region import Interval, Region
from planner.save_plan import BlobItem, SavePlan, WriteItem

_HEADER_STRUCT = struct.Struct(">I")
_MAX_FRAME_BYTES = 256 << 20  # 256 MiB 上限保护

OP_READY = "READY"
OP_MEMFD = "MEMFD"
OP_STARTUP = "STARTUP"
OP_PLAN = "PLAN"
OP_NEED_PLAN = "NEED_PLAN"
OP_SNAP = "SNAP"
OP_PROGRESS = "PROGRESS"
OP_DONE = "DONE"
OP_FAIL = "FAIL"
OP_PROTECT = "PROTECT"
OP_UPLOAD_DONE = "UPLOAD_DONE"
OP_UPLOAD_FAIL = "UPLOAD_FAIL"
OP_CLOSE = "CLOSE"
OP_BYE = "BYE"


class ProtocolError(OSError):
    """控制面帧格式错误或对端意外断开。"""


def encode_frame(payload: Mapping[str, Any]) -> bytes:
    """编码单个 UTF-8 JSON 帧（含 4 字节长度前缀）。"""
    body = json.dumps(
        payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    if len(body) > _MAX_FRAME_BYTES:
        raise ProtocolError(f"frame too large: {len(body)} bytes")
    return _HEADER_STRUCT.pack(len(body)) + body


def send_msg(
    sock: socket.socket,
    payload: Mapping[str, Any],
    *,
    fds: Sequence[int] = (),
) -> None:
    """发送一帧 JSON 消息；若提供 ``fds`` 则通过 ``SCM_RIGHTS`` 随头字节发送。"""
    body = json.dumps(
        payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    if len(body) > _MAX_FRAME_BYTES:
        raise ProtocolError(f"frame too large: {len(body)} bytes")
    header = _HEADER_STRUCT.pack(len(body))
    if fds:
        sent = socket.send_fds(sock, [header], list(fds))
        if sent <= 0:
            raise ProtocolError("send_fds sent 0 bytes")
        if sent < len(header):
            sock.sendall(header[sent:])
    else:
        sock.sendall(header)
    if body:
        sock.sendall(body)


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = nbytes
    while remaining > 0:
        try:
            chunk = sock.recv(remaining)
        except InterruptedError:
            continue
        if not chunk:
            raise EOFError("unexpected EOF while reading frame body")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_msg(
    sock: socket.socket, *, max_fds: int = 4
) -> tuple[dict[str, Any], list[int]] | None:
    """读取一帧 JSON 消息及附带的 fd；对端正常关闭返回 ``None``。"""
    header_parts: list[bytes] = []
    header_len = 0
    fds: list[int] = []
    while header_len < _HEADER_STRUCT.size:
        try:
            chunk, new_fds, _flags, _addr = socket.recv_fds(
                sock, _HEADER_STRUCT.size - header_len, max_fds
            )
        except InterruptedError:
            continue
        if new_fds:
            fds.extend(new_fds)
        if not chunk:
            for fd in fds:
                _safe_close_fd(fd)
            if header_len == 0:
                return None
            raise EOFError("unexpected EOF while reading frame header")
        header_parts.append(chunk)
        header_len += len(chunk)

    header = b"".join(header_parts)
    (length,) = _HEADER_STRUCT.unpack(header)
    if length > _MAX_FRAME_BYTES:
        for fd in fds:
            _safe_close_fd(fd)
        raise ProtocolError(f"incoming frame too large: {length} bytes")
    try:
        body = _recv_exact(sock, length) if length else b"{}"
        data = json.loads(body.decode("utf-8"))
    except BaseException:
        for fd in fds:
            _safe_close_fd(fd)
        raise
    if not isinstance(data, dict):
        for fd in fds:
            _safe_close_fd(fd)
        raise ProtocolError(f"expected JSON object frame, got {type(data).__name__}")
    return data, fds


def _safe_close_fd(fd: int) -> None:
    import os

    try:
        os.close(fd)
    except OSError:
        pass


def plan_to_dict(plan: SavePlan) -> dict[str, Any]:
    """把 ``SavePlan`` 序列化为纯 JSON 字典（不含 skipped，worker 落盘只需要写项与小状态）。"""
    items = []
    for item in plan.items:
        entry: dict[str, Any] = {
            "fqn": item.fqn,
            "file": item.file,
            "byte_offset": item.byte_offset,
            "length": item.length,
            "global_shape": list(item.global_shape),
            "dtype": item.dtype,
            "replica_index": item.replica_index,
            "padding": item.padding,
        }
        if item.flat is not None:
            entry["flat"] = [item.flat.start, item.flat.length]
        else:
            assert item.region is not None
            entry["region"] = [list(item.region.offsets), list(item.region.shape)]
        items.append(entry)

    blobs = [
        {
            "name": blob.name,
            "file": blob.file,
            "byte_offset": blob.byte_offset,
            "length": blob.length,
            "padding": blob.padding,
        }
        for blob in plan.blobs
    ]
    return {
        "rank": plan.rank,
        "file": plan.file,
        "items": items,
        "blobs": blobs,
    }


def plan_from_dict(data: Mapping[str, Any]) -> SavePlan:
    """从 JSON 字典还原 ``SavePlan``。"""
    items: list[WriteItem] = []
    for raw in data.get("items", ()):
        region = None
        flat = None
        if "flat" in raw:
            flat = Interval(int(raw["flat"][0]), int(raw["flat"][1]))
        else:
            reg = raw["region"]
            region = Region(tuple(int(v) for v in reg[0]), tuple(int(v) for v in reg[1]))
        items.append(
            WriteItem(
                fqn=str(raw["fqn"]),
                file=str(raw["file"]),
                byte_offset=int(raw["byte_offset"]),
                length=int(raw["length"]),
                global_shape=tuple(int(v) for v in raw["global_shape"]),
                dtype=str(raw["dtype"]),
                region=region,
                flat=flat,
                replica_index=int(raw.get("replica_index", 0)),
                padding=int(raw.get("padding", 0)),
            )
        )
    blobs = tuple(
        BlobItem(
            name=str(b["name"]),
            file=str(b["file"]),
            byte_offset=int(b["byte_offset"]),
            length=int(b["length"]),
            padding=int(b.get("padding", 0)),
        )
        for b in data.get("blobs", ())
    )
    return SavePlan(
        rank=int(data["rank"]),
        file=str(data["file"]),
        items=tuple(items),
        blobs=blobs,
    )


def compute_plan_id(
    plan: SavePlan,
    *,
    layout: Mapping[str, int] | None = None,
    alignment: int,
    crc_chunk_max: int,
) -> str:
    """计算规范化 SavePlan + 布局 + 对齐 + CRC 上限的 blake2b 指纹（D8）。"""
    canonical = {
        "plan": plan_to_dict(plan),
        "layout": {str(k): int(v) for k, v in sorted((layout or {}).items())},
        "alignment": int(alignment),
        "crc_chunk_max": int(crc_chunk_max),
    }
    raw = json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def encode_blobs(blobs: Mapping[str, bytes] | None) -> dict[str, str]:
    """把轻量 blobs 编码为 base64 字符串字典以便放入 SNAP 帧。"""
    if not blobs:
        return {}
    return {
        str(name): base64.b64encode(bytes(payload)).decode("ascii")
        for name, payload in sorted(blobs.items())
    }


def decode_blobs(raw: Mapping[str, str] | None) -> dict[str, bytes]:
    """解码 SNAP 帧中的 base64 blobs。"""
    if not raw:
        return {}
    return {
        str(name): base64.b64decode(encoded.encode("ascii"))
        for name, encoded in raw.items()
    }


__all__ = [
    "OP_BYE",
    "OP_CLOSE",
    "OP_DONE",
    "OP_FAIL",
    "OP_MEMFD",
    "OP_NEED_PLAN",
    "OP_PLAN",
    "OP_PROGRESS",
    "OP_PROTECT",
    "OP_READY",
    "OP_SNAP",
    "OP_STARTUP",
    "OP_UPLOAD_DONE",
    "OP_UPLOAD_FAIL",
    "ProtocolError",
    "compute_plan_id",
    "decode_blobs",
    "encode_blobs",
    "encode_frame",
    "plan_from_dict",
    "plan_to_dict",
    "recv_msg",
    "send_msg",
]
