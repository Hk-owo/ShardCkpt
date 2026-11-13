"""测试用的最小 S3 兼容服务。

只实现 checkpoint 读写用得到的操作，用来在没有 SeaweedFS / MinIO 的环境里
端到端验证 boto3 客户端：

- PUT / GET（含 ``Range``）/ HEAD / DELETE 对象
- ``PUT`` + ``x-amz-copy-source`` 的 copy（rename_or_put 依赖）
- ``GET ?list-type=2`` 的 ListObjectsV2 单页列举
- multipart：``POST ?uploads`` → ``PUT ?partNumber&uploadId`` → ``POST ?uploadId``，
  以及 ``DELETE ?uploadId`` 的 abort
- 对象级校验和：PUT / upload_part 带 ``x-amz-checksum-*`` 时**真的重算并比对**，
  不匹配回 400 ``BadDigest``；GET 带 ``x-amz-checksum-mode: ENABLED`` 时回传
  服务端记录的值（所以把 ``store.objects`` 里的内容改掉就能模拟静默损坏）
- 条件写：PUT 与 ``CompleteMultipartUpload`` 求值 ``If-Match`` / ``If-None-Match``
  （412 / 404 / 409 语义对齐 AWS S3），DELETE 只认 ``If-Match``；
  ``ignore_conditional_headers`` 可以模拟「接受条件头但根本不执行」的端点

不校验签名，不分页（对象数远小于 1000），也不校验分片大小。响应格式对齐 S3，
boto3 能直接解析；``ObjectStore.requests`` 记录请求流水，供测试断言走的哪条路径。
"""

from __future__ import annotations

import base64
import hashlib
import struct
import sys
import threading
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator
from urllib.parse import parse_qs, unquote, urlparse
from xml.sax.saxutils import escape

# --- 校验和 -----------------------------------------------------------------

_CHECKSUM_ALGORITHMS = ("crc32", "crc32c", "sha1", "sha256")
"""S3 的校验和头是小写算法名：``x-amz-checksum-crc32`` / ``crc32c`` / ``sha1`` / ``sha256``。"""

_CRC32C_POLY = 0x82F63B78
_CRC32C_TABLE: list[int] = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ _CRC32C_POLY if _c & 1 else _c >> 1
    _CRC32C_TABLE.append(_c)


def _crc32c(payload: bytes) -> int:
    """CRC32C（Castagnoli）。纯 Python 查表：桩只处理测试规模的数据。"""
    crc = 0xFFFFFFFF
    for byte in payload:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def checksum_b64(payload: bytes, algorithm: str) -> str:
    """按算法算 base64 摘要，格式与 ``x-amz-checksum-<alg>`` 头一致。"""
    if algorithm == "crc32":
        digest = struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    elif algorithm == "crc32c":
        digest = struct.pack(">I", _crc32c(payload))
    elif algorithm == "sha1":
        digest = hashlib.sha1(payload).digest()
    elif algorithm == "sha256":
        digest = hashlib.sha256(payload).digest()
    else:  # pragma: no cover - 调用点只传上面四种
        raise ValueError(f"unsupported checksum algorithm {algorithm!r}")
    return base64.b64encode(digest).decode("ascii")


class ObjectStore:
    """进程内对象表，服务与断言共用。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.objects: dict[str, bytes] = {}
        # upload_id -> {part_number: bytes}，complete 后合成 objects 里的对象
        self.uploads: dict[str, dict[int, bytes]] = {}
        # (method, path, query) 的请求流水，测试用它断言走的是哪条路径
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        # 测试注入：让 upload_part 返回 500，验证 abort 清理
        self.fail_upload_part = False
        # 对象 key -> {"algorithm", "value"}：服务端自己算的校验和，GET 带
        # x-amz-checksum-mode: ENABLED 时回传。只改 objects 里的内容而不动这里，
        # 就能模拟「存进去的字节坏了、而校验和还停在写入时的值」。
        self.checksums: dict[str, dict[str, str]] = {}
        # 客户端实际带来的校验和头：(object_key, algorithm, 头里的值)
        self.checksum_headers: list[tuple[str, str, str]] = []
        # 测试注入：打开后不带校验和头的 PUT 直接 400，用来证明客户端确实在发
        self.require_checksum = False
        # 条件写请求流水：(method, path, {条件头})。requests 是三元的、被既有用例
        # 按位置解包，所以条件头单独记，不动原来的形状。
        self.conditional_requests: list[tuple[str, str, dict[str, str]]] = []
        # 测试注入：打开后完全忽略 If-Match / If-None-Match（模拟不执行条件写的端点）
        self.ignore_conditional_headers = False
        # 测试注入：接下来 N 次 CompleteMultipartUpload 返回 409 ConditionalRequestConflict，
        # 且这次上传被服务端判死（客户端必须重新 CreateMultipartUpload）
        self.fail_complete_times = 0


def _etag(payload: bytes) -> str:
    return '"' + hashlib.md5(payload).hexdigest() + '"'


def _etag_in(header: str, etag: str) -> bool:
    """条件头里是否包含这个 ETag：支持逗号列表、``*`` 与带/不带引号。"""
    target = etag.strip().strip('"')
    for item in header.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "*" or item.strip('"') == target:
            return True
    return False


def _make_handler(store: ObjectStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:  # 静默，避免污染 pytest 输出
            pass

        def parse_request(self) -> bool:
            ok = super().parse_request()
            if ok:
                parsed = urlparse(self.path)
                query = {k: v[0] for k, v in parse_qs(parsed.query, keep_blank_values=True).items()}
                with store.lock:
                    store.requests.append((self.command, unquote(parsed.path), query))
            return ok

        # --- 请求分发 -------------------------------------------------------
        def do_PUT(self) -> None:
            bucket, key = self._target()
            if not key:
                self._error(400, "InvalidURI", "missing object key")
                return
            query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
            if "partNumber" in query and "uploadId" in query:
                self._upload_part(bucket, key, query)
                return
            length = int(self.headers.get("Content-Length") or 0)
            if self.headers.get("x-amz-copy-source"):
                source = unquote(self.headers["x-amz-copy-source"]).lstrip("/")
                source_bucket, _, source_key = source.partition("/")
                with store.lock:
                    payload = store.objects.get(source_key) if source_bucket == bucket else None
                    if payload is None:
                        self._error(404, "NoSuchKey", source_key)
                        return
                    store.objects[key] = payload
                body = (
                    "<CopyObjectResult><ETag>"
                    + escape(_etag(payload))
                    + "</ETag></CopyObjectResult>"
                ).encode()
                self._send(200, body, "application/xml")
                return

            payload = self.rfile.read(length) if length else b""
            with store.lock:
                existing = store.objects.get(key)
            if self._precondition_failed(
                key, _etag(existing) if existing is not None else None
            ):
                return
            if not self._verify_checksum(key, payload, required=store.require_checksum):
                return
            with store.lock:
                store.objects[key] = payload
            self._send(200, b"", etag=_etag(payload))

        def do_GET(self) -> None:
            bucket, key = self._target()
            query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
            if "list-type" in query:
                self._list(bucket, query.get("prefix", [""])[0])
                return
            with store.lock:
                payload = store.objects.get(key)
            if payload is None:
                self._error(404, "NoSuchKey", key)
                return

            range_header = self.headers.get("Range")
            if range_header:
                self._send_range(payload, range_header)
                return
            self._send(
                200,
                payload,
                "application/octet-stream",
                etag=_etag(payload),
                extra_headers=self._checksum_response_headers(key),
            )

        def do_HEAD(self) -> None:
            _, key = self._target()
            with store.lock:
                payload = store.objects.get(key)
            if payload is None:
                self._error(404, "NoSuchKey", key)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("ETag", _etag(payload))
            for name, value in self._checksum_response_headers(key).items():
                self.send_header(name, value)
            self.end_headers()

        def do_POST(self) -> None:
            bucket, key = self._target()
            query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
            if "uploads" in query:
                self._create_multipart(key)
                return
            if "uploadId" in query:
                self._complete_multipart(bucket, key, query["uploadId"][0])
                return
            self._error(400, "InvalidRequest", "unsupported POST")

        def do_DELETE(self) -> None:
            _, key = self._target()
            query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
            if "uploadId" in query:
                # abort_multipart_upload
                with store.lock:
                    store.uploads.pop(query["uploadId"][0], None)
                self._send(204, b"")
                return
            if_match = self.headers.get("If-Match")
            with store.lock:
                existing = store.objects.get(key)
                if if_match is not None and not store.ignore_conditional_headers:
                    store.conditional_requests.append(
                        ("DELETE", unquote(urlparse(self.path).path), {"If-Match": if_match})
                    )
            if if_match is not None and not store.ignore_conditional_headers:
                if existing is None:
                    self._error(404, "NoSuchKey", key)
                    return
                if not _etag_in(if_match, _etag(existing)):
                    self._error(412, "PreconditionFailed", f"{key}: If-Match not satisfied")
                    return
            with store.lock:
                store.objects.pop(key, None)
            self._send(204, b"")

        # --- 辅助 -----------------------------------------------------------
        def _target(self) -> tuple[str, str]:
            path = urlparse(self.path).path.lstrip("/")
            bucket, _, key = path.partition("/")
            return unquote(bucket), unquote(key)

        def _precondition_failed(self, key: str, existing_etag: str | None) -> bool:
            """按 RFC 7232 求值 ``If-Match`` / ``If-None-Match``；不满足时已回错误。

            ``If-None-Match: *`` 的语义是「只有对象不存在才通过」；精确 ETag 列表则是
            「当前 ETag 不在列表里才通过」。``If-Match`` 缺对象回 404、ETag 不符回 412
            —— 与 AWS S3 的 conditional writes 文档一致。
            """
            if store.ignore_conditional_headers:
                return False
            headers = {
                name: self.headers[name]
                for name in ("If-Match", "If-None-Match")
                if self.headers.get(name) is not None
            }
            if headers:
                with store.lock:
                    store.conditional_requests.append(
                        (self.command, unquote(urlparse(self.path).path), headers)
                    )
            if_none_match = headers.get("If-None-Match")
            if if_none_match is not None and existing_etag is not None:
                matched = (
                    True
                    if if_none_match.strip() == "*"
                    else _etag_in(if_none_match, existing_etag)
                )
                if matched:
                    self._error(412, "PreconditionFailed", f"{key}: If-None-Match not satisfied")
                    return True
            if_match = headers.get("If-Match")
            if if_match is not None:
                if existing_etag is None:
                    self._error(404, "NoSuchKey", key)
                    return True
                if not _etag_in(if_match, existing_etag):
                    self._error(412, "PreconditionFailed", f"{key}: If-Match not satisfied")
                    return True
            return False

        def _create_multipart(self, key: str) -> None:
            upload_id = uuid.uuid4().hex
            with store.lock:
                store.uploads[upload_id] = {}
            body = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                "<InitiateMultipartUploadResult>"
                f"<Key>{escape(key)}</Key>"
                f"<UploadId>{upload_id}</UploadId>"
                "</InitiateMultipartUploadResult>"
            ).encode()
            self._send(200, body, "application/xml")

        def _upload_part(self, bucket: str, key: str, query: dict[str, list[str]]) -> None:
            upload_id = query["uploadId"][0]
            number = int(query["partNumber"][0])
            length = int(self.headers.get("Content-Length") or 0)
            payload = self.rfile.read(length) if length else b""

            if store.fail_upload_part:
                self._error(500, "InternalError", "injected upload_part failure")
                return

            # upload_part_copy：从源对象的指定区间取数据，响应是 XML 而非 ETag 头
            copy_source = self.headers.get("x-amz-copy-source")
            if copy_source:
                source = unquote(copy_source).lstrip("/")
                source_bucket, _, source_key = source.partition("/")
                with store.lock:
                    source_payload = (
                        store.objects.get(source_key) if source_bucket == bucket else None
                    )
                if source_payload is None:
                    self._error(404, "NoSuchKey", source_key)
                    return
                payload = source_payload[self._copy_source_range(len(source_payload))]

            if not copy_source and not self._verify_checksum(
                key, payload, required=False, record=False
            ):
                return

            with store.lock:
                upload = store.uploads.get(upload_id)
                if upload is not None:
                    upload[number] = payload
            if upload is None:
                self._error(404, "NoSuchUpload", upload_id)
                return

            if copy_source:
                body = (
                    "<CopyPartResult><ETag>"
                    + escape(_etag(payload))
                    + "</ETag></CopyPartResult>"
                ).encode()
                self._send(200, body, "application/xml")
                return
            self._send(200, b"", etag=_etag(payload))

        def _copy_source_range(self, size: int) -> slice:
            header = self.headers.get("x-amz-copy-source-range")
            if not header:
                return slice(0, size)
            start_s, _, end_s = header.split("=", 1)[1].partition("-")
            start = int(start_s)
            return slice(start, min(int(end_s), size - 1) + 1)

        def _complete_multipart(self, bucket: str, key: str, upload_id: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)  # body 里的 Part 列表不校验，按 partNumber 排序合并

            # 注入的 409：条件写与并发删除撞上，服务端把这次 MPU 判死
            if store.fail_complete_times > 0:
                with store.lock:
                    store.fail_complete_times -= 1
                    store.uploads.pop(upload_id, None)
                self._error(409, "ConditionalRequestConflict", "injected conditional conflict")
                return

            with store.lock:
                existing = store.objects.get(key)
            if self._precondition_failed(
                key, _etag(existing) if existing is not None else None
            ):
                with store.lock:
                    store.uploads.pop(upload_id, None)
                return

            with store.lock:
                parts = store.uploads.pop(upload_id, None)
                payload = b""
                if parts is not None:
                    payload = b"".join(parts[n] for n in sorted(parts))
                    store.objects[key] = payload
                    # 合成后的整对象由服务端自己算一份，GET 回验用
                    store.checksums[key] = {
                        "algorithm": "crc32",
                        "value": checksum_b64(payload, "crc32"),
                    }
            if parts is None:
                self._error(404, "NoSuchUpload", upload_id)
                return
            body = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                "<CompleteMultipartUploadResult>"
                f"<Bucket>{escape(bucket)}</Bucket>"
                f"<Key>{escape(key)}</Key>"
                f"<ETag>{escape(_etag(payload))}</ETag>"
                "</CompleteMultipartUploadResult>"
            ).encode()
            self._send(200, body, "application/xml")

        def _send_range(self, payload: bytes, header: str) -> None:
            try:
                spec = header.split("=", 1)[1]
                start_s, _, end_s = spec.partition("-")
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else len(payload) - 1
            except (IndexError, ValueError):
                self._error(400, "InvalidRange", header)
                return
            if start < 0 or start >= len(payload) or end < start:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(payload)}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            chunk = payload[start : min(end, len(payload) - 1) + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Content-Range", f"bytes {start}-{start + len(chunk) - 1}/{len(payload)}")
            self.send_header("ETag", _etag(payload))
            self.end_headers()
            self.wfile.write(chunk)

        def _list(self, bucket: str, prefix: str) -> None:
            with store.lock:
                keys = sorted(k for k in store.objects if k.startswith(prefix))
                entries = [(k, len(store.objects[k]), _etag(store.objects[k])) for k in keys]
            body = [
                '<?xml version="1.0" encoding="UTF-8"?>',
                '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">',
                f"<Name>{escape(bucket)}</Name>",
                f"<Prefix>{escape(prefix)}</Prefix>",
                f"<KeyCount>{len(entries)}</KeyCount>",
                "<MaxKeys>1000</MaxKeys>",
                "<IsTruncated>false</IsTruncated>",
            ]
            for key, size, etag in entries:
                body.append(
                    "<Contents>"
                    f"<Key>{escape(key)}</Key>"
                    f"<Size>{size}</Size>"
                    f"<ETag>{escape(etag)}</ETag>"
                    "</Contents>"
                )
            body.append("</ListBucketResult>")
            self._send(200, "".join(body).encode(), "application/xml")

        def _error(self, status: int, code: str, message: str) -> None:
            body = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                f"<Error><Code>{escape(code)}</Code><Message>{escape(message)}</Message></Error>"
            ).encode()
            self._send(status, body, "application/xml")

        # --- 校验和 ---------------------------------------------------------
        def _verify_checksum(
            self, key: str, payload: bytes, *, required: bool, record: bool = True
        ) -> bool:
            """按客户端带来的 ``x-amz-checksum-*`` 头重算并比对。

            不匹配回 400 ``BadDigest``（与真实 S3 一致），返回 False 让调用方提前结束。
            ``record`` 为真时把服务端自己算的值记进 ``store.checksums``，供 GET 回传。
            """
            found = False
            for algorithm in _CHECKSUM_ALGORITHMS:
                header = self.headers.get(f"x-amz-checksum-{algorithm}")
                if not header:
                    continue
                found = True
                expected = checksum_b64(payload, algorithm)
                with store.lock:
                    store.checksum_headers.append((key, algorithm, header))
                if header != expected:
                    self._error(400, "BadDigest", f"{key}: checksum-{algorithm} mismatch")
                    return False
                if record:
                    with store.lock:
                        store.checksums[key] = {"algorithm": algorithm, "value": header}
            if required and not found:
                self._error(400, "InvalidRequest", f"{key}: checksum header required")
                return False
            if not found and record:
                # 真实 S3 总会在服务端算一份并存下来；不带头的上传也照样记录
                with store.lock:
                    store.checksums[key] = {
                        "algorithm": "crc32",
                        "value": checksum_b64(payload, "crc32"),
                    }
            return True

        def _checksum_response_headers(self, key: str) -> dict[str, str]:
            """GET/HEAD 带 ``x-amz-checksum-mode: ENABLED`` 时回传记录的校验和。"""
            mode = self.headers.get("x-amz-checksum-mode", "")
            if mode.upper() != "ENABLED":
                return {}
            with store.lock:
                entry = store.checksums.get(key)
            if not entry:
                return {}
            return {f"x-amz-checksum-{entry['algorithm']}": entry["value"]}

        def _send(
            self,
            status: int,
            body: bytes,
            content_type: str = "",
            *,
            etag: str = "",
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            if content_type:
                self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if etag:
                self.send_header("ETag", etag)
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            if body:
                self.wfile.write(body)

    return Handler


class _QuietServer(ThreadingHTTPServer):
    """boto3 重试 / 关闭连接时对端可能已经断开，这类噪声不打印 traceback。"""

    daemon_threads = True

    def handle_error(self, request, client_address) -> None:  # type: ignore[no-untyped-def]
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


class StubS3Server:
    """后台线程里的 S3 兼容服务。"""

    def __init__(self) -> None:
        self.store = ObjectStore()
        self._httpd = _QuietServer(("127.0.0.1", 0), _make_handler(self.store))
        host, port = self._httpd.server_address[:2]
        self.endpoint_url = f"http://{host}:{port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "StubS3Server":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)
        return False


def start() -> StubS3Server:
    return StubS3Server()


def iter_objects(server: StubS3Server) -> Iterator[tuple[str, bytes]]:
    with server.store.lock:
        yield from sorted(server.store.objects.items())
