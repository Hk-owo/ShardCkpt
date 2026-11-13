"""对象存储后端：S3 兼容（SeaweedFS / MinIO / AWS S3），基于 boto3。

实现覆盖 checkpoint 读写需要的对象操作：

- 整对象 PUT / GET，以及 Range GET（按张量读取时只取需要的字节）
- 大对象走 multipart：create → upload_part × N → complete，失败自动 abort
- ListObjectsV2 分页列举
- rename_or_put 用 copy + delete 模拟（对象存储没有 rename）；源对象超过 5 GiB
  时自动改走 UploadPartCopy，分片大小按 10000 片上限放大
- delete 幂等：对象不存在不报错

超过 ``multipart_threshold``（默认 64 MiB）的对象切成分片上传，
其余走单次 PUT。凭据为空时交给 boto3 的默认链
（环境变量 / ~/.aws/credentials / IAM role）。

**对象存储只负责「发回来的 = 存的」。** 本地层的块级 CRC-32C 由 ``save_local``
写进索引，恢复端把版本数据文件整文件下载到本地临时目录后，再由
``storage.local_load.load_local`` 按索引里的 CRC 块校验（理由见 README「本地分片
CRC 与对象存储校验的分工」）。对象存储这一侧的传输/落盘由这里的对象级校验和负责：
PUT / multipart 带 ``ChecksumAlgorithm``（默认 ``CRC32``），服务端独立重算并比对，
不匹配回 400 ``BadDigest``；整对象 GET / 下载带 ``ChecksumMode=ENABLED`` 让
botocore 回验返回值。不认识 ``x-amz-checksum-*`` 的端点把
``S3Config.checksum_algorithm`` 设成 ``None`` 即可退回旧行为。

**条件写**是写路径的硬前置（对象存储端的代际与 fence 依赖它）：``put_if_absent``
走 ``If-None-Match: *``，``put_if_match`` 走 ``If-Match: <etag>``，两者都由服务端
原子判定，``412 PreconditionFailed`` 即「我没赢」。大对象的条件挂在
``CompleteMultipartUpload`` 上 —— ``CreateMultipartUpload`` / ``UploadPart`` 在 AWS
上不接受条件头，所以「分片上传」与「条件可见化」是同一条路径的两半。

``S3Config.conditional_writes="off"`` 把这条连接降成**只读**端点：read / range /
download / head / list 照常，任何写动作（含 ``write`` / ``delete``）直接抛
:class:`storage.base.ConditionalWritesDisabled`。不做「无 CAS 的单写者写入」——
那条路径的 fence 是静默失效的。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from storage.base import (
    ConditionalWritesDisabled,
    ObjectMeta,
    ObjectNotFoundError,
    StorageBackend,
    StorageError,
    UnsupportedConditionalWrites,
    WriteOutcome,
)

# boto3 的 S3 client；不引入 mypy_boto3_s3 之类的额外类型依赖。
S3Client = Any

DEFAULT_REGION = "us-east-1"

DEFAULT_DOWNLOAD_CHUNK_BYTES = 8 << 20
"""整对象下载时每次从 ``StreamingBody`` 读多少字节写盘；不把整个对象读进内存。"""

# S3 规定：除最后一片外，每个分片至少 5 MiB；单次 PUT / copy_object 上限 5 GiB。
MIN_PART_SIZE = 5 * 1024 * 1024
DEFAULT_PART_SIZE = 8 * 1024 * 1024
DEFAULT_MULTIPART_THRESHOLD = 64 * 1024 * 1024
MAX_PART_SIZE = 5 * 1024 * 1024 * 1024
# copy_object 的源对象上限；超过就必须走 UploadPartCopy。
MAX_SINGLE_COPY_BYTES = MAX_PART_SIZE
# 一次 multipart 最多 10000 片，分片大小要据此放大。
MAX_MULTIPART_PARTS = 10000

DEFAULT_CHECKSUM_ALGORITHM = "CRC32"
"""上传时附带的对象级校验和算法。

「从本地读出来 -> 网络 -> 落盘」这条链路唯一的把关点就是这里：PUT 带
``x-amz-checksum-<alg>``，服务端独立重算并比对，不匹配直接 400 ``BadDigest``；
GET 带 ``ChecksumMode=ENABLED`` 让 SDK 回验返回值。

选 ``CRC32`` 而不是 ``CRC32C``，这是与本地层**故意不同**的选择，不是遗漏：

- 这里的校验和是**服务端**独立算的，客户端只负责声明算法，检错能力与 CRC32C 同
  量级（都是 32 位），换算法带不来任何强度收益。
- botocore 在没有 ``awscrt`` 时只实现了 ``crc32``/``sha1``/``sha256``/``sha512``
  （``_CHECKSUM_CLS`` 里没有 ``crc32c``），改成 ``CRC32C`` 要么引入 ``awscrt``，
  要么自己算并显式写 ``x-amz-checksum-crc32c`` 头（multipart 还要逐片处理）。
- 本地层（:mod:`storage.checksum`）用 CRC-32C，是因为那边**需要自己算**才有得比，
  而 ``google-crc32c`` 走 SSE4.2 / ARMv8 硬件指令。两层的算法互不影响：本地校验
  和不上传，上传校验和不落本地索引。
"""

SUPPORTED_CHECKSUM_ALGORITHMS = frozenset({"CRC32", "CRC32C", "SHA1", "SHA256"})
"""S3 认的校验和算法。"""

CONDITIONAL_WRITE_MODES = frozenset({"require", "off"})
"""条件写模式。

- ``require``（默认）：写路径必须走 CAS；首次使用时跑端点探针（见
  :meth:`S3Storage.ensure_conditional_writes`），探针不过就拒绝启动写入。
- ``off``：端点降为**只读**。read / range / download / head / list 照常，
  所有写动作抛 :class:`storage.base.ConditionalWritesDisabled`。
"""

PROBE_PREFIX = "._ckpt_probe"
"""端点探针写的一次性 key 前缀。远离 ``step_N/``，GC 的扫描不会把它当版本。"""


# 对象不存在时 S3 可能用这几种 Code / 状态码，统一映射到 ObjectNotFoundError。
_NOT_FOUND_CODES = frozenset({"NoSuchKey", "NoSuchBucket", "NotFound", "404"})


@dataclass(frozen=True, slots=True)
class S3Config:
    """一个 bucket + prefix 的定位信息与连接参数。

    endpoint_url 为空表示 AWS 官方端点；SeaweedFS / MinIO 需要显式给出并保持
    ``path_style=True``（它们的 DNS 不支持 virtual-host 风格）。
    """

    bucket: str
    prefix: str = ""
    endpoint_url: str | None = None
    region: str = DEFAULT_REGION
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None
    path_style: bool = True
    # 超过该字节数改用 multipart；分片大小对单次 PUT 也生效（单 PUT 上限 5 GiB）。
    multipart_threshold: int = DEFAULT_MULTIPART_THRESHOLD
    part_size: int = DEFAULT_PART_SIZE
    # 上传/回读附带的对象级校验和算法；None 表示这条连接不带校验头。
    # 老版本 SeaweedFS 之类不认识 x-amz-checksum-* 的端点需要显式设成 None。
    checksum_algorithm: str | None = DEFAULT_CHECKSUM_ALGORITHM
    # 条件写模式：require 要 CAS（写路径硬前置）；off 把端点降成只读。
    conditional_writes: str = "require"

    def __post_init__(self) -> None:
        if not self.bucket:
            raise ValueError("bucket is required")
        if self.conditional_writes not in CONDITIONAL_WRITE_MODES:
            raise ValueError(
                f"unsupported conditional_writes {self.conditional_writes!r}, "
                f"expected one of {sorted(CONDITIONAL_WRITE_MODES)}"
            )
        if self.multipart_threshold <= 0:
            raise ValueError(
                f"multipart_threshold must be > 0, got {self.multipart_threshold}"
            )
        if self.part_size < MIN_PART_SIZE:
            raise ValueError(
                f"part_size must be >= {MIN_PART_SIZE} bytes (5 MiB, S3 limit), "
                f"got {self.part_size}"
            )
        if self.part_size > MAX_PART_SIZE:
            raise ValueError(
                f"part_size must be <= {MAX_PART_SIZE} bytes (5 GiB, S3 limit), "
                f"got {self.part_size}"
            )
        if (
            self.checksum_algorithm is not None
            and self.checksum_algorithm not in SUPPORTED_CHECKSUM_ALGORITHMS
        ):
            raise ValueError(
                f"unsupported checksum_algorithm {self.checksum_algorithm!r}, "
                f"expected one of {sorted(SUPPORTED_CHECKSUM_ALGORITHMS)} or None"
            )
        object.__setattr__(self, "prefix", self.prefix.strip("/"))

    @classmethod
    def from_url(cls, url: str, **overrides: object) -> "S3Config":
        """解析 ``s3://bucket/prefix``。"""
        parsed = urlparse(url)
        if parsed.scheme != "s3":
            raise ValueError(f"expected s3:// URL, got {url!r}")
        if not parsed.netloc:
            raise ValueError(f"s3:// URL is missing a bucket: {url!r}")
        return cls(bucket=parsed.netloc, prefix=parsed.path, **overrides)  # type: ignore[arg-type]

    @property
    def url(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}" if self.prefix else f"s3://{self.bucket}"


def _make_client(config: S3Config) -> "S3Client":
    """按配置创建 boto3 S3 client。"""
    checksum_on = config.checksum_algorithm is not None
    kwargs: dict[str, object] = {
        "service_name": "s3",
        "region_name": config.region,
        "config": Config(
            s3={"addressing_style": "path" if config.path_style else "virtual"},
            retries={"max_attempts": 3, "mode": "standard"},
            # 校验头由 checksum_algorithm 控制：开启时让 SDK 自动算并校验，
            # 关闭时保持 when_required（老版 SeaweedFS 等不认识 x-amz-checksum-*）。
            request_checksum_calculation="when_supported" if checksum_on else "when_required",
            response_checksum_validation="when_supported" if checksum_on else "when_required",
        ),
    }
    if config.endpoint_url:
        kwargs["endpoint_url"] = config.endpoint_url
    if config.access_key_id:
        if not config.secret_access_key:
            raise ValueError("secret_access_key is required when access_key_id is set")
        kwargs["aws_access_key_id"] = config.access_key_id
        kwargs["aws_secret_access_key"] = config.secret_access_key
        if config.session_token:
            kwargs["aws_session_token"] = config.session_token
    return boto3.client(**kwargs)  # type: ignore[return-value]


def _translate(exc: ClientError, key: str) -> StorageError:
    """把 boto3 的 ClientError 收敛成 StorageError 子类。"""
    error = exc.response.get("Error", {})
    code = str(error.get("Code", ""))
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if code in _NOT_FOUND_CODES or status == 404:
        return ObjectNotFoundError(f"object not found: {key}")
    detail = error.get("Message") or str(exc)
    return StorageError(f"{key}: {code or status}: {detail}")


def _normalize_key(key: str) -> str:
    """校验相对 key：非空、不以 '/' 开头、不含 '.' / '..' 段。"""
    if not key:
        raise ValueError("key must not be empty")
    parts = key.split("/")
    if key.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"invalid object key: {key!r}")
    return key


def _join(prefix: str, key: str) -> str:
    return f"{prefix}/{key}" if prefix else key


def _condition_failure(exc: ClientError) -> str | None:
    """条件是**没满足**还是别的问题。

    返回 ``"precondition"``（412）、``"notfound"``（404，``If-Match`` 且对象不存在）
    或 ``None``（真正的错误，交给 :func:`_translate`）。
    """
    error = exc.response.get("Error", {})
    code = str(error.get("Code", ""))
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if status == 412 or code == "PreconditionFailed":
        return "precondition"
    if status == 404 or code in _NOT_FOUND_CODES:
        return "notfound"
    return None


def _is_conditional_conflict(exc: ClientError) -> bool:
    """409 ``ConditionalRequestConflict``：条件写与并发删除撞上。

    普通 PUT 可以直接重试；**multipart 必须重新 ``CreateMultipartUpload``** ——
    服务端已经把这次上传判死（``docs/generation.md`` 的失败模式矩阵）。
    """
    error = exc.response.get("Error", {})
    code = str(error.get("Code", ""))
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return status == 409 or code == "ConditionalRequestConflict"


def _payload_chunks(payload: bytes, part_size: int) -> Iterator[bytes]:
    for start in range(0, len(payload), part_size):
        yield payload[start : start + part_size]


def _file_chunks(path: Path, part_size: int) -> Iterator[bytes]:
    """按 ``part_size`` 流式读文件：峰值内存是一个分片，不是整个 checkpoint。"""
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(part_size)
            if not chunk:
                return
            yield chunk



@dataclass(frozen=True, slots=True)
class ConditionalProbe:
    """端点能力探针的结果。

    ``checks`` 是 ``(步骤名, 实际返回)`` 的流水，失败时直接打进异常信息 ——
    「端点不支持」和「端点响应异常」是两类完全不同的运维问题，不能只说一句
    "unsupported"。
    """

    supported: bool
    endpoint: str
    checks: tuple[tuple[str, str], ...] = ()
    detail: str = ""


class S3Storage(StorageBackend):
    """S3 兼容对象存储上的 checkpoint 根位置。

    ``key`` 参数一律是相对 prefix 的路径；对外接口与 fs 后端保持一致。
    """

    def __init__(self, config: S3Config, *, client: "S3Client | None" = None) -> None:
        self._config = config
        self._client = client if client is not None else _make_client(config)
        self._probe: ConditionalProbe | None = None

    @classmethod
    def from_url(cls, url: str, **overrides: object) -> "S3Storage":
        return cls(S3Config.from_url(url, **overrides))

    @property
    def config(self) -> S3Config:
        return self._config

    @property
    def client(self) -> "S3Client":
        return self._client

    @property
    def supports_conditional_writes(self) -> bool:
        """``conditional_writes="off"`` 的端点是只读的：没有任何写路径。"""
        return self._config.conditional_writes != "off"

    def object_key(self, key: str) -> str:
        """相对 key -> bucket 里的完整对象 key。"""
        return _join(self._config.prefix, _normalize_key(key))


    def _checksum_kwargs(self) -> dict[str, str]:
        """上传时附带对象级校验和：服务端独立重算并比对，不匹配回 400 ``BadDigest``。"""
        algorithm = self._config.checksum_algorithm
        return {"ChecksumAlgorithm": algorithm} if algorithm else {}

    def _checksum_mode_kwargs(self) -> dict[str, str]:
        """整对象 GET 时要求服务端回传校验和，由 botocore 回验返回值。

        只用于整对象读取：``Range`` GET 拿到的是对象的一部分，整对象校验和无法
        对局部内容验证，带上反而会被严格实现拒绝，所以 :meth:`read_range` 不传。
        """
        return {"ChecksumMode": "ENABLED"} if self._config.checksum_algorithm else {}

    def _ensure_writable(self) -> None:
        """只读端点在这里统一失败：不存在「无 CAS 的写入」这条降级路径。"""
        if not self.supports_conditional_writes:
            raise ConditionalWritesDisabled(
                f"{self._config.url}: conditional_writes=off 是只读端点，"
                "写路径要求 S3 conditional writes（If-None-Match / If-Match）"
            )

    def _condition_kwargs(
        self, *, if_none_match: bool = False, if_match: str | None = None
    ) -> dict[str, str]:
        """把条件翻译成 boto3 的参数名。两个条件是互斥的（RFC 7232）。"""
        self._ensure_writable()
        if if_none_match and if_match is not None:
            raise ValueError("If-None-Match 与 If-Match 不能同时给")
        if if_none_match:
            return {"IfNoneMatch": "*"}
        if if_match is not None:
            if not if_match:
                raise ValueError("if_match must be a non-empty ETag")
            return {"IfMatch": if_match}
        return {}

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    # --- 写 -----------------------------------------------------------------
    def write(self, key: str, data: bytes) -> None:
        self._ensure_writable()
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(f"data must be bytes-like, got {type(data).__name__}")
        object_key = self.object_key(key)
        payload = bytes(data)
        try:
            if len(payload) > self._config.multipart_threshold:
                self._write_multipart(
                    object_key,
                    lambda: _payload_chunks(payload, self._config.part_size),
                )
                return
            self._client.put_object(
                Bucket=self._config.bucket,
                Key=object_key,
                Body=payload,
                **self._checksum_kwargs(),
            )
        except ClientError as exc:
            raise _translate(exc, object_key) from exc
        except BotoCoreError as exc:
            raise StorageError(f"{object_key}: {exc}") from exc

    # --- 条件写（CAS）-------------------------------------------------------
    def put_if_absent(self, key: str, data: bytes) -> WriteOutcome:
        """``If-None-Match: *``：只有 key 不存在时才写入。"""
        return self._conditional_write(key, data, if_none_match=True)

    def put_if_match(self, key: str, data: bytes, etag: str) -> WriteOutcome:
        """``If-Match: <etag>``：只有当前 ETag 仍是这个值时才替换。"""
        return self._conditional_write(key, data, if_match=etag)

    def head(self, key: str) -> ObjectMeta | None:
        """取 ETag 与大小；对象不存在返回 ``None``（不是抛错）。"""
        object_key = self.object_key(key)
        try:
            response = self._client.head_object(
                Bucket=self._config.bucket, Key=object_key
            )
        except ClientError as exc:
            if _condition_failure(exc) == "notfound":
                return None
            raise _translate(exc, object_key) from exc
        except BotoCoreError as exc:
            raise StorageError(f"{object_key}: {exc}") from exc
        return ObjectMeta(
            key=key,
            etag=str(response.get("ETag", "")),
            size=int(response.get("ContentLength", 0)),
        )

    def write_file(
        self,
        key: str,
        path: "str | Path",
        *,
        if_none_match: bool = False,
        if_match: str | None = None,
    ) -> WriteOutcome:
        """从本地文件写入；超过阈值走流式分片，条件挂在 complete 上。

        「分片上传」与「条件可见化」是同一条路径：``UploadPart`` 不接受条件头，
        所以整个对象在 ``CompleteMultipartUpload`` 落地那一刻才参与 CAS，这也正是
        「同一 key 只有一个写者能把它变成可见对象」的判定点。
        """
        object_key = self.object_key(key)
        target = Path(path)
        condition = self._condition_kwargs(if_none_match=if_none_match, if_match=if_match)
        size = target.stat().st_size
        try:
            if size > self._config.multipart_threshold:
                self._write_multipart(
                    object_key,
                    lambda: _file_chunks(target, self._config.part_size),
                    condition=condition,
                )
            else:
                self._client.put_object(
                    Bucket=self._config.bucket,
                    Key=object_key,
                    Body=target.read_bytes(),
                    **self._checksum_kwargs(),
                    **condition,
                )
        except ClientError as exc:
            outcome = self._write_failure(exc, if_none_match=if_none_match)
            if outcome is not None:
                return outcome
            raise _translate(exc, object_key) from exc
        except BotoCoreError as exc:
            raise StorageError(f"{object_key}: {exc}") from exc
        return WriteOutcome.CREATED if if_none_match else WriteOutcome.SWAPPED

    def _conditional_write(
        self,
        key: str,
        data: bytes,
        *,
        if_none_match: bool = False,
        if_match: str | None = None,
    ) -> WriteOutcome:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(f"data must be bytes-like, got {type(data).__name__}")
        object_key = self.object_key(key)
        payload = bytes(data)
        condition = self._condition_kwargs(if_none_match=if_none_match, if_match=if_match)
        try:
            if len(payload) > self._config.multipart_threshold:
                self._write_multipart(
                    object_key,
                    lambda: _payload_chunks(payload, self._config.part_size),
                    condition=condition,
                )
            else:
                self._client.put_object(
                    Bucket=self._config.bucket,
                    Key=object_key,
                    Body=payload,
                    **self._checksum_kwargs(),
                    **condition,
                )
        except ClientError as exc:
            outcome = self._write_failure(exc, if_none_match=if_none_match)
            if outcome is not None:
                return outcome
            raise _translate(exc, object_key) from exc
        except BotoCoreError as exc:
            raise StorageError(f"{object_key}: {exc}") from exc
        return WriteOutcome.CREATED if if_none_match else WriteOutcome.SWAPPED

    @staticmethod
    def _write_failure(exc: ClientError, *, if_none_match: bool) -> WriteOutcome | None:
        """条件没满足 → 结构化的 outcome；其它错误返回 ``None`` 交给 ``_translate``。"""
        kind = _condition_failure(exc)
        if kind == "precondition":
            return WriteOutcome.EXISTS if if_none_match else WriteOutcome.CONFLICT
        if kind == "notfound":
            return WriteOutcome.GONE
        return None

    def _write_multipart(
        self,
        object_key: str,
        chunk_source: Callable[[], Iterable[bytes]],
        *,
        condition: dict[str, str] | None = None,
    ) -> None:
        """create → upload_part × N → complete；任何一步失败都 abort 掉本次上传。

        ``chunk_source`` 每次调用都应产出一份完整的分片序列（内存块或重新打开
        的文件句柄）：409 重开 MPU 时要能从头再读一遍，峰值内存只有一片；
        ``condition`` 只作用在 ``complete`` 上 —— 条件写与分片上传的唯一交点是
        「对象可见」那一刻。

        409 ``ConditionalRequestConflict``（条件写与并发删除撞上）时整条 MPU 已被
        服务端判死，必须重新 ``CreateMultipartUpload``，所以这里重开一次。
        """
        condition = dict(condition or {})
        try:
            self._write_multipart_once(object_key, chunk_source(), condition)
        except ClientError as exc:
            if not _is_conditional_conflict(exc):
                raise
            # 409：这次 MPU 已被服务端判死，必须重新 create —— 分片来源重新构造，
            # 所以它是 Callable 而不是一次性生成器。
            self._write_multipart_once(object_key, chunk_source(), condition)

    def _write_multipart_once(
        self,
        object_key: str,
        chunks: Iterable[bytes],
        condition: dict[str, str],
    ) -> None:
        try:
            upload_id = self._client.create_multipart_upload(
                Bucket=self._config.bucket, Key=object_key, **self._checksum_kwargs()
            )["UploadId"]
        except ClientError as exc:
            raise _translate(exc, object_key) from exc
        except BotoCoreError as exc:
            raise StorageError(f"{object_key}: {exc}") from exc

        try:
            parts: list[dict[str, Any]] = []
            for number, chunk in enumerate(chunks, start=1):
                response = self._client.upload_part(
                    Bucket=self._config.bucket,
                    Key=object_key,
                    UploadId=upload_id,
                    PartNumber=number,
                    Body=chunk,
                    **self._checksum_kwargs(),
                )
                # ETag 必须原样回传给 complete，不能去掉引号
                parts.append({"ETag": response["ETag"], "PartNumber": number})
            self._client.complete_multipart_upload(
                Bucket=self._config.bucket,
                Key=object_key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
                **condition,
            )
        except ClientError:
            self._abort_multipart(object_key, upload_id)
            raise
        except BotoCoreError as exc:
            self._abort_multipart(object_key, upload_id)
            raise StorageError(f"{object_key}: {exc}") from exc
        except BaseException:
            # 例如响应缺 ETag 之类的编程错误：也别把半截上传留在服务端
            self._abort_multipart(object_key, upload_id)
            raise

    def _abort_multipart(self, object_key: str, upload_id: str) -> None:
        """尽力清理未完成的上传；清理失败不能掩盖原始错误。"""
        try:
            self._client.abort_multipart_upload(
                Bucket=self._config.bucket, Key=object_key, UploadId=upload_id
            )
        except (ClientError, BotoCoreError):
            pass

    # --- 端点能力探针 -------------------------------------------------------
    def probe_conditional_writes(self) -> ConditionalProbe:
        """在一次性 key 上验证「条件真的被服务端执行了」。

        四个检查各自对应一条协议前提，任何一条不成立都说明 fence 会静默失效：

        1. 首写 ``If-None-Match: *`` → ``CREATED``
        2. 重写同一 key → ``EXISTS``，且**读回的内容没变**（防「接受条件但仍覆盖」）
        3. ``If-Match`` 用过期的 ETag → ``CONFLICT``
        4. ``If-Match`` 用当前 ETag → ``SWAPPED``，且内容确实换了

        不抛异常：结果放在 :class:`ConditionalProbe` 里，由
        :meth:`ensure_conditional_writes` 决定是否升级成错误 —— 探针本身也要能被
        测试直接断言。
        """
        endpoint = self._config.endpoint_url or self._config.url
        if not self.supports_conditional_writes:
            return ConditionalProbe(
                supported=False,
                endpoint=endpoint,
                detail="conditional_writes=off：只读端点，不运行探针",
            )

        key = f"{PROBE_PREFIX}/{uuid.uuid4().hex}"
        checks: list[tuple[str, str]] = []

        def verdict(ok: bool, detail: str = "") -> ConditionalProbe:
            return ConditionalProbe(
                supported=ok, endpoint=endpoint, checks=tuple(checks), detail=detail
            )

        try:
            first = self.put_if_absent(key, b"probe-v1")
            checks.append(("put_if_absent/first", first.value))
            if first is not WriteOutcome.CREATED:
                return verdict(False, f"首次 If-None-Match 写入返回 {first.value}，期望 created")

            second = self.put_if_absent(key, b"probe-v2")
            checks.append(("put_if_absent/second", second.value))
            if second is not WriteOutcome.EXISTS:
                return verdict(
                    False,
                    f"同一 key 的第二次 If-None-Match 写入返回 {second.value}，期望 exists："
                    "端点没有执行 If-None-Match（或 If-None-Match: * 未被支持）",
                )
            if self.read(key) != b"probe-v1":
                return verdict(False, "If-None-Match 返回 exists，但对象内容被覆盖了")

            meta = self.head(key)
            checks.append(("head", "missing" if meta is None else f"etag={meta.etag}"))
            if meta is None or not meta.etag:
                return verdict(False, "HEAD 拿不到 ETag：无法用 If-Match 做 CAS")

            stale_etag = '"00000000000000000000000000000000"'
            stale = self.put_if_match(key, b"probe-v3", stale_etag)
            checks.append(("put_if_match/stale", stale.value))
            if stale is not WriteOutcome.CONFLICT:
                return verdict(
                    False,
                    f"过期 ETag 的 If-Match 写入返回 {stale.value}，期望 conflict："
                    "端点没有执行 If-Match，CAS 会静默失效",
                )
            if self.read(key) != b"probe-v1":
                return verdict(False, "If-Match 失败却改动了对象内容")

            fresh = self.put_if_match(key, b"probe-v3", meta.etag)
            checks.append(("put_if_match/current", fresh.value))
            if fresh is not WriteOutcome.SWAPPED:
                return verdict(False, f"当前 ETag 的 If-Match 写入返回 {fresh.value}，期望 swapped")
            if self.read(key) != b"probe-v3":
                return verdict(False, "If-Match 返回 swapped，但内容没有换")
            return verdict(True)
        finally:
            self._drop_probe_object(key)

    def ensure_conditional_writes(self) -> ConditionalProbe:
        """写路径的前置检查：探针不过就拒绝写入（不降级）。"""
        if not self.supports_conditional_writes:
            raise ConditionalWritesDisabled(
                f"{self._config.url}: conditional_writes=off 是只读端点，"
                "写路径要求 S3 conditional writes（If-None-Match / If-Match）"
            )
        if self._probe is None:
            self._probe = self.probe_conditional_writes()
        if not self._probe.supported:
            detail = self._probe.detail or "条件没有被端点执行"
            checks = ", ".join(f"{name}={value}" for name, value in self._probe.checks)
            raise UnsupportedConditionalWrites(
                f"{self._probe.endpoint}: 不支持条件写（{detail}）；探针流水：{checks}"
            )
        return self._probe

    def _drop_probe_object(self, key: str) -> None:
        """删掉探针对象；清理失败不影响结论（它在 ``._ckpt_probe/`` 下，不参与恢复）。"""
        try:
            self._client.delete_object(
                Bucket=self._config.bucket, Key=self.object_key(key)
            )
        except (ClientError, BotoCoreError):
            pass

    def rename_or_put(self, src: str, dst: str) -> None:
        """对象存储没有 rename：超过 5 GiB 走 UploadPartCopy，否则 copy_object。"""
        self._ensure_writable()
        if src == dst:
            return
        src_key = self.object_key(src)
        dst_key = self.object_key(dst)

        size = self._object_size(src_key)
        if size > MAX_SINGLE_COPY_BYTES:
            self._copy_multipart(src_key, dst_key, size)
        else:
            try:
                self._client.copy_object(
                    Bucket=self._config.bucket,
                    Key=dst_key,
                    CopySource={"Bucket": self._config.bucket, "Key": src_key},
                )
            except ClientError as exc:
                raise _translate(exc, src_key) from exc
            except BotoCoreError as exc:
                raise StorageError(f"{src_key}: {exc}") from exc

        # copy 成功后再删源：中途失败时源对象仍然完整，调用方可以直接重试
        try:
            self._client.delete_object(Bucket=self._config.bucket, Key=src_key)
        except ClientError as exc:
            raise _translate(exc, src_key) from exc
        except BotoCoreError as exc:
            raise StorageError(f"{src_key}: {exc}") from exc

    def _object_size(self, full_key: str) -> int:
        """HEAD 拿对象大小，用来决定 copy 走单次还是分片。"""
        try:
            response = self._client.head_object(
                Bucket=self._config.bucket, Key=full_key
            )
            return int(response["ContentLength"])
        except ClientError as exc:
            raise _translate(exc, full_key) from exc
        except BotoCoreError as exc:
            raise StorageError(f"{full_key}: {exc}") from exc

    def _copy_part_size(self, size: int) -> int:
        """分片 copy 的 part 大小：不小于配置值，且总片数不超过 10000。"""
        needed = -(-size // MAX_MULTIPART_PARTS)  # ceil(size / 10000)
        part_size = max(self._config.part_size, needed)
        return min(max(part_size, MIN_PART_SIZE), MAX_PART_SIZE)

    def _copy_multipart(self, src_key: str, dst_key: str, size: int) -> None:
        """create → upload_part_copy × N → complete；失败 abort，源对象不动。"""
        part_size = self._copy_part_size(size)
        try:
            upload_id = self._client.create_multipart_upload(
                Bucket=self._config.bucket, Key=dst_key
            )["UploadId"]
        except ClientError as exc:
            raise _translate(exc, dst_key) from exc
        except BotoCoreError as exc:
            raise StorageError(f"{dst_key}: {exc}") from exc

        source = {"Bucket": self._config.bucket, "Key": src_key}
        try:
            parts: list[dict[str, Any]] = []
            for number, start in enumerate(range(0, size, part_size), start=1):
                end = min(start + part_size, size) - 1
                response = self._client.upload_part_copy(
                    Bucket=self._config.bucket,
                    Key=dst_key,
                    UploadId=upload_id,
                    PartNumber=number,
                    CopySource=source,
                    CopySourceRange=f"bytes={start}-{end}",
                )
                parts.append(
                    {
                        "ETag": response["CopyPartResult"]["ETag"],
                        "PartNumber": number,
                    }
                )
            self._client.complete_multipart_upload(
                Bucket=self._config.bucket,
                Key=dst_key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
        except ClientError as exc:
            self._abort_multipart(dst_key, upload_id)
            raise _translate(exc, dst_key) from exc
        except BotoCoreError as exc:
            self._abort_multipart(dst_key, upload_id)
            raise StorageError(f"{dst_key}: {exc}") from exc
        except BaseException:
            self._abort_multipart(dst_key, upload_id)
            raise

    def delete(self, key: str) -> None:
        self._ensure_writable()
        object_key = self.object_key(key)
        try:
            self._client.delete_object(Bucket=self._config.bucket, Key=object_key)
        except ClientError as exc:
            # S3 的 DELETE 本身是幂等的，只有权限 / 网络问题才会抛
            raise _translate(exc, object_key) from exc
        except BotoCoreError as exc:
            raise StorageError(f"{object_key}: {exc}") from exc

    # --- 读 -----------------------------------------------------------------
    def read(self, key: str) -> bytes:
        object_key = self.object_key(key)
        try:
            response = self._client.get_object(
                Bucket=self._config.bucket, Key=object_key, **self._checksum_mode_kwargs()
            )
            return response["Body"].read()
        except ClientError as exc:
            raise _translate(exc, object_key) from exc
        except BotoCoreError as exc:
            raise StorageError(f"{object_key}: {exc}") from exc

    def download(self, key: str, path: str | Path) -> int:
        """流式下载整个对象到本地文件，返回字节数。

        整对象 GET 带 ``ChecksumMode=ENABLED``，由 botocore 在读完流时回验；不把
        整个对象读进内存，峰值内存是 ``DEFAULT_DOWNLOAD_CHUNK_BYTES`` 一块。
        """
        object_key = self.object_key(key)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        body = None
        written = 0
        try:
            response = self._client.get_object(
                Bucket=self._config.bucket, Key=object_key, **self._checksum_mode_kwargs()
            )
            body = response["Body"]
            with target.open("wb") as fh:
                while True:
                    chunk = body.read(DEFAULT_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    fh.write(chunk)
                    written += len(chunk)
            return written
        except ClientError as exc:
            target.unlink(missing_ok=True)
            raise _translate(exc, object_key) from exc
        except (BotoCoreError, OSError) as exc:
            target.unlink(missing_ok=True)
            raise StorageError(f"{object_key}: {exc}") from exc
        finally:
            if body is not None:
                body.close()

    def read_range(self, key: str, offset: int, length: int) -> bytes:
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        if length <= 0:
            raise ValueError(f"length must be > 0, got {length}")
        object_key = self.object_key(key)
        try:
            # 不带 ChecksumMode：Range GET 只拿到对象的一部分，整对象校验和对局部
            # 内容无从验证；字节级的把关交给调用方（见 module docstring）。
            response = self._client.get_object(
                Bucket=self._config.bucket,
                Key=object_key,
                Range=f"bytes={offset}-{offset + length - 1}",
            )
        except ClientError as exc:
            raise _translate(exc, object_key) from exc
        except BotoCoreError as exc:
            raise StorageError(f"{object_key}: {exc}") from exc

        payload = response["Body"].read()
        if len(payload) != length:
            raise StorageError(
                f"{object_key}: short range read at offset {offset}, "
                f"got {len(payload)} of {length} bytes"
            )
        return payload

    # --- 枚举 ---------------------------------------------------------------
    def list(self, prefix: str = "") -> list[str]:
        """列出（去掉配置 prefix 之后的）key，按字典序返回。"""
        full_prefix = self._config.prefix
        if prefix:
            full_prefix = _join(self._config.prefix, _normalize_key(prefix))
        # 补 '/' 避免 "step_1" 顺带匹配 "step_10"
        if full_prefix:
            full_prefix += "/"

        keys: list[str] = []
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._config.bucket, Prefix=full_prefix):
                for item in page.get("Contents", []):
                    keys.append(self._relative(item["Key"]))
        except ClientError as exc:
            raise _translate(exc, full_prefix) from exc
        except BotoCoreError as exc:
            raise StorageError(f"{full_prefix}: {exc}") from exc
        return sorted(keys)

    def _relative(self, object_key: str) -> str:
        root = self._config.prefix
        if not root:
            return object_key
        return object_key[len(root) + 1 :] if object_key.startswith(root + "/") else object_key


__all__ = [
    "CONDITIONAL_WRITE_MODES",
    "ConditionalProbe",
    "DEFAULT_CHECKSUM_ALGORITHM",
    "PROBE_PREFIX",
    "SUPPORTED_CHECKSUM_ALGORITHMS",
    "S3Config",
    "S3Storage",
]
