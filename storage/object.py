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

**校验和是这条链路唯一的字节级把关点。** 本地层只做「临时文件 → fsync →
rename → 目录 fsync」，不存也不校验分片 CRC（理由见 README「为什么本地层不做
字节级校验」），于是「本地读出来 → 网络 → 落盘」这段的正确性由这里的对象级
校验和负责：PUT / multipart 带 ``ChecksumAlgorithm``（默认 ``CRC32``），服务端
独立重算并比对，不匹配回 400 ``BadDigest``；整对象 GET 带 ``ChecksumMode=
ENABLED`` 让 botocore 回验返回值。不认识 ``x-amz-checksum-*`` 的端点把
``S3Config.checksum_algorithm`` 设成 ``None`` 即可退回旧行为。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from storage.base import ObjectNotFoundError, StorageBackend, StorageError

# boto3 的 S3 client；不引入 mypy_boto3_s3 之类的额外类型依赖。
S3Client = Any

DEFAULT_REGION = "us-east-1"

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

本地层不再做字节级校验，于是「从本地读出来 -> 网络 -> 落盘」这条链路唯一的
把关点就是这里：PUT 带 ``x-amz-checksum-<alg>``，服务端独立重算并比对，不匹配
直接 400 ``BadDigest``；GET 带 ``ChecksumMode=ENABLED`` 让 SDK 回验返回值。

选 ``CRC32`` 而不是 ``CRC32C``：对象存储两种都认，而 Python 标准库只有
``zlib.crc32``（IEEE），本仓库其余部分也用它，保持一致；换成 ``CRC32C`` 只需改
这个默认值（botocore 自带两种算法的实现）。
"""

SUPPORTED_CHECKSUM_ALGORITHMS = frozenset({"CRC32", "CRC32C", "SHA1", "SHA256"})
"""S3 认的校验和算法。"""

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

    def __post_init__(self) -> None:
        if not self.bucket:
            raise ValueError("bucket is required")
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


class S3Storage(StorageBackend):
    """S3 兼容对象存储上的 checkpoint 根位置。

    ``key`` 参数一律是相对 prefix 的路径；对外接口与 fs 后端保持一致。
    """

    def __init__(self, config: S3Config, *, client: "S3Client | None" = None) -> None:
        self._config = config
        self._client = client if client is not None else _make_client(config)

    @classmethod
    def from_url(cls, url: str, **overrides: object) -> "S3Storage":
        return cls(S3Config.from_url(url, **overrides))

    @property
    def config(self) -> S3Config:
        return self._config

    @property
    def client(self) -> "S3Client":
        return self._client

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

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    # --- 写 -----------------------------------------------------------------
    def write(self, key: str, data: bytes) -> None:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(f"data must be bytes-like, got {type(data).__name__}")
        object_key = self.object_key(key)
        payload = bytes(data)
        if len(payload) > self._config.multipart_threshold:
            self._write_multipart(object_key, payload)
            return
        try:
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

    def _write_multipart(self, object_key: str, payload: bytes) -> None:
        """create → upload_part × N → complete；任何一步失败都 abort 掉本次上传。"""
        part_size = self._config.part_size
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
            for number, start in enumerate(range(0, len(payload), part_size), start=1):
                chunk = payload[start : start + part_size]
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
            )
        except ClientError as exc:
            self._abort_multipart(object_key, upload_id)
            raise _translate(exc, object_key) from exc
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

    def rename_or_put(self, src: str, dst: str) -> None:
        """对象存储没有 rename：超过 5 GiB 走 UploadPartCopy，否则 copy_object。"""
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
    "DEFAULT_CHECKSUM_ALGORITHM",
    "SUPPORTED_CHECKSUM_ALGORITHMS",
    "S3Config",
    "S3Storage",
]
