"""ShardCkpt 核心层：纯数据结构与异常，零依赖。

核心层不抛裸异常，所有失败都收敛到 CheckpointError 的子类，
接入其他语言的绑定层时可以直接按异常类型映射。
"""

from __future__ import annotations


class CheckpointError(Exception):
    """所有 ShardCkpt 错误的基类。"""


class FormatError(CheckpointError):
    """manifest 或数据文件格式不符合预期。"""


class MissingTensorError(CheckpointError):
    """请求的 tensor 不在 manifest 里，或对应文件缺失。"""


class ChecksumError(CheckpointError):
    """数据与 manifest 记录的 CRC32 不一致。"""


class UnsupportedDeviceError(CheckpointError):
    """设备没有可用的 backend（例如请求了 cuda 但既无 GPU 也无 mock）。"""


__all__ = [
    "CheckpointError",
    "ChecksumError",
    "FormatError",
    "MissingTensorError",
    "UnsupportedDeviceError",
]
