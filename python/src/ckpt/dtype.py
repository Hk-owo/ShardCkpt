"""dtype 名称与 numpy 的映射。

名称与 C++ 侧 DType 枚举、测试数据 manifest 里的写法完全一致。
bf16 没有 numpy 原生类型，用 uint16 承载原始位模式（与 make_data.py 一致）。
"""

from __future__ import annotations

import numpy as np

DTYPE_TO_NUMPY: dict[str, np.dtype] = {
    "f64": np.dtype(np.float64),
    "f32": np.dtype(np.float32),
    "f16": np.dtype(np.float16),
    "bf16": np.dtype(np.uint16),
    "i64": np.dtype(np.int64),
    "i32": np.dtype(np.int32),
    "i16": np.dtype(np.int16),
    "i8": np.dtype(np.int8),
    "u8": np.dtype(np.uint8),
    "bool": np.dtype(np.uint8),
}


def numpy_dtype(name: str) -> np.dtype:
    try:
        return DTYPE_TO_NUMPY[name]
    except KeyError:
        raise ValueError(f"unsupported dtype {name!r}") from None


def itemsize(name: str) -> int:
    return numpy_dtype(name).itemsize
