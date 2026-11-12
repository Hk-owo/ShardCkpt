"""pytest 公共夹具：把仓库内的 Python 包与数据生成器加入 sys.path。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tests", ROOT / "tests" / "fixtures", ROOT / "tests" / "sim"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


@pytest.fixture(autouse=True)
def _fresh_backends():
    """每个用例都从默认 backend 注册表开始，避免相互污染。

    进程级 pinned 池也在这里关掉：它是真的常驻内存，跨用例留着会让「池复用了
    几次」这类断言互相干扰。
    """
    import api as ckpt  # noqa: PLC0415

    ckpt.reset_registry()
    ckpt.close_default_pool()
    yield
    ckpt.reset_registry()
    ckpt.close_default_pool()


@pytest.fixture
def state():
    """一份默认规模的伪训练状态（42 个张量，约 148 KB），仿真与上传用例共用。"""
    import sim  # noqa: PLC0415

    return sim.make_state(step=0)


@pytest.fixture
def dataset(tmp_path):
    """一份 tiny 规模的测试数据（含 optimizer 状态）。

    生成数据本身会走 ``save_local``（进而用到进程级 pinned 池），生成完就把它关掉：
    用例看到的池统计应当是干净的。
    """
    import api as ckpt  # noqa: PLC0415
    import make_data

    out = tmp_path / "ckpt"
    make_data.write_dataset(out, seed=1234, scale="tiny")
    ckpt.close_default_pool()
    return out
