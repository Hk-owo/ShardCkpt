"""`examples/e2e_sim.py` 的冒烟测试：端到端剧本不能腐化。

demo 跑的是正式实现（只有 device 与对象存储是模拟的），所以它一旦跑不通，
说明保存 / 提交点 / 上传 / 恢复 / 回收这条链路真的坏了 —— 这条用例就是那个哨兵。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "e2e_sim.py"

#: 剧本里每个阶段至少会打出来的一行，用来确认流程真的走完了（而不是提前短路）。
PHASES = (
    "[1/7]",
    "[2/7]",
    "[3/7]",
    "[4/7]",
    "[5/7]",
    "[6/7]",
    "[7/7]",
)


@pytest.mark.parametrize("ranks", [1, 2])
def test_e2e_sim_script_runs_the_whole_scenario(ranks: int) -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--ranks", str(ranks)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    for phase in PHASES:
        assert phase in output, f"缺少阶段 {phase}：\n{output}"
    for needle in (
        # 崩溃注入真的把进程打死了
        "退出码：-9",
        # 提交点门禁真的拒了不完整版本
        "如期失败",
        # 恢复 + 续训是逐字节比对过的
        "逐字节一致",
        "全部阶段通过",
    ):
        assert needle in output, f"缺少断言痕迹 {needle!r}：\n{output}"


def test_e2e_sim_rejects_bad_rank_count() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--ranks", "0"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0
    assert "--ranks 必须 >= 1" in (proc.stdout + proc.stderr)
