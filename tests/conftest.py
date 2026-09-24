"""pytest 配置：让测试在任意工作目录下都能导入 src.problem2。

同时把"缺少实验产物"的测试降级为跳过，而不是失败 —— 例如还没跑过
`prepare_data` 时，依赖缓存的测试会被 skip，核心单元检查仍会执行。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers", "needs_artifacts: 需要先运行数据准备/训练脚本的测试"
    )
