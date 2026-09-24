"""一键自检：编译、单元测试、报告数字核对、工具冒烟（只读）。

用法::

    python -m tools.check_all

返回码 0 表示全部通过。适合在改动代码/文档后快速确认没有破坏既有结论。
"""

from __future__ import annotations

import argparse
import compileall
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# (显示名, 参数列表)
STEPS = [
    ("编译全部源码", ["-m", "compileall", "-q", "src", "tools", "tests"]),
    ("单元测试", ["-m", "pytest", "tests", "-q"]),
    ("报告数字核对", ["-m", "tools.audit_report_numbers", "--strict"]),
    ("指标表", ["-m", "tools.summary_table"]),
    ("聚合口径核对", ["-m", "tools.verify_aggregation"]),
    ("附件 3 推理自检（只读复核）", ["-m", "tools.check_attachment3"]),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="问题 2 一键自检")
    parser.add_argument("--skip", nargs="*", default=[], help="跳过的步骤名（子串匹配）")
    parser.add_argument("--quiet", action="store_true", help="只打印每步结果，不转发子进程输出")
    args = parser.parse_args(argv)

    print(f"仓库根目录: {REPO_ROOT}")
    print(f"Python: {sys.executable}\n")

    failed: list[str] = []
    for name, cmd in STEPS:
        if any(s in name for s in args.skip):
            print(f"[跳过] {name}")
            continue
        print(f"[执行] {name} ...", flush=True)
        res = subprocess.run(
            [sys.executable, *cmd],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        ok = res.returncode == 0
        print(f"[{'通过' if ok else '失败'}] {name} (exit={res.returncode})")
        if not ok:
            failed.append(name)
            tail = (res.stdout + res.stderr).strip().splitlines()[-15:]
            for line in tail:
                print(f"        {line}")
        elif not args.quiet:
            tail = (res.stdout or "").strip().splitlines()[-3:]
            for line in tail:
                print(f"        {line}")

    print()
    if failed:
        print(f"未通过 {len(failed)} 项: {failed}")
        return 1
    print(f"全部 {len(STEPS) - len(args.skip)} 项自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
