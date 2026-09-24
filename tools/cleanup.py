"""清理工作区：删除缓存与未被引用的中间产物。

保留：代码、配置、文档、全部最终产物（best.safetensors、eval_*.json、报告、表格、图、CSV）。
删除：__pycache__、.pytest_cache、以及仅用于"训练中断兜底"的 last.safetensors
      （代码中不读取该文件；需要时重跑训练即可再生成）。

用法::

    python -m tools.cleanup                 # 试运行，只列出将要删除的内容
    python -m tools.cleanup --yes           # 真正删除
    python -m tools.cleanup --yes --keep-last   # 保留 last.safetensors
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def mb(total: int) -> str:
    return f"{total / 2**20:.1f} MB"


def collect_targets(keep_last: bool) -> Tuple[List[Path], List[Path]]:
    """返回 (目录列表, 文件列表)。"""
    dirs: List[Path] = []
    files: List[Path] = []

    for name in ("__pycache__", ".pytest_cache"):
        for p in REPO_ROOT.rglob(name):
            if ".git" in p.parts:
                continue
            dirs.append(p)

    if not keep_last:
        for pat in ("outputs/problem2/runs/*/last.safetensors", "outputs/problem2/teacher/*/last.safetensors"):
            files.extend(sorted(REPO_ROOT.glob(pat)))

    return dirs, files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="清理缓存与未被引用的中间产物")
    parser.add_argument("--yes", action="store_true", help="真正执行删除（默认仅试运行）")
    parser.add_argument("--keep-last", action="store_true", help="保留 last.safetensors")
    args = parser.parse_args(argv)

    dirs, files = collect_targets(args.keep_last)

    total_dirs = 0
    for d in dirs:
        total_dirs += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    total_files = sum(f.stat().st_size for f in files if f.exists())

    print(f"工作区根目录: {REPO_ROOT}")
    print(f"缓存目录 {len(dirs)} 个，合计 {mb(total_dirs)}:")
    from collections import Counter

    for name, cnt in Counter(d.name for d in dirs).most_common():
        print(f"   - {name} × {cnt}")
    print(f"中间文件 {len(files)} 个，合计 {mb(total_files)}:")
    for f in files[:8]:
        print(f"   - {f.relative_to(REPO_ROOT)}  ({mb(f.stat().st_size)})")
    if len(files) > 8:
        print(f"   - … 其余 {len(files) - 8} 个")

    if not args.yes:
        print("\n试运行结束（未删除任何内容）。确认后加 --yes 执行。")
        return 0

    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
    for f in files:
        f.unlink(missing_ok=True)
    print(f"\n已删除：{len(dirs)} 个目录、{len(files)} 个文件，共释放约 {mb(total_dirs + total_files)}。")
    print("注：last.safetensors 是训练中断兜底权重，代码不读取；需要时重跑训练即可再生成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
