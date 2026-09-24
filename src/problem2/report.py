"""环境清单生成：写入 outputs/problem2/env/ 。"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

from .common import (
    REPO_ROOT,
    ensure_parent,
    environment_report,
    load_config,
    output_dir,
    resolve,
    save_json,
    sha256_file,
)


def pip_freeze() -> str:
    try:
        return subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=600
        ).stdout
    except Exception as exc:  # pragma: no cover
        return f"pip freeze failed: {exc!r}"


def freeze_with_header(freeze_text: str, report: dict) -> str:
    """给冻结清单加上可追溯的说明头（脚本可重复生成，不依赖人工维护）。

    特别提醒 torch 那一行：如果安装时用的是本地 wheel 文件，pip freeze 会记录
    ``torch @ file:///...whl``，复现时不能照抄，必须改用官方 index-url。
    """
    lines = [
        "# 本机实验环境的完整冻结，由 src/problem2/report.py 自动生成。",
        f"# 生成命令: python -m src.problem2.report --config configs/problem2.yaml",
        f"# Python: {report.get('python')}",
        f"# torch: {report.get('torch')} (CUDA build {report.get('torch_cuda_build')}), "
        f"cuda_available={report.get('cuda_available')}",
        f"# GPU: {report.get('cuda_device_name')}",
        "# 注意: 若 torch 一行形如 'torch @ file:///...whl'，说明当初用本地 wheel 安装，",
        "#       复现时请改用: pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.4.0",
        "# 直接依赖的版本区间见仓库根目录 requirements.txt。",
        "",
    ]
    return "\n".join(lines) + freeze_text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成环境清单")
    parser.add_argument("--config", default="configs/problem2.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    env_dir = output_dir(cfg) / "env"
    env_dir.mkdir(parents=True, exist_ok=True)

    report = environment_report()
    report["platform_detail"] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node": platform.node(),
    }
    try:
        import torch

        report["device_capability_probe"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else None,
            "arch_list": list(torch.cuda.get_arch_list()),
        }
    except Exception as exc:
        report["device_capability_probe"] = {"error": repr(exc)}

    save_json(env_dir / "environment.json", report)

    # pip freeze 失败时不要用空内容覆盖已经记录好的清单
    freeze_text = pip_freeze()
    if len([ln for ln in freeze_text.splitlines() if ln.strip()]) >= 5:
        (env_dir / "pip_freeze.txt").write_text(freeze_with_header(freeze_text, report), encoding="utf-8")
    else:
        print("[report] 警告：pip freeze 输出异常，保留原有的 pip_freeze.txt")

    # 记录本机实际执行的安装命令（手册要求把真实命令写入 install_commands.txt）
    install_cmds = env_dir / "install_commands.txt"
    if not install_cmds.exists():
        install_cmds.write_text(
            "# 由 src/problem2/report.py 生成的环境报告占位文件；"
            "完整的真实安装命令见 docs/problem2_guide.md 与 requirements.txt\n"
            f"# python: {sys.executable}\n",
            encoding="utf-8",
        )

    frozen = {
        "data/aligned_50.pkl": cfg["data"]["aligned_pkl"],
        "data/label.xlsx": cfg["data"]["labels_xlsx"],
    }
    hashes = {}
    for key, rel in frozen.items():
        p = resolve(rel)
        if p.exists():
            hashes[rel] = {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
        else:
            hashes[rel] = {"sha256": None, "error": "missing"}
    save_json(env_dir / "data_hashes.json", hashes)

    for k, v in report.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
