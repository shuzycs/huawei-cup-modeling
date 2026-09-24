"""打印附件 2 test 集与验证集完整视图的指标、混淆矩阵、逐类 F1（只读）。

用法::

    python -m tools.show_metrics
    python -m tools.show_metrics --experiment E5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.problem2 import CLASS_NAMES  # noqa: E402
from src.problem2.common import load_config, output_dir  # noqa: E402

EXPS = ("E0", "E1", "E2", "E3", "E4", "E5")
SEEDS = (42, 52, 62)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def print_confusion(cm: np.ndarray, title: str) -> None:
    print(f"   {title}")
    print("         " + "".join(f"{n:>10s}" for n in CLASS_NAMES) + f"{'行和':>10s}")
    for i, name in enumerate(CLASS_NAMES):
        cells = "".join(f"{int(cm[i, j]):>10d}" for j in range(3))
        print(f"   {name:6s}{cells}{int(cm[i].sum()):>10d}")
    print(
        "   预测和 "
        + "".join(f"{int(cm[:, j].sum()):>10d}" for j in range(3))
        + f"{int(cm.sum()):>10d}"
    )
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    print(f"   行归一化: {np.round(norm, 3).tolist()}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="打印 test / valid 完整视图指标")
    parser.add_argument("--config", default="configs/problem2.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    out = output_dir(load_config(args.config))
    exps = [args.experiment] if args.experiment else list(EXPS)

    print("=== 附件 2 test 独立检验（eval_test.json）===")
    print(f"{'exp':5s}{'macroF1':>10s}{'acc':>9s}{'MAE':>9s}{'Pearson':>10s}{'n':>6s}")
    for exp in exps:
        p = out / "runs" / f"{exp}_seed{args.seed}" / "eval_test.json"
        if not p.exists():
            continue
        c = read_json(p)["clean"]
        pear = f"{c['pearson']:.4f}" if c["pearson"] is not None else "n/a"
        print(
            f"{exp:5s}{c['macro_f1']:>10.4f}{c['accuracy']:>9.4f}{c['mae']:>9.4f}{pear:>10s}{c['n']:>6d}"
        )

    print()
    print("=== 验证集完整视图（eval_valid.json 的 clean 条件）===")
    for exp in exps:
        p = out / "runs" / f"{exp}_seed{args.seed}" / "eval_valid.json"
        if not p.exists():
            continue
        s = read_json(p)["condition_summary"]["clean"]
        cm = np.asarray(s.get("confusion_matrix_sum") or s.get("confusion_matrix"), dtype=float)
        ov = read_json(p)["overall"]
        pear = f"{ov['clean_pearson']:.4f}" if ov["clean_pearson"] is not None else "n/a"
        print(
            f"\n--- {exp} seed={args.seed}: macroF1={s['macro_f1']:.4f} acc={s['accuracy']:.4f} "
            f"MAE={s['mae']:.4f} Pearson={pear}"
        )
        print(
            "   逐类 F1: "
            + "  ".join(f"{name}={s['per_class'][name]['f1']:.4f}" for name in CLASS_NAMES)
        )
        print_confusion(cm, "混淆矩阵（行=真实，列=预测）:")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
