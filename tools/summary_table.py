"""打印主指标表 / 消融表的小型只读工具（比直接看 CSV 更易读）。

用法::

    python -m tools.summary_table                  # 打印主表与逐 run 表
    python -m tools.summary_table --csv /tmp/x.csv # 另存一份汇总 CSV
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.problem2.common import load_config, output_dir  # noqa: E402

EXPS = ("E0", "E1", "E2", "E3", "E4", "E5")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="打印主指标表")
    parser.add_argument("--config", default="configs/problem2.yaml")
    parser.add_argument("--csv", default=None, help="把 metric_summary.csv 另存到该路径")
    args = parser.parse_args(argv)

    out = output_dir(load_config(args.config))
    rows = read_csv_rows(out / "metric_summary.csv")
    if not rows:
        raise SystemExit(f"缺少 {out / 'metric_summary.csv'}，请先运行 python -m src.problem2.analyze")

    print("=== 主表（三种子均值，来自 metric_summary.csv）===")
    header = (
        f"{'exp':5s}{'seeds':>6s}{'cleanF1':>18s}{'cleanAcc':>10s}{'cleanMAE':>10s}"
        f"{'cleanPear':>11s}{'miss36F1':>18s}{'allMissF1':>11s}{'allMissMAE':>12s}"
    )
    print(header)
    for exp in EXPS:
        r = next((x for x in rows if x["experiment"] == exp), None)
        if r is None:
            continue

        def f(key: str) -> str:
            v = r.get(key)
            return f"{float(v):.4f}" if v not in (None, "") else "n/a"

        def ms(mean_key: str, std_key: str) -> str:
            m, s = r.get(mean_key), r.get(std_key)
            if m in (None, ""):
                return "n/a"
            return f"{float(m):.4f}±{float(s or 0):.4f}"

        print(
            f"{exp:5s}{r['n_seeds']:>6s}{ms('clean_macro_f1_mean','clean_macro_f1_std'):>18s}"
            f"{f('clean_accuracy_mean'):>10s}{f('clean_mae_mean'):>10s}{f('clean_pearson_mean'):>11s}"
            f"{ms('missing36_macro_f1_mean','missing36_macro_f1_std'):>18s}"
            f"{f('allmissing_macro_f1_mean'):>11s}{f('allmissing_mae_mean'):>12s}"
        )

    print()
    print("=== 逐 run 明细（ablation.csv 之外的训练记录见 runs/<E>_seed<S>/result.json）===")
    for exp in EXPS:
        r = next((x for x in rows if x["experiment"] == exp), None)
        if r is not None:
            print(f"   {exp}: {r['experiment_label']}  最佳 epoch(按 seed)={r['best_epochs']}")

    if args.csv:
        target = Path(args.csv)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((out / "metric_summary.csv").read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
        print(f"\n已另存: {target}")

    boot = read_csv_rows(out / "tables" / "paired_bootstrap.csv")
    if boot:
        print()
        print("=== 配对 bootstrap（accuracy）===")
        for r in boot:
            print(
                f"   {r['comparison']:12s} diff={float(r['diff_mean']):+.4f} "
                f"CI=[{float(r['ci_low']):+.4f},{float(r['ci_high']):+.4f}] 显著={r['significant_at_0.05']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
