"""核对门控权重与缺失位置离散度的聚合口径（只读）。

用法::

    python -m tools.verify_aggregation

用途：报告中的门控权重来自 `tables/modality_weights.csv`（先按 run 求均值、再对 run 求
均值±标准差），缺失位置的离散度来自 `tables/missing_position.csv`。
本脚本把这两处的原始数字重新打印一遍，便于人工对照报告。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.problem2.common import load_config, output_dir  # noqa: E402

VIEWS = ("valid|clean|k0", "valid|T|middle|10|k0", "valid|T|middle|70|k0")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="核对聚合口径")
    parser.add_argument("--config", default="configs/problem2.yaml")
    args = parser.parse_args(argv)

    out = output_dir(load_config(args.config))

    print("=== 门控权重（tables/modality_weights.csv：三种子均值 ± 样本标准差）===")
    rows = read_csv_rows(out / "tables" / "modality_weights.csv")
    for exp in ("E3", "E4", "E5"):
        for view in VIEWS:
            hit = [r for r in rows if r["experiment"] == exp and r["view_id"] == view]
            if not hit:
                print(f"   {exp} {view}: 无数据")
                continue
            r = hit[0]
            parts = []
            for m in ("T", "A", "V"):
                mean, std = r.get(f"alpha_{m}_mean"), r.get(f"alpha_{m}_std")
                if mean:
                    parts.append(f"a{m}={float(mean):.4f}±{float(std):.4f}")
            print(f"   {exp} {view:24s} " + "  ".join(parts) + f"  (n_runs={r['n_runs']})")

    print()
    print("=== 缺失位置离散度（tables/missing_position.csv，E5）===")
    pos = [r for r in read_csv_rows(out / "tables" / "missing_position.csv") if r["experiment"] == "E5"]
    for m in ("T", "A", "V"):
        for rate in ("0.1", "0.3", "0.5", "0.7"):
            v = [float(r["macro_f1_mean"]) for r in pos if r["modality"] == m and r["mask_rate"] == rate]
            if v:
                print(
                    f"   E5 {m} 目标缺失 {rate:>3s}: 三位置均值={np.mean(v):.4f} "
                    f"跨位置std={np.std(v, ddof=1):.4f} 各位置={[round(x, 4) for x in v]}"
                )

    print()
    print("=== 实际有效长度占比（tables/missing_rate.csv，E5）===")
    for r in read_csv_rows(out / "tables" / "missing_rate.csv"):
        if r["experiment"] == "E5":
            print(
                f"   {r['modality']} 目标缺失 {r['mask_rate']:>4s} → 实际有效 "
                f"{float(r['effective_ratio_min']):.4f} ~ {float(r['effective_ratio_max']):.4f} "
                f"(均值 {float(r['effective_ratio_mean']):.4f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
