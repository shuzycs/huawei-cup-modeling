"""自动核对实验总结报告/复现报告中引用的数字是否与真实结果文件一致。

用法::

    python -m tools.audit_report_numbers
    python -m tools.audit_report_numbers --report outputs/problem2/experiment_summary.md --strict

做法：从结果文件重算关键数字，再检查其字符串表示能否在报告文本中原样找到。
正负号写法（`+0.0250` / `−0.0017`）都会尝试匹配。
`--strict` 时只要有任一数字匹配不到就返回非零退出码（可用于 CI）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.problem2.common import load_config, output_dir  # noqa: E402

EXPS = ("E0", "E1", "E2", "E3", "E4", "E5")
SEEDS = (42, 52, 62)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def collect_checks(out: Path, detail: bool = True, include_gates: bool = False) -> List[Tuple[str, float]]:
    """返回 [(说明, 数值)]，全部由结果文件重算。

    detail=False 时只返回"汇总级"数字（三种子聚合、test、教师、bootstrap、规律表、门控），
    用于核对不逐 run 列指标的总结报告。
    """
    checks: List[Tuple[str, float]] = []

    # 逐 run + 三种子聚合
    for exp in EXPS:
        overalls = []
        for seed in SEEDS:
            p = out / "runs" / f"{exp}_seed{seed}" / "eval_valid.json"
            if not p.exists():
                continue
            o = read_json(p)["overall"]
            overalls.append(o)
            if detail:
                checks.append((f"{exp} seed{seed} cleanF1", o["clean_macro_f1"]))
                checks.append((f"{exp} seed{seed} cleanMAE", o["clean_mae"]))
                checks.append((f"{exp} seed{seed} miss36F1", o["primary_macro_f1_missing36"]))
        if not overalls:
            continue
        for key, label in (
            ("clean_macro_f1", "cleanF1"),
            ("primary_macro_f1_missing36", "miss36F1"),
            ("clean_mae", "cleanMAE"),
        ):
            v = np.array([o[key] for o in overalls], dtype=float)
            checks.append((f"{exp} 三种子 {label} mean", float(v.mean())))
            checks.append((f"{exp} 三种子 {label} std", float(v.std(ddof=1)) if v.size > 1 else 0.0))

    # test
    for exp in EXPS:
        p = out / "runs" / f"{exp}_seed42" / "eval_test.json"
        if p.exists():
            c = read_json(p)["clean"]
            checks.append((f"{exp} test macroF1", c["macro_f1"]))
            checks.append((f"{exp} test MAE", c["mae"]))

    # 教师
    for seed in SEEDS:
        p = out / "teacher" / f"mosei_text768_seed{seed}" / "result.json"
        if p.exists():
            b = read_json(p)["best"]
            checks.append((f"teacher seed{seed} F1", b["macro_f1"]))
            checks.append((f"teacher seed{seed} MAE", b["mae"]))

    # E4 蒸馏变体
    for d in sorted((out / "search").glob("E4_*")):
        p = d / "eval_valid.json"
        if p.exists():
            o = read_json(p)["overall"]
            checks.append((f"{d.name} cleanF1", o["clean_macro_f1"]))
            checks.append((f"{d.name} miss36F1", o["primary_macro_f1_missing36"]))

    # 配对 bootstrap
    for r in read_csv_rows(out / "tables" / "paired_bootstrap.csv"):
        checks.append((f"bootstrap {r['comparison']} diff", float(r["diff_mean"])))
        checks.append((f"bootstrap {r['comparison']} ci_low", float(r["ci_low"])))

    # 缺失类型 / 缺失率 / 缺失位置 / 类别强度
    for r in read_csv_rows(out / "tables" / "missing_modality_type.csv"):
        checks.append((f"type {r['experiment']}/{r['masked_modality']}", float(r["macro_f1_mean"])))
    for r in read_csv_rows(out / "tables" / "missing_rate.csv"):
        if r["experiment"] == "E5":
            checks.append((f"rate E5 {r['modality']}/{r['mask_rate']}", float(r["macro_f1_mean"])))
    for r in read_csv_rows(out / "tables" / "missing_position.csv"):
        if r["experiment"] == "E5" and r["mask_rate"] == "0.5":
            checks.append((f"pos E5 {r['modality']}/{r['position']}", float(r["macro_f1_mean"])))
    for r in read_csv_rows(out / "tables" / "class_intensity.csv"):
        if r["experiment"] == "E5" and r["seed"] == "42" and r["f1"]:
            checks.append((f"class {r['class_name']} f1", float(r["f1"])))

    # 门控权重（只有开启 --include-gates 才核对；E0 只有文本、E1/E2 无门控）
    if include_gates:
        for r in read_csv_rows(out / "tables" / "modality_weights.csv"):
            if r["experiment"] in ("E3", "E4", "E5") and r["view_id"] in (
                "valid|clean|k0",
                "valid|T|middle|70|k0",
            ):
                for m in ("T", "A", "V"):
                    key = f"alpha_{m}_mean"
                    if r.get(key):
                        checks.append((f"gate {r['experiment']}/{r['view_id']} a{m}", float(r[key])))

    return checks


def matches(text: str, value: float, nd: int = 4) -> bool:
    plus = f"{value:+.{nd}f}"
    plain = f"{value:.{nd}f}"
    return plus in text or plain in text or plain.replace("-", "−") in text or plus.replace("-", "−") in text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="核对报告数字与结果文件是否一致")
    parser.add_argument("--config", default="configs/problem2.yaml")
    parser.add_argument(
        "--report",
        action="append",
        default=None,
        help="要核对的报告（可多次指定；默认核对 outputs/problem2 下两份主要报告）",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="连逐 run 的明细数字一起核对（默认只核对汇总级数字：三种子聚合、test、教师、"
        "bootstrap、规律表）。逐 run 明细只在复现报告里逐行列出。",
    )
    parser.add_argument(
        "--include-gates",
        action="store_true",
        help="把门控权重也纳入核对（只有写明门控数字的报告才应开启）",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.9,
        help="--strict 下要求的最低匹配比例（默认 0.9；不同报告覆盖的指标集合本就不同）",
    )
    parser.add_argument("--strict", action="store_true", help="匹配比例低于阈值时返回非零退出码")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out = output_dir(cfg)

    # 默认目标：汇总报告逐项覆盖全部明细；复现报告只列汇总级数字（逐 run 明细见汇总报告）
    if args.report:
        targets = [
            (Path(p) if Path(p).is_absolute() else REPO_ROOT / p, args.detail, args.include_gates)
            for p in args.report
        ]
    elif args.detail or args.include_gates:
        targets = [(out / "experiment_summary.md", True, True)]
    else:
        targets = [
            (out / "experiment_summary.md", False, False),
            (out / "reproduction_report.md", False, False),
        ]

    ok_all = True
    for rp, detail, gates in targets:
        scope = ("明细+汇总" if detail else "汇总级") + ("+门控" if gates else "")
        checks = collect_checks(out, detail=detail, include_gates=gates)
        if not rp.exists():
            print(f"  [跳过] 报告不存在: {rp}")
            continue
        text = rp.read_text(encoding="utf-8")
        missing = [(label, value) for label, value in checks if not matches(text, value)]
        coverage = (len(checks) - len(missing)) / max(1, len(checks))
        ok = coverage >= args.min_coverage
        ok_all = ok_all and ok
        print(
            f"  [{'OK ' if ok else 'FAIL'}] {rp.name}（{scope}，{len(checks)} 个数字）: "
            f"覆盖 {coverage:.1%} ({len(checks) - len(missing)}/{len(checks)})"
        )
        if missing:
            print("        未匹配（该报告未列出这些指标，或存在数字漂移）：")
            for label, value in missing[:15]:
                print(f"          {label} = {value:+.4f}")
            if len(missing) > 15:
                print(f"          … 其余 {len(missing) - 15} 个省略")

    print("\n结论:", "各报告覆盖比例均达标，未发现数字漂移" if ok_all else "存在覆盖不足的报告，请检查")
    return 0 if (ok_all or not args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
