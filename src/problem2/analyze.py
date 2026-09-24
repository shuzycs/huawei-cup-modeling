"""规律分析与图表：`python -m src.problem2.analyze --config configs/problem2.yaml`

读取 outputs/problem2/runs/*/eval_valid.json，汇总主表 / 消融表 / 缺失规律 /
门控行为 / 错误分析，并生成图表。所有数字均来自真实日志文件，可回查。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import CLASS_NAMES, MODALITIES, __version__
from .common import output_dir, resolve, save_json
from .evaluate import (
    DUAL_CONDITIONS,
    MISSING_RATES,
    POSITIONS,
    SINGLE_MODALITY_CONDITIONS,
    single_condition_keys,
)
from .metrics import merge_metric_dicts

EXPERIMENT_LABELS = {
    "E0": "E0 文本单模态",
    "E1": "E1 固定50位平均+拼接MLP（仅完整输入）",
    "E2": "E2 E1+连续区间缺失训练",
    "E3": "E3 E2+掩码池化+动态门控（主模型）",
    "E4": "E4 E3+教师蒸馏（已实施）",
    "E5": "E5 E3+解冻文本编码器后两层（补充实验）",
}
METRIC_KEYS = ("macro_f1", "accuracy", "weighted_f1", "mae", "rmse", "pearson")


def _seed_std(values: Sequence[float]) -> Optional[float]:
    """三种子之间的样本标准差（ddof=1）；单种子返回 0.0，空返回 None。"""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    return float(v.std(ddof=1)) if v.size > 1 else 0.0


# --------------------------------------------------------------------------------------
# 载入
# --------------------------------------------------------------------------------------
def load_runs(cfg: Dict[str, Any], split: str = "valid") -> Dict[str, Dict[str, Any]]:
    root = output_dir(cfg) / "runs"
    out: Dict[str, Dict[str, Any]] = {}
    for run_dir in sorted(root.glob("*_seed*")):
        parts = run_dir.name.rsplit("_seed", 1)
        if len(parts) != 2:
            continue
        exp, seed = parts[0], int(parts[1])
        eval_path = run_dir / f"eval_{split}.json"
        samples_path = run_dir / f"eval_{split}_samples.npz"
        result_path = run_dir / "result.json"
        if not eval_path.exists():
            continue
        with open(eval_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        out.setdefault(exp, {})[str(seed)] = {
            "eval": data,
            "samples": samples_path if samples_path.exists() else None,
            "result": json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None,
            "dir": run_dir,
        }
    return out


def aggregate_over_seeds(per_seed: Dict[str, Dict[str, Any]], field: str) -> Dict[str, Any]:
    """把各 seed 的 metric dict 合并为 mean/std。"""
    vals = [rec["eval"][field] for rec in per_seed.values() if field in rec["eval"]]
    return merge_metric_dicts(vals)


# --------------------------------------------------------------------------------------
# 主表
# --------------------------------------------------------------------------------------
def build_metric_summary(runs: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    detail: Dict[str, Any] = {}
    for exp in sorted(runs):
        per_seed = runs[exp]
        overalls = [rec["eval"]["overall"] for rec in per_seed.values()]
        cond_all: Dict[str, List[Dict[str, Any]]] = {}
        for rec in per_seed.values():
            for cond, m in rec["eval"]["condition_summary"].items():
                cond_all.setdefault(cond, []).append(m)

        clean = merge_metric_dicts([
            rec["eval"]["condition_summary"].get("clean", {}) for rec in per_seed.values()
        ])
        primary = [merge_metric_dicts(cond_all[c]) for c in single_condition_keys() if c in cond_all]
        all_missing = [merge_metric_dicts(cond_all[c]) for c in cond_all if c != "clean"]
        missing_merged = merge_metric_dicts(all_missing)

        row = {
            "experiment": exp,
            "experiment_label": EXPERIMENT_LABELS.get(exp, exp),
            "n_seeds": len(per_seed),
            "seeds": ",".join(sorted(per_seed, key=int)),
            "best_epochs": ",".join(
                str(rec["eval"].get("seed")) + ":" + str(rec["result"]["best"]["epoch"] if rec.get("result") else "?")
                for rec in per_seed.values()
            ),
            "clean_macro_f1_mean": clean.get("macro_f1"),
            "clean_macro_f1_std": clean.get("macro_f1_std"),
            "clean_accuracy_mean": clean.get("accuracy"),
            "clean_accuracy_std": clean.get("accuracy_std"),
            "clean_mae_mean": clean.get("mae"),
            "clean_mae_std": clean.get("mae_std"),
            "clean_pearson_mean": clean.get("pearson"),
        }
        # 36 个单模态缺失条件的平均（主准则）
        primary = [merge_metric_dicts(cond_all[c]) for c in single_condition_keys() if c in cond_all]
        prim = merge_metric_dicts(primary)
        # 种子的真实离散度：先算每个种子的"跨条件平均"，再对种子取均值±样本标准差。
        # （直接用 conditions 的 std 混入合并会得到跨条件离散度，与报告口径不一致。）
        seed_level_miss36 = [
            float(np.mean([
                rec["eval"]["condition_summary"][c]["macro_f1"]
                for c in single_condition_keys()
                if c in rec["eval"]["condition_summary"]
            ]))
            for rec in per_seed.values()
        ]
        row.update(
            {
                "missing36_macro_f1_mean": prim.get("macro_f1"),
                "missing36_macro_f1_std": _seed_std(seed_level_miss36),
                "missing36_macro_f1_condition_std": prim.get("macro_f1_std"),
                "missing36_accuracy_mean": prim.get("accuracy"),
                "missing36_mae_mean": prim.get("mae"),
                "missing36_pearson_mean": prim.get("pearson"),
            }
        )
        row.update(
            {
                "allmissing_macro_f1_mean": missing_merged.get("macro_f1"),
                "allmissing_macro_f1_std": missing_merged.get("macro_f1_std"),
                "allmissing_mae_mean": missing_merged.get("mae"),
                "n_conditions_missing36": len(primary),
            }
        )
        # 每类 F1（完整输入）
        for cname in CLASS_NAMES:
            row[f"clean_f1_{cname}"] = (clean.get("per_class", {}).get(cname, {}) or {}).get("f1_mean")
        rows.append(row)
        detail[exp] = {
            "condition_summary": {c: merge_metric_dicts(v) for c, v in cond_all.items()},
            "clean": clean,
            "missing36": prim,
            "per_seed_overall": overalls,
        }
    return rows, detail


# --------------------------------------------------------------------------------------
# 缺失规律
# --------------------------------------------------------------------------------------
def missing_rate_table(
    runs: Dict[str, Dict[str, Any]], detail: Dict[str, Any], plan: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """按 (模态, 缺失率) 汇总；横轴用 validation_masks.json 中的实际有效长度占比。"""
    actual_rates = condition_actual_rates(plan)
    rows: List[Dict[str, Any]] = []
    for exp in sorted(runs):
        for m in MODALITIES:
            for rate in MISSING_RATES:
                keys = [f"{m}-{p}|{int(round(rate * 100))}" for p in POSITIONS]
                metrics = [detail[exp]["condition_summary"][k] for k in keys if k in detail[exp]["condition_summary"]]
                if not metrics:
                    continue
                rows.append(
                    {
                        "experiment": exp,
                        "modality": m,
                        "mask_rate": rate,
                        "effective_ratio_mean": actual_rates.get(f"{m}-middle|{int(round(rate * 100))}"),
                        "effective_ratio_min": min(
                            [v for k, v in actual_rates.items() if k.startswith(f"{m}-") and k.endswith(f"|{int(round(rate * 100))}")],
                            default=None,
                        ),
                        "effective_ratio_max": max(
                            [v for k, v in actual_rates.items() if k.startswith(f"{m}-") and k.endswith(f"|{int(round(rate * 100))}")],
                            default=None,
                        ),
                        "macro_f1_mean": float(np.mean([x["macro_f1"] for x in metrics])),
                        "macro_f1_std": float(np.std([x["macro_f1"] for x in metrics], ddof=1)) if len(metrics) > 1 else 0.0,
                        "macro_f1_seed_std": float(np.mean([x.get("macro_f1_std") or 0.0 for x in metrics])),
                        "accuracy_mean": float(np.mean([x["accuracy"] for x in metrics])),
                        "mae_mean": float(np.mean([x["mae"] for x in metrics])),
                        "n_positions": len(metrics),
                    }
                )
    return rows


def condition_actual_rates(plan: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """从 validation_masks.json 读取每个条件的平均实际缺失比例。"""
    out: Dict[str, float] = {}
    if not plan:
        return out
    for view in plan.get("views", []):
        if view.get("kind") == "clean":
            continue
        mm = view.get("masked_modalities") or []
        if len(mm) != 1:
            continue
        m = mm[0]
        rate = float(view.get("rate", 0.0))
        real = float(view.get("mean_actual_rate", {}).get(m, 0.0) or 0.0)
        key = f"{m}-{view.get('position')}|{int(round(rate * 100))}"
        out[key] = real
    return out


def missing_position_table(runs: Dict[str, Dict[str, Any]], detail: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for exp in sorted(runs):
        for m in MODALITIES:
            for pos in POSITIONS:
                for rate in MISSING_RATES:
                    c = f"{m}-{pos}|{int(round(rate * 100))}"
                    if c not in detail[exp]["condition_summary"]:
                        continue
                    cond = detail[exp]["condition_summary"][c]
                    rows.append(
                        {
                            "experiment": exp,
                            "modality": m,
                            "position": pos,
                            "mask_rate": rate,
                            "macro_f1_mean": cond.get("macro_f1"),
                            "macro_f1_std": cond.get("macro_f1_std"),
                            "mae_mean": cond.get("mae"),
                        }
                    )
    return rows


def modality_type_table(runs: Dict[str, Dict[str, Any]], detail: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for exp in sorted(runs):
        for m in MODALITIES:
            keys = [
                f"{m}-{p}|{int(round(r * 100))}"
                for p in POSITIONS
                for r in MISSING_RATES
            ]
            metrics = [detail[exp]["condition_summary"][k] for k in keys if k in detail[exp]["condition_summary"]]
            rows.append(
                {
                    "experiment": exp,
                    "masked_modality": m,
                    "n_conditions": len(metrics),
                    "macro_f1_mean": float(np.mean([x["macro_f1"] for x in metrics])) if metrics else None,
                    "macro_f1_seed_std": float(np.mean([x.get("macro_f1_std") or 0.0 for x in metrics])) if metrics else None,
                    "accuracy_mean": float(np.mean([x["accuracy"] for x in metrics])) if metrics else None,
                    "mae_mean": float(np.mean([x["mae"] for x in metrics])) if metrics else None,
                    "pearson_mean": float(np.mean([x["pearson"] for x in metrics if x.get("pearson") is not None])) if metrics else None,
                }
            )
        # 双模态缺失
        for pair in DUAL_CONDITIONS:
            keys = [f"{pair}|{int(round(r * 100))}" for r in (0.30, 0.50)]
            metrics = [detail[exp]["condition_summary"][k] for k in keys if k in detail[exp]["condition_summary"]]
            if not metrics:
                continue
            rows.append(
                {
                    "experiment": exp,
                    "masked_modality": pair,
                    "n_conditions": len(metrics),
                    "macro_f1_mean": float(np.mean([x["macro_f1"] for x in metrics])),
                    "macro_f1_seed_std": float(np.mean([x.get("macro_f1_std") or 0.0 for x in metrics])),
                    "accuracy_mean": float(np.mean([x["accuracy"] for x in metrics])),
                    "mae_mean": float(np.mean([x["mae"] for x in metrics])),
                    "pearson_mean": float(np.mean([x["pearson"] for x in metrics if x.get("pearson") is not None])),
                }
            )
    return rows


# --------------------------------------------------------------------------------------
# 门控行为 / 类别与强度 / 错误分析
# --------------------------------------------------------------------------------------
def load_sample_predictions(rec: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    path = rec.get("samples")
    if path is None or not Path(path).exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


MODALITIES_BY_EXPERIMENT = {
    "E0": ["T"],
    "E1": ["T", "A", "V"],
    "E2": ["T", "A", "V"],
    "E3": ["T", "A", "V"],
    "E4": ["T", "A", "V"],
    "E5": ["T", "A", "V"],
}


def modality_weight_table(runs: Dict[str, Dict[str, Any]], valid_ids: np.ndarray) -> List[Dict[str, Any]]:
    """按视图汇总各模态的平均融合权重（模态名以融合权重向量宽度为准，E0 只有文本）。

    聚合口径：先把每个 run 内该视图的所有样本取均值（每个 run 一个等权观测），
    再对 run 求均值与样本标准差，避免样本数不同的视图之间不可比。
    """
    per_run: Dict[Tuple[str, str], List[Tuple[List[str], np.ndarray, int]]] = {}
    for exp, seeds in runs.items():
        for seed, rec in seeds.items():
            sp = load_sample_predictions(rec)
            if sp is None or "fusion_weights" not in sp:
                continue
            w = np.atleast_2d(sp["fusion_weights"])
            views = [str(x) for x in sp["view_id"]]
            mods = [str(x) for x in sp["modalities"]] if "modalities" in sp else list(MODALITIES[: w.shape[1]])
            mods = mods[: w.shape[1]]
            for vid in sorted(set(views)):
                sel = np.asarray(views) == vid
                if not sel.any():
                    continue
                per_run.setdefault((exp, vid), []).append((mods, w[sel].mean(axis=0), int(sel.sum())))

    rows: List[Dict[str, Any]] = []
    for (exp, vid), entries in sorted(per_run.items()):
        mods = entries[0][0]
        width = len(mods)
        stack = np.stack([e[1][:width] for e in entries], axis=0)  # (n_runs, width)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(mean)
        row: Dict[str, Any] = {
            "experiment": exp,
            "view_id": vid,
            "modalities": ",".join(mods),
            "masked_modalities": view_masked_modalities(exp, vid),
            "n_runs": int(stack.shape[0]),
            "n_samples_per_run": int(entries[0][2]),
        }
        for i, m in enumerate(mods):
            row[f"alpha_{m}_mean"] = float(mean[i])
            row[f"alpha_{m}_std"] = float(std[i])
        rows.append(row)
    return rows


def view_masked_modalities(exp: str, view_id: str) -> str:
    if view_id.startswith("valid|clean"):
        return "clean"
    parts = view_id.split("|")
    if len(parts) < 3:
        return "unknown"
    tag = parts[1]
    if tag.startswith("dual-"):
        return tag
    return tag


def error_cases(runs: Dict[str, Dict[str, Any]], raw_text: np.ndarray, ids: np.ndarray, top_k: int = 60) -> List[Dict[str, Any]]:
    """选取主模型（优先 E3）验证集上的代表性错误。"""
    exp = "E3" if "E3" in runs else sorted(runs)[0]
    rec = sorted(runs[exp].items())[0][1]
    sp = load_sample_predictions(rec)
    if sp is None:
        return []
    text_by_id = {str(i): str(t) for i, t in zip(ids, raw_text)}
    rows: List[Dict[str, Any]] = []
    labels = sp["labels"]
    seed_values = sp.get("seed")
    for i in range(sp["pred_class"].shape[0]):
        pred = int(sp["pred_class"][i])
        cls_pred = int(np.argmax(sp["probs"][i]))
        true = int(labels[i])
        if cls_pred == true:
            continue
        rows.append(
            {
                "experiment": exp,
                "seed": int(np.asarray(seed_values).reshape(-1)[i]) if seed_values is not None else -1,
                "view_id": str(sp["view_id"][i]),
                "sample_id": str(sp["ids"][i]),
                "true_class": true,
                "true_class_name": CLASS_NAMES[true],
                "pred_class": cls_pred,
                "pred_class_name": CLASS_NAMES[cls_pred],
                "true_intensity": float(sp["targets"][i]),
                "pred_intensity": float(sp["intensity"][i]),
                "abs_error": float(abs(sp["targets"][i] - sp["intensity"][i])),
                "p_negative": float(sp["probs"][i][0]),
                "p_neutral": float(sp["probs"][i][1]),
                "p_positive": float(sp["probs"][i][2]),
                "alpha_T": float(sp["fusion_weights"][i][0]),
                "alpha_A": float(sp["fusion_weights"][i][1]),
                "alpha_V": float(sp["fusion_weights"][i][2]),
                "raw_text": text_by_id.get(str(sp["ids"][i]), "")[:400],
            }
        )
    rows.sort(key=lambda r: (-r["abs_error"], r["sample_id"]))
    return rows[:top_k]


def class_intensity_table(runs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for exp in sorted(runs):
        for seed, rec in sorted(runs[exp].items()):
            sp = load_sample_predictions(rec)
            if sp is None:
                continue
            labels = sp["labels"]
            targets = sp["targets"]
            intens = sp["intensity"]
            preds = sp["pred_class"]
            clean_mask = np.array([str(v).startswith("valid|clean") for v in sp["view_id"]])
            for branch, sel_extra in (("完整输入", clean_mask),):
                sel = sel_extra
                if sel.sum() == 0:
                    continue
                for c in range(3):
                    m = sel & (labels == c)
                    if m.sum() == 0:
                        continue
                    f1 = _f1_for_class(labels[m], preds[m], c)
                    rows.append(
                        {
                            "experiment": exp,
                            "seed": int(np.asarray(sp["seed"]).reshape(-1)[0]) if "seed" in sp else -1,
                            "branch": branch,
                            "class_id": c,
                            "class_name": CLASS_NAMES[c],
                            "n": int(m.sum()),
                            "f1": f1,
                            "mae": float(np.mean(np.abs(targets[m] - intens[m]))),
                        }
                    )
                for name, m in (
                    ("|y|<0.33", sel & (np.abs(targets) < 0.33)),
                    ("|y|>=1.0", sel & (np.abs(targets) >= 1.0)),
                ):
                    if m.sum() == 0:
                        continue
                    rows.append(
                        {
                            "experiment": exp,
                            "seed": int(np.asarray(sp["seed"]).reshape(-1)[0]) if "seed" in sp else -1,
                            "branch": branch,
                            "class_id": -1,
                            "class_name": name,
                            "n": int(m.sum()),
                            "f1": None,
                            "mae": float(np.mean(np.abs(targets[m] - intens[m]))),
                        }
                    )
    return rows


def _f1_for_class(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> Optional[float]:
    tp = int(((y_pred == c) & (y_true == c)).sum())
    fp = int(((y_pred == c) & (y_true != c)).sum())
    fn = int(((y_pred != c) & (y_true == c)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return None
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def paired_bootstrap_table(
    runs: Dict[str, Dict[str, Any]], n_boot: int, seed: int
) -> List[Dict[str, Any]]:
    """同一验证视图下 E5/E3 与基线的配对 bootstrap（准确率差的置信区间，向量化实现）。"""
    from .metrics import paired_bootstrap_accuracy_diff

    rows: List[Dict[str, Any]] = []
    main = "E5" if "E5" in runs else ("E3" if "E3" in runs else sorted(runs)[-1])
    # 除"最终模型 vs 基线"外，额外给出 E4 vs E3（蒸馏是否真的改善主模型）的配对比较
    pairs = [(main, e) for e in ("E1", "E2", "E3", "E4", "E0") if e in runs and e != main]
    if "E4" in runs and "E3" in runs and main != "E4":
        pairs.append(("E4", "E3"))
    for main_exp, base_exp in pairs:
        a_seed = sorted(runs[main_exp])[0]
        b_seed = sorted(runs[base_exp])[0]
        sa = load_sample_predictions(runs[main_exp][a_seed])
        sb = load_sample_predictions(runs[base_exp][b_seed])
        if sa is None or sb is None:
            continue
        va = [str(v) for v in sa["view_id"]]
        vb = [str(v) for v in sb["view_id"]]
        common = [v for v in sorted(set(va)) if v in set(vb)]
        if not common:
            continue
        ia = np.concatenate([np.flatnonzero(np.asarray(va) == v) for v in common])
        ib = np.concatenate([np.flatnonzero(np.asarray(vb) == v) for v in common])
        y_true = sa["labels"][ia]
        if not np.array_equal(y_true, sb["labels"][ib]):
            continue
        correct_a = (sa["pred_class"][ia] == y_true).astype(np.float64)
        correct_b = (sb["pred_class"][ib] == y_true).astype(np.float64)
        res = paired_bootstrap_accuracy_diff(correct_a, correct_b, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "comparison": f"{main_exp} vs {base_exp}",
                "metric": "accuracy",
                "n_paired_samples": int(len(y_true)),
                "n_views": len(common),
                "accuracy_main": res.get("accuracy_a"),
                "accuracy_base": res.get("accuracy_b"),
                "diff_mean": res.get("diff_mean"),
                "ci_low": res.get("ci_low"),
                "ci_high": res.get("ci_high"),
                "significant_at_0.05": res.get("significant"),
                "n_boot": res.get("n_boot"),
            }
        )
    return rows


# --------------------------------------------------------------------------------------
# 图表
# --------------------------------------------------------------------------------------
def make_plots(
    cfg: Dict[str, Any],
    rate_rows: List[Dict[str, Any]],
    position_rows: List[Dict[str, Any]],
    weight_rows: List[Dict[str, Any]],
    confusion: Dict[str, Any],
    out_dir: Path,
    runs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    exps = sorted({r["experiment"] for r in rate_rows})
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(exps))))

    # 缺失率曲线：横轴为有效长度占比
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, m in zip(axes, MODALITIES):
        for e_i, exp in enumerate(exps):
            xs, ys, es = [], [], []
            for r in rate_rows:
                if r["experiment"] == exp and r["modality"] == m:
                    eff = r.get("effective_ratio_mean")
                    xs.append(eff if eff is not None else 1 - r["mask_rate"])
                    ys.append(r["macro_f1_mean"])
                    es.append(r["macro_f1_std"] or 0.0)
            if not xs:
                continue
            order = np.argsort(xs)
            xs = np.asarray(xs)[order]
            ys = np.asarray(ys)[order]
            es = np.asarray(es)[order]
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=exp, color=colors[e_i])
        ax.set_title(f"遮蔽模态 {m}")
        ax.set_xlabel("实际有效长度占比")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("macro-F1")
    axes[-1].legend(fontsize=8)
    fig.suptitle("缺失率 vs 有效长度占比（验证集固定视图，三种子均值±标准差）")
    fig.tight_layout()
    fig.savefig(out_dir / "missing_rate.png", dpi=160)
    plt.close(fig)

    # 缺失位置
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    width = 0.2
    for ax, m in zip(axes, MODALITIES):
        for e_i, exp in enumerate(exps):
            means, stds = [], []
            for pos in POSITIONS:
                vals = [r["macro_f1_mean"] for r in position_rows if r["experiment"] == exp and r["modality"] == m and r["position"] == pos]
                means.append(np.mean(vals) if vals else np.nan)
                stds.append(np.std(vals) if vals else 0.0)
            x = np.arange(len(POSITIONS)) + (e_i - len(exps) / 2 + 0.5) * width
            ax.bar(x, means, width, yerr=stds, capsize=3, label=exp, color=colors[e_i])
        ax.set_xticks(np.arange(len(POSITIONS)))
        ax.set_xticklabels(["开头", "中间", "结尾"])
        ax.set_title(f"遮蔽模态 {m}")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("macro-F1（四种缺失率平均）")
    axes[-1].legend(fontsize=8)
    fig.suptitle("缺失位置对 macro-F1 的影响")
    fig.tight_layout()
    fig.savefig(out_dir / "missing_position.png", dpi=160)
    plt.close(fig)

    # 混淆矩阵（主模型完整输入，多种子求和）
    key = "E5" if "E5" in confusion else ("E3" if "E3" in confusion else (sorted(confusion)[0] if confusion else None))
    if key:
        cm = confusion[key]["clean"].get("confusion_matrix_sum")
        if cm is None:
            per_seed = (runs or {}).get(key, {})
            mats = [
                np.asarray(rec["eval"]["condition_summary"]["clean"]["confusion_matrix_sum"])
                for rec in per_seed.values()
                if rec["eval"].get("condition_summary", {}).get("clean", {}).get("confusion_matrix_sum")
                is not None
            ]
            cm = np.sum(mats, axis=0).astype(int).tolist() if mats else None
        if cm is None:
            print(f"[analyze] 跳过混淆矩阵图：{key} 没有可用的混淆矩阵")
        else:
            cm_arr = np.asarray(cm, dtype=float)
            fig, ax = plt.subplots(figsize=(5.2, 4.4))
            im = ax.imshow(cm_arr, cmap="Blues")
            ax.set_xticks(range(3), CLASS_NAMES)
            ax.set_yticks(range(3), CLASS_NAMES)
            ax.set_xlabel("预测")
            ax.set_ylabel("真实")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{int(cm_arr[i, j])}", ha="center", va="center",
                            color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black")
            ax.set_title(f"{key} 完整输入混淆矩阵（三种子求和）")
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.savefig(out_dir / "confusion_matrix.png", dpi=160)
            plt.close(fig)

    # 动态门控权重
    if weight_rows:
        conds = sorted({r["masked_modalities"] for r in weight_rows if r["experiment"] == "E3"})
        conds = [c for c in conds if c][:12]
        if conds:
            fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(conds) + 3), 4.4))
            bottom = np.zeros(len(conds))
            for k, comp in enumerate(("alpha_T_mean", "alpha_A_mean", "alpha_V_mean")):
                vals = []
                for c in conds:
                    sel = [r[comp] for r in weight_rows if r["experiment"] == "E3" and r["masked_modalities"] == c]
                    vals.append(np.mean(sel) if sel else 0.0)
                ax.bar(conds, vals, bottom=bottom, label=MODALITIES[k])
                bottom += np.asarray(vals)
            ax.set_ylabel("平均融合权重 α")
            ax.set_title("E3 动态门控在各类遮蔽配置下的平均权重")
            ax.tick_params(axis="x", rotation=45)
            ax.legend()
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(out_dir / "modality_weights.png", dpi=160)
            plt.close(fig)


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    if columns is None:
        # 取所有行的键的并集：不同实验的模态数不同（E0 只有文本），
        # 只看第一行会丢掉后续行才有的列（例如 alpha_A/alpha_V）。
        cols: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in cols:
                    cols.append(k)
    else:
        cols = list(columns)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c) for c in cols})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="问题 2 规律分析与图表")
    parser.add_argument("--config", default="configs/problem2.yaml")
    parser.add_argument("--split", default="valid")
    args = parser.parse_args(argv)

    from .common import load_config, load_json
    from .dataset import load_split

    cfg = load_config(args.config)
    out = output_dir(cfg)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir = out / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(cfg, args.split)
    if not runs:
        raise SystemExit("没有找到任何 eval_valid.json，请先运行 evaluate")
    print(f"[analyze] 已载入实验: {sorted(runs)}；每个实验的种子数: "
          f"{ {e: len(v) for e, v in sorted(runs.items())} }")

    plan_path = resolve(cfg["data"]["cache_dir"]) / "validation_masks.json"
    plan = load_json(plan_path) if plan_path.exists() else None

    rows, detail = build_metric_summary(runs)
    write_csv(table_dir / "metric_summary.csv", rows)
    print("[analyze] 主表完成")

    rate_rows = missing_rate_table(runs, detail, plan)
    write_csv(table_dir / "missing_rate.csv", rate_rows)
    position_rows = missing_position_table(runs, detail)
    write_csv(table_dir / "missing_position.csv", position_rows)
    type_rows = modality_type_table(runs, detail)
    write_csv(table_dir / "missing_modality_type.csv", type_rows)
    print("[analyze] 缺失规律表完成")

    valid = load_split(cfg, "valid")
    weight_rows = modality_weight_table(runs, valid.ids)
    write_csv(table_dir / "modality_weights.csv", weight_rows)
    ci_rows = class_intensity_table(runs)
    write_csv(table_dir / "class_intensity.csv", ci_rows)
    print("[analyze] 门控/类别强度表完成")

    err_rows = error_cases(runs, valid.raw_text, valid.ids)
    write_csv(table_dir / "error_cases.csv", err_rows)

    boot_rows = paired_bootstrap_table(
        runs, int(cfg.get("evaluation", {}).get("bootstrap_samples", 2000)),
        int(cfg.get("evaluation", {}).get("bootstrap_seed", 20240924)),
    )
    write_csv(table_dir / "paired_bootstrap.csv", boot_rows)
    print("[analyze] 错误分析与配对 bootstrap 完成")

    confusion = {exp: {"clean": d["clean"]} for exp, d in detail.items()}
    make_plots(cfg, rate_rows, position_rows, weight_rows, confusion, fig_dir, runs)

    # 消融表 = 主表核心列
    ablation_cols = [
        "experiment", "experiment_label", "n_seeds",
        "clean_macro_f1_mean", "clean_macro_f1_std", "clean_accuracy_mean", "clean_mae_mean", "clean_pearson_mean",
        "missing36_macro_f1_mean", "missing36_macro_f1_std", "missing36_accuracy_mean", "missing36_mae_mean",
        "allmissing_macro_f1_mean", "allmissing_mae_mean",
    ]
    write_csv(out / "ablation.csv", rows, ablation_cols)
    write_csv(out / "metric_summary.csv", rows)

    save_json(
        out / "analysis_report.json",
        {
            "script": "src.problem2.analyze",
            "script_version": __version__,
            "split": args.split,
            "experiments": sorted(runs),
            "seeds": {exp: sorted(v) for exp, v in runs.items()},
            "condition_summary": {exp: d["condition_summary"] for exp, d in detail.items()},
            "files": {
                "metric_summary": str(out / "metric_summary.csv"),
                "ablation": str(out / "ablation.csv"),
                "missing_rate": str(table_dir / "missing_rate.csv"),
                "missing_position": str(table_dir / "missing_position.csv"),
                "missing_modality_type": str(table_dir / "missing_modality_type.csv"),
                "modality_weights": str(table_dir / "modality_weights.csv"),
                "class_intensity": str(table_dir / "class_intensity.csv"),
                "error_cases": str(table_dir / "error_cases.csv"),
                "paired_bootstrap": str(table_dir / "paired_bootstrap.csv"),
                "figures": [
                    str(fig_dir / "missing_rate.png"),
                    str(fig_dir / "missing_position.png"),
                    str(fig_dir / "confusion_matrix.png"),
                    str(fig_dir / "modality_weights.png"),
                ],
            },
        },
    )
    print(f"[analyze] 完成：{len(runs)} 个实验，主表 {out / 'metric_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
