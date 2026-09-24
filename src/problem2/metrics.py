"""指标计算：分类（Accuracy / macro-F1 / 每类 P-R-F1 / 混淆矩阵）与回归（MAE / Pearson）。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from . import CLASS_NAMES

LABELS = [0, 1, 2]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out: Dict[str, Any] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["macro_f1"] = float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))
    out["weighted_f1"] = float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    out["per_class"] = {
        CLASS_NAMES[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i in range(3)
    }
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=LABELS).astype(int).tolist()
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    out: Dict[str, Any] = {}
    out["mae"] = float(np.mean(np.abs(y_true - y_pred)))
    out["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if y_pred.size < 2 or np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        out["pearson"] = None
        out["pearson_undefined_reason"] = (
            "预测或真值为常数（std < 1e-12），Pearson 未定义；已按未定义记录，不写 0"
        )
    else:
        out["pearson"] = float(np.corrcoef(y_true, y_pred)[0, 1])
        out["pearson_undefined_reason"] = None
    return out


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, targets: np.ndarray, intensities: np.ndarray
) -> Dict[str, Any]:
    out = classification_metrics(y_true, y_pred)
    out.update(regression_metrics(targets, intensities))
    out["n"] = int(len(y_true))
    return out


def merge_metric_dicts(metric_dicts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """把多个视图/种子的指标合并为 mean/std。"""
    keys = ["accuracy", "macro_f1", "weighted_f1", "mae", "rmse", "pearson"]
    out: Dict[str, Any] = {"n_runs": len(metric_dicts)}
    for k in keys:
        vals = [m.get(k) for m in metric_dicts]
        vals_f = [float(v) for v in vals if v is not None and np.isfinite(v)]
        if not vals_f:
            out[k] = None
            out[f"{k}_std"] = None
            out[f"{k}_n"] = 0
            continue
        out[k] = float(np.mean(vals_f))
        out[f"{k}_std"] = float(np.std(vals_f, ddof=1)) if len(vals_f) > 1 else 0.0
        out[f"{k}_n"] = len(vals_f)
    # 每类 F1 的合并
    per_class: Dict[str, Any] = {}
    for cname in CLASS_NAMES:
        vals = [m.get("per_class", {}).get(cname, {}).get("f1") for m in metric_dicts]
        vals_f = [float(v) for v in vals if v is not None and np.isfinite(v)]
        per_class[cname] = {
            "f1_mean": float(np.mean(vals_f)) if vals_f else None,
            "f1_std": float(np.std(vals_f, ddof=1)) if len(vals_f) > 1 else (0.0 if vals_f else None),
        }
    out["per_class"] = per_class
    # 混淆矩阵求和
    cms = [np.asarray(m["confusion_matrix"]) for m in metric_dicts if m.get("confusion_matrix") is not None]
    out["confusion_matrix_sum"] = np.sum(cms, axis=0).astype(int).tolist() if cms else None
    return out


def paired_bootstrap_accuracy_diff(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 20240924,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """配对 bootstrap（向量化）：acc(A) - acc(B) 的置信区间。

    correct_a / correct_b 为同一样本顺序下的 0/1 正确性向量。
    """
    a = np.asarray(correct_a, dtype=np.float64).reshape(-1)
    b = np.asarray(correct_b, dtype=np.float64).reshape(-1)
    n = a.size
    if n == 0 or n != b.size:
        return {"defined": False, "reason": "样本为空或长度不一致"}
    rng = np.random.default_rng(seed)
    n = a.size
    diffs = np.empty(n_boot, dtype=np.float64)
    chunk = max(1, min(200, n_boot))
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(m, n))  # 分块，避免 n_boot × n 的整数矩阵过大
        diffs[done : done + m] = a[idx].mean(axis=1) - b[idx].mean(axis=1)
        done += m
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "defined": True,
        "accuracy_a": float(a.mean()),
        "accuracy_b": float(b.mean()),
        "diff_mean": float(a.mean() - b.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "significant": bool(lo > 0 or hi < 0),
    }


def paired_bootstrap_diff(
    metric_fn,
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    extra_a: Optional[np.ndarray] = None,
    extra_b: Optional[np.ndarray] = None,
    n_boot: int = 2000,
    seed: int = 20240924,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """配对 bootstrap：模型 A 与模型 B 在同一批样本上的指标差置信区间（通用指标）。"""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    base_a = metric_fn(y_true, pred_a, *([] if extra_a is None else [extra_a]))
    base_b = metric_fn(y_true, pred_b, *([] if extra_b is None else [extra_b]))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = metric_fn(y_true[idx], pred_a[idx], *([] if extra_a is None else [extra_a[idx]]))
        b = metric_fn(y_true[idx], pred_b[idx], *([] if extra_b is None else [extra_b[idx]]))
        if a is None or b is None:
            continue
        diffs.append(a - b)
    if not diffs:
        return {"defined": False}
    diffs = np.asarray(diffs, dtype=np.float64)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "defined": True,
        "metric_a": float(base_a),
        "metric_b": float(base_b),
        "diff_mean": float(base_a - base_b),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": int(len(diffs)),
        "alpha": alpha,
    }
