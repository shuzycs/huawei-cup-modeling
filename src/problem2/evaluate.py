"""评估：固定验证缺失视图上的指标、逐样本预测输出、附件 2 test 独立检验。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import MODALITIES, __version__
from .common import (
    REPO_ROOT,
    environment_report,
    load_config,
    mean_std,
    output_dir,
    resolve,
    save_json,
    set_seed,
)
from .dataset import (
    SplitData,
    TextEmbedder,
    iter_batches,
    load_split,
    load_validation_plan,
    make_batch,
)
from .masks import view_observed_from_plan
from .metrics import compute_metrics, merge_metric_dicts
from .model import build_model_from_config

MISSING_RATES = (0.10, 0.30, 0.50, 0.70)
POSITIONS = ("start", "middle", "end")
SINGLE_MODALITY_CONDITIONS = [f"{m}-{p}" for m in MODALITIES for p in POSITIONS]
DUAL_CONDITIONS = ["dual-T-A", "dual-T-V", "dual-A-V"]


def single_condition_keys(rates: Sequence[float] = MISSING_RATES) -> List[str]:
    """36 个单模态缺失条件（模态 × 位置 × 比例）的条件键。"""
    return [
        f"{m}-{p}|{int(round(r * 100))}"
        for m in MODALITIES
        for p in POSITIONS
        for r in rates
    ]


# --------------------------------------------------------------------------------------
# 前向推理
# --------------------------------------------------------------------------------------
@torch.no_grad()
def predict_views(
    model,
    embedder: TextEmbedder,
    data: SplitData,
    observed: Optional[Dict[str, np.ndarray]],
    batch_size: int = 128,
    device: torch.device | str = "cpu",
) -> Dict[str, np.ndarray]:
    model.eval()
    probs, logits_all, intensity_all, idx_all, weights_all, avail_all = [], [], [], [], [], []
    for batch_idx in iter_batches(data.n, batch_size, shuffle=False):
        batch = make_batch(data, batch_idx, observed=observed, device=device, with_labels=False)
        text_emb = embedder.encode(batch["text_bert"], batch["text_observed"], batch_size=batch_size)
        out = model(
            text_emb,
            batch["audio"],
            batch["vision"],
            observed=batch["observed"],
            text_observed=batch["text_observed"],
        )
        probs.append(torch.softmax(out["class_logits"], dim=-1).cpu().numpy())
        logits_all.append(out["class_logits"].cpu().numpy())
        intensity_all.append(out["intensity"].cpu().numpy())
        weights_all.append(out["fusion_weights"].cpu().numpy())
        avail_all.append(out["available"].cpu().numpy())
        idx_all.append(batch_idx)
    return {
        "indices": np.concatenate(idx_all),
        "probs": np.concatenate(probs),
        "logits": np.concatenate(logits_all),
        "intensity": np.concatenate(intensity_all),
        "fusion_weights": np.concatenate(weights_all),
        "available": np.concatenate(avail_all),
        "pred_class": np.concatenate(probs).argmax(axis=-1).astype(np.int64),
    }


def metrics_from_pred(
    pred: Dict[str, np.ndarray], data: SplitData, indices: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    idx = pred["indices"] if indices is None else indices
    return compute_metrics(
        data.labels[idx], pred["pred_class"], data.targets[idx], pred["intensity"]
    )


# --------------------------------------------------------------------------------------
# 固定验证视图上的完整评估
# --------------------------------------------------------------------------------------
def condition_key(view: Dict[str, Any]) -> str:
    """视图条件键：clean 保持 'clean'，缺失视图带缺失率，避免不同比例被平均掉。"""
    if view.get("kind") == "clean":
        return "clean"
    return f"{view['condition']}|{int(round(float(view.get('rate', 0.0)) * 100))}"


def evaluate_fixed_views(
    model,
    embedder: TextEmbedder,
    data: SplitData,
    plan: Dict[str, Any],
    batch_size: int,
    device,
    log: Optional[Any] = None,
    collect_sample_predictions: bool = False,
) -> Dict[str, Any]:
    views = plan["views"]
    per_view: List[Dict[str, Any]] = []
    sample_pred: List[Dict[str, Any]] = []
    observed_map: Dict[str, Dict[str, np.ndarray]] = {}

    for i, view in enumerate(views):
        observed = view_observed_from_plan(plan, view, data.valid, data.n)
        observed_map[view["view_id"]] = observed
        pred = predict_views(model, embedder, data, observed, batch_size=batch_size, device=device)
        m = metrics_from_pred(pred, data)
        rec = {
            "view_id": view["view_id"],
            "condition": view["condition"],
            "condition_key": condition_key(view),
            "kind": view["kind"],
            "masked_modalities": view["masked_modalities"],
            "rate": view["rate"],
            "position": view["position"],
            "repeat": view["repeat"],
            "mean_actual_rate": view.get("mean_actual_rate"),
            "metrics": m,
        }
        per_view.append(rec)
        if collect_sample_predictions:
            sample_pred.append(
                {
                    "view_id": view["view_id"],
                    "indices": pred["indices"],
                    "pred_class": pred["pred_class"],
                    "probs": pred["probs"],
                    "intensity": pred["intensity"],
                    "fusion_weights": pred["fusion_weights"],
                    "available": pred["available"],
                }
            )
        if log is not None and (i + 1) % 10 == 0:
            log.info("  已评估 %d/%d 个固定视图", i + 1, len(views))

    # ---- 条件级聚合（键含缺失率，例如 "T-start|30"）----
    conditions: Dict[str, List[Dict[str, Any]]] = {}
    for rec in per_view:
        conditions.setdefault(rec["condition_key"], []).append(rec["metrics"])

    condition_summary: Dict[str, Any] = {}
    for cond, ms in conditions.items():
        merged = merge_metric_dicts(ms)
        merged["rate"] = 0.0 if cond == "clean" else float(cond.split("|")[-1]) / 100.0
        merged["position"] = "none" if cond == "clean" else cond.split("|")[0].split("-")[-1]
        merged["masked_modalities"] = (
            [] if cond == "clean"
            else (cond.split("-")[0].split("|")[0].split("dual")[-1].strip("-").split("-")
                  if cond.startswith("dual") else [cond.split("-")[0]])
        )
        condition_summary[cond] = merged

    clean_metrics = condition_summary.get("clean", {})
    single_cond_metrics = [
        condition_summary[c] for c in single_condition_keys() if c in condition_summary
    ]
    missing_cond_metrics = [v for k, v in condition_summary.items() if k != "clean"]

    def _avg(cfg_list, key):
        vals = [c[key] for c in cfg_list if c.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    overall = {
        "clean_macro_f1": clean_metrics.get("macro_f1"),
        "clean_accuracy": clean_metrics.get("accuracy"),
        "clean_mae": clean_metrics.get("mae"),
        "clean_pearson": clean_metrics.get("pearson"),
        "primary_macro_f1_missing36": _avg(single_cond_metrics, "macro_f1"),
        "primary_macro_f1_missing36_std": _avg(single_cond_metrics, "macro_f1_std"),
        "all_missing_macro_f1": _avg(missing_cond_metrics, "macro_f1"),
        "all_missing_mae": _avg(missing_cond_metrics, "mae"),
        "single_missing_macro_f1": _avg(single_cond_metrics, "macro_f1"),
        "single_missing_mae": _avg(single_cond_metrics, "mae"),
        "n_single_conditions": len(single_cond_metrics),
    }

    result = {
        "per_view": per_view,
        "condition_summary": condition_summary,
        "overall": overall,
    }
    if collect_sample_predictions:
        result["_sample_predictions"] = sample_pred
    return result


# --------------------------------------------------------------------------------------
# 命令行：对已训练权重做正式评估
# --------------------------------------------------------------------------------------
def save_sample_predictions(
    path: Path, sample_predictions: List[Dict[str, Any]], data: SplitData, seed: int, modalities: Sequence[str] = MODALITIES
) -> Path:
    """把逐样本预测（含缺失配置与融合权重）写成 npz，供 analyze 复核与画图。"""
    view_id: List[str] = []
    indices: List[np.ndarray] = []
    ids: List[str] = []
    labels: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    intens: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    available: List[np.ndarray] = []
    seed_arr: List[np.ndarray] = []
    for rec in sample_predictions:
        idx = np.asarray(rec["indices"])
        n = len(idx)
        view_id.extend([rec["view_id"]] * n)
        indices.append(idx)
        ids.extend([str(x) for x in data.ids[idx]])
        labels.append(data.labels[idx])
        targets.append(data.targets[idx])
        preds.append(np.asarray(rec["pred_class"]))
        intens.append(np.asarray(rec["intensity"]))
        probs.append(np.asarray(rec["probs"]))
        weights.append(np.asarray(rec["fusion_weights"]))
        available.append(np.asarray(rec["available"]))
        seed_arr.append(np.full(n, seed, dtype=np.int64))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        view_id=np.asarray(view_id),
        indices=np.concatenate(indices),
        ids=np.asarray(ids),
        labels=np.concatenate(labels),
        targets=np.concatenate(targets),
        pred_class=np.concatenate(preds),
        intensity=np.concatenate(intens),
        probs=np.concatenate(probs),
        fusion_weights=np.concatenate(weights),
        available=np.concatenate(available),
        seed=np.concatenate(seed_arr),
        modalities=np.asarray(list(modalities)),
    )
    return path


def run_dir(cfg: Dict[str, Any], experiment: str, seed: int) -> Path:
    return output_dir(cfg) / "runs" / f"{experiment}_seed{seed}"


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def load_checkpoint_model(cfg: Dict[str, Any], experiment: str, checkpoint: Path, device, embedder=None):
    from .infer import load_student

    return load_student(cfg, experiment, checkpoint, device, embedder=embedder)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="问题 2 评估")
    parser.add_argument("--config", default="configs/problem2.yaml")
    parser.add_argument("--split", default="valid", choices=["valid", "test", "train"])
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all-experiments", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device(cfg["training"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    batch_size = int(args.batch_size or cfg["training"].get("batch_size", 32) * 4)

    experiments = [args.experiment] if args.experiment else (["E0", "E1", "E2", "E3", "E4", "E5"] if args.all_experiments else [])
    if not experiments:
        raise SystemExit("请指定 --experiment 或 --all-experiments")

    data = load_split(cfg, args.split)
    plan = load_validation_plan(cfg) if args.split == "valid" else None

    embedder = TextEmbedder(
        cfg["text_model"]["repo_id"], cfg["text_model"].get("revision"), cfg["text_model"]["local_dir"], device
    )

    summary: Dict[str, Any] = {"split": args.split, "experiments": {}}
    for exp in experiments:
        seeds = [args.seed] if args.seed is not None else list(cfg["training"]["seeds"])
        summary["experiments"][exp] = {}
        for seed in seeds:
            ckpt = Path(args.checkpoint) if args.checkpoint else run_dir(cfg, exp, seed) / "best.safetensors"
            if not ckpt.exists():
                print(f"[evaluate] 跳过 {exp} seed={seed}：缺少权重 {ckpt}")
                continue
            model = load_checkpoint_model(cfg, exp, ckpt, device, embedder=embedder)
            set_seed(seed)
            t0 = time.time()
            if args.split == "valid":
                res = evaluate_fixed_views(
                    model, embedder, data, plan, batch_size, device, collect_sample_predictions=True
                )
            else:
                pred = predict_views(model, embedder, data, None, batch_size=batch_size, device=device)
                clean = metrics_from_pred(pred, data)
                res = {"clean": clean, "overall": {"clean_macro_f1": clean["macro_f1"], "clean_mae": clean["mae"]}}
            res["elapsed_seconds"] = round(time.time() - t0, 2)
            res["experiment"] = exp
            res["seed"] = seed
            res["checkpoint"] = str(ckpt)
            res["environment"] = environment_report()
            summary["experiments"][exp][str(seed)] = {k: v for k, v in res.items() if k != "_sample_predictions"}

            out_path = resolve(args.out) if args.out else run_dir(cfg, exp, seed) / f"eval_{args.split}.json"
            samples_path = None
            if "_sample_predictions" in res:
                samples_path = run_dir(cfg, exp, seed) / f"eval_{args.split}_samples.npz"
                save_sample_predictions(
                    samples_path, res["_sample_predictions"], data, seed, modalities=model.modalities
                )
            save_json(out_path, res)
            print(
                f"[evaluate] {exp} seed={seed} {args.split}: clean macro-F1="
                f"{_fmt(res['overall'].get('clean_macro_f1'))} missing36 macro-F1="
                f"{_fmt(res['overall'].get('primary_macro_f1_missing36'))} -> {out_path}"
            )

    out_summary = output_dir(cfg) / f"eval_summary_{args.split}.json"
    save_json(out_summary, summary)
    print(f"[evaluate] 汇总已写出 {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
