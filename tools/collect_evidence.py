"""汇总所有实验证据（只读，用于复核报告中的每一个数字）。

用法::

    python -m tools.collect_evidence                  # 全部小节
    python -m tools.collect_evidence --section A B    # 只打印指定小节
    python -m tools.collect_evidence --json /tmp/e.json

只读取 outputs/problem2 下真实存在的指标/日志/训练记录文件，
不做任何估计、补齐或平滑处理。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # 允许 python tools/collect_evidence.py 直接运行
    sys.path.insert(0, str(REPO_ROOT))

from src.problem2 import MODALITIES  # noqa: E402
from src.problem2.common import load_config, output_dir, save_json  # noqa: E402

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


def load_eval(out: Path, exp: str, seed: int, split: str = "valid") -> Optional[Dict[str, Any]]:
    p = out / "runs" / f"{exp}_seed{seed}" / f"eval_{split}.json"
    return read_json(p) if p.exists() else None


def _ms(values: List[float]) -> tuple:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None, None
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else 0.0)


def _fmt(value, nd: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{nd}f}"


# --------------------------------------------------------------------------------------
# A. 逐 run 指标
# --------------------------------------------------------------------------------------
def section_a(out: Path) -> Dict[str, Any]:
    print("=" * 104)
    print("A. 每个 run 的真实指标（runs/<E>_seed<S>/eval_valid.json）")
    print("=" * 104)
    print(
        f"{'exp':4s}{'seed':>5s}{'cleanF1':>9s}{'cleanAcc':>9s}{'cleanMAE':>9s}{'cleanPear':>10s}"
        f"{'miss36F1':>9s}{'allMissF1':>10s}{'allMissMAE':>11s}"
    )
    data: Dict[str, Any] = {}
    for exp in EXPS:
        for seed in SEEDS:
            d = load_eval(out, exp, seed)
            if d is None:
                print(f"{exp:4s}{seed:>5d}   <缺少 eval_valid.json>")
                continue
            o = d["overall"]
            data[f"{exp}_{seed}"] = o
            print(
                f"{exp:4s}{seed:>5d}{o['clean_macro_f1']:>9.4f}{o['clean_accuracy']:>9.4f}"
                f"{o['clean_mae']:>9.4f}{_fmt(o['clean_pearson']):>10s}"
                f"{o['primary_macro_f1_missing36']:>9.4f}"
                f"{_fmt(o['all_missing_macro_f1']):>10s}{o['all_missing_mae']:>11.4f}"
            )
    return data


# --------------------------------------------------------------------------------------
# B. 三种子聚合
# --------------------------------------------------------------------------------------
def section_b(out: Path) -> Dict[str, Any]:
    print()
    print("=" * 104)
    print("B. 三种子聚合（均值 ± 样本标准差 ddof=1）")
    print("=" * 104)
    print(
        f"{'exp':5s}{'n':>3s}{'clean_macroF1':>22s}{'clean_acc':>17s}{'clean_mae':>17s}"
        f"{'miss36_macroF1':>23s}{'all_miss_mae':>17s}{'clean_pearson':>16s}"
    )
    agg: Dict[str, Any] = {}
    for exp in EXPS:
        overalls = [d["overall"] for d in (load_eval(out, exp, s) for s in SEEDS) if d is not None]
        if not overalls:
            continue
        rec = {
            "n": len(overalls),
            "clean_macro_f1": _ms([o["clean_macro_f1"] for o in overalls]),
            "clean_accuracy": _ms([o["clean_accuracy"] for o in overalls]),
            "clean_mae": _ms([o["clean_mae"] for o in overalls]),
            "clean_pearson": _ms([o["clean_pearson"] for o in overalls]),
            "missing36_macro_f1": _ms([o["primary_macro_f1_missing36"] for o in overalls]),
            "all_missing_mae": _ms([o["all_missing_mae"] for o in overalls]),
        }
        agg[exp] = rec
        print(
            f"{exp:5s}{rec['n']:>3d}{rec['clean_macro_f1'][0]:>15.4f}±{rec['clean_macro_f1'][1]:.4f}"
            f"{rec['clean_accuracy'][0]:>10.4f}±0.0000"
            f"{rec['clean_mae'][0]:>10.4f}±{rec['clean_mae'][1]:.4f}"
            f"{rec['missing36_macro_f1'][0]:>16.4f}±{rec['missing36_macro_f1'][1]:.4f}"
            f"{rec['all_missing_mae'][0]:>10.4f}±{rec['all_missing_mae'][1]:.4f}"
            f"{rec['clean_pearson'][0]:>16.4f}"
        )
    return agg


# --------------------------------------------------------------------------------------
# C. 训练信息
# --------------------------------------------------------------------------------------
def section_c(out: Path) -> Dict[str, Any]:
    print()
    print("=" * 104)
    print("C. 每个 run 的训练信息（result.json / run_info.json / history.json）")
    print("=" * 104)
    data: Dict[str, Any] = {}
    for exp in EXPS:
        for seed in SEEDS:
            rd = out / "runs" / f"{exp}_seed{seed}"
            if not (rd / "result.json").exists():
                continue
            res, info, hist = (
                read_json(rd / "result.json"),
                read_json(rd / "run_info.json"),
                read_json(rd / "history.json"),
            )
            rec = {
                "best_epoch": res["best"]["epoch"],
                "epochs_run": res["epochs_run"],
                "elapsed_seconds": res["elapsed_seconds"],
                "trainable_params": res["model_params"]["trainable"],
                "mask_augmentation": info.get("mask_augmentation"),
                "class_weighted_loss": info.get("class_weighted_loss"),
                "unfrozen_text_encoder": info.get("unfrozen_text_encoder"),
                "overrides": info.get("overrides") or [],
                "checkpoint_sha256": res["checkpoint_sha256"],
                "final_train_loss": hist[-1]["train_loss"],
                "mean_actual_mask_rate": float(np.mean([h["mean_actual_mask_rate"] for h in hist])),
            }
            data[f"{exp}_{seed}"] = rec
            print(
                f"{exp} seed={seed}: best_epoch={rec['best_epoch']:>2d} epochs_run={rec['epochs_run']:>2d} "
                f"elapsed={rec['elapsed_seconds']:>6.1f}s trainable={rec['trainable_params']:>7d} "
                f"mask_aug={rec['mask_augmentation']} class_weighted={rec['class_weighted_loss']} "
                f"unfrozen={rec['unfrozen_text_encoder']}"
            )
            print(
                f"           final_train_loss={rec['final_train_loss']:.4f} "
                f"mean_mask_rate={rec['mean_actual_mask_rate']:.4f} "
                f"ckpt_sha256={rec['checkpoint_sha256'][:16]}… overrides={rec['overrides']}"
            )
    return data


# --------------------------------------------------------------------------------------
# D. 教师
# --------------------------------------------------------------------------------------
def section_d(out: Path) -> Dict[str, Any]:
    print()
    print("=" * 104)
    print("D. E4 教师（teacher/mosei_text768_seed<S>/）")
    print("=" * 104)
    data: Dict[str, Any] = {}
    for seed in SEEDS:
        rd = out / "teacher" / f"mosei_text768_seed{seed}"
        if not (rd / "result.json").exists():
            continue
        res, info = read_json(rd / "result.json"), read_json(rd / "run_info.json")
        rec = {
            "best_epoch": res["best"]["epoch"],
            "valid_macro_f1": res["best"]["macro_f1"],
            "valid_mae": res["best"]["mae"],
            "epochs_run": res["epochs_run"],
            "elapsed_seconds": res["elapsed_seconds"],
            "params": res["model_params"]["trainable"],
            "text_dim": info["text_dim"],
            "checkpoint_sha256": res["checkpoint_sha256"],
        }
        data[str(seed)] = rec
        print(
            f"teacher seed={seed}: best_epoch={rec['best_epoch']:>2d} valid_macroF1={rec['valid_macro_f1']:.4f} "
            f"valid_MAE={rec['valid_mae']:.4f} epochs_run={rec['epochs_run']} elapsed={rec['elapsed_seconds']:.1f}s "
            f"params={rec['params']} text_dim={rec['text_dim']}"
        )
    return data


# --------------------------------------------------------------------------------------
# E. 蒸馏搜索
# --------------------------------------------------------------------------------------
def section_e(out: Path) -> Dict[str, Any]:
    print()
    print("=" * 104)
    print("E. E4 蒸馏变体搜索（search/E4_*/eval_valid.json）")
    print("=" * 104)
    data: Dict[str, Any] = {}
    for d in sorted((out / "search").glob("E4_*")):
        ev = d / "eval_valid.json"
        if not ev.exists():
            continue
        o = read_json(ev)["overall"]
        info = read_json(d / "run_info.json") if (d / "run_info.json").exists() else {}
        rec = {
            "clean_macro_f1": o["clean_macro_f1"],
            "missing36_macro_f1": o["primary_macro_f1_missing36"],
            "clean_mae": o["clean_mae"],
            "overrides": info.get("overrides") or [],
        }
        data[d.name] = rec
        print(
            f"{d.name:24s} cleanF1={rec['clean_macro_f1']:.4f} missing36F1={rec['missing36_macro_f1']:.4f} "
            f"cleanMAE={rec['clean_mae']:.4f} overrides={rec['overrides']}"
        )
    return data


# --------------------------------------------------------------------------------------
# F. test 独立检验
# --------------------------------------------------------------------------------------
def section_f(out: Path) -> Dict[str, Any]:
    print()
    print("=" * 104)
    print("F. 附件 2 test 独立检验（eval_test.json）")
    print("=" * 104)
    data: Dict[str, Any] = {}
    for exp in EXPS:
        for seed in SEEDS:
            p = out / "runs" / f"{exp}_seed{seed}" / "eval_test.json"
            if not p.exists():
                continue
            c = read_json(p)["clean"]
            data[f"{exp}_{seed}"] = c
            print(
                f"{exp} seed={seed}: test macroF1={c['macro_f1']:.4f} acc={c['accuracy']:.4f} "
                f"mae={c['mae']:.4f} pearson={_fmt(c['pearson'])} n={c['n']}"
            )
    return data


# --------------------------------------------------------------------------------------
# G. 规律、门控、错误分析
# --------------------------------------------------------------------------------------
def section_g(out: Path) -> Dict[str, Any]:
    print()
    print("=" * 104)
    print("G. 缺失规律、门控、错误分析（tables/*.csv 与 eval_valid_samples.npz）")
    print("=" * 104)
    data: Dict[str, Any] = {}

    print("-- 缺失模态类型（按遮蔽模态对 12 个条件平均）")
    for r in read_csv_rows(out / "tables" / "missing_modality_type.csv"):
        print(
            f"   {r['experiment']:3s} {r['masked_modality']:10s} n={r['n_conditions']:>3s} "
            f"macroF1={float(r['macro_f1_mean']):.4f} mae={float(r['mae_mean']):.4f}"
        )

    print("-- 缺失率（E5，横轴 = 实际有效长度占比）")
    for r in read_csv_rows(out / "tables" / "missing_rate.csv"):
        if r["experiment"] == "E5":
            print(
                f"   E5 {r['modality']} 目标{r['mask_rate']:>4s} 实际有效={float(r['effective_ratio_mean']):.4f} "
                f"macroF1={float(r['macro_f1_mean']):.4f} mae={float(r['mae_mean']):.4f}"
            )

    print("-- 缺失位置（E5, 50% 缺失率）")
    for r in read_csv_rows(out / "tables" / "missing_position.csv"):
        if r["experiment"] == "E5" and r["mask_rate"] == "0.5":
            print(f"   E5 {r['modality']} {r['position']:7s} macroF1={float(r['macro_f1_mean']):.4f}")

    print("-- 配对 bootstrap")
    for r in read_csv_rows(out / "tables" / "paired_bootstrap.csv"):
        print(
            f"   {r['comparison']:12s} diff={float(r['diff_mean']):+.4f} "
            f"CI=[{float(r['ci_low']):+.4f},{float(r['ci_high']):+.4f}] "
            f"显著={r['significant_at_0.05']} n_paired={r['n_paired_samples']}"
        )

    print("-- 类别与强度（E5 seed42，完整视图）")
    for r in read_csv_rows(out / "tables" / "class_intensity.csv"):
        if r["experiment"] == "E5" and r["seed"] == "42":
            print(
                f"   {r['class_name']:10s} n={r['n']:>4s} "
                f"f1={_fmt(float(r['f1'])) if r['f1'] else 'n/a':>6s} mae={float(r['mae']):.4f}"
            )

    print("-- 门控权重（直接读 eval_valid_samples.npz）")
    for exp in ("E3", "E4", "E5"):
        p = out / "runs" / f"{exp}_seed42" / "eval_valid_samples.npz"
        if not p.exists():
            print(f"   {exp}: 缺少 {p.name}")
            continue
        with np.load(p) as z:
            vid = np.asarray([str(x) for x in z["view_id"]])
            w = z["fusion_weights"]
            mods = [str(x) for x in z["modalities"]] if "modalities" in z.files else list(MODALITIES)
        sel = vid == "valid|clean|k0"
        if sel.any():
            m = w[sel].mean(axis=0)
            print(f"   {exp} clean: " + "  ".join(f"a{x}={m[i]:.4f}" for i, x in enumerate(mods)))
        for rate in (10, 30, 50, 70):
            sel = vid == f"valid|T|middle|{rate}|k0"
            if sel.any():
                m = w[sel].mean(axis=0)
                print(f"   {exp} T|middle|{rate}: " + "  ".join(f"a{x}={m[i]:.4f}" for i, x in enumerate(mods)))

    p = out / "runs" / "E5_seed42" / "eval_valid_samples.npz"
    if p.exists():
        with np.load(p) as z:
            vid = np.asarray([str(x) for x in z["view_id"]])
            sel = vid == "valid|clean|k0"
            lab, prd = z["labels"][sel], z["pred_class"][sel]
        print(
            f"-- 验证集完整视图类别计数（E5 seed42）: "
            f"true={np.bincount(lab, minlength=3).tolist()} "
            f"pred={np.bincount(prd, minlength=3).tolist()} n={int(sel.sum())}"
        )

    err = read_csv_rows(out / "tables" / "error_cases.csv")
    print(f"-- 错误分析 error_cases.csv：{len(err)} 条（按 |强度误差| 降序），前 3 条：")
    for r in err[:3]:
        print(
            f"   {r['view_id']:24s} id={r['sample_id']} true={r['true_class_name']} "
            f"pred={r['pred_class_name']} |err|={float(r['abs_error']):.3f} aT={float(r['alpha_T']):.3f}"
        )
    return data


# --------------------------------------------------------------------------------------
# H. 附件 3
# --------------------------------------------------------------------------------------
def section_h(out: Path) -> Dict[str, Any]:
    print()
    print("=" * 104)
    print("H. 附件 3 推理（final/inference_report.json）")
    print("=" * 104)
    p = out / "final" / "inference_report.json"
    if not p.exists():
        print("   缺少 inference_report.json")
        return {}
    ir = read_json(p)
    cnt = Counter(r["class_name"] for r in ir["predictions"])
    ints = [r["intensity"] for r in ir["predictions"]]
    print(f"n_rows={ir['n_rows']} checks_passed={ir['checks']['passed']} ckpt_sha256={ir['checkpoint_sha256']}")
    print(f"类别分布={dict(cnt)} 强度范围=[{min(ints):.4f},{max(ints):.4f}]")
    return {"n_rows": ir["n_rows"], "checks_passed": ir["checks"]["passed"], "class_distribution": dict(cnt)}


# --------------------------------------------------------------------------------------
# I. 数据核查
# --------------------------------------------------------------------------------------
def section_i(out: Path) -> Dict[str, Any]:
    print()
    print("=" * 104)
    print("I. 数据核查（data_audit.json）")
    print("=" * 104)
    p = out / "data_audit.json"
    if not p.exists():
        print("   缺少 data_audit.json")
        return {}
    da = read_json(p)
    print(f"passed={da['passed']} 问题数={len(da['problems'])}")
    for split in ("train", "valid", "test"):
        s = da["attachment2"]["splits"][split]
        print(
            f"   {split}: n={s['n_samples']} 类别={s['class_distribution']} "
            f"文本内容长度均值={s['text_validity']['content_len_mean']} "
            f"语音有效行均值={s['audio_validity']['per_sample_mean']} "
            f"视觉有效行均值={s['vision_validity']['per_sample_mean']} "
            f"corr(内容长度,语音)={s['corr_contentlen_audio']}"
        )
        print(f"      标签核对={s['label_check']}")
    print(f"   附件3: n_files={da['attachment3']['n_files']} 编号完整={da['attachment3']['index_ok']}")
    return {"passed": da["passed"], "n_problems": len(da["problems"])}


# --------------------------------------------------------------------------------------
# J. 环境
# --------------------------------------------------------------------------------------
def section_j(out: Path) -> Dict[str, Any]:
    print()
    print("=" * 104)
    print("J. 环境与预训练编码器")
    print("=" * 104)
    env_path = out / "env" / "environment.json"
    if env_path.exists():
        env = read_json(env_path)
        for k in (
            "platform", "python", "torch", "torch_cuda_build", "cuda_available", "cuda_device_name",
            "cuda_device_capability", "cuda_device_memory_gb", "cudnn", "transformers", "numpy", "nvidia_smi",
        ):
            print(f"   {k} = {env.get(k)}")
    freeze = out / "env" / "pip_freeze.txt"
    if freeze.exists():
        print(f"   pip_freeze 行数 = {len(freeze.read_text(encoding='utf-8').splitlines())}")
    man = out / "pretrained" / "bert_uncased_L-2_H-128_A-2" / "manifest.json"
    if man.exists():
        m = read_json(man)
        print(f"   文本编码器 repo_id = {m['repo_id']}")
        print(f"   文本编码器 revision = {m['revision']} 总大小 = {m['total_size_mb']} MB")
        for name, rec in m["files"].items():
            print(f"      {name}: {rec['size_bytes']} B sha256={rec['sha256']}")
    return {}


SECTIONS = {
    "A": ("逐 run 指标", section_a),
    "B": ("三种子聚合", section_b),
    "C": ("训练信息", section_c),
    "D": ("E4 教师", section_d),
    "E": ("E4 蒸馏搜索", section_e),
    "F": ("附件2 test", section_f),
    "G": ("规律/门控/错误分析", section_g),
    "H": ("附件3 推理", section_h),
    "I": ("数据核查", section_i),
    "J": ("环境", section_j),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="汇总 problem2 实验的全部真实证据")
    parser.add_argument("--config", default="configs/problem2.yaml")
    parser.add_argument(
        "--section", nargs="*", choices=sorted(SECTIONS), default=sorted(SECTIONS),
        help="只打印指定小节（默认全部）",
    )
    parser.add_argument("--json", default=None, help="把结构化结果导出到该 JSON 文件")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out = output_dir(cfg)
    print(f"证据来源目录: {out}")
    print(f"小节: {', '.join(f'{k}={SECTIONS[k][0]}' for k in sorted(set(args.section)))}")

    result: Dict[str, Any] = {}
    for key in sorted(set(args.section)):
        result[key] = SECTIONS[key][1](out)

    if args.json:
        path = save_json(args.json, result)
        print(f"\n结构化结果已写出: {path}")
    print(f"\n共检查 {len(result)} 个小节。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
