"""数据准备：`python -m src.problem2.prepare_data --config configs/problem2.yaml`

1. 只反序列化一次近 1 GiB 的 aligned_50.pkl，把每个划分转成独立 .npz 缓存；
2. 用训练集**可观测**的语音/视觉行拟合每维均值/标准差（std 下界保护），
   保存 normalizer.npz；验证/测试/附件 3 一律使用训练集统计量；
3. 生成固定验证缺失视图计划 validation_masks.json（所有模型共用）。
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from . import MODALITIES, __version__
from .common import (
    REPO_ROOT,
    ensure_parent,
    environment_report,
    load_config,
    output_dir,
    resolve,
    save_json,
    set_seed,
    sha256_file,
)
from .data import SPLITS, build_label_lookup, load_aligned
from .masks import (
    build_validation_plan,
    content_validity_features,
    content_validity_text,
)

STD_FLOOR = 1e-3
CACHE_VERSION = "1.0.0"


def content_validity_audio(audio_raw: np.ndarray) -> np.ndarray:
    return content_validity_features(audio_raw)


def content_validity_vision(vision_raw: np.ndarray) -> np.ndarray:
    return content_validity_features(vision_raw)


def fit_normalizer(train_audio: np.ndarray, train_vision: np.ndarray, valid_a: np.ndarray, valid_v: np.ndarray):
    stats: Dict[str, Any] = {}
    for name, raw, valid in (("audio", train_audio, valid_a), ("vision", train_vision, valid_v)):
        rows = raw[valid]  # (M, D) 只在可观测行上拟合
        if rows.size == 0:
            raise RuntimeError(f"{name}: 训练集没有可观测行，无法拟合标准化统计量")
        mean = rows.mean(axis=0).astype(np.float64)
        std = rows.std(axis=0).astype(np.float64)
        near_zero = int((std < STD_FLOOR).sum())
        std = np.maximum(std, STD_FLOOR)
        stats[name] = {
            "mean": mean,
            "std": std,
            "n_rows_used": int(rows.shape[0]),
            "n_dims": int(rows.shape[1]),
            "dims_with_std_below_floor": near_zero,
            "raw_min": float(rows.min()),
            "raw_max": float(rows.max()),
        }
    return stats


def apply_normalizer(raw: np.ndarray, valid: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """按行（时间槽）布尔掩码标准化：只在有效行上 (x-mean)/std，其余保持零。"""
    raw = np.asarray(raw, dtype=np.float32)
    mask_rows = np.asarray(valid, dtype=bool)  # (N, L)
    out = np.zeros_like(raw, dtype=np.float32)
    if mask_rows.any():
        normalized = (raw - mean.astype(np.float32)) / std.astype(np.float32)
        out[mask_rows] = normalized[mask_rows]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="问题 2 数据准备与缓存")
    parser.add_argument("--config", default="configs/problem2.yaml")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的缓存")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = output_dir(cfg)
    cache_dir = resolve(cfg["data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    manifest: Dict[str, Any] = {
        "script": "src.problem2.prepare_data",
        "script_version": __version__,
        "cache_version": CACHE_VERSION,
        "python": platform.python_version(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "std_floor": STD_FLOOR,
        "splits": {},
    }

    aligned_path = resolve(cfg["data"]["aligned_pkl"])
    manifest["source"] = {
        "aligned_pkl": str(aligned_path.relative_to(REPO_ROOT)) if aligned_path.is_relative_to(REPO_ROOT) else str(aligned_path),
        "aligned_sha256": sha256_file(aligned_path),
        "labels_xlsx": cfg["data"]["labels_xlsx"],
    }

    print("[prepare_data] 反序列化 aligned_50.pkl ...")
    t0 = time.time()
    raw = load_aligned(cfg)
    print(f"[prepare_data] 反序列化完成，耗时 {time.time() - t0:.1f}s")

    # ---------------- 有效性掩码（基于原始值） ----------------
    valid: Dict[str, Dict[str, np.ndarray]] = {}
    for split in SPLITS:
        d = raw[split]
        tb = np.round(np.asarray(d["text_bert"])).astype(np.int64)
        valid[split] = {
            "T": content_validity_text(tb[:, 0, :], tb[:, 1, :]),
            "A": content_validity_audio(np.asarray(d["audio"])),
            "V": content_validity_vision(np.asarray(d["vision"])),
        }
        if valid[split]["T"].shape[1] != tb.shape[2] or valid[split]["A"].shape[1] != 50:
            raise RuntimeError(f"{split}: 有效掩码形状异常")

    manifest["validity_summary"] = {
        split: {
            m: {
                "total_valid": int(valid[split][m].sum()),
                "mean_per_sample": round(float(valid[split][m].sum(axis=1).mean()), 3),
                "samples_with_no_valid": int((valid[split][m].sum(axis=1) == 0).sum()),
            }
            for m in MODALITIES
        }
        for split in SPLITS
    }

    # ---------------- 标准化统计量（仅训练集可观测行） ----------------
    print("[prepare_data] 拟合训练集标准化统计量 ...")
    stats = fit_normalizer(
        np.asarray(raw["train"]["audio"]),
        np.asarray(raw["train"]["vision"]),
        valid["train"]["A"],
        valid["train"]["V"],
    )
    normalizer_path = cache_dir / "normalizer.npz"
    np.savez_compressed(
        normalizer_path,
        audio_mean=stats["audio"]["mean"],
        audio_std=stats["audio"]["std"],
        vision_mean=stats["vision"]["mean"],
        vision_std=stats["vision"]["std"],
    )
    manifest["normalizer"] = {
        name: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in rec.items()}
        for name, rec in stats.items()
    }
    print(f"[prepare_data] 已写出 {normalizer_path}")
    print(
        "[prepare_data] 语音 std 下界维度 {} / 视觉 std 下界维度 {}".format(
            stats["audio"]["dims_with_std_below_floor"], stats["vision"]["dims_with_std_below_floor"]
        )
    )

    # ---------------- 写出各划分缓存 ----------------
    for split in SPLITS:
        d = raw[split]
        path = cache_dir / f"{split}.npz"
        if path.exists() and not args.force:
            print(f"[prepare_data] 已存在，跳过: {path}")
        else:
            tb = np.round(np.asarray(d["text_bert"])).astype(np.int64)
            audio = apply_normalizer(np.asarray(d["audio"]), valid[split]["A"], stats["audio"]["mean"], stats["audio"]["std"])
            vision = apply_normalizer(np.asarray(d["vision"]), valid[split]["V"], stats["vision"]["mean"], stats["vision"]["std"])
            cl = np.round(np.asarray(d["classification_labels"])).astype(np.int64)
            rl = np.asarray(d["regression_labels"]).astype(np.float32)
            ids = np.asarray([str(x) for x in d["id"]])
            raw_text = np.asarray([str(x) for x in d["raw_text"]])
            np.savez_compressed(
                path,
                text_bert=tb,
                audio=audio,
                vision=vision,
                valid_T=valid[split]["T"].astype(np.uint8),
                valid_A=valid[split]["A"].astype(np.uint8),
                valid_V=valid[split]["V"].astype(np.uint8),
                classification_labels=cl,
                regression_labels=rl,
                ids=ids,
                raw_text=raw_text,
            )
            print(f"[prepare_data] 已写出 {path} ({path.stat().st_size / 2**20:.1f} MB)")

        manifest["splits"][split] = {
            "n_samples": int(len(raw[split]["id"])),
            "cache": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
            "cache_sha256": sha256_file(path) if path.stat().st_size < 2**28 else "skipped_large",
            "cache_size_mb": round(path.stat().st_size / 2**20, 3),
            "class_distribution": {
                str(i): int((np.round(np.asarray(raw[split]["classification_labels"])).astype(int) == i).sum())
                for i in range(3)
            },
        }

    # ---------------- E4 教师用的预计算 text（仅训练阶段辅助，不进入最终学生） ----------------
    teacher_cache = cache_dir / "teacher_text.npz"
    if teacher_cache.exists() and not args.force:
        print(f"[prepare_data] 已存在，跳过: {teacher_cache}")
    else:
        arrays = {}
        for split in SPLITS:
            if "text" not in raw[split]:
                print(f"[prepare_data] {split} 没有预计算 text 字段，跳过教师缓存")
                arrays = {}
                break
            arrays[split] = np.asarray(raw[split]["text"], dtype=np.float32)
        if arrays:
            np.savez_compressed(
                teacher_cache,
                **{f"text_{s}": a for s, a in arrays.items()},
                att_T_train=np.asarray(raw["train"]["text_bert"])[:, 1, :].astype(np.uint8),
                att_T_valid=np.asarray(raw["valid"]["text_bert"])[:, 1, :].astype(np.uint8),
            )
            print(
                f"[prepare_data] 已写出教师文本缓存 {teacher_cache} "
                f"({teacher_cache.stat().st_size / 2**20:.1f} MB, text_dim={arrays['train'].shape[-1]})"
            )
            manifest["teacher_text_cache"] = {
                "path": str(teacher_cache),
                "text_dim": int(arrays["train"].shape[-1]),
                "fields": sorted(arrays),
                "note": "附件 2 预计算的 text，仅 E4 教师训练使用；主模型输入只允许 text_bert",
                "validity_rule": "v_T = text_bert attention_mask==1（实测 first_n 与 bert_mask 池化给出相同探针结果）",
            }

    # ---------------- 固定验证缺失视图 ----------------
    vcfg = cfg.get("validation", {})
    seed = int(cfg.get("evaluation", {}).get("bootstrap_seed", 20240924))
    n_valid = int(valid["valid"]["T"].shape[0])
    print("[prepare_data] 生成固定验证缺失视图 ...")
    plan = build_validation_plan(
        valid["valid"],
        n_valid,
        seed=seed,
        views_per_condition=int(vcfg.get("views_per_condition", 3)),
        dual_pairs=vcfg.get("dual_modality_pairs"),
        dual_views=int(vcfg.get("dual_views_per_condition", 3)),
    )
    plan["cache_version"] = CACHE_VERSION
    plan["normalizer"] = str(normalizer_path)
    plan_path = save_json(cache_dir / "validation_masks.json", plan)
    manifest["validation_views"] = {
        "path": str(plan_path),
        "n_views": len(plan["views"]),
        "n_clean_views": sum(1 for v in plan["views"] if v["kind"] == "clean"),
        "n_missing_views": sum(1 for v in plan["views"] if v["kind"] != "clean"),
        "n_conditions": len({v["condition"] for v in plan["views"]}),
        "seed": seed,
    }
    print(f"[prepare_data] 已写出 {plan_path}（{len(plan['views'])} 个视图）")

    # 只读有效性掩码，便于后续快速校验
    np.savez_compressed(
        cache_dir / "validity.npz",
        **{f"{split}_{m}": valid[split][m].astype(np.uint8) for split in SPLITS for m in MODALITIES},
    )

    # label 连接与 Excel 的复核结果（供报告引用）
    lookup = build_label_lookup(cfg)
    manifest["excel_rows"] = len(lookup)
    manifest["environment"] = environment_report()
    manifest["elapsed_seconds"] = round(time.time() - t_start, 2)
    save_json(cache_dir / "prepare_manifest.json", manifest)
    print(f"[prepare_data] 完成，总耗时 {manifest['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
