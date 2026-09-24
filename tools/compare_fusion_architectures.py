"""Compare legacy gate and cross-attention runs on matched problem2 seeds/views."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = (42, 52, 62)
EXPERIMENTS = ("E3", "E4", "E5")
METRICS = {
    "valid": ("clean_macro_f1", "primary_macro_f1_missing36",
              "clean_accuracy", "clean_mae", "single_missing_mae"),
    "test": ("clean_macro_f1", "clean_mae"),
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def evaluation_path(old_root: Path, new_root: Path, arch: str, exp: str, seed: int, split: str):
    if arch == "gate" and split == "test" and seed != 42:
        return new_root / "legacy_gate_recheck" / f"{exp}_seed{seed}_eval_test.json"
    root = old_root if arch == "gate" else new_root
    return root / "runs" / f"{exp}_seed{seed}" / f"eval_{split}.json"


def summarize_metric(old_root, new_root, exp, split, key):
    values = {}
    for arch in ("gate", "cross"):
        values[arch] = np.asarray([
            read_json(evaluation_path(old_root, new_root, arch, exp, seed, split))["overall"][key]
            for seed in SEEDS
        ], dtype=np.float64)
    return {
        "gate_mean": float(values["gate"].mean()),
        "gate_sd": float(values["gate"].std(ddof=1)),
        "cross_mean": float(values["cross"].mean()),
        "cross_sd": float(values["cross"].std(ddof=1)),
        "paired_delta": float((values["cross"] - values["gate"]).mean()),
        "gate_per_seed": values["gate"].tolist(),
        "cross_per_seed": values["cross"].tolist(),
        "delta_per_seed": (values["cross"] - values["gate"]).tolist(),
    }


def bootstrap_sample_clusters(values: np.ndarray, rng: np.random.Generator, n_boot: int):
    # Each validation sample is a cluster; its repeated views and three seeds stay together.
    n = values.size
    indexes = rng.integers(0, n, size=(n_boot, n))
    draws = values[indexes].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"delta": float(values.mean()), "ci95": [float(lo), float(hi)],
            "n_samples": int(n), "n_boot": int(n_boot)}


def paired_predictions(old_root, new_root, exp, n_boot):
    rows = {subset: {"accuracy": [], "mae": []} for subset in ("clean", "missing36")}
    for seed in SEEDS:
        old_path = old_root / "runs" / f"{exp}_seed{seed}" / "eval_valid_samples.npz"
        new_path = new_root / "runs" / f"{exp}_seed{seed}" / "eval_valid_samples.npz"
        with np.load(old_path) as old, np.load(new_path) as new:
            for key in ("view_id", "indices", "ids", "labels", "targets"):
                if not np.array_equal(old[key], new[key]):
                    raise ValueError(f"{exp} seed {seed}: unmatched {key}")
            n = int(np.unique(old["indices"]).size)
            if n * 129 != old["indices"].size:
                raise ValueError(f"{exp} seed {seed}: unexpected number of views")
            view_ids = old["view_id"].reshape(-1, n)[:, 0]
            if not np.all(old["view_id"].reshape(-1, n) == view_ids[:, None]):
                raise ValueError(f"{exp} seed {seed}: views are not contiguous")
            labels = old["labels"].reshape(-1, n)
            targets = old["targets"].reshape(-1, n)
            old_class = old["pred_class"].reshape(-1, n)
            new_class = new["pred_class"].reshape(-1, n)
            old_intensity = old["intensity"].reshape(-1, n)
            new_intensity = new["intensity"].reshape(-1, n)
            selected = {
                "clean": np.flatnonzero(view_ids == "valid|clean|k0"),
                "missing36": np.asarray([
                    i for i, view in enumerate(view_ids)
                    if view.startswith(("valid|T|", "valid|A|", "valid|V|"))
                ], dtype=int),
            }
            if len(selected["clean"]) != 1 or len(selected["missing36"]) != 108:
                raise ValueError(f"{exp} seed {seed}: unexpected clean/missing view count")
            for subset, chosen in selected.items():
                accuracy = (
                    (new_class[chosen] == labels[chosen]).astype(float)
                    - (old_class[chosen] == labels[chosen]).astype(float)
                ).mean(axis=0)
                mae = (
                    np.abs(new_intensity[chosen] - targets[chosen])
                    - np.abs(old_intensity[chosen] - targets[chosen])
                ).mean(axis=0)
                rows[subset]["accuracy"].append(accuracy)
                rows[subset]["mae"].append(mae)
    result = {}
    for subset, metrics in rows.items():
        result[subset] = {}
        for metric, seed_rows in metrics.items():
            rng = np.random.default_rng(20240924 + len(subset) + len(metric))
            result[subset][metric] = bootstrap_sample_clusters(
                np.stack(seed_rows).mean(axis=0), rng, n_boot
            )
    return result


def condition_deltas(old_root, new_root, exp):
    groups = {"T": [], "A": [], "V": [], "T70": []}
    for seed in SEEDS:
        old = read_json(evaluation_path(old_root, new_root, "gate", exp, seed, "valid"))["condition_summary"]
        new = read_json(evaluation_path(old_root, new_root, "cross", exp, seed, "valid"))["condition_summary"]
        if set(old) != set(new):
            raise ValueError(f"{exp} seed {seed}: unmatched validation conditions")
        for condition in old:
            for group in ("T", "A", "V"):
                if condition.startswith(group + "-"):
                    groups[group].append(float(new[condition]["macro_f1"] - old[condition]["macro_f1"]))
                    if group == "T" and condition.endswith("|70"):
                        groups["T70"].append(float(new[condition]["macro_f1"] - old[condition]["macro_f1"]))
    return {
        group: {"mean_delta": float(np.mean(deltas)),
                "wins": int(np.count_nonzero(np.asarray(deltas) > 0)),
                "n": len(deltas)}
        for group, deltas in groups.items()
    }


def check_training_config(old_root, new_root, exp):
    keys = ("batch_size", "max_epochs", "learning_rate", "weight_decay",
            "regression_weight", "grad_clip", "class_weighted_loss",
            "masker", "unfrozen_text_encoder", "n_train", "n_valid")
    for seed in SEEDS:
        old = read_json(old_root / "runs" / f"{exp}_seed{seed}" / "run_info.json")
        new = read_json(new_root / "runs" / f"{exp}_seed{seed}" / "run_info.json")
        for key in keys:
            if old[key] != new[key]:
                raise ValueError(f"{exp} seed {seed}: changed training setting {key}")
        if old["smoke"] or new["smoke"]:
            raise ValueError(f"{exp} seed {seed}: smoke run in comparison")
        if exp == "E4":
            for key in ("temperature", "kl_weight", "intensity_consistency_weight"):
                if old["distillation"][key] != new["distillation"][key]:
                    raise ValueError(f"E4 seed {seed}: changed distillation setting {key}")


def format_mean(row, arch):
    return f"{row[arch + '_mean']:.4f} +/- {row[arch + '_sd']:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-root", type=Path, default=Path("outputs/problem2"))
    parser.add_argument("--cross-root", type=Path, default=Path("outputs/problem2_cross_attention"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    result = {"method": "matched seeds, data, masks and training settings",
              "seeds": list(SEEDS), "experiments": {}, "bootstrap_note":
              "95% validation-sample cluster bootstrap; seeds and selected checkpoints fixed."}
    for exp in EXPERIMENTS:
        check_training_config(args.gate_root, args.cross_root, exp)
        result["experiments"][exp] = {
            "valid": {key: summarize_metric(args.gate_root, args.cross_root, exp, "valid", key)
                      for key in METRICS["valid"]},
            "test": {key: summarize_metric(args.gate_root, args.cross_root, exp, "test", key)
                     for key in METRICS["test"]},
            "paired_validation": paired_predictions(args.gate_root, args.cross_root, exp, args.bootstrap),
            "condition_f1": condition_deltas(args.gate_root, args.cross_root, exp),
        }
    args.cross_root.mkdir(parents=True, exist_ok=True)
    json_path = args.cross_root / "comparison_with_gate.json"
    md_path = args.cross_root / "comparison_with_gate.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Gate versus cross-attention fusion",
        "",
        "Matched E3/E4/E5 training runs, seeds 42/52/62, and fixed validation views.",
        "Positive delta means cross-attention is higher. Lower MAE is better.",
        "",
        "| Experiment | Split and metric | Gate mean +/- SD | Cross mean +/- SD | Cross - gate |",
        "|---|---|---:|---:|---:|",
    ]
    table_metrics = (
        ("valid", "clean_macro_f1"),
        ("valid", "primary_macro_f1_missing36"),
        ("valid", "clean_mae"),
        ("valid", "single_missing_mae"),
        ("test", "clean_macro_f1"),
        ("test", "clean_mae"),
    )
    for exp in EXPERIMENTS:
        item = result["experiments"][exp]
        for split, key in table_metrics:
            row = item[split][key]
            lines.append(
                f"| {exp} | {split} {key} | {format_mean(row, 'gate')} | "
                f"{format_mean(row, 'cross')} | {row['paired_delta']:+.4f} |"
            )
    lines.extend([
        "",
        "## Paired validation differences",
        "",
        "The interval resamples 728 validation sample IDs; all repeated views and three",
        "fixed seeds for each sample stay in the same cluster. Checkpoints were selected",
        "on this validation set, so these intervals are descriptive rather than an",
        "independent generalization claim.",
        "",
        "| Experiment | Subset | Metric | Cross - gate | 95% cluster interval |",
        "|---|---|---|---:|---:|",
    ])
    for exp in EXPERIMENTS:
        for subset in ("clean", "missing36"):
            for metric in ("accuracy", "mae"):
                row = result["experiments"][exp]["paired_validation"][subset][metric]
                lines.append(
                    f"| {exp} | {subset} | {metric} | {row['delta']:+.4f} | "
                    f"[{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}] |"
                )
    lines.extend([
        "",
        "## Single-modality missing conditions",
        "",
        "Values are macro-F1 changes averaged across three seeds and 12 conditions",
        "per modality. T70 contains the three text positions at 70% missing.",
        "",
        "| Experiment | T | A | V | T70 |",
        "|---|---:|---:|---:|---:|",
    ])
    for exp in EXPERIMENTS:
        groups = result["experiments"][exp]["condition_f1"]
        lines.append("| " + exp + " | " + " | ".join(
            f"{groups[group]['mean_delta']:+.4f} ({groups[group]['wins']}/{groups[group]['n']} wins)"
            for group in ("T", "A", "V", "T70")
        ) + " |")
    lines.extend([
        "",
        "## Conclusion",
        "",
        "The new fusion does not improve the primary 36-condition validation macro-F1.",
        "E5 falls for every seed and its test MAE also rises. E4 has a small test",
        "macro-F1 gain but falls on validation and worsens test MAE; this is not",
        "enough evidence to replace the gate baseline.",
        "",
        "Source files: each run's config.json, run_info.json, eval_valid.json,",
        "eval_valid_samples.npz and eval_test.json. The legacy seed-52/62 test",
        "results are under legacy_gate_recheck/; historical artifacts were not overwritten.",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
