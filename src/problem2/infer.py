"""附件 3 最终推理：`python -m src.problem2.infer --config configs/problem2.yaml --checkpoint ... --input-dir ... --output ...`

- 按文件名编号 01..30 排序，每个文件恰好输出一行；
- 使用与训练/验证完全相同的文本编码器权重、训练集标准化参数与最终学生权重；
- 推理前检查字段/形状/NaN/Inf，推理后检查概率和、类别取值与强度范围；
- 不编造附件 3 的真实标签或准确率。
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from . import CLASS_NAMES, MODALITIES, __version__
from .common import (
    environment_report,
    load_config,
    output_dir,
    resolve,
    save_json,
    sha256_file,
    set_seed,
)
from .dataset import TextEmbedder, challenge_batch, load_challenge_file, load_normalizer
from .model import build_model_from_config
from .text_encoder import fetch_model

AUDIT_COLUMNS = [
    "file_name", "class_id", "class_name", "intensity",
    "p_negative", "p_neutral", "p_positive", "model_id",
]
SUBMISSION_COLUMNS = ["file_name", "class_id", "class_name", "intensity"]


def load_student(
    cfg: Dict[str, Any],
    experiment: str,
    checkpoint: Path,
    device,
    text_dim: Optional[int] = None,
    embedder: Optional[TextEmbedder] = None,
):
    """加载自训练层权重。

    checkpoint 中可能同时包含：
      * 融合/预测头参数（所有实验）；
      * 解冻实验（E5）中被微调的文本编码器层参数（bert.*）。
    BERT 的其余冻结层始终来自固定的预训练 revision。
    """
    from safetensors.torch import load_file

    if embedder is None:
        embedder = TextEmbedder(
            cfg["text_model"]["repo_id"], cfg["text_model"].get("revision"), cfg["text_model"]["local_dir"], device
        )
    text_dim = text_dim or embedder.hidden_size

    model = build_model_from_config(cfg, text_dim, experiment)
    state = load_file(str(checkpoint), device=str(device))
    head_state = {k: v for k, v in state.items() if not k.startswith("bert.")}
    bert_state = {k: v for k, v in state.items() if k.startswith("bert.")}
    missing, unexpected = model.load_state_dict(head_state, strict=False)
    if unexpected:
        raise RuntimeError(f"checkpoint 含未知参数: {unexpected[:5]}")
    if missing:
        raise RuntimeError(f"checkpoint 缺少参数: {missing[:5]}")
    if bert_state:
        # 保存时加了 "bert." 前缀以避免与融合头参数冲突；BertModel 本身的名字不带该前缀
        stripped = {k[len("bert.") :]: v for k, v in bert_state.items()}
        bert_missing, bert_unexpected = embedder.model.load_state_dict(stripped, strict=False)
        if bert_unexpected:
            raise RuntimeError(f"checkpoint 的文本编码器参数无法匹配: {bert_unexpected[:5]}")
        print(
            f"[infer] 已从 checkpoint 载入 {len(stripped)} 个文本编码器张量"
            f"（解冻实验；其余 {len(bert_missing)} 个张量使用固定预训练权重）"
        )
    model.to(device).eval()
    embedder.model.to(device).eval()
    return model


@torch.no_grad()
def predict_attachment3(
    cfg: Dict[str, Any],
    experiment: str,
    checkpoint: Path,
    input_dir: Path,
    device,
    embedder: Optional[TextEmbedder] = None,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    normalizer = load_normalizer(cfg)
    embedder = embedder or TextEmbedder(
        cfg["text_model"]["repo_id"], cfg["text_model"].get("revision"), cfg["text_model"]["local_dir"], device
    )
    model = load_student(cfg, experiment, checkpoint, device, text_dim=embedder.hidden_size, embedder=embedder)

    files = sorted(input_dir.glob("*.pkl"), key=lambda p: int(p.stem.split("_")[-1]))
    if len(files) != 30:
        print(f"[infer] 警告：附件 3 文件数为 {len(files)}，期望 30")

    rows: List[Dict[str, Any]] = []
    for path in files:
        sample = load_challenge_file(path)
        for field, expected in (("text_bert", (1, 3, 50)), ("audio", (1, 50, 74)), ("vision", (1, 50, 35))):
            arr = sample[field]
            if tuple(arr.shape) != expected:
                raise ValueError(f"{path.name}: {field} 形状 {tuple(arr.shape)} != {expected}")
            if arr.dtype.kind == "f" and not np.isfinite(arr).all():
                raise ValueError(f"{path.name}: {field} 含 NaN/Inf")
        batch = challenge_batch(sample, normalizer, device)
        text_emb = embedder.encode(batch["text_bert"], batch["text_observed"], batch_size=batch_size)
        out = model(
            text_emb,
            batch["audio"],
            batch["vision"],
            observed=batch["observed"],
            text_observed=batch["text_observed"],
        )
        probs = torch.softmax(out["class_logits"], dim=-1).cpu().numpy()[0]
        intensity = float(out["intensity"].cpu().numpy()[0])
        class_id = int(np.argmax(probs))
        weights = out["fusion_weights"].cpu().numpy()[0]
        rows.append(
            {
                "file_name": path.name,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "intensity": round(intensity, 6),
                "p_negative": round(float(probs[0]), 6),
                "p_neutral": round(float(probs[1]), 6),
                "p_positive": round(float(probs[2]), 6),
                "model_id": f"{experiment}|{checkpoint.name}",
                "fusion_weights": {m: round(float(w), 6) for m, w in zip(MODALITIES, weights)},
                "probs_sum": round(float(probs.sum()), 6),
            }
        )
    return rows


def validate_predictions(rows: List[Dict[str, Any]], input_dir: Path) -> Dict[str, Any]:
    problems: List[str] = []
    n_files = len(list(input_dir.glob("*.pkl")))
    if len(rows) != n_files:
        problems.append(f"输出行数 {len(rows)} != 输入文件数 {n_files}")
    names = [r["file_name"] for r in rows]
    if len(set(names)) != len(names):
        problems.append("存在重复文件名")
    expected_names = [p.name for p in sorted(input_dir.glob("*.pkl"), key=lambda p: int(p.stem.split("_")[-1]))]
    if names != expected_names:
        problems.append("输出顺序与 01..30 文件顺序不一致")
    for r in rows:
        if r["class_id"] not in (0, 1, 2):
            problems.append(f"{r['file_name']}: class_id={r['class_id']} 非法")
        if abs(r["probs_sum"] - 1.0) > 1e-4:
            problems.append(f"{r['file_name']}: 概率和 {r['probs_sum']} 偏离 1")
        if not (-3.0 - 1e-6 <= r["intensity"] <= 3.0 + 1e-6):
            problems.append(f"{r['file_name']}: 强度 {r['intensity']} 超出 [-3,3]")
        if abs(sum(r["fusion_weights"].values()) - 1.0) > 1e-4:
            problems.append(f"{r['file_name']}: 融合权重和不为 1")
    return {"n_rows": len(rows), "n_expected": n_files, "passed": not problems, "problems": problems}


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r[c] for c in columns})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="附件 3 最终推理")
    parser.add_argument("--config", default="configs/problem2.yaml")
    parser.add_argument("--experiment", default="E3")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output", default=None, help="审计用 CSV 路径")
    parser.add_argument("--submission-output", default=None, help="提交用 CSV 路径")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device(args.device or (cfg["training"].get("device", "cuda") if torch.cuda.is_available() else "cpu"))
    checkpoint = resolve(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"权重不存在: {checkpoint}")
    input_dir = resolve(args.input_dir or cfg["data"]["challenge_dir"])
    out_path = resolve(args.output) if args.output else output_dir(cfg) / "final" / "attachment3_predictions_audit.csv"
    sub_path = (
        resolve(args.submission_output) if args.submission_output else output_dir(cfg) / "final" / "attachment3_predictions.csv"
    )

    set_seed(0)
    t0 = time.time()
    rows = predict_attachment3(cfg, args.experiment, checkpoint, input_dir, device)
    checks = validate_predictions(rows, input_dir) if len(rows) == 30 else {
        "n_rows": len(rows), "passed": False, "problems": ["文件数不等于 30，未做完整校验"],
    }

    write_csv(out_path, rows, AUDIT_COLUMNS)
    write_csv(sub_path, rows, SUBMISSION_COLUMNS)

    report = {
        "script": "src.problem2.infer",
        "script_version": __version__,
        "experiment": args.experiment,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "input_dir": str(input_dir),
        "output_audit_csv": str(out_path),
        "output_submission_csv": str(sub_path),
        "n_rows": len(rows),
        "checks": checks,
        "predictions": rows,
        "elapsed_seconds": round(time.time() - t0, 2),
        "device": str(device),
        "environment": environment_report(),
        "text_model_manifest": fetch_model_manifest_only(cfg),
        "note": "附件 3 无标签，不计算任何指标；本文件仅记录推理过程与自检结果。",
    }
    save_json(output_dir(cfg) / "final" / "inference_report.json", report)

    print(f"[infer] 输出 {len(rows)} 行 -> {out_path}")
    print(f"[infer] 提交格式 -> {sub_path}")
    print(f"[infer] 自检通过: {checks['passed']}")
    for p in checks.get("problems", []):
        print(f"  - {p}")
    return 0 if checks["passed"] else 2


def fetch_model_manifest_only(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from .text_encoder import build_manifest

    local = resolve(cfg["text_model"]["local_dir"])
    if not (local / "config.json").exists():
        return {"error": "本地预训练目录不存在"}
    return build_manifest(cfg["text_model"]["repo_id"], str(cfg["text_model"].get("revision")), local)


if __name__ == "__main__":
    raise SystemExit(main())
