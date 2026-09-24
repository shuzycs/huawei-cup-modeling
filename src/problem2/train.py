"""训练入口：`python -m src.problem2.train --config configs/problem2.yaml --experiment E3 --seed 42`

要点
----
* 冻结 BERT：`eval()` + `no_grad()`；文本缺失在进入 BERT **之前**应用到 token IDs / attention mask；
* 主网络 AdamW，学习率 1e-3、权重衰减 1e-2、dropout 0.2、梯度裁剪 1.0；
* 早停依据验证集（完整输入视图）macro-F1，耐心 8 个 epoch，最多 40 epoch；
* 训练缺失由确定性 SpanMasker(seed, epoch, sample_index) 生成；E0/E1 不做缺失增强；
* 每个 run 独立目录，保存配置副本、最佳权重、逐 epoch 指标与耗时。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from . import MODALITIES, __version__
from .common import (
    REPO_ROOT,
    environment_report,
    load_config,
    output_dir,
    resolve,
    save_json,
    set_seed,
    sha256_file,
)
from .dataset import (
    SplitData,
    TextEmbedder,
    iter_batches,
    load_split,
    load_teacher_text,
    make_batch,
)
from .masks import SpanMasker, build_validation_plan, view_observed_from_plan
from .metrics import compute_metrics
from .model import (
    build_model_from_config,
    build_teacher_from_config,
    distillation_loss,
    inverse_frequency_class_weights,
    loss_function,
)

TRAIN_VIEW_CONDITIONS = [("T", "middle", 0.5), ("A", "middle", 0.5), ("V", "middle", 0.5)]
EXPERIMENTS = ("E0", "E1", "E2", "E3", "E4", "E5")
# E5 = E3 + 解冻文本编码器后两层（手册 1.1 明确要求的独立补充实验，不是偷偷改动基线）
UNFROZEN_EXPERIMENTS = ("E5",)
# E4 = E3 + 完整输入教师蒸馏（手册 3.3 可选扩展；教师可用附件 2 预计算 text）
DISTILLED_EXPERIMENTS = ("E4",)
TEACHER_ARCH = "mosei_text768"


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """把 --set a.b=value 应用到配置（value 用 yaml.safe_load 解析类型）。"""
    import yaml

    applied: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"格式应为 KEY=VALUE，收到 {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        parts = key.split(".")
        node: Any = cfg
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise KeyError(f"配置路径不存在: {key}")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"配置键不存在: {key}")
        try:
            value = yaml.safe_load(raw)
        except Exception:  # noqa: BLE001
            value = raw
        node[parts[-1]] = value
        applied[key] = value
    cfg.setdefault("training", {})["_overrides"] = [
        f"{k}={v}" for k, v in applied.items()
    ]
    return applied


class CosineSchedule:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, base_lr: float, min_lr_ratio: float = 0.05):
        self.opt = optimizer
        self.warmup = max(1, warmup_steps)
        self.total = max(self.warmup + 1, total_steps)
        self.base_lr = base_lr
        self.min_lr_ratio = min_lr_ratio
        self.step_num = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self) -> float:
        self.step_num += 1
        if self.step_num <= self.warmup:
            factor = self.step_num / self.warmup
        else:
            prog = (self.step_num - self.warmup) / max(1, self.total - self.warmup)
            factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * min(1.0, prog))
            )
        for g, base in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = base * factor
        return self.opt.param_groups[0]["lr"]


def uses_mask_augmentation(experiment: str) -> bool:
    return experiment in ("E2", "E3", "E4", "E5")


@torch.no_grad()
def teacher_logits(
    teacher,
    batch: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """教师前向：输入附件 2 的预计算 text + 完整音视频。"""
    teacher.eval()
    out = teacher(
        batch["teacher_text"],
        batch["audio"],
        batch["vision"],
        text_observed=batch["teacher_text_observed"],
    )
    return out["class_logits"], out["intensity"]


def load_teacher(cfg: Dict[str, Any], checkpoint: Path, text_dim: int, device):
    from safetensors.torch import load_file

    from .model import build_teacher_from_config

    teacher = build_teacher_from_config(cfg, text_dim=text_dim)
    state = load_file(str(checkpoint), device=str(device))
    missing, unexpected = teacher.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"教师权重不匹配：缺少 {missing[:5]}，多余 {unexpected[:5]}（{checkpoint}）"
        )
    return teacher.to(device).eval()


def run_teacher(
    cfg: Dict[str, Any],
    seed: int,
    max_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    smoke: bool = False,
    log: Optional[Any] = None,
) -> Dict[str, Any]:
    """训练 E4 教师：完整输入（附件 2 预计算 text + 音视频），结构与 E3 相同。"""
    import logging

    log = log or logging.getLogger("problem2")
    device = torch.device(cfg["training"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    tcfg = cfg["training"]
    dcfg = cfg.get("distillation", {})
    epochs = int(max_epochs if max_epochs is not None else dcfg.get("teacher_max_epochs", tcfg.get("max_epochs", 40)))
    if epochs < 1:
        raise ValueError(f"max_epochs 必须 >= 1（收到 {epochs}）")
    bsz = int(batch_size if batch_size is not None else tcfg.get("batch_size", 32))
    patience = int(dcfg.get("teacher_patience", tcfg.get("patience", 8)))
    lr = float(dcfg.get("teacher_learning_rate", tcfg.get("learning_rate", 1e-3)))
    wd = float(tcfg.get("weight_decay", 0.01))
    lam = float(tcfg.get("regression_weight", 0.5))
    grad_clip = float(tcfg.get("grad_clip", 1.0))

    set_seed(seed)
    train = load_split(cfg, "train")
    valid = load_split(cfg, "valid")
    teacher_cache = load_teacher_text(cfg)
    text_dim = teacher_cache.text_dim
    if smoke:
        keep = min(256, train.n)
        train = _subset(train, np.arange(keep))
        valid = _subset(valid, np.arange(min(128, valid.n)))
        epochs = min(epochs, 2)
        log.info("[TEACHER SMOKE] %d 训练样本 / %d 验证样本，%d epoch", train.n, valid.n, epochs)

    teacher = build_teacher_from_config(cfg, text_dim=text_dim).to(device)
    n_params = {
        "total": int(sum(p.numel() for p in teacher.parameters())),
        "trainable": int(sum(p.numel() for p in teacher.parameters() if p.requires_grad)),
    }
    class_weights = None
    if bool(tcfg.get("class_weighted_loss", False)):
        class_weights = inverse_frequency_class_weights(torch.from_numpy(train.labels)).to(device)
    log.info(
        "[教师 seed=%d] device=%s text_dim=%d 参数 total=%d trainable=%d class_weighted=%s",
        seed, device, text_dim, n_params["total"], n_params["trainable"], class_weights is not None,
    )

    save_teacher = lambda path: _save_checkpoint(teacher, path)  # noqa: E731
    run_dir = output_dir(cfg) / "teacher" / f"{TEACHER_ARCH}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", jsonable(copy.deepcopy(cfg)))
    save_json(
        run_dir / "run_info.json",
        {
            "role": "teacher",
            "arch": TEACHER_ARCH,
            "seed": seed,
            "text_source": "附件2 预计算 text (N,50,768)，注意力掩码取自 text_bert",
            "text_dim": text_dim,
            "batch_size": bsz,
            "max_epochs": epochs,
            "learning_rate": lr,
            "weight_decay": wd,
            "regression_weight": lam,
            "grad_clip": grad_clip,
            "class_weighted_loss": class_weights is not None,
            "model_params": n_params,
            "device": str(device),
            "n_train": train.n,
            "n_valid": valid.n,
            "environment": environment_report(),
            "smoke": smoke,
            "script_version": __version__,
        },
    )

    optimizer = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, math.ceil(train.n / bsz))
    schedule = CosineSchedule(
        optimizer, warmup_steps=max(1, steps_per_epoch), total_steps=steps_per_epoch * epochs,
        base_lr=lr, min_lr_ratio=float(cfg["text_model"].get("min_lr_ratio", 0.05)),
    )

    history: List[Dict[str, Any]] = []
    best = {"epoch": -1, "macro_f1": -1.0, "mae": float("inf")}
    bad = 0
    t_start = time.time()

    for epoch in range(epochs):
        teacher.train()
        tot = {"loss": 0.0, "ce": 0.0, "huber": 0.0}
        nb = 0
        for batch_idx in iter_batches(train.n, bsz, shuffle=True, seed=seed * 1000 + epoch):
            batch = make_batch(train, batch_idx, observed=None, device=device, with_labels=True)
            teacher_text = {
                "features": teacher_cache.get("train", batch_idx),
                "observed": teacher_cache.attention("train", batch_idx),
            }
            batch.update(
                {
                    "teacher_text": torch.from_numpy(teacher_text["features"]).to(device),
                    "teacher_text_observed": torch.from_numpy(teacher_text["observed"]).to(device),
                }
            )
            out = teacher(
                batch["teacher_text"], batch["audio"], batch["vision"],
                text_observed=batch["teacher_text_observed"],
            )
            losses = loss_function(
                out["class_logits"], out["intensity"], batch["labels"], batch["targets"],
                regression_weight=lam, class_weights=class_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            gn = torch.nn.utils.clip_grad_norm_(teacher.parameters(), grad_clip)
            if not torch.isfinite(gn):
                raise RuntimeError(f"教师梯度出现非有限值（epoch {epoch}）")
            optimizer.step()
            schedule.step()
            for k in tot:
                tot[k] += float(losses[k])
            nb += 1
        for k in tot:
            tot[k] /= max(1, nb)

        val = evaluate_clean(teacher, valid, teacher_cache, bsz * 4, device)
        rec = {
            "epoch": epoch,
            "train_loss": tot["loss"],
            "train_ce": tot["ce"],
            "train_huber": tot["huber"],
            "valid_clean_macro_f1": val["macro_f1"],
            "valid_clean_mae": val["mae"],
            "elapsed_seconds": round(time.time() - t_start, 2),
        }
        history.append(rec)
        log.info(
            "[教师 seed=%d] epoch %d/%d loss=%.4f ce=%.4f huber=%.4f | valid F1=%.4f MAE=%.4f | %.1fs",
            seed, epoch, epochs - 1, rec["train_loss"], rec["train_ce"], rec["train_huber"],
            val["macro_f1"], val["mae"], rec["elapsed_seconds"],
        )
        if val["macro_f1"] > best["macro_f1"] + 1e-6:
            best = {"epoch": epoch, "macro_f1": val["macro_f1"], "mae": val["mae"]}
            save_teacher(run_dir / "best.safetensors")
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log.info("[教师 seed=%d] 第 %d epoch 早停", seed, epoch)
                break

    save_teacher(run_dir / "last.safetensors")
    result = {
        "role": "teacher",
        "arch": TEACHER_ARCH,
        "seed": seed,
        "best": best,
        "epochs_run": len(history),
        "history": history,
        "elapsed_seconds": round(time.time() - t_start, 2),
        "checkpoint": str(run_dir / "best.safetensors"),
        "checkpoint_sha256": sha256_file(run_dir / "best.safetensors"),
        "model_params": n_params,
    }
    save_json(run_dir / "history.json", history)
    save_json(run_dir / "result.json", result)
    log.info("[教师 seed=%d] 完成：best epoch=%d valid macro-F1=%.4f", seed, best["epoch"], best["macro_f1"])
    return result


@torch.no_grad()
def evaluate_clean(model, data: SplitData, teacher_cache, batch_size: int, device, split: str = "valid") -> Dict[str, Any]:
    """完整输入（教师口径）下的验证指标。"""
    model.eval()
    preds, intens, idxs = [], [], []
    for batch_idx in iter_batches(data.n, batch_size, shuffle=False):
        batch = make_batch(data, batch_idx, observed=None, device=device, with_labels=True)
        batch["teacher_text"] = torch.from_numpy(teacher_cache.get(split, batch_idx)).to(device)
        batch["teacher_text_observed"] = torch.from_numpy(teacher_cache.attention(split, batch_idx)).to(device)
        out = model(
            batch["teacher_text"], batch["audio"], batch["vision"],
            text_observed=batch["teacher_text_observed"],
        )
        preds.append(out["class_logits"].argmax(dim=-1).cpu().numpy())
        intens.append(out["intensity"].cpu().numpy())
        idxs.append(batch_idx)
    idx = np.concatenate(idxs)
    return compute_metrics(
        data.labels[idx], np.concatenate(preds), data.targets[idx], np.concatenate(intens)
    )


def trainable_parameters(model) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def masked_views_for_batch(
    masker: SpanMasker,
    train: SplitData,
    idx: np.ndarray,
    epoch: int,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """为一批样本生成观测掩码（就地覆盖到该批的行）。"""
    observed = {m: np.zeros((len(idx), train.length), dtype=bool) for m in MODALITIES}
    metas: List[Dict[str, Any]] = []
    for j, sample_index in enumerate(idx):
        valid_i = {m: train.valid[m][sample_index] for m in MODALITIES}
        obs_i, meta = masker.sample(valid_i, epoch, int(sample_index))
        for m in MODALITIES:
            observed[m][j] = obs_i[m]
        metas.append(meta)
    return observed, metas


@torch.no_grad()
def quick_validate(
    model,
    embedder: TextEmbedder,
    valid: SplitData,
    views: List[Dict[str, Any]],
    batch_size: int,
    device,
) -> Dict[str, Any]:
    """每 epoch 的验证：完整视图 + 若干固定缺失视图。"""
    model.eval()
    per_view = []
    for view in views:
        observed = view["_observed"]
        preds, intens, idxs = [], [], []
        for batch_idx in iter_batches(valid.n, batch_size, shuffle=False):
            batch = make_batch(valid, batch_idx, observed=observed, device=device, with_labels=True)
            text_emb = embedder.encode(batch["text_bert"], batch["text_observed"], batch_size=batch_size)
            out = model(
                text_emb,
                batch["audio"],
                batch["vision"],
                observed=batch["observed"],
                text_observed=batch["text_observed"],
            )
            preds.append(out["class_logits"].argmax(dim=-1).cpu().numpy())
            intens.append(out["intensity"].cpu().numpy())
            idxs.append(batch_idx)
        idx = np.concatenate(idxs)
        m = compute_metrics(
            valid.labels[idx], np.concatenate(preds), valid.targets[idx], np.concatenate(intens)
        )
        per_view.append({"view_id": view["view_id"], "condition": view["condition"], "metrics": m})
    by_cond: Dict[str, List[Dict[str, Any]]] = {}
    for rec in per_view:
        # 每个视图单独成组：view_id 里已经含条件与比例（例如 valid|T|middle|50|k0），
        # 这样 clean 视图不会被稀释，早期停止依据才是真正的"完整输入 macro-F1"。
        by_cond.setdefault(rec["view_id"], []).append(rec["metrics"])
    agg = {}
    for cond, ms in by_cond.items():
        agg[cond] = {
            k: float(np.mean([m[k] for m in ms if m.get(k) is not None]))
            for k in ("macro_f1", "accuracy", "mae", "pearson")
            if any(m.get(k) is not None for m in ms)
        }
    missing = [v for k, v in agg.items() if not k.startswith("valid|clean")]
    clean = agg.get("valid|clean|k0") or next((v for k, v in agg.items() if k.startswith("valid|clean")), {})
    summary = {
        "clean_macro_f1": clean.get("macro_f1"),
        "clean_mae": clean.get("mae"),
        "missing_macro_f1_mean": float(np.mean([m["macro_f1"] for m in missing])) if missing else None,
        "missing_mae_mean": float(np.mean([m["mae"] for m in missing])) if missing else None,
    }
    return {"per_view": per_view, "condition": agg, "summary": summary}


def run(
    cfg: Dict[str, Any],
    experiment: str,
    seed: int,
    max_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    smoke: bool = False,
    log: Optional[Any] = None,
    teacher_checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    import logging

    log = log or logging.getLogger("problem2")
    if experiment not in EXPERIMENTS:
        raise ValueError(f"未知实验编号 {experiment}，可选 {EXPERIMENTS}")
    distill = experiment in DISTILLED_EXPERIMENTS

    device = torch.device(cfg["training"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    tcfg = cfg["training"]
    epochs = int(max_epochs if max_epochs is not None else tcfg.get("max_epochs", 40))
    if epochs < 1:
        raise ValueError(f"max_epochs 必须 >= 1（收到 {epochs}）；本脚本不会用 0 个 epoch 生成可提交权重")
    bsz = int(batch_size if batch_size is not None else tcfg.get("batch_size", 32))
    patience = int(tcfg.get("patience", 8))
    lr = float(tcfg.get("learning_rate", 1e-3))
    wd = float(tcfg.get("weight_decay", 0.01))
    lam = float(tcfg.get("regression_weight", 0.5))
    grad_clip = float(tcfg.get("grad_clip", 1.0))

    set_seed(seed)
    train = load_split(cfg, "train")
    valid = load_split(cfg, "valid")
    if smoke:
        keep = min(256, train.n)
        train = _subset(train, np.arange(keep))
        valid = _subset(valid, np.arange(min(128, valid.n)))
        epochs = min(epochs, 2)
        log.info("[SMOKE] 使用 %d 训练样本 / %d 验证样本，%d epoch", train.n, valid.n, epochs)

    embedder = TextEmbedder(
        cfg["text_model"]["repo_id"], cfg["text_model"].get("revision"), cfg["text_model"]["local_dir"], device
    )
    unfrozen = experiment in UNFROZEN_EXPERIMENTS
    text_lr = float(cfg["text_model"].get("unfrozen_learning_rate", 1e-4))
    if unfrozen:
        n_unfrozen = embedder.unfreeze_top_layers(int(cfg["text_model"].get("unfreeze_top_layers", 2)))
        log.info("[%s seed=%d] 解冻文本编码器后 %d 层（文本学习率 %g）", experiment, seed, n_unfrozen, text_lr)
    elif not cfg["text_model"].get("frozen", True):
        raise NotImplementedError("frozen:false 只允许配合 unfrozen_experiment（E5）使用")

    text_dim = embedder.hidden_size
    model = build_model_from_config(cfg, text_dim, experiment).to(device)
    n_params = {
        "total": int(sum(p.numel() for p in model.parameters())),
        "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    text_params = int(sum(p.numel() for p in embedder.model.parameters()))
    text_trainable = int(sum(p.numel() for p in embedder.model.parameters() if p.requires_grad))
    log.info(
        "[%s seed=%d] device=%s 模型参数 total=%d trainable=%d；文本编码器参数=%d（可训练 %d，%s）",
        experiment, seed, device, n_params["total"], n_params["trainable"], text_params, text_trainable,
        "解冻" if unfrozen else "冻结",
    )

    # ---- 每 epoch 验证用的固定视图（完整 + 单模态中段 50%）----
    quick_views: List[Dict[str, Any]] = []
    plan_small = build_validation_plan(
        valid.valid, valid.n, seed=int(cfg.get("evaluation", {}).get("bootstrap_seed", 20240924)),
        views_per_condition=1, dual_pairs=[], dual_views=0,
    )
    for view in plan_small["views"]:
        if view["kind"] == "clean":
            quick_views.append({**view, "_observed": None})
        elif (
            view["masked_modalities"][0],
            view["position"],
            float(view["rate"]),
        ) in TRAIN_VIEW_CONDITIONS:
            quick_views.append(
                {**view, "_observed": view_observed_from_plan(plan_small, view, valid.valid, valid.n)}
            )

    # ---- 优化器：解冻实验给文本编码器单独的小学习率参数组 ----
    groups: List[Dict[str, Any]] = [{"params": trainable_parameters(model), "lr": lr, "name": "head"}]
    if unfrozen:
        tp = [p for p in embedder.model.parameters() if p.requires_grad]
        if not tp:
            raise RuntimeError("解冻实验没有可训练的文本编码器参数")
        groups.append({"params": tp, "lr": text_lr, "name": "text_encoder"})
    optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, math.ceil(train.n / bsz))
    schedule = CosineSchedule(
        optimizer,
        warmup_steps=max(1, steps_per_epoch),
        total_steps=steps_per_epoch * epochs,
        base_lr=lr,
        min_lr_ratio=float(cfg["text_model"].get("min_lr_ratio", 0.05)),
    )

    masker = SpanMasker(
        seed=seed,
        clean_probability=float(tcfg.get("clean_probability", 0.25)),
        mask_rates=tuple(tcfg.get("mask_rates", [0.10, 0.30, 0.50, 0.70])),
        single_modality_probability=float(tcfg.get("single_modality_probability", 0.70)),
    )
    augment = uses_mask_augmentation(experiment)

    # 类别不均衡：可选按训练集频率倒数加权（默认关闭，开启时在 run_info 中记录）
    class_weights = None
    if bool(tcfg.get("class_weighted_loss", False)):
        class_weights = inverse_frequency_class_weights(torch.from_numpy(train.labels)).to(device)
        log.info("[%s seed=%d] 启用逆频率类别权重: %s", experiment, seed, class_weights.tolist())

    # ---- E4 教师（完整输入 + 附件 2 预计算 text），只在训练阶段使用 ----
    dcfg = cfg.get("distillation", {})
    teacher = None
    teacher_cache = None
    if distill:
        if teacher_checkpoint is None:
            raise ValueError("E4 需要 --teacher 指定教师权重（先运行 train --teacher）")
        teacher_cache = load_teacher_text(cfg)
        teacher = load_teacher(cfg, Path(teacher_checkpoint), teacher_cache.text_dim, device)
        log.info(
            "[%s seed=%d] 教师权重 %s（text_dim=%d, T=%.2f, KL 权重=%.2f, 强度一致性权重=%.2f）",
            experiment, seed, teacher_checkpoint, teacher_cache.text_dim,
            float(dcfg.get("temperature", 2.0)), float(dcfg.get("kl_weight", 1.0)),
            float(dcfg.get("intensity_consistency_weight", 0.5)),
        )

    run_dir = output_dir(cfg) / "runs" / f"{experiment}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", jsonable(copy.deepcopy(cfg)))
    save_json(
        run_dir / "run_info.json",
        {
            "experiment": experiment,
            "seed": seed,
            "mask_augmentation": augment,
            "masker": {
                "clean_probability": masker.clean_probability,
                "mask_rates": list(masker.mask_rates),
                "single_modality_probability": masker.single_modality_probability,
            },
            "batch_size": bsz,
            "max_epochs": epochs,
            "learning_rate": lr,
            "weight_decay": wd,
            "regression_weight": lam,
            "grad_clip": grad_clip,
            "class_weighted_loss": bool(tcfg.get("class_weighted_loss", False)),
            "overrides": tcfg.get("_overrides", []),
            "unfrozen_text_encoder": unfrozen,
            "distillation": (
                {
                    "teacher_checkpoint": str(teacher_checkpoint),
                    "teacher_checkpoint_sha256": sha256_file(Path(teacher_checkpoint)),
                    "teacher_text_source": "附件2 预计算 text (N,50,768)",
                    "temperature": float(dcfg.get("temperature", 2.0)),
                    "kl_weight": float(dcfg.get("kl_weight", 1.0)),
                    "intensity_consistency_weight": float(dcfg.get("intensity_consistency_weight", 0.5)),
                }
                if distill
                else None
            ),
            "n_train": train.n,
            "n_valid": valid.n,
            "model_params": n_params,
            "frozen_text_params": text_params,
            "text_model": {
                "repo_id": cfg["text_model"]["repo_id"],
                "revision": cfg["text_model"].get("revision"),
                "hidden_size": embedder.hidden_size,
            },
            "device": str(device),
            "environment": environment_report(),
            "smoke": smoke,
            "script_version": __version__,
        },
    )

    history: List[Dict[str, Any]] = []
    best = {"epoch": -1, "macro_f1": -1.0, "mae": float("inf")}
    epochs_without_improve = 0
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_losses = {"loss": 0.0, "ce": 0.0, "huber": 0.0, "kl": 0.0, "consistency": 0.0}
        n_batches = 0
        n_clean, n_missing = 0, 0
        mask_rate_acc: List[float] = []
        for batch_idx in iter_batches(train.n, bsz, shuffle=True, seed=seed * 1000 + epoch):
            if augment:
                observed, metas = masked_views_for_batch(masker, train, batch_idx, epoch)
                n_clean += sum(1 for m in metas if m["kind"].startswith("clean"))
                n_missing += sum(1 for m in metas if m["kind"] == "missing")
                for m in metas:
                    if m["kind"] == "missing":
                        for mm in m["masked_modalities"]:
                            d = m.get("detail", {}).get(mm)
                            if d:
                                mask_rate_acc.append(d["actual_rate"])
            else:
                observed = None
                n_clean += len(batch_idx)

            batch = make_batch(train, batch_idx, observed=observed, device=device, with_labels=True)
            if unfrozen:
                text_emb = embedder.encode_trainable(batch["text_bert"], batch["text_observed"])
            else:
                text_emb = embedder.encode(batch["text_bert"], batch["text_observed"], batch_size=bsz)
            out = model(
                text_emb,
                batch["audio"],
                batch["vision"],
                observed=batch["observed"],
                text_observed=batch["text_observed"],
            )
            if distill:
                batch["teacher_text"] = torch.from_numpy(
                    teacher_cache.get("train", batch_idx)
                ).to(device)
                batch["teacher_text_observed"] = torch.from_numpy(
                    teacher_cache.attention("train", batch_idx)
                ).to(device)
                t_logits, t_intensity = teacher_logits(teacher, batch)
                losses = distillation_loss(
                    out["class_logits"], out["intensity"], t_logits, t_intensity,
                    batch["labels"], batch["targets"],
                    regression_weight=lam,
                    temperature=float(dcfg.get("temperature", 2.0)),
                    kl_weight=float(dcfg.get("kl_weight", 1.0)),
                    intensity_consistency_weight=float(dcfg.get("intensity_consistency_weight", 0.5)),
                    class_weights=class_weights,
                    ce_class_weights=bool(tcfg.get("class_weighted_loss", False)),
                    normalize_kl=bool(dcfg.get("normalize_kl", True)),
                )
            else:
                losses = loss_function(
                    out["class_logits"], out["intensity"], batch["labels"], batch["targets"],
                    regression_weight=lam, class_weights=class_weights,
                )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), grad_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"梯度出现非有限值（epoch {epoch}）")
            optimizer.step()
            lr_now = schedule.step()
            for k in epoch_losses:
                v = losses.get(k)
                if v is not None:
                    epoch_losses[k] += float(v)
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(1, n_batches)

        # 验证
        val = quick_validate(model, embedder, valid, quick_views, bsz * 4, device)
        s = val["summary"]
        rec = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": epoch_losses["loss"],
            "train_ce": epoch_losses["ce"],
            "train_huber": epoch_losses["huber"],
            "train_kl": epoch_losses["kl"],
            "train_consistency": epoch_losses["consistency"],
            "n_clean_samples": n_clean,
            "n_missing_samples": n_missing,
            "mean_actual_mask_rate": float(np.mean(mask_rate_acc)) if mask_rate_acc else 0.0,
            "valid_clean_macro_f1": s["clean_macro_f1"],
            "valid_clean_mae": s["clean_mae"],
            "valid_missing_macro_f1": s["missing_macro_f1_mean"],
            "valid_missing_mae": s["missing_mae_mean"],
            "elapsed_seconds": round(time.time() - t_start, 2),
        }
        history.append(rec)
        log.info(
            "[%s seed=%d] epoch %d/%d loss=%.4f ce=%.4f huber=%.4f kl=%.4f cons=%.4f | clean F1=%.4f MAE=%.4f | "
            "missing F1=%.4f MAE=%.4f | mask_frac=%.3f | %.1fs",
            experiment, seed, epoch, epochs - 1, rec["train_loss"], rec["train_ce"], rec["train_huber"],
            rec["train_kl"], rec["train_consistency"],
            rec["valid_clean_macro_f1"] or float("nan"), rec["valid_clean_mae"] or float("nan"),
            rec["valid_missing_macro_f1"] or float("nan"), rec["valid_missing_mae"] or float("nan"),
            n_missing / max(1, n_clean + n_missing), rec["elapsed_seconds"],
        )

        # 保存最佳权重
        score = rec["valid_clean_macro_f1"]
        score = float(score) if score is not None else -1.0
        improved = score > best["macro_f1"] + 1e-6
        if improved:
            best = {
                "epoch": epoch,
                "macro_f1": score,
                "mae": rec["valid_clean_mae"],
                "missing_macro_f1": rec["valid_missing_macro_f1"],
            }
            _save_checkpoint(model, run_dir / "best.safetensors", embedder if unfrozen else None)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                log.info("[%s seed=%d] 第 %d epoch 早停（%d 个 epoch 无改进）", experiment, seed, epoch, patience)
                break

    _save_checkpoint(model, run_dir / "last.safetensors", embedder if unfrozen else None)
    result = {
        "experiment": experiment,
        "seed": seed,
        "best": best,
        "epochs_run": len(history),
        "history": history,
        "elapsed_seconds": round(time.time() - t_start, 2),
        "checkpoint": str(run_dir / "best.safetensors"),
        "checkpoint_sha256": sha256_file(run_dir / "best.safetensors"),
        "model_params": n_params,
    }
    save_json(run_dir / "history.json", history)
    save_json(run_dir / "result.json", result)
    log.info(
        "[%s seed=%d] 完成：best epoch=%d clean macro-F1=%.4f，耗时 %.1fs",
        experiment, seed, best["epoch"], best["macro_f1"], result["elapsed_seconds"],
    )
    return result


def _subset(data: SplitData, idx: np.ndarray) -> SplitData:
    obj = SplitData.__new__(SplitData)
    obj.text_bert = data.text_bert[idx]
    obj.audio = data.audio[idx]
    obj.vision = data.vision[idx]
    obj.valid = {m: data.valid[m][idx] for m in MODALITIES}
    obj.labels = data.labels[idx]
    obj.targets = data.targets[idx]
    obj.ids = data.ids[idx]
    obj.raw_text = data.raw_text[idx]
    obj.n = int(len(idx))
    obj.length = int(data.length)
    return obj


def _save_checkpoint(model, path: Path, embedder=None) -> None:
    """保存自训练层；解冻实验额外保存被微调的文本编码器层（bert.*）。"""
    from safetensors.torch import save_file

    state = {k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()}
    if embedder is not None and not getattr(embedder, "frozen", True):
        for name, param in embedder.model.named_parameters():
            if param.requires_grad:
                state[f"bert.{name}"] = param.detach().to("cpu").contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path))


def main(argv=None) -> int:
    from .common import setup_logging

    parser = argparse.ArgumentParser(description="问题 2 训练")
    parser.add_argument("--config", default="configs/problem2.yaml")
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENTS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="小样本 1-2 epoch 端到端冒烟")
    parser.add_argument("--log-file", default=None)
    parser.add_argument(
        "--teacher",
        nargs="?",
        const="auto",
        default=None,
        help="E4 蒸馏：指定教师权重；也可用 --teacher 训练/使用默认路径的教师",
    )
    parser.add_argument(
        "--train-teacher",
        action="store_true",
        help="训练 E4 教师（完整输入 + 附件 2 预计算 text）",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="临时覆盖配置项（小范围搜索用，例如 --set model.dropout=0.1 --set text_model.unfrozen_learning_rate=0.0003）；"
        "每次覆盖都会记录到 run 目录与日志，便于回查",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    try:
        overrides = apply_overrides(cfg, args.set)
    except Exception as exc:  # noqa: BLE001
        print(f"[train] --set 参数无效: {exc}")
        return 2

    if args.train_teacher:
        log_path = resolve(args.log_file) if args.log_file else output_dir(cfg) / "logs" / f"teacher_seed{args.seed}.log"
        log = setup_logging(log_path)
        log.info("=== 训练教师 seed=%d 开始（config=%s）===", args.seed, cfg["_config_path"])
        if overrides:
            log.info("命令行覆盖项: %s", overrides)
        try:
            run_teacher(cfg, args.seed, max_epochs=args.max_epochs, batch_size=args.batch_size,
                        smoke=args.smoke, log=log)
        except Exception as exc:  # noqa: BLE001
            log.exception("教师训练失败: %s", exc)
            return 1
        return 0

    teacher_path: Optional[Path] = None
    if args.teacher:
        if args.teacher == "auto":
            teacher_path = output_dir(cfg) / "teacher" / f"{TEACHER_ARCH}_seed{args.seed}" / "best.safetensors"
        else:
            teacher_path = resolve(args.teacher)
        if not teacher_path.exists():
            print(f"[train] 教师权重不存在: {teacher_path}")
            return 2

    log_path = resolve(args.log_file) if args.log_file else output_dir(cfg) / "logs" / f"train_{args.experiment}_seed{args.seed}.log"
    log = setup_logging(log_path)
    log.info("=== 训练 %s seed=%d 开始（config=%s）===", args.experiment, args.seed, cfg["_config_path"])
    if overrides:
        log.info("命令行覆盖项: %s", overrides)
    try:
        run(
            cfg,
            args.experiment,
            args.seed,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            smoke=args.smoke,
            log=log,
            teacher_checkpoint=teacher_path,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("训练失败: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
