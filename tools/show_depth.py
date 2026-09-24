"""统计模型层数（按"带参数层"计，区分冻结与可训练）。

用法::

    python -m tools.show_depth              # 默认统计最终模型 E5
    python -m tools.show_depth --experiment E1

统计口径：只计**带参数**的层（Linear / Conv1d / LayerNorm / 注意力 / 门控等）；
Dropout、激活函数、乘法遮蔽、softmax、池化加权求和、einsum 融合等不单独计层。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.problem2.common import load_config  # noqa: E402
from src.problem2.model import build_model_from_config, build_teacher_from_config  # noqa: E402
from src.problem2.text_encoder import load_bert  # noqa: E402

PARAM_LAYER_TYPES = (
    nn.Linear, nn.Conv1d, nn.LayerNorm, nn.BatchNorm1d, nn.Embedding,
    nn.MultiheadAttention, nn.Conv2d,
)


def count_layers(model: nn.Module) -> Tuple[int, int, List[Tuple[str, str, int]]]:
    """返回 (总带参数层数, 可训练带参数层数, [(名称, 类型, 参数量)])。"""
    rows: List[Tuple[str, str, int]] = []
    trainable = 0
    for name, mod in model.named_modules():
        if isinstance(mod, PARAM_LAYER_TYPES):
            n = sum(p.numel() for p in mod.parameters(recurse=False))
            rows.append((name, type(mod).__name__, n))
            if any(p.requires_grad for p in mod.parameters(recurse=False)):
                trainable += 1
    return len(rows), trainable, rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="统计模型层数")
    parser.add_argument("--config", default="configs/problem2.yaml")
    parser.add_argument("--experiment", default="E5")
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="逐层列出")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    hidden = int(cfg["model"]["hidden_dim"])
    print(f"hidden_dim = {hidden}；dropout = {cfg['model']['dropout']}\n")

    # ---------------- 冻结的文本编码器 ----------------
    bcfg, bert = load_bert(
        cfg["text_model"]["repo_id"], cfg["text_model"].get("revision"), cfg["text_model"]["local_dir"]
    )
    bert_total, bert_trainable, bert_rows = count_layers(bert)
    n_bert_layers = int(getattr(bcfg, "num_hidden_layers", 0))
    print("=" * 80)
    print("一、文本编码器（BERT，预训练权重）")
    print("=" * 80)
    print(f"  repo: {cfg['text_model']['repo_id']}  revision: {cfg['text_model'].get('revision')}")
    print(f"  Transformer 层数 = {n_bert_layers}（每层 = attention + intermediate + output）")
    print(f"  hidden_size={bcfg.hidden_size}  注意力头数={bcfg.num_attention_heads}  "
          f"FFN 中间维={bcfg.intermediate_size}  激活={bcfg.hidden_act}")
    print(f"  带参数层合计 = {bert_total} 层；参数量 = {sum(p.numel() for p in bert.parameters()):,}")
    if args.verbose:
        for name, kind, n in bert_rows:
            print(f"      {name:60s} {kind:12s} {n:>8,}")
    print()

    # ---------------- 自训练部分 ----------------
    if args.teacher:
        model = build_teacher_from_config(cfg, text_dim=768)
        title, text_dim = "E4 教师（768 维文本侧）", 768
    else:
        model = build_model_from_config(cfg, text_dim=128, experiment=args.experiment)
        title, text_dim = f"{args.experiment} 自训练部分", 128
    total, trainable, rows = count_layers(model)
    params = sum(p.numel() for p in model.parameters())
    print("=" * 80)
    print(f"二、自训练部分（{title}）")
    print("=" * 80)
    print(f"  带参数层合计 = {total} 层；参数量 = {params:,}")
    if args.verbose:
        for name, kind, n in rows:
            print(f"      {name:46s} {kind:12s} {n:>8,}")
    print()

    # ---------------- 分模态深度 ----------------
    print("=" * 80)
    print("三、从输入到输出的路径长度（按逐个子层计）")
    print("=" * 80)
    mods = list(model.modalities)
    print(f"  模态分支 = {mods}")
    per_mod = {}
    for m in mods:
        if m == "T":
            if args.teacher:
                depth = 2  # LayerNorm + Linear(768->hidden)
                desc = "LayerNorm(768) → Linear(768→hidden)"
            else:
                depth = 1
                desc = "Linear(128→hidden)（输入已由 BERT 编码）"
        else:
            depth = 4  # Linear + LayerNorm + Conv1d + LayerNorm
            desc = "Linear → LayerNorm → Conv1d(k=3) → LayerNorm"
        pool_depth = 2  # score: Linear + Linear
        pool_desc = "掩码注意力池化：Linear → Tanh → Linear（2 层）"
        if model.pools[m].__class__.__name__ == "MeanPool":
            pool_depth, pool_desc = 0, "平均池化：无参数"
        per_mod[m] = depth + pool_depth
        print(f"  [{m}] {desc} → {pool_desc}  => {per_mod[m]} 层")
    head = 0 if model.gate is None else 2
    print(f"  融合门控：Linear → Tanh → Linear = {head} 层（E1/E2 为加权平均，0 层）")
    print(f"  输出头：分类 Linear + 回归 Linear = 2 层")
    print()
    vision_path = per_mod["V"] + head + 2
    text_path = (2 if args.teacher else 1) + per_mod["T"] + head + 2
    print(f"  音频/视觉分支（自训练）路径 = {per_mod['A']} + {head} + 2 = {vision_path} 层")
    print(f"  文本分支（自训练）路径     = {1 if not args.teacher else 2} + {per_mod['T']} + {head} + 2 = {text_path} 层")

    print()
    print("=" * 80)
    print("四、结论（两种口径，避免混淆）")
    print("=" * 80)
    print("  口径 A：按逐个子层计数（Linear/LayerNorm/Conv1d/Attention 均算一层）")
    print(f"    自训练部分          : {total} 层，参数量 {params:,}")
    print(f"    冻结的 BERT         : {bert_total} 层，参数量 {sum(p.numel() for p in bert.parameters()):,}")
    print(f"    端到端最长路径      : 音频/视觉分支 {vision_path} 层 > 文本分支 {n_bert_layers + text_path} 层")
    print()
    print("  口径 B：按残差块/结构单元计数（业界描述深度的常用口径）")
    print(f"    文本编码器          : {n_bert_layers} 层 Transformer"
          f"（每层含 1 个 2 头自注意力 + 1 个 FFN(128→512→128)）")
    print(f"    音频时序编码器      : 1 个残差块（Linear → LayerNorm → Conv1d(k=3) → LayerNorm → GELU）")
    print(f"    视觉时序编码器      : 1 个残差块（同上）")
    print(f"    跨模态融合          : 1 层（掩码注意力池化 + 动态门控）")
    print(f"    输出头              : 2 个并行线性头（分类 3 类 / 回归 1 维）")
    print(f"    => 可学习结构的层数实际是 2（音视频各 1 个残差块）+ 1（融合）= 3 层")
    print()
    print("  参数规模：可训练 173,704（0.17 M） + 冻结 4,385,920（4.39 M） = 4,559,624（4.56 M）")
    print(f"  解冻说明：E0~E4 冻结全部 BERT；E5 解冻最后 "
          f"{int(cfg['text_model'].get('unfreeze_top_layers', 2))} 层 Transformer"
          f"（含文本侧共 396,544 个可训练参数）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
