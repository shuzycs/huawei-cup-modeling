"""统计模型里实际使用的激活函数（按模块树遍历 + 逐次前向计数）。

用法::

    python -m tools.show_activations            # 学生模型 E5（与最终提交一致）
    python -m tools.show_activations --experiment E1
    python -m tools.show_activations --teacher  # E4 教师（768 维文本）
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
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

# 激活函数模块类型 -> 显示名
ACT_TYPES: Tuple[type, ...] = (
    nn.GELU, nn.ReLU, nn.Tanh, nn.Sigmoid, nn.SiLU, nn.Mish, nn.ELU, nn.LeakyReLU,
    nn.Softmax, nn.LogSoftmax, nn.Softplus, nn.Hardswish, nn.PReLU,
)


def activation_name(mod: nn.Module) -> str | None:
    for t in ACT_TYPES:
        if isinstance(mod, t):
            return type(mod).__name__
    return None


def walk(model: nn.Module) -> List[Tuple[str, str]]:
    found: List[Tuple[str, str]] = []
    for name, mod in model.named_modules():
        act = activation_name(mod)
        if act is not None:
            found.append((name or "<root>", act))
    return found


def functional_calls(model: nn.Module, *args, **kwargs) -> Counter:
    """统计一次前向中 torch 激活函数的实际调用次数。

    同时覆盖 torch.nn.functional.* 与 torch.tanh/torch.sigmoid 这类顶层函数，
    否则会漏掉回归头的 ``3.0 * torch.tanh(...)``。
    """
    counter: Counter = Counter()
    patched_objs: List[Tuple[object, str, object]] = []

    def patch(obj, attr: str, label: str) -> None:
        if not hasattr(obj, attr):
            return
        orig = getattr(obj, attr)

        def wrapper(*a, **k):
            counter[label] += 1
            return orig(*a, **k)

        patched_objs.append((obj, attr, orig))
        setattr(obj, attr, wrapper)

    for name in ("gelu", "relu", "tanh", "sigmoid", "silu", "softmax", "log_softmax", "leaky_relu", "elu"):
        patch(torch.nn.functional, name, f"F.{name}")
    for name in ("tanh", "sigmoid", "softmax"):
        patch(torch, name, f"torch.{name}")
    try:
        model(*args, **kwargs)
    finally:
        for obj, attr, orig in patched_objs:
            setattr(obj, attr, orig)
    return counter


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="统计模型使用的激活函数")
    parser.add_argument("--config", default="configs/problem2.yaml")
    parser.add_argument("--experiment", default="E5")
    parser.add_argument("--teacher", action="store_true", help="统计 E4 教师（768 维文本侧）")
    parser.add_argument("--hidden-dim", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.hidden_dim:
        cfg["model"]["hidden_dim"] = args.hidden_dim
    hidden = int(cfg["model"]["hidden_dim"])
    print(f"配置文件: {cfg['_config_path']}")
    print(f"hidden_dim = {hidden}\n")

    if args.teacher:
        model = build_teacher_from_config(cfg, text_dim=768)
        text_dim, title = 768, "E4 教师（文本侧 768 维，不做降维）"
    else:
        model = build_model_from_config(cfg, text_dim=128, experiment=args.experiment)
        text_dim, title = 128, f"{args.experiment} 自训练部分"

    model.eval()
    print("=" * 78)
    print(f"一、自训练部分的激活函数模块（{title}）")
    print("=" * 78)
    mods = walk(model)
    kinds = Counter(act for _, act in mods)
    for act, cnt in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {act:12s} × {cnt}")
        for name, a in mods:
            if a == act:
                print(f"        {name}")
    if not mods:
        print("  （无激活模块：只有线性/归一化/softmax 等）")

    print()
    print("=" * 78)
    print("二、一次前向中的实际调用次数（含函数式调用）")
    print("=" * 78)
    b, length = 4, 50
    text = torch.randn(b, length, text_dim)
    audio = torch.randn(b, length, 74)
    vision = torch.randn(b, length, 35)
    text_obs = torch.zeros(b, length, dtype=torch.bool)
    text_obs[:, 1:10] = True
    with torch.no_grad():
        counter = functional_calls(model, text, audio, vision, text_observed=text_obs)
    for k, v in sorted(counter.items()):
        print(f"  {k:16s} × {v}")
    print("  说明：调用次数取决于 batch/长度，这里仅示意各激活是否被真正走到。")

    print()
    print("=" * 78)
    print("三、冻结文本编码器（google/bert_uncased_L-2_H-128_A-2）内部的激活")
    print("=" * 78)
    config, bert = load_bert(
        cfg["text_model"]["repo_id"], cfg["text_model"].get("revision"), cfg["text_model"]["local_dir"]
    )
    print(f"  hidden_act 配置项 = {getattr(config, 'hidden_act', None)}")
    print(f"  hidden_act 解析后 = {getattr(config, 'hidden_act', None)}（transformers 会映射到 ACT2FN）")
    bert_mods = Counter(act for _, act in walk(bert))
    for act, cnt in sorted(bert_mods.items(), key=lambda kv: -kv[1]):
        print(f"  {act:12s} × {cnt}")
    n_layers = int(getattr(config, "num_hidden_layers", 0))
    print(f"  层数 = {n_layers}；每层含 1 个中间激活 + 1 个输出激活（均为 {getattr(config, 'hidden_act', None)}）")
    print(f"  池化层激活 = {getattr(config, 'pooler_activation', '(未设置)')}")

    print()
    print("=" * 78)
    print("四、输出层")
    print("=" * 78)
    print("  分类头: nn.Linear(fusion_dim, 3) -> 直接输出 logits（训练用 F.cross_entropy，推理用 F.softmax）")
    print("  回归头: nn.Linear(fusion_dim, 1) -> 3.0 * tanh(.)  -> 强度落在 [-3, 3]")
    print("  融合权重: F.softmax(门控 logit)（不可用模态 logit 置 -inf）")
    print("  注意力池化权重: F.softmax(打分)（不可观测位置置 -inf）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
