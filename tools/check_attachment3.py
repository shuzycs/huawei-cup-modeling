"""检查附件 3 的可观测性与预测（只读，用于人工核查 30 条推理结果）。

用法::

    python -m tools.check_attachment3 --checkpoint outputs/problem2/runs/E5_seed42/best.safetensors
    python -m tools.check_attachment3 --experiment E3            # 用其它实验的结构核对

输出每个文件的 文本内容 token 数 / 各模态有效行数 / 三分类概率 / 强度 / 融合权重，
用于回答"附件 3 到底缺了哪个模态、模型是怎么处理的"这类问题。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.problem2 import CLASS_NAMES, MODALITIES  # noqa: E402
from src.problem2.common import load_config, resolve  # noqa: E402
from src.problem2.data import list_challenge_files  # noqa: E402
from src.problem2.dataset import (  # noqa: E402
    TextEmbedder,
    challenge_batch,
    load_challenge_file,
    load_normalizer,
)
from src.problem2.infer import load_student  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="附件 3 可观测性与预测核查")
    parser.add_argument("--config", default="configs/problem2.yaml")
    parser.add_argument("--experiment", default="E5")
    parser.add_argument("--checkpoint", default="outputs/problem2/runs/E5_seed42/best.safetensors")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device(
        args.device or (cfg["training"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    )
    ckpt = resolve(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"权重不存在: {ckpt}")
    input_dir = resolve(args.input_dir or cfg["data"]["challenge_dir"])

    normalizer = load_normalizer(cfg)
    embedder = TextEmbedder(
        cfg["text_model"]["repo_id"], cfg["text_model"].get("revision"), cfg["text_model"]["local_dir"], device
    )
    model = load_student(cfg, args.experiment, ckpt, device, embedder=embedder)

    files = list_challenge_files(cfg)
    print(f"实验={args.experiment} 权重={ckpt} 文件数={len(files)} 设备={device}")
    print(
        f"{'file':20s}{'T内容':>6s}{'att_len':>8s}{'vA':>5s}{'vV':>5s}"
        f"{'类别':>5s}{'p_neg':>8s}{'p_neu':>8s}{'p_pos':>8s}{'强度':>9s}"
        f"{'aT':>7s}{'aA':>7s}{'aV':>7s}"
    )
    counts: Counter = Counter()
    inconsistent = 0
    with torch.no_grad():
        for path in files:
            sample = load_challenge_file(path)
            batch = challenge_batch(sample, normalizer, device)
            tb = sample["text_bert"]
            att = tb[:, 1, :] > 0
            content = int((att & ~np.isin(tb[:, 0, :], [0, 100, 101, 102, 103])).sum())
            att_len = int(att.sum())
            v = batch["valid"]
            h = embedder.encode(batch["text_bert"], batch["text_observed"], batch_size=8)
            out = model(
                h,
                batch["audio"],
                batch["vision"],
                observed=batch["observed"],
                text_observed=batch["text_observed"],
            )
            probs = torch.softmax(out["class_logits"], dim=-1).cpu().numpy()[0]
            cls = int(np.argmax(probs))
            inten = float(out["intensity"].cpu().numpy()[0])
            w = out["fusion_weights"].cpu().numpy()[0]
            counts[CLASS_NAMES[cls]] += 1
            if (inten > 0.2 and cls == 0) or (inten < -0.2 and cls == 2):
                inconsistent += 1
            print(
                f"{path.name:20s}{content:>6d}{att_len:>8d}{int(v['A'].sum()):>5d}{int(v['V'].sum()):>5d}"
                f"{cls:>5d}{probs[0]:>8.3f}{probs[1]:>8.3f}{probs[2]:>8.3f}{inten:>9.3f}"
                + "".join(f"{x:>7.3f}" for x in w[: len(model.modalities)])
            )

    print()
    print(f"类别分布: {dict(counts)}")
    print(f"类别与强度符号明显冲突的行数（|强度|>0.2 且极性相反）: {inconsistent} / {len(files)}")
    print("注：附件 3 无真实标签，本脚本只做可观测性与自洽性核查，不计算任何指标。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
