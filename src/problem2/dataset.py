"""数据访问层：加载 .npz 缓存、构造批、应用缺失视图与文本编码。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import MODALITIES
from .common import REPO_ROOT, resolve
from .masks import (
    apply_missing_to_features,
    apply_missing_to_text,
    content_validity_text,
    view_observed_from_plan,
)

SPECIAL_TOKEN_IDS: Tuple[int, ...] = (0, 100, 101, 102, 103)


class SplitData:
    """一个划分的全部缓存内容（常驻内存，避免重复读 1 GiB Pickle）。"""

    def __init__(self, path: str | Path):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"缓存不存在，请先运行 prepare_data: {p}")
        with np.load(p, allow_pickle=False) as z:
            self.text_bert = z["text_bert"].astype(np.int64)
            self.audio = z["audio"].astype(np.float32)
            self.vision = z["vision"].astype(np.float32)
            self.valid = {
                "T": z["valid_T"].astype(bool),
                "A": z["valid_A"].astype(bool),
                "V": z["valid_V"].astype(bool),
            }
            self.labels = z["classification_labels"].astype(np.int64)
            self.targets = z["regression_labels"].astype(np.float32)
            self.ids = np.asarray([str(x) for x in z["ids"]])
            self.raw_text = (
                np.asarray([str(x) for x in z["raw_text"]])
                if "raw_text" in z.files
                else np.asarray([""] * self.text_bert.shape[0])
            )
        self.n = int(self.text_bert.shape[0])
        self.length = int(self.text_bert.shape[2])

    @property
    def text_observed_full(self) -> np.ndarray:
        return self.valid["T"]

    def __len__(self) -> int:
        return self.n


def cache_path(cfg: Dict[str, Any], split: str) -> Path:
    return resolve(cfg["data"]["cache_dir"]) / f"{split}.npz"


def load_split(cfg: Dict[str, Any], split: str) -> SplitData:
    return SplitData(cache_path(cfg, split))


def load_normalizer(cfg: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    path = resolve(cfg["data"]["cache_dir"]) / "normalizer.npz"
    if not path.exists():
        raise FileNotFoundError(f"缺少标准化参数，请先运行 prepare_data: {path}")
    with np.load(path) as z:
        return {
            "audio": {"mean": z["audio_mean"], "std": z["audio_std"]},
            "vision": {"mean": z["vision_mean"], "std": z["vision_std"]},
        }


class TeacherTextCache:
    """E4 教师输入：附件 2 的预计算 text（只用于训练阶段蒸馏）。

    用 mmap 打开，避免把整个 (N,50,768) 数组一次性读进内存。
    有效性规则与数据核查一致：v_T = text_bert 的 attention_mask。
    """

    def __init__(self, path: str | Path, allow_mmap: bool = True):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"缺少教师文本缓存，请先运行 prepare_data: {p}"
            )
        self.path = p
        self._z = np.load(p, mmap_mode="r" if allow_mmap else None)
        self.splits = {
            name[len("text_") :]: self._z[name]
            for name in self._z.files
            if name.startswith("text_")
        }
        self.text_dim = int(next(iter(self.splits.values())).shape[-1])

    def get(self, split: str, indices: Optional[Sequence[int]] = None) -> np.ndarray:
        if split not in self.splits:
            raise KeyError(f"教师缓存中没有划分 {split}，可选 {sorted(self.splits)}")
        arr = self.splits[split]
        return np.asarray(arr if indices is None else arr[np.asarray(indices, dtype=np.int64)])

    def attention(self, split: str, indices: Optional[Sequence[int]] = None) -> np.ndarray:
        key = f"att_T_{split}"
        if key not in self._z.files:
            raise KeyError(f"教师缓存中缺少 {key}")
        arr = self._z[key]
        out = arr if indices is None else arr[np.asarray(indices, dtype=np.int64)]
        return np.asarray(out).astype(bool)


def load_teacher_text(cfg: Dict[str, Any]) -> TeacherTextCache:
    return TeacherTextCache(resolve(cfg["data"]["cache_dir"]) / "teacher_text.npz")


def normalize_with(
    raw: np.ndarray, mean: np.ndarray, std: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """用训练集统计量标准化；只在可观测行上标准化，其余保留零。"""
    raw = np.asarray(raw, dtype=np.float32)
    mask_rows = np.asarray(valid, dtype=bool)
    out = np.zeros_like(raw, dtype=np.float32)
    if mask_rows.any():
        normalized = (raw - mean.astype(np.float32)) / std.astype(np.float32)
        out[mask_rows] = normalized[mask_rows]
    return out


# --------------------------------------------------------------------------------------
# 批构造
# --------------------------------------------------------------------------------------
def make_batch(
    data: SplitData,
    indices: Sequence[int],
    observed: Optional[Dict[str, np.ndarray]] = None,
    device: torch.device | str = "cpu",
    with_labels: bool = True,
    teacher_text: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """把索引与可观测掩码打包成模型输入。

    observed[m] 支持两种形态：
      * 全划分级 (N, 50)：按 idx 取行；
      * 批级 (B, 50)：与 idx 一一对应，直接使用。
    None 表示使用 data.valid[m]，即完整视图。

    teacher_text: 可选 {"features": (B,50,768), "observed": (B,50)}，仅 E4 教师使用。
    """
    idx = np.asarray(list(indices), dtype=np.int64)
    text_bert = data.text_bert[idx]  # (B,3,L)

    obs: Dict[str, np.ndarray] = {}
    for m in MODALITIES:
        if observed is None or observed.get(m) is None:
            obs[m] = data.valid[m][idx]
        else:
            arr = np.asarray(observed[m]).astype(bool)
            obs[m] = arr if arr.shape[0] == idx.shape[0] else arr[idx]

    # 文本：遮蔽只作用于内容 token，且必须在进入 BERT 之前完成
    text_bert_masked = apply_missing_to_text(text_bert, obs["T"])
    audio = apply_missing_to_features(data.audio[idx], obs["A"])
    vision = apply_missing_to_features(data.vision[idx], obs["V"])

    batch: Dict[str, Any] = {
        "indices": idx,
        "text_bert": torch.from_numpy(text_bert_masked).to(device),
        "text_observed": torch.from_numpy(obs["T"]).to(device),
        "audio": torch.from_numpy(audio).to(device),
        "vision": torch.from_numpy(vision).to(device),
        "observed": {m: torch.from_numpy(obs[m]).to(device) for m in MODALITIES},
        "ids": [data.ids[i] for i in idx],
    }
    if teacher_text is not None:
        # 教师用的预计算 text 不参与缺失模拟（教师在完整输入上训练）
        batch["teacher_text"] = torch.from_numpy(
            np.asarray(teacher_text.get("features"), dtype=np.float32)
        ).to(device)
        batch["teacher_text_observed"] = torch.from_numpy(
            np.asarray(teacher_text.get("observed"), dtype=bool)
        ).to(device)
    if with_labels:
        batch["labels"] = torch.from_numpy(data.labels[idx]).to(device)
        batch["targets"] = torch.from_numpy(data.targets[idx]).to(device)
    return batch


# --------------------------------------------------------------------------------------
# 文本编码（冻结 BERT，必须在缺失应用之后）
# --------------------------------------------------------------------------------------
class TextEmbedder:
    """冻结的小型 BERT -> 时序表示。

    关键点：传入的 text_bert 已经是"缺失已应用"的版本（不可观测内容 token 被置为
    [PAD] 且 attention mask=0），因此剩余 token 的上下文表示不会包含被遮蔽内容。
    编码后再把不可观测位置乘零。
    """

    def __init__(self, repo_id: str, revision: Optional[str], local_dir: str | Path, device: torch.device | str = "cpu"):
        from .text_encoder import load_bert

        self.device = torch.device(device)
        config, model = load_bert(repo_id, revision, local_dir)
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.hidden_size = int(config.hidden_size)
        self.max_position = int(getattr(config, "max_position_embeddings", 512))
        self.repo_id = repo_id
        self.revision = revision
        self.frozen = True

    # ------------------------------------------------------------------ 解冻支持
    def unfreeze_top_layers(self, n_layers: int = 2) -> int:
        """解冻文本编码器最后 n 层（含 pooler 之外的自注意力/前馈参数）。

        这是手册 1.1 明确要求的独立实验（E5），默认基线仍然完全冻结。
        返回实际解冻的层数。
        """
        self.frozen = False
        encoder_layers = getattr(self.model, "encoder", None)
        layers = list(getattr(encoder_layers, "layer", [])) if encoder_layers is not None else []
        n = min(int(n_layers), len(layers))
        for layer in layers[len(layers) - n :]:
            for p in layer.parameters():
                p.requires_grad_(True)
        return n

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def encode_trainable(self, text_bert: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        """可求导路径：文本缺失同样在进入 BERT 之前已应用（由 make_batch 完成）。"""
        input_ids = text_bert[:, 0, :].contiguous()
        attention = text_bert[:, 1, :].contiguous()
        token_type = text_bert[:, 2, :].contiguous()
        out = self.model(
            input_ids=input_ids, attention_mask=attention, token_type_ids=token_type
        )
        hidden = out.last_hidden_state.float()
        mask = observed.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        return hidden * mask

    @torch.no_grad()
    def encode(self, text_bert: torch.Tensor, observed: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
        """text_bert: (B,3,L) int64；observed: (B,L) bool -> (B,L,hidden)"""
        self.model.eval()
        outs: List[torch.Tensor] = []
        b = text_bert.shape[0]
        for s in range(0, b, batch_size):
            chunk = text_bert[s : s + batch_size]
            length = chunk.shape[2]
            if length > self.max_position:  # 本数据固定为 50，这里只做保护
                chunk = chunk[:, :, : self.max_position]
            input_ids = chunk[:, 0, :].contiguous()
            attention = chunk[:, 1, :].contiguous()
            token_type = chunk[:, 2, :].contiguous()
            out = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention.to(self.device),
                token_type_ids=token_type.to(self.device),
            )
            hidden = out.last_hidden_state
            if hidden.shape[1] < length:  # 长度被裁到 max_position 时补齐
                pad = torch.zeros(
                    hidden.shape[0], length - hidden.shape[1], hidden.shape[2],
                    dtype=hidden.dtype, device=hidden.device,
                )
                hidden = torch.cat([hidden, pad], dim=1)
            outs.append(hidden.float())
        hidden = torch.cat(outs, dim=0)  # (B,L,H)
        mask = observed.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        return (hidden * mask).to(dtype=torch.float32)

    def encode_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.encode(batch["text_bert"], batch["text_observed"])


# --------------------------------------------------------------------------------------
# 视图
# --------------------------------------------------------------------------------------
def load_validation_plan(cfg: Dict[str, Any]) -> Dict[str, Any]:
    import json

    path = resolve(cfg["data"]["cache_dir"]) / "validation_masks.json"
    if not path.exists():
        raise FileNotFoundError(f"缺少固定验证缺失视图，请先运行 prepare_data: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def view_observed(
    cfg: Dict[str, Any], plan: Dict[str, Any], view: Dict[str, Any], data: SplitData
) -> Dict[str, np.ndarray]:
    return view_observed_from_plan(plan, view, data.valid, data.n)


def iter_batches(n: int, batch_size: int, shuffle: bool = False, seed: int = 0) -> Iterable[np.ndarray]:
    order = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    for s in range(0, n, batch_size):
        yield order[s : s + batch_size]


# --------------------------------------------------------------------------------------
# 附件 3
# --------------------------------------------------------------------------------------
def load_challenge_file(path: str | Path) -> Dict[str, np.ndarray]:
    from .data import load_pickle

    d = load_pickle(path)
    if "test" not in d:
        raise KeyError(f"{path}: 缺少 'test' 字段")
    t = d["test"]
    return {
        "text_bert": np.round(np.asarray(t["text_bert"])).astype(np.int64),
        "audio": np.asarray(t["audio"], dtype=np.float64),
        "vision": np.asarray(t["vision"], dtype=np.float64),
    }


def challenge_batch(
    sample: Dict[str, np.ndarray],
    normalizer: Dict[str, Dict[str, np.ndarray]],
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """构造单文件（1 个样本）的批：全零行视为不可观测。"""
    tb = sample["text_bert"]
    audio_raw = sample["audio"]
    vision_raw = sample["vision"]
    valid_T = content_validity_text(tb[:, 0, :], tb[:, 1, :])
    valid_A = np.abs(audio_raw).sum(axis=-1) > 0
    valid_V = np.abs(vision_raw).sum(axis=-1) > 0

    audio = normalize_with(audio_raw, normalizer["audio"]["mean"], normalizer["audio"]["std"], valid_A)
    vision = normalize_with(vision_raw, normalizer["vision"]["mean"], normalizer["vision"]["std"], valid_V)

    tb_masked = apply_missing_to_text(tb, valid_T)
    audio = apply_missing_to_features(audio, valid_A)
    vision = apply_missing_to_features(vision, valid_V)

    return {
        "text_bert": torch.from_numpy(tb_masked).to(device),
        "text_observed": torch.from_numpy(valid_T).to(device),
        "audio": torch.from_numpy(audio).to(device),
        "vision": torch.from_numpy(vision).to(device),
        "observed": {m: torch.from_numpy(v).to(device) for m, v in (("T", valid_T), ("A", valid_A), ("V", valid_V))},
        "valid": {"T": valid_T, "A": valid_A, "V": valid_V},
    }
