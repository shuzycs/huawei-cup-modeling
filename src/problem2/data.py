"""数据读取：附件 2 对齐版 pkl、label.xlsx、附件 3 对齐版 30 个 pkl。

只反序列化可信的赛题文件（本仓库 data/ 目录下的官方附件）。
标签连接严格按 `video_id + '$_$' + clip_id`，不依赖行号。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import MODALITIES
from .common import REPO_ROOT, resolve

SPLITS = ("train", "valid", "test")
SPLIT_SIZES = {"train": 3395, "valid": 728, "test": 727}
LABEL_SEPARATOR = "$_$"

ATT2_FIELDS = ("text_bert", "audio", "vision", "raw_text", "id", "classification_labels", "regression_labels")


def load_pickle(path: str | Path) -> Any:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"数据文件不存在: {p}")
    with open(p, "rb") as fh:
        return pickle.load(fh)


def load_aligned(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return load_pickle(resolve(cfg["data"]["aligned_pkl"]))


def split_dir(cfg: Dict[str, Any]) -> Path:
    return resolve(cfg["data"]["challenge_dir"])


def list_challenge_files(cfg: Dict[str, Any]) -> List[Path]:
    """附件 3 对齐版文件，按文件名中的编号 01..30 明确排序。"""
    d = split_dir(cfg)
    files = sorted(d.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"附件 3 目录内没有 .pkl 文件: {d}")
    return files


def challenge_index(path: Path) -> int:
    stem = path.stem
    tail = stem.split("_")[-1]
    if not tail.isdigit():
        raise ValueError(f"无法从文件名解析编号: {path.name}")
    return int(tail)


def labels_dataframe(cfg: Dict[str, Any]):
    import pandas as pd

    xlsx = resolve(cfg["data"]["labels_xlsx"])
    if not xlsx.exists():
        raise FileNotFoundError(f"标签文件不存在: {xlsx}")
    return pd.read_excel(xlsx)


def build_label_lookup(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Excel -> {"{video_id}$_${clip_id}": {...}}，按 ID 连接而非行号。"""
    df = labels_dataframe(cfg)
    needed = {"video_id", "clip_id", "label", "annotation"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"label.xlsx 缺少列: {sorted(missing)}")
    lookup: Dict[str, Dict[str, Any]] = {}
    duplicated = 0
    for row in df.itertuples(index=False):
        key = f"{row.video_id}{LABEL_SEPARATOR}{int(row.clip_id)}"
        if key in lookup:
            duplicated += 1
        lookup[key] = {
            "label": float(row.label),
            "annotation": str(row.annotation),
            "mode": str(getattr(row, "mode", "")),
            "text": str(getattr(row, "text", "")),
        }
    if duplicated:
        raise ValueError(f"label.xlsx 存在 {duplicated} 个重复 ID")
    return lookup


def annotation_to_class(annotation: str) -> Optional[int]:
    a = str(annotation).strip().lower()
    if a.startswith("neg"):
        return 0
    if a.startswith("neu"):
        return 1
    if a.startswith("pos"):
        return 2
    return None


def load_split(cfg: Dict[str, Any], split: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = data if data is not None else load_aligned(cfg)
    if split not in data:
        raise KeyError(f"aligned pkl 中不存在划分: {split}")
    return data[split]
