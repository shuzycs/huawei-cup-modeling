"""确定性连续区间缺失生成器与固定验证缺失视图。

掩码定义（与手册 2.2 一致）
-------------------------
    v_{m,t} = 1  原始有效位置（真实内容）
    o_{m,t} = 1  本次输入中实际可观测（填充或局部缺失为 0）
局部缺失 = v=1 且 o=0，与原填充 v=0 分开记录。

所有随机性都来自 `stable_seed`（SHA256），不使用 Python 内置 hash()，
因此跨进程、跨机器、跨次运行完全可复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import MODALITIES
from .common import np_rng_for, stable_seed

POSITIONS: Tuple[str, ...] = ("start", "middle", "end")
ORIGINS: Dict[str, str] = {"start": "start", "middle": "center", "end": "end"}


# --------------------------------------------------------------------------------------
# 有效性与可观测性
# --------------------------------------------------------------------------------------
def text_special_token_mask(input_ids: np.ndarray) -> np.ndarray:
    """标注 [CLS]/[SEP]/[PAD] 等特殊 token 位置（True = 特殊 token）。"""
    special_ids = {0, 100, 101, 102, 103}
    return np.isin(np.asarray(input_ids), list(special_ids))


def content_validity_text(input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """文本内容有效性 v_T：attention mask=1 且非特殊 token。"""
    att = np.asarray(attention_mask) > 0
    return att & (~text_special_token_mask(input_ids))


def content_validity_features(raw: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """语音/视觉内容有效性 v：非零行（原始未标准化数值）。

    参数
    ----
    raw : (N, 50, D)
    eps : 行 L1 范数 > eps 视为有效行；默认严格非零。
    """
    l1 = np.abs(np.asarray(raw)).sum(axis=-1)
    return l1 > eps


def validity_from_masks(observed: Dict[str, np.ndarray], min_content: Dict[str, int] | None = None):
    """由可观测掩码推回 v（当前输入能推断的有效性）。"""
    min_content = min_content or {}
    out = {}
    for m in MODALITIES:
        o = np.asarray(observed[m]) > 0
        out[m] = o if min_content.get(m, 1) <= 1 else (o.sum(axis=-1) >= min_content[m])[..., None] & o
    return out


# --------------------------------------------------------------------------------------
# 区间起点
# --------------------------------------------------------------------------------------
def choose_span_start(n_valid: int, span: int, origin: str, rng: np.random.Generator) -> int:
    """在 [0, n_valid-span] 内选起点，origin ∈ {start, center, end, random}。"""
    hi = n_valid - span
    if hi <= 0:
        return 0
    if origin == "start":
        base = 0
    elif origin == "center":
        base = hi // 2
    elif origin == "end":
        base = hi
    else:
        return int(rng.integers(0, hi + 1))
    # 在基准附近轻微抖动，避免所有视图完全相同；抖动范围不超过可用区间
    jitter = int(rng.integers(-max(1, (hi + 1) // 8), max(1, (hi + 1) // 8) + 1))
    return int(np.clip(base + jitter, 0, hi))


def span_length(n_valid: int, rate: float) -> int:
    """按比例计算区间长度；短样本按比例缩短，并保证至少留下一个内容位置。"""
    if n_valid <= 1:
        return 0
    span = int(round(rate * n_valid))
    span = max(1, span)
    span = min(span, n_valid - 1)  # 至少保留 1 个内容位置可观测
    return span


def build_span_mask(
    valid: np.ndarray, rate: float, origin: str, rng: np.random.Generator
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """在 valid（bool, (L,)）的有效位置内选取一段连续区间。

    返回 (masked, info)；masked 为 True 表示该位置被局部缺失。
    """
    valid = np.asarray(valid).astype(bool)
    n_valid = int(valid.sum())
    masked = np.zeros_like(valid, dtype=bool)
    idx = np.flatnonzero(valid)
    if n_valid == 0:
        return masked, {"n_valid": 0, "span": 0, "actual_rate": 0.0, "start_index": None, "starts_at": None}
    span = span_length(n_valid, rate)
    if span == 0:
        return masked, {"n_valid": n_valid, "span": 0, "actual_rate": 0.0, "start_index": int(idx[0]), "starts_at": None}
    start = choose_span_start(n_valid, span, ORIGINS.get(origin, origin), rng)
    chosen = idx[start : start + span]
    masked[chosen] = True
    starts_at = "start" if start == 0 else ("end" if start == n_valid - span else "middle")
    return masked, {
        "n_valid": n_valid,
        "span": int(span),
        "actual_rate": float(span / n_valid),
        "start_index": int(chosen[0]),
        "starts_at": starts_at,
    }


def realize_missing(
    valid: Dict[str, np.ndarray],
    spec: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """按 spec 生成可观测掩码 o。

    valid[m] 可以是单样本的 (L,) 也可以是整批的 (N, L)；
    spec = {"masked_modalities": ["T"], "rates": {"T": 0.3}, "origins": {"T": "start"}}
    """
    any_arr = next(iter(valid.values()))
    arr = np.asarray(any_arr)
    if arr.ndim >= 2:
        n = arr.shape[0]
        per_sample = {m: np.zeros(arr.shape, dtype=bool) for m in MODALITIES}
        details: List[Dict[str, Any]] = []
        for i in range(n):
            one, info = realize_missing(
                {m: np.asarray(valid[m])[i] for m in MODALITIES}, spec, rng
            )
            for m in MODALITIES:
                per_sample[m][i] = one[m]
            details.append(info)
        return per_sample, {"per_sample": details}

    masked_mods = list(spec.get("masked_modalities", []))
    rates = spec.get("rates", {})
    origins = spec.get("origins", {})
    observed: Dict[str, np.ndarray] = {}
    detail: Dict[str, Any] = {}
    for m in MODALITIES:
        v = np.asarray(valid[m]).astype(bool)
        o = v.copy()
        if m in masked_mods:
            rate = float(rates.get(m, 0.3))
            origin = origins.get(m, "random")
            span_mask, info = build_span_mask(v, rate, origin, rng)
            o = v & (~span_mask)
            info["rate_target"] = rate
            info["origin"] = origin
            detail[m] = info
        else:
            detail[m] = {"n_valid": int(v.sum()), "span": 0, "actual_rate": 0.0,
                         "rate_target": 0.0, "origin": "none", "start_index": None, "starts_at": None}
        observed[m] = o
    out_spec = {
        "masked_modalities": masked_mods,
        "rates": {m: float(rates.get(m, 0.0)) for m in MODALITIES},
        "origins": {m: origins.get(m, "none") for m in MODALITIES},
        "detail": detail,
    }
    return observed, out_spec


# --------------------------------------------------------------------------------------
# 训练用的随机缺失生成器
# --------------------------------------------------------------------------------------
@dataclass
class SpanMasker:
    """确定性的 (seed, epoch, sample_index) -> 缺失视图 生成器。"""

    seed: int
    clean_probability: float = 0.25
    mask_rates: Sequence[float] = (0.10, 0.30, 0.50, 0.70)
    single_modality_probability: float = 0.70
    epochs: int = 40

    def _rng(self, epoch: int, sample_index: int) -> np.random.Generator:
        return np_rng_for("trainmask", self.seed, epoch, sample_index)

    def view_id(self, epoch: int, sample_index: int) -> str:
        return f"trainmask|seed={self.seed}|epoch={epoch}|idx={sample_index}"

    def sample(self, valid: Dict[str, np.ndarray], epoch: int, sample_index: int):
        """返回 (observed, meta)。clean 视图返回原样掩码。"""
        rng = self._rng(epoch, sample_index)
        if rng.random() < self.clean_probability:
            observed = {m: np.asarray(valid[m]).astype(bool).copy() for m in MODALITIES}
            meta = {
                "kind": "clean",
                "masked_modalities": [],
                "rates": {m: 0.0 for m in MODALITIES},
                "origins": {m: "none" for m in MODALITIES},
                "detail": {},
                "view_id": f"clean|seed={self.seed}|epoch={epoch}|idx={sample_index}",
            }
            return observed, meta

        available = [m for m in MODALITIES if np.asarray(valid[m]).sum() > 1]
        if not available:
            observed = {m: np.asarray(valid[m]).astype(bool).copy() for m in MODALITIES}
            meta = {"kind": "clean_fallback", "masked_modalities": [], "rates": {m: 0.0 for m in MODALITIES},
                    "origins": {m: "none" for m in MODALITIES}, "detail": {},
                    "view_id": f"cleanfb|seed={self.seed}|epoch={epoch}|idx={sample_index}"}
            return observed, meta

        n_mask = 1 if rng.random() < self.single_modality_probability else 2
        n_mask = min(n_mask, len(available))
        # 只用整数抽样决定模态组合，避免 numpy 按 dtype 分派导致的随机流差异
        picked: List[str] = []
        pool = list(available)
        for _ in range(n_mask):
            j = int(rng.integers(0, len(pool)))
            picked.append(pool.pop(j))
        chosen = picked

        rates, origins = {}, {}
        for m in chosen:
            rates[m] = float(self.mask_rates[int(rng.integers(0, len(self.mask_rates)))])
            origins[m] = POSITIONS[int(rng.integers(0, len(POSITIONS)))]

        # 把 rng 状态固定下来：realize_missing 内部再次使用同一个流
        observed, spec = realize_missing(
            valid,
            {"masked_modalities": chosen, "rates": rates, "origins": origins},
            rng,
        )
        spec["kind"] = "missing"
        spec["view_id"] = f"mask|seed={self.seed}|epoch={epoch}|idx={sample_index}"
        return observed, spec


# --------------------------------------------------------------------------------------
# 固定验证缺失图谱
# --------------------------------------------------------------------------------------
def fixed_view_specs(
    views_per_condition: int = 3,
    dual_pairs: Optional[Sequence[Sequence[str]]] = None,
    dual_views: int = 3,
    rates: Sequence[float] = (0.10, 0.30, 0.50, 0.70),
    positions: Sequence[str] = POSITIONS,
) -> List[Dict[str, Any]]:
    """生成验证集固定缺失图谱的说明列表（不含具体掩码，仅条件）。"""
    specs: List[Dict[str, Any]] = []

    # 完整视图
    for k in range(views_per_condition):
        specs.append(
            {
                "view_id": f"valid|clean|k{k}",
                "kind": "clean",
                "masked_modalities": [],
                "rates": {},
                "origins": {},
                "condition": "clean",
                "rate": 0.0,
                "position": "none",
                "repeat": k,
            }
        )

    # 单模态 × 位置 × 比例
    for m in MODALITIES:
        for pos in positions:
            for rate in rates:
                for k in range(views_per_condition):
                    specs.append(
                        {
                            "view_id": f"valid|{m}|{pos}|{int(round(rate * 100))}|k{k}",
                            "kind": "missing",
                            "masked_modalities": [m],
                            "rates": {m: float(rate)},
                            "origins": {m: pos},
                            "condition": f"{m}-{pos}",
                            "rate": float(rate),
                            "position": pos,
                            "repeat": k,
                        }
                    )

    # 双模态缺失（补充组）
    if dual_pairs is None:
        dual_pairs = (("T", "A"), ("T", "V"), ("A", "V"))
    for pair in dual_pairs:
        pair = list(pair)
        for rate in (0.30, 0.50):
            for k in range(dual_views):
                specs.append(
                    {
                        "view_id": f"valid|dual-{'-'.join(pair)}|{int(round(rate * 100))}|k{k}",
                        "kind": "missing",
                        "masked_modalities": pair,
                        "rates": {m: float(rate) for m in pair},
                        "origins": {m: "random" for m in pair},
                        "condition": f"dual-{'-'.join(pair)}",
                        "rate": float(rate),
                        "position": "random",
                        "repeat": k,
                    }
                )
    return specs


def build_validation_plan(
    valid: Dict[str, np.ndarray],
    n_samples: int,
    seed: int = 20240924,
    views_per_condition: int = 3,
    dual_pairs: Optional[Sequence[Sequence[str]]] = None,
    dual_views: int = 3,
    store_observed: bool = False,
) -> Dict[str, Any]:
    """为验证集构建固定的缺失视图（所有模型共用同一批视图）。

    每个视图的随机数流由 `stable_seed("validview", seed, view_id)` 决定，并按样本序号
    0..N-1 顺序消耗，因此掩码是 (seed, view_id, 样本序号) 的纯函数，可以随时确定性重建。
    默认不把逐样本掩码写进 JSON（728 样本 × 129 视图 ≈ 184 MB），改为用
    `materialize_view_observed` 在需要时重建；需要离线核对时可用 store_observed=True。
    """
    specs = fixed_view_specs(views_per_condition, dual_pairs, dual_views)
    # 序列长度由数据决定（本赛题为 50），不写死，便于在其它长度上复用
    length = max(int(np.asarray(valid[m]).shape[-1]) for m in MODALITIES)
    views: List[Dict[str, Any]] = []
    for spec in specs:
        rng = np_rng_for("validview", seed, spec["view_id"])
        observed_all = {m: np.zeros((n_samples, length), dtype=np.uint8) for m in MODALITIES}
        actual_rates = {m: [] for m in MODALITIES}
        for i in range(n_samples):
            if spec["kind"] == "clean":
                for m in MODALITIES:
                    observed_all[m][i] = np.asarray(valid[m][i]).astype(np.uint8)
                continue
            obs, info = realize_missing({m: valid[m][i] for m in MODALITIES}, spec, rng)
            for m in MODALITIES:
                observed_all[m][i] = obs[m].astype(np.uint8)
                if m in spec["masked_modalities"]:
                    actual_rates[m].append(info["detail"][m]["actual_rate"])
        entry = {
            "view_id": spec["view_id"],
            "condition": spec["condition"],
            "kind": spec["kind"],
            "masked_modalities": spec["masked_modalities"],
            "rates": spec["rates"],
            "origins": spec["origins"],
            "rate": spec["rate"],
            "position": spec["position"],
            "repeat": spec["repeat"],
            "mean_actual_rate": {
                m: (float(np.mean(actual_rates[m])) if actual_rates[m] else 0.0) for m in MODALITIES
            },
            "observed": (
                None
                if (spec["kind"] == "clean" or not store_observed)
                else {m: observed_all[m].reshape(-1).tolist() for m in MODALITIES}
            ),
        }
        views.append(entry)
    return {
        "seed": int(seed),
        "n_samples": int(n_samples),
        "views_per_condition": int(views_per_condition),
        "generator": "src.problem2.masks.build_validation_plan",
        "observed_stored": bool(store_observed),
        "regeneration": "materialize_view_observed(valid, view, seed) 逐样本重放同一随机流",
        "views": views,
    }


def materialize_view_observed(
    valid: Dict[str, np.ndarray], view: Dict[str, Any], seed: int, n_samples: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """确定性重放某个视图的逐样本掩码（与 build_validation_plan 完全一致）。"""
    if view.get("observed") is not None:
        n = int(np.asarray(view["observed"]["T"]).size // max(1, len(np.asarray(valid["T"])[0])))
        return {
            m: np.asarray(view["observed"][m], dtype=np.uint8).reshape(n, -1).astype(bool)
            for m in MODALITIES
        }
    arr = np.asarray(valid["T"])
    n = int(arr.shape[0]) if n_samples is None else int(n_samples)
    length = int(arr.shape[-1])
    if view["kind"] == "clean":
        return {m: np.asarray(valid[m]).astype(bool).copy() for m in MODALITIES}
    rng = np_rng_for("validview", seed, view["view_id"])
    out = {m: np.zeros((n, length), dtype=bool) for m in MODALITIES}
    for i in range(n):
        obs, _ = realize_missing({m: np.asarray(valid[m])[i] for m in MODALITIES}, view, rng)
        for m in MODALITIES:
            out[m][i] = obs[m]
    return out


def plan_view_observed(plan_entry: Dict[str, Any], valid: Dict[str, np.ndarray], n_samples: int):
    """从视图条目恢复 observed 掩码；JSON 中未存掩码时返回 None。"""
    if plan_entry.get("observed") is not None:
        length = int(np.asarray(valid["T"]).shape[-1])
        return {
            m: np.asarray(plan_entry["observed"][m], dtype=np.uint8).reshape(n_samples, length).astype(bool)
            for m in MODALITIES
        }
    return None


def view_observed_from_plan(
    plan: Dict[str, Any], view: Dict[str, Any], valid: Dict[str, np.ndarray], n_samples: int
) -> Dict[str, np.ndarray]:
    """统一的视图掩码获取入口：优先用 JSON 中存储的掩码，否则按 seed 确定性重建。"""
    stored = plan_view_observed(view, valid, n_samples)
    if stored is not None:
        return stored
    return materialize_view_observed(valid, view, int(plan.get("seed", 20240924)), n_samples)


# --------------------------------------------------------------------------------------
# 把缺失应用到模型输入
# --------------------------------------------------------------------------------------
def apply_missing_to_text(
    text_bert: np.ndarray,
    observed_text: np.ndarray,
    pad_id: int = 0,
) -> np.ndarray:
    """在进入 BERT 之前把不可观测的内容 token 设为 [PAD]、mask=0、token_type=0。

    text_bert: (B, 3, 50) int64
    observed_text: (B, 50) bool，内容 token 是否可观测（特殊 token 位置应为 False）
    特殊 token（[CLS]/[SEP]）保留原样。
    """
    out = np.array(text_bert, dtype=np.int64, copy=True)
    obs = np.asarray(observed_text).astype(bool)
    special = text_special_token_mask(out[:, 0, :])
    hide = (~obs) & (~special)
    out[:, 0, :][hide] = pad_id
    out[:, 1, :][hide] = 0
    out[:, 2, :][hide] = 0
    return out


def apply_missing_to_features(
    features: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    """标准化之后再次把不可观测行置零。"""
    return np.asarray(features, dtype=np.float32) * np.asarray(observed, dtype=np.float32)[..., None]
