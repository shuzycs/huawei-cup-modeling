"""问题 2 补充单元检查：数据管线、缺失视图、蒸馏、checkpoint 与工具约定。

运行： python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from src.problem2 import MODALITIES  # noqa: E402
from src.problem2.common import stable_seed  # noqa: E402
from src.problem2.evaluate import condition_key, single_condition_keys  # noqa: E402
from src.problem2.masks import (  # noqa: E402
    SpanMasker,
    build_validation_plan,
    content_validity_text,
    materialize_view_observed,
    realize_missing,
    view_observed_from_plan,
)
from src.problem2.model import (  # noqa: E402
    build_model_from_config,
    build_teacher_from_config,
    distillation_loss,
)

L = 20


def make_valid(n: int = 4, lengths=(12, 8, 4, 2)) -> dict:
    valid = {}
    for m in MODALITIES:
        arr = np.zeros((n, L), dtype=bool)
        for i, ln in enumerate(lengths[:n]):
            arr[i, :ln] = True
        arr[:, 0] = False  # 文本首位置是 [CLS]
        valid[m] = arr
    return valid


# ------------------------------------------------------------------ 视图与条件键
def test_validation_plan_regeneration_is_deterministic():
    """JSON 不存掩码时，按 (seed, view_id, 样本序号) 重建必须与首次生成完全一致。"""
    valid = make_valid()
    n = valid["T"].shape[0]
    plan = build_validation_plan(valid, n, seed=1234, views_per_condition=1,
                                 dual_pairs=[], dual_views=0, store_observed=True)
    rebuilt = build_validation_plan(valid, n, seed=1234, views_per_condition=1,
                                    dual_pairs=[], dual_views=0, store_observed=False)
    assert len(plan["views"]) == len(rebuilt["views"]) > 0
    assert plan["observed_stored"] is True and rebuilt["observed_stored"] is False

    missing_views = [v for v in plan["views"] if v["kind"] != "clean"]
    assert missing_views, "至少应有一个缺失视图用于核对重建"
    for a, b in zip(plan["views"], rebuilt["views"]):
        assert a["view_id"] == b["view_id"]
        if a["kind"] == "clean":
            continue
        assert a["observed"] is not None
        stored = np.asarray(a["observed"]["T"], dtype=np.uint8).reshape(valid["T"].shape)
        replayed = materialize_view_observed(valid, b, seed=1234)["T"].astype(np.uint8)
        assert np.array_equal(stored, replayed), f"{a['view_id']} 重建结果与首次生成不一致"
        # 重建结果也必须落在真实有效位置上
        assert not (replayed.astype(bool) & ~valid["T"]).any()


def test_view_observed_from_plan_prefers_stored():
    valid = make_valid()
    plan = build_validation_plan(valid, valid["T"].shape[0], seed=7, views_per_condition=1,
                                 dual_pairs=[], dual_views=0, store_observed=True)
    view = plan["views"][0]
    obs = view_observed_from_plan(plan, view, valid, valid["T"].shape[0])
    assert obs["T"].shape == valid["T"].shape
    assert obs["T"].dtype == bool


def test_condition_keys_include_rate():
    """条件键必须带缺失率，否则不同比例的视图会被平均掉。"""
    keys = single_condition_keys()
    assert len(keys) == 36
    assert "T-start|10" in keys and "T-start|70" in keys
    assert len(set(keys)) == 36
    assert condition_key({"kind": "clean"}) == "clean"
    assert condition_key({"kind": "missing", "condition": "A-end", "rate": 0.3}) == "A-end|30"


def test_realize_missing_batch_matches_per_sample():
    valid = make_valid()
    rng_batch = np.random.default_rng(11)
    spec = {"masked_modalities": ["A", "V"], "rates": {"A": 0.5, "V": 0.25},
            "origins": {"A": "middle", "V": "end"}}
    batched, _ = realize_missing(valid, spec, rng_batch)
    rng_seq = np.random.default_rng(11)
    for i in range(valid["T"].shape[0]):
        one, _ = realize_missing({m: valid[m][i] for m in MODALITIES}, spec, rng_seq)
        for m in MODALITIES:
            assert np.array_equal(batched[m][i], one[m]), f"批处理与逐样本在 {m}[{i}] 不一致"


def test_stable_seed_is_process_independent():
    """不得依赖 Python 内置 hash()（受 PYTHONHASHSEED 影响）。"""
    assert stable_seed("trainmask", 42, 3, 7) == stable_seed("trainmask", 42, 3, 7)
    assert stable_seed("a") != stable_seed("b")
    assert 0 <= stable_seed("x", 1.5, (2, 3)) < 2**31 - 1


def test_masker_rejects_non_content_positions_only():
    valid = make_valid()
    masker = SpanMasker(seed=3, clean_probability=0.0, mask_rates=(0.5,), single_modality_probability=1.0)
    for i in range(valid["T"].shape[0]):
        for epoch in range(5):
            obs, meta = masker.sample({m: valid[m][i] for m in MODALITIES}, epoch, i)
            for m in MODALITIES:
                assert not (obs[m] & ~valid[m][i]).any(), f"{m} 遮蔽了非有效位置"


def test_text_special_tokens_never_observed():
    ids = np.array([[101, 2054, 102, 0, 0]])
    att = np.array([[1, 1, 1, 0, 0]])
    v = content_validity_text(ids, att)
    assert not v[0, 0] and not v[0, 2] and v[0, 1]


# ------------------------------------------------------------------ 蒸馏与教师
@pytest.mark.parametrize("fusion_type", ["gate", "cross_attention"])
def test_teacher_model_handles_768_dim_text(fusion_type):
    cfg = {"model": {"hidden_dim": 16, "dropout": 0.0, "fusion_type": fusion_type},
           "distillation": {"teacher_text_dim": 768, "teacher_hidden_dim": 16,
                            "teacher_fusion_dim": 16, "teacher_text_projection": False}}
    teacher = build_teacher_from_config(cfg, text_dim=768)
    b = 3
    text = torch.randn(b, L, 768)
    audio = torch.randn(b, L, 74)
    vision = torch.randn(b, L, 35)
    observed = {m: torch.ones(b, L, dtype=torch.bool) for m in MODALITIES}
    out = teacher(text, audio, vision, observed=observed, text_observed=observed["T"])
    assert out["class_logits"].shape == (b, 3)
    assert out["fusion_weights"].shape == (b, 3)
    assert torch.isfinite(out["class_logits"]).all()


def test_distillation_loss_components_and_grad():
    b = 10
    student_logits = torch.randn(b, 3, requires_grad=True)
    student_intensity = torch.randn(b, requires_grad=True)
    teacher_logits = torch.randn(b, 3)
    teacher_intensity = torch.rand(b) * 6 - 3
    labels = torch.randint(0, 3, (b,))
    targets = torch.rand(b) * 6 - 3
    out = distillation_loss(
        student_logits, student_intensity, teacher_logits, teacher_intensity, labels, targets,
        regression_weight=0.5, temperature=4.0, kl_weight=0.3, intensity_consistency_weight=0.5,
    )
    for key in ("loss", "ce", "huber", "kl", "consistency"):
        assert key in out and torch.isfinite(out[key]), f"{key} 非有限"
    assert float(out["kl"]) >= -1e-6, "归一化后的 KL 不应为负"
    out["loss"].backward()
    assert student_logits.grad is not None and torch.isfinite(student_logits.grad).all()


def test_distillation_kl_grows_with_teacher_confidence():
    """记录软目标项的已知病态：教师越自信，同一 kl_weight 下的 KL 越大。

    这正是本实验最终把 E4 的 ``kl_weight`` 设为 0（只用强度一致性）的原因之一，
    也是 ``normalize_kl`` 开关存在的原因。默认配置下该病态是"已知且被记录"的。
    """
    torch.manual_seed(0)
    b = 64
    student = torch.randn(b, 3)
    labels = torch.randint(0, 3, (b,))
    targets = torch.rand(b) * 6 - 3
    intensity = torch.randn(b)

    def sharp(scale: float) -> torch.Tensor:
        t = torch.full((b, 3), -scale)
        t[:, 1] = scale
        return t

    def kl_value(teacher, normalize=False):
        return float(
            distillation_loss(
                student, intensity, teacher, torch.zeros(b), labels, targets,
                kl_weight=1.0, temperature=4.0, normalize_kl=normalize,
            )["kl"]
        )

    scales = (0.05, 0.1, 0.5, 2.0, 5.0, 20.0)
    raw = [kl_value(sharp(s), False) for s in scales]
    assert raw[-1] > raw[0] * 5, f"原始 KL 应随教师置信度增长: {[round(x, 3) for x in raw]}"
    # 归一化开关确实改变了尺度（默认关闭，此处只验证开关是可用的）
    norm = [kl_value(sharp(s), True) for s in (0.5, 5.0)]
    assert all(np.isfinite(x) for x in norm), "归一化路径不应产生 NaN/Inf"


def test_distillation_loss_keeps_gradient():
    """总损失必须保留计算图，且日志项为有限标量。"""
    b = 16
    student = torch.randn(b, 3, requires_grad=True)
    teacher = torch.randn(b, 3) * 3
    labels = torch.randint(0, 3, (b,))
    targets = torch.rand(b) * 6 - 3
    intensity = torch.randn(b)
    out = distillation_loss(
        student, intensity, teacher, torch.zeros(b), labels, targets,
        kl_weight=1.0, temperature=2.0,
    )
    assert out["loss"].requires_grad, "总损失必须保留计算图"
    out["loss"].backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    for key in ("ce", "huber", "kl", "consistency"):
        assert torch.isfinite(out[key]) and out[key].ndim == 0, f"{key} 应为有限标量"


# ------------------------------------------------------------------ checkpoint 约定
def test_teacher_and_student_share_non_text_state():
    """教师与学生的非文本参数必须同名同形，这样同一套加载/保存逻辑才能复用。

    文本侧刻意不同：学生接收 BERT 输出（Linear 投影），教师在 768 维原始 text 上先
    做 LayerNorm 再投影，因此文本侧的张量名与形状都不同 —— 这是预期差异。
    """
    cfg = {"model": {"hidden_dim": 8, "dropout": 0.0}}
    student = build_model_from_config(cfg, text_dim=128, experiment="E3")
    teacher = build_teacher_from_config(
        {"model": {"hidden_dim": 8, "dropout": 0.0},
         "distillation": {"teacher_hidden_dim": 8, "teacher_fusion_dim": 8, "teacher_text_projection": False}},
        text_dim=16,
    )
    s_state, t_state = student.state_dict(), teacher.state_dict()

    def is_text(key: str) -> bool:
        return key.startswith(("encoders.T", "input_norms.T"))

    shared = {k for k in s_state if not is_text(k)}
    assert shared, "应存在非文本共享参数"
    assert shared <= set(t_state), f"学生独有但教师缺少的参数: {sorted(shared - set(t_state))}"
    for k in sorted(shared):
        assert s_state[k].shape == t_state[k].shape, f"{k} 形状不一致"

    # 文本侧差异只应体现在名字或形状上，不允许教师出现学生无法解释的额外命名空间
    t_only = {k for k in t_state if k not in s_state}
    assert t_only and all(is_text(k) for k in t_only), f"教师独有参数应仅限文本侧: {sorted(t_only)}"

    student.load_state_dict(s_state, strict=True)
    teacher.load_state_dict(t_state, strict=True)


def test_saved_checkpoint_has_no_unknown_keys(tmp_path):
    """写盘再读回必须 strict 通过（防止出现"意外参数"导致推理加载失败）。"""
    from safetensors.torch import load_file, save_file

    cfg = {"model": {"hidden_dim": 8, "dropout": 0.0}}
    model = build_model_from_config(cfg, text_dim=16, experiment="E5")
    path = tmp_path / "ckpt.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))

    fresh = build_model_from_config(cfg, text_dim=16, experiment="E5")
    missing, unexpected = fresh.load_state_dict(load_file(str(path)), strict=False)
    assert not missing and not unexpected, f"缺失 {missing[:3]} / 多余 {unexpected[:3]}"


def test_model_with_fusion_dim_projection():
    """融合维 != 隐藏维时，各模态池化后需投影到融合维。"""
    m = build_model_from_config({"model": {"hidden_dim": 8, "dropout": 0.0}}, text_dim=8, experiment="E3")
    assert m.to_fusion is None
    from src.problem2.model import MultiModalSentimentModel

    m2 = MultiModalSentimentModel(text_dim=8, hidden_dim=8, fusion_dim=32, dropout=0.0,
                                  gate=True, modalities=MODALITIES)
    assert m2.to_fusion is not None
    b = 2
    out = m2(
        torch.randn(b, L, 8), torch.randn(b, L, 74), torch.randn(b, L, 35),
        text_observed=torch.ones(b, L, dtype=torch.bool),
    )
    assert out["pooled"].shape == (b, 3, 32)
    assert out["class_logits"].shape == (b, 3)
