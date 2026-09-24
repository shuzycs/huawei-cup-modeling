"""问题 2 单元检查（pytest）。

覆盖手册要求的掩码、文本遮蔽、池化、融合权重与端到端前反向检查。
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
from src.problem2.masks import (  # noqa: E402
    SpanMasker,
    apply_missing_to_features,
    apply_missing_to_text,
    build_span_mask,
    content_validity_features,
    content_validity_text,
    realize_missing,
)
from src.problem2.model import (  # noqa: E402
    MaskedAttentionPool,
    MultiModalSentimentModel,
    build_model_from_config,
    loss_function,
)

L = 50
H = 16


def make_valid(n: int = 6, lengths: tuple = (40, 20, 12, 6, 2, 50)) -> dict:
    valid = {}
    for m in MODALITIES:
        arr = np.zeros((n, L), dtype=bool)
        for i, ln in enumerate(lengths[:n]):
            arr[i, :ln] = True
        if m == "T":  # 文本内容 token 不含 [CLS]/[SEP] 位置
            arr = arr.copy()
            arr[:, 0] = False
        valid[m] = arr
    return valid


# ------------------------------------------------------------------ 掩码
def test_masks_only_hit_valid_positions():
    valid = make_valid()
    rng = np.random.default_rng(0)
    for m in MODALITIES:
        for rate in (0.1, 0.3, 0.5, 0.7):
            masked, info = build_span_mask(valid[m][0], rate, "middle", rng)
            assert not (masked & ~valid[m][0]).any(), "遮蔽了非有效位置"
            assert info["span"] >= 0


def test_masked_span_is_contiguous():
    valid = make_valid()
    rng = np.random.default_rng(1)
    for rate in (0.1, 0.3, 0.5, 0.7):
        for origin in ("start", "middle", "end"):
            masked, info = build_span_mask(valid["A"][0], rate, origin, rng)
            idx = np.flatnonzero(masked)
            if idx.size:
                assert np.array_equal(idx, np.arange(idx[0], idx[0] + idx.size)), "缺失区间不连续"
                assert idx.size == info["span"]
                assert idx.size <= valid["A"][0].sum() - 1, "至少应保留一个内容位置可观测"


def test_at_least_one_content_position_remains():
    valid = make_valid()
    rng = np.random.default_rng(2)
    for i in range(valid["A"].shape[0]):
        for rate in (0.1, 0.3, 0.5, 0.7, 0.99):
            masked, _ = build_span_mask(valid["A"][i], rate, "middle", rng)
            remaining = int((valid["A"][i] & ~masked).sum())
            n_valid = int(valid["A"][i].sum())
            if n_valid >= 1:
                assert remaining >= 1


def test_masker_labels_unchanged_and_unselected_modalities_untouched():
    valid = make_valid()
    masker = SpanMasker(seed=7, clean_probability=0.0, mask_rates=(0.5,), single_modality_probability=1.0)
    for idx in range(valid["T"].shape[0]):
        seen = set()
        for epoch in range(6):
            obs, meta = masker.sample({m: valid[m][idx] for m in MODALITIES}, epoch, idx)
            seen.add(tuple(meta["masked_modalities"]))
            for m in MODALITIES:
                if m not in meta["masked_modalities"]:
                    assert np.array_equal(obs[m], valid[m][idx]), "未选模态被改动"
                else:
                    assert not (obs[m] & ~valid[m][idx]).any(), "遮蔽了非有效位置"
        assert seen, "应当至少产生一个缺失视图"


def test_masker_is_deterministic():
    valid = make_valid()
    m1 = SpanMasker(seed=11, clean_probability=0.2)
    m2 = SpanMasker(seed=11, clean_probability=0.2)
    for i in range(valid["T"].shape[0]):
        for e in (0, 3):
            o1, meta1 = m1.sample({m: valid[m][i] for m in MODALITIES}, e, i)
            o2, meta2 = m2.sample({m: valid[m][i] for m in MODALITIES}, e, i)
            assert meta1["view_id"] == meta2["view_id"]
            for m in MODALITIES:
                assert np.array_equal(o1[m], o2[m])


def test_masks_and_zero_rows_consistent():
    """掩码与置零必须一致：不可观测位置在输入特征上必须为零。"""
    valid = make_valid()
    rng = np.random.default_rng(3)
    feats = rng.normal(size=(1, L, 5)).astype(np.float32)
    obs, _ = realize_missing(
        {m: valid[m][0] for m in MODALITIES},
        {"masked_modalities": ["A"], "rates": {"A": 0.5}, "origins": {"A": "middle"}},
        rng,
    )
    o = obs["A"]
    out = apply_missing_to_features(feats, obs["A"])
    assert np.array_equal(out[0][~o], np.zeros_like(feats[0][~o]))
    keep = o & valid["A"][0]
    assert np.array_equal(out[0][keep], feats[0][keep])


def test_realize_missing_supports_batched_valid():
    """整批 (N, L) 输入必须逐样本处理，且不越界。"""
    valid = make_valid()
    rng = np.random.default_rng(5)
    obs, info = realize_missing(
        valid,
        {"masked_modalities": ["A"], "rates": {"A": 0.5}, "origins": {"A": "middle"}},
        rng,
    )
    for m in MODALITIES:
        assert obs[m].shape == valid[m].shape
        assert not (obs[m] & ~valid[m]).any()
    assert len(info["per_sample"]) == valid["A"].shape[0]


def test_text_validity_excludes_special_tokens():
    ids = np.array([[101, 2054, 2003, 102, 0, 0]])
    att = np.array([[1, 1, 1, 1, 0, 0]])
    v = content_validity_text(ids, att)
    assert v.tolist() == [[False, True, True, False, False, False]]


def test_text_masking_applied_before_bert():
    text_bert = np.array([[[101, 2054, 2003, 102, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 0]]], dtype=np.int64)
    observed = np.array([[False, True, False, False, False]])
    out = apply_missing_to_text(text_bert, observed)
    assert out[0, 0, 0] == 101 and out[0, 0, 1] == 2054 and out[0, 0, 3] == 102, "特殊 token 应保留"
    assert out[0, 0, 2] == 0, "被遮蔽内容 token 应置为 [PAD]"
    assert out[0, 1, 2] == 0 and out[0, 2, 2] == 0
    assert out[0, 1, 1] == 1, "可观测 token 的 attention mask 不应被改"


def test_feature_validity_detects_zero_rows():
    raw = np.zeros((1, L, 3))
    raw[0, :5] = 1.0
    v = content_validity_features(raw)
    assert v[0, :5].all() and not v[0, 5:].any()


# ------------------------------------------------------------------ 模型
def test_pooling_all_masked_returns_zero_without_nan():
    pool = MaskedAttentionPool(H)
    x = torch.randn(2, L, H)
    observed = torch.zeros(2, L, dtype=torch.bool)
    pooled, w, available = pool(x, observed)
    assert not available.any()
    assert torch.isfinite(pooled).all()
    assert torch.allclose(pooled, torch.zeros_like(pooled))
    assert torch.allclose(w, torch.zeros_like(w))


def test_pooling_masked_weights_are_zero():
    pool = MaskedAttentionPool(H)
    x = torch.randn(3, L, H)
    observed = torch.zeros(3, L, dtype=torch.bool)
    observed[:, :7] = True
    _, w, _ = pool(x, observed)
    assert torch.allclose(w[:, 7:], torch.zeros(3, L - 7))
    assert torch.allclose(w.sum(dim=1), torch.ones(3))


def test_mean_pool_matches_manual_average():
    from src.problem2.model import MeanPool

    pool = MeanPool(H)
    x = torch.randn(2, L, H)
    x[0, 10:] = 0
    observed = torch.zeros(2, L, dtype=torch.bool)
    observed[0, :10] = True
    observed[1, :] = True
    pooled, w, available = pool(x * observed.unsqueeze(-1).float(), observed)
    manual0 = x[0, :10].mean(dim=0)
    manual1 = x[1].mean(dim=0)
    assert torch.allclose(pooled[0], manual0, atol=1e-5)
    assert torch.allclose(pooled[1], manual1, atol=1e-5)
    assert available.all()


@pytest.mark.parametrize("experiment", ["E0", "E1", "E2", "E3"])
def test_forward_shapes_and_constraints(experiment):
    cfg = {"model": {"hidden_dim": H, "dropout": 0.2}}
    model = build_model_from_config(cfg, text_dim=8, experiment=experiment)
    b = 4
    text = torch.randn(b, L, 8)
    audio = torch.randn(b, L, 74)
    vision = torch.randn(b, L, 35)
    observed = {
        "T": torch.zeros(b, L, dtype=torch.bool),
        "A": torch.zeros(b, L, dtype=torch.bool),
        "V": torch.zeros(b, L, dtype=torch.bool),
    }
    observed["T"][:, 1:11] = True
    observed["A"][:, :5] = True
    observed["V"][:, :0] = True  # 视觉完全缺失
    audio = audio * observed["A"].unsqueeze(-1).float()
    vision = vision * observed["V"].unsqueeze(-1).float()
    text = text * observed["T"].unsqueeze(-1).float()

    out = model(text, audio, vision, observed=observed, text_observed=observed["T"])
    assert out["class_logits"].shape == (b, 3)
    assert out["intensity"].shape == (b,)
    assert out["fusion_weights"].shape == (b, len(model.modalities))
    assert torch.isfinite(out["class_logits"]).all()
    assert torch.isfinite(out["intensity"]).all()
    assert (out["intensity"].abs() <= 3.0 + 1e-5).all(), "强度必须落在 [-3,3]"
    w = out["fusion_weights"]
    assert (w >= -1e-6).all()
    assert torch.allclose(w.sum(dim=1), torch.ones(b), atol=1e-5), "融合权重和应为 1"
    # 完全缺失的模态权重必须为 0
    if "V" in model.modalities:
        vi = model.modalities.index("V")
        assert torch.allclose(w[:, vi], torch.zeros(b), atol=1e-6)


def test_backward_has_finite_gradients():
    cfg = {"model": {"hidden_dim": H, "dropout": 0.2}}
    model = build_model_from_config(cfg, text_dim=8, experiment="E3")
    b = 6
    observed = {m: torch.zeros(b, L, dtype=torch.bool) for m in MODALITIES}
    observed["T"][:, 1:12] = True
    observed["A"][:, :6] = True
    observed["V"][:, 6:12] = True
    text = torch.randn(b, L, 8) * observed["T"].unsqueeze(-1).float()
    audio = torch.randn(b, L, 74) * observed["A"].unsqueeze(-1).float()
    vision = torch.randn(b, L, 35) * observed["V"].unsqueeze(-1).float()
    out = model(text, audio, vision, observed=observed)
    losses = loss_function(
        out["class_logits"], out["intensity"], torch.randint(0, 3, (b,)),
        torch.rand(b) * 6 - 3, regression_weight=0.5,
    )
    losses["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "反向传播未产生梯度"
    assert all(torch.isfinite(g).all() for g in grads), "梯度出现非有限值"


def test_cross_attention_observability_and_text_absent_fallback():
    cfg = {"model": {"hidden_dim": 8, "dropout": 0.0,
                     "fusion_type": "cross_attention", "attention_heads": 2}}
    model = build_model_from_config(cfg, text_dim=8, experiment="E3").eval()
    text = torch.randn(4, 5, 8)
    audio = torch.randn(4, 7, 74)
    vision = torch.randn(4, 4, 35)
    observed = {
        "T": torch.tensor([[0, 1, 0, 1, 0], [0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool),
        "A": torch.tensor([[1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool),
        "V": torch.tensor([[0, 0, 1, 0], [1, 0, 0, 0],
                           [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.bool),
    }
    out = model(text, audio, vision, observed=observed)
    assert torch.isfinite(out["class_logits"]).all()
    assert torch.isfinite(out["intensity"]).all()
    assert len(out["cross_attention_weights"]) == 2
    av_mask = torch.cat((observed["A"], observed["V"]), dim=1)
    for weights in out["cross_attention_weights"]:
        assert weights.shape == (4, 5, 11)
        assert (weights.masked_select(
            ~observed["T"][:, :, None].expand_as(weights)
        ) == 0).all()
        assert (weights.masked_select(~av_mask[:, None, :].expand_as(weights)) == 0).all()
    all_mask = torch.cat(tuple(observed[m] for m in MODALITIES), dim=1)
    assert (out["fusion_token_weights"][~all_mask] == 0).all()
    assert torch.allclose(out["fusion_weights"].sum(dim=1), torch.tensor([1., 1., 1., 0.]))
    assert out["fusion_weights"][1, 0] == 0
    assert out["fusion_weights"][2, 1] == 0 and out["fusion_weights"][2, 2] == 0
    assert out["fusion_weights"][1, 1:].sum() > 0

    changed_text = text.clone()
    changed_audio = audio.clone()
    changed_vision = vision.clone()
    changed_text[~observed["T"]] = 1000
    changed_audio[~observed["A"]] = 1000
    changed_vision[~observed["V"]] = 1000
    changed = model(changed_text, changed_audio, changed_vision, observed=observed)
    assert torch.allclose(out["class_logits"], changed["class_logits"], atol=1e-6)

    changed_audio = audio.clone()
    changed_audio[1, 1] += 10
    av_changed = model(text, changed_audio, vision, observed=observed)
    assert not torch.allclose(out["class_logits"][1], av_changed["class_logits"][1])


    model.train()
    trained = model(text, audio, vision, observed=observed)
    (trained["class_logits"].square().sum() + trained["intensity"].square().sum()).backward()
    assert model.cross_fusion.fusion_token.grad is not None
    assert torch.isfinite(model.cross_fusion.fusion_token.grad).all()
    grad = model.cross_fusion.cross_attention[0].in_proj_weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_audio_vision_transformer_respects_missing_positions_and_backpropagates():
    from src.problem2.model import ModalityTemporalTransformer

    encoder = ModalityTemporalTransformer(
        in_dim=74, hidden_dim=8, temporal_dim=16, layers=2, heads=4, dropout=0.0
    ).eval()
    x = torch.randn(3, 7, 74)
    observed = torch.tensor([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
    ], dtype=torch.bool)
    output = encoder(x, observed)
    assert output.shape == (3, 7, 8)
    assert torch.isfinite(output).all()
    assert (output[~observed] == 0).all()
    altered = x.clone()
    altered[~observed] = 1000
    assert torch.allclose(output, encoder(altered, observed), atol=1e-6)
    output[observed].square().sum().backward()
    assert encoder.layers[0].self_attn.in_proj_weight.grad is not None
    assert torch.isfinite(encoder.layers[0].self_attn.in_proj_weight.grad).all()

    cfg = {"model": {"hidden_dim": 8, "dropout": 0.0,
                     "temporal_encoder": "transformer", "audio_vision_dim": 16,
                     "audio_vision_layers": 2, "audio_vision_heads": 4}}
    model = build_model_from_config(cfg, text_dim=8, experiment="E3").eval()
    assert isinstance(model.encoders["A"], ModalityTemporalTransformer)
    assert isinstance(model.encoders["V"], ModalityTemporalTransformer)
    all_observed = {"T": observed, "A": observed, "V": observed}
    result = model(torch.randn(3, 7, 8), x, torch.randn(3, 7, 35),
                   observed=all_observed)
    assert torch.isfinite(result["class_logits"]).all()
    assert result["fusion_weights"][1].sum() == 0


def test_observed_stats_only_use_current_input():
    observed = torch.zeros(2, 4, 10, dtype=torch.bool)
    observed[0, 0, :3] = True
    observed[1, 1, :10] = True
    q = MultiModalSentimentModel.observed_stats(observed)
    assert q.shape == (2, 4, 2)
    assert float(q[0, 0, 0]) == 1.0 and abs(float(q[0, 0, 1]) - np.log1p(3)) < 1e-6
    assert float(q[0, 1, 0]) == 0.0


def test_safetensors_roundtrip(tmp_path):
    from safetensors.torch import load_file, save_file

    cfg = {"model": {"hidden_dim": H, "dropout": 0.0}}
    model = build_model_from_config(cfg, text_dim=8, experiment="E3")
    path = tmp_path / "m.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))
    model2 = build_model_from_config(cfg, text_dim=8, experiment="E3")
    missing, unexpected = model2.load_state_dict(load_file(str(path)), strict=False)
    assert not missing and not unexpected
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), model2.state_dict().items()):
        assert k1 == k2
        assert torch.allclose(v1, v2)
