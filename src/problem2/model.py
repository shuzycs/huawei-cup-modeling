"""统一模型：模态内时序建模 + 门控或交叉注意力融合 + 双任务输出。

接口
----
    forward(text_emb, audio, vision, observed=..., text_observed=...) ->
        {"class_logits": [B,3], "intensity": [B], "fusion_weights": [B,M],
         "available": [B,M], "pooled": [B,M,H], "token_weights": {...}}

观察到的 `observed` 掩码是唯一决定"哪些位置可观测"的输入：
不可观测位置在池化与跨模态注意力前已被置零，且其注意力权重被强制为 0。
特殊 token（[CLS]/[SEP]/[PAD]）由 dataset 在 text_observed 中标记为 False，
因此文本池化只在真实内容 token 上进行。

消融实验共用同一套数据管线与文本编码器，只切换：
    E0  文本单模态 + 掩码注意力池化
    E1  T/A/V 固定 50 位简单平均池化 + 拼接 MLP（仅完整输入训练）
    E2  同 E1，但训练时使用连续区间缺失增强
    E3  E2 + 掩码注意力池化 + 动态门控（可配置交叉注意力或音视频时序 Transformer）
    E4  E3 + 完整输入教师蒸馏
    E5  E3 + 解冻文本编码器后两层
"""

from __future__ import annotations

import math

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import MODALITIES

POOLING_MASKED_ATTENTION = "masked_attention"
POOLING_MEAN = "mean"


class MaskedAttentionPool(nn.Module):
    """把不可观测位置的注意力权重强制为 0 的注意力池化。

    若某模态没有任何可观测内容，返回全零向量且不做 softmax（避免空集合 NaN）。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(8, hidden_dim // 2), 1),
        )

    def forward(
        self, x: torch.Tensor, observed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B,L,H) 已屏蔽；observed: (B,L) bool -> (pooled (B,H), weights (B,L), available (B,))"""
        logits = self.score(x).squeeze(-1)
        mask = observed.to(dtype=torch.bool)
        available = mask.any(dim=1)
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        safe = torch.where(available[:, None], masked_logits, torch.zeros_like(masked_logits))
        weights = torch.softmax(safe, dim=1)
        weights = weights * mask.to(weights.dtype)
        pooled = torch.einsum("bl,blh->bh", weights, x)
        return pooled, weights, available


class MeanPool(nn.Module):
    """固定 50 位平均池化；零行也参与分母之前的平均（E1/E2 普通融合基线）。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(
        self, x: torch.Tensor, observed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = observed.to(dtype=x.dtype)
        denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        weights = mask / denom
        available = mask.sum(dim=1) > 0
        return pooled, weights, available


class ModalityTemporalEncoder(nn.Module):
    """线性投影 + LayerNorm + 一层核大小 3 的残差 1D 卷积；每层后屏蔽不可观测行。"""

    def __init__(self, in_dim: int, hidden_dim: int, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.observed_embed = nn.Parameter(torch.zeros(hidden_dim))
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        mask = observed.to(dtype=x.dtype).unsqueeze(-1)
        h = self.proj(x)
        h = self.norm(h)
        h = h + self.observed_embed.view(1, 1, -1)
        h = h * mask
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.norm2(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = h * mask
        return h


class FusionGate(nn.Module):
    """以 [池化向量, 可观测统计量] 为输入产生模态 logit，再在可用模态上 softmax。"""

    def __init__(self, hidden_dim: int, dropout: float = 0.2, stat_dim: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + stat_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, u: torch.Tensor, q: torch.Tensor, available: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.mlp(torch.cat([u, q], dim=-1)).squeeze(-1)  # (B,M)
        logits = logits.masked_fill(~available, float("-inf"))
        any_avail = available.any(dim=1, keepdim=True)
        safe = torch.where(any_avail, logits, torch.zeros_like(logits))
        weights = torch.softmax(safe, dim=1)
        weights = weights * available.to(weights.dtype)
        return weights, logits


def sinusoidal_positions(length: int, dim: int, reference: torch.Tensor) -> torch.Tensor:
    """Within-modality positions, without assuming aligned time steps."""
    pos = torch.arange(length, device=reference.device, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, dim, 2, device=reference.device, dtype=torch.float32)
                     * (-math.log(10000.0) / dim))
    angles = pos * freq
    encoding = torch.zeros(length, dim, device=reference.device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles[:, :dim // 2])
    return encoding.to(dtype=reference.dtype)[None]


class ModalityTemporalTransformer(nn.Module):
    """Encode observed positions within one modality before sequence pooling."""

    def __init__(self, in_dim: int, hidden_dim: int, temporal_dim: int = 256,
                 layers: int = 2, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        if temporal_dim % heads:
            raise ValueError("audio_vision_dim must be divisible by audio_vision_heads")
        if layers < 1:
            raise ValueError("audio_vision_layers must be positive")
        self.proj = nn.Linear(in_dim, temporal_dim)
        self.input_norm = nn.LayerNorm(temporal_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=temporal_dim, nhead=heads, dim_feedforward=2 * temporal_dim,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(layers)
        ])
        self.output_norm = nn.LayerNorm(temporal_dim)
        self.to_hidden = nn.Linear(temporal_dim, hidden_dim)

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        mask = observed.to(dtype=torch.bool)
        visible = mask.unsqueeze(-1)
        h = self.input_norm(self.proj(x))
        h = (h + sinusoidal_positions(x.shape[1], h.shape[2], h)) * visible
        # Attention needs one valid key even for a fully missing sequence.
        safe_mask = mask.clone()
        safe_mask[~mask.any(dim=1), 0] = True
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=~safe_mask) * visible
        return self.to_hidden(self.output_norm(h)) * visible


class CrossModalFusion(nn.Module):
    """Two text-to-AV attention blocks and one observable-only fusion token."""

    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by attention_heads={heads}")
        self.modality_embedding = nn.Parameter(torch.zeros(3, hidden_dim))
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
            for _ in range(2)
        ])
        self.cross_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
        self.cross_ff = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim))
            for _ in range(2)
        ])
        self.cross_output_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
        self.fusion_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.normal_(self.fusion_token, std=0.02)
        # No attention dropout here: returned modality masses sum to one in train mode.
        self.fusion_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=0.0, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_ff = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.fusion_output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, encoded: Dict[str, torch.Tensor], observed: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        text, audio, vision = (encoded[m] for m in ("T", "A", "V"))
        t_mask, a_mask, v_mask = (observed[m] for m in ("T", "A", "V"))
        t_pos, a_pos, v_pos = (
            sinusoidal_positions(x.shape[1], x.shape[2], x) for x in (text, audio, vision)
        )
        av_mask = torch.cat((a_mask, v_mask), dim=1)
        av = torch.cat((
            (audio + a_pos + self.modality_embedding[1]) * a_mask.unsqueeze(-1),
            (vision + v_pos + self.modality_embedding[2]) * v_mask.unsqueeze(-1),
        ), dim=1)
        # An all-masked row needs a temporary valid key to avoid NaN in softmax.
        av_available = av_mask.any(dim=1)
        safe_av_mask = av_mask.clone()
        safe_av_mask[~av_available, 0] = True
        cross_weights = []
        for attention, norm, ff, output_norm in zip(
            self.cross_attention, self.cross_norm, self.cross_ff, self.cross_output_norm
        ):
            query = (text + t_pos + self.modality_embedding[0]) * t_mask.unsqueeze(-1)
            update, weights = attention(
                query, av, av, key_padding_mask=~safe_av_mask,
                need_weights=True, average_attn_weights=True,
            )
            update = update * av_available[:, None, None].to(update.dtype)
            text = norm(text + self.dropout(update))
            text = output_norm(text + self.dropout(ff(text)))
            text = text * t_mask.unsqueeze(-1).to(text.dtype)
            weights = weights * (t_mask[:, :, None] & av_mask[:, None, :]).to(weights.dtype)
            cross_weights.append(weights)

        lengths = (text.shape[1], audio.shape[1], vision.shape[1])
        all_mask = torch.cat((t_mask, a_mask, v_mask), dim=1)
        positions = torch.cat((
            (text + t_pos + self.modality_embedding[0]) * t_mask.unsqueeze(-1),
            (audio + a_pos + self.modality_embedding[1]) * a_mask.unsqueeze(-1),
            (vision + v_pos + self.modality_embedding[2]) * v_mask.unsqueeze(-1),
        ), dim=1)
        available = all_mask.any(dim=1)
        safe_mask = all_mask.clone()
        safe_mask[~available, 0] = True
        token = self.fusion_token.expand(text.shape[0], -1, -1)
        summary, token_weights = self.fusion_attention(
            token, positions, positions, key_padding_mask=~safe_mask,
            need_weights=True, average_attn_weights=True,
        )
        summary = summary * available[:, None, None].to(summary.dtype)
        token = self.fusion_norm(token + self.dropout(summary))
        token = self.fusion_output_norm(token + self.dropout(self.fusion_ff(token)))
        token_weights = token_weights[:, 0] * all_mask.to(token_weights.dtype)
        modality_weights = torch.stack(
            [part.sum(dim=1) for part in token_weights.split(lengths, dim=1)], dim=1
        )
        return text, token[:, 0], modality_weights, cross_weights, token_weights


class MultiModalSentimentModel(nn.Module):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int = 74,
        vision_dim: int = 35,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        pooling: str = POOLING_MASKED_ATTENTION,
        gate: bool = True,
        modalities: Sequence[str] = MODALITIES,
        simple_concat: bool = False,
        num_classes: int = 3,
        fusion_dim: Optional[int] = None,
        text_projection: bool = True,
        fusion_type: str = "gate",
        attention_heads: int = 4,
        temporal_encoder: str = "conv",
        audio_vision_dim: int = 256,
        audio_vision_layers: int = 2,
        audio_vision_heads: int = 4,
    ):
        """fusion_dim 用于 E4 教师：文本侧保持原始维度（例如附件 2 的 768 维 text），
        用于融合输出投影；text_projection=False 时教师先归一化文本特征再投影到隐藏维。"""
        super().__init__()
        fusion_dim = int(fusion_dim or hidden_dim)
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim
        self.pooling_name = pooling
        self.use_gate = gate
        self.modalities: List[str] = list(modalities)
        self.simple_concat = simple_concat
        self.num_classes = num_classes
        self.text_projection = text_projection
        if fusion_type not in ("gate", "cross_attention"):
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        if fusion_type == "cross_attention" and (set(self.modalities) != set(MODALITIES) or simple_concat):
            raise ValueError("cross_attention requires T, A and V without simple_concat")
        self.fusion_type = fusion_type
        if temporal_encoder not in ("conv", "transformer"):
            raise ValueError(f"Unknown temporal_encoder: {temporal_encoder}")
        self.temporal_encoder = temporal_encoder

        encoders: Dict[str, nn.Module] = {}
        if "T" in self.modalities:
            if text_projection:
                # 学生：文本侧已经过编码器，直接投影到隐藏维
                encoders["T"] = nn.Linear(text_dim, hidden_dim)
            else:
                # 教师：在 768 维原始 text 上先归一化，再投影到隐藏维
                encoders["T"] = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim))
        encoder_cls = (ModalityTemporalTransformer if temporal_encoder == "transformer"
                       else ModalityTemporalEncoder)
        encoder_kwargs = (
            dict(temporal_dim=audio_vision_dim, layers=audio_vision_layers,
                 heads=audio_vision_heads, dropout=dropout)
            if temporal_encoder == "transformer" else dict(dropout=dropout)
        )
        if "A" in self.modalities:
            encoders["A"] = encoder_cls(audio_dim, hidden_dim, **encoder_kwargs)
        if "V" in self.modalities:
            encoders["V"] = encoder_cls(vision_dim, hidden_dim, **encoder_kwargs)
        self.encoders = nn.ModuleDict(encoders)

        # 文本侧若未投影，则在 encode 阶段已做 LayerNorm，这里不再重复归一化
        self.input_norms = nn.ModuleDict(
            {m: nn.LayerNorm(hidden_dim) for m in self.modalities if not (m == "T" and not text_projection)}
        )
        pool_cls = MaskedAttentionPool if pooling == POOLING_MASKED_ATTENTION else MeanPool
        # The fusion token supplies pooling weights in cross-attention mode.
        self.pools = (
            nn.ModuleDict({m: pool_cls(hidden_dim) for m in self.modalities})
            if fusion_type == "gate" else nn.ModuleDict()
        )
        self.to_fusion = (
            nn.ModuleDict({m: nn.Linear(hidden_dim, fusion_dim) for m in self.modalities})
            if fusion_type == "gate" and fusion_dim != hidden_dim else None
        )

        feat_dim = fusion_dim * len(self.modalities) if simple_concat else fusion_dim
        self.cross_fusion = (
            CrossModalFusion(hidden_dim, heads=attention_heads, dropout=dropout)
            if fusion_type == "cross_attention" else None
        )
        self.cross_to_fusion = (
            nn.Linear(hidden_dim, fusion_dim)
            if self.cross_fusion is not None and fusion_dim != hidden_dim else None
        )
        self.gate = FusionGate(fusion_dim, dropout) if gate and self.cross_fusion is None else None
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.regressor = nn.Linear(feat_dim, 1)

    @property
    def text_output_dim(self) -> int:
        return self.hidden_dim

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def observed_stats(observed: torch.Tensor) -> torch.Tensor:
        """(B,M,L) -> (B,M,2) = [是否至少一个可观测位置, log1p(可观测位置数)]。

        只依赖当前输入，训练与推理规则一致；不使用原始完整长度或真实缺失区间。
        """
        cnt = observed.sum(dim=-1)
        avail = (cnt > 0).to(observed.dtype)
        return torch.stack([avail, torch.log1p(cnt)], dim=-1)

    def _build_observed(
        self,
        audio: torch.Tensor,
        vision: torch.Tensor,
        observed: Optional[Dict[str, torch.Tensor]],
        text_observed: Optional[torch.Tensor],
        batch_size: int,
        length: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if observed:
            for m in self.modalities:
                if observed.get(m) is not None:
                    out[m] = observed[m].to(device=device, dtype=torch.bool)
        if "A" in self.modalities and "A" not in out:
            out["A"] = audio.abs().sum(dim=-1) > 0
        if "V" in self.modalities and "V" not in out:
            out["V"] = vision.abs().sum(dim=-1) > 0
        if "T" in self.modalities and "T" not in out:
            out["T"] = (
                text_observed.to(device=device, dtype=torch.bool)
                if text_observed is not None
                else torch.ones(batch_size, length, dtype=torch.bool, device=device)
            )
        return out

    # ------------------------------------------------------------------ 前向
    def forward(
        self,
        text_emb: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        observed: Optional[Dict[str, torch.Tensor]] = None,
        text_observed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, length = text_emb.shape[0], text_emb.shape[1]
        device = text_emb.device
        obs = self._build_observed(audio, vision, observed, text_observed, b, length, device)

        feats = {"T": text_emb, "A": audio, "V": vision}
        encoded: Dict[str, torch.Tensor] = {}
        for m in self.modalities:
            mask = obs[m]
            if m == "T":
                h = self.encoders[m](feats[m])
                if m in self.input_norms:
                    h = self.input_norms[m](h)
            else:
                h = self.encoders[m](feats[m], mask)
                h = self.input_norms[m](h)
            encoded[m] = h * mask.unsqueeze(-1).to(h.dtype)

        cross_weights = None
        fusion_token_weights = None
        if self.cross_fusion is not None:
            updated_text, cross_z, alpha, cross_weights, fusion_token_weights = (
                self.cross_fusion(encoded, obs)
            )
            encoded["T"] = updated_text

        pooled_list: List[torch.Tensor] = []
        avail_list: List[torch.Tensor] = []
        weights_map: Dict[str, torch.Tensor] = {}
        if self.cross_fusion is not None:
            lengths = [encoded[m].shape[1] for m in self.modalities]
            for i, (m, weights) in enumerate(zip(
                self.modalities, fusion_token_weights.split(lengths, dim=1)
            )):
                available = obs[m].any(dim=1)
                relative = weights / alpha[:, i:i + 1].clamp_min(1e-8)
                pooled = torch.einsum("bl,blh->bh", relative, encoded[m])
                if self.cross_to_fusion is not None:
                    pooled = self.cross_to_fusion(pooled)
                pooled = pooled * available.unsqueeze(-1).to(pooled.dtype)
                weights_map[m] = relative
                avail_list.append(available)
                pooled_list.append(pooled)
        else:
            for m in self.modalities:
                pooled, w, available = self.pools[m](encoded[m], obs[m])
                pooled = pooled * available.unsqueeze(-1).to(pooled.dtype)
                if self.to_fusion is not None:
                    pooled = self.to_fusion[m](pooled) * available.unsqueeze(-1).to(pooled.dtype)
                weights_map[m] = w
                avail_list.append(available)
                pooled_list.append(pooled)

        available_all = torch.stack(avail_list, dim=1)
        u = torch.stack(pooled_list, dim=1)
        if self.cross_fusion is not None:
            gate_logits = None
            z = self.cross_to_fusion(cross_z) if self.cross_to_fusion is not None else cross_z
        elif self.gate is not None:
            q = torch.stack([obs[m] for m in self.modalities], dim=1)
            alpha, gate_logits = self.gate(u, self.observed_stats(q), available_all)
            z = torch.einsum("bm,bmh->bh", alpha, u)
        else:
            w = available_all.to(u.dtype)
            w = w / torch.clamp(w.sum(dim=1, keepdim=True), min=1.0)
            alpha = w
            gate_logits = None
            z = torch.einsum("bm,bmh->bh", w, u)

        if self.simple_concat:
            z = torch.cat(pooled_list, dim=1)

        z = self.dropout(z)
        class_logits = self.classifier(z)
        intensity = 3.0 * torch.tanh(self.regressor(z).squeeze(-1))

        return {
            "class_logits": class_logits,
            "intensity": intensity,
            "fusion_weights": alpha,
            "available": available_all,
            "pooled": u,
            "token_weights": weights_map,
            "gate_logits": gate_logits,
            "cross_attention_weights": cross_weights,
            "fusion_token_weights": fusion_token_weights,
        }


def build_model_from_config(cfg: Dict, text_dim: int, experiment: str) -> MultiModalSentimentModel:
    """按实验编号构建模型（四个主实验共用文本编码器与数据预处理）。"""
    mcfg = cfg.get("model", {})
    hidden = int(mcfg.get("hidden_dim", 128))
    dropout = float(mcfg.get("dropout", 0.2))
    common = dict(text_dim=text_dim, hidden_dim=hidden, dropout=dropout)

    if experiment == "E0":
        return MultiModalSentimentModel(
            **common, pooling=POOLING_MASKED_ATTENTION, gate=False, modalities=("T",)
        )
    if experiment in ("E1", "E2"):
        return MultiModalSentimentModel(
            **common, pooling=POOLING_MEAN, gate=False, modalities=MODALITIES, simple_concat=True
        )
    if experiment in ("E3", "E4", "E5"):
        return MultiModalSentimentModel(
            **common, pooling=POOLING_MASKED_ATTENTION, gate=True, modalities=MODALITIES,
            fusion_type=str(mcfg.get("fusion_type", "gate")),
            attention_heads=int(mcfg.get("attention_heads", 4)),
            temporal_encoder=str(mcfg.get("temporal_encoder", "conv")),
            audio_vision_dim=int(mcfg.get("audio_vision_dim", 256)),
            audio_vision_layers=int(mcfg.get("audio_vision_layers", 2)),
            audio_vision_heads=int(mcfg.get("audio_vision_heads", 4)),
        )
    raise ValueError(f"未知实验编号: {experiment}")


def loss_function(
    class_logits: torch.Tensor,
    intensity: torch.Tensor,
    labels: torch.Tensor,
    targets: torch.Tensor,
    regression_weight: float = 0.5,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    ce = F.cross_entropy(class_logits, labels, weight=class_weights)
    huber = F.smooth_l1_loss(intensity, targets, beta=1.0)
    total = ce + regression_weight * huber
    return {"loss": total, "ce": ce.detach(), "huber": huber.detach()}


def inverse_frequency_class_weights(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """按训练集类别频率的倒数归一化（均值保持为 1），用于缓解类别不均衡。"""
    counts = torch.bincount(labels.to(torch.long), minlength=num_classes).to(torch.float32)
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    return weights / weights.mean()


# --------------------------------------------------------------------------------------
# E4：教师学生蒸馏
# --------------------------------------------------------------------------------------
def build_teacher_from_config(cfg: Dict, text_dim: int = 768) -> MultiModalSentimentModel:
    """E4 教师：与 E3 同结构、同动态门控，但文本侧使用附件 2 的预计算 text（不降维）。

    手册 §3.3 允许教师在**训练阶段**使用附件 2 的预计算 text；
    最终推理只使用能读取 text_bert/audio/vision 的学生。
    """
    dcfg = cfg.get("distillation", {})
    mcfg = cfg.get("model", {})
    hidden = int(dcfg.get("teacher_hidden_dim", mcfg.get("hidden_dim", 128)))
    dropout = float(dcfg.get("teacher_dropout", mcfg.get("dropout", 0.2)))
    return MultiModalSentimentModel(
        text_dim=text_dim,
        hidden_dim=hidden,
        dropout=dropout,
        pooling=POOLING_MASKED_ATTENTION,
        gate=True,
        modalities=MODALITIES,
        fusion_dim=int(dcfg.get("teacher_fusion_dim", text_dim)),
        text_projection=bool(dcfg.get("teacher_text_projection", False)),
        fusion_type=str(dcfg.get("teacher_fusion_type", mcfg.get("fusion_type", "gate"))),
        attention_heads=int(mcfg.get("attention_heads", 4)),
        temporal_encoder=str(mcfg.get("temporal_encoder", "conv")),
        audio_vision_dim=int(mcfg.get("audio_vision_dim", 256)),
        audio_vision_layers=int(mcfg.get("audio_vision_layers", 2)),
        audio_vision_heads=int(mcfg.get("audio_vision_heads", 4)),
    )


def distillation_loss(
    student_class_logits: torch.Tensor,
    student_intensity: torch.Tensor,
    teacher_class_logits: torch.Tensor,
    teacher_intensity: torch.Tensor,
    labels: torch.Tensor,
    targets: torch.Tensor,
    regression_weight: float = 0.5,
    temperature: float = 2.0,
    kl_weight: float = 1.0,
    intensity_consistency_weight: float = 0.5,
    class_weights: Optional[torch.Tensor] = None,
    ce_class_weights: bool = False,
    normalize_kl: bool = False,
) -> Dict[str, torch.Tensor]:
    """学生损失 = 真实标签损失 + KL(教师软分布 || 学生) + 强度一致性。

    KL 在固定温度 T 的软化分布上计算（标准 KD），并按 T^2 缩放。

    ``normalize_kl`` 默认 **False**，与最终报告采用的 E4 配置一致（该实验中
    ``kl_weight`` 最终取 0，即只用强度一致性，见复现报告 §3.3/§9）。

    选择 False 的原因：软目标项存在两种已知的尺度病态，本实验都实测到过：
      * 原样使用 KL：教师越训越自信，同一 ``kl_weight`` 下 KL 从 0.57 涨到 3.57，
        不同种子的教师之间不可比；
      * 除以 ``KL(p_T‖U)`` 做尺度归一化：教师分布接近均匀时分母趋近 0，
        学生只要偏离一点就会被放大成上千倍（单元测试覆盖了这一情形）。
    因此实现里保留了开关，但默认不做归一化，并如实记录该权衡。
    """
    ce = F.cross_entropy(
        student_class_logits, labels, weight=class_weights if ce_class_weights else None
    )
    huber = F.smooth_l1_loss(student_intensity, targets, beta=1.0)
    t = max(1e-3, float(temperature))
    log_student = F.log_softmax(student_class_logits / t, dim=-1)
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_class_logits / t, dim=-1)
    kl_raw = F.kl_div(log_student, teacher_probs, reduction="batchmean") * (t * t)
    if normalize_kl:
        # 可选：按教师信息量做尺度归一化。仅在校验过教师分布不接近均匀时使用，
        # 否则分母趋近 0 会放大损失（见函数文档）。
        with torch.no_grad():
            uniform = torch.full_like(teacher_probs, 1.0 / teacher_probs.shape[-1])
            ref = F.kl_div(torch.log(uniform + 1e-12), teacher_probs, reduction="batchmean")
            scale = torch.clamp(ref, min=1e-6, max=1.0)
        kl = kl_raw / scale
    else:
        kl = kl_raw
    consistency = F.smooth_l1_loss(student_intensity, teacher_intensity.detach(), beta=1.0)
    total = ce + regression_weight * huber + kl_weight * kl + intensity_consistency_weight * consistency
    return {
        "loss": total,
        "ce": ce.detach(),
        "huber": huber.detach(),
        "kl": kl.detach(),
        "consistency": consistency.detach(),
    }
