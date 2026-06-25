#!/usr/bin/env python3
"""
feature_importance.py
---------------------
计算并可视化 ICU 预测模型中各输入信号对最终预测结果的重要性权重。

支持三种互补的分析方法：
  1. Gradient × Input (Integrated Gradients) —— 时序 channel 级别的细粒度归因
  2. Ablation Study                          —— 模块级别贡献度
  3. Transformer 注意力权重提取              —— CLS token 对各 token 的关注分布

运行示例：
  # Demo 模式（随机数据，无需 checkpoint）
  python feature_importance.py --demo

  # 完整分析（需要 checkpoint 和 cache 文件）
  python feature_importance.py \
      --checkpoint output/20260623_.../model_tf_notes.pt \
      --cache      output/dataset_cache_clinicalbert_8dim_icuload.npz \
      --cache-meta output/dataset_cache_clinicalbert_meta_8dim_icuload.pkl \
      --n-samples 500 \
      --output-dir output/feature_importance
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("feature_importance")

# ---------------------------------------------------------------------------
# 模型定义（与 old_train.py 中 TransformerEarlyFusionWithNotes 完全对应）
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1)
        x = x + self.pe[: x.size(0)]
        x = self.dropout(x)
        return x.transpose(0, 1)


class TransformerEarlyFusionWithNotes(nn.Module):
    """复现 old_train.py 中的同名模型，用于加载 checkpoint。"""

    def __init__(
        self,
        seq_dim: int,
        static_dims: dict[str, int] | None = None,
        multihot_dims: dict[str, int] | None = None,
        hidden_dim: int = 64,
        note_dim: int = 768,
        use_notes: bool = True,
        nhead: int = 4,
        num_layers: int = 2,
        static_mode: str = "concat",
    ):
        super().__init__()
        self.use_notes = use_notes
        self.static_dims = static_dims or {}
        self.multihot_dims = multihot_dims or {}
        self.static_mode = static_mode

        # 1. 时序编码器（Transformer）
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=0.1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 2. 静态特征
        self.has_static = len(self.static_dims) > 0
        if self.has_static:
            if self.static_mode == "multi_token":
                self.static_heads = nn.ModuleDict()
                for name, vocab_size in self.static_dims.items():
                    if vocab_size <= 0:
                        self.static_heads[name] = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())
                    else:
                        self.static_heads[name] = nn.Embedding(vocab_size, hidden_dim)
            else:
                self.emb_dict = nn.ModuleDict()
                static_repr_dim = 0
                for name, vocab_size in self.static_dims.items():
                    if vocab_size <= 0:
                        static_repr_dim += 1
                    else:
                        emb_dim = max(4, min(16, vocab_size // 2))
                        self.emb_dict[name] = nn.Embedding(vocab_size, emb_dim)
                        static_repr_dim += emb_dim
                self.static_head = nn.Sequential(
                    nn.Linear(static_repr_dim, hidden_dim), nn.ReLU()
                )

        # 3. 多热稀疏特征（ICD / DRG / Proc / Rx）
        self.has_multihot = len(self.multihot_dims) > 0
        if self.has_multihot:
            self.mh_emb_dict = nn.ModuleDict()
            mh_repr_dim = 0
            for name, vocab_size in self.multihot_dims.items():
                emb_dim = max(8, min(32, vocab_size // 4))
                self.mh_emb_dict[name] = nn.Embedding(vocab_size, emb_dim)
                mh_repr_dim += emb_dim
            self.mh_head = nn.Sequential(
                nn.Linear(mh_repr_dim, hidden_dim), nn.ReLU()
            )

        # 4. 临床 Notes 嵌入
        if self.use_notes:
            self.note_head = nn.Sequential(
                nn.LayerNorm(note_dim),
                nn.Linear(note_dim, hidden_dim),
            )

        # 5. 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def _multihot_to_embedding(
        self, x_group: torch.Tensor, emb: nn.Embedding
    ) -> torch.Tensor:
        summed = x_group @ emb.weight
        denom = torch.clamp(x_group.sum(dim=1, keepdim=True), min=1.0)
        return summed / denom

    def forward(
        self,
        x_seq: torch.Tensor,
        x_static: torch.Tensor | None = None,
        x_mh: torch.Tensor | None = None,
        x_note: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x_seq.shape[0]
        x = self.seq_proj(x_seq)  # [B, T, H]

        extra_tokens: list[torch.Tensor] = []

        if self.has_static and x_static is not None:
            if self.static_mode == "multi_token":
                static_tokens = []
                col_idx = 0
                for name, vocab_size in self.static_dims.items():
                    val = x_static[:, col_idx]
                    if vocab_size <= 0:
                        token = self.static_heads[name](val.unsqueeze(1).float())
                    else:
                        token = self.static_heads[name](val.long())
                    static_tokens.append(token.unsqueeze(1))
                    col_idx += 1
                extra_tokens.extend(static_tokens)
            else:
                static_embs: list[torch.Tensor] = []
                col_idx = 0
                for name, vocab_size in self.static_dims.items():
                    val = x_static[:, col_idx]
                    if vocab_size <= 0:
                        static_embs.append(val.unsqueeze(1).float())
                    else:
                        static_embs.append(self.emb_dict[name](val.long()))
                    col_idx += 1
                static_repr = self.static_head(torch.cat(static_embs, dim=1))
                extra_tokens.append(static_repr.unsqueeze(1))

        if self.has_multihot and x_mh is not None:
            mh_embs: list[torch.Tensor] = []
            col_offset = 0
            for name, vocab_size in self.multihot_dims.items():
                group = x_mh[:, col_offset : col_offset + vocab_size]
                mh_embs.append(self._multihot_to_embedding(group, self.mh_emb_dict[name]))
                col_offset += vocab_size
            mh_repr = self.mh_head(torch.cat(mh_embs, dim=1))
            extra_tokens.append(mh_repr.unsqueeze(1))

        if self.use_notes and x_note is not None:
            x_note_norm = F.normalize(x_note, p=2, dim=1)
            note_repr = self.note_head(x_note_norm)
            extra_tokens.append(note_repr.unsqueeze(1))

        cls_tokens = self.cls_token.expand(B, -1, -1)
        if extra_tokens:
            x = torch.cat((cls_tokens, torch.cat(extra_tokens, dim=1), x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        cls_out = out[:, 0, :]
        return self.classifier(cls_out).squeeze(-1)


# ---------------------------------------------------------------------------
# 时序 Channel 名称（默认 8 维特征）
# ---------------------------------------------------------------------------

SEQ_CHANNEL_NAMES_8DIM = [
    "HR (心率)",
    "RR (呼吸频率)",
    "SPO2 (血氧)",
    "SBP (收缩压)",
    "DBP (舒张压)",
    "MAP (平均动脉压)",
    "TEMP (体温)",
    "GLUCOSE (血糖)",
]

# ---------------------------------------------------------------------------
# 方法 1：Integrated Gradients × Input
# ---------------------------------------------------------------------------

def gradient_x_input_importance(
    model: nn.Module,
    x_seq: torch.Tensor,
    x_static: torch.Tensor | None,
    x_mh: torch.Tensor | None,
    x_note: torch.Tensor | None,
    channel_names: list[str],
    device: torch.device,
    n_ig_steps: int = 50,
) -> dict[str, float]:
    """
    Integrated Gradients（积分梯度）计算每个时序 channel 对预测的贡献。

    公式：IG_i = (x_i - x'_i) * (1/m) * sum_k [ dF(x'+k/m*(x-x')) / dx_i ]
    基线 x' = 零向量（"无信号"参考态）。

    返回归一化重要性得分（sum=1）。
    """
    model.eval()
    x_seq = x_seq.to(device)
    if x_static is not None:
        x_static = x_static.to(device)
    if x_mh is not None:
        x_mh = x_mh.to(device)
    if x_note is not None:
        x_note = x_note.to(device)

    baseline = torch.zeros_like(x_seq)
    alphas = torch.linspace(0, 1, n_ig_steps, device=device)

    # 在插值路径上累积梯度
    grad_accum = torch.zeros_like(x_seq)

    for alpha in alphas:
        interp = (baseline + alpha * (x_seq - baseline)).detach().requires_grad_(True)

        # 静态输入不参与 IG（对 seq 归因），直接传入
        logits = model(interp, x_static, x_mh, x_note)
        logits.sum().backward()

        grad_accum = grad_accum + interp.grad.detach()

    # 梯形法则：平均梯度
    avg_grads = grad_accum / n_ig_steps

    # IG = delta_x * avg_grads，形状 [B, T, C]
    ig = (x_seq - baseline) * avg_grads
    channel_importance = ig.abs().mean(dim=(0, 1)).cpu().numpy()  # [C]

    total = channel_importance.sum()
    if total > 1e-8:
        channel_importance = channel_importance / total

    return {name: float(score) for name, score in zip(channel_names, channel_importance)}


# ---------------------------------------------------------------------------
# 方法 2：Ablation Study（模块级贡献）
# ---------------------------------------------------------------------------

def ablation_module_importance(
    model: nn.Module,
    x_seq: torch.Tensor,
    x_static: torch.Tensor | None,
    x_mh: torch.Tensor | None,
    x_note: torch.Tensor | None,
    device: torch.device,
) -> dict[str, Any]:
    """
    遮蔽各输入模块，测量预测概率变化幅度。

    返回：
        baseline:   基线平均预测概率
        no_seq:     遮蔽时序后的 |Δprob|
        no_static:  遮蔽静态特征后的 |Δprob|
        no_mh:      遮蔽诊断编码后的 |Δprob|
        no_note:    遮蔽 Notes 后的 |Δprob|
    """
    model.eval()
    x_seq = x_seq.to(device)
    if x_static is not None:
        x_static = x_static.to(device)
    if x_mh is not None:
        x_mh = x_mh.to(device)
    if x_note is not None:
        x_note = x_note.to(device)

    with torch.no_grad():
        baseline_probs = torch.sigmoid(
            model(x_seq, x_static, x_mh, x_note)
        ).cpu().numpy()

    results: dict[str, Any] = {
        "baseline": {
            "mean_prob": float(baseline_probs.mean()),
            "std_prob": float(baseline_probs.std()),
        }
    }

    def _record(logits: torch.Tensor, key: str) -> None:
        probs = torch.sigmoid(logits).cpu().numpy()
        delta = baseline_probs - probs
        results[key] = {
            "mean_delta": float(delta.mean()),
            "mean_abs_delta": float(np.abs(delta).mean()),
            "std_abs_delta": float(np.abs(delta).std()),
        }

    with torch.no_grad():
        # 遮蔽时序：用零序列替代
        _record(model(torch.zeros_like(x_seq), x_static, x_mh, x_note), "no_seq")

        # 遮蔽静态特征
        if x_static is not None:
            _record(model(x_seq, None, x_mh, x_note), "no_static")
        else:
            results["no_static"] = {"mean_abs_delta": 0.0, "note": "static not used"}

        # 遮蔽诊断编码
        if x_mh is not None:
            _record(model(x_seq, x_static, None, x_note), "no_mh")
        else:
            results["no_mh"] = {"mean_abs_delta": 0.0, "note": "multihot not used"}

        # 遮蔽 Notes
        if x_note is not None:
            _record(model(x_seq, x_static, x_mh, None), "no_note")
        else:
            results["no_note"] = {"mean_abs_delta": 0.0, "note": "notes not used"}

    return results


# ---------------------------------------------------------------------------
# 方法 3：Transformer 注意力权重
# ---------------------------------------------------------------------------

def extract_attention_weights(
    model: TransformerEarlyFusionWithNotes,
    x_seq: torch.Tensor,
    x_static: torch.Tensor | None,
    x_mh: torch.Tensor | None,
    x_note: torch.Tensor | None,
    channel_names: list[str],
    device: torch.device,
) -> dict[str, Any]:
    """
    提取所有 Transformer 层中 CLS token 对各 token 的平均注意力权重。

    Token 排列（与 forward 中 torch.cat 顺序一致）：
      [CLS] [STATIC] [CODES] [NOTE] t=0 t=1 ... t=T-1
    """
    model.eval()
    x_seq = x_seq.to(device)
    if x_static is not None:
        x_static = x_static.to(device)
    if x_mh is not None:
        x_mh = x_mh.to(device)
    if x_note is not None:
        x_note = x_note.to(device)

    # 捕获注意力权重
    captured_attn: list[torch.Tensor] = []

    def make_hook():
        def hook(module, input, output):
            # MultiheadAttention 输出 (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                captured_attn.append(output[1].detach().cpu())
        return hook

    hooks = []
    for layer in model.transformer_encoder.layers:
        # 强制 need_weights=True
        orig_forward = layer.self_attn.forward

        def patch(orig=orig_forward):
            def patched(q, k, v, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = True
                return orig(q, k, v, **kwargs)
            return patched

        layer.self_attn.forward = patch()
        hooks.append(layer.self_attn.register_forward_hook(make_hook()))

    with torch.no_grad():
        _ = model(x_seq, x_static, x_mh, x_note)

    for h in hooks:
        h.remove()

    # 构建 token 标签
    T = x_seq.shape[1]
    token_labels: list[str] = ["[CLS]"]
    if model.has_static and x_static is not None:
        token_labels.append("[STATIC]")
    if model.has_multihot and x_mh is not None:
        token_labels.append("[CODES]")
    if model.use_notes and x_note is not None:
        token_labels.append("[NOTE]")
    for t in range(T):
        token_labels.append(f"t={t}")

    if not captured_attn:
        return {"error": "No attention weights captured."}

    # CLS token（行 0）对所有 token（所有列）的注意力，对 batch 取均值
    attn_per_layer = []
    for aw in captured_attn:
        # aw: [B, T_full, T_full]
        cls_row = aw[:, 0, :].mean(dim=0).numpy()  # [T_full]
        attn_per_layer.append(cls_row)

    cls_attn_last = attn_per_layer[-1].tolist()
    cls_attn_avg = np.stack(attn_per_layer, axis=0).mean(axis=0).tolist()

    return {
        "token_labels": token_labels,
        "cls_attn_last_layer": cls_attn_last,
        "cls_attn_avg_all_layers": cls_attn_avg,
    }


def aggregate_attention_by_module(attn_result: dict[str, Any]) -> dict[str, float]:
    """把逐 token 注意力聚合成模块级权重。"""
    labels = attn_result.get("token_labels", [])
    weights = attn_result.get("cls_attn_avg_all_layers", [])
    if not labels or not weights:
        return {}

    groups: dict[str, float] = {
        "CLS (self)": 0.0,
        "静态人口统计": 0.0,
        "诊断编码 (ICD/DRG/Proc/Rx)": 0.0,
        "临床 Notes": 0.0,
        "时序生理信号": 0.0,
    }
    for label, w in zip(labels, weights):
        if label == "[CLS]":
            groups["CLS (self)"] += w
        elif label == "[STATIC]":
            groups["静态人口统计"] += w
        elif label == "[CODES]":
            groups["诊断编码 (ICD/DRG/Proc/Rx)"] += w
        elif label == "[NOTE]":
            groups["临床 Notes"] += w
        elif label.startswith("t="):
            groups["时序生理信号"] += w
    return groups


# ---------------------------------------------------------------------------
# 打印报告
# ---------------------------------------------------------------------------

def _bar(score: float, scale: float = 40.0) -> str:
    return "█" * max(0, int(score * scale))


def print_gxi_report(gxi: dict[str, float]) -> None:
    print("\n" + "=" * 62)
    print("  方法 1：Integrated Gradients × Input（时序 Channel 归因）")
    print("=" * 62)
    for rank, (name, score) in enumerate(
        sorted(gxi.items(), key=lambda kv: kv[1], reverse=True), start=1
    ):
        print(f"  #{rank:2d}  {name:<30s}  {score:.4f}  {_bar(score)}")
    print()


def print_ablation_report(abl: dict[str, Any]) -> None:
    print("=" * 62)
    print("  方法 2：Ablation Study（模块贡献度）")
    print("=" * 62)
    print(f"  基线平均风险概率：{abl['baseline']['mean_prob']:.4f}")
    print()
    for label, key in [
        ("遮蔽时序信号",       "no_seq"),
        ("遮蔽静态人口统计",   "no_static"),
        ("遮蔽诊断编码",       "no_mh"),
        ("遮蔽临床 Notes",     "no_note"),
    ]:
        d = abl.get(key, {})
        if "note" in d:
            print(f"  {label:<22s}  (未启用)")
        else:
            delta = d.get("mean_abs_delta", 0.0)
            print(f"  {label:<22s}  |Δprob| = {delta:.4f}  {_bar(delta, 200)}")
    print()


def print_attention_report(attn: dict[str, Any]) -> None:
    print("=" * 62)
    print("  方法 3：Transformer 注意力（CLS 对各 token 的关注度）")
    print("=" * 62)
    if "error" in attn:
        print(f"  错误：{attn['error']}")
        return

    module_weights = aggregate_attention_by_module(attn)
    non_self = {k: v for k, v in module_weights.items() if k != "CLS (self)"}
    total = sum(non_self.values()) or 1e-8

    print("  ── 模块级聚合（排除 CLS self-attention）")
    for name, w in sorted(non_self.items(), key=lambda kv: kv[1], reverse=True):
        nw = w / total
        print(f"    {name:<32s}  {nw:.4f}  {_bar(nw)}")
    print()

    # 最后一层的时间步注意力（Top-10）
    labels = attn.get("token_labels", [])
    weights_last = attn.get("cls_attn_last_layer", [])
    temporal = sorted(
        [(lbl, w) for lbl, w in zip(labels, weights_last) if lbl.startswith("t=")],
        key=lambda kv: kv[1], reverse=True
    )[:10]
    if temporal:
        print("  ── 最高关注时间步（最后一层，Top-10）")
        for lbl, w in temporal:
            print(f"    {lbl:<8s}  {w:.4f}  {_bar(w, 400)}")
    print()


# ---------------------------------------------------------------------------
# 方法 4：静态特征逐列贡献（Ablation + Integrated Gradients）
# ---------------------------------------------------------------------------

def static_feature_importance(
    model: nn.Module,
    x_seq: torch.Tensor,
    x_static: torch.Tensor,
    x_mh: torch.Tensor | None,
    x_note: torch.Tensor | None,
    static_dims: dict[str, int],
    device: torch.device,
    ig_steps: int = 50,
) -> dict[str, Any]:
    """
    对 x_static 的每一列逐列分析其对预测的贡献。

    策略：
      - 连续特征 (vocab_size=0): Ablation（置为列均值）+ Integrated Gradients
      - 分类特征 (vocab_size>0): Ablation（置为众数），IG 不适用于离散 Embedding

    返回：
        {
          "baseline_mean_prob": float,
          "ablation": {col_name: {"mean_abs_delta": float, "feature_type": str}},
          "integrated_gradients_continuous": {col_name: {"ig_score": float, "ig_ratio": float}},
        }
    """
    model.eval()
    x_seq    = x_seq.to(device)
    x_static = x_static.to(device)
    if x_mh   is not None: x_mh   = x_mh.to(device)
    if x_note is not None: x_note = x_note.to(device)

    static_cols = list(static_dims.items())   # [(name, vocab_size), ...]

    # ── 基线预测 ──────────────────────────────────────────────────────────
    with torch.no_grad():
        base_probs = torch.sigmoid(model(x_seq, x_static, x_mh, x_note)).cpu().numpy()
    baseline_prob = float(base_probs.mean())

    # ── A. Per-column Ablation ────────────────────────────────────────────
    ablation: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for col_i, (name, vocab_size) in enumerate(static_cols):
            x_abl = x_static.clone()
            if vocab_size <= 0:
                # 连续特征 → 置为列均值
                x_abl[:, col_i] = float(x_static[:, col_i].mean())
                ftype = "continuous"
            else:
                # 分类特征 → 置为众数
                vals = x_static[:, col_i].long().cpu().numpy()
                mode_val = int(np.bincount(vals).argmax())
                x_abl[:, col_i] = mode_val
                ftype = f"categorical({vocab_size})"

            abl_probs = torch.sigmoid(model(x_seq, x_abl, x_mh, x_note)).cpu().numpy()
            delta = base_probs - abl_probs
            ablation[name] = {
                "feature_type": ftype,
                "mean_delta":     float(delta.mean()),
                "mean_abs_delta": float(np.abs(delta).mean()),
                "std_abs_delta":  float(np.abs(delta).std()),
            }

    # ── B. Integrated Gradients（仅连续特征）─────────────────────────────
    continuous_idx = [(i, name) for i, (name, vs) in enumerate(static_cols) if vs <= 0]
    ig_continuous: dict[str, dict[str, float]] = {}

    if continuous_idx:
        x_static_f = x_static.float()
        col_means = x_static_f.mean(dim=0, keepdim=True)  # [1, C]

        # baseline: 连续列 → 列均值；分类列保持原值（不归因）
        baseline_s = x_static_f.clone()
        for col_i, _ in continuous_idx:
            baseline_s[:, col_i] = col_means[:, col_i]

        alphas = torch.linspace(0, 1, ig_steps, device=device)
        grad_accum = torch.zeros_like(x_static_f)

        for alpha in alphas:
            interp = (
                baseline_s + alpha * (x_static_f - baseline_s)
            ).detach().requires_grad_(True)
            logits = model(x_seq, interp, x_mh, x_note)
            logits.sum().backward()
            grad_accum = grad_accum + interp.grad.detach()

        avg_grads = grad_accum / ig_steps
        ig_val = (x_static_f - baseline_s) * avg_grads   # [B, C]
        ig_per_col = ig_val.abs().mean(dim=0).cpu().numpy()  # [C]

        total_ig = sum(float(ig_per_col[i]) for i, _ in continuous_idx) or 1e-8
        for col_i, name in continuous_idx:
            score = float(ig_per_col[col_i])
            ig_continuous[name] = {
                "ig_score": score,
                "ig_ratio": score / total_ig,
            }

    return {
        "baseline_mean_prob": baseline_prob,
        "ablation": ablation,
        "integrated_gradients_continuous": ig_continuous,
    }


def print_static_feature_report(
    sf: dict[str, Any], static_dims: dict[str, int]
) -> None:
    """打印静态特征逐列贡献报告。"""
    print("=" * 68)
    print("  方法 4：静态特征逐列贡献（Ablation + IG）")
    print("=" * 68)
    print(f"  基线平均风险概率：{sf['baseline_mean_prob']:.4f}")
    print()

    # A. Ablation 全列排名
    abl = sf["ablation"]
    sorted_abl = sorted(abl.items(), key=lambda kv: kv[1]["mean_abs_delta"], reverse=True)
    max_delta = max(v["mean_abs_delta"] for _, v in sorted_abl) or 1e-8
    total_delta = sum(v["mean_abs_delta"] for _, v in sorted_abl) or 1e-8

    print("  ── A. Per-column Ablation（全部 15 列，按贡献排序）")
    for rank, (name, d) in enumerate(sorted_abl, 1):
        delta = d["mean_abs_delta"]
        ftype = "连续" if static_dims.get(name, -1) <= 0 else f"分类"
        pct   = 100 * delta / total_delta
        bar   = "█" * int(delta / max_delta * 30)
        print(f"  #{rank:2d}  [{ftype}]  {name:<22s}  |Δp|={delta:.4f}  占比={pct:4.1f}%  {bar}")
    print()

    # B. IG 仅连续列
    ig = sf.get("integrated_gradients_continuous", {})
    if ig:
        sorted_ig = sorted(ig.items(), key=lambda kv: kv[1]["ig_ratio"], reverse=True)
        print("  ── B. Integrated Gradients（仅连续特征，归一化比例）")
        for rank, (name, vals) in enumerate(sorted_ig, 1):
            ratio = vals["ig_ratio"]
            bar   = "█" * int(ratio * 30)
            print(f"  #{rank:2d}  {name:<22s}  比例={ratio:.3f}  {bar}")
    print()


# ---------------------------------------------------------------------------
# 加载模型 & 数据
# ---------------------------------------------------------------------------

def load_model(
    ckpt: Path,
    static_dims: dict[str, int],
    multihot_dims: dict[str, int],
    seq_dim: int,
    note_dim: int,
    device: torch.device,
    hidden_dim: int | None = None,
    nhead: int = 4,
    num_layers: int = 2,
) -> TransformerEarlyFusionWithNotes:
    # 从 checkpoint 自动推断 hidden_dim（seq_proj.weight 形状为 [hidden_dim, seq_dim]）
    state = torch.load(ckpt, map_location=device)
    if hidden_dim is None:
        seq_proj_w = state.get("seq_proj.weight")
        if seq_proj_w is not None:
            hidden_dim = int(seq_proj_w.shape[0])
            logger.info("Auto-detected hidden_dim=%d from checkpoint", hidden_dim)
        else:
            hidden_dim = 64
            logger.warning("Could not detect hidden_dim, defaulting to %d", hidden_dim)

    # 自动探测 static_mode
    static_mode = "concat"
    if any(k.startswith("static_heads.") for k in state.keys()):
        static_mode = "multi_token"
        logger.info("Auto-detected static_mode='%s' from checkpoint", static_mode)
    else:
        logger.info("Auto-detected static_mode='concat' from checkpoint")

    model = TransformerEarlyFusionWithNotes(
        seq_dim=seq_dim,
        static_dims=static_dims,
        multihot_dims=multihot_dims,
        hidden_dim=hidden_dim,
        note_dim=note_dim,
        use_notes=True,
        nhead=nhead,
        num_layers=num_layers,
        static_mode=static_mode,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s", len(missing), missing[:8])
    if unexpected:
        logger.warning("Unexpected keys (%d): %s", len(unexpected), unexpected[:8])
    model.to(device).eval()
    logger.info("Checkpoint loaded from %s (hidden_dim=%d)", ckpt, hidden_dim)
    return model


def load_data(
    cache: Path, meta: Path
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, dict, dict]:
    with open(meta, "rb") as f:
        m = pickle.load(f)
    static_dims: dict[str, int] = m.get("static_dims", {})
    multihot_dims: dict[str, int] = m.get("multihot_dims", {})

    d = np.load(cache, allow_pickle=False)

    # 兼容大写键名（X_seq）和小写键名（x_seq）
    def _get(d, *names):
        for n in names:
            if n in d:
                return d[n].astype(np.float32)
        return None

    x_seq = _get(d, "x_seq", "X_seq")
    if x_seq is None:
        raise KeyError("Neither 'x_seq' nor 'X_seq' found in cache")
    x_static = _get(d, "x_static", "X_static")
    x_mh = _get(d, "x_mh", "X_mh")
    x_note = _get(d, "x_note", "X_note")
    logger.info(
        "Cache loaded | x_seq=%s x_static=%s x_mh=%s x_note=%s",
        x_seq.shape,
        x_static.shape if x_static is not None else None,
        x_mh.shape if x_mh is not None else None,
        x_note.shape if x_note is not None else None,
    )
    return x_seq, x_static, x_mh, x_note, static_dims, multihot_dims


# ---------------------------------------------------------------------------
# 核心分析流程
# ---------------------------------------------------------------------------

def run_analysis(
    model: nn.Module,
    x_seq: torch.Tensor,
    x_static: torch.Tensor | None,
    x_mh: torch.Tensor | None,
    x_note: torch.Tensor | None,
    channel_names: list[str],
    static_dims: dict[str, int],
    device: torch.device,
    output_dir: Path,
    label: str = "analysis",
    ig_steps: int = 50,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 方法 1
    logger.info("Running Integrated Gradients on temporal channels (steps=%d)...", ig_steps)
    gxi = gradient_x_input_importance(
        model, x_seq, x_static, x_mh, x_note, channel_names, device, ig_steps
    )
    print_gxi_report(gxi)

    # 方法 2
    logger.info("Running module-level Ablation Study...")
    abl = ablation_module_importance(model, x_seq, x_static, x_mh, x_note, device)
    print_ablation_report(abl)

    # 方法 3
    logger.info("Extracting Transformer attention weights...")
    attn = extract_attention_weights(
        model, x_seq, x_static, x_mh, x_note, channel_names, device
    )
    print_attention_report(attn)

    # 方法 4
    sf: dict[str, Any] = {}
    if x_static is not None and static_dims:
        logger.info("Running per-column static feature importance (steps=%d)...", ig_steps)
        sf = static_feature_importance(
            model, x_seq, x_static, x_mh, x_note, static_dims, device, ig_steps
        )
        print_static_feature_report(sf, static_dims)
    else:
        logger.info("Skipping static feature analysis (no static input).")

    # 保存完整 JSON
    report = {
        "integrated_gradients": gxi,
        "ablation": abl,
        "attention": {
            "token_labels": attn.get("token_labels", []),
            "cls_attn_last_layer": attn.get("cls_attn_last_layer", []),
            "cls_attn_avg_all_layers": attn.get("cls_attn_avg_all_layers", []),
            "module_aggregated": aggregate_attention_by_module(attn),
        },
        "static_feature_importance": sf,
    }
    json_path = output_dir / f"feature_importance_{label}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("JSON report -> %s", json_path)

    # 文本摘要
    txt_path = output_dir / f"feature_importance_{label}_summary.txt"
    _write_text_summary(report, txt_path, static_dims)
    logger.info("Text summary -> %s", txt_path)


def _write_text_summary(
    report: dict[str, Any], path: Path, static_dims: dict[str, int] | None = None
) -> None:
    lines = ["=" * 68, "  ICU 预测模型特征重要性分析报告", "=" * 68]

    lines.append("\n[1] Integrated Gradients（时序 Channel 归因）")
    for name, score in sorted(
        report["integrated_gradients"].items(), key=lambda kv: kv[1], reverse=True
    ):
        lines.append(f"  {name:<30s}  {score:.4f}  {_bar(score)}")

    lines.append("\n[2] Ablation Study（模块贡献度）")
    abl = report["ablation"]
    lines.append(f"  基线平均风险：{abl['baseline']['mean_prob']:.4f}")
    for label, key in [
        ("遮蔽时序信号",     "no_seq"),
        ("遮蔽静态人口统计", "no_static"),
        ("遮蔽诊断编码",     "no_mh"),
        ("遮蔽临床 Notes",   "no_note"),
    ]:
        d = abl.get(key, {})
        if "note" in d:
            lines.append(f"  {label:<22s}  (未启用)")
        else:
            delta = d.get("mean_abs_delta", 0.0)
            lines.append(f"  {label:<22s}  |Δprob| = {delta:.4f}  {_bar(delta, 200)}")

    lines.append("\n[3] Transformer 注意力（模块聚合）")
    mw = report["attention"].get("module_aggregated", {})
    total = sum(v for k, v in mw.items() if k != "CLS (self)") or 1e-8
    for name, w in sorted(mw.items(), key=lambda kv: kv[1], reverse=True):
        if name == "CLS (self)":
            continue
        nw = w / total
        lines.append(f"  {name:<30s}  {nw:.4f}  {_bar(nw)}")

    # [4] 静态特征逐列
    sf = report.get("static_feature_importance", {})
    if sf and static_dims:
        lines.append("\n[4] 静态特征逐列贡献（Ablation）")
        sf_abl = sf.get("ablation", {})
        sorted_abl = sorted(
            sf_abl.items(), key=lambda kv: kv[1]["mean_abs_delta"], reverse=True
        )
        total_delta = sum(v["mean_abs_delta"] for _, v in sorted_abl) or 1e-8
        max_delta = max(v["mean_abs_delta"] for _, v in sorted_abl) or 1e-8
        for rank, (name, d) in enumerate(sorted_abl, 1):
            delta = d["mean_abs_delta"]
            ftype = "连续" if static_dims.get(name, -1) <= 0 else "分类"
            pct   = 100 * delta / total_delta
            bar   = "█" * int(delta / max_delta * 25)
            lines.append(
                f"  #{rank:2d}  [{ftype}]  {name:<22s}  |Δp|={delta:.4f}  {pct:4.1f}%  {bar}"
            )

        ig = sf.get("integrated_gradients_continuous", {})
        if ig:
            lines.append("\n[4b] 连续特征 Integrated Gradients（归一化比例）")
            for rank, (name, vals) in enumerate(
                sorted(ig.items(), key=lambda kv: kv[1]["ig_ratio"], reverse=True), 1
            ):
                ratio = vals["ig_ratio"]
                bar   = "█" * int(ratio * 25)
                lines.append(f"  #{rank:2d}  {name:<22s}  比例={ratio:.3f}  {bar}")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Demo 模式
# ---------------------------------------------------------------------------

def run_demo(device: torch.device, output_dir: Path, ig_steps: int = 50) -> None:
    logger.info("DEMO mode: using synthetic random data")
    SEQ_DIM, T, B, NOTE_DIM = 8, 48, 32, 768

    static_dims = {
        "age": 0, "pre_icu_transfers": 0,
        "GENDER": 2, "MARITAL_STATUS": 8, "ETHNICITY": 41,
        "INSURANCE": 5, "ADMISSION_TYPE": 4, "ADMISSION_LOCATION": 9,
        "FIRST_CAREUNIT": 6, "surg_flag": 2,
    }
    multihot_dims = {"icd": 64, "drg": 64, "proc": 64, "rx": 64}

    model = TransformerEarlyFusionWithNotes(
        seq_dim=SEQ_DIM, static_dims=static_dims, multihot_dims=multihot_dims,
        hidden_dim=64, note_dim=NOTE_DIM, use_notes=True,
    ).to(device)

    rng = np.random.default_rng(42)
    x_seq = torch.from_numpy(rng.random((B, T, SEQ_DIM), dtype=np.float32)).to(device)

    x_static_np = np.zeros((B, len(static_dims)), dtype=np.float32)
    for i, (name, vs) in enumerate(static_dims.items()):
        if vs <= 0:
            x_static_np[:, i] = rng.random(B).astype(np.float32) * 80  # 连续列用随机小数
        else:
            x_static_np[:, i] = rng.integers(0, vs, B)
    x_static = torch.from_numpy(x_static_np).to(device)

    total_mh = sum(multihot_dims.values())
    x_mh = torch.from_numpy(
        (rng.random((B, total_mh)) > 0.9).astype(np.float32)
    ).to(device)
    x_note = torch.randn(B, NOTE_DIM, device=device)

    run_analysis(
        model=model,
        x_seq=x_seq, x_static=x_static, x_mh=x_mh, x_note=x_note,
        channel_names=SEQ_CHANNEL_NAMES_8DIM[:SEQ_DIM],
        static_dims=static_dims,
        device=device, output_dir=output_dir, label="demo", ig_steps=ig_steps,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_latest(output_dir: Path, glob: str) -> Path | None:
    candidates = sorted(output_dir.glob(glob), key=lambda p: str(p.parent), reverse=True)
    return candidates[0] if candidates else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ICU 预测模型特征重要性分析（IG + Ablation + Attention）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="模型 .pt 文件路径（不填则自动搜索 output/）")
    p.add_argument("--cache", type=Path, default=None,
                   help="数据集 .npz cache 文件路径")
    p.add_argument("--cache-meta", type=Path, default=None,
                   help="数据集 meta .pkl 文件路径")
    p.add_argument("--n-samples", type=int, default=200,
                   help="分析样本数（默认 200）")
    p.add_argument("--note-dim", type=int, default=768,
                   help="Note 嵌入维度（默认 768）")
    p.add_argument("--output-dir", type=Path, default=Path("output/feature_importance"),
                   help="输出目录（默认 output/feature_importance）")
    p.add_argument("--device", default=None,
                   help="cuda / cpu（默认自动检测）")
    p.add_argument("--demo", action="store_true",
                   help="Demo 模式：用随机数据演示，无需 checkpoint 和 cache")
    p.add_argument("--ig-steps", type=int, default=50,
                   help="Integrated Gradients 插值步数（默认 50）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", dev)

    out = Path(args.output_dir)

    if args.demo:
        run_demo(dev, out, ig_steps=args.ig_steps)
        return

    # 自动检测 checkpoint
    REPO = Path(__file__).resolve().parent
    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = _find_latest(REPO / "output", "*/model_tf_notes.pt")
        if ckpt is None:
            logger.error("No checkpoint found. Use --checkpoint or --demo.")
            sys.exit(1)
        logger.info("Auto-detected checkpoint: %s", ckpt)

    # 自动检测 cache (优先使用主训练脚本硬编码的默认路径)
    default_cache = REPO / "output" / "dataset_cache_clinicalbert_8dim_icuload.npz"
    default_meta = REPO / "output" / "dataset_cache_clinicalbert_meta_8dim_icuload.pkl"
    
    if args.cache:
        cache = args.cache
    else:
        cache = default_cache if default_cache.exists() else _find_latest(REPO / "output", "dataset_cache_clinicalbert_*icuload*.npz")
        
    if args.cache_meta:
        meta = args.cache_meta
    else:
        meta = default_meta if default_meta.exists() else _find_latest(REPO / "output", "dataset_cache_clinicalbert_meta_*icuload*.pkl")

    if cache is None or not cache.exists():
        logger.error("Cache .npz not found. Use --cache or --demo.")
        sys.exit(1)
    if meta is None or not meta.exists():
        logger.error("Meta .pkl not found. Use --cache-meta or --demo.")
        sys.exit(1)

    # 加载数据
    x_seq_all, x_static_all, x_mh_all, x_note_all, static_dims, multihot_dims = (
        load_data(cache, meta)
    )

    N = len(x_seq_all)
    n = min(args.n_samples, N)
    idx = np.random.default_rng(42).choice(N, size=n, replace=False)

    x_seq = torch.from_numpy(x_seq_all[idx]).to(dev)
    x_static = torch.from_numpy(x_static_all[idx]).to(dev) if x_static_all is not None else None
    x_mh = torch.from_numpy(x_mh_all[idx]).to(dev) if x_mh_all is not None else None
    x_note = torch.from_numpy(x_note_all[idx]).to(dev) if x_note_all is not None else None

    note_dim = x_note.shape[-1] if x_note is not None else args.note_dim
    seq_dim = x_seq.shape[-1]
    channel_names = SEQ_CHANNEL_NAMES_8DIM[:seq_dim]
    if len(channel_names) < seq_dim:
        channel_names += [f"ch_{i}" for i in range(len(channel_names), seq_dim)]

    model = load_model(ckpt, static_dims, multihot_dims, seq_dim, note_dim, dev)
    label = ckpt.parent.name[:40].replace(" ", "_")

    run_analysis(
        model=model,
        x_seq=x_seq, x_static=x_static, x_mh=x_mh, x_note=x_note,
        channel_names=channel_names,
        static_dims=static_dims,
        device=dev, output_dir=out, label=label, ig_steps=args.ig_steps,
    )


if __name__ == "__main__":
    main()
