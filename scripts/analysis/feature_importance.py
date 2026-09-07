#!/usr/bin/env python3
"""
feature_importance.py
---------------------
计算并可视化最新多模态 ICU 预测模型 Transformer E2E FineTune (+ Notes) 中
各输入信号对最终 30 天再入院预测结果的重要性权重。

支持四种互补的分析方法：
  1. Gradient × Input (Integrated Gradients) —— 48h 时序 channel 级别的细粒度归因
  2. Ablation Study                          —— 宏观四大输入模态的贡献度消融
  3. Transformer 注意力权重提取              —— CLS token 对 48 小时各时间步的动态关注轨迹
  4. 静态特征逐列分析                        —— 人口学/手术历程/ICU负荷指标的逐列重要性

运行示例：
  # 自动分析最新的 Transformer E2E 模型
  python scripts/analysis/feature_importance.py

  # 指定 checkpoint 和样本数
  python scripts/analysis/feature_importance.py \
      --checkpoint output/20260904_151210_.../model_tf_e2e.pt \
      --n-samples 200 \
      --output-dir output/feature_importance
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training.models import TransformerEndToEndWithNotes, TransformerEarlyFusionWithNotes
from scripts.training.config import EXP_CONFIG
from scripts.training.data import load_note_texts

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("feature_importance")

# ---------------------------------------------------------------------------
# 时序 Channel 名称（8 维生理指标）
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
# 方法 1：Integrated Gradients (时序生理通道细粒度归因)
# ---------------------------------------------------------------------------
def gradient_x_input_importance(
    model: nn.Module,
    x_seq: torch.Tensor,
    x_static: torch.Tensor | None,
    x_mh: torch.Tensor | None,
    x_note: torch.Tensor | None,
    x_note_tokens: dict[str, torch.Tensor] | None,
    channel_names: list[str],
    device: torch.device,
    n_ig_steps: int = 50,
) -> dict[str, float]:
    """
    基于 Integrated Gradients (积分梯度) 计算 8 个时序 channel 对预测的贡献。
    基线 x' = 零向量（参考态）。
    """
    model.eval()
    x_seq = x_seq.to(device)
    if x_static is not None:
        x_static = x_static.to(device)
    if x_mh is not None:
        x_mh = x_mh.to(device)
    if x_note is not None:
        x_note = x_note.to(device)
    if x_note_tokens is not None:
        x_note_tokens = {k: v.to(device) for k, v in x_note_tokens.items()}

    baseline = torch.zeros_like(x_seq)
    alphas = torch.linspace(0, 1, n_ig_steps, device=device)
    grad_accum = torch.zeros_like(x_seq)

    for alpha in alphas:
        interp = (baseline + alpha * (x_seq - baseline)).detach().requires_grad_(True)
        if hasattr(model, 'bert_model'):
            logits = model(interp, x_static, x_mh, x_note=x_note, x_note_tokens=x_note_tokens)
        else:
            logits = model(interp, x_static, x_mh, x_note=x_note)
        logits.sum().backward()
        grad_accum = grad_accum + interp.grad.detach()

    avg_grads = grad_accum / n_ig_steps
    ig = (x_seq - baseline) * avg_grads
    channel_importance = ig.abs().mean(dim=(0, 1)).cpu().numpy()

    total = channel_importance.sum()
    if total > 1e-8:
        channel_importance = channel_importance / total

    return {name: float(score) for name, score in zip(channel_names, channel_importance)}


# ---------------------------------------------------------------------------
# 方法 2：Ablation Study (四大输入模态宏观贡献度)
# ---------------------------------------------------------------------------
def ablation_module_importance(
    model: nn.Module,
    x_seq: torch.Tensor,
    x_static: torch.Tensor | None,
    x_mh: torch.Tensor | None,
    x_note: torch.Tensor | None,
    x_note_tokens: dict[str, torch.Tensor] | None,
    device: torch.device,
) -> dict[str, Any]:
    """
    遮蔽各输入模态，测量预测概率变化绝对幅度 |Δprob|。
    针对 TransformerEndToEndWithNotes，提取各分支表征并按需置零，
    以保证分类头 LayerNorm(320) 的维度一致性。
    """
    model.eval()
    x_seq = x_seq.to(device)
    if x_static is not None:
        x_static = x_static.to(device)
    if x_mh is not None:
        x_mh = x_mh.to(device)
    if x_note is not None:
        x_note = x_note.to(device)
    if x_note_tokens is not None:
        x_note_tokens = {k: v.to(device) for k, v in x_note_tokens.items()}

    with torch.no_grad():
        # 1. 提取各模态表征
        B = x_seq.shape[0]
        x = model.seq_proj(x_seq)
        cls_tokens = model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = model.pos_encoder(x)
        x = model.transformer_encoder(x)
        seq_repr = x[:, 0, :]

        static_repr = None
        if getattr(model, "has_static", False) and x_static is not None:
            static_embs = []
            col_idx = 0
            for name, _ in model.static_dims.items():
                val = x_static[:, col_idx]
                if model.static_dims[name] == 0:
                    static_embs.append(val.unsqueeze(1).float())
                else:
                    static_embs.append(model.emb_dict[name](val.long()))
                col_idx += 1
            static_repr = model.static_head(torch.cat(static_embs, dim=1))

        mh_repr = None
        if getattr(model, "has_multihot", False) and x_mh is not None:
            mh_embs = []
            col_offset = 0
            for name, vocab_size in model.multihot_dims.items():
                group = x_mh[:, col_offset : col_offset + vocab_size]
                mh_embs.append(model._multihot_to_embedding(group.float(), model.mh_emb_dict[name]))
                col_offset += vocab_size
            mh_repr = model.mh_head(torch.cat(mh_embs, dim=1))

        bert_repr = None
        if x_note_tokens is not None:
            bert_repr = model._encode_note_tokens(x_note_tokens)
        elif x_note is not None:
            if hasattr(model, "note_head"):
                bert_repr = model.note_head(F.normalize(x_note, p=2, dim=1))

        def _predict(s_r, st_r, m_r, b_r):
            parts = [s_r]
            if st_r is not None:
                parts.append(st_r)
            if m_r is not None:
                parts.append(m_r)
            if b_r is not None:
                parts.append(b_r)
            fused = torch.cat(parts, dim=1)
            return model.classifier(fused).squeeze(-1)

        base_logits = _predict(seq_repr, static_repr, mh_repr, bert_repr)
        baseline_probs = torch.sigmoid(base_logits).cpu().numpy()

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

        # 1. 遮蔽时序生理信号 (用零向量替代 seq_repr)
        _record(_predict(torch.zeros_like(seq_repr), static_repr, mh_repr, bert_repr), "no_seq")

        # 2. 遮蔽静态人口统计与管理特征
        if static_repr is not None:
            _record(_predict(seq_repr, torch.zeros_like(static_repr), mh_repr, bert_repr), "no_static")
        else:
            results["no_static"] = {"mean_abs_delta": 0.0, "note": "static not used"}

        # 3. 遮蔽高维稀疏诊断编码 (ICD/DRG/Proc/Rx)
        if mh_repr is not None:
            _record(_predict(seq_repr, static_repr, torch.zeros_like(mh_repr), bert_repr), "no_mh")
        else:
            results["no_mh"] = {"mean_abs_delta": 0.0, "note": "multihot not used"}

        # 4. 遮蔽临床病程 Notes
        if bert_repr is not None:
            _record(_predict(seq_repr, static_repr, mh_repr, torch.zeros_like(bert_repr)), "no_note")
        else:
            results["no_note"] = {"mean_abs_delta": 0.0, "note": "notes not used"}

    return results


# ---------------------------------------------------------------------------
# 方法 3：Transformer 自注意力机制提取 (时序关注轨迹)
# ---------------------------------------------------------------------------
def extract_temporal_attention(
    model: nn.Module,
    x_seq: torch.Tensor,
    x_static: torch.Tensor | None,
    x_mh: torch.Tensor | None,
    x_note: torch.Tensor | None,
    x_note_tokens: dict[str, torch.Tensor] | None,
    device: torch.device,
) -> dict[str, Any]:
    """
    提取 Transformer 编码器中 CLS token 对 48 个小时步的自注意力权重。
    """
    model.eval()
    x_seq = x_seq.to(device)
    if x_static is not None: x_static = x_static.to(device)
    if x_mh is not None: x_mh = x_mh.to(device)
    if x_note is not None: x_note = x_note.to(device)
    if x_note_tokens is not None:
        x_note_tokens = {k: v.to(device) for k, v in x_note_tokens.items()}

    if not hasattr(model, 'transformer_encoder'):
        return {"error": "Model does not have transformer_encoder."}

    captured_attn: list[torch.Tensor] = []

    def make_hook():
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                captured_attn.append(output[1].detach().cpu())
        return hook

    orig_forwards = []
    hooks = []
    for layer in model.transformer_encoder.layers:
        orig = layer.self_attn.forward
        orig_forwards.append((layer.self_attn, orig))
        def patch(orig_fn=orig):
            def patched(q, k, v, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = True
                return orig_fn(q, k, v, **kwargs)
            return patched
        layer.self_attn.forward = patch()
        hooks.append(layer.self_attn.register_forward_hook(make_hook()))

    with torch.no_grad():
        if hasattr(model, 'bert_model'):
            _ = model(x_seq, x_static, x_mh, x_note=x_note, x_note_tokens=x_note_tokens)
        else:
            _ = model(x_seq, x_static, x_mh, x_note=x_note)

    for h in hooks:
        h.remove()
    for self_attn, orig in orig_forwards:
        self_attn.forward = orig

    if not captured_attn:
        return {"error": "No attention weights captured."}

    T = x_seq.shape[1]
    # token 排列: [CLS], t=0, t=1, ..., t=T-1
    token_labels = ["[CLS]"] + [f"t={t}" for t in range(T)]

    attn_per_layer = []
    for aw in captured_attn:
        # aw: [B, seq_len, seq_len], 取 CLS (行 0) 对各列的注意力均值
        cls_row = aw[:, 0, :].mean(dim=0).numpy()
        attn_per_layer.append(cls_row)

    cls_attn_last = attn_per_layer[-1].tolist()
    cls_attn_avg = np.stack(attn_per_layer, axis=0).mean(axis=0).tolist()

    # 计算时间阶段分布 (排除 CLS 自身)
    weights_no_cls = cls_attn_avg[1:]
    total_w = sum(weights_no_cls) or 1e-8
    
    phase_acute = sum(weights_no_cls[:12]) / total_w       # 前 12 小时：入科急性期
    phase_stable = sum(weights_no_cls[12:36]) / total_w    # 12-36 小时：监护观察期
    phase_predischarge = sum(weights_no_cls[36:]) / total_w # 36-48 小时：出科前准备期

    return {
        "token_labels": token_labels,
        "cls_attn_last_layer": cls_attn_last,
        "cls_attn_avg_all_layers": cls_attn_avg,
        "phase_breakdown": {
            "入科急性期 (0-12h)": float(phase_acute),
            "监护观察期 (12-36h)": float(phase_stable),
            "出科准备期 (36-48h)": float(phase_predischarge),
        }
    }


# ---------------------------------------------------------------------------
# 方法 4：静态特征逐列贡献 (Ablation + IG)
# ---------------------------------------------------------------------------
def static_feature_importance(
    model: nn.Module,
    x_seq: torch.Tensor,
    x_static: torch.Tensor,
    x_mh: torch.Tensor | None,
    x_note: torch.Tensor | None,
    x_note_tokens: dict[str, torch.Tensor] | None,
    static_dims: dict[str, int],
    device: torch.device,
    ig_steps: int = 50,
) -> dict[str, Any]:
    """
    对 15 个静态特征逐列分析其对 30 天再入院预测的贡献。
    """
    model.eval()
    x_seq = x_seq.to(device)
    x_static = x_static.to(device)
    if x_mh is not None: x_mh = x_mh.to(device)
    if x_note is not None: x_note = x_note.to(device)
    if x_note_tokens is not None:
        x_note_tokens = {k: v.to(device) for k, v in x_note_tokens.items()}

    static_cols = list(static_dims.items())
    is_e2e = hasattr(model, 'bert_model')

    def _forward(st):
        if is_e2e:
            return model(x_seq, st, x_mh, x_note=x_note, x_note_tokens=x_note_tokens)
        return model(x_seq, st, x_mh, x_note=x_note)

    with torch.no_grad():
        base_probs = torch.sigmoid(_forward(x_static)).cpu().numpy()
    baseline_prob = float(base_probs.mean())

    # A. 逐列消融
    ablation: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for col_i, (name, vocab_size) in enumerate(static_cols):
            x_abl = x_static.clone()
            if vocab_size <= 0:
                x_abl[:, col_i] = float(x_static[:, col_i].mean())
                ftype = "continuous"
            else:
                vals = x_static[:, col_i].long().cpu().numpy()
                mode_val = int(np.bincount(vals).argmax())
                x_abl[:, col_i] = mode_val
                ftype = f"categorical({vocab_size})"

            abl_probs = torch.sigmoid(_forward(x_abl)).cpu().numpy()
            delta = base_probs - abl_probs
            ablation[name] = {
                "feature_type": ftype,
                "mean_delta": float(delta.mean()),
                "mean_abs_delta": float(np.abs(delta).mean()),
                "std_abs_delta": float(np.abs(delta).std()),
            }

    # B. 仅连续列的 Integrated Gradients
    continuous_idx = [(i, name) for i, (name, vs) in enumerate(static_cols) if vs <= 0]
    ig_continuous: dict[str, dict[str, float]] = {}

    if continuous_idx:
        x_static_f = x_static.float()
        col_means = x_static_f.mean(dim=0, keepdim=True)
        baseline_s = x_static_f.clone()
        for col_i, _ in continuous_idx:
            baseline_s[:, col_i] = col_means[:, col_i]

        alphas = torch.linspace(0, 1, ig_steps, device=device)
        grad_accum = torch.zeros_like(x_static_f)

        for alpha in alphas:
            interp = (baseline_s + alpha * (x_static_f - baseline_s)).detach().requires_grad_(True)
            logits = _forward(interp)
            logits.sum().backward()
            grad_accum = grad_accum + interp.grad.detach()

        avg_grads = grad_accum / ig_steps
        ig_val = (x_static_f - baseline_s) * avg_grads
        ig_per_col = ig_val.abs().mean(dim=0).cpu().numpy()

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


# ---------------------------------------------------------------------------
# 打印与保存报告
# ---------------------------------------------------------------------------
def _bar(score: float, scale: float = 40.0) -> str:
    return "█" * max(0, int(score * scale))

def print_report(report: dict[str, Any], static_dims: dict[str, int]) -> None:
    print("\n" + "=" * 68)
    print("  Transformer E2E FineTune (+ Notes) 特征重要性分析报告")
    print("=" * 68)

    # 1. IG
    print("\n[1] Integrated Gradients（时序生理通道细粒度归因）")
    for name, score in sorted(report["integrated_gradients"].items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:<30s}  {score:.4f}  {_bar(score, 30)}")

    # 2. 模态消融
    print("\n[2] Ablation Study（四大输入模态贡献度消融）")
    abl = report["ablation"]
    print(f"  基线平均风险预测概率：{abl['baseline']['mean_prob']:.4f}")
    mod_labels = [
        ("时序生理信号 (48h Vitals)", "no_seq"),
        ("静态人口学与管理特征", "no_static"),
        ("高维诊断编码 (ICD/DRG/Proc/Rx)", "no_mh"),
        ("非结构化病程文本 (ClinicalBERT)", "no_note"),
    ]
    for label, key in mod_labels:
        d = abl.get(key, {})
        delta = d.get("mean_abs_delta", 0.0)
        print(f"  {label:<32s}  |Δprob| = {delta:.4f}  {_bar(delta, 150)}")

    # 3. Transformer 时间步注意力
    print("\n[3] Transformer 时序自注意力机制（CLS 对 48 小时轨迹的关注分布）")
    attn = report.get("temporal_attention", {})
    if "error" in attn:
        print(f"  提示：{attn['error']}")
    else:
        pb = attn.get("phase_breakdown", {})
        print("  ── 临床观察阶段注意力占比：")
        for phase, ratio in pb.items():
            print(f"    {phase:<24s}  {ratio*100:5.1f}%  {_bar(ratio, 25)}")
        
        print("\n  ── 关注度最高的 10 个小时步 (Top-10 Hours)：")
        weights = attn.get("cls_attn_avg_all_layers", [])
        labels = attn.get("token_labels", [])
        if weights and labels:
            temporal_pairs = [(lbl, w) for lbl, w in zip(labels, weights) if lbl.startswith("t=")]
            temporal_pairs.sort(key=lambda kv: kv[1], reverse=True)
            for lbl, w in temporal_pairs[:10]:
                hour = lbl.replace("t=", "第 ") + " 小时"
                print(f"    {hour:<16s}  权重={w:.4f}  {_bar(w, 200)}")

    # 4. 静态特征
    print("\n[4] 静态特征逐列贡献度分析 (Ablation)")
    sf = report.get("static_feature_importance", {})
    sf_abl = sf.get("ablation", {})
    sorted_abl = sorted(sf_abl.items(), key=lambda kv: kv[1]["mean_abs_delta"], reverse=True)
    total_delta = sum(v["mean_abs_delta"] for _, v in sorted_abl) or 1e-8
    max_delta = max(v["mean_abs_delta"] for _, v in sorted_abl) or 1e-8
    for rank, (name, d) in enumerate(sorted_abl, 1):
        delta = d["mean_abs_delta"]
        ftype = "连续" if static_dims.get(name, -1) <= 0 else "分类"
        pct = 100 * delta / total_delta
        print(f"  #{rank:2d}  [{ftype}]  {name:<22s}  |Δp|={delta:.4f}  {pct:4.1f}%  {_bar(delta / max_delta, 20)}")

    ig_cont = sf.get("integrated_gradients_continuous", {})
    if ig_cont:
        print("\n[4b] 连续特征 Integrated Gradients 归一化比例：")
        for rank, (name, vals) in enumerate(sorted(ig_cont.items(), key=lambda kv: kv[1]["ig_ratio"], reverse=True), 1):
            ratio = vals["ig_ratio"]
            print(f"  #{rank:2d}  {name:<22s}  比例={ratio:.3f}  {_bar(ratio, 20)}")
    print("=" * 68 + "\n")


def write_text_summary(report: dict[str, Any], path: Path, static_dims: dict[str, int]) -> None:
    lines = ["=" * 68, "  Transformer E2E FineTune (+ Notes) 特征重要性分析报告", "=" * 68]

    lines.append("\n[1] Integrated Gradients（时序生理通道细粒度归因）")
    for name, score in sorted(report["integrated_gradients"].items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"  {name:<30s}  {score:.4f}  {_bar(score, 30)}")

    lines.append("\n[2] Ablation Study（四大输入模态贡献度消融）")
    abl = report["ablation"]
    lines.append(f"  基线平均风险预测概率：{abl['baseline']['mean_prob']:.4f}")
    mod_labels = [
        ("时序生理信号 (48h Vitals)", "no_seq"),
        ("静态人口学与管理特征", "no_static"),
        ("高维诊断编码 (ICD/DRG/Proc/Rx)", "no_mh"),
        ("非结构化病程文本 (ClinicalBERT)", "no_note"),
    ]
    for label, key in mod_labels:
        d = abl.get(key, {})
        delta = d.get("mean_abs_delta", 0.0)
        lines.append(f"  {label:<32s}  |Δprob| = {delta:.4f}  {_bar(delta, 150)}")

    lines.append("\n[3] Transformer 时序自注意力机制（CLS 对 48 小时轨迹的关注分布）")
    attn = report.get("temporal_attention", {})
    if "error" in attn:
        lines.append(f"  提示：{attn['error']}")
    else:
        pb = attn.get("phase_breakdown", {})
        lines.append("  ── 临床观察阶段注意力占比：")
        for phase, ratio in pb.items():
            lines.append(f"    {phase:<24s}  {ratio*100:5.1f}%  {_bar(ratio, 25)}")
        
        lines.append("\n  ── 关注度最高的 10 个小时步 (Top-10 Hours)：")
        weights = attn.get("cls_attn_avg_all_layers", [])
        labels = attn.get("token_labels", [])
        if weights and labels:
            temporal_pairs = [(lbl, w) for lbl, w in zip(labels, weights) if lbl.startswith("t=")]
            temporal_pairs.sort(key=lambda kv: kv[1], reverse=True)
            for lbl, w in temporal_pairs[:10]:
                hour = lbl.replace("t=", "第 ") + " 小时"
                lines.append(f"    {hour:<16s}  权重={w:.4f}  {_bar(w, 200)}")

    lines.append("\n[4] 静态特征逐列贡献度分析 (Ablation)")
    sf = report.get("static_feature_importance", {})
    sf_abl = sf.get("ablation", {})
    sorted_abl = sorted(sf_abl.items(), key=lambda kv: kv[1]["mean_abs_delta"], reverse=True)
    total_delta = sum(v["mean_abs_delta"] for _, v in sorted_abl) or 1e-8
    max_delta = max(v["mean_abs_delta"] for _, v in sorted_abl) or 1e-8
    for rank, (name, d) in enumerate(sorted_abl, 1):
        delta = d["mean_abs_delta"]
        ftype = "连续" if static_dims.get(name, -1) <= 0 else "分类"
        pct = 100 * delta / total_delta
        lines.append(f"  #{rank:2d}  [{ftype}]  {name:<22s}  |Δp|={delta:.4f}  {pct:4.1f}%  {_bar(delta / max_delta, 20)}")

    ig_cont = sf.get("integrated_gradients_continuous", {})
    if ig_cont:
        lines.append("\n[4b] 连续特征 Integrated Gradients 归一化比例：")
        for rank, (name, vals) in enumerate(sorted(ig_cont.items(), key=lambda kv: kv[1]["ig_ratio"], reverse=True), 1):
            ratio = vals["ig_ratio"]
            lines.append(f"  #{rank:2d}  {name:<22s}  比例={ratio:.3f}  {_bar(ratio, 20)}")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 主执行入口
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transformer E2E FineTune (+ Notes) 特征重要性分析")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="模型 .pt 文件路径（默认自动寻找最新的 model_tf_e2e.pt）")
    p.add_argument("--cache", type=Path, default=None,
                   help="数据集 .npz 缓存文件路径")
    p.add_argument("--cache-meta", type=Path, default=None,
                   help="数据集元数据 .pkl 文件路径")
    p.add_argument("--summaries", type=Path, default=Path("output/mimic3_note_summaries.pkl"),
                   help="病程摘要缓存 .pkl 路径（用于在线端到端分词）")
    p.add_argument("--n-samples", type=int, default=200,
                   help="分析样本数（默认 200）")
    p.add_argument("--output-dir", type=Path, default=Path("output/feature_importance"),
                   help="分析报告输出目录")
    p.add_argument("--device", default=None, help="cuda / cpu")
    p.add_argument("--ig-steps", type=int, default=50, help="IG 积分插值步数")
    return p.parse_args()


def find_latest_checkpoint(root: Path) -> Path | None:
    matches = list(root.glob("output/*/model_tf_e2e.pt"))
    if not matches:
        return None
    matches.sort(key=lambda p: str(p.parent), reverse=True)
    return matches[0]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = find_latest_checkpoint(REPO_ROOT)
        if ckpt_path is None or not ckpt_path.exists():
            logger.error("No model_tf_e2e.pt checkpoint found in output/*/")
            sys.exit(1)
    logger.info("Target Model Checkpoint: %s", ckpt_path)

    # 确定缓存路径
    cache_path = args.cache
    if cache_path is None:
        cache_path = REPO_ROOT / "output" / "dataset_cache_clinicalbert_8dim_icuload_nibp_drgemb_icdsplit.npz"
        if not cache_path.exists():
            candidates = list((REPO_ROOT / "output").glob("dataset_cache_*.npz"))
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates: cache_path = candidates[0]
            else:
                logger.error("Cache npz not found.")
                sys.exit(1)
    
    meta_path = args.cache_meta
    if meta_path is None:
        meta_path = REPO_ROOT / "output" / "dataset_cache_clinicalbert_meta_8dim_icuload_nibp_drgemb_icdsplit.pkl"
        if not meta_path.exists():
            candidates = list((REPO_ROOT / "output").glob("dataset_cache_*meta*.pkl"))
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates: meta_path = candidates[0]
            else:
                logger.error("Meta pkl not found.")
                sys.exit(1)

    logger.info("Loading cache from %s and %s ...", cache_path, meta_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    static_dims: dict[str, int] = meta.get("static_dims", {})
    multihot_dims: dict[str, int] = meta.get("multihot_dims", {})

    d = np.load(cache_path, allow_pickle=False)
    x_seq_all = d["X_seq"] if "X_seq" in d else d["x_seq"]
    x_static_all = d["X_static"] if "X_static" in d else d["x_static"]
    x_mh_all = d["X_mh"] if "X_mh" in d else d["x_mh"]
    x_note_all = d["X_note"] if "X_note" in d else d["x_note"]

    # 尝试加载 seq_scaler 进行时序归一化（保持与训练一致）
    exp_dir = ckpt_path.parent
    scaler_path = exp_dir / "seq_scaler.pkl"
    if scaler_path.exists():
        logger.info("Applying StandardScaler from %s to sequence features ...", scaler_path)
        with open(scaler_path, "rb") as f:
            seq_scaler = pickle.load(f)
        N, T, F = x_seq_all.shape
        x_seq_all = seq_scaler.transform(x_seq_all.reshape(-1, F)).reshape(N, T, F).astype(np.float32)

    # 加载文本摘要以支持端到端分词
    summaries_path = args.summaries
    note_texts_dict = {}
    if summaries_path.exists():
        logger.info("Loading note summaries for E2E tokenization from %s ...", summaries_path)
        with open(summaries_path, "rb") as f:
            raw_summaries = pickle.load(f)
        note_texts_dict = load_note_texts(raw_summaries)
        logger.info("Loaded note text for %d stays", len(note_texts_dict))

    # 实例化 TransformerEndToEndWithNotes
    seq_dim = x_seq_all.shape[-1]
    note_dim = x_note_all.shape[-1] if x_note_all is not None else 768
    hidden_dim = EXP_CONFIG.get("hidden_dim", 128)
    tf_num_layers = EXP_CONFIG.get("tf_num_layers", 2)
    tf_nhead = EXP_CONFIG.get("tf_nhead", 4)
    bert_model = EXP_CONFIG.get("e2e_bert_model", "emilyalsentzer/Bio_ClinicalBERT")

    logger.info("Instantiating TransformerEndToEndWithNotes (hidden_dim=%d, layers=%d, heads=%d) ...",
                hidden_dim, tf_num_layers, tf_nhead)
    model = TransformerEndToEndWithNotes(
        seq_dim=seq_dim,
        static_dims=static_dims,
        multihot_dims=multihot_dims,
        hidden_dim=hidden_dim,
        note_dim=note_dim,
        num_layers=tf_num_layers,
        nhead=tf_nhead,
        dropout=0.0,
        bert_model_name=bert_model,
    ).to(device)

    logger.info("Loading checkpoint weights from %s ...", ckpt_path)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # 选取测试分析样本
    N = len(x_seq_all)
    n = min(args.n_samples, N)
    idx = np.random.default_rng(42).choice(N, size=n, replace=False)

    x_seq = torch.from_numpy(x_seq_all[idx]).to(device)
    x_static = torch.from_numpy(x_static_all[idx]).to(device) if x_static_all is not None else None
    x_mh = torch.from_numpy(x_mh_all[idx]).to(device) if x_mh_all is not None else None
    x_note = torch.from_numpy(x_note_all[idx]).to(device) if x_note_all is not None else None

    # 构建端到端文本分词批数据
    x_note_tokens = None
    if model.tokenizer is not None and note_texts_dict:
        logger.info("Tokenizing note texts for %d evaluation samples ...", n)
        # 如果样本索引与字典键对齐，提取对应文本
        sample_texts = []
        for i in idx:
            text = note_texts_dict.get(int(i), "") or note_texts_dict.get(i, "")
            sample_texts.append(str(text))
        encoded = model.tokenizer(sample_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        x_note_tokens = {k: v.to(device) for k, v in encoded.items()}

    channel_names = SEQ_CHANNEL_NAMES_8DIM[:seq_dim]

    # 执行分析
    logger.info("Running Method 1: Integrated Gradients on 8 Vital Channels (steps=%d) ...", args.ig_steps)
    gxi = gradient_x_input_importance(
        model, x_seq, x_static, x_mh, x_note, x_note_tokens, channel_names, device, n_ig_steps=args.ig_steps
    )

    logger.info("Running Method 2: Module-Level Ablation Study ...")
    abl = ablation_module_importance(
        model, x_seq, x_static, x_mh, x_note, x_note_tokens, device
    )

    logger.info("Running Method 3: Transformer Temporal Attention Extraction ...")
    temporal_attn = extract_temporal_attention(
        model, x_seq, x_static, x_mh, x_note, x_note_tokens, device
    )

    logger.info("Running Method 4: Static Features Per-Column Importance ...")
    sf = static_feature_importance(
        model, x_seq, x_static, x_mh, x_note, x_note_tokens, static_dims, device, ig_steps=args.ig_steps
    )

    # 整合报告
    exp_label = exp_dir.name
    report = {
        "model_name": "Transformer E2E FineTune (+ Notes)",
        "checkpoint": str(ckpt_path),
        "dataset_cache": str(cache_path),
        "n_samples": n,
        "integrated_gradients": gxi,
        "ablation": abl,
        "temporal_attention": temporal_attn,
        "static_feature_importance": sf,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"feature_importance_{exp_label}.json"
    txt_path = out_dir / f"feature_importance_{exp_label}_summary.txt"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_text_summary(report, txt_path, static_dims)

    print_report(report, static_dims)
    logger.info("Analysis complete! Reports saved to:")
    logger.info("  JSON: %s", json_path)
    logger.info("  TXT:  %s", txt_path)


if __name__ == "__main__":
    main()
