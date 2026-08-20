import os
import pickle
import logging
from datetime import datetime
import json
import numpy as np
import pandas as pd
import duckdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb
import math

from config import EXP_CONFIG

def _compute_prauc(y_true, y_score):
    """Compute area under precision-recall curve via trapezoidal rule."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.sum() == 0:
        return float('nan')
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    tp_cumsum = np.cumsum(y_sorted)
    n_pos = int(y_true.sum())
    recalls = tp_cumsum / n_pos
    precisions = tp_cumsum / np.arange(1, len(y_sorted) + 1)
    recalls = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[1.0], precisions])
    return float(getattr(np, 'trapezoid', getattr(np, 'trapz', None))(precisions, recalls))


def _search_best_threshold_f1(y_true, y_prob):
    """Search threshold in [0.01, 0.99] that maximises F1."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 99):
        pred = (y_prob >= t).astype(np.int64)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2.0 * p * r / max(1e-12, p + r)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _compute_classification_metrics(y_true, y_prob, threshold=0.5):
    """Compute comprehensive binary classification metrics at a given threshold."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)
    
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    
    total = max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    accuracy  = (tp + tn) / total
    specificity = tn / max(1, tn + fp)
    
    # Brier score
    brier = float(np.mean((y_prob - y_true) ** 2))
    
    return {
        'threshold': round(threshold, 4),
        'accuracy':  round(accuracy, 4),
        'precision': round(precision, 4),
        'recall':    round(recall, 4),
        'f1':        round(f1, 4),
        'specificity': round(specificity, 4),
        'roc_auc':   round(roc_auc_score(y_true, y_prob), 4),
        'pr_auc':    round(_compute_prauc(y_true, y_prob), 4),
        'brier':     round(brier, 4),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


def _evaluate_xgb_probs(y_true, y_prob):
    """Compute full metrics for XGBoost predictions (already have probabilities)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    best_t = _search_best_threshold_f1(y_true, y_prob)
    return {
        'best_threshold': best_t,
        'metrics_at_best_t': _compute_classification_metrics(y_true, y_prob, threshold=best_t),
        'metrics_at_0.5':   _compute_classification_metrics(y_true, y_prob, threshold=0.5),
    }


def _log_eval_result(name, result):
    """Log evaluation metrics for one model."""
    m_best = result['metrics_at_best_t']
    m_05   = result['metrics_at_0.5']
    logging.info(
        f"  {name}:\n"
        f"    @threshold=0.5:          AUROC={m_05['roc_auc']:.4f}  PRAUC={m_05['pr_auc']:.4f}  "
        f"P={m_05['precision']:.4f}  R={m_05['recall']:.4f}  F1={m_05['f1']:.4f}  "
        f"Acc={m_05['accuracy']:.4f}  Brier={m_05['brier']:.4f}\n"
        f"    @best_threshold={m_best['threshold']:.3f}:  AUROC={m_best['roc_auc']:.4f}  PRAUC={m_best['pr_auc']:.4f}  "
        f"P={m_best['precision']:.4f}  R={m_best['recall']:.4f}  F1={m_best['f1']:.4f}  "
        f"Acc={m_best['accuracy']:.4f}  Brier={m_best['brier']:.4f}"
    )

def make_exp_dir() -> str:
    """Create a timestamped, parameter-tagged experiment output directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lr_str = f"lr{EXP_CONFIG['lr']:.0e}".replace('-', 'm')   # e.g. lr1e-3 -> lr1em3
    tag = (
        f"ep{EXP_CONFIG['epochs']}"
        f"_{lr_str}"
        f"_hd{EXP_CONFIG['hidden_dim']}"
        f"_bs{EXP_CONFIG['batch_size']}"
        f"{'_seqnorm' if EXP_CONFIG['opt_seq_norm'] else ''}"
        f"{'_cosinelr' if EXP_CONFIG['opt_cosine_lr'] else ''}"
        f"{'_posw' if EXP_CONFIG['opt_pos_weight'] else ''}"
        f"_icuload"
        f"_{EXP_CONFIG['note_embedding_type']}"
    )
    exp_dir = os.path.join('output', f"{ts}_{tag}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

