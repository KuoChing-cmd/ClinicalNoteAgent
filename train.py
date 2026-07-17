import logging
import os
import pickle
from datetime import datetime

os.makedirs("output", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("output/training.log"), logging.StreamHandler()],
)
import json
import math

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data_loader import fetch_mimic3_data, flatten_features
from src.engine import (
    _evaluate_xgb_probs,
    evaluate_model,
    pretrain_transformer,
    train_model,
)
from src.models import (
    CrossModalAttnFusion,
    GatedFusionWithNotes,
    LSTMLateFusionWithNotes,
    PretrainedTransformerCrossModalFusion,
    TransformerEarlyFusionWithNotes,
    TransformerPretrainer,
    TransformerSeqEncoder,
)

# ─── Experiment Configuration ────────────────────────────────────────────────
# Edit these values to configure a run. The exp dir name is auto-generated.
EXP_CONFIG = {
    "epochs": 200,
    "lr": 5e-4,  # ↓ from 1e-3; smaller LR to reduce overfitting / allow longer convergence
    "batch_size": 256,
    "hidden_dim": 128,
    "num_layers": 4,  # for LSTM
    "tf_num_layers": 2,  # ↓ from 4; reduce Transformer depth
    "tf_nhead": 4,  # ↓ from 8; reduce Transformer heads
    "dropout": 0.3,  # ↑ from 0.2; add more regularization globally
    "note_dim": None,  # auto-detected from embeddings (768 for ClinicalBERT, 4096 for Llama)
    "task": "readmission",  # "readmission" or "mortality"
    "top_k_codes": 64,  # top-K for ICD/DRG/Proc/Rx multi-hot
    "xgb_n_est": 200,
    "xgb_depth": 6,
    "lgb_n_est": 200,
    "lgb_depth": 6,
    "early_stop_patience": 30,  # ↑ from 15; give model more epochs to find better minimum
    # Note embedding selection
    "note_embedding_type": "clinicalbert",  # "clinicalbert" (768d) or "llama" (4096d)
    # Optimizations enabled in this run
    "opt_seq_norm": True,  # ① StandardScaler on sequence features
    "opt_cosine_lr": True,  # ② CosineAnnealingLR scheduler
    "opt_pos_weight": True,  # ③ pos_weight for class imbalance
    "static_mode": "multi_token",  # Enable fine-grained multi-token mode
}


def make_exp_dir() -> str:
    """Create a timestamped, parameter-tagged experiment output directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lr_str = f"lr{EXP_CONFIG['lr']:.0e}".replace("-", "m")  # e.g. lr1e-3 -> lr1em3
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
    exp_dir = os.path.join(
        "output", f"{ts}_{EXP_CONFIG.get('task', 'readmission')}_{tag}"
    )
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def run_experiment_for_task(task_name):
    EXP_CONFIG["task"] = task_name
    # ── Determine embedding file ──────────────────────────────────────────────
    emb_type = EXP_CONFIG["note_embedding_type"]
    if emb_type == "clinicalbert":
        emb_path = "output/mimic3_note_summaries_clinicalbert.pkl"
    else:
        emb_path = "output/mimic3_note_summaries.pkl"

    if not os.path.exists(emb_path):
        logging.error(f"Embeddings file not found: {emb_path}")
        if emb_type == "clinicalbert":
            logging.error("Run preprocess_clinicalbert_embeddings.py first.")
        else:
            logging.error("Run preprocess_note_embeddings.py first.")
        return

    # Create the experiment directory for this run
    exp_dir = make_exp_dir()
    run_start = datetime.now()
    logging.info(f"Experiment directory: {exp_dir}")
    # Add a per-experiment log handler
    exp_log_handler = logging.FileHandler(os.path.join(exp_dir, "training.log"))
    exp_log_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logging.getLogger().addHandler(exp_log_handler)

    logging.info(f"Note embedding type: {emb_type} (from {emb_path})")
    with open(emb_path, "rb") as f:
        embeddings_dict = pickle.load(f)

    # Auto-detect embedding dimension from data
    sample_val = next(iter(embeddings_dict.values()))
    if isinstance(sample_val, dict):
        note_dim_detected = len(sample_val.get("embedding", []))
    else:
        note_dim_detected = len(sample_val)
    EXP_CONFIG["note_dim"] = note_dim_detected
    logging.info(f"Auto-detected note_dim = {note_dim_detected}")

    # ── Dataset cache: skip 17-min DuckDB pipeline on repeat runs ─────────────
    # Cache is per embedding type to avoid conflicts
    task = EXP_CONFIG.get("task", "readmission")
    cache_tag = f"_{emb_type}" if emb_type != "llama" else ""
    cache_npz = f"output/dataset_cache_{task}{cache_tag}_15dim_filtered_icuload.npz"
    cache_meta = (
        f"output/dataset_cache_{task}{cache_tag}_meta_15dim_filtered_icuload.pkl"
    )

    if os.path.exists(cache_npz) and os.path.exists(cache_meta):
        logging.info(f"Loading cached dataset from {cache_npz} ...")
        import time as _time

        _t0 = _time.time()
        data = np.load(cache_npz)
        X_seq = data["X_seq"]
        X_static = data["X_static"]
        X_mh = data["X_mh"]
        X_note = data["X_note"]
        Y = data["Y"]
        with open(cache_meta, "rb") as f:
            meta = pickle.load(f)
        static_dims = meta["static_dims"]
        multihot_dims = meta["multihot_dims"]
        logging.info(
            f"Cache loaded: {len(Y)} samples, "
            f"X_seq={X_seq.shape}, X_note={X_note.shape} "
            f"({_time.time()-_t0:.1f}s)"
        )
    else:
        logging.info("No dataset cache found — running full DuckDB pipeline...")
        X_seq, X_static, X_mh, X_note, Y, static_dims, multihot_dims = (
            fetch_mimic3_data(
                embeddings_dict, note_emb_dim=note_dim_detected, task=task
            )
        )
        if X_seq is None:
            return
        # Save cache for future runs
        np.savez_compressed(
            cache_npz,
            X_seq=X_seq,
            X_static=X_static,
            X_mh=X_mh,
            X_note=X_note,
            Y=Y,
        )
        with open(cache_meta, "wb") as f:
            pickle.dump({"static_dims": static_dims, "multihot_dims": multihot_dims}, f)
        logging.info(f"Dataset cached to {cache_npz} + {cache_meta}")

    ordered_static_dims = static_dims

    np.random.seed(42)  # Bug4 fix: set seed for reproducible train/val/test split
    idx = np.random.permutation(len(Y))
    n_test = int(0.20 * len(Y))
    n_val = int(0.10 * len(Y))
    n_train = len(Y) - n_val - n_test
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    logging.info(
        f"Data split — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}"
    )

    # ── Optimization ①: Sequence Feature Normalization ────────────────────────
    # Fit StandardScaler ONLY on train split to prevent data leakage.
    # Reshape (N, T, F) -> (N*T, F) for fitting, then reshape back.
    logging.info("Applying StandardScaler to sequence features (fit on train only)...")
    N_train, T, F = X_seq[train_idx].shape
    seq_scaler = StandardScaler()
    X_seq_train_flat = X_seq[train_idx].reshape(-1, F)
    seq_scaler.fit(X_seq_train_flat)
    X_seq = (
        seq_scaler.transform(X_seq.reshape(-1, F))
        .reshape(X_seq.shape[0], T, F)
        .astype(np.float32)
    )
    pickle.dump(seq_scaler, open(os.path.join(exp_dir, "seq_scaler.pkl"), "wb"))
    logging.info(f"Sequence scaler saved to {exp_dir}/seq_scaler.pkl")

    # ── Optimization ③: Compute pos_weight from training labels ──────────────
    Y_train = Y[train_idx]
    n_pos = Y_train.sum()
    n_neg = len(Y_train) - n_pos
    pos_weight_val = float(n_neg / n_pos) if n_pos > 0 else 1.0
    logging.info(
        f"Class distribution — positives: {int(n_pos)}, negatives: {int(n_neg)}, pos_weight: {pos_weight_val:.2f}"
    )

    # Collect epoch histories for all models
    all_epoch_histories = {}

    logging.info(
        "\n================ ABLATION STUDY: CLINICAL ALIGNMENT ================"
    )

    logging.info(
        "1. Training Base LSTM (Clinical Series + Demographics + ICD/DRG/Proc/Rx, NO Notes)..."
    )
    model_base = LSTMLateFusionWithNotes(
        seq_dim=15,
        static_dims=ordered_static_dims,
        multihot_dims=multihot_dims,
        use_notes=False,
        num_layers=EXP_CONFIG["num_layers"],
        dropout=EXP_CONFIG["dropout"],
    )
    model_base, hist_base = train_model(
        model_base,
        X_seq[train_idx],
        Y[train_idx],
        X_static[train_idx],
        X_mh[train_idx],
        X_seq_val=X_seq[val_idx],
        Y_val=Y[val_idx],
        X_static_val=X_static[val_idx],
        X_mh_val=X_mh[val_idx],
        epochs=EXP_CONFIG["epochs"],
        lr=EXP_CONFIG["lr"],
        batch_size=EXP_CONFIG["batch_size"],
        early_stop_patience=EXP_CONFIG["early_stop_patience"],
        pos_weight=pos_weight_val,
    )
    all_epoch_histories["lstm_base"] = hist_base
    eval_base = evaluate_model(
        model_base, X_seq[test_idx], Y[test_idx], X_static[test_idx], X_mh[test_idx]
    )
    _log_eval_result("LSTM Base", eval_base)
    torch.save(model_base.state_dict(), os.path.join(exp_dir, "model_lstm_base.pt"))
    logging.info(f"Base LSTM model saved to {exp_dir}/model_lstm_base.pt")

    logging.info("2. Training Late Fusion LSTM (Base + LLM Notes Embedding)...")
    model_notes = LSTMLateFusionWithNotes(
        seq_dim=15,
        static_dims=ordered_static_dims,
        multihot_dims=multihot_dims,
        note_dim=EXP_CONFIG["note_dim"],
        use_notes=True,
        num_layers=EXP_CONFIG["num_layers"],
        dropout=EXP_CONFIG["dropout"],
    )
    model_notes, hist_notes = train_model(
        model_notes,
        X_seq[train_idx],
        Y[train_idx],
        X_static[train_idx],
        X_mh[train_idx],
        X_note[train_idx],
        X_seq_val=X_seq[val_idx],
        Y_val=Y[val_idx],
        X_static_val=X_static[val_idx],
        X_mh_val=X_mh[val_idx],
        X_note_val=X_note[val_idx],
        epochs=EXP_CONFIG["epochs"],
        lr=EXP_CONFIG["lr"],
        batch_size=EXP_CONFIG["batch_size"],
        early_stop_patience=EXP_CONFIG["early_stop_patience"],
        pos_weight=pos_weight_val,
    )
    all_epoch_histories["lstm_latefusion_notes"] = hist_notes
    eval_notes = evaluate_model(
        model_notes,
        X_seq[test_idx],
        Y[test_idx],
        X_static[test_idx],
        X_mh[test_idx],
        X_note[test_idx],
    )
    _log_eval_result("LSTM LateFusion", eval_notes)
    torch.save(model_notes.state_dict(), os.path.join(exp_dir, "model_lstm_notes.pt"))
    logging.info(f"Late Fusion LSTM model saved to {exp_dir}/model_lstm_notes.pt")

    logging.info("3. Training Early Fusion Transformer (Base + LLM Notes Embedding)...")
    model_tf_notes = TransformerEarlyFusionWithNotes(
        seq_dim=15,
        static_dims=ordered_static_dims,
        multihot_dims=multihot_dims,
        note_dim=EXP_CONFIG["note_dim"],
        use_notes=True,
        num_layers=EXP_CONFIG["tf_num_layers"],
        nhead=EXP_CONFIG["tf_nhead"],
        dropout=EXP_CONFIG["dropout"],
    )
    model_tf_notes, hist_tf = train_model(
        model_tf_notes,
        X_seq[train_idx],
        Y[train_idx],
        X_static[train_idx],
        X_mh[train_idx],
        X_note[train_idx],
        X_seq_val=X_seq[val_idx],
        Y_val=Y[val_idx],
        X_static_val=X_static[val_idx],
        X_mh_val=X_mh[val_idx],
        X_note_val=X_note[val_idx],
        epochs=EXP_CONFIG["epochs"],
        lr=EXP_CONFIG["lr"],
        batch_size=EXP_CONFIG["batch_size"],
        early_stop_patience=EXP_CONFIG["early_stop_patience"],
        pos_weight=pos_weight_val,
    )
    all_epoch_histories["transformer_earlyfusion_notes"] = hist_tf
    eval_tf = evaluate_model(
        model_tf_notes,
        X_seq[test_idx],
        Y[test_idx],
        X_static[test_idx],
        X_mh[test_idx],
        X_note[test_idx],
    )
    _log_eval_result("Transformer EarlyFusion", eval_tf)
    torch.save(model_tf_notes.state_dict(), os.path.join(exp_dir, "model_tf_notes.pt"))
    logging.info(f"Early Fusion Transformer model saved to {exp_dir}/model_tf_notes.pt")

    logging.info(
        "4. Training Cross-Modal Attention Fusion (LSTM × Note Cross-Attention)..."
    )
    model_cross = CrossModalAttnFusion(
        seq_dim=15,
        static_dims=ordered_static_dims,
        multihot_dims=multihot_dims,
        hidden_dim=EXP_CONFIG["hidden_dim"],
        note_dim=EXP_CONFIG["note_dim"],
        nhead=EXP_CONFIG["tf_nhead"],
        num_virtual_tokens=2,
        num_lstm_layers=EXP_CONFIG["num_layers"],
        dropout=EXP_CONFIG["dropout"],
    )
    model_cross, hist_cross = train_model(
        model_cross,
        X_seq[train_idx],
        Y[train_idx],
        X_static[train_idx],
        X_mh[train_idx],
        X_note[train_idx],
        X_seq_val=X_seq[val_idx],
        Y_val=Y[val_idx],
        X_static_val=X_static[val_idx],
        X_mh_val=X_mh[val_idx],
        X_note_val=X_note[val_idx],
        epochs=EXP_CONFIG["epochs"],
        lr=EXP_CONFIG["lr"],
        batch_size=EXP_CONFIG["batch_size"],
        early_stop_patience=EXP_CONFIG["early_stop_patience"],
        pos_weight=pos_weight_val,
    )
    all_epoch_histories["crossmodal_attention"] = hist_cross
    eval_cross = evaluate_model(
        model_cross,
        X_seq[test_idx],
        Y[test_idx],
        X_static[test_idx],
        X_mh[test_idx],
        X_note[test_idx],
    )
    _log_eval_result("CrossModal Attention", eval_cross)
    torch.save(model_cross.state_dict(), os.path.join(exp_dir, "model_cross_attn.pt"))
    logging.info(f"Cross-Modal Attention model saved to {exp_dir}/model_cross_attn.pt")

    logging.info("5. Training Gated Fusion (LSTM × Note Gated Blend)...")
    model_gated = GatedFusionWithNotes(
        seq_dim=15,
        static_dims=ordered_static_dims,
        multihot_dims=multihot_dims,
        hidden_dim=EXP_CONFIG["hidden_dim"],
        note_dim=EXP_CONFIG["note_dim"],
        num_lstm_layers=EXP_CONFIG["num_layers"],
        dropout=EXP_CONFIG["dropout"],
    )
    model_gated, hist_gated = train_model(
        model_gated,
        X_seq[train_idx],
        Y[train_idx],
        X_static[train_idx],
        X_mh[train_idx],
        X_note[train_idx],
        X_seq_val=X_seq[val_idx],
        Y_val=Y[val_idx],
        X_static_val=X_static[val_idx],
        X_mh_val=X_mh[val_idx],
        X_note_val=X_note[val_idx],
        epochs=EXP_CONFIG["epochs"],
        lr=EXP_CONFIG["lr"],
        batch_size=EXP_CONFIG["batch_size"],
        early_stop_patience=EXP_CONFIG["early_stop_patience"],
        pos_weight=pos_weight_val,
    )
    all_epoch_histories["gated_fusion"] = hist_gated
    eval_gated = evaluate_model(
        model_gated,
        X_seq[test_idx],
        Y[test_idx],
        X_static[test_idx],
        X_mh[test_idx],
        X_note[test_idx],
    )
    _log_eval_result("Gated Fusion", eval_gated)
    torch.save(model_gated.state_dict(), os.path.join(exp_dir, "model_gated_fusion.pt"))
    logging.info(f"Gated Fusion model saved to {exp_dir}/model_gated_fusion.pt")

    logging.info("6. Training Pretrained Transformer + Cross-Modal Attention...")
    # 1. Pretrain the encoder
    pretrained_encoder = TransformerSeqEncoder(
        seq_input_dim=15,
        hidden_dim=EXP_CONFIG["hidden_dim"],
        num_layers=EXP_CONFIG["tf_num_layers"],
        nhead=EXP_CONFIG["tf_nhead"],
        dropout=EXP_CONFIG["dropout"],
    )
    pretrained_encoder = pretrain_transformer(
        pretrained_encoder,
        X_seq[train_idx],
        seq_dim=15,
        hidden_dim=EXP_CONFIG["hidden_dim"],
        epochs=10,
        lr=1e-3,
        batch_size=EXP_CONFIG["batch_size"],
        mask_prob=0.15,
    )

    # 2. Instantiate downstream model
    model_pretrain_cross = PretrainedTransformerCrossModalFusion(
        encoder=pretrained_encoder,
        static_dims=ordered_static_dims,
        multihot_dims=multihot_dims,
        hidden_dim=EXP_CONFIG["hidden_dim"],
        note_dim=EXP_CONFIG["note_dim"],
        nhead=EXP_CONFIG["tf_nhead"],
        dropout=EXP_CONFIG["dropout"],
        num_virtual_tokens=2,
    )

    # 3. Fine-tune
    model_pretrain_cross, hist_pretrain_cross = train_model(
        model_pretrain_cross,
        X_seq[train_idx],
        Y[train_idx],
        X_static[train_idx],
        X_mh[train_idx],
        X_note[train_idx],
        X_seq_val=X_seq[val_idx],
        Y_val=Y[val_idx],
        X_static_val=X_static[val_idx],
        X_mh_val=X_mh[val_idx],
        X_note_val=X_note[val_idx],
        epochs=EXP_CONFIG["epochs"],
        lr=EXP_CONFIG["lr"],
        batch_size=EXP_CONFIG["batch_size"],
        early_stop_patience=EXP_CONFIG["early_stop_patience"],
        pos_weight=pos_weight_val,
    )
    all_epoch_histories["pretrained_crossmodal"] = hist_pretrain_cross
    eval_pretrain_cross = evaluate_model(
        model_pretrain_cross,
        X_seq[test_idx],
        Y[test_idx],
        X_static[test_idx],
        X_mh[test_idx],
        X_note[test_idx],
    )
    _log_eval_result("Pretrained CrossModal", eval_pretrain_cross)
    torch.save(
        model_pretrain_cross.state_dict(),
        os.path.join(exp_dir, "model_pretrained_cross_attn.pt"),
    )
    logging.info(
        f"Pretrained Cross-Modal Attention model saved to {exp_dir}/model_pretrained_cross_attn.pt"
    )

    # XGBoost natively handles class imbalance via scale_pos_weight (equivalent to pos_weight)
    X_xgb_base = flatten_features(X_seq, X_static, X_mh)
    xgb_base = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        scale_pos_weight=pos_weight_val,
        tree_method="hist",
        device="cuda",
    )
    xgb_base.fit(X_xgb_base[train_idx], Y[train_idx])
    xgb_base_probs = xgb_base.predict_proba(X_xgb_base[test_idx])[:, 1]
    eval_xgb_base = _evaluate_xgb_probs(Y[test_idx], xgb_base_probs)
    _log_eval_result("XGBoost Base", eval_xgb_base)
    xgb_base.save_model(os.path.join(exp_dir, "model_xgb_base.json"))
    logging.info(f"XGBoost Base model saved to {exp_dir}/model_xgb_base.json")

    X_xgb_notes = np.concatenate([X_xgb_base, X_note], axis=1)
    xgb_notes = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        scale_pos_weight=pos_weight_val,
        tree_method="hist",
        device="cuda",
    )
    xgb_notes.fit(X_xgb_notes[train_idx], Y[train_idx])
    xgb_notes_probs = xgb_notes.predict_proba(X_xgb_notes[test_idx])[:, 1]
    eval_xgb_notes = _evaluate_xgb_probs(Y[test_idx], xgb_notes_probs)
    _log_eval_result("XGBoost + Notes", eval_xgb_notes)
    xgb_notes.save_model(os.path.join(exp_dir, "model_xgb_notes.json"))
    logging.info(f"XGBoost + LLM Notes model saved to {exp_dir}/model_xgb_notes.json")

    # Find categorical indices for LightGBM
    # X_xgb_base layout: [mean_seq (8), last_seq (8), static (cont + cat), mh]
    num_seq_cols = X_seq.shape[2] * 2
    cat_indices = []
    current_idx = num_seq_cols
    for col, dim in ordered_static_dims.items():
        if dim > 0:  # dim > 0 indicates it's a categorical feature
            cat_indices.append(current_idx)
        current_idx += 1

    # LightGBM natively handles class imbalance via scale_pos_weight
    lgb_base = lgb.LGBMClassifier(
        n_estimators=EXP_CONFIG["lgb_n_est"],
        max_depth=EXP_CONFIG["lgb_depth"],
        scale_pos_weight=pos_weight_val,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_base.fit(X_xgb_base[train_idx], Y[train_idx], categorical_feature=cat_indices)
    lgb_base_probs = lgb_base.predict_proba(X_xgb_base[test_idx])[:, 1]
    eval_lgb_base = _evaluate_xgb_probs(Y[test_idx], lgb_base_probs)
    _log_eval_result("LightGBM Base", eval_lgb_base)
    lgb_base.booster_.save_model(os.path.join(exp_dir, "model_lgb_base.txt"))
    logging.info(f"LightGBM Base model saved to {exp_dir}/model_lgb_base.txt")

    lgb_notes = lgb.LGBMClassifier(
        n_estimators=EXP_CONFIG["lgb_n_est"],
        max_depth=EXP_CONFIG["lgb_depth"],
        scale_pos_weight=pos_weight_val,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_notes.fit(X_xgb_notes[train_idx], Y[train_idx], categorical_feature=cat_indices)
    lgb_notes_probs = lgb_notes.predict_proba(X_xgb_notes[test_idx])[:, 1]
    eval_lgb_notes = _evaluate_xgb_probs(Y[test_idx], lgb_notes_probs)
    _log_eval_result("LightGBM + Notes", eval_lgb_notes)
    lgb_notes.booster_.save_model(os.path.join(exp_dir, "model_lgb_notes.txt"))
    logging.info(f"LightGBM + LLM Notes model saved to {exp_dir}/model_lgb_notes.txt")

    # ── Collect all evaluation results ──────────────────────────────────────────
    all_eval_results = {
        "lstm_base": eval_base,
        "lstm_latefusion_notes": eval_notes,
        "transformer_earlyfusion_notes": eval_tf,
        "crossmodal_attention": eval_cross,
        "gated_fusion": eval_gated,
        "pretrained_crossmodal": eval_pretrain_cross,
        "xgb_base": eval_xgb_base,
        "xgb_notes": eval_xgb_notes,
        "lgb_base": eval_lgb_base,
        "lgb_notes": eval_lgb_notes,
    }
    # Remove non-serializable y_prob before saving
    eval_results_serializable = {}
    for k, v in all_eval_results.items():
        d = {kk: vv for kk, vv in v.items() if kk != "y_prob"}
        eval_results_serializable[k] = d

    eval_path = os.path.join(exp_dir, "evaluation_metrics.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results_serializable, f, indent=2, ensure_ascii=False)
    logging.info(f"Full evaluation metrics saved to {eval_path}")

    # ── Save epoch monitoring history to JSON ─────────────────────────────────
    epoch_monitor_path = os.path.join(exp_dir, "epoch_monitoring.json")
    with open(epoch_monitor_path, "w") as f:
        json.dump(all_epoch_histories, f, indent=2, ensure_ascii=False)
    logging.info(f"Epoch monitoring saved to {epoch_monitor_path}")

    # ── Helper to extract metrics for log / note ──────────────────────────────
    def _m(eval_result, key="metrics_at_best_t"):
        return eval_result[key]

    logging.info(
        "\n================ FINAL ABLATION RESULTS (Test Set) ================"
    )
    logging.info("---- @ Best F1 Threshold ----")
    for name, result in [
        ("XGBoost Base", eval_xgb_base),
        ("XGBoost + Notes", eval_xgb_notes),
        ("LightGBM Base", eval_lgb_base),
        ("LightGBM + Notes", eval_lgb_notes),
        ("LSTM Base", eval_base),
        ("LSTM LateFusion", eval_notes),
        ("Transformer EarlyFusion", eval_tf),
        ("CrossModal Attention", eval_cross),
        ("Gated Fusion", eval_gated),
        ("Transformer Pretrain+CrossModal", eval_pretrain_cross),
    ]:
        m = _m(result)
        logging.info(
            f"  {name:30s}  t={m['threshold']:.3f}  AUROC={m['roc_auc']:.4f}  PRAUC={m['pr_auc']:.4f}  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}  Brier={m['brier']:.4f}"
        )
    logging.info("========================================================\n")

    # ── Write experiment_note.md ──────────────────────────────────────────────
    run_end = datetime.now()
    duration = run_end - run_start
    h, rem = divmod(int(duration.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    duration_str = f"{h}h {m}m {s}s"

    # Build metrics table rows for experiment note
    def _note_row(name, result):
        mb = _m(result, "metrics_at_best_t")
        m5 = _m(result, "metrics_at_0.5")
        return (
            f"| {name} | {mb['threshold']:.3f} | {mb['roc_auc']:.4f} | {mb['pr_auc']:.4f} | "
            f"{mb['precision']:.4f} | {mb['recall']:.4f} | {mb['f1']:.4f} | {mb['accuracy']:.4f} | {mb['brier']:.4f} |"
        )

    def _note_row_05(name, result):
        m5 = _m(result, "metrics_at_0.5")
        return (
            f"| {name} | 0.500 | {m5['roc_auc']:.4f} | {m5['pr_auc']:.4f} | "
            f"{m5['precision']:.4f} | {m5['recall']:.4f} | {m5['f1']:.4f} | {m5['accuracy']:.4f} | {m5['brier']:.4f} |"
        )

    note_path = os.path.join(exp_dir, "experiment_note.md")
    with open(note_path, "w") as f:
        f.write(f"""# Experiment Note

## Run Info
| Item | Value |
|------|-------|
| Timestamp | {run_start.strftime('%Y-%m-%d %H:%M:%S')} |
| Duration  | {duration_str} |
| Exp Dir   | `{exp_dir}` |

## Dataset
| Item | Value |
|------|-------|
| Total samples  | {len(Y)} |
| Train / Val / Test | {len(train_idx)} / {len(val_idx)} / {len(test_idx)} |
| Positives (train) | {int(n_pos)} ({100*n_pos/len(Y_train):.1f}%) |
| Negatives (train) | {int(n_neg)} ({100*n_neg/len(Y_train):.1f}%) |
| pos_weight applied | {pos_weight_val:.2f} |

## Features
| Group | Features | Dim |
|-------|----------|-----|
| Vital signs (seq) | HR, RR, SpO₂, SBP, DBP, MAP, Temp, Glucose | 48×8 |
| Demographics (static) | age, gender, marital, ethnicity, insurance, adm_type, adm_loc, care_unit | 8 cols |
| Surgical (static) | surg_count, surg_flag, log_surg_gap | 3 cols |
| Admission (static) | pre_icu_transfers, log_ed_wait | 2 cols |
| **ICU pressure (static)** | **icu_speedup_los, log_icu_los** | **2 cols (NEW)** |
| Sparse codes (mh) | ICD-9 dx, DRG, ICD-9 proc, Rx (Top-64 each) | 256 |
| Clinical notes | ClinicalBERT discharge note embedding | {EXP_CONFIG['note_dim']} |

## Model Hyperparameters
| Param | Value |
|-------|-------|
| epochs     | {EXP_CONFIG['epochs']} |
| lr         | {EXP_CONFIG['lr']} |
| batch_size | {EXP_CONFIG['batch_size']} |
| hidden_dim | {EXP_CONFIG['hidden_dim']} |
| num_layers | {EXP_CONFIG['num_layers']} |
| tf_num_layers | {EXP_CONFIG['tf_num_layers']} |
| tf_nhead   | {EXP_CONFIG['tf_nhead']} |
| dropout    | {EXP_CONFIG['dropout']} |
| note_dim   | {EXP_CONFIG['note_dim']} |
| top_k_codes | {EXP_CONFIG['top_k_codes']} |
| xgb_n_estimators | {EXP_CONFIG['xgb_n_est']} |
| xgb_max_depth    | {EXP_CONFIG['xgb_depth']} |
| lgb_n_estimators | {EXP_CONFIG['lgb_n_est']} |
| lgb_max_depth    | {EXP_CONFIG['lgb_depth']} |
| early_stop_patience | {EXP_CONFIG['early_stop_patience']} |
| early_stop_min_delta | 1e-4 |

## Optimizations Applied
| # | Optimization | Enabled |
|---|---|---|
| ① | Sequence StandardScaler (fit on train only) | {'✅' if EXP_CONFIG['opt_seq_norm'] else '❌'} |
| ② | CosineAnnealingLR scheduler (eta_min = lr×0.01) | {'✅' if EXP_CONFIG['opt_cosine_lr'] else '❌'} |
| ③ | BCEWithLogitsLoss pos_weight / XGB scale_pos_weight | {'✅' if EXP_CONFIG['opt_pos_weight'] else '❌'} |
| ④ | Early Stopping (val_loss, patience={EXP_CONFIG['early_stop_patience']}) | ✅ |
| ⑤ | Best F1 threshold search | ✅ |
| ⑥ | ICU pressure features (load_index, speedup_los, log_icu_los) | ✅ |

## Ablation Study Results — Test Set @ Best F1 Threshold
| Model | Threshold | AUROC | PRAUC | Precision | Recall | F1 | Acc | Brier |
|-------|-----------|-------|-------|-----------|--------|----|-----|-------|
{_note_row('XGBoost Base', eval_xgb_base)}
{_note_row('XGBoost + LLM Notes', eval_xgb_notes)}
{_note_row('LightGBM Base', eval_lgb_base)}
{_note_row('LightGBM + LLM Notes', eval_lgb_notes)}
{_note_row('LSTM Base', eval_base)}
{_note_row('LSTM LateFusion (+ Notes)', eval_notes)}
{_note_row('Transformer EarlyFusion (+ Notes)', eval_tf)}
{_note_row('CrossModal Attn', eval_cross)}
{_note_row('Gated Fusion', eval_gated)}
{_note_row('**Pretrain+CrossModal**', eval_pretrain_cross)}

## Ablation Study Results — Test Set @ Fixed Threshold 0.5
| Model | Threshold | AUROC | PRAUC | Precision | Recall | F1 | Acc | Brier |
|-------|-----------|-------|-------|-----------|--------|----|-----|-------|
{_note_row_05('XGBoost Base', eval_xgb_base)}
{_note_row_05('XGBoost + LLM Notes', eval_xgb_notes)}
{_note_row_05('LightGBM Base', eval_lgb_base)}
{_note_row_05('LightGBM + LLM Notes', eval_lgb_notes)}
{_note_row_05('LSTM Base', eval_base)}
{_note_row_05('LSTM LateFusion (+ Notes)', eval_notes)}
{_note_row_05('Transformer EarlyFusion (+ Notes)', eval_tf)}
{_note_row_05('CrossModal Attn', eval_cross)}
{_note_row_05('Gated Fusion', eval_gated)}
{_note_row_05('**Pretrain+CrossModal**', eval_pretrain_cross)}

### Note Embedding Impact (AUROC)
- XGBoost: notes Δ AUC = {_m(eval_xgb_notes)['roc_auc'] - _m(eval_xgb_base)['roc_auc']:+.4f}
- LightGBM: notes Δ AUC = {_m(eval_lgb_notes)['roc_auc'] - _m(eval_lgb_base)['roc_auc']:+.4f}
- LSTM Late Fusion:       notes Δ AUC = {_m(eval_notes)['roc_auc'] - _m(eval_base)['roc_auc']:+.4f}
- CrossModal Attn Fusion: vs LSTM Base Δ AUC = {_m(eval_cross)['roc_auc'] - _m(eval_base)['roc_auc']:+.4f}
- **Gated Fusion: vs LSTM Base Δ AUC = {_m(eval_gated)['roc_auc'] - _m(eval_base)['roc_auc']:+.4f}**

## Saved Files
| File | Description |
|------|-------------|
| `model_lstm_base.pt`   | Base LSTM state_dict |
| `model_lstm_notes.pt`  | Late Fusion LSTM state_dict |
| `model_tf_notes.pt`    | Early Fusion Transformer state_dict |
| `model_cross_attn.pt`  | CrossModal Attention Fusion state_dict |
| `model_gated_fusion.pt` | Gated Fusion state_dict |
| `model_pretrained_cross_attn.pt` | Pretrained CrossModal Fusion state_dict |
| `model_xgb_base.json`  | XGBoost Base (XGBoost native format) |
| `model_xgb_notes.json` | XGBoost + Notes (XGBoost native format) |
| `model_lgb_base.txt`   | LightGBM Base |
| `model_lgb_notes.txt`  | LightGBM + Notes |
| `seq_scaler.pkl`       | StandardScaler for sequence features (required for inference) |
| `training.log`        | Full training log for this run |
| `epoch_monitoring.json` | Per-epoch train_loss / val_loss / val_AUROC / val_PRAUC for all models |
| `evaluation_metrics.json` | Full classification metrics for all models (threshold, P/R/F1, AUROC, PRAUC, Brier, confusion matrix) |
| `experiment_note.md`  | This file |
""")
    logging.info(f"Experiment note written to {note_path}")
    logging.getLogger().removeHandler(exp_log_handler)


def main():
    for t in ["readmission", "mortality"]:
        logging.info("=" * 60)
        logging.info(f"STARTING FULL PIPELINE FOR TASK: {t.upper()}")
        logging.info("=" * 60)
        run_experiment_for_task(t)


if __name__ == "__main__":
    main()
