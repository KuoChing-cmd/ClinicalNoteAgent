import os
import sys
import json
import pickle
import logging
from datetime import datetime
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from typing import cast

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

from config import EXP_CONFIG
from models import TransformerEarlyFusionEndToEndWithNotes
from data import load_note_texts
from trainer import train_e2e_model, evaluate_model

def _build_note_texts_from_summary_map(note_texts_map, sample_count):
    if sample_count <= 0:
        return []
    texts = [""] * sample_count
    for i in range(sample_count):
        texts[i] = str(note_texts_map.get(i, ""))
    return texts

def main():
    emb_type = "clinicalbert"
    emb_path = "output/mimic3_note_embeddings_clinicalbert.pkl"
    note_summary_path = "output/mimic3_note_summaries.pkl"
    
    if os.path.exists(note_summary_path):
        with open(note_summary_path, "rb") as f:
            note_summary_dict = pickle.load(f)
        note_texts_map = load_note_texts(note_summary_dict)
    else:
        note_texts_map = {}
        
    cache_tag = "_clinicalbert"
    cache_npz  = f"output/dataset_cache{cache_tag}_8dim_icuload_nibp_drgemb_icdsplit.npz"
    cache_meta = f"output/dataset_cache{cache_tag}_meta_8dim_icuload_nibp_drgemb_icdsplit.pkl"
    
    if not os.path.exists(cache_npz):
        logging.error("Cache missing, run full train.py first")
        return
        
    data = np.load(cache_npz)
    X_seq    = data["X_seq"]
    X_static = data["X_static"]
    X_mh     = data["X_mh"]
    Y        = data["Y"]
    
    with open(cache_meta, "rb") as f:
        meta = pickle.load(f)
    static_dims   = meta["static_dims"]
    multihot_dims = meta["multihot_dims"]
    ordered_static_dims = static_dims

    X_note_texts = _build_note_texts_from_summary_map(note_texts_map, len(Y))
    
    if EXP_CONFIG["opt_seq_norm"]:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        orig_shape = X_seq.shape
        X_seq_reshaped = X_seq.reshape(-1, orig_shape[-1])
        X_seq_scaled = scaler.fit_transform(X_seq_reshaped)
        X_seq = X_seq_scaled.reshape(orig_shape)
        
    n_pos = np.sum(Y == 1)
    n_neg = np.sum(Y == 0)
    pos_weight_val = float(n_neg / n_pos) if EXP_CONFIG["opt_pos_weight"] else 1.0

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(skf.split(np.zeros(len(Y)), Y))
    skf_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx_final, val_idx = next(skf_val.split(np.zeros(len(train_idx)), Y[train_idx]))
    val_idx = train_idx[val_idx]
    train_idx = train_idx[train_idx_final]

    model = TransformerEarlyFusionEndToEndWithNotes(
        seq_dim=8,
        static_dims=ordered_static_dims,
        multihot_dims=multihot_dims,
        hidden_dim=EXP_CONFIG["hidden_dim"],
        note_dim=768,
        num_layers=EXP_CONFIG["tf_num_layers"],
        nhead=EXP_CONFIG["tf_nhead"],
        dropout=EXP_CONFIG["dropout"],
        bert_model_name=EXP_CONFIG["e2e_bert_model"],
        freeze_layers=EXP_CONFIG.get("e2e_freeze_layers", 8),
        bert_lr_scale=EXP_CONFIG.get("e2e_bert_lr_scale", 0.1),
    )
    
    logging.info("Training EarlyFusion E2E for 5 epochs...")
    model, hist = train_e2e_model(
        model,
        X_seq[train_idx], Y[train_idx], X_static[train_idx], X_mh[train_idx],
        [X_note_texts[i] for i in train_idx],
        X_seq_val=X_seq[val_idx], Y_val=Y[val_idx], X_static_val=X_static[val_idx], X_mh_val=X_mh[val_idx],
        X_note_texts_val=[X_note_texts[i] for i in val_idx],
        epochs=5, lr=EXP_CONFIG["lr"], batch_size=EXP_CONFIG.get("e2e_batch_size", 32),
        early_stop_patience=EXP_CONFIG["early_stop_patience"], pos_weight=pos_weight_val,
        max_seq_len=EXP_CONFIG.get("e2e_max_seq_len", 512),
    )
    
    eval_res = evaluate_model(model, X_seq[test_idx], Y[test_idx], X_static[test_idx], X_mh[test_idx], X_note_texts=[X_note_texts[i] for i in test_idx], batch_size=EXP_CONFIG.get("e2e_batch_size", 32), tokenizer=model.tokenizer)
    
    metrics = eval_res["metrics_at_best_t"]
    logging.info(f"Test AUROC: {metrics["roc_auc"]:.4f}")
    logging.info(f"Test PR-AUC: {metrics["pr_auc"]:.4f}")
    logging.info(f"Test Best F1: {metrics["f1"]:.4f}")

if __name__ == "__main__":
    main()
