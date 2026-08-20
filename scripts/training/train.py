import os
import math

# Automatically set NUMEXPR_MAX_THREADS based on estimated idle CPU cores
try:
    _total_cores = os.cpu_count() or 4
    _load_1m, _, _ = os.getloadavg()
    _idle_cores = max(1, int(math.floor(_total_cores - _load_1m)))
    os.environ['NUMEXPR_MAX_THREADS'] = str(_idle_cores)
except Exception:
    pass
import pickle
import logging
from datetime import datetime

os.makedirs('output', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('output/training.log'),
        logging.StreamHandler()
    ]
)
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
from models import LSTMLateFusionWithNotes, TransformerLateFusionWithNotes, TransformerEarlyFusionWithNotes
from data import build_multihot_features, enrich_stays_with_features, compute_icu_load_features, fetch_mimic3_data, flatten_features
from metrics import _search_best_threshold_f1, _compute_classification_metrics, _evaluate_xgb_probs, _log_eval_result, make_exp_dir
from trainer import train_model, pretrain_transformer, evaluate_model

def main():
    # ── Determine embedding file ──────────────────────────────────────────────
    emb_type = EXP_CONFIG['note_embedding_type']
    if emb_type == 'clinicalbert':
        emb_path = 'output/mimic3_note_embeddings_clinicalbert.pkl'
    else:
        emb_path = 'output/mimic3_note_embeddings.pkl'
    
    if not os.path.exists(emb_path):
        logging.error(f"Embeddings file not found: {emb_path}")
        if emb_type == 'clinicalbert':
            logging.error("Run preprocess_clinicalbert.py first.")
        else:
            logging.error("Run preprocess_note_embeddings.py first.")
        return
    
    # Create the experiment directory for this run
    exp_dir = make_exp_dir()
    run_start = datetime.now()
    logging.info(f"Experiment directory: {exp_dir}")
    # Add a per-experiment log handler
    exp_log_handler = logging.FileHandler(os.path.join(exp_dir, 'training.log'))
    exp_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logging.getLogger().addHandler(exp_log_handler)
    
    logging.info(f"Note embedding type: {emb_type} (from {emb_path})")
    with open(emb_path, 'rb') as f:
        embeddings_dict = pickle.load(f)
    
    # Auto-detect embedding dimension from data
    sample_val = next(iter(embeddings_dict.values()))
    if isinstance(sample_val, dict):
        note_dim_detected = len(sample_val.get('embedding', []))
    else:
        note_dim_detected = len(sample_val)
    EXP_CONFIG['note_dim'] = note_dim_detected
    logging.info(f"Auto-detected note_dim = {note_dim_detected}")
    
    # ── Dataset cache: skip 17-min DuckDB pipeline on repeat runs ─────────────
    # Cache is per embedding type to avoid conflicts
    cache_tag = f'_{emb_type}' if emb_type != 'llama' else ''
    cache_npz  = f'output/dataset_cache{cache_tag}_8dim_icuload.npz'
    cache_meta = f'output/dataset_cache{cache_tag}_meta_8dim_icuload.pkl'
    
    if os.path.exists(cache_npz) and os.path.exists(cache_meta):
        logging.info(f"Loading cached dataset from {cache_npz} ...")
        import time as _time
        _t0 = _time.time()
        data = np.load(cache_npz)
        X_seq    = data['X_seq']
        X_static = data['X_static']
        X_mh     = data['X_mh']
        X_note   = data['X_note']
        Y        = data['Y']
        with open(cache_meta, 'rb') as f:
            meta = pickle.load(f)
        static_dims   = meta['static_dims']
        multihot_dims = meta['multihot_dims']
        logging.info(
            f"Cache loaded: {len(Y)} samples, "
            f"X_seq={X_seq.shape}, X_note={X_note.shape} "
            f"({_time.time()-_t0:.1f}s)"
        )
    else:
        logging.info("No dataset cache found — running full DuckDB pipeline...")
        X_seq, X_static, X_mh, X_note, Y, static_dims, multihot_dims = fetch_mimic3_data(
            embeddings_dict, note_emb_dim=note_dim_detected
        )
        if X_seq is None: return
        # Save cache for future runs
        np.savez_compressed(
            cache_npz,
            X_seq=X_seq, X_static=X_static, X_mh=X_mh, X_note=X_note, Y=Y,
        )
        with open(cache_meta, 'wb') as f:
            pickle.dump({'static_dims': static_dims, 'multihot_dims': multihot_dims}, f)
        logging.info(f"Dataset cached to {cache_npz} + {cache_meta}")
    
    ordered_static_dims = static_dims
                           
    np.random.seed(42)  # Bug4 fix: set seed for reproducible train/val/test split
    idx = np.random.permutation(len(Y))
    n_test = int(0.20 * len(Y))
    n_val  = int(0.10 * len(Y))
    n_train = len(Y) - n_val - n_test
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]
    logging.info(f"Data split — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    
    # ── Optimization ①: Sequence Feature Normalization ────────────────────────
    # Fit StandardScaler ONLY on train split to prevent data leakage.
    # Reshape (N, T, F) -> (N*T, F) for fitting, then reshape back.
    logging.info("Applying StandardScaler to sequence features (fit on train only)...")
    N_train, T, F = X_seq[train_idx].shape
    seq_scaler = StandardScaler()
    X_seq_train_flat = X_seq[train_idx].reshape(-1, F)
    seq_scaler.fit(X_seq_train_flat)
    X_seq = seq_scaler.transform(X_seq.reshape(-1, F)).reshape(X_seq.shape[0], T, F).astype(np.float32)
    pickle.dump(seq_scaler, open(os.path.join(exp_dir, 'seq_scaler.pkl'), 'wb'))
    logging.info(f"Sequence scaler saved to {exp_dir}/seq_scaler.pkl")
    
    # ── Optimization ③: Compute pos_weight from training labels ──────────────
    Y_train = Y[train_idx]
    n_pos = Y_train.sum()
    n_neg = len(Y_train) - n_pos
    pos_weight_val = float(n_neg / n_pos) if n_pos > 0 else 1.0
    logging.info(f"Class distribution — positives: {int(n_pos)}, negatives: {int(n_neg)}, pos_weight: {pos_weight_val:.2f}")
    
    # Collect epoch histories for all models
    all_epoch_histories = {}
    
    logging.info("\n================ ABLATION STUDY: CLINICAL ALIGNMENT ================")
    
    logging.info("1. Training Base LSTM (Clinical Series + Demographics + ICD/DRG/Proc/Rx, NO Notes)...")
    model_base = LSTMLateFusionWithNotes(seq_dim=8, static_dims=ordered_static_dims, multihot_dims=multihot_dims, use_notes=False, num_layers=EXP_CONFIG['num_layers'], dropout=EXP_CONFIG['dropout'])
    model_base, hist_base = train_model(
        model_base, X_seq[train_idx], Y[train_idx], X_static[train_idx], X_mh[train_idx],
        X_seq_val=X_seq[val_idx], Y_val=Y[val_idx], X_static_val=X_static[val_idx], X_mh_val=X_mh[val_idx],
        epochs=EXP_CONFIG['epochs'], lr=EXP_CONFIG['lr'], batch_size=EXP_CONFIG['batch_size'],
        early_stop_patience=EXP_CONFIG['early_stop_patience'], pos_weight=pos_weight_val)
    all_epoch_histories['lstm_base'] = hist_base
    eval_base = evaluate_model(model_base, X_seq[test_idx], Y[test_idx], X_static[test_idx], X_mh[test_idx])
    _log_eval_result('LSTM Base', eval_base)
    torch.save(model_base.state_dict(), os.path.join(exp_dir, 'model_lstm_base.pt'))
    logging.info(f"Base LSTM model saved to {exp_dir}/model_lstm_base.pt")
    
    logging.info("2. Training Late Fusion LSTM (Base + LLM Notes Embedding)...")
    model_notes = LSTMLateFusionWithNotes(seq_dim=8, static_dims=ordered_static_dims, multihot_dims=multihot_dims, note_dim=EXP_CONFIG['note_dim'], use_notes=True, num_layers=EXP_CONFIG['num_layers'], dropout=EXP_CONFIG['dropout'])
    model_notes, hist_notes = train_model(
        model_notes, X_seq[train_idx], Y[train_idx], X_static[train_idx], X_mh[train_idx], X_note[train_idx],
        X_seq_val=X_seq[val_idx], Y_val=Y[val_idx], X_static_val=X_static[val_idx], X_mh_val=X_mh[val_idx], X_note_val=X_note[val_idx],
        epochs=EXP_CONFIG['epochs'], lr=EXP_CONFIG['lr'], batch_size=EXP_CONFIG['batch_size'],
        early_stop_patience=EXP_CONFIG['early_stop_patience'], pos_weight=pos_weight_val)
    all_epoch_histories['lstm_latefusion_notes'] = hist_notes
    eval_notes = evaluate_model(model_notes, X_seq[test_idx], Y[test_idx], X_static[test_idx], X_mh[test_idx], X_note[test_idx])
    _log_eval_result('LSTM LateFusion', eval_notes)
    torch.save(model_notes.state_dict(), os.path.join(exp_dir, 'model_lstm_notes.pt'))
    logging.info(f"Late Fusion LSTM model saved to {exp_dir}/model_lstm_notes.pt")
    
    logging.info("3. Training Early Fusion Transformer (Base + LLM Notes Embedding)...")
    model_tf_notes = TransformerEarlyFusionWithNotes(seq_dim=8, static_dims=ordered_static_dims, multihot_dims=multihot_dims, note_dim=EXP_CONFIG['note_dim'], use_notes=True, num_layers=EXP_CONFIG['tf_num_layers'], nhead=EXP_CONFIG['tf_nhead'], dropout=EXP_CONFIG['dropout'])
    model_tf_notes, hist_tf = train_model(
        model_tf_notes, X_seq[train_idx], Y[train_idx], X_static[train_idx], X_mh[train_idx], X_note[train_idx],
        X_seq_val=X_seq[val_idx], Y_val=Y[val_idx], X_static_val=X_static[val_idx], X_mh_val=X_mh[val_idx], X_note_val=X_note[val_idx],
        epochs=EXP_CONFIG['epochs'], lr=EXP_CONFIG['lr'], batch_size=EXP_CONFIG['batch_size'],
        early_stop_patience=EXP_CONFIG['early_stop_patience'], pos_weight=pos_weight_val)
    all_epoch_histories['transformer_earlyfusion_notes'] = hist_tf
    eval_tf = evaluate_model(model_tf_notes, X_seq[test_idx], Y[test_idx], X_static[test_idx], X_mh[test_idx], X_note[test_idx])
    _log_eval_result('Transformer EarlyFusion', eval_tf)
    torch.save(model_tf_notes.state_dict(), os.path.join(exp_dir, 'model_tf_notes.pt'))
    logging.info(f"Early Fusion Transformer model saved to {exp_dir}/model_tf_notes.pt")
    logging.info("4. Training Late Fusion Transformer (Base + LLM Notes Embedding)...")
    model_tf_late = TransformerLateFusionWithNotes(seq_dim=8, static_dims=ordered_static_dims, multihot_dims=multihot_dims, note_dim=EXP_CONFIG['note_dim'], use_notes=True, num_layers=EXP_CONFIG['tf_num_layers'], nhead=EXP_CONFIG['tf_nhead'], dropout=EXP_CONFIG['dropout'])
    model_tf_late, hist_tf_late = train_model(
        model_tf_late, X_seq[train_idx], Y[train_idx], X_static[train_idx], X_mh[train_idx], X_note[train_idx],
        X_seq_val=X_seq[val_idx], Y_val=Y[val_idx], X_static_val=X_static[val_idx], X_mh_val=X_mh[val_idx], X_note_val=X_note[val_idx],
        epochs=EXP_CONFIG['epochs'], lr=EXP_CONFIG['lr'], batch_size=EXP_CONFIG['batch_size'],
        early_stop_patience=EXP_CONFIG['early_stop_patience'], pos_weight=pos_weight_val)
    all_epoch_histories['transformer_latefusion_notes'] = hist_tf_late
    eval_tf_late = evaluate_model(model_tf_late, X_seq[test_idx], Y[test_idx], X_static[test_idx], X_mh[test_idx], X_note[test_idx])
    _log_eval_result('Transformer LateFusion', eval_tf_late)
    torch.save(model_tf_late.state_dict(), os.path.join(exp_dir, 'model_tf_late.pt'))

    logging.info("5. Training LSTM Base (w/o ICU Pressure)...")
    X_static_no_icu = X_static.copy()
    X_static_no_icu[:, 5:7] = 0.0  # Zero out icu_speedup_los and log_icu_los
    model_no_icu = LSTMLateFusionWithNotes(seq_dim=8, static_dims=ordered_static_dims, multihot_dims=multihot_dims, use_notes=False, num_layers=EXP_CONFIG['num_layers'], dropout=EXP_CONFIG['dropout'])
    model_no_icu, hist_no_icu = train_model(
        model_no_icu, X_seq[train_idx], Y[train_idx], X_static_no_icu[train_idx], X_mh[train_idx],
        X_seq_val=X_seq[val_idx], Y_val=Y[val_idx], X_static_val=X_static_no_icu[val_idx], X_mh_val=X_mh[val_idx],
        epochs=EXP_CONFIG['epochs'], lr=EXP_CONFIG['lr'], batch_size=EXP_CONFIG['batch_size'],
        early_stop_patience=EXP_CONFIG['early_stop_patience'], pos_weight=pos_weight_val)
    all_epoch_histories['lstm_no_icu'] = hist_no_icu
    eval_no_icu = evaluate_model(model_no_icu, X_seq[test_idx], Y[test_idx], X_static_no_icu[test_idx], X_mh[test_idx])
    _log_eval_result('LSTM Base (w/o ICU)', eval_no_icu)

    logging.info("6. Training LSTM Base (w/o Admission)...")
    X_static_no_adm = X_static.copy()
    X_static_no_adm[:, 3:5] = 0.0  # Zero out admission features
    model_no_adm = LSTMLateFusionWithNotes(seq_dim=8, static_dims=ordered_static_dims, multihot_dims=multihot_dims, use_notes=False, num_layers=EXP_CONFIG['num_layers'], dropout=EXP_CONFIG['dropout'])
    model_no_adm, hist_no_adm = train_model(
        model_no_adm, X_seq[train_idx], Y[train_idx], X_static_no_adm[train_idx], X_mh[train_idx],
        X_seq_val=X_seq[val_idx], Y_val=Y[val_idx], X_static_val=X_static_no_adm[val_idx], X_mh_val=X_mh[val_idx],
        epochs=EXP_CONFIG['epochs'], lr=EXP_CONFIG['lr'], batch_size=EXP_CONFIG['batch_size'],
        early_stop_patience=EXP_CONFIG['early_stop_patience'], pos_weight=pos_weight_val)
    all_epoch_histories['lstm_no_adm'] = hist_no_adm
    eval_no_adm = evaluate_model(model_no_adm, X_seq[test_idx], Y[test_idx], X_static_no_adm[test_idx], X_mh[test_idx])
    _log_eval_result('LSTM Base (w/o Adm)', eval_no_adm)

    logging.info("7. Training LSTM Base (w/o Codes)...")
    X_mh_zero = np.zeros_like(X_mh)
    model_no_codes = LSTMLateFusionWithNotes(seq_dim=8, static_dims=ordered_static_dims, multihot_dims=multihot_dims, use_notes=False, num_layers=EXP_CONFIG['num_layers'], dropout=EXP_CONFIG['dropout'])
    model_no_codes, hist_no_codes = train_model(
        model_no_codes, X_seq[train_idx], Y[train_idx], X_static[train_idx], X_mh_zero[train_idx],
        X_seq_val=X_seq[val_idx], Y_val=Y[val_idx], X_static_val=X_static[val_idx], X_mh_val=X_mh_zero[val_idx],
        epochs=EXP_CONFIG['epochs'], lr=EXP_CONFIG['lr'], batch_size=EXP_CONFIG['batch_size'],
        early_stop_patience=EXP_CONFIG['early_stop_patience'], pos_weight=pos_weight_val)
    all_epoch_histories['lstm_no_codes'] = hist_no_codes
    eval_no_codes = evaluate_model(model_no_codes, X_seq[test_idx], Y[test_idx], X_static[test_idx], X_mh_zero[test_idx])
    _log_eval_result('LSTM Base (w/o Codes)', eval_no_codes)

    logging.info("8. Training LSTM Base (Only Vitals)...")
    X_static_zero = np.zeros_like(X_static)
    model_only_vitals = LSTMLateFusionWithNotes(seq_dim=8, static_dims=ordered_static_dims, multihot_dims=multihot_dims, use_notes=False, num_layers=EXP_CONFIG['num_layers'], dropout=EXP_CONFIG['dropout'])
    model_only_vitals, hist_only_vitals = train_model(
        model_only_vitals, X_seq[train_idx], Y[train_idx], X_static_zero[train_idx], X_mh_zero[train_idx],
        X_seq_val=X_seq[val_idx], Y_val=Y[val_idx], X_static_val=X_static_zero[val_idx], X_mh_val=X_mh_zero[val_idx],
        epochs=EXP_CONFIG['epochs'], lr=EXP_CONFIG['lr'], batch_size=EXP_CONFIG['batch_size'],
        early_stop_patience=EXP_CONFIG['early_stop_patience'], pos_weight=pos_weight_val)
    all_epoch_histories['lstm_only_vitals'] = hist_only_vitals
    eval_only_vitals = evaluate_model(model_only_vitals, X_seq[test_idx], Y[test_idx], X_static_zero[test_idx], X_mh_zero[test_idx])
    _log_eval_result('LSTM Base (Only Vitals)', eval_only_vitals)

    logging.info("9. Training LSTM Base (Only Static & Codes)...")
    X_seq_zero = np.zeros_like(X_seq)
    model_only_static = LSTMLateFusionWithNotes(seq_dim=8, static_dims=ordered_static_dims, multihot_dims=multihot_dims, use_notes=False, num_layers=EXP_CONFIG['num_layers'], dropout=EXP_CONFIG['dropout'])
    model_only_static, hist_only_static = train_model(
        model_only_static, X_seq_zero[train_idx], Y[train_idx], X_static[train_idx], X_mh[train_idx],
        X_seq_val=X_seq_zero[val_idx], Y_val=Y[val_idx], X_static_val=X_static[val_idx], X_mh_val=X_mh[val_idx],
        epochs=EXP_CONFIG['epochs'], lr=EXP_CONFIG['lr'], batch_size=EXP_CONFIG['batch_size'],
        early_stop_patience=EXP_CONFIG['early_stop_patience'], pos_weight=pos_weight_val)
    all_epoch_histories['lstm_only_static'] = hist_only_static
    eval_only_static = evaluate_model(model_only_static, X_seq_zero[test_idx], Y[test_idx], X_static[test_idx], X_mh[test_idx])
    _log_eval_result('LSTM Base (Only Static & Codes)', eval_only_static)

    
    # XGBoost natively handles class imbalance via scale_pos_weight (equivalent to pos_weight)
    X_xgb_base = flatten_features(X_seq, X_static, X_mh)
    xgb_base = xgb.XGBClassifier(n_estimators=200, max_depth=6,
                                   scale_pos_weight=pos_weight_val,
                                   tree_method='hist', device='cuda')
    xgb_base.fit(X_xgb_base[train_idx], Y[train_idx])
    xgb_base_probs = xgb_base.predict_proba(X_xgb_base[test_idx])[:, 1]
    eval_xgb_base = _evaluate_xgb_probs(Y[test_idx], xgb_base_probs)
    _log_eval_result('XGBoost Base', eval_xgb_base)
    xgb_base.save_model(os.path.join(exp_dir, 'model_xgb_base.json'))
    logging.info(f"XGBoost Base model saved to {exp_dir}/model_xgb_base.json")
    
    X_xgb_notes = np.concatenate([X_xgb_base, X_note], axis=1)
    xgb_notes = xgb.XGBClassifier(n_estimators=200, max_depth=6,
                                    scale_pos_weight=pos_weight_val,
                                    tree_method='hist', device='cuda')
    xgb_notes.fit(X_xgb_notes[train_idx], Y[train_idx])
    xgb_notes_probs = xgb_notes.predict_proba(X_xgb_notes[test_idx])[:, 1]
    eval_xgb_notes = _evaluate_xgb_probs(Y[test_idx], xgb_notes_probs)
    _log_eval_result('XGBoost + Notes', eval_xgb_notes)
    xgb_notes.save_model(os.path.join(exp_dir, 'model_xgb_notes.json'))
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
    lgb_base = lgb.LGBMClassifier(n_estimators=EXP_CONFIG['lgb_n_est'], max_depth=EXP_CONFIG['lgb_depth'],
                                  scale_pos_weight=pos_weight_val,
                                  n_jobs=-1, verbose=-1)
    lgb_base.fit(X_xgb_base[train_idx], Y[train_idx], categorical_feature=cat_indices)
    lgb_base_probs = lgb_base.predict_proba(X_xgb_base[test_idx])[:, 1]
    eval_lgb_base = _evaluate_xgb_probs(Y[test_idx], lgb_base_probs)
    _log_eval_result('LightGBM Base', eval_lgb_base)
    lgb_base.booster_.save_model(os.path.join(exp_dir, 'model_lgb_base.txt'))
    logging.info(f"LightGBM Base model saved to {exp_dir}/model_lgb_base.txt")

    lgb_notes = lgb.LGBMClassifier(n_estimators=EXP_CONFIG['lgb_n_est'], max_depth=EXP_CONFIG['lgb_depth'],
                                   scale_pos_weight=pos_weight_val,
                                   n_jobs=-1, verbose=-1)
    lgb_notes.fit(X_xgb_notes[train_idx], Y[train_idx], categorical_feature=cat_indices)
    lgb_notes_probs = lgb_notes.predict_proba(X_xgb_notes[test_idx])[:, 1]
    eval_lgb_notes = _evaluate_xgb_probs(Y[test_idx], lgb_notes_probs)
    _log_eval_result('LightGBM + Notes', eval_lgb_notes)
    lgb_notes.booster_.save_model(os.path.join(exp_dir, 'model_lgb_notes.txt'))
    logging.info(f"LightGBM + LLM Notes model saved to {exp_dir}/model_lgb_notes.txt")

    # ── Collect all evaluation results ──────────────────────────────────────────
    all_eval_results = {
        'lstm_base': eval_base,
        'lstm_latefusion_notes': eval_notes,
        'transformer_earlyfusion_notes': eval_tf,
        'xgb_base': eval_xgb_base,
        'xgb_notes': eval_xgb_notes,
        'lgb_base': eval_lgb_base,
        'lgb_notes': eval_lgb_notes,
    }
    # Remove non-serializable y_prob before saving
    
    # ── Calculate DCA and Robustness ──────────────────────────────────────────
    logging.info("\n================ DECISION CURVE ANALYSIS (DCA) ================")
    dca_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    def calc_dca(probs, y=Y[test_idx]):
        n = len(y)
        res = []
        for pt in dca_thresholds:
            preds = (probs >= pt).astype(int)
            tp = np.sum((preds == 1) & (y == 1))
            fp = np.sum((preds == 1) & (y == 0))
            if pt == 1.0:
                nb = 0.0
            else:
                nb = (tp / n) - (fp / n) * (pt / (1 - pt))
            res.append(nb)
        return res
    
    dca_results = {
        "Thresholds": dca_thresholds,
        "XGBoost Base": calc_dca(xgb_base_probs),
        "XGBoost + LLM Notes": calc_dca(xgb_notes_probs),
        "LSTM Base": calc_dca(np.array(eval_base['y_prob'])),
        "LSTM LateFusion (+ Notes)": calc_dca(np.array(eval_notes['y_prob'])),
        "Transformer EarlyFusion (+ Notes)": calc_dca(np.array(eval_tf['y_prob']))
    }
    dca_path = os.path.join(exp_dir, 'dca_results.json')
    with open(dca_path, 'w') as f:
        json.dump(dca_results, f, indent=2)
    logging.info(f"DCA results saved to {dca_path}")

    logging.info("\n================ MISSING DATA ROBUSTNESS (LSTM LateFusion) ================")
    drop_rates = [0.1, 0.2, 0.3]
    robustness_results = []
    
    for dr in drop_rates:
        X_seq_test_tensor = torch.tensor(X_seq[test_idx], dtype=torch.float32)
        mask = (torch.rand(X_seq_test_tensor.shape) > dr).float()
        X_seq_dropped = (X_seq_test_tensor * mask).numpy()
        
        eval_drop = evaluate_model(model_notes, X_seq_dropped, Y[test_idx], X_static[test_idx], X_mh[test_idx], X_note[test_idx])
        robustness_results.append((dr, eval_drop))
        _log_eval_result(f'LSTM LateFusion (Drop {int(dr*100)}%)', eval_drop)

    eval_results_serializable = {}
    for k, v in all_eval_results.items():
        d = {kk: vv for kk, vv in v.items() if kk != 'y_prob'}
        eval_results_serializable[k] = d
    
    eval_path = os.path.join(exp_dir, 'evaluation_metrics.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results_serializable, f, indent=2, ensure_ascii=False)
    logging.info(f"Full evaluation metrics saved to {eval_path}")

    # ── Save epoch monitoring history to JSON ─────────────────────────────────
    epoch_monitor_path = os.path.join(exp_dir, 'epoch_monitoring.json')
    with open(epoch_monitor_path, 'w') as f:
        json.dump(all_epoch_histories, f, indent=2, ensure_ascii=False)
    logging.info(f"Epoch monitoring saved to {epoch_monitor_path}")

    # ── Helper to extract metrics for log / note ──────────────────────────────
    def _m(eval_result, key='metrics_at_best_t'):
        return eval_result[key]

    logging.info("\n================ FINAL ABLATION RESULTS (Test Set) ================")
    logging.info("---- @ Best F1 Threshold ----")
    for name, result in [
        ('XGBoost Base', eval_xgb_base), ('XGBoost + Notes', eval_xgb_notes),
        ('LightGBM Base', eval_lgb_base), ('LightGBM + Notes', eval_lgb_notes),
        ('LSTM Base', eval_base), ('LSTM LateFusion', eval_notes),
        ('Transformer EarlyFusion', eval_tf), ('Transformer LateFusion', eval_tf_late),
        ('LSTM Base (w/o ICU)', eval_no_icu), ('LSTM Base (w/o Adm)', eval_no_adm),
        ('LSTM Base (w/o Codes)', eval_no_codes), ('LSTM Base (Only Vitals)', eval_only_vitals),
        ('LSTM Base (Only Static & Codes)', eval_only_static)
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
        mb = _m(result, 'metrics_at_best_t')
        m5 = _m(result, 'metrics_at_0.5')
        return (
            f"| {name} | {mb['threshold']:.3f} | {mb['roc_auc']:.4f} | {mb['pr_auc']:.4f} | "
            f"{mb['precision']:.4f} | {mb['recall']:.4f} | {mb['f1']:.4f} | {mb['accuracy']:.4f} | {mb['brier']:.4f} |"
        )
    
    def _note_row_05(name, result):
        m5 = _m(result, 'metrics_at_0.5')
        return (
            f"| {name} | 0.500 | {m5['roc_auc']:.4f} | {m5['pr_auc']:.4f} | "
            f"{m5['precision']:.4f} | {m5['recall']:.4f} | {m5['f1']:.4f} | {m5['accuracy']:.4f} | {m5['brier']:.4f} |"
        )

    note_path = os.path.join(exp_dir, 'experiment_note.md')
    with open(note_path, 'w') as f:
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

### Note Embedding Impact (AUROC)
- XGBoost: notes Δ AUC = {_m(eval_xgb_notes)['roc_auc'] - _m(eval_xgb_base)['roc_auc']:+.4f}
- LightGBM: notes Δ AUC = {_m(eval_lgb_notes)['roc_auc'] - _m(eval_lgb_base)['roc_auc']:+.4f}
- LSTM Late Fusion:       notes Δ AUC = {_m(eval_notes)['roc_auc'] - _m(eval_base)['roc_auc']:+.4f}


## Clinical Utility (Decision Curve Analysis)
| Threshold | XGB Base | XGB + Notes | LSTM Base | LSTM + Notes | TF EarlyFusion |
|-----------|----------|-------------|-----------|--------------|----------------|
| 0.10 | {dca_results['XGBoost Base'][0]:.4f} | {dca_results['XGBoost + LLM Notes'][0]:.4f} | {dca_results['LSTM Base'][0]:.4f} | {dca_results['LSTM LateFusion (+ Notes)'][0]:.4f} | {dca_results['Transformer EarlyFusion (+ Notes)'][0]:.4f} |\n| 0.20 | {dca_results['XGBoost Base'][1]:.4f} | {dca_results['XGBoost + LLM Notes'][1]:.4f} | {dca_results['LSTM Base'][1]:.4f} | {dca_results['LSTM LateFusion (+ Notes)'][1]:.4f} | {dca_results['Transformer EarlyFusion (+ Notes)'][1]:.4f} |\n| 0.30 | {dca_results['XGBoost Base'][2]:.4f} | {dca_results['XGBoost + LLM Notes'][2]:.4f} | {dca_results['LSTM Base'][2]:.4f} | {dca_results['LSTM LateFusion (+ Notes)'][2]:.4f} | {dca_results['Transformer EarlyFusion (+ Notes)'][2]:.4f} |\n| 0.40 | {dca_results['XGBoost Base'][3]:.4f} | {dca_results['XGBoost + LLM Notes'][3]:.4f} | {dca_results['LSTM Base'][3]:.4f} | {dca_results['LSTM LateFusion (+ Notes)'][3]:.4f} | {dca_results['Transformer EarlyFusion (+ Notes)'][3]:.4f} |\n| 0.50 | {dca_results['XGBoost Base'][4]:.4f} | {dca_results['XGBoost + LLM Notes'][4]:.4f} | {dca_results['LSTM Base'][4]:.4f} | {dca_results['LSTM LateFusion (+ Notes)'][4]:.4f} | {dca_results['Transformer EarlyFusion (+ Notes)'][4]:.4f} |\n
## Sensor Dropout Robustness (LSTM LateFusion)
| Drop Rate | Threshold | AUROC | PRAUC | Precision | Recall | F1 |
|-----------|-----------|-------|-------|-----------|--------|----|
| 10% | {_m(robustness_results[0][1])['threshold']:.3f} | {_m(robustness_results[0][1])['roc_auc']:.4f} | {_m(robustness_results[0][1])['pr_auc']:.4f} | {_m(robustness_results[0][1])['precision']:.4f} | {_m(robustness_results[0][1])['recall']:.4f} | {_m(robustness_results[0][1])['f1']:.4f} |\n| 20% | {_m(robustness_results[1][1])['threshold']:.3f} | {_m(robustness_results[1][1])['roc_auc']:.4f} | {_m(robustness_results[1][1])['pr_auc']:.4f} | {_m(robustness_results[1][1])['precision']:.4f} | {_m(robustness_results[1][1])['recall']:.4f} | {_m(robustness_results[1][1])['f1']:.4f} |\n| 30% | {_m(robustness_results[2][1])['threshold']:.3f} | {_m(robustness_results[2][1])['roc_auc']:.4f} | {_m(robustness_results[2][1])['pr_auc']:.4f} | {_m(robustness_results[2][1])['precision']:.4f} | {_m(robustness_results[2][1])['recall']:.4f} | {_m(robustness_results[2][1])['f1']:.4f} |\n\n## Saved Files
| File | Description |
|------|-------------|
| `model_lstm_base.pt`   | Base LSTM state_dict |
| `model_lstm_notes.pt`  | Late Fusion LSTM state_dict |
| `model_tf_notes.pt`    | Early Fusion Transformer state_dict |
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

if __name__ == "__main__":
    main()
