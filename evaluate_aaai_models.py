#!/usr/bin/env python3
"""
Unified benchmarking script to generate Table 2 for AAAI.
Supports 30-Day Readmission Prediction and In-Hospital Mortality Prediction.
"""

import os
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import duckdb
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, precision_recall_curve
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings

# Suppress sklearn/imblearn warnings
warnings.filterwarnings('ignore')

from train_readmission_mimic3_with_notes import (
    enrich_stays_with_features, 
    compute_icu_load_features, 
    build_multihot_features, 
    LSTMLateFusionWithNotes,
    flatten_features
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def fetch_mimic3_data_aaai(embeddings_dict, task='readmission', note_emb_dim=768):
    """
    Fetch MIMIC-III data and compute labels based on the specified task.
    task: 'readmission' or 'mortality'
    """
    logging.info(f"Connecting to DuckDB and loading MIMIC-III features for task: {task}...")
    con = duckdb.connect()
    
    stay_ids = list(embeddings_dict.keys())
    if not stay_ids: return None, None, None, None, None, None, None
        
    logging.info("Loading Demographics...")
    stays_df = con.query(f"""
        SELECT 
            s.SUBJECT_ID, s.HADM_ID, s.ICUSTAY_ID as stay_id, s.INTIME, s.OUTTIME, s.FIRST_CAREUNIT, s.LOS,
            a.ETHNICITY, a.MARITAL_STATUS, a.RELIGION, a.INSURANCE, a.ADMISSION_TYPE, a.ADMISSION_LOCATION,
            a.EDREGTIME, a.EDOUTTIME, a.HOSPITAL_EXPIRE_FLAG,
            p.GENDER, p.DOB
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/ICUSTAYS.csv', sample_size=-1) s
        JOIN read_csv_auto('/home/hanwen/data/mimic/iii/ADMISSIONS.csv', sample_size=-1) a ON s.HADM_ID = a.HADM_ID
        JOIN read_csv_auto('/home/hanwen/data/mimic/iii/PATIENTS.csv', sample_size=-1) p ON s.SUBJECT_ID = p.SUBJECT_ID
        WHERE s.ICUSTAY_ID IN {tuple(stay_ids)}
    """).df()
    
    stays_df['INTIME']  = pd.to_datetime(stays_df['INTIME'])
    stays_df['OUTTIME'] = pd.to_datetime(stays_df['OUTTIME'])
    stays_df = stays_df.sort_values(['SUBJECT_ID', 'INTIME']).reset_index(drop=True)

    labels = []
    if task == 'readmission':
        logging.info("Computing ICU readmission labels (30-day)...")
        for i, row in stays_df.iterrows():
            cond_a = stays_df[
                (stays_df['SUBJECT_ID'] == row['SUBJECT_ID']) &
                (stays_df['HADM_ID']    != row['HADM_ID']) &
                (stays_df['INTIME']      > row['OUTTIME']) &
                (stays_df['INTIME']     <= row['OUTTIME'] + pd.Timedelta(days=30))
            ]
            cond_b = stays_df[
                (stays_df['HADM_ID'] == row['HADM_ID']) &
                (stays_df['INTIME']   > row['OUTTIME'])
            ]
            flag = 1 if (len(cond_a) > 0 or len(cond_b) > 0) else 0
            labels.append(flag)
    elif task == 'mortality':
        logging.info("Computing In-Hospital Mortality labels...")
        labels = stays_df['HOSPITAL_EXPIRE_FLAG'].fillna(0).astype(int).tolist()
    else:
        raise ValueError("Invalid task")
        
    stays_df['label'] = labels
    pos_rate = stays_df['label'].mean()
    logging.info(f"Task: {task} | Positive rate: {pos_rate:.1%} ({sum(labels)} / {len(stays_df)})")

    stays_df['DOB'] = pd.to_datetime(stays_df['DOB'], errors='coerce')
    stays_df['age'] = (stays_df['INTIME'] - stays_df['DOB']).dt.days / 365.25
    stays_df['age'] = stays_df['age'].clip(0, 100)
    
    stays_df = enrich_stays_with_features(stays_df, con)
    stays_df = compute_icu_load_features(stays_df)
    
    cont_cols = ['age', 'pre_icu_transfers', 'surg_count', 'log_surg_gap', 'log_ed_wait', 'icu_speedup_los', 'log_icu_los', 'height', 'weight']
    cat_cols = ['GENDER', 'MARITAL_STATUS', 'RELIGION', 'ETHNICITY', 'INSURANCE', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'FIRST_CAREUNIT', 'surg_flag']
    
    static_encoders, static_dims = {}, {}
    for col in cont_cols:
        static_dims[col] = 0
    for col in cat_cols:
        stays_df[col] = stays_df[col].fillna('UNKNOWN').astype(str)
        le = LabelEncoder()
        stays_df[col] = le.fit_transform(stays_df[col])
        static_encoders[col] = le
        static_dims[col] = len(le.classes_)

    hadm_ids_tuple = tuple(stays_df['HADM_ID'].unique().tolist())

    logging.info("Loading Sparse Features...")
    icd_dict, icd_dim = build_multihot_features(con, 'DIAGNOSES_ICD', 'HADM_ID', 'ICD9_CODE', hadm_ids_tuple, top_k=64, trim=3)
    drg_dict, drg_dim = build_multihot_features(con, 'DRGCODES', 'HADM_ID', 'DRG_CODE', hadm_ids_tuple, top_k=64)
    proc_dict, proc_dim = build_multihot_features(con, 'PROCEDURES_ICD', 'HADM_ID', 'ICD9_CODE', hadm_ids_tuple, top_k=64, trim=3)
    rx_dict, rx_dim = build_multihot_features(con, 'PRESCRIPTIONS', 'HADM_ID', 'DRUG', hadm_ids_tuple, top_k=64)
    
    multihot_dims = {'icd': icd_dim, 'drg': drg_dim, 'proc': proc_dim, 'rx': rx_dim}

    item_map = {
        211: 0, 220045: 0, 618: 1, 220210: 1, 646: 2, 220277: 2, 51: 3, 220050: 3,
        8368: 4, 220051: 4, 52: 5, 220052: 5, 225312: 5, 678: 6, 223761: 6, 676: 6, 223762: 6,
        807: 7, 811: 7, 1529: 7, 225664: 7, 220621: 7,
        184: 8, 220739: 8,                # GCS Eye Opening
        454: 9, 223901: 9,                # GCS Motor Response
        723: 10, 223900: 10,              # GCS Verbal Response
        198: 11,                          # GCS Total
        780: 12, 1126: 12, 220274: 12,    # pH
        3420: 13, 190: 13, 223835: 13,    # FiO2
        3348: 14, 115: 14, 224308: 14, 8377: 14 # Capillary Refill Rate
    }
    
    events_df = con.query(f"""
        SELECT ICUSTAY_ID as stay_id, CHARTTIME, ITEMID, VALUENUM
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/CHARTEVENTS.csv', sample_size=-1)
        WHERE ICUSTAY_ID IN {tuple(stay_ids)} AND ITEMID IN {tuple(item_map.keys())} AND VALUENUM IS NOT NULL
    """).df()
    
    logging.info("Formatting dataset...")
    X_seq, X_static, X_mh, X_note, Y = [], [], [], [], []
    dropped_count = 0
    
    for _, stay in stays_df.iterrows():
        sid = stay['stay_id']
        hadm = stay['HADM_ID']
        
        evs = events_df[events_df['stay_id'] == sid].copy()
        if evs.empty:
            dropped_count += 1
            continue
            
        evs['CHARTTIME'] = pd.to_datetime(evs['CHARTTIME'])
        evs['hour'] = ((evs['CHARTTIME'] - stay['INTIME']).dt.total_seconds() / 3600).astype(int)
        evs = evs[(evs['hour'] >= 0) & (evs['hour'] < 48)]
        
        observed_channels = evs['ITEMID'].map(item_map).nunique()
        if observed_channels < 3:
            dropped_count += 1
            continue

        seq = np.full((48, 15), np.nan, dtype=np.float32)
        for _, e in evs.iterrows():
            seq[int(e['hour']), item_map[e['ITEMID']]] = e['VALUENUM']

        df_seq = pd.DataFrame(seq).ffill().fillna(0.0)
        X_seq.append(df_seq.values)
        
        X_static.append([stay[col] for col in cont_cols + cat_cols])
        
        mh_vecs = [
            icd_dict.get(hadm, np.zeros(icd_dim, dtype=np.float32)),
            drg_dict.get(hadm, np.zeros(drg_dim, dtype=np.float32)),
            proc_dict.get(hadm, np.zeros(proc_dim, dtype=np.float32)),
            rx_dict.get(hadm, np.zeros(rx_dim, dtype=np.float32))
        ]
        X_mh.append(np.concatenate(mh_vecs))
        
        val = embeddings_dict.get(sid)
        if val is None:
            val = np.zeros(note_emb_dim, dtype=np.float32)
        elif isinstance(val, dict):
            val = val.get('embedding', np.zeros(note_emb_dim, dtype=np.float32))
        X_note.append(np.array(val, dtype=np.float32))
        
        Y.append(stay['label'])
        
    logging.info(f"Dropped {dropped_count} stays due to extreme missingness (< 3 observed signals)")
    return np.array(X_seq), np.array(X_static, dtype=np.float32), np.array(X_mh, dtype=np.float32), np.array(X_note), np.array(Y), static_dims, multihot_dims

def compute_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec1 = precision_score(y_true, y_pred, zero_division=0)
    rec1 = recall_score(y_true, y_pred, zero_division=0)
    prec0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    auroc = roc_auc_score(y_true, y_score)
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    # Compatibility with older numpy via trapz instead of trapezoid
    auprc = float(np.trapz(precisions, recalls))
    
    return [acc, prec0, prec1, rec0, rec1, macro_f1, auroc, auprc]

def train_pytorch_lstm(X_seq_train, X_static_train, X_mh_train, X_note_train, Y_train,
                       X_seq_test, X_static_test, X_mh_test, X_note_test, Y_test,
                       static_dims, multihot_dims, use_notes=False, epochs=10):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMLateFusionWithNotes(
        seq_dim=X_seq_train.shape[-1], static_dims=static_dims, multihot_dims=multihot_dims,
        hidden_dim=128, note_dim=768 if use_notes else 0, use_notes=use_notes, num_layers=2
    ).to(device)
    
    pos_weight = torch.tensor([(len(Y_train) - Y_train.sum()) / max(1, Y_train.sum())]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 256
    
    def to_tensor(arr): return torch.tensor(arr, dtype=torch.float32).to(device)
    
    model.train()
    for ep in range(epochs):
        indices = np.random.permutation(len(Y_train))
        for i in range(0, len(Y_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            b_seq = to_tensor(X_seq_train[batch_idx])
            b_stat = to_tensor(X_static_train[batch_idx])
            b_mh = to_tensor(X_mh_train[batch_idx])
            b_note = to_tensor(X_note_train[batch_idx]) if use_notes else None
            b_y = to_tensor(Y_train[batch_idx])
            
            optimizer.zero_grad()
            logits = model(b_seq, b_stat, b_mh, b_note)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        b_seq = to_tensor(X_seq_test)
        b_stat = to_tensor(X_static_test)
        b_mh = to_tensor(X_mh_test)
        b_note = to_tensor(X_note_test) if use_notes else None
        logits = model(b_seq, b_stat, b_mh, b_note)
        probs = torch.sigmoid(logits).cpu().numpy()
        
    return probs

def main(task='readmission', sample_size=None):
    emb_path = 'output/mimic3_note_summaries_clinicalbert.pkl'
    if not os.path.exists(emb_path):
        logging.error(f"Need ClinicalBERT embeddings at {emb_path}")
        return
        
    with open(emb_path, 'rb') as f:
        emb_dict = pickle.load(f)
        
    if sample_size is not None:
        keys = list(emb_dict.keys())[:sample_size]
        emb_dict = {k: emb_dict[k] for k in keys}
        
    X_seq, X_static, X_mh, X_note, Y, static_dims, multihot_dims = fetch_mimic3_data_aaai(emb_dict, task=task)
    if X_seq is None: return
    
    np.random.seed(42)
    torch.manual_seed(42)
    indices = np.random.permutation(len(Y))
    
    # 80/20 train/test split for simple benchmarking
    train_size = int(0.8 * len(Y))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    def split(arr): return arr[train_idx], arr[test_idx]
    X_seq_tr, X_seq_te = split(X_seq)
    X_static_tr, X_static_te = split(X_static)
    X_mh_tr, X_mh_te = split(X_mh)
    X_note_tr, X_note_te = split(X_note)
    Y_tr, Y_te = split(Y)
    
    # Standard scale seq features
    scaler = StandardScaler()
    X_seq_tr_flat = X_seq_tr.reshape(X_seq_tr.shape[0], -1)
    X_seq_te_flat = X_seq_te.reshape(X_seq_te.shape[0], -1)
    scaler.fit(X_seq_tr_flat)
    X_seq_tr = scaler.transform(X_seq_tr_flat).reshape(X_seq_tr.shape)
    X_seq_te = scaler.transform(X_seq_te_flat).reshape(X_seq_te.shape)
    
    # Flatten features for ML models
    X_tr_flat = flatten_features(X_seq_tr, X_static_tr, X_mh_tr)
    X_te_flat = flatten_features(X_seq_te, X_static_te, X_mh_te)
    
    X_tr_flat_multi = np.concatenate([X_tr_flat, X_note_tr], axis=1)
    X_te_flat_multi = np.concatenate([X_te_flat, X_note_te], axis=1)
    
    pos_weight = float((len(Y_tr) - Y_tr.sum()) / max(1, Y_tr.sum()))
    
    models = {
        'LogReg (C=1.0)': LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000),
        'MLP': MLPClassifier(hidden_layer_sizes=(128,), max_iter=100),
        'BalancedRF': BalancedRandomForestClassifier(n_estimators=100, random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, scale_pos_weight=pos_weight, random_state=42)
    }
    
    results = {}
    
    logging.info("Training Base Models (Structured Data Only)...")
    for name, model in models.items():
        logging.info(f"Training {name} (Base)...")
        model.fit(X_tr_flat, Y_tr)
        probs = model.predict_proba(X_te_flat)[:, 1]
        results[name] = compute_metrics(Y_te, probs)
        
    logging.info("Training LSTM (Base)...")
    probs_lstm = train_pytorch_lstm(X_seq_tr, X_static_tr, X_mh_tr, X_note_tr, Y_tr,
                                    X_seq_te, X_static_te, X_mh_te, X_note_te, Y_te,
                                    static_dims, multihot_dims, use_notes=False)
    results['LSTM'] = compute_metrics(Y_te, probs_lstm)
    
    logging.info("Training KAMELEON Multimodal Models (Structured + Notes)...")
    for name, model in models.items():
        logging.info(f"Training KAMELEON-{name}...")
        model.fit(X_tr_flat_multi, Y_tr)
        probs = model.predict_proba(X_te_flat_multi)[:, 1]
        results[f'KAMELEON-{name.split(" ")[0]}'] = compute_metrics(Y_te, probs)
        
    logging.info("Training KAMELEON-LSTM...")
    probs_lstm_multi = train_pytorch_lstm(X_seq_tr, X_static_tr, X_mh_tr, X_note_tr, Y_tr,
                                    X_seq_te, X_static_te, X_mh_te, X_note_te, Y_te,
                                    static_dims, multihot_dims, use_notes=True)
    results['KAMELEON-LSTM'] = compute_metrics(Y_te, probs_lstm_multi)
    
    # Print Markdown Table
    cols = ['Acc', 'Prec0', 'Prec1', 'Rec0', 'Rec1', 'Macro F1', 'AUROC', 'AUPRC']
    print(f"\n\n### Task: {task.title()} Prediction")
    print("| Model | $D^{struct}$ | $D^{unstruct}$ | " + " | ".join(cols) + " |")
    print("|" + "-"*20 + "|" + "-"*12 + "|" + "-"*14 + "|" + "|".join(["-"*8]*len(cols)) + "|")
    
    ordered_keys = ['LogReg (C=1.0)', 'MLP', 'BalancedRF', 'LSTM', 'LightGBM', 'XGBoost',
                    'KAMELEON-LogReg', 'KAMELEON-MLP', 'KAMELEON-BalancedRF', 'KAMELEON-LSTM', 'KAMELEON-LightGBM', 'KAMELEON-XGBoost']
    
    for k in ordered_keys:
        if k not in results: continue
        vals = results[k]
        is_multi = k.startswith('KAMELEON')
        ds = "✓" if not is_multi else "✓"
        du = "–" if not is_multi else "✓"
        val_str = " | ".join([f"{v:.3f}" for v in vals])
        print(f"| {k:<18} | {ds:^10} | {du:^12} | {val_str} |")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="readmission", choices=['readmission', 'mortality'])
    parser.add_argument("--sample", type=int, default=None, help="Use a subset of data for quick testing")
    args = parser.parse_args()
    main(task=args.task, sample_size=args.sample)
