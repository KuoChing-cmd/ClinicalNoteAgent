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

# ---------------------------------------------------------
# Data Processing Pipeline (MIMIC-III DuckDB)
# ---------------------------------------------------------
def build_multihot_features(con, table, id_col, val_col, valid_ids, top_k, trim=None):
    """Generic function to build Top-K multi-hot encoding using DuckDB"""
    query_trim = f"SUBSTRING(CAST({val_col} AS VARCHAR), 1, {trim})" if trim else f"CAST({val_col} AS VARCHAR)"
    
    vocab_df = con.query(f"""
        SELECT {query_trim} as code, count(*) as cnt
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/{table}.csv', sample_size=-1)
        WHERE {id_col} IN {valid_ids} AND {val_col} IS NOT NULL
        GROUP BY code ORDER BY cnt DESC LIMIT {top_k}
    """).df()
    vocab = vocab_df['code'].tolist()
    
    raw_df = con.query(f"""
        SELECT {id_col} as target_id, {query_trim} as code
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/{table}.csv', sample_size=-1)
        WHERE {id_col} IN {valid_ids} AND {query_trim} IN {tuple(vocab) if len(vocab)>1 else f"('{vocab[0]}')"}
    """).df()
    
    feature_dict = {}
    for tid, group in raw_df.groupby('target_id'):
        vec = np.zeros(len(vocab), dtype=np.float32)
        for code in group['code']:
            if code in vocab:
                vec[vocab.index(code)] = 1.0
        feature_dict[tid] = vec
        
    return feature_dict, len(vocab)

def enrich_stays_with_features(stays_df, con):
    logging.info("Computing additional features for stays...")
    
    # Register temporary table for DuckDB
    con.register('stays_df_tmp', stays_df[['HADM_ID', 'stay_id', 'INTIME']])
    
    transfers_df = con.query("""
        SELECT t.HADM_ID, count(*) as pre_icu_transfers
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/TRANSFERS.csv', sample_size=-1) t
        JOIN stays_df_tmp s ON t.HADM_ID = s.HADM_ID
        WHERE CAST(t.OUTTIME AS TIMESTAMP) <= CAST(s.INTIME AS TIMESTAMP)
        GROUP BY t.HADM_ID
    """).df()
    
    surg_df = con.query("""
        SELECT 
            s.HADM_ID, 
            count(p.ITEMID) as surg_count,
            max(CAST(p.ENDTIME AS TIMESTAMP)) as last_surg_endtime
        FROM stays_df_tmp s
        JOIN read_csv_auto('/home/hanwen/data/mimic/iii/PROCEDUREEVENTS_MV.csv', sample_size=-1) p 
            ON s.HADM_ID = p.HADM_ID AND CAST(p.ENDTIME AS TIMESTAMP) <= CAST(s.INTIME AS TIMESTAMP)
        GROUP BY s.HADM_ID
    """).df()
    
    df = stays_df.copy()
    
    df = df.merge(transfers_df, on='HADM_ID', how='left')
    df['pre_icu_transfers'] = df['pre_icu_transfers'].fillna(0)
    
    df = df.merge(surg_df, on='HADM_ID', how='left')
    df['surg_count'] = df['surg_count'].fillna(0)
    df['surg_flag'] = (df['surg_count'] > 0).astype(int)
    
    df['last_surg_endtime'] = pd.to_datetime(df['last_surg_endtime'])
    df['INTIME'] = pd.to_datetime(df['INTIME'])
    gap_hours = (df['INTIME'] - df['last_surg_endtime']).dt.total_seconds() / 3600.0
    df['surg_gap'] = np.where(gap_hours > 0, gap_hours, 0)
    df['log_surg_gap'] = np.log1p(df['surg_gap'])
    
    df['EDREGTIME'] = pd.to_datetime(df['EDREGTIME'])
    df['EDOUTTIME'] = pd.to_datetime(df['EDOUTTIME'])
    ed_wait = (df['EDOUTTIME'] - df['EDREGTIME']).dt.total_seconds() / 3600.0
    df['log_ed_wait'] = np.log1p(np.where((ed_wait > 0) & (ed_wait < 240), ed_wait, 0))
    
    con.unregister('stays_df_tmp')
    
    return df


def compute_icu_load_features(stays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ICU stay-duration features using the pre-existing LOS column
    (Length of Stay in days) from ICUSTAYS.

    Note on MIMIC-III time-shifting:
      MIMIC-III shifts each patient's timestamps by an independent random offset,
      so comparing absolute timestamps *across* patients is meaningless.  We
      therefore do NOT compute any cross-patient concurrency metric (load_index).
      Instead, we derive two within-patient / population-mean features that are
      robust to the per-patient time shift:

      - log_icu_los     : log1p(LOS in hours) — captures stay duration on a
                          log scale, robust to outliers.
      - icu_speedup_los : unit_mean_LOS - actual_LOS — deviation from the
                          *same ICU unit*’s average LOS (grouped by FIRST_CAREUNIT).
                          Positive = shorter than peers in the same unit type,
                          suggesting potential capacity-driven early discharge.
                          Using unit-level mean avoids confounding from the large
                          baseline LOS differences across ICU types (e.g. MICU vs
                          CSRU vs TSICU).
    """
    logging.info("Computing ICU LOS features (unit-grouped speedup via FIRST_CAREUNIT)...")
    df = stays_df.copy()

    # LOS in ICUSTAYS is in days; convert to hours
    los_hours = df['LOS'].fillna(0.0).clip(lower=0.0) * 24.0
    df['_los_h'] = los_hours

    # Per-unit mean LOS (group by FIRST_CAREUNIT) — vectorised, no loop needed
    unit_mean_los = df.groupby('FIRST_CAREUNIT')['_los_h'].transform('mean')

    df['log_icu_los']     = np.log1p(los_hours).astype(np.float32)
    df['icu_speedup_los'] = (unit_mean_los - los_hours).astype(np.float32)  # positive = shorter than unit peers
    df.drop(columns=['_los_h'], inplace=True)

    # Log per-unit stats for sanity check
    unit_stats = stays_df.copy()
    unit_stats['_los_h'] = los_hours
    per_unit = unit_stats.groupby('FIRST_CAREUNIT')['_los_h'].agg(['mean', 'count'])
    stats_str = ', '.join(
        f"{u}: {row['mean']:.1f}h(n={int(row['count'])})"
        for u, row in per_unit.iterrows()
    )
    logging.info(f"ICU LOS by unit — {stats_str}")
    logging.info(
        f"icu_speedup_los (unit-adjusted): range=[{df['icu_speedup_los'].min():.1f}, "
        f"{df['icu_speedup_los'].max():.1f}]h, mean={df['icu_speedup_los'].mean():.2f}h"
    )
    return df


def load_note_texts(note_summary_dict):
    """Build a stay_id -> note text mapping from the Llama summary cache."""
    note_texts = {}
    if not note_summary_dict:
        return note_texts

    for stay_id, payload in note_summary_dict.items():
        parts = []
        try:
            sid = int(stay_id)
        except (TypeError, ValueError):
            sid = stay_id

        if isinstance(payload, dict):
            meta = payload.get('meta_summary', {}) if isinstance(payload.get('meta_summary', {}), dict) else {}
            if isinstance(meta, dict):
                final_summary = meta.get('final_summary')
                if final_summary:
                    parts.append(str(final_summary).strip())
                risk_score = meta.get('readmission_risk_score')
                if risk_score not in (None, ''):
                    parts.append(f"Risk score: {risk_score}")
                critical_factors = meta.get('critical_factors_list', [])
                if isinstance(critical_factors, list) and critical_factors:
                    parts.append("Critical factors: " + "; ".join(str(x) for x in critical_factors if str(x).strip()))
                elif critical_factors:
                    parts.append(f"Critical factors: {critical_factors}")

            category_summaries = payload.get('category_summaries', {})
            if isinstance(category_summaries, dict):
                for category, summary in category_summaries.items():
                    if isinstance(summary, str) and summary.strip():
                        parts.append(f"{category}: {summary.strip()}")
                    elif isinstance(summary, dict):
                        text = summary.get('summary') or summary.get('content') or summary.get('text')
                        if text:
                            parts.append(f"{category}: {str(text).strip()}")
            raw_summary = payload.get('summary')
            if raw_summary and str(raw_summary).strip():
                parts.append(str(raw_summary).strip())
        elif isinstance(payload, str):
            parts.append(payload.strip())

        combined = " ".join(part for part in parts if str(part).strip())
        note_texts[sid] = combined.strip()

    return note_texts


def fetch_mimic3_data(embeddings_dict, note_emb_dim=768):
    logging.info("Connecting to DuckDB and loading MIMIC-III features...")
    con = duckdb.connect()
    
    stay_ids = list(embeddings_dict.keys())
    if not stay_ids:
        return None, None, None, None, None, None, None, None
        
    # 1. Stays and Demographics
    logging.info("Loading Demographics...")
    stays_df = con.query(f"""
        SELECT 
            s.SUBJECT_ID, s.HADM_ID, s.ICUSTAY_ID as stay_id, s.INTIME, s.OUTTIME, s.FIRST_CAREUNIT, s.LOS,
            a.ETHNICITY, a.MARITAL_STATUS, a.INSURANCE, a.ADMISSION_TYPE, a.ADMISSION_LOCATION,
            a.EDREGTIME, a.EDOUTTIME,
            p.GENDER, p.DOB
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/ICUSTAYS.csv', sample_size=-1) s
        JOIN read_csv_auto('/home/hanwen/data/mimic/iii/ADMISSIONS.csv', sample_size=-1) a ON s.HADM_ID = a.HADM_ID
        JOIN read_csv_auto('/home/hanwen/data/mimic/iii/PATIENTS.csv', sample_size=-1) p ON s.SUBJECT_ID = p.SUBJECT_ID
        WHERE s.ICUSTAY_ID IN {tuple(stay_ids)}
    """).df()
    
    # Parse timestamps first, then compute ICU readmission labels.
    # Two conditions (OR logic) define a positive label:
    #   A. 跨次住院（Cross-admission）：同一 SUBJECT_ID 在本次出院后 30 天内
    #      有另一次 ICU 入院（不同 HADM_ID）。
    #   B. 院内再入ICU（Within-admission）：同一 HADM_ID 内存在另一次
    #      INTIME > 本次 OUTTIME 的 ICU 住院。
    stays_df['INTIME']  = pd.to_datetime(stays_df['INTIME'])
    stays_df['OUTTIME'] = pd.to_datetime(stays_df['OUTTIME'])
    stays_df = stays_df.sort_values(['SUBJECT_ID', 'INTIME']).reset_index(drop=True)

    logging.info("Computing ICU readmission labels (30-day cross-admission OR within-admission)...")
    readmitted_flags = []
    source_a_count = 0
    source_b_count = 0
    for i, row in stays_df.iterrows():
        # Condition A: 30天内跨次住院 ICU 再入院
        cond_a = stays_df[
            (stays_df['SUBJECT_ID'] == row['SUBJECT_ID']) &
            (stays_df['HADM_ID']    != row['HADM_ID']) &
            (stays_df['INTIME']      > row['OUTTIME']) &
            (stays_df['INTIME']     <= row['OUTTIME'] + pd.Timedelta(days=30))
        ]
        # Condition B: 同一次住院（相同 HADM_ID）内的 ICU 再入院
        cond_b = stays_df[
            (stays_df['HADM_ID'] == row['HADM_ID']) &
            (stays_df['INTIME']   > row['OUTTIME'])
        ]
        flag = 1 if (len(cond_a) > 0 or len(cond_b) > 0) else 0
        readmitted_flags.append(flag)
        if flag:
            if len(cond_a) > 0: source_a_count += 1
            if len(cond_b) > 0: source_b_count += 1
    stays_df['readmitted'] = readmitted_flags
    pos_rate = stays_df['readmitted'].mean()
    logging.info(
        f"ICU readmission rate: {pos_rate:.1%} "
        f"({stays_df['readmitted'].sum()} / {len(stays_df)} stays) | "
        f"Cross-admission(A): {source_a_count}, Within-admission(B): {source_b_count}"
    )

    stays_df['DOB'] = pd.to_datetime(stays_df['DOB'], errors='coerce')
    stays_df['age'] = (stays_df['INTIME'] - stays_df['DOB']).dt.days / 365.25
    stays_df['age'] = stays_df['age'].clip(0, 100)
    
    # Extract original features instead of fitting OLS
    stays_df = enrich_stays_with_features(stays_df, con)
    
    # Add ICU LOS pressure features (uses ICUSTAYS.LOS column, no cross-patient timestamp comparison)
    stays_df = compute_icu_load_features(stays_df)

    hadm_ids_tuple = tuple(stays_df['HADM_ID'].unique().tolist())

    # Load primary DRG per admission as a single categorical feature for nn.Embedding
    logging.info("Loading primary DRG code per admission...")
    drg_df = con.query(f"""
        SELECT HADM_ID, CAST(DRG_CODE AS VARCHAR) as drg_code
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/DRGCODES.csv', sample_size=-1)
        WHERE HADM_ID IN {hadm_ids_tuple} AND DRG_CODE IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY HADM_ID 
            ORDER BY CASE WHEN DRG_TYPE = 'HCFA' THEN 1 WHEN DRG_TYPE = 'MS' THEN 2 ELSE 3 END, ROW_ID
        ) = 1
    """).df()
    stays_df = stays_df.merge(drg_df, on='HADM_ID', how='left')
    stays_df['drg_code'] = stays_df['drg_code'].fillna('UNKNOWN').astype(str)
    top_drg_set = set(
        stays_df[stays_df['drg_code'] != 'UNKNOWN']['drg_code']
        .value_counts()
        .nlargest(128)
        .index
    )
    stays_df['drg_code'] = stays_df['drg_code'].apply(
        lambda x: x if (x == 'UNKNOWN' or x in top_drg_set) else 'OTHER'
    )

    # Option B: Extract Primary Diagnosis (SEQ_NUM=1) for single categorical nn.Embedding
    logging.info("Loading primary diagnosis ICD (SEQ_NUM=1) per admission...")
    primary_icd_df = con.query(f"""
        SELECT HADM_ID, SUBSTRING(CAST(ICD9_CODE AS VARCHAR), 1, 3) as primary_icd
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/DIAGNOSES_ICD.csv', sample_size=-1)
        WHERE HADM_ID IN {hadm_ids_tuple} AND SEQ_NUM = 1 AND ICD9_CODE IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (PARTITION BY HADM_ID ORDER BY ROW_ID) = 1
    """).df()
    stays_df = stays_df.merge(primary_icd_df, on='HADM_ID', how='left')
    stays_df['primary_icd'] = stays_df['primary_icd'].fillna('UNKNOWN').astype(str)
    top_icd_set = set(
        stays_df[stays_df['primary_icd'] != 'UNKNOWN']['primary_icd']
        .value_counts()
        .nlargest(128)
        .index
    )
    stays_df['primary_icd'] = stays_df['primary_icd'].apply(
        lambda x: x if (x == 'UNKNOWN' or x in top_icd_set) else 'OTHER'
    )
    
    cont_cols = ['age', 'pre_icu_transfers', 'surg_count', 'log_surg_gap', 'log_ed_wait',
                 'icu_speedup_los', 'log_icu_los']
    cat_cols = ['GENDER', 'MARITAL_STATUS', 'ETHNICITY', 'INSURANCE', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'FIRST_CAREUNIT', 'surg_flag', 'drg_code', 'primary_icd']
    
    static_encoders, static_dims = {}, {}
    for col in cont_cols:
        static_dims[col] = 0
    for col in cat_cols:
        stays_df[col] = stays_df[col].fillna('UNKNOWN').astype(str)
        le = LabelEncoder()
        stays_df[col] = le.fit_transform(stays_df[col])
        static_encoders[col] = le
        static_dims[col] = len(le.classes_)

    # 2. Extract High-Dim Sparse Features (Multi-hot without DRG)
    # Secondary Comorbidities (SEQ_NUM > 1, Top 32)
    logging.info("Loading Secondary Comorbidities ICD (SEQ_NUM > 1, Top 32)...")
    comorb_vocab_df = con.query(f"""
        SELECT SUBSTRING(CAST(ICD9_CODE AS VARCHAR), 1, 3) as code, count(*) as cnt
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/DIAGNOSES_ICD.csv', sample_size=-1)
        WHERE HADM_ID IN {hadm_ids_tuple} AND SEQ_NUM > 1 AND ICD9_CODE IS NOT NULL
        GROUP BY code ORDER BY cnt DESC LIMIT 32
    """).df()
    comorb_vocab = comorb_vocab_df['code'].tolist()
    comorb_raw_df = con.query(f"""
        SELECT HADM_ID as target_id, SUBSTRING(CAST(ICD9_CODE AS VARCHAR), 1, 3) as code
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/DIAGNOSES_ICD.csv', sample_size=-1)
        WHERE HADM_ID IN {hadm_ids_tuple} AND SEQ_NUM > 1 AND SUBSTRING(CAST(ICD9_CODE AS VARCHAR), 1, 3) IN {tuple(comorb_vocab)}
    """).df()
    icd_dict = {}
    for tid, group in comorb_raw_df.groupby('target_id'):
        vec = np.zeros(len(comorb_vocab), dtype=np.float32)
        for code_val in group['code']:
            if code_val in comorb_vocab:
                vec[comorb_vocab.index(code_val)] = 1.0
        icd_dict[tid] = vec
    icd_dim = len(comorb_vocab)
    
    logging.info("Loading Procedures ICD (Top 64)...")
    proc_dict, proc_dim = build_multihot_features(con, 'PROCEDURES_ICD', 'HADM_ID', 'ICD9_CODE', hadm_ids_tuple, top_k=64, trim=3)
    
    logging.info("Loading Pharmacy / Prescriptions (Top 64)...")
    rx_dict, rx_dim = build_multihot_features(con, 'PRESCRIPTIONS', 'HADM_ID', 'DRUG', hadm_ids_tuple, top_k=64)
    
    multihot_dims = {'icd': icd_dim, 'proc': proc_dim, 'rx': rx_dim}

    # 3. Dynamic Sequence Features (Vital Signs)
    # Includes both invasive (Arterial Line) and non-invasive (NIBP) blood pressure.
    item_map = {
        211: 0, 220045: 0,                           # HR
        618: 1, 220210: 1,                           # RR
        646: 2, 220277: 2,                           # SpO2
        51: 3, 220050: 3, 455: 3, 220179: 3,        # SBP (Arterial: 51, 220050; NIBP: 455, 220179)
        8368: 4, 220051: 4, 8441: 4, 220180: 4,     # DBP (Arterial: 8368, 220051; NIBP: 8441, 220180)
        52: 5, 220052: 5, 225312: 5, 456: 5, 220181: 5, # MAP (Arterial: 52, 220052, 225312; NIBP: 456, 220181)
        678: 6, 223761: 6, 676: 6, 223762: 6,       # Temperature
        807: 7, 811: 7, 1529: 7, 225664: 7, 220621: 7 # Glucose
    }
    logging.info(f"Querying CHARTEVENTS for sequences...")
    events_df = con.query(f"""
        SELECT ICUSTAY_ID as stay_id, CHARTTIME, ITEMID, VALUENUM
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/CHARTEVENTS.csv', sample_size=-1)
        WHERE ICUSTAY_ID IN {tuple(stay_ids)} AND ITEMID IN {tuple(item_map.keys())} AND VALUENUM IS NOT NULL
    """).df()
    
    logging.info("Formatting dataset...")
    X_seq, X_static, X_mh, X_note, Y = [], [], [], [], []
    
    X_note_texts = []
    for _, stay in stays_df.iterrows():
        sid = stay['stay_id']
        hadm = stay['HADM_ID']
        
        # Sequence
        evs = events_df[events_df['stay_id'] == sid].copy()
        seq = np.full((48, 8), np.nan, dtype=np.float32)
        if not evs.empty:
            evs['CHARTTIME'] = pd.to_datetime(evs['CHARTTIME'])
            evs['hour'] = ((evs['CHARTTIME'] - stay['INTIME']).dt.total_seconds() / 3600).astype(int)
            evs = evs[(evs['hour'] >= 0) & (evs['hour'] < 48)]
            # Prioritize invasive arterial line over cuff NIBP if measured within the same hour
            invasive_ids = {51, 8368, 52, 220050, 220051, 220052, 225312}
            evs['is_invasive'] = evs['ITEMID'].isin(invasive_ids).astype(int)
            evs = evs.sort_values(['hour', 'is_invasive', 'CHARTTIME'])
            for _, e in evs.iterrows():
                seq[int(e['hour']), item_map[e['ITEMID']]] = e['VALUENUM']

        df_seq = pd.DataFrame(seq).ffill().fillna(0.0)
        X_seq.append(df_seq.values)
        
        X_static.append([stay[col] for col in cont_cols + cat_cols])
        
        mh_vecs = []
        mh_vecs.append(icd_dict.get(hadm, np.zeros(icd_dim, dtype=np.float32)))
        mh_vecs.append(proc_dict.get(hadm, np.zeros(proc_dim, dtype=np.float32)))
        mh_vecs.append(rx_dict.get(hadm, np.zeros(rx_dim, dtype=np.float32)))
        X_mh.append(np.concatenate(mh_vecs))
        
        val = embeddings_dict.get(sid)
        if val is None:
            val = np.zeros(note_emb_dim, dtype=np.float32)
        elif isinstance(val, dict):
            val = val.get('embedding', np.zeros(note_emb_dim, dtype=np.float32))
        X_note.append(np.array(val, dtype=np.float32))

        note_text = load_note_texts({sid: embeddings_dict.get(sid, {})}).get(sid, '')
        if not note_text and isinstance(embeddings_dict.get(sid), dict):
            note_text = str(embeddings_dict[sid].get('summary', '') or '')
        X_note_texts.append(note_text)
        
        Y.append(stay['readmitted'])
        
    return np.array(X_seq), np.array(X_static, dtype=np.float32), np.array(X_mh, dtype=np.float32), np.array(X_note), np.array(Y), static_dims, multihot_dims, X_note_texts

def flatten_features(X_seq, X_static, X_mh):
    return np.concatenate([np.mean(X_seq, axis=1), X_seq[:, -1, :], X_static, X_mh], axis=1)

