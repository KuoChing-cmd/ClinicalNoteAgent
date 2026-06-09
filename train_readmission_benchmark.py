#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import shap
import matplotlib.pyplot as plt
from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.database import DatabaseConfig, DatabaseManager
from src.database.mimic4_query import MIMIC4DataExtractor
from src.monitoring.monitoring_agent import MonitoringAgent

# Import latefusion components
from scripts.train_monitoring_risk_model_latefusion import (
    fetch_stay_rows,
    build_dataset,
    train_and_evaluate
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=3307)
    parser.add_argument("--db-user", default="root")
    parser.add_argument("--db-password", default="hanwen123")
    parser.add_argument("--db-name", default="mimic4")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--horizon-hours", type=int, default=72)
    parser.add_argument("--output-dir", default="output/benchmark_latefusion")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def flatten_seq_features(x_seq: np.ndarray) -> np.ndarray:
    """
    Flattens [N, seq_len, channels] into [N, 4 * channels]
    by computing mean, max, min, last values over seq_len.
    """
    # x_seq is [N, seq_len, channels]
    N, seq_len, channels = x_seq.shape
    
    # Replace zeros with NaNs for proper min/mean (assuming 0 is padding/missing for some channels, but here we just use all)
    # Actually, in LSTM features, 0 might be padding. We will just compute standard stats.
    seq_mean = np.mean(x_seq, axis=1)
    seq_max = np.max(x_seq, axis=1)
    seq_min = np.min(x_seq, axis=1)
    seq_last = x_seq[:, -1, :]
    
    return np.concatenate([seq_mean, seq_max, seq_min, seq_last], axis=1)

def compute_operational_features(stay_rows, session) -> np.ndarray:
    """
    Computes Load Index, Speedup LOS, and Post-ready Delay for each stay.
    Returns: [N, 3] numpy array
    """
    # Fetch all stays to compute accurate load index
    q_all_stays = text("SELECT intime, outtime FROM icustays WHERE intime IS NOT NULL AND outtime IS NOT NULL")
    res_stays = session.execute(q_all_stays).mappings().all()
    stays_df = pd.DataFrame(res_stays)
    stays_df['intime'] = pd.to_datetime(stays_df['intime'], errors='coerce')
    stays_df['outtime'] = pd.to_datetime(stays_df['outtime'], errors='coerce')
    stays_df = stays_df.dropna(subset=['intime', 'outtime'])
    
    in_times = np.sort(stays_df['intime'].values)
    out_times = np.sort(stays_df['outtime'].values)
    
    out = []
    los_hours_list = []
    
    np.random.seed(42)
    
    for row in stay_rows:
        intime = pd.to_datetime(row.intime)
        outtime = pd.to_datetime(row.outtime)
        
        los_hours = (outtime - intime).total_seconds() / 3600.0
        los_hours_list.append(los_hours)
        
        started_before = np.searchsorted(in_times, outtime.to_datetime64(), side='right')
        ended_before = np.searchsorted(out_times, outtime.to_datetime64(), side='left')
        load_index = max(0, started_before - ended_before)
        
        # Post-ready delay proxy
        post_ready_delay = load_index * 0.1 + np.random.normal(0, 1)
        
        out.append([load_index, los_hours, post_ready_delay])
        
    out = np.array(out, dtype=np.float32)
    
    # Calculate Speedup LOS = Mean LOS - Actual LOS
    mean_los = np.mean(out[:, 1])
    out[:, 1] = mean_los - out[:, 1]  # Overwrite LOS with Speedup LOS
    
    return out

def run_benchmark(x_seq, x_static, op_features, y, static_schema, output_dir, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Feature Alignment ---")
    x_seq_flat = flatten_seq_features(x_seq)
    
    # Combine static and flattened seq for traditional models
    X_clinical = np.concatenate([x_static, x_seq_flat], axis=1)
    
    # Op features: 0=load_index, 1=speedup_los, 2=post_ready_delay
    X_model_b = np.concatenate([X_clinical, op_features[:, 0:2]], axis=1)
    X_full = np.concatenate([X_clinical, op_features], axis=1)
    
    print(f"LSTM Sequence Features: {x_seq.shape}")
    print(f"LSTM Static Features: {x_static.shape}")
    print(f"Flattened Sequence Features: {x_seq_flat.shape}")
    print(f"XGBoost Clinical Base Features (Model A): {X_clinical.shape}")
    print(f"XGBoost Model C Features: {X_full.shape}")
    
    # Train test split for traditional models
    np.random.seed(seed)
    n_samples = len(y)
    order = np.random.permutation(n_samples)
    train_size = int(n_samples * 0.8)
    
    train_idx = order[:train_size]
    test_idx = order[train_size:]
    
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    print("\n--- Training Traditional Incremental Models ---")
    
    # Model A
    xgb_a = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=seed)
    xgb_a.fit(X_clinical[train_idx], y_train)
    auc_a = roc_auc_score(y_test, xgb_a.predict_proba(X_clinical[test_idx])[:, 1])
    print(f"XGBoost Model A (Clinical Aligned with LSTM) AUC: {auc_a:.4f}")
    
    # Model B
    xgb_b = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=seed)
    xgb_b.fit(X_model_b[train_idx], y_train)
    auc_b = roc_auc_score(y_test, xgb_b.predict_proba(X_model_b[test_idx])[:, 1])
    print(f"XGBoost Model B (Clinical + Load + Speedup) AUC: {auc_b:.4f}")
    
    # Model C
    xgb_c = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=seed)
    xgb_c.fit(X_full[train_idx], y_train)
    auc_c = roc_auc_score(y_test, xgb_c.predict_proba(X_full[test_idx])[:, 1])
    print(f"XGBoost Model C (Full: Clinical + Operational) AUC: {auc_c:.4f}")
    
    print("\n--- Other Baselines (Full Features) ---")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=seed),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=seed, verbose=-1)
    }
    for name, model in models.items():
        model.fit(X_full[train_idx], y_train)
        preds = model.predict_proba(X_full[test_idx])[:, 1]
        auc = roc_auc_score(y_test, preds)
        print(f"{name} AUC: {auc:.4f}")
        
    print("\n--- Ablation Study on XGBoost ---")
    X_no_load = np.concatenate([X_clinical, op_features[:, 1:3]], axis=1) # skip load_index (0)
    xgb_no_load = xgb.XGBClassifier(random_state=seed)
    xgb_no_load.fit(X_no_load[train_idx], y_train)
    auc_no_load = roc_auc_score(y_test, xgb_no_load.predict_proba(X_no_load[test_idx])[:, 1])
    print(f"XGBoost (without Load Index) AUC: {auc_no_load:.4f} (Drop: {auc_c - auc_no_load:.4f})")
    
    # SHAP
    print("\n--- Generating SHAP Analysis ---")
    explainer = shap.TreeExplainer(xgb_c)
    shap_values = explainer.shap_values(X_full[test_idx])
    
    # Feature names for SHAP
    feature_names = [f"Static_{i}" for i in range(x_static.shape[1])]
    feature_names += [f"SeqFlat_{i}" for i in range(x_seq_flat.shape[1])]
    feature_names += ["Load_Index", "Speedup_LOS", "PostReady_Delay"]
    
    plt.figure()
    shap.summary_plot(shap_values, X_full[test_idx], feature_names=feature_names, show=False)
    shap_img_path = os.path.join(output_dir, 'shap_summary.png')
    plt.savefig(shap_img_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to {shap_img_path}")
    
    # Finally, train LSTM LateFusion to compare
    print("\n--- Training LSTM LateFusion (Clinical Baseline) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # For fair comparison, we use train_and_evaluate from train_monitoring_risk_model_latefusion
    # Note: the dataset splits in train_and_evaluate are randomized internally based on seed, 
    # we pass the same seed so it evaluates on a consistent split roughly.
    result = train_and_evaluate(
        x_seq=x_seq,
        x_static=x_static,
        y=y,
        static_schema=static_schema,
        epochs=15,  # short epochs for benchmark
        batch_size=128,
        lr=1e-3,
        seed=seed,
        val_ratio=0.1,
        test_ratio=0.2, # aligns roughly with 80/20 train/test
        loss_name="focal",
        focal_gamma=2.0,
        focal_alpha=0.75,
        threshold_policy="val_target_recall_max_precision",
        threshold_target_recall=0.35,
        early_stop_enabled=True,
        early_stop_patience=5,
        early_stop_min_delta=1e-4,
        early_stop_monitor="val_pr_auc",
        device=device,
    )
    
    lstm_auc = result["test_metrics_selected_threshold"]["roc_auc"]
    print(f"LSTM LateFusion (Clinical Baseline) Test AUC: {lstm_auc:.4f}")
    
    print("\n================ BENCHMARK SUMMARY ================")
    print(f"LSTM LateFusion (Clinical Base): {lstm_auc:.4f}")
    print(f"XGBoost Model A (Clinical Base): {auc_a:.4f}")
    print(f"XGBoost Model B (+ Load + Speedup): {auc_b:.4f}")
    print(f"XGBoost Model C (+ Operational Full): {auc_c:.4f}")
    print("===================================================")


def main():
    args = parse_args()
    config = DatabaseConfig(host=args.db_host, port=args.db_port, user=args.db_user, password=args.db_password, database=args.db_name)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    monitor_agent = MonitoringAgent(seed=args.seed)
    
    print("Fetching stay rows from database...")
    with DatabaseManager(config) as db:
        session = db.get_session()
        extractor = MIMIC4DataExtractor(session)
        
        stay_rows = fetch_stay_rows(
            session,
            limit=args.limit,
            horizon_hours=args.horizon_hours,
        )
        
        if not stay_rows:
            print("No data found!")
            return
            
        print(f"Found {len(stay_rows)} stay rows. Building dataset...")
        
        x_seq, x_static, y, static_schema = build_dataset(
            extractor=extractor,
            monitor_agent=monitor_agent,
            stay_rows=stay_rows,
            seq_len=48,
            max_query_rows=50000,
            include_all_chartevents=False,
            include_outputevents=True,
            include_datetimeevents=False,
            include_labevents=True,
            include_inputevents=False,
            include_omr=False,
            include_static_demographics=True,
            include_admission_context=True,
            include_icd_history=True,
            icd_top_k=64,
            icd_include_current_hadm=False,
            include_drg_onehot=True,
            drg_top_k=64,
            include_procedure_icd=True,
            procedure_top_k=64,
            include_hcpcs_events=True,
            hcpcs_top_k=64,
            include_pharmacy=True,
            pharmacy_top_k=64,
            pre_discharge_hours=48,
            use_core_channels=False,
            selected_channels=None,
            seq_aggregation="last",
            sql_in_batch_size=1000,
        )
        
        print("Dataset built. Computing operational features...")
        op_features = compute_operational_features(stay_rows, session)
        
    run_benchmark(x_seq, x_static, op_features, y, static_schema, args.output_dir, seed=args.seed)

if __name__ == '__main__':
    main()
