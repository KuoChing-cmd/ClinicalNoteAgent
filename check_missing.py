import duckdb
import pandas as pd
import numpy as np

def check_missing_rates():
    print("Connecting to DuckDB...")
    con = duckdb.connect(':memory:')
    
    print("Loading ICUSTAYS...")
    stays_df = con.query("""
        SELECT ICUSTAY_ID as stay_id, INTIME, OUTTIME
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/ICUSTAYS.csv', sample_size=-1)
        WHERE INTIME IS NOT NULL AND OUTTIME IS NOT NULL
    """).df()
    
    # Filter stays length > 48h for fair comparison (optional, but let's just do all valid stays)
    stays_df['INTIME'] = pd.to_datetime(stays_df['INTIME'])
    stays_df['OUTTIME'] = pd.to_datetime(stays_df['OUTTIME'])
    stays_df = stays_df[(stays_df['OUTTIME'] - stays_df['INTIME']).dt.total_seconds() >= 48*3600]
    
    # Take a random sample of 5000 stays to speed up calculation
    stays_df = stays_df.sample(5000, random_state=42)
    stay_ids = tuple(stays_df['stay_id'].tolist())
    
    item_map = {
        211: 0, 220045: 0, 618: 1, 220210: 1, 646: 2, 220277: 2, 51: 3, 220050: 3,
        8368: 4, 220051: 4,               # Diastolic BP
        52: 5, 220052: 5, 225312: 5,      # MAP
        678: 6, 223761: 6, 676: 6, 223762: 6, # Temperature
        807: 7, 811: 7, 1529: 7, 225664: 7, 220621: 7 # Glucose
    }
    
    feature_names = [
        "Heart Rate", "Respiratory Rate", "SpO2", "Systolic BP", 
        "Diastolic BP", "MAP", "Temperature", "Glucose"
    ]
    
    print("Querying CHARTEVENTS for the sampled 5000 stays...")
    events_df = con.query(f"""
        SELECT ICUSTAY_ID as stay_id, CHARTTIME, ITEMID
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/CHARTEVENTS.csv', sample_size=-1)
        WHERE ICUSTAY_ID IN {stay_ids} AND ITEMID IN {tuple(item_map.keys())}
    """).df()
    
    events_df['CHARTTIME'] = pd.to_datetime(events_df['CHARTTIME'])
    
    # Merge with INTIME
    events_df = events_df.merge(stays_df[['stay_id', 'INTIME']], on='stay_id')
    events_df['hour'] = ((events_df['CHARTTIME'] - events_df['INTIME']).dt.total_seconds() / 3600).astype(int)
    
    # Filter 0-48h
    events_df = events_df[(events_df['hour'] >= 0) & (events_df['hour'] < 48)]
    events_df['feat_idx'] = events_df['ITEMID'].map(item_map)
    
    # Count observed slots
    # We group by stay_id, hour, and feat_idx to see if a slot has AT LEAST one measurement
    observed = events_df.groupby(['stay_id', 'hour', 'feat_idx']).size().reset_index()
    
    num_stays = len(stay_ids)
    total_slots_per_feat = num_stays * 48
    
    print("\n================== 缺失率统计 (Missing Rates) ==================")
    print(f"统计范围: 5000名入住时长>48h的患者，提取前48小时的数据。总时间槽/特征 = 240,000")
    
    for feat_idx, name in enumerate(feature_names):
        feat_observed = observed[observed['feat_idx'] == feat_idx]
        observed_slots = len(feat_observed)
        missing_rate = 1.0 - (observed_slots / total_slots_per_feat)
        
        # Calculate patient-level missing rate (patients with ZERO measurements in 48h)
        patients_with_data = feat_observed['stay_id'].nunique()
        pat_missing_rate = 1.0 - (patients_with_data / num_stays)
        
        print(f"[{feat_idx}] {name:<18} | 时序点缺失率 (Hour-level): {missing_rate:.1%} | 完全无数据患者比例 (Patient-level): {pat_missing_rate:.1%}")

if __name__ == '__main__':
    check_missing_rates()
