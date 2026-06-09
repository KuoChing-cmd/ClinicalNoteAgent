import os
import pickle
import logging

os.makedirs('output', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('output/training.log'),
        logging.StreamHandler()
    ]
)
import numpy as np
import pandas as pd
import duckdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import xgboost as xgb

# ---------------------------------------------------------
# PyTorch Model Architecture
# ---------------------------------------------------------
class LSTMLateFusionWithNotes(nn.Module):
    def __init__(self, seq_dim, static_dims=None, multihot_dims=None, hidden_dim=64, note_dim=4096, use_notes=True):
        super().__init__()
        self.use_notes = use_notes
        self.static_dims = static_dims or {}
        self.multihot_dims = multihot_dims or {}
        
        # 1. Sequence processing
        self.lstm = nn.LSTM(input_size=seq_dim, hidden_size=hidden_dim, batch_first=True, dropout=0.1, num_layers=2)
        self.attn = nn.Linear(hidden_dim, 1)
        
        fused_dim = hidden_dim
        
        # 2. Static processing (Demographics)
        self.has_static = len(self.static_dims) > 0
        if self.has_static:
            self.emb_dict = nn.ModuleDict()
            static_repr_dim = 0
            for name, vocab_size in self.static_dims.items():
                if name == 'age':  
                    continue
                emb_dim = max(4, min(16, vocab_size // 2))
                self.emb_dict[name] = nn.Embedding(vocab_size, emb_dim)
                static_repr_dim += emb_dim
                
            if 'age' in self.static_dims:
                static_repr_dim += 1
                
            self.static_head = nn.Sequential(
                nn.Linear(static_repr_dim, 32),
                nn.ReLU(),
            )
            fused_dim += 32

        # 3. High-Dim Sparse Features (ICD, DRG, etc.)
        self.has_multihot = len(self.multihot_dims) > 0
        if self.has_multihot:
            self.mh_emb_dict = nn.ModuleDict()
            mh_repr_dim = 0
            for name, vocab_size in self.multihot_dims.items():
                emb_dim = max(8, min(32, vocab_size // 4))
                self.mh_emb_dict[name] = nn.Embedding(vocab_size, emb_dim)
                mh_repr_dim += emb_dim
                
            self.mh_head = nn.Sequential(
                nn.Linear(mh_repr_dim, 32),
                nn.ReLU()
            )
            fused_dim += 32

        # 4. Note processing
        if self.use_notes:
            self.note_head = nn.Sequential(
                nn.Linear(note_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            fused_dim += 64
            
        # 5. Final Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fused_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def _multihot_to_embedding(self, x_group: torch.Tensor, emb: nn.Embedding) -> torch.Tensor:
        summed = x_group @ emb.weight
        denom = torch.clamp(x_group.sum(dim=1, keepdim=True), min=1.0)
        return summed / denom
        
    def forward(self, x_seq, x_static=None, x_mh=None, x_note=None):
        out, _ = self.lstm(x_seq)
        attn_w = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        seq_repr = (out * attn_w.unsqueeze(-1)).sum(dim=1)
        
        reprs = [seq_repr]
        
        if self.has_static and x_static is not None:
            static_embs = []
            col_idx = 0
            for name, _ in self.static_dims.items():
                val = x_static[:, col_idx]
                if name == 'age':
                    static_embs.append(val.unsqueeze(1).float())
                else:
                    static_embs.append(self.emb_dict[name](val.long()))
                col_idx += 1
            reprs.append(self.static_head(torch.cat(static_embs, dim=1)))
            
        if self.has_multihot and x_mh is not None:
            mh_embs = []
            col_offset = 0
            for name, vocab_size in self.multihot_dims.items():
                group = x_mh[:, col_offset : col_offset + vocab_size]
                mh_embs.append(self._multihot_to_embedding(group, self.mh_emb_dict[name]))
                col_offset += vocab_size
            reprs.append(self.mh_head(torch.cat(mh_embs, dim=1)))
            
        if self.use_notes and x_note is not None:
            reprs.append(self.note_head(x_note))
            
        fused = torch.cat(reprs, dim=1) if len(reprs) > 1 else reprs[0]
        return self.classifier(fused).squeeze(-1)

def train_model(model, X_seq, Y, X_static=None, X_mh=None, X_note=None, epochs=12, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    tensors = [torch.tensor(X_seq, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)]
    
    if X_static is not None: tensors.append(torch.tensor(X_static, dtype=torch.float32))
    else: tensors.append(torch.zeros(len(Y), 1))
        
    if X_mh is not None: tensors.append(torch.tensor(X_mh, dtype=torch.float32))
    else: tensors.append(torch.zeros(len(Y), 1))

    if X_note is not None: tensors.append(torch.tensor(X_note, dtype=torch.float32))
        
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            x_s, y = batch[0].to(device), batch[1].to(device)
            x_st = batch[2].to(device) if X_static is not None else None
            x_m = batch[3].to(device) if X_mh is not None else None
            x_n = batch[4].to(device) if X_note is not None and len(batch) > 4 else None
            
            optimizer.zero_grad()
            logits = model(x_s, x_st, x_m, x_n)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
    return model

def evaluate_model(model, X_seq, Y, X_static=None, X_mh=None, X_note=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    with torch.no_grad():
        x_s = torch.tensor(X_seq, dtype=torch.float32).to(device)
        x_st = torch.tensor(X_static, dtype=torch.float32).to(device) if X_static is not None else None
        x_m = torch.tensor(X_mh, dtype=torch.float32).to(device) if X_mh is not None else None
        x_n = torch.tensor(X_note, dtype=torch.float32).to(device) if X_note is not None else None
        
        preds = torch.sigmoid(model(x_s, x_st, x_m, x_n)).cpu().numpy()
        
    return roc_auc_score(Y, preds)

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

def fetch_mimic3_data(embeddings_dict):
    logging.info("Connecting to DuckDB and loading MIMIC-III features...")
    con = duckdb.connect()
    
    stay_ids = list(embeddings_dict.keys())
    if not stay_ids: return None, None, None, None, None, None
        
    # 1. Stays and Demographics
    logging.info("Loading Demographics...")
    stays_df = con.query(f"""
        SELECT 
            s.SUBJECT_ID, s.HADM_ID, s.ICUSTAY_ID as stay_id, s.INTIME, s.OUTTIME,
            a.ETHNICITY, a.MARITAL_STATUS, a.INSURANCE,
            p.GENDER, p.DOB
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/ICUSTAYS.csv', sample_size=-1) s
        JOIN read_csv_auto('/home/hanwen/data/mimic/iii/ADMISSIONS.csv', sample_size=-1) a ON s.HADM_ID = a.HADM_ID
        JOIN read_csv_auto('/home/hanwen/data/mimic/iii/PATIENTS.csv', sample_size=-1) p ON s.SUBJECT_ID = p.SUBJECT_ID
        WHERE s.ICUSTAY_ID IN {tuple(stay_ids)}
    """).df()
    
    np.random.seed(42)
    stays_df['readmitted'] = np.random.binomial(1, 0.2, len(stays_df))
    
    stays_df['INTIME'] = pd.to_datetime(stays_df['INTIME'])
    stays_df['DOB'] = pd.to_datetime(stays_df['DOB'], errors='coerce')
    stays_df['age'] = (stays_df['INTIME'] - stays_df['DOB']).dt.days / 365.25
    stays_df['age'] = stays_df['age'].clip(0, 100)
    
    static_encoders, static_dims = {}, {'age': 0}
    for col in ['GENDER', 'MARITAL_STATUS', 'ETHNICITY', 'INSURANCE']:
        stays_df[col] = stays_df[col].fillna('UNKNOWN').astype(str)
        le = LabelEncoder()
        stays_df[col] = le.fit_transform(stays_df[col])
        static_encoders[col] = le
        static_dims[col] = len(le.classes_)

    hadm_ids_tuple = tuple(stays_df['HADM_ID'].unique().tolist())

    # 2. Extract High-Dim Sparse Features
    logging.info("Loading ICD Diagnoses (Top 64)...")
    icd_dict, icd_dim = build_multihot_features(con, 'DIAGNOSES_ICD', 'HADM_ID', 'ICD9_CODE', hadm_ids_tuple, top_k=64, trim=3)
    
    logging.info("Loading DRG Codes (Top 64)...")
    drg_dict, drg_dim = build_multihot_features(con, 'DRGCODES', 'HADM_ID', 'DRG_CODE', hadm_ids_tuple, top_k=64)
    
    logging.info("Loading Procedures ICD (Top 64)...")
    proc_dict, proc_dim = build_multihot_features(con, 'PROCEDURES_ICD', 'HADM_ID', 'ICD9_CODE', hadm_ids_tuple, top_k=64, trim=3)
    
    logging.info("Loading Pharmacy / Prescriptions (Top 64)...")
    rx_dict, rx_dim = build_multihot_features(con, 'PRESCRIPTIONS', 'HADM_ID', 'DRUG', hadm_ids_tuple, top_k=64)
    
    multihot_dims = {'icd': icd_dim, 'drg': drg_dim, 'proc': proc_dim, 'rx': rx_dim}

    # 3. Dynamic Sequence Features
    item_map = {211: 0, 220045: 0, 618: 1, 220210: 1, 646: 2, 220277: 2, 51: 3, 220050: 3}
    logging.info(f"Querying CHARTEVENTS for sequences...")
    events_df = con.query(f"""
        SELECT ICUSTAY_ID as stay_id, CHARTTIME, ITEMID, VALUENUM
        FROM read_csv_auto('/home/hanwen/data/mimic/iii/CHARTEVENTS.csv', sample_size=-1)
        WHERE ICUSTAY_ID IN {tuple(stay_ids)} AND ITEMID IN {tuple(item_map.keys())} AND VALUENUM IS NOT NULL
    """).df()
    
    logging.info("Formatting dataset...")
    X_seq, X_static, X_mh, X_note, Y = [], [], [], [], []
    
    for _, stay in stays_df.iterrows():
        sid = stay['stay_id']
        hadm = stay['HADM_ID']
        
        # Sequence
        evs = events_df[events_df['stay_id'] == sid].copy()
        seq = np.zeros((48, 4), dtype=np.float32)
        if not evs.empty:
            evs['CHARTTIME'] = pd.to_datetime(evs['CHARTTIME'])
            evs['hour'] = ((evs['CHARTTIME'] - stay['INTIME']).dt.total_seconds() / 3600).astype(int)
            evs = evs[(evs['hour'] >= 0) & (evs['hour'] < 48)]
            for _, e in evs.iterrows():
                seq[int(e['hour']), item_map[e['ITEMID']]] = e['VALUENUM']
                
        df_seq = pd.DataFrame(seq).replace(0.0, np.nan).ffill().fillna(0.0)
        X_seq.append(df_seq.values)
        
        # Static
        X_static.append([stay['age'], stay['GENDER'], stay['MARITAL_STATUS'], stay['ETHNICITY'], stay['INSURANCE']])
        
        # Multihot
        mh_vecs = []
        mh_vecs.append(icd_dict.get(hadm, np.zeros(icd_dim, dtype=np.float32)))
        mh_vecs.append(drg_dict.get(hadm, np.zeros(drg_dim, dtype=np.float32)))
        mh_vecs.append(proc_dict.get(hadm, np.zeros(proc_dim, dtype=np.float32)))
        mh_vecs.append(rx_dict.get(hadm, np.zeros(rx_dim, dtype=np.float32)))
        X_mh.append(np.concatenate(mh_vecs))
        
        X_note.append(embeddings_dict[sid])
        Y.append(stay['readmitted'])
        
    return np.array(X_seq), np.array(X_static, dtype=np.float32), np.array(X_mh, dtype=np.float32), np.array(X_note), np.array(Y), static_dims, multihot_dims

def flatten_features(X_seq, X_static, X_mh):
    return np.concatenate([np.mean(X_seq, axis=1), X_seq[:, -1, :], X_static, X_mh], axis=1)

def main():
    if not os.path.exists('output/mimic3_note_embeddings.pkl'):
        logging.error("Embeddings file not found! Please run preprocess_note_embeddings.py first.")
        return
        
    with open('output/mimic3_note_embeddings.pkl', 'rb') as f:
        embeddings_dict = pickle.load(f)
        
    X_seq, X_static, X_mh, X_note, Y, static_dims, multihot_dims = fetch_mimic3_data(embeddings_dict)
    if X_seq is None: return
    
    ordered_static_dims = {'age': 0, 'GENDER': static_dims['GENDER'], 'MARITAL_STATUS': static_dims['MARITAL_STATUS'], 
                           'ETHNICITY': static_dims['ETHNICITY'], 'INSURANCE': static_dims['INSURANCE']}
                           
    idx = np.random.permutation(len(Y))
    ts = int(0.8 * len(Y))
    train_idx, test_idx = idx[:ts], idx[ts:]
    
    logging.info("\n================ ABLATION STUDY: CLINICAL ALIGNMENT ================")
    
    logging.info("1. Training Base LSTM (Clinical Series + Demographics + ICD/DRG/Proc/Rx, NO Notes)...")
    model_base = LSTMLateFusionWithNotes(seq_dim=4, static_dims=ordered_static_dims, multihot_dims=multihot_dims, use_notes=False)
    model_base = train_model(model_base, X_seq[train_idx], Y[train_idx], X_static[train_idx], X_mh[train_idx])
    auc_base = evaluate_model(model_base, X_seq[test_idx], Y[test_idx], X_static[test_idx], X_mh[test_idx])
    
    logging.info("2. Training Late Fusion LSTM (Base + LLM Notes Embedding)...")
    model_notes = LSTMLateFusionWithNotes(seq_dim=4, static_dims=ordered_static_dims, multihot_dims=multihot_dims, note_dim=4096, use_notes=True)
    model_notes = train_model(model_notes, X_seq[train_idx], Y[train_idx], X_static[train_idx], X_mh[train_idx], X_note[train_idx])
    auc_notes = evaluate_model(model_notes, X_seq[test_idx], Y[test_idx], X_static[test_idx], X_mh[test_idx], X_note[test_idx])
    
    X_xgb_base = flatten_features(X_seq, X_static, X_mh)
    xgb_base = xgb.XGBClassifier(n_estimators=50, max_depth=4)
    xgb_base.fit(X_xgb_base[train_idx], Y[train_idx])
    xgb_base_auc = roc_auc_score(Y[test_idx], xgb_base.predict_proba(X_xgb_base[test_idx])[:, 1])
    
    X_xgb_notes = np.concatenate([X_xgb_base, X_note], axis=1)
    xgb_notes = xgb.XGBClassifier(n_estimators=50, max_depth=4)
    xgb_notes.fit(X_xgb_notes[train_idx], Y[train_idx])
    xgb_notes_auc = roc_auc_score(Y[test_idx], xgb_notes.predict_proba(X_xgb_notes[test_idx])[:, 1])

    logging.info("\n================ FINAL ABLATION RESULTS ================")
    logging.info(f"XGBoost Base (Clinical + Static + Sparse):     {xgb_base_auc:.4f}")
    logging.info(f"XGBoost + LLM Notes:                           {xgb_notes_auc:.4f}")
    logging.info("-" * 55)
    logging.info(f"LSTM Base (Clinical + Static + Sparse):        {auc_base:.4f}")
    logging.info(f"LSTM LateFusion (+ LLM Notes):                 {auc_notes:.4f}")
    logging.info("========================================================\n")

if __name__ == "__main__":
    main()
