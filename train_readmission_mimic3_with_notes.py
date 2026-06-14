import os
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
import math

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
                nn.LayerNorm(note_dim),
                nn.Linear(note_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            fused_dim += 64
            
        # 5. Final Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
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
            import torch.nn.functional as F
            x_note_norm = F.normalize(x_note, p=2, dim=1)
            reprs.append(self.note_head(x_note_norm))
            
        fused = torch.cat(reprs, dim=1) if len(reprs) > 1 else reprs[0]
        return self.classifier(fused).squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x.transpose(0, 1)

class TransformerEarlyFusionWithNotes(nn.Module):
    def __init__(self, seq_dim, static_dims=None, multihot_dims=None, hidden_dim=64, note_dim=4096, use_notes=True, nhead=4, num_layers=2):
        super().__init__()
        self.use_notes = use_notes
        self.static_dims = static_dims or {}
        self.multihot_dims = multihot_dims or {}
        
        # 1. Sequence processing (Transformer)
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=0.1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        fused_dim = hidden_dim # We'll just use CLS token output
        
        # 2. Static processing (Demographics)
        self.has_static = len(self.static_dims) > 0
        if self.has_static:
            self.emb_dict = nn.ModuleDict()
            static_repr_dim = 0
            for name, vocab_size in self.static_dims.items():
                if name == 'age': continue
                emb_dim = max(4, min(16, vocab_size // 2))
                self.emb_dict[name] = nn.Embedding(vocab_size, emb_dim)
                static_repr_dim += emb_dim
            if 'age' in self.static_dims:
                static_repr_dim += 1
            self.static_head = nn.Sequential(nn.Linear(static_repr_dim, hidden_dim), nn.ReLU())
            
        # 3. High-Dim Sparse Features (ICD, DRG, etc.)
        self.has_multihot = len(self.multihot_dims) > 0
        if self.has_multihot:
            self.mh_emb_dict = nn.ModuleDict()
            mh_repr_dim = 0
            for name, vocab_size in self.multihot_dims.items():
                emb_dim = max(8, min(32, vocab_size // 4))
                self.mh_emb_dict[name] = nn.Embedding(vocab_size, emb_dim)
                mh_repr_dim += emb_dim
            self.mh_head = nn.Sequential(nn.Linear(mh_repr_dim, hidden_dim), nn.ReLU())
            
        # 4. Note processing
        if self.use_notes:
            self.note_head = nn.Sequential(
                nn.LayerNorm(note_dim),
                nn.Linear(note_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
        # 5. Final Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
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
        B = x_seq.shape[0]
        x = self.seq_proj(x_seq) # [B, T, H]
        
        extra_tokens = []
        
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
            static_repr = self.static_head(torch.cat(static_embs, dim=1))
            extra_tokens.append(static_repr.unsqueeze(1))
            
        if self.has_multihot and x_mh is not None:
            mh_embs = []
            col_offset = 0
            for name, vocab_size in self.multihot_dims.items():
                group = x_mh[:, col_offset : col_offset + vocab_size]
                mh_embs.append(self._multihot_to_embedding(group, self.mh_emb_dict[name]))
                col_offset += vocab_size
            mh_repr = self.mh_head(torch.cat(mh_embs, dim=1))
            extra_tokens.append(mh_repr.unsqueeze(1))
            
        if self.use_notes and x_note is not None:
            import torch.nn.functional as F
            x_note_norm = F.normalize(x_note, p=2, dim=1)
            note_repr = self.note_head(x_note_norm)
            extra_tokens.append(note_repr.unsqueeze(1))
            
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        if len(extra_tokens) > 0:
            extra_tokens_tensor = torch.cat(extra_tokens, dim=1)
            x = torch.cat((cls_tokens, extra_tokens_tensor, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
            
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        
        cls_out = out[:, 0, :]
        return self.classifier(cls_out).squeeze(-1)


class CrossModalAttnFusion(nn.Module):
    """
    Cross-Modal Attention Fusion for readmission prediction.

    Architecture:
      1. LSTM encodes the physiological time series → hidden states H [B, T, d]
      2. Note embedding is projected into M 'virtual tokens' [B, M, d]
         (M > 1 gives the attention head diversity; default M=4)
      3. Cross-Attention: Q = H (time series), K = V = note tokens
         → each time step selectively reads the relevant note semantics
      4. Residual + LayerNorm stabilises gradients
      5. Temporal self-attention pools the enriched H → seq_repr [B, d]
      6. Demographics + sparse codes are fused via late concat → classifier

    Saved attention weights allow post-hoc visualisation of which time
    steps are most influenced by which note concepts.
    """
    def __init__(self, seq_dim, static_dims=None, multihot_dims=None,
                 hidden_dim=64, note_dim=4096, nhead=4,
                 num_virtual_tokens=4, num_lstm_layers=2):
        super().__init__()
        self.static_dims   = static_dims   or {}
        self.multihot_dims = multihot_dims or {}
        self.hidden_dim    = hidden_dim
        self.M             = num_virtual_tokens

        # ── 1. Physiological sequence encoder ────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=seq_dim, hidden_size=hidden_dim,
            num_layers=num_lstm_layers, batch_first=True,
            dropout=0.1 if num_lstm_layers > 1 else 0.0
        )

        # ── 2. Note → M virtual tokens ────────────────────────────────────────
        # Gradual compression avoids information bottleneck at 4096 → hidden_dim
        self.note_proj = nn.Sequential(
            nn.LayerNorm(note_dim),
            nn.Linear(note_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim * 4, hidden_dim * num_virtual_tokens),
        )
        # note_proj output will be reshaped to [B, M, hidden_dim]

        # ── 3. Cross-Attention block (seq queries ← note keys/values) ─────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=nhead,
            dropout=0.1, batch_first=True
        )
        self.cross_norm  = nn.LayerNorm(hidden_dim)   # post-attention norm
        self.cross_ff    = nn.Sequential(             # position-wise FFN
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.cross_ff_norm = nn.LayerNorm(hidden_dim)

        # ── 4. Temporal self-attention pooling ────────────────────────────────
        self.temporal_attn = nn.Linear(hidden_dim, 1)

        fused_dim = hidden_dim

        # ── 5. Static (demographics) ──────────────────────────────────────────
        self.has_static = len(self.static_dims) > 0
        if self.has_static:
            self.emb_dict      = nn.ModuleDict()
            static_repr_dim    = 0
            for name, vocab_size in self.static_dims.items():
                if name == 'age': continue
                emb_dim = max(4, min(16, vocab_size // 2))
                self.emb_dict[name] = nn.Embedding(vocab_size, emb_dim)
                static_repr_dim += emb_dim
            if 'age' in self.static_dims:
                static_repr_dim += 1
            self.static_head = nn.Sequential(
                nn.Linear(static_repr_dim, 32), nn.ReLU()
            )
            fused_dim += 32

        # ── 6. Sparse multi-hot (ICD / DRG / Proc / Rx) ───────────────────────
        self.has_multihot = len(self.multihot_dims) > 0
        if self.has_multihot:
            self.mh_emb_dict = nn.ModuleDict()
            mh_repr_dim = 0
            for name, vocab_size in self.multihot_dims.items():
                emb_dim = max(8, min(32, vocab_size // 4))
                self.mh_emb_dict[name] = nn.Embedding(vocab_size, emb_dim)
                mh_repr_dim += emb_dim
            self.mh_head = nn.Sequential(
                nn.Linear(mh_repr_dim, 32), nn.ReLU()
            )
            fused_dim += 32

        # ── 7. Classification head ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(0.3),
            nn.Linear(fused_dim, 64), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    # ── helpers ──────────────────────────────────────────────────────────────
    def _multihot_to_embedding(self, x_group, emb):
        summed = x_group @ emb.weight
        denom  = torch.clamp(x_group.sum(dim=1, keepdim=True), min=1.0)
        return summed / denom

    def _encode_static(self, x_static):
        static_embs, col_idx = [], 0
        for name, _ in self.static_dims.items():
            val = x_static[:, col_idx]
            if name == 'age':
                static_embs.append(val.unsqueeze(1).float())
            else:
                static_embs.append(self.emb_dict[name](val.long()))
            col_idx += 1
        return self.static_head(torch.cat(static_embs, dim=1))

    def _encode_multihot(self, x_mh):
        mh_embs, col_offset = [], 0
        for name, vocab_size in self.multihot_dims.items():
            group = x_mh[:, col_offset : col_offset + vocab_size]
            mh_embs.append(self._multihot_to_embedding(group, self.mh_emb_dict[name]))
            col_offset += vocab_size
        return self.mh_head(torch.cat(mh_embs, dim=1))

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x_seq, x_static=None, x_mh=None, x_note=None,
                return_attn=False):
        """
        Args:
            x_seq    : [B, T, seq_dim]   physiological time series
            x_static : [B, num_static]   demographic features
            x_mh     : [B, sum(vocab_k)] multi-hot sparse features
            x_note   : [B, note_dim]     LLM note embedding
            return_attn: if True, also return cross-attention weights [B, T, M]
        """
        import torch.nn.functional as F

        # 1. LSTM encode sequence  → H [B, T, d]
        H, _ = self.lstm(x_seq)

        # 2. Project note → M virtual tokens [B, M, d]
        if x_note is not None:
            x_note_norm = F.normalize(x_note, p=2, dim=1)
            note_tokens = self.note_proj(x_note_norm)          # [B, M*d]
            note_tokens = note_tokens.view(
                x_note.shape[0], self.M, self.hidden_dim       # [B, M, d]
            )
        else:
            # fallback: zero note tokens (model still works without notes)
            note_tokens = torch.zeros(
                H.shape[0], self.M, self.hidden_dim, device=H.device
            )

        # 3. Cross-Attention  Q=H, K=V=note_tokens
        attn_out, attn_weights = self.cross_attn(
            query=H, key=note_tokens, value=note_tokens
        )                                                        # [B, T, d]
        H = self.cross_norm(H + attn_out)                       # residual
        H = self.cross_ff_norm(H + self.cross_ff(H))            # FFN + residual

        # 4. Temporal self-attention pooling  → seq_repr [B, d]
        temp_w   = torch.softmax(self.temporal_attn(H).squeeze(-1), dim=1)  # [B, T]
        seq_repr = (H * temp_w.unsqueeze(-1)).sum(dim=1)        # [B, d]

        reprs = [seq_repr]

        # 5. Demographics
        if self.has_static and x_static is not None:
            reprs.append(self._encode_static(x_static))

        # 6. Sparse codes
        if self.has_multihot and x_mh is not None:
            reprs.append(self._encode_multihot(x_mh))

        fused  = torch.cat(reprs, dim=1) if len(reprs) > 1 else reprs[0]
        logits = self.classifier(fused).squeeze(-1)

        if return_attn:
            return logits, attn_weights   # attn_weights: [B, T, M]
        return logits

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
    return float(np.trapezoid(precisions, recalls))


def train_model(model, X_seq, Y, X_static=None, X_mh=None, X_note=None,
                X_seq_val=None, Y_val=None, X_static_val=None, X_mh_val=None, X_note_val=None,
                epochs=12, lr=1e-3, batch_size=256, pos_weight=None,
                early_stop_patience=15, early_stop_min_delta=1e-4):
    """
    Train a PyTorch model with:
      - Optimization ②: CosineAnnealingLR learning rate scheduling
      - Optimization ③: pos_weight in BCEWithLogitsLoss to handle class imbalance
      - Per-epoch validation loss / AUROC / PR-AUC tracking
      - Early stopping based on val_loss (patience + best checkpoint restore)
    
    Returns:
        (model, epoch_history) where epoch_history is a list of dicts with
        train_loss, val_loss, val_auroc, val_prauc, lr per epoch.
        Model weights are restored to the best val_loss checkpoint.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logging.info(f"Using device: {device} for training")
    
    tensors = [torch.tensor(X_seq, dtype=torch.float32).to(device), torch.tensor(Y, dtype=torch.float32).to(device)]
    
    if X_static is not None: tensors.append(torch.tensor(X_static, dtype=torch.float32).to(device))
    else: tensors.append(torch.zeros(len(Y), 1).to(device))
        
    if X_mh is not None: tensors.append(torch.tensor(X_mh, dtype=torch.float32).to(device))
    else: tensors.append(torch.zeros(len(Y), 1).to(device))

    if X_note is not None: tensors.append(torch.tensor(X_note, dtype=torch.float32).to(device))
        
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Prepare validation data on device (if provided)
    has_val = X_seq_val is not None and Y_val is not None
    if has_val:
        val_tensors = [
            torch.tensor(X_seq_val, dtype=torch.float32).to(device),
            torch.tensor(Y_val, dtype=torch.float32).to(device),
        ]
        if X_static_val is not None: val_tensors.append(torch.tensor(X_static_val, dtype=torch.float32).to(device))
        else: val_tensors.append(torch.zeros(len(Y_val), 1).to(device))
        if X_mh_val is not None: val_tensors.append(torch.tensor(X_mh_val, dtype=torch.float32).to(device))
        else: val_tensors.append(torch.zeros(len(Y_val), 1).to(device))
        if X_note_val is not None: val_tensors.append(torch.tensor(X_note_val, dtype=torch.float32).to(device))
        val_dataset = TensorDataset(*val_tensors)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
    
    # Optimization ③: class-imbalance-aware loss
    pw = torch.tensor([pos_weight], dtype=torch.float32).to(device) if pos_weight is not None else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    if pos_weight is not None:
        logging.info(f"  pos_weight = {pos_weight:.2f} (neg/pos ratio, correcting class imbalance)")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Optimization ②: cosine annealing LR — starts at lr, decays smoothly to 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    epoch_history = []
    
    # ── Early stopping state ──────────────────────────────────────────────────
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improve = 0
    best_state_dict = None
    if has_val:
        logging.info(f"  early_stopping: patience={early_stop_patience}, min_delta={early_stop_min_delta:.1e}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_s, y = batch[0], batch[1]
            x_st = batch[2] if X_static is not None else None
            x_m  = batch[3] if X_mh    is not None else None
            x_n  = batch[4] if X_note  is not None else None  # Bug1 fix: len(batch) guard was unreliable
            
            optimizer.zero_grad()
            logits = model(x_s, x_st, x_m, x_n)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        train_loss = epoch_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]
        
        # ── Per-epoch validation metrics ──────────────────────────────────────
        row = {
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 6),
            'lr': float(f"{current_lr:.2e}"),
        }
        
        if has_val:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            all_val_preds = []
            all_val_labels = []
            with torch.no_grad():
                for vbatch in val_loader:
                    vx_s, vy = vbatch[0], vbatch[1]
                    vx_st = vbatch[2] if X_static_val is not None else None
                    vx_m  = vbatch[3] if X_mh_val    is not None else None
                    vx_n  = vbatch[4] if X_note_val  is not None else None
                    
                    vlogits = model(vx_s, vx_st, vx_m, vx_n)
                    vloss = criterion(vlogits, vy)
                    val_loss_sum += vloss.item()
                    val_batches += 1
                    all_val_preds.append(torch.sigmoid(vlogits).cpu().numpy())
                    all_val_labels.append(vy.cpu().numpy())
            
            val_loss = val_loss_sum / max(1, val_batches)
            val_preds = np.concatenate(all_val_preds)
            val_labels = np.concatenate(all_val_labels)
            val_auroc = roc_auc_score(val_labels, val_preds)
            val_prauc = _compute_prauc(val_labels, val_preds)
            
            row['val_loss'] = round(val_loss, 6)
            row['val_auroc'] = round(val_auroc, 4)
            row['val_prauc'] = round(val_prauc, 4)
            
            # ── Early stopping check ─────────────────────────────────────────
            if val_loss < (best_val_loss - early_stop_min_delta):
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epochs_without_improve = 0
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_without_improve += 1
            
            row['best_val_loss'] = round(best_val_loss, 6)
            row['patience_counter'] = epochs_without_improve
            
            logging.info(
                f"Epoch {epoch+1}/{epochs} completed - "
                f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
                f"val_AUROC: {val_auroc:.4f}  val_PRAUC: {val_prauc:.4f}  "
                f"LR: {current_lr:.2e}  "
                f"best_val_loss: {best_val_loss:.4f} (ep{best_epoch})  "
                f"patience: {epochs_without_improve}/{early_stop_patience}"
            )
            
            # ── Trigger early stop ───────────────────────────────────────────
            if epochs_without_improve >= early_stop_patience:
                logging.info(
                    f"⚡ Early stopping triggered at epoch {epoch+1}. "
                    f"Best val_loss={best_val_loss:.6f} at epoch {best_epoch}. "
                    f"No improvement for {early_stop_patience} consecutive epochs."
                )
                row['early_stopped'] = True
                epoch_history.append(row)
                break
        else:
            logging.info(f"Epoch {epoch+1}/{epochs} completed - Loss: {train_loss:.4f}  LR: {current_lr:.2e}")
        
        epoch_history.append(row)
    
    # ── Restore best checkpoint ───────────────────────────────────────────────
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logging.info(
            f"✓ Restored best model from epoch {best_epoch} "
            f"(val_loss={best_val_loss:.6f})"
        )
    
    return model, epoch_history

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


def evaluate_model(model, X_seq, Y, X_static=None, X_mh=None, X_note=None, batch_size=512):
    """
    Evaluate a PyTorch model. Returns a dict containing:
      - y_prob: raw predicted probabilities
      - best_threshold: F1-optimal threshold
      - metrics_at_best_t: full metrics at best threshold
      - metrics_at_0.5: full metrics at 0.5
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    tensors = [torch.tensor(X_seq, dtype=torch.float32).to(device), torch.tensor(Y, dtype=torch.float32).to(device)]
    
    if X_static is not None: tensors.append(torch.tensor(X_static, dtype=torch.float32).to(device))
    else: tensors.append(torch.zeros(len(Y), 1).to(device))
        
    if X_mh is not None: tensors.append(torch.tensor(X_mh, dtype=torch.float32).to(device))
    else: tensors.append(torch.zeros(len(Y), 1).to(device))

    if X_note is not None: tensors.append(torch.tensor(X_note, dtype=torch.float32).to(device))
        
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            x_s = batch[0]
            x_st = batch[2] if X_static is not None else None
            x_m  = batch[3] if X_mh    is not None else None
            x_n  = batch[4] if X_note  is not None else None
            
            preds = torch.sigmoid(model(x_s, x_st, x_m, x_n))
            all_preds.append(preds.cpu().numpy())
            
    y_prob = np.concatenate(all_preds)
    y_true = np.asarray(Y, dtype=np.int64)
    
    best_t = _search_best_threshold_f1(y_true, y_prob)
    metrics_best = _compute_classification_metrics(y_true, y_prob, threshold=best_t)
    metrics_05   = _compute_classification_metrics(y_true, y_prob, threshold=0.5)
    
    return {
        'y_prob': y_prob,
        'best_threshold': best_t,
        'metrics_at_best_t': metrics_best,
        'metrics_at_0.5': metrics_05,
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
        # Bug3 fix: initialize with NaN so that un-observed slots are truly missing,
        # and real zero-valued measurements are NOT incorrectly treated as absent.
        seq = np.full((48, 4), np.nan, dtype=np.float32)
        if not evs.empty:
            evs['CHARTTIME'] = pd.to_datetime(evs['CHARTTIME'])
            evs['hour'] = ((evs['CHARTTIME'] - stay['INTIME']).dt.total_seconds() / 3600).astype(int)
            evs = evs[(evs['hour'] >= 0) & (evs['hour'] < 48)]
            for _, e in evs.iterrows():
                seq[int(e['hour']), item_map[e['ITEMID']]] = e['VALUENUM']

        # ffill: carry last observed value forward; fill remaining leading NaNs with 0
        df_seq = pd.DataFrame(seq).ffill().fillna(0.0)
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
        
        val = embeddings_dict.get(sid, np.zeros(4096, dtype=np.float32))
        if isinstance(val, dict):
            val = val.get('embedding', np.zeros(4096, dtype=np.float32))
        X_note.append(np.array(val, dtype=np.float32))
        
        Y.append(stay['readmitted'])
        
    return np.array(X_seq), np.array(X_static, dtype=np.float32), np.array(X_mh, dtype=np.float32), np.array(X_note), np.array(Y), static_dims, multihot_dims

def flatten_features(X_seq, X_static, X_mh):
    return np.concatenate([np.mean(X_seq, axis=1), X_seq[:, -1, :], X_static, X_mh], axis=1)

# ─── Experiment Configuration ────────────────────────────────────────────────
# Edit these values to configure a run. The exp dir name is auto-generated.
EXP_CONFIG = {
    "epochs":      200,
    "lr":          1e-3,
    "batch_size":  256,
    "hidden_dim":  64,
    "note_dim":    4096,
    "top_k_codes": 64,          # top-K for ICD/DRG/Proc/Rx multi-hot
    "xgb_n_est":   200,
    "xgb_depth":   6,
    # Optimizations enabled in this run
    "opt_seq_norm":    True,     # ① StandardScaler on sequence features
    "opt_cosine_lr":   True,     # ② CosineAnnealingLR scheduler
    "opt_pos_weight":  True,     # ③ pos_weight for class imbalance
}

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
    )
    exp_dir = os.path.join('output', f"{ts}_{tag}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def main():
    if not os.path.exists('output/mimic3_note_embeddings.pkl'):
        logging.error("Embeddings file not found! Please run preprocess_note_embeddings.py first.")
        return
    
    # Create the experiment directory for this run
    exp_dir = make_exp_dir()
    run_start = datetime.now()
    logging.info(f"Experiment directory: {exp_dir}")
    # Add a per-experiment log handler
    exp_log_handler = logging.FileHandler(os.path.join(exp_dir, 'training.log'))
    exp_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logging.getLogger().addHandler(exp_log_handler)
        
    with open('output/mimic3_note_embeddings.pkl', 'rb') as f:
        embeddings_dict = pickle.load(f)
        
    X_seq, X_static, X_mh, X_note, Y, static_dims, multihot_dims = fetch_mimic3_data(embeddings_dict)
    if X_seq is None: return
    
    ordered_static_dims = {'age': 0, 'GENDER': static_dims['GENDER'], 'MARITAL_STATUS': static_dims['MARITAL_STATUS'], 
                           'ETHNICITY': static_dims['ETHNICITY'], 'INSURANCE': static_dims['INSURANCE']}
                           
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
    model_base = LSTMLateFusionWithNotes(seq_dim=4, static_dims=ordered_static_dims, multihot_dims=multihot_dims, use_notes=False)
    model_base, hist_base = train_model(
        model_base, X_seq[train_idx], Y[train_idx], X_static[train_idx], X_mh[train_idx],
        X_seq_val=X_seq[val_idx], Y_val=Y[val_idx], X_static_val=X_static[val_idx], X_mh_val=X_mh[val_idx],
        epochs=EXP_CONFIG['epochs'], lr=EXP_CONFIG['lr'], batch_size=EXP_CONFIG['batch_size'],
        pos_weight=pos_weight_val)
    all_epoch_histories['lstm_base'] = hist_base
    eval_base = evaluate_model(model_base, X_seq[test_idx], Y[test_idx], X_static[test_idx], X_mh[test_idx])
    _log_eval_result('LSTM Base', eval_base)
    torch.save(model_base.state_dict(), os.path.join(exp_dir, 'model_lstm_base.pt'))
    logging.info(f"Base LSTM model saved to {exp_dir}/model_lstm_base.pt")
    
    logging.info("2. Training Late Fusion LSTM (Base + LLM Notes Embedding)...")
    model_notes = LSTMLateFusionWithNotes(seq_dim=4, static_dims=ordered_static_dims, multihot_dims=multihot_dims, note_dim=4096, use_notes=True)
    model_notes, hist_notes = train_model(
        model_notes, X_seq[train_idx], Y[train_idx], X_static[train_idx], X_mh[train_idx], X_note[train_idx],
        X_seq_val=X_seq[val_idx], Y_val=Y[val_idx], X_static_val=X_static[val_idx], X_mh_val=X_mh[val_idx], X_note_val=X_note[val_idx],
        epochs=EXP_CONFIG['epochs'], lr=EXP_CONFIG['lr'], batch_size=EXP_CONFIG['batch_size'],
        pos_weight=pos_weight_val)
    all_epoch_histories['lstm_latefusion_notes'] = hist_notes
    eval_notes = evaluate_model(model_notes, X_seq[test_idx], Y[test_idx], X_static[test_idx], X_mh[test_idx], X_note[test_idx])
    _log_eval_result('LSTM LateFusion', eval_notes)
    torch.save(model_notes.state_dict(), os.path.join(exp_dir, 'model_lstm_notes.pt'))
    logging.info(f"Late Fusion LSTM model saved to {exp_dir}/model_lstm_notes.pt")
    
    logging.info("3. Training Early Fusion Transformer (Base + LLM Notes Embedding)...")
    model_tf_notes = TransformerEarlyFusionWithNotes(seq_dim=4, static_dims=ordered_static_dims, multihot_dims=multihot_dims, note_dim=4096, use_notes=True)
    model_tf_notes, hist_tf = train_model(
        model_tf_notes, X_seq[train_idx], Y[train_idx], X_static[train_idx], X_mh[train_idx], X_note[train_idx],
        X_seq_val=X_seq[val_idx], Y_val=Y[val_idx], X_static_val=X_static[val_idx], X_mh_val=X_mh[val_idx], X_note_val=X_note[val_idx],
        epochs=EXP_CONFIG['epochs'], lr=EXP_CONFIG['lr'], batch_size=EXP_CONFIG['batch_size'],
        pos_weight=pos_weight_val)
    all_epoch_histories['transformer_earlyfusion_notes'] = hist_tf
    eval_tf = evaluate_model(model_tf_notes, X_seq[test_idx], Y[test_idx], X_static[test_idx], X_mh[test_idx], X_note[test_idx])
    _log_eval_result('Transformer EarlyFusion', eval_tf)
    torch.save(model_tf_notes.state_dict(), os.path.join(exp_dir, 'model_tf_notes.pt'))
    logging.info(f"Early Fusion Transformer model saved to {exp_dir}/model_tf_notes.pt")
    
    logging.info("4. Training Cross-Modal Attention Fusion (LSTM × Note Cross-Attention)...")
    model_cross = CrossModalAttnFusion(
        seq_dim=4, static_dims=ordered_static_dims, multihot_dims=multihot_dims,
        hidden_dim=EXP_CONFIG['hidden_dim'], note_dim=EXP_CONFIG['note_dim'],
        nhead=4, num_virtual_tokens=4
    )
    model_cross, hist_cross = train_model(
        model_cross, X_seq[train_idx], Y[train_idx], X_static[train_idx], X_mh[train_idx], X_note[train_idx],
        X_seq_val=X_seq[val_idx], Y_val=Y[val_idx], X_static_val=X_static[val_idx], X_mh_val=X_mh[val_idx], X_note_val=X_note[val_idx],
        epochs=EXP_CONFIG['epochs'], lr=EXP_CONFIG['lr'], batch_size=EXP_CONFIG['batch_size'],
        pos_weight=pos_weight_val)
    all_epoch_histories['crossmodal_attention'] = hist_cross
    eval_cross = evaluate_model(model_cross, X_seq[test_idx], Y[test_idx], X_static[test_idx], X_mh[test_idx], X_note[test_idx])
    _log_eval_result('CrossModal Attention', eval_cross)
    torch.save(model_cross.state_dict(), os.path.join(exp_dir, 'model_cross_attn.pt'))
    logging.info(f"Cross-Modal Attention model saved to {exp_dir}/model_cross_attn.pt")
    
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

    # ── Collect all evaluation results ──────────────────────────────────────────
    all_eval_results = {
        'lstm_base': eval_base,
        'lstm_latefusion_notes': eval_notes,
        'transformer_earlyfusion_notes': eval_tf,
        'crossmodal_attention': eval_cross,
        'xgb_base': eval_xgb_base,
        'xgb_notes': eval_xgb_notes,
    }
    # Remove non-serializable y_prob before saving
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
        ('LSTM Base', eval_base), ('LSTM LateFusion', eval_notes),
        ('Transformer EarlyFusion', eval_tf), ('CrossModal Attention', eval_cross),
    ]:
        m = _m(result)
        logging.info(
            f"  {name:30s}  t={m['threshold']:.3f}  AUROC={m['roc_auc']:.4f}  PRAUC={m['pr_auc']:.4f}  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  Brier={m['brier']:.4f}"
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
            f"{mb['precision']:.4f} | {mb['recall']:.4f} | {mb['f1']:.4f} | {mb['brier']:.4f} |"
        )
    
    def _note_row_05(name, result):
        m5 = _m(result, 'metrics_at_0.5')
        return (
            f"| {name} | 0.500 | {m5['roc_auc']:.4f} | {m5['pr_auc']:.4f} | "
            f"{m5['precision']:.4f} | {m5['recall']:.4f} | {m5['f1']:.4f} | {m5['brier']:.4f} |"
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

## Model Hyperparameters
| Param | Value |
|-------|-------|
| epochs     | {EXP_CONFIG['epochs']} |
| lr         | {EXP_CONFIG['lr']} |
| batch_size | {EXP_CONFIG['batch_size']} |
| hidden_dim | {EXP_CONFIG['hidden_dim']} |
| note_dim   | {EXP_CONFIG['note_dim']} |
| top_k_codes | {EXP_CONFIG['top_k_codes']} |
| xgb_n_estimators | {EXP_CONFIG['xgb_n_est']} |
| xgb_max_depth    | {EXP_CONFIG['xgb_depth']} |
| early_stop_patience | 15 |
| early_stop_min_delta | 1e-4 |

## Optimizations Applied
| # | Optimization | Enabled |
|---|---|---|
| ① | Sequence StandardScaler (fit on train only) | {'✅' if EXP_CONFIG['opt_seq_norm'] else '❌'} |
| ② | CosineAnnealingLR scheduler (eta_min = lr×0.01) | {'✅' if EXP_CONFIG['opt_cosine_lr'] else '❌'} |
| ③ | BCEWithLogitsLoss pos_weight / XGB scale_pos_weight | {'✅' if EXP_CONFIG['opt_pos_weight'] else '❌'} |
| ④ | Early Stopping (val_loss, patience=15) | ✅ |
| ⑤ | Best F1 threshold search | ✅ |

## Ablation Study Results — Test Set @ Best F1 Threshold
| Model | Threshold | AUROC | PRAUC | Precision | Recall | F1 | Brier |
|-------|-----------|-------|-------|-----------|--------|-----|-------|
{_note_row('XGBoost Base', eval_xgb_base)}
{_note_row('XGBoost + LLM Notes', eval_xgb_notes)}
{_note_row('LSTM Base', eval_base)}
{_note_row('LSTM LateFusion (+ Notes)', eval_notes)}
{_note_row('Transformer EarlyFusion (+ Notes)', eval_tf)}
{_note_row('**CrossModal Attn**', eval_cross)}

## Ablation Study Results — Test Set @ Fixed Threshold 0.5
| Model | Threshold | AUROC | PRAUC | Precision | Recall | F1 | Brier |
|-------|-----------|-------|-------|-----------|--------|-----|-------|
{_note_row_05('XGBoost Base', eval_xgb_base)}
{_note_row_05('XGBoost + LLM Notes', eval_xgb_notes)}
{_note_row_05('LSTM Base', eval_base)}
{_note_row_05('LSTM LateFusion (+ Notes)', eval_notes)}
{_note_row_05('Transformer EarlyFusion (+ Notes)', eval_tf)}
{_note_row_05('**CrossModal Attn**', eval_cross)}

### Note Embedding Impact (AUROC)
- XGBoost: notes Δ AUC = {_m(eval_xgb_notes)['roc_auc'] - _m(eval_xgb_base)['roc_auc']:+.4f}
- LSTM Late Fusion:       notes Δ AUC = {_m(eval_notes)['roc_auc'] - _m(eval_base)['roc_auc']:+.4f}
- CrossModal Attn Fusion: vs LSTM Base Δ AUC = {_m(eval_cross)['roc_auc'] - _m(eval_base)['roc_auc']:+.4f}

## Saved Files
| File | Description |
|------|-------------|
| `model_lstm_base.pt`   | Base LSTM state_dict |
| `model_lstm_notes.pt`  | Late Fusion LSTM state_dict |
| `model_tf_notes.pt`    | Early Fusion Transformer state_dict |
| `model_cross_attn.pt`  | CrossModal Attention Fusion state_dict |
| `model_xgb_base.json`  | XGBoost Base (XGBoost native format) |
| `model_xgb_notes.json` | XGBoost + Notes (XGBoost native format) |
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
