import os
import pickle
import logging
from datetime import datetime
from collections.abc import Mapping
import json
import numpy as np
import pandas as pd
import duckdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb
import math

try:
    from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel
except ImportError:  # pragma: no cover
    AutoModel = AutoTokenizer = BertConfig = BertModel = None

# ---------------------------------------------------------
# PyTorch Model Architecture
# ---------------------------------------------------------
class LSTMLateFusionWithNotes(nn.Module):
    def __init__(self, seq_dim, static_dims=None, multihot_dims=None, hidden_dim=64, note_dim=4096, use_notes=True, num_layers=4, dropout=0.2):
        super().__init__()
        self.use_notes = use_notes
        self.static_dims = static_dims or {}
        self.multihot_dims = multihot_dims or {}
        
        # 1. Sequence processing
        self.lstm = nn.LSTM(input_size=seq_dim, hidden_size=hidden_dim, batch_first=True, dropout=dropout, num_layers=num_layers, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        
        fused_dim = hidden_dim * 2
        
        # 2. Static processing (Demographics)
        self.has_static = len(self.static_dims) > 0
        if self.has_static:
            self.emb_dict = nn.ModuleDict()
            static_repr_dim = 0
            for name, vocab_size in self.static_dims.items():
                if self.static_dims[name] == 0:  
                    continue
                emb_dim = max(4, min(16, vocab_size // 2))
                self.emb_dict[name] = nn.Embedding(vocab_size, emb_dim)
                static_repr_dim += emb_dim
                
            static_repr_dim += sum(1 for v in self.static_dims.values() if v == 0)
                
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
                if self.static_dims[name] == 0:
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

class LSTMEndToEndWithNotes(LSTMLateFusionWithNotes):
    def __init__(self, seq_dim, static_dims=None, multihot_dims=None, hidden_dim=64, note_dim=768, use_notes=True, num_layers=4, dropout=0.2,
                 bert_model_name='emilyalsentzer/Bio_ClinicalBERT', freeze_layers=8, bert_lr_scale=0.1):
        super().__init__(seq_dim, static_dims=static_dims, multihot_dims=multihot_dims, hidden_dim=hidden_dim, note_dim=note_dim, use_notes=False, num_layers=num_layers, dropout=dropout)
        self.use_notes = use_notes
        self.bert_model_name = bert_model_name
        self.freeze_layers = freeze_layers
        self.bert_lr_scale = bert_lr_scale
        self.max_seq_len = 512
        self.bert_model, self.tokenizer = self._load_bert(bert_model_name)
        self.bert_dim = getattr(self.bert_model.config, 'hidden_size', note_dim)
        self._fused_dim = hidden_dim * 2 + 32 + 32
        self.note_head = nn.Sequential(
            nn.LayerNorm(self.bert_dim),
            nn.Linear(self.bert_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self._fused_dim + 64),
            nn.Dropout(0.3),
            nn.Linear(self._fused_dim + 64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def _load_bert(self, model_name):
        if AutoModel is None or AutoTokenizer is None:
            config = BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=2, num_attention_heads=8, intermediate_size=3072)
            model = BertModel(config)
            tokenizer = None
            return model, tokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
        except Exception:
            config = BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=2, num_attention_heads=8, intermediate_size=3072)
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') if AutoTokenizer is not None else None
            model = BertModel(config)

        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        for idx, layer in enumerate(getattr(model, 'encoder', getattr(model, 'bert', model)).layer[:self.freeze_layers]):
            for param in layer.parameters():
                param.requires_grad = False
        return model, tokenizer

    def _encode_note_tokens(self, x_note_tokens):
        if x_note_tokens is None:
            return None
        if isinstance(x_note_tokens, Mapping):
            input_ids = x_note_tokens.get('input_ids')
            attention_mask = x_note_tokens.get('attention_mask')
            if input_ids is None:
                return None
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            cls = outputs.last_hidden_state[:, 0, :]
        else:
            cls = x_note_tokens
        cls = F.normalize(cls, p=2, dim=1)
        return self.note_head(cls)

    def forward(self, x_seq, x_static=None, x_mh=None, x_note=None, x_note_tokens=None):
        out, _ = self.lstm(x_seq)
        attn_w = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        seq_repr = (out * attn_w.unsqueeze(-1)).sum(dim=1)
        reprs = [seq_repr]

        if self.has_static and x_static is not None:
            static_embs = []
            col_idx = 0
            for name, _ in self.static_dims.items():
                val = x_static[:, col_idx]
                if self.static_dims[name] == 0:
                    static_embs.append(val.unsqueeze(1).float())
                else:
                    static_embs.append(self.emb_dict[name](val.long()))
                col_idx += 1
            reprs.append(self.static_head(torch.cat(static_embs, dim=1)))

        if self.has_multihot and x_mh is not None:
            mh_embs = []
            col_offset = 0
            for name, vocab_size in self.multihot_dims.items():
                group = x_mh[:, col_offset:col_offset + vocab_size]
                mh_embs.append(self._multihot_to_embedding(group, self.mh_emb_dict[name]))
                col_offset += vocab_size
            reprs.append(self.mh_head(torch.cat(mh_embs, dim=1)))

        bert_repr = self._encode_note_tokens(x_note_tokens) if x_note_tokens is not None else None
        if bert_repr is None and x_note is not None:
            if isinstance(x_note, Mapping):
                bert_repr = self._encode_note_tokens(x_note)
            else:
                bert_repr = self.note_head(F.normalize(x_note, p=2, dim=1))
        if bert_repr is not None:
            reprs.append(bert_repr)

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

class TransformerLateFusionWithNotes(nn.Module):
    def __init__(self, seq_dim, static_dims=None, multihot_dims=None, hidden_dim=64, note_dim=4096, use_notes=True, num_layers=4, nhead=8, dropout=0.2):
        super().__init__()
        self.use_notes = use_notes
        self.static_dims = static_dims or {}
        self.multihot_dims = multihot_dims or {}
        
        # 1. Sequence processing (Transformer)
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        fused_dim = hidden_dim
        
        # 2. Static processing (Demographics)
        self.has_static = len(self.static_dims) > 0
        if self.has_static:
            self.emb_dict = nn.ModuleDict()
            static_repr_dim = 0
            for name, vocab_size in self.static_dims.items():
                if self.static_dims[name] == 0:  
                    continue
                emb_dim = max(4, min(16, vocab_size // 2))
                self.emb_dict[name] = nn.Embedding(vocab_size, emb_dim)
                static_repr_dim += emb_dim
                
            static_repr_dim += sum(1 for v in self.static_dims.values() if v == 0)
                
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
        B = x_seq.shape[0]
        x = self.seq_proj(x_seq) # [B, T, H]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # [B, T+1, H]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        seq_repr = x[:, 0, :] # Extract CLS token representation
        
        reprs = [seq_repr]
        
        if self.has_static and x_static is not None:
            static_embs = []
            col_idx = 0
            for name, _ in self.static_dims.items():
                val = x_static[:, col_idx]
                if self.static_dims[name] == 0:
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
                mh_embs.append(self._multihot_to_embedding(group.float(), self.mh_emb_dict[name]))
                col_offset += vocab_size
            reprs.append(self.mh_head(torch.cat(mh_embs, dim=1)))
            
        if self.use_notes and x_note is not None:
            import torch.nn.functional as F
            x_note_norm = F.normalize(x_note, p=2, dim=1)
            reprs.append(self.note_head(x_note_norm))
            
        fused = torch.cat(reprs, dim=1) if len(reprs) > 1 else reprs[0]
        return self.classifier(fused).squeeze(-1)

class TransformerEndToEndWithNotes(TransformerLateFusionWithNotes):
    def __init__(self, seq_dim, static_dims=None, multihot_dims=None, hidden_dim=64, note_dim=768, use_notes=True, num_layers=4, nhead=8, dropout=0.2,
                 bert_model_name='emilyalsentzer/Bio_ClinicalBERT', freeze_layers=8, bert_lr_scale=0.1):
        super().__init__(seq_dim, static_dims=static_dims, multihot_dims=multihot_dims, hidden_dim=hidden_dim, note_dim=note_dim, use_notes=False, num_layers=num_layers, nhead=nhead, dropout=dropout)
        self.use_notes = use_notes
        self.bert_model_name = bert_model_name
        self.freeze_layers = freeze_layers
        self.bert_lr_scale = bert_lr_scale
        self.max_seq_len = 512
        self.bert_model, self.tokenizer = self._load_bert(bert_model_name)
        self.bert_dim = getattr(self.bert_model.config, 'hidden_size', note_dim)
        self._fused_dim = hidden_dim + 32 + 32
        self.note_head = nn.Sequential(
            nn.LayerNorm(self.bert_dim),
            nn.Linear(self.bert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim + 32 + 32 + hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim + 32 + 32 + hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def _load_bert(self, model_name):
        if AutoModel is None or AutoTokenizer is None:
            config = BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=2, num_attention_heads=8, intermediate_size=3072)
            model = BertModel(config)
            tokenizer = None
            return model, tokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
        except Exception:
            config = BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=2, num_attention_heads=8, intermediate_size=3072)
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') if AutoTokenizer is not None else None
            model = BertModel(config)

        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        bert_backbone = getattr(model, 'encoder', getattr(model, 'bert', model))
        if hasattr(bert_backbone, 'layer') and len(bert_backbone.layer) > self.freeze_layers:
            for layer in bert_backbone.layer[:self.freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        return model, tokenizer

    def _encode_note_tokens(self, x_note_tokens):
        if x_note_tokens is None:
            return None
        if isinstance(x_note_tokens, Mapping):
            input_ids = x_note_tokens.get('input_ids')
            attention_mask = x_note_tokens.get('attention_mask')
            if input_ids is None:
                return None
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            cls = outputs.last_hidden_state[:, 0, :]
        else:
            cls = x_note_tokens
        cls = F.normalize(cls, p=2, dim=1)
        return self.note_head(cls)

    def forward(self, x_seq, x_static=None, x_mh=None, x_note=None, x_note_tokens=None):
        B = x_seq.shape[0]
        x = self.seq_proj(x_seq)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        seq_repr = x[:, 0, :]

        reprs = [seq_repr]

        if self.has_static and x_static is not None:
            static_embs = []
            col_idx = 0
            for name, _ in self.static_dims.items():
                val = x_static[:, col_idx]
                if self.static_dims[name] == 0:
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
                mh_embs.append(self._multihot_to_embedding(group.float(), self.mh_emb_dict[name]))
                col_offset += vocab_size
            reprs.append(self.mh_head(torch.cat(mh_embs, dim=1)))

        bert_repr = self._encode_note_tokens(x_note_tokens) if x_note_tokens is not None else None
        if bert_repr is None and x_note is not None:
            if isinstance(x_note, Mapping):
                bert_repr = self._encode_note_tokens(x_note)
            else:
                bert_repr = self.note_head(F.normalize(x_note, p=2, dim=1))
        if bert_repr is not None:
            reprs.append(bert_repr)

        fused = torch.cat(reprs, dim=1)
        return self.classifier(fused).squeeze(-1)

class TransformerEarlyFusionWithNotes(nn.Module):
    def __init__(self, seq_dim, static_dims=None, multihot_dims=None, hidden_dim=64, note_dim=4096, use_notes=True, num_layers=4, nhead=8, dropout=0.2, static_mode="concat"):
        super().__init__()
        self.use_notes = use_notes
        self.static_dims = static_dims or {}
        self.multihot_dims = multihot_dims or {}
        self.static_mode = static_mode
        
        # 1. Sequence processing (Transformer)
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        fused_dim = hidden_dim # We'll just use CLS token output
        
        # 2. Static processing (Demographics)
        self.has_static = len(self.static_dims) > 0
        if self.has_static:
            if self.static_mode == "multi_token":
                self.static_heads = nn.ModuleDict()
                for name, vocab_size in self.static_dims.items():
                    if vocab_size <= 0:
                        self.static_heads[name] = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())
                    else:
                        self.static_heads[name] = nn.Embedding(vocab_size, hidden_dim)
            else:
                self.emb_dict = nn.ModuleDict()
                static_repr_dim = 0
                for name, vocab_size in self.static_dims.items():
                    if self.static_dims[name] == 0: continue
                    emb_dim = max(4, min(16, vocab_size // 2))
                    self.emb_dict[name] = nn.Embedding(vocab_size, emb_dim)
                    static_repr_dim += emb_dim
                static_repr_dim += sum(1 for v in self.static_dims.values() if v == 0)
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
            if self.static_mode == "multi_token":
                static_tokens = []
                col_idx = 0
                for name, vocab_size in self.static_dims.items():
                    val = x_static[:, col_idx]
                    if vocab_size <= 0:
                        token = self.static_heads[name](val.unsqueeze(1).float())
                    else:
                        token = self.static_heads[name](val.long())
                    static_tokens.append(token.unsqueeze(1))
                    col_idx += 1
                extra_tokens.extend(static_tokens)
            else:
                static_embs = []
                col_idx = 0
                for name, _ in self.static_dims.items():
                    val = x_static[:, col_idx]
                    if self.static_dims[name] == 0:
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
