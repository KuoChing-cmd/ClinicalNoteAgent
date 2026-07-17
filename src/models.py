import math

import torch
import torch.nn as nn


# ---------------------------------------------------------
# PyTorch Model Architecture
# ---------------------------------------------------------
class LSTMLateFusionWithNotes(nn.Module):
    """
    LSTM-based late fusion architecture for ICU predictive tasks.

    This model integrates sequential vital signs via LSTM, static demographic features
    via embeddings, sparse multi-hot clinical codes via embedding sum-pooling, and
    pre-computed clinical note embeddings. It employs a late-fusion strategy where
    representations from all modalities are concatenated before the final classification head.

    Args:
        seq_dim (int): Dimensionality of the input sequential features (e.g., vital signs).
        static_dims (dict, optional): Dictionary mapping static feature names to their vocabulary sizes.
        multihot_dims (dict, optional): Dictionary mapping multi-hot feature names to their vocabulary sizes.
        hidden_dim (int): Hidden dimension size for the LSTM and subsequent fusion layers. Defaults to 64.
        note_dim (int): Dimensionality of the input clinical note embeddings. Defaults to 4096.
        use_notes (bool): Whether to include clinical notes in the fusion. Defaults to True.
        num_layers (int): Number of stacked LSTM layers. Defaults to 4.
        dropout (float): Dropout probability applied across layers. Defaults to 0.2.
    """
    def __init__(
        self,
        seq_dim,
        static_dims=None,
        multihot_dims=None,
        hidden_dim=64,
        note_dim=4096,
        use_notes=True,
        num_layers=4,
        dropout=0.2,
    ):
        super().__init__()
        self.use_notes = use_notes
        self.static_dims = static_dims or {}
        self.multihot_dims = multihot_dims or {}

        # 1. Sequence processing
        self.lstm = nn.LSTM(
            input_size=seq_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.attn = nn.Linear(hidden_dim, 1)

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

            self.mh_head = nn.Sequential(nn.Linear(mh_repr_dim, 32), nn.ReLU())
            fused_dim += 32

        # 4. Note processing
        if self.use_notes:
            self.note_head = nn.Sequential(
                nn.LayerNorm(note_dim),
                nn.Linear(note_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            fused_dim += 64

        # 5. Final Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(0.3),
            nn.Linear(fused_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def _multihot_to_embedding(
        self, x_group: torch.Tensor, emb: nn.Embedding
    ) -> torch.Tensor:
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
                mh_embs.append(
                    self._multihot_to_embedding(group, self.mh_emb_dict[name])
                )
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
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1)
        x = x + self.pe[: x.size(0)]
        x = self.dropout(x)
        return x.transpose(0, 1)


class TransformerEarlyFusionWithNotes(nn.Module):
    def __init__(
        self,
        seq_dim,
        static_dims=None,
        multihot_dims=None,
        hidden_dim=64,
        note_dim=4096,
        use_notes=True,
        num_layers=4,
        nhead=8,
        dropout=0.2,
        static_mode="concat",
    ):
        super().__init__()
        self.use_notes = use_notes
        self.static_dims = static_dims or {}
        self.multihot_dims = multihot_dims or {}
        self.static_mode = static_mode

        # 1. Sequence processing (Transformer)
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        fused_dim = hidden_dim  # We'll just use CLS token output

        # 2. Static processing (Demographics)
        self.has_static = len(self.static_dims) > 0
        if self.has_static:
            if self.static_mode == "multi_token":
                self.static_heads = nn.ModuleDict()
                for name, vocab_size in self.static_dims.items():
                    if vocab_size <= 0:
                        self.static_heads[name] = nn.Sequential(
                            nn.Linear(1, hidden_dim), nn.ReLU()
                        )
                    else:
                        self.static_heads[name] = nn.Embedding(vocab_size, hidden_dim)
            else:
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
                    nn.Linear(static_repr_dim, hidden_dim), nn.ReLU()
                )

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
                nn.Dropout(0.3),
            )

        # 5. Final Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(0.3),
            nn.Linear(fused_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def _multihot_to_embedding(
        self, x_group: torch.Tensor, emb: nn.Embedding
    ) -> torch.Tensor:
        summed = x_group @ emb.weight
        denom = torch.clamp(x_group.sum(dim=1, keepdim=True), min=1.0)
        return summed / denom

    def forward(self, x_seq, x_static=None, x_mh=None, x_note=None):
        B = x_seq.shape[0]
        x = self.seq_proj(x_seq)  # [B, T, H]

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
                mh_embs.append(
                    self._multihot_to_embedding(group, self.mh_emb_dict[name])
                )
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

    def __init__(
        self,
        seq_dim,
        static_dims=None,
        multihot_dims=None,
        hidden_dim=64,
        note_dim=4096,
        nhead=8,
        num_virtual_tokens=4,
        num_lstm_layers=4,
        dropout=0.2,
    ):
        super().__init__()
        self.static_dims = static_dims or {}
        self.multihot_dims = multihot_dims or {}
        self.hidden_dim = hidden_dim
        self.M = num_virtual_tokens

        # ── 1. Physiological sequence encoder ────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=seq_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # ── 2. Note → M virtual tokens ────────────────────────────────────────
        # Gradual compression avoids information bottleneck at 4096 → hidden_dim
        self.note_proj = nn.Sequential(
            nn.LayerNorm(note_dim),
            nn.Linear(note_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 4, hidden_dim * num_virtual_tokens),
        )
        # note_proj output will be reshaped to [B, M, hidden_dim]

        # ── 3. Cross-Attention block (seq queries ← note keys/values) ─────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)  # post-attention norm
        self.cross_ff = nn.Sequential(  # position-wise FFN
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
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
            self.emb_dict = nn.ModuleDict()
            static_repr_dim = 0
            for name, vocab_size in self.static_dims.items():
                if self.static_dims[name] == 0:
                    continue
                emb_dim = max(4, min(16, vocab_size // 2))
                self.emb_dict[name] = nn.Embedding(vocab_size, emb_dim)
                static_repr_dim += emb_dim
            static_repr_dim += sum(1 for v in self.static_dims.values() if v == 0)
            self.static_head = nn.Sequential(nn.Linear(static_repr_dim, 32), nn.ReLU())
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
            self.mh_head = nn.Sequential(nn.Linear(mh_repr_dim, 32), nn.ReLU())
            fused_dim += 32

        # ── 7. Classification head ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(0.3),
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    # ── helpers ──────────────────────────────────────────────────────────────
    def _multihot_to_embedding(self, x_group, emb):
        summed = x_group @ emb.weight
        denom = torch.clamp(x_group.sum(dim=1, keepdim=True), min=1.0)
        return summed / denom

    def _encode_static(self, x_static):
        static_embs, col_idx = [], 0
        for name, _ in self.static_dims.items():
            val = x_static[:, col_idx]
            if self.static_dims[name] == 0:
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
    def forward(self, x_seq, x_static=None, x_mh=None, x_note=None, return_attn=False):
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
            note_tokens = self.note_proj(x_note_norm)  # [B, M*d]
            note_tokens = note_tokens.view(
                x_note.shape[0], self.M, self.hidden_dim  # [B, M, d]
            )
        else:
            # fallback: zero note tokens (model still works without notes)
            note_tokens = torch.zeros(
                H.shape[0], self.M, self.hidden_dim, device=H.device
            )

        # 3. Cross-Attention  Q=H, K=V=note_tokens
        attn_out, attn_weights = self.cross_attn(
            query=H, key=note_tokens, value=note_tokens
        )  # [B, T, d]
        H = self.cross_norm(H + attn_out)  # residual
        H = self.cross_ff_norm(H + self.cross_ff(H))  # FFN + residual

        # 4. Temporal self-attention pooling  → seq_repr [B, d]
        temp_w = torch.softmax(self.temporal_attn(H).squeeze(-1), dim=1)  # [B, T]
        seq_repr = (H * temp_w.unsqueeze(-1)).sum(dim=1)  # [B, d]

        reprs = [seq_repr]

        # 5. Demographics
        if self.has_static and x_static is not None:
            reprs.append(self._encode_static(x_static))

        # 6. Sparse codes
        if self.has_multihot and x_mh is not None:
            reprs.append(self._encode_multihot(x_mh))

        fused = torch.cat(reprs, dim=1) if len(reprs) > 1 else reprs[0]
        logits = self.classifier(fused).squeeze(-1)

        if return_attn:
            return logits, attn_weights  # attn_weights: [B, T, M]
        return logits


class GatedFusionWithNotes(nn.Module):
    """
    Gated Fusion for readmission prediction.

    Instead of naively concatenating LSTM and note representations, a
    learnable gate decides how much to trust each modality:

        g = σ(W_h · h_lstm + W_e · f(e_llm) + b)
        x_fused = g ⊙ h_lstm + (1-g) ⊙ f(e_llm)

    When physiological signals already strongly indicate the outcome,
    the gate drives g→1 and suppresses note influence, preventing
    noisy or redundant text features from hurting performance.
    """

    def __init__(
        self,
        seq_dim,
        static_dims=None,
        multihot_dims=None,
        hidden_dim=64,
        note_dim=4096,
        num_lstm_layers=4,
        dropout=0.2,
    ):
        super().__init__()
        self.static_dims = static_dims or {}
        self.multihot_dims = multihot_dims or {}
        self.hidden_dim = hidden_dim

        # ── 1. Physiological sequence encoder ────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=seq_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden_dim, 1)

        # ── 2. Note projection → same dim as LSTM output ─────────────────────
        self.note_proj = nn.Sequential(
            nn.LayerNorm(note_dim),
            nn.Linear(note_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # ── 3. Gating mechanism ──────────────────────────────────────────────
        # g = σ(W_h · h + W_e · e + b)
        self.gate_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_e = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(hidden_dim))

        fused_dim = hidden_dim

        # ── 4. Static (demographics) ─────────────────────────────────────────
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
            self.static_head = nn.Sequential(nn.Linear(static_repr_dim, 32), nn.ReLU())
            fused_dim += 32

        # ── 5. Sparse multi-hot (ICD / DRG / Proc / Rx) ──────────────────────
        self.has_multihot = len(self.multihot_dims) > 0
        if self.has_multihot:
            self.mh_emb_dict = nn.ModuleDict()
            mh_repr_dim = 0
            for name, vocab_size in self.multihot_dims.items():
                emb_dim = max(8, min(32, vocab_size // 4))
                self.mh_emb_dict[name] = nn.Embedding(vocab_size, emb_dim)
                mh_repr_dim += emb_dim
            self.mh_head = nn.Sequential(nn.Linear(mh_repr_dim, 32), nn.ReLU())
            fused_dim += 32

        # ── 6. Classification head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(0.3),
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def _multihot_to_embedding(self, x_group, emb):
        summed = x_group @ emb.weight
        denom = torch.clamp(x_group.sum(dim=1, keepdim=True), min=1.0)
        return summed / denom

    def forward(self, x_seq, x_static=None, x_mh=None, x_note=None, return_gate=False):
        import torch.nn.functional as F

        # 1. LSTM → temporal attention pooling → h [B, hidden_dim]
        lstm_out, _ = self.lstm(x_seq)
        attn_w = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        h = (lstm_out * attn_w.unsqueeze(-1)).sum(dim=1)  # [B, d]

        # 2. Note projection → e [B, hidden_dim]
        if x_note is not None:
            e = self.note_proj(F.normalize(x_note, p=2, dim=1))  # [B, d]
        else:
            e = torch.zeros_like(h)

        # 3. Gated fusion: g = σ(W_h·h + W_e·e + b)
        g = torch.sigmoid(self.gate_h(h) + self.gate_e(e) + self.gate_bias)  # [B, d]
        fused_repr = g * h + (1.0 - g) * e  # [B, d]

        reprs = [fused_repr]

        # 4. Demographics
        if self.has_static and x_static is not None:
            static_embs, col_idx = [], 0
            for name, _ in self.static_dims.items():
                val = x_static[:, col_idx]
                if self.static_dims[name] == 0:
                    static_embs.append(val.unsqueeze(1).float())
                else:
                    static_embs.append(self.emb_dict[name](val.long()))
                col_idx += 1
            reprs.append(self.static_head(torch.cat(static_embs, dim=1)))

        # 5. Sparse codes
        if self.has_multihot and x_mh is not None:
            mh_embs, col_offset = [], 0
            for name, vocab_size in self.multihot_dims.items():
                group = x_mh[:, col_offset : col_offset + vocab_size]
                mh_embs.append(
                    self._multihot_to_embedding(group, self.mh_emb_dict[name])
                )
                col_offset += vocab_size
            reprs.append(self.mh_head(torch.cat(mh_embs, dim=1)))

        fused = torch.cat(reprs, dim=1) if len(reprs) > 1 else reprs[0]
        logits = self.classifier(fused).squeeze(-1)

        if return_gate:
            return logits, g.mean(dim=0)  # avg gate per hidden dim for analysis
        return logits


class TransformerSeqEncoder(nn.Module):
    def __init__(self, seq_input_dim, hidden_dim, num_layers=4, nhead=8, dropout=0.2):
        super().__init__()
        self.seq_proj = nn.Linear(seq_input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x_seq, mask_indices=None):
        x = self.seq_proj(x_seq)  # [B, T, H]
        if mask_indices is not None:
            # mask_indices: [B, T] boolean tensor
            expanded_mask = mask_indices.unsqueeze(-1).expand_as(x)
            x = torch.where(expanded_mask, self.mask_token, x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        return out


class TransformerPretrainer(nn.Module):
    def __init__(self, encoder, seq_input_dim, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, seq_input_dim),
        )

    def forward(self, x_seq, mask_indices):
        encoded_seq = self.encoder(x_seq, mask_indices)
        return self.reconstruction_head(encoded_seq)


class PretrainedTransformerCrossModalFusion(nn.Module):
    def __init__(
        self,
        encoder,
        static_dims=None,
        multihot_dims=None,
        hidden_dim=64,
        note_dim=4096,
        nhead=8,
        num_virtual_tokens=4,
        dropout=0.2,
    ):
        super().__init__()
        self.encoder = encoder
        self.static_dims = static_dims or {}
        self.multihot_dims = multihot_dims or {}
        self.hidden_dim = hidden_dim
        self.M = num_virtual_tokens

        self.note_proj = nn.Sequential(
            nn.LayerNorm(note_dim),
            nn.Linear(note_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 4, hidden_dim * num_virtual_tokens),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.cross_ff_norm = nn.LayerNorm(hidden_dim)

        self.temporal_attn = nn.Linear(hidden_dim, 1)

        fused_dim = hidden_dim

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
            self.static_head = nn.Sequential(nn.Linear(static_repr_dim, 32), nn.ReLU())
            fused_dim += 32

        self.has_multihot = len(self.multihot_dims) > 0
        if self.has_multihot:
            self.mh_emb_dict = nn.ModuleDict()
            mh_repr_dim = 0
            for name, vocab_size in self.multihot_dims.items():
                emb_dim = max(8, min(32, vocab_size // 4))
                self.mh_emb_dict[name] = nn.Embedding(vocab_size, emb_dim)
                mh_repr_dim += emb_dim
            self.mh_head = nn.Sequential(nn.Linear(mh_repr_dim, 32), nn.ReLU())
            fused_dim += 32

        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(0.3),
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def _multihot_to_embedding(self, x_group, emb):
        summed = x_group @ emb.weight
        denom = torch.clamp(x_group.sum(dim=1, keepdim=True), min=1.0)
        return summed / denom

    def _encode_static(self, x_static):
        static_embs, col_idx = [], 0
        for name, _ in self.static_dims.items():
            val = x_static[:, col_idx]
            if self.static_dims[name] == 0:
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

    def forward(self, x_seq, x_static=None, x_mh=None, x_note=None):
        import torch.nn.functional as F

        # 1. Transformer encode sequence  → H [B, T, d]
        H = self.encoder(x_seq)

        # 2. Project note → M virtual tokens [B, M, d]
        if x_note is not None:
            x_note_norm = F.normalize(x_note, p=2, dim=1)
            note_tokens = self.note_proj(x_note_norm)  # [B, M*d]
            note_tokens = note_tokens.view(
                x_note.shape[0], self.M, self.hidden_dim  # [B, M, d]
            )
        else:
            note_tokens = torch.zeros(
                H.shape[0], self.M, self.hidden_dim, device=H.device
            )

        # 3. Cross-Attention  Q=H, K=V=note_tokens
        attn_out, _ = self.cross_attn(
            query=H, key=note_tokens, value=note_tokens
        )  # [B, T, d]
        H = self.cross_norm(H + attn_out)  # residual
        H = self.cross_ff_norm(H + self.cross_ff(H))  # FFN + residual

        # 4. Temporal self-attention pooling  → seq_repr [B, d]
        temp_w = torch.softmax(self.temporal_attn(H).squeeze(-1), dim=1)  # [B, T]
        seq_repr = (H * temp_w.unsqueeze(-1)).sum(dim=1)  # [B, d]

        reprs = [seq_repr]

        if self.has_static and x_static is not None:
            reprs.append(self._encode_static(x_static))

        if self.has_multihot and x_mh is not None:
            reprs.append(self._encode_multihot(x_mh))

        fused = torch.cat(reprs, dim=1) if len(reprs) > 1 else reprs[0]
        logits = self.classifier(fused).squeeze(-1)

        return logits
