# Edit these values to configure a run. The exp dir name is auto-generated.
EXP_CONFIG = {
    "epochs":      200,
    "lr":          5e-4,         # ↓ from 1e-3; smaller LR to reduce overfitting / allow longer convergence
    "batch_size":  256,
    "hidden_dim":  128,
    "num_layers":  4,            # for LSTM
    "tf_num_layers": 2,          # ↓ from 4; reduce Transformer depth
    "tf_nhead":    4,            # ↓ from 8; reduce Transformer heads
    "dropout":     0.3,          # ↑ from 0.2; add more regularization globally
    "note_dim":    None,         # auto-detected from embeddings (768 for ClinicalBERT, 4096 for Llama)
    "top_k_codes": 64,           # top-K for ICD/DRG/Proc/Rx multi-hot
    "xgb_n_est":   200,
    "xgb_depth":   6,
    "lgb_n_est":   200,
    "lgb_depth":   6,
    "early_stop_patience": 30,   # ↑ from 15; give model more epochs to find better minimum
    # Note embedding selection
    "note_embedding_type": "clinicalbert",  # "clinicalbert" (768d) or "llama" (4096d)
    # Optimizations enabled in this run
    "opt_seq_norm":    True,     # ① StandardScaler on sequence features
    "opt_cosine_lr":   True,     # ② CosineAnnealingLR scheduler
    "opt_pos_weight":  True,     # ③ pos_weight for class imbalance
    "static_mode":     "multi_token", # 开启细粒度多 Token 模式
}

