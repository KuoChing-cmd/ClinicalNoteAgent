import os
import sys
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pytest
import torch

from scripts.training.data import load_note_texts
from scripts.training.models import LSTMEndToEndWithNotes, TransformerEndToEndWithNotes

try:
    from transformers import BatchEncoding as _BatchEncoding
    BatchEncodingType: type[Any] | None = _BatchEncoding
except Exception:  # pragma: no cover
    BatchEncodingType = None


def test_load_note_texts_extracts_text_from_summary_dict():
    payload = {
        101: {
            'meta_summary': {
                'final_summary': 'Summary text',
                'critical_factors_list': ['factor A', 'factor B'],
            },
            'category_summaries': {
                'history': 'history desc',
                'exam': 'exam desc',
            },
        }
    }
    texts = load_note_texts(payload)
    assert texts[101].startswith('Summary text')
    assert 'factor A' in texts[101]
    assert 'history desc' in texts[101]


def test_e2e_models_run_forward_pass():
    seq = torch.randn(2, 48, 8)
    static = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1], [1, 0, 1, 1, 0, 1, 1, 0]], dtype=torch.float32)
    mh = torch.randn(2, 64)
    tok = {
        'input_ids': torch.randint(0, 100, (2, 32)),
        'attention_mask': torch.ones(2, 32, dtype=torch.long),
    }

    if BatchEncodingType is not None:
        tok = BatchEncodingType(tok)

    model_lstm = LSTMEndToEndWithNotes(seq_dim=8, static_dims={"age": 0, "gender": 2, "unit": 3}, multihot_dims={'icd': 64}, note_dim=768, hidden_dim=16, num_layers=1, dropout=0.0)
    out_lstm = model_lstm(seq, static, mh, x_note_tokens=tok)
    assert out_lstm.shape == (2,)

    model_tf = TransformerEndToEndWithNotes(seq_dim=8, static_dims={"age": 0, "gender": 2, "unit": 3}, multihot_dims={'icd': 64}, note_dim=768, hidden_dim=16, num_layers=1, nhead=4, dropout=0.0)
    out_tf = model_tf(seq, static, mh, x_note_tokens=tok)
    assert out_tf.shape == (2,)


def test_fusion_models_with_variable_multihot_dims():
    from scripts.training.models import TransformerLateFusionWithNotes, TransformerEarlyFusionWithNotes, LSTMLateFusionWithNotes

    B = 2
    seq = torch.randn(B, 24, 8)
    static = torch.tensor([[0, 1, 2], [1, 0, 1]], dtype=torch.float32)
    static_dims = {"age": 0, "gender": 2, "unit": 3}
    multihot_dims = {'icd': 32, 'proc': 64, 'rx': 64}
    mh = torch.randn(B, 32 + 64 + 64)
    note = torch.randn(B, 768)

    model_tf_late = TransformerLateFusionWithNotes(seq_dim=8, static_dims=static_dims, multihot_dims=multihot_dims, note_dim=768, hidden_dim=16, num_layers=1, nhead=4, dropout=0.0)
    out_tf_late = model_tf_late(seq, static, mh, note)
    assert out_tf_late.shape == (B,)

    model_tf_early = TransformerEarlyFusionWithNotes(seq_dim=8, static_dims=static_dims, multihot_dims=multihot_dims, note_dim=768, hidden_dim=16, num_layers=1, nhead=4, dropout=0.0)
    out_tf_early = model_tf_early(seq, static, mh, note)
    assert out_tf_early.shape == (B,)

    model_lstm = LSTMLateFusionWithNotes(seq_dim=8, static_dims=static_dims, multihot_dims=multihot_dims, note_dim=768, hidden_dim=16, num_layers=1, dropout=0.0)
    out_lstm = model_lstm(seq, static, mh, note)
    assert out_lstm.shape == (B,)
