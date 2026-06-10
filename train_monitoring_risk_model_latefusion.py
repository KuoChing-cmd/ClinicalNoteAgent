#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import pickle
import time as pytime
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sqlalchemy import text, bindparam
from sqlalchemy.exc import OperationalError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.database import DatabaseConfig, DatabaseManager
from src.database.mimic4_query import DEFAULT_CHARTEVENT_LABEL_WHITELIST_TOP100, MIMIC4DataExtractor
from src.monitoring.monitoring_agent import MonitoringAgent

logger = logging.getLogger("monitoring_risk_latefusion_train")
DEFAULT_SQL_IN_BATCH_SIZE = 1000


def _auto_embedding_dim(cardinality: int) -> int:
    c = int(max(1, cardinality))
    return int(min(64, max(8, round(c**0.5) * 2)))


def _is_retryable_mysql_error(exc: Exception) -> bool:
    msg = str(getattr(exc, "orig", exc)).lower()
    return (
        "(2013" in msg
        or "(2006" in msg
        or "(2003" in msg
        or "lost connection to mysql server during query" in msg
        or "mysql server has gone away" in msg
        or "can't connect to mysql server" in msg
        or "connection refused" in msg
    )


@dataclass
class StayRow:
    subject_id: int
    hadm_id: int
    stay_id: int
    intime: Any
    outtime: Any
    label: int
    age: float | None = None
    gender: str | None = None
    marital_status: str | None = None
    insurance: str | None = None
    admission_type: str | None = None
    admit_to: str | None = None


def _normalize_cat(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _build_top_categories(values: list[str], top_k: int) -> list[str]:
    if top_k <= 0:
        return []
    cnt: Counter[str] = Counter(v for v in values if v)
    return [k for k, _ in cnt.most_common(top_k)]


def _build_static_feature_vocabs(
    stay_rows: list[StayRow],
    *,
    marital_top_k: int,
    insurance_top_k: int,
    admission_type_top_k: int,
    admit_to_top_k: int,
) -> tuple[list[str], list[str], list[str], list[str]]:
    marital_vals = [_normalize_cat(s.marital_status) for s in stay_rows]
    insurance_vals = [_normalize_cat(s.insurance) for s in stay_rows]
    admission_type_vals = [_normalize_cat(s.admission_type) for s in stay_rows]
    admit_to_vals = [_normalize_cat(s.admit_to) for s in stay_rows]
    return (
        _build_top_categories(marital_vals, top_k=marital_top_k),
        _build_top_categories(insurance_vals, top_k=insurance_top_k),
        _build_top_categories(admission_type_vals, top_k=admission_type_top_k),
        _build_top_categories(admit_to_vals, top_k=admit_to_top_k),
    )


def _build_static_vector(
    row: StayRow,
    *,
    include_demographics: bool,
    include_admission_context: bool,
    marital_vocab: list[str],
    insurance_vocab: list[str],
    admission_type_vocab: list[str],
    admit_to_vocab: list[str],
) -> np.ndarray:
    cols: list[float] = []

    if include_demographics:
        age = float(row.age) if row.age is not None else 65.0
        age = float(np.clip(age, 0.0, 100.0))
        gender = _normalize_cat(row.gender)
        cols.extend(
            [
                age / 100.0,
                1.0 if gender == "M" else 0.0,
                1.0 if gender == "F" else 0.0,
                1.0 if gender not in {"M", "F"} else 0.0,
            ]
        )

    if include_admission_context:
        marital = _normalize_cat(row.marital_status)
        insurance = _normalize_cat(row.insurance)
        admission_type = _normalize_cat(row.admission_type)
        admit_to = _normalize_cat(row.admit_to)

        marital_map = {k: i for i, k in enumerate(marital_vocab)}
        insurance_map = {k: i for i, k in enumerate(insurance_vocab)}
        admission_type_map = {k: i for i, k in enumerate(admission_type_vocab)}
        admit_to_map = {k: i for i, k in enumerate(admit_to_vocab)}

        marital_vec = np.zeros((len(marital_vocab) + 1,), dtype=np.float32)
        insurance_vec = np.zeros((len(insurance_vocab) + 1,), dtype=np.float32)
        admission_type_vec = np.zeros((len(admission_type_vocab) + 1,), dtype=np.float32)
        admit_to_vec = np.zeros((len(admit_to_vocab) + 1,), dtype=np.float32)

        marital_vec[marital_map.get(marital, len(marital_vocab))] = 1.0
        insurance_vec[insurance_map.get(insurance, len(insurance_vocab))] = 1.0
        admission_type_vec[admission_type_map.get(admission_type, len(admission_type_vocab))] = 1.0
        admit_to_vec[admit_to_map.get(admit_to, len(admit_to_vocab))] = 1.0

        cols.extend(marital_vec.tolist())
        cols.extend(insurance_vec.tolist())
        cols.extend(admission_type_vec.tolist())
        cols.extend(admit_to_vec.tolist())

    return np.asarray(cols, dtype=np.float32)


def _icd_prefix(code: Any, width: int = 3) -> str:
    if code is None:
        return ""
    s = str(code).strip().upper().replace(".", "")
    if not s:
        return ""
    return s[:width]


def _drg_code_norm(code: Any) -> str:
    if code is None:
        return ""
    s = str(code).strip().upper()
    if not s:
        return ""
    return s


def _pharmacy_term_norm(term: Any) -> str:
    if term is None:
        return ""
    s = " ".join(str(term).strip().upper().split())
    if not s:
        return ""
    return s


def _build_icd_history_vectors(
    session: Any,
    stay_rows: list[StayRow],
    *,
    top_k: int,
    include_current_hadm: bool,
    sql_in_batch_size: int,
) -> tuple[dict[int, np.ndarray], list[str]]:
    if top_k <= 0 or not stay_rows:
        return {}, []

    stay_ids = [int(s.stay_id) for s in stay_rows]
    hist_sql = text(
        """
        SELECT
            cur.stay_id,
            d.icd_code
        FROM icustays cur
        INNER JOIN admissions a_cur ON a_cur.hadm_id = cur.hadm_id
        INNER JOIN admissions a_hist ON a_hist.subject_id = cur.subject_id
        INNER JOIN diagnoses_icd d ON d.hadm_id = a_hist.hadm_id
        WHERE cur.stay_id IN :stay_ids
          AND d.icd_code IS NOT NULL
          AND (
            a_hist.admittime < a_cur.admittime
            OR (:include_current_hadm = 1 AND a_hist.hadm_id = a_cur.hadm_id)
          )
        """
    ).bindparams(bindparam("stay_ids", expanding=True))

    rows: list[dict[str, Any]] = []
    include_current_hadm_int = 1 if include_current_hadm else 0
    chunk_size = max(1, int(sql_in_batch_size))
    for i in range(0, len(stay_ids), chunk_size):
        batch_stay_ids = stay_ids[i : i + chunk_size]
        batch_rows = session.execute(
            hist_sql,
            {
                "stay_ids": batch_stay_ids,
                "include_current_hadm": include_current_hadm_int,
            },
        ).mappings().all()
        rows.extend(batch_rows)

    freq: dict[str, int] = {}
    by_stay: dict[int, list[str]] = {}
    for r in rows:
        sid = int(r["stay_id"])
        p3 = _icd_prefix(r.get("icd_code"), width=3)
        if not p3:
            continue
        freq[p3] = freq.get(p3, 0) + 1
        by_stay.setdefault(sid, []).append(p3)

    vocab = [k for k, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
    idx = {k: i for i, k in enumerate(vocab)}

    vectors: dict[int, np.ndarray] = {}
    for sid in stay_ids:
        vectors[int(sid)] = np.zeros((len(vocab),), dtype=np.float32)

    for sid, codes in by_stay.items():
        vec = vectors.get(int(sid))
        if vec is None:
            continue
        for c in codes:
            j = idx.get(c)
            if j is not None:
                vec[j] = 1.0

    return vectors, vocab


def _build_drg_vectors(
    session: Any,
    stay_rows: list[StayRow],
    *,
    top_k: int,
    sql_in_batch_size: int,
) -> tuple[dict[int, np.ndarray], list[str]]:
    if top_k <= 0 or not stay_rows:
        return {}, []

    stay_ids = [int(s.stay_id) for s in stay_rows]
    drg_sql = text(
        """
        SELECT
            cur.stay_id,
            d.drg_code
        FROM icustays cur
        INNER JOIN drgcodes d ON d.hadm_id = cur.hadm_id
        WHERE cur.stay_id IN :stay_ids
          AND d.drg_code IS NOT NULL
        """
    ).bindparams(bindparam("stay_ids", expanding=True))

    rows: list[dict[str, Any]] = []
    chunk_size = max(1, int(sql_in_batch_size))
    for i in range(0, len(stay_ids), chunk_size):
        batch_stay_ids = stay_ids[i : i + chunk_size]
        batch_rows = session.execute(
            drg_sql,
            {
                "stay_ids": batch_stay_ids,
            },
        ).mappings().all()
        rows.extend(batch_rows)

    freq: dict[str, int] = {}
    by_stay: dict[int, list[str]] = {}
    for r in rows:
        sid = int(r["stay_id"])
        drg = _drg_code_norm(r.get("drg_code"))
        if not drg:
            continue
        freq[drg] = freq.get(drg, 0) + 1
        by_stay.setdefault(sid, []).append(drg)

    vocab = [k for k, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
    idx = {k: i for i, k in enumerate(vocab)}

    vectors: dict[int, np.ndarray] = {}
    for sid in stay_ids:
        vectors[int(sid)] = np.zeros((len(vocab),), dtype=np.float32)

    for sid, codes in by_stay.items():
        vec = vectors.get(int(sid))
        if vec is None:
            continue
        for c in codes:
            j = idx.get(c)
            if j is not None:
                vec[j] = 1.0

    return vectors, vocab


def _build_procedure_vectors(
    session: Any,
    stay_rows: list[StayRow],
    *,
    top_k: int,
    sql_in_batch_size: int,
) -> tuple[dict[int, np.ndarray], list[str]]:
    if top_k <= 0 or not stay_rows:
        return {}, []

    stay_ids = [int(s.stay_id) for s in stay_rows]
    procedure_sql = text(
        """
        SELECT
            cur.stay_id,
            p.icd_code
        FROM icustays cur
        INNER JOIN procedures_icd p ON p.hadm_id = cur.hadm_id
        WHERE cur.stay_id IN :stay_ids
          AND p.icd_code IS NOT NULL
          AND (p.chartdate IS NULL OR p.chartdate <= cur.outtime)
        """
    ).bindparams(bindparam("stay_ids", expanding=True))

    rows: list[dict[str, Any]] = []
    chunk_size = max(1, int(sql_in_batch_size))
    for i in range(0, len(stay_ids), chunk_size):
        batch_stay_ids = stay_ids[i : i + chunk_size]
        batch_rows = session.execute(
            procedure_sql,
            {
                "stay_ids": batch_stay_ids,
            },
        ).mappings().all()
        rows.extend(batch_rows)

    freq: dict[str, int] = {}
    by_stay: dict[int, list[str]] = {}
    for r in rows:
        sid = int(r["stay_id"])
        p3 = _icd_prefix(r.get("icd_code"), width=3)
        if not p3:
            continue
        freq[p3] = freq.get(p3, 0) + 1
        by_stay.setdefault(sid, []).append(p3)

    vocab = [k for k, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
    idx = {k: i for i, k in enumerate(vocab)}

    vectors: dict[int, np.ndarray] = {}
    for sid in stay_ids:
        vectors[int(sid)] = np.zeros((len(vocab),), dtype=np.float32)

    for sid, codes in by_stay.items():
        vec = vectors.get(int(sid))
        if vec is None:
            continue
        for c in codes:
            j = idx.get(c)
            if j is not None:
                vec[j] = 1.0

    return vectors, vocab


def _hcpcs_code_norm(code: Any) -> str:
    if code is None:
        return ""
    s = " ".join(str(code).strip().upper().split())
    if not s:
        return ""
    return s


def _build_hcpcs_vectors(
    session: Any,
    stay_rows: list[StayRow],
    *,
    top_k: int,
    sql_in_batch_size: int,
) -> tuple[dict[int, np.ndarray], list[str]]:
    if top_k <= 0 or not stay_rows:
        return {}, []

    stay_ids = [int(s.stay_id) for s in stay_rows]
    hcpcs_sql = text(
        """
        SELECT
            cur.stay_id,
            h.hcpcs_cd
        FROM icustays cur
        INNER JOIN hcpcsevents h ON h.hadm_id = cur.hadm_id
        WHERE cur.stay_id IN :stay_ids
          AND h.hcpcs_cd IS NOT NULL
          AND (h.chartdate IS NULL OR h.chartdate <= cur.outtime)
        """
    ).bindparams(bindparam("stay_ids", expanding=True))

    rows: list[dict[str, Any]] = []
    chunk_size = max(1, int(sql_in_batch_size))
    for i in range(0, len(stay_ids), chunk_size):
        batch_stay_ids = stay_ids[i : i + chunk_size]
        batch_rows = session.execute(
            hcpcs_sql,
            {
                "stay_ids": batch_stay_ids,
            },
        ).mappings().all()
        rows.extend(batch_rows)

    freq: dict[str, int] = {}
    by_stay: dict[int, list[str]] = {}
    for r in rows:
        sid = int(r["stay_id"])
        code = _hcpcs_code_norm(r.get("hcpcs_cd"))
        if not code:
            continue
        freq[code] = freq.get(code, 0) + 1
        by_stay.setdefault(sid, []).append(code)

    vocab = [k for k, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
    idx = {k: i for i, k in enumerate(vocab)}

    vectors: dict[int, np.ndarray] = {}
    for sid in stay_ids:
        vectors[int(sid)] = np.zeros((len(vocab),), dtype=np.float32)

    for sid, codes in by_stay.items():
        vec = vectors.get(int(sid))
        if vec is None:
            continue
        for c in codes:
            j = idx.get(c)
            if j is not None:
                vec[j] = 1.0

    return vectors, vocab


def _build_pharmacy_vectors(
    session: Any,
    stay_rows: list[StayRow],
    *,
    top_k: int,
    sql_in_batch_size: int,
) -> tuple[dict[int, np.ndarray], list[str]]:
    if top_k <= 0 or not stay_rows:
        return {}, []

    stay_ids = [int(s.stay_id) for s in stay_rows]
    pharmacy_sql = text(
        """
        SELECT
            cur.stay_id,
            pr.drug,
            pr.route
        FROM icustays cur
        INNER JOIN prescriptions pr ON pr.hadm_id = cur.hadm_id
        WHERE cur.stay_id IN :stay_ids
          AND (
            pr.starttime IS NULL
            OR (
                pr.starttime <= cur.outtime
                AND (pr.stoptime IS NULL OR pr.stoptime >= cur.intime)
            )
          )
        """
    ).bindparams(bindparam("stay_ids", expanding=True))

    rows: list[dict[str, Any]] = []
    chunk_size = max(1, int(sql_in_batch_size))
    for i in range(0, len(stay_ids), chunk_size):
        batch_stay_ids = stay_ids[i : i + chunk_size]
        batch_rows = session.execute(
            pharmacy_sql,
            {
                "stay_ids": batch_stay_ids,
            },
        ).mappings().all()
        rows.extend(batch_rows)

    freq: dict[str, int] = {}
    by_stay: dict[int, list[str]] = {}
    for r in rows:
        sid = int(r["stay_id"])
        terms: list[str] = []

        drug = _pharmacy_term_norm(r.get("drug"))
        if drug:
            terms.append(f"DRUG::{drug}")

        route = _pharmacy_term_norm(r.get("route"))
        if route:
            terms.append(f"ROUTE::{route}")

        if not terms:
            continue

        for term in terms:
            freq[term] = freq.get(term, 0) + 1
            by_stay.setdefault(sid, []).append(term)

    vocab = [k for k, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
    idx = {k: i for i, k in enumerate(vocab)}

    vectors: dict[int, np.ndarray] = {}
    for sid in stay_ids:
        vectors[int(sid)] = np.zeros((len(vocab),), dtype=np.float32)

    for sid, terms in by_stay.items():
        vec = vectors.get(int(sid))
        if vec is None:
            continue
        for term in terms:
            j = idx.get(term)
            if j is not None:
                vec[j] = 1.0

    return vectors, vocab


class MonitoringRiskLateFusion(nn.Module):
    def __init__(
        self,
        seq_input_dim: int = 1,
        static_input_dim: int = 0,
        static_schema: dict[str, int] | None = None,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # dropout only applied between LSTM layers when num_layers > 1
        lstm_dropout = float(dropout) if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        # Attention pooling over LSTM output steps
        self.attn = nn.Linear(hidden_dim, 1)

        self.static_input_dim = int(max(0, static_input_dim))
        self.static_schema = static_schema or {}
        self.static_head: nn.Module
        fused_dim = hidden_dim
        if self.static_input_dim > 0:
            self.demographics_dim = int(self.static_schema.get("demographics_dim", 0))
            self.marital_dim = int(self.static_schema.get("marital_dim", 0))
            self.insurance_dim = int(self.static_schema.get("insurance_dim", 0))
            self.admission_type_dim = int(self.static_schema.get("admission_type_dim", 0))
            self.admit_to_dim = int(self.static_schema.get("admit_to_dim", 0))
            self.icd_dim = int(self.static_schema.get("icd_dim", 0))
            self.drg_dim = int(self.static_schema.get("drg_dim", 0))
            self.procedure_dim = int(self.static_schema.get("procedure_dim", 0))
            self.hcpcs_dim = int(self.static_schema.get("hcpcs_dim", 0))
            self.pharmacy_dim = int(self.static_schema.get("pharmacy_dim", 0))
            self.operational_dim = int(self.static_schema.get("operational_dim", 0))

            offset = 0
            self.slices: dict[str, tuple[int, int]] = {}
            for name, dim in [
                ("demographics", self.demographics_dim),
                ("marital", self.marital_dim),
                ("insurance", self.insurance_dim),
                ("admission_type", self.admission_type_dim),
                ("admit_to", self.admit_to_dim),
                ("icd", self.icd_dim),
                ("drg", self.drg_dim),
                ("procedure", self.procedure_dim),
                ("hcpcs", self.hcpcs_dim),
                ("pharmacy", self.pharmacy_dim),
                ("operational", self.operational_dim),
            ]:
                if dim > 0:
                    self.slices[name] = (offset, offset + dim)
                    offset += dim

            if offset != self.static_input_dim:
                raise ValueError(
                    f"static_schema size mismatch: schema_total={offset}, static_input_dim={self.static_input_dim}"
                )

            static_repr_dim = 0
            if self.demographics_dim > 0:
                self.demographics_head = nn.Sequential(
                    nn.Linear(self.demographics_dim, max(8, hidden_dim // 4)),
                    nn.ReLU(),
                )
                static_repr_dim += max(8, hidden_dim // 4)
                
            if self.operational_dim > 0:
                self.operational_head = nn.Sequential(
                    nn.Linear(self.operational_dim, max(8, hidden_dim // 4)),
                    nn.ReLU(),
                )
                static_repr_dim += max(8, hidden_dim // 4)

            self.marital_emb = nn.Embedding(self.marital_dim, _auto_embedding_dim(self.marital_dim)) if self.marital_dim > 0 else None
            self.insurance_emb = nn.Embedding(self.insurance_dim, _auto_embedding_dim(self.insurance_dim)) if self.insurance_dim > 0 else None
            self.admission_type_emb = nn.Embedding(self.admission_type_dim, _auto_embedding_dim(self.admission_type_dim)) if self.admission_type_dim > 0 else None
            self.admit_to_emb = nn.Embedding(self.admit_to_dim, _auto_embedding_dim(self.admit_to_dim)) if self.admit_to_dim > 0 else None
            self.icd_emb = nn.Embedding(self.icd_dim, _auto_embedding_dim(self.icd_dim)) if self.icd_dim > 0 else None
            self.drg_emb = nn.Embedding(self.drg_dim, _auto_embedding_dim(self.drg_dim)) if self.drg_dim > 0 else None
            self.procedure_emb = nn.Embedding(self.procedure_dim, _auto_embedding_dim(self.procedure_dim)) if self.procedure_dim > 0 else None
            self.hcpcs_emb = nn.Embedding(self.hcpcs_dim, _auto_embedding_dim(self.hcpcs_dim)) if self.hcpcs_dim > 0 else None
            self.pharmacy_emb = nn.Embedding(self.pharmacy_dim, _auto_embedding_dim(self.pharmacy_dim)) if self.pharmacy_dim > 0 else None

            for emb in [
                self.marital_emb,
                self.insurance_emb,
                self.admission_type_emb,
                self.admit_to_emb,
                self.icd_emb,
                self.drg_emb,
                self.procedure_emb,
                self.hcpcs_emb,
                self.pharmacy_emb,
            ]:
                if emb is not None:
                    static_repr_dim += int(emb.embedding_dim)

            static_hidden = max(16, hidden_dim // 2)
            self.static_head = nn.Sequential(
                nn.Linear(static_repr_dim, static_hidden),
                nn.ReLU(),
                nn.Dropout(p=dropout / 2),
            )
            fused_dim += static_hidden
        else:
            self.static_head = nn.Identity()

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _slice_tensor(self, x_static: torch.Tensor, name: str) -> torch.Tensor | None:
        sl = self.slices.get(name)
        if sl is None:
            return None
        s, e = sl
        return x_static[:, s:e]

    @staticmethod
    def _onehot_to_embedding(x_group: torch.Tensor, emb: nn.Embedding) -> torch.Tensor:
        idx = torch.argmax(x_group, dim=1)
        return emb(idx)

    @staticmethod
    def _multihot_to_embedding(x_group: torch.Tensor, emb: nn.Embedding) -> torch.Tensor:
        # Sum token embeddings weighted by multihot indicators, then normalize by active count.
        summed = x_group @ emb.weight
        denom = torch.clamp(x_group.sum(dim=1, keepdim=True), min=1.0)
        return summed / denom

    def _encode_static(self, x_static: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []

        x_demo = self._slice_tensor(x_static, "demographics")
        if x_demo is not None and x_demo.shape[1] > 0:
            parts.append(self.demographics_head(x_demo))
            
        x_op = self._slice_tensor(x_static, "operational")
        if x_op is not None and x_op.shape[1] > 0:
            parts.append(self.operational_head(x_op))

        for group_name, emb, is_multihot in [
            ("marital", self.marital_emb, False),
            ("insurance", self.insurance_emb, False),
            ("admission_type", self.admission_type_emb, False),
            ("admit_to", self.admit_to_emb, False),
            ("icd", self.icd_emb, True),
            ("drg", self.drg_emb, True),
            ("procedure", self.procedure_emb, True),
            ("hcpcs", self.hcpcs_emb, True),
            ("pharmacy", self.pharmacy_emb, True),
        ]:
            if emb is None:
                continue
            x_group = self._slice_tensor(x_static, group_name)
            if x_group is None or x_group.shape[1] == 0:
                continue
            if is_multihot:
                parts.append(self._multihot_to_embedding(x_group, emb))
            else:
                parts.append(self._onehot_to_embedding(x_group, emb))

        if not parts:
            raise ValueError("No static components found to encode while static_input_dim > 0")
        return torch.cat(parts, dim=1)

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.lstm(x_seq)  # out: [B, T, H]
        # Attention pooling: weighted sum over all time steps
        attn_w = torch.softmax(self.attn(out).squeeze(-1), dim=1)  # [B, T]
        seq_repr = (out * attn_w.unsqueeze(-1)).sum(dim=1)          # [B, H]
        if self.static_input_dim > 0:
            if x_static is None:
                raise ValueError("x_static is required when static_input_dim > 0")
            static_repr = self.static_head(self._encode_static(x_static))
            fused = torch.cat([seq_repr, static_repr], dim=1)
        else:
            fused = seq_repr
        logits = self.head(fused).squeeze(-1)
        return logits
        return logits


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


class TransformerSeqEncoder(nn.Module):
    def __init__(
        self,
        seq_input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        nhead: int,
    ) -> None:
        super().__init__()
        self.seq_proj = nn.Linear(seq_input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout, 
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x_seq: torch.Tensor, mask_indices: torch.Tensor | None = None, extra_tokens: torch.Tensor | None = None) -> torch.Tensor:
        x = self.seq_proj(x_seq)  # [B, T, H]
        
        if mask_indices is not None:
            # mask_indices: [B, T] boolean tensor
            expanded_mask = mask_indices.unsqueeze(-1).expand_as(x)
            x = torch.where(expanded_mask, self.mask_token, x)
            
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        if extra_tokens is not None:
            x = torch.cat((cls_tokens, extra_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
            
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        return out


class TransformerPretrainer(nn.Module):
    def __init__(self, encoder: TransformerSeqEncoder, seq_input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = encoder
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, seq_input_dim)
        )
        
    def forward(self, x_seq: torch.Tensor, mask_indices: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x_seq, mask_indices)
        encoded_seq = encoded[:, 1:, :]  # Drop CLS token
        return self.reconstruction_head(encoded_seq)


class MonitoringRiskTransformerLateFusion(MonitoringRiskLateFusion):
    def __init__(
        self,
        seq_input_dim: int = 1,
        static_input_dim: int = 0,
        static_schema: dict[str, int] | None = None,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        nhead: int = 4,
    ) -> None:
        super().__init__(seq_input_dim, static_input_dim, static_schema, hidden_dim, num_layers, dropout)
        
        self.lstm = None
        self.encoder = TransformerSeqEncoder(
            seq_input_dim=seq_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead
        )
        
        if self.static_input_dim > 0:
            static_hidden = max(16, hidden_dim // 2)
            self.static_to_hidden = nn.Linear(static_hidden, hidden_dim)
            
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor | None = None) -> torch.Tensor:
        if self.static_input_dim > 0:
            if x_static is None:
                raise ValueError("x_static is required when static_input_dim > 0")
            static_repr = self.static_head(self._encode_static(x_static))
            static_token = self.static_to_hidden(static_repr).unsqueeze(1)
            out = self.encoder(x_seq, extra_tokens=static_token)
        else:
            out = self.encoder(x_seq)
            
        cls_repr = out[:, 0, :]
        logits = self.head(cls_repr).squeeze(-1)
        return logits


class MonitoringRiskTransformerLateFusionWithNotes(MonitoringRiskTransformerLateFusion):
    def __init__(
        self,
        seq_input_dim: int = 1,
        static_input_dim: int = 0,
        static_schema: dict[str, int] | None = None,
        note_dim: int = 4096,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        nhead: int = 4,
    ) -> None:
        super().__init__(
            seq_input_dim=seq_input_dim,
            static_input_dim=static_input_dim,
            static_schema=static_schema,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead
        )
        
        self.note_dim = note_dim
        
        self.note_head = nn.Sequential(
            nn.LayerNorm(note_dim),
            nn.Dropout(p=dropout),
            nn.Linear(note_dim, hidden_dim),
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor | None = None, x_note: torch.Tensor | None = None) -> torch.Tensor:
        tokens_to_add = []
        
        if self.static_input_dim > 0:
            if x_static is None:
                raise ValueError("x_static is required when static_input_dim > 0")
            static_repr = self.static_head(self._encode_static(x_static))
            static_token = self.static_to_hidden(static_repr).unsqueeze(1)
            tokens_to_add.append(static_token)
            
        if x_note is not None:
            x_note_norm = torch.nn.functional.normalize(x_note, p=2, dim=1)
            note_repr = self.note_head(x_note_norm)
            note_token = note_repr.unsqueeze(1)
            tokens_to_add.append(note_token)
            
        if len(tokens_to_add) > 0:
            extra_tokens = torch.cat(tokens_to_add, dim=1)
            out = self.encoder(x_seq, extra_tokens=extra_tokens)
        else:
            out = self.encoder(x_seq)
            
        cls_repr = out[:, 0, :]
        logits = self.head(cls_repr).squeeze(-1)
        return logits

class FocalLossWithLogits(nn.Module):
    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: float | None = None,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        pt = torch.exp(-bce)
        focal = (1.0 - pt) ** self.gamma
        loss = focal * bce

        if self.alpha is not None:
            a = float(self.alpha)
            alpha_t = a * targets + (1.0 - a) * (1.0 - targets)
            loss = loss * alpha_t

        return loss.mean()


def _auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    rank_sum_pos = ranks[pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _prauc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute area under precision-recall curve via trapezoidal rule."""
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    tp_cumsum = np.cumsum(y_sorted)
    n_pos = int(y_true.sum())
    recalls = tp_cumsum / n_pos
    precisions = tp_cumsum / np.arange(1, len(y_sorted) + 1)
    # prepend sentinel
    recalls = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[1.0], precisions])
    return float(np.trapezoid(precisions, recalls))


def _best_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Search threshold in [0.01, 0.99] that maximises F1."""
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


def _threshold_max_precision_with_recall_floor(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    min_recall: float,
) -> float | None:
    """Choose threshold that maximises precision under recall >= min_recall."""
    target_recall = float(np.clip(min_recall, 0.0, 1.0))
    best_t: float | None = None
    best_p = -1.0
    best_r = -1.0

    for t in np.linspace(0.01, 0.99, 99):
        pred = (y_prob >= t).astype(np.int64)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)

        if r + 1e-12 < target_recall:
            continue

        if (p > best_p + 1e-12) or (abs(p - best_p) <= 1e-12 and r > best_r + 1e-12):
            best_p = float(p)
            best_r = float(r)
            best_t = float(t)

    return best_t


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    y_true = y_true.astype(np.int64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    total = max(1, tp + tn + fp + fn)
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2.0 * precision * recall / max(1e-12, (precision + recall))
    accuracy = (tp + tn) / total
    
    brier_score = float(np.mean((y_prob - y_true) ** 2))
    # Brier skill score: 1 - Brier / Brier_ref, where Brier_ref is always predicting the base rate
    base_rate = np.mean(y_true)
    brier_ref = float(np.mean((base_rate - y_true) ** 2))
    brier_skill_score = 1.0 - (brier_score / max(1e-12, brier_ref))

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": _auc_score(y_true, y_prob),
        "pr_auc": _prauc_score(y_true, y_prob),
        "brier_score": brier_score,
        "brier_skill_score": brier_skill_score,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def fetch_stay_rows(session, *, limit: int, horizon_hours: int) -> list[StayRow]:
    q = text(
        """
        WITH RankedStays AS (
            SELECT
                i.subject_id,
                i.hadm_id,
                i.stay_id,
                i.intime,
                i.outtime,
                (YEAR(i.intime) - p.anchor_year + p.anchor_age) AS age,
                p.gender,
                a.marital_status,
                a.insurance,
                a.admission_type,
                a.admission_location AS admit_to,
                LEAD(i.intime) OVER (PARTITION BY i.subject_id ORDER BY i.intime ASC) as next_intime
            FROM icustays i
            INNER JOIN admissions a ON a.hadm_id = i.hadm_id
            INNER JOIN patients p ON p.subject_id = i.subject_id
            WHERE i.outtime IS NOT NULL AND i.intime IS NOT NULL AND i.outtime > i.intime
        )
        SELECT 
            subject_id, hadm_id, stay_id, intime, outtime, age, gender, 
            marital_status, insurance, admission_type, admit_to,
            CASE 
                WHEN next_intime IS NOT NULL 
                 AND next_intime > outtime 
                 AND next_intime <= DATE_ADD(outtime, INTERVAL :horizon_hours HOUR) 
                THEN 1 ELSE 0 
            END AS label
        FROM RankedStays
        ORDER BY intime ASC
        LIMIT :limit_rows
        """
    )
    rows = session.execute(
        q,
        {"horizon_hours": int(horizon_hours), "limit_rows": int(limit)},
    ).mappings().all()

    out: list[StayRow] = []
    for r in rows:
        out.append(
            StayRow(
                subject_id=int(r["subject_id"]),
                hadm_id=int(r["hadm_id"]),
                stay_id=int(r["stay_id"]),
                intime=r["intime"],
                outtime=r["outtime"],
                label=int(r["label"]),
                age=float(r["age"]) if r.get("age") is not None else None,
                gender=str(r["gender"]) if r.get("gender") is not None else None,
                marital_status=str(r["marital_status"]) if r.get("marital_status") is not None else None,
                insurance=str(r["insurance"]) if r.get("insurance") is not None else None,
                admission_type=str(r["admission_type"]) if r.get("admission_type") is not None else None,
                admit_to=str(r["admit_to"]) if r.get("admit_to") is not None else None,
            )
        )
    return out


def build_dataset(
    *,
    extractor: MIMIC4DataExtractor,
    monitor_agent: MonitoringAgent,
    stay_rows: list[StayRow],
    seq_len: int,
    max_query_rows: int,
    include_all_chartevents: bool,
    include_outputevents: bool,
    include_datetimeevents: bool,
    include_labevents: bool,
    include_inputevents: bool,
    include_omr: bool,
    include_static_demographics: bool,
    include_admission_context: bool,
    include_icd_history: bool,
    icd_top_k: int,
    icd_include_current_hadm: bool,
    include_drg_onehot: bool,
    drg_top_k: int,
    include_procedure_icd: bool,
    procedure_top_k: int,
    include_hcpcs_events: bool,
    hcpcs_top_k: int,
    include_pharmacy: bool,
    pharmacy_top_k: int,
    pre_discharge_hours: int,
    use_core_channels: bool,
    selected_channels: list[str] | None,
    seq_aggregation: str,
    sql_in_batch_size: int,
    note_embeddings_dict: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, int]]:
    xs_seq: list[np.ndarray] = []
    xs_static: list[np.ndarray] = []
    xs_note: list[np.ndarray] = []
    ys: list[int] = []

    marital_vocab, insurance_vocab, admission_type_vocab, admit_to_vocab = _build_static_feature_vocabs(
        stay_rows,
        marital_top_k=8,
        insurance_top_k=8,
        admission_type_top_k=8,
        admit_to_top_k=8,
    )
    logger.info(
        "static vocab sizes | marital=%d insurance=%d admission_type=%d admit_to=%d",
        len(marital_vocab),
        len(insurance_vocab),
        len(admission_type_vocab),
        len(admit_to_vocab),
    )

    icd_vectors: dict[int, np.ndarray] = {}
    icd_vocab: list[str] = []
    if include_icd_history and int(icd_top_k) > 0:
        icd_vectors, icd_vocab = _build_icd_history_vectors(
            extractor.session,
            stay_rows,
            top_k=int(icd_top_k),
            include_current_hadm=bool(icd_include_current_hadm),
            sql_in_batch_size=int(sql_in_batch_size),
        )
        logger.info(
            "icd history features enabled | top_k=%d include_current_hadm=%s",
            len(icd_vocab),
            bool(icd_include_current_hadm),
        )

    drg_vectors: dict[int, np.ndarray] = {}
    drg_vocab: list[str] = []
    if include_drg_onehot and int(drg_top_k) > 0:
        drg_vectors, drg_vocab = _build_drg_vectors(
            extractor.session,
            stay_rows,
            top_k=int(drg_top_k),
            sql_in_batch_size=int(sql_in_batch_size),
        )
        logger.info("drg onehot features enabled | top_k=%d", len(drg_vocab))

    procedure_vectors: dict[int, np.ndarray] = {}
    procedure_vocab: list[str] = []
    if include_procedure_icd and int(procedure_top_k) > 0:
        procedure_vectors, procedure_vocab = _build_procedure_vectors(
            extractor.session,
            stay_rows,
            top_k=int(procedure_top_k),
            sql_in_batch_size=int(sql_in_batch_size),
        )
        logger.info("procedure icd features enabled | top_k=%d", len(procedure_vocab))

    hcpcs_vectors: dict[int, np.ndarray] = {}
    hcpcs_vocab: list[str] = []
    if include_hcpcs_events and int(hcpcs_top_k) > 0:
        hcpcs_vectors, hcpcs_vocab = _build_hcpcs_vectors(
            extractor.session,
            stay_rows,
            top_k=int(hcpcs_top_k),
            sql_in_batch_size=int(sql_in_batch_size),
        )
        logger.info("hcpcs features enabled | top_k=%d", len(hcpcs_vocab))

    pharmacy_vectors: dict[int, np.ndarray] = {}
    pharmacy_vocab: list[str] = []
    if include_pharmacy and int(pharmacy_top_k) > 0:
        pharmacy_vectors, pharmacy_vocab = _build_pharmacy_vectors(
            extractor.session,
            stay_rows,
            top_k=int(pharmacy_top_k),
            sql_in_batch_size=int(sql_in_batch_size),
        )
        logger.info("pharmacy features enabled | top_k=%d", len(pharmacy_vocab))

    def _clip_payload_to_pre_discharge_window(payload: dict[str, Any], hours: int = 24) -> dict[str, Any]:
        """Keep only records within [outtime-hours, outtime] for each stay window."""
        if not isinstance(payload, dict):
            return payload

        windows = payload.get("windows", [])
        events = payload.get("x_t", [])
        if not isinstance(windows, list) or len(windows) == 0:
            return payload

        latest_outtime: datetime | None = None
        earliest_intime: datetime | None = None
        for w in windows:
            if not isinstance(w, dict):
                continue
            intime = w.get("intime")
            outtime = w.get("outtime")
            if isinstance(intime, datetime):
                earliest_intime = intime if earliest_intime is None else min(earliest_intime, intime)
            if isinstance(outtime, datetime):
                latest_outtime = outtime if latest_outtime is None else max(latest_outtime, outtime)

        if latest_outtime is None:
            return payload

        window_start = latest_outtime - timedelta(hours=int(hours))

        if earliest_intime is not None:
            if window_start < earliest_intime:
                window_start = earliest_intime

        clipped_events: list[dict[str, Any]] = []
        if isinstance(events, list):
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                ct = ev.get("charttime")
                if not isinstance(ct, datetime):
                    continue
                if window_start <= ct <= latest_outtime:
                    clipped_events.append(ev)

        clipped_windows: list[dict[str, Any]] = []
        for w in windows:
            if not isinstance(w, dict):
                continue
            nw = dict(w)
            nw["intime"] = window_start
            nw["outtime"] = latest_outtime
            clipped_windows.append(nw)

        clipped_payload = dict(payload)
        clipped_payload["windows"] = clipped_windows
        clipped_payload["x_t"] = clipped_events
        return clipped_payload

    def _to_fixed_len_sequence(arr: np.ndarray, target_len: int, mode: str) -> np.ndarray:
        """Convert variable-length [T, C] sequence into fixed [target_len, C]."""
        if target_len <= 0:
            raise ValueError("target_len must be positive")

        t, c = arr.shape
        if t == 0:
            return np.zeros((target_len, c), dtype=np.float32)

        if t == target_len:
            return arr.astype(np.float32)

        if mode == "last":
            if t > target_len:
                return arr[-target_len:, :].astype(np.float32)
            out = np.zeros((target_len, c), dtype=np.float32)
            out[-t:, :] = arr
            out[: target_len - t, :] = arr[0, :]
            return out

        if mode == "first":
            if t > target_len:
                return arr[:target_len, :].astype(np.float32)
            out = np.zeros((target_len, c), dtype=np.float32)
            out[:t, :] = arr
            out[t:, :] = arr[-1, :]
            return out

        if mode == "full_resample":
            # Cover the whole stay by dividing timeline into target_len bins.
            boundaries = np.linspace(0, t, target_len + 1)
            out = np.zeros((target_len, c), dtype=np.float32)
            for i in range(target_len):
                s_idx = int(np.floor(boundaries[i]))
                e_idx = int(np.floor(boundaries[i + 1]))
                if e_idx <= s_idx:
                    s_idx = min(s_idx, t - 1)
                    out[i, :] = arr[s_idx, :]
                else:
                    out[i, :] = arr[s_idx:e_idx, :].mean(axis=0)
            return out.astype(np.float32)

        raise ValueError(f"Unsupported seq_aggregation mode: {mode}")

    batch_size_fetch = 500
    all_payloads = {}
    logger.info("Fetching all patient series in batches of %d...", batch_size_fetch)
    for i in range(0, len(stay_rows), batch_size_fetch):
        chunk = stay_rows[i:i + batch_size_fetch]
        for attempt in range(1, 9):
            try:
                chunk_payloads = extractor.build_icu_xt_series_batch(
                    stay_rows=chunk,
                    include_all_chartevents=include_all_chartevents,
                    chartevent_label_whitelist=DEFAULT_CHARTEVENT_LABEL_WHITELIST_TOP100,
                    include_outputevents=include_outputevents,
                    include_datetimeevents=include_datetimeevents,
                    include_labevents=include_labevents,
                    include_inputevents=include_inputevents,
                    include_omr=include_omr,
                    pre_discharge_hours=int(pre_discharge_hours) if pre_discharge_hours else None,
                    batch_size=batch_size_fetch,
                )
                all_payloads.update(chunk_payloads)
                break
            except OperationalError as exc:
                if not _is_retryable_mysql_error(exc):
                    raise
                wait_s = min(10.0, 0.8 * float(attempt))
                logger.warning(
                    "MySQL transient error on batch %d (attempt=%s/8, wait=%.1fs): %s",
                    i,
                    attempt,
                    wait_s,
                    exc,
                )
                try:
                    extractor.session.close()
                except Exception:
                    pass
                if attempt >= 8:
                    logger.error("Skip batch %d after exhausted DB retries", i)
                    break
                pytime.sleep(wait_s)

    for idx, s in enumerate(stay_rows, start=1):
        payload = all_payloads.get(s.stay_id)

        if payload is None:
            continue

        try:
            # Keep a defensive post-clip to align the final dataset window exactly.
            payload = _clip_payload_to_pre_discharge_window(payload, hours=int(pre_discharge_hours))
            channel_subset = selected_channels
            if channel_subset is None and use_core_channels:
                channel_subset = MonitoringAgent.DEFAULT_CORE_CHANNELS
            vital_values, vital_mask = monitor_agent.build_hourly_vital_channels_with_mask_from_xt(
                xt_payload=payload,
                step_hours=1,
                selected_channels=channel_subset,
            )
        except Exception:
            continue

        # delta = value[t] - value[t-1]; first step delta = 0
        vital_delta = np.zeros_like(vital_values)
        vital_delta[1:, :] = vital_values[1:, :] - vital_values[:-1, :]
        vital_delta = np.clip(vital_delta, -1.0, 1.0)

        # Concatenate: [40 values | 40 masks | 40 deltas] => 120 dims
        vital_channels = np.concatenate([vital_values, vital_mask, vital_delta], axis=1)

        # Optional source-level channels from inputevents / omr.
        aux_channels = _build_source_aux_channels(
            payload=payload,
            n_steps=vital_channels.shape[0],
            include_inputevents=include_inputevents,
            include_omr=include_omr,
        )
        if aux_channels.shape[1] > 0:
            vital_channels = np.concatenate([vital_channels, aux_channels], axis=1)

        static_vec = _build_static_vector(
            s,
            include_demographics=bool(include_static_demographics),
            include_admission_context=bool(include_admission_context),
            marital_vocab=marital_vocab,
            insurance_vocab=insurance_vocab,
            admission_type_vocab=admission_type_vocab,
            admit_to_vocab=admit_to_vocab,
        )
        icd_vec = np.zeros((0,), dtype=np.float32)
        if include_icd_history and len(icd_vocab) > 0:
            icd_vec = icd_vectors.get(int(s.stay_id), np.zeros((len(icd_vocab),), dtype=np.float32)).astype(np.float32)

        drg_vec = np.zeros((0,), dtype=np.float32)
        if include_drg_onehot and len(drg_vocab) > 0:
            drg_vec = drg_vectors.get(int(s.stay_id), np.zeros((len(drg_vocab),), dtype=np.float32)).astype(np.float32)

        procedure_vec = np.zeros((0,), dtype=np.float32)
        if include_procedure_icd and len(procedure_vocab) > 0:
            procedure_vec = procedure_vectors.get(
                int(s.stay_id),
                np.zeros((len(procedure_vocab),), dtype=np.float32),
            ).astype(np.float32)

        hcpcs_vec = np.zeros((0,), dtype=np.float32)
        if include_hcpcs_events and len(hcpcs_vocab) > 0:
            hcpcs_vec = hcpcs_vectors.get(
                int(s.stay_id),
                np.zeros((len(hcpcs_vocab),), dtype=np.float32),
            ).astype(np.float32)

        pharmacy_vec = np.zeros((0,), dtype=np.float32)
        if include_pharmacy and len(pharmacy_vocab) > 0:
            pharmacy_vec = pharmacy_vectors.get(
                int(s.stay_id),
                np.zeros((len(pharmacy_vocab),), dtype=np.float32),
            ).astype(np.float32)

        if static_vec.size > 0 or icd_vec.size > 0 or drg_vec.size > 0 or procedure_vec.size > 0 or hcpcs_vec.size > 0 or pharmacy_vec.size > 0:
            static_fused = np.concatenate(
                [x for x in (static_vec, icd_vec, drg_vec, procedure_vec, hcpcs_vec, pharmacy_vec) if x.size > 0],
                axis=0,
            ).astype(np.float32)
        else:
            static_fused = np.zeros((0,), dtype=np.float32)

        # Convert variable-length channel sequence to fixed-length for batching.
        seq = _to_fixed_len_sequence(vital_channels, target_len=seq_len, mode=seq_aggregation)
        xs_seq.append(seq)
        xs_static.append(static_fused)
        if note_embeddings_dict is not None:
            n_emb = note_embeddings_dict.get(int(s.stay_id), np.zeros((4096,), dtype=np.float32)).astype(np.float32)
            xs_note.append(n_emb)
        ys.append(int(s.label))

        if idx % 50 == 0:
            logger.info("dataset progress %d/%d | kept=%d", idx, len(stay_rows), len(xs_seq))

    if not xs_seq:
        raise RuntimeError("No valid samples produced from queried X_t payloads")

    x_seq = np.stack(xs_seq, axis=0).astype(np.float32)
    if xs_static and xs_static[0].size > 0:
        x_static = np.stack(xs_static, axis=0).astype(np.float32)
    else:
        x_static = np.zeros((len(xs_static), 0), dtype=np.float32)

    static_schema = {
        "demographics_dim": 4 if include_static_demographics else 0,
        "marital_dim": (len(marital_vocab) + 1) if include_admission_context else 0,
        "insurance_dim": (len(insurance_vocab) + 1) if include_admission_context else 0,
        "admission_type_dim": (len(admission_type_vocab) + 1) if include_admission_context else 0,
        "admit_to_dim": (len(admit_to_vocab) + 1) if include_admission_context else 0,
        "icd_dim": len(icd_vocab) if include_icd_history else 0,
        "drg_dim": len(drg_vocab) if include_drg_onehot else 0,
        "procedure_dim": len(procedure_vocab) if include_procedure_icd else 0,
        "hcpcs_dim": len(hcpcs_vocab) if include_hcpcs_events else 0,
        "pharmacy_dim": len(pharmacy_vocab) if include_pharmacy else 0,
    }

    y = np.asarray(ys, dtype=np.float32)
    x_note = np.stack(xs_note, axis=0).astype(np.float32) if xs_note else None
    
    logger.info("dataset built: x_seq.shape=%s, x_static.shape=%s, y.shape=%s", x_seq.shape, x_static.shape, y.shape)
    if x_note is not None:
        logger.info("x_note.shape=%s", x_note.shape)
        
    logger.info("static schema: %s", json.dumps(static_schema, ensure_ascii=False))
    return x_seq, x_static, y, x_note, static_schema


def _build_source_aux_channels(
    *,
    payload: dict[str, Any],
    n_steps: int,
    include_inputevents: bool,
    include_omr: bool,
) -> np.ndarray:
    if n_steps <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    windows = payload.get("windows", []) if isinstance(payload, dict) else []
    events = payload.get("x_t", []) if isinstance(payload, dict) else []
    if not windows:
        n_aux = (2 if include_inputevents else 0) + (2 if include_omr else 0)
        return np.zeros((n_steps, n_aux), dtype=np.float32)

    intime = windows[0].get("intime")
    outtime = windows[0].get("outtime")
    if intime is None or outtime is None or outtime <= intime:
        n_aux = (2 if include_inputevents else 0) + (2 if include_omr else 0)
        return np.zeros((n_steps, n_aux), dtype=np.float32)

    duration_seconds = (outtime - intime).total_seconds()
    step_seconds = max(1.0, duration_seconds / max(1, n_steps - 1))

    input_count = np.zeros((n_steps,), dtype=np.float32)
    input_sum = np.zeros((n_steps,), dtype=np.float32)
    omr_count = np.zeros((n_steps,), dtype=np.float32)
    omr_sum = np.zeros((n_steps,), dtype=np.float32)

    for row in events:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source") or "").strip().lower()
        if source not in {"inputevents", "omr"}:
            continue
        charttime = row.get("charttime")
        if charttime is None:
            continue
        if isinstance(charttime, date) and not isinstance(charttime, datetime):
            charttime = datetime.combine(charttime, datetime.min.time())
        if not isinstance(charttime, datetime):
            continue
        value = row.get("value")
        if value is None:
            continue
        try:
            fv = float(value)
        except Exception:
            continue

        sec = (charttime - intime).total_seconds()
        idx = int(sec // step_seconds)
        idx = max(0, min(idx, n_steps - 1))

        if source == "inputevents" and include_inputevents:
            input_count[idx] += 1.0
            input_sum[idx] += abs(fv)
        elif source == "omr" and include_omr:
            omr_count[idx] += 1.0
            omr_sum[idx] += fv

    cols: list[np.ndarray] = []
    if include_inputevents:
        input_count_norm = np.clip(input_count / 5.0, 0.0, 1.0)
        input_mean = np.zeros_like(input_sum)
        nz = input_count > 0
        input_mean[nz] = input_sum[nz] / input_count[nz]
        input_mean_norm = np.clip(input_mean / 500.0, 0.0, 1.0)
        cols.extend([input_count_norm, input_mean_norm])

    if include_omr:
        omr_count_norm = np.clip(omr_count / 3.0, 0.0, 1.0)
        omr_mean = np.zeros_like(omr_sum)
        nz = omr_count > 0
        omr_mean[nz] = omr_sum[nz] / omr_count[nz]
        omr_mean_norm = np.clip(omr_mean / 200.0, 0.0, 1.0)
        cols.extend([omr_count_norm, omr_mean_norm])

    if not cols:
        return np.zeros((n_steps, 0), dtype=np.float32)
    return np.stack(cols, axis=1).astype(np.float32)


def train_and_evaluate(
    *,
    x_seq: np.ndarray,
    x_static: np.ndarray,
    y: np.ndarray,
    x_note: np.ndarray | None = None,
    static_schema: dict[str, int] | None,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    loss_name: str,
    focal_gamma: float,
    focal_alpha: float | None,
    threshold_policy: str,
    threshold_target_recall: float,
    early_stop_enabled: bool,
    early_stop_patience: int,
    early_stop_min_delta: float,
    early_stop_monitor: str,
    device: torch.device,
    model_class: type = MonitoringRiskLateFusion,
    pretrained_encoder_state: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    train_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
) -> dict[str, Any]:
    n_total = len(x_seq)
    
    if train_idx is not None and val_idx is not None and test_idx is not None:
        logger.info("Using provided train/val/test splits")
        x_seq_train, x_seq_val, x_seq_test = x_seq[train_idx], x_seq[val_idx], x_seq[test_idx]
        x_static_train, x_static_val, x_static_test = x_static[train_idx], x_static[val_idx], x_static[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    else:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(x_seq))
        rng.shuffle(idx)

        x_seq = x_seq[idx]
        x_static = x_static[idx]
        y = y[idx]

        if val_ratio < 0.0 or test_ratio < 0.0 or (val_ratio + test_ratio) >= 1.0:
            raise ValueError("val_ratio and test_ratio must be >=0 and val_ratio + test_ratio < 1")

        n_test = int(math.floor(test_ratio * n_total))
        n_val = int(math.floor(val_ratio * n_total))
        n_train = n_total - n_val - n_test

        # Keep all splits non-empty when ratios are configured.
        if n_train <= 0:
            raise ValueError("Train split is empty. Reduce val_ratio/test_ratio or increase train-limit")
        if test_ratio > 0.0 and n_test == 0:
            n_test = 1
            n_train -= 1
        if val_ratio > 0.0 and n_val == 0:
            n_val = 1
            n_train -= 1
        if n_train <= 0:
            raise ValueError("Invalid split after enforcing non-empty val/test sets")

        train_end = n_train
        val_end = n_train + n_val
        x_seq_train, x_seq_val, x_seq_test = x_seq[:train_end], x_seq[train_end:val_end], x_seq[val_end:]
        x_static_train, x_static_val, x_static_test = (
            x_static[:train_end],
            x_static[train_end:val_end],
            x_static[val_end:],
        )
        if x_note is not None:
            x_note_train, x_note_val, x_note_test = (
                x_note[:train_end],
                x_note[train_end:val_end],
                x_note[val_end:],
            )
        else:
            x_note_train = x_note_val = x_note_test = None
            
        y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    logger.info("train uses full training split (no class downsampling)")

    seq_input_dim = int(x_seq_train.shape[-1])
    static_input_dim = int(x_static_train.shape[-1])
    
    kwargs = {
        "seq_input_dim": seq_input_dim,
        "static_input_dim": static_input_dim,
        "static_schema": static_schema,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.3,
    }
    if model_kwargs is not None:
        kwargs.update(model_kwargs)
        
    model = model_class(**kwargs)
    
    if pretrained_encoder_state is not None and hasattr(model, 'encoder'):
        logger.info("Loading pretrained encoder weights...")
        model.encoder.load_state_dict(pretrained_encoder_state)
    model = model.to(device)
    logger.info("model_input_dims | seq=%d static=%d", seq_input_dim, static_input_dim)
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    pw_value = n_neg / max(1, n_pos)
    pw = torch.tensor([pw_value], dtype=torch.float32, device=device)
    logger.info("pos_weight=%.2f  (n_pos=%d, n_neg=%d)", pw.item(), n_pos, n_neg)
    if loss_name == "focal":
        focal_loss = FocalLossWithLogits(gamma=focal_gamma, alpha=focal_alpha, pos_weight=pw)

        def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return focal_loss(logits, targets)

    else:
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pw)

        def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return bce_loss(logits, targets)

    logger.info("loss=%s%s", loss_name, f" (gamma={focal_gamma}, alpha={focal_alpha})" if loss_name == "focal" else "")
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=lr * 0.01)

    y_val_int = y_val.astype(np.int64)
    y_test_int = y_test.astype(np.int64)

    xv_seq = torch.from_numpy(x_seq_val).to(device)
    xv_static = torch.from_numpy(x_static_val).to(device)
    xv_note = torch.from_numpy(x_note_val).to(device) if x_note_val is not None else None
    yv = torch.from_numpy(y_val).to(device)

    monitor_mode = "min" if early_stop_monitor == "val_loss" else "max"
    best_monitor_value = float("inf") if monitor_mode == "min" else -float("inf")
    epochs_without_improve = 0
    best_epoch = 0
    stopped_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    epoch_monitoring: list[dict[str, float]] = []

    def _is_improved(metric_value: float, best_value: float) -> bool:
        if np.isnan(metric_value):
            return False
        if monitor_mode == "min":
            return metric_value < (best_value - float(early_stop_min_delta))
        return metric_value > (best_value + float(early_stop_min_delta))

    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n_train)
        epoch_loss = 0.0
        epoch_seen = 0

        for start in range(0, len(order), batch_size):
            sl = order[start : start + batch_size]
            xb_seq = torch.from_numpy(x_seq_train[sl]).to(device)
            xb_static = torch.from_numpy(x_static_train[sl]).to(device)
            xb_note = torch.from_numpy(x_note_train[sl]).to(device) if x_note_train is not None else None
            yb = torch.from_numpy(y_train[sl]).to(device)

            if xb_note is not None:
                logits = model(xb_seq, xb_static, xb_note)
            else:
                logits = model(xb_seq, xb_static)
                
            loss = compute_loss(logits, yb)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()

            epoch_loss += float(loss.item()) * len(sl)
            epoch_seen += len(sl)

        scheduler.step()
        epoch_loss /= max(1, epoch_seen)
        current_lr = scheduler.get_last_lr()[0]

        model.eval()
        with torch.no_grad():
            if xv_note is not None:
                val_logits = model(xv_seq, xv_static, xv_note)
            else:
                val_logits = model(xv_seq, xv_static)
            val_loss = float(compute_loss(val_logits, yv).item())
            val_probs = torch.sigmoid(val_logits).cpu().numpy().astype(np.float64)

        val_metrics_epoch = _classification_metrics(y_val_int, val_probs, threshold=0.5)
        epoch_row = {
            "epoch": float(ep),
            "train_loss": float(epoch_loss),
            "val_loss": float(val_loss),
            "val_f1_at_0.5": float(val_metrics_epoch["f1"]),
            "val_recall_at_0.5": float(val_metrics_epoch["recall"]),
            "val_precision_at_0.5": float(val_metrics_epoch["precision"]),
            "val_pr_auc": float(val_metrics_epoch["pr_auc"]),
            "val_roc_auc": float(val_metrics_epoch["roc_auc"]),
            "lr": float(current_lr),
        }
        epoch_monitoring.append(epoch_row)

        logger.info(
            "epoch %d/%d | train_loss=%.6f | val_loss=%.6f | val_pr_auc=%.6f | val_f1@0.5=%.6f | lr=%.2e",
            ep,
            epochs,
            epoch_loss,
            val_loss,
            val_metrics_epoch["pr_auc"],
            val_metrics_epoch["f1"],
            current_lr,
        )

        monitor_value = float(epoch_row[early_stop_monitor])
        if _is_improved(monitor_value, best_monitor_value):
            best_monitor_value = monitor_value
            best_epoch = int(ep)
            epochs_without_improve = 0
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improve += 1

        if bool(early_stop_enabled) and epochs_without_improve >= max(1, int(early_stop_patience)):
            stopped_epoch = int(ep)
            logger.info(
                "Early stopping triggered at epoch %d | monitor=%s best_epoch=%d best=%.6f patience=%d min_delta=%.3e",
                ep,
                early_stop_monitor,
                best_epoch,
                best_monitor_value,
                int(early_stop_patience),
                float(early_stop_min_delta),
            )
            break

    if stopped_epoch == 0:
        stopped_epoch = int(epochs)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info(
            "Loaded best checkpoint from epoch %d by %s=%.6f for final validation/test evaluation",
            best_epoch,
            early_stop_monitor,
            best_monitor_value,
        )

    model.eval()
    with torch.no_grad():
        xt_seq = torch.from_numpy(x_seq_test).to(device)
        xt_static = torch.from_numpy(x_static_test).to(device)
        xt_note = torch.from_numpy(x_note_test).to(device) if x_note_test is not None else None
        
        if xv_note is not None:
            val_probs = torch.sigmoid(model(xv_seq, xv_static, xv_note)).cpu().numpy()
            test_probs = torch.sigmoid(model(xt_seq, xt_static, xt_note)).cpu().numpy()
        else:
            val_probs = torch.sigmoid(model(xv_seq, xv_static)).cpu().numpy()
            test_probs = torch.sigmoid(model(xt_seq, xt_static)).cpu().numpy()

    val_probs_f64 = val_probs.astype(np.float64)
    test_probs_f64 = test_probs.astype(np.float64)

    best_t = _best_threshold_f1(y_val_int, val_probs_f64)
    logger.info("Best F1 threshold on validation set: %.3f", best_t)

    selected_t: float
    threshold_fallback_to_default = False
    threshold_candidate: float | None = None
    if threshold_policy == "fixed_0.5":
        selected_t = 0.5
    elif threshold_policy == "val_best_f1":
        selected_t = float(best_t)
    elif threshold_policy in {"val_recall_floor_precision_max", "val_target_recall_max_precision"}:
        t_candidate = _threshold_max_precision_with_recall_floor(
            y_val_int,
            val_probs_f64,
            min_recall=float(threshold_target_recall),
        )
        threshold_candidate = None if t_candidate is None else float(t_candidate)
        if t_candidate is None:
            logger.warning(
                "No threshold can satisfy recall >= %.3f on validation set; fallback to 0.5",
                float(threshold_target_recall),
            )
            selected_t = 0.5
            threshold_fallback_to_default = True
        else:
            selected_t = float(t_candidate)
    else:
        raise ValueError(f"Unsupported threshold_policy: {threshold_policy}")

    logger.info(
        "Selected operating threshold on validation set: %.3f | policy=%s",
        selected_t,
        threshold_policy,
    )

    val_metrics_default = _classification_metrics(y_val_int, val_probs_f64, threshold=0.5)
    val_metrics_best = _classification_metrics(y_val_int, val_probs_f64, threshold=best_t)
    test_metrics_default = _classification_metrics(y_test_int, test_probs_f64, threshold=0.5)
    test_metrics_val_best = _classification_metrics(y_test_int, test_probs_f64, threshold=best_t)
    val_metrics_selected = _classification_metrics(y_val_int, val_probs_f64, threshold=selected_t)
    test_metrics_selected = _classification_metrics(y_test_int, test_probs_f64, threshold=selected_t)

    result = {
        "samples_total": int(len(x_seq)),
        "samples_train": int(len(x_seq_train)),
        "samples_val": int(len(x_seq_val)),
        "samples_test": int(len(x_seq_test)),
        "positive_ratio_total": float(np.mean(y)),
        "positive_ratio_train": float(np.mean(y_train)),
        "positive_ratio_val": float(np.mean(y_val)),
        "positive_ratio_test": float(np.mean(y_test)),
        "threshold_policy": str(threshold_policy),
        "threshold_target_recall": float(threshold_target_recall),
        "threshold_candidate": None if threshold_candidate is None else float(threshold_candidate),
        "threshold_fallback_to_default": bool(threshold_fallback_to_default),
        "selected_threshold_from_validation": float(selected_t),
        "best_threshold_from_validation": float(best_t),
        "epoch_monitoring": epoch_monitoring,
        "early_stopping": {
            "enabled": bool(early_stop_enabled),
            "monitor": str(early_stop_monitor),
            "monitor_mode": str(monitor_mode),
            "patience": int(early_stop_patience),
            "min_delta": float(early_stop_min_delta),
            "best_epoch": int(best_epoch),
            "best_value": float(best_monitor_value),
            "stopped_epoch": int(stopped_epoch),
        },
        "val_metrics_threshold_0.5": val_metrics_default,
        "val_metrics_best_threshold": val_metrics_best,
        "val_metrics_selected_threshold": val_metrics_selected,
        "test_metrics_threshold_0.5": test_metrics_default,
        "test_metrics_validation_best_threshold": test_metrics_val_best,
        "test_metrics_selected_threshold": test_metrics_selected,
        "test_probs": test_probs_f64.tolist(),
        "epoch_monitoring": epoch_monitoring,
        "state_dict": model.state_dict(),
    }
    return result


def pretrain_transformer_encoder(
    x_seq: np.ndarray,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    mask_prob: float = 0.15,
    hidden_dim: int = 128,
    num_layers: int = 2,
    nhead: int = 4,
    device: torch.device | None = None,
    **model_kwargs,
) -> dict[str, Any]:
    """
    Masked time-series pretraining for the TransformerSeqEncoder.
    Returns the state dict of the pretrained encoder.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    seq_input_dim = x_seq.shape[-1]
    
    encoder = TransformerSeqEncoder(
        seq_input_dim=seq_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
        nhead=nhead,
        **model_kwargs,
    )
    model = TransformerPretrainer(encoder, seq_input_dim, hidden_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='none')
    
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(torch.from_numpy(x_seq))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    logger.info("Starting Masked Time-Series Pretraining (%d epochs)...", epochs)
    
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_items = 0
        
        for (batch_x,) in loader:
            batch_x = batch_x.to(device) # [B, T, F]
            # Generate mask
            mask = torch.rand(batch_x.shape[:2], device=device) < mask_prob
            
            optimizer.zero_grad()
            preds = model(batch_x, mask) # [B, T, F]
            
            # Loss only on masked positions
            loss_all = criterion(preds, batch_x) # [B, T, F]
            loss_all = loss_all.mean(dim=-1) # [B, T]
            
            # Mask out non-masked positions
            masked_loss = (loss_all * mask.float()).sum() / (mask.float().sum() + 1e-8)
            
            masked_loss.backward()
            optimizer.step()
            
            total_loss += masked_loss.item() * batch_x.size(0)
            total_items += batch_x.size(0)
            
        avg_loss = total_loss / total_items
        if epoch % 5 == 0 or epoch == 1:
            logger.info("Pretrain Epoch %d/%d | MSE Loss: %.4f", epoch, epochs, avg_loss)
            
    logger.info("Pretraining completed.")
    return encoder.state_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate Monitoring Agent risk model (Late Fusion)"
    )

    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=3307)
    parser.add_argument("--db-user", default="root")
    parser.add_argument("--db-password", default="hanwen123")
    parser.add_argument("--db-name", default="mimic4")
    parser.add_argument(
        "--data-source",
        choices=["db", "local_snapshot"],
        default="db",
        help="Dataset source: query DB online or load offline parquet snapshot.",
    )
    parser.add_argument(
        "--local-snapshot-dir",
        default="output/local_snapshots/monitoring_risk_latefusion/latest",
        help="Snapshot directory for --data-source=local_snapshot.",
    )

    parser.add_argument("--train-limit", type=int, default=120000)
    parser.add_argument("--max-query-rows", type=int, default=50000)
    parser.add_argument(
        "--sql-in-batch-size",
        type=int,
        default=DEFAULT_SQL_IN_BATCH_SIZE,
        help="Chunk size for SQLAlchemy expanding IN (:stay_ids) queries used by ICD/DRG feature extraction.",
    )
    parser.add_argument("--include-all-chartevents", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-outputevents", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-datetimeevents", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-labevents", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-inputevents", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-omr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-static-demographics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-admission-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-icd-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--icd-top-k", type=int, default=64)
    parser.add_argument(
        "--icd-include-current-hadm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include ICD from the current admission; may introduce leakage if diagnosis coding is post-hoc.",
    )
    parser.add_argument(
        "--include-drg-onehot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include DRG categorical one-hot features from drgcodes for each admission.",
    )
    parser.add_argument("--drg-top-k", type=int, default=64)
    parser.add_argument(
        "--include-procedure-icd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include procedures_icd categorical one-hot features using ICD prefix encoding.",
    )
    parser.add_argument("--procedure-top-k", type=int, default=64)
    parser.add_argument(
        "--include-hcpcs-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include hcpcsevents categorical one-hot features using full HCPCS code encoding.",
    )
    parser.add_argument("--hcpcs-top-k", type=int, default=64)
    parser.add_argument(
        "--include-pharmacy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include prescriptions-derived pharmacy one-hot features (drug/route).",
    )
    parser.add_argument("--pharmacy-top-k", type=int, default=64)
    parser.add_argument(
        "--use-core-channels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use MonitoringAgent.DEFAULT_CORE_CHANNELS for channel pruning.",
    )
    parser.add_argument(
        "--channels",
        default=None,
        help="Comma-separated channel names to override default/core channels, e.g. HR,RR,SPO2,SBP,DBP,MAP.",
    )

    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=720,
        help="Readmission label horizon in hours (default 720 = 30 days).",
    )
    parser.add_argument(
        "--pre-discharge-hours",
        type=int,
        default=48,
        help="Only use records from the final N hours before ICU outtime for feature building.",
    )
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument(
        "--seq-aggregation",
        choices=["last", "first", "full_resample"],
        default="last",
        help=(
            "How to map variable-length stay sequence to fixed seq-len: "
            "last=keep most recent window, first=keep earliest window, "
            "full_resample=compress full stay timeline into seq-len bins."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--loss", choices=["bce", "focal"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument(
        "--threshold-policy",
        choices=["fixed_0.5", "val_best_f1", "val_recall_floor_precision_max", "val_target_recall_max_precision"],
        default="val_target_recall_max_precision",
        help=(
            "Policy for selecting operating threshold from validation set. "
            "fixed_0.5=always 0.5; val_best_f1=legacy behavior; "
            "val_recall_floor_precision_max=max precision under recall floor; "
            "val_target_recall_max_precision=alias of recall-floor policy."
        ),
    )
    parser.add_argument(
        "--threshold-target-recall",
        type=float,
        default=0.25,
        help="Recall floor used when --threshold-policy is recall-target based.",
    )
    parser.add_argument(
        "--early-stop-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable early stopping based on validation monitor metric.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=15,
        help="Stop after this many epochs without monitor improvement.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum monitor change considered as improvement.",
    )
    parser.add_argument(
        "--early-stop-monitor",
        choices=["val_loss", "val_f1_at_0.5", "val_pr_auc"],
        default="val_pr_auc",
        help="Validation metric used by early stopping.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output-dir", default="output/latefusion")
    parser.add_argument(
        "--use-dataset-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache built dataset tensors (x/y) and reuse when dataset-defining args are unchanged.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        default="output/dataset_cache/monitoring_risk_latefusion",
        help="Directory for dataset cache files.",
    )
    
    parser.add_argument(
        "--use-notes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use clinical note embeddings.",
    )
    parser.add_argument(
        "--note-embeddings-path",
        default="output/mimic3_note_embeddings.pkl",
        help="Path to clinical note embeddings pkl file.",
    )
    
    return parser.parse_args()


def _configure_logging(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_log, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    return run_log


def _build_dataset_cache_key_from_args(args: argparse.Namespace) -> str:
    # Only include args that can change dataset content (x/y).
    data_source = str(args.data_source)
    payload = {
        "schema": 7,
        "script": str(Path(__file__).resolve()),
        "data_source": data_source,
        "local_snapshot_dir": str(Path(args.local_snapshot_dir).resolve()) if args.local_snapshot_dir else None,
        "train_limit": int(args.train_limit) if data_source == "db" else None,
        "horizon_hours": int(args.horizon_hours),
        "max_query_rows": int(args.max_query_rows) if data_source == "db" else None,
        "include_all_chartevents": bool(args.include_all_chartevents),
        "include_outputevents": bool(args.include_outputevents),
        "include_datetimeevents": bool(args.include_datetimeevents),
        "include_labevents": bool(args.include_labevents),
        "include_inputevents": bool(args.include_inputevents),
        "include_omr": bool(args.include_omr),
        "include_static_demographics": bool(args.include_static_demographics),
        "include_admission_context": bool(args.include_admission_context),
        "include_icd_history": bool(args.include_icd_history),
        "icd_top_k": int(args.icd_top_k),
        "icd_include_current_hadm": bool(args.icd_include_current_hadm),
        "include_drg_onehot": bool(args.include_drg_onehot),
        "drg_top_k": int(args.drg_top_k),
        "include_procedure_icd": bool(args.include_procedure_icd),
        "procedure_top_k": int(args.procedure_top_k),
        "include_hcpcs_events": bool(args.include_hcpcs_events),
        "hcpcs_top_k": int(args.hcpcs_top_k),
        "include_pharmacy": bool(args.include_pharmacy),
        "pharmacy_top_k": int(args.pharmacy_top_k),
        "pre_discharge_hours": int(args.pre_discharge_hours),
        "use_core_channels": bool(args.use_core_channels),
        "channels": str(args.channels) if args.channels is not None else None,
        "seq_len": int(args.seq_len),
        "seq_aggregation": str(args.seq_aggregation),
        "use_notes": bool(getattr(args, "use_notes", False)),
        "note_embeddings_path": str(getattr(args, "note_embeddings_path", "")) if getattr(args, "use_notes", False) else None,
    }
    if data_source == "db":
        payload.update(
            {
                "db_host": str(args.db_host),
                "db_port": int(args.db_port),
                "db_user": str(args.db_user),
                "db_name": str(args.db_name),
            }
        )
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _load_dataset_cache(cache_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, int]] | None:
    if not cache_file.exists():
        return None
    try:
        with np.load(cache_file) as data:
            x_seq = data["x_seq"].astype(np.float32)
            x_static = data["x_static"].astype(np.float32)
            y = data["y"].astype(np.float32)
            x_note = data["x_note"].astype(np.float32) if "x_note" in data else None
            static_schema_json = data["static_schema_json"].item()
            static_schema = json.loads(static_schema_json)
        return x_seq, x_static, y, x_note, static_schema
    except Exception as exc:
        logger.warning("dataset cache load failed (%s): %s", cache_file, exc)
        return None


def _save_dataset_cache(
    cache_file: Path,
    x_seq: np.ndarray,
    x_static: np.ndarray,
    y: np.ndarray,
    x_note: np.ndarray | None,
    static_schema: dict[str, int],
) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "x_seq": x_seq.astype(np.float32),
        "x_static": x_static.astype(np.float32),
        "y": y.astype(np.float32),
        "static_schema_json": np.asarray(json.dumps(static_schema, ensure_ascii=False)),
    }
    if x_note is not None:
        save_dict["x_note"] = x_note.astype(np.float32)
        
    np.savez(cache_file, **save_dict)


def _load_local_snapshot_dataset(snapshot_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, int]]:
    import pyarrow.parquet as pq

    meta_path = snapshot_dir / "meta.json"
    data_path = snapshot_dir / "dataset_snapshot.parquet"
    if not meta_path.exists() or not data_path.exists():
        raise FileNotFoundError(
            f"Local snapshot files not found under {snapshot_dir}. "
            "Expected meta.json and dataset_snapshot.parquet"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n_samples = int(meta["n_samples"])
    seq_shape = meta["x_seq_shape"]
    static_shape = meta["x_static_shape"]
    static_schema = meta.get("static_schema", {})

    seq_flat_dim = int(seq_shape[1] * seq_shape[2])
    static_dim = int(static_shape[1])
    x_seq = np.zeros((n_samples, int(seq_shape[1]), int(seq_shape[2])), dtype=np.float32)
    x_static = np.zeros((n_samples, static_dim), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.float32)

    pf = pq.ParquetFile(data_path)
    for batch in pf.iter_batches(batch_size=1024):
        b = batch.to_pydict()
        idxs = b["sample_idx"]
        labels = b["label"]
        seq_rows = b["seq_flat"]
        static_rows = b["static_flat"]

        for i, sample_idx in enumerate(idxs):
            si = int(sample_idx)
            seq_arr = np.asarray(seq_rows[i], dtype=np.float32)
            if seq_arr.size != seq_flat_dim:
                raise ValueError(
                    f"Invalid seq_flat length at sample_idx={si}: got {seq_arr.size}, expected {seq_flat_dim}"
                )
            x_seq[si] = seq_arr.reshape(int(seq_shape[1]), int(seq_shape[2]))

            st_arr = np.asarray(static_rows[i], dtype=np.float32)
            if st_arr.size != static_dim:
                raise ValueError(
                    f"Invalid static_flat length at sample_idx={si}: got {st_arr.size}, expected {static_dim}"
                )
            x_static[si] = st_arr
            y[si] = float(labels[i])

    logger.info(
        "Loaded local snapshot dataset from %s | x_seq.shape=%s x_static.shape=%s y.shape=%s",
        snapshot_dir,
        x_seq.shape,
        x_static.shape,
        y.shape,
    )
    return x_seq, x_static, y, None, static_schema


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    run_log_path = _configure_logging(output_dir)
    logger.info("Run log file: %s", run_log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    monitor_agent = MonitoringAgent(seed=args.seed)
    selected_channels = None
    if args.channels:
        selected_channels = [x.strip() for x in str(args.channels).split(",") if x.strip()]
    if selected_channels:
        logger.info("Using explicit channel list (%d): %s", len(selected_channels), ",".join(selected_channels))
    elif bool(args.use_core_channels):
        logger.info(
            "Using core channel subset (%d): %s",
            len(MonitoringAgent.DEFAULT_CORE_CHANNELS),
            ",".join(MonitoringAgent.DEFAULT_CORE_CHANNELS),
        )
    else:
        logger.info("Using full canonical channels (%d)", len(MonitoringAgent.DEFAULT_CHANNELS))
    logger.info("Train and evaluate Monitoring risk model (Late Fusion)")

    x_seq: np.ndarray | None = None
    x_static: np.ndarray | None = None
    x_note: np.ndarray | None = None
    y: np.ndarray | None = None
    static_schema: dict[str, int] | None = None
    cache_key = _build_dataset_cache_key_from_args(args)
    cache_dir = Path(args.dataset_cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    cache_file = cache_dir / f"dataset_{cache_key}.npz"

    cache_data = None
    if bool(args.use_dataset_cache):
        cache_data = _load_dataset_cache(cache_file)
        if cache_data is not None:
            x_seq, x_static, y, x_note, static_schema = cache_data
            logger.info(
                "dataset cache hit | key=%s file=%s x_seq.shape=%s x_static.shape=%s y.shape=%s",
                cache_key,
                cache_file,
                x_seq.shape,
                x_static.shape,
                y.shape,
            )
        else:
            logger.info("dataset cache miss | key=%s file=%s", cache_key, cache_file)

    if cache_data is None:
        if str(args.data_source) == "local_snapshot":
            snapshot_dir = Path(args.local_snapshot_dir)
            if not snapshot_dir.is_absolute():
                snapshot_dir = REPO_ROOT / snapshot_dir
            x_seq, x_static, y, x_note, static_schema = _load_local_snapshot_dataset(snapshot_dir)
        else:
            config = DatabaseConfig(
                host=args.db_host,
                port=args.db_port,
                user=args.db_user,
                password=args.db_password,
                database=args.db_name,
            )
            with DatabaseManager(config) as db:
                session = db.get_session()
                extractor = MIMIC4DataExtractor(session)
                train_rows = fetch_stay_rows(
                    session,
                    limit=int(args.train_limit),
                    horizon_hours=int(args.horizon_hours),
                )
                note_embeddings_dict = None
                if getattr(args, "use_notes", False):
                    note_path = Path(args.note_embeddings_path)
                    if not note_path.is_absolute():
                        note_path = REPO_ROOT / note_path
                    if note_path.exists():
                        logger.info("Loading clinical note embeddings from %s", note_path)
                        with open(note_path, "rb") as f:
                            note_embeddings_dict = pickle.load(f)
                    else:
                        logger.warning("Clinical note embeddings not found at %s. Proceeding without notes.", note_path)

                x_seq, x_static, y, x_note, static_schema = build_dataset(
                    extractor=extractor,
                    monitor_agent=monitor_agent,
                    stay_rows=train_rows,
                    seq_len=int(args.seq_len),
                    max_query_rows=int(args.max_query_rows),
                    include_all_chartevents=bool(args.include_all_chartevents),
                    include_outputevents=bool(args.include_outputevents),
                    include_datetimeevents=bool(args.include_datetimeevents),
                    include_labevents=bool(args.include_labevents),
                    include_inputevents=bool(args.include_inputevents),
                    include_omr=bool(args.include_omr),
                    include_static_demographics=bool(args.include_static_demographics),
                    include_admission_context=bool(args.include_admission_context),
                    include_icd_history=bool(args.include_icd_history),
                    icd_top_k=int(args.icd_top_k),
                    icd_include_current_hadm=bool(args.icd_include_current_hadm),
                    include_drg_onehot=bool(args.include_drg_onehot),
                    drg_top_k=int(args.drg_top_k),
                    include_procedure_icd=bool(args.include_procedure_icd),
                    procedure_top_k=int(args.procedure_top_k),
                    include_hcpcs_events=bool(args.include_hcpcs_events),
                    hcpcs_top_k=int(args.hcpcs_top_k),
                    include_pharmacy=bool(args.include_pharmacy),
                    pharmacy_top_k=int(args.pharmacy_top_k),
                    pre_discharge_hours=int(args.pre_discharge_hours),
                    use_core_channels=bool(args.use_core_channels),
                    selected_channels=selected_channels,
                    seq_aggregation=str(args.seq_aggregation),
                    sql_in_batch_size=int(args.sql_in_batch_size),
                    note_embeddings_dict=note_embeddings_dict,
                )

        if bool(args.use_dataset_cache):
            _save_dataset_cache(cache_file, x_seq, x_static, y, x_note, static_schema or {})
            logger.info("dataset cache saved | key=%s file=%s", cache_key, cache_file)

    if x_seq is None or x_static is None or y is None or static_schema is None:
        raise RuntimeError("dataset build failed: x_seq/x_static/y/static_schema are empty")
        
    model_cls = MonitoringRiskTransformerLateFusion
    if getattr(args, "use_notes", False):
        model_cls = MonitoringRiskTransformerLateFusionWithNotes
        logger.info("Using model: MonitoringRiskTransformerLateFusionWithNotes")
    else:
        logger.info("Using model: MonitoringRiskTransformerLateFusion")

    result = train_and_evaluate(
        x_seq=x_seq,
        x_static=x_static,
        x_note=x_note,
        y=y,
        static_schema=static_schema,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        loss_name=str(args.loss),
        focal_gamma=float(args.focal_gamma),
        focal_alpha=float(args.focal_alpha) if args.focal_alpha is not None else None,
        threshold_policy=str(args.threshold_policy),
        threshold_target_recall=float(args.threshold_target_recall),
        early_stop_enabled=bool(args.early_stop_enabled),
        early_stop_patience=int(args.early_stop_patience),
        early_stop_min_delta=float(args.early_stop_min_delta),
        early_stop_monitor=str(args.early_stop_monitor),
        device=device,
        model_class=model_cls,
    )

    state_dict = result.pop("state_dict")
    ckpt_path = output_dir / "monitoring_risk_latefusion_lstm.pth"
    torch.save(state_dict, ckpt_path)

    result_out = output_dir / "training_eval_metrics.json"
    result_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Saved model checkpoint to %s", ckpt_path)
    logger.info("Saved eval report to %s", result_out)
    logger.info("Validation metrics @0.5: %s", json.dumps(result["val_metrics_threshold_0.5"], ensure_ascii=False))
    logger.info("Validation metrics @best_t: %s", json.dumps(result["val_metrics_best_threshold"], ensure_ascii=False))
    logger.info(
        "Validation metrics @selected_t(policy=%s): %s",
        result["threshold_policy"],
        json.dumps(result["val_metrics_selected_threshold"], ensure_ascii=False),
    )
    logger.info("Test metrics @0.5: %s", json.dumps(result["test_metrics_threshold_0.5"], ensure_ascii=False))
    logger.info(
        "Test metrics @val_best_t: %s",
        json.dumps(result["test_metrics_validation_best_threshold"], ensure_ascii=False),
    )
    logger.info(
        "Test metrics @selected_t(policy=%s): %s",
        result["threshold_policy"],
        json.dumps(result["test_metrics_selected_threshold"], ensure_ascii=False),
    )


if __name__ == "__main__":
    main()
