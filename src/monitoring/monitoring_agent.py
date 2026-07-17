from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


class MonitoringLSTMPolicy(nn.Module):
    """LSTM policy that outputs discharge aggressiveness in [0, 1]."""

    def __init__(
        self, input_size: int = 3, hidden_size: int = 64, num_layers: int = 1
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        """Args: [batch, seq_len, 3] -> Returns: [batch]"""
        _, (hidden, _) = self.lstm(sequence_features)
        last_hidden = hidden[-1]
        out = self.head(last_hidden)
        return out.squeeze(-1)


@dataclass(frozen=True)
class CandidateSelection:
    aggressiveness: float
    risk_threshold: float
    transfer_capacity: int | None
    eligible_patient_ids: list[str]
    selected_patient_ids: list[str]


class MonitoringAgent:
    """ICU Monitoring Agent with risk-driven transfer selection.

    Risk model outputs per-patient risk: risk_i in [0, 1].

    Policy layer inputs:
    - risk vector {patient_id: risk_i}
    - ICU occupancy rate
    - queue pressure
    - transfer bed constraint (optional)

    Decision rule (deterministic):
    - Eligible set E = {i | r_i_stay(t) <= delta}
    - pressure score a_mon = f(occupancy, queue_pressure)
    - k = floor(a_mon * |E|), then clamp by transfer_capacity if provided
    - Select top-k lowest-risk patients from E
    """

    def __init__(
        self,
        *,
        risk_threshold: float = 0.35,
        hidden_size: int = 64,
        num_layers: int = 1,
        device: str | None = None,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        self.risk_threshold = float(np.clip(risk_threshold, 0.0, 1.0))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Keep legacy attributes for compatibility with older checkpoints/config paths.
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

    # Keep a stable canonical channel order for training/evaluation.
    DEFAULT_CHANNELS: list[str] = [
        # Vital signs
        "HR",
        "RR",
        "SPO2",
        "TEMP",
        # Hemodynamics
        "SBP",
        "DBP",
        "MAP",
        "CVP",
        "PAP",
        "PAPS",
        "PAPD",
        "PAWP",
        "ICP",
        # Blood gases
        "pH",
        "PCO2",
        "PO2",
        "HCO3",
        # Metabolic
        "GLUCOSE",
        "LACTATE",
        # Renal
        "CREATININE",
        "BUN",
        # Electrolytes
        "K",
        "NA",
        "CL",
        "CA",
        "MG",
        "PHOS",
        # Cell counts
        "WBC",
        "HGB",
        "HCT",
        "PLT",
        # Coagulation
        "PT",
        "PTT",
        # Liver
        "ALB",
        "BILI",
        "AST",
        "ALT",
        "ALP",
        # Output
        "UO",
        "DRAIN",
    ]

    # High-density + clinically core channels used for default pruning.
    DEFAULT_CORE_CHANNELS: list[str] = [
        "HR",
        "RR",
        "SPO2",
        "TEMP",
        "SBP",
        "DBP",
        "MAP",
        "GLUCOSE",
        "LACTATE",
        "CREATININE",
        "BUN",
        "K",
        "NA",
        "CL",
        "WBC",
        "HGB",
        "PLT",
        "PTT",
        "HCO3",
        "UO",
    ]

    @staticmethod
    def _normalize_sign_value(feature: str, value: float) -> float:
        """Map heterogeneous ICU event values to [0, 1] sign-intensity scale.

        Supports 40+ clinical variables with physiologically-informed normalization ranges.
        """
        name = str(feature or "").strip().lower()
        v = float(value)

        # Vital signs
        if "spo2" in name or "o2 saturation" in name:
            return float(np.clip((v - 70.0) / 30.0, 0.0, 1.0))
        if "heart rate" in name:
            return float(np.clip((v - 40.0) / 140.0, 0.0, 1.0))
        if "respiratory rate" in name:
            return float(np.clip((v - 8.0) / 32.0, 0.0, 1.0))
        if "temperature" in name:
            return float(np.clip((v - 34.0) / 8.0, 0.0, 1.0))

        # Hemodynamics (pressure in mmHg)
        is_map_alias = (
            "mean arterial pressure" in name
            or "arterial blood pressure mean" in name
            or "non invasive blood pressure mean" in name
            or "nibp mean" in name
            or "art bp mean" in name
            or "mean bp (vad)" in name
            or "hm ii- mean bp" in name
            or " map" in f" {name}"
            or "map " in f" {name}"
            or name == "map"
        )
        is_non_map_mean_pressure = (
            "mean airway pressure" in name
            or "pulmonary artery pressure mean" in name
            or "pa mean pressure" in name
            or "pcwp" in name
            or "ra (mean) pressure" in name
        )
        if is_map_alias and not is_non_map_mean_pressure:
            return float(np.clip((v - 40.0) / 80.0, 0.0, 1.0))
        if (
            "blood pressure" in name
            or "arterial blood pressure" in name
            or "nibp" in name
        ) and "systolic" in name:
            return float(np.clip((v - 70.0) / 110.0, 0.0, 1.0))
        if (
            "blood pressure" in name
            or "arterial blood pressure" in name
            or "nibp" in name
        ) and "diastolic" in name:
            return float(np.clip((v - 30.0) / 70.0, 0.0, 1.0))
        if "central venous pressure" in name or "cvp" in name or "jvp" in name:
            return float(np.clip((v - 2.0) / 12.0, 0.0, 1.0))
        if "pulmonary artery pressure" in name or "pap" in name:
            if "systolic" in name:
                return float(np.clip((v - 15.0) / 35.0, 0.0, 1.0))
            if "diastolic" in name:
                return float(np.clip((v - 5.0) / 20.0, 0.0, 1.0))
            return float(np.clip((v - 10.0) / 30.0, 0.0, 1.0))
        if ("wedge" in name or "occlusion" in name) and "pressure" in name:
            return float(np.clip((v - 5.0) / 20.0, 0.0, 1.0))
        if "intracranial pressure" in name or "icp" in name:
            return float(np.clip((v - 5.0) / 15.0, 0.0, 1.0))

        # Blood gases (ABG/VBG)
        if "ph" in name and ("blood" in name or "arterial" in name or "venous" in name):
            return float(np.clip((v - 7.2) / 0.3, 0.0, 1.0))
        if "pco2" in name or "pco₂" in name or "partial pressure co2" in name:
            return float(np.clip((v - 30.0) / 40.0, 0.0, 1.0))
        if "po2" in name or "po₂" in name or "partial pressure o2" in name:
            return float(np.clip((v - 50.0) / 100.0, 0.0, 1.0))
        if (
            "bicarbonate" in name
            or "hco3" in name
            or "hco₃" in name
            or "co2 content" in name
        ):
            return float(np.clip((v - 15.0) / 25.0, 0.0, 1.0))

        # Metabolic
        if "glucose" in name or "blood glucose" in name or "blood sugar" in name:
            return float(np.clip((v - 70.0) / 180.0, 0.0, 1.0))
        if "lactate" in name or "lactic acid" in name:
            return float(np.clip((v - 0.5) / 4.0, 0.0, 1.0))

        # Renal
        is_creat = "creatinine" in name
        is_creat_excluded = (
            "clearance" in name
            or "ratio" in name
            or "urine" in name
            or "ascites" in name
            or "body fluid" in name
            or "csf" in name
            or "joint fluid" in name
            or "pleural" in name
            or "stool" in name
            or "24 hr" in name
        )
        if is_creat and not is_creat_excluded:
            return float(np.clip((v - 0.5) / 3.0, 0.0, 1.0))
        if "bun" in name or "urea nitrogen" in name:
            return float(np.clip((v - 10.0) / 50.0, 0.0, 1.0))

        # Electrolytes (mEq/L)
        if "potassium" in name or "k+" in name:
            return float(np.clip((v - 3.0) / 2.0, 0.0, 1.0))
        if "sodium" in name or "na+" in name:
            return float(np.clip((v - 130.0) / 20.0, 0.0, 1.0))
        if "chloride" in name or "cl-" in name:
            return float(np.clip((v - 95.0) / 20.0, 0.0, 1.0))
        if "calcium" in name or "ca2+" in name:
            return float(np.clip((v - 7.0) / 3.0, 0.0, 1.0))
        if "magnesium" in name or "mg2+" in name:
            return float(np.clip((v - 1.5) / 1.5, 0.0, 1.0))
        if "phosphate" in name or "phosphorus" in name:
            return float(np.clip((v - 2.0) / 3.0, 0.0, 1.0))

        # Cell counts
        if "wbc" in name or "white blood cell" in name:
            return float(np.clip((v - 4.0) / 12.0, 0.0, 1.0))
        if "hemoglobin" in name or "hgb" in name:
            return float(np.clip((v - 10.0) / 8.0, 0.0, 1.0))
        if "hematocrit" in name or "hct" in name:
            return float(np.clip((v - 25.0) / 40.0, 0.0, 1.0))
        if "platelet" in name or "plt" in name:
            return float(np.clip((v - 100.0) / 150.0, 0.0, 1.0))

        # Coagulation (time in seconds)
        if "pt" in name and "partial thromboplastin" not in name:
            return float(np.clip((v - 12.0) / 10.0, 0.0, 1.0))
        if "ptt" in name or "partial thromboplastin time" in name or "aptt" in name:
            return float(np.clip((v - 30.0) / 30.0, 0.0, 1.0))
        if "inr" in name:
            return float(np.clip((v - 1.0) / 2.0, 0.0, 1.0))

        # Liver
        if "albumin" in name:
            return float(np.clip((v - 2.5) / 2.0, 0.0, 1.0))
        if "bilirubin" in name:
            return float(np.clip((v - 0.3) / 2.0, 0.0, 1.0))
        if "ast" in name or "serum glutamic oxaloacetic" in name:
            return float(np.clip((v - 30.0) / 100.0, 0.0, 1.0))
        if "alt" in name or "serum glutamic pyruvic" in name:
            return float(np.clip((v - 30.0) / 100.0, 0.0, 1.0))
        if "alp" in name or "alkaline phosphatase" in name:
            return float(np.clip((v - 40.0) / 80.0, 0.0, 1.0))

        # Output (volume in mL)
        if (
            "urine output" in name
            or "foley" in name
            or "urine out" in name
            or "u/o" in name
            or ("urine" in name and ("output" in name or "volume" in name))
        ):
            return float(np.clip(v / 500.0, 0.0, 1.0))
        if "drain" in name or "drainage" in name:
            return float(np.clip(v / 500.0, 0.0, 1.0))

        # Default: clip to [0,1]
        return float(np.clip(v, 0.0, 1.0))

    @staticmethod
    def _coerce_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value.strip():
            try:
                return datetime.fromisoformat(value.strip())
            except Exception:
                return None
        return None

    @staticmethod
    def _to_1d_series(values: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            raise ValueError("sign_series must contain at least one element")
        return np.clip(arr, 0.0, 1.0)

    def build_sequence_features(
        self,
        *,
        sign_series: Sequence[float] | np.ndarray,
        icu_occupancy_rate: float,
        queue_pressure: float,
    ) -> torch.Tensor:
        """Builds [1, seq_len, 3] tensor for LSTM input."""
        x = self._to_1d_series(sign_series)
        seq_len = x.shape[0]

        occ = float(np.clip(float(icu_occupancy_rate), 0.0, 1.0))
        pressure = float(np.clip(float(queue_pressure), 0.0, 1.0))

        occ_col = np.full((seq_len,), occ, dtype=np.float32)
        pressure_col = np.full((seq_len,), pressure, dtype=np.float32)
        features = np.stack([x, occ_col, pressure_col], axis=1)

        return torch.tensor(
            features, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

    def build_hourly_sign_series_from_xt(
        self,
        *,
        xt_payload: Mapping[str, Any],
        step_hours: int = 1,
    ) -> np.ndarray:
        """Resample event-style X_t payload to fixed-step hourly sign series."""
        if step_hours <= 0:
            raise ValueError("step_hours must be positive")

        windows = (
            xt_payload.get("windows", []) if isinstance(xt_payload, Mapping) else []
        )
        events = xt_payload.get("x_t", []) if isinstance(xt_payload, Mapping) else []
        if not windows:
            raise ValueError("xt_payload.windows is empty")

        valid_starts: list[datetime] = []
        valid_ends: list[datetime] = []
        for w in windows:
            if not isinstance(w, Mapping):
                continue
            s = self._coerce_datetime(w.get("intime"))
            e = self._coerce_datetime(w.get("outtime"))
            if s is not None:
                valid_starts.append(s)
            if e is not None:
                valid_ends.append(e)
        window_start = min(valid_starts) if valid_starts else None
        window_end = max(valid_ends) if valid_ends else None
        if window_start is None or window_end is None or window_end <= window_start:
            raise ValueError("xt_payload has invalid ICU windows")

        step_seconds = int(step_hours) * 3600
        duration_seconds = (window_end - window_start).total_seconds()
        n_steps = int(np.floor(duration_seconds / step_seconds)) + 1
        n_steps = max(1, n_steps)

        buckets: list[list[float]] = [[] for _ in range(n_steps)]
        for row in events:
            if not isinstance(row, Mapping):
                continue
            charttime = self._coerce_datetime(row.get("charttime"))
            if charttime is None or charttime < window_start or charttime > window_end:
                continue
            val = row.get("value")
            feature = str(row.get("feature") or "")
            if val is None:
                continue
            try:
                norm_v = self._normalize_sign_value(feature, float(val))
            except Exception:
                continue

            idx = int((charttime - window_start).total_seconds() // step_seconds)
            idx = max(0, min(idx, n_steps - 1))
            buckets[idx].append(norm_v)

        series = np.zeros((n_steps,), dtype=np.float32)
        global_values = [x for bucket in buckets for x in bucket]
        fallback = float(np.mean(global_values)) if global_values else 0.0

        last_value = fallback
        for i, bucket in enumerate(buckets):
            if bucket:
                last_value = float(np.mean(bucket))
            series[i] = last_value

        return np.clip(series, 0.0, 1.0)

    @staticmethod
    def _classify_vital(feature: str) -> str | None:
        """Map feature name to vital sign channel.

        Supports 20+ clinical channels across vital signs, hemodynamics, blood gas, labs, and output.
        Returns channel name or None if unrecognized.
        """
        name = str(feature or "").strip().lower()

        # Vital signs (CARDIO-RESPIRATORY)
        if "heart rate" in name:
            return "HR"
        if "respiratory rate" in name:
            return "RR"
        if "spo2" in name or "o2 saturation" in name:
            return "SPO2"
        if "temperature" in name:
            return "TEMP"

        # Blood Pressure
        if "arterial blood pressure" in name:
            if "systolic" in name:
                return "SBP"
            if "diastolic" in name:
                return "DBP"
        if "non invasive blood pressure" in name or "nibp" in name:
            if "systolic" in name:
                return "SBP"
            if "diastolic" in name:
                return "DBP"
        is_map_alias = (
            "mean arterial pressure" in name
            or "arterial blood pressure mean" in name
            or "non invasive blood pressure mean" in name
            or "nibp mean" in name
            or "art bp mean" in name
            or "mean bp (vad)" in name
            or "hm ii- mean bp" in name
            or " map" in f" {name}"
            or "map " in f" {name}"
            or name == "map"
        )
        is_non_map_mean_pressure = (
            "mean airway pressure" in name
            or "pulmonary artery pressure mean" in name
            or "pa mean pressure" in name
            or "pcwp" in name
            or "ra (mean) pressure" in name
        )
        if is_map_alias and not is_non_map_mean_pressure:
            return "MAP"

        # Invasive hemodynamics
        if "central venous pressure" in name or "cvp" in name or "jvp" in name:
            return "CVP"
        if "pulmonary artery pressure" in name or "pap" in name:
            if "systolic" in name:
                return "PAPS"
            if "diastolic" in name:
                return "PAPD"
            if "wedge" in name or "occlusion" in name:
                return "PAWP"
            return "PAP"
        if "intracranial pressure" in name or "icp" in name:
            return "ICP"
        if "jugular venous pressure" in name:
            return "CVP"

        # Blood gases (ABG/VBG)
        if "ph" in name and ("blood" in name or "arterial" in name or "venous" in name):
            return "pH"
        if "pco2" in name or "pCO2" in name or "partial pressure co2" in name:
            return "PCO2"
        if "po2" in name or "pO2" in name or "partial pressure o2" in name:
            return "PO2"
        if (
            "bicarbonate" in name
            or "hco3" in name
            or "hco₃" in name
            or "co2 content" in name
        ):
            return "HCO3"

        # Metabolic labs
        if "glucose" in name or "blood glucose" in name or "blood sugar" in name:
            return "GLUCOSE"
        if "lactate" in name or "lactic acid" in name:
            return "LACTATE"
        is_creat = "creatinine" in name
        is_creat_excluded = (
            "clearance" in name
            or "ratio" in name
            or "urine" in name
            or "ascites" in name
            or "body fluid" in name
            or "csf" in name
            or "joint fluid" in name
            or "pleural" in name
            or "stool" in name
            or "24 hr" in name
        )
        if is_creat and not is_creat_excluded:
            return "CREATININE"
        if "bun" in name or "urea nitrogen" in name:
            return "BUN"

        # Electrolytes
        if "potassium" in name or "k+" in name:
            return "K"
        if "sodium" in name or "na+" in name:
            return "NA"
        if "chloride" in name or "cl-" in name:
            return "CL"
        if "calcium" in name or "ca2+" in name:
            return "CA"
        if "magnesium" in name or "mg2+" in name:
            return "MG"
        if "phosphate" in name or "phosphorus" in name:
            return "PHOS"

        # Cell counts
        if "wbc" in name or "white blood cell" in name:
            return "WBC"
        if "hemoglobin" in name or "hgb" in name:
            return "HGB"
        if "hematocrit" in name or "hct" in name:
            return "HCT"
        if "platelet" in name or "plt" in name:
            return "PLT"

        # Coagulation
        if (
            "pt" in name
            and "partial thromboplastin" not in name
            and ("inr" in name or "prothrombin" in name)
        ):
            return "PT"
        if "ptt" in name or "partial thromboplastin time" in name or "aptt" in name:
            return "PTT"

        # Liver
        if "albumin" in name:
            return "ALB"
        if "bilirubin" in name:
            return "BILI"
        if "ast" in name or "serum glutamic oxaloacetic" in name:
            return "AST"
        if "alt" in name or "serum glutamic pyruvic" in name:
            return "ALT"
        if "alp" in name or "alkaline phosphatase" in name:
            return "ALP"

        # Output
        if (
            "urine output" in name
            or "foley" in name
            or "urine out" in name
            or "u/o" in name
            or ("urine" in name and ("output" in name or "volume" in name))
        ):
            return "UO"
        if "drains" in name or "drain output" in name:
            return "DRAIN"

        return None

    def build_hourly_vital_channels_with_mask_from_xt(
        self,
        *,
        xt_payload: Mapping[str, Any],
        step_hours: int = 1,
        selected_channels: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample X_t payload to fixed-step hourly time series with separate channels and missing mask.

        Supports 40+ clinical channels across:
        - Vital signs (HR, RR, SpO2, TEMP)
        - Hemodynamics (BP, CVP, PAP, MAP, ICP)
        - Blood gases (pH, PCO2, PO2, HCO3)
        - Metabolic (Glucose, Lactate)
        - Labs (Creatinine, BUN, Electrolytes, CBC, Liver)
        - Output (UO, Drains)

        Returns:
            values: np.ndarray of shape (seq_len, n_channels)
            mask: np.ndarray of shape (seq_len, n_channels), observed=1.0, filled=0.0
        """
        if step_hours <= 0:
            raise ValueError("step_hours must be positive")

        windows = (
            xt_payload.get("windows", []) if isinstance(xt_payload, Mapping) else []
        )
        events = xt_payload.get("x_t", []) if isinstance(xt_payload, Mapping) else []
        if not windows:
            raise ValueError("xt_payload.windows is empty")

        valid_starts: list[datetime] = []
        valid_ends: list[datetime] = []
        for w in windows:
            if not isinstance(w, Mapping):
                continue
            s = self._coerce_datetime(w.get("intime"))
            e = self._coerce_datetime(w.get("outtime"))
            if s is not None:
                valid_starts.append(s)
            if e is not None:
                valid_ends.append(e)
        window_start = min(valid_starts) if valid_starts else None
        window_end = max(valid_ends) if valid_ends else None
        if window_start is None or window_end is None or window_end <= window_start:
            raise ValueError("xt_payload has invalid ICU windows")

        if selected_channels is None:
            channels = list(self.DEFAULT_CHANNELS)
        else:
            canonical = set(self.DEFAULT_CHANNELS)
            channels = [
                str(ch).strip()
                for ch in selected_channels
                if str(ch).strip() in canonical
            ]
            if not channels:
                raise ValueError("selected_channels is empty after canonical filtering")
        n_channels = len(channels)
        channel_to_idx = {ch: i for i, ch in enumerate(channels)}

        step_seconds = int(step_hours) * 3600
        duration_seconds = (window_end - window_start).total_seconds()
        n_steps = int(np.floor(duration_seconds / step_seconds)) + 1
        n_steps = max(1, n_steps)

        # buckets[i][j] = list of values in step i for channel j
        buckets: list[list[list[float]]] = [
            [[] for _ in range(n_channels)] for _ in range(n_steps)
        ]

        for row in events:
            if not isinstance(row, Mapping):
                continue
            charttime = self._coerce_datetime(row.get("charttime"))
            if charttime is None or charttime < window_start or charttime > window_end:
                continue
            val = row.get("value")
            feature = str(row.get("feature") or "")
            if val is None:
                continue

            vital = self._classify_vital(feature)
            if vital is None:
                continue
            if vital not in channel_to_idx:
                continue
            ch_idx = channel_to_idx[vital]

            try:
                norm_v = self._normalize_sign_value(feature, float(val))
            except Exception:
                continue

            idx = int((charttime - window_start).total_seconds() // step_seconds)
            idx = max(0, min(idx, n_steps - 1))
            buckets[idx][ch_idx].append(norm_v)

        # Aggregate by mean within each bucket, then forward-fill.
        # mask[i, j] indicates whether this timestep/channel had direct observations.
        series = np.zeros((n_steps, n_channels), dtype=np.float32)
        mask = np.zeros((n_steps, n_channels), dtype=np.float32)

        # Compute fallback (global mean) for each channel
        fallbacks = np.zeros((n_channels,), dtype=np.float32)
        for ch_idx in range(n_channels):
            global_values = [x for step_data in buckets for x in step_data[ch_idx]]
            fallbacks[ch_idx] = float(np.mean(global_values)) if global_values else 0.5

        last_values = fallbacks.copy()
        for i in range(n_steps):
            for ch_idx in range(n_channels):
                if buckets[i][ch_idx]:
                    last_values[ch_idx] = float(np.mean(buckets[i][ch_idx]))
                    mask[i, ch_idx] = 1.0
                series[i, ch_idx] = last_values[ch_idx]

        return np.clip(series, 0.0, 1.0), mask

    def build_hourly_vital_channels_from_xt(
        self,
        *,
        xt_payload: Mapping[str, Any],
        step_hours: int = 1,
    ) -> np.ndarray:
        """Backward-compatible wrapper returning only values (without missing mask)."""
        values, _ = self.build_hourly_vital_channels_with_mask_from_xt(
            xt_payload=xt_payload,
            step_hours=step_hours,
        )
        return values

    def build_sequence_features_from_xt(
        self,
        *,
        xt_payload: Mapping[str, Any],
        icu_occupancy_rate: float,
        queue_pressure: float,
        step_hours: int = 1,
    ) -> torch.Tensor:
        """Convert extractor payload to fixed-step tensor [1, seq_len, 3]."""
        sign_series = self.build_hourly_sign_series_from_xt(
            xt_payload=xt_payload,
            step_hours=step_hours,
        )
        return self.build_sequence_features(
            sign_series=sign_series,
            icu_occupancy_rate=icu_occupancy_rate,
            queue_pressure=queue_pressure,
        )

    def build_sequence_features_from_mimic(
        self,
        *,
        extractor: Any,
        hadm_id: int,
        stay_id: int | None = None,
        icu_occupancy_rate: float,
        queue_pressure: float,
        step_hours: int = 1,
        vital_itemids: Sequence[int] | None = None,
        include_outputevents: bool = True,
        include_datetimeevents: bool = False,
        include_labevents: bool = False,
        lab_itemids: Sequence[int] | None = None,
        max_rows: int = 50000,
    ) -> torch.Tensor:
        """One-stop pipeline: MIMIC query -> fixed-step tensor for LSTM."""
        xt_payload = extractor.build_icu_xt_series(
            hadm_id=hadm_id,
            stay_id=stay_id,
            vital_itemids=vital_itemids,
            include_outputevents=include_outputevents,
            include_datetimeevents=include_datetimeevents,
            include_labevents=include_labevents,
            lab_itemids=lab_itemids,
            max_rows=max_rows,
        )
        return self.build_sequence_features_from_xt(
            xt_payload=xt_payload,
            icu_occupancy_rate=icu_occupancy_rate,
            queue_pressure=queue_pressure,
            step_hours=step_hours,
        )

    @torch.no_grad()
    def predict_aggressiveness(
        self,
        *,
        sign_series: Sequence[float] | np.ndarray,
        icu_occupancy_rate: float,
        queue_pressure: float,
    ) -> float:
        # sign_series is intentionally not used in the risk-driven policy.
        _ = sign_series
        occ = float(np.clip(float(icu_occupancy_rate), 0.0, 1.0))
        pressure = float(np.clip(float(queue_pressure), 0.0, 1.0))
        a_mon = 0.6 * occ + 0.4 * pressure
        return float(np.clip(a_mon, 0.0, 1.0))

    @staticmethod
    def _compute_transfer_count(
        *,
        n_eligible: int,
        aggressiveness: float,
        transfer_capacity: int | None,
    ) -> int:
        if n_eligible <= 0:
            return 0
        k = int(np.floor(float(np.clip(aggressiveness, 0.0, 1.0)) * n_eligible))
        k = max(0, min(k, n_eligible))
        if transfer_capacity is not None:
            cap = max(0, int(transfer_capacity))
            k = min(k, cap)
        return k

    def select_transfer_candidates(
        self,
        *,
        stay_risk_by_patient: Mapping[str, float],
        icu_occupancy_rate: float | None = None,
        queue_pressure: float | None = None,
        transfer_capacity: int | None = None,
        aggressiveness: float | None = None,
        risk_threshold: float | None = None,
    ) -> CandidateSelection:
        """Apply threshold + risk ranking + capacity-constrained top-k selection."""
        threshold = (
            float(np.clip(risk_threshold, 0.0, 1.0))
            if risk_threshold is not None
            else self.risk_threshold
        )

        if aggressiveness is None:
            occ = 0.0 if icu_occupancy_rate is None else float(icu_occupancy_rate)
            qp = 0.0 if queue_pressure is None else float(queue_pressure)
            a_mon = self.predict_aggressiveness(
                sign_series=np.array([0.0], dtype=np.float32),
                icu_occupancy_rate=occ,
                queue_pressure=qp,
            )
        else:
            a_mon = float(np.clip(aggressiveness, 0.0, 1.0))

        eligible_with_risk = []
        for pid, risk in stay_risk_by_patient.items():
            r = float(risk)
            if r <= threshold:
                eligible_with_risk.append((str(pid), r))

        eligible_with_risk.sort(key=lambda item: (item[1], item[0]))
        eligible = [pid for pid, _ in eligible_with_risk]

        n_eligible = len(eligible)
        k = self._compute_transfer_count(
            n_eligible=n_eligible,
            aggressiveness=a_mon,
            transfer_capacity=transfer_capacity,
        )
        selected = eligible[:k]

        return CandidateSelection(
            aggressiveness=a_mon,
            risk_threshold=threshold,
            transfer_capacity=(
                None if transfer_capacity is None else int(max(0, transfer_capacity))
            ),
            eligible_patient_ids=eligible,
            selected_patient_ids=selected,
        )

    def decide_transfer_out(
        self,
        *,
        sign_series: Sequence[float] | np.ndarray | None = None,
        icu_occupancy_rate: float,
        queue_pressure: float,
        stay_risk_by_patient: Mapping[str, float],
        transfer_capacity: int | None = None,
    ) -> CandidateSelection:
        # sign_series is kept only for backward compatibility of callsites.
        _ = sign_series
        return self.select_transfer_candidates(
            stay_risk_by_patient=stay_risk_by_patient,
            icu_occupancy_rate=icu_occupancy_rate,
            queue_pressure=queue_pressure,
            transfer_capacity=transfer_capacity,
        )


# Keep typo-compatible alias for external callers using the original wording.
MoniteringAgent = MonitoringAgent
