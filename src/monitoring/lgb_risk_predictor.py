import os
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np


class LightGBMRiskPredictor:
    """
    Wrapper for the trained LightGBM readmission/discharge risk model.
    Designed to be easily plugged into the simulation system (MonitoringAgent).
    """

    def __init__(self, model_path: str):
        """
        Initialize the predictor by loading a saved LightGBM Booster.

        Args:
            model_path: Path to the .txt model file (e.g. 'model_lgb_base.txt' or 'model_lgb_notes.txt')
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LightGBM model file not found at: {model_path}")

        print(f"Loading LightGBM model from {model_path}...")
        self.model = lgb.Booster(model_file=model_path)
        self.model_path = model_path

    def _flatten_features(
        self, X_seq: np.ndarray, X_static: np.ndarray, X_mh: np.ndarray
    ) -> np.ndarray:
        """
        Internal function to flatten sequences and concatenate with static/multi-hot features.
        This must match the logic in train_readmission_mimic3_with_notes.py exactly.

        Args:
            X_seq: [batch_size, seq_len, seq_channels]
            X_static: [batch_size, static_dim]
            X_mh: [batch_size, mh_dim]
        Returns:
            Flattened array [batch_size, (2 * seq_channels) + static_dim + mh_dim]
        """
        # Ensure correct dimensionality
        if X_seq.ndim == 2:
            X_seq = np.expand_dims(X_seq, axis=0)
        if X_static.ndim == 1:
            X_static = np.expand_dims(X_static, axis=0)
        if X_mh.ndim == 1:
            X_mh = np.expand_dims(X_mh, axis=0)

        seq_mean = np.mean(X_seq, axis=1)
        seq_last = X_seq[:, -1, :]

        return np.concatenate([seq_mean, seq_last, X_static, X_mh], axis=1)

    def predict_batch(
        self,
        X_seq: np.ndarray,
        X_static: np.ndarray,
        X_mh: np.ndarray,
        X_note: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predicts readmission/discharge risk for a batch of patients.

        Args:
            X_seq: Sequential features [batch, seq_len, channels]
            X_static: Static demographic/context features [batch, static_dim]
            X_mh: Multi-hot ICD/DRG/Procedure features [batch, mh_dim]
            X_note: (Optional) Clinical notes embeddings. If the model was trained with notes,
                    this must be provided. [batch, note_dim]

        Returns:
            np.ndarray of risk probabilities [batch]
        """
        X_base = self._flatten_features(X_seq, X_static, X_mh)

        if X_note is not None:
            if X_note.ndim == 1:
                X_note = np.expand_dims(X_note, axis=0)
            X_input = np.concatenate([X_base, X_note], axis=1)
        else:
            X_input = X_base

        # LightGBM predict returns probabilities for objective='binary'
        probs = self.model.predict(X_input)
        return probs

    def predict_for_simulation(
        self, patients_data: Dict[str, Dict[str, np.ndarray]], use_notes: bool = False
    ) -> Dict[str, float]:
        """
        Helper method specifically for the simulation system.
        Takes a dictionary mapping patient_id -> features, and returns patient_id -> risk.

        Args:
            patients_data: A dict where key is patient_id and value is a dict with keys
                           'X_seq', 'X_static', 'X_mh' (and 'X_note' if use_notes=True).
                           These should be unbatched arrays (e.g. X_seq is [seq_len, channels]).
            use_notes: Whether to look for and concatenate 'X_note'.

        Returns:
            Dict[str, float]: Mapping from patient_id to predicted risk probability.
                              This can be passed directly to MonitoringAgent.select_transfer_candidates.
        """
        if not patients_data:
            return {}

        pids = list(patients_data.keys())
        X_seq_list = []
        X_static_list = []
        X_mh_list = []
        X_note_list = []

        for pid in pids:
            data = patients_data[pid]
            X_seq_list.append(data["X_seq"])
            X_static_list.append(data["X_static"])
            X_mh_list.append(data["X_mh"])
            if use_notes:
                if "X_note" not in data:
                    raise ValueError(
                        f"use_notes is True but 'X_note' missing for patient {pid}"
                    )
                X_note_list.append(data["X_note"])

        # Stack into batches
        batch_seq = np.stack(X_seq_list)
        batch_static = np.stack(X_static_list)
        batch_mh = np.stack(X_mh_list)
        batch_note = np.stack(X_note_list) if use_notes else None

        probs = self.predict_batch(batch_seq, batch_static, batch_mh, batch_note)

        # Convert back to dict
        risk_by_patient = {pid: float(prob) for pid, prob in zip(pids, probs)}
        return risk_by_patient


# --- Example Usage for Simulation ---
if __name__ == "__main__":
    # Example snippet showing how it plugs into MonitoringAgent
    from src.monitoring.monitoring_agent import MonitoringAgent

    print("This is a module meant to be imported into your simulation code.")
    print("Example usage:")
    print("""
    # 1. Initialize predictor with the latest trained model
    predictor = LightGBMRiskPredictor("output/20260625_162712_ep200_lr5em04_hd128_bs256_seqnorm_cosinelr_posw_icuload_clinicalbert/model_lgb_base.txt")
    
    # 2. Collect features for patients currently in ICU (mock data)
    current_icu_patients = {
        "stay_101": {
            "X_seq": np.random.rand(48, 8),
            "X_static": np.random.rand(10),
            "X_mh": np.random.rand(64)
        },
        "stay_102": {
            "X_seq": np.random.rand(48, 8),
            "X_static": np.random.rand(10),
            "X_mh": np.random.rand(64)
        }
    }
    
    # 3. Predict risk for all patients
    stay_risk_dict = predictor.predict_for_simulation(current_icu_patients)
    print("Predicted Risks:", stay_risk_dict)
    
    # 4. Pass to MonitoringAgent for transfer decision
    agent = MonitoringAgent(risk_threshold=0.35)
    selection = agent.select_transfer_candidates(
        stay_risk_by_patient=stay_risk_dict,
        icu_occupancy_rate=0.85,
        queue_pressure=0.6,
        transfer_capacity=1
    )
    
    print("Selected for transfer out:", selection.selected_patient_ids)
    """)
