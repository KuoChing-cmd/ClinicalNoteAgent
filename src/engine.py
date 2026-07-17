import logging

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def _compute_prauc(y_true, y_score):
    """Compute area under precision-recall curve via trapezoidal rule."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(y_score)[::-1]
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recalls, precisions))


def train_model(
    model,
    X_seq,
    Y,
    X_static=None,
    X_mh=None,
    X_note=None,
    X_seq_val=None,
    Y_val=None,
    X_static_val=None,
    X_mh_val=None,
    X_note_val=None,
    epochs=12,
    lr=1e-3,
    batch_size=256,
    pos_weight=None,
    early_stop_patience=30,
    early_stop_min_delta=1e-4,
):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"Using device: {device} for training")

    tensors = [
        torch.tensor(X_seq, dtype=torch.float32).to(device),
        torch.tensor(Y, dtype=torch.float32).to(device),
    ]

    if X_static is not None:
        tensors.append(torch.tensor(X_static, dtype=torch.float32).to(device))
    else:
        tensors.append(torch.zeros(len(Y), 1).to(device))

    if X_mh is not None:
        tensors.append(torch.tensor(X_mh, dtype=torch.float32).to(device))
    else:
        tensors.append(torch.zeros(len(Y), 1).to(device))

    if X_note is not None:
        tensors.append(torch.tensor(X_note, dtype=torch.float32).to(device))

    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Prepare validation data on device (if provided)
    has_val = X_seq_val is not None and Y_val is not None
    if has_val:
        val_tensors = [
            torch.tensor(X_seq_val, dtype=torch.float32).to(device),
            torch.tensor(Y_val, dtype=torch.float32).to(device),
        ]
        if X_static_val is not None:
            val_tensors.append(
                torch.tensor(X_static_val, dtype=torch.float32).to(device)
            )
        else:
            val_tensors.append(torch.zeros(len(Y_val), 1).to(device))
        if X_mh_val is not None:
            val_tensors.append(torch.tensor(X_mh_val, dtype=torch.float32).to(device))
        else:
            val_tensors.append(torch.zeros(len(Y_val), 1).to(device))
        if X_note_val is not None:
            val_tensors.append(torch.tensor(X_note_val, dtype=torch.float32).to(device))
        val_dataset = TensorDataset(*val_tensors)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    # Optimization ③: class-imbalance-aware loss
    pw = (
        torch.tensor([pos_weight], dtype=torch.float32).to(device)
        if pos_weight is not None
        else None
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    if pos_weight is not None:
        logging.info(
            f"  pos_weight = {pos_weight:.2f} (neg/pos ratio, correcting class imbalance)"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Optimization ②: cosine annealing LR — starts at lr, decays smoothly to 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    epoch_history = []

    # ── Early stopping state ──────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improve = 0
    best_state_dict = None
    if has_val:
        logging.info(
            f"  early_stopping: patience={early_stop_patience}, min_delta={early_stop_min_delta:.1e}"
        )

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_s, y = batch[0], batch[1]
            x_st = batch[2] if X_static is not None else None
            x_m = batch[3] if X_mh is not None else None
            x_n = (
                batch[4] if X_note is not None else None
            )  # Bug1 fix: len(batch) guard was unreliable

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
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "lr": float(f"{current_lr:.2e}"),
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
                    vx_m = vbatch[3] if X_mh_val is not None else None
                    vx_n = vbatch[4] if X_note_val is not None else None

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

            row["val_loss"] = round(val_loss, 6)
            row["val_auroc"] = round(val_auroc, 4)
            row["val_prauc"] = round(val_prauc, 4)

            # ── Early stopping check ─────────────────────────────────────────
            if val_loss < (best_val_loss - early_stop_min_delta):
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epochs_without_improve = 0
                best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                epochs_without_improve += 1

            row["best_val_loss"] = round(best_val_loss, 6)
            row["patience_counter"] = epochs_without_improve

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
                row["early_stopped"] = True
                epoch_history.append(row)
                break
        else:
            logging.info(
                f"Epoch {epoch+1}/{epochs} completed - Loss: {train_loss:.4f}  LR: {current_lr:.2e}"
            )

        epoch_history.append(row)

    # ── Restore best checkpoint ───────────────────────────────────────────────
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logging.info(
            f"✓ Restored best model from epoch {best_epoch} "
            f"(val_loss={best_val_loss:.6f})"
        )

    return model, epoch_history


def pretrain_transformer(
    encoder,
    X_seq,
    seq_dim,
    hidden_dim,
    epochs=10,
    lr=1e-3,
    batch_size=256,
    mask_prob=0.15,
):
    """
    Pretrain the TransformerSeqEncoder using masked time-series reconstruction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrainer = TransformerPretrainer(encoder, seq_dim, hidden_dim).to(device)

    tensors = [torch.tensor(X_seq, dtype=torch.float32).to(device)]
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(pretrainer.parameters(), lr=lr)

    pretrainer.train()
    logging.info(f"Starting Pretraining for {epochs} epochs (mask_prob={mask_prob})...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            x_s = batch[0]
            # Create random mask
            B, T, _ = x_s.shape
            mask = torch.rand(B, T, device=device) < mask_prob

            optimizer.zero_grad()
            # Forward pass
            reconstructed = pretrainer(x_s, mask)

            # Compute loss only on masked elements
            if mask.sum() > 0:
                loss = criterion(reconstructed[mask], x_s[mask])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        logging.info(
            f"Pretrain Epoch {epoch+1}/{epochs} completed - MSE Loss: {epoch_loss / max(1, len(loader)):.4f}"
        )

    logging.info("Pretraining completed.")
    return encoder


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
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    accuracy = (tp + tn) / total
    specificity = tn / max(1, tn + fp)

    # Brier score
    brier = float(np.mean((y_prob - y_true) ** 2))

    return {
        "threshold": round(threshold, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(_compute_prauc(y_true, y_prob), 4),
        "brier": round(brier, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_model(
    model, X_seq, Y, X_static=None, X_mh=None, X_note=None, batch_size=512
):
    """
    Evaluate a PyTorch model. Returns a dict containing:
      - y_prob: raw predicted probabilities
      - best_threshold: F1-optimal threshold
      - metrics_at_best_t: full metrics at best threshold
      - metrics_at_0.5: full metrics at 0.5
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    tensors = [
        torch.tensor(X_seq, dtype=torch.float32).to(device),
        torch.tensor(Y, dtype=torch.float32).to(device),
    ]

    if X_static is not None:
        tensors.append(torch.tensor(X_static, dtype=torch.float32).to(device))
    else:
        tensors.append(torch.zeros(len(Y), 1).to(device))

    if X_mh is not None:
        tensors.append(torch.tensor(X_mh, dtype=torch.float32).to(device))
    else:
        tensors.append(torch.zeros(len(Y), 1).to(device))

    if X_note is not None:
        tensors.append(torch.tensor(X_note, dtype=torch.float32).to(device))

    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            x_s = batch[0]
            x_st = batch[2] if X_static is not None else None
            x_m = batch[3] if X_mh is not None else None
            x_n = batch[4] if X_note is not None else None

            preds = torch.sigmoid(model(x_s, x_st, x_m, x_n))
            all_preds.append(preds.cpu().numpy())

    y_prob = np.concatenate(all_preds)
    y_true = np.asarray(Y, dtype=np.int64)

    best_t = _search_best_threshold_f1(y_true, y_prob)
    metrics_best = _compute_classification_metrics(y_true, y_prob, threshold=best_t)
    metrics_05 = _compute_classification_metrics(y_true, y_prob, threshold=0.5)

    return {
        "y_prob": y_prob,
        "best_threshold": best_t,
        "metrics_at_best_t": metrics_best,
        "metrics_at_0.5": metrics_05,
    }


def _evaluate_xgb_probs(y_true, y_prob):
    """Compute full metrics for XGBoost predictions (already have probabilities)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    best_t = _search_best_threshold_f1(y_true, y_prob)
    return {
        "best_threshold": best_t,
        "metrics_at_best_t": _compute_classification_metrics(
            y_true, y_prob, threshold=best_t
        ),
        "metrics_at_0.5": _compute_classification_metrics(
            y_true, y_prob, threshold=0.5
        ),
    }


def _log_eval_result(name, result):
    """Log evaluation metrics for one model."""
    m_best = result["metrics_at_best_t"]
    m_05 = result["metrics_at_0.5"]
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
