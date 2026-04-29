"""
evaluate.py
-----------
Evaluation metrics for traffic forecasting.

Metrics
-------
  MAE   : mean absolute error
  RMSE  : root mean squared error
  MAPE  : mean absolute percentage error (ignores near-zero targets)
  SLA   : fraction of predictions within sla_threshold_pct of true value
"""

from __future__ import annotations

import numpy as np
import torch


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error — robust to near-zero denominators."""
    denom = np.sum(np.abs(y_true))
    if denom < 1e-10:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def sla_compliance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_pct: float = 20.0,
    min_active: float = 0.01,
) -> float:
    """Fraction of active predictions (dl_norm >= min_active) within threshold_pct% of true value."""
    mask = y_true >= min_active
    if mask.sum() == 0:
        return float("nan")
    rel_err = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
    return float(np.mean(rel_err <= threshold_pct))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_pct: float = 20.0,
) -> dict:
    return {
        "mae":             mae(y_true, y_pred),
        "rmse":            rmse(y_true, y_pred),
        "wape":            wape(y_true, y_pred),
        "sla_compliance":  sla_compliance(y_true, y_pred, threshold_pct),
    }


def _inverse_dl_norm(arr: np.ndarray, scaler) -> np.ndarray:
    """Inverse-transform dl_norm values (feature index 0) from scaled → original space."""
    return np.clip(arr * scaler.scale_[0] + scaler.mean_[0], 0.0, None)


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    threshold_pct: float = 20.0,
    edge_index=None,
    scaler=None,
) -> dict:
    """
    Run model on a DataLoader and return metric dict.

    Supports both LSTM (batch=(X,y,site)) and STGNN (batch=(X,y)) loaders.
    Pass edge_index for STGNN; omit (or None) for LSTM.
    Pass scaler to compute metrics in original dl_norm space (recommended).
    """
    model.eval()
    all_pred, all_true = [], []

    for batch in loader:
        if edge_index is not None:
            X, y = batch
            X, y = X.to(device), y.to(device)
            pred = model(X, edge_index)
        else:
            X, y, site = batch
            X, y, site = X.to(device), y.to(device), site.to(device)
            pred = model(X, site_idx=site)

        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())

    y_pred = np.concatenate(all_pred).ravel()
    y_true = np.concatenate(all_true).ravel()

    if scaler is not None:
        y_pred = _inverse_dl_norm(y_pred, scaler)
        y_true = _inverse_dl_norm(y_true, scaler)

    return compute_metrics(y_true, y_pred, threshold_pct)
