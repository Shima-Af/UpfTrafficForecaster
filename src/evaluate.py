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


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    mask = np.abs(y_true) > eps
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def sla_compliance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_pct: float = 20.0,
    eps: float = 1e-6,
) -> float:
    """Fraction of predictions within threshold_pct% of true value."""
    mask = np.abs(y_true) > eps
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
        "mape":            mape(y_true, y_pred),
        "sla_compliance":  sla_compliance(y_true, y_pred, threshold_pct),
    }


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    threshold_pct: float = 20.0,
    edge_index=None,
) -> dict:
    """
    Run model on a DataLoader and return metric dict.

    Supports both LSTM (batch=(X,y,site)) and STGNN (batch=(X,y)) loaders.
    Pass edge_index for STGNN; omit (or None) for LSTM.
    """
    model.eval()
    all_pred, all_true = [], []

    for batch in loader:
        if edge_index is not None:
            # STGNN batch: (X, y)  — y shape (B, N, horizon)
            X, y = batch
            X, y = X.to(device), y.to(device)
            pred = model(X, edge_index)
        else:
            # LSTM batch: (X, y, site_idx)  — y shape (B, horizon)
            X, y, site = batch
            X, y, site = X.to(device), y.to(device), site.to(device)
            pred = model(X, site_idx=site)

        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())

    y_pred = np.concatenate(all_pred).ravel()
    y_true = np.concatenate(all_true).ravel()
    return compute_metrics(y_true, y_pred, threshold_pct)
