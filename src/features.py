"""
features.py
-----------
Feature engineering for traffic time-series forecasting.

Input  : per-BS aggregate DataFrame [timestamp, site_id, dl_norm]
Output : feature matrix X (n_samples, n_features) and target y (n_samples, horizon)

Features built (8 total)
------------------------
- dl_norm            : current traffic value
- dl_norm_lag_96     : same slot yesterday (out-of-window)
- dl_norm_lag_192    : same slot 2 days ago (out-of-window)
- hour_sin, hour_cos : time-of-day (cyclic)
- dow_sin, dow_cos   : day-of-week (cyclic)
- is_weekend         : binary

month_sin/cos excluded — dataset spans only ~2.5 months (insufficient for seasonal learning).

Public API
----------
add_time_features(df) -> df with time columns added
add_lag_features(df, lags) -> df with lag columns added
make_sequences(series, seq_len, horizon) -> (X, y)
build_site_sequences(df, site_id, seq_len, horizon, feature_cols) -> (X, y, timestamps)
fit_scaler(X, method) -> scaler
apply_scaler(X, scaler) -> X_scaled
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ---------------------------------------------------------------------------
# Time feature engineering
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Add cyclic and categorical time features derived from timestamp.

    Adds: hour_sin, hour_cos, dow_sin, dow_cos, is_weekend
    (month_sin/cos excluded — dataset spans only ~2.5 months)
    """
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])

    slot_in_day = ts.dt.hour * 4 + ts.dt.minute // 15   # 0–95

    df["hour_sin"]   = np.sin(2 * math.pi * slot_in_day / 96)
    df["hour_cos"]   = np.cos(2 * math.pi * slot_in_day / 96)
    df["dow_sin"]    = np.sin(2 * math.pi * ts.dt.dayofweek / 7)
    df["dow_cos"]    = np.cos(2 * math.pi * ts.dt.dayofweek / 7)
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Lag feature engineering
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame,
    lags: Sequence[int],
    value_col: str = "dl_norm",
    group_col: str = "site_id",
) -> pd.DataFrame:
    """
    Add lag features within each group (site_id).

    Parameters
    ----------
    lags : list of integers (number of 15-min slots to look back)

    Adds columns: dl_norm_lag_1, dl_norm_lag_4, dl_norm_lag_96, ...
    """
    df = df.copy().sort_values([group_col, "timestamp"])
    for lag in lags:
        col_name = f"{value_col}_lag_{lag}"
        df[col_name] = df.groupby(group_col)[value_col].shift(lag)
    return df


# ---------------------------------------------------------------------------
# Sequence building for LSTM
# ---------------------------------------------------------------------------

def make_sequences(
    series: np.ndarray,
    seq_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over a 1-D (or 2-D) time series to produce (X, y) pairs.

    Parameters
    ----------
    series  : 1-D array (T,) or 2-D array (T, n_features)
    seq_len : number of input time steps
    horizon : number of output time steps to forecast

    Returns
    -------
    X : (n_samples, seq_len, n_features)   input sequences
    y : (n_samples, horizon)                forecast targets (dl_norm only)
    """
    if series.ndim == 1:
        series = series[:, np.newaxis]

    T, n_feat = series.shape
    n_samples = T - seq_len - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Series too short ({T}) for seq_len={seq_len} + horizon={horizon}"
        )

    X = np.stack([series[i : i + seq_len] for i in range(n_samples)])
    # Target: first feature (dl_norm) only
    y = np.stack([series[i + seq_len : i + seq_len + horizon, 0] for i in range(n_samples)])

    return X.astype(np.float32), y.astype(np.float32)


def build_site_sequences(
    df: pd.DataFrame,
    site_id: str,
    seq_len: int,
    horizon: int,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X, y) sequences for one site.

    Returns
    -------
    X          : (n_samples, seq_len, n_features)
    y          : (n_samples, horizon)
    timestamps : (n_samples,) — timestamp of the first predicted step
    """
    site_df = df[df["site_id"] == site_id].sort_values("timestamp").dropna(
        subset=feature_cols
    )
    arr        = site_df[feature_cols].values
    timestamps = site_df["timestamp"].values

    X, y = make_sequences(arr, seq_len, horizon)

    # Timestamps: align to the step just after the last input
    pred_ts = timestamps[seq_len : seq_len + len(X)]
    return X, y, pred_ts


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def fit_scaler(X: np.ndarray, method: str = "standard"):
    """Fit and return a scaler on the flattened feature matrix."""
    orig_shape = X.shape
    X_flat = X.reshape(-1, orig_shape[-1])
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "none":
        return None
    else:
        raise ValueError(f"Unknown scaler method: {method}")
    scaler.fit(X_flat)
    return scaler


def apply_scaler(X: np.ndarray, scaler) -> np.ndarray:
    """Apply a fitted scaler to (n_samples, seq_len, n_features) array."""
    if scaler is None:
        return X
    orig_shape = X.shape
    return scaler.transform(X.reshape(-1, orig_shape[-1])).reshape(orig_shape)
