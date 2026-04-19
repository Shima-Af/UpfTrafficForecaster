"""
dataset.py
----------
PyTorch Dataset and DataLoader wrappers for traffic forecasting.

TrafficDataset  : wraps (X, y) arrays into a PyTorch Dataset
MultiSiteDataset: builds one dataset from all sites, with site embedding index
split_by_date   : temporal train/val/test split (no shuffling across time)
make_loaders    : convenience function → (train_loader, val_loader, test_loader)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.features import (
    add_time_features,
    add_lag_features,
    build_site_sequences,
    fit_scaler,
    apply_scaler,
)

_ONE_WEEK_SLOTS = 96 * 7   # 672 slots = 7 days at 15-min resolution


def _load_bs_parquets(processed_dir) -> pd.DataFrame:
    """
    Load all processed parquets and sum across services to get total
    dl_norm per (site_id, timestamp).

    The processed parquets already contain BS-level data (site_id column)
    with one row per (timestamp, site_id, service).  aggregate_to_bs() must
    NOT be called again — it expects raw tile data with a cell_id column.
    """
    from pathlib import Path
    frames = []
    for pq in sorted(Path(processed_dir).glob("*.parquet")):
        frames.append(pd.read_parquet(pq))
    df = pd.concat(frames, ignore_index=True)
    df = (
        df.groupby(["timestamp", "site_id"], sort=False)["dl_norm"]
          .sum()
          .reset_index()
    )
    df.sort_values(["site_id", "timestamp"], inplace=True)
    return df


def apply_prev_week_fill(
    df: pd.DataFrame,
    anomaly_dates: list[str] | None = None,
    value_col: str = "dl_norm",
    timestamp_col: str = "timestamp",
    group_col: str = "site_id",
) -> pd.DataFrame:
    """
    Replace missing/anomalous values with the same slot from the previous week.

    Two cases handled:
    1. Anomaly dates (e.g. May 12 data-collection outage): ALL values for that
       calendar date are replaced with the value from exactly 7 days earlier.
    2. Remaining NaN values: filled with the value 7 days prior; if still NaN,
       14 days prior; if still NaN, forward-fill as a last resort.

    Parameters
    ----------
    df           : combined long-format DataFrame sorted by (site_id, timestamp)
    anomaly_dates: list of 'YYYYMMDD' strings to treat as full-day outages
    """
    df = df.copy().sort_values([group_col, timestamp_col])
    anomaly_dates = anomaly_dates or []

    # --- Mark anomaly-date rows as NaN so the fill logic handles them uniformly ---
    if anomaly_dates:
        anomaly_ts = set()
        for date_str in anomaly_dates:
            d = pd.Timestamp(date_str)
            # Flag the known outage window: 01:00–18:00 on the anomaly date
            # (00:00–00:45 and 18:15–23:45 were normal — preserve them)
            start = d + pd.Timedelta(hours=1)
            end   = d + pd.Timedelta(hours=18)
            mask = (
                (pd.to_datetime(df[timestamp_col]) >= start) &
                (pd.to_datetime(df[timestamp_col]) <= end)
            )
            df.loc[mask, value_col] = float("nan")

    # --- Build a lookup: (site_id, timestamp) → dl_norm for fast prev-week access ---
    df = df.set_index([group_col, timestamp_col])

    def _fill_series(s: pd.Series) -> pd.Series:
        """Fill NaN entries using prev-week (7d then 14d) then ffill."""
        site_id = s.index.get_level_values(0)[0]
        result = s.copy()
        nan_mask = result.isna()
        if not nan_mask.any():
            return result
        for lag_slots in [_ONE_WEEK_SLOTS, 2 * _ONE_WEEK_SLOTS]:
            still_nan = result.isna()
            if not still_nan.any():
                break
            lag_td = pd.Timedelta(minutes=15 * lag_slots)
            for ts in still_nan[still_nan].index.get_level_values(1):
                donor_key = (site_id, ts - lag_td)
                if donor_key in df.index:
                    result.loc[(site_id, ts)] = df.loc[donor_key, value_col]
        # Last-resort forward fill for any remaining NaN
        result = result.ffill()
        return result

    df[value_col] = df.groupby(level=0)[value_col].transform(_fill_series)
    df = df.reset_index()
    return df


class TrafficDataset(Dataset):
    """
    Simple dataset wrapping (X, y) arrays.

    X : (n_samples, seq_len, n_features)  float32
    y : (n_samples, horizon)              float32
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class MultiSiteDataset(Dataset):
    """
    Dataset spanning multiple sites.

    Returns (X, y, site_idx) where site_idx is an integer embedding index.
    """

    def __init__(
        self,
        X: np.ndarray,          # (n_total_samples, seq_len, n_features)
        y: np.ndarray,          # (n_total_samples, horizon)
        site_indices: np.ndarray,  # (n_total_samples,) int
    ):
        self.X    = torch.from_numpy(X)
        self.y    = torch.from_numpy(y)
        self.site = torch.from_numpy(site_indices.astype(np.int64))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.site[idx]


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def split_by_date(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame chronologically into train / val / test.

    No shuffling — time order is preserved to prevent leakage.
    """
    dates = np.sort(df[timestamp_col].unique())
    n     = len(dates)
    n_test = max(1, int(n * test_fraction))
    n_val  = max(1, int(n * val_fraction))
    n_train = n - n_val - n_test

    if n_train <= 0:
        raise ValueError(
            f"Not enough dates ({n}) for val={val_fraction} test={test_fraction}"
        )

    train_dates = dates[:n_train]
    val_dates   = dates[n_train : n_train + n_val]
    test_dates  = dates[n_train + n_val :]

    train_df = df[df[timestamp_col].isin(train_dates)]
    val_df   = df[df[timestamp_col].isin(val_dates)]
    test_df  = df[df[timestamp_col].isin(test_dates)]
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Build all datasets from processed parquet files
# ---------------------------------------------------------------------------

def build_datasets(
    processed_dir: str,
    params: dict,
    voronoi_map,
) -> tuple["MultiSiteDataset", "MultiSiteDataset", "MultiSiteDataset", object, list[str]]:
    """
    Load processed parquets, aggregate to BS level, engineer features,
    split by date, build MultiSiteDatasets.

    Parameters
    ----------
    processed_dir : path to data/netmob/processed/
    params        : full params dict
    voronoi_map   : pd.Series (tile_id → site_id)

    Returns
    -------
    train_ds, val_ds, test_ds, scaler, feature_cols
    """
    fp = params["features"]
    tp = params["training"]
    lp = params["lstm"]
    dq = params.get("data_quality", {})

    # ---- Load processed parquets (already BS-level; sum across services) ----
    df = _load_bs_parquets(processed_dir)

    # ---- Cross-day missing value fill (prev-week + anomaly dates) ----
    if dq.get("fill_cross_day") == "prev_week":
        anomaly_dates = dq.get("anomaly_dates", [])
        df = apply_prev_week_fill(df, anomaly_dates=anomaly_dates)

    # ---- Feature engineering ----
    if fp["include_time_features"]:
        df = add_time_features(df)
    df = add_lag_features(df, lags=fp["lags"])
    df.dropna(inplace=True)

    time_feature_cols = (
        ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
        if fp["include_time_features"] else []
    )
    lag_cols      = [f"dl_norm_lag_{lag}" for lag in fp["lags"]]
    feature_cols  = ["dl_norm"] + lag_cols + time_feature_cols

    # ---- Temporal split ----
    train_df, val_df, test_df = split_by_date(
        df,
        val_fraction=tp["val_fraction"],
        test_fraction=tp["test_fraction"],
    )

    # ---- Build sequences per site ----
    seq_len  = lp["seq_len"]
    horizon  = lp["horizon"]
    sites    = sorted(df["site_id"].unique())
    site2idx = {s: i for i, s in enumerate(sites)}

    def _build(split_df):
        Xs, ys, sidxs = [], [], []
        for site in sites:
            if site not in split_df["site_id"].values:
                continue
            try:
                X, y, _ = build_site_sequences(split_df, site, seq_len, horizon, feature_cols)
            except ValueError:
                continue
            Xs.append(X)
            ys.append(y)
            sidxs.append(np.full(len(X), site2idx[site], dtype=np.int64))
        if not Xs:
            raise RuntimeError("No sequences built — check data and seq_len/horizon")
        return (
            np.concatenate(Xs),
            np.concatenate(ys),
            np.concatenate(sidxs),
        )

    X_tr, y_tr, s_tr = _build(train_df)
    X_va, y_va, s_va = _build(val_df)
    X_te, y_te, s_te = _build(test_df)

    # ---- Fit scaler on training set ----
    scaler = fit_scaler(X_tr, method=fp["scaler"])
    X_tr   = apply_scaler(X_tr, scaler)
    X_va   = apply_scaler(X_va, scaler)
    X_te   = apply_scaler(X_te, scaler)

    return (
        MultiSiteDataset(X_tr, y_tr, s_tr),
        MultiSiteDataset(X_va, y_va, s_va),
        MultiSiteDataset(X_te, y_te, s_te),
        scaler,
        feature_cols,
    )


class GraphTrafficDataset(Dataset):
    """
    Dataset for STGNN training — each sample covers ALL nodes at once.

    Unlike MultiSiteDataset (which yields one site per sample), every
    sample here is a full graph snapshot: seq_len steps across N nodes.

    Shapes
    ------
    X : (n_samples, seq_len, N, n_features)  float32
    y : (n_samples, N, horizon)              float32
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Build graph datasets from processed parquets (STGNN)
# ---------------------------------------------------------------------------

def build_graph_datasets(
    processed_dir: str,
    params: dict,
    voronoi_map,
    node_index_df,
) -> tuple["GraphTrafficDataset", "GraphTrafficDataset", "GraphTrafficDataset", object, list[str], int]:
    """
    Build train/val/test GraphTrafficDatasets for STGNN training.

    Each sample is one sliding-window slice across ALL N gNodeBs simultaneously.
    The node ordering follows node_index_df (site_id → node_idx).

    Parameters
    ----------
    processed_dir  : path to data/netmob/processed/
    params         : full params dict
    voronoi_map    : pd.Series (tile_id → site_id)
    node_index_df  : pd.DataFrame with columns [site_id, node_idx]

    Returns
    -------
    train_ds, val_ds, test_ds, scaler, feature_cols, n_nodes
    """
    from src.features import add_time_features, add_lag_features

    fp = params["features"]
    tp = params["training"]
    sp = params["stgnn"]
    dq = params.get("data_quality", {})

    # ---- Load processed parquets (already BS-level; sum across services) ----
    df = _load_bs_parquets(processed_dir)

    # ---- Cross-day fill ----
    if dq.get("fill_cross_day") == "prev_week":
        df = apply_prev_week_fill(df, anomaly_dates=dq.get("anomaly_dates", []))

    # ---- Feature engineering ----
    if fp["include_time_features"]:
        df = add_time_features(df)
    df = add_lag_features(df, lags=fp["lags"])
    df.dropna(inplace=True)

    time_feature_cols = (
        ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
        if fp["include_time_features"] else []
    )
    lag_cols     = [f"dl_norm_lag_{lag}" for lag in fp["lags"]]
    feature_cols = ["dl_norm"] + lag_cols + time_feature_cols

    # ---- Map site_id → node_idx ----
    site2node = dict(zip(node_index_df["site_id"], node_index_df["node_idx"]))
    N = len(node_index_df)

    df["node_idx"] = df["site_id"].map(site2node)
    df = df.dropna(subset=["node_idx"])
    df["node_idx"] = df["node_idx"].astype(int)

    # ---- Build (T_total, N, F) aligned tensor ----
    all_timestamps = np.sort(df["timestamp"].unique())
    T_total        = len(all_timestamps)
    F              = len(feature_cols)
    ts2idx         = {ts: i for i, ts in enumerate(all_timestamps)}

    arr = np.zeros((T_total, N, F), dtype=np.float32)
    t_idx = df["timestamp"].map(ts2idx).values.astype(int)
    n_idx = df["node_idx"].values
    arr[t_idx, n_idx, :] = df[feature_cols].values.astype(np.float32)

    # Forward-fill any node-time slots that have no data (zeros → carry last value)
    for n in range(N):
        node_arr = arr[:, n, :]          # (T, F)
        # Find timesteps that were never written (all zeros on dl_norm)
        zero_mask = node_arr[:, 0] == 0.0
        if zero_mask.any() and not zero_mask.all():
            for f_i in range(F):
                s = pd.Series(node_arr[:, f_i])
                s[zero_mask] = np.nan
                node_arr[:, f_i] = s.ffill().bfill().values
            arr[:, n, :] = node_arr

    # ---- Temporal split by date ----
    n_test  = max(1, int(T_total * tp["test_fraction"]))
    n_val   = max(1, int(T_total * tp["val_fraction"]))
    n_train = T_total - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"Not enough timesteps ({T_total}) for given split fractions")

    # ---- Fit scaler on TRAINING portion of the raw (T, N, F) array ----
    # Fitting on the small (T_train*N, F) array avoids the gigantic
    # (n_samples * seq_len * N, F) reshape that causes OOM for large graphs.
    scaler = fit_scaler(arr[:n_train].reshape(-1, F), method=fp["scaler"])

    def _scale_arr(a: np.ndarray) -> np.ndarray:
        """Apply scaler to (T, N, F) → returns scaled copy."""
        sh = a.shape
        return apply_scaler(a.reshape(-1, F), scaler).reshape(sh)

    arr_train = _scale_arr(arr[:n_train])
    arr_val   = _scale_arr(arr[n_train : n_train + n_val])
    arr_te    = _scale_arr(arr[n_train + n_val :])

    seq_len = sp["seq_len"]
    horizon = sp["horizon"]

    def _slide(a: np.ndarray):
        """Slide window over (T, N, F) → X:(n, seq_len, N, F), y:(n, N, horizon)."""
        T = len(a)
        n_samples = T - seq_len - horizon + 1
        if n_samples <= 0:
            raise ValueError(f"Split too short ({T}) for seq_len={seq_len}+horizon={horizon}")
        X = np.stack([a[i      : i + seq_len]          for i in range(n_samples)])  # (n, seq, N, F)
        y = np.stack([a[i+seq_len : i+seq_len+horizon, :, 0] for i in range(n_samples)])  # (n, hor, N)
        y = y.transpose(0, 2, 1)   # → (n, N, horizon)
        return X.astype(np.float32), y.astype(np.float32)

    X_tr, y_tr = _slide(arr_train)
    X_va, y_va = _slide(arr_val)
    X_te, y_te = _slide(arr_te)

    print(f"[build_graph_datasets] N={N}  F={F}  "
          f"Train={len(X_tr)}  Val={len(X_va)}  Test={len(X_te)}")

    return (
        GraphTrafficDataset(X_tr, y_tr),
        GraphTrafficDataset(X_va, y_va),
        GraphTrafficDataset(X_te, y_te),
        scaler,
        feature_cols,
        N,
    )


def make_loaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
