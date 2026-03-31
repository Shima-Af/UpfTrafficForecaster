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
    from pathlib import Path
    from src.aggregate import aggregate_to_bs

    processed_dir = Path(processed_dir)
    fp = params["features"]
    tp = params["training"]
    lp = params["lstm"]

    # ---- Load and aggregate all parquet files ----
    frames = []
    for pq in sorted(processed_dir.glob("*.parquet")):
        day_df = pd.read_parquet(pq)
        bs_df  = aggregate_to_bs(day_df, voronoi_map)
        frames.append(bs_df)
    df = pd.concat(frames, ignore_index=True)
    df.sort_values(["site_id", "timestamp"], inplace=True)

    # ---- Feature engineering ----
    if fp["include_time_features"]:
        df = add_time_features(df)
    df = add_lag_features(df, lags=fp["lags"])
    df.dropna(inplace=True)

    time_feature_cols = (
        ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
         "month_sin", "month_cos", "is_weekend"]
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
