"""
dataset.py
----------
Shared data loading and PyTorch Dataset utilities for traffic forecasting.

_load_bs_parquets : load processed parquets → single dl_norm per (site_id, timestamp)
apply_prev_week_fill : fill anomaly dates with previous-week values
TrafficDataset    : wraps (X, y) arrays into a PyTorch Dataset
split_by_date     : temporal train/val/test split (no shuffling across time)
make_loaders      : convenience function → (train_loader, val_loader, test_loader)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

_ONE_WEEK_SLOTS = 96 * 7   # 672 slots = 7 days at 15-min resolution


def _load_bs_parquets(processed_dir, service: str = "total") -> pd.DataFrame:
    """
    Load all processed parquets and return a single dl_norm per (site_id, timestamp).

    Parameters
    ----------
    processed_dir : path to data/netmob/processed/
    service       : "total" to sum all services, or a specific service name
                    (e.g. "Netflix", "YouTube", "DailyMotion") to use that
                    service's signal only.

    With global_max_all_services normalization all service dl_norm values share
    the same scale, so summing is a direct addition — no byte conversion needed.
    """
    frames = []
    for pq in sorted(Path(processed_dir).glob("*.parquet")):
        frames.append(pd.read_parquet(pq))
    df = pd.concat(frames, ignore_index=True)

    if service == "total":
        df = (
            df.groupby(["timestamp", "site_id"], sort=False)["dl_norm"]
              .sum()
              .reset_index()
        )
    else:
        df = df[df["service"] == service][["timestamp", "site_id", "dl_norm"]].copy()

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
