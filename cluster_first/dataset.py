"""
dataset.py — Sliding-window dataset over coarsened cluster time series.

ClusterTrafficDataset wraps the (K, T_total) cluster aggregate signal into
(seq_len, K) input windows and (horizon, K) target windows for STGNN training.

Key design choices vs the original GraphTrafficDataset:
  - Input is (seq_len, K) not (seq_len, N, F): the cluster series has only one
    feature per node (the aggregate dl_norm), so no feature dimension is needed.
  - The graph topology (edge_index, edge_attr) is stored in the dataset and
    returned with every sample so the DataLoader collects it into a batch
    without needing a custom collate function.
  - Split is strictly temporal (train → val → test in time order) to prevent
    any form of temporal leakage from the future into training windows.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class ClusterTrafficDataset(Dataset):
    """
    Sliding-window dataset over the K-cluster aggregate time series.

    Each sample is one (seq_len, K) input window paired with a (horizon, K)
    target window immediately following it.  The model is trained to forecast
    all K cluster totals simultaneously, which directly maps to the input
    each of the K independent PPO UPF controllers will receive.

    Parameters
    ----------
    cluster_series : (K, T_total) numpy float32 — summed dl_norm per cluster
    edge_index     : (2, E_coarse) torch.LongTensor — coarsened graph edges
    edge_attr      : (E_coarse,) torch.FloatTensor — inter-cluster affinities
    seq_len        : number of 15-min slots in each input window (default 96 = 24 h)
    horizon        : number of future slots to predict (default 4 = 1 h ahead)
    split          : one of 'train', 'val', 'test'
    val_split      : fraction of total time for validation
    test_split     : fraction of total time for test
    """

    def __init__(
        self,
        cluster_series: np.ndarray,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        seq_len: int = 96,
        horizon: int = 4,
        split: str = "train",
        val_split: float = 0.15,
        test_split: float = 0.15,
    ) -> None:
        K, T_total = cluster_series.shape

        # ---- Temporal split (no shuffling — time order preserved) ----
        n_test  = max(horizon + seq_len, int(T_total * test_split))
        n_val   = max(horizon + seq_len, int(T_total * val_split))
        n_train = T_total - n_val - n_test

        if n_train <= seq_len + horizon:
            raise ValueError(
                f"Training split too short: T_total={T_total}, "
                f"seq_len={seq_len}, horizon={horizon}, "
                f"val_split={val_split}, test_split={test_split}. "
                f"Try reducing val/test fractions or increasing the dataset."
            )

        if split == "train":
            series_slice = cluster_series[:, :n_train]
        elif split == "val":
            series_slice = cluster_series[:, n_train: n_train + n_val]
        elif split == "test":
            series_slice = cluster_series[:, n_train + n_val:]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        # ---- Build sliding windows ----
        # series_slice: (K, T_split)
        T_split   = series_slice.shape[1]
        n_samples = T_split - seq_len - horizon + 1

        if n_samples <= 0:
            raise ValueError(
                f"Split '{split}' has only {T_split} timesteps, which is not "
                f"enough for seq_len={seq_len} + horizon={horizon}."
            )

        # X[i] : series_slice[:, i : i+seq_len].T  → (seq_len, K)
        # y[i] : series_slice[:, i+seq_len : i+seq_len+horizon].T  → (horizon, K)
        X_list, y_list = [], []
        for i in range(n_samples):
            X_list.append(series_slice[:, i: i + seq_len].T)            # (seq_len, K)
            y_list.append(series_slice[:, i + seq_len: i + seq_len + horizon].T)  # (horizon, K)

        self.X = torch.from_numpy(
            np.stack(X_list, axis=0).astype(np.float32)   # (n_samples, seq_len, K)
        )
        self.y = torch.from_numpy(
            np.stack(y_list, axis=0).astype(np.float32)   # (n_samples, horizon, K)
        )
        self.edge_index = edge_index   # (2, E_coarse)
        self.edge_attr  = edge_attr    # (E_coarse,)

        self.K         = K
        self.seq_len   = seq_len
        self.horizon   = horizon
        self.n_samples = n_samples
        self.split     = split

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dict so collation by DataLoader is straightforward.
        edge_index and edge_attr are the same for every sample (the coarsened
        graph topology doesn't change across time), but returning them here
        keeps the model's forward() signature simple.
        """
        return {
            "x":          self.X[idx],           # (seq_len, K)
            "y":          self.y[idx],            # (horizon, K)
            "edge_index": self.edge_index,        # (2, E_coarse)  — shared
            "edge_attr":  self.edge_attr,         # (E_coarse,)    — shared
        }

    def __repr__(self) -> str:
        return (
            f"ClusterTrafficDataset(split={self.split!r}, "
            f"n_samples={self.n_samples}, K={self.K}, "
            f"seq_len={self.seq_len}, horizon={self.horizon})"
        )
