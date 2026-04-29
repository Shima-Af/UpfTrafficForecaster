"""
forecast.py
-----------
Inference API: load trained model and produce multi-step traffic forecasts.

Usage
-----
    forecaster = Forecaster.load("models/best_model.pt", "models/scaler.pkl")
    # df: recent history DataFrame [timestamp, site_id, dl_norm]
    preds = forecaster.predict(df, site_id="102526", n_steps=4)
    # preds: np.ndarray shape (4,) — forecast dl_norm for next 4 slots
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features import (
    add_time_features,
    add_lag_features,
    apply_scaler,
)


class Forecaster:
    """
    Wraps a trained TrafficLSTM for single-site or batch inference.

    Parameters
    ----------
    model        : loaded TrafficLSTM in eval mode
    scaler       : fitted sklearn scaler (or None)
    feature_cols : list of feature column names used during training
    params       : full params dict
    device       : torch.device
    """

    def __init__(self, model, scaler, feature_cols: list[str], params: dict, device):
        self.model        = model
        self.scaler       = scaler
        self.feature_cols = feature_cols
        self.params       = params
        self.device       = device
        self.seq_len      = params["lstm"]["seq_len"]
        self.horizon      = params["lstm"]["horizon"]

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path = "models/best_model.pt",
        scaler_path: str | Path     = "models/scaler.pkl",
        device_str: str             = "auto",
    ) -> "Forecaster":
        from src.models.lstm import TrafficLSTM
        from src.train import get_device

        device = get_device(device_str)
        ckpt   = torch.load(checkpoint_path, map_location=device)

        model = TrafficLSTM.from_params(
            ckpt["params"],
            n_features=ckpt["n_features"],
            n_sites=ckpt["n_sites"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()

        scaler = None
        if Path(scaler_path).exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

        return cls(
            model=model,
            scaler=scaler,
            feature_cols=ckpt["feature_cols"],
            params=ckpt["params"],
            device=device,
        )

    def predict(
        self,
        history_df: pd.DataFrame,
        site_id: str,
        site_idx: int,
    ) -> np.ndarray:
        """
        Produce a forecast from recent history for one site.

        Parameters
        ----------
        history_df : DataFrame [timestamp, site_id, dl_norm] — must contain
                     at least seq_len + max(lags) rows for site_id
        site_id    : target site
        site_idx   : integer site index (as used during training)

        Returns
        -------
        np.ndarray shape (horizon,) — forecast dl_norm
        """
        fp = self.params["features"]

        df = history_df[history_df["site_id"] == site_id].copy()
        df = add_time_features(df)
        df = add_lag_features(df, lags=fp["lags"])
        df = df.dropna(subset=self.feature_cols).sort_values("timestamp")

        if len(df) < self.seq_len:
            raise ValueError(
                f"Need at least {self.seq_len} clean rows for site {site_id}, "
                f"got {len(df)}"
            )

        seq = df[self.feature_cols].values[-self.seq_len:]  # (seq_len, n_feat)
        seq = apply_scaler(seq[np.newaxis], self.scaler)    # (1, seq_len, n_feat)

        x = torch.from_numpy(seq).float().to(self.device)
        s = torch.tensor([site_idx], dtype=torch.long).to(self.device)

        with torch.no_grad():
            out = self.model(x, site_idx=s)                 # (1, horizon)

        return out.cpu().numpy().ravel()

    def predict_batch(
        self,
        history_df: pd.DataFrame,
        site_ids: list[str],
        site_id_to_idx: dict[str, int],
    ) -> dict[str, np.ndarray]:
        """Predict for multiple sites. Returns {site_id: forecast_array}."""
        return {
            sid: self.predict(history_df, sid, site_id_to_idx[sid])
            for sid in site_ids
            if sid in history_df["site_id"].values
        }
