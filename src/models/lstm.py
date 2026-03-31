"""
lstm.py
-------
LSTM-based traffic forecaster.

Architecture
------------
  Input  : (batch, seq_len, n_features)
  LSTM   : num_layers × hidden_size (uni- or bi-directional)
  Head   : Linear(hidden_size → horizon)
  Output : (batch, horizon)   — normalised dl_norm forecasts

Optionally wraps a site embedding (for multi-site training) that is
concatenated to the LSTM output before the head.

Usage
-----
    model = TrafficLSTM.from_params(params, n_features=9, n_sites=779)
    out   = model(x, site_idx=site_idx)   # out: (batch, horizon)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TrafficLSTM(nn.Module):
    """
    Multi-layer LSTM encoder with a linear forecasting head.

    Parameters
    ----------
    n_features    : number of input features per timestep
    hidden_size   : LSTM hidden state size
    num_layers    : number of stacked LSTM layers
    horizon       : number of future slots to predict
    dropout       : dropout between LSTM layers (ignored if num_layers == 1)
    bidirectional : use bidirectional LSTM
    n_sites       : if > 0, learn a site embedding concatenated to the LSTM output
    site_emb_dim  : embedding dimension for site ID
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        horizon: int = 4,
        dropout: float = 0.2,
        bidirectional: bool = False,
        n_sites: int = 0,
        site_emb_dim: int = 16,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.n_directions  = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.use_site_emb = n_sites > 0
        if self.use_site_emb:
            self.site_emb = nn.Embedding(n_sites, site_emb_dim)
            head_in = hidden_size * self.n_directions + site_emb_dim
        else:
            head_in = hidden_size * self.n_directions

        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(dropout),
            nn.Linear(head_in, horizon),
        )

    def forward(
        self,
        x: torch.Tensor,
        site_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : (batch, seq_len, n_features)
        site_idx : (batch,) int64  — required if n_sites > 0

        Returns
        -------
        (batch, horizon)
        """
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * n_directions, batch, hidden_size)
        # Take last layer, concatenate directions if bidirectional
        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (batch, 2*hidden)
        else:
            h = h_n[-1]                                  # (batch, hidden)

        if self.use_site_emb:
            if site_idx is None:
                raise ValueError("site_idx required when n_sites > 0")
            emb = self.site_emb(site_idx)                # (batch, site_emb_dim)
            h   = torch.cat([h, emb], dim=-1)

        return self.head(h)                              # (batch, horizon)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_params(
        cls,
        params: dict,
        n_features: int,
        n_sites: int = 0,
    ) -> "TrafficLSTM":
        lp = params["lstm"]
        return cls(
            n_features=n_features,
            hidden_size=lp["hidden_size"],
            num_layers=lp["num_layers"],
            horizon=lp["horizon"],
            dropout=lp["dropout"],
            bidirectional=lp["bidirectional"],
            n_sites=n_sites,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
