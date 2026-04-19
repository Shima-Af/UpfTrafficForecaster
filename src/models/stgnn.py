"""
stgnn.py
--------
Spatiotemporal Graph Attention Network for per-gNodeB traffic forecasting.

Architecture (temporal-then-spatial)
--------------------------------------
1. Temporal encoder  : GRU reads each node's seq_len-step history independently.
                       All B*N (batch × node) sequences processed in one GRU call.
2. Spatial encoder   : Two-layer Graph Attention Network (GAT) mixes information
                       across connected gNodeB neighbours.
3. Output head       : Linear(hidden → horizon) applied per node.

Input / output shapes
---------------------
  x          : (B, seq_len, N, n_features)   — one time window per sample
  edge_index : (2, E)                         — Voronoi-adjacency graph (fixed)
  output     : (B, N, horizon)               — per-node forecasts

The same edge_index is tiled into a block-diagonal batch graph inside forward(),
so no PyG Data/Batch objects are needed at training time — just plain tensors.

References
----------
  Veličković et al. "Graph Attention Networks" (ICLR 2018)
  Wu et al.        "Graph WaveNet" (IJCAI 2019)  — adaptive adjacency idea
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _tile_edge_index(edge_index: torch.Tensor, n_nodes: int, batch_size: int) -> torch.Tensor:
    """
    Replicate edge_index for a batch of independent graphs.

    Creates a block-diagonal adjacency with B copies of the N-node graph.
    Each copy is offset by i*n_nodes so nodes don't share edges across samples.

    Parameters
    ----------
    edge_index : (2, E)   source graph
    n_nodes    : N        number of nodes in the single graph
    batch_size : B

    Returns
    -------
    (2, B*E)  batched edge_index
    """
    offsets = torch.arange(batch_size, device=edge_index.device) * n_nodes  # (B,)
    # edge_index: (2, E) → (1, 2, E); offsets: (B,) → (B, 1, 1)
    ei_batch = edge_index.unsqueeze(0) + offsets.view(batch_size, 1, 1)      # (B, 2, E)
    return ei_batch.reshape(2, -1)                                             # (2, B*E)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TrafficSTGNN(nn.Module):
    """
    Spatiotemporal GAT forecaster for the Lyon gNodeB graph.

    Parameters
    ----------
    n_nodes     : total number of nodes (N = 965 for Lyon)
    n_features  : input features per timestep per node (F = 8)
    hidden_size : GRU hidden state size; also the GAT channel width
    num_layers  : number of GRU layers (depth of temporal encoder)
    horizon     : forecast horizon (number of future 15-min slots)
    dropout     : applied after GAT layer-1 and in GRU inter-layer
    num_heads   : number of attention heads in GAT layer-1
                  (layer-2 always uses 1 head → output is hidden_size)
    """

    def __init__(
        self,
        n_nodes: int,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        horizon: int = 4,
        dropout: float = 0.2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.n_nodes     = n_nodes
        self.hidden_size = hidden_size

        # --- Temporal encoder: shared GRU over all (batch × node) pairs ---
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --- Spatial encoder: 2-layer GAT ---
        # Layer 1: multi-head, concat=True → out_channels = hidden_size * num_heads
        self.gat1 = GATConv(
            in_channels=hidden_size,
            out_channels=hidden_size,
            heads=num_heads,
            dropout=dropout,
            concat=True,
        )
        # Layer 2: single head, concat=False → out_channels = hidden_size
        self.gat2 = GATConv(
            in_channels=hidden_size * num_heads,
            out_channels=hidden_size,
            heads=1,
            dropout=dropout,
            concat=False,
        )

        self.act     = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(hidden_size)

        # --- Output head: per-node linear projection ---
        self.head = nn.Linear(hidden_size, horizon)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (B, seq_len, N, F)
        edge_index : (2, E)  — single-graph adjacency (fixed for all samples)
        edge_attr  : (E,)    — optional edge weights (not used by GAT by default)

        Returns
        -------
        (B, N, horizon)
        """
        B, T, N, F = x.shape

        # ---- Temporal encoding ----
        # Merge batch and node dims so GRU processes B*N independent sequences
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, F)   # (B*N, T, F)
        _, h_n  = self.gru(x_flat)                               # h_n: (layers, B*N, H)
        h = h_n[-1]                                              # last layer: (B*N, H)

        # ---- Spatial encoding ----
        # Tile edge_index to cover the full batch of N-node graphs
        ei_batched = _tile_edge_index(edge_index, N, B)          # (2, B*E)

        h = self.gat1(h, ei_batched)                             # (B*N, H*heads)
        h = self.act(h)
        h = self.dropout(h)
        h = self.gat2(h, ei_batched)                             # (B*N, H)
        h = self.norm(h)

        # ---- Output ----
        out = self.head(h)                                       # (B*N, horizon)
        return out.reshape(B, N, -1)                             # (B, N, horizon)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_params(
        cls,
        params: dict,
        n_nodes: int,
        n_features: int,
    ) -> "TrafficSTGNN":
        sp = params["stgnn"]
        return cls(
            n_nodes=n_nodes,
            n_features=n_features,
            hidden_size=sp["hidden_size"],
            num_layers=sp["num_layers"],
            horizon=sp["horizon"],
            dropout=sp["dropout"],
            num_heads=sp["num_heads"],
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
