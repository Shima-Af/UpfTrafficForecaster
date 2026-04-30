"""
model.py — Cluster-level STGNN (GAT → GRU architecture).

ClusterSTGNN differs from the original TrafficSTGNN in two key ways:

  1. Spatial-first: GAT encoding happens at every time step before the GRU
     sees any temporal context.  This lets spatial message-passing propagate
     information across cluster boundaries before the GRU integrates the
     resulting spatially-enriched history.  In the original model, GRU runs
     first and GAT only mixes the final hidden states.

  2. Single feature input: each cluster node has only one input feature
     (the aggregate dl_norm) — no per-service breakdown or tile-level detail.
     An input projection Linear(1, hidden_dim) bridges the 1-D signal into
     the hidden space before GAT processing.

Architecture summary
--------------------
  Input x: (batch, seq_len, K)     — K cluster aggregates over time window

  For each time step t ∈ [0, seq_len):
    node_feat ← Linear(1, hidden_dim)(x[:, t, :, None])  # input projection
    for each GAT layer l:
        node_feat ← GATConv(hidden_dim, hidden_dim, heads, concat=False)(node_feat)
        node_feat ← LayerNorm(node_feat + residual)       # residual connection

  GRU reads the seq_len-length spatially-enriched sequence per cluster node:
    h_final ← GRU(input_size=hidden_dim, hidden_size=hidden_dim)(spatially enriched)

  Output head:
    forecast ← Linear(hidden_dim, horizon)(h_final)      # per-cluster forecast
    return (batch, horizon, K)

The (batch, horizon, K) output maps directly to the K cluster-level load
forecasts consumed by K independent PPO UPF controllers.

References
----------
Veličković et al. "Graph Attention Networks" (ICLR 2018)
Wu et al.        "Graph WaveNet" (IJCAI 2019)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


# ---------------------------------------------------------------------------
# Helper: block-diagonal edge tiling for batched graph processing
# ---------------------------------------------------------------------------

def _tile_edge_index(
    edge_index: torch.Tensor,
    n_nodes: int,
    n_copies: int,
) -> torch.Tensor:
    """
    Replicate edge_index to form a block-diagonal adjacency for n_copies
    independent graphs, each with n_nodes nodes.

    Used to process batch × seq_len graphs simultaneously in a single
    GATConv call instead of looping over batch samples and time steps.

    Parameters
    ----------
    edge_index : (2, E)   source single-graph edge list
    n_nodes    : N        nodes per graph
    n_copies   : B*T      total number of graph copies

    Returns
    -------
    (2, n_copies * E)  block-diagonal edge list
    """
    offsets = torch.arange(n_copies, device=edge_index.device) * n_nodes
    # edge_index: (2, E) → (1, 2, E); offsets: (B,) → (B, 1, 1)
    ei_batch = edge_index.unsqueeze(0) + offsets.view(n_copies, 1, 1)   # (B, 2, E)
    return ei_batch.reshape(2, -1)                                        # (2, B*E)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ClusterSTGNN(nn.Module):
    """
    Spatiotemporal STGNN for K-cluster traffic forecasting.

    Parameters
    ----------
    n_clusters  : K — number of UPF service area clusters
    hidden_dim  : width of GAT and GRU hidden representations
    gat_heads   : number of attention heads per GAT layer (concat=False,
                  so output is always hidden_dim regardless of head count)
    gat_layers  : depth of GAT spatial encoder (typically 2)
    gru_layers  : depth of GRU temporal encoder (typically 1 for small K)
    seq_len     : input sequence length (96 = 24 h at 15-min resolution)
    horizon     : forecast horizon (4 = 1 h ahead)
    dropout     : applied inside GATConv and between GRU layers
    """

    def __init__(
        self,
        n_clusters: int,
        hidden_dim: int = 64,
        gat_heads: int = 4,
        gat_layers: int = 2,
        gru_layers: int = 1,
        seq_len: int = 96,
        horizon: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.hidden_dim = hidden_dim
        self.seq_len    = seq_len
        self.horizon    = horizon

        # Input projection: single dl_norm value → hidden_dim space
        self.input_proj = nn.Linear(1, hidden_dim)

        # GAT spatial encoder — gat_layers layers with residual connections
        # concat=False: all heads averaged → output is always hidden_dim
        self.gat_layers_list = nn.ModuleList([
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=gat_heads,
                concat=False,
                edge_dim=1,        # edge_attr has 1 feature (affinity weight)
                dropout=dropout,
            )
            for _ in range(gat_layers)
        ])
        self.gat_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(gat_layers)
        ])
        self.gat_act = nn.ELU()

        # GRU temporal encoder — processes seq_len steps per cluster node
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # Output head: per-cluster linear projection
        self.head = nn.Linear(hidden_dim, horizon)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (batch, seq_len, K)  — cluster aggregate dl_norm
        edge_index : (2, E_coarse)        — coarsened graph edges
        edge_attr  : (E_coarse,)          — inter-cluster affinity weights

        Returns
        -------
        (batch, horizon, K)  — per-cluster forecasts for all horizon steps
        """
        B, T, K = x.shape
        E = edge_index.shape[1]

        # ---- Tile graph for batch × seq_len independent forward passes ----
        # Each (batch-sample, time-step) pair is treated as an independent
        # K-node graph.  Block-diagonal tiling avoids cross-contamination.
        BT = B * T
        tiled_ei   = _tile_edge_index(edge_index, K, BT)         # (2, BT*E)
        tiled_ea   = edge_attr.unsqueeze(-1).repeat(BT, 1)        # (BT*E, 1)
        #   edge_attr: (E,) → (E, 1) → repeat BT times → (BT*E, 1)

        # ---- Step 1: input projection + GAT spatial encoding ----
        # Reshape: (B, T, K) → (BT, K, 1) → (BT*K, 1)
        x_proj = x.reshape(BT, K, 1)                              # (BT, K, 1)
        x_proj = x_proj.reshape(BT * K, 1)                        # (BT*K, 1)
        h      = self.input_proj(x_proj)                           # (BT*K, hidden_dim)

        for gat, norm in zip(self.gat_layers_list, self.gat_norms):
            residual = h
            h = gat(h, tiled_ei, tiled_ea)                        # (BT*K, hidden_dim)
            h = self.gat_act(h)
            h = norm(h + residual)                                 # residual connection

        # Reshape back: (BT*K, hidden_dim) → (B, T, K, hidden_dim)
        h = h.reshape(BT, K, self.hidden_dim)                      # (BT, K, hidden_dim)
        h = h.reshape(B, T, K, self.hidden_dim)                    # (B, T, K, hidden_dim)

        # ---- Step 2: GRU temporal encoding per cluster node ----
        # Reshape: (B, T, K, hidden_dim) → (B*K, T, hidden_dim)
        h = h.permute(0, 2, 1, 3).reshape(B * K, T, self.hidden_dim)  # (B*K, T, H)
        _, h_n = self.gru(h)                                        # h_n: (layers, B*K, H)
        h = h_n[-1]                                                 # (B*K, hidden_dim)

        # ---- Step 3: output projection ----
        h = self.dropout(h)
        out = self.head(h)                                          # (B*K, horizon)
        out = out.reshape(B, K, self.horizon)                       # (B, K, horizon)
        out = out.permute(0, 2, 1)                                  # (B, horizon, K)

        return out

    # ------------------------------------------------------------------
    # Forward with internals (for interpretability / integration)
    # ------------------------------------------------------------------

    def forward_with_internals(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Same as forward() but also returns GAT attention weights and GRU
        hidden states.  These are used for Level 1 integration analysis:
        inspecting which cluster pairs attend to each other most strongly
        and which clusters drive the GRU's temporal memory.

        Parameters
        ----------
        (same as forward)

        Returns
        -------
        forecasts           : (batch, horizon, K)
        gat_attention_weights : list of (BT*E, heads) tensors, one per GAT layer
        gru_hidden_states   : (batch, K, hidden_dim) final GRU hidden state
        """
        B, T, K = x.shape
        E  = edge_index.shape[1]
        BT = B * T

        tiled_ei = _tile_edge_index(edge_index, K, BT)
        tiled_ea = edge_attr.unsqueeze(-1).repeat(BT, 1)

        x_proj = x.reshape(BT * K, 1)
        h      = self.input_proj(x_proj)

        gat_attention_weights: List[torch.Tensor] = []
        for gat, norm in zip(self.gat_layers_list, self.gat_norms):
            residual = h
            h, (_, alpha) = gat(h, tiled_ei, tiled_ea, return_attention_weights=True)
            h = self.gat_act(h)
            h = norm(h + residual)
            gat_attention_weights.append(alpha)              # (BT*E, heads)

        h = h.reshape(B, T, K, self.hidden_dim)
        h = h.permute(0, 2, 1, 3).reshape(B * K, T, self.hidden_dim)
        _, h_n = self.gru(h)
        h_final = h_n[-1]                                    # (B*K, hidden_dim)

        gru_hidden_states = h_final.reshape(B, K, self.hidden_dim)

        h_out = self.dropout(h_final)
        out   = self.head(h_out).reshape(B, K, self.horizon).permute(0, 2, 1)

        return out, gat_attention_weights, gru_hidden_states

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: dict, n_clusters: int) -> "ClusterSTGNN":
        """Instantiate from a config dict (cluster_first/config.yaml stgnn section)."""
        sp = config["stgnn"]
        return cls(
            n_clusters=n_clusters,
            hidden_dim=sp["hidden_dim"],
            gat_heads=sp["gat_heads"],
            gat_layers=sp["gat_layers"],
            gru_layers=sp["gru_layers"],
            seq_len=sp["seq_len"],
            horizon=sp["horizon"],
            dropout=sp["dropout"],
        )
