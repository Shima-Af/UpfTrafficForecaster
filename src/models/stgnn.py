"""
stgnn.py
--------
Spatiotemporal GNN forecaster (stub — implemented after LSTM baseline).

Architecture plan
-----------------
  Node features : per-gNodeB traffic time series (seq_len timesteps)
  Graph         : gNodeBs as nodes, edges from Voronoi adjacency + distance
  Encoder       : Temporal convolution or LSTM per node
  GNN layers    : GCN / GAT / SAGE for spatial message passing
  Decoder       : Linear head per node → horizon predictions

Reference
---------
  Wu et al. "Graph WaveNet for Deep Spatial-Temporal Graph Modeling" (2019)
  Li et al. "Diffusion Convolutional Recurrent Neural Network" (2018)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TrafficSTGNN(nn.Module):
    """Spatiotemporal GNN — stub awaiting torch-geometric installation."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "TrafficSTGNN is not yet implemented. "
            "Complete the LSTM baseline first, then implement STGNN. "
            "Requires: torch-geometric, torch-scatter, torch-sparse."
        )

    def forward(self, x, edge_index, edge_attr=None):
        raise NotImplementedError


def build_graph(voronoi_map, bs_df, params):
    """
    Build edge_index and edge_attr for the gNodeB graph.

    Strategy (from params → stgnn):
      - knn: connect each BS to its k nearest neighbours
      - threshold_distance: connect BSs within distance_threshold_m

    Returns
    -------
    edge_index : (2, n_edges) int64 tensor
    edge_attr  : (n_edges, 1) float32 tensor  — inverse distance weight
    """
    raise NotImplementedError("TODO: implement after LSTM baseline")
