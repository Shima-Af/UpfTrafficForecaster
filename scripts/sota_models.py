"""State-of-the-art forecasters for the comparison harness (scripts/compare_models.py).

Compact, faithful implementations conforming to the harness interface
    forward(x, edge_index, edge_attr) -> (B, horizon, K),  x: (B, seq_len, K).

Models (chosen for suitability to the cluster-level UPF problem: univariate-per-node,
small/weak graph, no time-of-day covariates in the data contract):
  - DLinear  (Zeng et al., AAAI 2023)  : trend/seasonal decomposition + linear. No graph.
  - STGCN    (Yu et al., IJCAI 2018)   : gated temporal conv + (1st-order) graph conv.
  - AGCRN    (Bai et al., NeurIPS 2020): node-adaptive GRU over a LEARNED adjacency
                                          (ignores the supplied graph by design).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Graph helpers (dense — K is small)
# --------------------------------------------------------------------------- #
def _dense_adj(edge_index, edge_attr, n, device):
    A = torch.zeros(n, n, device=device)
    A[edge_index[0], edge_index[1]] = edge_attr
    return torch.maximum(A, A.t())


def _sym_norm_adj(edge_index, edge_attr, n, device):
    """GCN-style symmetric normalisation  D^-1/2 (A+I) D^-1/2."""
    A = _dense_adj(edge_index, edge_attr, n, device) + torch.eye(n, device=device)
    dinv = A.sum(1).clamp(min=1e-6).pow(-0.5)
    return dinv.unsqueeze(1) * A * dinv.unsqueeze(0)


# --------------------------------------------------------------------------- #
#  DLinear (Zeng et al. 2023) — decomposition + linear, channel-independent
# --------------------------------------------------------------------------- #
class DLinear(nn.Module):
    def __init__(self, n_clusters, seq_len=96, horizon=4, kernel_size=25, **_):
        super().__init__()
        self.kernel_size = kernel_size + (kernel_size % 2 == 0)   # force odd
        self.lin_trend = nn.Linear(seq_len, horizon)
        self.lin_seasonal = nn.Linear(seq_len, horizon)

    def _moving_avg(self, z):                       # z: (B,K,L)
        pad = self.kernel_size // 2
        zp = F.pad(z, (pad, pad), mode="replicate")
        return F.avg_pool1d(zp, self.kernel_size, stride=1)

    def forward(self, x, edge_index=None, edge_attr=None):
        z = x.permute(0, 2, 1)                      # (B,K,L)
        trend = self._moving_avg(z)
        seasonal = z - trend
        out = self.lin_trend(trend) + self.lin_seasonal(seasonal)   # (B,K,H)
        return out.permute(0, 2, 1)                 # (B,H,K)


# --------------------------------------------------------------------------- #
#  STGCN (Yu et al. 2018) — gated temporal conv + graph conv
# --------------------------------------------------------------------------- #
class _TemporalGatedConv(nn.Module):
    def __init__(self, c_in, c_out, kt=3):
        super().__init__()
        self.conv = nn.Conv2d(c_in, 2 * c_out, (1, kt))     # conv along time

    def forward(self, x):                           # (B,C,N,T)
        p, q = torch.chunk(self.conv(x), 2, dim=1)
        return p * torch.sigmoid(q)                 # GLU; T -> T-kt+1


class _STConvBlock(nn.Module):
    def __init__(self, c_in, c_mid, c_out, kt=3):
        super().__init__()
        self.t1 = _TemporalGatedConv(c_in, c_mid, kt)
        self.gconv = nn.Conv2d(c_mid, c_mid, (1, 1))        # channel mix after graph agg
        self.t2 = _TemporalGatedConv(c_mid, c_out, kt)
        self.norm = nn.BatchNorm2d(c_out)

    def forward(self, x, Ahat):                     # x (B,C,N,T)
        x = self.t1(x)
        x = torch.einsum("nm,bcmt->bcnt", Ahat, x)  # spatial graph aggregation
        x = F.relu(self.gconv(x))
        return self.norm(self.t2(x))


class STGCN(nn.Module):
    def __init__(self, n_clusters, seq_len=96, horizon=4, channels=32, kt=3, **_):
        super().__init__()
        self.horizon = horizon
        self.b1 = _STConvBlock(1, channels, channels, kt)
        self.b2 = _STConvBlock(channels, channels, channels, kt)
        self.out_t = nn.Conv2d(channels, channels, (1, 1))
        self.head = nn.Conv2d(channels, horizon, (1, 1))

    def forward(self, x, edge_index, edge_attr):
        B, L, K = x.shape
        Ahat = _sym_norm_adj(edge_index, edge_attr, K, x.device)
        h = x.permute(0, 2, 1).unsqueeze(1)         # (B,1,K,L)
        h = self.b2(self.b1(h, Ahat), Ahat)         # (B,C,K,T'')
        h = self.head(F.relu(self.out_t(h)))        # (B,horizon,K,T'')
        return h[..., -1]                           # (B,horizon,K)


# --------------------------------------------------------------------------- #
#  AGCRN (Bai et al. 2020) — node-adaptive GRU over a learned adjacency
# --------------------------------------------------------------------------- #
class _AVWGCN(nn.Module):
    """Adaptive vertex-wise graph conv: graph + per-node weights from embeddings."""
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.empty(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.zeros(embed_dim, dim_out))
        nn.init.xavier_normal_(self.weights_pool)

    def forward(self, x, node_emb):                 # x (B,N,dim_in), node_emb (N,embed)
        N = node_emb.shape[0]
        A = F.softmax(F.relu(node_emb @ node_emb.t()), dim=1)        # learned adjacency
        supports = [torch.eye(N, device=x.device), A]
        for _ in range(2, self.cheb_k):
            supports.append(2 * A @ supports[-1] - supports[-2])
        supports = torch.stack(supports, 0)                          # (cheb_k,N,N)
        weights = torch.einsum("nd,dkio->nkio", node_emb, self.weights_pool)
        bias = node_emb @ self.bias_pool                             # (N,out)
        x_g = torch.einsum("knm,bmi->bkni", supports, x).permute(0, 2, 1, 3)  # (B,N,k,in)
        return torch.einsum("bnki,nkio->bno", x_g, weights) + bias   # (B,N,out)


class _AGCRNCell(nn.Module):
    def __init__(self, dim_in, dim_hidden, cheb_k, embed_dim):
        super().__init__()
        self.hidden = dim_hidden
        self.gate = _AVWGCN(dim_in + dim_hidden, 2 * dim_hidden, cheb_k, embed_dim)
        self.update = _AVWGCN(dim_in + dim_hidden, dim_hidden, cheb_k, embed_dim)

    def forward(self, x, state, node_emb):
        zr = torch.sigmoid(self.gate(torch.cat([x, state], -1), node_emb))
        z, r = torch.chunk(zr, 2, dim=-1)
        hc = torch.tanh(self.update(torch.cat([x, r * state], -1), node_emb))
        return z * state + (1 - z) * hc


class AGCRN(nn.Module):
    def __init__(self, n_clusters, horizon=4, hidden_dim=64, cheb_k=2, embed_dim=10, **_):
        super().__init__()
        self.hidden = hidden_dim; self.horizon = horizon
        self.node_emb = nn.Parameter(torch.randn(n_clusters, embed_dim) * 0.05)
        self.cell = _AGCRNCell(1, hidden_dim, cheb_k, embed_dim)
        self.head = nn.Conv2d(1, horizon, (1, hidden_dim))

    def forward(self, x, edge_index=None, edge_attr=None):   # graph is learned; given graph ignored
        B, L, K = x.shape
        state = torch.zeros(B, K, self.hidden, device=x.device)
        for t in range(L):
            state = self.cell(x[:, t, :].unsqueeze(-1), state, self.node_emb)
        out = self.head(state.unsqueeze(1))         # (B,horizon,K,1)
        return out.squeeze(-1)                      # (B,horizon,K)
