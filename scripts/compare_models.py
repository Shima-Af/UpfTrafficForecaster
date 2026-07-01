#!/usr/bin/env python3
"""Model-comparison study for the forecaster chapter.

Compares, on the SAME cluster-level data and protocol, four forecasters:
  - gru   : per-node GRU, NO graph (isolates what the graph buys)
  - dcrnn : Diffusion-Convolutional RNN (Li et al. 2018), compact faithful impl
  - gwn   : Graph WaveNet (Wu et al. 2019), compact faithful impl
  - stgnn : this thesis' cluster-first GAT+GRU (src/model.py)

It is deliberately self-contained and does NOT touch the production train.py /
evaluate_forecast.py pipeline. It reuses the cluster_series.npy + coarse graph that
those stages already produced (so run the main sweep first for the K you want).

Metrics are computed on the test split, apples-to-apples with the naive baselines
(persistence, seasonal-naive) recomputed on the identical test windows, so the skill
columns are directly comparable to the main sweep.

Usage:
  python scripts/compare_models.py --shapecheck         # instantiate + 1 forward each
  python scripts/compare_models.py --calibrate          # 2-epoch timing of every model @ K=10,50
  python scripts/compare_models.py                      # full study (see grids below)
  python scripts/compare_models.py --models gru stgnn --k 10 --seeds 0
"""
from __future__ import annotations

import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.cluster_dataset import ClusterTrafficDataset           # noqa: E402
from src.model import ClusterSTGNN, _tile_edge_index           # noqa: E402
from torch_geometric.nn import GATConv                          # noqa: E402
from sota_models import DLinear, STGCN, AGCRN                   # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "cluster_first" / "compare"

# ---- Grids (edit here) ------------------------------------------------------
SERVICE = "Netflix"
MODELS  = ["dlinear", "gru", "dcrnn", "gwn", "agcrn", "stgnn"]   # STGCN dropped (see SWEEP notes)
K_GRID  = [10, 50]            # small graph vs larger graph
SEEDS   = [0, 1, 2]
# ----------------------------------------------------------------------------


# =========================================================================
#  Shared graph helper
# =========================================================================
def build_transition(edge_index: torch.Tensor, edge_attr: torch.Tensor, n: int) -> torch.Tensor:
    """Random-walk normalised transition matrix P = D^-1 A (symmetric A)."""
    A = torch.zeros(n, n, device=edge_index.device)
    A[edge_index[0], edge_index[1]] = edge_attr
    A = torch.maximum(A, A.t())
    d = A.sum(1)
    d[d == 0] = 1.0
    return A / d.unsqueeze(1)


# =========================================================================
#  Baseline 1: per-node GRU (no graph)
# =========================================================================
class GRUForecaster(nn.Module):
    def __init__(self, n_clusters, hidden_dim=64, gru_layers=1, dropout=0.2, horizon=4, **_):
        super().__init__()
        self.horizon = horizon
        self.gru = nn.GRU(1, hidden_dim, num_layers=gru_layers, batch_first=True,
                          dropout=dropout if gru_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x, edge_index=None, edge_attr=None):
        B, L, K = x.shape
        xs = x.permute(0, 2, 1).reshape(B * K, L, 1)
        out, _ = self.gru(xs)
        pred = self.head(out[:, -1, :])                 # (B*K, horizon)
        return pred.reshape(B, K, self.horizon).permute(0, 2, 1)   # (B, horizon, K)


# =========================================================================
#  Baseline 2: DCRNN (diffusion-convolutional GRU, seq2seq)
# =========================================================================
class DiffusionConv(nn.Module):
    def __init__(self, c_in, c_out, max_diffusion=2):
        super().__init__()
        self.K = max_diffusion
        self.lin = nn.Linear(c_in * (max_diffusion + 1), c_out)

    def forward(self, X, P):                            # X: (B,N,c_in)
        out = [X]; Xk = X
        for _ in range(self.K):
            Xk = torch.einsum("nm,bmc->bnc", P, Xk)
            out.append(Xk)
        return self.lin(torch.cat(out, dim=-1))


class DCGRUCell(nn.Module):
    def __init__(self, in_dim, hidden_dim, max_diffusion=2):
        super().__init__()
        self.gate = DiffusionConv(in_dim + hidden_dim, 2 * hidden_dim, max_diffusion)
        self.cand = DiffusionConv(in_dim + hidden_dim, hidden_dim, max_diffusion)

    def forward(self, x, h, P):
        ru = torch.sigmoid(self.gate(torch.cat([x, h], -1), P))
        r, u = torch.chunk(ru, 2, dim=-1)
        c = torch.tanh(self.cand(torch.cat([x, r * h], -1), P))
        return u * h + (1 - u) * c


class DCRNN(nn.Module):
    def __init__(self, n_clusters, hidden_dim=64, max_diffusion=2, horizon=4, **_):
        super().__init__()
        self.hidden_dim = hidden_dim; self.horizon = horizon
        self.enc = DCGRUCell(1, hidden_dim, max_diffusion)
        self.dec = DCGRUCell(1, hidden_dim, max_diffusion)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr):
        B, L, K = x.shape
        P = build_transition(edge_index, edge_attr, K)
        h = torch.zeros(B, K, self.hidden_dim, device=x.device)
        for t in range(L):
            h = self.enc(x[:, t, :].unsqueeze(-1), h, P)
        preds = []; inp = x[:, -1, :].unsqueeze(-1)
        for _ in range(self.horizon):
            h = self.dec(inp, h, P)
            inp = self.proj(h)
            preds.append(inp)
        return torch.cat(preds, dim=-1).permute(0, 2, 1)   # (B, horizon, K)


# =========================================================================
#  Baseline 3: Graph WaveNet (dilated gated TCN + adaptive-adjacency GCN)
# =========================================================================
class GWNGraphConv(nn.Module):
    def __init__(self, c_in, c_out, support_len, order=2, dropout=0.3):
        super().__init__()
        self.order = order; self.dropout = dropout
        self.mlp = nn.Conv2d(c_in * (order * support_len + 1), c_out, (1, 1))

    def forward(self, x, supports):                    # x: (B,C,N,T)
        out = [x]
        for A in supports:
            xk = torch.einsum("nm,bcmt->bcnt", A, x); out.append(xk)
            for _ in range(2, self.order + 1):
                xk = torch.einsum("nm,bcmt->bcnt", A, xk); out.append(xk)
        h = self.mlp(torch.cat(out, dim=1))
        return F.dropout(h, self.dropout, training=self.training)


class GraphWaveNet(nn.Module):
    def __init__(self, n_clusters, horizon=4, residual=32, dilation_ch=32, skip=64,
                 emb=10, dropout=0.3, dilations=(1, 2, 4, 8, 16, 32), **_):
        super().__init__()
        self.horizon = horizon
        self.E1 = nn.Parameter(torch.randn(n_clusters, emb) * 0.05)
        self.E2 = nn.Parameter(torch.randn(emb, n_clusters) * 0.05)
        self.start = nn.Conv2d(1, residual, (1, 1))
        self.filt = nn.ModuleList(); self.gate = nn.ModuleList()
        self.skipc = nn.ModuleList(); self.gconv = nn.ModuleList(); self.bn = nn.ModuleList()
        support_len = 2                                  # physical + adaptive
        rf = 1
        for d in dilations:
            self.filt.append(nn.Conv2d(residual, dilation_ch, (1, 2), dilation=(1, d)))
            self.gate.append(nn.Conv2d(residual, dilation_ch, (1, 2), dilation=(1, d)))
            self.skipc.append(nn.Conv2d(dilation_ch, skip, (1, 1)))
            self.gconv.append(GWNGraphConv(dilation_ch, residual, support_len, dropout=dropout))
            self.bn.append(nn.BatchNorm2d(residual))
            rf += d
        self.receptive = rf
        self.end1 = nn.Conv2d(skip, skip * 2, (1, 1))
        self.end2 = nn.Conv2d(skip * 2, horizon, (1, 1))

    def forward(self, x, edge_index, edge_attr):
        B, L, K = x.shape
        X = x.permute(0, 2, 1).unsqueeze(1)             # (B,1,K,L)
        if L < self.receptive:
            X = F.pad(X, (self.receptive - L, 0, 0, 0))
        supports = [build_transition(edge_index, edge_attr, K),
                    F.softmax(F.relu(self.E1 @ self.E2), dim=1)]
        h = self.start(X); skip = 0
        for i in range(len(self.filt)):
            res = h
            h = torch.tanh(self.filt[i](h)) * torch.sigmoid(self.gate[i](h))
            s = self.skipc[i](h)
            skip = s if isinstance(skip, int) else skip[..., -s.size(3):] + s
            h = self.gconv[i](h, supports)
            h = h + res[..., -h.size(3):]
            h = self.bn[i](h)
        h = F.relu(skip); h = F.relu(self.end1(h)); h = self.end2(h)   # (B,horizon,K,T')
        return h[..., -1]                                # (B, horizon, K)


# =========================================================================
#  Improved STGNN: temporal-first, additive spatial refinement
# =========================================================================
class STGNNv2(nn.Module):
    """GRU per node FIRST (clean temporal signal), then an additive GAT spatial
    refinement on the resulting node embeddings. The residual makes spatial mixing
    optional, so the model's floor is the graph-free GRU rather than below it ---
    the failure mode of the spatial-then-temporal ClusterSTGNN."""
    def __init__(self, n_clusters, hidden_dim=64, gat_heads=4, gat_layers=1,
                 gru_layers=1, horizon=4, dropout=0.2, **_):
        super().__init__()
        self.horizon = horizon; self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(1, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=gru_layers,
                          batch_first=True, dropout=dropout if gru_layers > 1 else 0.0)
        self.gat = nn.ModuleList([GATConv(hidden_dim, hidden_dim, heads=gat_heads,
                                          concat=False, edge_dim=1, dropout=dropout)
                                  for _ in range(gat_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(gat_layers)])
        self.act = nn.ELU(); self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x, edge_index, edge_attr):
        B, T, K = x.shape
        # --- temporal first: per-node GRU on the clean projected sequence ---
        h = self.input_proj(x.reshape(B * T * K, 1)).reshape(B, T, K, self.hidden_dim)
        h = h.permute(0, 2, 1, 3).reshape(B * K, T, self.hidden_dim)
        _, hn = self.gru(h)
        hf = hn[-1]                                              # (B*K, hidden)
        # --- spatial second: additive GAT on the K-node graph (per sample) ---
        ei = _tile_edge_index(edge_index, K, B)                 # (2, B*E)
        ea = edge_attr.unsqueeze(-1).repeat(B, 1)               # (B*E, 1)
        for gat, norm in zip(self.gat, self.norms):
            hf = norm(hf + self.act(gat(hf, ei, ea)))           # additive residual
        out = self.head(self.dropout(hf)).reshape(B, K, self.horizon).permute(0, 2, 1)
        return out


# =========================================================================
#  Model factory
# =========================================================================
def build_model(name: str, n_clusters: int, cfg: dict):
    s = cfg["stgnn"]; H = s["horizon"]; hid = s["hidden_dim"]; dp = s["dropout"]
    if name == "gru":
        return GRUForecaster(n_clusters, hidden_dim=hid, gru_layers=s["gru_layers"],
                             dropout=dp, horizon=H)
    if name == "dcrnn":
        return DCRNN(n_clusters, hidden_dim=hid, max_diffusion=2, horizon=H)
    if name == "gwn":
        return GraphWaveNet(n_clusters, horizon=H, residual=32, dilation_ch=32,
                            skip=64, dropout=dp)
    if name == "stgnn":
        return ClusterSTGNN.from_config(cfg, n_clusters=n_clusters)
    if name == "stgnn2":
        return STGNNv2(n_clusters, hidden_dim=hid, gat_heads=s["gat_heads"],
                       gat_layers=1, gru_layers=s["gru_layers"], horizon=H, dropout=dp)
    if name == "dlinear":
        return DLinear(n_clusters, seq_len=s["seq_len"], horizon=H)
    if name == "stgcn":
        return STGCN(n_clusters, seq_len=s["seq_len"], horizon=H, channels=32)
    if name == "agcrn":
        return AGCRN(n_clusters, horizon=H, hidden_dim=hid, cheb_k=2, embed_dim=10)
    raise ValueError(f"unknown model {name}")


# =========================================================================
#  Data / metrics
# =========================================================================
def load_kdata(service: str, K: int):
    base = REPO_ROOT / "data" / "cluster_first" / service / f"K{K}"
    cs = np.load(base / "cluster_series.npy")
    ei = torch.from_numpy(np.load(base / "coarse_edge_index.npy")).long()
    ea = torch.from_numpy(np.load(base / "coarse_edge_attr.npy")).float()
    return cs, ei, ea


def wape(p, t):  return 100.0 * np.sum(np.abs(p - t)) / np.sum(np.abs(t))
def mae(p, t):   return float(np.mean(np.abs(p - t)))
def rmse(p, t):  return float(np.sqrt(np.mean((p - t) ** 2)))


@torch.no_grad()
def collect_test(model, loader, device):
    model.eval(); P, T = [], []
    for b in loader:
        x = b["x"].to(device); y = b["y"]
        ei = b["edge_index"][0].to(device); ea = b["edge_attr"][0].to(device)
        P.append(model(x, ei, ea).cpu().numpy()); T.append(y.numpy())
    return np.concatenate(P), np.concatenate(T)


def naive_from_windows(loader):
    """Persistence (last input) and seasonal-naive (same slot 24h ago) on test windows."""
    Xs, Ys = [], []
    for b in loader:
        Xs.append(b["x"].numpy()); Ys.append(b["y"].numpy())
    X = np.concatenate(Xs); Y = np.concatenate(Ys)          # (n,L,K),(n,H,K)
    H = Y.shape[1]
    persistence = np.repeat(X[:, -1:, :], H, axis=1)        # last value held
    seasonal = X[:, :H, :]                                  # slot t-96 (seq_len==96)
    return Y, persistence, seasonal


# =========================================================================
#  Train one (model, K, seed)
# =========================================================================
def run_one(name, K, seed, cfg, device, epochs, patience, service=SERVICE):
    torch.manual_seed(seed); np.random.seed(seed)
    cs, ei, ea = load_kdata(service, K)
    s = cfg["stgnn"]; tr = cfg["training"]
    sl, H = s["seq_len"], s["horizon"]
    dskw = dict(seq_len=sl, horizon=H, val_split=tr["val_split"], test_split=tr["test_split"])
    tr_ds = ClusterTrafficDataset(cs, ei, ea, split="train", **dskw)
    va_ds = ClusterTrafficDataset(cs, ei, ea, split="val", **dskw)
    te_ds = ClusterTrafficDataset(cs, ei, ea, split="test", **dskw)
    bs = tr["batch_size"]
    tl = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    vl = DataLoader(va_ds, batch_size=bs, shuffle=False)
    el = DataLoader(te_ds, batch_size=bs, shuffle=False)

    model = build_model(name, K, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    crit = nn.L1Loss()

    best_val, best_state, best_ep, wait = float("inf"), None, 0, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for b in tl:
            x = b["x"].to(device); y = b["y"].to(device)
            eib = b["edge_index"][0].to(device); eab = b["edge_attr"][0].to(device)
            opt.zero_grad()
            loss = crit(model(x, eib, eab), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval(); vtot = 0.0; n = 0
        with torch.no_grad():
            for b in vl:
                x = b["x"].to(device); y = b["y"].to(device)
                eib = b["edge_index"][0].to(device); eab = b["edge_attr"][0].to(device)
                vtot += torch.mean(torch.abs(model(x, eib, eab) - y)).item() * len(x); n += len(x)
        vmae = vtot / n
        if vmae < best_val - 1e-5:
            best_val, best_state, best_ep, wait = vmae, {k: v.cpu().clone() for k, v in model.state_dict().items()}, ep, 0
        else:
            wait += 1
            if wait >= patience:
                break
    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    preds, trues = collect_test(model, el, device)
    Yn, perst, seas = naive_from_windows(el)
    w = wape(preds, trues)
    lv, sn = wape(perst, Yn), wape(seas, Yn)
    return {
        "service": service, "model": name, "K": K, "seed": seed,
        "test_wape": round(w, 4), "test_mae": round(mae(preds, trues), 6),
        "test_rmse": round(rmse(preds, trues), 6),
        "lv_wape": round(lv, 4), "sn_wape": round(sn, 4),
        "skill_vs_persistence": round(1 - w / lv, 4),
        "skill_vs_seasonal": round(1 - w / sn, 4),
        "n_params": n_params, "epochs_run": best_ep, "train_time_s": round(train_time, 1),
    }


def aggregate(rows, out_name="compare.csv"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / out_name
    existing = []
    if out.exists():
        with open(out) as fh:
            existing = list(csv.DictReader(fh))
    allrows = existing + [{k: str(v) for k, v in r.items()} for r in rows]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(allrows)
    return out


def load_cfg():
    import yaml
    with open(REPO_ROOT / "config.yaml") as fh:
        return yaml.safe_load(fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapecheck", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--k", nargs="+", type=int, default=K_GRID)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--service", default=SERVICE, help="Netflix | total | DailyMotion | ...")
    ap.add_argument("--out", default="compare.csv", help="output csv name under results/.../compare/")
    args = ap.parse_args()
    cfg = load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    if args.shapecheck:
        for K in args.k:
            cs, ei, ea = load_kdata(SERVICE, K)
            x = torch.randn(4, cfg["stgnn"]["seq_len"], K).to(device)
            for m in args.models:
                mdl = build_model(m, K, cfg).to(device)
                out = mdl(x, ei.to(device), ea.to(device))
                np_ = sum(p.numel() for p in mdl.parameters())
                print(f"  K={K:3d} {m:6s} -> {tuple(out.shape)}  params={np_:,}")
        return 0

    if args.calibrate:
        print("=== 2-epoch timing (per-epoch s) ===")
        for K in args.k:
            for m in args.models:
                r = run_one(m, K, seed=0, cfg=cfg, device=device, epochs=2, patience=99)
                print(f"  K={K:3d} {m:6s}: {r['train_time_s']/2:6.1f} s/epoch  "
                      f"params={r['n_params']:,}")
        return 0

    jobs = [(m, K, sd) for m in args.models for K in args.k for sd in args.seeds]
    print(f"=== compare: {len(jobs)} runs (service={args.service} -> {args.out}) ===", flush=True)
    t0 = time.time(); rows = []
    for i, (m, K, sd) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {m} K={K} seed={sd}  (elapsed {time.time()-t0:.0f}s)", flush=True)
        r = run_one(m, K, sd, cfg, device, args.epochs, args.patience, service=args.service)
        rows.append(r); aggregate([r], out_name=args.out)
        print(f"    wape={r['test_wape']}  skill_persist={r['skill_vs_persistence']}  "
              f"epochs={r['epochs_run']}  {r['train_time_s']}s", flush=True)
    print(f"=== done {time.time()-t0:.0f}s -> {OUT_DIR/'compare.csv'} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
