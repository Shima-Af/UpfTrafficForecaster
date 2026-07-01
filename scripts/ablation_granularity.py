#!/usr/bin/env python3
"""Granularity ablation: cluster-first vs per-base-station, on a FIXED coarse target.

Question: does aggregating base stations into K clusters BEFORE forecasting
(cluster-first) discard spatial signal that forecasting at the base-station level and
summing afterwards could exploit? Both routes are scored on the SAME K-cluster target,
so the comparison is controlled (the K-sweep cannot answer this — its target moves with K).

  Route A (cluster-first): forecast K cluster signals directly   -> already in compare.csv
  Route B (per-BS):        forecast N=965 base stations, sum to K -> trained here

Spatial-attenuation test: compare the graph uplift (graph model minus no-graph GRU) at
the BS level vs. at the cluster level. If BS-level uplift >> cluster-level uplift, the
aggregation step smooths away spatial signal.

Output: results/cluster_first/ablation/ablation.csv
Usage:  python scripts/ablation_granularity.py --models gru gwn --k 10 --seeds 0 1 2
"""
from __future__ import annotations

import argparse, csv, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from src.cluster_dataset import ClusterTrafficDataset          # noqa: E402
from src.coarsen import _load_bs_level                         # noqa: E402
from compare_models import build_model, load_cfg, wape, mae, rmse   # noqa: E402

OUT = REPO / "results" / "cluster_first" / "ablation"


def build_bs_series(service: str):
    """(N=965, T) base-station series in node_idx order + (N,) K-cluster labels."""
    ni = pd.read_parquet(REPO / "data" / "graphs" / "node_index.parquet").sort_values("node_idx")
    df = _load_bs_level(REPO / "data" / "netmob" / "processed", service=service)
    ts = np.sort(df["timestamp"].unique())
    piv = (df.pivot_table(index="site_id", columns="timestamp", values="dl_norm", fill_value=0.0)
             .reindex(index=ni["site_id"].values, columns=ts, fill_value=0.0))
    series = piv.values.astype(np.float32)                     # (N, T)
    ei = torch.from_numpy(np.load(REPO / "data" / "graphs" / "edge_index.npy")).long()
    ea = torch.from_numpy(np.load(REPO / "data" / "graphs" / "edge_attr.npy")).float()
    return series, ni, ei, ea


def cluster_labels(ni: pd.DataFrame, K: int, service: str) -> np.ndarray:
    ca = pd.read_parquet(REPO / "data" / "cluster_first" / service / f"K{K}" / "cluster_assignments.parquet")
    return ni.merge(ca, on="site_id", how="left")["cluster_id"].values.astype(int)


def _agg(t: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Sum the last (node) dim of t (..., N) into clusters via one-hot M (N, K)."""
    return t @ M


@torch.no_grad()
def _eval_aggregated(model, loader, M, device):
    """Predict at BS level, aggregate preds & truth to clusters, return cluster metrics."""
    P, T, PERS, SEAS = [], [], [], []
    H = None
    for b in loader:
        x = b["x"].to(device); y = b["y"]
        ei = b["edge_index"][0].to(device); ea = b["edge_attr"][0].to(device)
        pred = model(x, ei, ea).cpu()                          # (B,H,N)
        H = pred.shape[1]
        P.append(_agg(pred, M.cpu()).numpy())
        T.append(_agg(y, M.cpu()).numpy())
        PERS.append(_agg(x[:, -1:, :].cpu().repeat(1, H, 1), M.cpu()).numpy())  # persistence
        SEAS.append(_agg(x[:, :H, :].cpu(), M.cpu()).numpy())                   # seasonal-naive
    P, T = np.concatenate(P), np.concatenate(T)
    pers, seas = np.concatenate(PERS), np.concatenate(SEAS)
    w, lv, sn = wape(P, T), wape(pers, T), wape(seas, T)
    return {"test_wape": round(w, 4), "test_mae": round(mae(P, T), 6),
            "lv_wape": round(lv, 4), "sn_wape": round(sn, 4),
            "skill_vs_persistence": round(1 - w / lv, 4),
            "skill_vs_seasonal": round(1 - w / sn, 4)}


def run_route_B(model_name, K, seed, cfg, device, epochs, patience, batch=8):
    torch.manual_seed(seed); np.random.seed(seed)
    series, ni, ei, ea = build_bs_series("Netflix")
    N = series.shape[0]
    labels = cluster_labels(ni, K, "Netflix")
    M = torch.zeros(N, K); M[torch.arange(N), torch.from_numpy(labels)] = 1.0   # (N,K) one-hot
    M = M.to(device)

    s, tr = cfg["stgnn"], cfg["training"]
    dskw = dict(seq_len=s["seq_len"], horizon=s["horizon"],
                val_split=tr["val_split"], test_split=tr["test_split"])
    tl = DataLoader(ClusterTrafficDataset(series, ei, ea, split="train", **dskw),
                    batch_size=batch, shuffle=True)
    vl = DataLoader(ClusterTrafficDataset(series, ei, ea, split="val", **dskw),
                    batch_size=batch)
    el = DataLoader(ClusterTrafficDataset(series, ei, ea, split="test", **dskw),
                    batch_size=batch)

    model = build_model(model_name, N, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    crit = nn.L1Loss()
    best, best_state, best_ep, wait = 1e9, None, 0, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for b in tl:
            x = b["x"].to(device); y = b["y"].to(device)
            eib = b["edge_index"][0].to(device); eab = b["edge_attr"][0].to(device)
            opt.zero_grad(); loss = crit(model(x, eib, eab), y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        model.eval(); vt, n = 0.0, 0
        with torch.no_grad():
            for b in vl:
                x = b["x"].to(device); y = b["y"].to(device)
                eib = b["edge_index"][0].to(device); eab = b["edge_attr"][0].to(device)
                vt += torch.mean(torch.abs(model(x, eib, eab) - y)).item() * len(x); n += len(x)
        v = vt / n
        if v < best - 1e-5:
            best, best_state, best_ep, wait = v, {k: vv.cpu().clone() for k, vv in model.state_dict().items()}, ep, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    m = _eval_aggregated(model, el, M, device)
    m.update({"route": "B_per_bs", "model": model_name, "target_K": K, "seed": seed,
              "n_nodes": N, "n_params": n_params, "epochs_run": best_ep,
              "train_time_s": round(time.time() - t0, 1)})
    return m


def aggregate(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "ablation.csv"
    prev = list(csv.DictReader(open(out))) if out.exists() else []
    allr = prev + [{k: str(v) for k, v in r.items()} for r in rows]
    cols = ["route", "model", "target_K", "seed", "n_nodes", "test_wape", "test_mae",
            "lv_wape", "sn_wape", "skill_vs_persistence", "skill_vs_seasonal",
            "n_params", "epochs_run", "train_time_s"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(allr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["gru", "gwn"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8,
                    help="small batch — N=965 GRU workspace is memory-heavy")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if args.calibrate:
        for m in args.models:
            r = run_route_B(m, args.k, 0, cfg, device, epochs=2, patience=99, batch=args.batch)
            print(f"  {m:6s} N={r['n_nodes']}: {r['train_time_s']/2:6.1f} s/epoch  "
                  f"params={r['n_params']:,}  skill_p(2ep)={r['skill_vs_persistence']}")
        return 0
    jobs = [(m, sd) for m in args.models for sd in args.seeds]
    print(f"=== ablation: {len(jobs)} per-BS runs (target K={args.k}) ===", flush=True)
    t0 = time.time()
    for i, (m, sd) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] route-B {m} seed={sd} (elapsed {time.time()-t0:.0f}s)", flush=True)
        r = run_route_B(m, args.k, sd, cfg, device, args.epochs, args.patience, args.batch)
        aggregate([r])
        print(f"    cluster-target WAPE={r['test_wape']}  skill_p={r['skill_vs_persistence']}  "
              f"epochs={r['epochs_run']}  {r['train_time_s']}s", flush=True)
    print(f"=== done {time.time()-t0:.0f}s -> {OUT/'ablation.csv'} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
