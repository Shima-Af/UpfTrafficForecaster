#!/usr/bin/env python3
"""Partition control: is it *geographic* aggregation that creates predictability, or aggregation per se?

The granularity ablation (ablation_granularity.py) shows that forecasting K aggregated signals
beats forecasting 965 base stations and summing. It does not show that the aggregation has to be
geographic. This script holds everything fixed (K, model, protocol, the 965 base-station series)
and changes only *how the base stations are partitioned*:

  skater  : the pipeline's contiguous geographic partition (data/cluster_first/<svc>/K<K>)
  kmeans  : k-means on projected coordinates (geographic, compact, contiguity not enforced)
  random  : the SKATER labels permuted over base stations (identical cluster sizes, no geography)

Each partition is coarsened with the same rule as src/coarsen.py (sum of member signals; clusters
adjacent iff any members are Voronoi-adjacent; weight = mean lat/lon RBF affinity) and forecast
with the common harness of compare_models.py. Test-window predictions are saved per run so that
per-cluster / per-horizon / peak diagnostics can be derived without retraining.

Outputs:
  results/cluster_first/partition/partition.csv
  results/cluster_first/partition/preds/<partition>_<model>_K<K>_seed<seed>.npz
Usage:
  python scripts/partition_control.py --models gru gwn --k 10 --seeds 0 1 2 --draws 3
"""
from __future__ import annotations

import argparse, csv, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from src.cluster_dataset import ClusterTrafficDataset                       # noqa: E402
from compare_models import build_model, load_cfg, wape, mae, rmse           # noqa: E402
from ablation_granularity import build_bs_series, cluster_labels            # noqa: E402

OUT = REPO / "results" / "cluster_first" / "partition"
COLS = ["service", "partition", "draw", "model", "K", "seed", "test_wape", "test_mae", "test_rmse",
        "lv_wape", "sn_wape", "skill_vs_persistence", "skill_vs_seasonal",
        "n_coarse_edges", "n_params", "epochs_run", "train_time_s"]


# =========================================================================
#  Partitions and coarsening
# =========================================================================
def bs_coords(ni: pd.DataFrame) -> np.ndarray:
    """(N, 2) [lat, lon] in node_idx order."""
    loc = pd.read_parquet(REPO / "data" / "graphs" / "bs_locations.parquet")
    return ni.merge(loc, on="site_id", how="left")[["lat", "lon"]].values.astype(np.float64)


def project_km(coords: np.ndarray) -> np.ndarray:
    """Equirectangular projection of [lat, lon] to (x_km, y_km) about the mean latitude."""
    lat0 = np.radians(coords[:, 0].mean())
    return np.column_stack([np.radians(coords[:, 1]) * 6371.0 * np.cos(lat0),
                            np.radians(coords[:, 0]) * 6371.0])


def make_partition(kind: str, K: int, draw: int, skater: np.ndarray, coords: np.ndarray) -> np.ndarray:
    if kind == "skater":
        return skater
    if kind == "kmeans":
        return KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(project_km(coords))
    if kind == "random":                      # size-matched: permute the SKATER labels
        return np.random.default_rng(1000 + draw).permutation(skater)
    raise ValueError(kind)


def coarsen(labels: np.ndarray, series: np.ndarray, fine_ei: np.ndarray, coords: np.ndarray):
    """Same rule as src/coarsen.py::build_coarsened_graph, for an arbitrary partition."""
    K = int(labels.max()) + 1
    M = np.zeros((len(labels), K), dtype=np.float32); M[np.arange(len(labels)), labels] = 1.0
    cs = (M.T @ series).astype(np.float32)                                    # (K, T)
    aff = rbf_kernel(coords, gamma=50.0)
    acc: dict = {}
    for i, j in zip(fine_ei[0], fine_ei[1]):
        ci, cj = int(labels[i]), int(labels[j])
        if ci != cj:
            acc.setdefault((ci, cj), []).append(float(aff[i, j]))
    src = [k[0] for k in acc] + list(range(K))
    dst = [k[1] for k in acc] + list(range(K))
    wts = [float(np.mean(v)) for v in acc.values()] + [1.0] * K
    return cs, torch.tensor([src, dst], dtype=torch.long), torch.tensor(wts, dtype=torch.float32)


def verify_skater(cs, ei, ea, service: str, K: int) -> None:
    """The re-derived SKATER coarsening must reproduce the pipeline's stored artefacts."""
    base = REPO / "data" / "cluster_first" / service / f"K{K}"
    ref = np.load(base / "cluster_series.npy")
    err = float(np.abs(ref - cs).max())
    ref_e = {(int(a), int(b)): float(w) for a, b, w in
             zip(*np.load(base / "coarse_edge_index.npy"), np.load(base / "coarse_edge_attr.npy"))}
    new_e = {(int(a), int(b)): float(w) for a, b, w in zip(ei[0], ei[1], ea)}
    assert err < 1e-3, f"cluster_series mismatch {err}"
    assert ref_e.keys() == new_e.keys(), "coarse edge set mismatch"
    werr = max(abs(ref_e[k] - new_e[k]) for k in ref_e)
    assert werr < 1e-4, f"coarse edge weight mismatch {werr}"
    print(f"  [verify] SKATER K={K}: series max|err|={err:.2e}, {len(ref_e)} edges match "
          f"(weight err {werr:.1e})", flush=True)


# =========================================================================
#  Train / evaluate one run (protocol identical to compare_models.run_one)
# =========================================================================
def train_eval(name, cs, ei, ea, seed, cfg, device, epochs, patience):
    torch.manual_seed(seed); np.random.seed(seed)
    s, tr = cfg["stgnn"], cfg["training"]
    dskw = dict(seq_len=s["seq_len"], horizon=s["horizon"],
                val_split=tr["val_split"], test_split=tr["test_split"])
    bs = tr["batch_size"]
    tl = DataLoader(ClusterTrafficDataset(cs, ei, ea, split="train", **dskw), batch_size=bs, shuffle=True)
    vl = DataLoader(ClusterTrafficDataset(cs, ei, ea, split="val", **dskw), batch_size=bs)
    te = ClusterTrafficDataset(cs, ei, ea, split="test", **dskw)
    el = DataLoader(te, batch_size=bs)

    model = build_model(name, cs.shape[0], cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    crit = nn.L1Loss()
    eid, ead = ei.to(device), ea.to(device)

    best, best_state, best_ep, wait = float("inf"), None, 0, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for b in tl:
            x = b["x"].to(device); y = b["y"].to(device)
            opt.zero_grad(); loss = crit(model(x, eid, ead), y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        model.eval(); vt, n = 0.0, 0
        with torch.no_grad():
            for b in vl:
                x = b["x"].to(device); y = b["y"].to(device)
                vt += torch.mean(torch.abs(model(x, eid, ead) - y)).item() * len(x); n += len(x)
        v = vt / n
        if v < best - 1e-5:
            best, best_state, best_ep, wait = v, {k: t.cpu().clone() for k, t in model.state_dict().items()}, ep, 0
        else:
            wait += 1
            if wait >= patience:
                break
    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval(); P = []
    with torch.no_grad():
        for b in el:
            P.append(model(b["x"].to(device), eid, ead).cpu().numpy())
    P = np.concatenate(P)                                   # (n, H, K)
    X, Y = te.X.numpy(), te.y.numpy()
    H = Y.shape[1]
    pers = np.repeat(X[:, -1:, :], H, axis=1)               # last value held
    seas = X[:, :H, :]                                      # same slot 24 h earlier (seq_len == 96)
    w, lv, sn = float(wape(P, Y)), float(wape(pers, Y)), float(wape(seas, Y))
    metrics = {"test_wape": round(w, 4), "test_mae": round(mae(P, Y), 6), "test_rmse": round(rmse(P, Y), 6),
               "lv_wape": round(lv, 4), "sn_wape": round(sn, 4),
               "skill_vs_persistence": round(1 - w / lv, 4), "skill_vs_seasonal": round(1 - w / sn, 4),
               "n_params": n_params, "epochs_run": best_ep, "train_time_s": round(train_time, 1)}
    return metrics, dict(pred=P, true=Y, persistence=pers, seasonal=seas)


def append_row(row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "partition.csv"
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["gru", "gwn"])
    ap.add_argument("--partitions", nargs="+", default=["skater", "kmeans", "random"])
    ap.add_argument("--k", nargs="+", type=int, default=[10])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--draws", type=int, default=3, help="number of random partitions")
    ap.add_argument("--draw-start", type=int, default=0, help="index of the first random partition")
    ap.add_argument("--service", default="Netflix")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()
    cfg = load_cfg(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    series, ni, fine_ei, _ = build_bs_series(args.service)
    fine_ei = fine_ei.numpy(); coords = bs_coords(ni)
    (OUT / "preds").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for K in args.k:
        skater = cluster_labels(ni, K, args.service)
        for kind in args.partitions:
            for draw in (range(args.draw_start, args.draw_start + args.draws) if kind == "random" else [0]):
                labels = make_partition(kind, K, draw, skater, coords)
                cs, ei, ea = coarsen(labels, series, fine_ei, coords)
                if kind == "skater":
                    verify_skater(cs, ei, ea, args.service, K)
                tag = f"{kind}{draw}" if kind == "random" else kind
                np.save(OUT / "preds" / f"labels_{tag}_K{K}.npy", labels)
                n_edges = int((ei[0] != ei[1]).sum()) // 2
                for m in args.models:
                    for sd in args.seeds:
                        r, arr = train_eval(m, cs, ei, ea, sd, cfg, device, args.epochs, args.patience)
                        r.update(service=args.service, partition=kind, draw=draw, model=m, K=K,
                                 seed=sd, n_coarse_edges=n_edges)
                        append_row(r)
                        np.savez_compressed(OUT / "preds" / f"{tag}_{m}_K{K}_seed{sd}.npz", **arr)
                        print(f"  K={K} {tag:8s} {m:4s} seed={sd}: wape={r['test_wape']} "
                              f"skill_p={r['skill_vs_persistence']} skill_s={r['skill_vs_seasonal']} "
                              f"ep={r['epochs_run']} {r['train_time_s']}s (elapsed {time.time()-t0:.0f}s)",
                              flush=True)
    print(f"=== done {time.time()-t0:.0f}s -> {OUT/'partition.csv'} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
