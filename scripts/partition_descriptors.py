#!/usr/bin/env python3
"""Structural descriptors of the partitions compared in partition_control.py (no training).

For each (K, partition): contiguity, compactness, within-cluster coherence, redundancy between
cluster signals and the intrinsic difficulty of the resulting target (naive-baseline WAPE).
Output: results/cluster_first/partition/descriptors.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from ablation_granularity import build_bs_series, cluster_labels            # noqa: E402
from partition_control import bs_coords, project_km, make_partition, coarsen, load_cfg, OUT  # noqa: E402
from src.cluster_dataset import ClusterTrafficDataset                       # noqa: E402


def describe(labels, series, fine_ei, coords, C, cfg):
    K = int(labels.max()) + 1; xy = project_km(coords); n = len(labels)
    A = csr_matrix((np.ones(fine_ei.shape[1]), (fine_ei[0], fine_ei[1])), shape=(n, n))
    comps, coh, rad = [], [], []
    for k in range(K):
        idx = np.where(labels == k)[0]
        comps.append(connected_components(A[idx][:, idx], directed=False)[0])
        rad.append(np.linalg.norm(xy[idx] - xy[idx].mean(0), axis=1).mean())
        if len(idx) > 1:
            coh.append(np.nanmean(C[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)]))
    cs, ei, ea = coarsen(labels, series, fine_ei, coords)
    s, tr = cfg["stgnn"], cfg["training"]
    te = ClusterTrafficDataset(cs, ei, ea, split="test", seq_len=s["seq_len"], horizon=s["horizon"],
                               val_split=tr["val_split"], test_split=tr["test_split"])
    X, Y = te.X.numpy(), te.y.numpy(); H = Y.shape[1]
    w = lambda p: float(100 * np.abs(p - Y).sum() / np.abs(Y).sum())
    return {
        "n_contiguous": int(sum(c == 1 for c in comps)), "mean_components": float(np.mean(comps)),
        "mean_radius_km": float(np.mean(rad)), "coherence": float(np.mean(coh)),
        "intercluster_corr": float(np.corrcoef(cs)[np.triu_indices(K, k=1)].mean()),
        "load_cv": float(cs.mean(1).std() / cs.mean(1).mean()),
        "n_coarse_edges": len({(min(a, b), max(a, b)) for a, b in zip(ei[0].tolist(), ei[1].tolist()) if a != b}),
        "persistence_wape": w(np.repeat(X[:, -1:, :], H, axis=1)), "seasonal_wape": w(X[:, :H, :]),
    }


def main():
    cfg = load_cfg()
    series, ni, fine_ei, _ = build_bs_series("Netflix")
    fine_ei = fine_ei.numpy(); coords = bs_coords(ni)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(series.astype(np.float64))
    rows = []
    for K in (10, 50):
        skater = cluster_labels(ni, K, "Netflix")
        for kind, draws in (("skater", 1), ("kmeans", 1), ("random", 3)):
            for d in range(draws):
                r = describe(make_partition(kind, K, d, skater, coords), series, fine_ei, coords, C, cfg)
                rows.append({"K": K, "partition": kind, "draw": d, **r})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "descriptors.csv", index=False)
    print(df.groupby(["K", "partition"], sort=False).mean(numeric_only=True).drop(columns="draw").round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
