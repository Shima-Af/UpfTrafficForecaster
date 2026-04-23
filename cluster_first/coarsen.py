"""
coarsen.py — Graph coarsening from 965 BSs to K cluster super-nodes.

After clustering, this module aggregates the fine-grained BS-level traffic
signals into K cluster-level signals and builds the coarsened K-node graph.

build_coarsened_graph():
    - Sums dl_norm across all BSs within each cluster at every 15-min slot,
      producing a (K, T_total) array of cluster aggregate traffic.
    - Builds a coarse adjacency: two clusters are connected if any BS in one
      is Voronoi-adjacent to any BS in the other; edge weights are the mean
      affinity across all such cross-cluster edges.
    - Adds self-loops with weight 1.0 (standard for message-passing stability).

verify_coarsening():
    - Checks that the sum of all cluster signals equals the sum of all BS
      signals at every timestamp (conservation of traffic — no double-counting).

The conservation property is the key correctness guarantee: the STGNN on the
coarsened graph is forecasting exactly the same total traffic as the original
fine-grained model, just partitioned into K geographically coherent chunks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Internal helper: load and aggregate parquets to BS level
# (mirrors src/dataset._load_bs_parquets without importing from src/)
# ---------------------------------------------------------------------------

def _load_bs_level(processed_dir: Path) -> pd.DataFrame:
    """
    Load all 77 daily parquets and produce a single byte-proportional dl_norm
    per (site_id, timestamp) — same logic as src/dataset._load_bs_parquets.

    Services are first converted back to raw bytes using the per-service
    scale_factor from metadata.json, summed across services, then
    re-normalised by the combined global max so dl_norm ∈ [0, 1].
    """
    metadata_path = processed_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {metadata_path}. "
            "Expected path: data/netmob/processed/metadata.json"
        )
    with open(metadata_path) as f:
        meta = json.load(f)
    service_scales: Dict[str, float] = {
        svc: info["scale_factor"] for svc, info in meta["services"].items()
    }

    parquets = sorted(processed_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"No parquet files found in {processed_dir}. "
            "Run `dvc repro preprocess` first."
        )

    frames: List[pd.DataFrame] = []
    for pq in tqdm(parquets, desc="Loading parquets", unit="day"):
        frames.append(pd.read_parquet(pq))
    df = pd.concat(frames, ignore_index=True)

    # Convert per-service dl_norm → raw bytes, then sum across services
    df["dl_bytes"] = df["dl_norm"] * df["service"].map(service_scales)
    df = (
        df.groupby(["timestamp", "site_id"], sort=False)["dl_bytes"]
          .sum()
          .reset_index()
    )

    # Re-normalise to [0, 1] relative to combined peak
    global_max = float(df["dl_bytes"].max())
    df["dl_norm"] = (df["dl_bytes"] / global_max).astype("float32")
    df.drop(columns="dl_bytes", inplace=True)
    df.sort_values(["site_id", "timestamp"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def build_coarsened_graph(
    cluster_labels: np.ndarray,
    adj_matrix_dense: np.ndarray,
    affinity_matrix: np.ndarray,
    daily_parquets_dir: str | Path,
    node_index_df: pd.DataFrame,
    output_dir: str | Path = "data/cluster_first",
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, Dict[int, List[str]]]:
    """
    Build a K-node coarsened graph from the 965-BS fine graph.

    Parameters
    ----------
    cluster_labels     : (N,) int array — cluster_id for each node in
                         node_index order (node_idx 0..N-1 → cluster 0..K-1)
    adj_matrix_dense   : (N, N) float — inverse-distance Voronoi weights,
                         used to determine which cluster pairs are connected
    affinity_matrix    : (N, N) float — composite affinity from clustering,
                         used as edge weights in the coarse graph
    daily_parquets_dir : path to data/netmob/processed/
    node_index_df      : DataFrame with columns [site_id, node_idx], mapping
                         BS identifiers to the integer order in cluster_labels
    output_dir         : directory to save coarsening artefacts

    Returns
    -------
    coarse_edge_index : (2, E_coarse) torch.LongTensor
    coarse_edge_attr  : (E_coarse,)   torch.FloatTensor — mean inter-cluster affinity
    cluster_series    : (K, T_total)  numpy float32 — summed dl_norm per cluster/slot
    cluster_bs_map    : dict {cluster_id: [site_id, ...]}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_parquets_dir = Path(daily_parquets_dir)

    K = int(cluster_labels.max()) + 1
    N = len(cluster_labels)

    # Map node_idx → cluster_id and site_id → cluster_id
    node2cluster: Dict[int, int] = {
        int(row.node_idx): int(cluster_labels[int(row.node_idx)])
        for row in node_index_df.itertuples()
    }
    site2cluster: Dict[str, int] = {
        str(row.site_id): node2cluster[int(row.node_idx)]
        for row in node_index_df.itertuples()
    }

    # Build reverse map: cluster_id → [site_id, ...]
    cluster_bs_map: Dict[int, List[str]] = {k: [] for k in range(K)}
    for site_id, cluster_id in site2cluster.items():
        cluster_bs_map[cluster_id].append(site_id)

    # ---- Step 1: build cluster_series (K, T_total) ----
    df = _load_bs_level(daily_parquets_dir)
    df["cluster_id"] = df["site_id"].map(site2cluster)

    # Drop any sites not in node_index (shouldn't happen, but be safe)
    n_dropped = df["cluster_id"].isna().sum()
    if n_dropped > 0:
        print(f"[coarsen] Warning: {n_dropped} rows have unknown site_id — dropped")
        df = df.dropna(subset=["cluster_id"])
    df["cluster_id"] = df["cluster_id"].astype(int)

    # Sum dl_norm across all BSs in the same cluster at the same timestamp
    cluster_df = (
        df.groupby(["timestamp", "cluster_id"], sort=True)["dl_norm"]
          .sum()
          .reset_index()
    )

    # Pivot to (T_total, K) then transpose to (K, T_total)
    pivot = cluster_df.pivot(index="timestamp", columns="cluster_id", values="dl_norm")
    pivot = pivot.sort_index()
    pivot = pivot.fillna(0.0)

    # Ensure all K clusters are present as columns
    for k in range(K):
        if k not in pivot.columns:
            pivot[k] = 0.0
    pivot = pivot[list(range(K))]

    cluster_series = pivot.values.T.astype(np.float32)   # (K, T_total)
    T_total = cluster_series.shape[1]
    print(f"[coarsen] cluster_series shape: ({K}, {T_total})  "
          f"(K clusters × {T_total} timestamps)")

    # ---- Step 2: build coarse adjacency ----
    # Two clusters k, l are connected if any fine edge (i→j) has
    # cluster_labels[i]=k and cluster_labels[j]=l (with k ≠ l).
    # Edge weight = mean affinity_matrix[i,j] over all such cross-cluster edges.
    src_nodes, dst_nodes = np.nonzero(adj_matrix_dense > 0)

    # Accumulate cross-cluster edge affinities
    cross_affinities: Dict[Tuple[int, int], List[float]] = {}
    for i, j in zip(src_nodes, dst_nodes):
        ci = int(cluster_labels[i])
        cj = int(cluster_labels[j])
        if ci == cj:
            continue
        key = (ci, cj)
        if key not in cross_affinities:
            cross_affinities[key] = []
        cross_affinities[key].append(float(affinity_matrix[i, j]))

    coarse_src: List[int] = []
    coarse_dst: List[int] = []
    coarse_wts: List[float] = []

    for (ci, cj), affs in cross_affinities.items():
        coarse_src.append(ci)
        coarse_dst.append(cj)
        coarse_wts.append(float(np.mean(affs)))

    # Self-loops with weight 1.0
    for k in range(K):
        coarse_src.append(k)
        coarse_dst.append(k)
        coarse_wts.append(1.0)

    coarse_edge_index = torch.tensor(
        [coarse_src, coarse_dst], dtype=torch.long
    )
    coarse_edge_attr = torch.tensor(coarse_wts, dtype=torch.float32)

    print(
        f"[coarsen] coarse_edge_index: {coarse_edge_index.shape}  "
        f"({len(coarse_src) - K} inter-cluster edges + {K} self-loops)"
    )

    # ---- Step 3: save artefacts ----
    np.save(output_dir / "cluster_series.npy",    cluster_series)
    np.save(output_dir / "coarse_edge_index.npy", coarse_edge_index.numpy())
    np.save(output_dir / "coarse_edge_attr.npy",  coarse_edge_attr.numpy())

    # Save cluster_bs_map as JSON for downstream use
    with open(output_dir / "cluster_bs_map.json", "w") as f:
        json.dump(cluster_bs_map, f, indent=2)

    return coarse_edge_index, coarse_edge_attr, cluster_series, cluster_bs_map


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_coarsening(
    cluster_series: np.ndarray,
    daily_parquets_dir: str | Path,
    tol: float = 1e-4,
) -> None:
    """
    Sanity check: the sum of all cluster signals at each timestamp must equal
    the sum of all BS dl_norm values at that timestamp.

    This confirms that graph coarsening is lossless — no traffic is dropped
    or double-counted when forming cluster aggregates.  The pipeline should
    be aborted if this check fails, as downstream UPF energy estimates would
    be incorrect.

    Parameters
    ----------
    cluster_series     : (K, T_total) float32 — summed cluster signals
    daily_parquets_dir : path to data/netmob/processed/
    tol                : maximum allowed absolute error per timestamp

    Raises
    ------
    AssertionError if max absolute error > tol
    """
    daily_parquets_dir = Path(daily_parquets_dir)
    df = _load_bs_level(daily_parquets_dir)

    # Sum of all BSs at each timestamp
    bs_totals = (
        df.groupby("timestamp", sort=True)["dl_norm"]
          .sum()
          .sort_index()
          .values.astype(np.float32)
    )

    # Sum of all clusters at each timestamp: cluster_series.sum(axis=0)
    cluster_totals = cluster_series.sum(axis=0)   # (T_total,)

    if len(bs_totals) != len(cluster_totals):
        raise AssertionError(
            f"verify_coarsening: T mismatch — BS totals has {len(bs_totals)} "
            f"timestamps, cluster_series has {len(cluster_totals)}"
        )

    abs_err = np.abs(bs_totals - cluster_totals)
    max_err = float(abs_err.max())
    mean_err = float(abs_err.mean())
    print(
        f"[verify_coarsening] max_abs_err={max_err:.6f}  mean_abs_err={mean_err:.6f}  "
        f"tol={tol}  {'PASS ✓' if max_err <= tol else 'FAIL ✗'}"
    )

    if max_err > tol:
        raise AssertionError(
            f"Coarsening conservation check FAILED: max absolute error "
            f"{max_err:.6f} exceeds tolerance {tol}. Check cluster_labels "
            f"ordering matches node_index_df and all BSs have cluster assignments."
        )
