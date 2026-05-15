"""
evaluate_forecast.py — Cluster quality and STGNN forecast evaluation.

Evaluates trained ClusterSTGNN checkpoints across all K values in the sweep.
Produces two categories of metrics:

  1. Clustering quality  — size balance, geographic compactness, traffic
     balance, coarsening conservation, contiguity in the Voronoi graph.
  2. Forecast accuracy   — overall MAE / RMSE / WAPE, per-cluster and
     per-horizon breakdowns, peak/low-load period analysis, error percentile
     distribution.  Compared against last-value, historical-mean, and
     seasonal-naive baselines.

No UPF assignment, energy, or SLA metrics are computed here.  Those belong
to the downstream orchestration / digital-twin repository.

Outputs
-------
results/<service>/K<K>/forecast_eval.json     — per-K detailed results
results/<service>/K<K>/predictions_test.npy   — raw model predictions on test window (n_test, horizon, K)
results/<service>/K<K>/targets_test.npy       — ground-truth targets on test window  (n_test, horizon, K)
results/<service>/K<K>/predictions_train.npy  — in-sample model predictions on train window (n_train, horizon, K)
results/<service>/K<K>/targets_train.npy      — ground-truth targets on train window         (n_train, horizon, K)
results/<service>/forecast_eval_summary.json  — K-sweep comparison table
figures/<service>/K<K>/                        — plots

The train-window predictions are in-sample (same checkpoint that was trained on
this window). They are intended for downstream RL training in UpfRLControllers,
which uses the train window for PPO episodes and reserves the test window for
held-out evaluation.

Run with:
    python -m src.evaluate_forecast [--config config.yaml] [--with-coherence]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.cm as mpl_cm
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.cluster_dataset import ClusterTrafficDataset
from src.model import ClusterSTGNN


# ---------------------------------------------------------------------------
# Geographic helper
# ---------------------------------------------------------------------------

def _haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorised Haversine distance in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Contiguity check
# ---------------------------------------------------------------------------

def _check_contiguity(
    cluster_assignments: pd.DataFrame,
    node_index_df: pd.DataFrame,
    edge_index_np: np.ndarray,
    n_clusters: int,
) -> Dict[int, Dict[str, Any]]:
    """
    For each cluster, verify whether member BSs form a connected subgraph
    in the fine-grained Voronoi adjacency.
    """
    merged = cluster_assignments.merge(node_index_df, on="site_id", how="inner")
    node2cluster = dict(
        zip(merged["node_idx"].astype(int), merged["cluster_id"].astype(int))
    )

    src_all, dst_all = edge_index_np[0], edge_index_np[1]
    results: Dict[int, Dict[str, Any]] = {}

    for k in range(n_clusters):
        nodes_k = np.array([n for n, c in node2cluster.items() if c == k], dtype=np.int64)

        if len(nodes_k) < 2:
            results[k] = {"n_components": 1, "is_connected": True}
            continue

        node_set = set(nodes_k.tolist())
        mask = np.fromiter(
            (s in node_set and d in node_set for s, d in zip(src_all, dst_all)),
            dtype=bool,
            count=len(src_all),
        )
        sub_src = src_all[mask]
        sub_dst = dst_all[mask]

        if len(sub_src) == 0:
            results[k] = {"n_components": int(len(nodes_k)), "is_connected": False}
            continue

        uniq = np.unique(nodes_k)
        remap = {int(old): new for new, old in enumerate(uniq)}
        n_k = len(uniq)
        sub_src_r = np.array([remap[int(s)] for s in sub_src])
        sub_dst_r = np.array([remap[int(d)] for d in sub_dst])
        rows = np.concatenate([sub_src_r, sub_dst_r])
        cols = np.concatenate([sub_dst_r, sub_src_r])
        adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_k, n_k))
        n_comp, _ = connected_components(adj, directed=False)
        results[k] = {"n_components": int(n_comp), "is_connected": n_comp == 1}

    return results


# ---------------------------------------------------------------------------
# Baseline forecasters
# All return shape (n_samples, horizon, n_clusters)
# ---------------------------------------------------------------------------

def _last_value_forecast(
    cluster_series: np.ndarray,
    seq_len: int,
    horizon: int,
    test_start: int,
    n_samples: int,
) -> np.ndarray:
    """ŝ(t+h) = s(t) for all h — last observed value repeated."""
    preds = []
    for i in range(n_samples):
        t_last = test_start + i + seq_len - 1
        last_val = cluster_series[:, t_last]                  # (K,)
        preds.append(np.stack([last_val] * horizon, axis=0))  # (horizon, K)
    return np.stack(preds, axis=0)                             # (n, horizon, K)


def _historical_mean_forecast(
    cluster_series: np.ndarray,
    n_train: int,
    horizon: int,
    n_samples: int,
) -> np.ndarray:
    """ŝ(t+h) = mean(s_train) for all h."""
    train_mean = cluster_series[:, :n_train].mean(axis=1)     # (K,)
    pred_row   = np.stack([train_mean] * horizon, axis=0)     # (horizon, K)
    return np.stack([pred_row] * n_samples, axis=0)           # (n, horizon, K)


def _seasonal_naive_forecast(
    cluster_series: np.ndarray,
    seq_len: int,
    horizon: int,
    test_start: int,
    n_samples: int,
    season_len: int = 672,   # 7 × 96 slots = 7 days
) -> np.ndarray:
    """ŝ(t+h) = s(t+h − season_len) — same slot one week ago."""
    _k, t_total = cluster_series.shape
    preds = []
    for i in range(n_samples):
        t_target = test_start + i + seq_len          # first target slot (global)
        t_season = t_target - season_len
        if t_season >= 0 and t_season + horizon <= t_total:
            val = cluster_series[:, t_season: t_season + horizon]  # (K, horizon)
            preds.append(val.T)                                     # (horizon, K)
        else:
            # Fall back to last-value if not enough history
            t_last = test_start + i + seq_len - 1
            last_val = cluster_series[:, t_last]
            preds.append(np.stack([last_val] * horizon, axis=0))
    return np.stack(preds, axis=0)   # (n, horizon, K)


# ---------------------------------------------------------------------------
# Forecast metrics
# ---------------------------------------------------------------------------

def _forecast_metrics(
    preds: np.ndarray,
    trues: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute forecast quality metrics for arrays of shape (n, horizon, K).

    WAPE (Weighted Absolute Percentage Error) is used instead of MAPE because
    MAPE explodes on near-zero sparse traffic values.
    """
    _n, horizon, n_clusters = preds.shape
    abs_err = np.abs(preds - trues)

    mae  = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    wape = float(abs_err.sum() / (np.abs(trues).sum() + 1e-12) * 100)

    per_cluster_mae  = [float(np.mean(abs_err[:, :, k])) for k in range(n_clusters)]
    per_cluster_rmse = [
        float(np.sqrt(np.mean((preds[:, :, k] - trues[:, :, k]) ** 2)))
        for k in range(n_clusters)
    ]
    per_horizon_mae  = [float(np.mean(abs_err[:, h, :])) for h in range(horizon)]

    # Peak / low-load split — based on mean actual traffic at h=1 across all clusters
    sample_load = trues[:, 0, :].mean(axis=1)
    q75  = float(np.percentile(sample_load, 75))
    q25  = float(np.percentile(sample_load, 25))
    peak_mask = sample_load >= q75
    low_mask  = sample_load <= q25
    peak_mae  = float(np.mean(abs_err[peak_mask])) if peak_mask.any() else None
    low_mae   = float(np.mean(abs_err[low_mask]))  if low_mask.any()  else None

    ae_flat = abs_err.ravel()
    return {
        "mae":                 mae,
        "rmse":                rmse,
        "wape":                wape,
        "per_cluster_mae":     per_cluster_mae,
        "per_cluster_rmse":    per_cluster_rmse,
        "per_horizon_mae":     per_horizon_mae,
        "peak_period_mae":     peak_mae,
        "low_load_period_mae": low_mae,
        "abs_err_median":      float(np.median(ae_flat)),
        "abs_err_p90":         float(np.percentile(ae_flat, 90)),
        "abs_err_p95":         float(np.percentile(ae_flat, 95)),
    }


# ---------------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------------

def _clustering_metrics(
    cluster_assignments: pd.DataFrame,
    bs_locations: pd.DataFrame,
    cluster_series: np.ndarray,
    node_index_df: pd.DataFrame,
    edge_index_np: np.ndarray,
    n_clusters: int,
    processed_dir: Optional[Path],
    service: str,
    with_coherence: bool,
) -> Dict[str, Any]:
    """Compute clustering quality metrics for a K-partition."""
    df = cluster_assignments.merge(bs_locations, on="site_id", how="left")

    # ---- Cluster sizes ----
    sizes = np.array(
        [(df["cluster_id"] == k).sum() for k in range(n_clusters)], dtype=np.int64
    )
    size_stats: Dict[str, Any] = {
        "per_cluster_n_bss": sizes.tolist(),
        "min":  int(sizes.min()),
        "max":  int(sizes.max()),
        "mean": float(sizes.mean()),
        "std":  float(sizes.std()),
    }

    # ---- Geographic compactness ----
    compactness_per_cluster: List[Optional[float]] = []
    for k in range(n_clusters):
        subset = df[df["cluster_id"] == k][["lat", "lon"]].dropna()
        if len(subset) < 1:
            compactness_per_cluster.append(None)
            continue
        lat_c = float(subset["lat"].mean())
        lon_c = float(subset["lon"].mean())
        dists = _haversine_km(
            subset["lat"].values,
            subset["lon"].values,
            np.full(len(subset), lat_c),
            np.full(len(subset), lon_c),
        )
        compactness_per_cluster.append(float(dists.mean()))

    valid_c = [v for v in compactness_per_cluster if v is not None]
    geo_compactness: Dict[str, Any] = {
        "mean_dist_to_centroid_km_per_cluster": compactness_per_cluster,
        "overall_mean_km": float(np.mean(valid_c))  if valid_c else None,
        "overall_max_km":  float(max(valid_c))       if valid_c else None,
    }

    # ---- Traffic balance ----
    total_load = cluster_series.sum(axis=1)    # (K,)
    mean_load  = cluster_series.mean(axis=1)   # (K,)
    load_cv    = float(total_load.std() / (total_load.mean() + 1e-12))
    traffic_balance: Dict[str, Any] = {
        "total_load_per_cluster": total_load.tolist(),
        "mean_load_per_cluster":  mean_load.tolist(),
        "load_cv":                load_cv,
    }

    # ---- Contiguity in original Voronoi graph ----
    contiguity: Dict[str, Any] = {}
    try:
        cont = _check_contiguity(cluster_assignments, node_index_df, edge_index_np, n_clusters)
        contiguity = {
            "per_cluster_n_components": [cont[k]["n_components"] for k in range(n_clusters)],
            "per_cluster_is_connected": [cont[k]["is_connected"]  for k in range(n_clusters)],
            "all_connected":            all(cont[k]["is_connected"] for k in range(n_clusters)),
        }
    except Exception as exc:  # noqa: BLE001
        contiguity = {"error": str(exc)}

    # ---- Coarsening conservation ----
    conservation: Dict[str, Any] = {
        "cluster_series_total_sum": float(cluster_series.sum()),
    }
    if with_coherence and processed_dir is not None:
        try:
            from src.coarsen import verify_coarsening  # noqa: PLC0415
            valid_site_ids = set(node_index_df["site_id"].tolist())
            verify_coarsening(
                cluster_series=cluster_series,
                daily_parquets_dir=processed_dir,
                valid_site_ids=valid_site_ids,
                service=service,
            )
            conservation["passed"] = True
        except AssertionError as exc:
            conservation["passed"] = False
            conservation["message"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            conservation["error"] = str(exc)
    else:
        conservation["note"] = (
            "Full verification skipped — pass --with-coherence to enable. "
            "Conservation is verified during training by verify_coarsening()."
        )

    # ---- Temporal coherence (expensive, optional) ----
    temporal_coherence: Optional[Dict[str, Any]] = None
    if with_coherence and processed_dir is not None:
        print("    [coherence] Loading parquets for temporal coherence …")
        try:
            from src.coarsen import _load_bs_level  # noqa: PLC0415
            bs_df = _load_bs_level(processed_dir, service=service)
            pivot = (
                bs_df.pivot(index="site_id", columns="timestamp", values="dl_norm")
                     .fillna(0.0)
            )
            coherence_per_cluster: List[Optional[float]] = []
            for k in range(n_clusters):
                site_ids_k = cluster_assignments[
                    cluster_assignments["cluster_id"] == k
                ]["site_id"].tolist()
                available = [s for s in site_ids_k if s in pivot.index]
                if len(available) < 2:
                    coherence_per_cluster.append(None)
                    continue
                x_mat = pivot.loc[available].values
                corr  = np.corrcoef(x_mat)
                upper = corr[np.triu_indices(len(available), k=1)]
                coherence_per_cluster.append(float(np.nanmean(upper)))

            valid_coh = [v for v in coherence_per_cluster if v is not None]
            temporal_coherence = {
                "mean_within_cluster_corr_per_cluster": coherence_per_cluster,
                "overall_mean": float(np.mean(valid_coh)) if valid_coh else None,
            }
        except Exception as exc:  # noqa: BLE001
            temporal_coherence = {"error": str(exc)}

    return {
        "size_stats":         size_stats,
        "geo_compactness":    geo_compactness,
        "traffic_balance":    traffic_balance,
        "contiguity":         contiguity,
        "conservation":       conservation,
        "temporal_coherence": temporal_coherence,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_forecast_vs_actual(
    preds: np.ndarray,
    trues: np.ndarray,
    n_clusters: int,
    figures_dir: Path,
) -> None:
    p = preds[:, 0, :]   # h=1 step ahead
    t = trues[:, 0, :]
    n_cols = min(n_clusters, 2)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), squeeze=False)
    for k in range(n_clusters):
        ax = axes[k // n_cols][k % n_cols]
        ax.plot(t[:, k], label="Actual",    color="steelblue", linewidth=0.7, alpha=0.9)
        ax.plot(p[:, k], label="Predicted", color="tomato",    linewidth=0.7, alpha=0.8)
        mae_k = float(np.mean(np.abs(p[:, k] - t[:, k])))
        ax.set_title(f"Cluster {k}  (MAE={mae_k:.4f})", fontsize=8)
        ax.set_xlabel("Test sample index")
        ax.set_ylabel("dl_norm (cluster sum)")
        ax.legend(fontsize=7)
    for k in range(n_clusters, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_visible(False)
    fig.suptitle(f"Predicted vs Actual — K={n_clusters}, h=1 step ahead", fontsize=11)
    plt.tight_layout()
    _savefig(fig, figures_dir / f"forecast_vs_actual_K{n_clusters}.png")


def _plot_per_horizon_mae(
    metrics_dict: Dict[str, Dict[str, Any]],
    n_clusters: int,
    figures_dir: Path,
) -> None:
    horizon = len(next(iter(metrics_dict.values()))["per_horizon_mae"])
    methods  = list(metrics_dict.keys())
    x        = np.arange(horizon)
    n_m      = len(methods)
    width    = 0.7 / n_m
    fig, ax  = plt.subplots(figsize=(8, 4))
    for i, method in enumerate(methods):
        offset = (i - n_m / 2 + 0.5) * width
        ax.bar(x + offset, metrics_dict[method]["per_horizon_mae"], width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h+1}" for h in range(horizon)])
    ax.set_ylabel("MAE (dl_norm units)")
    ax.set_title(f"Per-Horizon MAE — K={n_clusters}")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, figures_dir / f"per_horizon_mae_K{n_clusters}.png")


def _plot_per_cluster_mae(
    metrics_dict: Dict[str, Dict[str, Any]],
    n_clusters: int,
    figures_dir: Path,
) -> None:
    methods = list(metrics_dict.keys())
    x       = np.arange(n_clusters)
    n_m     = len(methods)
    width   = 0.7 / n_m
    fig, ax = plt.subplots(figsize=(max(8, n_clusters * 0.6), 4))
    for i, method in enumerate(methods):
        offset = (i - n_m / 2 + 0.5) * width
        ax.bar(x + offset, metrics_dict[method]["per_cluster_mae"], width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{k}" for k in range(n_clusters)], fontsize=8)
    ax.set_ylabel("MAE (dl_norm units)")
    ax.set_title(f"Per-Cluster MAE — K={n_clusters}")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, figures_dir / f"per_cluster_mae_K{n_clusters}.png")


def _plot_cluster_sizes(
    sizes: List[int],
    n_clusters: int,
    figures_dir: Path,
) -> None:
    mean_size = float(np.mean(sizes))
    fig, ax = plt.subplots(figsize=(max(6, n_clusters * 0.6), 4))
    ax.bar(range(n_clusters), sizes, color="steelblue")
    ax.axhline(mean_size, color="tomato", linestyle="--", label=f"mean = {mean_size:.1f}")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of BSs")
    ax.set_title(f"Cluster Size Distribution — K={n_clusters}")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, figures_dir / f"cluster_sizes_K{n_clusters}.png")


def _plot_cluster_map(
    cluster_assignments: pd.DataFrame,
    bs_locations: pd.DataFrame,
    n_clusters: int,
    figures_dir: Path,
) -> None:
    df = cluster_assignments.merge(bs_locations, on="site_id", how="left")
    if df[["lat", "lon"]].isna().any().any():
        return
    cmap   = mpl_cm.get_cmap("tab20" if n_clusters <= 20 else "hsv")
    fig, ax = plt.subplots(figsize=(9, 8))
    for k in range(n_clusters):
        subset = df[df["cluster_id"] == k]
        ax.scatter(
            subset["lon"], subset["lat"],
            c=[cmap(k / n_clusters)],
            s=6, alpha=0.7, edgecolors="none",
            label=f"C{k}" if n_clusters <= 12 else None,
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Geographic Cluster Map — K={n_clusters}")
    if n_clusters <= 12:
        ax.legend(title="Cluster", markerscale=2, fontsize=7, ncol=2)
    plt.tight_layout()
    _savefig(fig, figures_dir / f"cluster_map_K{n_clusters}.png")


# ---------------------------------------------------------------------------
# Per-K evaluation
# ---------------------------------------------------------------------------

def _evaluate_k(
    n_clusters: int,
    service: str,
    output_base: Path,
    ckpt_base: Path,
    results_base: Path,
    figures_base: Path,
    graphs_dir: Path,
    processed_dir: Path,
    stgnn_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    cfg: Dict[str, Any],
    with_coherence: bool,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    k_output_dir  = output_base  / f"K{n_clusters}"
    k_ckpt_dir    = ckpt_base    / f"K{n_clusters}"
    k_results_dir = results_base / f"K{n_clusters}"
    k_figures_dir = figures_base / f"K{n_clusters}"

    required = [
        k_output_dir / "cluster_series.npy",
        k_output_dir / "coarse_edge_index.npy",
        k_output_dir / "coarse_edge_attr.npy",
        k_output_dir / "cluster_assignments.parquet",
        k_ckpt_dir   / "best_model.pt",
        graphs_dir   / "bs_locations.parquet",
        graphs_dir   / "edge_index.npy",
        graphs_dir   / "node_index.parquet",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print(f"[evaluate_forecast] K={n_clusters}: skipping — missing files:")
        for m in missing:
            print(f"    {m}")
        return None

    print(f"\n{'='*60}")
    print(f" K={n_clusters}")
    print(f"{'='*60}")

    # ---- Load artefacts ----
    cluster_series      = np.load(k_output_dir / "cluster_series.npy")       # (K, T)
    coarse_ei           = torch.from_numpy(
        np.load(k_output_dir / "coarse_edge_index.npy")).long()
    coarse_ea           = torch.from_numpy(
        np.load(k_output_dir / "coarse_edge_attr.npy"))
    cluster_assignments = pd.read_parquet(k_output_dir / "cluster_assignments.parquet")
    bs_locations        = pd.read_parquet(graphs_dir   / "bs_locations.parquet")
    edge_index_np       = np.load(graphs_dir           / "edge_index.npy")
    node_index_df       = pd.read_parquet(graphs_dir   / "node_index.parquet")

    # ---- Model ----
    ckpt  = torch.load(k_ckpt_dir / "best_model.pt", map_location=device)
    model = ClusterSTGNN.from_config(cfg, n_clusters=n_clusters).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(
        f"[evaluate_forecast] Checkpoint loaded  "
        f"epoch={ckpt['epoch']}  val_mae={ckpt['val_mae']:.5f}"
    )

    # ---- Temporal split boundaries ----
    seq_len    = stgnn_cfg["seq_len"]
    horizon    = stgnn_cfg["horizon"]
    val_split  = training_cfg["val_split"]
    test_split = training_cfg["test_split"]
    batch_size = training_cfg["batch_size"]

    _k_dim, t_total = cluster_series.shape
    n_test    = max(horizon + seq_len, int(t_total * test_split))
    n_val     = max(horizon + seq_len, int(t_total * val_split))
    n_train   = t_total - n_val - n_test
    test_start = n_train + n_val

    def _run_inference(split_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Run the trained model on a given split and return (preds, trues)."""
        ds = ClusterTrafficDataset(
            cluster_series=cluster_series,
            edge_index=coarse_ei, edge_attr=coarse_ea,
            seq_len=seq_len, horizon=horizon,
            split=split_name,
            val_split=val_split, test_split=test_split,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        p_chunks, t_chunks = [], []
        with torch.no_grad():
            for batch in loader:
                x_b  = batch["x"].to(device)
                y_b  = batch["y"].to(device)
                ei_b = batch["edge_index"][0].to(device)
                ea_b = batch["edge_attr"][0].to(device)
                p_chunks.append(model(x_b, ei_b, ea_b).cpu())
                t_chunks.append(y_b.cpu())
        return (
            torch.cat(p_chunks, dim=0).numpy(),   # (n, horizon, K)
            torch.cat(t_chunks, dim=0).numpy(),
        )

    # ---- Test-window inference (held-out for downstream PPO evaluation) ----
    preds_np, trues_np = _run_inference("test")
    n_samples = preds_np.shape[0]

    # ---- Train-window in-sample inference (for downstream PPO training) ----
    # Same checkpoint, same input-window logic, just applied to the train range.
    # In-sample is the standard pragmatic choice; walk-forward CV is out of scope.
    preds_train_np, trues_train_np = _run_inference("train")

    # ---- Save raw predictions for both splits ----
    k_results_dir.mkdir(parents=True, exist_ok=True)
    np.save(k_results_dir / "predictions_test.npy",  preds_np)
    np.save(k_results_dir / "targets_test.npy",      trues_np)
    np.save(k_results_dir / "predictions_train.npy", preds_train_np)
    np.save(k_results_dir / "targets_train.npy",     trues_train_np)
    print(
        f"[evaluate_forecast] Predictions saved → {k_results_dir}\n"
        f"    test:  preds={preds_np.shape}  targets={trues_np.shape}\n"
        f"    train: preds={preds_train_np.shape}  targets={trues_train_np.shape}"
    )

    # ---- Forecast metrics ----
    model_metrics = _forecast_metrics(preds_np, trues_np)

    lv_preds = _last_value_forecast(cluster_series, seq_len, horizon, test_start, n_samples)
    hm_preds = _historical_mean_forecast(cluster_series, n_train, horizon, n_samples)
    sn_preds = _seasonal_naive_forecast(cluster_series, seq_len, horizon, test_start, n_samples)

    lv_metrics = _forecast_metrics(lv_preds[:n_samples], trues_np)
    hm_metrics = _forecast_metrics(hm_preds[:n_samples], trues_np)
    sn_metrics = _forecast_metrics(sn_preds[:n_samples], trues_np)

    print(
        f"\n  Forecast WAPE (lower is better):\n"
        f"    Model:           {model_metrics['wape']:.2f}%\n"
        f"    Last-value:      {lv_metrics['wape']:.2f}%\n"
        f"    Historical-mean: {hm_metrics['wape']:.2f}%\n"
        f"    Seasonal-naive:  {sn_metrics['wape']:.2f}%"
    )

    # ---- Clustering metrics ----
    print(f"\n  Computing clustering metrics …")
    clust = _clustering_metrics(
        cluster_assignments=cluster_assignments,
        bs_locations=bs_locations,
        cluster_series=cluster_series,
        node_index_df=node_index_df,
        edge_index_np=edge_index_np,
        n_clusters=n_clusters,
        processed_dir=processed_dir if with_coherence else None,
        service=service,
        with_coherence=with_coherence,
    )
    print(
        f"  Sizes: min={clust['size_stats']['min']}  "
        f"max={clust['size_stats']['max']}  "
        f"mean={clust['size_stats']['mean']:.1f}  "
        f"std={clust['size_stats']['std']:.1f}"
    )
    mean_km = clust["geo_compactness"]["overall_mean_km"]
    print(f"  Geo compactness (mean dist to centroid): {mean_km:.2f} km")
    print(f"  Traffic load CoV: {clust['traffic_balance']['load_cv']:.3f}")
    print(f"  All clusters connected: {clust['contiguity'].get('all_connected', 'N/A')}")

    # ---- Figures ----
    print(f"\n  Generating figures → {k_figures_dir}")
    all_metrics = {
        "Model":           model_metrics,
        "Last-value":      lv_metrics,
        "Hist-mean":       hm_metrics,
        "Seasonal-naive":  sn_metrics,
    }
    for fn, args in [
        (_plot_forecast_vs_actual, (preds_np, trues_np, n_clusters, k_figures_dir)),
        (_plot_per_horizon_mae,    (all_metrics, n_clusters, k_figures_dir)),
        (_plot_per_cluster_mae,    (all_metrics, n_clusters, k_figures_dir)),
        (_plot_cluster_sizes,      (clust["size_stats"]["per_cluster_n_bss"], n_clusters, k_figures_dir)),
        (_plot_cluster_map,        (cluster_assignments, bs_locations, n_clusters, k_figures_dir)),
    ]:
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001
            print(f"    {fn.__name__} skipped: {exc}")

    # ---- Assemble and save per-K result ----
    result: Dict[str, Any] = {
        "K":         n_clusters,
        "service":   service,
        "checkpoint": {
            "epoch":   int(ckpt["epoch"]),
            "val_mae": float(ckpt["val_mae"]),
        },
        "clustering": clust,
        "forecast": {
            "model":           model_metrics,
            "last_value":      lv_metrics,
            "hist_mean":       hm_metrics,
            "seasonal_naive":  sn_metrics,
        },
    }
    result_path = k_results_dir / "forecast_eval.json"
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n  Saved → {result_path}")
    return result


# ---------------------------------------------------------------------------
# Best-K selection
# ---------------------------------------------------------------------------

_WAPE_TOLERANCE = 0.05   # accept K if WAPE ≤ best_WAPE × (1 + tolerance)
_MAX_SIZE_CV    = 1.0    # cluster size std/mean — above this is "extreme imbalance"


def _select_best_k(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Recommend the smallest K whose forecast quality is within 5 % of the best,
    subject to clustering sanity checks.

    Stage 1 — both checks must pass:
      - all clusters are connected in the Voronoi graph
      - cluster size CV (std / mean) < _MAX_SIZE_CV

    Stage 2 — WAPE budget:
      Among passing K values keep those with test_wape ≤ best_wape × 1.05.
      Return the smallest (simplest partition).

    Fallback: relax to connected-only if both checks fail; then WAPE-only.
    """
    if not summaries:
        return {"best_k": None, "reason": "no completed K evaluations"}

    def _size_cv(r: Dict) -> float:
        return r["size_std"] / (r["size_mean"] + 1e-12)

    checks = [
        (lambda r: r.get("all_connected") is True and _size_cv(r) < _MAX_SIZE_CV,
         "connected + size_cv < 1.0"),
        (lambda r: r.get("all_connected") is True,
         "connected only (size balance relaxed)"),
        (lambda _: True,
         "no sanity checks passed — WAPE only"),
    ]

    for check_fn, label in checks:
        candidates = [r for r in summaries if check_fn(r)]
        if not candidates:
            continue
        best_wape = min(r["test_wape"] for r in candidates)
        eligible  = [r for r in candidates
                     if r["test_wape"] <= best_wape * (1.0 + _WAPE_TOLERANCE)]
        chosen    = min(eligible, key=lambda r: r["K"])
        return {
            "best_k":            chosen["K"],
            "best_k_wape":       round(chosen["test_wape"], 4),
            "best_wape_in_pool": round(best_wape, 4),
            "wape_gap_pct":      round(
                (chosen["test_wape"] - best_wape) / (best_wape + 1e-12) * 100, 2
            ),
            "sanity_filter": label,
        }

    return {"best_k": None, "reason": "unexpected state in _select_best_k"}


# ---------------------------------------------------------------------------
# MLflow helpers (evaluation side)
# ---------------------------------------------------------------------------

def _mlflow_setup(mlflow_cfg: Dict[str, Any]) -> None:
    """Configure tracking URI and experiment; no-op if MLflow is unavailable."""
    try:
        import mlflow  # noqa: PLC0415
        uri = mlflow_cfg.get("tracking_uri", "")
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(
            mlflow_cfg.get("experiment_name", "cluster_first_stgnn_forecasting")
        )
    except Exception:  # noqa: BLE001
        pass


def _mlflow_log_eval(
    mlflow_cfg: Dict[str, Any],
    n_clusters: int,
    cfg: Dict[str, Any],
    result: Dict[str, Any],
    k_results_dir: Path,
    k_figures_dir: Path,
    is_best: bool,
) -> None:
    """
    Open a dedicated MLflow eval run for one K and log params, metrics,
    clustering quality, and artifacts.  Completely non-blocking — any
    MLflow error is silently swallowed so evaluation output is never lost.
    """
    try:
        import mlflow  # noqa: PLC0415

        _mlflow_setup(mlflow_cfg)

        stgnn_cfg    = cfg["stgnn"]
        training_cfg = cfg["training"]
        service      = cfg.get("data", {}).get("service", "total")
        clust_method = cfg.get("clustering", {}).get("method", "skater")

        fm    = result["forecast"]["model"]
        clust = result["clustering"]
        geo_km = clust["geo_compactness"]["overall_mean_km"]
        size_cv = clust["size_stats"]["std"] / (clust["size_stats"]["mean"] + 1e-12)

        with mlflow.start_run(run_name=f"cluster_first_K{n_clusters}_eval"):

            mlflow.log_params({
                "K":                 n_clusters,
                "service":           service,
                "clustering_method": clust_method,
                "hidden_dim":        stgnn_cfg["hidden_dim"],
                "gat_heads":         stgnn_cfg["gat_heads"],
                "gat_layers":        stgnn_cfg["gat_layers"],
                "gru_layers":        stgnn_cfg["gru_layers"],
                "dropout":           stgnn_cfg["dropout"],
                "seq_len":           stgnn_cfg["seq_len"],
                "horizon":           stgnn_cfg["horizon"],
                "lr":                training_cfg["lr"],
                "batch_size":        training_cfg["batch_size"],
                "seed":              training_cfg["seed"],
            })

            # Forecast quality
            mlflow.log_metrics({
                "val_mae":             result["checkpoint"]["val_mae"],
                "test_mae":            fm["mae"],
                "test_rmse":           fm["rmse"],
                "test_wape":           fm["wape"],
                "peak_period_mae":     fm["peak_period_mae"]     or float("nan"),
                "low_load_period_mae": fm["low_load_period_mae"] or float("nan"),
                "abs_err_median":      fm["abs_err_median"],
                "abs_err_p90":         fm["abs_err_p90"],
                "abs_err_p95":         fm["abs_err_p95"],
                "lv_wape":             result["forecast"]["last_value"]["wape"],
                "hm_wape":             result["forecast"]["hist_mean"]["wape"],
                "sn_wape":             result["forecast"]["seasonal_naive"]["wape"],
            })

            # Per-horizon MAE: test_mae_h1 … test_mae_h{horizon}
            for h, mae_h in enumerate(fm["per_horizon_mae"], start=1):
                mlflow.log_metric(f"test_mae_h{h}", mae_h)

            # Clustering quality
            mlflow.log_metrics({
                "cluster_size_mean":  clust["size_stats"]["mean"],
                "cluster_size_std":   clust["size_stats"]["std"],
                "cluster_size_cv":    size_cv,
                "geo_compactness_km": geo_km if geo_km is not None else float("nan"),
                "traffic_load_cv":    clust["traffic_balance"]["load_cv"],
                "all_connected":      float(clust["contiguity"].get("all_connected") or 0),
            })

            mlflow.set_tag("is_best_k", str(is_best))

            # Artifacts
            for fname in [
                "forecast_eval.json",
                "predictions_test.npy", "targets_test.npy",
                "predictions_train.npy", "targets_train.npy",
            ]:
                p = k_results_dir / fname
                if p.exists():
                    mlflow.log_artifact(str(p))
            if k_figures_dir.exists():
                mlflow.log_artifacts(str(k_figures_dir), artifact_path="figures")

    except Exception:  # noqa: BLE001
        pass   # MLflow is optional — never block evaluation output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run cluster quality and forecast evaluation across all sweep_k values."""
    parser = argparse.ArgumentParser(
        description="Cluster quality and STGNN forecast evaluation"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--with-coherence",
        action="store_true",
        help=(
            "Enable expensive optional metrics: temporal coherence (within-cluster "
            "BS correlation) and coarsening conservation check. Both require loading "
            "all processed parquets."
        ),
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    service      = cfg.get("data", {}).get("service", "total")
    paths        = cfg["paths"]
    stgnn_cfg    = cfg["stgnn"]
    training_cfg = cfg["training"]
    mlflow_cfg   = cfg.get("mlflow", {})
    sweep_k: List[int] = training_cfg["sweep_k"]

    output_base   = Path(paths["output_dir"])      / service
    ckpt_base     = Path(paths["checkpoints_dir"]) / service
    results_base  = Path(paths["results_dir"])     / service
    # ---- Separate summary results path (hardcoded to 'total' for DVC) ----
    summary_results_base = Path(paths["results_dir"]) / "total"
    figures_base  = Path(paths["figures_dir"])     / service
    graphs_dir    = Path(paths["graphs_dir"])
    processed_dir = Path(paths["processed_dir"])

    device = torch.device("cpu")   # evaluation always runs on CPU

    summaries:  List[Dict[str, Any]]          = []
    completed:  List[tuple[int, Dict[str, Any]]] = []   # (K, full result) for MLflow

    for k_val in sweep_k:
        result = _evaluate_k(
            n_clusters=k_val,
            service=service,
            output_base=output_base,
            ckpt_base=ckpt_base,
            results_base=results_base,
            figures_base=figures_base,
            graphs_dir=graphs_dir,
            processed_dir=processed_dir,
            stgnn_cfg=stgnn_cfg,
            training_cfg=training_cfg,
            cfg=cfg,
            with_coherence=args.with_coherence,
            device=device,
        )
        if result is None:
            continue

        completed.append((k_val, result))
        geo_km = result["clustering"]["geo_compactness"]["overall_mean_km"]
        summaries.append({
            "K":           result["K"],
            "val_mae":     result["checkpoint"]["val_mae"],
            "test_mae":    result["forecast"]["model"]["mae"],
            "test_rmse":   result["forecast"]["model"]["rmse"],
            "test_wape":   result["forecast"]["model"]["wape"],
            "lv_wape":     result["forecast"]["last_value"]["wape"],
            "hm_wape":     result["forecast"]["hist_mean"]["wape"],
            "sn_wape":     result["forecast"]["seasonal_naive"]["wape"],
            "size_mean":   result["clustering"]["size_stats"]["mean"],
            "size_std":    result["clustering"]["size_stats"]["std"],
            "load_cv":     result["clustering"]["traffic_balance"]["load_cv"],
            "geo_mean_km": geo_km,
            "all_connected": result["clustering"]["contiguity"].get("all_connected"),
        })

    # ---- Best-K selection ----
    best_k_info = _select_best_k(summaries)
    best_k_val  = best_k_info.get("best_k")

    if best_k_val is not None:
        print(
            f"\n[evaluate_forecast] Best K = {best_k_val}  "
            f"(WAPE={best_k_info['best_k_wape']:.2f}%  "
            f"gap={best_k_info['wape_gap_pct']:.2f}%  "
            f"filter: {best_k_info['sanity_filter']})"
        )
    else:
        print(f"\n[evaluate_forecast] Best-K selection: {best_k_info.get('reason')}")

    # ---- Combined summary with safety fallback ----
    summary_obj: Dict[str, Any] = {
        "service":  service,
        "sweep_k":  sweep_k,
        "best_k":   best_k_info,
        "results":  summaries,
    }
    summary_results_base.mkdir(parents=True, exist_ok=True)
    summary_path = summary_results_base / "forecast_eval_summary.json"
    try:
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary_obj, fh, indent=2)
        print(f"[evaluate_forecast] Summary saved → {summary_path}")
    except Exception as e:
        print(f"[evaluate_forecast] WARNING: Failed to save summary to {summary_path}: {e}")
        print(f"[evaluate_forecast] Creating fallback empty summary...")
        try:
            summary_results_base.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "service": service,
                    "sweep_k": sweep_k,
                    "best_k": best_k_info,
                    "results": [],
                    "error": str(e)
                }, fh, indent=2)
            print(f"[evaluate_forecast] Fallback summary created at {summary_path}")
        except Exception as e2:
            print(f"[evaluate_forecast] ERROR: Could not create fallback summary: {e2}")
            raise

    # ---- MLflow logging (one eval run per K) ----
    for k_val, result in completed:
        _mlflow_log_eval(
            mlflow_cfg=mlflow_cfg,
            n_clusters=k_val,
            cfg=cfg,
            result=result,
            k_results_dir=results_base  / f"K{k_val}",
            k_figures_dir=figures_base  / f"K{k_val}",
            is_best=(k_val == best_k_val),
        )

    # ---- Print comparison table ----
    if summaries:
        hdr = (
            f"{'K':>4}  {'val_MAE':>8}  {'test_MAE':>8}  {'test_WAPE':>9}  "
            f"{'LV_WAPE':>7}  {'SN_WAPE':>7}  "
            f"{'geo_km':>6}  {'load_cv':>7}  {'connected':>9}  {'best':>4}"
        )
        print(f"\n{'='*len(hdr)}")
        print(" K-SWEEP FORECAST EVALUATION SUMMARY")
        print(f"{'='*len(hdr)}")
        print(hdr)
        print("-" * len(hdr))
        for r in summaries:
            marker = " <--" if r["K"] == best_k_val else ""
            print(
                f"{r['K']:>4}  {r['val_mae']:>8.5f}  {r['test_mae']:>8.5f}  "
                f"{r['test_wape']:>8.2f}%  {r['lv_wape']:>6.2f}%  {r['sn_wape']:>6.2f}%  "
                f"{(r['geo_mean_km'] or 0.0):>6.2f}  {r['load_cv']:>7.3f}  "
                f"{str(r['all_connected']):>9}{marker}"
            )


if __name__ == "__main__":
    main()
