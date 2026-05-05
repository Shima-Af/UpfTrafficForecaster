"""
train.py — Full cluster-first STGNN training pipeline.

Execution order:
  1. Load config and set seeds.
  2. Compute per-BS statistics (mean/peak/shape features) from all 77 daily parquets.
  3. Build 965×965 dense fine adjacency from edge_index.npy + edge_attr.npy.
  4. Sweep over K ∈ sweep_k:
       a. Run AttributedSpectralClustering → cluster assignments
       b. Build coarsened graph + cluster_series via build_coarsened_graph()
       c. Verify coarsening conservation
       d. Build ClusterTrafficDataset (train / val / test splits)
       e. Instantiate ClusterSTGNN, train with Adam + CosineAnnealingLR
       f. Early stopping on val MAE; save best checkpoint
       g. Log metrics to MLflow (or fallback JSON)
  5. Print K-sweep comparison table.

Run with:
    python -m src.train [--config config.yaml]
"""

from __future__ import annotations

import sys
# Force UTF-8 output on Windows (cp1252 console cannot encode arrows, dashes, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy.stats import entropy as scipy_entropy
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cluster import AttributedSpectralClustering, SkaterGeographicClustering
from src.coarsen import build_coarsened_graph, verify_coarsening
from src.cluster_dataset import ClusterTrafficDataset
from src.model import ClusterSTGNN


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# BS statistics
# ---------------------------------------------------------------------------

def compute_bs_stats(
    processed_dir: Path,
    node_index_df: pd.DataFrame,
    service: str = "total",
) -> pd.DataFrame:
    """
    Compute per-BS traffic statistics from all daily parquets.

    Features computed
    -----------------
    mean_load       : time-average of dl_norm (0–1)
    peak_load       : 95th-percentile dl_norm (robust to outliers)
    std_load        : temporal standard deviation
    peak_to_mean    : peak_load / (mean_load + 1e-8)
    temporal_entropy: Shannon entropy of the normalised time distribution
    coeff_variation : std_load / (mean_load + 1e-8)
    night_day_ratio : mean load slots 0–23 / mean load slots 32–79

    Parameters
    ----------
    processed_dir : path to data/netmob/processed/
    node_index_df : DataFrame [site_id, node_idx]
    service       : "total" or a specific service name (Netflix/YouTube/DailyMotion)
    """
    parquets = sorted(processed_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"No parquets in {processed_dir}. Run `dvc repro preprocess` first."
        )

    frames: List[pd.DataFrame] = []
    for pq in tqdm(parquets, desc="Loading BS stats", unit="day"):
        frames.append(pd.read_parquet(pq))
    df = pd.concat(frames, ignore_index=True)

    # With global_max normalization, dl_norm values are directly summable
    if service == "total":
        df = (
            df.groupby(["timestamp", "site_id"], sort=False)["dl_norm"]
              .sum()
              .reset_index()
        )
    else:
        df = df[df["service"] == service][["timestamp", "site_id", "dl_norm"]].copy()

    # Add slot-in-day column for night/day ratio
    df["slot"] = df["timestamp"].dt.hour * 4 + df["timestamp"].dt.minute // 15

    stats_rows: List[Dict] = []
    valid_sites = set(node_index_df["site_id"].astype(str))

    for site_id, grp in tqdm(
        df.groupby("site_id"), desc="Computing BS stats", unit="BS"
    ):
        if str(site_id) not in valid_sites:
            continue
        vals = grp["dl_norm"].values.astype(np.float64)
        if len(vals) == 0:
            continue

        mean_load  = float(np.mean(vals))
        peak_load  = float(np.percentile(vals, 95))
        std_load   = float(np.std(vals))
        p2m        = peak_load / (mean_load + 1e-8)
        cv         = std_load  / (mean_load + 1e-8)

        # Temporal entropy: normalise daily-average profile over slots 0–95
        slot_mean = grp.groupby("slot")["dl_norm"].mean().reindex(
            range(96), fill_value=0.0
        ).values.astype(np.float64)
        total = slot_mean.sum()
        if total > 0:
            te = float(scipy_entropy(slot_mean / total))
        else:
            te = 0.0

        # Night/day ratio
        night_mask = grp["slot"].isin(range(0, 24))        # 00:00–05:45
        day_mask   = grp["slot"].isin(range(32, 80))       # 08:00–19:45
        night_mean = float(grp.loc[night_mask, "dl_norm"].mean()) if night_mask.any() else 0.0
        day_mean   = float(grp.loc[day_mask,   "dl_norm"].mean()) if day_mask.any()   else 1e-8
        ndr = night_mean / (day_mean + 1e-8)

        stats_rows.append({
            "site_id":          str(site_id),
            "mean_load":        mean_load,
            "peak_load":        peak_load,
            "std_load":         std_load,
            "peak_to_mean":     p2m,
            "temporal_entropy": te,
            "coeff_variation":  cv,
            "night_day_ratio":  ndr,
        })

    stats_df = pd.DataFrame(stats_rows).set_index("site_id")

    # Reindex to match node_index ordering for alignment with adj_matrix
    ordered_sites = node_index_df.sort_values("node_idx")["site_id"].astype(str).tolist()
    stats_df = stats_df.reindex(ordered_sites)

    n_missing = stats_df.isnull().any(axis=1).sum()
    if n_missing > 0:
        print(f"[compute_bs_stats] Warning: {n_missing} BSs have missing stats — filling with 0")
        stats_df = stats_df.fillna(0.0)

    return stats_df


# ---------------------------------------------------------------------------
# Fine adjacency matrix
# ---------------------------------------------------------------------------

def build_fine_adj(
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """
    Build a dense (N, N) adjacency matrix from COO edge_index + edge_attr.

    Parameters
    ----------
    edge_index : (2, E) int64 — source/target node indices
    edge_attr  : (E,)  float32 — edge weights (inverse-distance)
    n_nodes    : N

    Returns
    -------
    (N, N) float64 dense adjacency (asymmetric, zeros on diagonal)
    """
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    adj[edge_index[0], edge_index[1]] = edge_attr.astype(np.float64)
    return adj


# ---------------------------------------------------------------------------
# Training / evaluation epoch helpers
# ---------------------------------------------------------------------------

def _train_epoch(
    model: ClusterSTGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        x          = batch["x"].to(device)          # (B, seq_len, K)
        y          = batch["y"].to(device)          # (B, horizon, K)
        edge_index = batch["edge_index"][0].to(device)  # (2, E)
        edge_attr  = batch["edge_attr"][0].to(device)   # (E,)

        optimizer.zero_grad()
        pred  = model(x, edge_index, edge_attr)     # (B, horizon, K)
        loss  = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(x)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def _eval_epoch(
    model: ClusterSTGNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Returns (loss, mae, rmse)."""
    model.eval()
    total_loss = 0.0
    all_pred: List[torch.Tensor] = []
    all_true: List[torch.Tensor] = []

    for batch in loader:
        x          = batch["x"].to(device)
        y          = batch["y"].to(device)
        edge_index = batch["edge_index"][0].to(device)
        edge_attr  = batch["edge_attr"][0].to(device)

        pred = model(x, edge_index, edge_attr)
        total_loss += criterion(pred, y).item() * len(x)
        all_pred.append(pred.cpu())
        all_true.append(y.cpu())

    preds = torch.cat(all_pred, dim=0)   # (N_samples, horizon, K)
    trues = torch.cat(all_true, dim=0)

    mae  = torch.mean(torch.abs(preds - trues)).item()
    rmse = torch.sqrt(torch.mean((preds - trues) ** 2)).item()

    return total_loss / len(loader.dataset), mae, rmse


# ---------------------------------------------------------------------------
# MLflow helper
# ---------------------------------------------------------------------------

def _try_mlflow(run_name: str, params: dict, experiment_name: str, tracking_uri: str) -> object:
    """Start an MLflow run if available; return None otherwise."""
    try:
        import mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_name)
        mlflow.log_params(params)
        return run
    except Exception:
        return None


def _mlflow_log(run, key: str, value: float, step: int) -> None:
    try:
        import mlflow
        if run is not None:
            mlflow.log_metric(key, value, step=step)
    except Exception:
        pass


def _mlflow_end(run) -> None:
    try:
        import mlflow
        if run is not None:
            mlflow.end_run()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str = "config.yaml") -> None:
    # ---- Load config ----
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = get_device()
    print(f"[train] device={device}")

    paths        = cfg["paths"]
    service       = cfg.get("data", {}).get("service", "total")
    processed_dir = Path(paths["processed_dir"])
    graphs_dir    = Path(paths["graphs_dir"])
    output_dir    = Path(paths["output_dir"]) / service
    ckpt_dir      = Path(paths["checkpoints_dir"]) / service
    results_dir   = Path(paths["results_dir"]) / service
    # ---- Separate sweep results path (hardcoded to 'total' for DVC) ----
    sweep_results_dir = Path(paths["results_dir"]) / "total"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    sweep_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] service={service}")

    # ---- Load graph topology ----
    edge_index_path  = graphs_dir / "edge_index.npy"
    edge_attr_path   = graphs_dir / "edge_attr.npy"
    node_index_path  = graphs_dir / "node_index.parquet"

    for p in [edge_index_path, edge_attr_path, node_index_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required graph file not found: {p}. "
                "Run `dvc repro build_graph` first."
            )

    edge_index_np  = np.load(edge_index_path)                # (2, E)
    edge_attr_np   = np.load(edge_attr_path)                  # (E,)
    node_index_df  = pd.read_parquet(node_index_path)
    node_index_df["site_id"] = node_index_df["site_id"].astype(str)
    N = len(node_index_df)
    print(f"[train] Graph: N={N} nodes, E={edge_index_np.shape[1]} edges")

    # ---- Load BS locations (lat/lon) for geographic affinity ----
    bs_locations_path = graphs_dir / "bs_locations.parquet"
    if not bs_locations_path.exists():
        raise FileNotFoundError(f"bs_locations.parquet not found at {bs_locations_path}")
    bs_locations_df = pd.read_parquet(bs_locations_path)
    bs_locations_df["site_id"] = bs_locations_df["site_id"].astype(str)
    # Align to node_index order so rows match bs_stats_df
    ordered_sites = node_index_df.sort_values("node_idx")["site_id"].tolist()
    bs_locations_df = (
        bs_locations_df.set_index("site_id")
        .reindex(ordered_sites)
        .reset_index()
    )

    # ---- Compute BS statistics (once — shared across all K values) ----
    print(f"[train] Computing per-BS statistics (service={service}) ...")
    bs_stats_df = compute_bs_stats(processed_dir, node_index_df, service=service)
    print(f"[train] bs_stats_df: {bs_stats_df.shape}")

    # ---- Build fine adj matrix (once — kept for coarsening edge weights) ----
    adj_dense = build_fine_adj(edge_index_np, edge_attr_np, N)

    # ---- K sweep ----
    sweep_k: List[int] = cfg["training"]["sweep_k"]
    clust_cfg = cfg["clustering"]
    train_cfg = cfg["training"]
    stgnn_cfg = cfg["stgnn"]

    sweep_results: List[Dict] = []

    for K in sweep_k:
        print(f"\n{'='*60}")
        print(f" K = {K}")
        print(f"{'='*60}")
        k_output_dir = output_dir / f"K{K}"
        k_ckpt_dir   = ckpt_dir   / f"K{K}"
        k_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ---- Step 4a: clustering ----
        method = clust_cfg.get("method", "skater")

        if method == "skater":
            clustering = SkaterGeographicClustering(
                n_clusters=K,
                floor=clust_cfg.get("skater", {}).get("floor", 3),
                random_state=clust_cfg["random_state"],
            )
            clustering.fit(
                bs_stats_df, bs_locations_df, edge_index_np,
                output_dir=k_output_dir,
            )
        else:
            clustering = AttributedSpectralClustering(
                n_clusters=K,
                w_graph=clust_cfg["affinity_weights"]["graph"],
                w_density=clust_cfg["affinity_weights"]["density"],
                w_shape=clust_cfg["affinity_weights"]["shape"],
                density_features=clust_cfg["density_features"],
                shape_features=clust_cfg["shape_features"],
                random_state=clust_cfg["random_state"],
            )
            clustering.fit(bs_stats_df, bs_locations_df, output_dir=k_output_dir)

        # Print cluster summary
        cluster_stats = clustering.get_cluster_stats(bs_stats_df)
        summary_cols = ["n_members", "mean_load", "peak_load"]
        if "dominant_shape" in cluster_stats.columns:
            summary_cols.append("dominant_shape")
        print("\nCluster summary:")
        print(cluster_stats[summary_cols].to_string())

        # ---- Step 4b: coarsening ----
        coarse_ei, coarse_ea, cluster_series, cluster_bs_map = build_coarsened_graph(
            cluster_labels=clustering.labels_,
            adj_matrix_dense=adj_dense,
            affinity_matrix=clustering.affinity_matrix_,
            daily_parquets_dir=processed_dir,
            node_index_df=node_index_df,
            output_dir=k_output_dir,
            service=service,
        )

        # Build the set of site_ids that actually received cluster assignments.
        # Some BSs in the parquets are not in node_index (outside graph coverage)
        # and are deliberately excluded from the model; the conservation check
        # must compare against the same subset.
        assigned_site_ids: set = {
            site for sites in cluster_bs_map.values() for site in sites
        }

        # ---- Step 4c: verify coarsening conservation ----
        try:
            verify_coarsening(cluster_series, processed_dir,
                              valid_site_ids=assigned_site_ids,
                              service=service)
        except AssertionError as e:
            print(f"[train] ABORT: coarsening verification failed for K={K}: {e}")
            continue

        # ---- Step 4d: datasets ----
        seq_len  = stgnn_cfg["seq_len"]
        horizon  = stgnn_cfg["horizon"]
        val_sp   = train_cfg["val_split"]
        test_sp  = train_cfg["test_split"]

        ds_kwargs = dict(
            cluster_series=cluster_series,
            edge_index=coarse_ei,
            edge_attr=coarse_ea,
            seq_len=seq_len,
            horizon=horizon,
            val_split=val_sp,
            test_split=test_sp,
        )
        train_ds = ClusterTrafficDataset(**ds_kwargs, split="train")
        val_ds   = ClusterTrafficDataset(**ds_kwargs, split="val")
        test_ds  = ClusterTrafficDataset(**ds_kwargs, split="test")

        print(
            f"[train] Datasets — train: {len(train_ds)}  "
            f"val: {len(val_ds)}  test: {len(test_ds)}"
        )

        batch_size   = train_cfg["batch_size"]
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        # ---- Step 4e: model ----
        model = ClusterSTGNN.from_config(cfg, n_clusters=K).to(device)
        n_params = model.count_parameters()
        print(f"[train] ClusterSTGNN K={K}: {n_params:,} parameters")

        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"], eta_min=1e-6
        )
        criterion = nn.L1Loss()   # MAE loss — directly minimises forecast error

        # ---- MLflow ----
        mlflow_cfg = cfg.get("mlflow", {})
        mlflow_run = _try_mlflow(
            run_name=f"cluster_first_K{K}",
            experiment_name=mlflow_cfg.get("experiment_name", "cluster_first_stgnn_forecasting"),
            tracking_uri=mlflow_cfg.get("tracking_uri", ""),
            params={
                "K":                K,
                "service":          service,
                "clustering_method": method,
                "hidden_dim":       stgnn_cfg["hidden_dim"],
                "gat_heads":        stgnn_cfg["gat_heads"],
                "gat_layers":       stgnn_cfg["gat_layers"],
                "gru_layers":       stgnn_cfg["gru_layers"],
                "dropout":          stgnn_cfg["dropout"],
                "seq_len":          seq_len,
                "horizon":          horizon,
                "lr":               train_cfg["lr"],
                "batch_size":       batch_size,
                "seed":             train_cfg["seed"],
            },
        )

        # ---- Training loop ----
        best_val_mae   = float("inf")
        best_epoch     = 0
        patience_count = 0
        history: List[Dict] = []
        t_start = time.time()

        epochs = train_cfg["epochs"]
        pbar   = tqdm(range(1, epochs + 1), desc=f"K={K} training", unit="epoch")

        for epoch in pbar:
            train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_mae, val_rmse = _eval_epoch(model, val_loader, criterion, device)
            scheduler.step()

            _mlflow_log(mlflow_run, "train_loss", train_loss, epoch)
            _mlflow_log(mlflow_run, "val_loss",   val_loss,   epoch)
            _mlflow_log(mlflow_run, "val_mae",    val_mae,    epoch)
            _mlflow_log(mlflow_run, "val_rmse",   val_rmse,   epoch)

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "val_mae":    val_mae,
                "val_rmse":   val_rmse,
            })

            pbar.set_postfix(
                train_loss=f"{train_loss:.5f}",
                val_mae=f"{val_mae:.5f}",
                best=f"{best_val_mae:.5f}",
            )

            # Early stopping + checkpoint
            if val_mae < best_val_mae:
                best_val_mae   = val_mae
                best_epoch     = epoch
                patience_count = 0
                torch.save(
                    {
                        "epoch":       epoch,
                        "val_mae":     val_mae,
                        "val_rmse":    val_rmse,
                        "model_state": model.state_dict(),
                        "K":           K,
                        "config":      cfg,
                    },
                    k_ckpt_dir / "best_model.pt",
                )
            else:
                patience_count += 1
                if patience_count >= train_cfg["early_stopping_patience"]:
                    print(
                        f"\n[train] Early stopping at epoch {epoch} "
                        f"(best val_mae={best_val_mae:.5f} at epoch {best_epoch})"
                    )
                    break

        training_time = time.time() - t_start
        _mlflow_end(mlflow_run)

        # Save training history
        with open(k_output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        sweep_results.append({
            "K":               K,
            "val_MAE":         best_val_mae,
            "val_RMSE":        min(h["val_rmse"] for h in history),
            "n_params":        n_params,
            "best_epoch":      best_epoch,
            "training_time_s": round(training_time, 1),
        })

        print(
            f"\n[train] K={K} done — best val_MAE={best_val_mae:.5f} "
            f"at epoch {best_epoch}, {training_time:.0f}s"
        )

    # ---- Sweep comparison table ----
    print(f"\n{'='*60}")
    print(" K-SWEEP RESULTS")
    print(f"{'='*60}")
    header = f"{'K':>5}  {'val_MAE':>10}  {'val_RMSE':>10}  {'n_params':>10}  {'best_epoch':>12}  {'time_s':>8}"
    print(header)
    print("-" * len(header))
    for r in sweep_results:
        print(
            f"{r['K']:>5}  {r['val_MAE']:>10.5f}  {r['val_RMSE']:>10.5f}  "
            f"{r['n_params']:>10,}  {r['best_epoch']:>12}  {r['training_time_s']:>8.1f}"
        )

    # ---- Save sweep results with safety fallback ----
    sweep_results_path = sweep_results_dir / "sweep_results.json"
    try:
        sweep_results_dir.mkdir(parents=True, exist_ok=True)
        with open(sweep_results_path, "w") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\n[train] Sweep results saved → {sweep_results_path}")
    except Exception as e:
        print(f"\n[train] WARNING: Failed to save sweep results to {sweep_results_path}: {e}")
        print(f"[train] Creating fallback empty sweep results...")
        try:
            sweep_results_dir.mkdir(parents=True, exist_ok=True)
            with open(sweep_results_path, "w") as f:
                json.dump([], f, indent=2)
            print(f"[train] Fallback sweep results created at {sweep_results_path}")
        except Exception as e2:
            print(f"[train] ERROR: Could not create fallback file: {e2}")
            raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cluster-first STGNN training")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    args = parser.parse_args()
    main(config_path=args.config)
