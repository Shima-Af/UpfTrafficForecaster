"""
evaluate.py — Forecast evaluation and UPF assignment analysis.

Loads a trained ClusterSTGNN checkpoint and evaluates it on the test split.
For each forecast step, converts cluster-level dl_norm to Gbps via the
calibration parameter α, assigns each cluster to a UPF configuration type,
and computes energy consumption relative to all-DPDK and all-USR baselines.

UPF assignment logic
---------------------
Given predicted cluster aggregate ŝ_k(t+h) and calibration parameter α:
    λ_k = ŝ_k × α  (Gbps)

    λ_k < 0.12 Gbps   → USR  (safe operating region)
    0.12 ≤ λ_k < 0.17 → DPDK (above USR/DPDK crossover)
    λ_k ≥ 0.17        → DANGER (USR saturation — must use DPDK, flag as overload)

Energy model
-------------
    E_USR(λ)  = energy_usr_slope × λ  (W)  — linear pre-saturation model
    E_DPDK    = energy_dpdk_w          (W)  — flat regardless of load

Run with:
    python -m cluster_first.evaluate --n_clusters 10 [--config cluster_first/config.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server environments

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from cluster_first.dataset import ClusterTrafficDataset
from cluster_first.model import ClusterSTGNN


# ---------------------------------------------------------------------------
# UPF assignment helpers
# ---------------------------------------------------------------------------

def assign_upf(
    lambda_gbps: np.ndarray,
    threshold_usr: float = 0.12,
    threshold_danger: float = 0.17,
) -> np.ndarray:
    """
    Assign each cluster load value to a UPF configuration type.

    The USR/DPDK crossover threshold (0.12 Gbps) is the primary decision
    boundary.  The danger threshold (0.17 Gbps) flags when USR would be
    in saturation regardless of controller choice — used for SLA analysis.

    Parameters
    ----------
    lambda_gbps      : arbitrary-shape float array of cluster loads
    threshold_usr    : above this → DPDK is preferred (default 0.12 Gbps)
    threshold_danger : above this → saturation risk (default 0.17 Gbps)

    Returns
    -------
    string array of same shape with values 'USR', 'DPDK', 'DANGER'
    """
    labels = np.where(lambda_gbps < threshold_usr, "USR", "DPDK")
    labels = np.where(lambda_gbps >= threshold_danger, "DANGER", labels)
    return labels


def compute_energy(
    lambda_gbps: np.ndarray,
    upf_types: np.ndarray,
    slope: float = 6.8,
    flat: float = 0.82,
) -> np.ndarray:
    """
    Compute per-cluster energy (W) for given load and UPF assignment.

    Parameters
    ----------
    lambda_gbps : cluster load (Gbps), same shape as upf_types
    upf_types   : string array 'USR' | 'DPDK' | 'DANGER'
    slope       : USR power model coefficient (W / Gbps)
    flat        : DPDK flat power draw (W)

    Returns
    -------
    float array of energy values (W), same shape as inputs
    """
    energy = np.where(
        upf_types == "USR",
        slope * lambda_gbps,
        np.full_like(lambda_gbps, flat),
    )
    # DANGER: cluster is forced to DPDK (saturation risk → conservative choice)
    energy = np.where(upf_types == "DANGER", flat, energy)
    return energy


# ---------------------------------------------------------------------------
# Baseline forecasters
# ---------------------------------------------------------------------------

def last_value_forecast(
    cluster_series: np.ndarray,
    seq_len: int,
    horizon: int,
    n_test: int,
) -> np.ndarray:
    """Naive last-value baseline: ŝ(t+h) = s(t) for all h."""
    T_total = cluster_series.shape[1]
    t_test_start = T_total - n_test
    preds = []
    for i in range(n_test - seq_len - horizon + 1):
        t = t_test_start + i + seq_len - 1   # last observed slot
        preds.append(cluster_series[:, t])    # (K,) broadcast over horizon
    preds = np.stack(preds, axis=0)           # (n_test_samples, K)
    # Expand to (n_test_samples, horizon, K)
    return np.stack([preds] * horizon, axis=1)


def historical_mean_forecast(
    cluster_series: np.ndarray,
    n_train: int,
    horizon: int,
    n_test_samples: int,
) -> np.ndarray:
    """Historical mean baseline: ŝ(t+h) = mean(s_train) for all h."""
    train_mean = cluster_series[:, :n_train].mean(axis=1)   # (K,)
    pred = np.stack([train_mean] * horizon, axis=0)          # (horizon, K)
    return np.stack([pred] * n_test_samples, axis=0)         # (n, horizon, K)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def forecast_metrics(
    preds: np.ndarray,
    trues: np.ndarray,
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, MAPE for (n_samples, horizon, K) arrays.

    Returns per-cluster and overall averaged metrics.
    """
    mae  = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))

    # MAPE: only where |true| > 1e-4 (avoid division by near-zero)
    mask = np.abs(trues) > 1e-4
    if mask.any():
        mape = float(np.mean(np.abs((preds[mask] - trues[mask]) / trues[mask])) * 100)
    else:
        mape = float("nan")

    # Per-cluster MAE
    per_cluster_mae = {
        f"cluster_{k}_mae": float(np.mean(np.abs(preds[:, :, k] - trues[:, :, k])))
        for k in range(preds.shape[2])
    }

    return {"mae": mae, "rmse": rmse, "mape": mape, **per_cluster_mae}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_forecast_vs_actual(
    preds: np.ndarray,
    trues: np.ndarray,
    K: int,
    figures_dir: Path,
    filename: str,
) -> None:
    """K subplots showing predicted vs actual cluster traffic over test period."""
    n_cols = min(K, 2)
    n_rows = (K + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), squeeze=False)

    # Use first horizon step for plotting (h=1 ahead)
    p = preds[:, 0, :]   # (n_samples, K)
    t = trues[:, 0, :]

    for k in range(K):
        ax = axes[k // n_cols][k % n_cols]
        ax.plot(t[:, k], label="Actual",    color="steelblue", linewidth=0.8, alpha=0.9)
        ax.plot(p[:, k], label="Predicted", color="tomato",    linewidth=0.8, alpha=0.8)
        mae_k = float(np.mean(np.abs(p[:, k] - t[:, k])))
        ax.set_title(f"Cluster {k}  (MAE={mae_k:.4f})", fontsize=9)
        ax.set_xlabel("Test sample")
        ax.set_ylabel("dl_norm sum")
        ax.legend(fontsize=7)

    # Hide unused subplots
    for k in range(K, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_visible(False)

    fig.suptitle(f"Predicted vs Actual — K={K}, horizon=1 step", fontsize=11)
    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_upf_assignment_map(
    cluster_assignments: pd.DataFrame,
    bs_locations: pd.DataFrame,
    upf_types_mean: np.ndarray,
    alpha: float,
    K: int,
    figures_dir: Path,
    filename: str,
) -> None:
    """
    Lyon BS map coloured by UPF assignment.

    Each BS inherits the UPF type of its cluster, determined by the mean
    predicted load over the test period.
    """
    color_map = {"USR": "forestgreen", "DPDK": "steelblue", "DANGER": "crimson"}

    # upf_types_mean: (K,) string — one type per cluster (based on mean test-period load)
    # Map cluster_id → UPF type → colour
    df = cluster_assignments.merge(bs_locations, on="site_id", how="left")
    df["upf_type"] = df["cluster_id"].map(
        {k: upf_types_mean[k] for k in range(K)}
    )
    df["color"] = df["upf_type"].map(color_map).fillna("grey")

    fig, ax = plt.subplots(figsize=(9, 8))
    for upf_type, color in color_map.items():
        subset = df[df["upf_type"] == upf_type]
        ax.scatter(subset["lon"], subset["lat"], c=color, s=8, alpha=0.7,
                   label=upf_type, edgecolors="none")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"UPF Assignment Map — K={K}, α={alpha:.3f} Gbps/unit")
    ax.legend(title="UPF type", markerscale=2)
    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_energy_comparison(
    methods: List[str],
    alpha_values: List[float],
    energy_per_method_alpha: Dict[str, Dict[float, float]],
    K: int,
    figures_dir: Path,
    filename: str,
) -> None:
    """Grouped bar chart comparing energy (W) across methods and α scenarios."""
    x   = np.arange(len(alpha_values))
    w   = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, method in enumerate(methods):
        vals = [energy_per_method_alpha[method].get(a, 0.0) for a in alpha_values]
        ax.bar(x + i * w - 0.4 + w / 2, vals, width=w, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels([f"α={a:.3f}" for a in alpha_values])
    ax.set_ylabel("Total UPF energy (W)")
    ax.set_title(f"Energy Comparison — K={K}")
    ax.legend()
    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(n_clusters: int, config_path: str = "cluster_first/config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    paths       = cfg["paths"]
    upf_cfg     = cfg["upf"]
    stgnn_cfg   = cfg["stgnn"]
    train_cfg   = cfg["training"]

    K = n_clusters
    graphs_dir   = Path(paths["graphs_dir"])
    output_dir   = Path(paths["output_dir"]) / f"K{K}"
    ckpt_path    = Path(paths["checkpoints_dir"]) / f"K{K}" / "best_model.pt"
    results_dir  = Path(paths["results_dir"])
    figures_dir  = Path(paths["figures_dir"])

    for p in [ckpt_path, output_dir / "cluster_series.npy",
              output_dir / "coarse_edge_index.npy",
              output_dir / "coarse_edge_attr.npy",
              output_dir / "cluster_assignments.parquet"]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}. Run cluster_first/train.py first."
            )

    device = torch.device("cpu")  # evaluation can always run on CPU

    # ---- Load artefacts ----
    cluster_series = np.load(output_dir / "cluster_series.npy")       # (K, T_total)
    coarse_ei      = torch.from_numpy(np.load(output_dir / "coarse_edge_index.npy")).long()
    coarse_ea      = torch.from_numpy(np.load(output_dir / "coarse_edge_attr.npy"))
    cluster_assignments = pd.read_parquet(output_dir / "cluster_assignments.parquet")
    bs_locations        = pd.read_parquet(graphs_dir / "bs_locations.parquet")

    # ---- Load model ----
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = ClusterSTGNN.from_config(cfg, n_clusters=K).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[evaluate] Loaded K={K} checkpoint (epoch {ckpt['epoch']}, "
          f"val_mae={ckpt['val_mae']:.5f})")

    # ---- Build test dataset ----
    seq_len = stgnn_cfg["seq_len"]
    horizon = stgnn_cfg["horizon"]
    test_ds = ClusterTrafficDataset(
        cluster_series=cluster_series,
        edge_index=coarse_ei,
        edge_attr=coarse_ea,
        seq_len=seq_len,
        horizon=horizon,
        split="test",
        val_split=train_cfg["val_split"],
        test_split=train_cfg["test_split"],
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    n_test_samples = len(test_ds)

    # ---- Inference ----
    all_preds: List[torch.Tensor] = []
    all_trues: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in test_loader:
            x          = batch["x"].to(device)
            y          = batch["y"].to(device)
            edge_index = batch["edge_index"][0].to(device)
            edge_attr  = batch["edge_attr"][0].to(device)
            pred = model(x, edge_index, edge_attr)
            all_preds.append(pred.cpu())
            all_trues.append(y.cpu())

    preds_np = torch.cat(all_preds, dim=0).numpy()   # (n, horizon, K)
    trues_np = torch.cat(all_trues, dim=0).numpy()   # (n, horizon, K)

    # ---- Forecast metrics ----
    model_metrics = forecast_metrics(preds_np, trues_np)
    print("\n[evaluate] Model metrics:")
    for k, v in model_metrics.items():
        if not k.startswith("cluster_"):
            print(f"  {k}: {v:.5f}")

    # ---- Baselines ----
    T_total   = cluster_series.shape[1]
    n_test_ts = max(horizon + seq_len, int(T_total * train_cfg["test_split"]))
    n_val_ts  = max(horizon + seq_len, int(T_total * train_cfg["val_split"]))
    n_train_ts = T_total - n_val_ts - n_test_ts

    lv_preds   = last_value_forecast(cluster_series, seq_len, horizon, n_test_ts)
    # lv_preds may differ in length — align to n_test_samples
    lv_preds   = lv_preds[:n_test_samples]
    hm_preds   = historical_mean_forecast(cluster_series, n_train_ts, horizon, n_test_samples)

    lv_metrics = forecast_metrics(lv_preds, trues_np)
    hm_metrics = forecast_metrics(hm_preds, trues_np)

    print("\n[evaluate] Last-value baseline:")
    print(f"  MAE={lv_metrics['mae']:.5f}  RMSE={lv_metrics['rmse']:.5f}")
    print("[evaluate] Historical-mean baseline:")
    print(f"  MAE={hm_metrics['mae']:.5f}  RMSE={hm_metrics['rmse']:.5f}")

    # ---- UPF assignment for each α scenario ----
    alpha_scenarios: Dict[str, float] = upf_cfg["alpha_scenarios"]
    thresh_usr    = upf_cfg["threshold_dpdk_gbps"]    # 0.12 = USR/DPDK crossover
    thresh_danger = upf_cfg["threshold_danger_gbps"]  # 0.17 = saturation onset
    dpdk_w        = upf_cfg["energy_dpdk_w"]
    usr_slope     = upf_cfg["energy_usr_slope"]

    all_results: Dict = {
        "K":           K,
        "model":       model_metrics,
        "last_value":  lv_metrics,
        "hist_mean":   hm_metrics,
        "upf_analysis": {},
    }

    energy_per_method_alpha: Dict[str, Dict[float, float]] = {
        "Model": {}, "Oracle": {}, "All-DPDK": {}, "All-USR": {}, "Last-Value": {}
    }

    for scenario_name, alpha in alpha_scenarios.items():
        # Convert dl_norm forecast → Gbps
        lambda_pred   = preds_np * alpha         # (n, horizon, K)
        lambda_true   = trues_np * alpha
        lambda_lv     = lv_preds * alpha

        # Assignments
        types_model  = assign_upf(lambda_pred,  thresh_usr, thresh_danger)
        types_oracle = assign_upf(lambda_true,  thresh_usr, thresh_danger)
        types_lv     = assign_upf(lambda_lv,    thresh_usr, thresh_danger)

        # Assignment accuracy
        acc = float(np.mean(types_model == types_oracle))

        # SLA violation rate (actual load exceeds danger threshold)
        sla_violations = float(np.mean(lambda_true >= thresh_danger))

        # Energy
        e_model  = compute_energy(lambda_pred,  types_model,  usr_slope, dpdk_w)
        e_oracle = compute_energy(lambda_true,  types_oracle, usr_slope, dpdk_w)
        e_alldpdk = np.full_like(lambda_true, dpdk_w)
        e_allusr  = usr_slope * lambda_true
        e_lv     = compute_energy(lambda_lv,    types_lv,     usr_slope, dpdk_w)

        mean_e_model   = float(e_model.sum(axis=-1).mean())   # sum over K, mean over (n, h)
        mean_e_oracle  = float(e_oracle.sum(axis=-1).mean())
        mean_e_alldpdk = float(e_alldpdk.sum(axis=-1).mean())
        mean_e_allusr  = float(e_allusr.sum(axis=-1).mean())
        mean_e_lv      = float(e_lv.sum(axis=-1).mean())

        energy_per_method_alpha["Model"][alpha]      = mean_e_model
        energy_per_method_alpha["Oracle"][alpha]     = mean_e_oracle
        energy_per_method_alpha["All-DPDK"][alpha]   = mean_e_alldpdk
        energy_per_method_alpha["All-USR"][alpha]    = mean_e_allusr
        energy_per_method_alpha["Last-Value"][alpha] = mean_e_lv

        savings_vs_dpdk = (mean_e_alldpdk - mean_e_model) / mean_e_alldpdk * 100

        print(f"\n[evaluate] α={alpha} ({scenario_name}):")
        print(f"  Assignment accuracy: {acc:.3f}")
        print(f"  SLA violation rate:  {sla_violations:.4f}")
        print(f"  Energy — Model: {mean_e_model:.3f}W  Oracle: {mean_e_oracle:.3f}W  "
              f"All-DPDK: {mean_e_alldpdk:.3f}W  All-USR: {mean_e_allusr:.3f}W")
        print(f"  Energy savings vs all-DPDK: {savings_vs_dpdk:.1f}%")

        all_results["upf_analysis"][scenario_name] = {
            "alpha":                alpha,
            "assignment_accuracy":  acc,
            "sla_violation_rate":   sla_violations,
            "energy_model_w":       mean_e_model,
            "energy_oracle_w":      mean_e_oracle,
            "energy_alldpdk_w":     mean_e_alldpdk,
            "energy_allusr_w":      mean_e_allusr,
            "energy_lastvalue_w":   mean_e_lv,
            "savings_vs_dpdk_pct":  savings_vs_dpdk,
        }

        # ---- Figures ----
        # UPF assignment map (mean predicted load over test period)
        mean_lambda_pred = preds_np.mean(axis=(0, 1)) * alpha   # (K,) mean over (n, horizon)
        upf_types_mean   = assign_upf(mean_lambda_pred, thresh_usr, thresh_danger)

        try:
            plot_upf_assignment_map(
                cluster_assignments=cluster_assignments,
                bs_locations=bs_locations,
                upf_types_mean=upf_types_mean,
                alpha=alpha,
                K=K,
                figures_dir=figures_dir,
                filename=f"upf_assignment_map_K{K}_alpha{alpha:.3f}.png",
            )
        except Exception as exc:
            print(f"[evaluate] Map plot skipped ({exc})")

    # Forecast vs actual plot
    try:
        plot_forecast_vs_actual(
            preds=preds_np, trues=trues_np, K=K,
            figures_dir=figures_dir,
            filename=f"forecast_vs_actual_K{K}.png",
        )
    except Exception as exc:
        print(f"[evaluate] Forecast plot skipped ({exc})")

    # Energy comparison plot
    try:
        plot_energy_comparison(
            methods=["Model", "Oracle", "All-DPDK", "All-USR"],
            alpha_values=list(alpha_scenarios.values()),
            energy_per_method_alpha=energy_per_method_alpha,
            K=K,
            figures_dir=figures_dir,
            filename=f"energy_comparison_K{K}.png",
        )
    except Exception as exc:
        print(f"[evaluate] Energy plot skipped ({exc})")

    # ---- Save results JSON ----
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"K{K}_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[evaluate] Results saved → {results_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cluster-first STGNN evaluation")
    parser.add_argument(
        "--n_clusters", type=int, required=True,
        help="K — number of clusters (must match a trained checkpoint)"
    )
    parser.add_argument(
        "--config", default="cluster_first/config.yaml",
        help="Path to config YAML"
    )
    args = parser.parse_args()
    evaluate(n_clusters=args.n_clusters, config_path=args.config)
