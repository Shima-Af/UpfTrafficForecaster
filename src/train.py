"""
train.py
--------
DVC pipeline stage: train traffic forecasting model with optional MLflow tracking.

CLI usage (called by dvc.yaml):
    python -m src.train

Supported models (set params.yaml → training.model):
    lstm  : per-site LSTM with site embedding  (MultiSiteDataset)
    stgnn : graph attention + GRU on full gNodeB graph (GraphTrafficDataset)

Steps
-----
1. Load processed parquets + aggregate to BS level via Voronoi map
2. Engineer features, split train/val/test by date
3. Build model and DataLoaders (model-specific)
4. Train with early stopping + cosine LR scheduler
5. Evaluate on test set, log metrics and artefacts

Reads  : data/netmob/processed/, data/graphs/voronoi_map.parquet,
         data/graphs/edge_index.npy, data/graphs/node_index.parquet  (stgnn only)
Writes : models/best_model.pt, models/scaler.pkl, results/metrics_train.json
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

try:
    import mlflow
    import mlflow.pytorch
    _MLFLOW = True
except ImportError:
    _MLFLOW = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# LSTM training loop  (batch = (X, y, site_idx))
# ---------------------------------------------------------------------------

def _train_epoch_lstm(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for X, y, site in loader:
        X, y, site = X.to(device), y.to(device), site.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X, site_idx=site), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)


@torch.no_grad()
def _eval_epoch_lstm(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for X, y, site in loader:
        X, y, site = X.to(device), y.to(device), site.to(device)
        total += criterion(model(X, site_idx=site), y).item() * len(X)
    return total / len(loader.dataset)


# ---------------------------------------------------------------------------
# STGNN training loop  (batch = (X, y), edge_index fixed on device)
# ---------------------------------------------------------------------------

def _train_epoch_stgnn(model, loader, optimizer, criterion, edge_index, device):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X, edge_index)    # (B, N, horizon)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)


@torch.no_grad()
def _eval_epoch_stgnn(model, loader, criterion, edge_index, device):
    model.eval()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        total += criterion(model(X, edge_index), y).item() * len(X)
    return total / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(processed_dir: Path, models_dir: Path, results_dir: Path, params: dict) -> None:
    tp     = params["training"]
    model_type = tp["model"]           # "lstm" or "stgnn"
    seed   = tp["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device(tp["device"])
    print(f"[train] Model={model_type}  Device={device}")

    # ---- Voronoi map (produced by build_graph DVC stage) ----
    voronoi_path = Path("data/graphs/voronoi_map.parquet")
    if not voronoi_path.exists():
        raise FileNotFoundError(f"{voronoi_path} not found — run 'dvc repro build_graph' first.")
    voronoi_map = pd.read_parquet(voronoi_path).squeeze()
    voronoi_map.index.name = None

    # ================================================================
    # Build datasets — branch on model type
    # ================================================================
    if model_type == "stgnn":
        # ---- Load node index and edge graph ----
        node_index_path = Path("data/graphs/node_index.parquet")
        edge_index_path = Path("data/graphs/edge_index.npy")
        if not node_index_path.exists() or not edge_index_path.exists():
            raise FileNotFoundError(
                "Graph files not found — run 'dvc repro build_graph' first."
            )
        node_index_df = pd.read_parquet(node_index_path)
        edge_index_np = np.load(edge_index_path)                    # (2, E)
        edge_index    = torch.from_numpy(edge_index_np).to(device)  # long tensor

        from src.dataset import build_graph_datasets, make_loaders
        print("[train] Building graph datasets (this may take ~1 min)...")
        train_ds, val_ds, test_ds, scaler, feature_cols, n_nodes = build_graph_datasets(
            processed_dir, params, voronoi_map, node_index_df
        )
        n_features = len(feature_cols)
        train_loader, val_loader, test_loader = make_loaders(
            train_ds, val_ds, test_ds, batch_size=tp["batch_size"]
        )

        from src.models.stgnn import TrafficSTGNN
        model = TrafficSTGNN.from_params(params, n_nodes=n_nodes, n_features=n_features)
        model.to(device)
        print(f"[train] STGNN  N={n_nodes}  F={n_features}  "
              f"Params={model.count_parameters():,}")

        def _train_ep(m, ldr, opt, crit, dev):
            return _train_epoch_stgnn(m, ldr, opt, crit, edge_index, dev)

        def _eval_ep(m, ldr, crit, dev):
            return _eval_epoch_stgnn(m, ldr, crit, edge_index, dev)

        extra_ckpt = {"n_nodes": n_nodes, "n_features": n_features}

    else:   # lstm (default)
        from src.dataset import build_datasets, make_loaders
        print("[train] Building per-site datasets...")
        train_ds, val_ds, test_ds, scaler, feature_cols = build_datasets(
            processed_dir, params, voronoi_map
        )
        n_features = len(feature_cols)
        n_sites    = train_ds.site.max().item() + 1
        print(f"[train] LSTM  F={n_features}  Sites={n_sites}  "
              f"Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")
        train_loader, val_loader, test_loader = make_loaders(
            train_ds, val_ds, test_ds, batch_size=tp["batch_size"]
        )

        from src.models.lstm import TrafficLSTM
        model = TrafficLSTM.from_params(params, n_features=n_features, n_sites=n_sites)
        model.to(device)
        print(f"[train] LSTM  Params={model.count_parameters():,}")

        def _train_ep(m, ldr, opt, crit, dev):
            return _train_epoch_lstm(m, ldr, opt, crit, dev)

        def _eval_ep(m, ldr, crit, dev):
            return _eval_epoch_lstm(m, ldr, crit, dev)

        extra_ckpt = {"n_features": n_features, "n_sites": n_sites}

    # ================================================================
    # Shared training loop
    # ================================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=tp["lr"])
    criterion = nn.MSELoss()

    if tp["lr_scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tp["epochs"]
        )
    else:
        scheduler = None

    if _MLFLOW:
        mlflow.set_experiment("upf_traffic_forecaster")
        mlflow.start_run()
        mlflow.log_params({**params.get(model_type, {}), **tp})

    best_val   = float("inf")
    patience   = tp["early_stopping_patience"]
    no_improve = 0
    best_path  = models_dir / "best_model.pt"

    for epoch in range(1, tp["epochs"] + 1):
        train_loss = _train_ep(model, train_loader, optimizer, criterion, device)
        val_loss   = _eval_ep(model, val_loader,   criterion, device)

        if scheduler:
            scheduler.step()

        if _MLFLOW:
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save({
                "epoch":        epoch,
                "model_type":   model_type,
                "model_state":  model.state_dict(),
                "val_loss":     val_loss,
                "params":       params,
                "feature_cols": feature_cols,
                **extra_ckpt,
            }, best_path)
        else:
            no_improve += 1

        if epoch % 10 == 0 or no_improve == 0:
            marker = "  ← best" if no_improve == 0 else ""
            print(f"  Epoch {epoch:4d}  train={train_loss:.6f}  val={val_loss:.6f}{marker}")

        if no_improve >= patience:
            print(f"[train] Early stopping at epoch {epoch}")
            break

    # ---- Save scaler ----
    with open(models_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ---- Test evaluation ----
    from src.evaluate import evaluate_loader
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    if model_type == "stgnn":
        test_metrics = evaluate_loader(model, test_loader, device, edge_index=edge_index)
    else:
        test_metrics = evaluate_loader(model, test_loader, device)

    print(f"[train] Test metrics: {test_metrics}")

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "metrics_train.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    if _MLFLOW:
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.pytorch.log_model(model, "model")
        mlflow.end_run()

    print(f"[train] Done.  Best val_loss={best_val:.6f}  → {best_path}")


if __name__ == "__main__":
    with open("params.yaml", encoding="utf-8") as f:
        _params = yaml.safe_load(f)
    _models_dir = Path("models")
    _models_dir.mkdir(exist_ok=True)
    run(Path("data/netmob/processed"), _models_dir, Path("results"), _params)
