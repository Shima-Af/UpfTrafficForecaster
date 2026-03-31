"""
train.py
--------
DVC pipeline stage: train traffic forecasting model with MLflow tracking.

CLI usage (called by dvc.yaml):
    python -m src.train

Steps
-----
1. Load processed parquets + aggregate to BS level via Voronoi map
2. Engineer features (lags, time encodings)
3. Split train / val / test by date
4. Train model (LSTM or STGNN) with early stopping
5. Log metrics and artefacts to MLflow
6. Save best checkpoint to models/

Reads  : data/netmob/processed/, data/graphs/voronoi_map.parquet
Writes : models/best_model.pt, models/scaler.pkl, results/metrics_train.json
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# Optional MLflow — skip gracefully if not installed
try:
    import mlflow
    import mlflow.pytorch
    _MLFLOW = True
except ImportError:
    _MLFLOW = False


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        X, y, site = batch
        X, y, site = X.to(device), y.to(device), site.to(device)
        optimizer.zero_grad()
        pred = model(X, site_idx=site)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        X, y, site = batch
        X, y, site = X.to(device), y.to(device), site.to(device)
        pred = model(X, site_idx=site)
        total_loss += criterion(pred, y).item() * len(X)
    return total_loss / len(loader.dataset)


def run(processed_dir: Path, models_dir: Path, results_dir: Path, params: dict) -> None:
    from src.aggregate import build_voronoi_map
    from src.dataset import build_datasets, make_loaders
    from src.models.lstm import TrafficLSTM

    tp    = params["training"]
    lp    = params["lstm"]
    seed  = tp["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device(tp["device"])
    print(f"[train] Device: {device}")

    # ---- Build Voronoi map ----
    voronoi_map_path = Path("data/graphs/voronoi_map.parquet")
    if voronoi_map_path.exists():
        import pandas as pd
        voronoi_map = pd.read_parquet(voronoi_map_path).squeeze()
        voronoi_map.index.name = None
    else:
        print("[train] Building Voronoi map (first run)...")
        import numpy as np
        import pandas as pd
        sample_pq = next(sorted(processed_dir.glob("*.parquet")))
        tile_ids = pd.read_parquet(sample_pq)["cell_id"].unique()
        voronoi_map = build_voronoi_map(tile_ids.astype(np.int64), params)
        voronoi_map_path.parent.mkdir(parents=True, exist_ok=True)
        voronoi_map.to_frame().to_parquet(voronoi_map_path)

    # ---- Build datasets ----
    print("[train] Building datasets...")
    train_ds, val_ds, test_ds, scaler, feature_cols = build_datasets(
        processed_dir, params, voronoi_map
    )
    n_features = len(feature_cols)
    n_sites    = train_ds.site.max().item() + 1
    print(f"[train] Features={n_features}  Sites={n_sites}  "
          f"Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")

    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, batch_size=tp["batch_size"]
    )

    # ---- Build model ----
    model = TrafficLSTM.from_params(params, n_features=n_features, n_sites=n_sites)
    model.to(device)
    print(f"[train] Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=tp["lr"])
    criterion = nn.MSELoss()

    if tp["lr_scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tp["epochs"]
        )
    else:
        scheduler = None

    # ---- MLflow run ----
    if _MLFLOW:
        mlflow.set_experiment("upf_traffic_forecaster")
        run_ctx = mlflow.start_run()
        mlflow.log_params({**lp, **tp})
    else:
        run_ctx = None

    # ---- Training loop ----
    best_val  = float("inf")
    patience  = tp["early_stopping_patience"]
    no_improve = 0
    best_path = models_dir / "best_model.pt"

    for epoch in range(1, tp["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = eval_epoch(model, val_loader,   criterion, device)

        if scheduler:
            scheduler.step()

        if _MLFLOW:
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        if val_loss < best_val:
            best_val    = val_loss
            no_improve  = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "n_features":  n_features,
                "n_sites":     n_sites,
                "params":      params,
                "feature_cols": feature_cols,
            }, best_path)
        else:
            no_improve += 1

        if epoch % 10 == 0 or no_improve == 0:
            print(f"  Epoch {epoch:4d}  train={train_loss:.6f}  val={val_loss:.6f}"
                  + ("  ← best" if no_improve == 0 else ""))

        if no_improve >= patience:
            print(f"[train] Early stopping at epoch {epoch}")
            break

    # ---- Save scaler ----
    scaler_path = models_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # ---- Evaluate on test set ----
    from src.evaluate import evaluate_loader
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate_loader(model, test_loader, device)
    print(f"[train] Test metrics: {test_metrics}")

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "metrics_train.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    if _MLFLOW:
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.pytorch.log_model(model, "model")
        if run_ctx:
            mlflow.end_run()

    print(f"[train] Done. Best val_loss={best_val:.6f}  → {best_path}")


if __name__ == "__main__":
    with open("params.yaml", encoding="utf-8") as f:
        _params = yaml.safe_load(f)
    _models_dir = Path("models")
    _models_dir.mkdir(exist_ok=True)
    run(Path("data/netmob/processed"), _models_dir, Path("results"), _params)
