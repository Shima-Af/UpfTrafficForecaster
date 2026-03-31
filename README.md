# UPF Traffic Forecaster

Traffic forecasting module for a 5G UPF digital twin framework.
Predicts future per-gNodeB downlink load from NetMob 2023 mobile traffic traces,
using LSTM and (upcoming) spatiotemporal GNN models.

Part of a broader research framework:

```
Physical Testbed
      ↓
UPF Profiling Campaign          Traffic Forecaster   ← this repo
(power & QoS models)            (LSTM / STGNN)
      ↓                                ↓
                  Digital Twin
            (simulation + control)
                      ↓
              UPF Controllers
         (PPO / Rule-based / DP)
```

## Dataset

**NetMob 2023** — privacy-preserving aggregate mobile traffic data.
- City: Lyon, France
- Services: Netflix, YouTube, DailyMotion
- Period: 16 March – 31 May 2019
- Resolution: 15-minute bins, 100×100 m grid tiles
- Direction: Downlink only

Raw data is versioned with DVC (remote: AWS S3).

## Pipeline

```
raw NetMob files
      ↓  dvc repro preprocess
data/netmob/processed/       (normalised per-day parquets)
      ↓  dvc repro build_graph
data/graphs/voronoi_map.parquet   (tile → gNodeB assignment)
      ↓  dvc repro train
models/best_model.pt         (trained LSTM checkpoint)
      ↓  dvc repro evaluate
results/metrics_test.json    (MAE, RMSE, MAPE, SLA compliance)
```

## Quickstart

```bash
# Clone and restore data
git clone https://github.com/Shima-Af/UpfTrafficForecaster.git
cd UpfTrafficForecaster
pip install -r requirements.txt
dvc pull

# Run full pipeline
dvc repro
```

## Project Structure

```
src/
  netmob_loader.py   — parse raw NetMob files
  preprocess.py      — DVC stage: raw → normalised parquet
  aggregate.py       — Voronoi tessellation: tiles → gNodeB time series
  features.py        — lag + cyclic time feature engineering
  dataset.py         — PyTorch Dataset + temporal train/val/test split
  models/
    lstm.py          — multi-layer LSTM with site embeddings
    stgnn.py         — spatiotemporal GNN (upcoming)
  train.py           — MLflow training loop with early stopping
  evaluate.py        — MAE / RMSE / MAPE / SLA compliance metrics
  forecast.py        — inference API
data/
  netmob/raw/        — raw NetMob files (DVC-tracked)
  netmob/processed/  — normalised parquets (DVC-tracked)
  graphs/            — Voronoi map + BS locations (DVC-tracked)
models/              — trained checkpoints (DVC-tracked)
results/             — evaluation metrics and forecast plots
```

## Key Design Choices

- **No physical unit assumption** — NetMob values are dimensionless privacy-preserving
  aggregates. Normalisation is data-driven (global max across the dataset).
- **Voronoi aggregation** — 54,013 NetMob tiles are assigned to their nearest gNodeB
  using base station locations from Cartoradio (ANFR). Reduces the problem to
  ~700 spatially meaningful nodes.
- **Temporal split** — train/val/test split is strictly chronological to prevent
  data leakage.

## Related Repos

- [UpfProfilingCampaign](https://github.com/Shima-Af/UpfProfilingCampaign) — Phase 1:
  lab measurement campaign and UPF power/QoS model training.
