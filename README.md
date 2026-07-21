# UPF Traffic Forecaster

Traffic forecasting module for a 5G UPF digital twin framework.
Predicts near-future downlink load for **K cluster-level service areas**
(prospective MEC / UPF sites) in Lyon, from NetMob 2023 mobile traffic traces,
using a **cluster-first Spatio-Temporal GNN (GAT + GRU)**.

> `PIPELINE.md` is the authoritative, code-verified description of the pipeline.
> This README summarises it and documents the downstream export contract.
> Where the two disagree, `PIPELINE.md` wins.

Part of a broader research framework:

```
Physical Testbed
      ↓
UPF Profiling Campaign          Traffic Forecaster   ← this repo
(power & QoS models)            (cluster-first STGNN)
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
- Services: Netflix, YouTube, DailyMotion (current model: **Netflix**)
- Period: 16 March – 31 May 2019
- Resolution: 15-minute bins, 96 slots/day, 100×100 m grid tiles
- Direction: Downlink only

Raw values are **already dimensionless** — privacy-preserving aggregate
indicators with no physical unit. They are *not* bit/s. Raw data is versioned
with DVC (remote: AWS S3).

## Model

Forecasts are produced in **normalised `[0,1]` units** (`dl_norm`). Mapping to
physical capacity (Gbps) is a *downstream scenario concern* — the controller's
`alpha` — applied **after** forecasting, never before.

| | |
|---|---|
| Spatial | GAT, 4 heads, 2 layers |
| Temporal | GRU, 1 layer |
| Hidden dim | 64 |
| Lookback | 24 h (`seq_len: 96` × 15 min) |
| Horizon | 1 h ahead (`horizon: 4` × 15 min) |
| Split | strictly chronological (no leakage), ~50 / ~10.5 / ~10.5 days |
| Metrics | MAE, RMSE, **WAPE** (not MAPE — traffic is sparse/near-zero) |
| Baselines | last-value (persistence), historical-mean, seasonal-naive |

The STGNN beats all baselines at every K. **K=10 is a deployment choice** (number
of MEC/UPF controllers, ~2.7 km cluster footprint), *not* the accuracy-optimal K —
the evaluator's automatic `best_k` is 5 by WAPE. See `PIPELINE.md` for the
per-K results table.

## Pipeline

```
data/netmob/raw/                     raw NetMob files (DVC-tracked)
      ↓  dvc repro build_graph       Voronoi tessellation, ~965 gNodeBs
data/graphs/                         voronoi_map, bs_locations, edge_index, edge_attr
      ↓  dvc repro preprocess        dl_norm = bs_tile_sum / global_max
data/netmob/processed/               normalised per-day parquets
      ↓  dvc repro train             SKATER clustering (K) + coarse graph + STGNN
data/cluster_first/<svc>/K<k>/       cluster_series, assignments, bs_map, coarse graph
checkpoints/cluster_first/           trained STGNN checkpoints
      ↓  dvc repro evaluate_forecast MAE / RMSE / WAPE vs 3 baselines
results/cluster_first/<svc>/K<k>/    predictions_{train,val,test}, targets_{train,val,test}
      ↓  (frozen) export_for_twin    flat bundle for downstream consumers
exports/traffic_forecaster/
```

## Export contract (downstream handoff)

`UpfRLControllers` and `UPF_NDT` consume a **flat** artifact set from
`exports/traffic_forecaster/`, produced by the `export_for_twin` stage
(`scripts/export_for_twin.py`) and tracked as a **cached DVC output** — so the
whole bundle is fetchable from the S3 remote and importable at a pinned rev.

The bundle is 11 files + `EXPORT_MANIFEST.json` (which records an md5 per file):

| File | Source |
|---|---|
| `predictions_{train,val,test}.npy` | `results/cluster_first/<svc>/K<k>/` |
| `targets_{train,val,test}.npy` | `results/cluster_first/<svc>/K<k>/` |
| `cluster_series.npy` | `data/cluster_first/<svc>/K<k>/` |
| `cluster_assignments.parquet` | `data/cluster_first/<svc>/K<k>/` |
| `cluster_bs_map.json` | `data/cluster_first/<svc>/K<k>/` |
| `bs_locations.parquet` | `data/graphs/` |
| `forecast_eval_summary.json` | `results/cluster_first/total/` |

Note on `forecast_eval_summary.json`: as a metric of `evaluate_forecast` it is
declared `cache: false`, so *that* copy lives in no DVC remote. The copy inside
`exports/` **is** cached — that is what makes the summary fetchable downstream.

### Consuming it

Import at the pinned release tag rather than copying files by hand:

```bash
dvc import --rev thesis-v1 \
  https://github.com/Shima-Af/UpfTrafficForecaster.git \
  exports/traffic_forecaster/predictions_test.npy \
  -o data/external/traffic_forecaster/predictions_test.npy
```

Verify what you got against the `md5` block in `EXPORT_MANIFEST.json`.

### ⚠ The bundle is frozen on purpose

The `export_for_twin` stage is marked `frozen: true` in `dvc.yaml`, so
`dvc repro` will **not** re-run it. This is deliberate, not an oversight:

- `checkpoints/cluster_first/Netflix/K10/best_model.pt` was **retrained
  2026-06-11**. The `predictions_*.npy` now produced by `evaluate_forecast`
  therefore **differ** from the arrays in `exports/`.
- The arrays in `exports/` are the outputs of the *pre-retrain* checkpoint —
  the ones behind every downstream Phase 2–9 result and the MASCOTS paper.
  `predictions_{train,val}.npy` were recovered from the DVC cache at commit
  `ff90c8a`. They are **not reproducible** from the current working tree.
- The live `results/cluster_first/total/forecast_eval_summary.json` has since
  been overwritten by a **DailyMotion** sweep; the Netflix summary survives only
  inside `exports/`.

Unfreezing and re-running the stage would silently replace the thesis bundle
with current-model bytes and invalidate downstream results. Read the
`provenance` field of `EXPORT_MANIFEST.json` before touching it. To publish a
*new* model, export to a different directory and cut a new tag — do not
overwrite this one.

## Quickstart

```bash
git clone https://github.com/Shima-Af/UpfTrafficForecaster.git
cd UpfTrafficForecaster
pip install -r requirements.txt
dvc pull

dvc repro          # export_for_twin is frozen and will be skipped
```

## Project Structure

```
src/
  netmob_loader.py     — parse raw NetMob files
  preprocess.py        — DVC stage: raw → normalised parquet
  aggregate.py         — Voronoi tessellation: tiles → gNodeB time series
  cluster.py           — SKATER geographic clustering (min cluster size 3)
  coarsen.py           — cluster-level coarse graph + conservation check
  cluster_dataset.py   — PyTorch Dataset over cluster series
  dataset.py           — temporal train/val/test split
  features.py          — lag + cyclic time feature engineering
  model.py             — cluster-first STGNN (GAT + GRU)
  train.py             — MLflow training loop with early stopping
  evaluate_forecast.py — MAE / RMSE / WAPE + baseline comparison
scripts/
  export_for_twin.py   — stage the flat downstream bundle
  run_sweep.py         — multi-seed / multi-K sweeps
  compare_models.py    — STGNN vs SOTA baselines
  ablation_granularity.py, benchmark_training_speed.py
  plot_forecaster_figures.py
data/
  netmob/raw/          — raw NetMob files (DVC-tracked)
  netmob/processed/    — normalised parquets (DVC-tracked)
  graphs/              — Voronoi map + BS locations (DVC-tracked)
  cluster_first/       — per-(service,K) cluster artifacts (DVC-tracked)
checkpoints/           — trained STGNN checkpoints (DVC-tracked)
results/               — metrics + forecast arrays (DVC-tracked)
exports/               — frozen downstream bundle (DVC-tracked)
```

## Key Design Choices

- **No physical unit assumption** — NetMob values are dimensionless
  privacy-preserving aggregates. Normalisation is data-driven:
  `dl_norm = bs_tile_sum / global_max`, one global max across **all** services
  (`global_max = 148178608.0`, `data/netmob/processed/metadata.json`). There is
  no byte conversion.
- **Voronoi aggregation** — NetMob tiles are assigned to their nearest gNodeB
  using Cartoradio (ANFR) base station locations, giving ~965 spatially
  meaningful nodes.
- **Cluster-first** — ~965 BSs is too many for the model and the controllers, so
  BSs are clustered with SKATER *before* forecasting; each cluster is one graph
  node. Cluster signal = sum of member `dl_norm` (valid as a direct sum because
  all services share one normaliser).
- **`alpha` applied downstream** — forecast in `[0,1]`, scale at the control
  boundary. Pre-scaling carries zero predictive information, de-tunes training
  (MSE scales with α²), and destroys scenario reusability.
- **Temporal split** — strictly chronological, to prevent leakage.

## Related Repos

- [UpfProfilingCampaign](https://github.com/Shima-Af/UpfProfilingCampaign) — Phase 1:
  lab measurement campaign and UPF power/QoS model training.
- UpfRLControllers — downstream consumer of `exports/traffic_forecaster/`.
