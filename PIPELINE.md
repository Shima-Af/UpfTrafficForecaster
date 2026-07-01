# UPF Traffic Forecaster — Pipeline (authoritative, code-verified)

This document is the **ground-truth** description of the pipeline as implemented in
`src/` and configured in `config.yaml` / `params.yaml`. Where the older
`README.md` / `PROJECT_SUMMARY.md` disagree with this file, this file reflects the
actual code (see "Known doc inconsistencies" at the bottom).

## Goal

Predict near-future downlink traffic load for **K cluster-level service areas**
(prospective MEC / UPF sites) in Lyon, from NetMob 2023 mobile-traffic traces,
using a Spatio-Temporal GNN. Forecasts are produced in **normalized [0,1] units**;
mapping to physical capacity (Gbps) is a *downstream* scenario concern (the
controller's `alpha`), not part of this model.

## Step-by-step

1. **Read NetMob 2023 traffic.**
   Raw values are **already dimensionless** — privacy-preserving aggregate
   indicators, normalized across the full NetMob dataset, with *no physical unit*
   (`src/netmob_loader.py`). They are *not* bit/s. 15-min bins, 96 slots/day,
   100 m tiles, downlink only.

2. **Re-scale to a clean [0,1] (`dl_norm`).**
   `dl_norm = bs_tile_sum / global_max`, a single global max across **all** services
   (`global_max = 148178608.0`, `data/netmob/processed/metadata.json`).
   This is *why* the controller needs `alpha`: there is no physical unit to recover,
   so `alpha` ([0,1] → Gbps) is an assumed scenario mapping, applied **after**
   forecasting — never before (see "alpha placement" below).

3. **Tile → base-station assignment (Voronoi).**
   gNodeB locations (Cartoradio) define a Voronoi tessellation; each 100 m tile is
   assigned to its containing cell. ~965 gNodeBs. (`src/aggregate.py`, `method: voronoi`.)

4. **Cluster gNodeBs into K nodes (SKATER).**
   ~965 BSs is too many for the model/controllers, so they are clustered.
   Method = **SKATER** (geographic MST), min cluster size 3. Swept **K ∈ {5,10,20}**.

5. **Build the coarse graph.**
   Each cluster = one node. Two clusters share an edge iff **any** member BS of one is
   Voronoi-adjacent to **any** member BS of the other; edge weight = mean cross-cluster
   affinity. Self-loops added. (`src/coarsen.py`.)

6. **Aggregate cluster signal (sum).**
   At each 15-min slot, sum `dl_norm` over all member BSs → cluster signal. Valid as a
   direct sum because all services share one normalizer. Conservation is checked by
   `verify_coarsening`.

7. **STGNN forecast.**
   GAT (4 heads, 2 layers) for space + GRU (1 layer) for time, hidden dim 64.
   **Lookback = 24 h** (`seq_len: 96` × 15 min), **horizon = 1 h ahead**
   (`horizon: 4` × 15 min). Chronological train/val/test split (no leakage).

8. **Evaluate (`src/evaluate_forecast.py`).**
   Metrics in `dl_norm` units: MAE, RMSE, **WAPE** (used instead of MAPE because traffic
   is sparse/near-zero). Per-cluster, per-horizon, and peak vs low-load breakdowns.
   Baselines: last-value (persistence), historical-mean, seasonal-naive.

## alpha placement (verdict)

`alpha` is a single global constant (1.0 → 1 Gbps/unit). Apply it **after** the forecast,
in the controller — not before. Pre-scaling carries zero predictive information (it does
not change traffic *shape*), de-tunes training (MSE scales with α²), and destroys scenario
reusability (one trained model must serve every α). Forecast in [0,1]; scale at the
control boundary.

## Current results (service = Netflix, seed 42)

| K  | test MAE | test RMSE | test WAPE | persistence WAPE | seasonal-naive WAPE |
|----|----------|-----------|-----------|------------------|---------------------|
| 5  | 0.1666   | 0.2739    | **17.77** | 23.56            | 29.46               |
| 10 | 0.1107   | 0.1855    | 23.61     | 29.45            | 35.56               |
| 20 | 0.0641   | 0.1158    | 27.33     | 34.99            | 41.62               |

STGNN beats all baselines at every K. Note WAPE **rises** with K (smaller, burstier
per-cluster signals) while MAE falls. The evaluator's automatic `best_k = 5` by WAPE; the
**K=10 decision is a deployment choice (number of MEC/UPF controllers, ~2.7 km cluster
footprint), not an accuracy-optimal one** — and should be documented as such.

## Known doc inconsistencies (to reconcile)

- **Normalization:** `PROJECT_SUMMARY.md` describes a byte-proportional per-service
  renormalization (26.72 Mbps/unit, 653.9 Mbps peak). The **implemented** method is plain
  `global_max` with *no byte conversion* (metadata.json + `src/preprocess.py`). Pick one and
  align docs to code.
- **README.md** still describes the older per-gNodeB **LSTM** design (`src/models/lstm.py`,
  "stgnn upcoming", MAPE, SLA compliance, ~700 nodes / 54,013 tiles). Current code is the
  STGNN cluster-first pipeline (965 nodes / ~122k tiles, WAPE, `src/model.py`).
