# CLAUDE.md — UPF Digital Twin: Session Memory & Continuation Guide

This file is the primary context document for Claude Code sessions on this repository.
Read it fully before doing anything else. It records the research context, all decisions
already made, the current state of the codebase, and the immediate next steps.

---

## 1. Who Is the User and What Is This Project?

**Researcher:** Shima Afshar Borji — PhD student at DITEN, University of Genova / CNIT S2N National Lab.

**Research goal:** Energy-efficient orchestration of 5G User Plane Functions (UPFs) using
AI-based Digital Twins.

**Published baseline (Phase 1):**
> "AI-Driven Energy-Efficient Orchestration of UPFs in 5G Core Networks"
> (Afshar Borji et al., submitted to Computer Communications, 2025)

Phase 1 is a single-UPF system:
- Physical testbed: DPDK vs. USR×1 vs. USR×2 UPF configurations
- Digital Twin: MLP regressors for power consumption and QoS metrics
- PPO controller: 3-action selection (config 0/1/2)
- Traffic forecaster: city-level LSTM (scalar ρ̂_{t+1}) fed by Netflix/Lyon aggregate

**This repository (Phase 2) extends Phase 1 to distributed multi-UPF:**
- Replace the scalar city-level LSTM with a **Spatio-Temporal Graph Neural Network (STGNN)**
  that produces per-gNodeB forecasts for all 965 gNodeBs in Lyon simultaneously
- Cluster gNodeBs into **K=4 geographic UPF service areas** using traffic-weighted k-means
- Run **K=4 independent PPO controllers** (one per UPF, reusing Phase 1 logic exactly)
- PhD contribution framing: the STGNN upgrades the "Forecasting Module" box in Figure 3
  of the Phase 1 paper, enabling the single-node controller to scale to K distributed nodes

**Collaboration style:**
- Shima prefers to discuss and agree on approach BEFORE implementation begins
- She wants to learn STGNN/GNN concepts through building — explain as we go
- Strong domain expertise in 5G core, UPF, NFV, DRL, testbed measurement
- Less familiar with ML architecture design and PyTorch Geometric — be patient

---

## 2. Dataset: NetMob 2023 Lyon

- **Source:** NetMob 2023 challenge dataset — aggregated 4G/5G traffic from a French operator
- **City:** Lyon
- **Period:** 77 days (2019-03-16 to 2019-06-01, approximately)
- **Resolution:** 15-minute bins → 96 slots/day
- **Services:** Netflix, YouTube, DailyMotion (3 services)
- **Grid:** 100×100 m tiles covering Lyon (287 cols × 426 rows = ~122k tiles)
- **gNodeBs:** 965 base stations from Cartoradio (Lyon bounding box: lat 45.60–45.90, lon 4.70–5.05)

**Processed data schema** (in `data/netmob/processed/Lyon_YYYYMMDD.parquet`, 79 files):
```
timestamp  : datetime64[ns]   (15-min resolution, Paris local time)
site_id    : str               (Cartoradio gNodeB identifier)
service    : str               (DailyMotion | Netflix | YouTube)
dl_norm    : float32           (∈ [0, 1], normalised by per-service global max)
```

**Key data quality issue — May 12, 2019:**
Data collection system outage from ~01:00 to ~18:00 on May 12. Values are near-zero
across ALL sites and ALL services simultaneously (impossible in real traffic), with sharp
on/off boundaries. NOT real traffic — NOT a public holiday or social event.
Fix: replace the 01:00–18:00 window with same-slot values from exactly 7 days prior (May 5).

**Voronoi tile-to-gNodeB assignment:**
Each of the 122k tiles is assigned to the nearest gNodeB using Voronoi tessellation.
The mapping is stored in `data/graphs/voronoi_map.parquet` (tile_id → site_id).

---

## 3. Architecture Decisions (all agreed with researcher)

### 3.1 Input Features (8 total)

| Feature | Description |
|---|---|
| `dl_norm` | Current normalised total traffic (sum of 3 services) |
| `dl_norm_lag_96` | Same slot yesterday (96 × 15 min = 24 h back) |
| `dl_norm_lag_192` | Same slot 2 days ago (192 × 15 min = 48 h back) |
| `hour_sin`, `hour_cos` | Cyclic time-of-day encoding (slot 0–95 mapped to circle) |
| `dow_sin`, `dow_cos` | Cyclic day-of-week encoding |
| `is_weekend` | Binary 0/1 |

**Excluded:** month_sin/cos (dataset is only 77 days — insufficient for seasonal learning).
**Excluded:** short in-window lags (lag_1, lag_2, lag_4) — the LSTM/GRU temporal encoder
already captures intra-window dependencies; only out-of-window lags add new information.

### 3.2 STGNN Architecture

**File:** `src/models/stgnn.py`

Temporal-then-spatial design:
1. **GRU temporal encoder** — processes each node's seq_len-step input series independently
   (all B×N sequences packed into one GRU call for efficiency)
2. **2-layer Graph Attention Network (GAT)** — mixes information across Voronoi-adjacent
   gNodeB neighbours; layer 1 uses `num_heads=4` (concat), layer 2 uses 1 head (plain)
3. **Linear output head** — per-node projection from hidden_size → horizon

Key shapes:
- Input `x`: `(B, seq_len, N, F)` — B time windows, N=965 nodes, F=8 features
- Output: `(B, N, horizon)` — per-node forecasts for all 965 gNodeBs

Graph is tiled into a block-diagonal batch structure inside `forward()` — no PyG
Data/Batch objects needed at training time.

**Current hyperparameters (params.yaml → stgnn):**
```yaml
seq_len: 96       # 24 h lookback
horizon: 4        # 1 h ahead (4 × 15 min)
hidden_size: 64
num_layers: 2
dropout: 0.2
num_heads: 4
```

### 3.3 K=4 UPF Clusters

**Why K=4:**
- Calibration: Netflix city-level peak aggregate = 400 Mbps (from Phase 1 paper)
- City-level peak across all services: 1341.5 Mbps (computed from dataset)
- USR×1 UPF saturation threshold: 400 Mbps
- K = ceil(1341.5 / 400) = **4**

**Method:** Traffic-weighted k-means on standardised (lat, lon, mean_traffic) features.
Pure lat/lon k-means produced a single 485-node cluster at ~368 Mbps (near saturation).
Traffic weighting spreads load more evenly.

**Results** (saved in `data/graphs/upf_clusters.parquet`):

| Cluster | gNodeBs | Peak (Mbps) |
|---|---|---|
| 0 | 231 | ~272 |
| 1 | 302 | ~103 |
| 2 | 361 | ~91 |
| 3 | 71 | ~66 |

Calibration factor: `1 dl_norm_unit = 26.62 Mbps` (saved in `upf_cluster_config.json`).

### 3.4 Multi-UPF Control (Scenario A — agreed)

**NOT multi-agent RL.** K=4 independent single-agent MDPs.

Each UPF controller:
- Receives the aggregate cluster-level traffic forecast from the STGNN
- Runs the same PPO logic as in Phase 1 (3-action config selection)
- Has no communication with other UPF controllers

**Justification:** 3GPP TS 23.501 PDU session binding — each UE session is pinned to one
UPF for its lifetime. Cross-UPF interference doesn't exist at the traffic level.
PhD contribution is in the STGNN forecaster, not the controller design.

---

## 4. DVC Pipeline Status

```
build_graph  ✅  data/graphs/voronoi_map.parquet + bs_locations + graph_edges
preprocess   ✅  data/netmob/processed/ (79 day-parquets, 77 valid days)
train        ❌  NOT YET RUN (requires GPU — see Section 6)
evaluate     ❌  NOT YET RUN
```

**Note:** `dvc.yaml` currently lists only `lstm` deps in the train stage.
After training runs successfully, update dvc.yaml to add stgnn deps
(`src/models/stgnn.py`, `data/graphs/edge_index.npy`, `data/graphs/node_index.parquet`).

---

## 5. Key Files Reference

```
params.yaml                         ← all tunable parameters (single source of truth)
src/
  models/
    lstm.py                         ← LSTM baseline (working, not yet trained)
    stgnn.py                        ← STGNN model (GRU + GAT, implemented)
  dataset.py                        ← TrafficDataset, MultiSiteDataset, GraphTrafficDataset,
                                       build_datasets(), build_graph_datasets(),
                                       apply_prev_week_fill()
  features.py                       ← add_time_features(), add_lag_features(),
                                       make_sequences(), fit_scaler(), apply_scaler()
  train.py                          ← unified training loop (lstm + stgnn)
  evaluate.py                       ← metrics (MAE, RMSE, MAPE, SLA compliance)
  aggregate.py                      ← Voronoi assignment, build_graph stage
  preprocess.py                     ← raw tiles → BS-level parquets
  twin.py                           ← Digital Twin adapter (Phase 1 interface)
  forecast.py                       ← inference wrapper
data/
  netmob/
    raw/Lyon/                       ← raw .txt files + GeoJSON + Cartoradio CSV
    processed/                      ← 79 day-parquets (already BS-level, 3 services)
  graphs/
    voronoi_map.parquet             ← tile_id → site_id
    node_index.parquet              ← site_id → node_idx (0-based, canonical ordering)
    edge_index.npy                  ← (2, 5544) int64 — Voronoi adjacency
    edge_attr.npy                   ← (5544,) float32 — inverse distance weights
    bs_locations.parquet            ← site_id, lat, lon
    upf_clusters.parquet            ← site_id, cluster (0–3), lat, lon, mean_traffic
    upf_cluster_config.json         ← K, mbps calibration, method metadata
models/                             ← best_model.pt, scaler.pkl (written by train stage)
results/                            ← metrics_train.json, metrics_test.json (written by eval)
```

---

## 6. Immediate Next Steps (GPU Server)

### Step 1 — Environment setup

```bash
# Match torch version to your CUDA version, e.g. CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# torch-geometric (match torch + CUDA):
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.x.0+cu118.html

# Remaining dependencies:
pip install -r requirements.txt
```

### Step 2 — Verify data is present

```bash
ls data/netmob/processed/   # should show 79 .parquet files
ls data/graphs/              # should show voronoi_map, node_index, edge_index, etc.
```

If processed data is missing, re-run the DVC pipeline:
```bash
dvc repro build_graph
dvc repro preprocess
```

### Step 3 — Run STGNN training

```bash
# params.yaml already has: training.model: stgnn  and  training.device: auto
python -m src.train
```

Expected output:
```
[train] Model=stgnn  Device=cuda
[train] Building graph datasets (this may take ~1 min)...
[build_graph_datasets] N=965  F=8  Train=XXXX  Val=XXXX  Test=XXXX
[train] STGNN  N=965  F=8  Params=73,284
  Epoch    1  train=X.XXXXXX  val=X.XXXXXX  ← best
  Epoch    2  ...
```

Loss should decrease steadily. If it plateaus early, try:
- Increase `stgnn.hidden_size` to 128
- Increase `training.lr` to 0.003 for first 10 epochs

### Step 4 — Run LSTM baseline (for comparison)

```bash
# Change params.yaml:  training.model: lstm
python -m src.train
# saves to models/best_model.pt — rename first to avoid overwriting STGNN checkpoint
cp models/best_model.pt models/best_model_lstm.pt
```

### Step 5 — After training: cluster-level aggregation

The STGNN outputs `(B, N=965, horizon=4)` per-gNodeB forecasts.
To feed K=4 PPO controllers, aggregate node forecasts to cluster level:

```python
import pandas as pd, torch
clusters = pd.read_parquet('data/graphs/upf_clusters.parquet')
# cluster_load[k] = sum of dl_norm forecasts for all nodes in cluster k
# Then convert to Mbps: cluster_load_mbps = cluster_load * 26.62
```

This aggregation step (STGNN → cluster signals → K PPO controllers) is the key
pipeline that connects Phase 2 forecasting to Phase 1 control logic.
It is NOT yet implemented — write it in `src/forecast.py` or a new `src/controller.py`.

---

## 7. Known Issues / Things to Watch

1. **dvc.yaml train stage is incomplete:** It lists `src/models/lstm.py` as a dep but
   not `stgnn.py` or graph files. Update after confirming training works.

2. **Cluster 3 is small (71 nodes, ~66 Mbps peak):** This may reflect a real suburban
   low-load area of Lyon, but check its geographic extent before concluding. If it causes
   load imbalance in the PPO controller, consider merging into an adjacent cluster (K=3).

3. **prev_week fill performance:** `apply_prev_week_fill()` loops over NaN entries —
   it may be slow on first run with the full 77-day dataset (a few minutes). This runs
   inside `build_graph_datasets()` every training run. If it becomes a bottleneck, cache
   the filled DataFrame to a parquet file.

4. **Scaler is fitted on dl_norm total (sum of 3 services):** The StandardScaler sees
   values in approximately [0, 3] (sum of three [0,1]-normalised services). This is fine
   but means the scaler's mean/std should be checked after training — they should be
   roughly mean≈0.1, std≈0.15 given sparse traffic patterns.

5. **evaluate.py and forecast.py** are not yet adapted for STGNN outputs. After training,
   update `src/evaluate.py:evaluate_loader` (already done) and `src/forecast.py` to handle
   the `(B, N, horizon)` output tensor shape.

---

## 8. Research Framing (for writing / presentations)

The contribution is cleanly decomposable as:

```
Phase 1 paper:   scalar ρ̂_{t+1}  →  1 PPO controller  →  1 UPF
                 (city-level LSTM)

Phase 2 (this):  per-gNodeB STGNN  →  spatial aggregation  →  K cluster signals
                  (965-node GAT+GRU)     (Voronoi k-means)    →  K independent PPO controllers
                                                               →  K UPFs
```

The STGNN replaces the "Forecasting Module" box in Figure 3 of the Phase 1 paper.
Everything downstream (PPO controller, Digital Twin profiling models) is reused unchanged.

**Why STGNN over plain LSTM:**
Neighbouring gNodeBs share load during events (concerts, football matches, commuting peaks).
A city-level LSTM loses spatial resolution — it cannot distinguish which geographic area
is peaking. The STGNN captures spatial correlations via GAT message-passing, allowing
each of the K UPF controllers to receive a geographically meaningful load forecast
rather than a fraction of the city-level total.

**Why GAT over GCN:**
GAT learns attention weights per edge (per-neighbour importance), which is more expressive
than GCN's fixed normalised adjacency. In practice, not all Voronoi neighbours are equally
correlated — a busy commercial district next to a residential area should have different
message weight than two neighbouring commercial cells.
