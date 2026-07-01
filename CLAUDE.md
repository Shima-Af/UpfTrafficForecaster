# CLAUDE.md — UPF Digital Twin: Session Memory & Continuation Guide

> ⚠️ **STALE — describes the superseded v1 (per-gNodeB STGNN + LSTM) branch.**
> The current branch is `cluster-first-stgnn`, which clusters **first** (SKATER → K nodes)
> and forecasts cluster signals directly. For the accurate, code-verified pipeline see
> [`PIPELINE.md`](PIPELINE.md). In particular, this branch uses a **single global-max**
> normaliser with **no per-service byte conversion** (`dl_norm = bs_tile_sum / global_max`,
> `metadata.json`), and physical units are a downstream scenario concern (`alpha`), not
> applied here. Ignore the "byte-proportional / 26.72 Mbps" guidance below.

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
- Digital Twin: 2-layer ML model (Layer 1: throughput/CPU/loss/delay regressors; Layer 2: power model via stacking)
- PPO controller: 3-action selection (config 0/1/2)
- Traffic forecaster: city-level LSTM (scalar ρ̂_{t+1}) fed by Netflix/Lyon aggregate

**This repository (Phase 2) extends Phase 1 to distributed multi-UPF:**
- Replace the scalar city-level LSTM with a **Spatio-Temporal Graph Neural Network (STGNN)**
  that produces per-gNodeB forecasts for all 965 gNodeBs in Lyon simultaneously
- Cluster gNodeBs into **K geographic UPF service areas** (K to be decided — see Section 6)
- Run **K independent PPO controllers** (one per UPF, reusing Phase 1 logic exactly)
- PhD contribution: STGNN upgrades the "Forecasting Module" box in Figure 3 of Phase 1 paper

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

**⚠️ SUPERSEDED on `cluster-first-stgnn` — do NOT apply on this branch.** The text below
describes the old v1 byte-proportional normalization. The current branch uses a single
global-max normaliser with no byte conversion (see the banner at the top and `PIPELINE.md`).
Original v1 note follows for history only:

**~~IMPORTANT — normalization fix (implemented, do not revert):~~**
The raw parquets store per-service dl_norm. Summing directly distorts relative contributions
(0.5 Netflix ≠ 0.5 YouTube in bytes — Netflix scale_factor is 5× larger).
Fix in `src/dataset.py:_load_bs_parquets()`: convert each service's dl_norm back to raw bytes
using `metadata.json` scale_factors, sum across services, then re-normalise by combined global max.
- Old (wrong): `dl_norm_total = dl_norm_Netflix + dl_norm_YouTube + dl_norm_DailyMotion`
- New (correct): `dl_bytes = dl_norm × scale_factor` → sum → divide by combined global_max
- City-level peak CORRECTED: **653.9 Mbps** (was 1341.5 Mbps with distorted normalization)
- Calibration factor: **26.72 Mbps per dl_norm unit** (node-level)

**Key data quality issue — May 12, 2019:**
Data collection system outage from ~01:00 to ~18:00 on May 12. Fix: replace with same-slot
values from exactly 7 days prior (May 5). Handled by `apply_prev_week_fill()`.

**Voronoi tile-to-gNodeB assignment:**
Each of the 122k tiles is assigned to the nearest gNodeB using Voronoi tessellation.
The mapping is stored in `data/graphs/voronoi_map.parquet` (tile_id → site_id).

---

## 3. Architecture Decisions (all agreed with researcher)

### 3.1 Input Features (8 total)

| Feature | Description |
|---|---|
| `dl_norm` | Current normalised total traffic (byte-proportional sum of 3 services) |
| `dl_norm_lag_96` | Same slot yesterday (96 × 15 min = 24 h back) |
| `dl_norm_lag_192` | Same slot 2 days ago (192 × 15 min = 48 h back) |
| `hour_sin`, `hour_cos` | Cyclic time-of-day encoding (slot 0–95 mapped to circle) |
| `dow_sin`, `dow_cos` | Cyclic day-of-week encoding |
| `is_weekend` | Binary 0/1 |

**Excluded:** month_sin/cos (dataset is only 77 days — insufficient for seasonal learning).
**Excluded:** short in-window lags (lag_1, lag_2, lag_4) — the GRU temporal encoder already
captures intra-window dependencies; only out-of-window lags add new information.

### 3.2 STGNN Architecture

**File:** `src/models/stgnn.py`

Temporal-then-spatial design:
1. **GRU temporal encoder** — processes each node's seq_len-step input series independently
   (all B×N sequences packed into one GRU call for efficiency)
2. **2-layer Graph Attention Network (GAT)** — mixes information across Voronoi-adjacent
   gNodeB neighbours; layer 1 uses `num_heads=4` (concat → H*4 channels), layer 2 uses 1 head
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

**Checkpoint format** — saved as a dict, NOT a plain state_dict:
```python
ckpt = torch.load('models/best_model.pt', map_location=device)
model.load_state_dict(ckpt['model_state'])   # key is 'model_state', not the dict itself
# Also contains: ckpt['epoch'], ckpt['val_loss'], ckpt['params'], ckpt['feature_cols']
```

### 3.3 LSTM Baseline

**File:** `src/models/lstm.py` — per-site LSTM with site embedding.
- Site embedding: `Embedding(965, 16)` concatenated with LSTM hidden state before output head
- Scaler: StandardScaler fitted on X features only (targets are in original dl_norm space)
- **DO NOT** pass `scaler` to `evaluate_loader()` for LSTM — targets are already in dl_norm space
- **DO** pass `scaler` to `evaluate_loader()` for STGNN — targets are in scaled space

### 3.4 Training Results (as of this session)

| Model | Norm | Hidden | WAPE | SLA (≤20%) | Params |
|---|---|---|---|---|---|
| STGNN v1 | distorted | 64 | — | 26.7% | 73,284 |
| STGNN v2 | distorted | 128 | — | 41.4% | — |
| STGNN v3 | **fixed** | 64 | **36.2%** | **42.8%** | 73,284 |
| LSTM baseline | fixed | 128 | 0.9%* | 99.98%* | 219,060 |

*LSTM WAPE/SLA is artificially low: site embeddings memorise each node's pattern (no spatial
generalisation). At cluster level, STGNN errors cancel out and WAPE improves significantly.

**Saved model files:**
- `models/best_model.pt` — STGNN v3 (current best, fixed normalization)
- `models/scaler.pkl` — STGNN v3 StandardScaler (mean≈0.010, scale≈0.012)
- `models/best_model_lstm.pt` — LSTM baseline
- `models/scaler_lstm.pkl` — LSTM StandardScaler (fitted on X only)
- `models/best_model_stgnn_v1.pt`, `models/scaler_stgnn_v1.pkl` — archived v1

### 3.5 K=? UPF Clusters — BLOCKING DECISION

**Status: under discussion with colleague (as of April 2026).**

Method agreed: pure lat/lon k-means (geographic proximity → low RAN-to-UPF latency).
Old traffic-weighted clustering is superseded by this decision.

Three options being considered:
- **Option A (capacity):** K = ⌈653.9 / C_UPF⌉. With C_UPF=400 Mbps → K=2; with 200 Mbps → K=4
- **Option B (geographic/latency):** fix K by coverage radius (≤5 ms → ≤5 km → K≈4–5) ← **preferred**
- **Option C (energy-aware):** sweep K and pick best energy savings (requires full control loop first)

**Do not implement clustering until K is decided.**
The `upf_clusters.parquet` currently has K=4 traffic-weighted clusters — these are stale
and will be replaced once K is decided and lat/lon k-means is re-run.

### 3.6 Multi-UPF Control (Scenario A — agreed)

**NOT multi-agent RL.** K independent single-agent MDPs.

Each UPF controller:
- Receives the aggregate cluster-level traffic forecast from the STGNN
- Runs the same PPO logic as in Phase 1 (3-action config selection)
- Has no communication with other UPF controllers

**Justification:** 3GPP TS 23.501 PDU session binding — each UE session is pinned to one
UPF for its lifetime. Cross-UPF interference doesn't exist at the traffic level.

---

## 4. DVC Pipeline Status

```
build_graph  ✅  data/graphs/voronoi_map.parquet + bs_locations + graph_edges
preprocess   ✅  data/netmob/processed/ (79 day-parquets, 77 valid days)
train        ✅  STGNN v3 trained (28 epochs, early stop). models/best_model.pt
evaluate     ❌  NOT YET RUN (src/evaluate.py stage not adapted yet)
```

**dvc.yaml train stage** — already updated to include STGNN deps:
```yaml
deps: [src/train.py, src/dataset.py, src/features.py, src/models/lstm.py,
       src/models/stgnn.py, src/evaluate.py, data/netmob/processed,
       data/graphs/edge_index.npy, data/graphs/node_index.parquet]
```

---

## 5. Key Files Reference

```
params.yaml                         ← all tunable parameters (single source of truth)
CLAUDE.md                           ← this file — session memory
src/
  models/
    lstm.py                         ← LSTM baseline with site embedding
    stgnn.py                        ← STGNN model (GRU + GAT), implemented & trained
  dataset.py                        ← _load_bs_parquets() [byte-proportional norm],
                                       build_graph_datasets(), apply_prev_week_fill()
  features.py                       ← add_time_features(), add_lag_features(),
                                       build_site_sequences(), fit_scaler(), apply_scaler()
  train.py                          ← unified training loop (lstm + stgnn, MLflow logging)
  evaluate.py                       ← mae, rmse, wape [NOT mape], sla_compliance,
                                       _inverse_dl_norm(), evaluate_loader(scaler=...)
  aggregate.py                      ← Voronoi assignment, build_graph stage
  preprocess.py                     ← raw tiles → BS-level parquets
  twin.py                           ← Digital Twin adapter (Phase 1 interface)
  forecast.py                       ← inference wrapper (not yet updated for STGNN)
  scripts/
    benchmark_training_speed.py     ← times forward+backward pass per batch
notebooks/
  01_analytics.ipynb                ← spatial/temporal data exploration (complete)
  02_forecasting_analysis.ipynb     ← prediction results, all plots (complete)
reports/
  progress_report.tex / .pdf        ← 2-page colleague briefing (gitignored)
data/
  netmob/
    raw/Lyon/                       ← raw .txt files + GeoJSON + Cartoradio CSV
    processed/                      ← 79 day-parquets (BS-level, 3 services)
  graphs/
    voronoi_map.parquet             ← tile_id → site_id
    node_index.parquet              ← site_id → node_idx (0-based, canonical ordering)
    edge_index.npy                  ← (2, 5544) int64 — Voronoi adjacency
    edge_attr.npy                   ← (5544,) float32 — inverse distance weights
    bs_locations.parquet            ← site_id, lat, lon
    upf_clusters.parquet            ← STALE — will be replaced after K decision
    upf_cluster_config.json         ← STALE — will be replaced after K decision
models/                             ← DVC-tracked, gitignored
  best_model.pt                     ← STGNN v3 checkpoint (dict with 'model_state' key)
  scaler.pkl                        ← STGNN v3 StandardScaler
  best_model_lstm.pt                ← LSTM baseline checkpoint
  scaler_lstm.pkl                   ← LSTM StandardScaler (X only)
results/                            ← DVC-tracked, gitignored
  metrics_train.json                ← STGNN v3 test metrics
  training_log_stgnn_v3_fixednorm.txt
  training_log_lstm.txt
```

---

## 6. Immediate Next Steps

### Step 1 — Decide K (BLOCKING)
Discuss with colleague. Options in Section 3.5. Once K is decided:
```bash
# Re-run lat/lon k-means with agreed K
python -m src.cluster --method latlon --k <K>    # (script to be written)
# Update upf_clusters.parquet and upf_cluster_config.json
```

### Step 2 — Implement STGNN → cluster aggregation in src/forecast.py
```python
clusters = pd.read_parquet('data/graphs/upf_clusters.parquet')
node_index = pd.read_parquet('data/graphs/node_index.parquet')
# For each cluster k: sum pred_dl[:, node_idx_k, :] * 26.72 → Mbps signal
```

### Step 3 — PPO controller (reuse Phase 1 code)
Wire K independent controllers, each receiving one cluster's STGNN forecast.

### Step 4 — Evaluation
Compare: Single-UPF DPDK | K-UPF STGNN+PPO | K-UPF LSTM+PPO | oracle

---

## 7. Known Issues / Things to Watch

1. **K clustering is stale:** `upf_clusters.parquet` has old traffic-weighted K=4 clusters.
   Do not use for PPO until re-done with lat/lon k-means and agreed K.

2. **CUDA env on this server:** Do NOT set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   — GRID A100D-40C virtual GPU does not support CUDA VMM and will crash.

3. **STGNN batch_size must be 4:** GRU workspace ∝ B×N×seq_len×hidden. At batch_size=64,
   GRU alone needs ~18 GB. Keep batch_size=4 for STGNN.

4. **LSTM batch_size must be 512:** At batch_size=4, LSTM has 1.19M batches/epoch (~3h).
   Use batch_size=512 (~9300 batches, ~90 sec/epoch).

5. **evaluate.py uses WAPE not MAPE:** MAPE explodes on sparse traffic (near-zero denominators).
   WAPE = Σ|y_true - y_pred| / Σ|y_true| × 100 — robust to near-zero values.

6. **prev_week fill is slow:** `apply_prev_week_fill()` runs inside `build_graph_datasets()`
   every training run (~a few minutes on 77 days × 965 nodes). Cache if it becomes a bottleneck.

7. **forecast.py not updated for STGNN:** Still handles scalar output. Needs update for
   (B, N, horizon) tensor shape before evaluation stage can run.

---

## 8. Research Framing (for writing / presentations)

The contribution is cleanly decomposable as:

```
Phase 1 paper:   scalar ρ̂_{t+1}  →  1 PPO controller  →  1 UPF
                 (city-level LSTM)

Phase 2 (this):  per-gNodeB STGNN  →  spatial aggregation  →  K cluster signals
                  (965-node GRU+GAT)    (lat/lon k-means)    →  K independent PPO controllers
                                                               →  K UPFs
```

**Why STGNN over plain LSTM:**
Neighbouring gNodeBs share load during events (concerts, football matches, commuting peaks).
A city-level LSTM loses spatial resolution. The STGNN captures spatial correlations via GAT
message-passing, allowing each UPF controller to receive a geographically meaningful forecast.
LSTM's 0.9% WAPE is misleading — it memorises per-node patterns via site embeddings and cannot
generalise spatially. STGNN's 36% node-level WAPE improves significantly at cluster level.

**Why GAT over GCN:**
GAT learns attention weights per edge (per-neighbour importance). In practice, not all Voronoi
neighbours are equally correlated — a busy commercial cell next to a residential area should have
different message weight than two neighbouring commercial cells.

**Calibration chain:**
dl_norm=1.0 → 26.72 Mbps (node-level) → city peak = 653.9 Mbps (sum of 965 nodes × mean_dl)
