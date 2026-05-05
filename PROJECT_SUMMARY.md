# UPF Digital Twin — Phase 2: STGNN-Based Multi-UPF Orchestration
## Summary as a Thesis Module

---

## 1. Module Goal

This module is **Phase 2** of a PhD research system for energy-efficient orchestration of 5G User Plane Functions (UPFs). The overarching contribution replaces the scalar, city-level traffic forecast from Phase 1 with a **Spatio-Temporal Graph Neural Network (STGNN)** that produces per-cluster traffic predictions. These predictions drive **K independent PPO controllers**, one per geographic UPF service area, enabling distributed, energy-aware orchestration of a multi-UPF 5G core.

**Phase 1 (published baseline):**
`city-level LSTM (scalar ρ̂_{t+1}) → 1 PPO controller → 1 UPF`

**Phase 2 (this module):**
`per-gNodeB STGNN (965 nodes) → spatial aggregation → K cluster signals → K PPO controllers → K UPFs`

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | NetMob 2023 — French operator 4G/5G data |
| City | Lyon |
| Period | 77 days (2019-03-16 to 2019-06-01) |
| Temporal resolution | 15-minute bins → 96 slots/day |
| Services | Netflix, YouTube, DailyMotion |
| Spatial units | 965 gNodeBs (Cartoradio, Lyon bbox) covering ~122k 100m×100m tiles |
| Data schema | `(timestamp, site_id, service, dl_norm)` where `dl_norm ∈ [0,1]` |

**Normalization fix (critical):** Each service's `dl_norm` is first converted back to raw bytes via per-service `scale_factor` (from `metadata.json`), then summed across services, then re-normalized by the combined global max. This ensures Netflix's 5× larger scale does not distort aggregation. Resulting city-level peak: **653.9 Mbps** (26.72 Mbps per dl_norm unit at node level).

**Anomaly fix:** May 12, 2019 outage (~01:00–18:00) is filled using same-slot values from May 5 (7 days prior).

---

## 3. Pipeline Stages

### Stage 1 — Build Graph (`src/aggregate.py` + `src/preprocess.py`)

| | Detail |
|---|---|
| **Input** | Raw `.txt` tiles + Cartoradio gNodeB CSV |
| **Operation** | Voronoi tessellation: assign each of 122k tiles to nearest gNodeB; compute Voronoi adjacency edges |
| **Output (shapes)** | `voronoi_map.parquet`: `(~122k, 2)` — `tile_id → site_id` |
| | `node_index.parquet`: `(965, 2)` — `site_id → node_idx` (0-based) |
| | `edge_index.npy`: `(2, 5544)` int64 — COO adjacency |
| | `edge_attr.npy`: `(5544,)` float32 — inverse-distance weights |
| | `bs_locations.parquet`: `(965, 3)` — `site_id, lat, lon` |

---

### Stage 2 — Preprocess (`src/preprocess.py` → `src/dataset.py`)

| | Detail |
|---|---|
| **Input** | Raw tile-level `.txt` files (per service, per day) |
| **Operation** | Aggregate tiles → gNodeBs via Voronoi map; apply byte-proportional normalization; fill May 12 anomaly; write per-day parquets |
| **Output (shapes)** | `Lyon_YYYYMMDD.parquet` × 79 files |
| | Each file: `(N_valid_rows, 4)` — `(timestamp, site_id, service, dl_norm)` |
| | After loading all days: dense matrix `(T_total=7392, N=965)` of `dl_norm` |

---

### Stage 3 — Feature Engineering (`src/features.py`)

Applied per node independently.

| Feature | Formula | Purpose |
|---|---|---|
| `dl_norm` | current slot value | Traffic level |
| `dl_norm_lag_96` | slot 96 steps back (24 h) | Yesterday same slot |
| `dl_norm_lag_192` | slot 192 steps back (48 h) | Day-before-yesterday |
| `hour_sin/cos` | `sin/cos(2π × slot/96)` | Time-of-day cycle |
| `dow_sin/cos` | `sin/cos(2π × dow/7)` | Day-of-week cycle |
| `is_weekend` | `{0,1}` | Weekend flag |

**Feature tensor shape:** `(T_total, N=965, F=8)`

**Scaler:** `StandardScaler` fitted on training X features only (not targets). Targets stay in `dl_norm` space for interpretability.

---

### Stage 4 — Dataset Construction (`src/dataset.py`)

**Temporal split (no shuffling — preserves temporal order):**

```
T_total = 7392 slots
train : 70% → T_train = 5174
val   : 15% → T_val  = 1109
test  : 15% → T_test = 1109
```

**Sliding window with stride=1:**

| Split | n_samples formula | Approximate count |
|---|---|---|
| Train | T_train − seq_len − horizon + 1 | ~5074 |
| Val | T_val − seq_len − horizon + 1 | ~1009 |
| Test | T_test − seq_len − horizon + 1 | ~1009 |

**Per-sample shapes (STGNN):**

| Tensor | Shape | Description |
|---|---|---|
| `x` | `(seq_len=96, N=965, F=8)` | 24 h input window, all nodes, all features |
| `y` | `(N=965, horizon=4)` | Next-1h ground truth, all nodes |

**After DataLoader batching:**

| Tensor | Shape |
|---|---|
| `x` | `(B=4, seq_len=96, N=965, F=8)` |
| `y` | `(B=4, N=965, horizon=4)` |

*(B=4 is mandatory — GRU workspace ∝ B×N×seq_len×hidden; larger batch exceeds GPU RAM)*

---

### Stage 5 — STGNN Model Forward Pass (`src/models/stgnn.py`)

**Architecture:** Temporal-first, then Spatial (GRU → GAT)

| Step | Layer | Input shape | Output shape | Notes |
|---|---|---|---|---|
| 1 | Reshape | `(B, 96, 965, 8)` | `(B×965, 96, 8)` | Pack all nodes into batch dimension |
| 2 | GRU temporal encoder | `(B×965, 96, 8)` | `(B×965, hidden=64)` | 2-layer GRU; captures intra-window dynamics per node independently |
| 3 | Reshape | `(B×965, 64)` | `(B, 965, 64)` | Restore node dimension |
| 4 | Tile graph edges | `(2, 5544)` | `(2, B×5544)` | Block-diagonal for PyG batched GAT |
| 5 | GAT layer 1 | `(B×965, 64)` | `(B×965, 64×4=256)` | 4 attention heads, concat |
| 6 | GAT layer 2 | `(B×965, 256)` | `(B×965, 64)` | 1 head, no concat |
| 7 | Linear output head | `(B×965, 64)` | `(B×965, horizon=4)` | Per-node projection |
| 8 | Reshape | `(B×965, 4)` | `(B, 965, 4)` | **Final output** |

**Key hyperparameters (from `params.yaml`):**

```yaml
seq_len:     96    # 24 h lookback
horizon:     4     # 1 h ahead forecast
hidden_size: 64
num_layers:  2     # GRU layers
dropout:     0.2
num_heads:   4     # GAT attention heads
```

**Total parameters:** 73,284

---

### Stage 6 — Training (`src/train.py`)

| Setting | Value |
|---|---|
| Loss | MAE (L1Loss) |
| Optimizer | Adam, lr=0.001 |
| Scheduler | CosineAnnealingLR (T_max=50, η_min=1e-6) |
| Gradient clipping | max_norm=1.0 |
| Early stopping | patience=10 on val loss |
| Batch size | 4 (memory constraint) |
| Epochs run | 28 (early stop) |
| Logging | MLflow |

**Checkpoint format:**
```python
{
  'model_state': state_dict,   # load with model.load_state_dict(ckpt['model_state'])
  'epoch':       int,
  'val_loss':    float,
  'params':      dict,
  'feature_cols': list
}
```

---

### Stage 7 — Cluster Aggregation (`src/forecast.py` — to be updated)

After STGNN inference:

```
STGNN output:  (B, N=965, horizon=4)   dl_norm per node
     ↓
Voronoi cluster assignment: node_idx → cluster_k
     ↓
Aggregate:  sum pred_dl[:, node_idx_k, :] × 26.72  → Mbps signal per cluster
     ↓
K cluster forecasts:  (B, K, horizon=4)  in Mbps
     ↓
K independent PPO controllers (one per UPF)
```

---

### Stage 8 — Evaluation (`src/evaluate.py`)

**Metrics:**

| Metric | Formula | Why |
|---|---|---|
| WAPE | `Σ|y−ŷ| / Σ|y| × 100` | Robust to near-zero sparse traffic (MAPE explodes) |
| MAE | `mean(|y−ŷ|)` | Absolute forecast error |
| RMSE | `sqrt(mean((y−ŷ)²))` | Penalises large errors |
| SLA compliance | Fraction of slots where error ≤ 20% | Operational bound |

**Inverse calibration for evaluation:** `dl_norm → Mbps` via `× 26.72`

---

## 4. Results Summary

| Model | Norm fix | Hidden | Node WAPE | SLA (≤20%) | Params |
|---|---|---|---|---|---|
| STGNN v1 | ✗ distorted | 64 | — | 26.7% | 73,284 |
| STGNN v2 | ✗ distorted | 128 | — | 41.4% | — |
| **STGNN v3** | **✓ fixed** | **64** | **36.2%** | **42.8%** | **73,284** |
| LSTM baseline | ✓ fixed | 128 | 0.9%* | 99.98%* | 219,060 |

*LSTM figures are misleading: site embeddings memorize per-node patterns with no spatial generalization. At cluster level, STGNN error cancels and WAPE improves significantly. LSTM cannot generalize to new or merged coverage areas.

---

## 5. Why STGNN Over LSTM (Thesis Justification)

| Axis | LSTM baseline | STGNN |
|---|---|---|
| Spatial awareness | None — treats each gNodeB independently | GAT message-passing captures inter-cell correlations |
| Generalisation | Memorises per-site patterns via embedding | Learns transferable spatial-temporal patterns |
| Event handling | Cannot model load spill from neighbouring cells | Neighbouring nodes pass scaled messages during events |
| Phase 2 compatibility | Single scalar ρ̂ per site, no cluster signal | Native `(B, N, horizon)` output → direct aggregation to K clusters |
| Attention interpretability | None | Per-edge attention weights are inspectable and publishable |

**Why GAT over GCN:** GAT learns per-edge attention weights. In practice Voronoi neighbours are not equally correlated — a busy commercial cell next to a residential area should weight neighbours differently than two co-located commercial cells.

---

## 6. Data & Model Flow — One-Line Summary per Stage

```
Raw tiles (122k × 77 days)
  → [Stage 1] Voronoi tessellation → 965-node graph + edge_index (2, 5544)
  → [Stage 2] Preprocess + normalize → dl_norm per (timestamp, site_id)
  → [Stage 3] Feature engineering → (T, 965, 8) feature tensor
  → [Stage 4] Sliding windows → batches of (4, 96, 965, 8) input / (4, 965, 4) target
  → [Stage 5] STGNN forward pass → (4, 965, 4) per-node forecasts
  → [Stage 6] Cluster aggregation → (K, 4) Mbps signals
  → [Stage 7] K × PPO controllers → K UPF config decisions (0/1/2)
  → [Stage 8] Evaluate WAPE / SLA / energy savings vs. baselines
```

---

## 7. Constraints & Engineering Decisions

| Constraint | Value | Reason |
|---|---|---|
| STGNN batch size | **4** | GRU workspace ∝ B×N×seq_len×hidden; B=64 needs ~18 GB GPU RAM |
| LSTM batch size | **512** | At B=4, 1.19M batches/epoch (~3 h); B=512 → ~90 sec/epoch |
| CUDA alloc | No `expandable_segments` | GRID A100D-40C vGPU does not support CUDA VMM |
| Metric | WAPE not MAPE | MAPE diverges on sparse near-zero traffic |
| Clustering | Lat/lon k-means | Pure geographic proximity minimises RAN-to-UPF latency |
| Multi-UPF control | K independent MDPs (not MARL) | 3GPP TS 23.501 PDU session binding — no cross-UPF interference |

---

## 8. Pending Steps (Blocking on K Decision)

1. **Decide K** — three options: capacity-based (K=2–4), latency-based (K=4–5, ≤5 km radius), or energy sweep
2. **Re-run lat/lon k-means** with agreed K → update `upf_clusters.parquet`
3. **Update `src/forecast.py`** for `(B, N, horizon)` tensor output
4. **Wire K PPO controllers** (reuse Phase 1 PPO logic exactly)
5. **Run DVC evaluate stage** and compare: Single-UPF DPDK | K-UPF STGNN+PPO | K-UPF LSTM+PPO | Oracle

---

This module is self-contained as a thesis chapter: it takes raw operator network data, constructs a spatial graph of 965 gNodeBs, trains a GRU+GAT STGNN to forecast per-node traffic 1 hour ahead, aggregates forecasts to K geographic UPF service areas, and feeds those signals to K independent PPO controllers for energy-optimal UPF configuration selection.
