"""
cluster_first — cluster-first STGNN pipeline for UPF Digital Twin Phase 2.

Pipeline order (inverted from the original approach):
    1. cluster.py   → AttributedSpectralClustering groups 965 BSs into K areas
    2. coarsen.py   → graph coarsening builds K-node super-graph + aggregate signals
    3. dataset.py   → ClusterTrafficDataset wraps cluster-level time series
    4. model.py     → ClusterSTGNN (GAT→GRU) forecasts K cluster aggregates
    5. train.py     → full training loop with K-sweep and MLflow logging
    6. evaluate.py  → forecast metrics, UPF assignment, and energy analysis

This design aligns what the model optimises (cluster-level MAE) with what
the UPF controller needs (cluster-level load threshold decisions), removing
the mismatch in the original pipeline where per-BS MAE was minimised but
cluster aggregate error drove UPF decisions.
"""
