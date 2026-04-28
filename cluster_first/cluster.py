"""
cluster.py — Attributed Spectral Clustering for gNodeB grouping.

Groups the 965 Lyon gNodeBs into K UPF service areas by combining three
complementary affinity signals:

  A_graph   : Voronoi border adjacency (captures geographic proximity and
              the topology of the physical RAN — nearby cells should be
              served by the same UPF to keep RAN-to-UPF latency low)

  A_density : RBF on load-magnitude features (clusters cells with similar
              absolute traffic load so no single UPF is overloaded while
              another sits idle)

  A_shape   : RBF on temporal-shape features (clusters cells with similar
              diurnal patterns so the STGNN sees coherent aggregate signals
              rather than mixtures of out-of-phase peak times)

The composite affinity A = w1·A_graph + w2·A_density + w3·A_shape is fed
to sklearn's SpectralClustering(affinity='precomputed'), which embeds the
graph into its eigenvector space before k-means.

References
----------
Ng, Jordan, Weiss. "On Spectral Clustering" (NeurIPS 2002)
Shi & Malik. "Normalized Cuts and Image Segmentation" (TPAMI 2000)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler


class AttributedSpectralClustering:
    """
    Spectral clustering that combines graph structure, load-density, and
    temporal-shape information into a single composite affinity matrix.

    Parameters
    ----------
    n_clusters      : K — number of UPF service areas
    w_graph         : weight on Voronoi adjacency affinity
    w_density       : weight on load-magnitude RBF affinity
    w_shape         : weight on temporal-shape RBF affinity
    density_features: column names in bs_stats_df for density affinity
    shape_features  : column names in bs_stats_df for shape affinity
    random_state    : seed for spectral clustering k-means step
    """

    def __init__(
        self,
        n_clusters: int,
        w_graph: float,
        w_density: float,
        w_shape: float,
        density_features: List[str],
        shape_features: List[str],
        random_state: int = 42,
    ) -> None:
        self.n_clusters       = n_clusters
        self.w_graph          = w_graph
        self.w_density        = w_density
        self.w_shape          = w_shape
        self.density_features = density_features
        self.shape_features   = shape_features
        self.random_state     = random_state

        # Set after fit()
        self.labels_: np.ndarray | None = None          # (N,) int
        self.affinity_matrix_: np.ndarray | None = None  # (N, N) float

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        bs_stats_df: pd.DataFrame,
        bs_locations_df: pd.DataFrame,
        output_dir: str | Path = "data/cluster_first",
    ) -> "AttributedSpectralClustering":
        """
        Compute composite affinity and run spectral clustering.

        The geographic component (A_graph) is a dense lat/lon RBF kernel —
        every BS pair receives a geographic affinity based on actual distance,
        not just Voronoi neighbours.  This is critical: the old sparse Voronoi
        adjacency only connected 5-6 neighbours per BS, so its effective weight
        was diluted by the dense A_density and A_shape matrices, producing
        geographically scattered clusters and singletons.

        gamma_geo controls the distance decay.  Default 50 corresponds to
        sigma ~ 0.1 degrees (~11 km); BSs further than ~20 km apart have
        near-zero affinity, enforcing geographic compactness.

        Saves cluster_assignments.parquet and affinity_matrix.npy to
        output_dir so downstream coarsening can reload them without
        re-running the expensive spectral decomposition.

        Parameters
        ----------
        bs_stats_df    : DataFrame indexed by site_id (string), columns must
                         include all density_features and shape_features.
                         Rows must be in node_index order (same as bs_locations_df).
        bs_locations_df: DataFrame with columns [site_id, lat, lon], indexed
                         or aligned to the same node order as bs_stats_df.
        output_dir     : directory for saved artefacts

        Returns
        -------
        self (for method chaining)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        N = len(bs_stats_df)
        if N != len(bs_locations_df):
            raise ValueError(
                f"bs_stats_df has {N} rows but bs_locations_df has "
                f"{len(bs_locations_df)} rows — they must match."
            )

        # ---- Step 1-2: standardise feature blocks ----
        scaler_density = StandardScaler()
        scaler_shape   = StandardScaler()

        X_density = scaler_density.fit_transform(
            bs_stats_df[self.density_features].values.astype(np.float64)
        )
        X_shape = scaler_shape.fit_transform(
            bs_stats_df[self.shape_features].values.astype(np.float64)
        )

        # ---- Step 3: RBF affinity matrices ----
        A_density = rbf_kernel(X_density, gamma=1.0)   # (N, N), in [0, 1]
        A_shape   = rbf_kernel(X_shape,   gamma=1.0)   # (N, N), in [0, 1]

        # ---- Step 4: dense geographic affinity from lat/lon ----
        # gamma=50 -> sigma~0.1 deg (~11 km): BSs >20 km apart get ~0 affinity,
        # ensuring clusters are geographically compact contiguous zones.
        # This replaces the sparse Voronoi adjacency which caused scattered
        # singletons because it only connected 5-6 neighbours per BS.
        coords  = bs_locations_df[["lat", "lon"]].values.astype(np.float64)
        A_graph = rbf_kernel(coords, gamma=50.0)        # (N, N), in [0, 1]

        # ---- Step 5-6: composite affinity, symmetrised ----
        A = (
            self.w_graph   * A_graph   +
            self.w_density * A_density +
            self.w_shape   * A_shape
        )
        A = (A + A.T) / 2                               # ensure exact symmetry

        # ---- Step 7: spectral clustering ----
        sc = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity="precomputed",
            random_state=self.random_state,
            n_init=10,
            assign_labels="kmeans",
        )
        sc.fit(A)

        # ---- Step 8: store results ----
        self.labels_          = sc.labels_.astype(np.int32)   # (N,)
        self.affinity_matrix_ = A

        # ---- Save artefacts ----
        site_ids = bs_stats_df.index.tolist()
        assignments = pd.DataFrame(
            {"site_id": site_ids, "cluster_id": self.labels_}
        )
        assignments.to_parquet(output_dir / "cluster_assignments.parquet", index=False)
        np.save(output_dir / "affinity_matrix.npy", A.astype(np.float32))

        print(
            f"[AttributedSpectralClustering] K={self.n_clusters}  "
            f"saved -> {output_dir}/cluster_assignments.parquet"
        )
        return self

    # ------------------------------------------------------------------
    # Cluster summary
    # ------------------------------------------------------------------

    def get_cluster_stats(self, bs_stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarise each cluster by member count and feature centroids.

        Returns a DataFrame indexed by cluster_id with:
          - n_members        : how many BSs belong to this cluster
          - mean of every feature column in bs_stats_df
          - dominant_shape   : qualitative label derived from the shape
                               features (highest-valued feature name)

        Useful for deciding whether a cluster corresponds to a dense
        commercial area, a residential area, a transport corridor, etc.
        """
        if self.labels_ is None:
            raise RuntimeError("Call fit() before get_cluster_stats().")

        df = bs_stats_df.copy()
        df["cluster_id"] = self.labels_

        numeric_cols = [c for c in df.columns if c != "cluster_id"]
        stats = df.groupby("cluster_id")[numeric_cols].mean()
        stats["n_members"] = df.groupby("cluster_id").size()

        # Qualitative label: which shape feature has the highest z-score
        shape_block = stats[self.shape_features]
        z_scores    = (shape_block - shape_block.mean()) / (shape_block.std() + 1e-8)
        stats["dominant_shape"] = z_scores.idxmax(axis=1)

        return stats.sort_index()
