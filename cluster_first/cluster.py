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
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
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


# =============================================================================
# SKATER Geographic Clustering
# =============================================================================

def _within_ssq(node_list: List[int], coords: np.ndarray) -> float:
    """Sum of squared deviations from centroid for a set of nodes."""
    if len(node_list) <= 1:
        return 0.0
    X = coords[node_list]
    return float(np.sum((X - X.mean(axis=0)) ** 2))


def _bfs(start: int, cut_u: int, cut_v: int,
         mst_adj: Dict[int, List[int]], region: Set[int]) -> Set[int]:
    """
    BFS within `region`, treating the undirected edge (cut_u, cut_v) as removed.
    Returns the connected component reachable from `start`.
    """
    visited: Set[int] = {start}
    queue = [start]
    while queue:
        node = queue.pop()
        for nb in mst_adj[node]:
            if nb in visited or nb not in region:
                continue
            if (node == cut_u and nb == cut_v) or (node == cut_v and nb == cut_u):
                continue
            visited.add(nb)
            queue.append(nb)
    return visited


class SkaterGeographicClustering:
    """
    SKATER-based geographic clustering for UPF service area delineation.

    Produces K geographically contiguous zones by iteratively partitioning
    the Minimum Spanning Tree of the Voronoi adjacency graph.  Each cut
    maximises the reduction in within-cluster geographic variance, so the
    resulting clusters are compact geographic regions with no islands or
    singletons.

    Why SKATER over k-means for this problem:
      - K-means produces circular Voronoi-like zones but ignores the actual
        network topology; a cluster can span a river or motorway that has no
        Voronoi edges, creating unrealistic UPF service areas.
      - SKATER only cuts edges that exist in the Voronoi graph, so every
        cluster boundary is a real geographic discontinuity in the RAN.
      - Guaranteed contiguity: every cluster is a single connected region,
        which directly maps to a contiguous UPF coverage polygon.

    Algorithm (Assuncao et al. 2006):
      1. Build MST of the Voronoi graph, edge weights = geodesic distance.
      2. Repeat K-1 times:
           For each candidate edge in the current tree, compute the within-
           cluster SSQ gain if that edge is cut.  Cut the edge with the
           highest gain, subject to both resulting sub-regions having at
           least `floor` members.

    Parameters
    ----------
    n_clusters : K
    floor      : minimum number of BSs per cluster (prevents singletons)
    random_state : not used (SKATER is deterministic), kept for API compat
    """

    def __init__(
        self,
        n_clusters: int,
        floor: int = 3,
        random_state: int = 42,
    ) -> None:
        self.n_clusters   = n_clusters
        self.floor        = floor
        self.random_state = random_state

        self.labels_: np.ndarray | None = None
        self.affinity_matrix_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        bs_stats_df: pd.DataFrame,
        bs_locations_df: pd.DataFrame,
        edge_index: np.ndarray,
        output_dir: str | Path = "data/cluster_first",
    ) -> "SkaterGeographicClustering":
        """
        Partition 965 BSs into K contiguous geographic zones.

        Parameters
        ----------
        bs_stats_df     : DataFrame indexed by site_id — used only for
                          get_cluster_stats(); not used for clustering itself.
        bs_locations_df : DataFrame with columns [site_id, lat, lon] in
                          node_index order (row i = node_idx i).
        edge_index      : (2, E) int64 Voronoi adjacency in COO format.
        output_dir      : where to save cluster_assignments.parquet and
                          affinity_matrix.npy.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        N = len(bs_locations_df)
        coords = bs_locations_df[["lat", "lon"]].values.astype(np.float64)

        # ---- Build MST of Voronoi graph weighted by geographic distance ----
        src = edge_index[0].astype(int)
        dst = edge_index[1].astype(int)
        mask = src != dst
        src, dst = src[mask], dst[mask]

        # Euclidean distance in lat/lon degrees
        # (sufficient for intra-city scale where 1 deg lat ~ 1 deg lon ~ 111 km)
        dists = np.sqrt(np.sum((coords[src] - coords[dst]) ** 2, axis=1))

        adj = csr_matrix(
            (np.concatenate([dists, dists]),
             (np.concatenate([src, dst]), np.concatenate([dst, src]))),
            shape=(N, N),
        )

        # Check connectivity — SKATER requires a connected graph
        n_components, comp_labels = connected_components(adj, directed=False)
        if n_components > 1:
            print(
                f"[SkaterGeographicClustering] Warning: Voronoi graph has "
                f"{n_components} connected components. Nodes in small components "
                f"will be assigned to the nearest main cluster."
            )

        mst = minimum_spanning_tree(adj)        # (N-1) edges, minimum weight
        mst_coo = mst.tocoo()

        # Build MST adjacency list (undirected)
        mst_adj: Dict[int, List[int]] = {i: [] for i in range(N)}
        mst_edge_list: List[Tuple[int, int, float]] = []
        for u, v, w in zip(mst_coo.row, mst_coo.col, mst_coo.data):
            u, v = int(u), int(v)
            mst_adj[u].append(v)
            mst_adj[v].append(u)
            mst_edge_list.append((u, v, float(w)))

        # ---- SKATER: iterative MST partitioning ----
        # Each region is tracked as a set of node indices.
        # We also track which MST edges belong to each region.
        regions: List[Set[int]] = [set(range(N))]

        for _ in range(self.n_clusters - 1):
            best_gain     = -np.inf
            best_reg_idx  = -1
            best_edge     = (-1, -1)
            best_halves: Tuple[Set[int], Set[int]] = (set(), set())

            for r_idx, region in enumerate(regions):
                region_nodes = list(region)
                region_ssq   = _within_ssq(region_nodes, coords)

                for u, v, _ in mst_edge_list:
                    if u not in region or v not in region:
                        continue

                    # Component reachable from u after cutting (u, v)
                    half_u = _bfs(u, u, v, mst_adj, region)
                    half_v = region - half_u

                    if len(half_u) < self.floor or len(half_v) < self.floor:
                        continue

                    gain = (region_ssq
                            - _within_ssq(list(half_u), coords)
                            - _within_ssq(list(half_v), coords))

                    if gain > best_gain:
                        best_gain    = gain
                        best_reg_idx = r_idx
                        best_edge    = (u, v)
                        best_halves  = (half_u, half_v)

            if best_reg_idx == -1:
                print(
                    f"[SkaterGeographicClustering] Warning: no valid cut found "
                    f"with floor={self.floor}. Stopping at {len(regions)} clusters."
                )
                break

            # Remove the cut edge from mst_adj so future BFS respects all cuts
            u_cut, v_cut = best_edge
            mst_adj[u_cut].remove(v_cut)
            mst_adj[v_cut].remove(u_cut)

            regions.pop(best_reg_idx)
            regions.append(best_halves[0])
            regions.append(best_halves[1])

        # ---- Assign integer labels 0..K-1 ----
        labels = np.zeros(N, dtype=np.int32)
        for label, region in enumerate(regions):
            for node in region:
                labels[node] = label

        self.labels_ = labels

        # ---- Affinity matrix: lat/lon RBF (used by coarsen.py for edge weights) ----
        self.affinity_matrix_ = rbf_kernel(coords, gamma=50.0).astype(np.float32)

        # ---- Save artefacts ----
        site_ids    = bs_locations_df["site_id"].tolist()
        assignments = pd.DataFrame({"site_id": site_ids, "cluster_id": self.labels_})
        assignments.to_parquet(output_dir / "cluster_assignments.parquet", index=False)
        np.save(output_dir / "affinity_matrix.npy", self.affinity_matrix_)

        sizes = np.bincount(self.labels_)
        print(
            f"[SkaterGeographicClustering] K={len(regions)}  "
            f"min_size={sizes.min()}  max_size={sizes.max()}  "
            f"saved -> {output_dir}/cluster_assignments.parquet"
        )
        return self

    # ------------------------------------------------------------------
    # Cluster summary (API-compatible with AttributedSpectralClustering)
    # ------------------------------------------------------------------

    def get_cluster_stats(self, bs_stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarise each cluster by member count and mean traffic features.

        Returns a DataFrame indexed by cluster_id with n_members and
        the mean of every numeric column in bs_stats_df.
        """
        if self.labels_ is None:
            raise RuntimeError("Call fit() before get_cluster_stats().")

        df = bs_stats_df.copy()
        df["cluster_id"] = self.labels_

        numeric_cols = [c for c in df.columns if c != "cluster_id"]
        stats = df.groupby("cluster_id")[numeric_cols].mean()
        stats["n_members"] = df.groupby("cluster_id").size()
        return stats.sort_index()
