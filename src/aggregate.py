"""
aggregate.py
------------
Build gNodeB coverage areas and tile-to-BS assignment for the UPF Digital Twin.

Pipeline (logical order)
------------------------
1. Load gNodeB site locations (Cartoradio CSV, bbox-filtered, deduplicated).
2. Build Voronoi coverage polygons clipped to the city bbox:
     - scipy.spatial.Voronoi in projected (x, y) metre space
     - Infinite boundary regions bounded via far-point padding
     - ridge_points gives BS adjacency directly (reused for graph edges)
3. Load tile centroids from NetMob GeoJSON (exact polygon means).
4. Assign tiles → BSs via spatial join (tile centroid inside which coverage polygon).
   Tiles outside all coverage polygons are dropped — not forced to nearest BS.
5. Determine active BSs (those with ≥ min_tiles_per_bs assigned tiles).
6. Build graph edges from Voronoi ridge_points, trimmed to active BSs only.
7. Save voronoi_map.parquet, bs_locations.parquet, graph_edges.parquet.

Public API
----------
load_base_stations(params)                                -> pd.DataFrame
build_tile_centroids(tile_ids, params)                    -> pd.DataFrame
build_voronoi(bs_df, params)                              -> VoronoiResult
build_voronoi_map(tile_df, bs_df, vresult, params)        -> pd.Series
build_graph_edges(bs_df, vresult, active_sites, params)   -> pd.DataFrame
aggregate_to_bs(df, voronoi_map, per_service)             -> pd.DataFrame
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from scipy.spatial import Voronoi
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, shape
    from shapely.ops import transform, unary_union
    _GEO_OK = True
except ImportError:
    _GEO_OK = False


# ---------------------------------------------------------------------------
# Result container — keeps Voronoi artefacts together so they are computed once
# ---------------------------------------------------------------------------

@dataclass
class VoronoiResult:
    """Everything produced by build_voronoi(), shared by downstream steps."""
    polygons : list          # shapely Polygon (or None) per BS, clipped to bbox
    vor      : object        # scipy.spatial.Voronoi (includes ridge_points)
    bs_xy    : np.ndarray    # projected BS coordinates (x_m, y_m)
    lat_ref  : float         # reference latitude used for projection
    n_bs     : int           # number of original BSs (excludes far padding points)


# ---------------------------------------------------------------------------
# Step 1 — Load base stations
# ---------------------------------------------------------------------------

def load_geojson(params: dict) -> dict:
    """Load the NetMob GeoJSON once and reuse it across build steps."""
    geojson_path = Path(params["aggregation"]["geojson_path"])
    with open(geojson_path, encoding="utf-8") as f:
        return json.load(f)


def load_city_boundary(geojson: dict):
    """Build the city footprint by unioning all tile polygons from the GeoJSON."""
    polygons = [shape(feature["geometry"]) for feature in geojson["features"] if feature.get("geometry")]
    boundary = unary_union(polygons)
    if boundary.is_empty:
        raise ValueError("GeoJSON did not produce a valid city boundary")
    return boundary


def load_base_stations(params: dict, city_boundary=None) -> pd.DataFrame:
    """
    Load gNodeB site locations from Cartoradio CSV.

        Filtering:
            - City footprint if provided, otherwise bbox fallback.
      - Deduplicate co-located antennas at identical coordinates.

    Returns DataFrame: site_id, lon, lat
    """
    cfg      = params["aggregation"]
    csv_path = Path(cfg["bs_csv"])
    enc      = cfg.get("bs_encoding", "latin-1")

    df = pd.read_csv(csv_path, sep=";", encoding=enc)
    df.columns = [c.encode("ascii", "ignore").decode().strip() for c in df.columns]

    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "num" in cl and "ro" in cl:
            col_map[c] = "site_id"
        elif "longitude" in cl:
            col_map[c] = "lon"
        elif "latitude" in cl:
            col_map[c] = "lat"
    df.rename(columns=col_map, inplace=True)

    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df.dropna(subset=["lon", "lat"], inplace=True)

    if city_boundary is not None:
        mask = [city_boundary.covers(Point(float(lon), float(lat))) for lon, lat in zip(df["lon"], df["lat"])]
        df = df[pd.Series(mask, index=df.index)].reset_index(drop=True)
    else:
        bb   = cfg["bbox"]
        mask = (
            (df["lat"] >= bb["lat_min"]) & (df["lat"] <= bb["lat_max"]) &
            (df["lon"] >= bb["lon_min"]) & (df["lon"] <= bb["lon_max"])
        )
        df = df[mask].reset_index(drop=True)
    df["site_id"] = df["site_id"].astype(str)

    n_before = len(df)
    df = df.drop_duplicates(subset=["lat", "lon"], keep="first").reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed:
        print(f"[aggregate] Removed {n_removed} co-located BS duplicates "
              f"({n_before} -> {len(df)} unique positions)")

    print(f"[aggregate] Loaded {len(df)} base stations")
    return df[["site_id", "lon", "lat"]]


# ---------------------------------------------------------------------------
# Step 2 — Build Voronoi coverage polygons
# ---------------------------------------------------------------------------

def build_voronoi(bs_df: pd.DataFrame, params: dict, city_boundary=None) -> VoronoiResult:
    """
    Build Voronoi coverage polygons for all base stations, clipped to the city footprint.

    Strategy for infinite boundary regions
    ----------------------------------------
    scipy.spatial.Voronoi produces infinite ridges for BSs on the convex hull.
    We pad the input with 4 far-away dummy points (one per cardinal direction,
    10× the point-cloud span away). This forces all original BS regions to be
    finite. The dummy points are excluded from ridge_points filtering downstream.

    All geometry is in projected (x_m, y_m) metre space for accuracy.

    Returns VoronoiResult containing polygons, the Voronoi object, projected
    coordinates, reference latitude, and the number of original BSs.
    """
    if not _SCIPY_OK:
        raise ImportError("scipy is required: pip install scipy")
    if not _GEO_OK:
        raise ImportError("geopandas and shapely are required: pip install geopandas shapely")

    cfg     = params["aggregation"]
    lat_ref = float(bs_df["lat"].mean())

    # Project BS locations to metre space
    bs_xy = _latlon_to_xy(bs_df["lat"].values, bs_df["lon"].values, lat_ref)
    n_bs  = len(bs_df)

    if city_boundary is not None:
        boundary = _project_geometry(city_boundary, lat_ref)
    else:
        # Fallback to bbox if no city footprint is available.
        bb      = cfg["bbox"]
        bbox_corners_lat = [bb["lat_min"], bb["lat_min"], bb["lat_max"], bb["lat_max"]]
        bbox_corners_lon = [bb["lon_min"], bb["lon_max"], bb["lon_max"], bb["lon_min"]]
        bbox_xy          = _latlon_to_xy(
            np.array(bbox_corners_lat), np.array(bbox_corners_lon), lat_ref
        )
        boundary = Polygon(bbox_xy)

    # Add 4 far dummy points to bound all Voronoi regions
    center = bs_xy.mean(axis=0)
    span   = max(float(bs_xy[:, 0].max() - bs_xy[:, 0].min()),
                 float(bs_xy[:, 1].max() - bs_xy[:, 1].min()))
    far    = span * 10
    extra_xy = np.array([
        center + [0,  far],
        center + [0, -far],
        center + [ far, 0],
        center + [-far, 0],
    ])
    all_xy = np.vstack([bs_xy, extra_xy])

    vor = Voronoi(all_xy)

    # Build clipped polygon for each original BS
    polygons = []
    for i in range(n_bs):
        region_idx = vor.point_region[i]
        region     = vor.regions[region_idx]
        if not region or -1 in region:
            # Should not occur with far-point padding; keep as None
            polygons.append(None)
            continue
        vertices = vor.vertices[region]
        poly     = Polygon(vertices)
        if not poly.is_valid:
            poly = poly.buffer(0)
        poly = poly.intersection(boundary)
        polygons.append(poly if (poly is not None and not poly.is_empty) else None)

    valid_count = sum(1 for p in polygons if p is not None)
    print(f"[aggregate] Built {valid_count} Voronoi coverage polygons "
          f"(clipped to city footprint, {n_bs - valid_count} empty)")

    return VoronoiResult(
        polygons=polygons,
        vor=vor,
        bs_xy=bs_xy,
        lat_ref=lat_ref,
        n_bs=n_bs,
    )


# ---------------------------------------------------------------------------
# Step 3 — Load tile centroids
# ---------------------------------------------------------------------------

def build_tile_centroids(tile_ids: np.ndarray, params: dict, geojson: Optional[dict] = None, city_boundary=None) -> pd.DataFrame:
    """
    Load exact tile centroids from the NetMob GeoJSON file.

    Centroid = mean of polygon vertices (geographically exact).
    Only tiles within the city footprint are returned when a boundary is provided.

    Returns DataFrame: tile_id, lat, lon
    """
    if geojson is None:
        geojson = load_geojson(params)

    tile_id_set = set(tile_ids.tolist())
    rows = []
    for feature in geojson["features"]:
        tid = int(feature["properties"]["tile_id"])
        if tid not in tile_id_set:
            continue
        coords = feature["geometry"]["coordinates"][0][:-1]
        lon_c  = sum(c[0] for c in coords) / len(coords)
        lat_c  = sum(c[1] for c in coords) / len(coords)
        rows.append({"tile_id": tid, "lat": lat_c, "lon": lon_c})

    tile_df = pd.DataFrame(rows)

    n_before = len(tile_df)
    if city_boundary is not None:
        mask = [city_boundary.covers(Point(float(lon), float(lat))) for lon, lat in zip(tile_df["lon"], tile_df["lat"])]
        tile_df = tile_df[pd.Series(mask, index=tile_df.index)].reset_index(drop=True)
    else:
        bb      = params["aggregation"]["bbox"]
        tile_df  = tile_df[
            (tile_df["lat"] >= bb["lat_min"]) & (tile_df["lat"] <= bb["lat_max"]) &
            (tile_df["lon"] >= bb["lon_min"]) & (tile_df["lon"] <= bb["lon_max"])
        ].reset_index(drop=True)

    n_removed = n_before - len(tile_df)
    if n_removed:
        print(f"[aggregate] Tile footprint filter: removed {n_removed} edge tiles "
              f"({n_before} -> {len(tile_df)})")
    return tile_df


# ---------------------------------------------------------------------------
# Step 4 — Assign tiles → BSs via spatial join
# ---------------------------------------------------------------------------

def build_voronoi_map(
    tile_df: pd.DataFrame,
    bs_df: pd.DataFrame,
    vresult: VoronoiResult,
    params: dict,
) -> pd.Series:
    """
    Assign each tile to a BS by checking which Voronoi coverage polygon
    contains the tile centroid (spatial join in projected metre space).

    Tiles whose centroids fall outside all coverage polygons are dropped —
    they are not forced onto the nearest BS.

    Returns pd.Series indexed by tile_id, values = site_id.
    BSs with fewer than min_tiles_per_bs assigned tiles are also dropped.
    """
    if not _GEO_OK:
        raise ImportError("geopandas and shapely are required: pip install geopandas shapely")

    # Build GeoDataFrame of Voronoi coverage polygons (projected space)
    records = [
        {"site_id": bs_df.iloc[i]["site_id"], "geometry": poly}
        for i, poly in enumerate(vresult.polygons)
        if poly is not None
    ]
    voronoi_gdf = gpd.GeoDataFrame(records)

    # Build GeoDataFrame of tile centroid Points (same projected space)
    tile_xy = _latlon_to_xy(tile_df["lat"].values, tile_df["lon"].values, vresult.lat_ref)
    tile_gdf = gpd.GeoDataFrame(
        {"tile_id": tile_df["tile_id"].values},
        geometry=[Point(float(x), float(y)) for x, y in tile_xy],
    )

    # Spatial join: which coverage polygon contains each tile centroid?
    joined = gpd.sjoin(tile_gdf, voronoi_gdf, how="inner", predicate="within")

    # Deduplicate tiles on polygon borders (keep first match)
    joined = joined[~joined.index.duplicated(keep="first")]

    result    = pd.Series(joined["site_id"].values, index=joined["tile_id"].values, name="site_id")
    n_dropped = len(tile_df) - len(result)
    if n_dropped:
        print(f"[aggregate] {n_dropped} tiles dropped (outside all coverage polygons)")

    min_tiles = params["aggregation"].get("min_tiles_per_bs", 1)
    counts    = result.value_counts()
    valid     = counts[counts >= min_tiles].index
    result    = result[result.isin(valid)]

    print(f"[aggregate] {len(result)} tiles assigned to {result.nunique()} active BSs")
    return result


# ---------------------------------------------------------------------------
# Step 6 — Graph edges from Voronoi ridge_points
# ---------------------------------------------------------------------------

def build_graph_edges(
    bs_df: pd.DataFrame,
    vresult: VoronoiResult,
    active_sites: set,
    params: dict,
) -> pd.DataFrame:
    """
    Build spatial graph edges between base stations.

    voronoi_border (default)
        Reads ridge_points from the Voronoi object built in Step 2 — no
        recomputation. Ridges involving the 4 far dummy padding points are
        filtered out automatically (index >= vresult.n_bs).

    knn
        K nearest neighbours by Haversine distance
        (params → aggregation → k_neighbors, default 8).

    In both cases edges where either endpoint is not in active_sites are dropped,
    keeping the graph consistent with BSs that have actual traffic data.

    Returns DataFrame: src_site_id, dst_site_id, distance_km
    Each undirected edge appears once.
    """
    cfg    = params["aggregation"]
    method = cfg.get("edge_method", "voronoi_border")

    if method == "voronoi_border":
        edges = _edges_voronoi_border(bs_df, vresult)
    elif method == "knn":
        k     = cfg.get("k_neighbors", 8)
        edges = _edges_knn(bs_df, vresult.bs_xy, bs_df["site_id"].values, k)
    else:
        raise ValueError(f"Unknown edge_method: {method!r}. "
                         "Choose 'voronoi_border' or 'knn'.")

    df_edges  = pd.DataFrame(edges, columns=["src_site_id", "dst_site_id", "distance_km"])

    # Drop edges involving BSs with no traffic data
    n_before  = len(df_edges)
    mask      = (
        df_edges["src_site_id"].isin(active_sites) &
        df_edges["dst_site_id"].isin(active_sites)
    )
    df_edges  = df_edges[mask].reset_index(drop=True)
    n_dropped = n_before - len(df_edges)
    if n_dropped:
        print(f"[aggregate] Dropped {n_dropped} edges to inactive BSs")

    print(f"[aggregate] Graph: {len(active_sites)} nodes, "
          f"{len(df_edges)} edges (method={method})")
    return df_edges


def _edges_voronoi_border(
    bs_df: pd.DataFrame,
    vresult: VoronoiResult,
    min_shared_border_m: float = 1.0,
) -> list[tuple]:
    """
    Extract edges from Voronoi ridge_points, verified against clipped polygons.

    ridge_points comes from the unconstrained Voronoi (before city-boundary
    clipping). Two BSs that are adjacent in the full tessellation may no longer
    share a border after their polygons are clipped — e.g. two peripheral BSs
    separated by a gap in the city boundary. We discard such phantom edges by
    checking that the intersection of the two clipped polygons is a line of
    positive length (not empty or a single point).

    min_shared_border_m: minimum shared border length in metres to keep an edge.
    """
    site_ids = bs_df["site_id"].values
    edges    = []
    for p, q in vresult.vor.ridge_points:
        if p >= vresult.n_bs or q >= vresult.n_bs:
            continue   # skip ridges to dummy padding points

        poly_p = vresult.polygons[p]
        poly_q = vresult.polygons[q]
        if poly_p is None or poly_q is None:
            continue   # one side has no coverage polygon

        shared = poly_p.intersection(poly_q)
        if shared.is_empty or shared.length < min_shared_border_m:
            continue   # no real shared border after city-boundary clipping

        dist = _haversine_km(
            bs_df.iloc[p]["lat"], bs_df.iloc[p]["lon"],
            bs_df.iloc[q]["lat"], bs_df.iloc[q]["lon"],
        )
        edges.append((site_ids[p], site_ids[q], round(dist, 4)))
    return edges


def _edges_knn(
    bs_df: pd.DataFrame,
    bs_xy: np.ndarray,
    site_ids: np.ndarray,
    k: int,
) -> list[tuple]:
    """K-nearest-neighbour edges (undirected, no duplicates)."""
    from scipy.spatial import cKDTree
    tree               = cKDTree(bs_xy)
    _, nbr_indices     = tree.query(bs_xy, k=k + 1)  # +1: first result is self

    seen  = set()
    edges = []
    for i, neighbours in enumerate(nbr_indices):
        for j in neighbours[1:]:
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            dist = _haversine_km(
                bs_df.iloc[i]["lat"], bs_df.iloc[i]["lon"],
                bs_df.iloc[j]["lat"], bs_df.iloc[j]["lon"],
            )
            edges.append((site_ids[i], site_ids[j], round(dist, 4)))
    return edges


# ---------------------------------------------------------------------------
# Aggregate tile-level DataFrame to BS-level
# ---------------------------------------------------------------------------

def aggregate_to_bs(
    df: pd.DataFrame,
    voronoi_map: pd.Series,
    per_service: bool = False,
) -> pd.DataFrame:
    """
    Sum per-tile traffic into per-BS time series using the voronoi_map assignment.

    Parameters
    ----------
    df           : DataFrame [timestamp, cell_id, service, dl_norm, ul_norm]
    voronoi_map  : Series tile_id → site_id
    per_service  : if True, keep service as a groupby key

    Returns DataFrame [timestamp, site_id, (service,) dl_norm, ul_norm]
    """
    df = df.copy()
    df = df[df["cell_id"].isin(voronoi_map.index)]
    df["site_id"] = voronoi_map.reindex(df["cell_id"]).values

    group_keys = ["timestamp", "site_id", "service"] if per_service else ["timestamp", "site_id"]
    return (
        df.groupby(group_keys, sort=False)
          .agg(dl_norm=("dl_norm", "sum"), ul_norm=("ul_norm", "sum"))
          .reset_index()
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _latlon_to_xy(lat: np.ndarray, lon: np.ndarray, lat_ref: float) -> np.ndarray:
    """Equirectangular projection → (x_m, y_m). Accurate enough for ~30 km area."""
    R = 6_371_000.0
    x = np.radians(lon) * R * math.cos(math.radians(lat_ref))
    y = np.radians(lat) * R
    return np.column_stack([x, y])


def _project_geometry(geometry, lat_ref: float):
    """Project a lon/lat shapely geometry into the metre space used by Voronoi."""
    R = 6_371_000.0
    cos_lat_ref = math.cos(math.radians(lat_ref))

    def _project(x, y, z=None):
        x = np.asarray(x)
        y = np.asarray(y)
        return (np.radians(x) * R * cos_lat_ref, np.radians(y) * R)

    return transform(_project, geometry)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    R    = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


# ---------------------------------------------------------------------------
# CLI entry point  (DVC stage: python -m src.aggregate)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    params_path = "params.yaml"
    out_dir     = Path("data/graphs")

    with open(params_path, encoding="utf-8") as fh:
        cfg_params = yaml.safe_load(fh)

    geojson = load_geojson(cfg_params)
    city_boundary = load_city_boundary(geojson)

    # ------------------------------------------------------------------
    # Step 1: Load base stations inside the city footprint
    # ------------------------------------------------------------------
    bs_df = load_base_stations(cfg_params, city_boundary=city_boundary)

    # ------------------------------------------------------------------
    # Step 2: Build Voronoi coverage polygons + adjacency (once)
    # ------------------------------------------------------------------
    vresult = build_voronoi(bs_df, cfg_params, city_boundary=city_boundary)

    # ------------------------------------------------------------------
    # Step 3: Load tile centroids from GeoJSON
    # ------------------------------------------------------------------
    all_tile_ids = np.array(
        [int(feat["properties"]["tile_id"]) for feat in geojson["features"]],
        dtype=np.int64,
    )
    print(f"[aggregate] {len(all_tile_ids)} tile IDs in GeoJSON")
    tile_df = build_tile_centroids(all_tile_ids, cfg_params, geojson=geojson, city_boundary=city_boundary)

    # ------------------------------------------------------------------
    # Step 4: Assign tiles → BSs via spatial join (coverage polygon check)
    # ------------------------------------------------------------------
    voronoi_map = build_voronoi_map(tile_df, bs_df, vresult, cfg_params)

    # ------------------------------------------------------------------
    # Step 5: Determine active BSs (those with ≥ 1 tile assigned)
    # ------------------------------------------------------------------
    active_sites = set(voronoi_map.values)
    bs_active    = bs_df[bs_df["site_id"].isin(active_sites)].reset_index(drop=True)
    print(f"[aggregate] Active BSs: {len(bs_active)} / {len(bs_df)}")

    # ------------------------------------------------------------------
    # Step 6: Build graph edges — reuses vresult.vor.ridge_points
    # ------------------------------------------------------------------
    edges_df = build_graph_edges(bs_df, vresult, active_sites, cfg_params)

    # ------------------------------------------------------------------
    # Step 7: Save
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    voronoi_out = voronoi_map.reset_index()
    voronoi_out.columns = ["tile_id", "site_id"]
    voronoi_out.to_parquet(out_dir / "voronoi_map.parquet",   index=False, compression="snappy")
    print(f"[aggregate] Saved voronoi_map.parquet   ({len(voronoi_out)} rows)")

    bs_active.to_parquet(out_dir / "bs_locations.parquet",    index=False, compression="snappy")
    print(f"[aggregate] Saved bs_locations.parquet  ({len(bs_active)} base stations)")

    edges_df.to_parquet(out_dir / "graph_edges.parquet",      index=False, compression="snappy")
    print(f"[aggregate] Saved graph_edges.parquet   ({len(edges_df)} edges)")

    print("[aggregate] Done.")
