"""
aggregate.py
------------
Voronoi-based aggregation of NetMob 100×100 m tiles to gNodeB (base station) level.

Pipeline
--------
1. Load gNodeB site locations from Cartoradio CSV (filtered to Lyon bbox).
2. Convert each tile_id to its centroid (lat, lon) using the Lyon grid config.
3. Build a Voronoi diagram: assign each tile to its nearest gNodeB site.
4. Aggregate per-tile traffic DataFrame to per-gNodeB time series.

Tile geometry
-------------
tile_id = row * n_cols + col
Centroid lat/lon computed from grid origin (params.yaml → grid section).
Tiles run South↓ (lat decreases with row) and East→ (lon increases with col).
100 m tile ≈ 0.000899° lat, ≈ 0.001272° lon at latitude 45.75°.

Public API
----------
load_base_stations(params)  -> GeoDataFrame (site_id, lon, lat, geometry)
build_tile_centroids(tile_ids, params) -> GeoDataFrame (tile_id, row, col, geometry)
build_voronoi_map(tile_ids, params) -> pd.Series (tile_id → site_id)
aggregate_to_bs(df, voronoi_map) -> pd.DataFrame  (timestamp, site_id, dl_norm)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# Optional spatial imports — graceful error if not installed
try:
    from scipy.spatial import cKDTree
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

try:
    import geopandas as gpd
    from shapely.geometry import Point, box
    from shapely.ops import voronoi_diagram
    from shapely.geometry import MultiPoint
    _GEO_OK = True
except ImportError:
    _GEO_OK = False


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

# Approximate degrees per 100 m at latitude ~45.75°
_DEG_LAT_PER_100M = 100 / 111_320          # ≈ 0.000899°
_LAT_REF          = 45.75                  # reference latitude for lon scaling
_DEG_LON_PER_100M = 100 / (111_320 * math.cos(math.radians(_LAT_REF)))  # ≈ 0.001272°


def tile_id_to_rowcol(tile_ids: np.ndarray, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Split tile_id into (row, col) arrays."""
    rows = tile_ids // n_cols
    cols = tile_ids % n_cols
    return rows, cols


def rowcol_to_latlon(
    rows: np.ndarray,
    cols: np.ndarray,
    origin_lat: float,
    origin_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert grid (row, col) to tile centroid (lat, lon).

    Origin is the top-left corner of tile (0, 0).
    Centroid = origin + (col + 0.5) * tile_size_lon  [East]
                       - (row + 0.5) * tile_size_lat  [South]
    """
    lat = origin_lat - (rows + 0.5) * _DEG_LAT_PER_100M
    lon = origin_lon + (cols + 0.5) * _DEG_LON_PER_100M
    return lat, lon


# ---------------------------------------------------------------------------
# Load base stations
# ---------------------------------------------------------------------------

def load_base_stations(params: dict) -> pd.DataFrame:
    """
    Load gNodeB site locations from Cartoradio CSV.

    Returns DataFrame with columns: site_id, lon, lat
    Filtered to bbox defined in params → aggregation → bbox.
    """
    cfg = params["aggregation"]
    csv_path = Path(cfg["bs_csv"])
    enc = cfg.get("bs_encoding", "latin-1")

    df = pd.read_csv(csv_path, sep=";", encoding=enc)
    # Normalise column names (handle encoding artefacts in header)
    df.columns = [c.encode("ascii", "ignore").decode().strip() for c in df.columns]
    # Rename to standard names
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

    # Filter to Lyon bounding box
    bb = cfg["bbox"]
    mask = (
        (df["lat"] >= bb["lat_min"]) & (df["lat"] <= bb["lat_max"]) &
        (df["lon"] >= bb["lon_min"]) & (df["lon"] <= bb["lon_max"])
    )
    df = df[mask].reset_index(drop=True)
    df["site_id"] = df["site_id"].astype(str)
    return df[["site_id", "lon", "lat"]]


# ---------------------------------------------------------------------------
# Build tile centroids
# ---------------------------------------------------------------------------

def build_tile_centroids(tile_ids: np.ndarray, params: dict) -> pd.DataFrame:
    """
    Compute centroid lat/lon for each tile_id.

    Returns DataFrame with columns: tile_id, row, col, lat, lon
    """
    grid = params["grid"]
    rows, cols = tile_id_to_rowcol(tile_ids, grid["n_cols"])
    lat, lon = rowcol_to_latlon(rows, cols, grid["origin_lat"], grid["origin_lon"])
    return pd.DataFrame({
        "tile_id": tile_ids,
        "row":     rows,
        "col":     cols,
        "lat":     lat,
        "lon":     lon,
    })


# ---------------------------------------------------------------------------
# Voronoi: assign each tile to nearest base station
# ---------------------------------------------------------------------------

def build_voronoi_map(tile_ids: np.ndarray, params: dict) -> pd.Series:
    """
    Assign each tile to its nearest base station using a KD-tree (Voronoi equivalent).

    Parameters
    ----------
    tile_ids : array of tile IDs present in the processed data
    params   : full params dict

    Returns
    -------
    pd.Series indexed by tile_id, values = site_id (nearest BS)
    """
    if not _SCIPY_OK:
        raise ImportError("scipy is required for Voronoi mapping: pip install scipy")

    bs_df     = load_base_stations(params)
    tile_df   = build_tile_centroids(tile_ids, params)

    # Project to approximate flat metric coords (metres) for KD-tree
    # Simple equirectangular: good enough for ~30 km area
    lat_ref = params["grid"]["origin_lat"] - params["grid"]["n_rows"] / 2 * _DEG_LAT_PER_100M
    bs_xy   = _latlon_to_xy(bs_df["lat"].values,  bs_df["lon"].values,  lat_ref)
    tile_xy = _latlon_to_xy(tile_df["lat"].values, tile_df["lon"].values, lat_ref)

    tree         = cKDTree(bs_xy)
    _, indices   = tree.query(tile_xy, k=1)

    min_tiles = params["aggregation"].get("min_tiles_per_bs", 1)
    assigned_sites = bs_df.iloc[indices]["site_id"].values

    result = pd.Series(assigned_sites, index=tile_df["tile_id"].values, name="site_id")

    # Drop BSs with too few assigned tiles
    counts = result.value_counts()
    valid  = counts[counts >= min_tiles].index
    result = result[result.isin(valid)]

    print(f"[aggregate] {len(bs_df)} base stations, "
          f"{len(result)} tiles assigned, "
          f"{result.nunique()} active BSs")
    return result


# ---------------------------------------------------------------------------
# Aggregate tile-level DataFrame to BS-level
# ---------------------------------------------------------------------------

def aggregate_to_bs(
    df: pd.DataFrame,
    voronoi_map: pd.Series,
) -> pd.DataFrame:
    """
    Aggregate per-tile traffic to per-BS time series.

    Parameters
    ----------
    df           : DataFrame with columns [timestamp, cell_id, dl_norm, ul_norm]
    voronoi_map  : Series tile_id → site_id

    Returns
    -------
    DataFrame with columns [timestamp, site_id, dl_norm, ul_norm]
    dl_norm aggregated as sum (total load across tiles in each BS Voronoi cell).
    """
    df = df.copy()
    df = df[df["cell_id"].isin(voronoi_map.index)]
    df["site_id"] = voronoi_map.reindex(df["cell_id"]).values

    agg = (
        df.groupby(["timestamp", "site_id"], sort=False)
          .agg(dl_norm=("dl_norm", "sum"), ul_norm=("ul_norm", "sum"))
          .reset_index()
    )
    return agg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _latlon_to_xy(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_ref: float,
) -> np.ndarray:
    """Equirectangular projection → (x_m, y_m) relative to lat_ref."""
    R = 6_371_000  # Earth radius in metres
    x = np.radians(lon) * R * math.cos(math.radians(lat_ref))
    y = np.radians(lat) * R
    return np.column_stack([x, y])
