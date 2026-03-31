"""
aggregate.py
------------
Voronoi-based aggregation of NetMob 100×100 m tiles to gNodeB (base station) level.

Pipeline
--------
1. Load gNodeB site locations from Cartoradio CSV (filtered to Lyon bbox).
2. Load tile centroids from the NetMob GeoJSON file (exact polygon coordinates).
3. Build a Voronoi diagram: assign each tile to its nearest gNodeB site.
4. Aggregate per-tile traffic DataFrame to per-gNodeB time series.

Tile geometry
-------------
tile_id = row * n_cols + col
Tile centroids are read directly from the NetMob GeoJSON file (Lyon.geojson),
which contains the exact polygon for every tile. No grid-origin formula is used.

Public API
----------
load_base_stations(params)          -> pd.DataFrame (site_id, lon, lat)
build_tile_centroids(tile_ids, params) -> pd.DataFrame (tile_id, lat, lon)
build_voronoi_map(tile_ids, params) -> pd.Series (tile_id → site_id)
aggregate_to_bs(df, voronoi_map)    -> pd.DataFrame (timestamp, site_id, dl_norm)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


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
# Build tile centroids from GeoJSON
# ---------------------------------------------------------------------------

def build_tile_centroids(tile_ids: np.ndarray, params: dict) -> pd.DataFrame:
    """
    Load exact tile centroids from the NetMob GeoJSON file.

    Each feature in the GeoJSON is a polygon; the centroid is the mean of
    its vertices. This is geographically exact — no grid-origin formula.

    Returns DataFrame with columns: tile_id, lat, lon
    Only tiles present in tile_ids are returned.
    """
    geojson_path = Path(params["aggregation"]["geojson_path"])
    with open(geojson_path, encoding="utf-8") as f:
        geojson = json.load(f)

    tile_id_set = set(tile_ids.tolist())
    rows = []
    for feature in geojson["features"]:
        tid = int(feature["properties"]["tile_id"])
        if tid not in tile_id_set:
            continue
        # Outer ring; last point repeats first — exclude it
        coords = feature["geometry"]["coordinates"][0][:-1]
        lon_c = sum(c[0] for c in coords) / len(coords)
        lat_c = sum(c[1] for c in coords) / len(coords)
        rows.append({"tile_id": tid, "lat": lat_c, "lon": lon_c})

    return pd.DataFrame(rows)


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

    bs_df   = load_base_stations(params)
    tile_df = build_tile_centroids(tile_ids, params)

    # Use mean tile latitude as reference for equirectangular projection
    lat_ref = float(tile_df["lat"].mean())
    bs_xy   = _latlon_to_xy(bs_df["lat"].values,  bs_df["lon"].values,  lat_ref)
    tile_xy = _latlon_to_xy(tile_df["lat"].values, tile_df["lon"].values, lat_ref)

    tree       = cKDTree(bs_xy)
    _, indices = tree.query(tile_xy, k=1)

    min_tiles      = params["aggregation"].get("min_tiles_per_bs", 1)
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
    """Equirectangular projection → (x_m, y_m). Good enough for a ~30 km area."""
    earth_radius_m = 6_371_000
    x = np.radians(lon) * earth_radius_m * math.cos(math.radians(lat_ref))
    y = np.radians(lat) * earth_radius_m
    return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# CLI entry point (DVC stage: python -m src.aggregate)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import yaml

    params_path   = "params.yaml"
    processed_dir = Path("data/netmob/processed")
    out_dir       = Path("data/graphs")

    with open(params_path, encoding="utf-8") as fh:
        cfg_params = yaml.safe_load(fh)

    # Collect all unique tile_ids from processed parquets (cell_id column only)
    print("[aggregate] Scanning processed parquets for tile IDs...")
    parquet_files = sorted(processed_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"[aggregate] ERROR: No parquets found in {processed_dir}", file=sys.stderr)
        sys.exit(1)

    tile_id_sets = [
        pd.read_parquet(p, columns=["cell_id"])["cell_id"].unique()
        for p in parquet_files
    ]
    all_tile_ids = np.unique(np.concatenate(tile_id_sets))
    print(f"[aggregate] {len(all_tile_ids)} unique tile IDs across {len(parquet_files)} days")

    # Build Voronoi map
    voronoi_result = build_voronoi_map(all_tile_ids, cfg_params)

    # Load BS locations (filtered to bbox) for saving
    bs_result = load_base_stations(cfg_params)

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    voronoi_df = voronoi_result.reset_index()
    voronoi_df.columns = ["tile_id", "site_id"]
    voronoi_df.to_parquet(out_dir / "voronoi_map.parquet", index=False, compression="snappy")
    print(f"[aggregate] Saved voronoi_map.parquet  ({len(voronoi_df)} rows)")

    bs_result.to_parquet(out_dir / "bs_locations.parquet", index=False, compression="snappy")
    print(f"[aggregate] Saved bs_locations.parquet ({len(bs_result)} base stations)")

    print("[aggregate] Done.")
