"""
preprocess.py
-------------
DVC pipeline stage: raw NetMob tiles → BS-level normalised Parquet.

Pipeline
--------
1. Load Voronoi map (output of build_graph stage).
2. Pass 1 — find the global max BS-level traffic value across ALL services, days, BSs.
3. Pass 2 — for each day and service:
   a. Parse raw tile-level .txt file.
   b. Aggregate tiles to BS level (sum over each Voronoi cell).
   c. Normalise by the single global_max.
4. Write one Parquet per day containing all services.

Input layout
------------
data/netmob/raw/{city}/{service}/{YYYYMMDD}/{city}_{service}_{YYYYMMDD}_DL.txt

Output layout
-------------
data/netmob/processed/
    {city}_{YYYYMMDD}.parquet  — one file per day
    manifest.csv               — summary per processed day
    metadata.json              — single global_max scale factor

Each parquet schema
-------------------
timestamp  : datetime64[ns]   (local Paris naive time, 15-min resolution)
site_id    : str              (gNodeB identifier from Cartoradio)
service    : str              (DailyMotion | Netflix | YouTube)
dl_norm    : float32          (dimensionless ∈ [0,1], scaled by global max across all services)

Normalisation
-------------
dl_norm = bs_raw_sum / global_max
where bs_raw_sum is the sum of raw tile values within the Voronoi cell of each BS,
and global_max is the dataset-wide maximum across ALL services, BSs, and timestamps.
Using a single normaliser means services are on a common scale and their dl_norm
values can be summed or compared directly without any byte-conversion step.

DST handling (20190331)
-----------------------
File has 92 slots; 4 missing slots (02:00–02:45 local) are filled with 0.

CLI usage
---------
python -m src.preprocess [raw_dir] [out_dir] [graphs_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.netmob_loader import (
    _DST_DATE,
    _DST_MISSING_SLOTS,
    _SLOTS_NORMAL,
    _SLOTS_DST,
    _scale_traffic,
    _detect_services,
    _make_timestamps,
    parse_file,
)


# ---------------------------------------------------------------------------
# Aggregate raw tile traffic to BS level
# ---------------------------------------------------------------------------

def _aggregate_to_bs(
    cell_ids: np.ndarray,
    traffic: np.ndarray,
    vm_series: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sum tile-level raw traffic into BS-level traffic using Voronoi map.

    Parameters
    ----------
    cell_ids  : int64 array (n_cells,)
    traffic   : float32 array (n_cells, n_slots)
    vm_series : Series indexed by tile_id, values = site_id

    Returns
    -------
    site_ids   : object array of site_id strings (n_bs,)
    bs_traffic : float32 array (n_bs, n_slots)
    """
    valid_mask = np.isin(cell_ids, vm_series.index)
    cell_ids = cell_ids[valid_mask]
    traffic  = traffic[valid_mask]

    if len(cell_ids) == 0:
        return np.array([]), np.zeros((0, traffic.shape[1]), dtype=np.float32)

    assigned = vm_series.reindex(cell_ids).values   # site_id per tile

    # pandas groupby sum — efficient and clean
    df = pd.DataFrame(traffic.astype(np.float32))
    df["site_id"] = assigned
    agg = df.groupby("site_id", sort=True).sum()

    return agg.index.values, agg.values.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-service single-day parser (tile level, before aggregation)
# ---------------------------------------------------------------------------

def _parse_service_raw(
    city_dir: Path,
    date_str: str,
    service: str,
    direction: str,
    cells_subset: list[int] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Parse a single service file for one day.

    Returns (cell_ids, traffic) with raw tile values, or None if file absent.
    traffic shape: (n_cells, 96)
    """
    day_dir = city_dir / service / date_str
    if not day_dir.exists():
        return None
    txt_files = list(day_dir.glob(f"*_{direction}.txt"))
    if not txt_files:
        return None

    cell_ids, traffic = parse_file(txt_files[0])

    # DST day: 92 slots → pad to 96 with zeros at missing positions
    if date_str == _DST_DATE and traffic.shape[1] == _SLOTS_DST:
        padded = np.zeros((traffic.shape[0], _SLOTS_NORMAL), dtype=np.float32)
        real_cols = [i for i in range(_SLOTS_NORMAL) if i not in _DST_MISSING_SLOTS]
        padded[:, real_cols] = traffic
        traffic = padded

    if cells_subset is not None:
        mask = np.isin(cell_ids, cells_subset)
        cell_ids = cell_ids[mask]
        traffic  = traffic[mask]

    return cell_ids, traffic


# ---------------------------------------------------------------------------
# Build day DataFrame from BS-level traffic
# ---------------------------------------------------------------------------

def _build_day_df(
    site_ids: np.ndarray,
    traffic: np.ndarray,
    date_str: str,
    scale: float,
    fill_missing: str,
    service: str,
) -> pd.DataFrame:
    """
    Convert (site_ids, BS-level raw traffic) to a normalised long-format DataFrame.

    Columns: timestamp, site_id, service, dl_norm
    """
    dl_norm    = _scale_traffic(traffic, scale)      # (n_bs, 96)
    timestamps = _make_timestamps(date_str)          # 96 entries, None at DST gaps

    valid_slots     = [(i, ts) for i, ts in enumerate(timestamps) if ts is not None]
    slot_indices    = np.array([i  for i, _  in valid_slots], dtype=np.int32)
    slot_timestamps = [ts          for _, ts in valid_slots]

    n_bs    = len(site_ids)
    n_valid = len(valid_slots)

    sid_col = np.tile(site_ids, n_valid)                  # (n_valid * n_bs,)
    ts_col  = np.repeat(slot_timestamps, n_bs)            # (n_valid * n_bs,)
    dl_col  = dl_norm[:, slot_indices].T.reshape(-1)      # (n_valid * n_bs,)

    df = pd.DataFrame({
        "timestamp": ts_col,
        "site_id":   sid_col,
        "service":   service,
        "dl_norm":   dl_col.astype(np.float32),
    })

    if fill_missing == "ffill" and date_str != _DST_DATE:
        df["dl_norm"] = (
            df.sort_values("timestamp")
              .groupby("site_id")["dl_norm"]
              .transform(lambda x: x.ffill())
        )

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(raw_dir: Path, out_dir: Path, graphs_dir: Path, params: dict) -> None:
    """
    Two-pass BS-level preprocessing pipeline:
      Pass 1 — aggregate tiles→BS for every day/service to compute scale factors.
      Pass 2 — aggregate, normalise, and write one parquet per day.
    """
    pp              = params["preprocess"]
    fill_missing    = pp.get("fill_missing", "ffill")
    cells_subset    = pp.get("cells_subset")
    norm_method     = pp.get("normalization", "global_max")
    norm_percentile = pp.get("normalization_percentile", 99.9)

    # Load Voronoi map produced by build_graph
    vm_path = graphs_dir / "voronoi_map.parquet"
    vm_df   = pd.read_parquet(vm_path)
    vm_series = vm_df.set_index("tile_id")["site_id"]
    print(f"[preprocess] Voronoi map: {len(vm_df):,} tiles -> "
          f"{vm_df['site_id'].nunique()} BSs")

    city_dirs = [d for d in sorted(raw_dir.iterdir()) if d.is_dir()]
    if not city_dirs:
        raise FileNotFoundError(f"No city directories found under {raw_dir}")

    # ------------------------------------------------------------------
    # Pass 1: compute single global_max across ALL services
    # ------------------------------------------------------------------
    print("[preprocess] Pass 1 — computing global max across all services and days...")
    all_bs_maxes: list[float] = []

    for city_dir in city_dirs:
        services  = _detect_services(city_dir)
        all_dates = _collect_dates(city_dir, services)
        for svc in services:
            for date_str in sorted(all_dates):
                result = _parse_service_raw(
                    city_dir, date_str, svc, "DL", cells_subset
                )
                if result is None:
                    continue
                cell_ids, traffic = result
                _, bs_traffic = _aggregate_to_bs(cell_ids, traffic, vm_series)
                if len(bs_traffic) > 0:
                    all_bs_maxes.append(float(bs_traffic.max()))

    if not all_bs_maxes:
        raise RuntimeError("No data found — cannot compute global max")

    if norm_method == "global_max":
        global_scale = float(max(all_bs_maxes))
    elif norm_method == "percentile":
        global_scale = float(np.percentile(all_bs_maxes, norm_percentile))
    else:
        raise ValueError(f"Unknown normalization method: {norm_method}")

    print(f"[preprocess] global_max ({norm_method}): {global_scale:.4f}")

    # Save metadata — single scale factor shared by all services
    metadata = {
        "normalization":  norm_method,
        "global_max":     global_scale,
        "aggregation_level": "BS (Voronoi sum)",
        "note": (
            "dl_norm = bs_tile_sum / global_max (single scale across all services). "
            "bs_tile_sum = sum of raw tile values in the BS Voronoi cell. "
            "Services share the same normaliser so their dl_norm values are "
            "directly comparable and summable without byte conversion."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------------
    # Pass 2: aggregate, normalise and write parquets
    # ------------------------------------------------------------------
    print("[preprocess] Pass 2 — aggregating to BS, normalising, writing parquets...")
    manifest_rows = []

    for city_dir in city_dirs:
        city      = city_dir.name
        services  = _detect_services(city_dir)
        all_dates = _collect_dates(city_dir, services)
        print(f"  City={city}  Services={services}  Days={len(all_dates)}")

        for date_str in sorted(all_dates):
            out_path = out_dir / f"{city}_{date_str}.parquet"
            if out_path.exists():
                continue

            day_dfs: list[pd.DataFrame] = []
            for svc in services:
                result = _parse_service_raw(
                    city_dir, date_str, svc, "DL", cells_subset
                )
                if result is None:
                    continue
                cell_ids, traffic = result
                site_ids, bs_traffic = _aggregate_to_bs(cell_ids, traffic, vm_series)
                if len(site_ids) == 0:
                    continue
                day_dfs.append(
                    _build_day_df(
                        site_ids, bs_traffic, date_str, global_scale, fill_missing, svc
                    )
                )

            if not day_dfs:
                print(f"  [warn] {date_str}: no data for any service, skipping")
                continue

            df = pd.concat(day_dfs, ignore_index=True)

            n_bs    = df["site_id"].nunique()
            n_slots = df["timestamp"].nunique()
            is_dst  = date_str == _DST_DATE

            df.to_parquet(out_path, index=False, compression="snappy")
            print(f"  [{date_str}] BSs={n_bs}  slots={n_slots}"
                  + ("  [DST]" if is_dst else ""))

            manifest_rows.append({
                "city":     city,
                "date":     date_str,
                "n_bs":     n_bs,
                "n_slots":  n_slots,
                "dst_day":  is_dst,
                "services": ",".join(services),
                "out_file": out_path.name,
            })

    pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)
    print(f"\n[preprocess] Done. {len(manifest_rows)} days -> {out_dir}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _collect_dates(city_dir: Path, services: list[str]) -> set[str]:
    dates: set[str] = set()
    for svc in services:
        svc_dir = city_dir / svc
        if svc_dir.exists():
            dates.update(
                d.name for d in svc_dir.iterdir()
                if d.is_dir() and len(d.name) == 8 and d.name.isdigit()
            )
    return dates


if __name__ == "__main__":
    with open("params.yaml", encoding="utf-8") as _fh:
        _params = yaml.safe_load(_fh)

    _raw_dir    = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/netmob/raw")
    _out_dir    = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/netmob/processed")
    _graphs_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/graphs")
    _out_dir.mkdir(parents=True, exist_ok=True)

    run(_raw_dir, _out_dir, _graphs_dir, _params)
