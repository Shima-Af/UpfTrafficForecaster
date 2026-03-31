"""
preprocess.py
-------------
DVC pipeline stage: raw NetMob → cleaned, normalised Parquet.

Input layout
------------
data/netmob/raw/{city}/{service}/{YYYYMMDD}/{city}_{service}_{YYYYMMDD}_DL.txt

Output layout
-------------
data/netmob/processed/
    {city}_{YYYYMMDD}.parquet  — one file per day
    manifest.csv               — summary per processed day
    metadata.json              — global scale factor used for dl_norm

Each parquet schema
-------------------
timestamp  : datetime64[ns]   (local Paris naive time, 15-min resolution)
cell_id    : int64
dl_norm    : float32           (dimensionless ∈ [0, 1], scaled by global max)
ul_norm    : float32           (0.0 — DL-only dataset)

Normalisation
-------------
Raw NetMob values are privacy-preserving aggregate indicators with no physical
unit. dl_norm = raw_value / global_scale, where global_scale is the dataset-wide
maximum (or configurable percentile) computed in a first pass over all files.
The scale factor is saved to metadata.json for reproducibility.

DST handling (20190331)
-----------------------
File has 92 slots; 4 missing slots (02:00–02:45 local) are filled with 0.

CLI usage
---------
python -m src.preprocess [raw_dir] [out_dir]
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
# Per-day processing  (returns raw values — normalisation applied in run())
# ---------------------------------------------------------------------------

def _parse_day_raw(
    city_dir: Path,
    date_str: str,
    services: list[str],
    direction: str,
    cells_subset: list[int] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Parse and sum service files for one day.

    Returns (cell_ids, traffic_sum) with raw values, or None if no files found.
    traffic_sum shape: (n_cells, 96)
    """
    traffic_sum: np.ndarray | None = None
    cell_ids_ref: np.ndarray | None = None

    for svc in services:
        day_dir = city_dir / svc / date_str
        if not day_dir.exists():
            continue
        txt_files = list(day_dir.glob(f"*_{direction}.txt"))
        if not txt_files:
            continue

        cell_ids, traffic = parse_file(txt_files[0])

        # DST day: 92 slots → pad to 96 with zeros at missing positions
        if date_str == _DST_DATE and traffic.shape[1] == _SLOTS_DST:
            padded = np.zeros((traffic.shape[0], _SLOTS_NORMAL), dtype=np.float32)
            real_cols = [i for i in range(_SLOTS_NORMAL) if i not in _DST_MISSING_SLOTS]
            padded[:, real_cols] = traffic
            traffic = padded

        if traffic_sum is None:
            cell_ids_ref = cell_ids
            traffic_sum = traffic.copy()
        else:
            if not np.array_equal(cell_ids_ref, cell_ids):
                raise ValueError(
                    f"[{date_str}] Cell ID mismatch across services"
                )
            traffic_sum += traffic

    if traffic_sum is None:
        return None

    if cells_subset is not None:
        mask = np.isin(cell_ids_ref, cells_subset)
        cell_ids_ref = cell_ids_ref[mask]
        traffic_sum  = traffic_sum[mask]

    return cell_ids_ref, traffic_sum


def _build_day_df(
    cell_ids: np.ndarray,
    traffic: np.ndarray,
    date_str: str,
    scale: float,
    fill_missing: str,
) -> pd.DataFrame:
    """
    Convert (cell_ids, raw traffic array) to a normalised long-format DataFrame.

    Columns: timestamp, cell_id, dl_norm, ul_norm
    """
    dl_norm    = _scale_traffic(traffic, scale)          # (n_cells, 96)
    timestamps = _make_timestamps(date_str)              # 96 entries, None at DST gaps

    valid_slots    = [(i, ts) for i, ts in enumerate(timestamps) if ts is not None]
    slot_indices   = np.array([i  for i, _  in valid_slots], dtype=np.int32)
    slot_timestamps = [ts          for _, ts in valid_slots]

    n_cells = len(cell_ids)
    n_valid = len(valid_slots)

    cid_col = np.tile(cell_ids, n_valid)                 # (n_valid * n_cells,)
    ts_col  = np.repeat(slot_timestamps, n_cells)        # (n_valid * n_cells,)
    dl_col  = dl_norm[:, slot_indices].T.reshape(-1)     # (n_valid * n_cells,)

    df = pd.DataFrame({
        "timestamp": ts_col,
        "cell_id":   cid_col,
        "dl_norm":   dl_col.astype(np.float32),
        "ul_norm":   np.zeros(len(cid_col), dtype=np.float32),
    })

    # Forward-fill any NaN cells on normal days (DST zeros are left as-is)
    if fill_missing == "ffill" and date_str != _DST_DATE:
        df["dl_norm"] = (
            df.sort_values("timestamp")
              .groupby("cell_id")["dl_norm"]
              .transform(lambda x: x.ffill())
        )

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(raw_dir: Path, out_dir: Path, params: dict) -> None:
    """
    Two-pass preprocessing pipeline:
      Pass 1 — scan all raw files to compute the global scale factor.
      Pass 2 — parse, normalise, and write one parquet per day.
    """
    pp    = params["preprocess"]
    fill_missing   = pp.get("fill_missing", "ffill")
    cells_subset   = pp.get("cells_subset")
    norm_method    = pp.get("normalization", "global_max")
    norm_percentile = pp.get("normalization_percentile", 99.9)

    city_dirs = [d for d in sorted(raw_dir.iterdir()) if d.is_dir()]
    if not city_dirs:
        raise FileNotFoundError(f"No city directories found under {raw_dir}")

    # ------------------------------------------------------------------
    # Pass 1: compute global scale
    # ------------------------------------------------------------------
    print("[preprocess] Pass 1 — computing global scale factor...")
    all_maxes: list[float] = []

    for city_dir in city_dirs:
        services = _detect_services(city_dir)
        all_dates = _collect_dates(city_dir, services)
        for date_str in sorted(all_dates):
            result = _parse_day_raw(city_dir, date_str, services, "DL", cells_subset)
            if result is not None:
                _, traffic = result
                all_maxes.append(float(traffic.max()))

    if not all_maxes:
        raise RuntimeError("No data found — cannot compute scale factor")

    if norm_method == "global_max":
        scale = float(max(all_maxes))
    elif norm_method == "percentile":
        scale = float(np.percentile(all_maxes, norm_percentile))
    else:
        raise ValueError(f"Unknown normalization method: {norm_method}")

    print(f"[preprocess] Scale factor ({norm_method}): {scale:.2f}")

    # Save metadata
    metadata = {
        "normalization":            norm_method,
        "normalization_percentile": norm_percentile,
        "scale_factor":             scale,
        "note": (
            "dl_norm = raw_value / scale_factor. "
            "Raw NetMob values are dimensionless privacy-preserving aggregates."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------------
    # Pass 2: normalise and write parquets
    # ------------------------------------------------------------------
    print("[preprocess] Pass 2 — normalising and writing parquets...")
    manifest_rows = []

    for city_dir in city_dirs:
        city     = city_dir.name
        services = _detect_services(city_dir)
        all_dates = _collect_dates(city_dir, services)
        print(f"  City={city}  Services={services}  Days={len(all_dates)}")

        for date_str in sorted(all_dates):
            out_path = out_dir / f"{city}_{date_str}.parquet"
            if out_path.exists():
                continue

            result = _parse_day_raw(city_dir, date_str, services, "DL", cells_subset)
            if result is None:
                print(f"  [warn] {date_str}: no files, skipping")
                continue

            cell_ids, traffic = result
            df = _build_day_df(cell_ids, traffic, date_str, scale, fill_missing)

            n_cells = df["cell_id"].nunique()
            n_slots = df["timestamp"].nunique()
            is_dst  = date_str == _DST_DATE

            df.to_parquet(out_path, index=False, compression="snappy")
            print(f"  [{date_str}] cells={n_cells}  slots={n_slots}"
                  + ("  [DST]" if is_dst else ""))

            manifest_rows.append({
                "city":     city,
                "date":     date_str,
                "n_cells":  n_cells,
                "n_slots":  n_slots,
                "dst_day":  is_dst,
                "services": ",".join(services),
                "out_file": out_path.name,
            })

    pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)
    print(f"\n[preprocess] Done. {len(manifest_rows)} days → {out_dir}")


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
    with open("params.yaml", encoding="utf-8") as f:
        _params = yaml.safe_load(f)

    _raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/netmob/raw")
    _out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/netmob/processed")
    _out_dir.mkdir(parents=True, exist_ok=True)

    run(_raw_dir, _out_dir, _params)
