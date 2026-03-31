"""
netmob_loader.py
----------------
Parse raw NetMob 2023 traffic files.

File format (per .txt file)
---------------------------
- Space-separated, no header
- One row per tile (cell): tile_id  v_0  v_1 ... v_95
- Normal day  : 96 time slots (15-min bins, 00:00–23:45 local Paris time)
- DST day     : 92 slots (20190331 — clock jumps 02:00→03:00; slots 8–11 absent)
- Values      : privacy-preserving aggregate traffic indicators — dimensionless,
                normalised across the full NetMob 2023 dataset.
                No physical unit (not kbits, not bytes).
                Do NOT apply any rate or capacity conversion to these values.

Directory layout expected
--------------------------
raw_dir/{city}/{service}/{YYYYMMDD}/{city}_{service}_{YYYYMMDD}_{direction}.txt

Public API
----------
parse_file(path)
    → cell_ids (np.ndarray int64, shape n_cells)
    + traffic  (np.ndarray float32, shape n_cells × n_slots)

load_day(date_dir, services, direction="DL")
    → cell_ids, traffic_sum  (services summed, n_cells × 96)

load_traces(raw_dir, params) -> pd.DataFrame
    Columns: timestamp (local Paris naive), cell_id, dl_raw
    Raw (unnormalised) values. Normalisation is applied in src/preprocess.py.
    For use in notebooks / small subsets only.
    For full preprocessing use src/preprocess.py (DVC stage).
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

# March 31 2019 — DST day in France (92 slots, missing 02:00–02:45)
_DST_DATE = "20190331"
_DST_MISSING_SLOTS = (8, 9, 10, 11)   # indices in the full 96-slot array
_SLOTS_NORMAL = 96
_SLOTS_DST = 92


def parse_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse one NetMob .txt file.

    Returns
    -------
    cell_ids : int64 array, shape (n_cells,)
    traffic  : float32 array, shape (n_cells, n_slots)
               Values in kbits per 15-min slot.
    """
    path = Path(path)
    raw = np.loadtxt(path, dtype=np.float64)   # shape (n_cells, 1 + n_slots)
    cell_ids = raw[:, 0].astype(np.int64)
    traffic = raw[:, 1:].astype(np.float32)
    return cell_ids, traffic


def _make_timestamps(date_str: str) -> list[pd.Timestamp]:
    """
    Return list of 96 naive (local Paris time) Timestamps for date_str.
    For the DST day (20190331) the 4 missing slots get NaT — callers handle fill.
    """
    d = pd.Timestamp(date_str)
    if date_str == _DST_DATE:
        # Build the 92 real timestamps (02:00–02:45 don't exist in local time)
        # Slot 0–7: 00:00–01:45, slot 8–91: 03:00–23:45
        pre  = [d + pd.Timedelta(minutes=15 * i) for i in range(8)]
        post = [d + pd.Timedelta(hours=3, minutes=15 * i) for i in range(84)]
        return pre + [None] * 4 + post    # 96 entries, 4 None gaps
    else:
        return [d + pd.Timedelta(minutes=15 * i) for i in range(96)]


def load_day(
    date_dir: str | Path,
    services: Sequence[str],
    direction: str = "DL",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and sum traffic across services for one city/date directory.

    Parameters
    ----------
    date_dir  : path like raw/{city}/{service}/{YYYYMMDD}/ — parent of service dirs
                OR path like raw/{city}/  (then services subdirs are explored)
    services  : list of service names, e.g. ["Netflix", "YouTube", "DailyMotion"]
    direction : "DL" or "UL"

    Returns
    -------
    cell_ids : int64 (n_cells,)
    traffic  : float32 (n_cells, 96)  — DST day padded with 0 at missing slots
    """
    date_dir = Path(date_dir)
    # Infer date from path (last directory component that looks like YYYYMMDD)
    date_str = date_dir.name if re.match(r"\d{8}$", date_dir.name) else None

    traffic_sum: np.ndarray | None = None
    cell_ids: np.ndarray | None = None

    for svc in services:
        # Search up the tree for a matching file
        # Expected: date_dir is raw/{city}/{service}/{YYYYMMDD}/
        txt_files = list(date_dir.glob(f"*_{direction}.txt"))
        if not txt_files:
            raise FileNotFoundError(
                f"No {direction} file found in {date_dir} for service {svc}"
            )
        ids, traf = parse_file(txt_files[0])

        # Pad DST day: insert zeros at missing slot positions
        if date_str == _DST_DATE and traf.shape[1] == _SLOTS_DST:
            padded = np.zeros((traf.shape[0], _SLOTS_NORMAL), dtype=np.float32)
            dst_cols = [i for i in range(_SLOTS_NORMAL) if i not in _DST_MISSING_SLOTS]
            padded[:, dst_cols] = traf
            traf = padded

        if cell_ids is None:
            cell_ids = ids
            traffic_sum = traf.copy()
        else:
            if not np.array_equal(cell_ids, ids):
                raise ValueError(
                    f"Cell ID mismatch between services for {date_dir}"
                )
            traffic_sum += traf

    return cell_ids, traffic_sum


def load_traces(
    raw_dir: str | Path,
    params: dict | str | Path = "params.yaml",
) -> pd.DataFrame:
    """
    Load all raw files and return a long-format DataFrame with raw values.

    Parameters
    ----------
    raw_dir : root of raw data, e.g. data/netmob/raw/Lyon
    params  : params dict or path to params.yaml

    Returns
    -------
    DataFrame with columns: timestamp, cell_id, dl_raw
    dl_raw contains the raw dimensionless NetMob values (no unit conversion).
    To get dl_norm ∈ [0,1] divide by the global scale factor from
    data/netmob/processed/metadata.json (computed by src/preprocess.py).
    Sorted by (timestamp, cell_id).

    Note: For large datasets use src/preprocess.py (DVC stage) instead.
    """
    if not isinstance(params, dict):
        with open(params, encoding="utf-8") as f:
            params = yaml.safe_load(f)

    raw_dir = Path(raw_dir)
    pp = params["preprocess"]
    services = _detect_services(raw_dir)
    cells_subset = pp.get("cells_subset")

    records = []
    for svc_dir in sorted(raw_dir.iterdir()):
        if not svc_dir.is_dir() or svc_dir.name not in services:
            continue
        for day_dir in sorted(svc_dir.iterdir()):
            if not day_dir.is_dir():
                continue
            date_str = day_dir.name
            cell_ids, traffic = load_day(day_dir, [svc_dir.name])

            if cells_subset is not None:
                mask = np.isin(cell_ids, cells_subset)
                cell_ids = cell_ids[mask]
                traffic = traffic[mask]

            timestamps = _make_timestamps(date_str)

            for slot_idx, ts in enumerate(timestamps):
                if ts is None:
                    continue
                for ci, cid in enumerate(cell_ids):
                    records.append({
                        "timestamp": ts,
                        "cell_id":   int(cid),
                        "dl_raw":    float(traffic[ci, slot_idx]),
                    })

    df = pd.DataFrame(records)
    df.sort_values(["timestamp", "cell_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scale_traffic(traffic: np.ndarray, scale: float) -> np.ndarray:
    """
    Divide raw NetMob values by a global scale factor to produce dl_norm ∈ [0, 1].

    The scale factor is computed by src/preprocess.py (e.g. dataset-wide max or
    99th-percentile) and stored in data/netmob/processed/metadata.json.
    Raw values have no physical unit — they are privacy-preserving aggregate
    indicators normalised across the full NetMob 2023 dataset.
    """
    return (traffic / scale).astype(np.float32)


def _detect_services(city_dir: Path) -> list[str]:
    """Return list of service subdirectories found under city_dir."""
    return [d.name for d in sorted(city_dir.iterdir()) if d.is_dir()]
