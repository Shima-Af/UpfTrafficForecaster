#!/usr/bin/env python3
"""Stage the forecaster artifacts the digital twin / controllers consume.

UPF_NDT's ``traffic_loader`` reads seven files, flat, from
``data/external/traffic_forecaster/`` (see UPF_NDT/configs/paths.yaml). This
repo produces all seven, but scattered across ``results/`` and ``data/``.
This script gathers them for one ``(service, K)`` into a single flat directory,
ready to be DVC-pushed / copied to the twin.

Usage (from repo root):
    python scripts/export_for_twin.py --service Netflix --k 10
    python scripts/export_for_twin.py --service Netflix --k 10 --out exports/traffic_forecaster

Exit code is 0 only when every required artifact was found and copied.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _sources(service: str, k: int) -> dict[str, Path]:
    """Map each twin-expected filename to its source path in this repo."""
    res = REPO_ROOT / "results" / "cluster_first" / service / f"K{k}"
    dat = REPO_ROOT / "data" / "cluster_first" / service / f"K{k}"
    graphs = REPO_ROOT / "data" / "graphs"
    summary = REPO_ROOT / "results" / "cluster_first" / "total" / "forecast_eval_summary.json"
    return {
        "predictions_test.npy":        res / "predictions_test.npy",
        "targets_test.npy":            res / "targets_test.npy",
        "cluster_series.npy":          dat / "cluster_series.npy",
        "cluster_assignments.parquet": dat / "cluster_assignments.parquet",
        "cluster_bs_map.json":         dat / "cluster_bs_map.json",
        "bs_locations.parquet":        graphs / "bs_locations.parquet",
        "forecast_eval_summary.json":  summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--service", required=True, help="e.g. Netflix")
    ap.add_argument("--k", type=int, required=True, help="number of clusters, e.g. 10")
    ap.add_argument(
        "--out", default="exports/traffic_forecaster",
        help="destination directory (default: exports/traffic_forecaster)",
    )
    args = ap.parse_args()

    sources = _sources(args.service, args.k)
    missing = {name: p for name, p in sources.items() if not p.exists()}
    if missing:
        print("ERROR: missing artifacts — run the train/evaluate_forecast stages first:")
        for name, p in missing.items():
            print(f"  - {name}: {p.relative_to(REPO_ROOT)}")
        return 1

    # Guard: the summary must describe the service being exported, since the
    # twin validates summary['service'] == scenario service.
    summary = json.loads(sources["forecast_eval_summary.json"].read_text())
    if summary.get("service") != args.service:
        print(
            f"ERROR: forecast_eval_summary.json service='{summary.get('service')}' "
            f"!= requested '{args.service}'. Re-run evaluate_forecast for this service."
        )
        return 1

    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, src in sources.items():
        shutil.copy2(src, out_dir / name)
        print(f"  ✓ {name:<32} ← {src.relative_to(REPO_ROOT)}")

    (out_dir / "EXPORT_MANIFEST.json").write_text(json.dumps({
        "service": args.service,
        "K": args.k,
        "files": sorted(sources.keys()),
        "consumed_by": "UPF_NDT data/external/traffic_forecaster/",
    }, indent=2))

    print(f"\nExported {len(sources)} artifacts for service={args.service} K={args.k} → {out_dir}")
    print("Next: DVC-push this dir, or copy it to UPF_NDT/data/external/traffic_forecaster/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
