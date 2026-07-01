#!/usr/bin/env python3
"""Layer-A evaluation sweep: STGNN forecast quality vs graph size (K) and traffic type.

Two claims this produces evidence for:
  1. Graph-size robustness — does the STGNN hold its skill from ~10 to ~100 nodes?
     => PRIMARY grid: one service (Netflix), wide K, several seeds.
  2. Traffic-type robustness — is the result Netflix-specific?
     => GENERALIZATION grid: aggregate (total) + sparse (DailyMotion) at candidate K.

It does NOT touch the model/pipeline: for each (service, seed) it writes a temp config
(overriding data.service, training.seed, training.sweep_k), runs `src.train` then
`src.evaluate_forecast`, and archives that run's forecast_eval_summary.json (which already
contains STGNN + last-value/historical-mean/seasonal-naive WAPE per K). Results are
aggregated into a tidy CSV with skill-over-baseline columns.

Outputs (under results/cluster_first/sweep/):
  <service>_seed<seed>.json   — raw per-run summary (archived before the next seed overwrites)
  aggregate.csv               — tidy (service, K, seed, metrics, skill...) rebuilt after every run

Usage:
  python scripts/run_sweep.py            # full sweep (long — run detached, see SWEEP_PLAN.md)
  python scripts/run_sweep.py --smoke    # 1 service / 1 K / 1 seed / 2 epochs — plumbing check
  python scripts/run_sweep.py --services Netflix --k 8 10 --seeds 0   # custom subset
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = REPO_ROOT / "results" / "cluster_first" / "sweep"
TMP_DIR = REPO_ROOT / ".sweep_tmp"
SUMMARY_SRC = REPO_ROOT / "results" / "cluster_first" / "total" / "forecast_eval_summary.json"

# ---- Grids (edit here) ------------------------------------------------------
PRIMARY_SERVICE = "Netflix"
PRIMARY_K = [4, 6, 8, 10, 15, 20, 30, 50, 100]   # "10 to 100 nodes"
GENERALIZATION_SERVICES = ["total", "DailyMotion"]
GENERALIZATION_K = [8, 10, 15]                    # candidate region only
SEEDS = [0, 1, 2]
# ----------------------------------------------------------------------------


def _write_config(service: str, seed: int, k_list: list[int], epochs: int | None) -> Path:
    """Clone config.yaml, override the three sweep fields, return temp path."""
    with open(REPO_ROOT / "config.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("data", {})["service"] = service
    cfg["training"]["seed"] = seed
    cfg["training"]["sweep_k"] = list(k_list)
    if epochs is not None:                      # smoke mode
        cfg["training"]["epochs"] = epochs
        cfg["training"]["early_stopping_patience"] = epochs
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / f"config_{service}_seed{seed}.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path


def _run(stage: str, config_path: Path) -> bool:
    """Run `python -m src.<stage> --config <cfg>` from REPO_ROOT. Returns success."""
    cmd = [sys.executable, "-m", f"src.{stage}", "--config", str(config_path)]
    print(f"    $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        print(f"    !! {stage} FAILED (exit {proc.returncode})", flush=True)
        return False
    return True


def _archive(service: str, seed: int) -> Path | None:
    """Copy this run's summary into a (service, seed)-tagged file, tagging provenance."""
    if not SUMMARY_SRC.exists():
        print(f"    !! no summary produced at {SUMMARY_SRC}", flush=True)
        return None
    data = json.loads(SUMMARY_SRC.read_text())
    data["_sweep_service"] = service
    data["_sweep_seed"] = seed
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    dst = SWEEP_DIR / f"{service}_seed{seed}.json"
    dst.write_text(json.dumps(data, indent=2))
    return dst


def _aggregate() -> Path:
    """Rebuild aggregate.csv from every archived <service>_seed<seed>.json."""
    rows: list[dict] = []
    for jf in sorted(SWEEP_DIR.glob("*_seed*.json")):
        data = json.loads(jf.read_text())
        service = data.get("_sweep_service", data.get("service", "?"))
        seed = data.get("_sweep_seed", "?")
        for r in data.get("results", []):
            wape = r.get("test_wape")
            lv, sn = r.get("lv_wape"), r.get("sn_wape")
            rows.append({
                "service": service,
                "seed": seed,
                "K": r.get("K"),
                "test_wape": wape,
                "test_mae": r.get("test_mae"),
                "test_rmse": r.get("test_rmse"),
                "lv_wape": lv,
                "hm_wape": r.get("hm_wape"),
                "sn_wape": sn,
                # skill = how much WAPE the model removes vs the baseline (higher = better)
                "skill_vs_persistence": round(1 - wape / lv, 4) if wape and lv else None,
                "skill_vs_seasonal": round(1 - wape / sn, 4) if wape and sn else None,
            })
    rows.sort(key=lambda d: (str(d["service"]), d["K"] if d["K"] is not None else 0, d["seed"]))
    out = SWEEP_DIR / "aggregate.csv"
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["service"])
        w.writeheader()
        w.writerows(rows)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="fast plumbing check: Netflix / K=4 / seed=0 / 2 epochs")
    ap.add_argument("--services", nargs="+", help="override service list")
    ap.add_argument("--k", nargs="+", type=int, help="override K list (applied to all services)")
    ap.add_argument("--seeds", nargs="+", type=int, help="override seed list")
    args = ap.parse_args()

    # Build the (service -> K list) plan and seed list.
    if args.smoke:
        plan = {"Netflix": [4]}
        seeds = [0]
        epochs = 2
    else:
        plan = {PRIMARY_SERVICE: PRIMARY_K}
        for s in GENERALIZATION_SERVICES:
            plan[s] = GENERALIZATION_K
        seeds = SEEDS
        epochs = None
    if args.services:
        k_list = args.k or PRIMARY_K
        plan = {s: (args.k or k_list) for s in args.services}
    elif args.k:
        plan = {s: args.k for s in plan}
    if args.seeds:
        seeds = args.seeds

    jobs = [(s, seed, klist) for s, klist in plan.items() for seed in seeds]
    print(f"=== sweep: {len(jobs)} (service,seed) runs ===", flush=True)
    for s, klist in plan.items():
        print(f"  {s}: K={klist}  seeds={seeds}", flush=True)

    t0 = time.time()
    ok, fail = 0, 0
    for i, (service, seed, klist) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] service={service} seed={seed} K={klist}  "
              f"(elapsed {time.time()-t0:.0f}s)", flush=True)
        cfg = _write_config(service, seed, klist, epochs)
        if not _run("train", cfg):
            fail += 1
            continue
        if not _run("evaluate_forecast", cfg):
            fail += 1
            continue
        if _archive(service, seed):
            ok += 1
            agg = _aggregate()              # rebuild after every run so partial results survive
            print(f"    ✓ archived; aggregate → {agg.relative_to(REPO_ROOT)}", flush=True)

    print(f"\n=== done: {ok} ok, {fail} failed, {time.time()-t0:.0f}s ===", flush=True)
    if (SWEEP_DIR / "aggregate.csv").exists():
        print(f"Results: {(SWEEP_DIR / 'aggregate.csv').relative_to(REPO_ROOT)}", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
