# Layer-A Evaluation Sweep — STGNN forecast quality vs graph size & traffic type

Driver: [`scripts/run_sweep.py`](scripts/run_sweep.py). Edit the grids at the top of that file.

## What it establishes

1. **Graph-size robustness (primary claim).** Does the STGNN hold its skill as the graph grows
   from ~10 to ~100 nodes? K is a *model* axis here, not a deployment choice.
2. **Traffic-type robustness (generalization claim).** Is the result Netflix-specific, or does it
   hold for aggregate (`total`) and sparse (`DailyMotion`) traffic on the same geographic clusters?

## Grids

| | service(s) | K | seeds | runs |
|---|---|---|---|---|
| **Primary** | Netflix | 4, 6, 8, 10, 15, 20, 30, 50, 100 | 0,1,2 | 27 (K,seed) |
| **Generalization** | total, DailyMotion | 8, 10, 15 | 0,1,2 | 18 (K,seed) |

Each `(service, seed)` is one `train` + one `evaluate_forecast` call that sweeps its K list
internally (data is loaded once per call and reused across K). 15 train calls total.

## Headline metric: skill over baselines — NOT raw WAPE

Raw WAPE **rises** with K because smaller clusters aggregate fewer base stations → less spatial
averaging → burstier, intrinsically harder targets. That is a property of the signal, not the
model. To show the model itself scales, read **skill = 1 − WAPE_STGNN / WAPE_baseline** (vs
persistence and vs seasonal-naive) at each K. If skill stays roughly flat across K, the STGNN
holds its margin while the problem degrades around it — that is the graph-size-robustness result.
Both columns are in `aggregate.csv`.

## Run it (survives closing the IDE / SSH)

```bash
cd /home/ubuntu/UPF_Forecasting/UpfTrafficForecaster

# 0) plumbing check (~1 min): Netflix / K=4 / seed=0 / 2 epochs
./.venv/bin/python scripts/run_sweep.py --smoke

# 1) full sweep, detached — keeps running after the terminal/IDE closes
nohup ./.venv/bin/python scripts/run_sweep.py > sweep.out 2>&1 & disown
echo $! > sweep.pid
```

`nohup` ignores the hang-up signal sent when the terminal closes; `disown` detaches it from the
shell. The process keeps running and writes to `sweep.out`.

### Monitor / control
```bash
tail -f sweep.out                               # live progress log
cat results/cluster_first/sweep/aggregate.csv   # partial results (rebuilt after every run)
ps -p "$(cat sweep.pid)" >/dev/null && echo running || echo done   # is it alive?
kill "$(cat sweep.pid)"                          # stop early (partial results are kept)
```

### Custom subsets
```bash
./.venv/bin/python scripts/run_sweep.py --services Netflix --k 8 10 12 --seeds 0
```

## Outputs (`results/cluster_first/sweep/`)

- `<service>_seed<seed>.json` — raw per-run summary (STGNN + 3 baselines per K).
- `aggregate.csv` — tidy `(service, seed, K, test_wape, test_mae, test_rmse, lv_wape, hm_wape,
  sn_wape, skill_vs_persistence, skill_vs_seasonal)`, rebuilt after every run.

## Notes

- Per-K checkpoints/outputs under `data|results/cluster_first/<service>/K<K>/` are **not**
  seed-tagged and are overwritten between seeds — intentional; the metrics are archived per seed
  before that happens. Only the model weights are transient, not the recorded results.
- Runtime is dominated by the large-K Netflix runs (K=50, 100). Expect a few hours total; the
  detached run is the right call. If you want a fast first look, scout the K curve at one seed,
  then add seeds only around the interesting region:
  `./.venv/bin/python scripts/run_sweep.py --services Netflix --seeds 0`.
