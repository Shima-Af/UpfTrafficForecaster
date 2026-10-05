#!/usr/bin/env python3
"""Generate the forecaster-chapter figures into reports/figures/.

Reads the sweep / comparison CSVs and produces (PDF for the thesis + PNG preview):
  fig_skill_vs_k.{pdf,png}       <- results/cluster_first/sweep/aggregate.csv   (data ready)
  fig_generalization.{pdf,png}   <- results/cluster_first/sweep/aggregate.csv   (data ready)
  fig_model_comparison.{pdf,png} <- results/cluster_first/compare/compare.csv   (after compare run)

Re-run any time; figures present in the CSVs are (re)generated, missing data is skipped.
Usage:  python scripts/plot_forecaster_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs                                                    # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SWEEP = REPO / "results" / "cluster_first" / "sweep" / "aggregate.csv"
COMPARE = REPO / "results" / "cluster_first" / "compare" / "compare.csv"
CHARAC = REPO / "results" / "cluster_first" / "compare" / "charac.csv"   # GWN across K & services
FIGS = fs.FIGS
ERR = dict(ecolor=fs.INK, elinewidth=0.7, capsize=1.5)
_save = fs.save


def fig_skill_vs_k():
    if not CHARAC.exists():
        print("  - skip skill_vs_k (no charac.csv)"); return
    df = pd.read_csv(CHARAC)
    d = df[(df.service == "Netflix") & (df.model == "gwn")].copy()
    if d.empty:
        print("  - skip skill_vs_k (no Netflix rows)"); return
    g = d.groupby("K").agg(
        sp_m=("skill_vs_persistence", "mean"), sp_s=("skill_vs_persistence", "std"),
        ss_m=("skill_vs_seasonal", "mean"),    ss_s=("skill_vs_seasonal", "std"),
        w_m=("test_wape", "mean"),             w_s=("test_wape", "std"),
        lv=("lv_wape", "mean"),                sn=("sn_wape", "mean"),
    ).reset_index().sort_values("K")
    K = g.K.values
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXTWIDTH, 2.6), gridspec_kw={"wspace": 0.3})

    ax = axes[0]                               # skill: flat in K
    ax.errorbar(K, g.ss_m, yerr=g.ss_s.fillna(0), marker="^", color=fs.AQUA, capsize=1.5,
                label="vs. seasonal-naive")
    ax.errorbar(K, g.sp_m, yerr=g.sp_s.fillna(0), marker="o", color=fs.BLUE, capsize=1.5,
                label="vs. persistence")
    ax.set_ylim(0, 0.5); ax.set_ylabel("skill"); ax.legend(loc="upper right", borderaxespad=0.1)
    fs.panel(ax, "a", "skill over the baselines")

    ax = axes[1]                               # raw error: rises for model and baselines alike
    ax.plot(K, g.sn, marker="^", color=fs.AQUA, label="seasonal-naive")
    ax.plot(K, g.lv, marker="s", color=fs.ORANGE, label="persistence")
    ax.errorbar(K, g.w_m, yerr=g.w_s.fillna(0), marker="o", color=fs.BLUE, capsize=1.5,
                label="Graph WaveNet")
    ax.set_ylim(0, None); ax.set_ylabel("test WAPE (%)"); ax.legend(loc="upper left", borderaxespad=0.1)
    fs.panel(ax, "b", "raw error")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xticks(K); ax.set_xticklabels([str(k) for k in K])
        ax.minorticks_off(); ax.set_xlabel("number of clusters $K$")
    _save(fig, "fig_skill_vs_k")


def fig_generalization():
    if not CHARAC.exists():
        print("  - skip generalization (no charac.csv)"); return
    df = pd.read_csv(CHARAC); df = df[df.model == "gwn"]
    services = [s for s in ["total", "Netflix", "DailyMotion"] if s in df.service.unique()]
    Ks = [8, 10, 15]
    sub = df[df.service.isin(services) & df.K.isin(Ks)]
    if sub.empty:
        print("  - skip generalization (no total/DailyMotion rows yet)"); return
    # colour follows the service (as in fig_traffic); the all-service aggregate is neutral
    colors = {"total": fs.INK2, "Netflix": fs.BLUE, "DailyMotion": fs.AQUA}
    names = {"total": "total (aggregate)", "Netflix": "Netflix", "DailyMotion": "DailyMotion (sparse)"}
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXTWIDTH, 2.5), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (col, title), letter in zip(axes, [("skill_vs_persistence", "skill vs. persistence"),
                                               ("skill_vs_seasonal", "skill vs. seasonal-naive")], "ab"):
        w = 0.25
        for i, svc in enumerate(services):
            g = sub[sub.service == svc].groupby("K")[col].agg(["mean", "std"]).reindex(Ks)
            x = np.arange(len(Ks)) + (i - (len(services) - 1) / 2) * (w + 0.015)
            ax.bar(x, g["mean"], w, yerr=g["std"].fillna(0), color=colors[svc], label=names[svc],
                   error_kw=ERR)
        ax.set_xticks(range(len(Ks))); ax.set_xticklabels([f"$K={k}$" for k in Ks])
        ax.grid(axis="x", visible=False); fs.panel(ax, letter, title)
    axes[0].set_ylabel("skill"); axes[0].set_ylim(0, 0.5)
    axes[0].legend(loc="upper left", borderaxespad=0.1, labelspacing=0.25)
    _save(fig, "fig_generalization")


def fig_model_comparison():
    if not COMPARE.exists():
        print("  - skip model_comparison (no compare.csv yet — run scripts/compare_models.py)")
        return
    df = pd.read_csv(COMPARE)
    order = [m for m in ["dlinear", "gru", "stgnn", "agcrn", "gwn", "dcrnn"]   # STGCN dropped
             if m in df.model.unique()]
    Ks = sorted(df.K.unique())
    if df.empty or not order:
        print("  - skip model_comparison (compare.csv empty)"); return
    labels = {"dlinear": "DLinear", "gru": "GRU", "stgcn": "STGCN", "dcrnn": "DCRNN",
              "gwn": "Graph\nWaveNet", "agcrn": "AGCRN", "stgnn": "GAT-GRU"}
    graph_free = {"dlinear", "gru"}
    fig, axes = plt.subplots(1, len(Ks), figsize=(fs.TEXTWIDTH, 2.6), sharey=True, squeeze=False,
                             gridspec_kw={"wspace": 0.08})
    for ax, K, letter in zip(axes[0], Ks, "abcd"):
        g = df[df.K == K].groupby("model")["skill_vs_persistence"].agg(["mean", "std"]).reindex(order)
        x = np.arange(len(order))
        ax.bar(x, g["mean"], 0.66, yerr=g["std"].fillna(0), error_kw=ERR,
               color=[fs.MUTED if m in graph_free else fs.BLUE for m in order])
        ax.set_xticks(x); ax.set_xticklabels([labels[m] for m in order], fontsize=7)
        ax.grid(axis="x", visible=False); fs.panel(ax, letter, f"$K={K}$")
    ax0 = axes[0][0]
    ax0.set_ylabel("skill vs. persistence"); ax0.set_ylim(0, 0.32)
    ax0.legend(handles=[Patch(color=fs.MUTED, label="graph-free"), Patch(color=fs.BLUE, label="graph-based")],
               loc="upper left", ncol=2, borderaxespad=0.1, columnspacing=1.0)
    _save(fig, "fig_model_comparison")


def fig_ablation():
    abl = REPO / "results" / "cluster_first" / "ablation" / "ablation.csv"
    if not (abl.exists() and COMPARE.exists()):
        print("  - skip ablation (need ablation.csv + compare.csv)"); return
    b = pd.read_csv(abl)                                    # Route B (per-BS -> sum)
    a = pd.read_csv(COMPARE); a = a[a.K == 10]              # Route A (cluster-first, direct)
    models = [m for m in ["gru", "stgnn2"] if m in b.model.unique()]
    if not models:
        print("  - skip ablation (no matching models)"); return
    labels = {"gru": "GRU (no graph)", "stgnn2": "GAT (sparse graph)"}
    rA = a.groupby("model")["skill_vs_persistence"].agg(["mean", "std"]).reindex(models)
    rB = b.groupby("model")["skill_vs_persistence"].agg(["mean", "std"]).reindex(models)
    fig, ax = plt.subplots(figsize=(0.62 * fs.TEXTWIDTH, 2.6))
    x = np.arange(len(models)); w = 0.34
    ax.bar(x - w / 2 - 0.01, rA["mean"], w, yerr=rA["std"].fillna(0), error_kw=ERR,
           color=fs.BLUE, label="cluster-first (aggregate, then forecast)")
    ax.bar(x + w / 2 + 0.01, rB["mean"], w, yerr=rB["std"].fillna(0), error_kw=ERR,
           color=fs.ORANGE, label="per-base-station (forecast, then sum)")
    ax.axhline(0, color=fs.INK2, lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([labels[m] for m in models])
    ax.set_ylabel("skill vs. persistence ($K=10$ target)"); ax.set_ylim(-0.13, 0.36)
    ax.grid(axis="x", visible=False); ax.legend(loc="upper right", borderaxespad=0.1, labelspacing=0.25)
    _save(fig, "fig_ablation")


def main():
    fs.apply()
    print(f"writing figures -> {FIGS}")
    fig_skill_vs_k()
    fig_generalization()
    fig_model_comparison()
    fig_ablation()
    return 0


if __name__ == "__main__":
    sys.exit(main())
