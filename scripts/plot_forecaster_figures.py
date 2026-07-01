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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SWEEP = REPO / "results" / "cluster_first" / "sweep" / "aggregate.csv"
COMPARE = REPO / "results" / "cluster_first" / "compare" / "compare.csv"
CHARAC = REPO / "results" / "cluster_first" / "compare" / "charac.csv"   # GWN across K & services
FIGS = REPO / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    print(f"  ✓ {name}.pdf / .png")


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
    ).reset_index().sort_values("K")
    K = g.K.values
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.errorbar(K, g.sp_m, yerr=g.sp_s.fillna(0), marker="o", lw=1.8,
                color="#1b7837", label="skill vs. persistence", capsize=3)
    ax.errorbar(K, g.ss_m, yerr=g.ss_s.fillna(0), marker="s", lw=1.8,
                color="#2166ac", label="skill vs. seasonal-naive", capsize=3)
    ax.set_xscale("log"); ax.set_xticks(K); ax.set_xticklabels([str(k) for k in K])
    ax.set_xlabel("number of clusters  $K$  (log scale)")
    ax.set_ylabel("skill  $=1-\\mathrm{WAPE}_\\mathrm{model}/\\mathrm{WAPE}_\\mathrm{base}$")
    ax.set_ylim(0, max(0.5, g.ss_m.max() * 1.2))
    ax.annotate("model skill preserved", (K[len(K)//2], g.sp_m.iloc[len(K)//2]),
                textcoords="offset points", xytext=(0, -28), color="#1b7837", fontsize=8)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(K, g.w_m, marker="^", ls="--", color="#b2182b", label="raw test WAPE")
    ax2.set_ylabel("raw test WAPE (%)", color="#b2182b")
    ax2.tick_params(axis="y", labelcolor="#b2182b")
    ax2.annotate("intrinsic difficulty $\\uparrow$", (K[-3], g.w_m.iloc[-3]),
                 textcoords="offset points", xytext=(-70, 6), color="#b2182b", fontsize=8)
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="upper center", fontsize=8, ncol=1, framealpha=0.9)
    ax.set_title("Graph WaveNet skill vs. graph size (Netflix)")
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
    colors = {"total": "#1b7837", "Netflix": "#2166ac", "DailyMotion": "#b2182b"}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, (col, title) in zip(axes, [("skill_vs_persistence", "vs. persistence"),
                                       ("skill_vs_seasonal", "vs. seasonal-naive")]):
        w = 0.25
        for i, svc in enumerate(services):
            g = sub[sub.service == svc].groupby("K")[col].agg(["mean", "std"]).reindex(Ks)
            x = np.arange(len(Ks)) + (i - (len(services) - 1) / 2) * w
            ax.bar(x, g["mean"], w, yerr=g["std"].fillna(0), capsize=3,
                   color=colors.get(svc, None), label=svc)
        ax.set_xticks(range(len(Ks))); ax.set_xticklabels([f"K={k}" for k in Ks])
        ax.set_title(title); ax.axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("skill over baseline")
    axes[0].legend(fontsize=8, title="service")
    fig.suptitle("Generalization across traffic types")
    _save(fig, "fig_generalization")


def fig_model_comparison():
    if not COMPARE.exists():
        print("  - skip model_comparison (no compare.csv yet — run scripts/compare_models.py)")
        return
    df = pd.read_csv(COMPARE)
    order = [m for m in ["dlinear", "gru", "dcrnn", "gwn", "agcrn", "stgnn"]   # STGCN dropped
             if m in df.model.unique()]
    Ks = sorted(df.K.unique())
    if df.empty or not order:
        print("  - skip model_comparison (compare.csv empty)"); return
    labels = {"dlinear": "DLinear", "gru": "GRU\n(no graph)", "stgcn": "STGCN", "dcrnn": "DCRNN",
              "gwn": "Graph\nWaveNet", "agcrn": "AGCRN", "stgnn": "GAT-GRU\n(in-house)"}
    palette = {"dlinear": "#bdbdbd", "gru": "#969696", "stgcn": "#74add1", "dcrnn": "#4575b4",
               "gwn": "#5aae61", "agcrn": "#d6604d", "stgnn": "#1b7837"}
    fig, axes = plt.subplots(1, len(Ks), figsize=(5.4 * len(Ks), 4.0), sharey=True, squeeze=False)
    for ax, K in zip(axes[0], Ks):
        sub = df[df.K == K]
        g = sub.groupby("model")["skill_vs_persistence"].agg(["mean", "std"]).reindex(order)
        x = np.arange(len(order))
        ax.bar(x, g["mean"], 0.65, yerr=g["std"].fillna(0), capsize=3,
               color=[palette[m] for m in order])
        ax.set_xticks(x); ax.set_xticklabels([labels[m] for m in order], fontsize=7.5)
        ax.axhline(0, color="k", lw=0.6); ax.set_title(f"K = {K}")
    axes[0][0].set_ylabel("skill vs. persistence")
    fig.suptitle("Model comparison at two graph sizes (Netflix)")
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
    labels = {"gru": "GRU\n(no graph)", "stgnn2": "GAT\n(sparse)"}
    rA = a.groupby("model")["skill_vs_persistence"].agg(["mean", "std"]).reindex(models)
    rB = b.groupby("model")["skill_vs_persistence"].agg(["mean", "std"]).reindex(models)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    x = np.arange(len(models)); w = 0.36
    ax.bar(x - w / 2, rA["mean"], w, yerr=rA["std"].fillna(0), capsize=3,
           color="#1b7837", label="cluster-first (aggregate $\\rightarrow$ forecast)")
    ax.bar(x + w / 2, rB["mean"], w, yerr=rB["std"].fillna(0), capsize=3,
           color="#b2182b", label="per-base-station (forecast $\\rightarrow$ sum)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([labels[m] for m in models])
    ax.set_ylabel("skill vs. persistence  (K=10 target)")
    ax.set_title("Cluster-first vs. per-base-station forecasting")
    ax.legend(fontsize=8, loc="upper right")
    ax.annotate("per-BS $\\approx$ persistence (skill $\\approx$ 0)",
                (x[-1] + w / 2, 0.0), textcoords="offset points", xytext=(0, 10),
                fontsize=8, color="#b2182b", ha="center")
    _save(fig, "fig_ablation")


def main():
    print(f"writing figures -> {FIGS}")
    fig_skill_vs_k()
    fig_generalization()
    fig_model_comparison()
    fig_ablation()
    return 0


if __name__ == "__main__":
    sys.exit(main())
