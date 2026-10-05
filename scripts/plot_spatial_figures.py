#!/usr/bin/env python3
"""Spatial-substrate, clustering and diagnostic figures for the forecaster chapter.

Complements plot_forecaster_figures.py (benchmark / K-sweep / generalization / ablation).
Everything is derived from pipeline artefacts; nothing is retrained here.

  fig_voronoi_graph      Voronoi coverage cells (mean load) + base-station adjacency graph
  fig_spatial_stats      cell area, node degree, edge length distributions
  fig_traffic            diurnal profiles, per-station sparsity, autocorrelation, correlation vs hops
  fig_cluster_maps       SKATER service areas for K = 5, 10, 20
  fig_cluster_k10        K = 10 coarse graph, cluster sizes and loads
  fig_partition_control  SKATER vs k-means vs random partition   (needs partition_control.py output)
  fig_diagnostics        per-cluster / per-horizon / load-regime error of Graph WaveNet, K = 10
  fig_forecast_examples  forecast vs. actual traces, K = 10

Also writes the numbers quoted in the chapter to results/cluster_first/spatial/*.csv|json.
Usage:  python scripts/plot_spatial_figures.py [--only NAME ...]
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from shapely.ops import polylabel, unary_union

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from src import aggregate as ag                                   # noqa: E402
import figstyle as fs                                             # noqa: E402

GRAPHS = REPO / "data" / "graphs"
CF = REPO / "data" / "cluster_first"
PART = REPO / "results" / "cluster_first" / "partition"
STATS = REPO / "results" / "cluster_first" / "spatial"
SERVICES = ["Netflix", "YouTube", "DailyMotion"]
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#e8f1fd", "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
_cache: dict = {}


# =========================================================================
#  Data access (memoised)
# =========================================================================
def nodes() -> pd.DataFrame:
    if "nodes" not in _cache:
        ni = pd.read_parquet(GRAPHS / "node_index.parquet").sort_values("node_idx")
        _cache["nodes"] = ni.merge(pd.read_parquet(GRAPHS / "bs_locations.parquet"), on="site_id")
    return _cache["nodes"]


def bs_series() -> tuple[dict, np.ndarray]:
    """{service: (N, T) float32 in node order}, timestamps (T,)."""
    if "series" not in _cache:
        files = sorted((REPO / "data" / "netmob" / "processed").glob("*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        ts = np.sort(df["timestamp"].unique())
        out = {}
        for svc in SERVICES:
            piv = (df[df.service == svc]
                   .pivot_table(index="site_id", columns="timestamp", values="dl_norm", fill_value=0.0)
                   .reindex(index=nodes()["site_id"].values, columns=ts, fill_value=0.0))
            out[svc] = piv.values.astype(np.float32)
        _cache["series"] = (out, pd.DatetimeIndex(ts))
    return _cache["series"]


def geometry():
    """Voronoi cells (projected km, origin at the station centroid), station xy, city outline."""
    if "geom" not in _cache:
        params = yaml.safe_load(open(REPO / "params.yaml"))
        params["aggregation"]["geojson_path"] = str(REPO / params["aggregation"]["geojson_path"])
        boundary = ag.load_city_boundary(ag.load_geojson(params))
        vr = ag.build_voronoi(nodes(), params, city_boundary=boundary)
        origin = vr.bs_xy.mean(axis=0)
        from shapely.affinity import affine_transform
        tokm = lambda g: affine_transform(g, [1e-3, 0, 0, 1e-3, -origin[0] / 1e3, -origin[1] / 1e3])
        cells = [tokm(p) for p in vr.polygons]
        _cache["geom"] = (cells, (vr.bs_xy - origin) / 1e3, tokm(ag._project_geometry(boundary, vr.lat_ref)))
    return _cache["geom"]


def fine_edges() -> np.ndarray:
    """(E, 2) undirected base-station adjacency (each pair once)."""
    ei = np.load(GRAPHS / "edge_index.npy")
    e = np.sort(ei.T[ei[0] != ei[1]], axis=1)
    return np.unique(e, axis=0)


def labels_for(K: int, service: str = "Netflix") -> np.ndarray:
    ca = pd.read_parquet(CF / service / f"K{K}" / "cluster_assignments.parquet")
    return nodes().merge(ca, on="site_id", how="left")["cluster_id"].values.astype(int)


def corr_matrix(service: str = "Netflix") -> np.ndarray:
    key = f"corr_{service}"
    if key not in _cache:
        x = bs_series()[0][service].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            _cache[key] = np.corrcoef(x)
    return _cache[key]


def _polys(geom):
    return [np.asarray(g.exterior.coords) for g in getattr(geom, "geoms", [geom]) if not g.is_empty]


def _map_axes(ax, outline, pad=0.6):
    x0, y0, x1, y1 = outline.bounds
    ax.set_xlim(x0 - pad, x1 + pad); ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal"); ax.axis("off")


def _scalebar(ax, outline, km=5):
    x0, y0, _, _ = outline.bounds
    ax.plot([x0, x0 + km], [y0 - 0.2, y0 - 0.2], color=fs.INK, lw=1.2, solid_capstyle="butt")
    ax.text(x0 + km / 2, y0 - 0.5, f"{km} km", ha="center", va="top", fontsize=7, color=fs.INK2)


def _outline(ax, outline, **kw):
    for ring in _polys(outline):
        ax.plot(ring[:, 0], ring[:, 1], **{"color": fs.INK2, "lw": 0.6, **kw})


# =========================================================================
#  Fig: Voronoi tessellation + adjacency graph
# =========================================================================
def fig_voronoi_graph():
    cells, xy, outline = geometry()
    load = bs_series()[0]["Netflix"].mean(axis=1) / np.array([c.area for c in cells])   # per km^2
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXTWIDTH, 3.25), gridspec_kw={"wspace": 0.04})

    ax = axes[0]
    verts, vals = [], []
    for c, v in zip(cells, load):
        for ring in _polys(c):
            verts.append(ring); vals.append(v)
    vals = np.array(vals); lo, hi = np.percentile(vals, [2, 99.5])
    pc = PolyCollection(verts, array=np.clip(vals, lo, hi), cmap=SEQ_BLUE, norm=LogNorm(lo, hi),
                        edgecolors="white", linewidths=0.15)
    ax.add_collection(pc); _outline(ax, outline); _map_axes(ax, outline); _scalebar(ax, outline)
    cb = fig.colorbar(pc, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.55,
                      anchor=(0.85, 1.0))
    cb.set_label(r"mean Netflix load density ($\hat d$ per km$^2$)", fontsize=7.5)
    cb.outline.set_visible(False); cb.ax.tick_params(length=2, labelsize=7)
    fs.panel(ax, "a", "Voronoi coverage cells")

    ax = axes[1]
    e = fine_edges()
    ax.add_collection(LineCollection(xy[e], colors=fs.MUTED, linewidths=0.35))
    ax.scatter(xy[:, 0], xy[:, 1], s=2.2, color=fs.BLUE, linewidths=0, zorder=3)
    _outline(ax, outline); _map_axes(ax, outline); _scalebar(ax, outline)
    fs.panel(ax, "b", "base-station adjacency graph")
    fs.save(fig, "fig_voronoi_graph")


# =========================================================================
#  Fig: spatial statistics
# =========================================================================
def fig_spatial_stats():
    cells, xy, _ = geometry()
    area = np.array([c.area for c in cells])
    e = fine_edges()
    deg = np.bincount(e.ravel(), minlength=len(xy))
    length = np.linalg.norm(xy[e[:, 0]] - xy[e[:, 1]], axis=1)
    vm = pd.read_parquet(GRAPHS / "voronoi_map.parquet")
    tiles = vm.groupby("site_id").size().reindex(nodes()["site_id"]).fillna(0).values

    fig, axes = plt.subplots(1, 3, figsize=(fs.TEXTWIDTH, 2.1), gridspec_kw={"wspace": 0.38})

    def median_rule(ax, v, fmt):
        m = float(np.median(v))
        ax.axvline(m, color=fs.INK, lw=0.8, ls=(0, (3, 2)))
        ax.annotate(f"median {fmt.format(m)}", (m, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -1), textcoords="offset points", va="top", fontsize=7, color=fs.INK2,
                    bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.9))

    ax = axes[0]
    ax.hist(area, bins=np.logspace(np.log10(area.min()), np.log10(area.max()), 28),
            color=fs.BLUE, edgecolor="white", linewidth=0.4)
    ax.set_xscale("log"); ax.set_xlabel(r"cell area (km$^2$)"); ax.set_ylabel("base stations")
    median_rule(ax, area, "{:.2f} km$^2$"); fs.panel(ax, "a")

    ax = axes[1]
    ax.bar(np.arange(deg.max() + 1), np.bincount(deg), width=0.8, color=fs.BLUE,
           edgecolor="white", linewidth=0.4)
    ax.set_xlabel("node degree"); ax.set_ylabel("base stations")
    median_rule(ax, deg, "{:.0f}"); fs.panel(ax, "b")

    ax = axes[2]
    ax.hist(length, bins=np.linspace(0, np.percentile(length, 99.5), 30), color=fs.BLUE,
            edgecolor="white", linewidth=0.4)
    ax.set_xlabel("edge length (km)"); ax.set_ylabel("edges")
    median_rule(ax, length, "{:.2f} km"); fs.panel(ax, "c")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fs.save(fig, "fig_spatial_stats")

    STATS.mkdir(parents=True, exist_ok=True)
    json.dump({
        "n_bs": int(len(xy)), "n_tiles": int(len(vm)), "city_area_km2": float(area.sum()),
        "tiles_per_bs": {"min": int(tiles.min()), "median": float(np.median(tiles)),
                         "mean": float(tiles.mean()), "max": int(tiles.max())},
        "cell_area_km2": {"min": float(area.min()), "median": float(np.median(area)),
                          "mean": float(area.mean()), "p90": float(np.percentile(area, 90)),
                          "max": float(area.max())},
        "n_edges": int(len(e)),
        "degree": {"min": int(deg.min()), "median": float(np.median(deg)), "mean": float(deg.mean()),
                   "max": int(deg.max()), "n_isolated": int((deg == 0).sum())},
        "edge_length_km": {"median": float(np.median(length)), "mean": float(length.mean()),
                           "p90": float(np.percentile(length, 90)), "max": float(length.max())},
    }, open(STATS / "graph_stats.json", "w"), indent=2)


# =========================================================================
#  Fig: traffic characterisation
# =========================================================================
def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
    v = float(np.dot(x, x))
    return np.array([1.0] + [float(np.dot(x[:-k], x[k:])) / v for k in range(1, max_lag + 1)])


def fig_traffic():
    series, ts = bs_series()
    fig, axes = plt.subplots(2, 2, figsize=(fs.TEXTWIDTH, 4.5), gridspec_kw={"wspace": 0.3, "hspace": 0.52})
    colors = dict(zip(SERVICES, [fs.BLUE, fs.ORANGE, fs.AQUA]))
    stats: dict = {}

    # (a) diurnal profile of the city aggregate, relative to each service's own mean
    ax = axes[0, 0]
    slot = ts.hour * 4 + ts.minute // 15
    wknd = ts.dayofweek >= 5
    hours = np.arange(96) / 4
    for svc in SERVICES:
        city = series[svc].sum(axis=0)
        for mask, ls in ((~wknd, "-"), (wknd, (0, (4, 2)))):
            prof = pd.Series(city[mask]).groupby(slot[mask]).mean().reindex(range(96)).values
            ax.plot(hours, prof / city.mean(), color=colors[svc], ls=ls,
                    label=svc if ls == "-" else None)
        stats[f"mean_city_load_{svc}"] = float(city.mean())
    ax.plot([], [], color=fs.INK2, ls="-", label="weekday"); ax.plot([], [], color=fs.INK2, ls=(0, (4, 2)), label="weekend")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 6)); ax.set_xlabel("hour of day (local time)")
    ax.set_ylabel("city load / service mean"); ax.set_ylim(0, 3.1); ax.set_yticks([0, 1, 2, 3])
    ax.legend(ncol=2, loc="upper left", columnspacing=1.0, borderaxespad=0.1, labelspacing=0.25)
    fs.panel(ax, "a", "diurnal profile")

    # (b) sparsity: share of each station's slots that are (near-)empty
    ax = axes[0, 1]
    for svc in SERVICES:
        frac = np.sort((series[svc] < 0.01).mean(axis=1))
        ax.plot(frac * 100, np.arange(1, len(frac) + 1) / len(frac), color=colors[svc], label=svc)
        stats[f"share_slots_below_0.01_{svc}"] = float((series[svc] < 0.01).mean())
        stats[f"median_bs_share_below_0.01_{svc}"] = float(np.median(frac))
        stats[f"p10_bs_share_below_0.01_{svc}"] = float(np.percentile(frac, 10))
    ax.set_xlabel(r"share of slots with $\hat d<0.01$ (%)"); ax.set_ylabel("fraction of base stations")
    ax.set_xlim(0, 100); ax.set_ylim(0, 1); ax.legend(loc="upper left", borderaxespad=0.2)
    fs.panel(ax, "b", "per-station sparsity")

    # (c) autocorrelation at three aggregation levels (Netflix)
    ax = axes[1, 0]
    x = series["Netflix"]; L = 192; lag_h = np.arange(L + 1) / 4
    active = np.where(x.std(axis=1) > 0)[0]
    acf_bs = np.array([_acf(x[i].astype(np.float64), L) for i in active])
    cl = np.load(CF / "Netflix" / "K10" / "cluster_series.npy").astype(np.float64)
    acf_cl = np.array([_acf(c, L) for c in cl])
    acf_city = _acf(x.sum(axis=0).astype(np.float64), L)
    q25, q50, q75 = np.percentile(acf_bs, [25, 50, 75], axis=0)
    ax.fill_between(lag_h, q25, q75, color=fs.ORANGE, alpha=0.18, lw=0)
    ax.plot(lag_h, q50, color=fs.ORANGE, label="base station (median, IQR)")
    ax.plot(lag_h, np.median(acf_cl, axis=0), color=fs.BLUE, label="cluster, $K=10$ (median)")
    ax.plot(lag_h, acf_city, color=fs.INK2, ls=(0, (4, 2)), label="city aggregate")
    ax.axhline(0, color=fs.INK2, lw=0.6)
    ax.set_xlim(0, 48); ax.set_xticks(range(0, 49, 12)); ax.set_ylim(-0.45, 1.85)
    ax.set_yticks([-0.4, 0, 0.4, 0.8])
    ax.set_xlabel("lag (h)"); ax.set_ylabel("autocorrelation"); ax.legend(loc="upper right", borderaxespad=0.1, labelspacing=0.25)
    fs.panel(ax, "c", "autocorrelation by aggregation level")
    for name, a in (("bs_median", q50), ("cluster_median", np.median(acf_cl, axis=0)), ("city", acf_city)):
        stats[f"acf_{name}_lag15min"] = float(a[1]); stats[f"acf_{name}_lag1h"] = float(a[4])
        stats[f"acf_{name}_lag24h"] = float(a[96])

    # (d) correlation between stations vs. hop distance on the Voronoi graph (Netflix)
    ax = axes[1, 1]
    C = corr_matrix(); n = len(C); e = fine_edges()
    A = csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)); A = A + A.T
    D = shortest_path(A, unweighted=True, directed=False)
    iu = np.triu_indices(n, k=1); d = D[iu]; c = C[iu]; ok = np.isfinite(c) & np.isfinite(d)
    hops = np.arange(1, 13)
    med = [np.median(c[ok & (d == h)]) for h in hops]
    lo = [np.percentile(c[ok & (d == h)], 25) for h in hops]
    hi = [np.percentile(c[ok & (d == h)], 75) for h in hops]
    ax.fill_between(hops, lo, hi, color=fs.BLUE, alpha=0.18, lw=0)
    ax.plot(hops, med, color=fs.BLUE, marker="o", label="station pairs (median, IQR)")
    ax.axhline(np.median(c[ok]), color=fs.INK2, ls=(0, (4, 2)), lw=1.0, label="all pairs (median)")
    ax.set_xticks(hops[::1]); ax.set_xlim(0.5, 12.5); ax.set_ylim(0, None)
    ax.set_xlabel("graph distance (hops)"); ax.set_ylabel("Pearson correlation")
    ax.legend(loc="upper right", borderaxespad=0.1)
    fs.panel(ax, "d", "spatial correlation")
    stats.update({f"corr_hop{h}": float(m) for h, m in zip(hops, med)})
    stats["corr_all_pairs_median"] = float(np.median(c[ok]))
    stats["n_bs_zero_variance_Netflix"] = int(n - len(active))

    fs.save(fig, "fig_traffic")
    STATS.mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(STATS / "traffic_stats.json", "w"), indent=2)


# =========================================================================
#  Clustering figures
# =========================================================================
def coarse_adjacency(labels: np.ndarray) -> set:
    e = fine_edges(); a, b = labels[e[:, 0]], labels[e[:, 1]]
    return {(int(min(i, j)), int(max(i, j))) for i, j in zip(a, b) if i != j}


def greedy_colouring(K: int, adj: set) -> np.ndarray:
    """Neighbour-distinct colour index per cluster (largest-degree-first greedy)."""
    nb = {k: set() for k in range(K)}
    for i, j in adj:
        nb[i].add(j); nb[j].add(i)
    col = -np.ones(K, dtype=int)
    for k in sorted(range(K), key=lambda k: -len(nb[k])):
        used = {col[j] for j in nb[k] if col[j] >= 0}
        col[k] = next(c for c in range(K) if c not in used)
    return col


def _draw_partition(ax, labels, label_ids=True, fontsize=7):
    cells, xy, outline = geometry()
    K = int(labels.max()) + 1
    col = greedy_colouring(K, coarse_adjacency(labels))
    assert col.max() < len(fs.TINTS), "more neighbour-distinct colours needed than tints defined"
    verts, fc = [], []
    for c, lab in zip(cells, labels):
        for ring in _polys(c):
            verts.append(ring); fc.append(fs.TINTS[col[lab]])
    ax.add_collection(PolyCollection(verts, facecolors=fc, edgecolors="white", linewidths=0.12))
    regions = []
    for k in range(K):
        region = unary_union([cells[i].buffer(1e-4) for i in np.where(labels == k)[0]])
        regions.append(region)
        for ring in _polys(region):
            ax.plot(ring[:, 0], ring[:, 1], color=fs.INK2, lw=0.6)
        if label_ids:
            big = max(getattr(region, "geoms", [region]), key=lambda g: g.area)
            p = polylabel(big, tolerance=0.05)
            ax.text(p.x, p.y, str(k), ha="center", va="center", fontsize=fontsize, color=fs.INK)
    _map_axes(ax, outline)
    return regions


def fig_cluster_maps(Ks=(5, 10, 20)):
    fig, axes = plt.subplots(1, len(Ks), figsize=(fs.TEXTWIDTH, 2.55), gridspec_kw={"wspace": 0.02})
    _, _, outline = geometry()
    for ax, K, letter in zip(axes, Ks, "abc"):
        _draw_partition(ax, labels_for(K), fontsize=7 if K <= 10 else 6)
        fs.panel(ax, letter, f"$K={K}$")
    _scalebar(axes[0], outline)
    fs.save(fig, "fig_cluster_maps")


def cluster_table(Ks=(4, 5, 6, 8, 10, 15, 20, 30, 50, 100), n_null=200) -> pd.DataFrame:
    """Structure of the SKATER partition at every K + within-cluster coherence vs a size-matched null."""
    cells, xy, _ = geometry()
    area = np.array([c.area for c in cells]); C = corr_matrix()
    rng = np.random.default_rng(0)

    def coherence(lab):
        vals = []
        for k in range(lab.max() + 1):
            idx = np.where(lab == k)[0]
            if len(idx) > 1:
                sub = C[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)]
                vals.append(np.nanmean(sub))
        return float(np.nanmean(vals))

    rows = []
    for K in Ks:
        lab = labels_for(K); n = np.bincount(lab, minlength=K)
        cs = np.load(CF / "Netflix" / f"K{K}" / "cluster_series.npy")
        radius = [np.linalg.norm(xy[lab == k] - xy[lab == k].mean(0), axis=1).mean() for k in range(K)]
        null = [coherence(rng.permutation(lab)) for _ in range(n_null)]
        cc = np.corrcoef(cs)[np.triu_indices(K, k=1)]
        rows.append({
            "K": K, "min_bs": int(n.min()), "median_bs": float(np.median(n)), "max_bs": int(n.max()),
            "largest_share_bs": float(n.max() / n.sum()),
            "mean_radius_km": float(np.mean(radius)), "max_radius_km": float(np.max(radius)),
            "mean_area_km2": float(np.mean([area[lab == k].sum() for k in range(K)])),
            "load_cv": float(cs.mean(1).std() / cs.mean(1).mean()),
            "largest_share_load": float(cs.mean(1).max() / cs.mean(1).sum()),
            "peak_load_max": float(cs.max()),
            "n_coarse_edges": len(coarse_adjacency(lab)),
            "mean_coarse_degree": 2 * len(coarse_adjacency(lab)) / K,
            "coherence_skater": coherence(lab),
            "coherence_random_mean": float(np.mean(null)), "coherence_random_std": float(np.std(null)),
            "intercluster_corr_mean": float(cc.mean()),
        })
    df = pd.DataFrame(rows)
    STATS.mkdir(parents=True, exist_ok=True)
    df.to_csv(STATS / "cluster_structure.csv", index=False)
    print(df.round(3).to_string(index=False))
    return df


def fig_cluster_k10(K=10):
    cells, xy, outline = geometry()
    lab = labels_for(K)
    cs = np.load(CF / "Netflix" / f"K{K}" / "cluster_series.npy")
    n = np.bincount(lab, minlength=K); mean, p99 = cs.mean(1), np.percentile(cs, 99, axis=1)
    order = np.argsort(-mean)

    fig = plt.figure(figsize=(fs.TEXTWIDTH, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 0.72, 0.9], wspace=0.14)
    ax = fig.add_subplot(gs[0])
    _draw_partition(ax, lab, label_ids=False)
    cen = np.array([xy[lab == k].mean(0) for k in range(K)])
    adj = sorted(coarse_adjacency(lab))
    ax.add_collection(LineCollection([cen[[i, j]] for i, j in adj], colors=fs.INK, linewidths=0.7, zorder=3))
    ax.scatter(cen[:, 0], cen[:, 1], s=30 + 330 * mean / mean.max(), color=fs.BLUE, edgecolors="white",
               linewidths=0.8, zorder=4)
    for k in range(K):
        ax.annotate(str(k), cen[k], ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold", zorder=5)
    _scalebar(ax, outline)
    fs.panel(ax, "a", "service areas and coarse graph")

    y = np.arange(K)
    ax1 = fig.add_subplot(gs[1])
    ax1.barh(y, n[order], height=0.62, color=fs.BLUE)
    for yi, v in zip(y, n[order]):
        ax1.text(v + 6, yi, str(v), va="center", fontsize=7, color=fs.INK2)
    ax1.set_yticks(y); ax1.set_yticklabels([str(k) for k in order]); ax1.invert_yaxis()
    ax1.set_ylabel("cluster"); ax1.set_xlabel("base stations"); ax1.set_xlim(0, n.max() * 1.22)
    ax1.grid(axis="y", visible=False); ax1.tick_params(axis="y", length=0)
    fs.panel(ax1, "b", "size")

    ax2 = fig.add_subplot(gs[2], sharey=ax1)
    ax2.hlines(y, mean[order], p99[order], color=fs.RULE, lw=1.6, zorder=1)
    ax2.scatter(mean[order], y, s=18, color=fs.BLUE, zorder=3, label="mean")
    ax2.scatter(p99[order], y, s=20, color=fs.ORANGE, marker="D", zorder=3, label="99th pct.")
    ax2.set_xlabel(r"cluster load $\hat d$"); ax2.set_xlim(0, None)
    ax2.tick_params(axis="y", labelleft=False, length=0); ax2.grid(axis="y", visible=False)
    ax2.legend(loc="lower right", borderaxespad=0.2, handletextpad=0.3)
    fs.panel(ax2, "c", "load")
    fs.save(fig, "fig_cluster_k10")


# =========================================================================
#  Forecast diagnostics (Graph WaveNet on the SKATER partition; partition_control.py output)
# =========================================================================
def _runs(tag: str, model: str, K: int) -> list[dict]:
    files = sorted((PART / "preds").glob(f"{tag}_{model}_K{K}_seed*.npz"))
    return [dict(np.load(f)) for f in files]


def _wape(p, t, axis=None):
    return 100.0 * np.abs(p - t).sum(axis=axis) / np.abs(t).sum(axis=axis)


def fig_diagnostics(K=10, model="gwn"):
    runs = _runs("skater", model, K)
    if not runs:
        print("  - skip diagnostics (run scripts/partition_control.py first)"); return
    true, pers, seas = runs[0]["true"], runs[0]["persistence"], runs[0]["seasonal"]
    preds = np.stack([r["pred"] for r in runs])                      # (S, n, H, K)
    cs = np.load(CF / "Netflix" / f"K{K}" / "cluster_series.npy")
    order = np.argsort(-cs.mean(1))
    H = true.shape[1]; stats: dict = {"K": K, "model": model, "n_seeds": len(runs)}

    fig, axes = plt.subplots(1, 3, figsize=(fs.TEXTWIDTH, 2.35), gridspec_kw={"wspace": 0.42})

    # (a) skill per cluster
    ax = axes[0]
    w_m = np.stack([_wape(p, true, axis=(0, 1)) for p in preds])     # (S, K)
    skill = 1 - w_m / _wape(pers, true, axis=(0, 1))
    pooled = float(np.mean([1 - _wape(p, true) / _wape(pers, true) for p in preds]))
    x = np.arange(K)
    ax.bar(x, skill.mean(0)[order], 0.66, yerr=skill.std(0)[order], color=fs.BLUE,
           error_kw=dict(ecolor=fs.INK, elinewidth=0.7, capsize=1.5))
    ax.axhline(pooled, color=fs.INK, lw=0.8, ls=(0, (3, 2)))
    ax.text(0.98, 0.97, f"dashed: pooled ({pooled:.3f})", transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=fs.INK2)
    ax.set_xticks(x); ax.set_xticklabels([str(k) for k in order])
    ax.set_xlabel("cluster (by decreasing load)"); ax.set_ylabel("skill vs. persistence")
    ax.set_ylim(0, max(0.36, skill.mean(0).max() * 1.25)); ax.grid(axis="x", visible=False)
    fs.panel(ax, "a", "by cluster")
    stats["skill_per_cluster"] = {int(k): float(v) for k, v in enumerate(skill.mean(0))}
    stats["wape_per_cluster"] = {int(k): float(v) for k, v in enumerate(w_m.mean(0))}
    stats["skill_pooled"] = pooled

    # (b) error by lead time
    ax = axes[1]
    lead = (np.arange(H) + 1) * 15
    wm = np.stack([_wape(p, true, axis=(0, 2)) for p in preds])
    wp, ws = _wape(pers, true, axis=(0, 2)), _wape(seas, true, axis=(0, 2))
    ax.plot(lead, ws, color=fs.AQUA, marker="^", label="seasonal-naive")
    ax.plot(lead, wp, color=fs.ORANGE, marker="s", label="persistence")
    ax.errorbar(lead, wm.mean(0), yerr=wm.std(0), color=fs.BLUE, marker="o", capsize=1.5,
                label="Graph WaveNet")
    ax.set_xticks(lead); ax.set_xlabel("lead time (min)"); ax.set_ylabel("WAPE (%)")
    ax.set_ylim(0, ws.max() * 1.45); ax.legend(loc="upper left", borderaxespad=0.1, labelspacing=0.25)
    fs.panel(ax, "b", "by lead time")
    stats["wape_by_lead"] = {"model": wm.mean(0).tolist(), "persistence": wp.tolist(), "seasonal": ws.tolist()}
    stats["skill_by_lead"] = (1 - wm.mean(0) / wp).tolist()

    # (c) error by load regime (per-cluster quartiles of the realised load)
    ax = axes[2]
    q = np.quantile(true, [0.25, 0.5, 0.75], axis=(0, 1))            # (3, K)
    regime = (true[..., None] > q.T[None, None]).sum(-1)             # 0..3
    wm_r = np.array([[_wape(p[regime == r], true[regime == r]) for r in range(4)] for p in preds])
    wp_r = np.array([_wape(pers[regime == r], true[regime == r]) for r in range(4)])
    bias = np.array([[(p[regime == r] - true[regime == r]).sum() / true[regime == r].sum() * 100
                      for r in range(4)] for p in preds])
    xr = np.arange(4); bw = 0.36
    ax.bar(xr - bw / 2 - 0.01, wp_r, bw, color=fs.ORANGE, label="persistence")
    ax.bar(xr + bw / 2 + 0.01, wm_r.mean(0), bw, yerr=wm_r.std(0), color=fs.BLUE, label="Graph WaveNet",
           error_kw=dict(ecolor=fs.INK, elinewidth=0.7, capsize=1.5))
    ax.set_xticks(xr); ax.set_xticklabels(["Q1\n(low)", "Q2", "Q3", "Q4\n(peak)"])
    ax.set_xlabel("load quartile"); ax.set_ylabel("WAPE (%)"); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", borderaxespad=0.1, labelspacing=0.25)
    fs.panel(ax, "c", "by load regime")
    stats["wape_by_quartile"] = {"model": wm_r.mean(0).tolist(), "persistence": wp_r.tolist()}
    stats["bias_pct_by_quartile"] = bias.mean(0).tolist()
    top = true >= np.quantile(true, 0.99, axis=(0, 1))
    stats["bias_pct_top1pct"] = float(np.mean([(p[top] - true[top]).sum() / true[top].sum() * 100 for p in preds]))
    stats["mae"] = float(np.mean([np.abs(p - true).mean() for p in preds]))
    stats["wape"] = float(np.mean([_wape(p, true) for p in preds]))

    fs.save(fig, "fig_diagnostics")
    STATS.mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(STATS / f"diagnostics_{model}_K{K}.json", "w"), indent=2)
    print(json.dumps(stats, indent=1)[:1800])


def fig_forecast_examples(K=10, model="gwn", clusters=(9, 2, 0), days=5, lead=4):
    runs = _runs("skater", model, K)
    if not runs:
        print("  - skip forecast_examples (run scripts/partition_control.py first)"); return
    r = runs[0]; _, ts = bs_series()
    cfg = yaml.safe_load(open(REPO / "config.yaml"))
    T = len(ts); n_test = int(T * cfg["training"]["test_split"]); L = cfg["stgnn"]["seq_len"]
    t = ts[T - n_test + L + (lead - 1): T - n_test + L + (lead - 1) + len(r["true"])]   # target times
    start = int(np.argmax((t.hour == 0) & (t.minute == 0)))          # first midnight in the window
    sl = slice(start, start + days * 96)
    n = np.bincount(labels_for(K), minlength=K)

    fig, axes = plt.subplots(len(clusters), 1, figsize=(fs.TEXTWIDTH, 1.25 * len(clusters) + 0.5),
                             sharex=True, gridspec_kw={"hspace": 0.32})
    for ax, k, letter in zip(axes, clusters, "abc"):
        y, p = r["true"][sl, lead - 1, k], r["pred"][sl, lead - 1, k]
        ax.plot(t[sl], y, color=fs.INK2, lw=0.9, label="observed")
        ax.plot(t[sl], p, color=fs.BLUE, lw=1.3, label=f"Graph WaveNet, {lead * 15} min ahead")
        ax.set_ylim(0, None); ax.set_ylabel(r"load $\hat d$")
        fs.panel(ax, letter, f"cluster {k} ({n[k]} base stations)")
        ax.margins(x=0)
    axes[0].legend(loc="upper right", ncol=2, borderaxespad=0.0, bbox_to_anchor=(1.0, 1.3))
    import matplotlib.dates as mdates
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    axes[-1].xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    fs.save(fig, "fig_forecast_examples")
    print(f"  example window: {t[sl][0]} .. {t[sl][-1]}")


# =========================================================================
#  Partition control (partition_control.py + partition_descriptors.py output)
# =========================================================================
def fig_partition_control(Ks=(10, 50)):
    f = PART / "partition.csv"; g = PART / "descriptors.csv"
    if not (f.exists() and g.exists()):
        print("  - skip partition_control (run partition_control.py and partition_descriptors.py)"); return
    df = pd.read_csv(f); desc = pd.read_csv(g)
    Ks = [K for K in Ks if K in df.K.unique()]
    kinds = ["skater", "kmeans", "random"]
    names = {"skater": "SKATER", "kmeans": "$k$-means", "random": "random"}
    kcol = dict(zip(Ks, [fs.BLUE, fs.ORANGE]))               # colour encodes K in all three panels
    gwn = df[df.model == "gwn"]
    fig, axes = plt.subplots(1, 3, figsize=(fs.TEXTWIDTH, 2.5), gridspec_kw={"wspace": 0.45})
    x = np.arange(len(kinds)); w = 0.34
    off = {K: (i - (len(Ks) - 1) / 2) * (w + 0.02) for i, K in enumerate(Ks)}

    def runs_on(ax, col):                                    # bar = median over runs, dots = every run
        for K in Ks:
            sub = gwn[gwn.K == K]
            ax.bar(x + off[K], sub.groupby("partition")[col].median().reindex(kinds), w,
                   color=kcol[K], label=f"$K={K}$")
            for xi, kind in zip(x + off[K], kinds):
                v = np.sort(sub[sub.partition == kind][col].values)
                jit = np.linspace(-0.09, 0.09, len(v)) if len(v) > 1 else np.zeros(len(v))
                ax.scatter(xi + jit, v, s=6, color=fs.INK, edgecolors="white", linewidths=0.4, zorder=3)

    ax = axes[0]                                             # raw error and the target's difficulty
    runs_on(ax, "test_wape")
    for K in Ks:
        lv = gwn[gwn.K == K].groupby("partition")["lv_wape"].mean().reindex(kinds)
        ax.hlines(lv, x + off[K] - w / 2, x + off[K] + w / 2, color=fs.INK, lw=1.3, zorder=4,
                  label="persistence" if K == Ks[-1] else None)
    ax.set_ylabel("test WAPE (%)"); ax.set_ylim(0, 58)
    ax.legend(loc="upper right", borderaxespad=0.0, labelspacing=0.2, handlelength=1.2)
    fs.panel(ax, "a", "raw error")

    ax = axes[1]
    runs_on(ax, "skill_vs_persistence")
    ax.set_ylabel("skill vs. persistence"); ax.set_ylim(0, 0.36)
    fs.panel(ax, "b", "skill")

    ax = axes[2]                                             # redundancy between the K cluster signals
    for K in Ks:
        d = desc[desc.K == K].groupby("partition")["intercluster_corr"].mean().reindex(kinds)
        ax.bar(x + off[K], d.values, w, color=kcol[K])
    ax.set_ylim(0, 1.0); ax.set_ylabel("mean correlation between\ncluster signals")
    fs.panel(ax, "c", "redundancy")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([names[k] for k in kinds]); ax.grid(axis="x", visible=False)
    fs.save(fig, "fig_partition_control")

    summ = (df.groupby(["K", "partition", "model"], sort=False)
              .agg(n=("seed", "size"), wape=("test_wape", "median"), wape_min=("test_wape", "min"),
                   wape_max=("test_wape", "max"), lv=("lv_wape", "mean"), sn=("sn_wape", "mean"),
                   skill_p=("skill_vs_persistence", "median"), skill_p_min=("skill_vs_persistence", "min"),
                   skill_p_max=("skill_vs_persistence", "max"), skill_p_mean=("skill_vs_persistence", "mean"),
                   skill_s=("skill_vs_seasonal", "median"), mae=("test_mae", "median"),
                   epochs_min=("epochs_run", "min")).reset_index())
    summ.to_csv(PART / "partition_summary.csv", index=False)
    print(summ.round(4).to_string(index=False))
    print(desc.groupby(["K", "partition"], sort=False).mean(numeric_only=True).drop(columns="draw").round(3).to_string())


FIGURES = {
    "voronoi_graph": fig_voronoi_graph, "spatial_stats": fig_spatial_stats, "traffic": fig_traffic,
    "cluster_maps": fig_cluster_maps, "cluster_table": cluster_table, "cluster_k10": fig_cluster_k10,
    "diagnostics": fig_diagnostics, "forecast_examples": fig_forecast_examples,
    "partition_control": fig_partition_control,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", choices=list(FIGURES), default=list(FIGURES))
    args = ap.parse_args()
    fs.apply()
    print(f"writing figures -> {fs.FIGS}")
    for name in args.only:
        FIGURES[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
