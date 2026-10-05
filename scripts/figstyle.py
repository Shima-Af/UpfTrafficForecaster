"""Shared matplotlib style for the forecaster-chapter figures (reports/figures/).

One place for typography, the colour roles and the save routine, so every chapter figure
reads as part of the same set. Colours are a fixed categorical order (never cycled); maps
with more regions than hues use a neighbour-distinct colouring plus direct labels instead.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FIGS = REPO / "reports" / "figures"

# Categorical slots, fixed order (colour-vision-deficiency-checked as adjacent pairs).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
INK, INK2, MUTED, RULE = "#0b0b0b", "#52514e", "#8a8983", "#d9d8d3"
# Pale fills for region maps (neighbour-distinct colouring; identity is carried by labels).
TINTS = ["#b7d3f6", "#f9cdb9", "#b5e6d2", "#f7dc9c", "#f5c9d9", "#c9c3ea", "#d5d4cf"]

TEXTWIDTH = 6.3   # inches; figures are drawn at final size so type stays ~8 pt in print


def apply() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "lines.linewidth": 1.4, "lines.markersize": 4,
        "legend.frameon": False, "legend.handlelength": 1.6,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel(ax, letter: str, title: str = "") -> None:
    """Panel tag '(a) title' set flush-left above the axes."""
    ax.set_title(f"({letter})" + (f" {title}" if title else ""), loc="left", pad=5)


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")
