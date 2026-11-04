"""Figures. One visual system for every chart:

* categorical colour = sector (8 validated slots; the 9th sector is neutral gray),
* sequential blue = strength of a link,
* diverging blue <-> red with a gray midpoint = signed correlation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from .config import SECTOR_OF, SECTORS

# ── Palette ──────────────────────────────────────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots in fixed order (validated: adjacent CVD dE >= 9.1).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#b4b2a9"

SECTOR_COLORS: dict[str, str] = {
    s: (CATEGORICAL[i] if i < len(CATEGORICAL) else OTHER) for i, s in enumerate(SECTORS)
}

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#3987e5", "#b7d3f6", "#f0efec", "#f5b8b7", "#e34948", "#8f1f1f"]
)


def use_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "font.size": 10,
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def _save(fig, path: str | Path | None):
    if path is not None:
        fig.savefig(path)
    return fig


def sector_legend(ax, sectors=None, loc="upper left", **kw):
    sectors = list(SECTORS) if sectors is None else sectors
    handles = [Line2D([0], [0], marker="o", lw=0, markersize=8,
                      markerfacecolor=SECTOR_COLORS[s], markeredgecolor=SURFACE, label=s)
               for s in sectors]
    return ax.legend(handles=handles, loc=loc, title="Sector", title_fontsize=9,
                     fontsize=9, alignment="left", **kw)


def sector_order(tickers) -> list[str]:
    """Tickers ordered by sector (config order), then alphabetically."""
    rank = {s: i for i, s in enumerate(SECTORS)}
    return sorted(tickers, key=lambda t: (rank.get(SECTOR_OF.get(t), 99), t))


# ── Figures ──────────────────────────────────────────────────────────────
def plot_correlation_heatmap(corr: pd.DataFrame, title: str, path=None):
    """Correlation heatmap with tickers grouped by sector and sector bands on the axes."""
    order = sector_order(corr.columns)
    c = corr.loc[order, order]
    n = len(order)

    fig, ax = plt.subplots(figsize=(12, 10.5))
    im = ax.imshow(c.values, cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n), order, rotation=90, fontsize=7.5)
    ax.set_yticks(range(n), order, fontsize=7.5)
    ax.tick_params(length=0, pad=16)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)

    # sector bands + separators
    bounds, start = [], 0
    for s in SECTORS:
        k = sum(SECTOR_OF.get(t) == s for t in order)
        if k:
            bounds.append((s, start, start + k))
            start += k
    for s, a, b in bounds:
        col = SECTOR_COLORS[s]
        ax.add_patch(plt.Rectangle((-1.45, a - 0.5), 0.75, b - a, color=col, clip_on=False, lw=0))
        ax.add_patch(plt.Rectangle((a - 0.5, n - 0.3), b - a, 0.75, color=col, clip_on=False, lw=0))
        if a:
            ax.axhline(a - 0.5, color=SURFACE, lw=1.5)
            ax.axvline(a - 0.5, color=SURFACE, lw=1.5)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Pearson ρ of daily log-returns", color=INK_2)
    ax.set_title(title, pad=12)
    sector_legend(ax, loc="upper left", bbox_to_anchor=(1.12, 1.0))
    return _save(fig, path)


def plot_correlation_distribution(pairs: pd.DataFrame, title: str, path=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(pairs["rho"], bins=40, color=CATEGORICAL[0], edgecolor=SURFACE, linewidth=1)
    mean = pairs["rho"].mean()
    ax.axvline(mean, color=INK_2, lw=1)
    ax.annotate(f"mean ρ = {mean:.2f}", (mean, ax.get_ylim()[1] * 0.92),
                xytext=(6, 0), textcoords="offset points", color=INK_2, fontsize=9)
    ax.set_xlabel("Pairwise correlation ρ")
    ax.set_ylabel("Number of pairs")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_title(title)
    return _save(fig, path)


def mst_layout(tree: nx.Graph) -> dict:
    """Kamada-Kawai on hop distance: deterministic and readable for trees."""
    return nx.kamada_kawai_layout(tree, weight=None)


def plot_mst(tree: nx.Graph, title: str, pos=None, path=None, highlight=None):
    """MST: node colour = sector, node size = degree, edge width/colour = correlation."""
    pos = mst_layout(tree) if pos is None else pos
    fig, ax = plt.subplots(figsize=(14, 11))

    rhos = np.array([d["rho"] for *_, d in tree.edges(data=True)])
    norm = Normalize(vmin=0.2, vmax=0.9)
    nx.draw_networkx_edges(
        tree, pos, ax=ax,
        width=1.0 + 5.0 * norm(rhos).clip(0, 1),
        edge_color=[SEQ_BLUE(0.25 + 0.75 * v) for v in norm(rhos).clip(0, 1)],
    )
    nodes = list(tree.nodes())
    deg = dict(tree.degree())
    nx.draw_networkx_nodes(
        tree, pos, ax=ax, nodelist=nodes,
        node_color=[SECTOR_COLORS.get(SECTOR_OF.get(n), OTHER) for n in nodes],
        node_size=[260 + 170 * deg[n] for n in nodes],
        edgecolors=SURFACE, linewidths=2,
    )
    if highlight:
        nx.draw_networkx_nodes(tree, pos, ax=ax, nodelist=highlight, node_color="none",
                               node_size=[260 + 170 * deg[n] + 520 for n in highlight],
                               edgecolors=INK, linewidths=1.4)
    for n, (x, y) in pos.items():
        ax.annotate(n, (x, y), xytext=(0, -(9 + 2.3 * deg[n])), textcoords="offset points",
                    ha="center", va="top", fontsize=8.5, color=INK, fontweight="bold",
                    path_effects=[patheffects.withStroke(linewidth=3, foreground=SURFACE)])

    sector_legend(ax, loc="lower left")
    sm = plt.cm.ScalarMappable(cmap=LinearSegmentedColormap.from_list(
        "edge", [SEQ_BLUE(0.25), SEQ_BLUE(1.0)]), norm=norm)
    cax = ax.inset_axes([0.73, 0.04, 0.24, 0.018])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_label("Edge correlation ρ (colour and width)", color=INK_2, fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax.set_title(title)
    ax.axis("off")
    ax.margins(0.04)
    return _save(fig, path)


def plot_centrality(metrics: pd.DataFrame, top: int = 12, path=None):
    """Top-N tickers by degree, betweenness and closeness — small multiples, one axis each."""
    cols = [("degree", "Degree"), ("betweenness", "Betweenness centrality"),
            ("closeness", "Closeness centrality (weighted)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (col, label) in zip(axes, cols):
        s = metrics[col].sort_values(ascending=False).head(top)[::-1]
        colors = [SECTOR_COLORS.get(SECTOR_OF.get(t), OTHER) for t in s.index]
        ax.barh(s.index, s.values, color=colors, height=0.72, edgecolor=SURFACE, linewidth=1)
        ax.set_title(label, fontsize=11)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        fmt = "{:.0f}" if col == "degree" else "{:.2f}"
        ax.bar_label(ax.containers[0], labels=[fmt.format(v) for v in s.values],
                     padding=3, fontsize=8, color=INK_2)
        ax.margins(x=0.15)
    present = [s for s in SECTORS if s in set(metrics["sector"])]
    fig.legend(handles=[Line2D([0], [0], marker="s", lw=0, markersize=9,
                               markerfacecolor=SECTOR_COLORS[s], markeredgecolor=SURFACE, label=s)
                        for s in present],
               loc="lower center", ncol=len(present), fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Most central stocks in the MST (top {top} per metric)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, path)


def plot_dendrogram(dist: pd.DataFrame, title: str, method: str = "single", path=None):
    """Hierarchical clustering on d_ij. Single linkage is equivalent to cutting the MST."""
    z = linkage(squareform(dist.to_numpy(), checks=False), method=method)
    fig, ax = plt.subplots(figsize=(15, 5.5))
    with plt.rc_context({"lines.linewidth": 1.2}):
        dn = dendrogram(z, labels=list(dist.columns), ax=ax, color_threshold=0,
                        above_threshold_color=INK_2, leaf_font_size=8.5)
    # sector marker under each leaf (identity shown by ticker label + colour)
    # start the axis just below the first merge: everything under it is empty
    bottom = np.floor(z[:, 2].min() * 10) / 10 - 0.05
    top = ax.get_ylim()[1]
    ax.set_ylim(bottom, top)
    xs = np.arange(5, 10 * len(dn["ivl"]), 10)
    ax.scatter(xs, np.full(len(xs), bottom - 0.02 * (top - bottom)), clip_on=False, zorder=3,
               s=46, c=[SECTOR_COLORS.get(SECTOR_OF.get(t), OTHER) for t in dn["ivl"]],
               edgecolors=SURFACE, linewidths=1)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_ylabel("Distance d_ij")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(title)
    sector_legend(ax, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return _save(fig, path)
