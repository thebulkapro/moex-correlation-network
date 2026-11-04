"""Topological metrics of the MST and sector-level summaries."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .config import SECTOR_OF


def node_metrics(tree: nx.Graph, sector_of: dict[str, str] = SECTOR_OF) -> pd.DataFrame:
    """Per-node degree, betweenness, closeness and eccentricity.

    Betweenness is computed on hops (in a tree the shortest path between two
    vertices is unique, so weights do not change it). Closeness and
    eccentricity use the distance d_ij as edge length.
    """
    degree = dict(tree.degree())
    betweenness = nx.betweenness_centrality(tree)
    closeness = nx.closeness_centrality(tree, distance="weight")
    eccentricity = nx.eccentricity(tree, weight="weight")
    df = pd.DataFrame({
        "sector": {n: sector_of.get(n, "Other") for n in tree.nodes()},
        "degree": degree,
        "betweenness": betweenness,
        "closeness": closeness,
        "eccentricity": eccentricity,
    })
    df.index.name = "ticker"
    return df.sort_values(["degree", "betweenness"], ascending=False)


def tree_summary(tree: nx.Graph) -> dict:
    """Global descriptors: length, diameter, radius, average path lengths, centre."""
    ecc = nx.eccentricity(tree, weight="weight")
    radius = min(ecc.values())
    diameter = max(ecc.values())
    return {
        "nodes": tree.number_of_nodes(),
        "edges": tree.number_of_edges(),
        "total_length": tree.size(weight="weight"),
        "mean_edge_rho": float(np.mean([d["rho"] for *_, d in tree.edges(data=True)])),
        "diameter": diameter,
        "diameter_hops": nx.diameter(tree),
        "radius": radius,
        "avg_path_length": nx.average_shortest_path_length(tree, weight="weight"),
        "avg_path_hops": nx.average_shortest_path_length(tree),
        "center": sorted(n for n, e in ecc.items() if np.isclose(e, radius)),
        "periphery": sorted(n for n, e in ecc.items() if np.isclose(e, diameter)),
    }


def classify_nodes(metrics: pd.DataFrame) -> dict[str, list[str]]:
    """Hubs: degree >= mean + 1 sd and betweenness >= mean. Leaves: degree == 1."""
    deg, bc = metrics["degree"], metrics["betweenness"]
    hubs = metrics.index[(deg >= deg.mean() + deg.std(ddof=0)) & (bc >= bc.mean())]
    leaves = metrics.index[deg == 1]
    return {"hubs": list(hubs), "leaves": list(leaves)}


def sector_edge_stats(tree: nx.Graph, sector_of: dict[str, str] = SECTOR_OF) -> pd.DataFrame:
    """Each MST edge labelled as intra- or inter-sector."""
    rows = []
    for u, v, d in tree.edges(data=True):
        su, sv = sector_of.get(u, "Other"), sector_of.get(v, "Other")
        rows.append({"u": u, "v": v, "sector_u": su, "sector_v": sv,
                     "rho": d["rho"], "intra": su == sv})
    return pd.DataFrame(rows).sort_values("rho", ascending=False).reset_index(drop=True)


def sector_cohesion(corr: pd.DataFrame, sector_of: dict[str, str] = SECTOR_OF) -> pd.DataFrame:
    """Mean pairwise correlation inside each sector vs. with the rest of the market."""
    rows = []
    sectors = pd.Series({t: sector_of.get(t, "Other") for t in corr.columns})
    for sector, members in sectors.groupby(sectors).groups.items():
        members = list(members)
        others = [t for t in corr.columns if t not in members]
        block = corr.loc[members, members].to_numpy()
        iu = np.triu_indices(len(members), k=1)
        rows.append({
            "sector": sector,
            "n": len(members),
            "mean_rho_within": float(block[iu].mean()) if len(members) > 1 else np.nan,
            "mean_rho_outside": float(corr.loc[members, others].to_numpy().mean()),
        })
    return pd.DataFrame(rows).set_index("sector").sort_values("mean_rho_within", ascending=False)
