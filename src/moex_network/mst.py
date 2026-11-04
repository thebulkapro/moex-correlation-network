"""Minimum spanning tree: a from-scratch Kruskal plus NetworkX helpers."""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
import pandas as pd


class DisjointSet:
    """Union-find with path halving and union by size."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def kruskal_mst(dist: np.ndarray | pd.DataFrame, labels: Sequence[str] | None = None):
    """Kruskal (1956) on a complete graph given by a symmetric distance matrix.

    1. Every vertex starts in its own set.
    2. Sort all N(N-1)/2 edges by weight.
    3. Add an edge iff it joins two different sets (otherwise it closes a cycle).
    4. Stop after N-1 edges.

    Complexity O(E log E) with E = N(N-1)/2.

    Returns (edges, total_length) where edges is a list of (u, v, weight).
    """
    if isinstance(dist, pd.DataFrame):
        labels = list(dist.columns) if labels is None else list(labels)
        dist = dist.to_numpy()
    n = dist.shape[0]
    labels = list(range(n)) if labels is None else list(labels)

    iu, ju = np.triu_indices(n, k=1)
    order = np.argsort(dist[iu, ju], kind="stable")

    dsu = DisjointSet(n)
    edges, total = [], 0.0
    for k in order:
        i, j = int(iu[k]), int(ju[k])
        if dsu.union(i, j):
            w = float(dist[i, j])
            edges.append((labels[i], labels[j], w))
            total += w
            if len(edges) == n - 1:
                break
    return edges, total


def complete_graph(dist: pd.DataFrame) -> nx.Graph:
    """Complete weighted graph with weight = distance."""
    g = nx.Graph()
    cols = list(dist.columns)
    g.add_nodes_from(cols)
    values = dist.to_numpy()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            g.add_edge(cols[i], cols[j], weight=float(values[i, j]))
    return g


def graph_from_edges(edges, nodes: Sequence[str] | None = None) -> nx.Graph:
    g = nx.Graph()
    if nodes is not None:
        g.add_nodes_from(nodes)
    for u, v, w in edges:
        g.add_edge(u, v, weight=w, rho=1.0 - w**2 / 2.0)
    return g


def build_mst(dist: pd.DataFrame) -> nx.Graph:
    """MST via the hand-written Kruskal, with `weight` (=d) and `rho` edge attributes."""
    edges, _ = kruskal_mst(dist)
    return graph_from_edges(edges, nodes=dist.columns)


def edge_set(g: nx.Graph) -> set[frozenset]:
    return {frozenset(e) for e in g.edges()}


def edges_frame(g: nx.Graph) -> pd.DataFrame:
    rows = [
        {"source": u, "target": v, "distance": d["weight"], "rho": 1.0 - d["weight"] ** 2 / 2.0}
        for u, v, d in g.edges(data=True)
    ]
    return pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)
