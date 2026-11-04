import networkx as nx
import numpy as np
import pandas as pd
import pytest

from moex_network.correlation import correlation_to_distance, distance_to_correlation
from moex_network.mst import build_mst, complete_graph, edge_set, kruskal_mst


def random_distance(n, seed):
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal((250, n)) + rng.standard_normal((250, 1))  # common factor
    labels = [f"T{i}" for i in range(n)]
    corr = pd.DataFrame(returns, columns=labels).corr()
    return correlation_to_distance(corr)


@pytest.mark.parametrize("seed", range(5))
def test_kruskal_matches_networkx(seed):
    dist = random_distance(25, seed)
    ours = build_mst(dist)
    ref = nx.minimum_spanning_tree(complete_graph(dist))
    assert edge_set(ours) == edge_set(ref)
    assert ours.size(weight="weight") == pytest.approx(ref.size(weight="weight"))


def test_tree_shape():
    dist = random_distance(30, 42)
    edges, total = kruskal_mst(dist)
    g = build_mst(dist)
    assert len(edges) == 29
    assert nx.is_tree(g)
    assert total == pytest.approx(sum(w for *_, w in edges))


def test_small_known_case():
    d = np.array([[0, 1, 4, 3],
                  [1, 0, 2, 5],
                  [4, 2, 0, 6],
                  [3, 5, 6, 0]], dtype=float)
    edges, total = kruskal_mst(d, labels="ABCD")
    assert {frozenset(e[:2]) for e in edges} == {frozenset("AB"), frozenset("BC"), frozenset("AD")}
    assert total == 6


def test_distance_roundtrip_and_bounds():
    corr = pd.DataFrame([[1.0, 0.5, -1.0], [0.5, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    d = correlation_to_distance(corr)
    assert np.allclose(np.diag(d), 0)
    assert d.iloc[0, 2] == pytest.approx(2.0)
    assert d.iloc[1, 2] == pytest.approx(np.sqrt(2))
    assert np.allclose(distance_to_correlation(d), corr)
