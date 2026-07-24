"""Flag-complex construction tests (SI S4.1, S4.2, S4.5, S13.4).

The flag complex is the top-dimensional clique complex of the lifted Hebbian
graph: a ``d``-simplex is present exactly when its ``d+1`` vertices form a lifted
clique and the local intrinsic-dimension rule permits a ``d``-simplex in that star
(S4.1). Greedy Chaining (S4.2 / S13.4) enumerates rank-ordered ``d``-subsets per
seed node, verifies clique completion, and deduplicates canonical simplices. These
tests pin the construction on small graphs with known cliques, including a
heterogeneous-dimension junction.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from proteus.stage2.flag_complex import (
    FlagComplexConfig,
    build_flag_complex,
    simplex_volume,
)


def _complete_graph(n: int) -> dict[int, list[int]]:
    return {i: [j for j in range(n) if j != i] for i in range(n)}


def test_triangle_is_single_two_simplex() -> None:
    """A 3-clique with d_final=2 yields exactly one 2-simplex (S4.1)."""
    graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    res = build_flag_complex(graph, [2, 2, 2], pos)

    assert res.simplices == [(0, 1, 2)]
    assert res.complex.intrinsic_dim == 2
    assert list(res.incidence_counts) == [1, 1, 1]
    assert res.orphan_ids == []
    # Volume of the unit right triangle is 1/2.
    assert res.complex.simplices[0].volume == pytest.approx(0.5)


def test_k4_is_single_three_simplex() -> None:
    """A 4-clique with d_final=3 yields exactly one 3-simplex (S4.1)."""
    graph = _complete_graph(4)
    pos = np.eye(4)[:, :3]  # 4 vertices; but need R^3 unit-corner tetra
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    res = build_flag_complex(graph, [3, 3, 3, 3], pos)

    assert res.simplices == [(0, 1, 2, 3)]
    assert res.complex.intrinsic_dim == 3
    # Volume of the unit corner tetrahedron is 1/6.
    assert res.complex.simplices[0].volume == pytest.approx(1.0 / 6.0)


def test_cycle_at_dim_one_has_only_edges() -> None:
    """A 1-manifold (d_final=1) flag complex is its 1-skeleton: edges, no faces."""
    n = 6
    graph = {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}
    pos = np.array(
        [[math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)] for i in range(n)]
    )
    res = build_flag_complex(graph, [1] * n, pos)

    assert all(len(s) == 2 for s in res.simplices)
    assert res.complex.intrinsic_dim == 1
    # A closed 1-ring has exactly n edges.
    assert len(res.simplices) == n
    assert res.orphan_ids == []


def test_dim_one_ignores_triangle_cliques() -> None:
    """d_final=1 builds only edges even when a 3-clique is present (S4.1 rule)."""
    graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    res = build_flag_complex(graph, [1, 1, 1], pos)

    assert all(len(s) == 2 for s in res.simplices)
    assert set(res.simplices) == {(0, 1), (0, 2), (1, 2)}
    assert res.complex.intrinsic_dim == 1


def test_heterogeneous_junction_dimension() -> None:
    """A 2-simplex patch meeting a 1-simplex chain keeps per-star dimension (S4.2).

    Nodes 0,1,2 form a triangle (d_final=2); node 2 also anchors an edge chain
    2-3-4 (d_final=1). The complex must carry the 2-simplex and the two edges.
    """
    graph = {0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2, 4], 4: [3]}
    pos = np.array([[0, 0], [1, 0], [0, 1], [-1, 1], [-2, 1]], dtype=float)
    res = build_flag_complex(graph, [2, 2, 2, 1, 1], pos)

    assert (0, 1, 2) in res.simplices
    assert (2, 3) in res.simplices
    assert (3, 4) in res.simplices
    # No spurious higher simplex on the 1-D chain.
    assert all(len(s) <= 3 for s in res.simplices)
    # Node 2 (the junction) participates in the triangle and the edge to 3.
    assert res.incidence_counts[2] == 2
    assert res.orphan_ids == []


def test_canonical_dedup_across_seeds() -> None:
    """The same simplex discovered from multiple seeds is inserted once (S13.4)."""
    graph = _complete_graph(4)
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    res = build_flag_complex(graph, [2, 2, 2, 2], pos)

    # d_final=2 over K4: the four triangular faces, each stored once, sorted.
    assert len(res.simplices) == len(set(res.simplices))
    assert all(tuple(sorted(s)) == s for s in res.simplices)
    for s in res.simplices:
        assert len(s) == 3


def test_isolated_node_is_orphan() -> None:
    """A node with fewer neighbours than its dimension is logged as an orphan."""
    # Node 3 is isolated; nodes 0,1,2 form a triangle at d_final=2.
    graph = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: []}
    pos = np.array([[0, 0], [1, 0], [0, 1], [5, 5]], dtype=float)
    res = build_flag_complex(graph, [2, 2, 2, 2], pos)

    assert 3 in res.orphan_ids
    assert res.incidence_counts[3] == 0


def test_max_simplex_dim_cap() -> None:
    """The operational cap clamps the enumerated dimension below d_final."""
    graph = _complete_graph(5)
    pos = np.random.default_rng(0).normal(size=(5, 4))
    capped = build_flag_complex(
        graph, [4, 4, 4, 4, 4], pos, config=FlagComplexConfig(max_simplex_dim=2)
    )
    assert all(len(s) <= 3 for s in capped.simplices)
    assert capped.complex.intrinsic_dim <= 2


def test_simplex_volume_degenerate_is_zero() -> None:
    """Collinear / coincident vertices yield zero d-volume (S4.5)."""
    collinear = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert simplex_volume(collinear) == pytest.approx(0.0)
    assert simplex_volume(np.array([[1.0, 2.0]])) == 0.0
