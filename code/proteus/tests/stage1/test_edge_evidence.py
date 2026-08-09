"""Unit tests for hollow-edge (empty-region) evidence (OPEN_ISSUES #44, A2-T28)."""

from __future__ import annotations

import numpy as np

from proteus.stage1.edge_evidence import (
    HollowEdgeConfig,
    gabriel_diameter_empty,
    hollowness_scores,
    hollow_edge_mask,
    prune_hollow_edges,
)


def test_hollowness_scores_bridge_vs_support() -> None:
    """Bridge over a void has H≈0; within-blob edge has H=O(1)."""

    rng = np.random.default_rng(0)
    blob_a = rng.normal(loc=[-2.0, 0.0], scale=0.15, size=(80, 2))
    blob_b = rng.normal(loc=[2.0, 0.0], scale=0.15, size=(80, 2))
    data = np.vstack([blob_a, blob_b])
    # Scaffold endpoints: within A, within B, and a cross bridge.
    positions = np.array(
        [
            [-2.0, 0.0],
            [-1.9, 0.05],
            [2.0, 0.0],
            [1.9, -0.05],
        ],
        dtype=float,
    )
    edges = [(0, 1), (2, 3), (0, 2)]  # support, support, bridge
    H = hollowness_scores(positions, edges, data, mid_radius_frac=0.35)
    assert H[0] > H[2]
    assert H[1] > H[2]
    assert H[2] < 0.35
    assert H[0] > 0.5


def test_gabriel_fallback_cuts_empty_diameter() -> None:
    """Empty diameter ball ⇒ Gabriel-empty True; filled ball ⇒ False."""

    data = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]], dtype=float)
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
    # Edge 0-1 has midpoint sample; edge 2-3 spans empty space above.
    edges = [(0, 1), (2, 3)]
    empty = gabriel_diameter_empty(positions, edges, data)
    assert empty[0] is np.False_ or empty[0] == False
    assert bool(empty[1]) is True


def test_prune_hollow_edges_separates_two_blobs() -> None:
    """Pruning hollow bridges leaves two components on a toy lifted graph."""

    rng = np.random.default_rng(1)
    blob_a = rng.normal(loc=[-3.0, 0.0], scale=0.2, size=(100, 2))
    blob_b = rng.normal(loc=[3.0, 0.0], scale=0.2, size=(100, 2))
    data = np.vstack([blob_a, blob_b])
    # 3+3 nodes with dense within edges + two bridges.
    positions = np.array(
        [
            [-3.1, 0.0], [-3.0, 0.1], [-2.9, -0.1],
            [2.9, 0.1], [3.0, 0.0], [3.1, -0.1],
        ],
        dtype=float,
    )
    edges = [
        (0, 1), (1, 2), (0, 2),
        (3, 4), (4, 5), (3, 5),
        (0, 3), (2, 5),  # bridges
    ]
    kept = prune_hollow_edges(
        positions, edges, data,
        config=HollowEdgeConfig(mid_radius_frac=0.35, h0=0.35, min_end_count=0.5),
    )
    # Bridges should be gone; within-blob edges retained.
    assert (0, 3) not in kept and (3, 0) not in kept
    assert (2, 5) not in kept and (5, 2) not in kept
    assert (0, 1) in kept
    assert (3, 4) in kept


def test_swiss_guard_shortcuts_cut_sheet_stays_connected() -> None:
    """Inter-wrap shortcut edges are hollow; the sheet path stays one CC.

    Synthetic 2-D 'swiss': two parallel arcs (wraps) with along-arc edges
    plus a chord shortcut across the gap.  Hollow prune must cut the
    shortcut while leaving each arc's chain intact and — via end
    connections representing the roll's continuous sheet in this toy —
    a single major component when wraps share an endpoint bridge that
    *is* occupied (the roll's intrinsic path).
    """

    rng = np.random.default_rng(2)
    # Dense samples along two parallel lines (wraps) connected at one end.
    t = np.linspace(0.0, 1.0, 60)
    wrap0 = np.column_stack([t, np.zeros_like(t)]) + rng.normal(0, 0.01, (60, 2))
    wrap1 = np.column_stack([t, np.full_like(t, 1.0)]) + rng.normal(0, 0.01, (60, 2))
    # Intrinsic sheet connection near t=0 (roll continuity).
    junction = np.column_stack([
        np.full(20, 0.0),
        np.linspace(0.0, 1.0, 20),
    ]) + rng.normal(0, 0.01, (20, 2))
    data = np.vstack([wrap0, wrap1, junction])

    # Scaffold: 4 nodes on wrap0, 4 on wrap1, plus junction node.
    positions = np.array(
        [
            [0.0, 0.0], [0.33, 0.0], [0.66, 0.0], [1.0, 0.0],
            [0.0, 1.0], [0.33, 1.0], [0.66, 1.0], [1.0, 1.0],
        ],
        dtype=float,
    )
    along = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (0, 4)]  # sheet path
    shortcuts = [(1, 5), (2, 6), (3, 7)]  # inter-wrap chords over void
    edges = along + shortcuts
    mask = hollow_edge_mask(
        positions, edges, data,
        config=HollowEdgeConfig(mid_radius_frac=0.35, h0=0.35, min_end_count=0.5),
    )
    cut = {e for e, c in zip(edges, mask) if c}
    # Shortcuts should be cut; along-arc + junction kept.
    assert (1, 5) in cut or (5, 1) in cut
    assert (2, 6) in cut or (6, 2) in cut
    assert (0, 1) not in cut
    assert (0, 4) not in cut

    kept = [e for e in edges if e not in cut]
    # Connected components on 8 nodes.
    parent = list(range(8))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in kept:
        union(i, j)
    n_cc = len({find(i) for i in range(8)})
    assert n_cc == 1
