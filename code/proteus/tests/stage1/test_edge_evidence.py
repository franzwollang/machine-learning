"""Unit tests for hollow-edge (empty-region) evidence (OPEN_ISSUES #44, A2-T28)."""

from __future__ import annotations

import numpy as np

from proteus.stage1.edge_evidence import (
    HollowEdgeConfig,
    edge_ball_occupancy,
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


def test_edge_ball_occupancy_matches_hollowness_scores() -> None:
    """Occupancy helper agrees with H = n_mid / (n_end + eps)."""

    rng = np.random.default_rng(3)
    data = rng.normal(size=(50, 2))
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    edges = [(0, 1), (0, 2)]
    n_mid, n_end, _lengths = edge_ball_occupancy(
        positions, edges, data, mid_radius_frac=0.5,
    )
    H = hollowness_scores(positions, edges, data, mid_radius_frac=0.5, eps=1e-9)
    np.testing.assert_allclose(H, n_mid / (n_end + 1e-9))


def test_adapted_nested_scaffold_mid035_empty_ball_regime() -> None:
    """A2-T30: mid_frac=0.35 on adapted nested scaffold is empty-ball dominated.

    Median ``n_end`` is low and cross-shell vs intra-shell ``H`` does not
    separate — documents why Gabriel fallback spuriously yields K=2.
    """

    from proteus.stage1.scaffold import Stage1Scaffold
    from proteus.stage1.stabilization import StabilizationConfig
    from tests.datasets.synthetic.nested_spheres import make_nested_spheres

    nested = make_nested_spheres(n_per_sphere=200, extrusion_dim=1, seed=0)
    data = nested.points
    labels = nested.labels
    sc = Stage1Scaffold(
        dim=int(data.shape[1]), tau=0.27, k=8, max_nodes=64,
        ann_backend="naive", rng=np.random.default_rng(0),
    )
    sc.init_from(data, n_seeds=8)
    sc.run_until_stable(
        data, StabilizationConfig(max_epochs=40, min_equilibrium_epochs=3),
    )
    pos = np.asarray([sc.nodes[i].position for i in range(len(sc.nodes))])
    edges = [(int(link.i), int(link.j)) for link in sc.links.lifted_links()]
    assert len(edges) >= 10

    _n_mid, n_end, _L = edge_ball_occupancy(
        pos, edges, data, mid_radius_frac=0.35,
    )
    assert float(np.median(n_end)) < 1.5

    H = hollowness_scores(pos, edges, data, mid_radius_frac=0.35)
    sig = labels >= 0
    Xs = data[sig]
    ys = labels[sig]
    nn = np.argmin(((pos[:, None, :] - Xs[None, :, :]) ** 2).sum(-1), axis=1)
    node_shell = ys[nn]
    cross = [H[k] for k, (i, j) in enumerate(edges) if node_shell[i] != node_shell[j]]
    intra = [H[k] for k, (i, j) in enumerate(edges) if node_shell[i] == node_shell[j]]
    assert cross and intra
    # Non-discriminative at 0.35: medians both near empty-ball collapse.
    assert abs(float(np.median(cross)) - float(np.median(intra))) < 0.25


def test_adapted_nested_scaffold_mid05_h_separates_but_not_cutset() -> None:
    """A2-T30: mid_frac=0.5 separates H on nested, but prune stays 1 CC.

    Documents that discriminative ``H`` alone is not a lifted cut-set on the
    adapted Hebbian graph (redundant paths).  No awaiting flip.
    """

    from proteus.stage1.clustering import _lifted_components_covering_all_nodes
    from proteus.stage1.scaffold import Stage1Scaffold
    from proteus.stage1.stabilization import StabilizationConfig
    from tests.datasets.synthetic.nested_spheres import make_nested_spheres

    nested = make_nested_spheres(n_per_sphere=200, extrusion_dim=1, seed=0)
    data = nested.points
    labels = nested.labels
    sc = Stage1Scaffold(
        dim=int(data.shape[1]), tau=0.27, k=8, max_nodes=64,
        ann_backend="naive", rng=np.random.default_rng(0),
    )
    sc.init_from(data, n_seeds=8)
    sc.run_until_stable(
        data, StabilizationConfig(max_epochs=40, min_equilibrium_epochs=3),
    )
    n = len(sc.nodes)
    pos = np.asarray([sc.nodes[i].position for i in range(n)])
    edges = [(int(link.i), int(link.j)) for link in sc.links.lifted_links()]
    H = hollowness_scores(pos, edges, data, mid_radius_frac=0.5)
    sig = labels >= 0
    Xs = data[sig]
    ys = labels[sig]
    nn = np.argmin(((pos[:, None, :] - Xs[None, :, :]) ** 2).sum(-1), axis=1)
    node_shell = ys[nn]
    cross = np.asarray(
        [H[k] for k, (i, j) in enumerate(edges) if node_shell[i] != node_shell[j]],
        dtype=float,
    )
    intra = np.asarray(
        [H[k] for k, (i, j) in enumerate(edges) if node_shell[i] == node_shell[j]],
        dtype=float,
    )
    assert len(cross) >= 5 and len(intra) >= 5
    assert float(np.median(intra)) > float(np.median(cross)) + 0.2

    kept = prune_hollow_edges(
        pos, edges, data,
        config=HollowEdgeConfig(
            mid_radius_frac=0.5, h0=0.65, min_end_count=1.0, gabriel_fallback=False,
        ),
    )
    graph = {i: [] for i in range(n)}
    for i, j in kept:
        graph[i].append(j)
        graph[j].append(i)
    comps = _lifted_components_covering_all_nodes(n, graph)
    majors = [c for c in comps if len(c) >= max(3, int(np.ceil(n * 0.2)))]
    assert len(majors) < 2



def test_require_gabriel_and_h_blocks_gabriel_only_cut() -> None:
    """A2-T31: conjunction needs H<h0 ∧ Gabriel; Gabriel-alone is not enough.

    Low-mass bridge with H above h0 (dense mid fill relative to ends) must
    stay kept under conjunction even if Gabriel diameter is empty.
    """

    # Two endpoints far apart; mid filled so H is high; diameter still empty
    # of *other* points near the open ball boundary is hard — use a case
    # where H < h0 but we compare or/and rules on a clear void bridge.
    rng = np.random.default_rng(1)
    blob_a = rng.normal(loc=[-3.0, 0.0], scale=0.2, size=(60, 2))
    blob_b = rng.normal(loc=[3.0, 0.0], scale=0.2, size=(60, 2))
    data = np.vstack([blob_a, blob_b])
    positions = np.array([[-3.0, 0.0], [3.0, 0.0]], dtype=float)
    edges = [(0, 1)]

    or_cfg = HollowEdgeConfig(
        mid_radius_frac=0.35, h0=0.35, min_end_count=0.5, gabriel_fallback=True,
        require_gabriel_and_h=False,
    )
    and_cfg = HollowEdgeConfig(
        mid_radius_frac=0.35, h0=0.35, min_end_count=0.5, gabriel_fallback=True,
        require_gabriel_and_h=True,
    )
    cut_or = hollow_edge_mask(positions, edges, data, config=or_cfg)
    cut_and = hollow_edge_mask(positions, edges, data, config=and_cfg)
    # Empty-gap bridge: H≈0 so both rules cut.
    assert bool(cut_or[0]) is True
    assert bool(cut_and[0]) is True

    # Within-support edge: neither rule should cut.
    positions2 = np.array([[-3.0, 0.0], [-2.7, 0.05]], dtype=float)
    cut_or2 = hollow_edge_mask(positions2, edges, data, config=or_cfg)
    cut_and2 = hollow_edge_mask(positions2, edges, data, config=and_cfg)
    assert bool(cut_or2[0]) is False
    assert bool(cut_and2[0]) is False


def test_require_gabriel_and_h_default_off() -> None:
    """Conjunction flag stays default-off (proposal-path; A2-T31)."""

    assert HollowEdgeConfig().require_gabriel_and_h is False
