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



def test_require_gabriel_and_h_is_conjunction() -> None:
    """A2-T31: ``require_gabriel_and_h`` cuts iff ``H < h0`` ∧ Gabriel-empty.

    Also: conjunction ⊆ OR-with-Gabriel on a void-bridge + within-edge set.
    """

    rng = np.random.default_rng(1)
    blob_a = rng.normal(loc=[-2.0, 0.0], scale=0.15, size=(80, 2))
    blob_b = rng.normal(loc=[2.0, 0.0], scale=0.15, size=(80, 2))
    data = np.vstack([blob_a, blob_b])
    positions = np.array(
        [
            [-2.0, 0.0],
            [2.0, 0.0],
            [-1.9, 0.05],
            [1.9, -0.05],
        ],
        dtype=float,
    )
    edges = [(0, 1), (0, 2), (1, 3)]
    or_cfg = HollowEdgeConfig(
        mid_radius_frac=0.35, h0=0.35, min_end_count=0.5, gabriel_fallback=True,
        require_gabriel_and_h=False,
    )
    and_cfg = HollowEdgeConfig(
        mid_radius_frac=0.35, h0=0.35, min_end_count=0.5, gabriel_fallback=True,
        require_gabriel_and_h=True,
    )
    H = hollowness_scores(positions, edges, data, mid_radius_frac=0.35)
    gab = gabriel_diameter_empty(positions, edges, data)
    cut_or = hollow_edge_mask(positions, edges, data, config=or_cfg)
    cut_and = hollow_edge_mask(positions, edges, data, config=and_cfg)
    for k in range(len(edges)):
        assert bool(cut_and[k]) == (bool(H[k] < 0.35) and bool(gab[k]))
    assert not np.any(cut_and & ~cut_or)
    # Void bridge is hollow under OR.
    assert bool(cut_or[0]) is True
    # Within-blob edges stay under conjunction.
    assert bool(cut_and[1]) is False
    assert bool(cut_and[2]) is False


def test_require_gabriel_and_h_default_off() -> None:
    """Conjunction flag stays default-off (proposal-path; A2-T31)."""

    assert HollowEdgeConfig().require_gabriel_and_h is False


def test_a4_roc_primary_config_preset() -> None:
    """#44 / A2-T33: A4 primary preset mid=0.5 h0=0.7 gabriel=False."""

    from proteus.stage1.edge_evidence import (
        A4_PRIMARY_GABRIEL_FALLBACK,
        A4_PRIMARY_H0,
        A4_PRIMARY_MID_RADIUS_FRAC,
        A4_PRIMARY_MIN_END_COUNT,
        a4_roc_primary_config,
    )

    cfg = a4_roc_primary_config()
    assert cfg.mid_radius_frac == A4_PRIMARY_MID_RADIUS_FRAC == 0.5
    assert cfg.h0 == A4_PRIMARY_H0 == 0.7
    assert cfg.min_end_count == A4_PRIMARY_MIN_END_COUNT == 0.5
    assert cfg.gabriel_fallback is A4_PRIMARY_GABRIEL_FALLBACK is False
    assert cfg.mst_critical_only is False
    assert cfg.bridge_critical_only is False
    assert HollowEdgeConfig().mst_critical_only is False
    assert HollowEdgeConfig().bridge_critical_only is False


def test_mst_critical_only_intersects_hollow_mask() -> None:
    """#44 / A2-T34: MST-critical hollow cuts only MST∩hollow edges.

    Two blobs linked by a long bridge plus a short redundant path: the
    Euclidean MST prefers the shorter support edges; a hollow long bridge
    is cut under H-only but kept when ``mst_critical_only`` (not in MST).
    """

    from proteus.stage1.edge_evidence import mst_edge_mask

    rng = np.random.default_rng(1)
    blob_a = rng.normal(loc=[-3.0, 0.0], scale=0.12, size=(60, 2))
    blob_b = rng.normal(loc=[3.0, 0.0], scale=0.12, size=(60, 2))
    data = np.vstack([blob_a, blob_b])
    # Nodes: A0, A1 close; B0, B1 close; long A0–B0 bridge; short A1–B1 chord.
    positions = np.array(
        [
            [-3.0, 0.0],
            [-2.85, 0.2],
            [3.0, 0.0],
            [2.85, 0.2],
        ],
        dtype=float,
    )
    edges = [(0, 1), (2, 3), (0, 2), (1, 3)]  # supports, long bridge, short chord
    lengths = [
        float(np.linalg.norm(positions[i] - positions[j])) for i, j in edges
    ]
    assert lengths[2] > lengths[3]  # long bridge longer than short chord
    mst = mst_edge_mask(positions, edges)
    # Long bridge should not be selected into the MST (shorter chord wins).
    assert bool(mst[2]) is False

    cfg_all = HollowEdgeConfig(
        mid_radius_frac=0.35, h0=0.35, min_end_count=0.5, gabriel_fallback=False,
        mst_critical_only=False,
    )
    cfg_mst = HollowEdgeConfig(
        mid_radius_frac=0.35, h0=0.35, min_end_count=0.5, gabriel_fallback=False,
        mst_critical_only=True,
    )
    cut_all = hollow_edge_mask(positions, edges, data, config=cfg_all)
    cut_mst = hollow_edge_mask(positions, edges, data, config=cfg_mst)
    # Long void bridge is hollow under H-only.
    assert bool(cut_all[2]) is True
    # Not MST-critical → MST-only mode does not cut it.
    assert bool(cut_mst[2]) is False
    assert not np.any(cut_mst & ~mst)
    assert not np.any(cut_mst & ~cut_all)


def test_bridge_critical_only_intersects_hollow_mask() -> None:
    """#44: bridge-critical hollow cuts only graph bridges ∩ hollow.

    Two blobs linked by a single long void bridge: the bridge is both MST
    and a graph-theoretic bridge.  Adding a redundant short chord removes
    the bridge property from the long edge — ``bridge_critical_only`` then
    keeps it (not a cut-set), while H-only still cuts it.
    """

    from proteus.stage1.edge_evidence import bridge_edge_mask, mst_edge_mask

    rng = np.random.default_rng(2)
    blob_a = rng.normal(loc=[-3.0, 0.0], scale=0.12, size=(60, 2))
    blob_b = rng.normal(loc=[3.0, 0.0], scale=0.12, size=(60, 2))
    data = np.vstack([blob_a, blob_b])
    positions = np.array(
        [
            [-3.0, 0.0],
            [-2.85, 0.2],
            [3.0, 0.0],
            [2.85, 0.2],
        ],
        dtype=float,
    )
    # Supports + single long bridge: long edge is a bridge.
    edges_single = [(0, 1), (2, 3), (0, 2)]
    br_single = bridge_edge_mask(edges_single, n_nodes=4)
    assert bool(br_single[2]) is True

    cfg_bridge = HollowEdgeConfig(
        mid_radius_frac=0.35, h0=0.35, min_end_count=0.5, gabriel_fallback=False,
        bridge_critical_only=True,
    )
    cut_single = hollow_edge_mask(positions, edges_single, data, config=cfg_bridge)
    assert bool(cut_single[2]) is True  # hollow + bridge → cut

    # Redundant short chord: long edge is no longer a bridge.
    edges_cycle = [(0, 1), (2, 3), (0, 2), (1, 3)]
    br_cycle = bridge_edge_mask(edges_cycle, n_nodes=4)
    assert bool(br_cycle[2]) is False
    mst = mst_edge_mask(positions, edges_cycle)
    # Long bridge still not in Euclidean MST (short chord wins).
    assert bool(mst[2]) is False

    cfg_all = HollowEdgeConfig(
        mid_radius_frac=0.35, h0=0.35, min_end_count=0.5, gabriel_fallback=False,
        bridge_critical_only=False,
    )
    cut_all = hollow_edge_mask(positions, edges_cycle, data, config=cfg_all)
    cut_br = hollow_edge_mask(positions, edges_cycle, data, config=cfg_bridge)
    assert bool(cut_all[2]) is True
    assert bool(cut_br[2]) is False  # not a bridge → keep under bridge-only
    assert not np.any(cut_br & ~br_cycle)
    assert not np.any(cut_br & ~cut_all)


def test_bridge_critical_default_off() -> None:
    """Bridge-critical flag stays default-off (proposal-path)."""

    assert HollowEdgeConfig().bridge_critical_only is False


def test_soft_capacity_only_intersects_hollow_mask() -> None:
    """#44 / A2-T37: soft-capacity keeps only high-betweenness ∩ hollow.

    Path graph 0-1-2-3: the middle edge has higher Brandes betweenness than
    the ends.  With ``soft_capacity_frac`` near 1.0 only the peak-betweenness
    edge may be cut; a hollow end-edge is suppressed.
    """

    from proteus.stage1.edge_evidence import (
        edge_betweenness_scores,
        soft_capacity_edge_mask,
    )

    # Path of 4 nodes: edges (0,1), (1,2), (2,3). Mid edge has highest bet.
    pos = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=float,
    )
    edges = [(0, 1), (1, 2), (2, 3)]
    # Data only near endpoints so mid and end edges look hollow under H.
    rng = np.random.default_rng(0)
    data = np.vstack([
        rng.normal([0.0, 0.0], 0.05, size=(40, 2)),
        rng.normal([3.0, 0.0], 0.05, size=(40, 2)),
    ])
    scores = edge_betweenness_scores(edges, n_nodes=4)
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]
    soft = soft_capacity_edge_mask(edges, n_nodes=4, frac=0.9)
    assert bool(soft[1]) is True
    assert bool(soft[0]) is False or bool(soft[2]) is False

    cfg_all = HollowEdgeConfig(
        mid_radius_frac=0.5, h0=0.9, gabriel_fallback=False,
        soft_capacity_only=False,
    )
    cfg_soft = HollowEdgeConfig(
        mid_radius_frac=0.5, h0=0.9, gabriel_fallback=False,
        soft_capacity_only=True, soft_capacity_frac=0.9,
    )
    cut_all = hollow_edge_mask(pos, edges, data, cfg_all)
    cut_soft = hollow_edge_mask(pos, edges, data, cfg_soft)
    # Soft capacity never cuts more than unrestricted hollow.
    assert not np.any(cut_soft & ~cut_all)
    assert not np.any(cut_soft & ~soft)
    assert HollowEdgeConfig().soft_capacity_only is False


def test_soft_capacity_default_off() -> None:
    """Soft-capacity flag stays default-off (proposal-path; A2-T37)."""

    assert HollowEdgeConfig().soft_capacity_only is False
    assert HollowEdgeConfig().soft_capacity_frac == 0.25
    assert HollowEdgeConfig().soft_capacity_method == "betweenness"


def test_bridge_mass_soft_capacity_method() -> None:
    """#44 / A2-T39: bridge_mass scores only bridges; soft gate by mass.

    Path 0-1-2-3: every edge is a bridge; middle edge has mass min(2,2)=2,
    ends have mass min(1,3)=1.  ``frac=0.9`` keeps only the peak-mass mid.
    """

    from proteus.stage1.edge_evidence import (
        bridge_mass_scores,
        soft_capacity_edge_mask,
    )

    pos = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=float,
    )
    edges = [(0, 1), (1, 2), (2, 3)]
    rng = np.random.default_rng(1)
    data = np.vstack([
        rng.normal([0.0, 0.0], 0.05, size=(40, 2)),
        rng.normal([3.0, 0.0], 0.05, size=(40, 2)),
    ])
    mass = bridge_mass_scores(edges, n_nodes=4)
    assert mass[1] > mass[0]
    assert mass[1] > mass[2]
    soft = soft_capacity_edge_mask(
        edges, n_nodes=4, frac=0.9, method="bridge_mass",
    )
    assert bool(soft[1]) is True
    assert bool(soft[0]) is False
    assert bool(soft[2]) is False

    cfg = HollowEdgeConfig(
        mid_radius_frac=0.5, h0=0.9, gabriel_fallback=False,
        soft_capacity_only=True, soft_capacity_frac=0.9,
        soft_capacity_method="bridge_mass",
    )
    cut = hollow_edge_mask(pos, edges, data, cfg)
    assert not np.any(cut & ~soft)
    assert HollowEdgeConfig().soft_capacity_method == "betweenness"def test_poisson_null_h0_calibration_export_table() -> None:
    """#44 / A2-T38: export Poisson-null sheet H quantiles for A3/A4 SI.

    Recomputes sheet-null quantiles via the A4 adversarial harness and
    checks the frozen export in ``edge_evidence`` stays within tolerance.
    Documents A4 primary ``h0=0.7 ≤ sheet q01`` at mid=0.5.  Defaults
    remain off; no awaiting flip.
    """

    from proteus.stage1.edge_evidence import (
        A4_PRIMARY_H0,
        A4_PRIMARY_MID_RADIUS_FRAC,
        POISSON_NULL_PRIMARY_H0,
        POISSON_NULL_PRIMARY_MID,
        POISSON_NULL_PRIMARY_SHEET_Q01,
        POISSON_NULL_SHEET_H_QUANTILES,
        POISSON_NULL_SHEET_N_EDGES,
        POISSON_NULL_SI_NOTE,
        a4_roc_primary_config,
        format_poisson_null_h0_table,
    )
    from tests.scenarios.synthetic.hollow_edge_nulls import sheet_null_h_quantiles

    assert HollowEdgeConfig().h0 == 0.35  # operational default unchanged
    assert a4_roc_primary_config().h0 == A4_PRIMARY_H0 == POISSON_NULL_PRIMARY_H0
    assert POISSON_NULL_PRIMARY_MID == A4_PRIMARY_MID_RADIUS_FRAC == 0.5

    for mid, snap in POISSON_NULL_SHEET_H_QUANTILES.items():
        live = sheet_null_h_quantiles(mid_radius_frac=float(mid), seed=0)
        assert live.n_edges == POISSON_NULL_SHEET_N_EDGES
        assert abs(live.mean_h - snap["mean_h"]) < 0.05
        for key in ("q0.01", "q0.05", "q0.1", "q0.25", "q0.5"):
            assert abs(live.quantiles[key] - snap[key]) < 0.08

    # Primary discipline: h0 at/below sheet lower tail at mid=0.5.
    q01_mid05 = POISSON_NULL_SHEET_H_QUANTILES[0.5]["q0.01"]
    assert POISSON_NULL_PRIMARY_H0 <= POISSON_NULL_PRIMARY_SHEET_Q01
    assert POISSON_NULL_PRIMARY_H0 <= q01_mid05 + 0.05  # allow snap rounding

    tsv = format_poisson_null_h0_table()
    assert tsv.splitlines()[0].startswith("mid\t")
    assert "primary mid=0.5" in tsv
    assert "sample-ARI" in POISSON_NULL_SI_NOTE
    assert "defaults off" in POISSON_NULL_SI_NOTE
