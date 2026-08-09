"""Integration test for the Stage 1 recursion orchestrator."""

from __future__ import annotations

import numpy as np
import pytest

from proteus.stage1.controller import ScaleSearchConfig
from proteus.stage1.recursion import (
    RecursionConfig,
    RecursionNode,
    RecursionTree,
    run_recursive_discovery,
)
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.ground_truth import ClusterNode
from tests.datasets.synthetic.circles import make_circle
from tests.datasets.synthetic.hierarchical_gaussian import make_hierarchical_gaussian
from tests.harness.hierarchy_recovery import (
    adjusted_rand_vs_coarse_fine,
    assert_fine_ari_at_least,
    assert_leaf_partition_covers_dataset,
    assert_recursion_matches_gt_hierarchy_unimodal_levels,
    assert_terminal_leaf_count_equals_fine_components,
    leaf_partition_by_region_id,
    per_sample_leaf_labels,
)


def test_hierarchical_gaussian_recursion_matches_gt() -> None:
    """Recursion vs hierarchical GT: six fine leaves, ARI, full-depth unimodal harness."""

    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=0,
    )
    data = dataset.points
    all_labels = np.asarray(dataset.labels, dtype=int)
    gt = dataset.ground_truth
    assert gt.topology is not None
    n_fine = int(gt.topology.connected_components)
    grid = gt.tau_grid_hint
    assert grid is not None, "hierarchical Gaussian fixture must set tau_grid_hint"
    tau_lo, tau_hi = grid
    n = int(data.shape[0])
    blob_mask = all_labels >= 0
    fine = all_labels[blob_mask]
    coarse = fine // 2

    config = RecursionConfig(
        scale_search=ScaleSearchConfig(
            tau_min=tau_lo,
            tau_max=tau_hi,
            max_grid_points=4,
            k=8,
            n_seeds=8,
            ann_backend="naive",
            stabilization=StabilizationConfig(
                min_equilibrium_epochs=2,
                max_epochs=8,
            ),
            seed=42,
        ),
        min_samples=30,
        max_depth=3,
        seed=42,
    )

    tree = run_recursive_discovery(data, dim=4, config=config)
    root = tree.nodes[0]

    assert_leaf_partition_covers_dataset(tree, n)
    assert_terminal_leaf_count_equals_fine_components(tree, n_fine=n_fine)
    assert root.n_clusters >= 3, (
        f"expected >= 3 root clusters (GT has 3 coarse, 6 fine), got {root.n_clusters}"
    )

    leaf_part = leaf_partition_by_region_id(tree)
    leaf_y = per_sample_leaf_labels(n, leaf_part)
    leaf_y_blobs = leaf_y[blob_mask]
    ari_c, _ = adjusted_rand_vs_coarse_fine(leaf_y_blobs, coarse, fine)

    # A flat 6-leaf refinement of a 3-coarse partition gives ARI ~0.57;
    # a hierarchical tree (3 coarse leaves) would give ~1.0.  Accept both.
    assert ari_c >= 0.55, f"coarse ARI too low: {ari_c:.4f}"
    assert_fine_ari_at_least(leaf_y_blobs, fine, min_ari=0.95)

    assert_recursion_matches_gt_hierarchy_unimodal_levels(
        data, tree, gt.cluster_hierarchy,
        min_samples=5,
        levels={0},
        required_levels={0},
    )


def test_persistence_gate_circle_is_single_feature() -> None:
    """Persistence accept-gate (SI S2.6.2): a circle has no persistent split.

    Under ``require_persistent_split`` a uniform ring produces no multi-cluster
    partition that survives across adjacent scales, so the region is terminal ---
    the recursion returns a single leaf without relying on the single-scale
    cleanup heuristics of S2.6.1 (OPEN_ISSUES #27).
    """

    dataset = make_circle(
        n_samples=1200, radius=1.0, noise=0.02, extrusion_dim=2, seed=21,
    )
    data = dataset.points
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint

    config = RecursionConfig(
        scale_search=ScaleSearchConfig(
            tau_min=tau_lo,
            tau_max=tau_hi,
            max_grid_points=8,
            k=8,
            n_seeds=8,
            ann_backend="naive",
            stabilization=StabilizationConfig(
                min_equilibrium_epochs=3,
                max_epochs=15,
            ),
            seed=77,
        ),
        min_samples=100,
        max_depth=3,
        require_persistent_split=True,
        seed=77,
    )

    tree = run_recursive_discovery(data, dim=gt.ambient_dim, config=config)

    # No persistent split -> the ring is one intrinsic feature (a single leaf).
    assert len(tree.nodes) == 1
    assert tree.nodes[0].is_leaf
    assert len(tree.leaves) == 1


def test_persistence_gate_hierarchy_matches_gt() -> None:
    """Persistence accept-gate (SI S2.6.2): hierarchy still resolves six leaves.

    The gate must not suppress genuine multi-modal structure: the 3-coarse /
    6-fine hierarchical Gaussian recurses to six terminal leaves with high fine
    ARI, exactly as under the default (heuristic) acceptance path.
    """

    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=0,
    )
    data = dataset.points
    all_labels = np.asarray(dataset.labels, dtype=int)
    gt = dataset.ground_truth
    assert gt.topology is not None
    n_fine = int(gt.topology.connected_components)
    tau_lo, tau_hi = gt.tau_grid_hint
    n = int(data.shape[0])
    blob_mask = all_labels >= 0
    fine = all_labels[blob_mask]
    coarse = fine // 2

    config = RecursionConfig(
        scale_search=ScaleSearchConfig(
            tau_min=tau_lo,
            tau_max=tau_hi,
            max_grid_points=4,
            k=8,
            n_seeds=8,
            ann_backend="naive",
            stabilization=StabilizationConfig(
                min_equilibrium_epochs=2,
                max_epochs=8,
            ),
            seed=42,
        ),
        min_samples=30,
        max_depth=3,
        require_persistent_split=True,
        seed=42,
    )

    tree = run_recursive_discovery(data, dim=4, config=config)
    root = tree.nodes[0]

    assert_leaf_partition_covers_dataset(tree, n)
    assert_terminal_leaf_count_equals_fine_components(tree, n_fine=n_fine)
    assert root.n_clusters >= 3

    leaf_part = leaf_partition_by_region_id(tree)
    leaf_y = per_sample_leaf_labels(n, leaf_part)
    leaf_y_blobs = leaf_y[blob_mask]
    ari_c, _ = adjusted_rand_vs_coarse_fine(leaf_y_blobs, coarse, fine)
    assert ari_c >= 0.55, f"coarse ARI too low: {ari_c:.4f}"
    assert_fine_ari_at_least(leaf_y_blobs, fine, min_ari=0.95)


def _build_three_level_gt(sigma2: float = 0.01) -> list[ClusterNode]:
    """Root(L0) -> one coarse(L1) -> two fine leaves(L2), centers ±1 on axis 0."""

    cov_leaf = np.eye(2, dtype=float) * float(sigma2)
    cov_big = np.eye(2, dtype=float)
    return [
        ClusterNode(cluster_id=0, level=0, parent_id=None, weight=1.0,
                    center=np.zeros(2), covariance=cov_big, is_leaf=False),
        ClusterNode(cluster_id=1, level=1, parent_id=0, weight=1.0,
                    center=np.zeros(2), covariance=cov_big, is_leaf=False),
        ClusterNode(cluster_id=2, level=2, parent_id=1, weight=0.5,
                    center=np.array([-1.0, 0.0]), covariance=cov_leaf, is_leaf=True),
        ClusterNode(cluster_id=3, level=2, parent_id=1, weight=0.5,
                    center=np.array([1.0, 0.0]), covariance=cov_leaf, is_leaf=True),
    ]


def _build_three_level_tree(tau_leaf: float) -> tuple[np.ndarray, RecursionTree]:
    """Matching recursion tree with a *fixed* leaf-level ``tau_star`` and a small
    mean offset baked into the samples.

    Root/mid frames carry a large ``tau_star`` (1.0); the two level-2 leaves carry
    ``tau_leaf``.  Each leaf's 50 samples sit at ``center + [0.15, 0]`` plus tiny
    jitter, so the leaf mean is displaced by 0.15 from the GT fine center.
    """

    rng = np.random.default_rng(0)
    offset = np.array([0.15, 0.0])
    a = np.array([-1.0, 0.0]) + offset + rng.normal(scale=0.08, size=(50, 2))
    b = np.array([1.0, 0.0]) + offset + rng.normal(scale=0.08, size=(50, 2))
    data = np.vstack([a, b])
    all_idx = np.arange(100, dtype=int)
    nodes = [
        RecursionNode(region_id=0, level=0, parent_id=None, tau_star=1.0,
                      n_samples=100, dim=2, n_clusters=1, children=[1],
                      is_leaf=False, sample_indices=all_idx.copy()),
        RecursionNode(region_id=1, level=1, parent_id=0, tau_star=1.0,
                      n_samples=100, dim=2, n_clusters=2, children=[2, 3],
                      is_leaf=False, sample_indices=all_idx.copy()),
        RecursionNode(region_id=2, level=2, parent_id=1, tau_star=float(tau_leaf),
                      n_samples=50, dim=2, n_clusters=1, children=[],
                      is_leaf=True, sample_indices=np.arange(0, 50, dtype=int)),
        RecursionNode(region_id=3, level=2, parent_id=1, tau_star=float(tau_leaf),
                      n_samples=50, dim=2, n_clusters=1, children=[],
                      is_leaf=True, sample_indices=np.arange(50, 100, dtype=int)),
    ]
    return data, RecursionTree(nodes=nodes)


def test_unimodal_harness_uses_per_frame_tau() -> None:
    """The unimodal harness must smooth GT at each frame's own ``tau_star`` (#31, SI S2.5.4).

    A 0.15 leaf-mean displacement is *significant* at the fine leaf scale
    (``Σ_smooth = 0.01·I``  →  Hotelling ≈ 56 ≫ χ²₀.₉₅) but *insignificant* at the
    root scale (``Σ_smooth = 1.0·I``  →  Hotelling ≈ 1.1).  So the harness raises iff
    it consults the per-frame leaf ``tau_star``; a global root-scale harness would
    silently pass.  The large-``tau_leaf`` control confirms the failure is scale-driven,
    not offset-driven.
    """

    hierarchy = _build_three_level_gt()

    # Fine leaf scale: the displacement is significant -> the gate must fire.
    data_fine, tree_fine = _build_three_level_tree(tau_leaf=0.01)
    with pytest.raises(AssertionError, match="Hotelling"):
        assert_recursion_matches_gt_hierarchy_unimodal_levels(
            data_fine, tree_fine, hierarchy,
            min_samples=5, levels={2}, required_levels={2},
        )

    # Control: with a coarse leaf scale the same displacement is within tolerance.
    data_coarse, tree_coarse = _build_three_level_tree(tau_leaf=1.0)
    assert_recursion_matches_gt_hierarchy_unimodal_levels(
        data_coarse, tree_coarse, hierarchy,
        min_samples=5, levels={2}, required_levels={2},
    )


def test_finer_research_flag_defaults_off() -> None:
    """#44: allow_finer_research is proposed/operational and default-off."""

    cfg = RecursionConfig()
    assert cfg.allow_finer_research is False
    assert 0.0 < cfg.finer_tau_cap_ratio < 1.0
    assert cfg.max_finer_scale_steps >= 1
    assert cfg.prefer_disconnected_prepass is False
    assert 0.0 < cfg.finer_prepass_min_frac <= 0.5
    assert cfg.prefer_radial_gap_prepass is False
    assert cfg.finer_radial_min_gap_ratio > 0.0
    assert cfg.prefer_radial_band_prepass is False
    assert cfg.finer_radial_hist_bins >= 8
    assert cfg.prefer_noncentroid_radial_band_prepass is False
    assert cfg.finer_radial_min_trough_rel == 0.0
    assert cfg.prefer_signal_density_band_prepass is False
    assert 0.0 < cfg.finer_signal_density_keep_frac <= 1.0
    assert cfg.prefer_pca_axis_gap_prepass is False


def test_major_lifted_component_partition_requires_two_majors() -> None:
    """#44c: prepass returns None unless ≥2 major lifted components exist."""

    from proteus.stage1.recursion import _major_lifted_component_partition
    from proteus.types import Link

    class _Links:
        def __init__(self, edges: list[tuple[int, int]]):
            self._edges = edges

        def neighbour_graph(self, n: int) -> dict[int, list[int]]:
            g = {i: [] for i in range(n)}
            for i, j in self._edges:
                g[i].append(j)
                g[j].append(i)
            return g

        def lifted_links(self):
            return [
                Link(i=i, j=j, count_ij=1.0, count_ji=1.0, lifted=True)
                for i, j in self._edges
            ]

    class _Node:
        def __init__(self, pos, hits=1.0):
            self.position = np.asarray(pos, dtype=float)
            self.hit_count = hits
            self.d_final = 1

    class _Scaf:
        def __init__(self, nodes, edges):
            self.nodes = nodes
            self.links = _Links(edges)
            self.tau = 0.1

    # One connected component of 6 nodes — no prepass hit.
    nodes = [_Node([float(i), 0.0]) for i in range(6)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    assert _major_lifted_component_partition(
        _Scaf(nodes, edges), min_frac=0.2, min_abs=2,
    ) is None

    # Two majors of 3 + isolated tiny absorbed into nearest.
    nodes = [
        _Node([0.0, 0.0]), _Node([0.1, 0.0]), _Node([0.2, 0.0]),
        _Node([5.0, 0.0]), _Node([5.1, 0.0]), _Node([5.2, 0.0]),
        _Node([0.15, 0.05]),  # tiny near first major
    ]
    edges = [(0, 1), (1, 2), (3, 4), (4, 5)]
    pre = _major_lifted_component_partition(
        _Scaf(nodes, edges), min_frac=0.2, min_abs=2,
    )
    assert pre is not None
    assert pre.n_clusters == 2
    assert pre.partition_q_score > 0.0
    # Tiny node 6 absorbed into cluster of nodes 0-2.
    assert int(pre.labels[6]) == int(pre.labels[0])
    assert int(pre.labels[0]) != int(pre.labels[3])


def test_radial_gap_partition_recovers_concentric_rings() -> None:
    """#44: radial-gap prepass splits concentric shells that stay lifted-CC."""

    from proteus.stage1.recursion import _radial_gap_partition
    from proteus.types import Link

    class _Links:
        def __init__(self, edges: list[tuple[int, int]]):
            self._edges = edges

        def neighbour_graph(self, n: int) -> dict[int, list[int]]:
            g = {i: [] for i in range(n)}
            for i, j in self._edges:
                g[i].append(j)
                g[j].append(i)
            return g

        def lifted_links(self):
            return [
                Link(i=i, j=j, count_ij=1.0, count_ji=1.0, lifted=True)
                for i, j in self._edges
            ]

    class _Node:
        def __init__(self, pos, hits=1.0):
            self.position = np.asarray(pos, dtype=float)
            self.hit_count = hits
            self.d_final = 1

    class _Scaf:
        def __init__(self, nodes, edges):
            self.nodes = nodes
            self.links = _Links(edges)
            self.tau = 0.1

    # Two rings (r≈1 and r≈3), fully connected so lifted CC = 1.
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    inner = [ _Node([np.cos(a), np.sin(a)]) for a in angles ]
    outer = [ _Node([3.0 * np.cos(a), 3.0 * np.sin(a)]) for a in angles ]
    nodes = inner + outer
    # Cycle within each ring + one bridge (still one CC, but radial gap clear).
    edges = (
        [(i, (i + 1) % 8) for i in range(8)]
        + [(8 + i, 8 + ((i + 1) % 8)) for i in range(8)]
        + [(0, 8)]
    )
    pre = _radial_gap_partition(
        _Scaf(nodes, edges), min_frac=0.2, min_abs=3, min_gap_ratio=0.25,
    )
    assert pre is not None
    assert pre.n_clusters == 2
    assert pre.partition_q_score > 0.0
    assert set(int(x) for x in pre.labels[:8]) == {0} or set(
        int(x) for x in pre.labels[:8]
    ) == {1}
    assert int(pre.labels[0]) != int(pre.labels[8])

    # Single ring: no large radial gap → None at default min_gap_ratio.
    single = [ _Node([np.cos(a), np.sin(a)]) for a in angles ]
    single_edges = [(i, (i + 1) % 8) for i in range(8)]
    assert _radial_gap_partition(
        _Scaf(single, single_edges), min_frac=0.2, min_abs=2, min_gap_ratio=0.25,
    ) is None


def test_radial_band_gap_partition_ignores_midband_bridges() -> None:
    """#44: trough-masked radial gap recovers shells when mid-band fills continuum."""

    from proteus.stage1.recursion import (
        _radial_band_gap_partition,
        _radial_gap_partition,
    )
    from proteus.types import Link

    class _Links:
        def __init__(self, edges: list[tuple[int, int]]):
            self._edges = edges

        def neighbour_graph(self, n: int) -> dict[int, list[int]]:
            g = {i: [] for i in range(n)}
            for i, j in self._edges:
                g[i].append(j)
                g[j].append(i)
            return g

        def lifted_links(self):
            return [
                Link(i=i, j=j, count_ij=1.0, count_ji=1.0, lifted=True)
                for i, j in self._edges
            ]

    class _Node:
        def __init__(self, pos, hits=1.0):
            self.position = np.asarray(pos, dtype=float)
            self.hit_count = hits
            self.d_final = 1

    class _Scaf:
        def __init__(self, nodes, edges):
            self.nodes = nodes
            self.links = _Links(edges)
            self.tau = 0.1

    # Concentric shells at r=1 and r=3 with a dense mid-radius continuum
    # that dilutes plain radial gaps; mid nodes stay unlinked so Q reflects
    # the shell cut (e2e scaffolds carry Hebbian weights instead).
    def _ring(radius: float, count: int) -> list:
        angs = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        return [
            _Node([radius * np.cos(a), radius * np.sin(a)]) for a in angs
        ]

    inner = _ring(1.0, 12)
    outer = _ring(3.0, 12)
    mid: list = []
    for rm in np.linspace(1.4, 2.6, 7):
        mid.extend(_ring(float(rm), 4))
    nodes = inner + outer + mid
    edges = (
        [(i, j) for i in range(12) for j in range(i + 1, 12)]
        + [(12 + i, 12 + j) for i in range(12) for j in range(i + 1, 12)]
        + [(0, 12)]  # one shell bridge → single lifted CC
    )
    scaf = _Scaf(nodes, edges)
    # Plain radial gap fails (continuum) at the default ratio.
    assert _radial_gap_partition(
        scaf, min_frac=0.15, min_abs=3, min_gap_ratio=0.25,
    ) is None
    pre = _radial_band_gap_partition(
        scaf, min_frac=0.15, min_abs=3, min_gap_ratio=0.25, hist_bins=12,
    )
    assert pre is not None
    assert pre.n_clusters == 2
    assert pre.partition_q_score > 0.0
    # Inner ring (0..11) single label, distinct from outer (12..23).
    assert len(set(int(x) for x in pre.labels[:12])) == 1
    assert len(set(int(x) for x in pre.labels[12:24])) == 1
    assert int(pre.labels[0]) != int(pre.labels[12])


def test_radial_band_trough_gate_and_coord_median_origin() -> None:
    """#44 / A2-T13: trough-depth gate rejects weak bimodality; coord-median recovers."""

    from proteus.stage1.recursion import (
        _coordinate_median,
        _radial_band_gap_partition,
    )
    from proteus.types import Link

    class _Links:
        def __init__(self, edges: list[tuple[int, int]]):
            self._edges = edges

        def neighbour_graph(self, n: int) -> dict[int, list[int]]:
            g = {i: [] for i in range(n)}
            for i, j in self._edges:
                g[i].append(j)
                g[j].append(i)
            return g

        def lifted_links(self):
            return [
                Link(i=i, j=j, count_ij=1.0, count_ji=1.0, lifted=True)
                for i, j in self._edges
            ]

    class _Node:
        def __init__(self, pos, hits=1.0):
            self.position = np.asarray(pos, dtype=float)
            self.hit_count = hits
            self.d_final = 1

    class _Scaf:
        def __init__(self, nodes, edges):
            self.nodes = nodes
            self.links = _Links(edges)
            self.tau = 0.1

    def _ring(radius: float, count: int, center=(0.0, 0.0)) -> list:
        angs = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        cx, cy = center
        return [
            _Node([cx + radius * np.cos(a), cy + radius * np.sin(a)])
            for a in angs
        ]

    # Nearly filled continuum: peaks exist but trough is shallow → gate rejects.
    shallow_nodes = _ring(1.0, 10) + _ring(3.0, 10)
    for rm in np.linspace(1.2, 2.8, 9):
        shallow_nodes.extend(_ring(float(rm), 6))
    shallow_edges = (
        [(i, j) for i in range(10) for j in range(i + 1, 10)]
        + [(10 + i, 10 + j) for i in range(10) for j in range(i + 1, 10)]
    )
    shallow = _Scaf(shallow_nodes, shallow_edges)
    assert _radial_band_gap_partition(
        shallow, min_frac=0.15, min_abs=3, min_gap_ratio=0.25,
        hist_bins=12, min_trough_rel=0.55,
    ) is None

    # Clear two-shell mid continuum still recovers under a moderate trough gate.
    inner = _ring(1.0, 12)
    outer = _ring(3.0, 12)
    mid: list = []
    for rm in np.linspace(1.4, 2.6, 7):
        mid.extend(_ring(float(rm), 4))
    clear_nodes = inner + outer + mid
    clear_edges = (
        [(i, j) for i in range(12) for j in range(i + 1, 12)]
        + [(12 + i, 12 + j) for i in range(12) for j in range(i + 1, 12)]
        + [(0, 12)]
    )
    clear = _Scaf(clear_nodes, clear_edges)
    pre = _radial_band_gap_partition(
        clear, min_frac=0.15, min_abs=3, min_gap_ratio=0.25,
        hist_bins=12, min_trough_rel=0.25,
    )
    assert pre is not None and pre.n_clusters == 2 and pre.partition_q_score > 0.0

    # Dense one-sided tissue pulls the mean; coordinate-median stays nearer 0.
    # Fixture tuned so mean-origin band fails while coord-median recovers shells
    # (integrator: original n=10@x=4.5 was over-polluted vs contiguous peak support).
    rng = np.random.default_rng(7)
    skew_nodes = _ring(1.0, 12) + _ring(3.0, 12)
    for _ in range(6):
        skew_nodes.append(_Node([3.8 + 0.15 * rng.normal(), 0.2 * rng.normal()]))
    for rm in np.linspace(1.4, 2.6, 5):
        skew_nodes.extend(_ring(float(rm), 3))
    skew_edges = (
        [(i, j) for i in range(12) for j in range(i + 1, 12)]
        + [(12 + i, 12 + j) for i in range(12) for j in range(i + 1, 12)]
        + [(0, 12)]
    )
    skew = _Scaf(skew_nodes, skew_edges)
    pts = np.asarray([nd.position for nd in skew_nodes], dtype=float)
    mean_c = pts.mean(axis=0)
    med_c = _coordinate_median(pts)
    assert float(abs(med_c[0])) < float(abs(mean_c[0]))
    pre_mean = _radial_band_gap_partition(
        skew, min_frac=0.12, min_abs=3, min_gap_ratio=0.2,
        hist_bins=14, origin="mean", min_trough_rel=0.2,
    )
    assert pre_mean is None
    pre_nc = _radial_band_gap_partition(
        skew, min_frac=0.12, min_abs=3, min_gap_ratio=0.2,
        hist_bins=14, origin="coord_median", min_trough_rel=0.2,
    )
    assert pre_nc is not None
    assert pre_nc.n_clusters == 2
    assert pre_nc.partition_q_score > 0.0
    assert len(set(int(x) for x in pre_nc.labels[:12])) == 1
    assert len(set(int(x) for x in pre_nc.labels[12:24])) == 1
    assert int(pre_nc.labels[0]) != int(pre_nc.labels[12])


def test_signal_density_band_prefers_shell_arcs() -> None:
    """#44 / A2-T14: knn×radial denseness mask keeps shell arcs over mid continuum.

    Mid-radius continuum tissue fills radial bins so plain band-gap fails;
    ``signal_density_keep_frac`` with score ``rho_knn * rho_radial`` recovers
    the two shells.  (Divide residual upweights sparse mid bins — rejected.)
    """

    from proteus.stage1.recursion import _radial_band_gap_partition
    from proteus.types import Link

    class _Links:
        def __init__(self, edges: list[tuple[int, int]]):
            self._edges = edges

        def neighbour_graph(self, n: int) -> dict[int, list[int]]:
            g = {i: [] for i in range(n)}
            for i, j in self._edges:
                g[i].append(j)
                g[j].append(i)
            return g

        def lifted_links(self):
            return [
                Link(i=i, j=j, count_ij=1.0, count_ji=1.0, lifted=True)
                for i, j in self._edges
            ]

    class _Node:
        def __init__(self, pos, hits=1.0):
            self.position = np.asarray(pos, dtype=float)
            self.hit_count = hits
            self.d_final = 1

    class _Scaf:
        def __init__(self, nodes, edges):
            self.nodes = nodes
            self.links = _Links(edges)
            self.tau = 0.1

    def _ring(radius: float, count: int) -> list:
        angs = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        return [
            _Node([radius * np.cos(a), radius * np.sin(a)]) for a in angs
        ]

    inner = _ring(1.0, 12)
    outer = _ring(3.0, 12)
    # Mid-radius continuum bridges (dense radial fill, weaker knn than shells).
    mid: list = []
    for rm in np.linspace(1.4, 2.6, 7):
        mid.extend(_ring(float(rm), 4))
    nodes = inner + outer + mid
    edges = (
        [(i, j) for i in range(12) for j in range(i + 1, 12)]
        + [(12 + i, 12 + j) for i in range(12) for j in range(i + 1, 12)]
        + [(0, 12)]
    )
    scaf = _Scaf(nodes, edges)
    # Plain band may or may not fire depending on trough depth; denseness
    # mask must recover two shell labels under knn×radial keep.
    pre = _radial_band_gap_partition(
        scaf, min_frac=0.15, min_abs=3, min_gap_ratio=0.25,
        hist_bins=12, signal_density_keep_frac=0.7,
    )
    assert pre is not None
    assert pre.n_clusters == 2
    assert pre.partition_q_score > 0.0
    assert len(set(int(x) for x in pre.labels[:12])) == 1
    assert len(set(int(x) for x in pre.labels[12:24])) == 1
    assert int(pre.labels[0]) != int(pre.labels[12])


def test_pca_axis_gap_recovers_offset_rings() -> None:
    """#44 / A2-T21: PCA-axis gap splits laterally offset rings (non-radial).

    Two side-by-side rings share no radial-from-origin trough but separate
    cleanly on PC1.  Concentric rings should fail the gap-ratio gate.
    """

    from proteus.stage1.recursion import (
        _pca_axis_gap_partition,
        _radial_gap_partition,
    )
    from proteus.types import Link

    class _Links:
        def __init__(self, edges: list[tuple[int, int]]):
            self._edges = edges

        def neighbour_graph(self, n: int) -> dict[int, list[int]]:
            g = {i: [] for i in range(n)}
            for i, j in self._edges:
                g[i].append(j)
                g[j].append(i)
            return g

        def lifted_links(self):
            return [
                Link(i=i, j=j, count_ij=1.0, count_ji=1.0, lifted=True)
                for i, j in self._edges
            ]

    class _Node:
        def __init__(self, pos, hits=1.0):
            self.position = np.asarray(pos, dtype=float)
            self.hit_count = hits
            self.d_final = 1

    class _Scaf:
        def __init__(self, nodes, edges):
            self.nodes = nodes
            self.links = _Links(edges)
            self.tau = 0.1

    def _ring(cx: float, cy: float, radius: float, count: int) -> list:
        angs = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        return [
            _Node([cx + radius * np.cos(a), cy + radius * np.sin(a)])
            for a in angs
        ]

    left = _ring(0.0, 0.0, 1.0, 16)
    right = _ring(3.5, 0.0, 1.0, 16)
    nodes = left + right
    # Intra-ring complete graphs + one weak bridge (lifted-connected).
    edges = (
        [(i, j) for i in range(16) for j in range(i + 1, 16)]
        + [(16 + i, 16 + j) for i in range(16) for j in range(i + 1, 16)]
        + [(0, 16)]
    )
    scaf = _Scaf(nodes, edges)
    # Radial gap is the wrong cue for offset rings (may still fire on a
    # diameter); PCA-axis must recover left/right membership.
    pre = _pca_axis_gap_partition(
        scaf, min_frac=0.2, min_abs=3, min_gap_ratio=0.25,
    )
    assert pre is not None
    assert pre.n_clusters == 2
    assert pre.partition_q_score > 0.0
    assert len(set(int(x) for x in pre.labels[:16])) == 1
    assert len(set(int(x) for x in pre.labels[16:])) == 1
    assert int(pre.labels[0]) != int(pre.labels[16])

    # Concentric control: PCA diameter has no deep trough → None.
    inner = _ring(0.0, 0.0, 1.0, 16)
    outer = _ring(0.0, 0.0, 3.0, 16)
    conc = _Scaf(
        inner + outer,
        (
            [(i, j) for i in range(16) for j in range(i + 1, 16)]
            + [(16 + i, 16 + j) for i in range(16) for j in range(i + 1, 16)]
            + [(0, 16)]
        ),
    )
    assert _pca_axis_gap_partition(
        conc, min_frac=0.2, min_abs=3, min_gap_ratio=0.25,
    ) is None
    # Radial gap still the right cue for concentric.
    rad = _radial_gap_partition(
        conc, min_frac=0.2, min_abs=3, min_gap_ratio=0.25,
    )
    assert rad is not None
    assert rad.n_clusters == 2


def test_research_finer_split_rejects_invalid_cap() -> None:
    """#44: finer re-search is a no-op when the cap is not strictly inside (tau_min, tau*)."""

    from proteus.stage1.recursion import _research_finer_split

    data = np.zeros((40, 2), dtype=float)
    cfg = RecursionConfig(
        scale_search=ScaleSearchConfig(tau_min=1e-3, tau_max=1.0),
        allow_finer_research=True,
        finer_tau_cap_ratio=0.5,
        max_finer_scale_steps=3,
    )
    # parent_tau * ratio <= tau_min -> cannot form a valid finer window
    assert _research_finer_split(data, 2, cfg, parent_tau=1e-3) is None
    assert _research_finer_split(data, 2, cfg, parent_tau=1.5e-3) is None


def test_finer_research_circle_does_not_shatter() -> None:
    """#44 / A2-T3: allow_finer_research must not shatter a uniform circle.

    Pairing with persistence (the intended composition) keeps a single leaf.
    Flag-alone without persistence over-fragments and is out of scope for
    this guard (documented in REQUEST_TRACKER for #44).
    """

    dataset = make_circle(
        n_samples=600, radius=1.0, noise=0.02, extrusion_dim=2, seed=21,
    )
    data = dataset.points
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint

    config = RecursionConfig(
        scale_search=ScaleSearchConfig(
            tau_min=tau_lo,
            tau_max=tau_hi,
            max_grid_points=6,
            k=8,
            n_seeds=8,
            ann_backend="naive",
            stabilization=StabilizationConfig(
                min_equilibrium_epochs=2,
                max_epochs=6,
            ),
            seed=11,
        ),
        min_samples=80,
        max_depth=3,
        allow_finer_research=True,
        require_persistent_split=True,
        max_finer_scale_steps=4,
        seed=11,
    )
    tree = run_recursive_discovery(data, dim=gt.ambient_dim, config=config)
    assert len(tree.leaves) == 1
    assert tree.nodes[0].is_leaf


def test_finer_research_swiss_does_not_shatter() -> None:
    """#44 / A2-T3: allow_finer_research+persist keeps swiss roll as one leaf."""

    from tests.datasets.synthetic.swiss_roll import make_swiss_roll

    dataset = make_swiss_roll(n_samples=600, noise=0.02, seed=7)
    data = dataset.points
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint

    config = RecursionConfig(
        scale_search=ScaleSearchConfig(
            tau_min=tau_lo,
            tau_max=tau_hi,
            max_grid_points=6,
            k=8,
            n_seeds=8,
            ann_backend="naive",
            stabilization=StabilizationConfig(
                min_equilibrium_epochs=2,
                max_epochs=6,
            ),
            seed=11,
        ),
        min_samples=80,
        max_depth=3,
        allow_finer_research=True,
        require_persistent_split=True,
        max_finer_scale_steps=4,
        seed=11,
    )
    tree = run_recursive_discovery(data, dim=gt.ambient_dim, config=config)
    assert len(tree.leaves) == 1
    assert tree.nodes[0].is_leaf


def test_finer_research_nested_spheres_aspiration_sketch() -> None:
    """#44 sketch: with flag+DM, nested_spheres should aim for cc=2 leaves.

    Default path (flag off) terminates at K=1 at coarse tau*.  This test
    documents the aspiration and only asserts the default-off contract is
    unchanged; do **not** assert leaf-count==2 until recovery is green.

    A2-T4 measurements (tissue_fraction=0): recommended pairing
    (persist + allow_finer_research + steps<=4 + min_samples>=80) →
    nested_spheres and linked_tori both stay at **1 leaf** (n=160 and n=240).
    Deepening (steps=12–18 + prepass, min_samples=40) over-fragments
    (nested leaves 5–20, tori 6–12) with ARI≈0 / <0.23 — not shell recovery.

    A2-T7/T8: circle-safe persist+prepass grid (n_seeds=8, min_frac 0.15–0.4,
    steps≤12) still leaves nested at 1 leaf; steps≥12 or require_dm_split
    without persist yields 5–9 leaves with ARI≲0.09 — still not recovery.

    A2 diagnostic: major lifted-CC prepass misses shells (graph stays 1 CC or
    noise-fragments). Flag-gated ``prefer_radial_gap_prepass`` is the next
    proposed path; do not assert leaf-count==2 until e2e recovery is green.

    A2-T10/T11: mid-band bridges keep lifted CC=1 and dilute plain radial
    gaps; ``prefer_radial_band_prepass`` (histogram-trough exclusion) recovers
    shell ARI on unit scaffolds with mid fillers — e2e nested recovery still
    not asserted here.

    A2-T15/T18: ``prefer_signal_density_band_prepass`` with knn×radial score
    recovers nested unit harness (steps≥8 → 2 leaves ARI=1.0); linked_tori
    stays 1 leaf (radial origin not suited to offset linked rings). Do not
    flip awaiting until A1 confirms + tori path exists. Swiss: steps≤4 → 1
    leaf; steps=8 shatters — keep ``max_finer_scale_steps≤4`` for uniforms.

    A2-T21: ``prefer_pca_axis_gap_prepass`` recovers offset (non-linked) rings
    on unit scaffolds; interlocking linked_tori e2e still unrecovered — hold
    awaiting.
    """

    from tests.datasets.synthetic.nested_spheres import make_nested_spheres

    dataset = make_nested_spheres(n_per_sphere=64, extrusion_dim=1, seed=0)
    data = dataset.points
    gt = dataset.ground_truth
    assert gt.topology is not None
    assert int(gt.topology.connected_components) == 2

    base = RecursionConfig(
        scale_search=ScaleSearchConfig(
            tau_min=1e-4,
            tau_max=2.0,
            max_grid_points=6,
            k=6,
            n_seeds=6,
            ann_backend="naive",
            stabilization=StabilizationConfig(
                min_equilibrium_epochs=2,
                max_epochs=6,
            ),
            seed=3,
        ),
        min_samples=20,
        max_depth=2,
        seed=3,
    )
    tree_off = run_recursive_discovery(data, dim=gt.ambient_dim, config=base)
    # Flag off: coarse K=1 terminates — at most one leaf (current #44 defect).
    assert len(tree_off.leaves) == 1

    # Aspiration (not asserted yet): allow_finer_research + radial-band /
    # DM should recover ~2 leaves.  Keep the config construction here as the
    # living sketch for follow-on validation runs.
    _aspirational = RecursionConfig(
        scale_search=base.scale_search,
        min_samples=base.min_samples,
        max_depth=base.max_depth,
        allow_finer_research=True,
        prefer_disconnected_prepass=True,
        prefer_radial_gap_prepass=True,
        prefer_radial_band_prepass=True,
        prefer_noncentroid_radial_band_prepass=True,
        prefer_signal_density_band_prepass=True,
        prefer_pca_axis_gap_prepass=True,
        require_dm_split=True,
        finer_tau_cap_ratio=0.5,
        max_finer_scale_steps=12,
        seed=base.seed,
    )
    assert _aspirational.allow_finer_research is True
    assert _aspirational.prefer_disconnected_prepass is True
    assert _aspirational.prefer_radial_gap_prepass is True
    assert _aspirational.prefer_radial_band_prepass is True
    assert _aspirational.prefer_noncentroid_radial_band_prepass is True
    assert _aspirational.prefer_signal_density_band_prepass is True
    assert _aspirational.prefer_pca_axis_gap_prepass is True
    assert _aspirational.max_finer_scale_steps == 12
