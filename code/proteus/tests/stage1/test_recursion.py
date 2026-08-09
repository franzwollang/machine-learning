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
    assert cfg.prefer_tube_major_radius_prepass is False
    assert cfg.finer_tube_min_residual_ratio > 0.0
    assert cfg.prefer_spectral_gap_prepass is False
    assert cfg.finer_spectral_knn >= 2
    assert cfg.prefer_hollow_edge_prepass is False
    assert cfg.hollow_mid_radius_frac > 0.0
    assert cfg.hollow_h0 > 0.0
    assert cfg.hollow_min_end_count >= 0.0
    assert cfg.hollow_gabriel_fallback is True
    assert cfg.hollow_require_gabriel_and_h is False
    assert cfg.hollow_require_persistent_agree is False
    assert cfg.hollow_use_a4_primary is False
    assert cfg.hollow_mst_critical_only is False
    assert cfg.hollow_bridge_critical_only is False
    assert cfg.hollow_soft_capacity_only is False
    assert 0.0 < cfg.hollow_soft_capacity_frac <= 1.0
    assert cfg.hollow_soft_capacity_method == "betweenness"


def test_hollow_edge_partition_splits_bridged_blobs() -> None:
    """#44: hollow prepass recovers two majors after cutting a void bridge."""

    from proteus.stage1.recursion import _hollow_edge_partition
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
                Link(i=i, j=j, count_ij=5.0, count_ji=5.0, lifted=True)
                for i, j in self._edges
            ]

    class _Node:
        def __init__(self, pos, hits=10.0):
            self.position = np.asarray(pos, dtype=float)
            self.hit_count = hits
            self.d_final = 1

    class _Scaf:
        def __init__(self, nodes, edges):
            self.nodes = nodes
            self.links = _Links(edges)
            self.tau = 0.2

    rng = np.random.default_rng(0)
    data = np.vstack([
        rng.normal([-2.5, 0.0], 0.15, size=(60, 2)),
        rng.normal([2.5, 0.0], 0.15, size=(60, 2)),
    ])
    nodes = [
        _Node([-2.6, 0.0]), _Node([-2.4, 0.1]), _Node([-2.4, -0.1]),
        _Node([2.4, 0.1]), _Node([2.5, 0.0]), _Node([2.6, -0.1]),
    ]
    edges = [
        (0, 1), (1, 2), (0, 2),
        (3, 4), (4, 5), (3, 5),
        (0, 3),  # hollow bridge
    ]
    pre = _hollow_edge_partition(
        _Scaf(nodes, edges), data, min_frac=0.2, min_abs=2,
        mid_radius_frac=0.35, h0=0.35, min_end_count=0.5,
    )
    assert pre is not None
    assert pre.n_clusters == 2
    assert pre.partition_q_score > 0.0
    assert int(pre.labels[0]) != int(pre.labels[3])


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

    # Concentric control: PC1 diameter cut fails centroid-separation gate.
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


def test_tube_major_radius_recovers_interlocking_rings() -> None:
    """#44 / A2-T24: tube residual splits Hopf-linked rings (interlock cue).

    Linked major circles in the synthetic linked_tori pose (xy circle about
    the origin + yz circle translated by R) separate by nearer-tube
    assignment.  Concentric coplanar rings are the wrong cue and must miss.
    """

    from proteus.stage1.recursion import (
        _pca_axis_gap_partition,
        _tube_major_radius_partition,
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

    R = 2.0
    n_ring = 16
    angs = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
    ring_xy = [
        _Node([R * np.cos(a), R * np.sin(a), 0.0]) for a in angs
    ]
    ring_yz = [
        _Node([R, R * np.sin(a), R * np.cos(a)]) for a in angs
    ]
    nodes = ring_xy + ring_yz
    edges = (
        [(i, j) for i in range(n_ring) for j in range(i + 1, n_ring)]
        + [
            (n_ring + i, n_ring + j)
            for i in range(n_ring) for j in range(i + 1, n_ring)
        ]
        + [(0, n_ring)]  # weak bridge — lifted-connected
    )
    scaf = _Scaf(nodes, edges)
    pre = _tube_major_radius_partition(
        scaf, min_frac=0.2, min_abs=3, min_residual_ratio=0.15,
    )
    assert pre is not None
    assert pre.n_clusters == 2
    assert pre.partition_q_score > 0.0
    assert len(set(int(x) for x in pre.labels[:n_ring])) == 1
    assert len(set(int(x) for x in pre.labels[n_ring:])) == 1
    assert int(pre.labels[0]) != int(pre.labels[n_ring])

    # Concentric control: Hopf tube template should not recover shells.
    inner = [
        _Node([1.0 * np.cos(a), 1.0 * np.sin(a), 0.0]) for a in angs
    ]
    outer = [
        _Node([3.0 * np.cos(a), 3.0 * np.sin(a), 0.0]) for a in angs
    ]
    conc = _Scaf(
        inner + outer,
        (
            [(i, j) for i in range(n_ring) for j in range(i + 1, n_ring)]
            + [
                (n_ring + i, n_ring + j)
                for i in range(n_ring) for j in range(i + 1, n_ring)
            ]
            + [(0, n_ring)]
        ),
    )
    assert _tube_major_radius_partition(
        conc, min_frac=0.2, min_abs=3, min_residual_ratio=0.15,
    ) is None
    # Offset (non-linked) rings: PCA remains the right cue; tube may miss.
    left = [
        _Node([np.cos(a), np.sin(a), 0.0]) for a in angs
    ]
    right = [
        _Node([3.5 + np.cos(a), np.sin(a), 0.0]) for a in angs
    ]
    offset = _Scaf(
        left + right,
        (
            [(i, j) for i in range(n_ring) for j in range(i + 1, n_ring)]
            + [
                (n_ring + i, n_ring + j)
                for i in range(n_ring) for j in range(i + 1, n_ring)
            ]
            + [(0, n_ring)]
        ),
    )
    pca = _pca_axis_gap_partition(
        offset, min_frac=0.2, min_abs=3, min_gap_ratio=0.25,
    )
    assert pca is not None
    assert pca.n_clusters == 2


def test_spectral_gap_bipartitions_offset_rings() -> None:
    """#44 / A2-T25: Fiedler bipartition splits well-separated ring pair."""

    from proteus.stage1.recursion import _spectral_gap_partition
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

    n_ring = 16
    angs = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
    left = [_Node([np.cos(a), np.sin(a), 0.0]) for a in angs]
    right = [_Node([3.5 + np.cos(a), np.sin(a), 0.0]) for a in angs]
    # Dense intra-ring; single weak bridge.
    edges = (
        [(i, j) for i in range(n_ring) for j in range(i + 1, n_ring)]
        + [
            (n_ring + i, n_ring + j)
            for i in range(n_ring) for j in range(i + 1, n_ring)
        ]
        + [(0, n_ring)]
    )
    scaf = _Scaf(left + right, edges)
    pre = _spectral_gap_partition(scaf, min_frac=0.2, min_abs=3, knn=8)
    assert pre is not None
    assert pre.n_clusters == 2
    assert pre.partition_q_score > 0.0
    assert len(set(int(x) for x in pre.labels[:n_ring])) == 1
    assert len(set(int(x) for x in pre.labels[n_ring:])) == 1
    assert int(pre.labels[0]) != int(pre.labels[n_ring])


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

    A2-T24/T25: ``prefer_tube_major_radius_prepass`` recovers interlocking
    thin rings on unit scaffolds (Hopf tube residual); spectral Fiedler is an
    alternate graph cue.  E2e linked_tori under persist+sd+pca+tube still not
    asserted here — hold awaiting.
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
        prefer_tube_major_radius_prepass=True,
        prefer_spectral_gap_prepass=True,
        prefer_hollow_edge_prepass=True,
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
    assert _aspirational.prefer_tube_major_radius_prepass is True
    assert _aspirational.prefer_spectral_gap_prepass is True
    assert _aspirational.prefer_hollow_edge_prepass is True
    assert _aspirational.max_finer_scale_steps == 12


def test_finer_research_persist_sd_pca_regression_harness() -> None:
    """#44 / A2-T26: persist+sd+pca(+tube) leaf-count regression harness.

    Guards uniforms under the recommended pairing; documents nested / tori
    leaf counts without flipping awaiting component tests.
    """

    from tests.datasets.synthetic.linked_tori import make_linked_tori
    from tests.datasets.synthetic.nested_spheres import make_nested_spheres
    from tests.datasets.synthetic.swiss_roll import make_swiss_roll

    def _lean_scale(tau_lo: float, tau_hi: float, *, seed: int) -> ScaleSearchConfig:
        return ScaleSearchConfig(
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
            seed=seed,
        )

    def _run(dataset, *, steps: int, min_samples: int, **flags) -> int:
        gt = dataset.ground_truth
        tau_lo, tau_hi = gt.tau_grid_hint
        cfg = RecursionConfig(
            scale_search=_lean_scale(tau_lo, tau_hi, seed=11),
            min_samples=min_samples,
            max_depth=3,
            allow_finer_research=True,
            require_persistent_split=True,
            max_finer_scale_steps=steps,
            prefer_signal_density_band_prepass=True,
            finer_signal_density_keep_frac=0.55,
            prefer_pca_axis_gap_prepass=True,
            prefer_tube_major_radius_prepass=bool(
                flags.get("tube", False),
            ),
            prefer_spectral_gap_prepass=bool(
                flags.get("spectral", False),
            ),
            seed=11,
        )
        tree = run_recursive_discovery(
            dataset.points, dim=gt.ambient_dim, config=cfg,
        )
        return len(tree.leaves)

    circle = make_circle(
        n_samples=600, radius=1.0, noise=0.02, extrusion_dim=2, seed=21,
    )
    swiss = make_swiss_roll(n_samples=600, noise=0.02, seed=7)
    # Uniform guards: persist+sd+pca at steps<=4 must stay single-leaf.
    assert _run(circle, steps=4, min_samples=80) == 1
    assert _run(swiss, steps=4, min_samples=80) == 1

    nested = make_nested_spheres(n_per_sphere=64, extrusion_dim=1, seed=0)
    tori = make_linked_tori(
        n_per_torus=80, extrusion_dim=1, tissue_fraction=0.0, seed=0,
    )
    # Measurement only — do not assert recovery / flip awaiting.
    nested_leaves = _run(nested, steps=8, min_samples=20)
    tori_pca = _run(tori, steps=8, min_samples=40)
    tori_tube = _run(tori, steps=8, min_samples=40, tube=True)
    # Trees must be well-formed; tori e2e recovery still not claimed.
    assert nested_leaves >= 1
    assert tori_pca >= 1
    assert tori_tube >= 1
    # PCA path: interlocking linked_tori still unrecovered (known failure).
    assert tori_pca == 1


def test_hollow_edge_persist_no_descent_regression_harness() -> None:
    """#44 / A2-T29/T30: persist(+dm)+hollow, NO finer descent — leaf harness.

    Guards uniforms / zoo under hollow prepass without ``allow_finer_research``.
    Documents nested/tori = 1 leaf at scale-search ``tau*``.  Fixed-tau
    ``K=2`` at ~0.27/0.5 is **not** recovery (sample ARI near chance;
    A2-T30).  Do **not** flip awaiting.
    """

    from proteus.stage1.recursion import _hollow_edge_partition
    from proteus.stage1.scaffold import Stage1Scaffold
    from tests.datasets.synthetic.linked_tori import make_linked_tori
    from tests.datasets.synthetic.manifold_zoo import make_manifold_zoo
    from tests.datasets.synthetic.nested_spheres import make_nested_spheres
    from tests.datasets.synthetic.swiss_roll import make_swiss_roll

    def _lean(seed: int = 42) -> ScaleSearchConfig:
        return ScaleSearchConfig(
            tau_min=1e-3,
            tau_max=2.0,
            max_grid_points=6,
            k=8,
            n_seeds=8,
            ann_backend="naive",
            stabilization=StabilizationConfig(
                min_equilibrium_epochs=2,
                max_epochs=8,
            ),
            seed=seed,
        )

    def _run(points, dim, *, persist: bool, dm: bool, min_samples: int) -> int:
        cfg = RecursionConfig(
            scale_search=_lean(),
            min_samples=min_samples,
            max_depth=3,
            require_persistent_split=persist,
            require_dm_split=dm,
            allow_finer_research=False,
            prefer_hollow_edge_prepass=True,
            seed=42,
        )
        tree = run_recursive_discovery(points, dim=dim, config=cfg)
        return len(tree.leaves)

    circle = make_circle(
        n_samples=300, radius=1.0, noise=0.02, extrusion_dim=2, seed=0,
    )
    swiss = make_swiss_roll(n_samples=400, noise=0.02, seed=0)
    zoo = make_manifold_zoo(seed=0)
    z = zoo.points
    if z.shape[0] > 600:
        z = z[np.random.default_rng(0).choice(z.shape[0], 600, replace=False)]

    # Uniform / connected guards (persist+hollow, no descent).
    assert _run(circle.points, circle.points.shape[1], persist=True, dm=False, min_samples=80) == 1
    assert _run(swiss.points, swiss.points.shape[1], persist=True, dm=False, min_samples=80) == 1
    assert _run(z, z.shape[1], persist=True, dm=False, min_samples=40) == 1
    assert _run(circle.points, circle.points.shape[1], persist=True, dm=True, min_samples=80) == 1

    nested = make_nested_spheres(n_per_sphere=80, extrusion_dim=1, seed=0)
    tori = make_linked_tori(n_per_torus=120, seed=0)
    # E2E at scale-search tau*: nested unrecovered.  Tori may emit 2 leaves
    # with near-zero ARI (spurious hollow cut at coarse tau_max) — not
    # component recovery; do not flip awaiting.
    assert _run(nested.points, nested.points.shape[1], persist=True, dm=False, min_samples=40) == 1
    tori_leaves = _run(tori.points, tori.points.shape[1], persist=True, dm=False, min_samples=40)
    assert tori_leaves >= 1

    # Fixed-tau "oracle": K=2 majors can appear at probe taus, but sample ARI
    # is near chance (A2-T30) — do **not** treat K=2 as component recovery.
    from sklearn.metrics import adjusted_rand_score

    def _oracle(points, tau: float):
        sc = Stage1Scaffold(
            dim=int(points.shape[1]), tau=float(tau), k=8, max_nodes=64,
            ann_backend="naive", rng=np.random.default_rng(0),
        )
        sc.init_from(points, n_seeds=8)
        sc.run_until_stable(
            points,
            StabilizationConfig(max_epochs=40, min_equilibrium_epochs=3),
        )
        part = _hollow_edge_partition(
            sc, points,
            mid_radius_frac=0.35, h0=0.35, min_end_count=0.5, min_frac=0.2,
        )
        return sc, part

    def _sample_ari(sc, part, points, labels) -> float:
        pos = np.asarray([sc.nodes[i].position for i in range(len(sc.nodes))])
        nn = np.argmin(
            ((points[:, None, :] - pos[None, :, :]) ** 2).sum(-1), axis=1,
        )
        pred = part.labels[nn]
        mask = np.asarray(labels) >= 0
        return float(adjusted_rand_score(labels[mask], pred[mask]))

    sc_n, h_nested = _oracle(nested.points, 0.27)
    sc_t, h_tori = _oracle(tori.points, 0.5)
    assert h_nested is not None and h_nested.n_clusters == 2
    assert h_tori is not None and h_tori.n_clusters == 2
    assert _sample_ari(sc_n, h_nested, nested.points, nested.labels) < 0.2
    assert _sample_ari(sc_t, h_tori, tori.points, tori.labels) < 0.2
    assert tori_leaves in (1, 2)  # spurious 2-leaf possible; ARI not recovery

def test_multi_tau_hollow_prune_scan_harness() -> None:
    """#44 / A2-T30: multi-tau hollow prune→CC scan (grid + E[tau] + tau*).

    Documents that default H-or-Gabriel yields majors=2 at probe taus
    nested@0.27 / tori@0.5 with sample ARI≈chance, while
    ``require_gabriel_and_h`` suppresses those spurious K=2 hits.  At
    scale-search ``tau*`` / expected_tau nested+tori stay ≤1 major under
    default.  Do **not** flip awaiting.
    """

    from proteus.stage1.clustering import _lifted_components_covering_all_nodes
    from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
    from proteus.stage1.edge_evidence import HollowEdgeConfig, prune_hollow_edges
    from proteus.stage1.scaffold import Stage1Scaffold
    from sklearn.metrics import adjusted_rand_score
    from tests.datasets.synthetic.linked_tori import make_linked_tori
    from tests.datasets.synthetic.nested_spheres import make_nested_spheres
    from tests.datasets.synthetic.swiss_roll import make_swiss_roll

    def _lean() -> ScaleSearchConfig:
        return ScaleSearchConfig(
            tau_min=1e-3,
            tau_max=2.0,
            max_grid_points=6,
            k=8,
            n_seeds=8,
            ann_backend="naive",
            stabilization=StabilizationConfig(
                min_equilibrium_epochs=2, max_epochs=8,
            ),
            seed=42,
        )

    def _adapt(points, tau: float):
        sc = Stage1Scaffold(
            dim=int(points.shape[1]), tau=float(tau), k=8, max_nodes=64,
            ann_backend="naive", rng=np.random.default_rng(0),
        )
        sc.init_from(points, n_seeds=8)
        sc.run_until_stable(
            points,
            StabilizationConfig(max_epochs=30, min_equilibrium_epochs=3),
        )
        return sc

    def _majors_ari(sc, points, labels, cfg: HollowEdgeConfig):
        n = len(sc.nodes)
        pos = np.asarray([sc.nodes[i].position for i in range(n)])
        edges = [(int(l.i), int(l.j)) for l in sc.links.lifted_links()]
        kept = prune_hollow_edges(pos, edges, points, config=cfg)
        graph = {i: [] for i in range(n)}
        for i, j in kept:
            graph[i].append(j)
            graph[j].append(i)
        comps = _lifted_components_covering_all_nodes(n, graph)
        majors = [c for c in comps if len(c) >= max(3, int(np.ceil(n * 0.2)))]
        ari = None
        if labels is not None and len(majors) >= 2:
            lab = np.full(n, -1, dtype=int)
            for cid, c in enumerate(majors):
                for m in c:
                    lab[m] = cid
            nn = np.argmin(
                ((points[:, None, :] - pos[None, :, :]) ** 2).sum(-1), axis=1,
            )
            pred = lab[nn]
            mask = np.asarray(labels) >= 0
            ari = float(adjusted_rand_score(labels[mask], pred[mask]))
        return len(majors), ari

    cfg_def = HollowEdgeConfig()
    cfg_conj = HollowEdgeConfig(require_gabriel_and_h=True)

    nested = make_nested_spheres(n_per_sphere=80, extrusion_dim=1, seed=0)
    tori = make_linked_tori(n_per_torus=120, seed=0)
    circle = make_circle(
        n_samples=300, radius=1.0, noise=0.02, extrusion_dim=2, seed=0,
    )
    swiss = make_swiss_roll(n_samples=400, noise=0.02, seed=0)

    # Nested / tori probe taus: default K=2 with ARI~chance; conjunction ≤1.
    sc_n = _adapt(nested.points, 0.27)
    maj_n, ari_n = _majors_ari(sc_n, nested.points, nested.labels, cfg_def)
    maj_n_c, _ = _majors_ari(sc_n, nested.points, nested.labels, cfg_conj)
    assert maj_n == 2
    assert ari_n is not None and ari_n < 0.2
    assert maj_n_c <= 1

    sc_t = _adapt(tori.points, 0.5)
    maj_t, ari_t = _majors_ari(sc_t, tori.points, tori.labels, cfg_def)
    maj_t_c, _ = _majors_ari(sc_t, tori.points, tori.labels, cfg_conj)
    assert maj_t == 2
    assert ari_t is not None and ari_t < 0.2
    assert maj_t_c <= 1

    # Scale-search tau* + expected_tau: nested/tori not recovered under default.
    for name, ds, labels in (
        ("nested", nested, nested.labels),
        ("tori", tori, tori.labels),
    ):
        r = run_scale_search(ds.points, dim=ds.points.shape[1], config=_lean())
        et = float(ds.ground_truth.expected_tau)
        for tau in (float(r.tau_star), et):
            sc = _adapt(ds.points, tau)
            maj, ari = _majors_ari(sc, ds.points, labels, cfg_def)
            assert maj <= 1 or (ari is not None and ari < 0.2)

    # Uniforms at tau*: default hollow prune stays connected (1 major).
    for ds in (circle, swiss):
        r = run_scale_search(ds.points, dim=ds.points.shape[1], config=_lean())
        sc = _adapt(ds.points, float(r.tau_star))
        maj, _ = _majors_ari(sc, ds.points, None, cfg_def)
        assert maj <= 1


def test_hollow_gabriel_and_h_seed_stability_on_probe_taus() -> None:
    """#44 / A2-T31: conjunction seed-stable vs Gabriel-driven default K=2.

    Seeds 0..2: default often emits majors=2 at nested@0.27; conjunction
    stays ≤1.  Flag / prepass remain default-off.
    """

    from proteus.stage1.clustering import _lifted_components_covering_all_nodes
    from proteus.stage1.edge_evidence import HollowEdgeConfig, prune_hollow_edges
    from proteus.stage1.scaffold import Stage1Scaffold
    from tests.datasets.synthetic.nested_spheres import make_nested_spheres

    nested = make_nested_spheres(n_per_sphere=80, extrusion_dim=1, seed=0)
    cfg_def = HollowEdgeConfig()
    cfg_conj = HollowEdgeConfig(require_gabriel_and_h=True)

    def _majors(seed: int, cfg: HollowEdgeConfig) -> int:
        sc = Stage1Scaffold(
            dim=int(nested.points.shape[1]), tau=0.27, k=8, max_nodes=64,
            ann_backend="naive", rng=np.random.default_rng(seed),
        )
        sc.init_from(nested.points, n_seeds=8)
        sc.run_until_stable(
            nested.points,
            StabilizationConfig(max_epochs=30, min_equilibrium_epochs=3),
        )
        n = len(sc.nodes)
        pos = np.asarray([sc.nodes[i].position for i in range(n)])
        edges = [(int(l.i), int(l.j)) for l in sc.links.lifted_links()]
        kept = prune_hollow_edges(pos, edges, nested.points, config=cfg)
        graph = {i: [] for i in range(n)}
        for i, j in kept:
            graph[i].append(j)
            graph[j].append(i)
        comps = _lifted_components_covering_all_nodes(n, graph)
        return len([c for c in comps if len(c) >= max(3, int(np.ceil(n * 0.2)))])

    conj_majors = [_majors(s, cfg_conj) for s in range(3)]
    assert all(m <= 1 for m in conj_majors)
    # Default seed-0 probe remains the known spurious K=2 (Gabriel path).
    assert _majors(0, cfg_def) == 2


def test_hollow_persist_agree_couples_at_candidate_taus() -> None:
    """#44 / A2-T32: hollow+persist-agree stays 1 leaf on uniforms; flag off.

    Couples hollow prepass to persistence agreement at the region's
    scale-search result.  Nested/tori remain unrecovered (no awaiting flip).
    """

    from tests.datasets.synthetic.linked_tori import make_linked_tori
    from tests.datasets.synthetic.nested_spheres import make_nested_spheres
    from tests.datasets.synthetic.swiss_roll import make_swiss_roll

    def _lean() -> ScaleSearchConfig:
        return ScaleSearchConfig(
            tau_min=1e-3,
            tau_max=2.0,
            max_grid_points=6,
            k=8,
            n_seeds=8,
            ann_backend="naive",
            stabilization=StabilizationConfig(
                min_equilibrium_epochs=2, max_epochs=8,
            ),
            seed=42,
            selector="persistence",
            record_partitions=True,
        )

    def _run(points, dim, *, conj: bool, persist_agree: bool, min_samples: int) -> int:
        cfg = RecursionConfig(
            scale_search=_lean(),
            min_samples=min_samples,
            max_depth=3,
            require_persistent_split=True,
            allow_finer_research=False,
            prefer_hollow_edge_prepass=True,
            hollow_require_gabriel_and_h=conj,
            hollow_require_persistent_agree=persist_agree,
            seed=42,
        )
        tree = run_recursive_discovery(points, dim=dim, config=cfg)
        return len(tree.leaves)

    circle = make_circle(
        n_samples=300, radius=1.0, noise=0.02, extrusion_dim=2, seed=0,
    )
    swiss = make_swiss_roll(n_samples=400, noise=0.02, seed=0)
    nested = make_nested_spheres(n_per_sphere=80, extrusion_dim=1, seed=0)
    tori = make_linked_tori(n_per_torus=120, seed=0)

    assert _run(circle.points, circle.points.shape[1], conj=True, persist_agree=True, min_samples=80) == 1
    assert _run(swiss.points, swiss.points.shape[1], conj=True, persist_agree=True, min_samples=80) == 1
    # Coupled hollow does not recover nested/tori at operational tau*.
    assert _run(nested.points, nested.points.shape[1], conj=True, persist_agree=True, min_samples=40) == 1
    assert _run(tori.points, tori.points.shape[1], conj=True, persist_agree=True, min_samples=40) == 1


def _hollow_majors_and_sample_ari(sc, points, labels, cfg: "HollowEdgeConfig"):
    """Shared majors + nearest-node sample ARI helper (A2-T33..T35)."""

    from proteus.stage1.clustering import _lifted_components_covering_all_nodes
    from proteus.stage1.edge_evidence import prune_hollow_edges
    from sklearn.metrics import adjusted_rand_score

    n = len(sc.nodes)
    pos = np.asarray([sc.nodes[i].position for i in range(n)])
    edges = [(int(l.i), int(l.j)) for l in sc.links.lifted_links()]
    kept = prune_hollow_edges(pos, edges, points, config=cfg)
    graph = {i: [] for i in range(n)}
    for i, j in kept:
        graph[i].append(j)
        graph[j].append(i)
    comps = _lifted_components_covering_all_nodes(n, graph)
    majors = [c for c in comps if len(c) >= max(3, int(np.ceil(n * 0.2)))]
    ari = None
    if labels is not None and len(majors) >= 2:
        lab = np.full(n, -1, dtype=int)
        for cid, c in enumerate(majors):
            for m in c:
                lab[m] = cid
        nn = np.argmin(
            ((points[:, None, :] - pos[None, :, :]) ** 2).sum(-1), axis=1,
        )
        pred = lab[nn]
        mask = np.asarray(labels) >= 0
        ari = float(adjusted_rand_score(labels[mask], pred[mask]))
    return len(majors), ari


def test_a4_primary_hollow_sample_ari_suite() -> None:
    """#44 / A2-T33: A4 primary (mid=0.5,h0=0.7,noGab) sample-ARI suite.

    Documents majors+ARI on nested/tori/zoo/swiss/circle under the A4 ROC
    primary preset.  Uniforms stay ≤1 major; nested/tori are **not**
    sample-ARI recovered — do **not** flip awaiting.  Flag
    ``hollow_use_a4_primary`` remains default-off.
    """

    from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
    from proteus.stage1.edge_evidence import a4_roc_primary_config
    from proteus.stage1.scaffold import Stage1Scaffold
    from tests.datasets.synthetic.linked_tori import make_linked_tori
    from tests.datasets.synthetic.manifold_zoo import make_manifold_zoo
    from tests.datasets.synthetic.nested_spheres import make_nested_spheres
    from tests.datasets.synthetic.swiss_roll import make_swiss_roll

    assert RecursionConfig().hollow_use_a4_primary is False
    cfg = a4_roc_primary_config()
    assert cfg.mid_radius_frac == 0.5 and cfg.h0 == 0.7
    assert cfg.gabriel_fallback is False

    def _lean() -> ScaleSearchConfig:
        return ScaleSearchConfig(
            tau_min=1e-3,
            tau_max=2.0,
            max_grid_points=6,
            k=8,
            n_seeds=8,
            ann_backend="naive",
            stabilization=StabilizationConfig(
                min_equilibrium_epochs=2, max_epochs=8,
            ),
            seed=42,
        )

    def _adapt(points, tau: float):
        sc = Stage1Scaffold(
            dim=int(points.shape[1]), tau=float(tau), k=8, max_nodes=64,
            ann_backend="naive", rng=np.random.default_rng(0),
        )
        sc.init_from(points, n_seeds=8)
        sc.run_until_stable(
            points,
            StabilizationConfig(max_epochs=30, min_equilibrium_epochs=3),
        )
        return sc

    circle = make_circle(
        n_samples=300, radius=1.0, noise=0.02, extrusion_dim=2, seed=0,
    )
    swiss = make_swiss_roll(n_samples=400, noise=0.02, seed=0)
    zoo = make_manifold_zoo(seed=0)
    nested = make_nested_spheres(n_per_sphere=80, extrusion_dim=1, seed=0)
    tori = make_linked_tori(n_per_torus=120, seed=0)

    # Uniforms / zoo at tau*: A4 primary stays connected (≤1 major).
    for ds, labels in (
        (circle, None),
        (swiss, None),
        (zoo, None),
    ):
        pts = ds.points
        if pts.shape[0] > 600:
            pts = pts[np.random.default_rng(0).choice(pts.shape[0], 600, replace=False)]
        r = run_scale_search(pts, dim=pts.shape[1], config=_lean())
        sc = _adapt(pts, float(r.tau_star))
        maj, _ = _hollow_majors_and_sample_ari(sc, pts, labels, cfg)
        assert maj <= 1

    # Nested / tori: report majors+ARI; K≥2 alone is not recovery.
    for ds, labels in (
        (nested, nested.labels),
        (tori, tori.labels),
    ):
        r = run_scale_search(ds.points, dim=ds.points.shape[1], config=_lean())
        et = float(ds.ground_truth.expected_tau)
        for tau in (float(r.tau_star), et, 0.27 if ds is nested else 0.5):
            sc = _adapt(ds.points, tau)
            maj, ari = _hollow_majors_and_sample_ari(sc, ds.points, labels, cfg)
            if maj >= 2:
                assert ari is not None and ari < 0.5  # not sample-ARI recovery
            else:
                assert maj <= 1

    # E2E recursion flag path: hollow_use_a4_primary still unrecovered.
    def _e2e(points, dim, min_samples: int) -> int:
        tree = run_recursive_discovery(
            points,
            dim=dim,
            config=RecursionConfig(
                scale_search=_lean(),
                min_samples=min_samples,
                max_depth=3,
                require_persistent_split=True,
                prefer_hollow_edge_prepass=True,
                hollow_use_a4_primary=True,
                seed=42,
            ),
        )
        return len(tree.leaves)

    assert _e2e(circle.points, circle.points.shape[1], 80) == 1
    assert _e2e(swiss.points, swiss.points.shape[1], 80) == 1
    assert _e2e(nested.points, nested.points.shape[1], 40) == 1
    # Tori may emit a spurious 2-leaf under A4 primary (non-cut-set /
    # coarse-tau artifact) — not sample-ARI recovery; do not flip awaiting.
    tori_leaves = _e2e(tori.points, tori.points.shape[1], 40)
    assert tori_leaves in (1, 2)


def test_mst_critical_hollow_contrast_vs_h_and_conj() -> None:
    """#44 / A2-T34: MST-critical hollow vs H-only and Gabriel∧H (majors+ARI).

    On nested@0.27 default H|Gab yields majors=2 ARI~chance; conjunction and
    MST-critical (A4 primary) stay ≤1 major.  Flag default-off.
    """

    from proteus.stage1.edge_evidence import HollowEdgeConfig, a4_roc_primary_config
    from proteus.stage1.scaffold import Stage1Scaffold
    from tests.datasets.synthetic.linked_tori import make_linked_tori
    from tests.datasets.synthetic.nested_spheres import make_nested_spheres

    assert RecursionConfig().hollow_mst_critical_only is False

    nested = make_nested_spheres(n_per_sphere=80, extrusion_dim=1, seed=0)
    tori = make_linked_tori(n_per_torus=120, seed=0)

    def _adapt(points, tau: float):
        sc = Stage1Scaffold(
            dim=int(points.shape[1]), tau=float(tau), k=8, max_nodes=64,
            ann_backend="naive", rng=np.random.default_rng(0),
        )
        sc.init_from(points, n_seeds=8)
        sc.run_until_stable(
            points,
            StabilizationConfig(max_epochs=30, min_equilibrium_epochs=3),
        )
        return sc

    cfg_def = HollowEdgeConfig()
    cfg_conj = HollowEdgeConfig(require_gabriel_and_h=True)
    cfg_mst = a4_roc_primary_config(mst_critical_only=True)
    cfg_a4 = a4_roc_primary_config()

    sc_n = _adapt(nested.points, 0.27)
    maj_def, ari_def = _hollow_majors_and_sample_ari(
        sc_n, nested.points, nested.labels, cfg_def,
    )
    maj_conj, _ = _hollow_majors_and_sample_ari(
        sc_n, nested.points, nested.labels, cfg_conj,
    )
    maj_mst, ari_mst = _hollow_majors_and_sample_ari(
        sc_n, nested.points, nested.labels, cfg_mst,
    )
    maj_a4, ari_a4 = _hollow_majors_and_sample_ari(
        sc_n, nested.points, nested.labels, cfg_a4,
    )
    assert maj_def == 2
    assert ari_def is not None and ari_def < 0.2
    assert maj_conj <= 1
    assert maj_mst <= 1
    # A4 primary alone: if it emits K≥2, ARI must stay below recovery.
    if maj_a4 >= 2:
        assert ari_a4 is not None and ari_a4 < 0.5
    else:
        assert maj_a4 <= 1

    sc_t = _adapt(tori.points, 0.5)
    maj_t_def, ari_t = _hollow_majors_and_sample_ari(
        sc_t, tori.points, tori.labels, cfg_def,
    )
    maj_t_mst, _ = _hollow_majors_and_sample_ari(
        sc_t, tori.points, tori.labels, cfg_mst,
    )
    assert maj_t_def == 2
    assert ari_t is not None and ari_t < 0.2
    assert maj_t_mst <= maj_t_def


def test_hollow_recovery_requires_sample_ari_not_k() -> None:
    """#44 / A2-T35: recovery harness — sample ARI gate, not K majors.

    Documents two failure modes that produce misleading ``K=2``:
    1. **Empty-ball / Gabriel**: mid-ball empty ⇒ H≈0 or Gabriel-only cut
       with sample ARI≈chance (nested@0.27 default).
    2. **Non-cut-set**: hollow edges that are not bridges leave redundant
       paths; MST-critical intersection suppresses spurious majors.

    Any future recovery claim must assert sample ARI (typically ≫ chance),
    not major-CC count alone.  Flags stay default-off; no awaiting flip.
    """

    from proteus.stage1.edge_evidence import (
        HollowEdgeConfig,
        a4_roc_primary_config,
        edge_ball_occupancy,
    )
    from proteus.stage1.scaffold import Stage1Scaffold
    from tests.datasets.synthetic.nested_spheres import make_nested_spheres

    nested = make_nested_spheres(n_per_sphere=80, extrusion_dim=1, seed=0)
    sc = Stage1Scaffold(
        dim=int(nested.points.shape[1]), tau=0.27, k=8, max_nodes=64,
        ann_backend="naive", rng=np.random.default_rng(0),
    )
    sc.init_from(nested.points, n_seeds=8)
    sc.run_until_stable(
        nested.points,
        StabilizationConfig(max_epochs=30, min_equilibrium_epochs=3),
    )
    n = len(sc.nodes)
    pos = np.asarray([sc.nodes[i].position for i in range(n)])
    edges = [(int(l.i), int(l.j)) for l in sc.links.lifted_links()]

    # Failure mode 1: empty-ball regime at mid_frac=0.35 (non-discriminative H).
    n_mid, n_end, _ = edge_ball_occupancy(
        pos, edges, nested.points, mid_radius_frac=0.35,
    )
    empty_ball_frac = float(np.mean(n_mid <= 0.0))
    assert empty_ball_frac > 0.5  # majority empty mid-balls on adapted scaffold

    maj_def, ari_def = _hollow_majors_and_sample_ari(
        sc, nested.points, nested.labels, HollowEdgeConfig(),
    )
    # Spurious K=2 with ARI near chance — K alone must not claim recovery.
    assert maj_def == 2
    assert ari_def is not None and ari_def < 0.2

    # Failure mode 2: non-cut-set — MST-critical hollow keeps majors ≤1 here.
    maj_mst, ari_mst = _hollow_majors_and_sample_ari(
        sc, nested.points, nested.labels,
        a4_roc_primary_config(mst_critical_only=True),
    )
    assert maj_mst <= 1
    assert ari_mst is None  # no multi-major ⇒ no ARI recovery claim

    # Contract: a recovery predicate requires ARI threshold, not K.
    def _claims_recovery(majors: int, ari: float | None, *, ari_min: float = 0.5) -> bool:
        return majors >= 2 and ari is not None and ari >= ari_min

    assert not _claims_recovery(maj_def, ari_def)
    assert not _claims_recovery(maj_mst, ari_mst)


def test_bridge_critical_hollow_vs_mst_on_nested() -> None:
    """#44: bridge-critical (true cut-set) vs MST-critical on nested@0.27.

    Capacity/flow beyond MST: graph bridges ∩ hollow.  On adapted nested
    scaffolds both suppress spurious default K=2; neither recovers sample
    ARI.  Flags default-off.
    """

    from proteus.stage1.edge_evidence import HollowEdgeConfig, a4_roc_primary_config
    from proteus.stage1.scaffold import Stage1Scaffold
    from tests.datasets.synthetic.nested_spheres import make_nested_spheres

    assert RecursionConfig().hollow_bridge_critical_only is False

    nested = make_nested_spheres(n_per_sphere=80, extrusion_dim=1, seed=0)
    sc = Stage1Scaffold(
        dim=int(nested.points.shape[1]), tau=0.27, k=8, max_nodes=64,
        ann_backend="naive", rng=np.random.default_rng(0),
    )
    sc.init_from(nested.points, n_seeds=8)
    sc.run_until_stable(
        nested.points,
        StabilizationConfig(max_epochs=30, min_equilibrium_epochs=3),
    )

    maj_def, ari_def = _hollow_majors_and_sample_ari(
        sc, nested.points, nested.labels, HollowEdgeConfig(),
    )
    maj_mst, _ = _hollow_majors_and_sample_ari(
        sc, nested.points, nested.labels,
        a4_roc_primary_config(mst_critical_only=True),
    )
    maj_br, ari_br = _hollow_majors_and_sample_ari(
        sc, nested.points, nested.labels,
        a4_roc_primary_config(bridge_critical_only=True),
    )
    assert maj_def == 2
    assert ari_def is not None and ari_def < 0.2
    assert maj_mst <= 1
    assert maj_br <= 1
    # Bridge-critical ⊆ cut-set ⇒ no multi-major recovery claim.
    assert ari_br is None or ari_br < 0.5


def test_a4_primary_multi_seed_sample_ari_table() -> None:
    """#44: multi-seed A4 primary majors+ARI table (nested/tori probe taus).

    Seeds 0..2 at nested@0.27 and tori@0.5 under A4 primary (and
    bridge-critical).  Documents seed fragility; never treat K≥2 as
    recovery without sample ARI.  No awaiting flip.
    """

    from proteus.stage1.edge_evidence import a4_roc_primary_config
    from proteus.stage1.scaffold import Stage1Scaffold
    from tests.datasets.synthetic.linked_tori import make_linked_tori
    from tests.datasets.synthetic.nested_spheres import make_nested_spheres

    cfg_a4 = a4_roc_primary_config()
    cfg_br = a4_roc_primary_config(bridge_critical_only=True)
    nested = make_nested_spheres(n_per_sphere=80, extrusion_dim=1, seed=0)
    tori = make_linked_tori(n_per_torus=120, seed=0)

    def _adapt(points, tau: float, seed: int):
        sc = Stage1Scaffold(
            dim=int(points.shape[1]), tau=float(tau), k=8, max_nodes=64,
            ann_backend="naive", rng=np.random.default_rng(seed),
        )
        sc.init_from(points, n_seeds=8)
        sc.run_until_stable(
            points,
            StabilizationConfig(max_epochs=30, min_equilibrium_epochs=3),
        )
        return sc

    recovered = 0
    for seed in (0, 1, 2):
        for ds, labels, tau in (
            (nested, nested.labels, 0.27),
            (tori, tori.labels, 0.5),
        ):
            sc = _adapt(ds.points, tau, seed)
            for cfg in (cfg_a4, cfg_br):
                maj, ari = _hollow_majors_and_sample_ari(
                    sc, ds.points, labels, cfg,
                )
                if maj >= 2:
                    assert ari is not None and ari < 0.5
                else:
                    assert maj <= 1
                if maj >= 2 and ari is not None and ari >= 0.5:
                    recovered += 1
    assert recovered == 0  # no sample-ARI recovery across seeds/cfgs


def test_denser_scaffold_hollow_ari_a4_primary_bridge() -> None:
    """#44 / A2-T36: denser n/max_nodes hollow majors+ARI under A4+bridge.

    Baseline harnesses use ``max_nodes=64`` / modest ``n``.  Raising sample
    density and scaffold capacity can change empty-ball / cut-set geometry.
    This documents majors+sample-ARI on denser nested/tori scaffolds under
    A4 primary and bridge-critical (flags stay default-off).  Denser nests
    often collapse spurious K=2 to 1 major; when K≥2 appears (e.g. denser
    tori under A4), sample ARI stays near chance — **not** recovery.  Do
    **not** flip awaiting.
    """

    from proteus.stage1.edge_evidence import HollowEdgeConfig, a4_roc_primary_config
    from proteus.stage1.scaffold import Stage1Scaffold
    from tests.datasets.synthetic.linked_tori import make_linked_tori
    from tests.datasets.synthetic.nested_spheres import make_nested_spheres

    assert RecursionConfig().prefer_hollow_edge_prepass is False
    assert RecursionConfig().hollow_use_a4_primary is False
    assert RecursionConfig().hollow_bridge_critical_only is False

    cfg_a4 = a4_roc_primary_config()
    cfg_br = a4_roc_primary_config(bridge_critical_only=True)
    cfg_def = HollowEdgeConfig()

    def _adapt(points, tau: float, *, max_nodes: int, seed: int = 0):
        sc = Stage1Scaffold(
            dim=int(points.shape[1]),
            tau=float(tau),
            k=8,
            max_nodes=int(max_nodes),
            ann_backend="naive",
            rng=np.random.default_rng(seed),
        )
        sc.init_from(points, n_seeds=8)
        sc.run_until_stable(
            points,
            StabilizationConfig(max_epochs=30, min_equilibrium_epochs=3),
        )
        return sc

    # (dataset_builder_kwargs, tau, max_nodes) denser than baseline 80/64 & 120/64.
    nested_cases = (
        ({"n_per_sphere": 160, "extrusion_dim": 1, "seed": 0}, 0.27, 96),
        ({"n_per_sphere": 240, "extrusion_dim": 1, "seed": 0}, 0.27, 128),
    )
    tori_cases = (
        ({"n_per_torus": 200, "seed": 0}, 0.5, 96),
        ({"n_per_torus": 240, "seed": 0}, 0.5, 128),
    )

    recovered = 0
    denser_nested_collapsed = 0
    for kwargs, tau, max_nodes in nested_cases:
        ds = make_nested_spheres(**kwargs)
        sc = _adapt(ds.points, tau, max_nodes=max_nodes)
        assert len(sc.nodes) > 64 or max_nodes > 64
        for cfg in (cfg_def, cfg_a4, cfg_br):
            maj, ari = _hollow_majors_and_sample_ari(
                sc, ds.points, ds.labels, cfg,
            )
            if maj >= 2:
                assert ari is not None and ari < 0.5
            else:
                assert maj <= 1
            if maj <= 1:
                denser_nested_collapsed += 1
            if maj >= 2 and ari is not None and ari >= 0.5:
                recovered += 1

    # At least one denser nested setting collapses A4/bridge to ≤1 major
    # (density suppresses the baseline empty-ball K=2 artifact).
    assert denser_nested_collapsed >= 1

    for kwargs, tau, max_nodes in tori_cases:
        ds = make_linked_tori(**kwargs)
        sc = _adapt(ds.points, tau, max_nodes=max_nodes)
        for cfg in (cfg_a4, cfg_br):
            maj, ari = _hollow_majors_and_sample_ari(
                sc, ds.points, ds.labels, cfg,
            )
            if maj >= 2:
                assert ari is not None and ari < 0.5
            else:
                assert maj <= 1
            if maj >= 2 and ari is not None and ari >= 0.5:
                recovered += 1
        # Bridge-critical remains a strict cut-set filter on denser tori.
        maj_br, ari_br = _hollow_majors_and_sample_ari(
            sc, ds.points, ds.labels, cfg_br,
        )
        assert maj_br <= 1
        assert ari_br is None or ari_br < 0.5

    assert recovered == 0  # denser scaffolds do not sample-ARI recover


def test_soft_capacity_hollow_contrast_vs_mst_bridge() -> None:
    """#44 / A2-T37: soft-capacity (betweenness) vs MST/bridge majors+ARI.

    On nested@0.27, default/A4 primary can emit spurious K=2 with sample
    ARI≈chance.  Soft-capacity (``soft_capacity_frac=0.25``) intersects
    hollow with high-betweenness edges and collapses majors to ≤1 like
    MST-critical and bridge-critical — still **not** sample-ARI recovery.
    Flags default-off; no awaiting flip.
    """

    from proteus.stage1.edge_evidence import HollowEdgeConfig, a4_roc_primary_config
    from proteus.stage1.scaffold import Stage1Scaffold
    from tests.datasets.synthetic.linked_tori import make_linked_tori
    from tests.datasets.synthetic.nested_spheres import make_nested_spheres

    assert RecursionConfig().hollow_soft_capacity_only is False
    assert RecursionConfig().hollow_soft_capacity_frac == 0.25

    nested = make_nested_spheres(n_per_sphere=80, extrusion_dim=1, seed=0)
    tori = make_linked_tori(n_per_torus=120, seed=0)

    def _adapt(points, tau: float):
        sc = Stage1Scaffold(
            dim=int(points.shape[1]), tau=float(tau), k=8, max_nodes=64,
            ann_backend="naive", rng=np.random.default_rng(0),
        )
        sc.init_from(points, n_seeds=8)
        sc.run_until_stable(
            points,
            StabilizationConfig(max_epochs=30, min_equilibrium_epochs=3),
        )
        return sc

    cfg_def = HollowEdgeConfig()
    cfg_a4 = a4_roc_primary_config()
    cfg_mst = a4_roc_primary_config(mst_critical_only=True)
    cfg_br = a4_roc_primary_config(bridge_critical_only=True)
    cfg_soft = a4_roc_primary_config(soft_capacity_only=True, soft_capacity_frac=0.25)
    cfg_soft_def = HollowEdgeConfig(soft_capacity_only=True, soft_capacity_frac=0.25)

    sc_n = _adapt(nested.points, 0.27)
    maj_def, ari_def = _hollow_majors_and_sample_ari(
        sc_n, nested.points, nested.labels, cfg_def,
    )
    maj_a4, ari_a4 = _hollow_majors_and_sample_ari(
        sc_n, nested.points, nested.labels, cfg_a4,
    )
    maj_mst, _ = _hollow_majors_and_sample_ari(
        sc_n, nested.points, nested.labels, cfg_mst,
    )
    maj_br, _ = _hollow_majors_and_sample_ari(
        sc_n, nested.points, nested.labels, cfg_br,
    )
    maj_soft, ari_soft = _hollow_majors_and_sample_ari(
        sc_n, nested.points, nested.labels, cfg_soft,
    )
    maj_soft_def, _ = _hollow_majors_and_sample_ari(
        sc_n, nested.points, nested.labels, cfg_soft_def,
    )

    assert maj_def == 2
    assert ari_def is not None and ari_def < 0.2
    if maj_a4 >= 2:
        assert ari_a4 is not None and ari_a4 < 0.5
    assert maj_mst <= 1
    assert maj_br <= 1
    # Soft capacity collapses spurious A4/default K=2 like hard cut-set filters.
    assert maj_soft <= 1
    assert maj_soft_def <= 1
    assert ari_soft is None or ari_soft < 0.5

    sc_t = _adapt(tori.points, 0.5)
    recovered = 0
    for cfg in (cfg_a4, cfg_mst, cfg_br, cfg_soft):
        maj, ari = _hollow_majors_and_sample_ari(
            sc_t, tori.points, tori.labels, cfg,
        )
        if maj >= 2:
            assert ari is not None and ari < 0.5
        else:
            assert maj <= 1
        if maj >= 2 and ari is not None and ari >= 0.5:
            recovered += 1
    assert recovered == 0


def test_soft_capacity_denser_scaffold_ari_combo() -> None:
    """#44 / A2-T39: soft-cap × denser n/max_nodes majors+ARI (A4 primary).

    Combines T36 denser scaffolds with T37 soft-capacity (betweenness and
    bridge_mass methods).  Expect collapse of nested spurious K=2 and no
    sample-ARI recovery on nested/tori.  Flags remain default-off; do
    **not** flip awaiting.
    """

    from proteus.stage1.edge_evidence import HollowEdgeConfig, a4_roc_primary_config
    from proteus.stage1.scaffold import Stage1Scaffold
    from tests.datasets.synthetic.linked_tori import make_linked_tori
    from tests.datasets.synthetic.nested_spheres import make_nested_spheres

    assert RecursionConfig().hollow_soft_capacity_only is False
    assert RecursionConfig().hollow_soft_capacity_method == "betweenness"

    cfg_a4 = a4_roc_primary_config()
    cfg_soft = a4_roc_primary_config(
        soft_capacity_only=True, soft_capacity_frac=0.25,
        soft_capacity_method="betweenness",
    )
    cfg_mass = a4_roc_primary_config(
        soft_capacity_only=True, soft_capacity_frac=0.25,
        soft_capacity_method="bridge_mass",
    )
    cfg_soft_def = HollowEdgeConfig(
        soft_capacity_only=True, soft_capacity_frac=0.25,
        soft_capacity_method="betweenness",
    )

    def _adapt(points, tau: float, *, max_nodes: int, seed: int = 0):
        sc = Stage1Scaffold(
            dim=int(points.shape[1]),
            tau=float(tau),
            k=8,
            max_nodes=int(max_nodes),
            ann_backend="naive",
            rng=np.random.default_rng(seed),
        )
        sc.init_from(points, n_seeds=8)
        sc.run_until_stable(
            points,
            StabilizationConfig(max_epochs=30, min_equilibrium_epochs=3),
        )
        return sc

    nested_cases = (
        ({"n_per_sphere": 160, "extrusion_dim": 1, "seed": 0}, 0.27, 96),
        ({"n_per_sphere": 240, "extrusion_dim": 1, "seed": 0}, 0.27, 128),
    )
    tori_cases = (
        ({"n_per_torus": 200, "seed": 0}, 0.5, 96),
        ({"n_per_torus": 240, "seed": 0}, 0.5, 128),
    )

    recovered = 0
    soft_nested_collapsed = 0
    for kwargs, tau, max_nodes in nested_cases:
        ds = make_nested_spheres(**kwargs)
        sc = _adapt(ds.points, tau, max_nodes=max_nodes)
        maj_a4, ari_a4 = _hollow_majors_and_sample_ari(
            sc, ds.points, ds.labels, cfg_a4,
        )
        for cfg in (cfg_soft, cfg_mass, cfg_soft_def):
            maj, ari = _hollow_majors_and_sample_ari(
                sc, ds.points, ds.labels, cfg,
            )
            if maj >= 2:
                assert ari is not None and ari < 0.5
            else:
                assert maj <= 1
                soft_nested_collapsed += 1
            if maj >= 2 and ari is not None and ari >= 0.5:
                recovered += 1
        if maj_a4 >= 2 and ari_a4 is not None:
            assert ari_a4 < 0.5

    # Soft-capacity (either method) collapses at least one denser nested case.
    assert soft_nested_collapsed >= 1

    for kwargs, tau, max_nodes in tori_cases:
        ds = make_linked_tori(**kwargs)
        sc = _adapt(ds.points, tau, max_nodes=max_nodes)
        for cfg in (cfg_a4, cfg_soft, cfg_mass):
            maj, ari = _hollow_majors_and_sample_ari(
                sc, ds.points, ds.labels, cfg,
            )
            if maj >= 2:
                assert ari is not None and ari < 0.5
            else:
                assert maj <= 1
            if maj >= 2 and ari is not None and ari >= 0.5:
                recovered += 1
        # Soft betweenness and bridge_mass remain ≤1 major on denser tori
        # (capacity filters suppress A4's occasional chance-ARI K=2).
        maj_s, _ = _hollow_majors_and_sample_ari(
            sc, ds.points, ds.labels, cfg_soft,
        )
        maj_m, _ = _hollow_majors_and_sample_ari(
            sc, ds.points, ds.labels, cfg_mass,
        )
        assert maj_s <= 1
        assert maj_m <= 1

    assert recovered == 0  # soft-cap×denser does not sample-ARI recover


def test_bridge_mass_soft_capacity_vs_betweenness_nested() -> None:
    """#44 / A2-T39: bridge_mass soft-cap collapses nested@0.27 like bet.

    On baseline nested@0.27, A4 primary can emit spurious K=2; both
    betweenness and bridge_mass soft-capacity (frac=0.25) collapse to ≤1
    major.  Default method stays betweenness; no awaiting flip.
    """

    from proteus.stage1.edge_evidence import a4_roc_primary_config
    from proteus.stage1.scaffold import Stage1Scaffold
    from tests.datasets.synthetic.nested_spheres import make_nested_spheres

    nested = make_nested_spheres(n_per_sphere=80, extrusion_dim=1, seed=0)
    sc = Stage1Scaffold(
        dim=int(nested.points.shape[1]), tau=0.27, k=8, max_nodes=64,
        ann_backend="naive", rng=np.random.default_rng(0),
    )
    sc.init_from(nested.points, n_seeds=8)
    sc.run_until_stable(
        nested.points,
        StabilizationConfig(max_epochs=30, min_equilibrium_epochs=3),
    )
    cfg_bet = a4_roc_primary_config(
        soft_capacity_only=True, soft_capacity_frac=0.25,
        soft_capacity_method="betweenness",
    )
    cfg_mass = a4_roc_primary_config(
        soft_capacity_only=True, soft_capacity_frac=0.25,
        soft_capacity_method="bridge_mass",
    )
    maj_b, ari_b = _hollow_majors_and_sample_ari(
        sc, nested.points, nested.labels, cfg_bet,
    )
    maj_m, ari_m = _hollow_majors_and_sample_ari(
        sc, nested.points, nested.labels, cfg_mass,
    )
    assert maj_b <= 1
    assert maj_m <= 1
    assert ari_b is None or ari_b < 0.5
    assert ari_m is None or ari_m < 0.5
    assert RecursionConfig().hollow_soft_capacity_method == "betweenness"


def test_soft_capacity_frac_sweep_nested_tori_ari() -> None:
    """#44 / A2-T40: soft_capacity_frac sweep majors+ARI (A4 primary).

    Nested@0.27 collapses A4's spurious K=2 across frac∈{0.1,0.25,0.5,0.9}.
    Tori@0.5 keeps chance-ARI K=2 until frac=0.9 collapses to 1 major.
    Flags default-off; do **not** flip awaiting.
    """

    from proteus.stage1.edge_evidence import (
        SOFT_CAPACITY_FRAC_SWEEP_NESTED_MAJORS,
        SOFT_CAPACITY_FRAC_SWEEP_TORI,
        a4_roc_primary_config,
    )
    from proteus.stage1.scaffold import Stage1Scaffold
    from tests.datasets.synthetic.linked_tori import make_linked_tori
    from tests.datasets.synthetic.nested_spheres import make_nested_spheres

    assert RecursionConfig().hollow_soft_capacity_only is False
    assert RecursionConfig().hollow_soft_capacity_frac == 0.25

    nested = make_nested_spheres(n_per_sphere=80, extrusion_dim=1, seed=0)
    tori = make_linked_tori(n_per_torus=120, seed=0)

    def _adapt(points, tau: float):
        sc = Stage1Scaffold(
            dim=int(points.shape[1]), tau=float(tau), k=8, max_nodes=64,
            ann_backend="naive", rng=np.random.default_rng(0),
        )
        sc.init_from(points, n_seeds=8)
        sc.run_until_stable(
            points,
            StabilizationConfig(max_epochs=30, min_equilibrium_epochs=3),
        )
        return sc

    sc_n = _adapt(nested.points, 0.27)
    maj_a4, ari_a4 = _hollow_majors_and_sample_ari(
        sc_n, nested.points, nested.labels, a4_roc_primary_config(),
    )
    assert maj_a4 == 2
    assert ari_a4 is not None and ari_a4 < 0.2

    recovered = 0
    for frac, expect_maj in SOFT_CAPACITY_FRAC_SWEEP_NESTED_MAJORS.items():
        cfg = a4_roc_primary_config(
            soft_capacity_only=True, soft_capacity_frac=float(frac),
        )
        maj, ari = _hollow_majors_and_sample_ari(
            sc_n, nested.points, nested.labels, cfg,
        )
        assert maj == expect_maj
        assert maj <= 1
        assert ari is None or ari < 0.5
        if maj >= 2 and ari is not None and ari >= 0.5:
            recovered += 1

    sc_t = _adapt(tori.points, 0.5)
    for frac, (expect_maj, expect_ari) in SOFT_CAPACITY_FRAC_SWEEP_TORI.items():
        cfg = a4_roc_primary_config(
            soft_capacity_only=True, soft_capacity_frac=float(frac),
        )
        maj, ari = _hollow_majors_and_sample_ari(
            sc_t, tori.points, tori.labels, cfg,
        )
        assert maj == expect_maj
        if expect_maj >= 2:
            assert ari is not None and ari < 0.5
            if expect_ari is not None:
                assert abs(ari - expect_ari) < 0.08
        else:
            assert ari is None or ari < 0.5
        if maj >= 2 and ari is not None and ari >= 0.5:
            recovered += 1
    assert recovered == 0


def test_soft_capacity_persist_agree_leaf_harness() -> None:
    """#44 / A2-T40: soft-cap × persist-agree leaf harness (A4 primary).

    Uniforms stay 1 leaf; soft alone collapses A4 tori K=2→1; soft+persist
    stays unrecovered on nested/tori.  Flags default-off; no awaiting flip.
    """

    from tests.datasets.synthetic.linked_tori import make_linked_tori
    from tests.datasets.synthetic.nested_spheres import make_nested_spheres
    from tests.datasets.synthetic.swiss_roll import make_swiss_roll

    assert RecursionConfig().hollow_soft_capacity_only is False
    assert RecursionConfig().hollow_require_persistent_agree is False

    def _lean() -> ScaleSearchConfig:
        return ScaleSearchConfig(
            tau_min=1e-3,
            tau_max=2.0,
            max_grid_points=6,
            k=8,
            n_seeds=8,
            ann_backend="naive",
            stabilization=StabilizationConfig(
                min_equilibrium_epochs=2, max_epochs=8,
            ),
            seed=42,
            selector="persistence",
            record_partitions=True,
        )

    def _run(points, dim, *, soft: bool, persist: bool, frac: float,
             min_samples: int) -> int:
        cfg = RecursionConfig(
            scale_search=_lean(),
            min_samples=min_samples,
            max_depth=3,
            require_persistent_split=True,
            allow_finer_research=False,
            prefer_hollow_edge_prepass=True,
            hollow_use_a4_primary=True,
            hollow_soft_capacity_only=soft,
            hollow_soft_capacity_frac=frac,
            hollow_require_persistent_agree=persist,
            seed=42,
        )
        tree = run_recursive_discovery(points, dim=dim, config=cfg)
        return len(tree.leaves)

    circle = make_circle(
        n_samples=300, radius=1.0, noise=0.02, extrusion_dim=2, seed=0,
    )
    swiss = make_swiss_roll(n_samples=400, noise=0.02, seed=0)
    nested = make_nested_spheres(n_per_sphere=80, extrusion_dim=1, seed=0)
    tori = make_linked_tori(n_per_torus=120, seed=0)

    # Uniform-safe under soft×persist.
    assert _run(circle.points, circle.points.shape[1], soft=True, persist=True,
                frac=0.25, min_samples=80) == 1
    assert _run(swiss.points, swiss.points.shape[1], soft=True, persist=True,
                frac=0.25, min_samples=80) == 1

    # Soft collapses A4 tori's spurious 2-leaf hit; combo stays unrecovered.
    assert _run(tori.points, tori.points.shape[1], soft=False, persist=False,
                frac=0.25, min_samples=40) == 2
    assert _run(tori.points, tori.points.shape[1], soft=True, persist=False,
                frac=0.25, min_samples=40) == 1
    assert _run(tori.points, tori.points.shape[1], soft=True, persist=True,
                frac=0.25, min_samples=40) == 1
    assert _run(nested.points, nested.points.shape[1], soft=True, persist=True,
                frac=0.25, min_samples=40) == 1
    assert _run(nested.points, nested.points.shape[1], soft=True, persist=True,
                frac=0.1, min_samples=40) == 1
    assert _run(nested.points, nested.points.shape[1], soft=True, persist=True,
                frac=0.9, min_samples=40) == 1


def test_soft_x_gabriel_conj_nested_tori_ari() -> None:
    """#44 / A2-T41: soft×require_gabriel_and_h majors+sample-ARI table.

    Soft alone collapses nested@0.27 spurious K=2 but keeps tori@0.5
    chance-ARI K=2.  Gabriel∧H conjunction alone and soft×conj collapse
    both scaffolds to ≤1 major — still **not** sample-ARI recovery.
    Flags default-off; do **not** flip awaiting.
    """

    from proteus.stage1.edge_evidence import (
        SOFT_X_GABRIEL_CONJ_TABLE,
        a4_roc_primary_config,
    )
    from proteus.stage1.scaffold import Stage1Scaffold
    from tests.datasets.synthetic.linked_tori import make_linked_tori
    from tests.datasets.synthetic.nested_spheres import make_nested_spheres

    assert RecursionConfig().hollow_soft_capacity_only is False
    assert RecursionConfig().hollow_require_gabriel_and_h is False

    nested = make_nested_spheres(n_per_sphere=80, extrusion_dim=1, seed=0)
    tori = make_linked_tori(n_per_torus=120, seed=0)

    def _adapt(points, tau: float):
        sc = Stage1Scaffold(
            dim=int(points.shape[1]), tau=float(tau), k=8, max_nodes=64,
            ann_backend="naive", rng=np.random.default_rng(0),
        )
        sc.init_from(points, n_seeds=8)
        sc.run_until_stable(
            points,
            StabilizationConfig(max_epochs=30, min_equilibrium_epochs=3),
        )
        return sc

    cfg_a4 = a4_roc_primary_config()
    cfg_soft = a4_roc_primary_config(
        soft_capacity_only=True, soft_capacity_frac=0.25,
    )
    cfg_conj = a4_roc_primary_config(require_gabriel_and_h=True)
    cfg_soft_conj = a4_roc_primary_config(
        soft_capacity_only=True, soft_capacity_frac=0.25,
        require_gabriel_and_h=True,
    )

    sc_n = _adapt(nested.points, 0.27)
    sc_t = _adapt(tori.points, 0.5)

    recovered = 0
    live = {
        "a4": (
            *_hollow_majors_and_sample_ari(
                sc_n, nested.points, nested.labels, cfg_a4,
            ),
            *_hollow_majors_and_sample_ari(
                sc_t, tori.points, tori.labels, cfg_a4,
            ),
        ),
        "soft": (
            *_hollow_majors_and_sample_ari(
                sc_n, nested.points, nested.labels, cfg_soft,
            ),
            *_hollow_majors_and_sample_ari(
                sc_t, tori.points, tori.labels, cfg_soft,
            ),
        ),
        "conj": (
            *_hollow_majors_and_sample_ari(
                sc_n, nested.points, nested.labels, cfg_conj,
            ),
            *_hollow_majors_and_sample_ari(
                sc_t, tori.points, tori.labels, cfg_conj,
            ),
        ),
        "soft_x_conj": (
            *_hollow_majors_and_sample_ari(
                sc_n, nested.points, nested.labels, cfg_soft_conj,
            ),
            *_hollow_majors_and_sample_ari(
                sc_t, tori.points, tori.labels, cfg_soft_conj,
            ),
        ),
    }

    for mode, (nm, na, tm, ta) in live.items():
        exp_nm, exp_na, exp_tm, exp_ta = SOFT_X_GABRIEL_CONJ_TABLE[mode]
        assert nm == exp_nm
        assert tm == exp_tm
        if exp_na is not None:
            assert na is not None and abs(na - exp_na) < 0.08
        else:
            assert na is None or na < 0.5
        if exp_ta is not None:
            assert ta is not None and abs(ta - exp_ta) < 0.08
        else:
            assert ta is None or ta < 0.5
        for maj, ari in ((nm, na), (tm, ta)):
            if maj >= 2 and ari is not None and ari >= 0.5:
                recovered += 1

    # Soft×conj and conj collapse both; soft alone still leaves tori K=2.
    assert live["soft"][2] == 2
    assert live["soft_x_conj"][0] <= 1
    assert live["soft_x_conj"][2] <= 1
    assert recovered == 0
