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
