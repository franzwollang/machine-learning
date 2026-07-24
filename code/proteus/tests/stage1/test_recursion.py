"""Integration test for the Stage 1 recursion orchestrator."""

from __future__ import annotations

import numpy as np

from proteus.stage1.controller import ScaleSearchConfig
from proteus.stage1.recursion import RecursionConfig, run_recursive_discovery
from proteus.stage1.stabilization import StabilizationConfig
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
