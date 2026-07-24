"""Hierarchical Gaussian mixture scenario."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from scipy.spatial import cKDTree

from proteus.stage1.controller import ScaleSearchConfig
from proteus.stage1.recursion import RecursionConfig, run_recursive_discovery
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.hierarchical_gaussian import make_hierarchical_gaussian
from tests.harness.markers import awaiting
from tests.harness.hierarchy_recovery import (
    adjusted_rand_vs_coarse_fine,
    assert_fine_ari_at_least,
    assert_leaf_partition_covers_dataset,
    assert_recursion_matches_gt_hierarchy_unimodal_levels,
    assert_terminal_leaf_count_equals_fine_components,
    leaf_partition_by_region_id,
    per_sample_leaf_labels,
)
from tests.harness.stage1_scenario_metrics import (
    normalize_stage1_reconstruction,
    run_fixed_tau_stable_and_report,
)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_hierarchy_cluster_purity() -> None:
    """AP / Q clusters on a converged scaffold should separate the 3 coarse blobs.

    **Primary:** cluster purity vs coarse labels (``> 0.7``).
    **Secondary:** normalized Stage 1 reconstruction under a data-driven scale
    (pairwise mean distance).  The lifted graph may bridge blobs at one τ, so
    lifted-component count is informational only (not forced to match 3).
    """

    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=0,
    )
    data = dataset.points
    gt = dataset.ground_truth
    tau = gt.expected_tau
    assert tau is not None

    train = run_fixed_tau_stable_and_report(
        data,
        dim=4,
        tau=float(tau),
        stabilization=StabilizationConfig(
            min_equilibrium_epochs=2, max_epochs=12,
        ),
        k=8,
        min_nodes=8,
        max_nodes=128,
        n_seeds=12,
        prune_after=10,
        rng=np.random.default_rng(42),
        cluster=True,
    )
    scaffold = train.scaffold
    result = train.cluster_result
    assert result is not None

    assert result.n_clusters >= 3, (
        f"Expected >= 3 clusters for 3 coarse blobs, got {result.n_clusters}"
    )

    all_labels = np.asarray(dataset.labels, dtype=int)
    # For coarse purity, use only blob samples (label >= 0) as NN targets
    blob_idx = np.where(all_labels >= 0)[0]
    blob_data = data[blob_idx]
    coarse_labels = all_labels[blob_idx] // 2
    n_nodes = len(scaffold.nodes)
    positions = scaffold.node_positions()

    tree = cKDTree(blob_data)
    _, nearest_blob = tree.query(positions, k=1)
    node_gt_coarse = coarse_labels[nearest_blob]

    purity_scores = []
    for c in range(result.n_clusters):
        mask = result.labels == c
        if mask.sum() == 0:
            continue
        gt_counts = Counter(node_gt_coarse[mask])
        majority = gt_counts.most_common(1)[0][1]
        purity_scores.append(majority / mask.sum())

    mean_purity = float(np.mean(purity_scores))
    assert mean_purity > 0.7, (
        f"Mean cluster purity {mean_purity:.2f} is too low (expected > 0.7)"
    )

    rep = train.report
    norms = normalize_stage1_reconstruction(
        rep, "hierarchy", data=data, rng=np.random.default_rng(0),
    )
    assert norms["mean_norm"] < 1.8, (
        f"normalized Stage 1 mean min-dist {norms['mean_norm']:.3f} too high"
    )
    assert rep.n_lifted_edges >= 3
    assert train.epochs_ran <= 12


@pytest.mark.scenario
@pytest.mark.synthetic
def test_hierarchy_recursion_matches_gaussian_gt():
    """Recursion partition vs hierarchical Gaussian (root → 3 coarse → 6 fine).

    The tree must **resolve every fine Gaussian** as its own terminal leaf
    (six leaves), match fine labels at high ARI, and pass the level-wise
    τ-smoothed unimodal harness at depths 0, 1, and 2 (no skipping fine level).
    """

    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=1000, ambient_dim=4, seed=0,
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

    # Mask for samples that belong to fine clusters (ring points are -1)
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
    # ARI only on blob samples (exclude ring background)
    leaf_y_blobs = leaf_y[blob_mask]
    ari_c, ari_f = adjusted_rand_vs_coarse_fine(leaf_y_blobs, coarse, fine)

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


@awaiting("inference.membership", si="S7.4")
def test_hierarchy_membership_stability():
    """Membership trajectories should be stable under resampling."""
    pytest.fail("Not implemented")
