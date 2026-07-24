"""Unit tests for the Q-partition persistence signal (SI S2.6.2)."""

from __future__ import annotations

import numpy as np

from proteus.stage1.persistence import (
    PartitionSnapshot,
    PersistenceConfig,
    compute_persistence,
    mean_matched_jaccard,
)


def _snap(idx: int, tau: float, labels: list[int], q: float = 1.0) -> PartitionSnapshot:
    arr = np.asarray(labels, dtype=int)
    n_clusters = int(np.unique(arr[arr >= 0]).size)
    return PartitionSnapshot(
        grid_index=idx,
        tau=tau,
        labels=arr,
        n_clusters=n_clusters,
        partition_q_score=q,
    )


def test_mean_matched_jaccard_identical_is_one() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert mean_matched_jaccard(labels, labels) == 1.0


def test_mean_matched_jaccard_relabel_invariant() -> None:
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([2, 2, 0, 0, 1, 1])  # same partition, permuted labels
    assert mean_matched_jaccard(a, b) == 1.0


def test_mean_matched_jaccard_penalizes_cluster_count_mismatch() -> None:
    a = np.array([0, 0, 1, 1, 2, 2])       # 3 clusters
    b = np.array([0, 0, 0, 0, 0, 0])       # 1 cluster
    # Matched pair Jaccard 2/6, averaged over max(3, 1) = 3 slots.
    assert mean_matched_jaccard(a, b) < 0.2


def test_single_cluster_grid_has_no_persistence() -> None:
    snaps = [_snap(i, 1.0 / (i + 1), [0, 0, 0, 0], q=0.0) for i in range(5)]
    res = compute_persistence(snaps, PersistenceConfig())
    assert res.tau_star_index is None
    assert res.tau_star is None
    assert np.all(res.run_lengths == 0)


def test_persistent_multicluster_partition_is_detected() -> None:
    # Coarsest point is single-cluster; then the same 3-way split persists for
    # three adjacent grid points before fragmenting at the finest scale.
    three_way = [0, 0, 1, 1, 2, 2]
    snaps = [
        _snap(0, 1.00, [0, 0, 0, 0, 0, 0], q=0.0),
        _snap(1, 0.70, three_way),
        _snap(2, 0.50, three_way),
        _snap(3, 0.35, three_way),
        _snap(4, 0.25, [0, 1, 2, 3, 4, 5]),  # over-refined
    ]
    res = compute_persistence(snaps, PersistenceConfig(min_persistence=2))
    # Coarsest persistent multi-cluster split starts at index 1 (tau = 0.70).
    assert res.tau_star_index == 1
    assert res.tau_star == 0.70
    assert res.run_lengths[1] >= 3


def test_transient_split_below_min_persistence_is_rejected() -> None:
    # A single isolated multi-cluster grid point surrounded by single clusters
    # is transient and must not be accepted.
    snaps = [
        _snap(0, 1.00, [0, 0, 0, 0, 0, 0], q=0.0),
        _snap(1, 0.70, [0, 0, 1, 1, 2, 2]),
        _snap(2, 0.50, [0, 0, 0, 0, 0, 0], q=0.0),
    ]
    res = compute_persistence(snaps, PersistenceConfig(min_persistence=2))
    assert res.tau_star_index is None


def test_min_persistence_threshold_is_respected() -> None:
    # Two adjacent identical partitions: accepted at min_persistence=2, rejected
    # at min_persistence=3.
    split = [0, 0, 1, 1]
    snaps = [_snap(0, 0.7, split), _snap(1, 0.5, split)]
    assert compute_persistence(snaps, PersistenceConfig(min_persistence=2)).tau_star_index == 0
    assert compute_persistence(snaps, PersistenceConfig(min_persistence=3)).tau_star_index is None
