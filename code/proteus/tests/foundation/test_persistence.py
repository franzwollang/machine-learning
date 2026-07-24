"""Unit tests for the Q-partition persistence signal (SI S2.6.2)."""

from __future__ import annotations

import numpy as np

from proteus.stage1.persistence import (
    PartitionSnapshot,
    PersistenceConfig,
    compute_persistence,
    interval_is_persistent,
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


def _fine_end_false_positive_snaps() -> list[PartitionSnapshot]:
    """Coarse scales churn through incompatible multi-cluster partitions; only an
    isolated two-point block survives at the fine end.

    This mirrors the warm-start artifact SI S2.6.2 warns about: a uniform
    manifold whose arc-partition never stabilizes coarsely, but whose two finest
    grid points chance-coincide (OPEN_ISSUES #27).  ``two_a``/``six`` alternate so
    no adjacent coarse pair agrees; ``three`` repeats only at indices 4--5.
    """

    two_a = [0, 0, 0, 1, 1, 1]
    six = [0, 1, 2, 3, 4, 5]
    three = [0, 0, 1, 1, 2, 2]
    return [
        _snap(0, 0.500, two_a),
        _snap(1, 0.250, six),
        _snap(2, 0.125, two_a),
        _snap(3, 0.062, six),
        _snap(4, 0.031, three),
        _snap(5, 0.016, three),
    ]


def test_coarse_anchored_rejects_isolated_fine_end_block() -> None:
    # Coarse-anchored (canonical): the coarsest multi-cluster partition (index 0)
    # does not itself persist, so the region has no coarse-anchored feature --- the
    # fine-end coincidence is not accepted (SI S2.6.2 hardening, #27).
    snaps = _fine_end_false_positive_snaps()
    res = compute_persistence(snaps, PersistenceConfig(min_persistence=2))
    assert res.tau_star_index is None
    # The isolated fine-end block still shows up in the raw run-length signal.
    assert res.run_lengths[4] == 2


def test_legacy_rule_accepts_isolated_fine_end_block() -> None:
    # Legacy rule (coarse_anchored=False) selects the first persistent block even
    # when it is an isolated fine-end coincidence --- the behavior the hardening
    # fixes.  Kept to document the difference.
    snaps = _fine_end_false_positive_snaps()
    res = compute_persistence(
        snaps, PersistenceConfig(min_persistence=2, coarse_anchored=False),
    )
    assert res.tau_star_index == 4


# ---------------------------------------------------------------------------
# interval_is_persistent --- the pure test used by the controller's (default
# off) cold-start path-independence recheck (SI S2.6.2, OPEN_ISSUES #27).
# ---------------------------------------------------------------------------


def test_interval_is_persistent_accepts_agreeing_multicluster_block() -> None:
    three = [0, 0, 1, 1, 2, 2]
    snaps = [_snap(0, 0.5, three), _snap(1, 0.35, three)]
    assert interval_is_persistent(snaps, PersistenceConfig(min_persistence=2))


def test_interval_is_persistent_rejects_disagreeing_block() -> None:
    # Two adjacent multi-cluster partitions that disagree (different cluster
    # counts / assignments) do not form a persistent interval --- this is the
    # signal that trips on cold-started resolution-level variance (the reason
    # the recheck over-rejects genuine features and is left off by default).
    six = [0, 1, 2, 3, 4, 5]
    three = [0, 0, 1, 1, 2, 2]
    snaps = [_snap(0, 0.5, six), _snap(1, 0.35, three)]
    assert not interval_is_persistent(snaps, PersistenceConfig(min_persistence=2))


def test_interval_is_persistent_rejects_short_interval() -> None:
    three = [0, 0, 1, 1, 2, 2]
    snaps = [_snap(0, 0.5, three)]
    assert not interval_is_persistent(snaps, PersistenceConfig(min_persistence=2))
