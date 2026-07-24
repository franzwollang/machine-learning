"""Q-partition persistence across adjacent scales (SI S2.6.1, S2.6.2).

The single-scale graph-local Q-score is only a *proposal-level* screen: a
uniform manifold sampled coarsely (a ring resolved into a few arcs) and a
genuinely multi-modal region can present indistinguishable single-scale
statistics (SI S2.6.1, "Scope of the single-scale criterion").  The canonical
arbiter is **feature persistence** (SI S2.6.2): an intrinsic multi-cluster
feature satisfies the cluster definition over an *interval* of adjacent scales,
whereas the transient arc-partition of a uniform manifold does not.

This module operationalizes that signal.  Along the coarse-to-fine ``tau`` grid
already swept by :mod:`proteus.stage1.controller`, we record the accepted
partition at each grid point and measure how far a given multi-cluster partition
persists toward finer scales.  Partitions are compared in **sample space**: the
dataset samples carry a stable identity across every grid point (unlike scaffold
node indices, which shift under splits and prunes), so this is a robust
instantiation of the SI's node-ID-overlap tracking that relies on the same
warm-started scaffold.

The signal has two consumers, both acceptance-path (S2.6.2):

* characteristic-scale selection --- the coarsest ``tau`` at which a
  multi-cluster partition first *persists* (issue #28 secondary signal), and
* recursion timing --- a proposed split is accepted only if it persists,
  replacing the provisional single-scale cleanup stand-ins of S2.6.1 (issue #27).

Both are wired incrementally behind a flag while the legacy load-band selector
remains the transition default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class PersistenceConfig:
    """Configuration for the Q-partition persistence signal (SI S2.6.2).

    Attributes
    ----------
    min_persistence:
        Number of consecutive grid points over which a multi-cluster partition
        must survive to count as an intrinsic (rather than transient) feature.
        The SI prescribes "an interval of adjacent scales"; the minimal
        non-trivial interval is two adjacent grid points (derived from the
        persistence definition, not a tuned value).
    overlap_threshold:
        Minimum mean matched-cluster Jaccard overlap for two adjacent
        partitions to be judged "the same partition".  Operational default
        (OPEN_ISSUES #28 / S14.3): the persistence *interval* count, not this
        threshold, does the acceptance work; the threshold only defines partition
        identity and is backstopped by ``min_persistence``.
    min_clusters:
        Smallest cluster count that counts as a multi-cluster partition (the
        one-cluster null is never "persistent" in the sense that matters).
    """

    min_persistence: int = 2
    overlap_threshold: float = 0.5
    min_clusters: int = 2


@dataclass
class PartitionSnapshot:
    """Accepted partition at one grid point, in sample space (SI S2.6.2).

    ``labels`` assigns every dataset sample to a cluster id, so snapshots at
    different grid points are directly comparable regardless of how the
    underlying scaffold's node set changed.
    """

    grid_index: int
    tau: float
    labels: np.ndarray
    n_clusters: int
    partition_q_score: float
    stabilized: bool = True


@dataclass
class PersistenceResult:
    """Result of tracking partition persistence across the tau grid.

    Attributes
    ----------
    run_lengths:
        ``run_lengths[i]`` is the number of grid points in the maximal
        consecutive multi-cluster block starting at grid index ``i`` (toward
        finer scales) over which each adjacent pair is "the same partition".
        Zero where the grid point is not multi-cluster.
    match_overlaps:
        ``match_overlaps[t]`` is the mean matched-cluster Jaccard overlap
        between grid points ``t`` and ``t+1`` (``nan`` when either side is not
        multi-cluster).
    tau_star_index / tau_star:
        Coarsest grid index (and its ``tau``) whose multi-cluster partition
        persists for at least ``min_persistence`` grid points, or ``None`` if no
        partition persists (a terminal / single-feature region).
    """

    run_lengths: np.ndarray
    match_overlaps: np.ndarray
    tau_star_index: Optional[int]
    tau_star: Optional[float]
    snapshots: list[PartitionSnapshot] = field(default_factory=list)


def route_samples_to_labels(
    scaffold: Any,
    data: np.ndarray,
    node_labels: np.ndarray,
) -> np.ndarray:
    """Assign each sample to the cluster of its best-matching-unit node.

    Returns an ``(n_samples,)`` integer array of cluster labels.  Samples whose
    BMU query fails (empty scaffold) are labeled ``-1``.
    """

    data_arr = np.asarray(data, dtype=float)
    n_samples = data_arr.shape[0]
    labels = np.full(n_samples, -1, dtype=int)
    node_labels = np.asarray(node_labels, dtype=int)
    if node_labels.size == 0:
        return labels
    for i in range(n_samples):
        bmu_ids, _ = scaffold.ann.query_knn(data_arr[i], k=1)
        if len(bmu_ids) == 0:
            continue
        bmu = int(bmu_ids[0])
        if 0 <= bmu < node_labels.size:
            labels[i] = int(node_labels[bmu])
    return labels


def _contingency(labels_a: np.ndarray, labels_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the cluster contingency matrix over shared (non-``-1``) samples.

    Rows index unique labels of ``a``, columns index unique labels of ``b``;
    entries count samples assigned to both.
    """

    labels_a = np.asarray(labels_a, dtype=int)
    labels_b = np.asarray(labels_b, dtype=int)
    valid = (labels_a >= 0) & (labels_b >= 0)
    a = labels_a[valid]
    b = labels_b[valid]
    ua = np.unique(a)
    ub = np.unique(b)
    a_index = {lab: r for r, lab in enumerate(ua)}
    b_index = {lab: c for c, lab in enumerate(ub)}
    table = np.zeros((ua.size, ub.size), dtype=float)
    for la, lb in zip(a, b):
        table[a_index[int(la)], b_index[int(lb)]] += 1.0
    return table, ua, ub


def mean_matched_jaccard(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Mean Jaccard overlap of optimally matched clusters between two partitions.

    Clusters are matched by maximum-intersection assignment (Hungarian on the
    contingency matrix).  The mean is taken over ``max(K_a, K_b)`` slots, so an
    unmatched cluster on either side contributes zero --- differing cluster
    counts are penalized, which is what "the same partition" should require.
    Returns ``0.0`` when either partition is empty.
    """

    table, _, _ = _contingency(labels_a, labels_b)
    if table.size == 0:
        return 0.0
    k_a, k_b = table.shape
    row_tot = table.sum(axis=1)
    col_tot = table.sum(axis=0)
    # Maximize total intersection -> minimize negative intersection.
    rows, cols = linear_sum_assignment(-table)
    jaccards: list[float] = []
    for r, c in zip(rows, cols):
        inter = table[r, c]
        union = row_tot[r] + col_tot[c] - inter
        jaccards.append(inter / union if union > 0.0 else 0.0)
    denom = max(k_a, k_b)
    if denom == 0:
        return 0.0
    return float(np.sum(jaccards) / denom)


def _partitions_agree(
    a: PartitionSnapshot,
    b: PartitionSnapshot,
    config: PersistenceConfig,
) -> tuple[bool, float]:
    """Whether two adjacent snapshots represent the same multi-cluster partition."""

    if a.n_clusters < config.min_clusters or b.n_clusters < config.min_clusters:
        return False, float("nan")
    overlap = mean_matched_jaccard(a.labels, b.labels)
    return overlap >= config.overlap_threshold, overlap


def compute_persistence(
    snapshots: list[PartitionSnapshot],
    config: PersistenceConfig | None = None,
) -> PersistenceResult:
    """Measure multi-cluster partition persistence across a coarse-to-fine grid.

    ``snapshots`` must be ordered as the controller sweeps ``tau`` --- coarsest
    first.  See :class:`PersistenceResult` for the returned signal.
    """

    config = config if config is not None else PersistenceConfig()
    n = len(snapshots)
    run_lengths = np.zeros(n, dtype=int)
    match_overlaps = np.full(max(n - 1, 0), np.nan, dtype=float)

    agree = np.zeros(max(n - 1, 0), dtype=bool)
    for t in range(n - 1):
        ok, overlap = _partitions_agree(snapshots[t], snapshots[t + 1], config)
        agree[t] = ok
        match_overlaps[t] = overlap

    for i in range(n):
        if snapshots[i].n_clusters < config.min_clusters:
            run_lengths[i] = 0
            continue
        length = 1
        t = i
        while (
            t < n - 1
            and agree[t]
            and snapshots[t + 1].n_clusters >= config.min_clusters
        ):
            length += 1
            t += 1
        run_lengths[i] = length

    tau_star_index: Optional[int] = None
    for i in range(n):
        if run_lengths[i] >= config.min_persistence:
            tau_star_index = i
            break

    tau_star = snapshots[tau_star_index].tau if tau_star_index is not None else None

    return PersistenceResult(
        run_lengths=run_lengths,
        match_overlaps=match_overlaps,
        tau_star_index=tau_star_index,
        tau_star=tau_star,
        snapshots=list(snapshots),
    )
