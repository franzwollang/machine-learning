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
  intended to replace the single-scale cleanup stand-ins of S2.6.1 (issue #27).

Characteristic-scale *resolution* defaults to the SI S2.5.1
``load_crossover`` selector (``ScaleSearchConfig.selector="load_crossover"``).
``selector="persistence"`` is the structural / recursion-timing path: it uses
this module's interval as the accept/reject arbiter and, by default, lands
``tau*`` at the coarse end of that interval.  Optional hybrid / experimental refinement
(:attr:`PersistenceConfig.resolve_within_interval`, default ``"none"``) can
re-pick ``tau*`` via ``load_crossover`` or experimental ``mid_interval``
*within* the accepted persistent subgrid without changing the accept/reject
arbiter (OPEN_ISSUES #28).  The legacy ``load_band`` scale selector is gone
from the acceptance path; the controller keeps a deprecated alias that
warns and redirects to ``load_crossover``.

.. note::
   The acceptance rule is **coarse-anchored** by default
   (:attr:`PersistenceConfig.coarse_anchored`): the coarsest multi-cluster
   partition must itself persist, which removes the warm-start fine-end false
   positive that the bare two-point rule admitted on uniform manifolds (SI
   S2.6.2).  With the stand-ins removed and ``require_persistent_split`` on, this
   reduces the circle to a single leaf and keeps the hierarchical Gaussian at six
   leaves.  It is **not yet** a full replacement for the S2.6.1 stand-ins: a
   developable manifold (swiss roll) whose coarsest partition is a few arcs can
   still produce a coarse-anchored run whose adjacent overlap sits just above
   ``overlap_threshold``, so the stand-ins remain load-bearing for that residual.
   The principled resolution is the Stage 2 DM evidence gate (S3.4, M4); see the
   "residual limitation" note in SI S2.6.2 and OPEN_ISSUES #27.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional

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
    coarse_anchored:
        When ``True`` (canonical, SI S2.6.2) the characteristic split must be
        **coarse-anchored**: the region's *coarsest* multi-cluster partition must
        itself persist for ``min_persistence`` grid points.  A persistent block
        that only appears at finer scales --- after coarser scales showed
        different, non-agreeing multi-cluster partitions --- is rejected as a
        resolution artifact rather than an intrinsic feature.  This is the
        S2.6.2 hardening that removes the warm-start fine-end false positive
        (OPEN_ISSUES #27): on a warm-started sweep a uniform manifold's
        arc-partition can chance-coincide over an isolated two-point block at the
        finest scales, spuriously satisfying the bare persistence rule; requiring
        the block to reach back to the emergence scale (where the region first
        ceases to be one cluster) removes it.  When ``False`` the legacy rule is
        used: the coarsest grid index whose block persists, scanning past any
        non-persistent coarser multi-cluster points.  Operational (S14.3).
    cold_start_recheck:
        Experimental **path-independence recheck** of a coarse-anchored
        candidate, **default off and empirically refuted as an acceptance gate**
        (SI S2.6.2 residual note, OPEN_ISSUES #27).  When ``True`` the grid
        points of the candidate persistence interval are re-fit from
        *cold-started* scaffolds (each seeded fresh at its own ``tau`` on an
        independent RNG stream, rather than warm-started from the coarser
        scale), and the candidate is kept only if the interval still persists
        under those independent fits.  The intent (GPT cross-family audit,
        turn 10) was to catch the residual warm-start artifact that
        coarse-anchoring alone does not: on a developable manifold the warm
        sweep can carry a coarse arc-partition forward so it *marginally*
        persists.  In practice the recheck **over-rejects genuine multi-level
        features**: independently cold-fitted scaffolds at adjacent coarse
        scales settle at different *resolution levels* (e.g. the hierarchical
        Gaussian's coarse anchor warm-fits to a stable 3-way partition but
        cold-fits to 6-way vs 3-way at adjacent scales, matched overlap
        ``~0.27 < overlap_threshold``), so the interval "fails" the recheck and
        the true root split is rejected.  The partition-overlap statistic cannot
        separate this resolution-level variance from a true absence of
        structure, which is exactly the job the Stage 2 DM evidence gate (S3.4,
        M4) does with a proper model-comparison margin.  The flag is retained
        (default ``False``) as a reproducible diagnostic and to document the
        negative result; it must not be enabled on the acceptance path.  The
        recheck runs in the controller
        (:func:`proteus.stage1.controller.run_scale_search`), not in
        :func:`compute_persistence`; it is ignored when ``coarse_anchored`` is
        ``False``.  Operational (S14.3).
    resolve_within_interval:
        Optional **hybrid / experimental resolution** when
        ``ScaleSearchConfig.selector="persistence"`` (OPEN_ISSUES #28).
        ``"none"`` (default) keeps today's behavior: ``tau*`` is the coarsest
        persistent multi-cluster grid index from :func:`compute_persistence`.
        ``"load_crossover"`` keeps persistence as the accept/reject arbiter but
        re-picks ``tau*`` by running the SI S2.5.1 load-crossover rule on the
        accepted persistent *subgrid* only (indices ``[i_lo, i_hi]`` of the
        coarse-anchored block).  ``"mid_interval"`` is an **experimental**
        probe that lands ``tau*`` at the integer midpoint of that same block
        (for coarse-vs-mid comparisons; not SI-justified).  Applied in the
        controller, not in :func:`compute_persistence` (the
        ``PersistenceResult.tau_star*`` fields still report the coarse-end
        arbiter index).  Default off; do not flip until a SI-justified
        within-interval signal exists.  Operational (S14.3).
    """

    min_persistence: int = 2
    overlap_threshold: float = 0.5
    min_clusters: int = 2
    coarse_anchored: bool = True
    cold_start_recheck: bool = False
    resolve_within_interval: Literal["none", "load_crossover", "mid_interval"] = "none"


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
        Selected characteristic-scale grid index (and its ``tau``), or ``None``
        if no partition qualifies (a terminal / single-feature region).  Under
        the canonical ``coarse_anchored=True`` rule this is the *coarsest
        multi-cluster* grid index, accepted only if that partition itself
        persists for ``min_persistence`` grid points --- so it can be ``None``
        even when a finer, isolated persistent block exists (that block is judged
        a warm-start artifact).  Under the legacy rule it is the coarsest grid
        index whose block persists, scanning past non-persistent coarser
        multi-cluster points.
    cold_start_rejected:
        ``True`` when a coarse-anchored candidate existed on the warm sweep but
        was rejected by the cold-start path-independence recheck
        (:attr:`PersistenceConfig.cold_start_recheck`); in that case
        ``tau_star_index`` / ``tau_star`` are ``None``.  ``False`` otherwise
        (no candidate, recheck disabled, or candidate survived the recheck).
    """

    run_lengths: np.ndarray
    match_overlaps: np.ndarray
    tau_star_index: Optional[int]
    tau_star: Optional[float]
    snapshots: list[PartitionSnapshot] = field(default_factory=list)
    cold_start_rejected: bool = False


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
    if config.coarse_anchored:
        # Coarse-anchored rule (SI S2.6.2): only the coarsest multi-cluster
        # partition may anchor the characteristic scale.  If it does not itself
        # persist, the region has no coarse-anchored feature and is terminal ---
        # any persistent block appearing further toward the fine end (after the
        # partition has already churned through incompatible multi-cluster
        # states) is treated as a warm-start artifact, not an intrinsic feature.
        for i in range(n):
            if snapshots[i].n_clusters >= config.min_clusters:
                if run_lengths[i] >= config.min_persistence:
                    tau_star_index = i
                break
    else:
        # Legacy rule: coarsest grid index whose block persists, scanning past
        # any non-persistent (transient) coarser multi-cluster points.
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


def interval_is_persistent(
    snapshots: list[PartitionSnapshot],
    config: PersistenceConfig | None = None,
) -> bool:
    """Whether coarse-first ``snapshots`` form one persistent multi-cluster block.

    Returns ``True`` when the maximal same-partition multi-cluster run starting
    at the *coarsest* supplied snapshot spans at least ``min_persistence`` grid
    points.  Used by the controller's cold-start recheck
    (:attr:`PersistenceConfig.cold_start_recheck`) to test whether a candidate
    persistence interval, re-fit path-independently, still persists.  This is a
    pure function of the supplied snapshots (the ``coarse_anchored`` /
    ``cold_start_recheck`` policy flags do not affect the run-length signal).
    """

    config = config if config is not None else PersistenceConfig()
    if len(snapshots) < config.min_persistence:
        return False
    result = compute_persistence(
        snapshots,
        replace(config, coarse_anchored=False, cold_start_recheck=False),
    )
    return bool(result.run_lengths[0] >= config.min_persistence)
