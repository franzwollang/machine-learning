"""Intrinsic-dimension estimators for the Stage-1 scaffold (SI S1.4.1).

Two estimators of the per-node intrinsic dimension ``d_final`` are provided:

* :func:`estimate_d_final` -- the **degree proxy** on the lifted structural
  graph (``max(1, degree - 1)`` with neighbor-median smoothing). It is the
  operational Stage-1 default and is *validated* (SI S1.4.1, OPEN_ISSUES #39):
  on the uniform d-ball reference ensemble (the same ensemble used to calibrate
  ``c_{d,k}`` / ``C_Q(d)``, SI S2.5.5/S3.3) its per-node median matches the true
  dimension for ``d in {1,2,3,4}``. Its documented failure mode is an *upward*
  bias on thin curved manifolds embedded in a higher ambient dimension, where
  off-manifold lifted edges (the high-dimensional Delaunay slivers that S3.1--3.2
  link-pruning targets) inflate node degree.

* :func:`estimate_d_final_mle` -- a **Levina--Bickel maximum-likelihood**
  estimator computed from node/point positions directly. It recovers the true
  dimension on well-sampled (locally Poissonian) manifolds and is the principled
  cross-check for absolute-dimension reporting. Because it assumes Poisson local
  sampling, it *under-reads* on the near-regular lattice of an equilibrated
  coarse scaffold at ``d >= 3``; use it on the dense routed sample set (or as an
  independent validation estimator), not as a coarse-node drop-in.

See SI S1.4.1 for the validation protocol and the estimator-selection guidance
for dimension-sensitive consumers (junction detection S8.4, simplex dimension
S4.2).
"""

from __future__ import annotations

import numpy as np


def estimate_d_final(
    neighbour_graph: dict[int, list[int]],
    *,
    dim_floor: int = 1,
    ambient_dim: int | None = None,
) -> np.ndarray:
    """Estimate smoothed local intrinsic dimension from graph degree.

    Raw values are ``max(dim_floor, degree(i) - 1)``.  The returned value for
    node ``i`` is the rounded median of the raw values over ``{i} ∪ N(i)``.
    """

    if dim_floor < 1:
        raise ValueError("dim_floor must be >= 1")
    if not neighbour_graph:
        return np.empty(0, dtype=int)

    n_nodes = max(neighbour_graph.keys()) + 1
    raw = np.full(n_nodes, int(dim_floor), dtype=float)
    for idx in range(n_nodes):
        degree = len(neighbour_graph.get(idx, []))
        raw[idx] = max(dim_floor, degree - 1)

    smoothed = np.empty(n_nodes, dtype=int)
    for idx in range(n_nodes):
        neighborhood = [idx] + list(neighbour_graph.get(idx, []))
        valid = [j for j in neighborhood if 0 <= j < n_nodes]
        value = int(round(float(np.median(raw[valid]))))
        if ambient_dim is not None:
            value = min(value, int(ambient_dim))
        smoothed[idx] = max(int(dim_floor), value)
    return smoothed


def estimate_d_final_mle(
    positions: np.ndarray,
    *,
    k_min: int = 6,
    k_max: int = 12,
    dim_floor: int = 1,
    ambient_dim: int | None = None,
) -> np.ndarray:
    """Per-node Levina--Bickel MLE intrinsic dimension from positions (SI S1.4.1).

    For each point ``i`` with sorted nearest-neighbor distances
    ``T_1 <= ... <= T_{k}`` (excluding itself), the Levina--Bickel maximum-
    likelihood estimate at neighbor count ``k`` is

        m_k(i) = [ (1/(k-1)) * sum_{j=1}^{k-1} log(T_k(i) / T_j(i)) ]^{-1}.

    Following MacKay--Ghahramani, the per-node estimate averages the *inverses*
    ``1/m_k(i)`` over ``k in [k_min, k_max]`` and inverts the mean; this is the
    numerically stable pooling of the individual scale estimates. The raw
    per-node values are then neighbor-median smoothed over each point's
    ``k_min`` nearest neighbors (mirroring the smoothing of
    :func:`estimate_d_final`) and rounded to an integer.

    Unlike the degree proxy this estimator uses geometry directly, so it is not
    inflated by off-manifold structural edges. It assumes locally Poissonian
    sampling and therefore *under-reads* on the near-regular node lattice of an
    equilibrated coarse scaffold at ``d >= 3``; prefer it on the dense routed
    sample set or as an independent cross-check (SI S1.4.1, OPEN_ISSUES #39).
    """

    if dim_floor < 1:
        raise ValueError("dim_floor must be >= 1")
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2:
        raise ValueError("positions must be a 2-D (n, d) array")
    n = positions.shape[0]
    if n == 0:
        return np.empty(0, dtype=int)
    if n <= 2:
        # Fewer than three points cannot form the T_j / T_k distance ratios the
        # MLE needs; fall back to the dimension floor.
        return np.full(n, int(dim_floor), dtype=int)

    from scipy.spatial import cKDTree

    k_hi = min(int(k_max), n - 1)
    k_lo = min(max(int(k_min), 2), k_hi)
    tree = cKDTree(positions)
    dists, idxs = tree.query(positions, k=k_hi + 1)
    dists = np.atleast_2d(np.asarray(dists, dtype=float))
    idxs = np.atleast_2d(np.asarray(idxs, dtype=int))
    # Column 0 is the point itself (distance 0); drop it.
    neighbour_dists = np.maximum(dists[:, 1:], 1e-12)

    raw = np.full(n, float(dim_floor), dtype=float)
    for i in range(n):
        inverse_terms: list[float] = []
        for kk in range(k_lo, k_hi + 1):
            reference = neighbour_dists[i, kk - 1]
            log_ratios = np.log(reference / neighbour_dists[i, : kk - 1])
            mean_log = float(np.mean(log_ratios))
            if mean_log > 0.0:
                inverse_terms.append(mean_log)  # this equals 1 / m_kk(i)
        if inverse_terms:
            raw[i] = 1.0 / float(np.mean(inverse_terms))
    raw = np.maximum(raw, float(dim_floor))

    smoothed = np.empty(n, dtype=int)
    for i in range(n):
        neighbourhood = idxs[i, : k_lo + 1]
        value = int(round(float(np.median(raw[neighbourhood]))))
        if ambient_dim is not None:
            value = min(value, int(ambient_dim))
        smoothed[i] = max(int(dim_floor), value)
    return smoothed
