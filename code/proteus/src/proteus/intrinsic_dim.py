"""Minimal intrinsic-dimension fallback estimators.

This pass uses a graph-degree proxy with neighbor-median smoothing.  The
full Levina-Bickel style estimator is intentionally deferred until the scale
controller needs it.
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
