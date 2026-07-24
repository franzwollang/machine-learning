"""Persistent homology metrics for Proteus evaluation.

Uses gudhi for Vietoris-Rips complex computation.  Falls back to a
stub that raises ImportError if gudhi is not installed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def compute_persistence_diagrams(
    points: np.ndarray,
    max_dim: int = 2,
    max_edge_length: float = np.inf,
) -> list[np.ndarray]:
    """Compute persistence diagrams up to ``max_dim``.

    Returns a list of (n_features, 2) arrays, one per homology dimension.
    """
    try:
        import gudhi
    except ImportError as e:
        raise ImportError(
            "gudhi is required for persistent homology metrics. "
            "Install with: pip install gudhi"
        ) from e

    rips = gudhi.RipsComplex(points=points, max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=max_dim + 1)
    st.compute_persistence()

    diagrams: list[np.ndarray] = []
    for dim in range(max_dim + 1):
        pairs = st.persistence_intervals_in_dimension(dim)
        if len(pairs) == 0:
            diagrams.append(np.empty((0, 2)))
        else:
            diagrams.append(np.array(pairs))
    return diagrams


def bottleneck_distance(
    dgm1: np.ndarray, dgm2: np.ndarray,
) -> float:
    """Bottleneck distance between two persistence diagrams."""
    try:
        import gudhi.bottleneck
    except ImportError as e:
        raise ImportError("gudhi is required for bottleneck distance.") from e

    if dgm1.size == 0 and dgm2.size == 0:
        return 0.0
    if dgm1.size == 0:
        dgm1 = np.empty((0, 2))
    if dgm2.size == 0:
        dgm2 = np.empty((0, 2))
    return float(gudhi.bottleneck.bottleneck_distance(dgm1, dgm2))


def wasserstein_distance(
    dgm1: np.ndarray,
    dgm2: np.ndarray,
    order: float = 2.0,
) -> float:
    """Wasserstein distance between two persistence diagrams."""
    try:
        import gudhi.wasserstein
    except ImportError as e:
        raise ImportError("gudhi is required for Wasserstein distance.") from e

    if dgm1.size == 0 and dgm2.size == 0:
        return 0.0
    if dgm1.size == 0:
        dgm1 = np.empty((0, 2))
    if dgm2.size == 0:
        dgm2 = np.empty((0, 2))
    return float(gudhi.wasserstein.wasserstein_distance(dgm1, dgm2, order=order))


def betti_numbers(
    points: np.ndarray,
    threshold: float,
    max_dim: int = 2,
) -> tuple[int, ...]:
    """Compute Betti numbers at a fixed filtration threshold."""
    diagrams = compute_persistence_diagrams(points, max_dim=max_dim,
                                            max_edge_length=threshold * 1.5)
    result: list[int] = []
    for dim in range(max_dim + 1):
        dgm = diagrams[dim]
        if dgm.size == 0:
            result.append(0)
            continue
        alive = ((dgm[:, 0] <= threshold) & (dgm[:, 1] > threshold))
        result.append(int(alive.sum()))
    return tuple(result)
