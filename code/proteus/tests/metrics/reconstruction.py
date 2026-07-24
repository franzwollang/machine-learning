"""Reconstruction and projection error metrics for Proteus evaluation."""
from __future__ import annotations

import numpy as np


def mean_min_distance(
    data: np.ndarray, representatives: np.ndarray,
) -> float:
    """Average distance from each data point to its nearest representative."""
    if representatives.shape[0] == 0:
        return float("inf")
    sq = (
        np.sum(data ** 2, axis=1, keepdims=True)
        + np.sum(representatives ** 2, axis=1, keepdims=True).T
        - 2.0 * data @ representatives.T
    )
    return float(np.sqrt(np.maximum(sq, 0.0)).min(axis=1).mean())


def max_min_distance(
    data: np.ndarray, representatives: np.ndarray,
) -> float:
    """Hausdorff-style: max over data of distance to nearest representative."""
    if representatives.shape[0] == 0:
        return float("inf")
    sq = (
        np.sum(data ** 2, axis=1, keepdims=True)
        + np.sum(representatives ** 2, axis=1, keepdims=True).T
        - 2.0 * data @ representatives.T
    )
    return float(np.sqrt(np.maximum(sq, 0.0)).min(axis=1).max())


def projection_error(
    data: np.ndarray,
    projections: np.ndarray,
) -> float:
    """Mean L2 distance between data points and their projections."""
    return float(np.linalg.norm(data - projections, axis=1).mean())
