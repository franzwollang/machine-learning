"""Routing weights for Stage 1 k-neighbor updates."""

from __future__ import annotations

import numpy as np


def rank_decay_weights(k: int) -> np.ndarray:
    """Return rank-decay weights ``2^{-r}`` for ranks ``r=0..k-1``."""

    if int(k) < 1:
        raise ValueError("k must be >= 1")
    return np.array([2.0 ** (-rank) for rank in range(int(k))], dtype=float)


def gaussian_relative_weights(distances: np.ndarray, tau: float) -> np.ndarray:
    """Return relative Gaussian weights over a k-neighbor distance set.

    The nearest neighbor receives weight 1.0; other neighbors decay by their
    squared-distance excess relative to the nearest neighbor.  This preserves a
    stable BMU update while using metric information when low-dimensional
    distances are reliable.
    """

    distances_arr = np.asarray(distances, dtype=float)
    if distances_arr.ndim != 1:
        raise ValueError("distances must be a 1D array")
    if distances_arr.size == 0:
        return np.empty(0, dtype=float)
    if float(tau) <= 0.0:
        raise ValueError("tau must be positive")
    d2 = distances_arr * distances_arr
    return np.exp(-(d2 - float(np.min(d2))) / (2.0 * float(tau)))


def routing_weights(
    distances: np.ndarray,
    tau: float,
    ambient_dim: int,
    *,
    gaussian_cutoff_dim: int = 8,
) -> np.ndarray:
    """Choose Gaussian or rank-decay routing weights.

    For ambient dimension <= ``gaussian_cutoff_dim`` we use a relative
    Gaussian kernel on the returned k-neighbor distances.  For higher
    dimensions we fall back to the rank-decay approximation.
    """

    distances_arr = np.asarray(distances, dtype=float)
    if int(ambient_dim) <= int(gaussian_cutoff_dim):
        return gaussian_relative_weights(distances_arr, tau)
    return rank_decay_weights(distances_arr.size)
