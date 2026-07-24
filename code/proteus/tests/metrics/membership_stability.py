"""Membership trajectory stability metrics for Proteus evaluation."""
from __future__ import annotations

import numpy as np


def trajectory_rank_correlation(
    trajectories_a: np.ndarray,
    trajectories_b: np.ndarray,
) -> float:
    """Mean Spearman rank correlation of membership trajectories.

    Each input is (N, L) where L is the number of hierarchy levels.
    Compares trajectories from two independent runs or resamples.
    """
    from scipy.stats import spearmanr

    n, L = trajectories_a.shape
    correlations = np.zeros(L)
    for level in range(L):
        rho, _ = spearmanr(trajectories_a[:, level], trajectories_b[:, level])
        correlations[level] = rho if not np.isnan(rho) else 0.0
    return float(np.mean(correlations))


def trajectory_l2_stability(
    trajectories_a: np.ndarray,
    trajectories_b: np.ndarray,
) -> float:
    """Mean L2 distance between membership trajectories across runs."""
    return float(np.linalg.norm(trajectories_a - trajectories_b, axis=1).mean())
