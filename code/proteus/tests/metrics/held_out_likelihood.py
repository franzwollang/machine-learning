"""Held-out log-likelihood metric for Proteus evaluation.

This module provides a generic evaluator that consumes any callable
density function, enabling comparison across Proteus and baselines.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def held_out_log_likelihood(
    test_points: np.ndarray,
    log_density_fn: Callable[[np.ndarray], np.ndarray],
) -> float:
    """Mean held-out log-density over test points.

    Parameters
    ----------
    test_points:
        (N, D) array of held-out samples.
    log_density_fn:
        Callable that takes (N, D) and returns (N,) log-densities.
    """
    log_densities = log_density_fn(test_points)
    return float(np.mean(log_densities))


def per_point_log_likelihood(
    test_points: np.ndarray,
    log_density_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Per-point log-density array for diagnostic inspection."""
    return np.asarray(log_density_fn(test_points), dtype=float)
