"""Simplex mass distribution diagnostics for Proteus evaluation (SI S9.2)."""
from __future__ import annotations

import numpy as np


def mass_cv(simplex_masses: np.ndarray) -> float:
    """Coefficient of variation of simplex masses (CV_S)."""
    if simplex_masses.size == 0:
        return float("inf")
    mean = float(np.mean(simplex_masses))
    if mean <= 0:
        return float("inf")
    return float(np.std(simplex_masses) / mean)


def mass_histogram(
    simplex_masses: np.ndarray, n_bins: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of simplex masses for visual diagnostics."""
    counts, edges = np.histogram(simplex_masses, bins=n_bins)
    return counts, edges
