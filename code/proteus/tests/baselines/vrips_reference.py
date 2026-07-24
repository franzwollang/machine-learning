"""Vietoris-Rips topology baseline."""
from __future__ import annotations

import numpy as np


def vrips_betti_numbers(
    points: np.ndarray,
    threshold: float,
    max_dim: int = 2,
) -> tuple[int, ...]:
    """Compute Betti numbers at a fixed filtration value using gudhi."""
    from tests.metrics.persistent_homology import betti_numbers
    return betti_numbers(points, threshold=threshold, max_dim=max_dim)
