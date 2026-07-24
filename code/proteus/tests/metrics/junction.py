"""Junction score computation for Proteus evaluation (SI S8.4).

Computes the five-indicator J_i score for each vertex star.  This metric
module does not depend on any Proteus implementation; it consumes raw
incidence and diagnostic arrays.
"""
from __future__ import annotations

import numpy as np


def junction_score(
    incidence_counts: np.ndarray,
    neighbor_dims: np.ndarray,
    directional_asymmetry: np.ndarray,
    split_reject_rates: np.ndarray,
    local_torsion_q90: np.ndarray,
    theta_dip: float = 0.05,
    theta_A: float = 0.5,
    theta_rej: float = 0.8,
) -> np.ndarray:
    """Compute the five-indicator junction score J_i for each vertex.

    Each input is a 1D array of length N (one value per vertex).
    Returns an integer array of shape (N,) with J_i in [0, 5].
    """
    n = incidence_counts.shape[0]
    J = np.zeros(n, dtype=int)
    J += (incidence_counts > theta_dip).astype(int)
    J += (neighbor_dims >= 1).astype(int)
    J += (directional_asymmetry > theta_A).astype(int)
    J += (split_reject_rates > theta_rej).astype(int)
    J += (local_torsion_q90 >= 0.30).astype(int)
    return J


def is_junction_frozen(
    J_history: np.ndarray,
    threshold: int = 3,
    consecutive_windows: int = 2,
) -> bool:
    """Whether a vertex should be junction-frozen.

    ``J_history`` is a 1D array of J_i values over consecutive gate windows.
    Returns True if J_i >= threshold for the last ``consecutive_windows``.
    """
    if J_history.shape[0] < consecutive_windows:
        return False
    return bool(np.all(J_history[-consecutive_windows:] >= threshold))
