"""Mass and flux conservation diagnostics for Proteus evaluation (SI S6.2)."""
from __future__ import annotations

import numpy as np


def epsilon_mass(simplex_masses: np.ndarray) -> float:
    """Post-normalization mass deviation |sum(m_S) - 1|."""
    return float(abs(np.sum(simplex_masses) - 1.0))


def epsilon_flux(
    pressure_vectors: list[np.ndarray],
    divergence_stencils: list[np.ndarray],
    face_pressures_global: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Global flux conservation residual.

    Parameters
    ----------
    pressure_vectors:
        Per-simplex pressure vectors p_S, each of shape (d+1,).
    divergence_stencils:
        Per-simplex A_S matrices, each of shape (d, d+1).
    face_pressures_global:
        All face pressures in a flat array for normalization.
    """
    numerator = 0.0
    for p_S, A_S in zip(pressure_vectors, divergence_stencils):
        flux = A_S @ p_S
        numerator += float(np.dot(flux, flux))
    denominator = float(np.dot(face_pressures_global, face_pressures_global)) + eps
    return numerator / denominator
