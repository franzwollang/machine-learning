"""Moment-update helpers for Proteus.

These are the SI S2.2/S2.3 primitives used by Stage 1 and later tests.
They are intentionally stateless and small so downstream modules can share
one implementation of the EWMA and derived diagnostics.
"""

from __future__ import annotations

import numpy as np


def ewma_update(
    m: np.ndarray,
    s: np.ndarray,
    e: np.ndarray,
    alpha: float,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a rank-weighted EWMA update to first and second moments.

    The update is

    ``m <- (1 - alpha * weight) * m + alpha * weight * e``

    and analogously for ``s`` using the elementwise squared residual
    ``e * e``.
    """

    gain = float(alpha) * float(weight)
    if gain < 0.0:
        raise ValueError("alpha * weight must be non-negative")
    m_arr = np.asarray(m, dtype=float)
    s_arr = np.asarray(s, dtype=float)
    e_arr = np.asarray(e, dtype=float)
    if m_arr.shape != e_arr.shape or s_arr.shape != e_arr.shape:
        raise ValueError("m, s, and e must have the same shape")
    m_new = (1.0 - gain) * m_arr + gain * e_arr
    s_new = (1.0 - gain) * s_arr + gain * (e_arr * e_arr)
    return m_new, s_new


def variance_from_moments(m: np.ndarray, s: np.ndarray) -> float:
    """Return ``tr(s - m * m)`` clipped at zero for numerical stability."""

    m_arr = np.asarray(m, dtype=float)
    s_arr = np.asarray(s, dtype=float)
    if m_arr.shape != s_arr.shape:
        raise ValueError("m and s must have the same shape")
    variance_vec = s_arr - m_arr * m_arr
    return float(np.maximum(variance_vec, 0.0).sum())


def incoherence_ratio(m: np.ndarray, sigma: float, eps: float = 1e-8) -> float:
    """Return ``||m|| / (sigma + eps)``."""

    return float(np.linalg.norm(np.asarray(m, dtype=float)) / (float(sigma) + eps))
