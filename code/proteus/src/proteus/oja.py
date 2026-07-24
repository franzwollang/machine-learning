"""Oja-direction helper for local residual geometry."""

from __future__ import annotations

import numpy as np


def update_oja(u: np.ndarray, e: np.ndarray, eta: float) -> np.ndarray:
    """Apply one rank-1 Oja update and return a unit vector.

    The update is ``u <- u + eta * (e * (e . u) - ||e||^2 * u)``.
    If the resulting vector degenerates, a canonical axis is returned.
    """

    u_arr = np.asarray(u, dtype=float)
    e_arr = np.asarray(e, dtype=float)
    if u_arr.shape != e_arr.shape:
        raise ValueError("u and e must have the same shape")
    updated = u_arr + float(eta) * (
        e_arr * float(np.dot(e_arr, u_arr))
        - float(np.dot(e_arr, e_arr)) * u_arr
    )
    norm = float(np.linalg.norm(updated))
    if norm == 0.0 or not np.isfinite(norm):
        fallback = np.zeros_like(u_arr)
        fallback[0] = 1.0
        return fallback
    return updated / norm
