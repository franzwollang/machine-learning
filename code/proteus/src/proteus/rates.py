"""Rate helpers for Stage 1 dynamics.

These functions implement the operational formulas from SI S2.3.
"""

from __future__ import annotations

import math


def eta_gng(sigma_sq: float, tau: float, k: int) -> float:
    """Effective radial drift rate (SI S2.3, analysis quantity only).

    Returns ``(ln 2 / (2k)) * max(0, 1 - sigma_sq / tau)``.

    This is *not* part of the operational motion law: Stage 1 moves nodes solely
    through the deferred nudge (``eta_cent`` committed at ``delta_min``). ``eta_gng``
    is retained as the effective drift rate on cap-satisfied intervals for the S12
    convergence analysis and is intentionally not called by the scaffold loop.
    """

    _validate_tau(tau)
    _validate_k(k)
    ratio = 1.0 - float(sigma_sq) / float(tau)
    return float(math.log(2.0) / (2.0 * int(k)) * max(0.0, ratio))


def eta_cent(kappa: float, r: float, k: int) -> float:
    """Centering rate ``kappa * (1-r) / k``."""

    _validate_kappa(kappa)
    _validate_grid_ratio(r)
    _validate_k(k)
    return float(kappa) * (1.0 - float(r)) / int(k)


def delta_min(kappa: float, r: float, tau: float) -> float:
    """Deferred-nudge threshold ``kappa * (1-r) * sqrt(tau)``."""

    _validate_kappa(kappa)
    _validate_grid_ratio(r)
    _validate_tau(tau)
    return float(kappa) * (1.0 - float(r)) * math.sqrt(float(tau))


def _validate_tau(tau: float) -> None:
    if float(tau) <= 0.0:
        raise ValueError("tau must be positive")


def _validate_k(k: int) -> None:
    if int(k) < 1:
        raise ValueError("k must be >= 1")


def _validate_grid_ratio(r: float) -> None:
    if not 0.0 < float(r) < 1.0:
        raise ValueError("grid ratio r must satisfy 0 < r < 1")


def _validate_kappa(kappa: float) -> None:
    if float(kappa) <= 0.0:
        raise ValueError("kappa must be positive")
