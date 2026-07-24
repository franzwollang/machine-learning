"""Deferred-nudge helpers for Stage 1 node motion."""

from __future__ import annotations

import numpy as np

from proteus.moments import incoherence_ratio
from proteus.types import NodeState


def accumulate_nudge(
    node: NodeState,
    m_step: np.ndarray,
    eta_cent_value: float,
    eps: float = 1e-8,
    rho_max: float = 10.0,
) -> None:
    """Accumulate a deferred nudge on ``node``.

    The scale factor is ``eta_cent * rho * step`` where
    ``rho = ||m_i|| / (sigma_i + eps)`` and ``sigma_i = sqrt(variance)``.

    Gated by ``node.update_count > 0`` so that nodes whose hard-Voronoi
    variance has not yet been seeded (``sigma_i`` still at the EWMA prior)
    do not accumulate nudges from soft-routing residual means with a
    near-zero denominator.  ``rho`` is also clamped at ``rho_max``
    (default 10.0) as a defensive bound against pathological inputs;
    in healthy operation rho stays well below this cap.
    """

    step = np.asarray(m_step, dtype=float)
    if step.shape != node.nudge.shape:
        raise ValueError("m_step must have the same shape as node.nudge")
    if node.update_count <= 0:
        return
    sigma = float(np.sqrt(max(node.variance, 0.0)))
    rho = incoherence_ratio(node.residual_mean, sigma=sigma, eps=eps)
    rho = min(rho, float(rho_max))
    node.nudge = node.nudge + float(eta_cent_value) * rho * step


def apply_if_threshold(node: NodeState, delta_min_value: float) -> bool:
    """Apply and reset ``node.nudge`` if it exceeds ``delta_min_value``."""

    if float(delta_min_value) <= 0.0:
        raise ValueError("delta_min_value must be positive")
    if float(np.linalg.norm(node.nudge)) < float(delta_min_value):
        return False
    node.position = node.position + node.nudge
    node.nudge = np.zeros_like(node.nudge)
    return True
