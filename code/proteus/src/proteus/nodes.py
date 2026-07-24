"""Node-state helpers for Proteus Stage 1 (SI S2.3).

The canonical ``NodeState`` type lives in :mod:`proteus.types` (promoted out of
``tests/contracts`` per OPEN_ISSUES #38); this module provides the helpers that
construct and update it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from proteus.moments import ewma_update, variance_from_moments
from proteus.types import NodeState


def make_node(
    position: np.ndarray,
    dim: int,
    *,
    principal_dir: Optional[np.ndarray] = None,
    d_final: int = 1,
) -> NodeState:
    """Create a well-formed ``NodeState`` at ``position``.

    Residual moments and the deferred-nudge accumulator are initialized to
    zero.  The principal direction is normalized; if none is supplied, the
    first coordinate axis is used as a deterministic default.
    """

    position_arr = np.asarray(position, dtype=float)
    if position_arr.shape != (dim,):
        raise ValueError(f"position must have shape ({dim},), got {position_arr.shape}")
    if principal_dir is None:
        principal = np.zeros(dim, dtype=float)
        principal[0] = 1.0
    else:
        principal = np.asarray(principal_dir, dtype=float)
        if principal.shape != (dim,):
            raise ValueError(
                f"principal_dir must have shape ({dim},), got {principal.shape}"
            )
        norm = float(np.linalg.norm(principal))
        if norm == 0.0:
            principal = np.zeros(dim, dtype=float)
            principal[0] = 1.0
        else:
            principal = principal / norm

    zeros = np.zeros(dim, dtype=float)
    return NodeState(
        position=position_arr.copy(),
        residual_mean=zeros.copy(),
        residual_sq=zeros.copy(),
        nudge=zeros.copy(),
        principal_dir=principal.copy(),
        hit_count=0.0,
        variance=0.0,
        d_final=int(d_final),
        update_count=0,
        m_pos=zeros.copy(),
        s_pos=zeros.copy(),
        h_pos=0.0,
        update_count_pos=0,
        m_neg=zeros.copy(),
        s_neg=zeros.copy(),
        h_neg=0.0,
        update_count_neg=0,
    )


def update_node_moments(
    node: NodeState,
    e: np.ndarray,
    alpha: float,
    weight: float,
    *,
    is_bmu: bool = True,
) -> NodeState:
    """Update node moments in place and return ``node``.

    The residual mean ``m_i`` (and shadow means ``m_pos`` / ``m_neg``)
    always update under the soft routing kernel.  The second moment
    ``s_i`` (and shadow ``s_pos`` / ``s_neg``), variance, update counts,
    and shadow hit counts update only when ``is_bmu`` is True, so that
    the variance estimator measures the hard-Voronoi catchment (SI S2.3.1).
    """

    e_arr = np.asarray(e, dtype=float)

    # Residual mean: always (soft routing)
    gain = float(alpha) * float(weight)
    node.residual_mean = (1.0 - gain) * node.residual_mean + gain * e_arr

    if is_bmu:
        # Second moment and derived variance: BMU only (hard Voronoi)
        node.residual_sq = (1.0 - gain) * node.residual_sq + gain * (e_arr * e_arr)
        node.variance = variance_from_moments(node.residual_mean, node.residual_sq)
        node.update_count += 1

    proj = float(np.dot(e_arr, node.principal_dir))
    if proj > 0.0:
        node.m_pos = (1.0 - gain) * node.m_pos + gain * e_arr
        if is_bmu:
            node.s_pos = (1.0 - gain) * node.s_pos + gain * (e_arr * e_arr)
            node.h_pos += float(weight)
            node.update_count_pos += 1
    elif proj < 0.0:
        node.m_neg = (1.0 - gain) * node.m_neg + gain * e_arr
        if is_bmu:
            node.s_neg = (1.0 - gain) * node.s_neg + gain * (e_arr * e_arr)
            node.h_neg += float(weight)
            node.update_count_neg += 1

    return node


def accumulate_hit(node: NodeState, weight: float) -> None:
    """Increment ``node.hit_count`` by a non-negative rank weight."""

    weight_f = float(weight)
    if weight_f < 0.0:
        raise ValueError("hit-count weight must be non-negative")
    node.hit_count += weight_f
