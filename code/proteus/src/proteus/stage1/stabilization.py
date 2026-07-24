"""Thermal-equilibrium checks for Stage 1 scaffolds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from proteus.moments import incoherence_ratio


def cv_threshold(k: int, buffer: float = 1.5) -> float:
    """EWMA noise-floor threshold for variance CV: ``buffer * sqrt(2/k)``."""

    return float(buffer) * float(np.sqrt(2.0 / max(int(k), 1)))


@dataclass(frozen=True)
class StabilizationConfig:
    """Configuration for fixed-tau thermal-equilibrium stopping.

    When ``cv_tolerance`` is None the threshold is derived automatically
    from the scaffold's ``k`` via ``cv_threshold(k, cv_buffer)``.
    """

    cv_tolerance: Optional[float] = None
    cv_buffer: float = 1.5
    min_equilibrium_epochs: int = 3
    max_epochs: int = 40


def compute_variance_cv(scaffold: Any, *, eps: float = 1e-12) -> float:
    """CV of sigma^2 over mature nodes (update_count >= prune_after).

    Returns inf when fewer than 4 mature nodes are available.
    """

    prune_after = getattr(scaffold, "prune_after", 0)
    mature = [n for n in scaffold.nodes if n.update_count >= prune_after]
    if len(mature) < 4:
        return float("inf")
    variances = np.array([n.variance for n in mature], dtype=float)
    mean_var = float(variances.mean())
    if mean_var <= eps:
        return float("inf")
    return float(np.std(variances) / mean_var)


def compute_neighbor_normalized_cv(scaffold: Any, eps: float = 1e-8) -> float:
    """CV over neighbor-normalized incoherence ratios (diagnostic only)."""

    if not scaffold.nodes:
        return float("inf")
    graph = scaffold.neighbour_graph()
    rhos = np.array([
        incoherence_ratio(
            node.residual_mean,
            sigma=float(np.sqrt(max(node.variance, 0.0))),
            eps=eps,
        )
        for node in scaffold.nodes
    ])
    global_mean = float(np.mean(rhos)) if rhos.size else 0.0
    rho_tilde = np.empty_like(rhos)
    for idx, rho in enumerate(rhos):
        neighbours = graph.get(idx, [])
        if neighbours:
            neighbour_mean = float(np.mean(rhos[neighbours]))
        else:
            neighbour_mean = global_mean
        rho_tilde[idx] = rho / (neighbour_mean + eps)

    mean = float(np.mean(rho_tilde))
    if mean <= eps:
        return 0.0
    return float(np.std(rho_tilde) / max(mean, eps))


def is_stable(
    history: list[float],
    config: StabilizationConfig,
    scaffold: Any = None,
) -> bool:
    """Return true if the last required epochs are below CV tolerance."""

    if len(history) < config.min_equilibrium_epochs:
        return False
    if config.cv_tolerance is not None:
        tol = config.cv_tolerance
    elif scaffold is not None:
        tol = cv_threshold(scaffold.k, config.cv_buffer)
    else:
        return False
    tail = history[-config.min_equilibrium_epochs:]
    return all(value < tol for value in tail)
