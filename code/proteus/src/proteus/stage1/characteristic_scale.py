"""Compensated node-count characteristic-scale signal (SI S2.5, S2.5.1).

This module implements the **primary** characteristic-scale signal called for by
OPEN_ISSUES #28: the compensated node-count / support-volume trace.  The raw
Lindeberg response ``R_i(tau)`` is self-normalizing at equilibrium (a converged
scaffold equalizes hit masses and locks the k-NN radius to the cap, so the
tau-dependence cancels), which is why the legacy variance-load band selector had
to lean on empirical constants.  The signal that equilibration *cannot*
normalize away is how the equilibrium node count scales with ``tau``.

On a ``d``-dimensional support each settled node occupies a catchment of volume
``~tau^{d/2}``, so the equilibrium node count obeys ``N(tau) ~ V_supp /
tau^{d/2}`` once the support is well resolved.  The **compensated node count**

    C(tau) = N(tau) * tau^{d/2}   (equivalently the support-volume trace V_C)

therefore rises while the support is under-resolved (node count floored) and
flattens onto a plateau at ``~V_supp`` once every intrinsic direction is
resolved.  The coarse edge of that plateau --- the coarsest ``tau`` at which the
scaffold has just become well resolved --- is the characteristic scale
(``v_death^+`` in the S2.6.2 persistence language: the coarsest grid point at
which the resolved structure first appears).  This is exactly the "transition
between unresolved support and redundant over-refinement" that S2.5.1 asks the
support trace to cross-check.

The signal is exposed behind the controller ``selector`` flag while the legacy
load-band selector remains the transition default (OPEN_ISSUES #28); flipping the
default is migrated scenario by scenario so scale-search regressions can bisect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CharacteristicScaleConfig:
    """Configuration for the compensated node-count knee selector (SI S2.5.1).

    Attributes
    ----------
    plateau_tol:
        Maximum relative change in the compensated node count ``C(tau)`` between
        adjacent stabilized grid points for the pair to count as "on the
        plateau".  Operational default backstopped by the scale-search
        falsifiers (circle / swiss-roll tau* recovery); OPEN_ISSUES #28, S14.3.
    min_plateau:
        Number of adjacent grid points that must share the plateau before its
        coarse edge is accepted as the characteristic scale.  The minimal
        non-trivial plateau is two grid points (one adjacent pair), matching the
        persistence interval of S2.6.2; derived rather than tuned.
    use_estimated_dim:
        When ``True`` the compensation exponent ``d/2`` uses the per-grid-point
        estimated intrinsic dimension (degree proxy, S1.4); when ``False`` it
        uses the working/ambient dimension.  The intrinsic dimension is correct
        for embedded manifolds (a circle in R^2 has ``d=1``), so this is the
        default.
    """

    plateau_tol: float = 0.15
    min_plateau: int = 2
    use_estimated_dim: bool = True


@dataclass
class CharacteristicScaleResult:
    """Result of the compensated node-count knee search (SI S2.5.1).

    Attributes
    ----------
    compensated_trace:
        ``C(tau_j) = N(tau_j) * tau_j^{d_j/2}`` per grid point (``nan`` where the
        grid point did not stabilize).
    rel_changes:
        Relative change of ``C`` between adjacent grid points; ``rel_changes[j]``
        compares grid points ``j`` (coarser) and ``j+1`` (finer).  ``nan`` where
        either endpoint is unusable.
    d_trace:
        Compensation dimension used at each grid point.
    knee_index / knee_tau:
        Coarsest grid index (and its ``tau``) that begins a plateau of at least
        ``min_plateau`` grid points, or ``None`` if no plateau is found (the
        searched band never resolves the support).
    """

    compensated_trace: np.ndarray
    rel_changes: np.ndarray
    d_trace: np.ndarray
    knee_index: Optional[int]
    knee_tau: Optional[float]


def compensated_node_count(
    node_counts: np.ndarray,
    tau_grid: np.ndarray,
    d_trace: np.ndarray,
) -> np.ndarray:
    """Return the compensated node-count trace ``C(tau) = N(tau) * tau^{d/2}``.

    ``d_trace`` supplies the (possibly per-grid-point) compensation dimension.
    """

    node_counts = np.asarray(node_counts, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    d_trace = np.asarray(d_trace, dtype=float)
    safe_tau = np.maximum(tau_grid, 0.0)
    return node_counts * np.power(safe_tau, d_trace / 2.0)


def select_characteristic_scale_by_node_count(
    node_counts: np.ndarray,
    tau_grid: np.ndarray,
    d_trace: np.ndarray,
    stabilized: list[bool],
    config: CharacteristicScaleConfig | None = None,
) -> CharacteristicScaleResult:
    """Select the characteristic scale from the compensated node-count plateau.

    ``tau_grid`` is descending (coarse to fine), matching the controller sweep.
    The characteristic scale is the **coarse edge** of the compensated-count
    plateau: the coarsest stabilized grid point from which the relative change of
    ``C`` stays within ``plateau_tol`` for at least ``min_plateau`` consecutive
    grid points.
    """

    config = config if config is not None else CharacteristicScaleConfig()
    node_counts = np.asarray(node_counts, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    d_trace = np.asarray(d_trace, dtype=float)
    n = len(tau_grid)

    compensated = compensated_node_count(node_counts, tau_grid, d_trace)
    stab = np.array(stabilized, dtype=bool)
    masked = np.where(stab, compensated, np.nan)

    rel_changes = np.full(max(n - 1, 0), np.nan, dtype=float)
    for j in range(n - 1):
        a = masked[j]
        b = masked[j + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        denom = max(abs(a), abs(b), 1e-12)
        rel_changes[j] = abs(b - a) / denom

    knee_index: Optional[int] = None
    # A plateau of length ``L`` spans ``L`` grid points and ``L-1`` adjacent
    # pairs; the coarsest index that starts such a plateau is the knee.
    needed_pairs = max(config.min_plateau - 1, 1)
    for i in range(n - needed_pairs):
        if not stab[i]:
            continue
        window = rel_changes[i : i + needed_pairs]
        if window.size < needed_pairs:
            break
        if np.all(np.isfinite(window)) and np.all(window <= config.plateau_tol):
            knee_index = i
            break

    knee_tau = float(tau_grid[knee_index]) if knee_index is not None else None
    return CharacteristicScaleResult(
        compensated_trace=compensated,
        rel_changes=rel_changes,
        d_trace=d_trace,
        knee_index=knee_index,
        knee_tau=knee_tau,
    )
