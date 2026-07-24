"""Expressivity error ledger for Proteus evaluation (SI S9.2)."""
from __future__ import annotations

import numpy as np


def expressivity_ledger(
    mass_cv: float,
    junction_residual: float,
    torsion_q95: float,
    held_out_ll_delta: float,
    stat_scale: float,
    weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0),
) -> float:
    """Compute the E_tot operational error ledger.

    Parameters follow SI S9.2:
      w_mass * CV_S + w_junc * (eta_s + beta_max) + w_tors * q_95(R_S)
      + w_ll * max(0, -delta_ell_heldout) + w_stat * Delta_stat
    """
    w_mass, w_junc, w_tors, w_ll, w_stat = weights
    return (
        w_mass * mass_cv
        + w_junc * junction_residual
        + w_tors * torsion_q95
        + w_ll * max(0.0, -held_out_ll_delta)
        + w_stat * stat_scale
    )


def is_expressively_saturated(
    mass_cv: float,
    torsion_q95: float,
    junction_residual: float,
    held_out_ll_delta: float,
    cv_threshold: float = 0.01,
    torsion_threshold: float = 0.30,
    junction_threshold: float = 0.10,
    ll_se_threshold: float = 1.0,
) -> bool:
    """Check the stopping rule from SI S9.2 / S12.1."""
    return (
        mass_cv < cv_threshold
        and torsion_q95 < torsion_threshold
        and junction_residual < junction_threshold
        and held_out_ll_delta < ll_se_threshold
    )
