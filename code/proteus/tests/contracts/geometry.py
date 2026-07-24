"""Torsion, edge-ratio, and warp contracts (SI S5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TorsionState:
    """Torsion diagnostic for a single simplex (SI S5.1, S5.2)."""
    simplex_id: int
    omega_S: np.ndarray              # antisymmetric 2-form
    kappa_S: float                   # ||Omega_S||_F
    R_S: float                       # kappa_S / tau*
    ladder_band: str                 # "ignore", "monitor", "geometric_fix", "warp"


@dataclass
class EdgeRatioCheck:
    """Split-placement edge-ratio fallback result (SI S5.3)."""
    simplex_id: int
    r_LS: float                      # ell_max / ell_min
    passed: bool                     # r_LS <= 5


@dataclass
class WarpAttachment:
    """A local warp attached to a patch (SI S5.5, S5.6)."""
    patch_simplex_ids: list[int]
    warp_type: str                   # "mini_nsf" or "global_glow"
    pre_R_S_median: float
    post_R_S_median: float
    held_out_ll_delta: float
    accepted: bool
