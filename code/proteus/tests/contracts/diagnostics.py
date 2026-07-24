"""Junction, ledger, and boundary diagnostics contracts (SI S8.4, S9.2, S6.3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


@dataclass
class JunctionScore:
    """Per-vertex junction score (SI S8.4)."""
    vertex_id: int
    J_i: int                         # 0..5
    frozen: bool = False


class BoundaryType(Enum):
    TRUE_MANIFOLD = "true_manifold"
    COMPUTATIONAL = "computational"
    ORIENTATION_SEAM = "orientation_seam"


@dataclass
class BoundaryClassification:
    """Per-facet boundary classification (SI S6.3)."""
    facet_id: int
    boundary_type: BoundaryType


@dataclass
class ExpressivityLedger:
    """Snapshot of the operational error ledger (SI S9.2)."""
    mass_cv: float
    junction_residual: float         # eta_s + beta_max
    torsion_q95: float
    held_out_ll_delta: float
    stat_scale: float                # Delta_stat
    e_tot: float
    saturated: bool
