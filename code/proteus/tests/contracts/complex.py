"""Simplicial complex, mass field, and face-pressure contracts (SI S4, S6, S10)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Simplex:
    """A top-dimensional simplex in the flag complex (SI S4.1)."""
    vertex_ids: tuple[int, ...]
    volume: float = 0.0              # |S|_d
    mass: float = 0.0                # m_S


@dataclass
class Complex:
    """A simplicial complex for a single region (SI S4)."""
    simplices: list[Simplex]
    vertex_positions: np.ndarray     # (N_nodes, d)
    intrinsic_dim: int


@dataclass
class MassField:
    """Settled simplex mass field for a region (SI S10)."""
    masses: np.ndarray               # (M,) one per simplex
    region_id: int


@dataclass
class FacePressureField:
    """Conservative face-pressure field for a region (SI S6)."""
    empirical_tallies: np.ndarray    # hat{p}_f, (F,)
    conservative_pressures: np.ndarray  # p_f, (F,)
    region_id: int
