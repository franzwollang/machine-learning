"""Scale-parameter, response-trace, and selection contracts (SI S2.4, S2.5)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ScaleParameter:
    """A single candidate or selected scale (SI S2.4).

    ``tau_global`` is the operational scale primitive. The bounded controller value
    ``s_control`` of SI S2.4 is a normalization remark only
    (``tau = -d_subspace * log(1 - s_control)``) and is not stored or consumed.
    """
    tau_global: float                # variance cap tau (uniform within a run)
    d_subspace: int


@dataclass
class ScaleResponseTrace:
    """Phi_C(tau) and V_C(tau) evaluated on a geometric grid (SI S2.5)."""
    tau_values: np.ndarray           # (J,) grid of tau
    phi_values: np.ndarray           # (J,) cluster response
    support_values: np.ndarray       # (J,) support trace V_C


@dataclass
class ScaleSelection:
    """Result of characteristic-scale selection (SI S2.5.1)."""
    tau_star: float
    response_at_star: float
    bracket_indices: tuple[int, int]
    trace: ScaleResponseTrace
