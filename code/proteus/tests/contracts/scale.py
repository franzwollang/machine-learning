"""Scale-parameter, response-trace, and selection contracts (SI S2.4, S2.5)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ScaleParameter:
    """A single candidate or selected scale (SI S2.4).

    ``tau_global`` is the operational scale primitive. ``s_control`` is a
    reporting/normalization convention only -- ``1 - exp(-tau / d_subspace)``
    -- with no operational consumer (OPEN_ISSUES #35).
    """
    tau_global: float                # operational scale primitive
    d_subspace: int
    s_control: float | None = None   # reporting only; 1 - exp(-tau / d_subspace)


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
