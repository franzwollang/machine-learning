"""Gaussian summary, membership, and trajectory contracts (SI S7)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GaussianSummary:
    """Per-region Gaussian fit used for canonical membership (SI S7.1, S7.2)."""
    region_id: int
    mu_C: np.ndarray                 # cluster mode in y-space
    Sigma_C: np.ndarray              # covariance in y-space
    has_warp: bool = False


@dataclass
class Membership:
    """A single-level membership score (SI S7.2)."""
    region_id: int
    score: float                     # mu_C(x) in (0, 1]


@dataclass
class MembershipTrajectory:
    """Multiscale fuzzy membership trajectory (SI S7.4)."""
    path: list[Membership]           # root to leaf

    @property
    def scores(self) -> np.ndarray:
        return np.array([m.score for m in self.path])

    @property
    def region_ids(self) -> list[int]:
        return [m.region_id for m in self.path]

    @property
    def depth(self) -> int:
        return len(self.path)
