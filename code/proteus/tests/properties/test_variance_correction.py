"""Variance-correction rate invariants (SI S2.3)."""
from __future__ import annotations

import numpy as np

from proteus.rates import eta_gng


def test_eta_gng_formula():
    """eta_GNG,i must match (ln2 / 2k)(1 - sigma_i^2 / tau)."""
    tau = 2.0
    k = 8
    ratios = np.linspace(0.0, 0.95, 10)
    for ratio in ratios:
        sigma_sq = ratio * tau
        expected = np.log(2.0) / (2.0 * k) * (1.0 - sigma_sq / tau)
        assert np.isclose(eta_gng(sigma_sq=sigma_sq, tau=tau, k=k), expected)


def test_eta_gng_zero_at_cap():
    """eta_GNG,i must be zero when sigma_i^2 == tau."""
    assert eta_gng(sigma_sq=1.0, tau=1.0, k=8) == 0.0


def test_eta_gng_positive_under_cap():
    """eta_GNG,i must be positive when sigma_i^2 < tau."""
    assert eta_gng(sigma_sq=0.5, tau=1.0, k=8) > 0.0
