"""Unit tests for Stage 1 rate helpers."""

from __future__ import annotations

import numpy as np
import pytest

from proteus.rates import delta_min, eta_cent, eta_gng


def test_eta_gng_formula_under_cap() -> None:
    expected = np.log(2.0) / (2.0 * 8) * (1.0 - 0.25)

    assert np.isclose(eta_gng(sigma_sq=0.25, tau=1.0, k=8), expected)


def test_eta_gng_zero_at_and_above_cap() -> None:
    assert eta_gng(sigma_sq=1.0, tau=1.0, k=8) == 0.0
    assert eta_gng(sigma_sq=2.0, tau=1.0, k=8) == 0.0


def test_eta_cent_formula() -> None:
    assert np.isclose(eta_cent(kappa=0.5, r=1.0 / np.sqrt(2.0), k=8), 0.01830582617584078)


def test_delta_min_formula() -> None:
    value = delta_min(kappa=0.5, r=1.0 / np.sqrt(2.0), tau=4.0)
    assert np.isclose(value, 0.5 * (1.0 - 1.0 / np.sqrt(2.0)) * 2.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_sq": 0.0, "tau": 0.0, "k": 8},
        {"sigma_sq": 0.0, "tau": 1.0, "k": 0},
    ],
)
def test_eta_gng_validates_inputs(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        eta_gng(**kwargs)
