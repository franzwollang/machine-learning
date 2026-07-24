"""Mass and flux conservation invariants (SI S6.2)."""
from __future__ import annotations

import numpy as np
import pytest

from tests.harness.markers import awaiting


@awaiting("stage2.dual_flow", si="S6.2")
def test_epsilon_mass_after_normalization():
    """epsilon_mass must be <= 1e-6 after simplex-mass normalization."""
    pytest.fail("Not implemented")


@awaiting("stage2.dual_flow", si="S6.2")
def test_epsilon_flux_threshold():
    """epsilon_flux must be below 1e-3; warn if above."""
    pytest.fail("Not implemented")
