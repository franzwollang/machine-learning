"""Star-matrix conditioning invariants (SI S10.4)."""
from __future__ import annotations

import numpy as np
import pytest

from tests.harness.markers import awaiting


@awaiting("evidence.star_matrix", si="S10.4")
def test_conditioning_above_rho_min():
    """sigma_min(K_i)/sigma_max(K_i) must be >= rho_min after edits."""
    pytest.fail("Not implemented")


@awaiting("evidence.star_matrix", si="S10.4")
def test_ill_conditioned_stars_quarantined():
    """Stars with conditioning below rho_min must not contribute to F_DM."""
    pytest.fail("Not implemented")
