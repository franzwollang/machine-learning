"""SVD-limit reduction test (SI S11)."""
from __future__ import annotations

import numpy as np
import pytest

from tests.harness.markers import awaiting


@awaiting("stage1.scaffold", si="S11")
def test_frozen_topology_ewma_converges_to_local_pca():
    """Under frozen topology and isotropic large-k neighborhoods, EWMA
    moments must converge to Gaussian-windowed local PCA at bandwidth
    sigma = sqrt(tau) / c_{d,k}."""
    pytest.fail("Not implemented")


@awaiting("stage1.scaffold", si="S11")
def test_oja_recovers_principal_direction():
    """Under the SVD-limit conditions, the Oja direction must converge to
    the dominant eigenvector of the local Gaussian-windowed covariance."""
    pytest.fail("Not implemented")
