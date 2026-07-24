"""Dirichlet-multinomial evidence gate invariants (SI S3.4, S3.5)."""
from __future__ import annotations

import numpy as np
import pytest

from tests.harness.markers import awaiting


@awaiting("evidence.dm_score", si="S3.4")
def test_fdm_closed_form():
    """F_DM must be insensitive to any optimizer iteration count."""
    pytest.fail("Not implemented")


@awaiting("evidence.dm_score", si="S3.5")
def test_fdm_monotone_in_evidence():
    """F_DM for the true topology must decrease as H_R grows."""
    pytest.fail("Not implemented")
