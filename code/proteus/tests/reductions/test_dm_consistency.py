"""DM gate consistency reduction test (SI S3.5)."""
from __future__ import annotations

import numpy as np
import pytest

from tests.harness.markers import awaiting


@awaiting("evidence.dm_score", si="S3.5")
def test_dm_selects_true_topology():
    """As H_R -> inf, the DM marginal must select the correct topology
    among a finite candidate set with probability -> 1."""
    pytest.fail("Not implemented")


@awaiting("evidence.dm_score", si="S3.5")
def test_dm_margin_dominates_occam():
    """The O(H_R * Delta_R) likelihood margin must dominate the O(log H_R)
    Occam factor and fixed log(tau_BF) threshold."""
    pytest.fail("Not implemented")
