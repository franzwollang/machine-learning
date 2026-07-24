"""Scale-grid evaluation wall-time benchmark."""

from __future__ import annotations
import pytest
from tests.harness.markers import awaiting


@awaiting("stage1.scale_grid", si="S2.5")
def test_scale_grid_time_small():
    """Full grid evaluation on N~1000 must complete within small budget."""
    pytest.fail("Not implemented")
