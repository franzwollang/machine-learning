"""Dual-flow GBP iteration count benchmark."""

from __future__ import annotations
import pytest
from tests.harness.markers import awaiting


@awaiting("stage2.dual_flow", si="S6.2")
def test_dual_flow_converges_within_budget():
    """Loopy GBP must converge within the iteration budget."""
    pytest.fail("Not implemented")
