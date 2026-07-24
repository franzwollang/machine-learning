"""Dimensionality-junction scenario."""
from __future__ import annotations
import pytest
from tests.harness.markers import awaiting

@awaiting("diagnostics.junction", si="S8.4")
def test_junction_detected():
    """J_i >= 3 should fire at stars near the 1D/2D junction."""
    pytest.fail("Not implemented")

@awaiting("diagnostics.junction", si="S8.4")
def test_junction_freeze_prevents_oscillation():
    """Frozen junction stars should block cross-junction splits."""
    pytest.fail("Not implemented")
