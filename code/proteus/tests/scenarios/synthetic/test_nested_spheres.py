"""Nested-sphere topology scenario."""
from __future__ import annotations
import pytest
from tests.harness.markers import awaiting

@awaiting("stage1.controller", si="S2.5")
def test_nested_spheres_two_scales():
    """Scale controller should find two characteristic scales."""
    pytest.fail("Not implemented")

@awaiting("stage2.flag_complex", si="S4.1")
def test_nested_spheres_topology():
    """PH should recover expected Betti numbers per component."""
    pytest.fail("Not implemented")
