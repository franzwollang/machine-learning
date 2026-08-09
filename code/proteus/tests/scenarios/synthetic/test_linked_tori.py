"""Linked-tori topology scenario."""
from __future__ import annotations
import pytest
from tests.harness.markers import awaiting

@awaiting("stage1.controller", si="S2.5")
def test_linked_tori_component_separation():
    """AP clustering should identify two components."""
    pytest.fail("Not implemented")

@awaiting("stage2.flag_complex", si="S4.1")
def test_linked_tori_betti_numbers():
    """PH should recover b1>=2 for each torus component.

    Intended path (OPEN_ISSUES #41; keep @awaiting until green on fitted regions):
    split accepted regions → ``per_region_topology(..., reading='lifetime')`` on
    node positions (not the lifted-graph flag complex) with
    ``sigma_star = sqrt(tau_star)``. Do not flip this test by weakening thresholds.
    """
    pytest.fail("Not implemented — awaiting green per-region PH on fitted tori")
