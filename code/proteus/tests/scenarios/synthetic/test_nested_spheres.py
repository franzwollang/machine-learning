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
    """PH should recover expected Betti numbers per component.

    Intended path (OPEN_ISSUES #41; keep @awaiting until green on fitted regions):
    ``topology_from_accepted_regions`` (labels → ``per_region_topology``) on
    accepted-region node positions; expect per-shell ``b0 = 1`` and
    ``b_{sphere_dim} = 1``. Prefer lifetime reading over fixed ``1.5 sigma_star``
    if tissue pollution births short loops — fitted-circle probe still red at SI
    default (see ``test_ph_fitted_circle_probe``).
    """
    pytest.fail("Not implemented — awaiting green per-region PH on fitted spheres")
