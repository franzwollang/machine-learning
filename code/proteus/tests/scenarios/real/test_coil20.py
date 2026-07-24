"""COIL-20 topology recovery scenario."""
from __future__ import annotations

import pytest

from tests.harness.markers import awaiting, real_data


@real_data
@awaiting("stage1.controller", si="S2.5")
def test_coil20_component_count():
    """Should recover 20 connected components."""
    pytest.fail("Not implemented")


@real_data
@awaiting("stage2.flag_complex", si="S4.1")
def test_coil20_loop_topology():
    """Each object component should have b1 >= 1 (loop)."""
    pytest.fail("Not implemented")


@real_data
@awaiting("inference.membership", si="S7.4")
def test_coil20_viewpoint_trajectory_smoothness():
    """Intra-object membership trajectory should be smooth in viewpoint angle."""
    pytest.fail("Not implemented")
