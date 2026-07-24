"""CMU MoCap scale-space and junction scenario."""
from __future__ import annotations

import pytest

from tests.harness.markers import awaiting, real_data


@real_data
@awaiting("stage1.controller", si="S2.5")
def test_mocap_two_scale_levels():
    """Scale-space should recover within-activity and activity-level scales."""
    pytest.fail("Not implemented")


@real_data
@awaiting("diagnostics.junction", si="S8.4")
def test_mocap_activity_junction_detection():
    """Junction detector should fire at activity transitions."""
    pytest.fail("Not implemented")
