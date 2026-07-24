"""OpenFace CK+ emotion and intensity scenario."""
from __future__ import annotations

import pytest

from tests.harness.markers import awaiting, real_data


@real_data
@awaiting("stage1.controller", si="S2.5")
def test_ck_plus_emotion_cluster_separation():
    """Emotion clusters should separate at Stage 1 with purity above threshold."""
    pytest.fail("Not implemented")


@real_data
@awaiting("inference.membership", si="S7.4")
def test_ck_plus_intensity_continuity():
    """Onset/apex/offset phases should appear as continuous within-cluster trajectories."""
    pytest.fail("Not implemented")
