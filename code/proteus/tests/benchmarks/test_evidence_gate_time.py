"""Evidence gate scoring wall-time benchmark."""

from __future__ import annotations
import pytest
from tests.harness.markers import awaiting


@awaiting("evidence.gate", si="S3.6")
def test_evidence_gate_time_small():
    """F_DM scoring on a small affected region must complete within budget."""
    pytest.fail("Not implemented")
