"""Peak memory envelope benchmark."""

from __future__ import annotations
import pytest
from tests.harness.markers import awaiting


@awaiting("stage1.scaffold", si="S4.6")
def test_peak_memory_small():
    """Full pipeline on N~1000 must stay within small memory budget."""
    pytest.fail("Not implemented")
