"""Variable-density routing scenario."""
from __future__ import annotations
import pytest
from tests.harness.markers import awaiting

@awaiting("stage1.scaffold", si="S2.3")
def test_variable_density_node_distribution():
    """Nodes should be denser in the high-density region."""
    pytest.fail("Not implemented")

@awaiting("stage2.dual_flow", si="S6.2")
def test_variable_density_mass_cv():
    """Mass CV should be below threshold despite density variation."""
    pytest.fail("Not implemented")
