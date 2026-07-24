"""PBMC 3k hierarchical clustering scenario."""
from __future__ import annotations

import pytest

from tests.harness.markers import awaiting, real_data


@real_data
@awaiting("stage1.controller", si="S2.5")
def test_pbmc3k_hierarchy_depth():
    """Scaffold hierarchy should have depth >= 2."""
    pytest.fail("Not implemented")


@real_data
@awaiting("stage1.ap_clustering", si="S2.6")
def test_pbmc3k_coarse_cluster_purity():
    """Coarse AP clusters should align with lymphoid/myeloid labels."""
    pytest.fail("Not implemented")


@real_data
@awaiting("inference.membership", si="S7.4")
def test_pbmc3k_fine_cluster_purity():
    """Fine AP clusters should align with T/B/NK/monocyte labels."""
    pytest.fail("Not implemented")
