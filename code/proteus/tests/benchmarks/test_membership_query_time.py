"""Membership query wall-time benchmark."""

from __future__ import annotations
import pytest
from tests.harness.markers import awaiting


@awaiting("inference.membership", si="S7")
def test_membership_query_time_small():
    """Single membership-trajectory query must complete within budget."""
    pytest.fail("Not implemented")
