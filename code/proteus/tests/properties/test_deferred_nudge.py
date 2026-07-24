"""Deferred-nudge invariants (SI S2.3)."""
from __future__ import annotations

import numpy as np

from proteus.deferred import apply_if_threshold
from proteus.nodes import make_node
from proteus.rates import eta_cent


def test_position_unchanged_below_threshold():
    """w_i must not change when ||a_i|| < delta_min."""
    node = make_node(np.zeros(2), dim=2)
    node.nudge = np.array([0.1, 0.0])
    assert not apply_if_threshold(node, delta_min_value=1.0)
    np.testing.assert_allclose(node.position, np.zeros(2))
    np.testing.assert_allclose(node.nudge, np.array([0.1, 0.0]))


def test_position_updates_at_threshold():
    """w_i must update when ||a_i|| >= delta_min, and a_i resets to zero."""
    node = make_node(np.zeros(2), dim=2)
    node.nudge = np.array([1.0, 0.0])
    assert apply_if_threshold(node, delta_min_value=0.5)
    np.testing.assert_allclose(node.position, np.array([1.0, 0.0]))
    np.testing.assert_allclose(node.nudge, np.zeros(2))


def test_centering_rate_formula():
    """eta_cent must equal kappa * (1-r) / k numerically."""
    kappa = 0.5
    r = 1.0 / np.sqrt(2.0)
    k = 8
    assert np.isclose(eta_cent(kappa, r, k), kappa * (1.0 - r) / k)
