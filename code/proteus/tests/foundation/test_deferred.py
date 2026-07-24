"""Unit tests for deferred-nudge helpers."""

from __future__ import annotations

import numpy as np

from proteus.deferred import accumulate_nudge, apply_if_threshold
from proteus.nodes import make_node


def test_no_fire_below_threshold() -> None:
    node = make_node(np.zeros(2), dim=2)
    node.nudge = np.array([0.1, 0.0])

    fired = apply_if_threshold(node, delta_min_value=1.0)

    assert not fired
    np.testing.assert_allclose(node.position, np.zeros(2))
    np.testing.assert_allclose(node.nudge, np.array([0.1, 0.0]))


def test_fire_updates_position_and_resets() -> None:
    node = make_node(np.zeros(2), dim=2)
    node.nudge = np.array([1.0, 0.0])

    fired = apply_if_threshold(node, delta_min_value=0.5)

    assert fired
    np.testing.assert_allclose(node.position, np.array([1.0, 0.0]))
    np.testing.assert_allclose(node.nudge, np.zeros(2))


def test_accumulate_nudge_uses_rho_scaled_step() -> None:
    node = make_node(np.zeros(2), dim=2)
    node.residual_mean = np.array([3.0, 4.0])
    node.variance = 4.0
    node.update_count = 5

    accumulate_nudge(node, np.array([1.0, 0.0]), eta_cent_value=0.1, eps=0.0)

    # rho = ||m|| / sigma = 5 / 2 = 2.5; below rho_max=10.0
    np.testing.assert_allclose(node.nudge, np.array([0.25, 0.0]))


def test_accumulate_nudge_blocked_when_no_updates() -> None:
    node = make_node(np.zeros(2), dim=2)
    node.residual_mean = np.array([3.0, 4.0])
    node.variance = 0.0  # no BMU updates yet
    node.update_count = 0

    accumulate_nudge(node, np.array([1.0, 0.0]), eta_cent_value=0.1, eps=1e-8)

    np.testing.assert_allclose(node.nudge, np.zeros(2))
