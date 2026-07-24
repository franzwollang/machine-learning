"""Tests for Stage 1 routing weight selection."""

from __future__ import annotations

import numpy as np

from proteus.stage1.routing_weights import (
    gaussian_relative_weights,
    rank_decay_weights,
    routing_weights,
)


def test_rank_decay_weights() -> None:
    np.testing.assert_allclose(
        rank_decay_weights(4),
        np.array([1.0, 0.5, 0.25, 0.125]),
    )


def test_gaussian_relative_weights_anchor_bmu_at_one() -> None:
    distances = np.array([0.2, 0.3, 0.6])
    weights = gaussian_relative_weights(distances, tau=0.05)

    assert np.isclose(weights[0], 1.0)
    assert np.all(weights[1:] < 1.0)
    assert weights[1] > weights[2]


def test_low_dimensional_routing_uses_gaussian_weights() -> None:
    distances = np.array([0.1, 0.2, 0.4])

    weights = routing_weights(distances, tau=0.05, ambient_dim=3)

    np.testing.assert_allclose(weights, gaussian_relative_weights(distances, 0.05))


def test_high_dimensional_routing_uses_rank_decay() -> None:
    distances = np.array([0.1, 0.2, 0.4])

    weights = routing_weights(distances, tau=0.05, ambient_dim=9)

    np.testing.assert_allclose(weights, rank_decay_weights(3))
