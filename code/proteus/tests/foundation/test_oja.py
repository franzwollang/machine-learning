"""Unit tests for Oja direction updates."""

from __future__ import annotations

import numpy as np

from proteus.oja import update_oja


def test_oja_output_unit_norm() -> None:
    updated = update_oja(np.array([1.0, 0.0]), np.array([1.0, 1.0]), eta=0.1)

    assert np.isclose(np.linalg.norm(updated), 1.0)


def test_oja_zero_norm_fallback() -> None:
    updated = update_oja(np.array([0.0, 0.0]), np.array([0.0, 0.0]), eta=0.1)

    np.testing.assert_allclose(updated, np.array([1.0, 0.0]))


def test_oja_aligns_with_repeated_residual_direction() -> None:
    rng = np.random.default_rng(0)
    target = np.array([1.0, 0.0, 0.0])
    u = np.array([0.1, 1.0, 0.0])
    u = u / np.linalg.norm(u)
    for _ in range(400):
        e = target + rng.normal(scale=0.05, size=3)
        u = update_oja(u, e, eta=0.02)

    assert abs(float(np.dot(u, target))) > 0.90
