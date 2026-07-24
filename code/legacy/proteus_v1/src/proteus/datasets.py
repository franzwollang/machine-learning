"""Utility dataset generators for Proteus tests and demos."""
from __future__ import annotations

import numpy as np


def make_circle(n_samples: int, radius: float = 1.0, noise: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate points on a noisy circle in R^2."""
    if rng is None:
        rng = np.random.default_rng()
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    points = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
    if noise > 0:
        points += rng.normal(scale=noise, size=points.shape)
    return points


def make_swiss_roll(
    n_samples: int,
    height: float = 1.0,
    twists: float = 3.0,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a Swiss-roll style manifold in R^3.

    This mirrors the classic manifold-learning benchmark used in the
    Proteus documentation.  ``twists`` controls the number of windings
    (default corresponds to the range 3π–9π often used in the literature).
    """

    if rng is None:
        rng = np.random.default_rng()
    t = rng.uniform(1.5 * np.pi, 1.5 * np.pi + twists * np.pi, size=n_samples)
    h = rng.uniform(0.0, height, size=n_samples)
    x = t * np.cos(t)
    z = t * np.sin(t)
    roll = np.stack([x, h, z], axis=1)
    roll /= (1.5 * np.pi + twists * np.pi)
    if noise > 0:
        roll += rng.normal(scale=noise, size=roll.shape)
    return roll


__all__ = ["make_circle", "make_swiss_roll"]
