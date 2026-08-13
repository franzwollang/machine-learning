"""Nested-sphere clean-shell PH stepping stone (OPEN_ISSUES #41, A4-T8).

Evidence-gathering on *clean* Fibonacci-sphere shells via
``topology_from_accepted_regions`` and signal-label filtering. Does **not**
flip ``test_nested_spheres_topology``, ``test_linked_tori_betti_numbers``, or
circle ``b1=1`` ``@awaiting`` recovery tests.

Measured probe findings (Fibonacci S², not lat/lon — grids birth spurious
``b1`` under lifetime):
  * Single shell ``n=400``, ``radius=1``, ``sigma=0.4``: both
    ``fixed_threshold`` and ``lifetime`` with ``lifetime_frac=0.5`` recover
    ``(b0,b1,b2)=(1,0,1)``.
  * Nested shells radii ``(1.0, 2.0)``, ``~175`` Fibonacci samples/shell,
    ``sigma`` proportional to radius ``[0.4, 0.8]``: per-region
    ``fixed_threshold`` recovers ``(1,0,1)`` per shell; at this modest ``n``
    ``lifetime_frac=0.5`` inflates ``b0`` (essential components per point) —
    ``lifetime_frac=0.75`` restores ``(1,0,1)`` per shell.
  * Whole-cloud tissue pollution (``tissue_fraction≈0.2``) spoils Betti;
    ``include_labels=[1, 2]`` signal filter restores per-shell readings.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.datasets.synthetic.tissue import append_uniform_tissue
from tests.metrics.persistent_homology import (
    DEFAULT_LIFETIME_FRAC,
    compare_readings,
    topology_from_accepted_regions,
)


def fibonacci_sphere(
    n: int,
    radius: float = 1.0,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Even-ish samples on ``S^2`` via golden spiral (not a lat/lon grid)."""
    rng = np.random.default_rng(seed)
    i = np.arange(n, dtype=float)
    phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden * i
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    pts = np.column_stack([x, y, z])
    return pts + 1e-4 * rng.normal(size=pts.shape)


def _nested_fibonacci_shells(
    n_per_shell: int = 175,
    radii: tuple[float, float] = (1.0, 2.0),
    sigma_scale: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Two labeled Fibonacci shells with per-shell ``sigma ∝ radius``."""
    r_inner, r_outer = radii
    inner = fibonacci_sphere(n_per_shell, r_inner, seed=1)
    outer = fibonacci_sphere(n_per_shell, r_outer, seed=2)
    points = np.vstack([inner, outer])
    labels = np.array([1] * n_per_shell + [2] * n_per_shell, dtype=int)
    sigmas = [sigma_scale * r_inner, sigma_scale * r_outer]
    return points, labels, sigmas


@pytest.mark.scenario
@pytest.mark.synthetic
def test_fibonacci_sphere_both_readings_recover_shell_betti() -> None:
    """Dense Fibonacci shell: fixed and lifetime (0.5) both see ``(1,0,1)``."""
    pts = fibonacci_sphere(400, radius=1.0, seed=0)
    sigma = 0.4
    cmp = compare_readings(pts, sigma, max_dim=2, lifetime_frac=0.5)
    assert cmp.n_points == 400
    assert cmp.fixed_threshold == (1, 0, 1)
    assert cmp.lifetime == (1, 0, 1)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_shells_per_region_fixed_threshold() -> None:
    """Per accepted shell label, fixed_threshold recovers ``(1,0,1)`` each."""
    points, labels, sigmas = _nested_fibonacci_shells()
    reports = topology_from_accepted_regions(
        points,
        labels,
        sigmas,
        include_labels=[1, 2],
        reading="fixed_threshold",
        max_dim=2,
    )
    assert [r.region_id for r in reports] == [1, 2]
    for rep in reports:
        assert rep.betti == (1, 0, 1)
        assert rep.sigma_star in (0.4, 0.8)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_shells_lifetime_frac_window_at_modest_n() -> None:
    """Modest ``n`` per shell: ``lifetime_frac=0.5`` inflates ``b0``; 0.75 OK."""
    points, labels, sigmas = _nested_fibonacci_shells()

    inflated = topology_from_accepted_regions(
        points,
        labels,
        sigmas,
        include_labels=[1, 2],
        reading="lifetime",
        max_dim=2,
        lifetime_frac=DEFAULT_LIFETIME_FRAC,
    )
    for rep in inflated:
        assert rep.betti[2] == 1
        assert rep.betti[1] == 0
        assert rep.betti[0] > 1  # measured: essential H0 overcount at modest n

    recovered = topology_from_accepted_regions(
        points,
        labels,
        sigmas,
        include_labels=[1, 2],
        reading="lifetime",
        max_dim=2,
        lifetime_frac=0.75,
    )
    assert [r.region_id for r in recovered] == [1, 2]
    for rep in recovered:
        assert rep.betti == (1, 0, 1)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_shells_tissue_filter_restores_per_shell_betti() -> None:
    """Whole-cloud pollution spoils Betti; signal labels restore per shell."""
    points, labels, sigmas = _nested_fibonacci_shells()
    polluted, plabels, meta = append_uniform_tissue(
        points,
        labels,
        rng=np.random.default_rng(11),
        tissue_fraction=0.2,
    )
    assert meta["tissue_count"] > 0

    whole = compare_readings(
        polluted, sigmas[1], max_dim=2, lifetime_frac=DEFAULT_LIFETIME_FRAC,
    )
    assert whole.lifetime[0] > 1  # whole-cloud lifetime polluted

    fixed = topology_from_accepted_regions(
        polluted,
        plabels,
        sigmas,
        include_labels=[1, 2],
        reading="fixed_threshold",
        max_dim=2,
    )
    for rep in fixed:
        assert rep.betti == (1, 0, 1)

    life = topology_from_accepted_regions(
        polluted,
        plabels,
        sigmas,
        include_labels=[1, 2],
        reading="lifetime",
        max_dim=2,
        lifetime_frac=0.75,
    )
    for rep in life:
        assert rep.betti == (1, 0, 1)
