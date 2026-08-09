"""Harness smoke for per-region Vietoris--Rips PH (SI S14.2, OPEN_ISSUES #41).

These tests exercise ``tests.metrics.persistent_homology`` on *clean geometric*
point clouds. They do **not** flip the Stage-2 topology-recovery scenario
assertions (``test_nested_spheres_topology``, ``test_linked_tori_betti_numbers``,
circle ``b1=1`` on tissue-polluted scaffolds) — those stay ``@awaiting`` until
filtration/reading choices are green on real fitted regions.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.metrics.persistent_homology import (
    DEFAULT_LIFETIME_FRAC,
    FILTRATION_MULTIPLIER,
    extract_region_node_positions,
    lifetime_betti_numbers,
    per_region_topology,
    region_betti_numbers,
    sigma_star_from_tau,
    topology_from_accepted_regions,
)


def _clean_circle(n: int = 48, radius: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    # Tiny jitter so pairwise distances are generic; keep << radius.
    jitter = 1e-4 * rng.normal(size=(n, 2))
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)]) + jitter


@pytest.mark.scenario
@pytest.mark.synthetic
def test_ph_harness_lifetime_recovers_clean_circle_b1() -> None:
    """Lifetime reading on a clean circle recovers b0=1, b1=1."""
    pts = _clean_circle()
    # Chord length between adjacent samples ~ 2π/n; choose sigma well above that.
    sigma = 0.4
    betti = lifetime_betti_numbers(pts, sigma, max_dim=1, lifetime_frac=0.25)
    assert betti[0] >= 1
    assert betti[1] == 1


@pytest.mark.scenario
@pytest.mark.synthetic
def test_ph_harness_per_region_two_circles() -> None:
    """Per-region assembly returns one report per component with b1=1 each."""
    c0 = _clean_circle(seed=1)
    c1 = _clean_circle(seed=2) + np.array([5.0, 0.0])
    sigma = 0.4
    reports = per_region_topology([c0, c1], sigma, reading="lifetime", max_dim=1)
    assert len(reports) == 2
    for rep in reports:
        assert rep.reading == "lifetime"
        assert rep.n_points == c0.shape[0]
        assert rep.betti[1] == 1
        assert rep.filtration_radius == pytest.approx(FILTRATION_MULTIPLIER * sigma)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_ph_harness_fixed_threshold_mode_and_label_split() -> None:
    """fixed_threshold reading + label split helper stay wired for SI default."""
    c0 = _clean_circle(seed=3)
    c1 = _clean_circle(seed=4) + np.array([0.0, 5.0])
    all_pts = np.vstack([c0, c1])
    labels = np.array([0] * len(c0) + [1] * len(c1))
    regions = extract_region_node_positions(all_pts, labels)
    assert len(regions) == 2

    tau_star = 0.16
    sigma = sigma_star_from_tau(tau_star)
    betti = region_betti_numbers(
        regions[0], sigma, reading="fixed_threshold", max_dim=1,
    )
    # Clean circle at 1.5*sigma with sigma=0.4 should still see the loop.
    assert betti[0] >= 1
    assert DEFAULT_LIFETIME_FRAC > 0.0
