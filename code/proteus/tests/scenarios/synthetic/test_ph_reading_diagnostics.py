"""Reading-mode diagnostics for Stage-2 topology recovery (OPEN_ISSUES #41).

Compares ``fixed_threshold`` vs ``lifetime`` Vietoris--Rips readings and
documents why tissue/signal filtering (per-region assembly) is required.

These are *evidence-gathering* smokes. They do **not** flip
``test_nested_spheres_topology``, ``test_linked_tori_betti_numbers``, or the
circle ``b1=1`` tissue-scaffold target — those stay ``@awaiting`` until green
on real fitted regions.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.datasets.synthetic.tissue import append_uniform_tissue
from tests.metrics.persistent_homology import (
    DEFAULT_LIFETIME_FRAC,
    FILTRATION_MULTIPLIER,
    compare_readings,
    extract_region_node_positions,
    region_betti_numbers,
)


def _clean_circle(n: int = 64, radius: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    jitter = 1e-4 * rng.normal(size=(n, 2))
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)]) + jitter


def _clean_torus_grid(
    n_theta: int = 24,
    n_phi: int = 12,
    major: float = 2.0,
    minor: float = 0.5,
) -> np.ndarray:
    """Regular torus grid (diagnostic geometry; not a fitted scaffold)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    x = (major + minor * np.cos(ph)) * np.cos(th)
    y = (major + minor * np.cos(ph)) * np.sin(th)
    z = minor * np.sin(ph)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


@pytest.mark.scenario
@pytest.mark.synthetic
def test_compare_readings_clean_circle_both_recover_b1() -> None:
    """On clean geometry both readings recover the circle loop."""
    pts = _clean_circle()
    sigma = 0.4
    cmp = compare_readings(pts, sigma, max_dim=1, lifetime_frac=0.25)
    assert cmp.n_points == pts.shape[0]
    assert cmp.filtration_radius == pytest.approx(FILTRATION_MULTIPLIER * sigma)
    assert cmp.fixed_threshold[0] == 1
    assert cmp.fixed_threshold[1] == 1
    assert cmp.lifetime[0] == 1
    assert cmp.lifetime[1] == 1


@pytest.mark.scenario
@pytest.mark.synthetic
def test_tissue_pollution_requires_signal_filter_not_lifetime_alone() -> None:
    """Whole-scaffold PH is polluted; signal-label filter recovers b1=1.

    Measured finding for #41 item 2: at tissue_fraction≈0.2, whole-cloud
    ``fixed_threshold`` births spurious H1 (b1>1). Lifetime on the *same*
    whole cloud does not restore clean Betti — it can inflate b0 via short
    essential/long bars from tissue. Filtering to signal labels (stand-in for
    per-accepted-region / non-tissue nodes) restores b1=1 under both readings.
    """
    signal = _clean_circle(seed=7)
    labels = np.zeros(len(signal), dtype=int)
    polluted, plabels, meta = append_uniform_tissue(
        signal,
        labels,
        rng=np.random.default_rng(11),
        tissue_fraction=0.2,
    )
    assert meta["tissue_count"] > 0

    sigma = 0.4
    whole = compare_readings(polluted, sigma, max_dim=1, lifetime_frac=DEFAULT_LIFETIME_FRAC)
    # Spurious topology on the tissue-polluted whole cloud.
    assert whole.fixed_threshold[1] > 1
    assert whole.lifetime[0] > 1  # lifetime alone does not clean tissue

    signal_only = extract_region_node_positions(
        polluted, plabels, include_labels=[0],
    )[0]
    filtered = compare_readings(
        signal_only, sigma, max_dim=1, lifetime_frac=DEFAULT_LIFETIME_FRAC,
    )
    assert filtered.fixed_threshold == (1, 1)
    assert filtered.lifetime == (1, 1)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_clean_torus_lifetime_recovers_two_loops() -> None:
    """Clean torus grid: lifetime reading recovers b0=1, b1=2 (diagnostic).

    Does not touch linked-tori fitted-scaffold ``@awaiting`` assertions.
    Grid + ``sigma≈1.0`` / ``lifetime_frac=0.5`` is a measured operating
    window where both loops persist; random sparse samples overcount H0
    essentials under a finite VR radius.
    """
    pts = _clean_torus_grid()
    sigma = 1.0
    betti = region_betti_numbers(
        pts, sigma, reading="lifetime", max_dim=1, lifetime_frac=0.5,
    )
    assert betti[0] == 1
    assert betti[1] == 2
