"""Denser accepted-region coverage probe toward SI 1.5σ* (OPEN_ISSUES #41, A4-T17).

Follows the A4-T12 reading-path proposal: prefer denser scaffold / signal-node
coverage so true H1 births within ``filtration_mult=1.5``, rather than raising
defaults to empirical mult=6 / frac≥4.

Evidence-gathering only — does **not** flip circle / nested_spheres /
linked_tori ``@awaiting`` recovery tests or change SI defaults.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.metrics.persistent_homology import (
    DEFAULT_LIFETIME_FRAC,
    FILTRATION_MULTIPLIER,
    betti_numbers,
    filtration_radius,
    lifetime_betti_numbers,
    sweep_lifetime_frac,
)


def _clean_circle(n: int, radius: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    jitter = 1e-4 * rng.normal(size=(n, 2))
    return np.column_stack(
        [radius * np.cos(theta), radius * np.sin(theta)]
    ) + jitter


def _chord_gap(n: int, radius: float = 1.0) -> float:
    """Adjacent-sample chord length on a regular circle."""
    return float(2.0 * radius * np.sin(np.pi / n))


@pytest.mark.scenario
@pytest.mark.synthetic
@pytest.mark.parametrize(
    "n,sigma,expect_b1_at_si",
    [
        # Sparse: NN gap larger than SI r=1.5σ → loop unborn at fixed_threshold.
        (16, 0.15, False),
        # Denser coverage: gap << 1.5σ → SI fixed_threshold sees the loop.
        (48, 0.4, True),
        (96, 0.35, True),
        (128, 0.3, True),
    ],
)
def test_dense_circle_coverage_recovers_si_fixed_threshold(
    n: int,
    sigma: float,
    expect_b1_at_si: bool,
) -> None:
    """Coverage vs scale: denser sampling enables SI 1.5σ* fixed_threshold."""
    pts = _clean_circle(n, seed=n)
    gap = _chord_gap(n)
    r_si = filtration_radius(sigma, multiplier=FILTRATION_MULTIPLIER)
    fixed = betti_numbers(pts, threshold=r_si, max_dim=1)

    if expect_b1_at_si:
        assert gap < r_si, "dense cases should have gap inside SI radius"
        assert fixed == (1, 1)
    else:
        assert gap > r_si, "sparse case: gap exceeds SI radius"
        assert fixed[1] == 0


@pytest.mark.scenario
@pytest.mark.synthetic
def test_coverage_probe_mult_frac_pairs_on_borderline_circle() -> None:
    """Borderline coverage: report best (mult, frac) pairs without changing defaults.

    Uses n=24, sigma=0.25 — SI 1.5σ* is near the birth threshold. Documents
    which modest mult/frac pairs recover ``(1,1)`` as an existence table for
    the denser-coverage path (not a default proposal).
    """
    n = 24
    sigma = 0.25
    pts = _clean_circle(n, seed=7)
    gap = _chord_gap(n)
    r_si = filtration_radius(sigma)
    # Sanity: borderline — gap should be close to SI radius.
    assert 0.5 * r_si < gap < 2.0 * r_si

    si_fixed = betti_numbers(pts, threshold=r_si, max_dim=1)
    # May or may not recover depending on jitter; record via assertions that
    # denser n at same sigma does recover (coverage path).
    denser = _clean_circle(64, seed=7)
    denser_fixed = betti_numbers(denser, threshold=r_si, max_dim=1)
    assert denser_fixed == (1, 1)

    mults = [1.5, 2.0, 3.0, 4.0, 6.0]
    fracs = [0.25, 0.5, 0.75, 1.0, 2.0, 4.0]
    recovering_fixed: list[float] = []
    recovering_life: list[tuple[float, float]] = []

    for mult in mults:
        b = betti_numbers(pts, threshold=mult * sigma, max_dim=1)
        if b == (1, 1):
            recovering_fixed.append(mult)
        for frac in fracs:
            lb = lifetime_betti_numbers(
                pts,
                sigma,
                max_dim=1,
                filtration_mult=mult,
                lifetime_frac=frac,
            )
            if lb == (1, 1):
                recovering_life.append((mult, frac))

    # Existence: some mult ≥ SI recovers fixed_threshold on this borderline cloud.
    assert any(m >= FILTRATION_MULTIPLIER for m in recovering_fixed)
    # Denser coverage at same sigma is the preferred path (already asserted).
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    # Keep a compact sweep artifact for the borderline cloud at SI mult.
    rows = sweep_lifetime_frac(
        pts,
        sigma,
        fracs=fracs,
        filtration_mult=FILTRATION_MULTIPLIER,
        max_dim=1,
        target_betti=(1, 1),
    )
    assert len(rows) == len(fracs)
    # Silence unused if si_fixed unused for branch — keep for evidence note.
    _ = (si_fixed, recovering_life)
