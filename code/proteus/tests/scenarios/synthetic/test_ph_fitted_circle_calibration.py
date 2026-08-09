"""Fitted-circle filtration_mult / lifetime_frac calibration (OPEN_ISSUES #41, A4-T9).

Evidence-gathering only on the Stage-1 ``scaffold_at_star`` from the same
circle recipe as ``test_ph_fitted_circle_probe`` (seed=21 / scale seed=77).
Does **not** flip circle ``b1=1`` / nested_spheres / linked_tori ``@awaiting``
tests, and does **not** change SI ``FILTRATION_MULTIPLIER=1.5`` or
``DEFAULT_LIFETIME_FRAC=0.5``.

Measured calibration (NN signal-label filter on scaffold nodes; ``max_dim=1``):
  * ``fixed_threshold``: SI ``mult=1.5`` → ``b1=0``; first recovery of
    ``(b0,b1)=(1,1)`` at ``mult=6``; holds through ``mult∈[6,10]``; fills
    again by ``mult=12`` (``b1=0``).
  * ``lifetime`` at SI ``mult=1.5``: no probed ``lifetime_frac`` recovers
    ``b1=1`` (loop unborn inside ``r_max``).
  * ``lifetime`` recovers ``(1,1)`` only when ``filtration_mult≥6`` *and*
    ``lifetime_frac≥4`` (default ``0.5`` leaves ``b0≫1`` from long-lived
    finite H0 merge bars). Existence proof only — not a defended default.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.circles import make_circle
from tests.metrics.persistent_homology import (
    DEFAULT_LIFETIME_FRAC,
    FILTRATION_MULTIPLIER,
    betti_numbers,
    lifetime_betti_numbers,
    sigma_star_from_tau,
)


@pytest.fixture(scope="module")
def fitted_circle_signal_nodes():
    """Fit circle once; return ``(signal_node_positions, sigma_star)``."""
    dataset = make_circle(
        n_samples=1200, radius=1.0, noise=0.02, extrusion_dim=2, seed=21,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    config = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=8,
        ann_backend="naive",
        stabilization=StabilizationConfig(
            min_equilibrium_epochs=3, max_epochs=15,
        ),
        seed=77,
    )
    result = run_scale_search(dataset.points, dim=gt.ambient_dim, config=config)
    pos = result.scaffold_at_star.node_positions()
    sigma = sigma_star_from_tau(result.tau_star)

    nn = NearestNeighbors(n_neighbors=1).fit(dataset.points)
    _, idx = nn.kneighbors(pos)
    node_labels = np.asarray(dataset.labels)[idx[:, 0]]
    signal = pos[node_labels == 0]
    assert signal.shape[0] >= 8
    return signal, sigma


def _min_fixed_mult_recovering_b1(
    signal: np.ndarray,
    sigma: float,
    mults: list[float],
) -> float | None:
    for mult in mults:
        b = betti_numbers(signal, threshold=mult * sigma, max_dim=1)
        if b[0] == 1 and b[1] == 1:
            return float(mult)
    return None


@pytest.mark.scenario
@pytest.mark.synthetic
def test_fitted_circle_fixed_threshold_min_mult_recovers_b1(
    fitted_circle_signal_nodes,
) -> None:
    """Document min ``filtration_mult`` where fixed_threshold sees b1=1."""
    signal, sigma = fitted_circle_signal_nodes

    si = betti_numbers(
        signal, threshold=FILTRATION_MULTIPLIER * sigma, max_dim=1,
    )
    assert si[1] == 0  # SI default still fails on signal-filtered nodes

    mults = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
    min_ok = _min_fixed_mult_recovering_b1(signal, sigma, mults)
    assert min_ok == pytest.approx(6.0)

    # Window: recover at 6 and 8; filled again by 12.
    assert betti_numbers(signal, threshold=6.0 * sigma, max_dim=1) == (1, 1)
    assert betti_numbers(signal, threshold=8.0 * sigma, max_dim=1) == (1, 1)
    assert betti_numbers(signal, threshold=12.0 * sigma, max_dim=1)[1] == 0


@pytest.mark.scenario
@pytest.mark.synthetic
def test_fitted_circle_lifetime_needs_large_mult_and_frac(
    fitted_circle_signal_nodes,
) -> None:
    """Lifetime recovers (1,1) only with mult≥6 and lifetime_frac≥4 (probe)."""
    signal, sigma = fitted_circle_signal_nodes

    # SI mult + default / nearby fracs: loop unborn → b1=0.
    for frac in (0.25, DEFAULT_LIFETIME_FRAC, 0.75, 1.0, 2.0):
        b = lifetime_betti_numbers(
            signal,
            sigma,
            max_dim=1,
            filtration_mult=FILTRATION_MULTIPLIER,
            lifetime_frac=frac,
        )
        assert b[1] == 0

    # At recovering mult, default frac still inflates b0.
    b_default = lifetime_betti_numbers(
        signal,
        sigma,
        max_dim=1,
        filtration_mult=6.0,
        lifetime_frac=DEFAULT_LIFETIME_FRAC,
    )
    assert b_default[1] == 1
    assert b_default[0] > 1

    # Existence: (mult, frac)=(6, 4) recovers clean (1,1) — not a default claim.
    b_hit = lifetime_betti_numbers(
        signal,
        sigma,
        max_dim=1,
        filtration_mult=6.0,
        lifetime_frac=4.0,
    )
    assert b_hit == (1, 1)

    # Coarser existence points still hold; SI defaults unchanged by this file.
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
