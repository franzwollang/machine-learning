"""Denser fitted-scaffold coverage experiment toward SI 1.5σ* (#41, A4-T19).

Varies ``ScaleSearchConfig.max_nodes`` on the circle ``scaffold_at_star``
recipe (same seeds as ``test_ph_fitted_circle_probe``). Evidence-gathering
only — does **not** flip circle / nested_spheres / linked_tori ``@awaiting``
recovery tests or change SI ``FILTRATION_MULTIPLIER=1.5``.

Measured finding (seed=21 data / seed=77 scale; NN signal-label filter):
  * ``max_nodes=64`` (default cap path): ``n_sig≈40``, SI fixed → ``(4,0)``
  * ``max_nodes=128``: ``n_sig≈80``, SI fixed → ``(1,1)`` recovered
  * ``max_nodes=256``: ``n_sig≈134``, SI fixed → ``(1,1)`` recovered

Affirms the A4-T12/T17 denser-coverage reading path on *fitted* scaffolds
(prefer denser nodes over raising filtration_mult to 6).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.circles import make_circle
from tests.metrics.persistent_homology import (
    FILTRATION_MULTIPLIER,
    betti_numbers,
    filtration_radius,
    nearest_data_labels,
    sigma_star_from_tau,
)


@dataclass(frozen=True)
class FittedCoverageRow:
    max_nodes: int
    n_nodes: int
    n_signal: int
    sigma_star: float
    median_nn_gap: float
    r_si: float
    betti_si: tuple[int, ...]
    recovers_si_b1: bool


def _fit_circle_coverage(max_nodes: int) -> FittedCoverageRow:
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
        max_nodes=int(max_nodes),
        ann_backend="naive",
        stabilization=StabilizationConfig(
            min_equilibrium_epochs=3, max_epochs=15,
        ),
        seed=77,
    )
    result = run_scale_search(dataset.points, dim=gt.ambient_dim, config=config)
    pos = result.scaffold_at_star.node_positions()
    sigma = sigma_star_from_tau(result.tau_star)
    node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
    signal = pos[node_labels == 0]
    if signal.shape[0] >= 2:
        nn = NearestNeighbors(n_neighbors=2).fit(signal)
        dists, _ = nn.kneighbors(signal)
        med_gap = float(np.median(dists[:, 1]))
    else:
        med_gap = float("nan")
    r_si = filtration_radius(sigma, multiplier=FILTRATION_MULTIPLIER)
    betti = betti_numbers(signal, threshold=r_si, max_dim=1)
    return FittedCoverageRow(
        max_nodes=int(max_nodes),
        n_nodes=int(pos.shape[0]),
        n_signal=int(signal.shape[0]),
        sigma_star=float(sigma),
        median_nn_gap=med_gap,
        r_si=float(r_si),
        betti_si=tuple(int(x) for x in betti),
        recovers_si_b1=bool(betti == (1, 1)),
    )


@pytest.fixture(scope="module")
def fitted_coverage_table() -> dict[int, FittedCoverageRow]:
    """Fit once per ``max_nodes`` setting for the coverage table."""
    return {m: _fit_circle_coverage(m) for m in (64, 128, 256)}


@pytest.mark.scenario
@pytest.mark.synthetic
def test_fitted_coverage_table_node_sigma_pairs(fitted_coverage_table) -> None:
    """Report node/σ pairs: denser ``max_nodes`` enables SI 1.5σ* on circle."""
    rows = fitted_coverage_table
    assert set(rows) == {64, 128, 256}

    # Baseline default-scale cap path still fails SI fixed_threshold.
    assert rows[64].recovers_si_b1 is False
    assert rows[64].betti_si[1] == 0
    assert rows[64].n_signal < rows[128].n_signal

    # Denser scaffolds recover SI (1,1) — coverage path, not mult=6.
    assert rows[128].recovers_si_b1 is True
    assert rows[128].betti_si == (1, 1)
    assert rows[256].recovers_si_b1 is True
    assert rows[256].betti_si == (1, 1)

    # Gap vs radius: recovered rows keep median NN gap inside SI radius.
    for m in (128, 256):
        assert rows[m].median_nn_gap < rows[m].r_si


@pytest.mark.scenario
@pytest.mark.synthetic
def test_fitted_coverage_table_is_monotone_in_signal_nodes(
    fitted_coverage_table,
) -> None:
    """Higher ``max_nodes`` yields more signal-associated scaffold nodes."""
    n64 = fitted_coverage_table[64].n_signal
    n128 = fitted_coverage_table[128].n_signal
    n256 = fitted_coverage_table[256].n_signal
    assert n64 < n128 <= n256
