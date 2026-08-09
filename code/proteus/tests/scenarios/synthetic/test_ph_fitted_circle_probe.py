"""Fitted-circle scaffold PH probe (OPEN_ISSUES #41, A4-T7).

Evidence-gathering only: runs Vietoris--Rips readings on a real Stage-1
circle ``scaffold_at_star`` via ``topology_from_accepted_regions``. Does
**not** flip circle ``b1=1`` / nested_spheres / linked_tori ``@awaiting``
recovery tests.

Measured finding (seed=21 circle recipe matching flag-complex scaffold):
  * Whole / single accepted cluster at SI ``1.5 σ*``: disconnected,
    ``b1=0`` (filtration too fine vs scaffold NN gaps).
  * Lifetime at the same SI radius inflates ``b0`` and still yields ``b1=0``.
  * NN-to-data signal filter (drop tissue-labeled nodes) recovers ``b1=1``
    only at a *larger* fixed threshold (``~8 σ*`` here) — not at SI default.
Keep recovery assertions awaiting until a defended reading is green.
"""
from __future__ import annotations

import numpy as np
import pytest

from proteus.stage1.clustering import run_clustering
from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.circles import make_circle
from tests.metrics.persistent_homology import (
    FILTRATION_MULTIPLIER,
    betti_numbers,
    compare_readings,
    filtration_radius,
    nearest_data_labels,
    sigma_star_from_tau,
    topology_from_accepted_regions,
)


def _fit_circle_scaffold():
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
    return dataset, result


@pytest.mark.scenario
@pytest.mark.synthetic
def test_fitted_circle_si_default_does_not_recover_b1() -> None:
    """SI ``1.5 σ*`` on fitted accepted-region nodes does not yet yield b1=1."""
    dataset, result = _fit_circle_scaffold()
    scaffold = result.scaffold_at_star
    pos = scaffold.node_positions()
    sigma = sigma_star_from_tau(result.tau_star)
    assert filtration_radius(sigma) == pytest.approx(
        FILTRATION_MULTIPLIER * sigma,
    )

    cr = run_clustering(scaffold)
    # Circle at tau* is typically one accepted partition (tissue may still
    # pollute node positions inside that single label).
    assert cr.n_clusters >= 1

    whole = compare_readings(pos, sigma, max_dim=1)
    # Documented failure mode at SI default — do not weaken this evidence.
    assert whole.fixed_threshold[1] == 0
    assert whole.lifetime[1] == 0
    assert whole.lifetime[0] > whole.fixed_threshold[0]

    reports = topology_from_accepted_regions(
        pos, cr.labels, sigma, reading="lifetime", max_dim=1,
    )
    assert len(reports) == cr.n_clusters
    assert all(r.betti[1] == 0 for r in reports)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_fitted_circle_signal_filter_needs_larger_than_si_radius() -> None:
    """NN signal-label filter recovers b1=1 only beyond SI 1.5 σ* (probe)."""
    dataset, result = _fit_circle_scaffold()
    pos = result.scaffold_at_star.node_positions()
    sigma = sigma_star_from_tau(result.tau_star)

    node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
    assert (node_labels == 0).sum() >= 8  # some signal-associated nodes

    si_reports = topology_from_accepted_regions(
        pos,
        node_labels,
        sigma,
        include_labels=[0],
        reading="fixed_threshold",
        max_dim=1,
    )
    assert si_reports[0].betti[1] == 0  # SI default still fails on signal nodes

    signal_pts = pos[node_labels == 0]
    # Existence proof that *some* larger fixed threshold sees the loop —
    # not a license to flip awaiting tests or change SI 1.5 default.
    betti_wide = betti_numbers(signal_pts, threshold=8.0 * sigma, max_dim=1)
    assert betti_wide[0] == 1
    assert betti_wide[1] == 1
