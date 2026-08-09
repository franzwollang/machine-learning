"""Linked-tori topology scenario."""
from __future__ import annotations

import numpy as np
import pytest

from tests.harness.markers import awaiting
from tests.metrics.persistent_homology import (
    format_per_region_ph_diagnostics,
    run_per_region_ph,
)
from tests.scenarios.synthetic.test_ph_reading_diagnostics import (
    _clean_torus_grid,
)


def _two_clean_tori_scaffold(
    n_theta: int = 24,
    n_phi: int = 12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Separated clean torus grids (stand-in for fitted linked_tori regions)."""
    t0 = _clean_torus_grid(n_theta=n_theta, n_phi=n_phi, major=2.0, minor=0.5)
    t1 = _clean_torus_grid(n_theta=n_theta, n_phi=n_phi, major=2.0, minor=0.5)
    t1 = t1 + np.array([8.0, 0.0, 0.0])
    points = np.vstack([t0, t1])
    labels = np.array([1] * len(t0) + [2] * len(t1), dtype=int)
    sigma = 0.55
    return points, labels, sigma


@awaiting("stage1.controller", si="S2.5")
def test_linked_tori_component_separation():
    """AP clustering should identify two components."""
    pytest.fail("Not implemented")


@awaiting("stage2.flag_complex", si="S4.1")
def test_linked_tori_betti_numbers():
    """PH should recover b1>=2 for each torus component.

    Intended path (OPEN_ISSUES #41; keep @awaiting until green on fitted regions):
    ``run_per_region_ph(..., reading='lifetime' or 'fixed_threshold')`` on
    accepted-region node positions (not the lifted-graph flag complex) with
    ``sigma_star = sqrt(tau_star)``. Do not flip by weakening thresholds —
    denser fitted coverage helps circles (``test_ph_fitted_coverage_experiment``)
    but interlocking tori remain unrecovered on Stage-1 scaffolds.

    Scaffolding (A4-T20): clean torus grids exercise ``run_per_region_ph`` and
    emit diagnostics; this test still fails under strict xfail until the
    *fitted* Stage-1 path is green.
    """
    points, labels, sigma = _two_clean_tori_scaffold()
    clean = run_per_region_ph(
        points,
        labels,
        sigma,
        scenario="linked_tori_clean_grids_scaffold",
        include_labels=[1, 2],
        reading="fixed_threshold",
        max_dim=2,
        expected_betti=(1, 2, 1),
    )
    diag = format_per_region_ph_diagnostics(clean)
    pytest.fail(
        "awaiting green per-region PH on fitted tori; "
        f"clean-grid harness all_match={clean.all_match}\n{diag}"
    )
