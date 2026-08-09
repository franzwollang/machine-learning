"""Per-region PH runner prototype for nested_spheres / linked_tori (#41, A4-T16).

Scaffolds ``run_per_region_ph`` against clean geometric stand-ins for the
scenario recovery path. Fitted Stage-1 recovery assertions stay ``@awaiting``
/ may remain xfail — this file maximizes useful harness wiring without
weakening those targets.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.metrics.persistent_homology import (
    DEFAULT_LIFETIME_FRAC,
    FILTRATION_MULTIPLIER,
    run_per_region_ph,
)
from tests.scenarios.synthetic.test_ph_nested_spheres_clean_shells import (
    _nested_fibonacci_shells,
)
from tests.scenarios.synthetic.test_ph_reading_diagnostics import (
    _clean_torus_grid,
)


def _two_clean_tori(
    n_theta: int = 20,
    n_phi: int = 10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Two separated torus grids with labels 1/2 (linked topology not required)."""
    t0 = _clean_torus_grid(n_theta=n_theta, n_phi=n_phi, major=2.0, minor=0.5)
    t1 = _clean_torus_grid(n_theta=n_theta, n_phi=n_phi, major=2.0, minor=0.5)
    # Translate second torus so clouds do not interleave for this scaffold test.
    t1 = t1 + np.array([8.0, 0.0, 0.0])
    points = np.vstack([t0, t1])
    labels = np.array([1] * len(t0) + [2] * len(t1), dtype=int)
    # Tube-scale sigma: large enough to birth torus loops on the grid.
    sigma = 0.55
    return points, labels, sigma


@pytest.mark.scenario
@pytest.mark.synthetic
def test_run_per_region_ph_nested_spheres_clean_shells() -> None:
    """Clean nested Fibonacci shells: runner reports ``(1,0,1)`` per shell."""
    points, labels, sigmas = _nested_fibonacci_shells()
    result = run_per_region_ph(
        points,
        labels,
        sigmas,
        scenario="nested_spheres_clean_shells",
        include_labels=[1, 2],
        reading="fixed_threshold",
        max_dim=2,
        expected_betti=(1, 0, 1),
    )
    assert result.scenario == "nested_spheres_clean_shells"
    assert result.reading == "fixed_threshold"
    assert result.filtration_mult == FILTRATION_MULTIPLIER
    assert [r.region_id for r in result.reports] == [1, 2]
    assert result.all_match is True
    for rep in result.reports:
        assert rep.betti == (1, 0, 1)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_run_per_region_ph_nested_spheres_lifetime_needs_frac() -> None:
    """Modest-n shells: SI lifetime_frac fails match; 0.75 succeeds."""
    points, labels, sigmas = _nested_fibonacci_shells()
    si = run_per_region_ph(
        points,
        labels,
        sigmas,
        scenario="nested_spheres_clean_shells",
        include_labels=[1, 2],
        reading="lifetime",
        max_dim=2,
        lifetime_frac=DEFAULT_LIFETIME_FRAC,
        expected_betti=(1, 0, 1),
    )
    assert si.all_match is False

    ok = run_per_region_ph(
        points,
        labels,
        sigmas,
        scenario="nested_spheres_clean_shells",
        include_labels=[1, 2],
        reading="lifetime",
        max_dim=2,
        lifetime_frac=0.75,
        expected_betti=(1, 0, 1),
    )
    assert ok.all_match is True


@pytest.mark.scenario
@pytest.mark.synthetic
def test_run_per_region_ph_linked_tori_clean_grids() -> None:
    """Clean separated torus grids: each region recovers ``b1 >= 2``.

    Scaffold for ``test_linked_tori_betti_numbers`` intended path. Does **not**
    flip that ``@awaiting`` test — geometry here is a regular grid, not a
    fitted Stage-1 scaffold on interlocking tori.
    """
    points, labels, sigma = _two_clean_tori()
    result = run_per_region_ph(
        points,
        labels,
        sigma,
        scenario="linked_tori_clean_grids",
        include_labels=[1, 2],
        reading="lifetime",
        max_dim=2,
        lifetime_frac=0.25,
        expected_betti=None,  # b2 may vary on coarse grids; check b1 below
    )
    assert result.scenario == "linked_tori_clean_grids"
    assert result.all_match is None
    assert len(result.reports) == 2
    for rep in result.reports:
        assert rep.n_points == points.shape[0] // 2
        assert rep.betti[0] == 1
        assert rep.betti[1] >= 2


@pytest.mark.scenario
@pytest.mark.synthetic
def test_run_per_region_ph_empty_include_yields_no_match_false() -> None:
    """Runner with expected_betti on empty selection is a hard no-match."""
    points = np.zeros((0, 3))
    labels = np.zeros((0,), dtype=int)
    result = run_per_region_ph(
        points,
        labels,
        0.4,
        scenario="empty",
        include_labels=[],
        expected_betti=(1, 0, 1),
        max_dim=2,
    )
    assert result.reports == ()
    assert result.all_match is False
