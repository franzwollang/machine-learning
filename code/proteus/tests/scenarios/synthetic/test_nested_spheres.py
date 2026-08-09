"""Nested-sphere topology scenario."""
from __future__ import annotations

import pytest

from tests.harness.markers import awaiting
from tests.metrics.persistent_homology import (
    format_per_region_ph_diagnostics,
    run_per_region_ph,
)
from tests.scenarios.synthetic.test_ph_nested_spheres_clean_shells import (
    _nested_fibonacci_shells,
)


@awaiting("stage1.controller", si="S2.5")
def test_nested_spheres_two_scales():
    """Scale controller should find two characteristic scales."""
    pytest.fail("Not implemented")


@awaiting("stage2.flag_complex", si="S4.1")
def test_nested_spheres_topology():
    """PH should recover expected Betti numbers per component.

    Intended path (OPEN_ISSUES #41; keep @awaiting until green on fitted regions):
    ``run_per_region_ph`` / ``topology_from_accepted_regions`` on accepted-region
    node positions; expect per-shell ``b0 = 1`` and ``b_{sphere_dim} = 1``.
    Prefer lifetime reading over fixed ``1.5 sigma_star`` if tissue pollution
    births short loops — fitted-circle probe still red at SI default unless
    denser ``max_nodes`` coverage is used (see
    ``test_ph_fitted_coverage_experiment``).

    Scaffolding (A4-T20): clean Fibonacci shells exercise ``run_per_region_ph``
    and emit diagnostics; this test still fails under strict xfail until the
    *fitted* Stage-1 path is green.
    """
    points, labels, sigmas = _nested_fibonacci_shells()
    clean = run_per_region_ph(
        points,
        labels,
        sigmas,
        scenario="nested_spheres_clean_shells_scaffold",
        include_labels=[1, 2],
        reading="fixed_threshold",
        max_dim=2,
        expected_betti=(1, 0, 1),
    )
    diag = format_per_region_ph_diagnostics(clean)
    # Clean-shell harness is green; fitted Stage-1 recovery remains awaiting.
    pytest.fail(
        "awaiting green per-region PH on fitted spheres; "
        f"clean-shell harness all_match={clean.all_match}\n{diag}"
    )
