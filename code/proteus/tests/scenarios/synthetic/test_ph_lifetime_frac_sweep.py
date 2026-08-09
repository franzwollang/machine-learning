"""lifetime_frac sweep harness (OPEN_ISSUES #41, A4-T15).

Parametrized / table-artifact evidence for clean Fibonacci shells and
fitted-circle signal nodes. Does **not** flip nested_spheres / linked_tori /
circle tissue ``@awaiting`` recovery tests, and does **not** change SI
``FILTRATION_MULTIPLIER=1.5`` or ``DEFAULT_LIFETIME_FRAC=0.5``.
"""
from __future__ import annotations

import numpy as np
import pytest

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.circles import make_circle
from tests.metrics.persistent_homology import (
    DEFAULT_LIFETIME_FRAC,
    FILTRATION_MULTIPLIER,
    format_lifetime_frac_sweep_table,
    nearest_data_labels,
    sigma_star_from_tau,
    sweep_lifetime_frac,
    sweep_lifetime_frac_per_region,
)
from tests.scenarios.synthetic.test_ph_nested_spheres_clean_shells import (
    _nested_fibonacci_shells,
    fibonacci_sphere,
)

# Shared frac grid covering SI default, modest-n recovery, and existence proofs.
SWEEP_FRACS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 2.0, 4.0)


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
    node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
    signal = pos[node_labels == 0]
    assert signal.shape[0] >= 8
    return signal, sigma


@pytest.mark.scenario
@pytest.mark.synthetic
@pytest.mark.parametrize("frac", SWEEP_FRACS)
def test_clean_single_shell_lifetime_frac_parametrized(frac: float) -> None:
    """Dense Fibonacci shell: SI-window fracs recover ``(1,0,1)``."""
    pts = fibonacci_sphere(400, radius=1.0, seed=0)
    sigma = 0.4
    rows = sweep_lifetime_frac(
        pts,
        sigma,
        fracs=[frac],
        max_dim=2,
        target_betti=(1, 0, 1),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.n_points == 400
    if frac < 0.5:
        # Too-aggressive floor: many short H0 bars survive → b0 inflate.
        assert row.betti[2] == 1
        assert row.matches_target is False
    else:
        assert row.matches_target is True
        assert row.betti == (1, 0, 1)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_shells_lifetime_frac_sweep_table() -> None:
    """Modest-n nested shells: table marks frac≥0.75 as target match."""
    points, labels, sigmas = _nested_fibonacci_shells()
    rows = sweep_lifetime_frac_per_region(
        points,
        labels,
        sigmas,
        fracs=SWEEP_FRACS,
        include_labels=[1, 2],
        max_dim=2,
        target_betti=(1, 0, 1),
    )
    assert len(rows) == len(SWEEP_FRACS) * 2
    table = format_lifetime_frac_sweep_table(rows)
    assert "region_id" in table.splitlines()[0]

    by_frac: dict[float, list] = {}
    for row in rows:
        by_frac.setdefault(row.lifetime_frac, []).append(row)

    for row in by_frac[DEFAULT_LIFETIME_FRAC]:
        assert row.betti[2] == 1
        assert row.betti[1] == 0
        assert row.matches_target is False  # b0 inflate at modest n

    for frac in (0.75, 1.0, 2.0, 4.0):
        for row in by_frac[frac]:
            assert row.matches_target is True
            assert row.betti == (1, 0, 1)

    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5


@pytest.mark.scenario
@pytest.mark.synthetic
def test_fitted_circle_lifetime_frac_sweep_table(fitted_circle_signal_nodes) -> None:
    """Fitted-circle: SI mult never recovers; mult=6 needs frac≥4."""
    signal, sigma = fitted_circle_signal_nodes

    si_rows = sweep_lifetime_frac(
        signal,
        sigma,
        fracs=SWEEP_FRACS,
        filtration_mult=FILTRATION_MULTIPLIER,
        max_dim=1,
        target_betti=(1, 1),
    )
    assert all(row.betti[1] == 0 for row in si_rows)
    assert all(row.matches_target is False for row in si_rows)

    wide_rows = sweep_lifetime_frac(
        signal,
        sigma,
        fracs=SWEEP_FRACS,
        filtration_mult=6.0,
        max_dim=1,
        target_betti=(1, 1),
    )
    by_frac = {row.lifetime_frac: row for row in wide_rows}
    # Default frac sees the loop but inflates b0.
    assert by_frac[DEFAULT_LIFETIME_FRAC].betti[1] == 1
    assert by_frac[DEFAULT_LIFETIME_FRAC].matches_target is False
    # Existence proof only — not a defended default.
    assert by_frac[4.0].matches_target is True
    assert by_frac[4.0].betti == (1, 1)

    table = format_lifetime_frac_sweep_table(wide_rows)
    assert "4" in table
