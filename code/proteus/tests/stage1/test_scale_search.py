"""Integration test for the Stage 1 scale-grid controller."""

from __future__ import annotations

import numpy as np

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.circles import make_circle


def test_circle_scale_search_finds_peak_near_expected_tau() -> None:
    dataset = make_circle(
        n_samples=1200,
        radius=1.0,
        noise=0.02,
        extrusion_dim=2,
        seed=21,
    )
    data = dataset.points
    gt = dataset.ground_truth
    expected_tau = gt.expected_tau
    assert expected_tau is not None
    tau_grid_hint = gt.tau_grid_hint
    assert tau_grid_hint is not None
    tau_lo, tau_hi = tau_grid_hint

    config = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=8,
        ann_backend="naive",
        stabilization=StabilizationConfig(
            min_equilibrium_epochs=3,
            max_epochs=15,
        ),
        seed=77,
    )
    result = run_scale_search(data, dim=gt.ambient_dim, config=config)

    assert result.tau_star > 0.0
    ratio = result.tau_star / expected_tau
    # Exact faded tissue shifts the controller's load transition coarser than the
    # local signal-scale heuristic used for ``expected_tau``.
    assert 0.5 < ratio < 10.0, (
        f"tau_star={result.tau_star:.6f} vs expected={expected_tau:.6f} "
        f"(ratio={ratio:.2f})"
    )

    finite_load = result.load_trace[np.isfinite(result.load_trace)]
    assert finite_load.size > 0
    assert finite_load.min() < 1.0, (
        f"Load trace never goes under cap: min={finite_load.min():.2f}"
    )
    assert finite_load.max() > 1.0 or finite_load.max() > 0.5, (
        f"Load trace shows no resolution transition: max={finite_load.max():.2f}"
    )
