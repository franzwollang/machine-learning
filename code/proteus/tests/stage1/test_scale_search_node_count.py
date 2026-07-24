"""Compensated node-count characteristic-scale signal on scaffolds (SI S2.5.1).

The compensated node-count trace ``C(tau) = N(tau) * tau^{d/2}`` is the intended
*primary* characteristic-scale signal (OPEN_ISSUES #28).  Under the current
scaffold it does not yet plateau cleanly enough to dominate the legacy load-band
selector --- the degree-proxy intrinsic dimension is biased high (#39) and the
node count saturates against ``max_nodes`` at fine scales --- so the
``node_count`` and ``combined`` selectors are wired behind a flag with a safe
fall-back to ``load_band`` while that theory gap is closed.  These tests pin the
plumbing: the signal is computed and recorded, and selecting it never regresses
the returned ``tau_star`` below the legacy path.
"""

from __future__ import annotations

import numpy as np

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.circles import make_circle


def _config(selector: str):
    dataset = make_circle(
        n_samples=800, radius=1.0, noise=0.02, extrusion_dim=2, seed=21,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    config = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=6,
        k=8,
        n_seeds=8,
        ann_backend="naive",
        selector=selector,
        stabilization=StabilizationConfig(min_equilibrium_epochs=3, max_epochs=12),
        seed=77,
    )
    return dataset, config


def test_node_count_selector_records_compensated_trace() -> None:
    dataset, config = _config("node_count")
    gt = dataset.ground_truth
    result = run_scale_search(dataset.points, dim=gt.ambient_dim, config=config)

    assert result.node_count_trace is not None
    assert result.node_count_trace.shape == result.tau_grid.shape
    cr = result.characteristic_scale_result
    assert cr is not None
    assert cr.compensated_trace.shape == result.tau_grid.shape
    # Compensation dimension is a positive per-grid-point estimate.
    assert np.all(cr.d_trace >= 1.0)
    # A positive characteristic scale is always returned (knee or fall-back).
    assert result.tau_star > 0.0


def test_combined_selector_records_both_signals() -> None:
    dataset, config = _config("combined")
    gt = dataset.ground_truth
    result = run_scale_search(dataset.points, dim=gt.ambient_dim, config=config)

    # Combined uses the node-count knee as primary and persistence as a
    # structural cross-check, so both diagnostics are populated.
    assert result.characteristic_scale_result is not None
    assert result.persistence_result is not None
    assert result.partition_snapshots is not None
    assert result.tau_star > 0.0


def test_node_count_selector_matches_legacy_fallback_when_no_knee() -> None:
    # Until the compensated trace plateaus (OPEN_ISSUES #28), the node-count
    # selector must not regress the returned tau* relative to the legacy path.
    dataset, config = _config("node_count")
    gt = dataset.ground_truth
    result = run_scale_search(dataset.points, dim=gt.ambient_dim, config=config)

    legacy_config = ScaleSearchConfig(
        tau_min=config.tau_min,
        tau_max=config.tau_max,
        max_grid_points=config.max_grid_points,
        k=config.k,
        n_seeds=config.n_seeds,
        ann_backend=config.ann_backend,
        selector="load_band",
        stabilization=config.stabilization,
        seed=config.seed,
    )
    legacy = run_scale_search(dataset.points, dim=gt.ambient_dim, config=legacy_config)

    cr = result.characteristic_scale_result
    assert cr is not None
    if cr.knee_index is None:
        assert result.tau_star == legacy.tau_star
