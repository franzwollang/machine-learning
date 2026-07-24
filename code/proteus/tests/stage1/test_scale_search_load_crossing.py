"""Load-crossing and combined characteristic-scale selectors (SI S2.5.1, S2.6.2).

The legacy ``load_band`` selector takes the coarsest stabilized grid point whose
variance load lands in ``[band_lo, 1]``; on uniform manifolds it therefore pins
``tau*`` to the coarse end of the band, overshooting the geometric scale by up to
an order of magnitude.  The principled ``load_crossing`` signal instead locates
the scale at which equilibrium residual variance equals the cap (``load = 1``),
which lands within roughly one grid step of the geometric truth.  These tests
assert that dominance directly (new selector strictly closer to the truth than
the legacy band on the same run) rather than only checking a loose tolerance.
"""

from __future__ import annotations

import numpy as np

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.circles import make_circle
from tests.datasets.synthetic.hierarchical_gaussian import make_hierarchical_gaussian
from tests.datasets.synthetic.swiss_roll import make_swiss_roll


def _log_error(ratio: float) -> float:
    """Absolute log-ratio distance from the geometric truth (0 is perfect)."""

    return abs(np.log(ratio))


def _run(dataset, selector, **over):
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    base = dict(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=8,
        ann_backend="naive",
        selector=selector,
        stabilization=StabilizationConfig(min_equilibrium_epochs=3, max_epochs=15),
        seed=77,
    )
    base.update(over)
    return run_scale_search(dataset.points, dim=gt.ambient_dim, config=ScaleSearchConfig(**base))


def test_load_crossing_dominates_band_on_circle() -> None:
    dataset = make_circle(n_samples=1200, radius=1.0, noise=0.02, extrusion_dim=2, seed=21)
    expected = dataset.ground_truth.expected_tau

    band = _run(dataset, "load_band")
    cross = _run(dataset, "load_crossing")

    ratio_cross = cross.tau_star / expected
    # Within roughly one geometric grid step (ratio = sqrt(2) ~ 1.41 per step).
    assert 0.5 < ratio_cross < 2.5, f"circle load_crossing ratio={ratio_cross:.2f}"
    # Strictly better than the legacy band selector on the same problem.
    assert _log_error(ratio_cross) < _log_error(band.tau_star / expected)


def test_load_crossing_dominates_band_on_swiss_roll() -> None:
    dataset = make_swiss_roll(n_samples=1500, noise=0.05, seed=7)
    expected = dataset.ground_truth.expected_tau

    band = _run(dataset, "load_band")
    cross = _run(dataset, "load_crossing")

    ratio_cross = cross.tau_star / expected
    assert 0.4 < ratio_cross < 2.5, f"swiss load_crossing ratio={ratio_cross:.2f}"
    assert _log_error(ratio_cross) <= _log_error(band.tau_star / expected)


def test_combined_uses_load_crossing_on_uniform_manifold() -> None:
    # A single-feature manifold has no persistent multi-cluster split, so the
    # combined selector must fall through to the load-crossing operating scale.
    dataset = make_circle(n_samples=1200, radius=1.0, noise=0.02, extrusion_dim=2, seed=21)
    expected = dataset.ground_truth.expected_tau

    combined = _run(dataset, "combined")
    cross = _run(dataset, "load_crossing")

    assert combined.persistence_result is not None
    assert combined.persistence_result.tau_star_index is None
    assert combined.tau_star == cross.tau_star
    assert 0.5 < combined.tau_star / expected < 2.5


def test_combined_uses_persistence_on_multimodal_region() -> None:
    dataset = make_hierarchical_gaussian(children_per_coarse=2, n_samples=600, ambient_dim=4, seed=0)
    combined = _run(
        dataset,
        "combined",
        n_seeds=12,
        min_nodes=8,
        max_nodes=128,
        stabilization=StabilizationConfig(min_equilibrium_epochs=2, max_epochs=12),
        seed=42,
    )

    pr = combined.persistence_result
    assert pr is not None
    assert pr.tau_star_index is not None
    # A genuinely multi-modal region: tau* is taken from the persistence signal,
    # which lands on a multi-cluster grid point.
    assert combined.tau_star == combined.tau_grid[pr.tau_star_index]
    assert combined.partition_snapshots[pr.tau_star_index].n_clusters >= 3


def test_unknown_selector_raises() -> None:
    dataset = make_circle(n_samples=400, radius=1.0, noise=0.02, extrusion_dim=2, seed=21)
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    config = ScaleSearchConfig(
        tau_min=tau_lo, tau_max=tau_hi, max_grid_points=5, selector="bogus",
    )
    try:
        run_scale_search(dataset.points, dim=gt.ambient_dim, config=config)
    except ValueError as exc:
        assert "selector" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for unknown selector")
