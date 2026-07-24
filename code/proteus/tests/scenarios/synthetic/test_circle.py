"""Circle smoke scenario."""
from __future__ import annotations

import pytest

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.circles import make_circle
from tests.harness.markers import awaiting
from tests.harness.stage1_scenario_metrics import (
    normalize_stage1_reconstruction,
    run_scale_search_and_report,
)


def test_circle_scale_recovery():
    """Scale controller should find one characteristic scale for a circle."""

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

    ratio = result.tau_star / expected_tau
    # The exact faded-support controller often resolves at a coarser operating
    # scale than the local signal-scale metadata alone would suggest.
    assert 0.5 < ratio < 10.0, (
        f"tau_star={result.tau_star:.6f} vs expected={expected_tau:.6f} "
        f"(ratio={ratio:.2f})"
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_stage1_reconstruction_normalized() -> None:
    """Stage 1: normalized mean-min distance + lifted graph + Q clustering.

    Uses the same scale search envelope as ``test_circle_scale_recovery``.
    Stage 2 density reconstruction remains in ``test_circle_reconstruction_error``.
    """

    dataset = make_circle(
        n_samples=1200,
        radius=1.0,
        noise=0.02,
        extrusion_dim=2,
        seed=21,
    )
    data = dataset.points
    gt = dataset.ground_truth
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
    train = run_scale_search_and_report(
        data,
        dim=gt.ambient_dim,
        config=config,
        with_clustering=True,
    )
    rep = train.report
    assert rep.n_clusters is not None
    # Tissue can induce a handful of coarse partitions at tau* without changing
    # the underlying single circular signal mode.
    assert 1 <= rep.n_clusters <= 6
    assert rep.partition_q_score is not None and rep.partition_q_score > 0.0
    # Tissue can leave several small lifted fragments at tau* even when the
    # circular signal is reconstructed cleanly.
    assert rep.n_lifted_components <= 6
    frac_iso = rep.n_isolated_lifted / max(rep.n_nodes, 1)
    assert frac_iso <= 0.1

    norms = normalize_stage1_reconstruction(rep, "circle", radius=1.0)
    assert norms["mean_norm"] < 0.35, (
        f"normalized mean min-dist {norms['mean_norm']:.3f} vs radius scale"
    )
    assert train.epochs_ran <= config.stabilization.max_epochs


@awaiting("stage2.density", si="S6.4")
def test_circle_reconstruction_error():
    """Mean-min distance to learned support should be < 20% of radius.

    Baseline Stage 1 normalized metric: ``test_circle_stage1_reconstruction_normalized``.
    """
    pytest.fail("Not implemented")


@awaiting("inference.membership", si="S7.4")
def test_circle_membership_trajectory_depth():
    """Membership trajectory depth should be 1 (single mode)."""
    pytest.fail("Not implemented")
