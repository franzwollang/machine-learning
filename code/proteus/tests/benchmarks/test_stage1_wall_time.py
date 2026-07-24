"""Stage 1 scaffold construction wall-time benchmark."""

from __future__ import annotations

import pytest

from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.circles import make_circle
from tests.datasets.synthetic.hierarchical_gaussian import make_hierarchical_gaussian
from tests.harness.budgets import load_budgets
from tests.harness.stage1_scenario_metrics import run_fixed_tau_stable_and_report


@pytest.mark.benchmark
def test_stage1_wall_time_small() -> None:
    """Fixed-tau scaffold on N~1200 must complete within small budget."""

    budget = load_budgets("small")["stage1_wall_time"]
    dataset = make_circle(
        n_samples=1200,
        radius=1.0,
        noise=0.02,
        extrusion_dim=2,
        seed=21,
    )
    data = dataset.points
    gt = dataset.ground_truth
    tau = gt.expected_tau
    assert tau is not None

    train = run_fixed_tau_stable_and_report(
        data,
        dim=gt.ambient_dim,
        tau=float(tau),
        stabilization=StabilizationConfig(
            min_equilibrium_epochs=3,
            max_epochs=15,
        ),
        k=8,
        min_nodes=4,
        max_nodes=128,
        n_seeds=8,
        prune_after=10,
        cluster=False,
    )
    assert train.wall_seconds < float(budget), (
        f"wall {train.wall_seconds:.2f}s exceeds budget {budget}s"
    )


@pytest.mark.benchmark
@pytest.mark.slow
def test_stage1_wall_time_medium() -> None:
    """Fixed-tau hierarchy subsample must complete within medium budget."""

    budget = load_budgets("medium")["stage1_wall_time"]
    dataset = make_hierarchical_gaussian(
        children_per_coarse=2,
        n_samples=8000,
        ambient_dim=4,
        seed=3,
    )
    data = dataset.points
    tau = dataset.ground_truth.expected_tau
    assert tau is not None

    train = run_fixed_tau_stable_and_report(
        data,
        dim=4,
        tau=float(tau),
        stabilization=StabilizationConfig(
            min_equilibrium_epochs=2,
            max_epochs=12,
        ),
        k=8,
        min_nodes=8,
        max_nodes=256,
        n_seeds=16,
        prune_after=10,
        cluster=False,
    )
    assert train.wall_seconds < float(budget), (
        f"wall {train.wall_seconds:.2f}s exceeds budget {budget}s"
    )
