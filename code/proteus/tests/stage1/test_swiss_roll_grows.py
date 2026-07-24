"""Stabilized Stage 1 scaffold on a noisy swiss roll."""

from __future__ import annotations

import numpy as np

from proteus.stage1 import Stage1Scaffold
from proteus.stage1.stabilization import StabilizationConfig, cv_threshold
from tests.datasets.synthetic.swiss_roll import make_swiss_roll
from tests.metrics.reconstruction import mean_min_distance


def _run_swiss_roll_with_expected_tau() -> tuple[
    Stage1Scaffold,
    dict[str, list[float]],
    np.ndarray,
    int,
    int,
]:
    dataset = make_swiss_roll(
        n_samples=1600,
        height=1.0,
        twists=3.5,
        noise=0.01,
        extrusion_dim=1,
        seed=22,
    )
    data = dataset.points
    gt = dataset.ground_truth
    tau = gt.expected_tau
    assert tau is not None
    lower = gt.max_ent_node_lower(multiplier=0.5)
    upper = gt.max_ent_node_upper(multiplier=2.0)
    scaffold = Stage1Scaffold(
        dim=gt.ambient_dim,
        tau=tau,
        k=8,
        min_nodes=8,
        max_nodes=upper + 1,
        prune_after=10,
        ann_backend="naive",
        rng=np.random.default_rng(88),
    )
    scaffold.init_from(data, n_seeds=12)
    history = scaffold.run_until_stable(
        data,
        StabilizationConfig(
            min_equilibrium_epochs=3,
            max_epochs=20,
        ),
    )
    return scaffold, history, data, lower, upper


def test_swiss_roll_grows_and_covers_with_expected_tau() -> None:
    scaffold, history, data, lower, upper = _run_swiss_roll_with_expected_tau()
    data_diameter = float(np.linalg.norm(data.max(axis=0) - data.min(axis=0)))
    assert lower <= len(scaffold.nodes) <= upper
    assert (
        mean_min_distance(data, scaffold.node_positions())
        < 0.08 * data_diameter
    )
    assert len(history["cv"]) <= 20


def test_swiss_roll_reaches_variance_cv_threshold() -> None:
    scaffold, history, _, _, _ = _run_swiss_roll_with_expected_tau()
    tol = cv_threshold(scaffold.k)
    assert history["cv"][-1] < tol
