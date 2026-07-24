"""Stabilized Stage 1 scaffold on a noisy circle."""

from __future__ import annotations

import numpy as np

from proteus.stage1 import Stage1Scaffold
from proteus.stage1.stabilization import StabilizationConfig, cv_threshold
from tests.datasets.synthetic.circles import make_circle
from tests.metrics.reconstruction import mean_min_distance


def _run_circle_with_expected_tau() -> tuple[
    Stage1Scaffold,
    dict[str, list[float]],
    np.ndarray,
    int,
    int,
]:
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
    lower = gt.max_ent_node_lower(multiplier=0.5)
    upper = gt.max_ent_node_upper(multiplier=2.0)
    scaffold = Stage1Scaffold(
        dim=gt.ambient_dim,
        tau=tau,
        k=8,
        min_nodes=4,
        max_nodes=upper + 1,
        prune_after=10,
        ann_backend="naive",
        rng=np.random.default_rng(77),
    )
    scaffold.init_from(data, n_seeds=8)
    history = scaffold.run_until_stable(
        data,
        StabilizationConfig(min_equilibrium_epochs=3, max_epochs=20),
    )
    return scaffold, history, data, lower, upper


def test_circle_grows_and_covers_with_expected_tau() -> None:
    scaffold, history, data, lower, upper = _run_circle_with_expected_tau()
    assert lower <= len(scaffold.nodes) <= upper + 1
    # Exact faded tissue keeps a slightly larger halo than the earlier append
    # model, so reconstruction tolerance needs a small bump.
    assert mean_min_distance(data, scaffold.node_positions()) < 0.11
    assert len(history["cv"]) <= 20


def test_circle_reaches_variance_cv_threshold() -> None:
    scaffold, history, _, _, _ = _run_circle_with_expected_tau()
    tol = cv_threshold(scaffold.k)
    assert history["cv"][-1] < tol
