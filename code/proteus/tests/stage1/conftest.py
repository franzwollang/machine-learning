"""Shared fixtures for Stage 1 integration tests."""

from __future__ import annotations

import numpy as np
import pytest

from proteus.stage1 import Stage1Scaffold
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.circles import make_circle


@pytest.fixture(scope="module")
def converged_circle_scaffold() -> tuple[Stage1Scaffold, np.ndarray]:
    """A converged circle scaffold for reuse across tests in a module."""

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
    upper = gt.max_ent_node_upper(2.0)

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
    scaffold.run_until_stable(
        data,
        StabilizationConfig(min_equilibrium_epochs=3, max_epochs=20),
    )
    return scaffold, data
