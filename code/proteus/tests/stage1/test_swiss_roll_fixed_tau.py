"""Fixed-tau Stage 1 scaffold on a noisy swiss roll."""

from __future__ import annotations

import numpy as np

from proteus.stage1 import Stage1Scaffold
from tests.datasets.synthetic.swiss_roll import make_swiss_roll
from tests.metrics.reconstruction import mean_min_distance


def test_swiss_roll_fixed_tau_loop() -> None:
    dataset = make_swiss_roll(
        n_samples=1500,
        height=1.0,
        twists=3.5,
        noise=0.01,
        extrusion_dim=1,
        seed=11,
    )
    data = dataset.points
    gt = dataset.ground_truth
    tau = gt.expected_tau
    assert tau is not None
    scaffold = Stage1Scaffold(
        dim=gt.ambient_dim,
        tau=tau,
        k=8,
        ann_backend="naive",
        rng=np.random.default_rng(456),
    )
    scaffold.init_from(data, n_seeds=48)

    for _ in range(6):
        scaffold.run_epoch(data)

    data_diameter = float(np.linalg.norm(data.max(axis=0) - data.min(axis=0)))
    assert mean_min_distance(data, scaffold.node_positions()) < 0.05 * data_diameter
