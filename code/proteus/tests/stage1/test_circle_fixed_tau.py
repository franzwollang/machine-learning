"""Fixed-tau Stage 1 scaffold on a noisy circle."""

from __future__ import annotations

import numpy as np

from proteus.stage1 import Stage1Scaffold
from tests.datasets.synthetic.circles import make_circle
from tests.metrics.reconstruction import mean_min_distance


def test_circle_fixed_tau_loop() -> None:
    dataset = make_circle(
        n_samples=1000,
        radius=1.0,
        noise=0.02,
        extrusion_dim=2,
        seed=10,
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
        rng=np.random.default_rng(123),
    )
    scaffold.init_from(data, n_seeds=24)

    epoch_stats = [scaffold.run_epoch(data) for _ in range(6)]
    positions = scaffold.node_positions()

    # The faded circle leaves more low-density tissue mass away from the ring
    # than the earlier append-only background model.
    assert mean_min_distance(data, positions) < 0.16
    assert np.mean([node.hit_count for node in scaffold.nodes]) > 0.0
    fired_epochs = sum(int(stats["deferred_fires"] > 0.0) for stats in epoch_stats)
    assert fired_epochs / len(epoch_stats) >= 0.30
