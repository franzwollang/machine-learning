"""Quantitative tests for the Proteus Stage 1 implementation."""
from __future__ import annotations

import numpy as np

from proteus.datasets import make_circle, make_swiss_roll
from proteus.stage1 import ProteusStage1


def _mean_min_distance(points: np.ndarray, representatives: np.ndarray) -> float:
    """Compute the average distance to the closest representative."""
    deltas = points[:, None, :] - representatives[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    return float(distances.min(axis=1).mean())


def test_stage1_converges_on_circle() -> None:
    rng = np.random.default_rng(7)
    data = make_circle(1200, radius=1.0, noise=0.02, rng=rng)
    tau = float(np.var(data, axis=0).mean())
    stage1 = ProteusStage1(
        max_epochs=35,
        max_nodes=96,
        min_nodes=8,
        prune_after=20,
        rng=np.random.default_rng(13),
    )
    stage1.fit(data, tau=tau)
    nodes = stage1.get_node_positions()
    mean_distance = _mean_min_distance(data, nodes)
    assert stage1.cv_history, "Stage 1 should record at least one CV entry"
    # The circle has radius 1.0; a well-formed scaffold should cover it
    # with <20\% quantisation error.
    assert mean_distance < 0.2
    # Thermal equilibrium is approximated numerically; we require a
    # substantial reduction in the coefficient of variation and a final
    # value well below the initial pass.
    assert stage1.last_cv < 1.1
    assert stage1.last_cv < stage1.cv_history[0]


def test_stage1_recovers_swiss_roll_structure() -> None:
    rng = np.random.default_rng(11)
    data = make_swiss_roll(1500, height=1.0, twists=3.5, noise=0.01, rng=rng)
    tau = float(np.var(data, axis=0).mean())
    stage1 = ProteusStage1(
        max_epochs=45,
        max_nodes=128,
        min_nodes=10,
        prune_after=25,
        rng=np.random.default_rng(23),
    )
    stage1.fit(data, tau=tau)
    nodes = stage1.get_node_positions()
    mean_distance = _mean_min_distance(data, nodes)
    data_scale = float(np.linalg.norm(data, axis=1).mean())
    assert stage1.cv_history, "Stage 1 should record at least one CV entry"
    # Quantitative bound: the learned scaffold should approximate the
    # manifold with an average nearest-node distance well below 10% of the
    # dataset's characteristic scale.
    assert mean_distance / data_scale < 0.1
    # Splitting should expand the mesh beyond the initial seed nodes.
    assert nodes.shape[0] > stage1.min_nodes
    assert stage1.last_cv < 0.8
    assert stage1.last_cv < stage1.cv_history[0]
