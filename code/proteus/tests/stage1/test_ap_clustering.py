"""Integration tests for Q-score clustering on converged scaffolds."""

from __future__ import annotations

import numpy as np

from proteus.stage1 import Stage1Scaffold
from proteus.stage1.clustering import run_ap_clustering
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.circles import make_circle


def test_circle_clustering_runs_without_error() -> None:
    """Clustering should produce a valid labeling on a converged circle scaffold."""

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

    scaffold = Stage1Scaffold(
        dim=gt.ambient_dim, tau=tau, k=8, min_nodes=4,
        max_nodes=gt.max_ent_node_upper(2.0) + 1,
        prune_after=10, ann_backend="naive",
        rng=np.random.default_rng(77),
    )
    scaffold.init_from(data, n_seeds=8)
    scaffold.run_until_stable(
        data, StabilizationConfig(min_equilibrium_epochs=3, max_epochs=20),
    )

    result = run_ap_clustering(scaffold)

    assert result.n_clusters >= 1
    assert result.labels.shape == (len(scaffold.nodes),)
    assert len(result.exemplar_indices) == result.n_clusters


def test_circle_clustering_produces_one_cluster() -> None:
    """A uniform circle should not be split by Q-score clustering.

    The Q-score framework tests local intra-correlation vs boundary
    inter-correlation.  On a uniform ring, every proposed cut has
    boundary inter comparable to local intra, so all seeds merge into
    a single cluster.
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
    tau = gt.expected_tau
    assert tau is not None

    scaffold = Stage1Scaffold(
        dim=gt.ambient_dim, tau=tau, k=8, min_nodes=4,
        max_nodes=gt.max_ent_node_upper(2.0) + 1,
        prune_after=10, ann_backend="naive",
        rng=np.random.default_rng(77),
    )
    scaffold.init_from(data, n_seeds=8)
    scaffold.run_until_stable(
        data, StabilizationConfig(min_equilibrium_epochs=3, max_epochs=20),
    )

    result = run_ap_clustering(scaffold)

    assert result.n_clusters == 1
