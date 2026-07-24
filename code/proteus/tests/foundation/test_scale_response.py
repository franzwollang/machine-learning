"""Unit tests for Lindeberg-normalized scale response (SI S2.5)."""

from __future__ import annotations

import numpy as np

from proteus.stage1 import Stage1Scaffold
from proteus.stage1.scale_response import (
    cluster_response,
    node_response,
    support_trace,
)


def _circle_scaffold(n_nodes: int = 8, tau: float = 0.1) -> Stage1Scaffold:
    """Build a small converged scaffold on a unit circle."""

    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    scaffold = Stage1Scaffold(
        dim=2, tau=tau, k=4, min_nodes=4, max_nodes=n_nodes,
        ann_backend="naive", enable_topology_edits=False,
        rng=np.random.default_rng(0),
    )
    scaffold.init_from(points, n_seeds=n_nodes)
    scaffold.run_epoch(points)
    return scaffold


def test_node_response_all_positive() -> None:
    scaffold = _circle_scaffold()

    R = node_response(scaffold, scaffold.tau, d_working=2)

    assert R.shape == (len(scaffold.nodes),)
    assert np.all(R > 0.0)


def test_cluster_response_equals_sum_of_node_responses() -> None:
    scaffold = _circle_scaffold()

    R = node_response(scaffold, scaffold.tau, d_working=2)
    phi = cluster_response(scaffold, scaffold.tau, d_working=2)

    assert np.isclose(phi, R.sum())


def test_support_trace_positive() -> None:
    scaffold = _circle_scaffold()

    V = support_trace(scaffold, scaffold.tau, d_working=2)

    assert V > 0.0


def test_support_trace_increases_with_tau() -> None:
    scaffold = _circle_scaffold(tau=0.1)

    V_small = support_trace(scaffold, 0.01, d_working=2)
    V_large = support_trace(scaffold, 1.0, d_working=2)

    assert V_large >= V_small
