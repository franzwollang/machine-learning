"""Unit tests for Stage 1 stabilization helpers."""

from __future__ import annotations

import numpy as np

from proteus.links import LinkCounters
from proteus.nodes import make_node
from proteus.stage1.stabilization import (
    StabilizationConfig,
    compute_neighbor_normalized_cv,
    compute_variance_cv,
    cv_threshold,
    is_stable,
)


class _TinyScaffold:
    def __init__(self, n: int = 3, prune_after: int = 0) -> None:
        self.nodes = [make_node(np.zeros(2), dim=2) for _ in range(n)]
        self.links = LinkCounters()
        self.links.increment_directed(0, 1, 1.0, lift=True)
        if n > 2:
            self.links.increment_directed(1, 2, 1.0, lift=True)
        self.prune_after = prune_after
        self.k = 8

    def neighbour_graph(self) -> dict[int, list[int]]:
        return self.links.neighbour_graph(len(self.nodes))


def test_neighbor_normalized_cv_zero_for_uniform_rho() -> None:
    scaffold = _TinyScaffold()
    for node in scaffold.nodes:
        node.residual_mean = np.array([1.0, 0.0])
        node.variance = 1.0

    assert np.isclose(compute_neighbor_normalized_cv(scaffold), 0.0)


def test_neighbor_normalized_cv_positive_for_nonuniform_rho() -> None:
    scaffold = _TinyScaffold()
    scaffold.nodes[0].residual_mean = np.array([1.0, 0.0])
    scaffold.nodes[1].residual_mean = np.array([2.0, 0.0])
    scaffold.nodes[2].residual_mean = np.array([4.0, 0.0])
    for node in scaffold.nodes:
        node.variance = 1.0

    assert compute_neighbor_normalized_cv(scaffold) > 0.0


def test_is_stable_requires_consecutive_tail() -> None:
    config = StabilizationConfig(cv_tolerance=0.5, min_equilibrium_epochs=3)

    assert not is_stable([0.1, 0.1], config)
    assert is_stable([0.9, 0.1, 0.2, 0.3], config)
    assert not is_stable([0.1, 0.6, 0.1], config)


def test_is_stable_auto_threshold_from_scaffold() -> None:
    config = StabilizationConfig(min_equilibrium_epochs=2)
    tol = cv_threshold(8, config.cv_buffer)

    class _MockScaffold:
        k = 8

    assert is_stable([tol - 0.01, tol - 0.01], config, scaffold=_MockScaffold())
    assert not is_stable([tol + 0.01, tol - 0.01], config, scaffold=_MockScaffold())


def test_cv_threshold_formula() -> None:
    assert np.isclose(cv_threshold(8, 1.5), 1.5 * np.sqrt(2.0 / 8))
    assert np.isclose(cv_threshold(16, 2.0), 2.0 * np.sqrt(2.0 / 16))
    assert cv_threshold(1, 1.0) > 0.0


def test_variance_cv_zero_for_uniform_variance() -> None:
    scaffold = _TinyScaffold(n=5, prune_after=0)
    for node in scaffold.nodes:
        node.variance = 0.5
        node.update_count = 10

    assert np.isclose(compute_variance_cv(scaffold), 0.0)


def test_variance_cv_positive_for_heterogeneous_variance() -> None:
    scaffold = _TinyScaffold(n=5, prune_after=0)
    for i, node in enumerate(scaffold.nodes):
        node.variance = 0.1 * (i + 1)
        node.update_count = 10

    assert compute_variance_cv(scaffold) > 0.0


def test_variance_cv_inf_when_insufficient_mature_nodes() -> None:
    scaffold = _TinyScaffold(n=5, prune_after=100)

    assert compute_variance_cv(scaffold) == float("inf")
