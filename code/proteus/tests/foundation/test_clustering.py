"""Unit tests for scale-conditioned Q-score clustering on Stage 1 scaffolds."""

from __future__ import annotations

import numpy as np

from proteus.links import LinkCounters
from proteus.nodes import make_node
from proteus.stage1.clustering import (
    compute_edge_weights,
    partition_q_score,
    q_score,
    run_ap_clustering,
)


class _TwoComponentScaffold:
    """Scaffold with two disconnected 3-node chains."""

    def __init__(self) -> None:
        self.nodes = [make_node(np.array([float(i), 0.0]), dim=2) for i in range(6)]
        self.links = LinkCounters()
        for i, j, w in [(0, 1, 10.0), (1, 2, 10.0), (3, 4, 10.0), (4, 5, 10.0)]:
            self.links.increment_directed(i, j, w, lift=True)
            self.links.increment_directed(j, i, w, lift=True)
        for i, node in enumerate(self.nodes):
            node.hit_count = 100.0
            node.update_count = 50
            node.d_final = 1
        self.tau = 1.0
        self.prune_after = 10
        self.k = 4

    def neighbour_graph(self) -> dict[int, list[int]]:
        return self.links.neighbour_graph(len(self.nodes))


def test_edge_weights_positive_for_linked_pairs() -> None:
    scaffold = _TwoComponentScaffold()

    W = compute_edge_weights(scaffold)

    assert len(W) > 0
    for key, w in W.items():
        assert w > 0.0, f"W_v{key} should be positive"


def test_edge_weights_zero_for_unlinked_pairs() -> None:
    scaffold = _TwoComponentScaffold()

    W = compute_edge_weights(scaffold)

    assert (0, 3) not in W
    assert (2, 5) not in W


def test_q_score_positive_for_tight_cluster() -> None:
    scaffold = _TwoComponentScaffold()
    W = compute_edge_weights(scaffold)
    graph = scaffold.links.neighbour_graph(6)

    q = q_score({0, 1, 2}, W, graph)

    assert q > 0.0, f"Q-score should be positive for tight cluster, got {q}"


def test_ap_finds_two_disconnected_components() -> None:
    scaffold = _TwoComponentScaffold()

    result = run_ap_clustering(scaffold)

    assert result.n_clusters == 2
    assert result.labels[0] == result.labels[1] == result.labels[2]
    assert result.labels[3] == result.labels[4] == result.labels[5]
    assert result.labels[0] != result.labels[3]


def test_partition_q_score_positive_for_valid_partition() -> None:
    scaffold = _TwoComponentScaffold()
    W = compute_edge_weights(scaffold)
    graph = scaffold.links.neighbour_graph(6)
    clusters = [{0, 1, 2}, {3, 4, 5}]

    pq = partition_q_score(clusters, 6, W, graph)

    assert pq > 0.0, f"Partition Q-score should be positive, got {pq}"
