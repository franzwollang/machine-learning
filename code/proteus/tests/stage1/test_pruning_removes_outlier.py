"""Stage 1 pruning removes unsupported outlier nodes."""

from __future__ import annotations

import numpy as np

from proteus.nodes import make_node
from proteus.stage1 import Stage1Scaffold
from proteus.stage1.pruning import prune_nodes


def test_pruning_removes_outlier_node() -> None:
    scaffold = Stage1Scaffold(dim=2, tau=1.0, min_nodes=3, ann_backend="naive")
    scaffold.nodes = [
        make_node(np.array([0.0, 0.0]), dim=2),
        make_node(np.array([1.0, 0.0]), dim=2),
        make_node(np.array([0.0, 1.0]), dim=2),
        make_node(np.array([100.0, 100.0]), dim=2),
    ]
    scaffold.tau_local = np.ones(4)
    for node in scaffold.nodes:
        node.hit_count = 100.0
        node.variance = 1.0
        node.update_count = scaffold.prune_after
    scaffold.nodes[3].hit_count = 1.0
    scaffold.nodes[3].variance = 0.1
    scaffold.links.increment_directed(0, 1, 10.0, lift=True)
    scaffold.links.increment_directed(0, 2, 10.0, lift=True)
    scaffold.links.increment_directed(1, 2, 10.0, lift=True)
    scaffold.links.increment_directed(3, 0, 1.0, lift=True)
    scaffold.links.increment_directed(3, 1, 1.0, lift=True)
    scaffold.links.increment_directed(3, 2, 1.0, lift=True)
    scaffold.rebuild_ann()

    removed = prune_nodes(scaffold)

    assert removed == [3]
    assert len(scaffold.nodes) == 3
