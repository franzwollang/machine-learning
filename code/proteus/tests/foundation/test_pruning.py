"""Unit tests for Stage 1 pruning helpers."""

from __future__ import annotations

import numpy as np

from proteus.nodes import make_node
from proteus.stage1 import Stage1Scaffold
from proteus.stage1.pruning import (
    demote_lifted_by_cluster,
    directed_floor_mass,
    prune_links,
    prune_nodes,
)


def test_prune_nodes_removes_low_hit_low_variance_node() -> None:
    scaffold = Stage1Scaffold(dim=2, tau=1.0, min_nodes=2, ann_backend="naive")
    scaffold.nodes = [
        make_node(np.array([0.0, 0.0]), dim=2),
        make_node(np.array([1.0, 0.0]), dim=2),
        make_node(np.array([2.0, 0.0]), dim=2),
    ]
    scaffold.tau_local = np.ones(3)
    for idx, node in enumerate(scaffold.nodes):
        node.hit_count = 100.0
        node.variance = 1.0
        node.update_count = scaffold.prune_after
    scaffold.nodes[1].hit_count = 1.0
    scaffold.nodes[1].variance = 0.1
    scaffold.links.increment_directed(0, 1, 10.0, lift=True)
    scaffold.links.increment_directed(1, 2, 10.0, lift=True)
    scaffold.links.increment_directed(0, 2, 10.0, lift=True)
    scaffold.rebuild_ann()

    removed = prune_nodes(scaffold)

    assert removed == [1]
    assert len(scaffold.nodes) == 2


def test_prune_links_removes_weak_edge_by_mass_fraction() -> None:
    """A weak edge below the directed mass floor should be pruned (bilateral).

    With ``dim=2`` and ``beta=0.5``, ``m_floor = (0.5 / 4) * T = 0.125 * T``.
    Node 0 has ``T = 10.1`` (edges to 1 and 2); ``m_floor ≈ 1.26``, so
    ``C(0→1)=0.1`` yields a prune vote from 0.  Node 1 is symmetric toward 0.
    Bilateral agreement removes the weak edge.
    """

    scaffold = Stage1Scaffold(dim=2, tau=1.0, min_nodes=1, ann_backend="naive")
    scaffold.nodes = [make_node(np.zeros(2), dim=2) for _ in range(4)]
    scaffold.tau_local = np.ones(4)
    scaffold.iteration = 100
    scaffold.links.increment_directed(0, 1, 0.1, lift=True)
    scaffold.links.increment_directed(1, 0, 0.1, lift=True)
    scaffold.links.increment_directed(0, 2, 10.0, lift=True)
    scaffold.links.increment_directed(2, 0, 10.0, lift=True)
    scaffold.links.increment_directed(1, 3, 10.0, lift=True)
    scaffold.links.increment_directed(3, 1, 10.0, lift=True)

    verdicts = prune_links(scaffold)

    removed_pairs = {(v.i, v.j) for v in verdicts if v.accepted}
    assert (0, 1) in removed_pairs


def test_prune_links_bilateral_protects_asymmetric_edge() -> None:
    """An edge that is in the tail for one endpoint but dominant for the
    other should NOT be pruned (bilateral agreement required)."""

    scaffold = Stage1Scaffold(dim=2, tau=1.0, min_nodes=1, ann_backend="naive")
    scaffold.nodes = [make_node(np.zeros(2), dim=2) for _ in range(3)]
    scaffold.tau_local = np.ones(3)
    scaffold.iteration = 100
    scaffold.links.increment_directed(0, 1, 100.0, lift=True)
    scaffold.links.increment_directed(0, 2, 1.0, lift=True)
    scaffold.links.increment_directed(2, 0, 50.0, lift=True)

    verdicts = prune_links(scaffold)

    removed_pairs = {(v.i, v.j) for v in verdicts if v.accepted}
    assert (0, 2) not in removed_pairs


def test_prune_links_floor_equality_survives() -> None:
    """Counts exactly at ``m_floor`` do not receive a prune vote (strict ``<``)."""

    scaffold = Stage1Scaffold(dim=2, tau=1.0, min_nodes=1, ann_backend="naive")
    scaffold.nodes = [make_node(np.zeros(2), dim=2) for _ in range(4)]
    scaffold.tau_local = np.ones(4)
    scaffold.iteration = 100
    # T = 8, m_floor = 0.125 * 8 = 1.0 per endpoint with beta=0.5, dim=2
    scaffold.links.increment_directed(0, 1, 1.0, lift=True)
    scaffold.links.increment_directed(1, 0, 1.0, lift=True)
    scaffold.links.increment_directed(0, 2, 7.0, lift=True)
    scaffold.links.increment_directed(2, 0, 7.0, lift=True)
    scaffold.links.increment_directed(1, 3, 7.0, lift=True)
    scaffold.links.increment_directed(3, 1, 7.0, lift=True)

    m = directed_floor_mass(2, scaffold.prune_beta, 8.0)
    assert np.isclose(m, 1.0)

    verdicts = prune_links(scaffold)
    removed_pairs = {(v.i, v.j) for v in verdicts if v.accepted}
    assert (0, 1) not in removed_pairs


def test_prune_links_floor_strictly_below_removed() -> None:
    """Directed counts strictly below ``m_floor`` vote prune (bilateral removes)."""

    scaffold = Stage1Scaffold(dim=2, tau=1.0, min_nodes=1, ann_backend="naive")
    scaffold.nodes = [make_node(np.zeros(2), dim=2) for _ in range(4)]
    scaffold.tau_local = np.ones(4)
    scaffold.iteration = 100
    scaffold.links.increment_directed(0, 1, 0.99, lift=True)
    scaffold.links.increment_directed(1, 0, 0.99, lift=True)
    scaffold.links.increment_directed(0, 2, 7.01, lift=True)
    scaffold.links.increment_directed(2, 0, 7.01, lift=True)
    scaffold.links.increment_directed(1, 3, 7.01, lift=True)
    scaffold.links.increment_directed(3, 1, 7.01, lift=True)

    verdicts = prune_links(scaffold)
    removed_pairs = {(v.i, v.j) for v in verdicts if v.accepted}
    assert (0, 1) in removed_pairs


def test_prune_links_skips_removal_when_protected() -> None:
    """Bilateral prune votes do not delete a link until ``protected_until``."""

    scaffold = Stage1Scaffold(dim=2, tau=1.0, min_nodes=1, ann_backend="naive")
    scaffold.nodes = [make_node(np.zeros(2), dim=2) for _ in range(4)]
    scaffold.tau_local = np.ones(4)
    scaffold.iteration = 10
    scaffold.links.increment_directed(0, 1, 0.1, lift=True)
    scaffold.links.increment_directed(1, 0, 0.1, lift=True)
    scaffold.links.increment_directed(0, 2, 10.0, lift=True)
    scaffold.links.increment_directed(2, 0, 10.0, lift=True)
    scaffold.links.increment_directed(1, 3, 10.0, lift=True)
    scaffold.links.increment_directed(3, 1, 10.0, lift=True)
    for link in scaffold.links.as_list():
        if (link.i, link.j) == (0, 1):
            link.protected_until = 50
            break

    prune_links(scaffold)
    pairs = {(link.i, link.j) for link in scaffold.links.as_list()}
    assert (0, 1) in pairs


def test_demote_lifted_by_cluster_demotes_weak_lifted_edge() -> None:
    """Stage 2 demotion should convert a weak lifted edge to shadow."""

    scaffold = Stage1Scaffold(dim=2, tau=1.0, min_nodes=1, ann_backend="naive")
    scaffold.nodes = [make_node(np.zeros(2), dim=2) for _ in range(4)]
    scaffold.tau_local = np.ones(4)
    for node in scaffold.nodes:
        node.hit_count = 100.0
        node.d_final = 1
    scaffold.links.increment_directed(0, 1, 0.1, lift=True)
    scaffold.links.increment_directed(1, 0, 0.1, lift=True)
    scaffold.links.increment_directed(0, 2, 10.0, lift=True)
    scaffold.links.increment_directed(2, 0, 10.0, lift=True)
    scaffold.links.increment_directed(1, 3, 10.0, lift=True)
    scaffold.links.increment_directed(3, 1, 10.0, lift=True)

    labels = np.array([0, 0, 0, 0], dtype=int)
    verdicts = demote_lifted_by_cluster(scaffold, labels, beta=0.5)

    demoted = {(v.i, v.j) for v in verdicts if v.accepted}
    assert (0, 1) in demoted

    link_01 = None
    for link in scaffold.links.as_list():
        if (link.i, link.j) == (0, 1):
            link_01 = link
            break
    assert link_01 is not None, "Edge 0-1 should still exist as shadow"
    assert not link_01.lifted, "Edge 0-1 should be demoted to shadow"
