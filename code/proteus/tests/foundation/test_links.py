"""Unit tests for Hebbian link counters with shadow/lifted semantics."""

from __future__ import annotations

import numpy as np

from proteus.links import LinkCounters


def test_record_neighborhood_sets_protected_until() -> None:
    links = LinkCounters()
    links.record_neighborhood([5, 2, 9], [1.0, 0.5, 0.25], protected_until=42)

    by_pair = {(link.i, link.j): link for link in links.as_list()}
    assert by_pair[(2, 5)].protected_until == 42
    assert by_pair[(5, 9)].protected_until == 42


def test_increment_directed_tracks_orientation() -> None:
    links = LinkCounters()

    links.increment_directed(1, 2, 0.5)
    links.increment_directed(2, 1, 0.25)

    [link] = links.as_list()
    assert (link.i, link.j) == (1, 2)
    assert np.isclose(link.count_ij, 0.5)
    assert np.isclose(link.count_ji, 0.25)
    assert np.isclose(links.total_count(2, 1), 0.75)


def test_increment_directed_extends_protected_until_on_lift() -> None:
    links = LinkCounters()
    links.increment_directed(0, 1, 1.0, lift=False, protected_until=10)
    edge = {(l.i, l.j): l for l in links.as_list()}[(0, 1)]
    assert edge.protected_until == 10
    links.increment_directed(0, 1, 0.5, lift=True, protected_until=99)
    assert edge.lifted is True
    assert edge.protected_until == 99


def test_record_neighborhood_lifts_bmu_pair_only() -> None:
    links = LinkCounters()

    links.record_neighborhood([5, 2, 9], [1.0, 0.5, 0.25])

    by_pair = {(link.i, link.j): link for link in links.as_list()}
    assert by_pair[(2, 5)].lifted is True
    assert np.isclose(by_pair[(2, 5)].count_ji, 0.5)
    assert (5, 9) in by_pair  # shadow edge created
    assert by_pair[(5, 9)].lifted is False
    assert np.isclose(by_pair[(5, 9)].count_ij, 0.25)


def test_neighbour_graph_returns_only_lifted() -> None:
    links = LinkCounters()
    links.record_neighborhood([0, 1, 2], [1.0, 0.5, 0.25])

    graph = links.neighbour_graph(3)
    full = links.full_graph(3)

    assert 1 in graph[0]
    assert 2 not in graph[0]
    assert 1 in full[0]
    assert 2 in full[0]


def test_shadow_edge_promoted_by_bmu_coactivation() -> None:
    links = LinkCounters()
    links.record_neighborhood([0, 1, 2], [1.0, 0.5, 0.25])

    assert not links.as_list()[0].lifted or links.as_list()[1].lifted is False
    edge_02 = {(l.i, l.j): l for l in links.as_list()}[(0, 2)]
    assert edge_02.lifted is False

    links.record_neighborhood([0, 2], [1.0, 0.5])
    edge_02 = {(l.i, l.j): l for l in links.as_list()}[(0, 2)]
    assert edge_02.lifted is True


def test_self_edges_are_ignored() -> None:
    links = LinkCounters()

    links.increment_directed(1, 1, 1.0)

    assert links.as_list() == []
