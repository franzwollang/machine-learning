"""Tests for scaffold-native density level-set clustering (OPEN_ISSUES #44)."""

from __future__ import annotations

import numpy as np
import pytest

from proteus.links import LinkCounters
from proteus.stage1.dm_cluster import dm_partition_background_verdict
from proteus.stage1.level_set import (
    LevelSetConfig,
    build_level_set_tree,
    select_level_set_partition,
)


class _Node:
    def __init__(self, position: np.ndarray) -> None:
        self.position = np.asarray(position, dtype=float)
        self.hit_count = 1.0
        self.d_final = 2
        self.variance = 0.1


class _Scaffold:
    def __init__(
        self,
        positions: np.ndarray,
        edges: list[tuple[int, int, float]],
    ) -> None:
        self.nodes = [_Node(p) for p in positions]
        self.links = LinkCounters()
        for i, j, count in edges:
            self.links.increment_directed(i, j, float(count), lift=True)
            self.links.increment_directed(j, i, float(count), lift=True)
        self.tau = 1.0


def _two_blobs_with_background() -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    rng = np.random.default_rng(4)
    left = rng.normal(loc=(-2.0, 0.0), scale=0.05, size=(8, 2))
    right = rng.normal(loc=(2.0, 0.0), scale=0.05, size=(8, 2))
    background = np.array([[0.0, 1.5], [0.0, -1.5]])
    positions = np.vstack([left, right, background])

    edges: list[tuple[int, int, float]] = []
    for block in (range(0, 8), range(8, 16)):
        ids = list(block)
        for i, j in zip(ids, ids[1:] + ids[:1], strict=True):
            edges.append((i, j, 200.0))
    edges.extend([(7, 8, 1.0), (16, 0, 5.0), (16, 8, 5.0)])
    edges.extend([(17, 3, 5.0), (17, 11, 5.0)])
    return positions, edges


def test_build_level_set_tree_preserves_background() -> None:
    """Dense components activate before sparse nodes, which remain label -1."""

    positions, _ = _two_blobs_with_background()
    tree = build_level_set_tree(
        positions,
        LevelSetConfig(
            k_neighbors=4,
            alpha=1.0,
            min_cluster_size=4,
            n_levels=60,
        ),
    )

    candidates = [level for level in tree.levels if level.n_clusters == 2]
    assert candidates
    assert any(level.n_background == 2 for level in candidates)
    level = next(level for level in candidates if level.n_background == 2)
    assert set(level.labels[:8]) == {0}
    assert set(level.labels[8:16]) == {1}
    assert np.all(level.labels[16:] == -1)


def test_level_set_defaults_are_validated_reader() -> None:
    """The production default matches the scaffold probe reader."""

    config = LevelSetConfig()
    assert config.k_neighbors == 8
    assert config.alpha == 1.0
    assert config.min_cluster_size == 4
    assert config.n_levels == 120


def test_build_level_set_tree_validates_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        build_level_set_tree(np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        build_level_set_tree(np.array([[0.0], [np.nan]]))
    with pytest.raises(ValueError, match="positive"):
        build_level_set_tree(np.zeros((3, 2)), LevelSetConfig(alpha=0.0))


def test_selects_coarsest_dm_accepted_partition_without_expected_k() -> None:
    """Automatic extraction uses DM evidence, never a supplied cluster count."""

    positions, edges = _two_blobs_with_background()
    scaffold = _Scaffold(positions, edges)
    config = LevelSetConfig(
        k_neighbors=4,
        alpha=1.0,
        min_cluster_size=4,
        n_levels=60,
    )
    selection = select_level_set_partition(scaffold, config)

    assert selection.accepted
    assert selection.cluster_result is not None
    assert selection.cluster_result.n_clusters == 2
    assert selection.log_bf > np.log(3.0)
    labels = selection.cluster_result.labels
    assert set(labels[:8]) == {0}
    assert set(labels[8:16]) == {1}
    assert np.all(labels[16:] == -1)

    # Coarsest accepted means no later (coarser) level also clears the gate.
    assert selection.selected_level is not None
    for level in selection.tree.levels[selection.selected_level + 1:]:
        if level.n_clusters < 2:
            continue
        clusters = [
            set(np.where(level.labels == label)[0].tolist())
            for label in range(level.n_clusters)
        ]
        background = set(np.where(level.labels < 0)[0].tolist())
        _, accepted = dm_partition_background_verdict(
            scaffold, clusters, background,
        )
        assert not accepted


def test_uniform_ring_has_no_evidence_bearing_level_set_split() -> None:
    """A uniform manifold must remain one feature, not finite-sample arcs."""

    n = 64
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    positions = np.c_[np.cos(theta), np.sin(theta)]
    edges = [
        (i, (i + 1) % n, 100.0)
        for i in range(n)
    ]
    scaffold = _Scaffold(positions, edges)

    selection = select_level_set_partition(scaffold)
    assert not selection.accepted
    assert selection.cluster_result is None
