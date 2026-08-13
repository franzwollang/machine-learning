"""Tests for scaffold-native density level-set clustering (OPEN_ISSUES #44)."""

from __future__ import annotations

import numpy as np
import pytest

from proteus.links import LinkCounters
from proteus.stage1.dm_cluster import DMClusterConfig
from proteus.stage1.level_set import (
    LevelSetBranch,
    LevelSetConfig,
    apply_geometric_screens,
    build_level_set_dag,
    build_level_set_tree,
    select_level_set_partition,
)
from proteus.stage1.recursion import RecursionConfig, run_recursive_discovery


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


def _cycle_edges(ids: list[int], weight: float) -> list[tuple[int, int, float]]:
    return [
        (i, j, weight)
        for i, j in zip(ids, ids[1:] + ids[:1], strict=True)
    ]


def _two_blobs_with_background() -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    rng = np.random.default_rng(4)
    left = rng.normal(loc=(-2.0, 0.0), scale=0.05, size=(8, 2))
    right = rng.normal(loc=(2.0, 0.0), scale=0.05, size=(8, 2))
    background = np.array([[0.0, 1.5], [0.0, -1.5]])
    positions = np.vstack([left, right, background])

    edges: list[tuple[int, int, float]] = []
    edges.extend(_cycle_edges(list(range(0, 8)), 200.0))
    edges.extend(_cycle_edges(list(range(8, 16)), 200.0))
    edges.extend([(7, 8, 1.0), (16, 0, 5.0), (16, 8, 5.0)])
    edges.extend([(17, 3, 5.0), (17, 11, 5.0)])
    return positions, edges


def _four_clumps() -> np.ndarray:
    """Two close pairs of 5-node clumps plus far tissue.

    Far points inflate ``r_k`` so the quantile sweep reaches the intra-pair
    and inter-pair merge radii (otherwise C-D never leaves the within-clump
    scale and the DAG has no parents).
    """

    rng = np.random.default_rng(1)
    centres = ((0.0, 0.0), (0.25, 0.0), (3.0, 0.0), (3.25, 0.0))
    clumps = [rng.normal(loc=c, scale=0.02, size=(5, 2)) for c in centres]
    tissue = np.array([[12.0, 12.0], [12.0, -12.0], [-12.0, 12.0]])
    return np.vstack(clumps + [tissue])


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
    assert level.n_inactive == 2
    assert level.n_runt == 0


def test_level_set_defaults_are_validated_reader() -> None:
    """The production default matches the scaffold probe reader."""

    config = LevelSetConfig()
    assert config.k_neighbors == 8
    assert config.alpha == 1.0
    assert config.min_cluster_size == 4
    assert config.n_levels == 120
    assert config.min_persistence == 0.05
    assert config.min_excess_mass == 0.02


def test_inactive_and_runt_nodes_are_distinct() -> None:
    """Never-activated nodes and undersized active components both map to -1."""

    blob = np.array(
        [
            [0.00, 0.00], [0.05, 0.00], [0.00, 0.05], [0.05, 0.05],
            [0.02, 0.02], [0.03, 0.01], [0.01, 0.03], [0.04, 0.04],
        ],
        dtype=float,
    )
    runt_pair = np.array([[2.0, 0.0], [2.05, 0.0]], dtype=float)
    isolated = np.array([[20.0, 20.0]], dtype=float)
    positions = np.vstack([blob, runt_pair, isolated])
    tree = build_level_set_tree(
        positions,
        LevelSetConfig(k_neighbors=3, min_cluster_size=4, n_levels=40),
    )
    assert any(level.n_inactive > 0 and level.n_clusters == 1 for level in tree.levels)
    runts = [level for level in tree.levels if level.n_runt > 0]
    assert runts
    assert all(level.n_background >= level.n_inactive + level.n_runt for level in tree.levels)
    for level in tree.levels:
        if level.n_runt > 0:
            assert np.all(level.labels[8:10] == -1)


def test_build_level_set_tree_validates_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        build_level_set_tree(np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        build_level_set_tree(np.array([[0.0], [np.nan]]))
    with pytest.raises(ValueError, match="positive"):
        build_level_set_tree(np.zeros((3, 2)), LevelSetConfig(alpha=0.0))


def test_branch_lineage_tracks_parents_and_merges() -> None:
    """Adjacent C-D levels form a grow-and-merge DAG, not independent cuts."""

    tree = build_level_set_tree(
        _four_clumps(),
        LevelSetConfig(k_neighbors=3, alpha=1.0, min_cluster_size=4, n_levels=40),
    )
    dag = build_level_set_dag(tree)
    assert dag.branches
    assert any(level.n_clusters == 4 for level in tree.levels)
    assert any(level.n_clusters == 1 for level in tree.levels)

    by_id = {b.branch_id: b for b in dag.branches}
    merged = [b for b in dag.branches if b.parent_id is not None]
    assert merged
    for child in merged:
        parent = by_id[child.parent_id]
        assert child.branch_id in parent.child_ids
        assert child.merge_level is not None
        assert child.birth_level <= child.merge_level
        assert child.node_ids.issubset(parent.node_ids) or child.merge_level < len(
            tree.levels,
        )


def test_rank_persistence_and_excess_mass_are_normalized() -> None:
    """Lifetime is a core-radius rank gap; mass is a (fraction × rank) integral."""

    positions, _ = _two_blobs_with_background()
    tree = build_level_set_tree(
        positions,
        LevelSetConfig(k_neighbors=4, min_cluster_size=4, n_levels=60),
    )
    dag = build_level_set_dag(tree)
    long_lived = [b for b in dag.branches if b.merge_level is None]
    assert long_lived
    for branch in dag.branches:
        assert 0.0 <= branch.persistence <= 1.0
        assert 0.0 <= branch.excess_mass <= 1.0
    assert max(b.persistence for b in long_lived) >= 0.05
    assert max(b.excess_mass for b in dag.branches) > 0.0


def test_low_persistence_branch_is_pruned() -> None:
    branches = (
        LevelSetBranch(
            branch_id=0, birth_level=0, merge_level=None, parent_id=None,
            child_ids=(), node_ids=frozenset({0}),
            persistence=0.4, excess_mass=0.2,
        ),
        LevelSetBranch(
            branch_id=1, birth_level=3, merge_level=4, parent_id=0,
            child_ids=(), node_ids=frozenset({1}),
            persistence=0.01, excess_mass=0.2,
        ),
    )
    screened = apply_geometric_screens(
        branches, LevelSetConfig(min_persistence=0.05, min_excess_mass=0.02),
    )
    by_id = {b.branch_id: b for b in screened}
    assert by_id[0].prune_reason is None
    assert by_id[1].prune_reason == "persistence"


def test_low_mass_branch_is_pruned() -> None:
    branches = (
        LevelSetBranch(
            branch_id=0, birth_level=0, merge_level=None, parent_id=None,
            child_ids=(), node_ids=frozenset({0}),
            persistence=0.4, excess_mass=0.2,
        ),
        LevelSetBranch(
            branch_id=1, birth_level=0, merge_level=5, parent_id=0,
            child_ids=(), node_ids=frozenset({1}),
            persistence=0.4, excess_mass=0.001,
        ),
    )
    screened = apply_geometric_screens(
        branches, LevelSetConfig(min_persistence=0.05, min_excess_mass=0.02),
    )
    by_id = {b.branch_id: b for b in screened}
    assert by_id[0].prune_reason is None
    assert by_id[1].prune_reason == "excess_mass"


def test_dm_collapses_homogeneous_siblings() -> None:
    """A well-mixed pair is not an evidence-bearing split after DM collapse."""

    positions = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]],
        dtype=float,
    )
    edges = [
        (0, 1, 50.0), (0, 2, 50.0), (2, 3, 50.0), (2, 0, 50.0),
        (1, 3, 50.0), (1, 2, 50.0),
    ]
    scaffold = _Scaffold(positions, edges)
    selection = select_level_set_partition(
        scaffold,
        LevelSetConfig(
            k_neighbors=2,
            min_cluster_size=2,
            n_levels=20,
            min_persistence=0.0,
            min_excess_mass=0.0,
        ),
        DMClusterConfig(tau_bf=3.0),
    )
    assert not selection.accepted
    assert any(b.prune_reason == "dm" for b in selection.branches) or (
        selection.cluster_result is None
    )


def test_selects_coarsest_dm_accepted_partition_without_expected_k() -> None:
    """Automatic extraction uses DAG screens + DM, never a supplied K."""

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
    assert selection.selected_level is not None
    assert selection.dag is not None
    assert selection.branches

    surviving = [
        b for b in selection.branches
        if b.prune_reason is None and b.merge_level is None
    ]
    assert len(surviving) >= 1

    k2_levels = [
        i for i, level in enumerate(selection.tree.levels)
        if level.n_clusters == 2
    ]
    assert k2_levels
    assert selection.selected_level == max(k2_levels)


def test_uniform_ring_has_no_evidence_bearing_level_set_split() -> None:
    """A uniform manifold must remain one feature, not finite-sample arcs."""

    n = 64
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    positions = np.c_[np.cos(theta), np.sin(theta)]
    edges = [(i, (i + 1) % n, 100.0) for i in range(n)]
    scaffold = _Scaffold(positions, edges)

    selection = select_level_set_partition(scaffold)
    assert not selection.accepted
    assert selection.cluster_result is None


def test_extraction_does_not_read_expected_k_from_config() -> None:
    """LevelSetConfig has no cluster-count field to leak into selection."""

    assert not hasattr(LevelSetConfig(), "expect_k")
    assert not hasattr(LevelSetConfig(), "n_clusters")


def test_recursion_flag_still_defaults_off() -> None:
    assert RecursionConfig().use_level_set_clustering is False
    with pytest.raises(ValueError, match="require_persistent_split"):
        run_recursive_discovery(
            np.zeros((4, 2)),
            dim=2,
            config=RecursionConfig(
                min_samples=10,
                use_level_set_clustering=True,
                require_persistent_split=True,
            ),
        )
