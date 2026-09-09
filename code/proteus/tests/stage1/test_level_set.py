"""Tests for scaffold-native density level-set clustering (OPEN_ISSUES #44)."""

from __future__ import annotations

import numpy as np
import pytest

from proteus.links import LinkCounters
from proteus.stage1.dm_cluster import DMClusterConfig
from proteus.stage1.level_set import (
    LevelSetBranch,
    LevelSetConfig,
    LevelSetSelection,
    LevelSetTree,
    ValleyResolvability,
    ValleyVerdict,
    apply_geometric_screens,
    assess_valley_resolvability,
    at_shot_noise_scale,
    build_level_set_dag,
    build_level_set_tree,
    finer_walk_needs_node_budget,
    mean_neighbor_radius,
    mesh_is_scale_matched,
    next_node_budget,
    null_bottleneck_ratio,
    select_level_set_partition,
    studentized_bottleneck,
)
from proteus.stage1.recursion import (
    RecursionConfig,
    _grow_underresolved_level_set,
    _level_set_finer_walk_exhausted,
    _level_set_should_finer_walk,
    run_recursive_discovery,
)
from tests.datasets.synthetic.circles import make_circle
from tests.datasets.synthetic.hierarchical_gaussian import (
    make_hierarchical_gaussian,
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
        max_nodes: int | None = None,
    ) -> None:
        self.nodes = [_Node(p) for p in positions]
        self.links = LinkCounters()
        for i, j, count in edges:
            self.links.increment_directed(i, j, float(count), lift=True)
            self.links.increment_directed(j, i, float(count), lift=True)
        self.tau = 1.0
        if max_nodes is not None:
            self.max_nodes = int(max_nodes)


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
    assert config.min_cluster_frac == 0.15
    assert config.max_bottleneck_ratio == 0.25
    assert config.grow_nodes_when_underresolved is True
    assert config.node_growth_factor == 2.0
    assert config.max_node_growth_steps == 5
    assert config.mesh_scale_match_ratio == 2.0


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


def _knn_edges(positions: np.ndarray, k: int = 8) -> list[tuple[int, int, float]]:
    from scipy.spatial import cKDTree

    n = int(positions.shape[0])
    k_use = max(1, min(int(k), n - 1))
    _, idx = cKDTree(positions).query(positions, k=k_use + 1)
    edges: list[tuple[int, int, float]] = []
    for i, nbrs in enumerate(idx):
        for j in nbrs[1:]:
            edges.append((i, int(j), 1.0))
    return edges


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


def test_coarse_tissue_satellite_is_not_a_split() -> None:
    """A few-percent coarse satellite is background, not a second feature."""

    rng = np.random.default_rng(0)
    blob = rng.normal((0.0, 0.0), 0.05, size=(40, 2))
    speck = rng.normal((8.0, 0.0), 0.02, size=(4, 2))
    positions = np.vstack([blob, speck])
    selection = select_level_set_partition(_Scaffold(positions, _knn_edges(positions)))
    assert not selection.accepted


def test_faded_circle_is_one_feature() -> None:
    """Tissue-polluted circle must not split into arcs (the #44 false-split)."""

    ds = make_circle(n_samples=400, tissue_fraction=0.03, seed=0)
    selection = select_level_set_partition(
        _Scaffold(ds.points, _knn_edges(ds.points)),
    )
    assert not selection.accepted


def test_hierarchy_returns_coarsest_three_way_split() -> None:
    """Root extraction returns the coarse blobs; fine children are recursion."""

    ds = make_hierarchical_gaussian(n_samples=600, seed=0)
    selection = select_level_set_partition(
        _Scaffold(ds.points, _knn_edges(ds.points)),
    )
    assert selection.accepted
    assert selection.cluster_result is not None
    assert selection.cluster_result.n_clusters == 3


def _two_arcs(gap_flow: float) -> _Scaffold:
    """A ring with two position gaps whose coarsest filtered cut is 2 arcs.

    ``gap_flow`` sets the Hebbian counts on the edges crossing the gaps.
    High gap flow is fitted-manifold physics (traffic crosses a sampling
    gap on a connected feature); near-zero gap flow is a genuine density
    valley between two weakly linked features.
    """

    arc_a = np.linspace(0.0, 110.0, 12) * np.pi / 180.0
    arc_b = np.linspace(180.0, 290.0, 12) * np.pi / 180.0
    theta = np.concatenate([arc_a, arc_b])
    positions = np.c_[np.cos(theta), np.sin(theta)]
    edges = [(i, i + 1, 6.0) for i in range(11)]
    edges += [(i, i + 1, 6.0) for i in range(12, 23)]
    edges += [(11, 12, gap_flow), (23, 0, gap_flow)]
    return _Scaffold(positions, edges)


def test_bottleneck_guard_rejects_arc_cut_with_manifold_flow() -> None:
    """A balanced arc cut whose boundary carries manifold flow is rejected.

    Position statistics cannot separate sampling-gap arcs from true
    valleys (persistence, excess mass, and stability were all measured
    inseparable); the cross-cut flow bottleneck can.
    """

    scaffold = _two_arcs(gap_flow=6.0)
    selection = select_level_set_partition(
        scaffold,
        LevelSetConfig(k_neighbors=4, min_cluster_size=4, n_levels=60),
    )
    assert not selection.accepted
    assert selection.cluster_result is None


def test_weak_bridge_split_passes_bottleneck_guard() -> None:
    """The same geometry with near-zero bridge flow is a real valley."""

    scaffold = _two_arcs(gap_flow=0.05)
    selection = select_level_set_partition(
        scaffold,
        LevelSetConfig(k_neighbors=4, min_cluster_size=4, n_levels=60),
    )
    assert selection.accepted
    assert selection.cluster_result is not None
    assert selection.cluster_result.n_clusters == 2
    labels = selection.cluster_result.labels
    assert len(set(labels[:12]) | set(labels[12:])) == 2


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


def test_two_blob_split_is_resolved_split() -> None:
    positions, edges = _two_blobs_with_background()
    selection = select_level_set_partition(
        _Scaffold(positions, edges),
        LevelSetConfig(k_neighbors=4, min_cluster_size=4, n_levels=60),
    )
    assert selection.accepted
    assert selection.resolvability is not None
    assert selection.resolvability.verdict == ValleyVerdict.RESOLVED_SPLIT
    assert selection.resolvability.saw_balanced_cut
    assert selection.bottleneck_ratio is not None
    assert selection.bottleneck_ratio <= LevelSetConfig().max_bottleneck_ratio
    assert selection.candidate_level is not None
    assert selection.studentized_ratio is not None
    assert selection.studentized_ratio <= LevelSetConfig().max_bottleneck_ratio


def test_arc_cut_at_node_cap_is_resolved_null() -> None:
    """Bottleneck-rejected arcs must not trigger cap growth, even at max_nodes."""

    scaffold = _two_arcs(gap_flow=6.0)
    n = len(scaffold.nodes)
    scaffold.max_nodes = n
    selection = select_level_set_partition(
        scaffold,
        LevelSetConfig(k_neighbors=4, min_cluster_size=4, n_levels=60),
    )
    assert not selection.accepted
    assert selection.resolvability is not None
    assert selection.resolvability.verdict == ValleyVerdict.RESOLVED_NULL
    assert selection.resolvability.at_node_cap
    assert selection.resolvability.saw_balanced_cut
    assert selection.resolvability.reject_reason == "bottleneck"
    assert selection.bottleneck_ratio is not None
    assert selection.bottleneck_ratio > LevelSetConfig().max_bottleneck_ratio
    assert selection.candidate_level is not None

    grown_result, _, grown_sel, _ = _grow_underresolved_level_set(
        np.zeros((n, 2)),
        dim=2,
        config=RecursionConfig(use_level_set_clustering=True),
        scale_search_config=RecursionConfig().scale_search,
        scaffold=scaffold,
        selection=selection,
    )
    assert grown_result is None
    assert grown_sel.resolvability is not None
    assert grown_sel.resolvability.verdict == ValleyVerdict.RESOLVED_NULL


def test_weak_bridge_is_resolved_split() -> None:
    scaffold = _two_arcs(gap_flow=0.05)
    selection = select_level_set_partition(
        scaffold,
        LevelSetConfig(k_neighbors=4, min_cluster_size=4, n_levels=60),
    )
    assert selection.accepted
    assert selection.resolvability is not None
    assert selection.resolvability.verdict == ValleyVerdict.RESOLVED_SPLIT


def test_capped_scaffold_without_balanced_cut_is_under_resolved() -> None:
    rng = np.random.default_rng(0)
    blob = rng.normal((0.0, 0.0), 0.02, size=(16, 2))
    scaffold = _Scaffold(blob, _knn_edges(blob), max_nodes=16)
    selection = select_level_set_partition(
        scaffold,
        LevelSetConfig(
            k_neighbors=4,
            min_cluster_size=4,
            min_cluster_frac=0.4,
            n_levels=40,
        ),
    )
    assert not selection.accepted
    assert selection.resolvability is not None
    assert selection.resolvability.verdict == ValleyVerdict.UNDER_RESOLVED
    assert selection.resolvability.at_node_cap
    assert not selection.resolvability.saw_balanced_cut


def test_next_node_budget_respects_ceiling() -> None:
    assert next_node_budget(1024, 2.0, 10_000) == 2048
    assert next_node_budget(1024, 2.0, 1536) == 1536
    assert next_node_budget(5, 2.0, 100) == 10
    assert next_node_budget(1, 2.0, 1) == 1


def test_child_finer_walk_only_when_under_resolved() -> None:
    """Density children split only at their own tau*; the finer walk is root-only."""

    no_cut = LevelSetSelection(
        tree=LevelSetTree(core_radii=np.empty(0), levels=()),
        cluster_result=None,
        selected_level=None,
        log_bf=float("-inf"),
        resolvability=ValleyResolvability(
            verdict=ValleyVerdict.UNDER_RESOLVED,
            saw_balanced_cut=False,
            at_node_cap=True,
            n_nodes=64,
            max_nodes=64,
            reject_reason="no_cut",
        ),
    )
    bottleneck = LevelSetSelection(
        tree=no_cut.tree,
        cluster_result=None,
        selected_level=None,
        log_bf=float("-inf"),
        resolvability=ValleyResolvability(
            verdict=ValleyVerdict.RESOLVED_NULL,
            saw_balanced_cut=True,
            at_node_cap=True,
            n_nodes=64,
            max_nodes=64,
            reject_reason="bottleneck",
        ),
    )
    assert _level_set_should_finer_walk(0, no_cut) is True
    assert _level_set_should_finer_walk(0, bottleneck) is True
    assert _level_set_should_finer_walk(1, no_cut) is False
    assert _level_set_should_finer_walk(1, bottleneck) is False


def test_finer_walk_does_not_stop_on_bottleneck() -> None:
    """Composites show bottleneck-rejected arcs before tau_sep (#48)."""

    from types import SimpleNamespace

    scaffold = _two_arcs(gap_flow=6.0)
    rng = np.random.default_rng(0)
    data = rng.normal(size=(400, 2))
    cfg = RecursionConfig(use_level_set_clustering=True)
    bottleneck = SimpleNamespace(
        resolvability=SimpleNamespace(reject_reason="bottleneck"),
    )
    shot = SimpleNamespace(
        resolvability=SimpleNamespace(reject_reason="one_feature_null"),
    )
    assert _level_set_finer_walk_exhausted(bottleneck, scaffold, data, cfg) is False
    assert _level_set_finer_walk_exhausted(shot, scaffold, data, cfg) is True


def test_studentized_bottleneck_rejects_circle_probe_first_accept() -> None:
    """Circle f7: raw φ sits in the true-split band; φ/φ_0 does not (#48)."""

    ceiling = LevelSetConfig().max_bottleneck_ratio
    assert studentized_bottleneck(0.052, 0.05) > ceiling
    assert studentized_bottleneck(0.05, 0.80) <= ceiling
    assert studentized_bottleneck(0.05, None) is None
    assert studentized_bottleneck(0.05, 0.0) is None


def test_shot_noise_floor_is_node_rk_versus_sample_knn() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(size=(40, 2))
    assert at_shot_noise_scale(_Scaffold(sample, _knn_edges(sample)), sample, k=4)
    coarse = sample[::5]
    assert mean_neighbor_radius(coarse, 4) > mean_neighbor_radius(sample, 4)
    assert not at_shot_noise_scale(_Scaffold(coarse, _knn_edges(coarse)), sample, k=4)


def test_no_cut_at_cap_is_resolved_null_at_shot_floor() -> None:
    rng = np.random.default_rng(0)
    blob = rng.normal((0.0, 0.0), 0.02, size=(16, 2))
    scaffold = _Scaffold(blob, _knn_edges(blob), max_nodes=16)
    resolvability = assess_valley_resolvability(
        scaffold,
        accepted=False,
        saw_balanced_cut=False,
        reject_reason="no_cut",
        at_shot_floor=True,
    )
    assert resolvability.verdict == ValleyVerdict.RESOLVED_NULL
    assert resolvability.at_node_cap


def test_mesh_scale_match_is_rk_versus_residual() -> None:
    """Thin uniforms have r_k >> sqrt(tau*); coarse composites do not (#48)."""

    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    ring = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    scaffold = _Scaffold(ring, _knn_edges(ring, k=4), max_nodes=16)
    scaffold.tau = 1e-3
    assert not mesh_is_scale_matched(
        scaffold, scaffold.tau, k=4, ratio=LevelSetConfig().mesh_scale_match_ratio,
    )
    scaffold.tau = 4.0
    assert mesh_is_scale_matched(
        scaffold, scaffold.tau, k=4, ratio=LevelSetConfig().mesh_scale_match_ratio,
    )


def test_finer_walk_needs_budget_only_when_parent_scale_matched() -> None:
    """Circle flickers bottleneck→no_cut; only scale-matched parents grow."""

    no_cut = LevelSetSelection(
        tree=LevelSetTree(core_radii=np.empty(0), levels=()),
        cluster_result=None,
        selected_level=None,
        log_bf=float("-inf"),
        resolvability=ValleyResolvability(
            verdict=ValleyVerdict.UNDER_RESOLVED,
            saw_balanced_cut=False,
            at_node_cap=True,
            n_nodes=64,
            max_nodes=64,
            reject_reason="no_cut",
        ),
    )
    bottleneck = LevelSetSelection(
        tree=no_cut.tree,
        cluster_result=None,
        selected_level=None,
        log_bf=float("-inf"),
        resolvability=ValleyResolvability(
            verdict=ValleyVerdict.RESOLVED_NULL,
            saw_balanced_cut=True,
            at_node_cap=True,
            n_nodes=64,
            max_nodes=64,
            reject_reason="bottleneck",
        ),
    )
    assert finer_walk_needs_node_budget(
        no_cut, parent_scale_matched=True, at_shot_floor=False,
    )
    assert not finer_walk_needs_node_budget(
        no_cut, parent_scale_matched=False, at_shot_floor=False,
    )
    assert not finer_walk_needs_node_budget(
        no_cut, parent_scale_matched=True, at_shot_floor=True,
    )
    assert not finer_walk_needs_node_budget(
        bottleneck, parent_scale_matched=True, at_shot_floor=False,
    )


def test_grow_skips_thin_uniform_no_cut() -> None:
    """L=1 circle/swiss no_cut must not raise N (rk >> sqrt(tau*))."""

    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    ring = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    scaffold = _Scaffold(ring, _knn_edges(ring, k=4), max_nodes=16)
    scaffold.tau = 1e-3
    selection = LevelSetSelection(
        tree=LevelSetTree(core_radii=np.empty(0), levels=()),
        cluster_result=None,
        selected_level=None,
        log_bf=float("-inf"),
        resolvability=ValleyResolvability(
            verdict=ValleyVerdict.UNDER_RESOLVED,
            saw_balanced_cut=False,
            at_node_cap=True,
            n_nodes=16,
            max_nodes=16,
            reject_reason="no_cut",
        ),
    )
    grown_result, grown_sc, _, budget = _grow_underresolved_level_set(
        ring,
        dim=2,
        config=RecursionConfig(use_level_set_clustering=True),
        scale_search_config=RecursionConfig().scale_search,
        scaffold=scaffold,
        selection=selection,
    )
    assert grown_result is None
    assert grown_sc is scaffold
    assert budget == 16


def test_equal_weak_diameters_are_one_feature_null() -> None:
    """Linearly separable arcs with a matching orthogonal hole must not fail-open."""

    base = _two_arcs(gap_flow=0.05)
    positions = np.asarray([node.position for node in base.nodes])
    edges = [(i, i + 1, 6.0) for i in range(11) if (i, i + 1) != (5, 6)]
    edges += [(i, i + 1, 6.0) for i in range(12, 23) if (i, i + 1) != (17, 18)]
    edges += [(11, 12, 0.05), (23, 0, 0.05), (5, 6, 0.05), (17, 18, 0.05)]
    selection = select_level_set_partition(
        _Scaffold(positions, edges),
        LevelSetConfig(k_neighbors=4, min_cluster_size=4, n_levels=60),
    )
    assert not selection.accepted
    assert selection.resolvability is not None
    assert selection.resolvability.reject_reason in {
        "one_feature_null", "bottleneck",
    }


def test_true_valley_has_studentized_ratio_below_ceiling() -> None:
    scaffold = _two_arcs(gap_flow=0.05)
    selection = select_level_set_partition(
        scaffold,
        LevelSetConfig(k_neighbors=4, min_cluster_size=4, n_levels=60),
    )
    assert selection.accepted
    assert selection.null_bottleneck_ratio is not None
    assert selection.studentized_ratio is not None
    assert selection.studentized_ratio <= LevelSetConfig().max_bottleneck_ratio
    positions = np.asarray([node.position for node in scaffold.nodes])
    labels = selection.cluster_result.labels
    phi0 = null_bottleneck_ratio(scaffold, positions, labels)
    assert phi0 is not None
    assert phi0 > selection.bottleneck_ratio
