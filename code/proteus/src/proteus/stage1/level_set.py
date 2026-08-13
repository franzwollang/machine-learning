"""Density level-set clustering on an equilibrated Stage-1 scaffold.

This module implements the OPEN_ISSUES #44 pivot: the scaffold's equalized
hit mass makes per-node counts nearly flat, but its node spacing remains a
monotone proxy for sample density.  A Chaudhuri--Dasgupta robust
single-linkage sweep over node positions therefore estimates the Hartigan
density cluster tree without volumetric cells or a known magnification
exponent.

Tree construction and automatic extraction are proposal-path and default-off:
inspect candidate partitions from coarse to fine and select the coarsest one
whose signal blocks clear the background-aware Dirichlet--multinomial
homogeneity Bayes factor. This is the intended acceptance reduction, but the
finite-sample branch null remains unresolved (#44). Nodes inactive at the
chosen density level retain label ``-1`` as an explicit background tier rather
than being forcibly absorbed into a signal cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from proteus.stage1.clustering import (
    ClusterResult,
    compute_edge_weights,
    partition_q_score,
)
from proteus.stage1.dm_cluster import (
    DMClusterConfig,
    dm_partition_background_verdict,
)

__all__ = [
    "LevelSetConfig",
    "LevelSetLevel",
    "LevelSetTree",
    "LevelSetSelection",
    "build_level_set_tree",
    "select_level_set_partition",
]


@dataclass(frozen=True)
class LevelSetConfig:
    """Configuration for scaffold-native robust single linkage.

    ``k_neighbors=8`` and ``alpha=1`` are the validated proposal-path reader
    from the #44 probes.  ``n_levels`` samples the monotone density sweep; it
    does not choose the accepted partition.  ``min_cluster_size`` suppresses
    components too small to be evidence-bearing.  The DM gate is the sole
    automatic split arbiter.
    """

    k_neighbors: int = 8
    alpha: float = 1.0
    min_cluster_size: int = 4
    n_levels: int = 120


@dataclass(frozen=True)
class LevelSetLevel:
    """One density level in the estimated cluster tree."""

    radius: float
    labels: np.ndarray
    n_clusters: int
    n_background: int


@dataclass(frozen=True)
class LevelSetTree:
    """Ordered fine-to-coarse C-D level sweep."""

    core_radii: np.ndarray
    levels: tuple[LevelSetLevel, ...]


@dataclass(frozen=True)
class LevelSetSelection:
    """Automatic extraction result and diagnostics."""

    tree: LevelSetTree
    cluster_result: ClusterResult | None
    selected_level: int | None
    log_bf: float

    @property
    def accepted(self) -> bool:
        return self.cluster_result is not None


def _validate_positions(positions: np.ndarray) -> np.ndarray:
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2:
        raise ValueError("positions must have shape (n, d)")
    if not np.all(np.isfinite(points)):
        raise ValueError("positions must be finite")
    return points


def build_level_set_tree(
    positions: np.ndarray,
    config: LevelSetConfig | None = None,
) -> LevelSetTree:
    """Estimate the density cluster tree by robust single linkage.

    At radius ``r``, activate nodes whose distance to their k-th *other*
    neighbour is at most ``r`` and connect active pairs within ``alpha * r``.
    Components with fewer than ``min_cluster_size`` nodes remain background.
    Thus label ``-1`` operationally combines inactive low-density nodes and
    active runt components; distinguishing those background subtypes is part
    of the unresolved branch-extraction work. Levels are ordered from
    fine/high-density to coarse/low-density.
    """

    config = config or LevelSetConfig()
    points = _validate_positions(positions)
    n = int(points.shape[0])
    if n == 0:
        return LevelSetTree(
            core_radii=np.empty(0, dtype=float),
            levels=(),
        )
    if n == 1:
        labels = np.array([-1], dtype=int)
        return LevelSetTree(
            core_radii=np.array([0.0], dtype=float),
            levels=(
                LevelSetLevel(
                    radius=0.0,
                    labels=labels,
                    n_clusters=0,
                    n_background=1,
                ),
            ),
        )

    k = max(1, min(int(config.k_neighbors), n - 1))
    alpha = float(config.alpha)
    if not (alpha > 0.0):
        raise ValueError("alpha must be positive")
    min_size = max(1, int(config.min_cluster_size))
    n_levels = max(2, int(config.n_levels))

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    core_radii = np.asarray(dists[:, -1], dtype=float)
    quantiles = np.linspace(0.02, 1.0, n_levels)
    radii = np.unique(np.quantile(core_radii, quantiles))

    levels: list[LevelSetLevel] = []
    previous: np.ndarray | None = None
    for radius in radii:
        active = np.where(core_radii <= float(radius))[0]
        labels = np.full(n, -1, dtype=int)
        if active.size >= min_size:
            sub = points[active]
            pairs = cKDTree(sub).query_pairs(
                alpha * float(radius),
                output_type="ndarray",
            )
            m = int(active.size)
            if pairs.size:
                graph = csr_matrix(
                    (
                        np.ones(len(pairs), dtype=float),
                        (pairs[:, 0], pairs[:, 1]),
                    ),
                    shape=(m, m),
                )
            else:
                graph = csr_matrix((m, m))
            _, component = connected_components(graph, directed=False)
            sizes = np.bincount(component)
            major = np.where(sizes >= min_size)[0]
            remap = {int(c): i for i, c in enumerate(major)}
            for node_id, comp_id in zip(active, component, strict=True):
                labels[int(node_id)] = remap.get(int(comp_id), -1)

        # Multiple adjacent radii can induce the same partition.  Keep only
        # structural changes; labels are already dense and deterministic.
        if previous is not None and np.array_equal(labels, previous):
            continue
        previous = labels.copy()
        live = labels[labels >= 0]
        n_clusters = len(set(int(v) for v in live))
        levels.append(
            LevelSetLevel(
                radius=float(radius),
                labels=labels,
                n_clusters=n_clusters,
                n_background=int(np.sum(labels < 0)),
            ),
        )

    return LevelSetTree(
        core_radii=core_radii,
        levels=tuple(levels),
    )


def _label_sets(labels: np.ndarray) -> tuple[list[set[int]], set[int]]:
    clusters = [
        set(np.where(labels == label)[0].tolist())
        for label in sorted(set(int(v) for v in labels if v >= 0))
    ]
    background = set(np.where(labels < 0)[0].tolist())
    return clusters, background


def select_level_set_partition(
    scaffold: Any,
    config: LevelSetConfig | None = None,
    dm_config: DMClusterConfig | None = None,
) -> LevelSetSelection:
    """Select the coarsest evidence-bearing split in the node-density tree.

    Candidate levels are inspected from coarse to fine.  The first partition
    with at least two signal components that clears the background-aware DM
    margin is returned.  This is hierarchy-preserving: recursion receives the
    coarsest accepted split, then can discover finer branches inside each
    child.  No expected cluster count or ground-truth label enters selection.
    """

    config = config or LevelSetConfig()
    dm_config = dm_config or DMClusterConfig()
    positions = np.asarray(
        [node.position for node in scaffold.nodes],
        dtype=float,
    )
    tree = build_level_set_tree(positions, config)
    if not tree.levels:
        return LevelSetSelection(tree, None, None, float("-inf"))

    best_rejected_bf = float("-inf")
    for level_index in range(len(tree.levels) - 1, -1, -1):
        level = tree.levels[level_index]
        if level.n_clusters < 2:
            continue
        clusters, background = _label_sets(level.labels)
        log_bf, accepted = dm_partition_background_verdict(
            scaffold,
            clusters,
            background,
            dm_config,
        )
        best_rejected_bf = max(best_rejected_bf, float(log_bf))
        if not accepted:
            continue

        hits = np.asarray(
            [float(node.hit_count) for node in scaffold.nodes],
            dtype=float,
        )
        exemplars = np.asarray(
            [int(max(c, key=lambda node_id: hits[node_id])) for c in clusters],
            dtype=int,
        )
        graph_lifted = scaffold.links.neighbour_graph(len(scaffold.nodes))
        weights = compute_edge_weights(scaffold)
        q_value = partition_q_score(
            clusters,
            len(scaffold.nodes),
            weights,
            graph_lifted,
        )
        result = ClusterResult(
            labels=level.labels.copy(),
            exemplar_indices=exemplars,
            n_clusters=len(clusters),
            partition_q_score=float(q_value),
        )
        return LevelSetSelection(
            tree=tree,
            cluster_result=result,
            selected_level=level_index,
            log_bf=float(log_bf),
        )

    return LevelSetSelection(
        tree=tree,
        cluster_result=None,
        selected_level=None,
        log_bf=best_rejected_bf,
    )
