"""Density level-set clustering on an equilibrated Stage-1 scaffold.

This module implements the OPEN_ISSUES #44 pivot: the scaffold's equalized
hit mass makes per-node counts nearly flat, but its node spacing remains a
monotone proxy for sample density.  A Chaudhuri--Dasgupta robust
single-linkage sweep over node positions therefore estimates the Hartigan
density cluster tree without volumetric cells or a known magnification
exponent.

Tree construction is proposal-path and default-off.  Automatic extraction
builds an explicit merge DAG, prunes short-lived and low-mass branches
(ToMATo-style rank persistence and normalized excess mass), then confirms
surviving sibling groups with the background-aware Dirichlet--multinomial
homogeneity Bayes factor.  The split criterion is a Hartigan density
valley, not support connectivity: a geometrically connected manifold is
still split when a superlevel set disconnects into evidence-bearing
modes.  Uniform-density manifolds (circle, swiss roll, disk) remain one
feature because they have no valley; sampling-gap arcs are rejected by
the flow-bottleneck guard.  A valley-resolvability trichotomy (resolved
split / resolved null / under-resolved) classifies the current read;
finer descent re-seeds the mesh at each finer ``tau`` with ``N`` free
up to ``n/k`` (SI S2.6.2 / #48).  This path is not a default.  Nodes
inactive or runt-sized at the chosen density level
retain label ``-1`` as an explicit background tier rather than being
forcibly absorbed into a signal cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import ceil
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, maximum_flow
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
    "LevelSetBranch",
    "LevelSetDAG",
    "LevelSetSelection",
    "ValleyVerdict",
    "ValleyResolvability",
    "assess_valley_resolvability",
    "mean_neighbor_radius",
    "shot_noise_node_cap",
    "at_shot_noise_scale",
    "null_bottleneck_ratio",
    "studentized_bottleneck",
    "build_level_set_tree",
    "build_level_set_dag",
    "apply_geometric_screens",
    "apply_dm_sibling_collapse",
    "select_level_set_partition",
]


@dataclass(frozen=True)
class LevelSetConfig:
    """Configuration for scaffold-native robust single linkage.

    ``k_neighbors=8`` and ``alpha=1`` are the validated proposal-path reader
    from the #44 probes.  ``n_levels`` samples the monotone density sweep; it
    does not choose the accepted partition.  ``min_cluster_size`` suppresses
    components too small to be evidence-bearing.

    ``min_persistence`` and ``min_excess_mass`` are family-wise geometric
    screens (SI S14.3).  Connected-manifold nulls (ring, disk, line) produce
    spurious-branch envelopes that overlap true two-blob / four-clump splits,
    so the defaults prune only short-lived or low-mass runts.

    Extraction is coarse-anchored on *mass-filtered* cuts: the single
    candidate is the coarsest level whose cut still has ``K >= 2`` clusters
    after the relative-mass floor.  ``min_cluster_frac`` is a fraction of
    the *region node budget*, so a coarse tissue satellite or a pair of
    still-inactive core fragments cannot be an evidence-bearing split.
    Nested features whose valley is bridged by tissue before the coarse
    tail (nested shells, linked tori) surface as balanced mid-tree cuts and
    are now visited.

    ``max_bottleneck_ratio`` is the flow-bottleneck guard that makes the
    mid-tree visit safe.  Sampling-gap arcs of a connected manifold pass
    every position-only screen (persistence, excess mass, subsample and
    ``k`` stability all measured inseparable on 2026-08-13), but the cut
    boundary of an arc is not a *flow* bottleneck: the cross-cut max-flow
    is comparable to either arc's own internal throughput (ratio 0.6-1.3
    measured on fitted circles), while true valleys carry near-zero cross
    flow relative to each side's internal bisection flow (0.0-0.07 on
    fitted hierarchy / tori / nested shells).  All of these are operational
    proposal-path defaults, not promoted acceptance constants.

    ``growth_policy="track_tau"`` is the landed finer-walk policy
    (SI S2.6.2 / OPEN_ISSUES #48): each finer ``τ`` re-seeds the mesh
    (``fit_scaffold_at_tau``) with ``N`` free up to the derived bound
    ``n/k``.  There is no node-budget gate, no ``no_cut`` trigger, and
    no scale-match ratio.  Descent ends when the bound binds or on
    ``one_feature_null``.

    Raw ``φ`` is not an acceptance-path floor: the circle finer-walk
    probe (2026-09-09) accepted a shot-noise cut at ``φ=0.052``, inside
    the fitted true-split band.  The #48 studentized ratio
    ``φ / φ_0`` compares the candidate to typical *disagreeing*
    intrinsic connected cuts of the same flow graph (SI S2.6.2).  The
    same ``max_bottleneck_ratio`` ceiling is applied to that ratio.
    """

    k_neighbors: int = 8
    alpha: float = 1.0
    min_cluster_size: int = 4
    n_levels: int = 120
    min_persistence: float = 0.05
    min_excess_mass: float = 0.02
    min_cluster_frac: float = 0.15
    max_bottleneck_ratio: float = 0.25
    growth_policy: str = "track_tau"

    def __post_init__(self) -> None:
        if self.growth_policy not in {"track_tau"}:
            raise ValueError(
                "growth_policy must be 'track_tau', "
                f"got {self.growth_policy!r}"
            )


@dataclass(frozen=True)
class LevelSetLevel:
    """One density level in the estimated cluster tree.

    Public ``labels`` map runt and inactive nodes to ``-1``.  ``n_inactive``
    counts nodes with ``r_k > r``; ``n_runt`` counts active components below
    ``min_cluster_size``.  Both contribute to ``n_background``.
    """

    radius: float
    labels: np.ndarray
    n_clusters: int
    n_background: int
    n_inactive: int = 0
    n_runt: int = 0


@dataclass(frozen=True)
class LevelSetTree:
    """Ordered fine-to-coarse C-D level sweep."""

    core_radii: np.ndarray
    levels: tuple[LevelSetLevel, ...]


@dataclass(frozen=True)
class LevelSetBranch:
    """One persistent component in the C-D merge DAG.

    ``persistence`` is the core-radius rank lifetime
    ``rank(merge) - rank(birth)`` (or ``1 - rank(birth)`` if the branch
    survives to the coarsest level).  ``excess_mass`` is the integral of
    per-level node-mass fraction along that rank interval.  ``prune_reason``
    is ``None`` for retained branches and one of ``"persistence"``,
    ``"excess_mass"``, or ``"dm"`` after screening.
    """

    branch_id: int
    birth_level: int
    merge_level: int | None
    parent_id: int | None
    child_ids: tuple[int, ...]
    node_ids: frozenset[int]
    persistence: float
    excess_mass: float
    prune_reason: str | None = None


@dataclass(frozen=True)
class LevelSetDAG:
    """Merge DAG plus per-level cluster-to-branch map.

    ``level_branch_ids[ℓ][k]`` is the branch id of public cluster ``k`` at
    structural level ``ℓ`` (fine-to-coarse, matching ``tree.levels``).
    """

    branches: tuple[LevelSetBranch, ...]
    level_branch_ids: tuple[tuple[int, ...], ...]


class ValleyVerdict(str, Enum):
    """Trichotomy at a fitted ``(tau, N)`` scaffold (SI S2.6.2)."""

    RESOLVED_SPLIT = "resolved_split"
    RESOLVED_NULL = "resolved_null"
    UNDER_RESOLVED = "under_resolved"


@dataclass(frozen=True)
class ValleyResolvability:
    """Node-growth classification for one level-set extraction.

    ``reject_reason`` is ``None`` on an accepted split, otherwise one of
    ``"bottleneck"``, ``"one_feature_null"``, ``"dm"``, or ``"no_cut"``.
    """

    verdict: ValleyVerdict
    saw_balanced_cut: bool
    at_node_cap: bool
    n_nodes: int
    max_nodes: int | None
    reject_reason: str | None = None


@dataclass(frozen=True)
class LevelSetSelection:
    """Automatic extraction result and diagnostics."""

    tree: LevelSetTree
    cluster_result: ClusterResult | None
    selected_level: int | None
    log_bf: float
    dag: LevelSetDAG | None = None
    branches: tuple[LevelSetBranch, ...] = ()
    resolvability: ValleyResolvability | None = None
    candidate_level: int | None = None
    bottleneck_ratio: float | None = None
    null_bottleneck_ratio: float | None = None
    studentized_ratio: float | None = None

    @property
    def accepted(self) -> bool:
        return self.cluster_result is not None


def assess_valley_resolvability(
    scaffold: Any,
    *,
    accepted: bool,
    saw_balanced_cut: bool,
    reject_reason: str | None = None,
    at_shot_floor: bool = False,
) -> ValleyResolvability:
    """Classify a fitted scaffold as split / null / under-resolved (SI S2.6.2).

    The trichotomy governs node-cap growth, not whether a finer ``tau`` may
    still be probed.  A resolved null at coarse ``L=1`` can still be a
    composite feature that separates only below ``tau_sep``.  ``no_cut``
    at the cap is ``under_resolved`` only when the mesh is still coarser
    than the sample ``k``NN (a hidden valley remains possible).  At the
    shot-noise floor further ``N`` growth only resolves sample atoms.
    """

    n_nodes = len(getattr(scaffold, "nodes", ()))
    raw_cap = getattr(scaffold, "max_nodes", None)
    max_nodes = int(raw_cap) if raw_cap is not None else None
    at_cap = max_nodes is not None and n_nodes >= max_nodes
    if accepted:
        verdict = ValleyVerdict.RESOLVED_SPLIT
    elif saw_balanced_cut:
        verdict = ValleyVerdict.RESOLVED_NULL
    elif at_cap and not at_shot_floor:
        verdict = ValleyVerdict.UNDER_RESOLVED
    else:
        verdict = ValleyVerdict.RESOLVED_NULL
    return ValleyResolvability(
        verdict=verdict,
        saw_balanced_cut=saw_balanced_cut,
        at_node_cap=at_cap,
        n_nodes=n_nodes,
        max_nodes=max_nodes,
        reject_reason=reject_reason,
    )


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
    Components with fewer than ``min_cluster_size`` nodes remain background
    (runts), distinct in the ``n_runt`` count from never-activated nodes.
    Levels are ordered from fine/high-density to coarse/low-density.
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
                    n_inactive=1,
                    n_runt=0,
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
        n_runt = 0
        if active.size:
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
            n_runt = int(np.sum(sizes[sizes < min_size]))

        if previous is not None and np.array_equal(labels, previous):
            continue
        previous = labels.copy()
        live = labels[labels >= 0]
        n_clusters = len(set(int(v) for v in live))
        n_inactive = int(n - active.size)
        levels.append(
            LevelSetLevel(
                radius=float(radius),
                labels=labels,
                n_clusters=n_clusters,
                n_background=int(np.sum(labels < 0)),
                n_inactive=n_inactive,
                n_runt=n_runt,
            ),
        )

    return LevelSetTree(
        core_radii=core_radii,
        levels=tuple(levels),
    )


def _cluster_sets(labels: np.ndarray) -> dict[int, set[int]]:
    groups: dict[int, set[int]] = {}
    for i, lab in enumerate(np.asarray(labels, dtype=int)):
        if lab < 0:
            continue
        groups.setdefault(int(lab), set()).add(int(i))
    return groups


def _map_fine_to_coarse(
    fine: dict[int, set[int]],
    coarse: dict[int, set[int]],
) -> dict[int, int | None]:
    """Map each fine cluster to the coarse cluster of maximum overlap.

    Empty overlap is background absorption.  C-D components only grow and
    merge as radius increases, so this recovers the unique tree parent.
    """

    mapping: dict[int, int | None] = {}
    for fine_id, fine_nodes in fine.items():
        best_id: int | None = None
        best_overlap = 0
        for coarse_id, coarse_nodes in coarse.items():
            overlap = len(fine_nodes & coarse_nodes)
            if overlap > best_overlap:
                best_overlap = overlap
                best_id = int(coarse_id)
        mapping[int(fine_id)] = best_id if best_overlap > 0 else None
    return mapping


def _level_ranks(tree: LevelSetTree) -> np.ndarray:
    """Empirical core-radius CDF at each structural level (monotone rank)."""

    radii = np.asarray(tree.core_radii, dtype=float)
    n_levels = len(tree.levels)
    ranks = np.zeros(n_levels, dtype=float)
    if radii.size == 0:
        if n_levels:
            ranks[:] = np.linspace(0.0, 1.0, n_levels)
        return ranks
    for i, level in enumerate(tree.levels):
        ranks[i] = float(np.mean(radii <= float(level.radius)))
    return ranks


def _branch_stats(
    nodes_by_level: dict[int, frozenset[int]],
    birth_level: int,
    merge_level: int | None,
    ranks: np.ndarray,
    n_nodes: int,
) -> tuple[float, float]:
    """Return ``(persistence, excess_mass)`` on the rank scale."""

    n_rank = int(ranks.shape[0])
    birth_rank = float(ranks[birth_level]) if n_rank else 0.0
    if merge_level is None:
        death_rank = 1.0
        last = n_rank
    else:
        death_rank = float(ranks[merge_level]) if merge_level < n_rank else 1.0
        last = int(merge_level)
    persistence = max(0.0, death_rank - birth_rank)
    if n_nodes <= 0 or last <= birth_level:
        return persistence, 0.0

    mass = 0.0
    denom = float(n_nodes)
    living = sorted(ℓ for ℓ in nodes_by_level if birth_level <= ℓ < last)
    for idx, level in enumerate(living):
        if idx + 1 < len(living):
            nxt = living[idx + 1]
            delta = float(ranks[nxt] - ranks[level])
        else:
            delta = death_rank - float(ranks[level])
        if delta <= 0.0:
            continue
        mass += (len(nodes_by_level[level]) / denom) * delta
    return persistence, float(mass)


def build_level_set_dag(tree: LevelSetTree) -> LevelSetDAG:
    """Recover stable branch identities from adjacent C-D partitions.

    Matching uses set overlap.  When several fine components merge, the
    oldest (earliest birth, then larger mass) continues and the others die
    into it — the ToMATo survivor rule.  Unmatched fine components are
    absorbed into background.
    """

    if not tree.levels:
        return LevelSetDAG(branches=(), level_branch_ids=())

    n_nodes = int(tree.levels[0].labels.shape[0])
    ranks = _level_ranks(tree)
    next_id = 0
    records: dict[int, dict[str, Any]] = {}
    level_maps: list[dict[int, int]] = []

    first = _cluster_sets(tree.levels[0].labels)
    cmap: dict[int, int] = {}
    for lab, nodes in first.items():
        bid = next_id
        next_id += 1
        records[bid] = {
            "birth_level": 0,
            "merge_level": None,
            "parent_id": None,
            "child_ids": [],
            "nodes_by_level": {0: frozenset(nodes)},
        }
        cmap[int(lab)] = bid
    level_maps.append(cmap)

    for i in range(len(tree.levels) - 1):
        fine = _cluster_sets(tree.levels[i].labels)
        coarse = _cluster_sets(tree.levels[i + 1].labels)
        mapping = _map_fine_to_coarse(fine, coarse)
        prev_cmap = level_maps[i]
        next_cmap: dict[int, int] = {}
        children_of: dict[int, list[int]] = {int(c): [] for c in coarse}
        unmatched: list[int] = []
        for fine_id, coarse_id in mapping.items():
            if coarse_id is None:
                unmatched.append(int(fine_id))
            else:
                children_of[int(coarse_id)].append(int(fine_id))

        for fine_id in unmatched:
            bid = prev_cmap[fine_id]
            records[bid]["merge_level"] = i + 1

        for coarse_id, fine_ids in children_of.items():
            cnodes = frozenset(coarse[coarse_id])
            if not fine_ids:
                bid = next_id
                next_id += 1
                records[bid] = {
                    "birth_level": i + 1,
                    "merge_level": None,
                    "parent_id": None,
                    "child_ids": [],
                    "nodes_by_level": {i + 1: cnodes},
                }
                next_cmap[coarse_id] = bid
                continue
            if len(fine_ids) == 1:
                bid = prev_cmap[fine_ids[0]]
                records[bid]["nodes_by_level"][i + 1] = cnodes
                next_cmap[coarse_id] = bid
                continue
            child_bids = [prev_cmap[f] for f in fine_ids]
            survivor = min(
                child_bids,
                key=lambda b: (
                    int(records[b]["birth_level"]),
                    -len(records[b]["nodes_by_level"][i]),
                    b,
                ),
            )
            for bid in child_bids:
                if bid == survivor:
                    continue
                records[bid]["merge_level"] = i + 1
                records[bid]["parent_id"] = survivor
                records[survivor]["child_ids"].append(bid)
            records[survivor]["nodes_by_level"][i + 1] = cnodes
            next_cmap[coarse_id] = survivor
        level_maps.append(next_cmap)

    branches: list[LevelSetBranch] = []
    for bid, rec in sorted(records.items()):
        persistence, excess = _branch_stats(
            rec["nodes_by_level"],
            int(rec["birth_level"]),
            rec["merge_level"],
            ranks,
            n_nodes,
        )
        last_level = max(rec["nodes_by_level"])
        branches.append(
            LevelSetBranch(
                branch_id=int(bid),
                birth_level=int(rec["birth_level"]),
                merge_level=rec["merge_level"],
                parent_id=rec["parent_id"],
                child_ids=tuple(int(c) for c in rec["child_ids"]),
                node_ids=frozenset(rec["nodes_by_level"][last_level]),
                persistence=float(persistence),
                excess_mass=float(excess),
            ),
        )

    level_branch_ids = tuple(
        tuple(cmap[k] for k in sorted(cmap))
        for cmap in level_maps
    )
    return LevelSetDAG(
        branches=tuple(branches),
        level_branch_ids=level_branch_ids,
    )


def apply_geometric_screens(
    branches: tuple[LevelSetBranch, ...],
    config: LevelSetConfig | None = None,
) -> tuple[LevelSetBranch, ...]:
    """Prune branches that fail rank persistence or excess-mass floors."""

    config = config or LevelSetConfig()
    min_p = float(config.min_persistence)
    min_m = float(config.min_excess_mass)
    out: list[LevelSetBranch] = []
    for branch in branches:
        reason = branch.prune_reason
        if reason is None and branch.persistence < min_p:
            reason = "persistence"
        if reason is None and branch.excess_mass < min_m:
            reason = "excess_mass"
        out.append(branch if reason == branch.prune_reason else replace(
            branch, prune_reason=reason,
        ))
    return tuple(out)


def _filter_relative_mass(
    labels: np.ndarray,
    min_size: int,
    min_frac: float,
) -> np.ndarray:
    """Map clusters below the relative-mass floor to background.

    Floor is ``max(min_size, ceil(min_frac * n))`` of the region node
    budget, so a coarse tissue satellite or a still-inactive core fragment
    cannot be an evidence-bearing sibling.
    """

    out = np.asarray(labels, dtype=int).copy()
    n = int(out.shape[0])
    if n == 0:
        return out
    floor = max(int(min_size), int(ceil(float(min_frac) * n)))
    for lab in set(int(v) for v in out if v >= 0):
        if int(np.sum(out == lab)) < floor:
            out[out == lab] = -1
    remain = sorted(set(int(v) for v in out if v >= 0))
    remap = {old: i for i, old in enumerate(remain)}
    compacted = np.full_like(out, -1)
    for i, lab in enumerate(out):
        if lab >= 0:
            compacted[i] = remap[int(lab)]
    return compacted


_FLOW_SCALE = 100.0
"""Integer capacity scale for ``scipy`` max-flow on Hebbian counts."""


def _flow_graph(scaffold: Any) -> tuple[list[int], list[int], list[int]]:
    """Undirected integer-capacity edge lists from the Hebbian link counts."""

    rows: list[int] = []
    cols: list[int] = []
    caps: list[int] = []
    for link in scaffold.links.as_list():
        weight = float(link.count_ij) + float(link.count_ji)
        cap = int(round(_FLOW_SCALE * weight))
        if cap > 0:
            i, j = int(link.i), int(link.j)
            rows += [i, j]
            cols += [j, i]
            caps += [cap, cap]
    return rows, cols, caps


def _set_maxflow(
    graph: tuple[list[int], list[int], list[int]],
    n: int,
    source_set: list[int],
    sink_set: list[int],
) -> float:
    """Max flow between two node sets, routed through the full flow graph."""

    if not source_set or not sink_set:
        return 0.0
    rows, cols, caps = graph
    big = 2 ** 30
    r = rows + [n] * len(source_set) + list(sink_set)
    c = cols + list(source_set) + [n + 1] * len(sink_set)
    p = caps + [big] * (len(source_set) + len(sink_set))
    matrix = csr_matrix((p, (r, c)), shape=(n + 2, n + 2))
    return float(maximum_flow(matrix, n, n + 1).flow_value) / _FLOW_SCALE


def _induced_adjacency(
    graph: tuple[list[int], list[int], list[int]],
    members: np.ndarray,
) -> csr_matrix:
    """Induced undirected weight matrix on ``members`` (local ``0..m-1``)."""

    members = np.asarray(members, dtype=int)
    index = {int(g): i for i, g in enumerate(members)}
    m = int(members.shape[0])
    rows, cols, caps = graph
    loc_r: list[int] = []
    loc_c: list[int] = []
    weights: list[float] = []
    for i, j, cap in zip(rows, cols, caps, strict=True):
        if i in index and j in index and i != j:
            loc_r.append(index[i])
            loc_c.append(index[j])
            weights.append(float(cap))
    if not weights:
        return csr_matrix((m, m), dtype=float)
    return csr_matrix((weights, (loc_r, loc_c)), shape=(m, m), dtype=float)


def _hop_adjacency(adj: csr_matrix) -> csr_matrix:
    """Symmetrized unweighted copy: hop length 1 on every positive edge."""

    if adj.nnz == 0:
        return adj
    binary = adj.maximum(adj.T).tocsr()
    binary.data = np.ones(binary.data.shape[0], dtype=float)
    return binary


def _hop_distances(adj: csr_matrix, source: int) -> np.ndarray:
    return dijkstra(
        _hop_adjacency(adj),
        directed=False,
        indices=int(source),
        unweighted=True,
    )


def _farthest_pair(adj: csr_matrix) -> tuple[int, int] | None:
    """2-approximation to the hop-diameter endpoints."""

    m = int(adj.shape[0])
    if m < 2 or adj.nnz == 0:
        return None
    binary = _hop_adjacency(adj)
    seed_dist = dijkstra(binary, directed=False, indices=0, unweighted=True)
    finite = np.isfinite(seed_dist)
    if not np.any(finite):
        return None
    start = int(np.argmax(np.where(finite, seed_dist, -1.0)))
    far_dist = dijkstra(binary, directed=False, indices=start, unweighted=True)
    finite_far = np.isfinite(far_dist)
    if not np.any(finite_far):
        return None
    end = int(np.argmax(np.where(finite_far, far_dist, -1.0)))
    if start == end:
        return None
    return start, end


def _normalized_laplacian_vectors(
    adj: csr_matrix,
    n_eigs: int,
) -> np.ndarray | None:
    """First ``n_eigs`` non-trivial eigenvectors of the normalized Laplacian.

    ``L = I - D^{-1/2} A D^{-1/2}`` on the symmetrized weights.  Column 0 of
    ``eigh`` is the trivial (or first-component) mode and is skipped.
    Returns shape ``(m, k)`` or ``None`` when the solve is undefined.
    """

    m = int(adj.shape[0])
    k = min(int(n_eigs), m - 1)
    if m < 2 or k < 1:
        return None
    dense = np.asarray((0.5 * (adj + adj.T)).toarray(), dtype=float)
    deg = dense.sum(axis=1)
    d_inv_sqrt = np.zeros(m, dtype=float)
    positive = deg > 0.0
    if not np.any(positive):
        return None
    d_inv_sqrt[positive] = 1.0 / np.sqrt(deg[positive])
    scaled = dense * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    laplacian = np.eye(m) - scaled
    try:
        _vals, vecs = np.linalg.eigh(laplacian)
    except np.linalg.LinAlgError:
        return None
    return vecs[:, 1 : 1 + k]


def _median_split_mask(values: np.ndarray) -> np.ndarray | None:
    """Boolean median split; ``None`` if both sides cannot be populated."""

    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    if int(np.sum(finite)) < 2:
        return None
    filled = vals.copy()
    fill = float(np.median(vals[finite]))
    filled[~finite] = fill
    side = filled > float(np.median(filled))
    if int(side.sum()) == 0 or int((~side).sum()) == 0:
        order = np.argsort(filled, kind="stable")
        side = np.zeros(filled.shape[0], dtype=bool)
        side[order[filled.shape[0] // 2 :]] = True
    if int(side.sum()) == 0 or int((~side).sum()) == 0:
        return None
    return side


def _component_count(adj: csr_matrix, local_ids: np.ndarray) -> int:
    if local_ids.size == 0:
        return 0
    if local_ids.size == 1:
        return 1
    n_comp, _ = connected_components(
        adj[local_ids][:, local_ids], directed=False,
    )
    return int(n_comp)


def _both_sides_connected(adj: csr_matrix, mask: np.ndarray) -> bool:
    return (
        _component_count(adj, np.where(mask)[0]) == 1
        and _component_count(adj, np.where(~mask)[0]) == 1
    )


def _largest_component(adj: csr_matrix, local_ids: np.ndarray) -> np.ndarray:
    if local_ids.size <= 1:
        return local_ids
    n_comp, labels = connected_components(
        adj[local_ids][:, local_ids], directed=False,
    )
    if n_comp <= 1:
        return local_ids
    keep = int(np.argmax(np.bincount(labels)))
    return local_ids[labels == keep]


def _intrinsic_split_mask(adj: csr_matrix) -> np.ndarray | None:
    """Fiedler median split, else hop-geodesic median from the diameter."""

    m = int(adj.shape[0])
    if m < 2:
        return None
    mask = None
    if m >= 4:
        vecs = _normalized_laplacian_vectors(adj, 1)
        if vecs is not None:
            mask = _median_split_mask(vecs[:, 0])
    if mask is None:
        pair = _farthest_pair(adj)
        if pair is None:
            return None
        mask = _median_split_mask(_hop_distances(adj, pair[0]))
    return mask


def _hyperplane_bisection_flow(
    graph: tuple[list[int], list[int], list[int]],
    n: int,
    members: np.ndarray,
    positions: np.ndarray,
) -> float:
    """Legacy ambient-hyperplane internal throughput (#48 contrast).

    Bisects ``members`` by the max-variance coordinate median.  Kept so
    tests can show that a curled sheet inflates φ when a side is
    disconnected in the flow graph.
    """

    points = positions[members]
    axis = int(np.argmax(points.var(axis=0)))
    median = float(np.median(points[:, axis]))
    half_a = members[points[:, axis] <= median].tolist()
    half_b = members[points[:, axis] > median].tolist()
    if not half_a or not half_b:
        order = np.argsort(points[:, axis])
        mid = max(1, len(members) // 2)
        half_a = members[order[:mid]].tolist()
        half_b = members[order[mid:]].tolist()
    return _set_maxflow(graph, n, half_a, half_b)


def _bisection_flow(
    graph: tuple[list[int], list[int], list[int]],
    n: int,
    members: np.ndarray,
    positions: np.ndarray,
) -> float:
    """Internal throughput: max flow across an intrinsic connected bisection.

    Bisects the side by the Fiedler vector of the induced flow subgraph
    (symmetrized weights, normalized Laplacian).  Falls back to a hop-count
    graph-geodesic median split from the side's farthest-point pair when
    the eigen-solve fails or the side has fewer than 4 nodes.  If a half
    is disconnected, the largest component of that half is the source or
    sink.  Returns 0 when the side has no measurable internal throughput.
    ``positions`` is unused; kept so existing call sites stay valid.
    """

    del positions
    members = np.asarray(members, dtype=int)
    if members.size < 2:
        return 0.0
    adj = _induced_adjacency(graph, members)
    mask = _intrinsic_split_mask(adj)
    if mask is None:
        return 0.0
    local_a = _largest_component(adj, np.where(mask)[0])
    local_b = _largest_component(adj, np.where(~mask)[0])
    if local_a.size == 0 or local_b.size == 0:
        return 0.0
    return _set_maxflow(
        graph, n, members[local_a].tolist(), members[local_b].tolist(),
    )


def _flow_bottleneck_ratio(
    scaffold: Any,
    labels: np.ndarray,
    positions: np.ndarray,
) -> float:
    """Worst pairwise cross-cut max-flow over internal bisection flow.

    Dimensionless arc-vs-valley discriminant (SI S2.6.2): for a sampling-gap
    arc cut of a connected manifold the cross-cut max-flow matches either
    side's own internal throughput (ratio near 1), while a true density
    valley carries near-zero cross flow.  Returns ``inf`` when a block has
    no measurable internal throughput, which rejects the cut.
    """

    n = int(labels.shape[0])
    graph = _flow_graph(scaffold)
    keys = sorted(set(int(v) for v in labels if v >= 0))
    blocks = [np.where(labels == key)[0] for key in keys]
    internal = [
        _bisection_flow(graph, n, members, positions) for members in blocks
    ]
    worst = 0.0
    for a in range(len(blocks)):
        for b in range(a + 1, len(blocks)):
            cross = _set_maxflow(
                graph, n, blocks[a].tolist(), blocks[b].tolist(),
            )
            denom = min(internal[a], internal[b])
            if denom <= 0.0:
                return float("inf")
            worst = max(worst, cross / denom)
    return worst


_NULL_CUT_RANDOM = 16
"""Random geodesic seed count for the one-feature null φ_0 estimator."""

_NULL_CUT_EIGS = 3
"""Non-trivial Laplacian modes used as intrinsic null bisections."""


def mean_neighbor_radius(points: np.ndarray, k: int) -> float:
    """Mean ``k``-neighbour radius of a point set (SI S2.6.2 ``τ_shot``)."""

    arr = np.asarray(points, dtype=float)
    n = int(arr.shape[0])
    if n < 2:
        return float("inf")
    k_use = max(1, min(int(k), n - 1))
    dists, _ = cKDTree(arr).query(arr, k=k_use + 1)
    return float(np.asarray(dists[:, -1], dtype=float).mean())


def shot_noise_node_cap(n_samples: int, k: int, min_nodes: int = 1) -> int:
    """Mesh resolution bound ``N ≤ n/k`` (SI S2.6.2 / OPEN_ISSUES #48).

    A node's catchment must hold at least the ``k`` samples that define
    the local density scale everywhere else in the reader, i.e. mean
    samples per node ``n/N ≥ k``, equivalently ``N ≤ n/k``.  Derived
    from the reader's own neighbour count ``k``; no calibrated constant.
    """

    return max(int(min_nodes), int(n_samples) // max(int(k), 1))


def at_shot_noise_scale(
    scaffold: Any,
    data: np.ndarray,
    k: int,
) -> bool:
    """True when mesh resolution has reached the bound ``N ≥ n/k``.

    The meeting point is the catchment condition: each node must hold at
    least the ``k`` samples that define the local density scale, i.e.
    ``n/N ≥ k``, equivalently the resolution bound ``N ≤ n/k``.  The
    previous node-``r_k``-vs-sample-``r_k`` comparison was spacing
    equality at ``N ≈ n`` and never fired before ``max_nodes``
    (SI S2.6.2 / OPEN_ISSUES #48).
    """

    n_nodes = len(getattr(scaffold, "nodes", ()))
    if n_nodes < 2:
        return False
    return n_nodes >= shot_noise_node_cap(
        int(np.asarray(data).shape[0]), k, 1,
    )


def _two_set_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Max label-flip accuracy on indices that are signal in both cuts."""

    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    mask = (a >= 0) & (b >= 0)
    if int(np.sum(mask)) < 2:
        return 1.0
    a_s = a[mask]
    b_s = b[mask]
    ua = sorted(set(int(v) for v in a_s))
    ub = sorted(set(int(v) for v in b_s))
    if len(ua) < 2 or len(ub) < 2:
        return 1.0
    a2 = (a_s == ua[0]).astype(int)
    b2 = (b_s == ub[0]).astype(int)
    same = float(np.mean(a2 == b2))
    return max(same, 1.0 - same)


def _hyperplane_cut_labels(
    positions: np.ndarray,
    signal: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    labels = np.full(int(positions.shape[0]), -1, dtype=int)
    pts = positions[signal]
    if pts.shape[0] < 2:
        return labels
    axis = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        return labels
    proj = pts @ (axis / norm)
    median = float(np.median(proj))
    side = (proj > median).astype(int)
    if int(side.min()) == int(side.max()):
        order = np.argsort(proj)
        side = np.zeros(pts.shape[0], dtype=int)
        side[order[pts.shape[0] // 2:]] = 1
    labels[signal] = side
    return labels


def _null_cut_directions(
    dim: int,
    rng: np.random.Generator,
    n_random: int = _NULL_CUT_RANDOM,
) -> list[np.ndarray]:
    """Legacy ambient directions (#48 contrast / tests)."""

    directions: list[np.ndarray] = []
    for i in range(max(int(dim), 1)):
        axis = np.zeros(dim, dtype=float)
        axis[i] = 1.0
        directions.append(axis)
    for _ in range(max(int(n_random), 0)):
        vec = rng.normal(size=dim)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            directions.append(vec / norm)
    return directions


def _labels_from_local_mask(
    n: int,
    members: np.ndarray,
    local_mask: np.ndarray,
) -> np.ndarray:
    labels = np.full(int(n), -1, dtype=int)
    side = np.zeros(int(members.shape[0]), dtype=int)
    side[local_mask] = 1
    labels[members] = side
    return labels


def _iter_intrinsic_null_cuts(
    adj: csr_matrix,
    n: int,
    members: np.ndarray,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Connected intrinsic bisections of the signal-induced flow graph.

    (a) median splits of the first ``_NULL_CUT_EIGS`` non-trivial
    normalized-Laplacian eigenvectors; (b) hop-count geodesic median
    splits from ``_NULL_CUT_RANDOM`` random seeds.  A cut is kept only
    when both sides are connected in the induced subgraph.
    """

    cuts: list[np.ndarray] = []
    m = int(adj.shape[0])
    if m < 2:
        return cuts
    vecs = _normalized_laplacian_vectors(adj, _NULL_CUT_EIGS)
    if vecs is not None:
        for k in range(int(vecs.shape[1])):
            mask = _median_split_mask(vecs[:, k])
            if mask is None or not _both_sides_connected(adj, mask):
                continue
            cuts.append(_labels_from_local_mask(n, members, mask))
    n_seeds = min(int(_NULL_CUT_RANDOM), m)
    if n_seeds > 0:
        seeds = rng.choice(m, size=n_seeds, replace=False)
        for seed in seeds:
            mask = _median_split_mask(_hop_distances(adj, int(seed)))
            if mask is None or not _both_sides_connected(adj, mask):
                continue
            cuts.append(_labels_from_local_mask(n, members, mask))
    return cuts


def null_bottleneck_ratio(
    scaffold: Any,
    positions: np.ndarray,
    candidate_labels: np.ndarray,
    rng: np.random.Generator | None = None,
) -> float | None:
    """Typical one-feature ``φ``: median bottleneck of disagreeing cuts.

    Null cuts are intrinsic bisections of the signal-induced flow graph:
    median splits of the first ``_NULL_CUT_EIGS`` non-trivial
    normalized-Laplacian eigenvectors, plus hop-count graph-geodesic
    median splits from ``_NULL_CUT_RANDOM`` random seed nodes.

    Hop count (not ``1/weight``) is the geodesic length so that ``φ_0``
    estimates the typical ``φ`` of a *geometric* connected cut.  Density-
    weighted lengths would systematically recover a true valley — the
    Fiedler cut of the weighted graph already does that, and the
    agreement filter drops it.  Cuts whose sides are not each connected
    in the induced subgraph are discarded.  Cuts that recreate the
    candidate (agreement ``≥ 0.5`` after a label flip) are dropped so a
    true valley that coincides with the Fiedler cut does not contaminate
    the null.  Returns ``None`` when the pool is empty.  SI S2.6.2 / #48.
    """

    rng = rng if rng is not None else np.random.default_rng(0)
    labels = np.asarray(candidate_labels)
    members = np.where(labels >= 0)[0]
    if members.size < 4:
        return None
    n = int(labels.shape[0])
    graph = _flow_graph(scaffold)
    adj = _induced_adjacency(graph, members)
    pool: list[float] = []
    agreeing: list[tuple[float, float]] = []
    for cut in _iter_intrinsic_null_cuts(adj, n, members, rng):
        signal = cut[members]
        if len(set(int(v) for v in signal)) < 2:
            continue
        phi = _flow_bottleneck_ratio(scaffold, cut, positions)
        if not np.isfinite(phi) or phi < 0.0:
            continue
        agree = _two_set_agreement(labels, cut)
        if agree <= 0.5:
            pool.append(float(phi))
        else:
            agreeing.append((agree, float(phi)))
    if not pool and agreeing:
        agreeing.sort(reverse=True)
        leftover = [phi for _, phi in agreeing[1:]]
        pool = leftover
    if not pool:
        return None
    return float(np.median(np.asarray(pool, dtype=float)))


def studentized_bottleneck(
    phi_candidate: float,
    phi_null: float | None,
) -> float | None:
    """``φ / φ_0``; ``None`` when the null scale is unusable."""

    if phi_null is None or (not np.isfinite(phi_null)) or float(phi_null) <= 0.0:
        return None
    if not np.isfinite(phi_candidate):
        return float("inf")
    return float(phi_candidate) / float(phi_null)


def _label_sets(labels: np.ndarray) -> tuple[list[set[int]], set[int]]:
    clusters = [
        set(np.where(labels == label)[0].tolist())
        for label in sorted(set(int(v) for v in labels if v >= 0))
    ]
    background = set(np.where(labels < 0)[0].tolist())
    return clusters, background


def _branch_alive(branch: LevelSetBranch, level: int) -> bool:
    if branch.birth_level > level:
        return False
    if branch.merge_level is None:
        return True
    return level < int(branch.merge_level)


def _branch_nodes_at(
    tree: LevelSetTree,
    dag: LevelSetDAG,
    branch_id: int,
    level: int,
) -> set[int]:
    if not (0 <= level < len(dag.level_branch_ids)):
        return set()
    ids = dag.level_branch_ids[level]
    labels = tree.levels[level].labels
    for cluster_id, bid in enumerate(ids):
        if int(bid) == int(branch_id):
            return set(np.where(labels == cluster_id)[0].tolist())
    return set()


def _labels_from_branches(
    tree: LevelSetTree,
    dag: LevelSetDAG,
    alive: list[LevelSetBranch],
    level: int,
) -> np.ndarray:
    n = int(tree.levels[level].labels.shape[0])
    labels = np.full(n, -1, dtype=int)
    for new_id, branch in enumerate(alive):
        for node in _branch_nodes_at(tree, dag, branch.branch_id, level):
            labels[int(node)] = new_id
    return labels


def apply_dm_sibling_collapse(
    tree: LevelSetTree,
    dag: LevelSetDAG,
    branches: tuple[LevelSetBranch, ...],
    scaffold: Any,
    dm_config: DMClusterConfig,
) -> tuple[LevelSetBranch, ...]:
    """Collapse geometrically surviving siblings that fail the DM split test."""

    by_id = {int(b.branch_id): b for b in branches}
    parents = [
        b for b in branches
        if b.child_ids and b.prune_reason is None
    ]
    parents.sort(
        key=lambda p: min(
            (
                int(by_id[c].merge_level)
                for c in p.child_ids
                if c in by_id and by_id[c].merge_level is not None
            ),
            default=0,
        ),
    )
    for parent in parents:
        parent = by_id[parent.branch_id]
        if parent.prune_reason is not None:
            continue
        kids = [
            by_id[c] for c in parent.child_ids
            if c in by_id and by_id[c].prune_reason is None
        ]
        if not kids:
            continue
        merge_at = min(
            int(k.merge_level) for k in kids if k.merge_level is not None
        ) if any(k.merge_level is not None for k in kids) else None
        if merge_at is None or merge_at < 1:
            continue
        level = merge_at - 1
        group = [parent, *kids]
        clusters = [
            _branch_nodes_at(tree, dag, g.branch_id, level) for g in group
        ]
        clusters = [c for c in clusters if c]
        if len(clusters) < 2:
            continue
        assigned: set[int] = set()
        for cluster in clusters:
            assigned |= cluster
        background = set(range(int(tree.levels[level].labels.shape[0]))) - assigned
        _log_bf, accepted = dm_partition_background_verdict(
            scaffold, clusters, background, dm_config,
        )
        if accepted:
            continue
        for kid in kids:
            by_id[kid.branch_id] = replace(kid, prune_reason="dm")
    return tuple(by_id[k] for k in sorted(by_id))


def _cluster_result_from_labels(
    scaffold: Any,
    labels: np.ndarray,
) -> ClusterResult:
    clusters, _background = _label_sets(labels)
    n = int(labels.shape[0])
    hits = np.asarray(
        [float(node.hit_count) for node in scaffold.nodes],
        dtype=float,
    )
    exemplars = np.asarray(
        [int(max(c, key=lambda node_id: hits[node_id])) for c in clusters],
        dtype=int,
    ) if clusters else np.empty(0, dtype=int)
    graph_lifted = scaffold.links.neighbour_graph(n)
    weights = compute_edge_weights(scaffold)
    q_value = partition_q_score(clusters, n, weights, graph_lifted) if clusters else 0.0
    return ClusterResult(
        labels=labels.copy(),
        exemplar_indices=exemplars,
        n_clusters=len(clusters),
        partition_q_score=float(q_value),
    )


def select_level_set_partition(
    scaffold: Any,
    config: LevelSetConfig | None = None,
    dm_config: DMClusterConfig | None = None,
    data: np.ndarray | None = None,
) -> LevelSetSelection:
    """Select the coarsest evidence-bearing split (SI S2.6.2).

    Pipeline: C-D tree → merge DAG (diagnostics + sibling collapse) →
    coarsest *mass-filtered* ``K >= 2`` cut → flow-bottleneck guard →
    studentized one-feature floor → background-aware DM.  Levels whose
    cut collapses below ``K = 2`` after
    the relative-mass floor (satellite-only structure) are skipped, so a
    balanced mid-tree cut — nested shells or linked tori whose valley is
    tissue-bridged before the coarse tail — is reachable.  The single
    candidate is that coarsest filtered cut: if it fails the bottleneck
    guard or DM, the region is rejected outright, never walked finer.  No
    expected cluster count enters selection.  The returned
    ``resolvability`` trichotomy tells the orchestrator whether to grow
    ``N`` (under-resolved at the node cap) or not (resolved null).
    """

    config = config or LevelSetConfig()
    dm_config = dm_config or DMClusterConfig()
    positions = np.asarray(
        [node.position for node in scaffold.nodes],
        dtype=float,
    )
    at_shot = False
    if data is not None:
        at_shot = at_shot_noise_scale(scaffold, data, config.k_neighbors)
    tree = build_level_set_tree(positions, config)
    if not tree.levels:
        return LevelSetSelection(
            tree,
            None,
            None,
            float("-inf"),
            resolvability=assess_valley_resolvability(
                scaffold,
                accepted=False,
                saw_balanced_cut=False,
                reject_reason="no_cut",
                at_shot_floor=at_shot,
            ),
        )

    dag = build_level_set_dag(tree)
    screened = apply_geometric_screens(dag.branches, config)
    screened = apply_dm_sibling_collapse(
        tree, dag, screened, scaffold, dm_config,
    )

    best_rejected_bf = float("-inf")
    saw_balanced_cut = False
    reject_reason: str | None = "no_cut"
    candidate_level: int | None = None
    bottleneck_ratio: float | None = None
    phi_null: float | None = None
    phi_rel: float | None = None
    for level_index in range(len(tree.levels) - 1, -1, -1):
        if tree.levels[level_index].n_clusters < 2:
            continue
        labels = _filter_relative_mass(
            tree.levels[level_index].labels,
            config.min_cluster_size,
            config.min_cluster_frac,
        )
        clusters, background = _label_sets(labels)
        if len(clusters) < 2:
            # Satellite-only structure at this level; a genuine balanced
            # cut may still exist finer (tissue-bridged nested features).
            continue
        saw_balanced_cut = True
        candidate_level = level_index
        # Coarse-anchor: this is the coarsest mass-filtered K>=2 cut and
        # the only candidate.  Guard against sampling-gap arcs, studentize
        # against the local one-feature null, then confirm with DM.
        ratio = _flow_bottleneck_ratio(scaffold, labels, positions)
        bottleneck_ratio = float(ratio)
        phi_null = null_bottleneck_ratio(scaffold, positions, labels)
        phi_rel = studentized_bottleneck(float(ratio), phi_null)
        if ratio > config.max_bottleneck_ratio:
            reject_reason = "bottleneck"
            break
        if phi_rel is not None and phi_rel > config.max_bottleneck_ratio:
            reject_reason = "one_feature_null"
            break
        log_bf, accepted = dm_partition_background_verdict(
            scaffold, clusters, background, dm_config,
        )
        best_rejected_bf = max(best_rejected_bf, float(log_bf))
        if accepted:
            result = _cluster_result_from_labels(scaffold, labels)
            return LevelSetSelection(
                tree=tree,
                cluster_result=result,
                selected_level=level_index,
                log_bf=float(log_bf),
                dag=replace(dag, branches=screened),
                branches=screened,
                resolvability=assess_valley_resolvability(
                    scaffold,
                    accepted=True,
                    saw_balanced_cut=True,
                    at_shot_floor=at_shot,
                ),
                candidate_level=level_index,
                bottleneck_ratio=float(ratio),
                null_bottleneck_ratio=phi_null,
                studentized_ratio=phi_rel,
            )
        reject_reason = "dm"
        break

    return LevelSetSelection(
        tree=tree,
        cluster_result=None,
        selected_level=None,
        log_bf=best_rejected_bf,
        dag=replace(dag, branches=screened),
        branches=screened,
        resolvability=assess_valley_resolvability(
            scaffold,
            accepted=False,
            saw_balanced_cut=saw_balanced_cut,
            reject_reason=reject_reason,
            at_shot_floor=at_shot,
        ),
        candidate_level=candidate_level,
        bottleneck_ratio=bottleneck_ratio,
        null_bottleneck_ratio=phi_null,
        studentized_ratio=phi_rel,
    )
