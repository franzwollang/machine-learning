"""Recursive scale-discovery orchestrator for Stage 1 (SI S2.5, S4.4, S2.6.2)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np

from proteus.stage1.clustering import (
    ClusterResult,
    _lifted_components_covering_all_nodes,
    compute_edge_weights,
    partition_q_score,
    run_clustering,
)
from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.dm_cluster import (
    DMClusterConfig,
    dm_partition_verdict,
    run_clustering_dm,
)
from proteus.stage1.pruning import demote_lifted_by_cluster
from proteus.stage1.transfer import apply_t2_transfer


@dataclass
class RecursionConfig:
    """Configuration for the recursive scale-discovery loop.

    ``require_persistent_split`` turns on the SI S2.6.2 **feature-persistence
    accept-gate**: a region's proposed multi-cluster split is accepted only if a
    multi-cluster partition *persists* across adjacent ``tau`` grid points (the
    cross-scale arbiter), rather than trusting a single-scale partition alone.
    When enabled the scale search is forced to record partitions and select the
    persistence scale (``selector="persistence"``); a region whose split does not
    persist is treated as terminal (a single intrinsic feature).  This is the
    *intended* canonical replacement for the single-scale cleanup heuristics of
    S2.6.1 (OPEN_ISSUES #27).  The persistence rule is **coarse-anchored**
    (S2.6.2), which removes the warm-start fine-end false positive: with the
    S2.6.1 stand-ins removed and this gate on, the circle collapses to a single
    leaf and the hierarchical Gaussian keeps its six leaves.  It is **not yet safe
    to delete those stand-ins**, however: a developable manifold (swiss roll)
    whose coarsest partition is a few arcs still over-fragments under the gate
    alone, pending the Stage 2 DM evidence gate (S3.4, M4).  See the "residual
    limitation" note in SI S2.6.2.  Default off during the M2 transition.

    ``require_dm_split`` turns on the **DM cluster-acceptance path** (SI S3.4
    reduction, proposed S2.6.3, OPEN_ISSUES #27): AP proposals are merged by
    the Dirichlet--multinomial block-homogeneity Bayes factor
    (:func:`proteus.stage1.dm_cluster.run_clustering_dm`) instead of the
    S2.6.1 single-scale cleanup stand-ins, and a region's split is retained
    only if the K-way block partition clears the ``log(tau_bf)`` margin
    against the one-feature null -- the non-degenerate likelihood-ratio null
    the graph-local ``Q`` cannot provide.  Independent of and complementary to
    persistence (the two flags may be combined).  It is a **proposed /
    operational** path pending validation, and is **not** a licence to delete
    the S2.6.1 stand-ins.  Default off.

    ``allow_finer_research`` (OPEN_ISSUES #44, **proposed / operational,
    default off**) enables a single finer-than-``tau*`` re-search when the
    region's characteristic-scale partition is ``K<=1`` (or ``Q<=0``), or when
    persistence finds no split at the coarse grid.  Without this flag the
    orchestrator treats such a region as terminal and never probes scales where
    disconnected sub-structure (nested shells, linked tori) becomes visible.
    When enabled, the re-search caps ``tau_max`` at
    ``tau_star * finer_tau_cap_ratio`` (strictly ``< tau_star``) and reuses the
    existing persistence / DM accept gates on the finer proposal.  **Stop
    guarantee:** at most one finer attempt per region invocation; if the
    capped search still yields ``K<=1`` / ``Q<=0`` or the gate rejects, the
    region is terminal; ``min_samples`` / ``max_depth`` still bound the tree.

    **Recommended pairing for uniform manifolds (required):** enable
    ``require_persistent_split`` together with ``allow_finer_research``, keep
    ``max_finer_scale_steps <= 4``, and ``min_samples >= 80``.  A2-T3 measured
    circle/swiss = 1 leaf under that pairing; flag alone over-fragments the
    circle (~21 leaves).  A2-T4: the same recommended pairing still yields
    **1 leaf** on nested_spheres / linked_tori (gt cc=2) at n≈160–240 — shell
    recovery is **not** solved by this pairing; deeper walks / prepass tend to
    over-fragment with near-zero ARI.  Do not flip awaiting component tests.
    Optional SI prose for gate-owns-stop + persist pairing: see A3 mailbox
    Opt A+C sketch (REQUEST_TRACKER).

    ``max_finer_scale_steps`` bounds how many successive geometric shrinks of
    the ``tau_max`` cap are attempted inside one region (each step multiplies
    the cap by ``finer_tau_cap_ratio``).  Nested multi-component scenes can
    require many grid steps below the coarse ``tau*`` (OPEN_ISSUES #44
    evidence: ~80x).  Default ``8`` is an operational budget; the flag remains
    off so the default acceptance path is unchanged.

    ``prefer_disconnected_prepass`` (OPEN_ISSUES #44c, **proposed /
    operational, default off**) short-circuits the finer re-search walk when
    the lifted Hebbian graph at a capped scale has **≥2 major connected
    components** (each at least ``finer_prepass_min_frac`` of the scaffold
    nodes, and at least 3 nodes).  Tiny components are absorbed into the
    nearest major by Euclidean position.  Zero inter-component lifted edges
    imply block-diagonal flow, so this is the cheap obvious-disconnect path
    before the general AP/DM finer search.  Pair with a modest
    ``max_finer_scale_steps`` (and preferably ``require_persistent_split``)
    so uniform manifolds that only fracture at extreme fine scales do not
    trigger a false prepass hit.
    """

    scale_search: ScaleSearchConfig = field(default_factory=ScaleSearchConfig)
    min_samples: int = 100
    max_depth: int = 5
    r_min: int = 3
    explained_energy: float = 0.999
    require_persistent_split: bool = False
    require_dm_split: bool = False
    dm_cluster: DMClusterConfig = field(default_factory=DMClusterConfig)
    allow_finer_research: bool = False
    finer_tau_cap_ratio: float = 1.0 / np.sqrt(2.0)
    max_finer_scale_steps: int = 8
    prefer_disconnected_prepass: bool = False
    finer_prepass_min_frac: float = 0.2
    seed: int = 42


@dataclass
class RecursionNode:
    """A node in the recursion tree.

    ``sample_indices`` lists original dataset row indices belonging to this
    region (same order as rows of ``data`` passed into the frame that created
    the node); leaves partition ``0 .. n-1`` at the root.
    """

    region_id: int
    level: int
    parent_id: Optional[int]
    tau_star: Optional[float]
    n_samples: int
    dim: int
    n_clusters: int
    children: list[int] = field(default_factory=list)
    is_leaf: bool = True
    sample_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=int),
    )


@dataclass
class RecursionTree:
    """The full recursion tree produced by recursive discovery."""

    nodes: list[RecursionNode] = field(default_factory=list)

    @property
    def depth(self) -> int:
        if not self.nodes:
            return 0
        return max(n.level for n in self.nodes) + 1

    @property
    def leaves(self) -> list[RecursionNode]:
        return [n for n in self.nodes if n.is_leaf]


def _clusters_from_labels(labels: np.ndarray) -> list[set[int]]:
    """Group node ids by non-negative cluster label into member sets."""

    groups: dict[int, set[int]] = {}
    for node_id, lbl in enumerate(np.asarray(labels, dtype=int)):
        if lbl < 0:
            continue
        groups.setdefault(int(lbl), set()).add(int(node_id))
    return [groups[k] for k in sorted(groups)]


def assign_samples_to_clusters(
    data: np.ndarray,
    scaffold: "Stage1Scaffold",  # noqa: F821
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Map each data sample to its BMU's cluster label.

    Returns a dict mapping cluster label -> array of sample indices.
    """

    n_samples = data.shape[0]
    cluster_indices: dict[int, list[int]] = {}
    for i in range(n_samples):
        bmu_ids, _ = scaffold.ann.query_knn(data[i], k=1)
        bmu = int(bmu_ids[0])
        label = int(labels[bmu])
        cluster_indices.setdefault(label, []).append(i)
    return {k: np.array(v, dtype=int) for k, v in cluster_indices.items()}


def _cluster_scaffold(
    scaffold: "Stage1Scaffold",  # noqa: F821
    config: RecursionConfig,
):
    """Run AP or DM clustering on ``scaffold`` per config flags."""

    if config.require_dm_split:
        return run_clustering_dm(scaffold, config.dm_cluster)
    return run_clustering(scaffold)


def _major_lifted_component_partition(
    scaffold: "Stage1Scaffold",  # noqa: F821
    *,
    min_frac: float = 0.2,
    min_abs: int = 3,
) -> ClusterResult | None:
    """Return a partition from ≥2 major lifted components, else ``None``.

    OPEN_ISSUES #44c cheap prepass: when the lifted graph already separates
    into multiple large components, skip AP and treat those components as the
    candidate split.  Components smaller than
    ``max(min_abs, ceil(n * min_frac))`` are absorbed into the nearest major
    component by Euclidean node position.  Requires ``Q > 0`` on the resulting
    partition.
    """

    n = len(scaffold.nodes)
    if n < 2:
        return None
    frac = float(min_frac)
    if not (0.0 < frac <= 0.5):
        frac = 0.2
    threshold = max(int(min_abs), int(np.ceil(n * frac)))

    graph_lifted = scaffold.links.neighbour_graph(n)
    comps = _lifted_components_covering_all_nodes(n, graph_lifted)
    majors = [c for c in comps if len(c) >= threshold]
    if len(majors) < 2:
        return None

    positions = np.asarray(
        [scaffold.nodes[i].position for i in range(n)], dtype=float,
    )
    major_centroids = [
        positions[sorted(c)].mean(axis=0) for c in majors
    ]

    labels = np.full(n, -1, dtype=int)
    for cid, members in enumerate(majors):
        for m in members:
            labels[int(m)] = cid

    # Absorb non-major nodes into nearest major by centroid distance.
    leftovers = [c for c in comps if len(c) < threshold]
    for tiny in leftovers:
        for m in tiny:
            dists = [
                float(np.sum((positions[int(m)] - cen) ** 2))
                for cen in major_centroids
            ]
            labels[int(m)] = int(np.argmin(dists))

    clusters: list[set[int]] = [
        set(np.where(labels == cid)[0].tolist()) for cid in range(len(majors))
    ]
    # Drop empties (should not happen after absorb).
    clusters = [c for c in clusters if c]
    if len(clusters) < 2:
        return None

    W = compute_edge_weights(scaffold)
    pq = partition_q_score(clusters, n, W, graph_lifted)
    if pq <= 0.0:
        return None

    hits = np.array([node.hit_count for node in scaffold.nodes], dtype=float)
    exemplars = [int(max(c, key=lambda g: hits[g])) for c in clusters]
    # Relabel densely 0..K-1
    dense = np.full(n, -1, dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            dense[int(m)] = cid

    return ClusterResult(
        labels=dense,
        exemplar_indices=np.array(exemplars, dtype=int),
        n_clusters=len(clusters),
        partition_q_score=float(pq),
    )


def _research_finer_split(
    data: np.ndarray,
    dim: int,
    config: RecursionConfig,
    parent_tau: float,
):
    """Capped multi-step scale re-search strictly finer than ``parent_tau`` (#44).

    Walks ``tau_max`` downward geometrically (``finer_tau_cap_ratio`` per step,
    at most ``max_finer_scale_steps`` times) until a multi-cluster proposal with
    ``Q > 0`` appears, else returns ``None``.  Does not recurse into children;
    the caller applies accept gates and child descent.  **Stop guarantee:**
    step budget + ``tau_min`` bound the walk; a failed / gated-out proposal
    leaves the region terminal.

    When ``prefer_disconnected_prepass`` is on, each step first tries the
    major-lifted-component short-circuit (#44c) before the general AP/DM path.
    """

    ratio = float(config.finer_tau_cap_ratio)
    if not (0.0 < ratio < 1.0):
        ratio = float(config.scale_search.grid_ratio)
    tau_min = float(config.scale_search.tau_min)
    tau_cap = float(parent_tau) * ratio
    max_steps = max(1, int(config.max_finer_scale_steps))

    for _step in range(max_steps):
        if not (tau_min < tau_cap < float(parent_tau)):
            return None

        scale_search_config = replace(config.scale_search, tau_max=tau_cap)
        if config.require_persistent_split:
            scale_search_config = replace(
                scale_search_config,
                selector="persistence",
                record_partitions=True,
            )

        result = run_scale_search(data, dim, scale_search_config)
        scaffold = result.scaffold_at_star
        if scaffold is None or len(scaffold.nodes) < 2:
            tau_cap *= ratio
            continue

        if config.require_persistent_split:
            persistence = result.persistence_result
            if persistence is None or persistence.tau_star_index is None:
                tau_cap *= ratio
                continue

        # #44c: cheap disconnected-lifted prepass before general clustering.
        if config.prefer_disconnected_prepass:
            pre = _major_lifted_component_partition(
                scaffold,
                min_frac=float(config.finer_prepass_min_frac),
            )
            if (
                pre is not None
                and pre.n_clusters > 1
                and pre.partition_q_score > 0.0
            ):
                return result, scaffold, pre

        cluster_result = _cluster_scaffold(scaffold, config)
        if (
            cluster_result.n_clusters > 1
            and cluster_result.partition_q_score > 0.0
        ):
            return result, scaffold, cluster_result

        tau_cap *= ratio

    return None


def _dm_accepts_split(
    scaffold: "Stage1Scaffold",  # noqa: F821
    cluster_result: Any,
    config: RecursionConfig,
) -> bool:
    """True if DM gate is off, or the K-way partition clears ``log(tau_bf)``."""

    if not config.require_dm_split:
        return True
    clusters = _clusters_from_labels(cluster_result.labels)
    _log_bf, accepted = dm_partition_verdict(
        scaffold, clusters, config.dm_cluster,
    )
    return bool(accepted)


def _descend_into_clusters(
    *,
    data_arr: np.ndarray,
    dim: int,
    config: RecursionConfig,
    tree: RecursionTree,
    node: RecursionNode,
    region_id: int,
    _level: int,
    orig_rows: np.ndarray,
    scaffold: "Stage1Scaffold",  # noqa: F821
    cluster_result: Any,
) -> RecursionTree:
    """Demote, map samples, create children, and recurse."""

    demote_lifted_by_cluster(
        scaffold, cluster_result.labels,
        beta=float(getattr(scaffold, "prune_beta", 0.5)),
    )

    sample_map = assign_samples_to_clusters(
        data_arr, scaffold, cluster_result.labels,
    )

    children_created: list[int] = []
    for label, child_indices in sorted(sample_map.items()):
        if len(child_indices) < config.min_samples:
            child_id = len(tree.nodes)
            global_child = orig_rows[np.asarray(child_indices, dtype=int)]
            tree.nodes.append(RecursionNode(
                region_id=child_id,
                level=_level + 1,
                parent_id=region_id,
                tau_star=None,
                n_samples=len(child_indices),
                dim=dim,
                n_clusters=0,
                is_leaf=True,
                sample_indices=global_child.copy(),
            ))
            children_created.append(child_id)
            continue

        cluster_node_ids = np.where(cluster_result.labels == label)[0]
        # Operationally d_final == working dim (SI S1.4.1 refresh semantics),
        # so d_hat reduces to the region working dim unless refreshed.
        d_finals = [scaffold.nodes[int(i)].d_final for i in cluster_node_ids]
        d_hat = int(np.median(d_finals)) if d_finals else dim

        t2 = apply_t2_transfer(
            data_arr, child_indices, dim, d_hat,
            r_min=config.r_min,
            explained_energy=config.explained_energy,
        )

        child_config = RecursionConfig(
            scale_search=config.scale_search,
            min_samples=config.min_samples,
            max_depth=config.max_depth,
            r_min=config.r_min,
            explained_energy=config.explained_energy,
            require_persistent_split=config.require_persistent_split,
            require_dm_split=config.require_dm_split,
            dm_cluster=config.dm_cluster,
            allow_finer_research=config.allow_finer_research,
            finer_tau_cap_ratio=config.finer_tau_cap_ratio,
            max_finer_scale_steps=config.max_finer_scale_steps,
            prefer_disconnected_prepass=config.prefer_disconnected_prepass,
            finer_prepass_min_frac=config.finer_prepass_min_frac,
            seed=config.seed + region_id + label,
        )

        child_id = len(tree.nodes)
        children_created.append(child_id)

        child_orig_globals = orig_rows[np.asarray(child_indices, dtype=int)]

        run_recursive_discovery(
            t2.child_data,
            t2.child_dim,
            child_config,
            _level=_level + 1,
            _parent_id=region_id,
            _tree=tree,
            _sample_indices=child_orig_globals,
        )

    node.children = children_created
    if children_created:
        node.is_leaf = False

    return tree


def run_recursive_discovery(
    data: np.ndarray,
    dim: int,
    config: RecursionConfig | None = None,
    *,
    _level: int = 0,
    _parent_id: int | None = None,
    _tree: RecursionTree | None = None,
    _sample_indices: np.ndarray | None = None,
) -> RecursionTree:
    """Recursively discover scale structure via scale search + clustering + T2.

    At each level: run scale search to find tau_star, cluster the
    converged scaffold via Q-score seed merging, and for each cluster
    with enough samples apply the T2 PCA transfer and recurse into the
    child.  Recursion is gated by Q(P; v) > 0 on the proposed partition.
    """

    config = config if config is not None else RecursionConfig()
    tree = _tree if _tree is not None else RecursionTree()
    data_arr = np.asarray(data, dtype=float)
    n_samples = data_arr.shape[0]

    if _sample_indices is None:
        orig_rows = np.arange(n_samples, dtype=int)
    else:
        orig_rows = np.asarray(_sample_indices, dtype=int, copy=True)
    if orig_rows.shape[0] != n_samples:
        raise ValueError(
            "_sample_indices length must match data row count "
            f"({orig_rows.shape[0]} vs {n_samples})",
        )

    region_id = len(tree.nodes)
    node = RecursionNode(
        region_id=region_id,
        level=_level,
        parent_id=_parent_id,
        tau_star=None,
        n_samples=n_samples,
        dim=dim,
        n_clusters=0,
        is_leaf=True,
        sample_indices=orig_rows.copy(),
    )
    tree.nodes.append(node)

    if n_samples < config.min_samples:
        return tree

    if _level >= config.max_depth:
        return tree

    scale_search_config = config.scale_search
    if config.require_persistent_split:
        # The persistence accept-gate (SI S2.6.2) needs the per-grid-point
        # partitions and the persistence-selected characteristic scale.
        scale_search_config = replace(
            scale_search_config,
            selector="persistence",
            record_partitions=True,
        )

    result = run_scale_search(data_arr, dim, scale_search_config)
    node.tau_star = result.tau_star
    scaffold = result.scaffold_at_star

    if scaffold is None or len(scaffold.nodes) < 2:
        return tree

    need_finer = False
    cluster_result = None

    if config.require_persistent_split:
        # Accept a split only if a multi-cluster partition persists across
        # adjacent scales; a region with no persistent split is a single
        # intrinsic feature (terminal leaf), regardless of any transient
        # single-scale fragmentation (SI S2.6.2, OPEN_ISSUES #27).
        # OPEN_ISSUES #44: when ``allow_finer_research`` is on, defer the
        # terminal decision so a capped finer re-search can still run.
        persistence = result.persistence_result
        if persistence is None or persistence.tau_star_index is None:
            need_finer = True
        else:
            cluster_result = _cluster_scaffold(scaffold, config)
            node.n_clusters = cluster_result.n_clusters
            if (
                cluster_result.n_clusters <= 1
                or cluster_result.partition_q_score <= 0.0
            ):
                need_finer = True
    else:
        cluster_result = _cluster_scaffold(scaffold, config)
        node.n_clusters = cluster_result.n_clusters
        if (
            cluster_result.n_clusters <= 1
            or cluster_result.partition_q_score <= 0.0
        ):
            need_finer = True

    if need_finer:
        # OPEN_ISSUES #44: optionally re-search strictly finer than this
        # region's characteristic tau* before declaring the region terminal.
        if not config.allow_finer_research or node.tau_star is None:
            return tree
        researched = _research_finer_split(
            data_arr, dim, config, float(node.tau_star),
        )
        if researched is None:
            return tree
        # Keep node.tau_star as the coarse characteristic scale (hierarchy
        # root); use the finer scaffold only for the accepted split.
        _finer_result, scaffold, cluster_result = researched
        node.n_clusters = cluster_result.n_clusters

    assert cluster_result is not None

    if not _dm_accepts_split(scaffold, cluster_result, config):
        # Region-level accept gate: retain the multi-cluster split only if it
        # clears the Bayes-factor margin against the one-feature null. A region
        # whose split is not evidence-bearing is a single intrinsic feature
        # (terminal leaf), the non-degenerate likelihood-ratio null that the
        # graph-local Q cannot supply (SI S2.6.1 / S3.4, OPEN_ISSUES #27).
        node.n_clusters = 1
        return tree

    return _descend_into_clusters(
        data_arr=data_arr,
        dim=dim,
        config=config,
        tree=tree,
        node=node,
        region_id=region_id,
        _level=_level,
        orig_rows=orig_rows,
        scaffold=scaffold,
        cluster_result=cluster_result,
    )
