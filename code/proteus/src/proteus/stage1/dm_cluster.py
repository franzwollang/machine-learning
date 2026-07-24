"""DM cluster-acceptance reduction (PROPOSED spec; SI S3.4/S2.6.2).

**Status: proposed / operational, default off.**  The SI defines the
Dirichlet--multinomial (DM) evidence gate (S3.4) for Stage-2 *node/star
topology edits* -- ``J_i(M)`` outgoing outcomes per node, scored on directed
transition counts on affected stars.  S2.6.1 names that gate as the eventual
arbiter of the Stage-1 cluster-count ambiguity (the non-degenerate
likelihood-ratio null that the graph-local ``Q`` cannot supply), but it does
**not** specify how a
partition-into-K decision reduces to an ``F_DM`` comparison.  This module
implements that missing reduction and is documented as OPEN_ISSUES #27 /
proposed SI S2.6.3; it is validated against the S2.6.1 stand-ins and the
persistence gate before any promotion.

Reduction (stochastic-block-model homogeneity test)
---------------------------------------------------
A candidate partition of a region into blocks ``C_1..C_K`` induces a directed
block-flow matrix ``N`` with ``N[k, l] = sum_{a in C_k, b in C_l} n_{a->b}``
from the Hebbian transition counts.  The question "is this region one feature
or K?" becomes a DM test of **homogeneity of the K block-to-block routing
rows** over a fixed ``K``-outcome space (the blocks):

* *keep* (one feature): the K rows share a single routing distribution ->
  score the pooled row ``N.sum(axis=0)``;
* *split* (K features): each block routes with its own distribution -> score
  each row ``N[k]`` separately.

By the standard Dirichlet--multinomial contingency Bayes factor this is exactly
``evaluate_edit`` (SI S3.4) with ``keep = [pooled row]`` and
``edit = [row_0, .., row_{K-1}]``, all over ``J = K`` outcomes and the BDeu
concentration ``alpha_0 = 1/(d_final + 1)`` (S2.7).  The split is accepted iff

    log BF = sum_k log m(N[k]) - log m(N.sum(0)) > log tau_BF,

matching the two-subpopulation fixture of S3.5 (K=2 rows).  The single-cluster
null is *non-degenerate* here (unlike ``Q(P;v) -> +inf``), which is the whole
point of routing cluster acceptance through the DM marginal.

Known limitation (measured; see OPEN_ISSUES #27): at a *single* scale this test
inherits the S2.6.1 under-determination -- a uniform manifold sampled into arcs
has band-concentrated block rows that the homogeneity test can read as
heterogeneous.  It is therefore intended to compose with the cross-scale
persistence signal (S2.6.2), not to replace it single-handedly.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any

import numpy as np

from proteus.evidence.dm_score import bdeu_alpha, node_log_marginal
from proteus.stage1.clustering import (
    ClusterResult,
    _clusters_from_ap_labels,
    _lifted_components_covering_all_nodes,
    _run_ap,
    _smoothed_pmi_similarities,
    compute_edge_weights,
    partition_q_score,
)

__all__ = [
    "DMClusterConfig",
    "block_flow_matrix",
    "dm_partition_logbf",
    "dm_pair_logbf",
    "dm_gated_merge",
    "dm_partition_verdict",
    "run_clustering_dm",
]


@dataclass(frozen=True)
class DMClusterConfig:
    """Configuration for the DM cluster-acceptance reduction (proposed S2.6.3).

    Attributes
    ----------
    tau_bf:
        Bayes-factor threshold; the acceptance margin is ``log(tau_bf)``.
        Shares the S14.3 ``tau_BF`` operational default with the Stage-2 gate
        (S3.6); ``tau_bf = 3.0`` gives ``log 3 ~ 1.10``.
    """

    tau_bf: float = 3.0


def _region_alpha0(scaffold: Any, members: set[int] | None = None) -> float:
    """BDeu concentration for a region (SI S2.7).

    Uses the median ``d_final`` over the region's nodes; ``d_final`` is
    operationally the region working dim (SI S1.4.1), so this is the region
    branching prior ``alpha_0 = 1/(d_final + 1)``.
    """

    nodes = scaffold.nodes
    if members:
        d_vals = [int(nodes[i].d_final) for i in members]
    else:
        d_vals = [int(n.d_final) for n in nodes]
    d_eff = int(np.median(d_vals)) if d_vals else 1
    return bdeu_alpha(max(d_eff, 1))


def block_flow_matrix(scaffold: Any, clusters: list[set[int]]) -> np.ndarray:
    """Directed block-flow ``N[k, l] = sum_{a in C_k, b in C_l} n_{a->b}``.

    Aggregates the directed Hebbian transition counts (shadow + lifted, per SI
    S6/S3.4 "regardless of tier") over the candidate partition. The diagonal
    ``N[k, k]`` is within-block flow.
    """

    k = len(clusters)
    node_block: dict[int, int] = {}
    for b, members in enumerate(clusters):
        for g in members:
            node_block[int(g)] = b
    N = np.zeros((k, k), dtype=float)
    for link in scaffold.links.as_list():
        bi = node_block.get(int(link.i))
        bj = node_block.get(int(link.j))
        if bi is None or bj is None:
            continue
        N[bi, bj] += float(link.count_ij)  # i -> j
        N[bj, bi] += float(link.count_ji)  # j -> i
    return N


def dm_partition_logbf(N: np.ndarray, alpha_0: float) -> float:
    """K-way block-homogeneity log-Bayes-factor (SI S3.4 reduction).

    ``log BF = sum_k log m(N[k]) - log m(N.sum(0))`` over ``J = K`` outcomes.
    Positive means the K-block split is favoured over the one-feature null;
    accept iff it exceeds ``log(tau_bf)``. Returns ``-inf`` for ``K < 2``.
    """

    k = int(N.shape[0])
    if k < 2:
        return float("-inf")
    log_split = float(
        sum(node_log_marginal(N[b], k, alpha_0) for b in range(k))
    )
    log_keep = float(node_log_marginal(N.sum(axis=0), k, alpha_0))
    return log_split - log_keep


def dm_pair_logbf(N: np.ndarray, a: int, b: int, alpha_0: float) -> float:
    """Two-row homogeneity log-BF for blocks ``a`` and ``b`` over the current
    ``K``-outcome space: ``log m(N[a]) + log m(N[b]) - log m(N[a] + N[b])``.

    Low (below ``log tau_bf``) means the two blocks route indistinguishably
    and should be merged; high means their split is evidence-bearing.
    """

    j = int(N.shape[1])
    ra = N[a]
    rb = N[b]
    return float(
        node_log_marginal(ra, j, alpha_0)
        + node_log_marginal(rb, j, alpha_0)
        - node_log_marginal(ra + rb, j, alpha_0)
    )


def dm_partition_verdict(
    scaffold: Any,
    clusters: list[set[int]],
    config: DMClusterConfig | None = None,
) -> tuple[float, bool]:
    """Region-level DM accept verdict for a candidate partition (SI S3.4).

    Returns ``(log_bf, accepted)`` where ``accepted`` is the K-way block
    split clearing the ``log(tau_bf)`` margin against the one-feature null.
    A partition with ``K < 2`` returns ``(-inf, False)`` (nothing to accept).
    """

    config = config or DMClusterConfig()
    live = [set(c) for c in clusters if c]
    if len(live) < 2:
        return float("-inf"), False
    members: set[int] = set()
    for c in live:
        members |= c
    a0 = _region_alpha0(scaffold, members)
    N = block_flow_matrix(scaffold, live)
    log_bf = dm_partition_logbf(N, a0)
    return log_bf, log_bf > float(log(max(config.tau_bf, 1.0)))


def _clusters_adjacent(
    ca: set[int], cb: set[int], graph: dict[int, list[int]],
) -> bool:
    """True iff any node of ``ca`` is graph-adjacent to a node of ``cb``."""

    smaller, larger = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
    for g in smaller:
        for nb in graph.get(g, []):
            if nb in larger:
                return True
    return False


def dm_gated_merge(
    clusters: list[set[int]],
    scaffold: Any,
    alpha_0: float,
    log_tau_bf: float,
    graph: dict[int, list[int]],
) -> list[set[int]]:
    """Agglomerative merge of adjacent blocks whose split lacks evidence.

    The outcome space is **fixed** at the initial fragment blocks: the
    fragment-to-fragment flow matrix ``N`` is computed once, and merging two
    source groups pools their *rows only* (destination columns never contract).
    With the columns fixed, the pairwise homogeneity log-BF
    ``log m(row_a) + log m(row_b) - log m(row_a + row_b)`` equals the exact
    Dirichlet--multinomial edit delta ``F_DM(after) - F_DM(before)`` for that
    merge -- the other groups' rows are unchanged and cancel -- so the greedy
    is a sequence of exact fixed-outcome homogeneity gates rather than a
    contracting surrogate. Each step merges the adjacent pair with the lowest
    log-BF (most indistinguishable) until every remaining adjacent pair clears
    ``log_tau_bf``. Adjacency is evaluated on ``graph`` (which must match the
    tiers scored by :func:`block_flow_matrix`, i.e. the full shadow+lifted
    graph). Proposed replacement for the S2.6.1 cleanup stand-ins (#27).
    """

    fragments = [set(c) for c in clusters if c]
    if len(fragments) <= 1:
        return fragments

    # Fixed outcome space: one column per initial fragment.
    N = block_flow_matrix(scaffold, fragments)
    groups: list[list[int]] = [[k] for k in range(len(fragments))]
    node_sets: list[set[int]] = [set(f) for f in fragments]

    def _row(members: list[int]) -> np.ndarray:
        return N[members].sum(axis=0)

    j = int(N.shape[1])
    while len(groups) > 1:
        rows = [_row(g) for g in groups]
        best_pair: tuple[int, int] | None = None
        best_bf = float("inf")
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if not _clusters_adjacent(node_sets[a], node_sets[b], graph):
                    continue
                bf = float(
                    node_log_marginal(rows[a], j, alpha_0)
                    + node_log_marginal(rows[b], j, alpha_0)
                    - node_log_marginal(rows[a] + rows[b], j, alpha_0)
                )
                if bf < best_bf:
                    best_bf = bf
                    best_pair = (a, b)
        if best_pair is None or best_bf >= log_tau_bf:
            break
        a, b = best_pair
        groups[a].extend(groups[b])
        node_sets[a] |= node_sets[b]
        del groups[b]
        del node_sets[b]
    return node_sets


def _ap_fragments(
    scaffold: Any, comp_list: list[int],
) -> list[set[int]]:
    """AP proposal fragments for one component (no S2.6.1 cleanup passes)."""

    if len(comp_list) == 1:
        return [{comp_list[0]}]
    S, _pref = _smoothed_pmi_similarities(scaffold, comp_list)
    ap_out = _run_ap(S)
    if ap_out is None:
        return [set(comp_list)]
    labels_l, exemplar_idx_l = ap_out
    clusters, _ex = _clusters_from_ap_labels(
        comp_list, labels_l, exemplar_idx_l,
    )
    return clusters


def run_clustering_dm(
    scaffold: Any, config: DMClusterConfig | None = None,
) -> ClusterResult:
    """Cluster a scaffold via AP proposals + DM-gated merge (S3.4 reduction).

    Mirrors :func:`proteus.stage1.clustering.run_clustering` but replaces the
    single-scale cleanup stand-ins (Q-merge / boundary refine / absorb /
    ``<=3`` collapse) with the DM block-homogeneity merge. Default off; used
    only when
    the DM cluster-acceptance path is explicitly selected (OPEN_ISSUES #27).
    """

    config = config or DMClusterConfig()
    n = len(scaffold.nodes)
    if n < 2:
        return ClusterResult(
            labels=np.zeros(n, dtype=int),
            exemplar_indices=np.array([0] if n else [], dtype=int),
            n_clusters=max(n, 0),
            partition_q_score=0.0,
        )

    W = compute_edge_weights(scaffold)
    graph_lifted = scaffold.links.neighbour_graph(n)
    graph_full = scaffold.links.full_graph(n)
    components = _lifted_components_covering_all_nodes(n, graph_lifted)
    hits = np.array([node.hit_count for node in scaffold.nodes], dtype=float)
    log_tau_bf = float(log(max(config.tau_bf, 1.0)))

    all_clusters: list[set[int]] = []
    for comp in components:
        comp_list = sorted(comp)
        frags = _ap_fragments(scaffold, comp_list)
        a0 = _region_alpha0(scaffold, set(comp_list))
        # Adjacency graph must match the tiers scored by block_flow_matrix
        # (shadow + lifted), so use the full Hebbian graph, not lifted-only.
        merged = dm_gated_merge(frags, scaffold, a0, log_tau_bf, graph_full)
        all_clusters.extend(merged)

    if not all_clusters:
        all_clusters = [set(range(n))]

    labels = np.full(n, -1, dtype=int)
    for cluster_id, members in enumerate(all_clusters):
        for m in members:
            labels[m] = cluster_id
    orphans = np.where(labels < 0)[0]
    if orphans.size > 0:
        positions = np.array([scaffold.nodes[i].position for i in range(n)])
        for orphan in orphans:
            assigned = np.where(labels >= 0)[0]
            if assigned.size == 0:
                labels[orphan] = 0
                continue
            dists = np.linalg.norm(
                positions[assigned] - positions[orphan], axis=1,
            )
            labels[orphan] = labels[assigned[int(np.argmin(dists))]]

    exemplars = [int(max(c, key=lambda g: hits[g])) for c in all_clusters]
    pq = partition_q_score(all_clusters, n, W, graph_lifted)

    return ClusterResult(
        labels=labels,
        exemplar_indices=np.array(exemplars, dtype=int),
        n_clusters=len(all_clusters),
        partition_q_score=pq,
    )
