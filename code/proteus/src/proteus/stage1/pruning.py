"""Stage 1 pruning gauntlets.

Two-stage link control (SI S3.1 / S3.2, OPEN_ISSUES #29):

1. **Stage 1 (ambient):** ``prune_links`` applies a **directed floor** on all
   edges (shadow + lifted).  For each endpoint ``src`` with total outgoing
   mass ``T``, an edge ``src -> dst`` receives a prune vote from ``src`` when
   ``C(src->dst) < m_floor`` with ``m_floor = (prune_beta / (2 * D)) * T`` and
   ``D = scaffold.dim``.  ``prune_beta`` scales how large a directed count must
   be relative to a **uniform** ``2D``-slot model: under uniform splitting of
   ``T`` across ``2D`` notional slots, each slot carries ``T / (2D)``; the
   floor keeps edges whose share is at least ``prune_beta`` times that slot
   mass.  Bilateral agreement is required for hard deletion.

2. **Stage 2 (post-clustering):** ``demote_lifted_by_cluster`` uses the same
   floor rule on **lifted** edges only, with ``D`` replaced by the cluster's
   ``median(d_final)`` per node; demotes to shadow instead of deleting.

Guards: ``Link.protected_until`` vs ``scaffold.iteration`` blocks **removal
or demotion** until the link has aged (all edges still enter outgoing totals
for floor computation).  ``prune_after`` gates node pruning maturity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from proteus.links import LinkCounters
from proteus.types import Link


@dataclass(frozen=True)
class LinkPruneVerdict:
    """Per-link prune vote summary."""

    i: int
    j: int
    vote_i: bool
    vote_j: bool

    @property
    def accepted(self) -> bool:
        return self.vote_i and self.vote_j


def wilson_upper(p: float, n: float, z: float = 1.2816) -> float:
    """One-sided Wilson upper bound for a link proportion."""

    if n <= 0.0:
        return 1.0
    denom = 1.0 + (z * z) / n
    centre = p + (z * z) / (2.0 * n)
    rad = z * np.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n))
    return float((centre + rad) / denom)


def directed_floor_mass(dim: int, beta: float, total_out: float) -> float:
    """Minimum outgoing directed mass for an edge to avoid a prune vote.

    ``m_floor = (beta / (2 * D)) * total_out`` with ``D = max(dim, 1)``.
    An edge with ``C(src->dst) < m_floor`` receives a prune vote from
    ``src`` (strict inequality so equality survives).
    """

    d = max(int(dim), 1)
    return float(beta / (2.0 * d)) * float(total_out)


def count_mature_lifted_isolated(scaffold: Any) -> int:
    """Count mature nodes with zero lifted neighbours (diagnostic)."""

    graph = scaffold.neighbour_graph()
    n = len(scaffold.nodes)
    out = 0
    for i in range(n):
        if getattr(scaffold.nodes[i], "update_count", 0) < scaffold.prune_after:
            continue
        if len(graph.get(i, [])) == 0:
            out += 1
    return out


def prune_links(scaffold: Any) -> list[LinkPruneVerdict]:
    """Directed-floor bilateral pruning on all edges (Stage 1 ambient).

    Uses ``scaffold.dim`` as ``D`` and ``scaffold.prune_beta`` as ``beta``.
    Operates on *all* edges (shadow + lifted).  Hard-deletes only when both
    endpoints vote to prune.
    """

    all_links = scaffold.links.as_list()
    if not all_links:
        return []
    beta = float(getattr(scaffold, "prune_beta", 0.5))
    dim = int(scaffold.dim)

    outgoing: dict[int, list[tuple[int, float]]] = {}
    for link in all_links:
        outgoing.setdefault(link.i, []).append((link.j, link.count_ij))
        outgoing.setdefault(link.j, []).append((link.i, link.count_ji))

    votes: dict[tuple[int, int], bool] = {}
    for src, rows in outgoing.items():
        total_out = sum(count for _, count in rows)
        if total_out <= 0.0 or len(rows) < 2:
            continue
        m_floor = directed_floor_mass(dim, beta, total_out)
        for dst, count in rows:
            votes[(src, dst)] = count < m_floor

    verdicts: list[LinkPruneVerdict] = []
    for link in all_links:
        vote_i = votes.get((link.i, link.j), False)
        vote_j = votes.get((link.j, link.i), False)
        verdict = LinkPruneVerdict(link.i, link.j, vote_i, vote_j)
        verdicts.append(verdict)
        if verdict.accepted and scaffold.iteration >= link.protected_until:
            scaffold.links.remove(link.i, link.j)
    return verdicts


def demote_lifted_by_cluster(
    scaffold: Any,
    labels: np.ndarray,
    beta: float = 0.5,
) -> list[LinkPruneVerdict]:
    """Stage 2 post-clustering demotion of lifted edges to shadow.

    For each cluster, uses the cluster's median ``d_final`` as ``D`` in the
    floor formula.  Lifted edges below the directed floor at an endpoint
    receive a prune vote; bilateral agreement demotes to shadow.
    """

    lifted = scaffold.links.lifted_links()
    if not lifted:
        return []

    n = len(scaffold.nodes)
    labels_arr = np.asarray(labels, dtype=int)

    cluster_ids = set(int(c) for c in labels_arr)
    cluster_dim: dict[int, int] = {}
    for cid in cluster_ids:
        members = np.where(labels_arr == cid)[0]
        # d_final is the diagnostically-refreshed estimate (SI S1.4.1); within a
        # run it stays at the working dim, so cluster_dim == D_subspace.
        d_finals = [scaffold.nodes[int(m)].d_final for m in members]
        cluster_dim[cid] = int(np.median(d_finals)) if d_finals else scaffold.dim

    node_dim: dict[int, int] = {}
    for i in range(n):
        cid = int(labels_arr[i])
        node_dim[i] = max(cluster_dim[cid], 1)

    outgoing: dict[int, list[tuple[int, float]]] = {}
    for link in lifted:
        outgoing.setdefault(link.i, []).append((link.j, link.count_ij))
        outgoing.setdefault(link.j, []).append((link.i, link.count_ji))

    votes: dict[tuple[int, int], bool] = {}
    for src, rows in outgoing.items():
        total_out = sum(count for _, count in rows)
        if total_out <= 0.0 or len(rows) < 2:
            continue
        m_floor = directed_floor_mass(node_dim[src], beta, total_out)
        for dst, count in rows:
            votes[(src, dst)] = count < m_floor

    verdicts: list[LinkPruneVerdict] = []
    for link in lifted:
        vote_i = votes.get((link.i, link.j), False)
        vote_j = votes.get((link.j, link.i), False)
        verdict = LinkPruneVerdict(link.i, link.j, vote_i, vote_j)
        verdicts.append(verdict)
        if verdict.accepted and scaffold.iteration >= link.protected_until:
            link.lifted = False
    return verdicts


def prune_nodes(scaffold: Any) -> list[int]:
    """Apply low-hit, low-variance node pruning in place."""

    n_nodes = len(scaffold.nodes)
    if n_nodes <= scaffold.min_nodes:
        return []
    graph = scaffold.neighbour_graph()
    hits = np.array([node.hit_count for node in scaffold.nodes], dtype=float)
    removed: list[int] = []

    for idx, node in enumerate(scaffold.nodes):
        if n_nodes - len(removed) <= scaffold.min_nodes:
            break
        if idx in removed:
            continue
        if getattr(node, "update_count", 0) < scaffold.prune_after:
            continue
        neighbours = graph.get(idx, [])
        if not neighbours:
            continue
        neighbour_hits = [hits[j] for j in neighbours if j not in removed]
        if not neighbour_hits:
            continue
        low_hit = node.hit_count < 0.5 * float(np.mean(neighbour_hits))
        low_var = node.variance < 0.5 * float(scaffold.tau_local[idx])
        if low_hit and low_var and not _would_disconnect(graph, idx, removed):
            removed.append(idx)

    if removed:
        _remove_nodes(scaffold, removed)
    return removed


def _would_disconnect(
    graph: dict[int, list[int]],
    idx: int,
    removed: list[int],
) -> bool:
    neighbours = [n for n in graph.get(idx, []) if n not in removed]
    if len(neighbours) <= 1:
        return False
    remaining = set(neighbours)
    for n in neighbours:
        remaining.update(
            j for j in graph.get(n, []) if j not in removed and j != idx
        )
    return not set(neighbours).issubset(remaining)


def _remove_nodes(scaffold: Any, removed: list[int]) -> None:
    removed_set = set(removed)
    old_to_new: dict[int, int] = {}
    new_nodes = []
    for old_idx, node in enumerate(scaffold.nodes):
        if old_idx in removed_set:
            continue
        old_to_new[old_idx] = len(new_nodes)
        new_nodes.append(node)

    new_links: list[Link] = []
    for link in scaffold.links.as_list():
        if link.i in removed_set or link.j in removed_set:
            continue
        i_new = old_to_new[link.i]
        j_new = old_to_new[link.j]
        if i_new <= j_new:
            new_links.append(
                Link(i_new, j_new, link.count_ij, link.count_ji, link.protected_until, link.lifted)
            )
        else:
            new_links.append(
                Link(j_new, i_new, link.count_ji, link.count_ij, link.protected_until, link.lifted)
            )

    scaffold.nodes = new_nodes
    retained = [old_idx for old_idx, _ in sorted(old_to_new.items(), key=lambda item: item[1])]
    scaffold.tau_local = np.array([scaffold.tau_local[old_idx] for old_idx in retained])
    scaffold.links = LinkCounters()
    scaffold.links.rebuild_from_links(new_links)
    scaffold.rebuild_ann()
