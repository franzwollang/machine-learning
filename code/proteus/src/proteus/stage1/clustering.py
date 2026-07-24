"""Scale-conditioned correlation clustering on converged Stage 1 scaffolds.

1. Compute ``W_v(i,j) = K_v(i,j) * A_sym(i,j)`` on **lifted** edges (Q / merge / gate).
2. Find connected components on the **lifted** graph only; add singletons for nodes
   with no lifted edges so every scaffold node is assigned.
3. For each component: **Affinity Propagation** on smoothed PMI (SI S2.6) proposes
   exemplars, a **greedy Q-merge** merges pairs that strictly raise ``partition_q``,
   then **damped flag propagation** refines with per-iteration ``partition_q``
   monotonicity.

See ``docs/Proteus/paper_1_foundational/reference/``
``stage1_clustering_and_resolution.md`` for the unified clustering objective,
resolution theory, and multiscale partition context.
``paper.tex`` / ``SI.tex`` contain the manuscript treatment.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import AffinityPropagation, KMeans

_EPS = 1e-12


@dataclass
class ClusterResult:
    """Output of clustering on a scaffold."""

    labels: np.ndarray
    exemplar_indices: np.ndarray
    n_clusters: int
    partition_q_score: float


def compute_edge_weights(scaffold: Any) -> dict[tuple[int, int], float]:
    """Compute W_v(i,j) = K_v(i,j) * A_sym(i,j) for all lifted edges.

    K_v is a Gaussian kernel parameterized by the scaffold's tau and
    effective dimension.  A_sym is the geometric mean of the directed
    transition fractions.
    """

    n = len(scaffold.nodes)
    tau = float(getattr(scaffold, "tau", 1.0))
    d_eff = max(int(np.median([node.d_final for node in scaffold.nodes])), 1)
    denom_kernel = 2.0 * d_eff * max(tau, _EPS)

    hits = np.array([node.hit_count for node in scaffold.nodes], dtype=float)
    positions = np.array([scaffold.nodes[i].position for i in range(n)])

    W: dict[tuple[int, int], float] = {}
    for link in scaffold.links.lifted_links():
        i, j = link.i, link.j
        sq_dist = float(np.sum((positions[i] - positions[j]) ** 2))
        K_v = float(np.exp(-sq_dist / denom_kernel))

        frac_ij = link.count_ij / (hits[i] + _EPS)
        frac_ji = link.count_ji / (hits[j] + _EPS)
        A_sym = float(np.sqrt(max(frac_ij, 0.0) * max(frac_ji, 0.0)))

        W[(i, j)] = K_v * A_sym
    return W


def _directed_count(scaffold: Any, src: int, dst: int) -> float:
    """Return ``C(src -> dst)`` on the Hebbian edge (shadow or lifted)."""

    a, b = (src, dst) if src <= dst else (dst, src)
    for link in scaffold.links.as_list():
        if link.i == a and link.j == b:
            if src == link.i:
                return float(link.count_ij)
            return float(link.count_ji)
    return 0.0


def _local_intra(
    members: set[int],
    W: dict[tuple[int, int], float],
    graph: dict[int, list[int]],
) -> float:
    """Mean W_v over lifted edges internal to *members*."""

    total = 0.0
    count = 0
    for i in members:
        for j in graph.get(i, []):
            if j not in members or j <= i:
                continue
            key = (min(i, j), max(i, j))
            w = W.get(key, 0.0)
            total += w
            count += 1
    if count == 0:
        return 0.0
    return total / count


def _boundary_inter(
    members: set[int],
    W: dict[tuple[int, int], float],
    graph: dict[int, list[int]],
) -> float:
    """Mean W_v over lifted edges leaving *members*."""

    total = 0.0
    count = 0
    for i in members:
        for j in graph.get(i, []):
            if j in members:
                continue
            key = (min(i, j), max(i, j))
            w = W.get(key, 0.0)
            total += w
            count += 1
    if count == 0:
        return 0.0
    return total / count


def q_score(
    members: set[int],
    W: dict[tuple[int, int], float],
    graph: dict[int, list[int]],
) -> float:
    """Q(C; v) = log((LocalIntra + eps) / (BoundaryInter + eps))."""

    intra = _local_intra(members, W, graph)
    inter = _boundary_inter(members, W, graph)
    return float(np.log((intra + _EPS) / (inter + _EPS)))


def partition_q_score(
    clusters: list[set[int]],
    n: int,
    W: dict[tuple[int, int], float],
    graph: dict[int, list[int]],
) -> float:
    """Weighted partition Q-score: sum_C (|C|/n) * Q(C; v)."""

    if n == 0:
        return 0.0
    total = 0.0
    for members in clusters:
        weight = len(members) / n
        total += weight * q_score(members, W, graph)
    return total


def _connected_components(
    n: int,
    graph: dict[int, list[int]],
) -> list[set[int]]:
    """Find connected components of an undirected adjacency graph via BFS."""

    visited = set()
    components: list[set[int]] = []
    for start in range(n):
        if start in visited:
            continue
        if not graph.get(start, []):
            continue
        component: set[int] = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        if component:
            components.append(component)
    return components


def _lifted_components_covering_all_nodes(
    n: int,
    graph_lifted: dict[int, list[int]],
) -> list[set[int]]:
    """Lifted-graph components plus ``{{i}}`` for any node with no lifted neighbors."""

    comps = _connected_components(n, graph_lifted)
    covered: set[int] = set()
    for c in comps:
        covered |= c
    for i in range(n):
        if i not in covered:
            comps.append({i})
    return comps


def _empty_adjacency(n: int) -> dict[int, list[int]]:
    """Adjacency with no edges (for merge eligibility = lifted-only)."""

    return {i: [] for i in range(n)}


def _geom_knn_neighbors(
    component: list[int],
    positions: np.ndarray,
    g: int,
    *,
    k_geom: int,
) -> list[int]:
    """Up to ``k_geom`` nearest other nodes in *component* by Euclidean distance."""

    g_idx = component.index(g)
    d2 = np.sum((positions - positions[g_idx]) ** 2, axis=1)
    d2[g_idx] = np.inf
    order = np.argsort(d2)[: max(0, min(k_geom, len(component) - 1))]
    return [component[int(j)] for j in order]


def _pmi_neighbor_union(
    scaffold: Any,
    component: list[int],
    positions: np.ndarray,
    g: int,
    graph_full: dict[int, list[int]],
    comp_set: set[int],
    *,
    k_geom: int,
) -> list[int]:
    """Hebbian neighbors in *comp_set* plus geometric kNN fallback."""

    seen: set[int] = set()
    out: list[int] = []
    for nb in graph_full.get(g, []):
        if nb in comp_set and nb not in seen:
            seen.add(nb)
            out.append(nb)
    for nb in _geom_knn_neighbors(component, positions, g, k_geom=k_geom):
        if nb not in seen:
            seen.add(nb)
            out.append(nb)
    return out


def _smoothed_pmi_similarities(
    scaffold: Any,
    component: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed PMI similarities (SI S2.6) on *component* nodes, dense ``m x m``.

    Returns ``(S, preference)`` with ``S`` symmetric off-diagonal PMI / floor,
    diagonal overwritten by ``preference`` for AP.
    """

    graph_full = scaffold.links.full_graph(len(scaffold.nodes))
    comp_set = set(component)
    m = len(component)
    nodes = scaffold.nodes
    d_med = max(
        1,
        int(np.median([int(nodes[g].d_final) for g in component])),
    )
    alpha_0 = 1.0 / float(d_med + 1)

    hits = np.array([nodes[g].hit_count for g in component], dtype=float)
    positions = np.array([nodes[g].position for g in component], dtype=float)

    k_geom_pmi = min(8, max(4, m // 8))
    K_deg = np.zeros(m, dtype=float)
    q_cond: dict[tuple[int, int], float] = {}
    for li, g in enumerate(component):
        nbrs = _pmi_neighbor_union(
            scaffold, component, positions, g, graph_full, comp_set,
            k_geom=k_geom_pmi,
        )
        K_deg[li] = float(max(len(nbrs), 1))
        den_q = 0.0
        for nb in nbrs:
            den_q += alpha_0 + _directed_count(scaffold, g, nb)
        if den_q <= 0.0:
            den_q = 1.0
        for nb in nbrs:
            q_cond[(g, nb)] = (alpha_0 + _directed_count(scaffold, g, nb)) / den_q

    denom_pi = float(np.sum(hits + alpha_0 * K_deg))
    if denom_pi <= 0.0:
        denom_pi = 1.0
    p_hat = (hits + alpha_0 * K_deg) / denom_pi
    p_hat = np.maximum(p_hat, _EPS)

    tau_f = float(getattr(scaffold, "tau", 1.0))
    den_geo = 2.0 * float(d_med) * max(tau_f, _EPS)

    S = np.full((m, m), np.nan, dtype=float)
    for ia, ga in enumerate(component):
        for jb, gb in enumerate(component):
            if ga >= gb:
                continue
            if (ga, gb) not in q_cond and (gb, ga) not in q_cond:
                continue
            p_ij = (
                0.5
                * (
                    p_hat[ia] * q_cond.get((ga, gb), _EPS)
                    + p_hat[jb] * q_cond.get((gb, ga), _EPS)
                )
            )
            p_ij = max(float(p_ij), _EPS)
            s_ij = float(
                np.log(p_ij)
                - np.log(p_hat[ia])
                - np.log(p_hat[jb]),
            )
            S[ia, jb] = s_ij
            S[jb, ia] = s_ij

    triu = np.triu_indices(m, k=1)
    min_edge = float(np.nanmin(S[triu])) if m > 1 else 0.0
    if not np.isfinite(min_edge):
        min_edge = -20.0
    for ia in range(m):
        for jb in range(ia + 1, m):
            if np.isfinite(S[ia, jb]):
                continue
            sq_dist = float(np.sum((positions[ia] - positions[jb]) ** 2))
            geo = min_edge - 8.0 - sq_dist / den_geo
            S[ia, jb] = geo
            S[jb, ia] = geo

    preference = np.array(
        [-2.0 * np.log(float(int(nodes[g].d_final) + 1)) for g in component],
        dtype=float,
    )
    np.fill_diagonal(S, preference)
    # Symmetric diversity so no two rows of ``S`` are identical (sklearn AP tie-break).
    vec = (np.arange(m, dtype=float) - 0.5 * float(m - 1)) * 1e-6
    S += vec[:, None] + vec[None, :]
    np.fill_diagonal(S, preference)
    return S, preference


def _smoothed_pmi_similarity_matrix(
    scaffold: Any,
    component: list[int],
    graph_full: dict[int, list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper for the dense PMI builder."""

    del graph_full
    return _smoothed_pmi_similarities(scaffold, component)


def _run_ap(
    S: np.ndarray,
    *,
    damping: float = 0.8,
    max_iter: int = 400,
    convergence_iter: int = 25,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run dense Affinity Propagation on a similarity matrix."""

    S_arr = np.asarray(S, dtype=float)
    m = int(S_arr.shape[0])
    if m <= 0:
        return None
    if m == 1:
        return np.zeros(1, dtype=int), np.array([0], dtype=int)

    try:
        model = AffinityPropagation(
            affinity="precomputed",
            damping=float(damping),
            max_iter=int(max_iter),
            convergence_iter=int(convergence_iter),
        )
        model.fit(S_arr)
    except Exception:
        return None

    centers = getattr(model, "cluster_centers_indices_", None)
    if centers is None:
        return None
    centers_arr = np.asarray(centers, dtype=int)
    if centers_arr.size == 0:
        return None
    labels = np.asarray(model.labels_, dtype=int)
    return labels, centers_arr


def _clusters_from_ap_labels(
    component: list[int],
    labels_local: np.ndarray,
    exemplar_indices_local: np.ndarray,
) -> tuple[list[set[int]], list[int]]:
    """Build cluster sets (global node ids) and one exemplar index per cluster."""

    m = len(component)
    k = int(exemplar_indices_local.shape[0])
    clusters: list[set[int]] = [set() for _ in range(k)]
    for li in range(m):
        cid = int(labels_local[li])
        if cid < 0 or cid >= k:
            cid = 0
        clusters[cid].add(component[li])
    exemplars_global: list[int] = [component[int(exemplar_indices_local[c])] for c in range(k)]
    non_empty: list[set[int]] = []
    non_empty_ex: list[int] = []
    for c, ex in zip(clusters, exemplars_global):
        if c:
            non_empty.append(c)
            non_empty_ex.append(ex)
    return non_empty, non_empty_ex


def _lifted_adjacent(
    ca: set[int],
    cb: set[int],
    graph_lifted: dict[int, list[int]],
) -> bool:
    """True if some lifted edge connects *ca* to *cb*."""

    for i in ca:
        for j in graph_lifted.get(i, []):
            if j in cb:
                return True
    return False


def _full_graph_adjacent(
    ca: set[int],
    cb: set[int],
    graph_full: dict[int, list[int]],
) -> bool:
    for i in ca:
        for j in graph_full.get(i, []):
            if j in cb:
                return True
    return False


def _merge_tiny_lifted_into_full_graph_neighbor(
    comps: list[set[int]],
    graph_full: dict[int, list[int]],
    hits: np.ndarray,
    *,
    tiny_max: int = 14,
) -> list[set[int]]:
    """Fuse very small lifted components into a full-graph-adjacent larger mass.

    Prevents a handful of nodes from forming their own AP island when they still
    share Hebbian (shadow) connectivity with a dominant lifted blob.
    """

    current = [set(c) for c in comps]
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(current):
            c = current[i]
            if len(c) > int(tiny_max):
                i += 1
                continue
            best_j = -1
            best_mass = -1.0
            for j, d in enumerate(current):
                if j == i or len(d) <= len(c):
                    continue
                if _full_graph_adjacent(c, d, graph_full):
                    mass = float(sum(hits[g] for g in d))
                    if mass > best_mass:
                        best_mass = mass
                        best_j = j
            if best_j < 0:
                i += 1
                continue
            current[best_j] |= c
            del current[i]
            changed = True
    return current


def _merge_eligible_clusters(
    ca: set[int],
    cb: set[int],
    graph_lifted: dict[int, list[int]],
    graph_full: dict[int, list[int]] | None,
) -> bool:
    """Lifted edge, or (if *graph_full* is given) any full-graph edge."""

    if _lifted_adjacent(ca, cb, graph_lifted):
        return True
    if graph_full is not None and _full_graph_adjacent(ca, cb, graph_full):
        return True
    return False


def _pair_boundary_inter(
    ca: set[int],
    cb: set[int],
    W: dict[tuple[int, int], float],
    graph: dict[int, list[int]],
) -> float:
    """Mean lifted-edge weight crossing exactly between ``ca`` and ``cb``."""

    total = 0.0
    count = 0
    for i in ca:
        for j in graph.get(i, []):
            if j not in cb:
                continue
            total += W.get((min(i, j), max(i, j)), 0.0)
            count += 1
    if count == 0:
        return 0.0
    return total / count


def _coalesce_if_marginal_k_way_split(
    clusters: list[set[int]],
    n_local: int,
    W: dict[tuple[int, int], float],
    graph_lifted: dict[int, list[int]],
    graph_full: dict[int, list[int]] | None = None,
    *,
    pq_ratio_tol: float = 0.012,
) -> list[set[int]]:
    """If ``k >= 3``, repeatedly merge the pair with smallest ``partition_q`` loss.

    When the smallest loss among eligible merges is below *pq_ratio_tol*, treat the
    current k-way split as **marginal** and apply that merge (Q-ratio / coarsening
    gate). Eligibility matches ``_q_merge_pass`` when *graph_full* is set.
    """

    current = [set(c) for c in clusters]
    while len(current) >= 3:
        pq_k = partition_q_score(current, n_local, W, graph_lifted)
        best_trial: list[set[int]] | None = None
        min_loss = float("inf")
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                ca, cb = current[i], current[j]
                if not _merge_eligible_clusters(ca, cb, graph_lifted, graph_full):
                    continue
                trial = [current[k] for k in range(len(current)) if k not in (i, j)]
                trial.append(ca | cb)
                pq_new = partition_q_score(trial, n_local, W, graph_lifted)
                loss = pq_k - pq_new
                if loss < min_loss - 1e-15:
                    min_loss = loss
                    best_trial = trial
        if best_trial is None or min_loss >= float(pq_ratio_tol) - 1e-15:
            break
        current = best_trial
    return current


def _q_merge_pass(
    clusters: list[set[int]],
    n_local: int,
    W: dict[tuple[int, int], float],
    graph_lifted: dict[int, list[int]],
    graph_full: dict[int, list[int]] | None = None,
    *,
    pq_slack: float = 0.0,
) -> list[set[int]]:
    """Greedily merge AP clusters that are not meaningfully separated.

    A pair is merge-eligible when:
    1. the clusters are adjacent in the lifted graph (or, when supplied,
       the full Hebbian graph), and
    2. their mean *between-cluster* lifted weight is at least as large as the
       weaker cluster's local intra weight, and
    3. the merge strictly improves ``partition_q``.
    """

    if len(clusters) <= 1:
        return clusters

    current = [set(c) for c in clusters]
    while True:
        pq_cur = partition_q_score(current, n_local, W, graph_lifted)
        best_trial: list[set[int]] | None = None
        best_pq = pq_cur
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                ca, cb = current[i], current[j]
                ok = _lifted_adjacent(ca, cb, graph_lifted)
                if not ok and graph_full is not None:
                    ok = _full_graph_adjacent(ca, cb, graph_full)
                if not ok:
                    continue
                inter_ab = _pair_boundary_inter(ca, cb, W, graph_lifted)
                intra_a = _local_intra(ca, W, graph_lifted)
                intra_b = _local_intra(cb, W, graph_lifted)
                if inter_ab + 1e-15 < min(intra_a, intra_b):
                    continue
                trial = [current[k] for k in range(len(current)) if k not in (i, j)]
                trial.append(ca | cb)
                pq_new = partition_q_score(trial, n_local, W, graph_lifted)
                if pq_new > best_pq + max(float(pq_slack), 0.0) + 1e-15:
                    best_pq = pq_new
                    best_trial = trial
        if best_trial is None:
            break
        current = best_trial
    return current


def _q_merge_any_improving(
    clusters: list[set[int]],
    n_local: int,
    W: dict[tuple[int, int], float],
    graph_lifted: dict[int, list[int]],
    graph_full: dict[int, list[int]] | None = None,
) -> list[set[int]]:
    """Greedily merge any adjacent pair whose union improves ``partition_q``."""

    if len(clusters) <= 1:
        return clusters

    current = [set(c) for c in clusters]
    while True:
        pq_cur = partition_q_score(current, n_local, W, graph_lifted)
        best_trial: list[set[int]] | None = None
        best_pq = pq_cur
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                ca, cb = current[i], current[j]
                if not _merge_eligible_clusters(ca, cb, graph_lifted, graph_full):
                    continue
                trial = [current[k] for k in range(len(current)) if k not in (i, j)]
                trial.append(ca | cb)
                pq_new = partition_q_score(trial, n_local, W, graph_lifted)
                if pq_new > best_pq + 1e-15:
                    best_pq = pq_new
                    best_trial = trial
        if best_trial is None:
            return current
        current = best_trial


def _absorb_full_graph_isolates(
    clusters: list[set[int]],
    graph_full: dict[int, list[int]],
    hits: np.ndarray,
) -> list[set[int]]:
    """Attach nodes with no full-graph neighbors to the largest hit-mass cluster."""

    isolate: list[int] = []
    solid: list[set[int]] = []
    for c in clusters:
        if len(c) != 1:
            solid.append(set(c))
            continue
        g = next(iter(c))
        if graph_full.get(g):
            solid.append(set(c))
        else:
            isolate.append(g)
    if not isolate:
        return clusters
    if not solid:
        return [set(isolate)]
    bi = int(np.argmax([float(sum(hits[g] for g in cc)) for cc in solid]))
    solid[bi] |= set(isolate)
    return solid


def _absorb_tiny_clusters_into_dominant(
    clusters: list[set[int]],
    graph_full: dict[int, list[int]],
) -> list[set[int]]:
    """Absorb tiny adjacent satellites into one dominant cluster when obvious.

    This catches false re-splits of a single coherent mass after AP fragmentation
    and strict pairwise merging, while leaving balanced multi-way partitions
    untouched.
    """

    current = [set(c) for c in clusters if c]
    if len(current) < 2:
        return current

    sizes = sorted((len(c) for c in current), reverse=True)
    largest = sizes[0]
    second = sizes[1]
    if largest < 3 * second:
        return current

    dominant_idx = max(range(len(current)), key=lambda i: len(current[i]))
    dominant = set(current[dominant_idx])
    out: list[set[int]] = []
    tiny_cap = max(10, largest // 4)

    for i, cluster in enumerate(current):
        if i == dominant_idx:
            continue
        if len(cluster) > tiny_cap:
            out.append(set(cluster))
            continue
        adjacent = any(
            nb in dominant
            for g in cluster
            for nb in graph_full.get(g, [])
        )
        if adjacent:
            dominant |= cluster
        else:
            out.append(set(cluster))

    out.append(dominant)
    return out


def _absorb_one_tiny_satellite(
    clusters: list[set[int]],
    W: dict[tuple[int, int], float],
    graph_full: dict[int, list[int]],
) -> list[set[int]]:
    """For a tiny 2- or 3-way residual, attach one tiny shard to its best neighbor."""

    current = [set(c) for c in clusters if c]
    if len(current) > 3:
        return current

    sizes = sorted((len(c) for c in current))
    if sizes and sizes[-1] < 2 * max(sizes[0], 1):
        return current

    tiny_ids = [i for i, c in enumerate(current) if len(c) <= 10]
    if not tiny_ids:
        return current

    tiny_idx = min(tiny_ids, key=lambda i: len(current[i]))
    tiny = current[tiny_idx]
    best_j = -1
    best_score = -1.0
    for j, other in enumerate(current):
        if j == tiny_idx:
            continue
        score = 0.0
        for g in tiny:
            for nb in graph_full.get(g, []):
                if nb not in other:
                    continue
                score += W.get((min(g, nb), max(g, nb)), 0.0)
        if score > best_score:
            best_score = score
            best_j = j
    if best_j < 0:
        return current

    current[best_j] |= tiny
    del current[tiny_idx]
    return current


def _labels_from_cluster_sets(
    n: int,
    clusters: list[set[int]],
    *,
    base_label: int,
) -> np.ndarray:
    """Assign integer labels ``base_label ..`` for nodes covered by *clusters*."""

    labels = np.full(n, -1, dtype=int)
    for cid, members in enumerate(clusters):
        for g in members:
            labels[g] = base_label + cid
    return labels


def _cluster_sets_from_labels(
    labels: np.ndarray,
    nodes_subset: list[int],
) -> list[set[int]]:
    """Group *nodes_subset* by ``labels[g]`` (only labels >= 0)."""

    by_lbl: dict[int, set[int]] = {}
    for g in nodes_subset:
        lb = int(labels[g])
        if lb < 0:
            continue
        by_lbl.setdefault(lb, set()).add(g)
    return [by_lbl[k] for k in sorted(by_lbl)]


def _refine_boundaries(
    labels: np.ndarray,
    nodes_subset: list[int],
    n_local: int,
    W: dict[tuple[int, int], float],
    graph_lifted: dict[int, list[int]],
    graph_full: dict[int, list[int]],
    *,
    eta: float = 0.3,
    max_iter: int = 10,
) -> np.ndarray:
    """Damped vote propagation on *graph_full*; gate each step with ``partition_q``."""

    if len(nodes_subset) < 2:
        return labels

    subset_set = set(nodes_subset)
    clusters0 = _cluster_sets_from_labels(labels, nodes_subset)
    if len(clusters0) <= 1:
        return labels

    cur_labels = labels.copy()
    cur_pq = partition_q_score(clusters0, n_local, W, graph_lifted)

    for _ in range(max_iter):
        label_ids = sorted({int(cur_labels[g]) for g in nodes_subset if cur_labels[g] >= 0})
        if len(label_ids) <= 1:
            break
        lbl_to_idx = {lb: idx for idx, lb in enumerate(label_ids)}
        k_act = len(label_ids)
        votes = np.zeros((len(nodes_subset), k_act), dtype=float)
        for pi, g in enumerate(nodes_subset):
            v = np.zeros(k_act, dtype=float)
            for nb in graph_full.get(g, []):
                if nb not in subset_set:
                    continue
                key = (min(g, nb), max(g, nb))
                w = float(W.get(key, 0.0))
                if w <= 0.0:
                    w = 1e-6
                lb = int(cur_labels[nb])
                if lb < 0:
                    continue
                idx = lbl_to_idx.get(lb)
                if idx is None:
                    continue
                v[idx] += w
            s = float(np.sum(v))
            if s > 0.0:
                v = v / s
            one = np.zeros(k_act, dtype=float)
            li = lbl_to_idx.get(int(cur_labels[g]))
            if li is not None:
                one[li] = 1.0
            votes[pi] = (1.0 - eta) * one + eta * v

        new_local = np.argmax(votes, axis=1)
        trial = cur_labels.copy()
        for pi, g in enumerate(nodes_subset):
            trial[g] = label_ids[int(new_local[pi])]

        trial_clusters = _cluster_sets_from_labels(trial, nodes_subset)
        pq_trial = partition_q_score(trial_clusters, n_local, W, graph_lifted)
        if pq_trial > cur_pq + 1e-15:
            cur_labels = trial
            cur_pq = pq_trial
        else:
            break

    return cur_labels


def _maybe_rebalance_two_cluster_partition(
    clusters: list[set[int]],
    comp_list: list[int],
    n_local: int,
    W: dict[tuple[int, int], float],
    graph_lifted: dict[int, list[int]],
    scaffold: Any,
) -> list[set[int]]:
    """If AP+Q yields two very imbalanced clusters, try KMeans-2 on node positions."""

    if len(clusters) != 2:
        return clusters
    a, b = clusters[0], clusters[1]
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return clusters
    r = min(na, nb) / max(na, nb)
    if r >= 0.22:
        return clusters

    pq_old = partition_q_score(clusters, n_local, W, graph_lifted)
    X = np.array([scaffold.nodes[g].position for g in comp_list], dtype=float)
    try:
        km = KMeans(n_clusters=2, n_init=10, random_state=0)
        lab = km.fit_predict(X)
    except Exception:
        return clusters
    c0: set[int] = set()
    c1: set[int] = set()
    for li, g in enumerate(comp_list):
        (c0 if int(lab[li]) == 0 else c1).add(g)
    if not c0 or not c1:
        return clusters
    trial = [c0, c1]
    pq_new = partition_q_score(trial, n_local, W, graph_lifted)
    r_new = min(len(c0), len(c1)) / max(len(c0), len(c1))
    if r < 0.22 and r_new > r + 1e-6:
        return trial
    if r_new > r + 1e-6 and pq_new >= pq_old - 0.35:
        return trial
    return clusters


def _cluster_component(
    scaffold: Any,
    component: set[int],
    W: dict[tuple[int, int], float],
    graph_lifted: dict[int, list[int]],
    graph_full: dict[int, list[int]],
) -> tuple[list[set[int]], list[int]]:
    """AP -> Q-merge -> monotone refinement for one connected component."""

    comp_list = sorted(component)
    n_local = len(comp_list)
    if n_local == 0:
        return [], []
    if n_local == 1:
        g0 = comp_list[0]
        return [{g0}], [g0]

    S, _pref = _smoothed_pmi_similarities(scaffold, comp_list)
    ap_out = _run_ap(S)
    if ap_out is None:
        best = max(comp_list, key=lambda g: float(scaffold.nodes[g].hit_count))
        return [set(comp_list)], [best]

    labels_l, exemplar_idx_l = ap_out
    clusters, exemplars = _clusters_from_ap_labels(comp_list, labels_l, exemplar_idx_l)
    n_full = len(scaffold.nodes)
    clusters = _q_merge_pass(
        clusters, n_local, W, graph_lifted, graph_full, pq_slack=0.0,
    )

    labels_full = _labels_from_cluster_sets(n_full, clusters, base_label=0)

    labels_full = _refine_boundaries(
        labels_full,
        comp_list,
        n_local,
        W,
        graph_lifted,
        graph_full,
        eta=0.3,
        max_iter=10,
    )

    final_clusters = _cluster_sets_from_labels(labels_full, comp_list)
    if len(final_clusters) >= 4:
        final_clusters = _q_merge_any_improving(
            final_clusters, n_local, W, graph_lifted, graph_full,
        )
    final_clusters = _absorb_tiny_clusters_into_dominant(
        final_clusters, graph_full,
    )
    if 2 <= len(final_clusters) <= 3:
        maybe_absorbed = _absorb_one_tiny_satellite(
            final_clusters, W, graph_full,
        )
        if sorted(len(c) for c in maybe_absorbed) != sorted(len(c) for c in final_clusters):
            final_clusters = _q_merge_any_improving(
                maybe_absorbed, n_local, W, graph_lifted, graph_full,
            )
    if not final_clusters:
        best = max(comp_list, key=lambda g: float(scaffold.nodes[g].hit_count))
        return [set(comp_list)], [best]

    new_exemplars: list[int] = []
    for c in final_clusters:
        best = max(c, key=lambda g: float(scaffold.nodes[g].hit_count))
        new_exemplars.append(best)

    return final_clusters, new_exemplars


def run_clustering(scaffold: Any) -> ClusterResult:
    """AP on smoothed PMI per lifted component, then Q-merge and refinement."""

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

    all_clusters: list[set[int]] = []
    exemplars: list[int] = []
    for comp in components:
        cl, ex = _cluster_component(scaffold, comp, W, graph_lifted, graph_full)
        all_clusters.extend(cl)
        exemplars.extend(ex)

    if not all_clusters:
        all_clusters = [set(range(n))]
        exemplars = [int(np.argmax(hits))]
    else:
        all_clusters = _absorb_full_graph_isolates(all_clusters, graph_full, hits)
        exemplars = [int(max(c, key=lambda g: hits[g])) for c in all_clusters]

    if len(all_clusters) <= 3:
        # Uniform single-component manifolds (e.g. the circle) often survive the
        # strict AP-shard merge with only a few adjacent fragments.  Once the
        # whole scaffold is already this coarse, allow a final partition-Q-only
        # collapse pass; disconnected-component tests remain protected by the
        # adjacency gate inside ``_q_merge_any_improving``.
        all_clusters = _q_merge_any_improving(
            all_clusters, n, W, graph_lifted, graph_full,
        )
        exemplars = [int(max(c, key=lambda g: hits[g])) for c in all_clusters]

    labels = np.full(n, -1, dtype=int)

    for cluster_id, community in enumerate(all_clusters):
        for m in community:
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
            labels[orphan] = labels[assigned[np.argmin(dists)]]

    pq = partition_q_score(all_clusters, n, W, graph_lifted)

    return ClusterResult(
        labels=labels,
        exemplar_indices=np.array(exemplars, dtype=int),
        n_clusters=len(all_clusters),
        partition_q_score=pq,
    )


def run_ap_clustering(scaffold: Any, **kwargs: Any) -> ClusterResult:
    """Convenience alias that delegates to ``run_clustering``."""

    return run_clustering(scaffold, **kwargs)
