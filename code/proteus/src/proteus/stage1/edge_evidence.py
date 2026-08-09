"""Empty-region (hollow-edge) evidence for Stage 1 (OPEN_ISSUES #44).

Batch hollowness for a lifted edge ``(i, j)`` with endpoints ``x_i, x_j`` and
length ``L`` (theory note ``reference/empty_region_evidence_and_scale.md``):

- ``n_mid`` = data count in ball of radius ``r = mid_radius_frac * L`` about
  the midpoint;
- ``n_end`` = mean of the same counts about ``x_i`` and ``x_j``;
- ``H = n_mid / (n_end + eps)``.

Within-support edges have ``H = O(1)``; bridges over a void have ``H ≈ 0``.
At low endpoint mass, fall back to the Gabriel empty-diameter test (cut when
the open diameter ball contains no data).  Proposal-path defaults; the cut
threshold ``h_0`` is acceptance-path and needs Poisson-null calibration
before any awaiting flip (S14.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-9


@dataclass(frozen=True)
class HollowEdgeConfig:
    """Operational defaults for batch hollow-edge scoring / pruning.

    ``mid_radius_frac`` and ``h0`` are proposal-path / operational until a
    Poisson lens-null calibration lands (OPEN_ISSUES #44).  Probe A2-T27 found
    joint nested+tori major-CC recovery on seed 0 near
    ``mid_radius_frac=0.35``, ``h0=0.35`` (note's ``L/4`` alone false-hollows
    when ``n_end≈0``); multi-seed fragile — do not flip awaiting.

    A2-T30 audit (adapted nested/tori scaffolds): at ``mid_radius_frac=0.35``
    mid-balls are typically smaller than the node→data gap, so ``H≈0`` on
    *both* cross-shell and intra-shell lifted edges (non-discriminative).
    Around ``mid_radius_frac=0.5`` cross vs intra ``H`` separates on nested
    lifted edges, but hollow-pruning still fails as a cut-set (redundant
    Hebbian paths) and fixed-tau ``K=2`` majors have sample ARI≈chance —
    do **not** treat major-CC count as recovery.  Gabriel fallback at low
    ``n_end`` amplifies spurious cuts.  Keep flag default-off.

    A2-T30 multi-tau scan + A4 sheet null (q01≈0.57 > h0=0.35): default
    ``H-or-Gabriel`` yields spurious majors=2 at probe taus (nested@0.27,
    tori@0.5) with sample ARI≈chance, driven by Gabriel at low ``n_end``.
    ``require_gabriel_and_h=True`` (cut iff ``H < h0`` ∧ Gabriel-empty)
    suppresses those spurious K=2 hits on the probe grid while keeping
    ``prefer_hollow_edge_prepass`` default-off.  Raising ``min_end_count``
    alone *increases* Gabriel usage; prefer conjunction or
    ``gabriel_fallback=False`` with a calibrated ``h0`` / mid_frac.

    A2-T33 / A4-T24 primary ROC export (sheet FPR≈0, bridge TPR≈0.9):
    ``mid_radius_frac=0.5``, ``h0=0.7``, ``gabriel_fallback=False``,
    ``min_end_count=0.5`` — see :func:`a4_roc_primary_config`.  Sheet-null
    safe ≠ nested cut-set / sample-ARI recovery; keep default-off.

    A2-T34: ``mst_critical_only=True`` restricts cuts to hollow edges that
    also lie on a Euclidean MST of the lifted graph (conservative
    capacity/bridge proxy).  Contrast vs H-only and Gabriel∧H; default off.

    A2 capacity/flow follow-on: ``bridge_critical_only=True`` intersects
    hollow cuts with *graph-theoretic bridges* of the lifted undirected
    graph (edges whose removal increases the CC count).  Bridges ⊆ every
    spanning tree, so this is a stricter true cut-set than MST-critical;
    default off.  Mutually independent of ``mst_critical_only`` (both may
    apply as successive intersections).

    A2-T37 soft-capacity: ``soft_capacity_only=True`` intersects hollow
    cuts with edges whose Brandes betweenness is at least
    ``soft_capacity_frac * max(betweenness)`` (operational default
    ``0.25``).  Continuous capacity/flow proxy between hard bridges and
    unrestricted hollow; default off. Independent of MST/bridge flags
    (successive intersections when combined).

    A2-T39 follow-on: ``soft_capacity_method`` selects the score —
    ``"betweenness"`` (default Brandes) or ``"bridge_mass"`` (min-cut
    mass on bridges: ``min(|comp_u|,|comp_v|)`` after removing a bridge;
    non-bridges score 0).  Operational / proposal-path; default method
    remains betweenness.
    """

    mid_radius_frac: float = 0.35
    h0: float = 0.35
    min_end_count: float = 0.5
    gabriel_fallback: bool = True
    require_gabriel_and_h: bool = False
    mst_critical_only: bool = False
    bridge_critical_only: bool = False
    soft_capacity_only: bool = False
    soft_capacity_frac: float = 0.25
    soft_capacity_method: str = "betweenness"
    eps: float = _EPS


# A4-T24 → A2-T33 primary HollowEdgeConfig (flag-gated; do not flip defaults).
A4_PRIMARY_MID_RADIUS_FRAC: float = 0.5
A4_PRIMARY_H0: float = 0.7
A4_PRIMARY_MIN_END_COUNT: float = 0.5
A4_PRIMARY_GABRIEL_FALLBACK: bool = False


# ---------------------------------------------------------------------------
# Poisson-null ``h0`` calibration export (A2-T38 → A3/A4 SI sync)
# ---------------------------------------------------------------------------
# Snapshot of sheet-null H quantiles (connected density-gradient sheet,
# seed=0, n=49 edges) from ``tests.scenarios.synthetic.hollow_edge_nulls``.
# Under a locally homogeneous Poisson field ``E[H]≈1``; the lower tail is
# the practical null for choosing acceptance-path ``h0`` without fixture
# seed-tuning.  Live harness may re-check within tolerance; do not flip
# RecursionConfig / HollowEdgeConfig defaults from these numbers alone.

POISSON_NULL_SHEET_SEED: int = 0
POISSON_NULL_SHEET_N_EDGES: int = 49

# mid_radius_frac → {quantile_label: H}
POISSON_NULL_SHEET_H_QUANTILES: dict[float, dict[str, float]] = {
    0.25: {
        "q0.01": 0.15,
        "q0.05": 0.4087,
        "q0.1": 0.6925,
        "q0.25": 0.8077,
        "q0.5": 1.0,
        "mean_h": 1.0177,
    },
    0.35: {
        "q0.01": 0.4265,
        "q0.05": 0.6596,
        "q0.1": 0.7438,
        "q0.25": 0.8913,
        "q0.5": 1.0164,
        "mean_h": 1.0328,
    },
    0.5: {
        "q0.01": 0.7571,
        "q0.05": 0.8164,
        "q0.1": 0.8621,
        "q0.25": 0.9362,
        "q0.5": 1.0087,
        "mean_h": 1.0177,
    },
}

# A4 recommend_hollow_edge_configs primary (sheet FPR≈0, bridge TPR≈0.9):
# h0=0.7 ≤ sheet q01≈0.82 at mid=0.5 with gabriel off.  SI should note
# sheet-null safe ≠ nested/tori sample-ARI recovery.
POISSON_NULL_PRIMARY_MID: float = A4_PRIMARY_MID_RADIUS_FRAC
POISSON_NULL_PRIMARY_H0: float = A4_PRIMARY_H0
POISSON_NULL_PRIMARY_SHEET_Q01: float = 0.82
POISSON_NULL_SI_NOTE: str = (
    "Poisson-null sheet H: mid=0.25/0.35/0.5 q01≈0.15/0.43/0.76 (meanH≈1); "
    "A4 primary mid=0.5 h0=0.7≤q01≈0.82 gabriel=False (sheet FPR≈0, bridge "
    "TPR≈0.9). Sheet-null safe ≠ nested cut-set / sample-ARI recovery; "
    "keep HollowEdgeConfig / RecursionConfig defaults off."
)


def format_poisson_null_h0_table(
    quantiles: dict[float, dict[str, float]] | None = None,
) -> str:
    """Compact TSV of sheet-null H quantiles for A3/A4 SI handoff (A2-T38)."""

    qmap = POISSON_NULL_SHEET_H_QUANTILES if quantiles is None else quantiles
    header = "mid\tq01\tq05\tq10\tq25\tq50\tmeanH"
    lines = [header]
    for mid in sorted(qmap):
        row = qmap[mid]
        lines.append(
            f"{mid:g}\t{row['q0.01']:.4f}\t{row['q0.05']:.4f}\t"
            f"{row['q0.1']:.4f}\t{row['q0.25']:.4f}\t{row['q0.5']:.4f}\t"
            f"{row['mean_h']:.4f}"
        )
    lines.append(
        f"# primary mid={POISSON_NULL_PRIMARY_MID:g} "
        f"h0={POISSON_NULL_PRIMARY_H0:g} "
        f"sheet_q01≈{POISSON_NULL_PRIMARY_SHEET_Q01:g} "
        f"gabriel={A4_PRIMARY_GABRIEL_FALLBACK}"
    )
    lines.append(f"# {POISSON_NULL_SI_NOTE}")
    return "\n".join(lines)


def a4_roc_primary_config(**overrides: object) -> HollowEdgeConfig:
    """A4 sheet/bridge ROC primary preset (OPEN_ISSUES #44 / A2-T33).

    Primary preference from ``recommend_hollow_edge_configs``: mid=0.5,
    h0=0.7, gabriel off, min_end=0.5.  Operational / proposal-path until
    sample-ARI recovery is demonstrated; never the RecursionConfig default.
    """

    base = dict(
        mid_radius_frac=A4_PRIMARY_MID_RADIUS_FRAC,
        h0=A4_PRIMARY_H0,
        min_end_count=A4_PRIMARY_MIN_END_COUNT,
        gabriel_fallback=A4_PRIMARY_GABRIEL_FALLBACK,
        require_gabriel_and_h=False,
        mst_critical_only=False,
        bridge_critical_only=False,
        soft_capacity_only=False,
        soft_capacity_frac=0.25,
        soft_capacity_method="betweenness",
    )
    base.update(overrides)
    return HollowEdgeConfig(**base)  # type: ignore[arg-type]


def edge_ball_occupancy(
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    data: np.ndarray,
    *,
    mid_radius_frac: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(n_mid, n_end, lengths)`` per edge for hollow diagnostics.

    ``n_end`` is the mean of the endpoint-ball counts (same balls as
    :func:`hollowness_scores`).  Used to detect the empty-ball regime
    where ``H`` collapses without discriminating bridges.
    """

    pos = np.asarray(positions, dtype=float)
    pts = np.asarray(data, dtype=float)
    if pts.ndim != 2:
        raise ValueError("data must be 2-D")
    frac = float(mid_radius_frac)
    if frac <= 0.0:
        raise ValueError("mid_radius_frac must be positive")
    n_mid = np.empty(len(edges), dtype=float)
    n_end = np.empty(len(edges), dtype=float)
    lengths = np.empty(len(edges), dtype=float)
    for k, (i, j) in enumerate(edges):
        xi = pos[int(i)]
        xj = pos[int(j)]
        length = float(np.linalg.norm(xi - xj))
        lengths[k] = length
        if length <= 0.0:
            n_mid[k] = 0.0
            n_end[k] = 0.0
            continue
        radius = frac * length
        mid = 0.5 * (xi + xj)
        n_mid[k] = float(np.sum(np.linalg.norm(pts - mid, axis=1) <= radius))
        n_i = float(np.sum(np.linalg.norm(pts - xi, axis=1) <= radius))
        n_j = float(np.sum(np.linalg.norm(pts - xj, axis=1) <= radius))
        n_end[k] = 0.5 * (n_i + n_j)
    return n_mid, n_end, lengths


def hollowness_scores(
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    data: np.ndarray,
    *,
    mid_radius_frac: float = 0.35,
    eps: float = _EPS,
) -> np.ndarray:
    """Return ``H(i,j) = n_mid / (n_end + eps)`` for each edge.

    Parameters
    ----------
    positions:
        ``(n_nodes, d)`` scaffold node positions.
    edges:
        Lifted undirected edges as ``(i, j)`` index pairs.
    data:
        ``(n_samples, d)`` raw sample positions (data-side evidence).
    mid_radius_frac:
        Mid / endpoint ball radius as a fraction of edge length ``L``.
    """

    n_mid, n_end, lengths = edge_ball_occupancy(
        positions, edges, data, mid_radius_frac=mid_radius_frac,
    )
    scores = np.empty(len(edges), dtype=float)
    for k in range(len(edges)):
        if lengths[k] <= 0.0:
            scores[k] = 1.0
        else:
            scores[k] = float(n_mid[k]) / (float(n_end[k]) + float(eps))
    return scores


def gabriel_diameter_empty(
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    data: np.ndarray,
) -> np.ndarray:
    """True iff the open diameter ball of edge ``(i,j)`` contains no data.

    Used as the low-``n_end`` fallback: empty diameter ⇒ treat as hollow bridge
    (cut).  This is the geometric emptiness test, not construction of the
    Gabriel graph (which *keeps* empty-diameter edges).
    """

    pos = np.asarray(positions, dtype=float)
    pts = np.asarray(data, dtype=float)
    out = np.zeros(len(edges), dtype=bool)
    for k, (i, j) in enumerate(edges):
        xi = pos[int(i)]
        xj = pos[int(j)]
        length = float(np.linalg.norm(xi - xj))
        if length <= 0.0:
            out[k] = False
            continue
        mid = 0.5 * (xi + xj)
        radius = 0.5 * length
        out[k] = not bool(np.any(np.linalg.norm(pts - mid, axis=1) < radius - 1e-12))
    return out


def mst_edge_mask(
    positions: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    """Boolean mask ``True`` iff edge is in a Euclidean MST (Kruskal).

    Used by ``mst_critical_only`` hollow pruning (A2-T34): only MST edges
    are capacity-critical bridges in a tree sense; cutting non-MST hollow
    edges often leaves redundant Hebbian paths (non-cut-set failure mode).
    """

    pos = np.asarray(positions, dtype=float)
    n = int(pos.shape[0])
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    ranked: list[tuple[float, int, int, int]] = []
    for k, (i, j) in enumerate(edges):
        ii, jj = int(i), int(j)
        if ii == jj:
            continue
        length = float(np.linalg.norm(pos[ii] - pos[jj]))
        ranked.append((length, k, ii, jj))
    ranked.sort(key=lambda t: t[0])
    in_mst = np.zeros(len(edges), dtype=bool)
    used = 0
    for _, k, ii, jj in ranked:
        if union(ii, jj):
            in_mst[k] = True
            used += 1
            if used >= max(0, n - 1):
                break
    return in_mst


def bridge_edge_mask(
    edges: list[tuple[int, int]],
    *,
    n_nodes: int | None = None,
) -> np.ndarray:
    """Boolean mask ``True`` iff edge is a bridge of the undirected graph.

    Tarjan-style DFS discovery: an edge ``(u,v)`` is a bridge when it is a
    tree edge and ``low[v] > disc[u]`` (no back-edge from ``v``'s subtree
    reaches ``u`` or an ancestor).  Used by ``bridge_critical_only`` hollow
    pruning (capacity/flow beyond MST): only true cut-set edges may be cut.
    """

    if not edges:
        return np.zeros(0, dtype=bool)
    if n_nodes is None:
        n_nodes = 0
        for i, j in edges:
            n_nodes = max(n_nodes, int(i) + 1, int(j) + 1)
    n = int(n_nodes)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for k, (i, j) in enumerate(edges):
        ii, jj = int(i), int(j)
        if ii == jj or ii < 0 or jj < 0 or ii >= n or jj >= n:
            continue
        adj[ii].append((jj, k))
        adj[jj].append((ii, k))

    disc = [-1] * n
    low = [-1] * n
    parent = [-1] * n
    time = 0
    is_bridge = np.zeros(len(edges), dtype=bool)

    def dfs(u: int) -> None:
        nonlocal time
        disc[u] = time
        low[u] = time
        time += 1
        for v, ek in adj[u]:
            if disc[v] == -1:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    is_bridge[ek] = True
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for s in range(n):
        if disc[s] == -1 and adj[s]:
            dfs(s)
    return is_bridge


def edge_betweenness_scores(
    edges: list[tuple[int, int]],
    *,
    n_nodes: int | None = None,
) -> np.ndarray:
    """Brandes edge betweenness on the undirected multigraph of ``edges``.

    Soft capacity / flow proxy (A2-T37): high-betweenness edges carry more
    shortest paths and approximate min-cut mass without requiring a hard
    bridge.  Returns one score per input edge (0 for self-loops / OOB).
    """

    from collections import deque

    if not edges:
        return np.zeros(0, dtype=float)
    if n_nodes is None:
        n_nodes = 0
        for i, j in edges:
            n_nodes = max(n_nodes, int(i) + 1, int(j) + 1)
    n = int(n_nodes)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for k, (i, j) in enumerate(edges):
        ii, jj = int(i), int(j)
        if ii == jj or ii < 0 or jj < 0 or ii >= n or jj >= n:
            continue
        adj[ii].append((jj, k))
        adj[jj].append((ii, k))

    cb = np.zeros(len(edges), dtype=float)
    for s in range(n):
        if not adj[s]:
            continue
        stack: list[int] = []
        pred: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=float)
        sigma[s] = 1.0
        dist = [-1] * n
        dist[s] = 0
        q: deque[int] = deque([s])
        while q:
            v = q.popleft()
            stack.append(v)
            for w, ek in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append((v, ek))
        delta = np.zeros(n, dtype=float)
        while stack:
            w = stack.pop()
            for v, ek in pred[w]:
                if sigma[w] > 0.0:
                    c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                else:
                    c = 0.0
                cb[ek] += c
                delta[v] += c
    # Undirected convention: each undirected edge counted twice.
    return cb * 0.5


def bridge_mass_scores(
    edges: list[tuple[int, int]],
    *,
    n_nodes: int | None = None,
) -> np.ndarray:
    """Min-cut mass scores: bridge ``min(|comp_u|,|comp_v|)``, else 0.

    Operational soft-capacity alternative to Brandes betweenness
    (A2-T39).  Only true bridges carry positive mass; the mass equals
    the smaller side of the cut after removing that edge (unit-capacity
    global min-cut contribution when the edge is the unique cut edge).
    """

    if not edges:
        return np.zeros(0, dtype=float)
    if n_nodes is None:
        n_nodes = 0
        for i, j in edges:
            n_nodes = max(n_nodes, int(i) + 1, int(j) + 1)
    n = int(n_nodes)
    is_br = bridge_edge_mask(edges, n_nodes=n)
    scores = np.zeros(len(edges), dtype=float)
    if not np.any(is_br):
        return scores

    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for k, (i, j) in enumerate(edges):
        ii, jj = int(i), int(j)
        if ii == jj or ii < 0 or jj < 0 or ii >= n or jj >= n:
            continue
        adj[ii].append((jj, k))
        adj[jj].append((ii, k))

    for k, (i, j) in enumerate(edges):
        if not bool(is_br[k]):
            continue
        ii, jj = int(i), int(j)
        # BFS from ii avoiding edge k; mass = min(|reach|, n-|reach|).
        seen = [False] * n
        stack = [ii]
        seen[ii] = True
        reached = 0
        while stack:
            u = stack.pop()
            reached += 1
            for v, ek in adj[u]:
                if ek == k or seen[v]:
                    continue
                seen[v] = True
                stack.append(v)
        scores[k] = float(min(reached, n - reached))
    return scores


def soft_capacity_edge_mask(
    edges: list[tuple[int, int]],
    *,
    n_nodes: int | None = None,
    frac: float = 0.25,
    method: str = "betweenness",
) -> np.ndarray:
    """Boolean mask ``True`` iff capacity score ≥ ``frac * max``.

    Operational soft-capacity gate (A2-T37 / A2-T39).  ``method`` is
    ``"betweenness"`` (Brandes) or ``"bridge_mass"`` (min-cut mass on
    bridges).  ``frac`` in ``(0, 1]``; values ≤0 keep all edges, values
    >1 keep none with positive max.
    """

    if not edges:
        return np.zeros(0, dtype=bool)
    m = str(method).strip().lower()
    if m in ("bridge_mass", "mincut_mass", "min_cut_mass"):
        scores = bridge_mass_scores(edges, n_nodes=n_nodes)
    else:
        scores = edge_betweenness_scores(edges, n_nodes=n_nodes)
    f = float(frac)
    if f <= 0.0:
        return np.ones(len(edges), dtype=bool)
    peak = float(np.max(scores)) if scores.size else 0.0
    if peak <= 0.0:
        return np.zeros(len(edges), dtype=bool)
    return scores >= (f * peak)


def hollow_edge_mask(
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    data: np.ndarray,
    config: HollowEdgeConfig | None = None,
) -> np.ndarray:
    """Boolean mask ``True`` = cut (hollow) for each edge.

    Default rule (``require_gabriel_and_h=False``):
    - ``n_end >= min_end_count`` → cut iff ``H < h0``;
    - else if ``gabriel_fallback`` → cut iff Gabriel diameter ball is empty;
    - else → keep.

    Conjunction rule (``require_gabriel_and_h=True``, A2-T31 / A4 ROC):
    cut iff ``H < h0`` **and** Gabriel diameter ball is empty.  Suppresses
    empty-ball Gabriel-only spurious cuts; keep proposal-path / default-off.

    When ``mst_critical_only`` is set, intersect the hollow mask with the
    Euclidean MST edge mask (A2-T34).  When ``bridge_critical_only`` is set,
    further (or instead) intersect with graph-theoretic bridges (capacity /
    flow cut-set beyond the MST proxy).      When ``soft_capacity_only`` is set,
    intersect with high soft-capacity scores (A2-T37 betweenness /
    A2-T39 bridge-mass min-cut; see ``soft_capacity_method``).
    """

    cfg = config if config is not None else HollowEdgeConfig()
    pos = np.asarray(positions, dtype=float)
    pts = np.asarray(data, dtype=float)
    H = hollowness_scores(
        pos, edges, pts,
        mid_radius_frac=float(cfg.mid_radius_frac),
        eps=float(cfg.eps),
    )
    _, end_mass, _ = edge_ball_occupancy(
        pos, edges, pts, mid_radius_frac=float(cfg.mid_radius_frac),
    )

    need_gab = bool(cfg.gabriel_fallback) or bool(cfg.require_gabriel_and_h)
    gab = (
        gabriel_diameter_empty(pos, edges, pts)
        if need_gab
        else np.zeros(len(edges), dtype=bool)
    )
    min_end = float(cfg.min_end_count)
    h0 = float(cfg.h0)
    cut = np.zeros(len(edges), dtype=bool)
    if cfg.require_gabriel_and_h:
        for k in range(len(edges)):
            cut[k] = bool(H[k] < h0) and bool(gab[k])
    else:
        for k in range(len(edges)):
            if end_mass[k] >= min_end:
                cut[k] = bool(H[k] < h0)
            elif cfg.gabriel_fallback:
                cut[k] = bool(gab[k])
            else:
                cut[k] = False
    if cfg.mst_critical_only and len(edges) > 0:
        cut = np.logical_and(cut, mst_edge_mask(pos, edges))
    if cfg.bridge_critical_only and len(edges) > 0:
        cut = np.logical_and(
            cut, bridge_edge_mask(edges, n_nodes=int(pos.shape[0])),
        )
    if cfg.soft_capacity_only and len(edges) > 0:
        cut = np.logical_and(
            cut,
            soft_capacity_edge_mask(
                edges,
                n_nodes=int(pos.shape[0]),
                frac=float(cfg.soft_capacity_frac),
                method=str(cfg.soft_capacity_method),
            ),
        )
    return cut


def prune_hollow_edges(
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    data: np.ndarray,
    config: HollowEdgeConfig | None = None,
) -> list[tuple[int, int]]:
    """Return lifted edges that survive hollow-edge pruning."""

    if not edges:
        return []
    cut = hollow_edge_mask(positions, edges, data, config=config)
    return [e for e, c in zip(edges, cut) if not bool(c)]
