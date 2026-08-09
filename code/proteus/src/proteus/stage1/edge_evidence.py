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
    """

    mid_radius_frac: float = 0.35
    h0: float = 0.35
    min_end_count: float = 0.5
    gabriel_fallback: bool = True
    require_gabriel_and_h: bool = False
    eps: float = _EPS


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
        return cut
    for k in range(len(edges)):
        if end_mass[k] >= min_end:
            cut[k] = bool(H[k] < h0)
        elif cfg.gabriel_fallback:
            cut[k] = bool(gab[k])
        else:
            cut[k] = False
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
