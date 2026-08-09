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
    """

    mid_radius_frac: float = 0.35
    h0: float = 0.35
    min_end_count: float = 0.5
    gabriel_fallback: bool = True
    eps: float = _EPS


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

    pos = np.asarray(positions, dtype=float)
    pts = np.asarray(data, dtype=float)
    if pts.ndim != 2:
        raise ValueError("data must be 2-D")
    scores = np.empty(len(edges), dtype=float)
    frac = float(mid_radius_frac)
    if frac <= 0.0:
        raise ValueError("mid_radius_frac must be positive")
    for k, (i, j) in enumerate(edges):
        xi = pos[int(i)]
        xj = pos[int(j)]
        length = float(np.linalg.norm(xi - xj))
        if length <= 0.0:
            scores[k] = 1.0
            continue
        radius = frac * length
        mid = 0.5 * (xi + xj)
        n_mid = int(np.sum(np.linalg.norm(pts - mid, axis=1) <= radius))
        n_i = int(np.sum(np.linalg.norm(pts - xi, axis=1) <= radius))
        n_j = int(np.sum(np.linalg.norm(pts - xj, axis=1) <= radius))
        n_end = 0.5 * (n_i + n_j)
        scores[k] = float(n_mid) / (float(n_end) + float(eps))
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
    """Boolean mask ``True`` = cut (hollow) for each edge."""

    cfg = config if config is not None else HollowEdgeConfig()
    pos = np.asarray(positions, dtype=float)
    pts = np.asarray(data, dtype=float)
    H = hollowness_scores(
        pos, edges, pts,
        mid_radius_frac=float(cfg.mid_radius_frac),
        eps=float(cfg.eps),
    )
    # Endpoint mass for the decidability gate (same balls as H).
    end_mass = np.empty(len(edges), dtype=float)
    frac = float(cfg.mid_radius_frac)
    for k, (i, j) in enumerate(edges):
        xi = pos[int(i)]
        xj = pos[int(j)]
        length = float(np.linalg.norm(xi - xj))
        if length <= 0.0:
            end_mass[k] = 0.0
            continue
        radius = frac * length
        n_i = int(np.sum(np.linalg.norm(pts - xi, axis=1) <= radius))
        n_j = int(np.sum(np.linalg.norm(pts - xj, axis=1) <= radius))
        end_mass[k] = 0.5 * (n_i + n_j)

    cut = np.zeros(len(edges), dtype=bool)
    gab = (
        gabriel_diameter_empty(pos, edges, pts)
        if cfg.gabriel_fallback
        else np.zeros(len(edges), dtype=bool)
    )
    min_end = float(cfg.min_end_count)
    h0 = float(cfg.h0)
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
