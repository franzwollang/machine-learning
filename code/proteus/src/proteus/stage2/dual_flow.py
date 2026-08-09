"""Experimental Stage-2 dual / face-graph adjacency stub (SI S6 / S10.4; #43).

This module is a *proposal-path* producer for the evidence gate's affected
dual-subgraph connectivity check (SI S10.4 dynamic preservation A2). It builds
an undirected dual adjacency whose vertices are simplex ids and whose edges
join simplices that share a facet (codim-1 face) — the S6 face/factor graph
shape documented on :class:`proteus.evidence.gate.DualAdjacency`.

**What this stub is not.** Full SI S6 dual-flow remains M4 / OPEN_ISSUES #43:

* **S6.1** online face-pressure tallies — fractional residual → facet normals
  land behind ``enable_face_tallies`` (proposed; default off). Not wired into
  live sample routing yet.
* **S6.2** loopy Gaussian BP conservative reconstruction (real factor-graph
  solve; this module only sketches an identity / damped copy behind
  ``enable_conservative_bp``)
* **S6.3** boundary-face taxonomy — manifold / computational / orientation
  seams land behind ``enable_boundary_taxonomy`` (proposed; default off).
  Heuristic single-owner → true-manifold; hint sets override.
* **S6.4** simplex-local PL density formula — still missing.

Mass-conservation and density ``@awaiting("stage2.dual_flow")`` tests stay
xfail until that producer lands. This file unblocks adjacency → gate wiring
and experimental dry-run / BP / tally / taxonomy sketches only.

Flags (proposal-path, SI S14.3 operational defaults — all default **off**):

* ``DualFlowConfig.enable_dual_adjacency`` — builders / dry-run return ``None``
  adjacency when off so
  :func:`proteus.evidence.gate.affected_dual_subgraph_connected` keeps its
  conservative ``True`` default (acceptance path unchanged).
* ``DualFlowConfig.enable_conservative_bp`` — when off,
  :func:`solve_conservative_pressures` returns ``None``; when on, returns an
  identity/damped sketch (``p ≈ hat p``), **not** the SI quadratic BP solve.
* ``DualFlowConfig.enable_face_tallies`` — when off,
  :func:`accumulate_face_pressure_tally` returns ``None``.
* ``DualFlowConfig.enable_boundary_taxonomy`` — when off,
  :func:`classify_boundary_facets` returns ``None``.
* Call sites that opt in (tests / experimental dry-runs) pass flags ``True``
  and feed results into the gate or diagnostics.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from proteus.evidence.gate import (
    DualAdjacency,
    affected_dual_subgraph_connected,
)
from proteus.types import (
    BoundaryClassification,
    BoundaryType,
    Complex,
    EditProposal,
    Simplex,
)

__all__ = [
    "DualFlowConfig",
    "DualAdjacencyDict",
    "DualDryRunResult",
    "ConservativeBPResult",
    "FaceTallyResult",
    "build_dual_adjacency",
    "build_dual_adjacency_from_complex",
    "dry_run_dual_from_edit",
    "solve_conservative_pressures",
    "simplex_outward_normals",
    "accumulate_face_pressure_tally",
    "classify_boundary_facets",
    "affected_subgraph_connected",
    "resolve_dual_connected",
]

# Concrete DualAdjacency realization used by this stub (SI S6.2 contract).
DualAdjacencyDict: TypeAlias = dict[Hashable, tuple[Hashable, ...]]


@dataclass(frozen=True)
class DualFlowConfig:
    """Proposal-path flags for the dual-flow stub (SI S6 / S14.3).

    Attributes
    ----------
    enable_dual_adjacency:
        When ``False`` (default), builders / dry-run return ``None`` adjacency
        and the evidence gate keeps its conservative connectivity default.
        When ``True``, builders emit a facet-sharing dual adjacency for
        experimental / dry-run wiring. Operational default — not derived or
        calibrated; acceptance-path code must leave this off until full S6
        dual-flow is ready (#43).
    enable_conservative_bp:
        When ``False`` (default), :func:`solve_conservative_pressures` returns
        ``None``. When ``True``, returns a *sketch* that copies / damps
        empirical tallies toward themselves — **not** the SI S6.2 loopy
        Gaussian BP solve on the face/factor graph. Proposed path only; do not
        flip mass-conservation ``@awaiting`` tests on this sketch.
    enable_face_tallies:
        When ``False`` (default), :func:`accumulate_face_pressure_tally`
        returns ``None``. When ``True``, applies SI S6.1
        ``Δp̂_f ∝ max{0,(x-w̄_S)^T n_f}`` increments (proposal-path helper;
        not wired into live routing).
    enable_boundary_taxonomy:
        When ``False`` (default), :func:`classify_boundary_facets` returns
        ``None``. When ``True``, labels single-owner facets via SI S6.3
        taxonomy (heuristic true-manifold + optional computational /
        orientation-seam hint sets).
    bp_damping:
        Operational damping in ``[0, 1]`` for the BP sketch
        (``p <- (1-d)*hat_p + d*p_prev``). Default ``0.5``.
    bp_max_iters:
        Sketch iteration count (default ``1``). Real S6.2 needs convergence
        monitoring via ``r_data`` / ``r_cons``; not implemented here.
    tally_scale:
        Operational scale on S6.1 increments (default ``1.0``). Not calibrated.
    """

    enable_dual_adjacency: bool = False
    enable_conservative_bp: bool = False
    enable_face_tallies: bool = False
    enable_boundary_taxonomy: bool = False
    bp_damping: float = 0.5
    bp_max_iters: int = 1
    tally_scale: float = 1.0


@dataclass(frozen=True)
class DualDryRunResult:
    """Post-edit dry-run dual path for the evidence gate (SI S10.4 A2 / #43).

    Attributes
    ----------
    dual_adjacency:
        Facet-sharing adjacency on the *post-edit* complex, or ``None`` when
        ``enable_dual_adjacency`` is off.
    affected_simplices:
        Post-edit simplex ids (enumeration indices) touched by the edit —
        survivors that share a vertex with a removed/added simplex and/or
        contain an ``affected_node_id``, plus newly added simplex ids.
    dual_connected:
        Induced dual-subgraph connectivity on ``affected_simplices``. When
        adjacency is ``None``, this is ``True`` (conservative open default).
    post_edit_complex:
        Complex after removals/additions (same ``vertex_positions`` /
        ``intrinsic_dim`` as the input).
    """

    dual_adjacency: DualAdjacencyDict | None
    affected_simplices: tuple[Hashable, ...]
    dual_connected: bool
    post_edit_complex: Complex


@dataclass(frozen=True)
class ConservativeBPResult:
    """Sketch output for SI S6.2 conservative reconstruction (proposal-path).

    Not a real loopy Gaussian BP solve. ``pressures`` is an identity/damped
    copy of ``empirical``; residuals are reported for API shape only.
    """

    empirical: np.ndarray
    pressures: np.ndarray
    r_data: float
    r_cons: float
    iters: int
    note: str = (
        "sketch only: p≈hat_p; full loopy Gaussian BP (SI S6.2) not implemented"
    )


@dataclass(frozen=True)
class FaceTallyResult:
    """One-sample SI S6.1 face-pressure tally update (proposal-path).

    ``increments[i]`` is the nonnegative contribution to the facet opposite
    vertex ``i``; ``tallies`` is ``prior + increments`` (or just increments
    when no prior is supplied).
    """

    increments: np.ndarray
    tallies: np.ndarray
    barycenter: np.ndarray
    normals: np.ndarray


def _as_vertex_frozenset(vertices: Sequence[Hashable]) -> frozenset[Hashable]:
    verts = frozenset(vertices)
    if len(verts) != len(list(vertices)):
        raise ValueError("simplex vertex ids must be unique")
    if not verts:
        raise ValueError("simplex must have at least one vertex")
    return verts


def _facets(vertices: frozenset[Hashable]) -> list[frozenset[Hashable]]:
    """Codim-1 faces of ``vertices`` (drop one vertex each)."""

    if len(vertices) <= 1:
        return []
    return [vertices - {v} for v in vertices]


def build_dual_adjacency(
    simplices: Sequence[Sequence[Hashable]] | Mapping[Hashable, Sequence[Hashable]],
    *,
    config: DualFlowConfig | None = None,
) -> DualAdjacencyDict | None:
    """Build undirected dual adjacency from simplex vertex lists (SI S6.2).

    Parameters
    ----------
    simplices:
        Either a sequence of vertex-id sequences (simplex id = enumeration
        index ``0..n-1``) or a mapping ``simplex_id -> vertex_ids``.
    config:
        When ``enable_dual_adjacency`` is false, returns ``None`` immediately.

    Returns
    -------
    DualAdjacencyDict | None
        Symmetric adjacency list suitable for
        :func:`affected_dual_subgraph_connected`, or ``None`` when the flag is
        off. Isolated simplices appear with an empty neighbor tuple.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_dual_adjacency:
        return None

    if isinstance(simplices, Mapping):
        items: list[tuple[Hashable, frozenset[Hashable]]] = [
            (sid, _as_vertex_frozenset(verts)) for sid, verts in simplices.items()
        ]
    else:
        items = [
            (i, _as_vertex_frozenset(verts)) for i, verts in enumerate(simplices)
        ]

    # facet -> simplex ids that own that facet (codim-1 face).
    facet_owners: dict[frozenset[Hashable], list[Hashable]] = defaultdict(list)
    for sid, verts in items:
        for facet in _facets(verts):
            facet_owners[facet].append(sid)

    nbrs: dict[Hashable, set[Hashable]] = {sid: set() for sid, _ in items}
    for owners in facet_owners.values():
        if len(owners) < 2:
            continue
        # Every pair of simplices sharing this facet is dual-adjacent.
        for i, a in enumerate(owners):
            for b in owners[i + 1 :]:
                if a == b:
                    continue
                nbrs[a].add(b)
                nbrs[b].add(a)

    return {sid: tuple(sorted(nbrs[sid], key=repr)) for sid, _ in items}


def build_dual_adjacency_from_complex(
    complex: Complex,
    *,
    config: DualFlowConfig | None = None,
    id_fn: None = None,
) -> DualAdjacencyDict | None:
    """Build dual adjacency from a :class:`~proteus.types.Complex` (SI S6.2).

    Simplex ids default to enumeration indices into ``complex.simplices``.
    """

    del id_fn  # reserved for stable external ids; unused in this stub
    verts_by_index = [tuple(s.vertex_ids) for s in complex.simplices]
    return build_dual_adjacency(verts_by_index, config=config)


def dry_run_dual_from_edit(
    complex: Complex,
    *,
    remove_simplex_indices: Sequence[int] | None = None,
    add_simplices: Sequence[Sequence[int]] | None = None,
    affected_node_ids: Sequence[int] | None = None,
    proposal: EditProposal | None = None,
    config: DualFlowConfig | None = None,
) -> DualDryRunResult:
    """Dry-run a complex edit → affected simplices → dual adjacency (SI S10.4).

    Proposal-path helper (#43 / A5-T34). Applies removals then additions to a
    copy of ``complex.simplices``, rebuilds enumeration ids, selects the
    post-edit affected set, and optionally builds facet-sharing dual adjacency.

    Affected set (post-edit ids only, per SI S6.6 / S10.4 A2):

    * every survivor that shares a vertex with a *removed* simplex, or
    * every survivor / new simplex that shares a vertex with an *added*
      simplex, or
    * every post-edit simplex containing an ``affected_node_id`` (from the
      explicit arg or ``proposal.affected_node_ids``).

    When ``enable_dual_adjacency`` is off, ``dual_adjacency`` is ``None`` and
    ``dual_connected`` is ``True`` (acceptance path unchanged).
    """

    cfg = config or DualFlowConfig()
    n = len(complex.simplices)
    remove_set = set(remove_simplex_indices or ())
    for idx in remove_set:
        if idx < 0 or idx >= n:
            raise IndexError(f"remove_simplex_indices out of range: {idx}")

    node_ids: set[int] = set(affected_node_ids or ())
    if proposal is not None:
        node_ids.update(proposal.affected_node_ids)

    removed_vertex_sets = [
        frozenset(complex.simplices[i].vertex_ids) for i in sorted(remove_set)
    ]
    added_vertex_lists = [tuple(v) for v in (add_simplices or ())]
    for verts in added_vertex_lists:
        _as_vertex_frozenset(verts)

    survivors: list[Simplex] = [
        s for i, s in enumerate(complex.simplices) if i not in remove_set
    ]
    new_simplices = [
        Simplex(vertex_ids=tuple(int(v) for v in verts))
        for verts in added_vertex_lists
    ]
    post_simplices = survivors + new_simplices
    post_edit = Complex(
        simplices=post_simplices,
        vertex_positions=complex.vertex_positions,
        intrinsic_dim=complex.intrinsic_dim,
    )

    touch_vertices: set[int] = set(node_ids)
    for vs in removed_vertex_sets:
        touch_vertices.update(int(v) for v in vs)
    for verts in added_vertex_lists:
        touch_vertices.update(int(v) for v in verts)

    affected: list[Hashable] = []
    for new_id, s in enumerate(post_simplices):
        if touch_vertices and any(int(v) in touch_vertices for v in s.vertex_ids):
            affected.append(new_id)
        elif not touch_vertices and (remove_set or added_vertex_lists):
            # No node/vertex hint: treat all post-edit simplices as affected
            # when an edit was requested (conservative dry-run).
            affected.append(new_id)

    # If nothing was edited and no node hint, affected stays empty (vacuous).
    affected_t = tuple(affected)

    adj = build_dual_adjacency_from_complex(post_edit, config=cfg)
    connected = affected_subgraph_connected(adj, affected_t)
    return DualDryRunResult(
        dual_adjacency=adj,
        affected_simplices=affected_t,
        dual_connected=connected,
        post_edit_complex=post_edit,
    )


def solve_conservative_pressures(
    empirical_pressures: np.ndarray,
    *,
    simplex_facet_indices: Sequence[Sequence[int]] | None = None,
    config: DualFlowConfig | None = None,
) -> ConservativeBPResult | None:
    """Sketch SI S6.2 conservative reconstruction (proposal-path; #43 / A5-T35).

    When ``enable_conservative_bp`` is off, returns ``None``. When on, returns
    a damped identity sketch ``p ≈ hat p`` — **not** the loopy Gaussian BP
    solve on the face/factor graph. ``simplex_facet_indices`` is accepted for
    API shape (face/factor incidence) but unused by the sketch; ``r_cons`` is
    reported as ``0.0`` because no ``A_S p_S`` residual is computed.

    Do **not** flip ``@awaiting("stage2.dual_flow")`` mass / density tests on
    this sketch.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_conservative_bp:
        return None

    hat = np.asarray(empirical_pressures, dtype=float).reshape(-1)
    if hat.ndim != 1:
        raise ValueError("empirical_pressures must be 1-D")
    del simplex_facet_indices  # reserved for real A_S / factor-graph wiring

    damp = float(cfg.bp_damping)
    if not 0.0 <= damp <= 1.0:
        raise ValueError("bp_damping must be in [0, 1]")
    iters = int(cfg.bp_max_iters)
    if iters < 1:
        raise ValueError("bp_max_iters must be >= 1")

    p = hat.copy()
    for _ in range(iters):
        # Identity attractor with optional damping (sketch only).
        p = (1.0 - damp) * hat + damp * p

    eps = 1e-12
    r_data = float(np.sum((p - hat) ** 2) / (np.sum(hat**2) + eps))
    r_cons = 0.0  # no A_S available in sketch
    return ConservativeBPResult(
        empirical=hat,
        pressures=p,
        r_data=r_data,
        r_cons=r_cons,
        iters=iters,
    )


def simplex_outward_normals(
    vertex_positions: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Outward unit normals for each facet opposite vertex ``i`` (SI S6.1/S6.2).

    ``vertex_positions`` has shape ``(d+1, D)`` with ``D >= d``. Facet ``i`` is
    the codim-1 face excluding vertex ``i``. The normal is oriented away from
    the opposite vertex (out of the simplex through that facet). Degenerate
    facets yield a zero row.
    """

    P = np.asarray(vertex_positions, dtype=float)
    if P.ndim != 2:
        raise ValueError("vertex_positions must be 2-D (n_vertices, D)")
    n, _D = P.shape
    if n < 2:
        raise ValueError("simplex needs at least 2 vertices")
    normals = np.zeros_like(P)
    for i in range(n):
        facet = np.delete(P, i, axis=0)
        facet_c = facet.mean(axis=0)
        # Direction from opposite vertex through facet (outward-ish raw).
        raw = facet_c - P[i]
        if facet.shape[0] == 1:
            nvec = raw
        else:
            # Nullspace of facet affine span → candidate normal(s).
            V = facet[1:] - facet[0]
            # V: (d-1, D). Right singular vectors with small singular values.
            _u, _s, vh = np.linalg.svd(V, full_matrices=True)
            # Prefer the last row of vh (smallest singular direction).
            nvec = vh[-1].copy()
            if np.dot(nvec, raw) < 0.0:
                nvec = -nvec
            # If SVD normal is nearly orthogonal to raw (flat / high ambient),
            # fall back to raw projected off the facet span.
            if abs(np.dot(nvec, raw)) < eps * (np.linalg.norm(raw) + eps):
                nvec = raw.copy()
                for row in V:
                    denom = float(np.dot(row, row)) + eps
                    nvec = nvec - (np.dot(nvec, row) / denom) * row
        norm = float(np.linalg.norm(nvec))
        if norm < eps:
            continue
        normals[i] = nvec / norm
    return normals


def accumulate_face_pressure_tally(
    sample: np.ndarray,
    vertex_positions: np.ndarray,
    *,
    prior_tallies: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    config: DualFlowConfig | None = None,
) -> FaceTallyResult | None:
    """Online SI S6.1 face-pressure tally for one sample (proposal-path; #43).

    When ``enable_face_tallies`` is off, returns ``None``. When on, computes
    nonnegative increments

        Δp̂_f ∝ max{0, (x − w̄_S)^T n_f}

    with outward facet normals ``n_f`` and simplex barycenter ``w̄_S``. Does
    **not** flip density / mass ``@awaiting`` tests — routing integration is
    still pending.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_face_tallies:
        return None

    x = np.asarray(sample, dtype=float).reshape(-1)
    P = np.asarray(vertex_positions, dtype=float)
    if P.ndim != 2:
        raise ValueError("vertex_positions must be 2-D")
    n, D = P.shape
    if x.shape[0] != D:
        raise ValueError(
            f"sample dim {x.shape[0]} != vertex ambient dim {D}"
        )
    nrm = normals if normals is not None else simplex_outward_normals(P)
    nrm = np.asarray(nrm, dtype=float)
    if nrm.shape != P.shape:
        raise ValueError("normals must match vertex_positions shape")

    bary = P.mean(axis=0)
    residual = x - bary
    scale = float(cfg.tally_scale)
    if scale < 0.0:
        raise ValueError("tally_scale must be >= 0")
    increments = np.array(
        [scale * max(0.0, float(np.dot(residual, nrm[i]))) for i in range(n)],
        dtype=float,
    )
    if prior_tallies is None:
        tallies = increments.copy()
    else:
        prior = np.asarray(prior_tallies, dtype=float).reshape(-1)
        if prior.shape != (n,):
            raise ValueError(f"prior_tallies must have shape ({n},)")
        tallies = prior + increments
    return FaceTallyResult(
        increments=increments,
        tallies=tallies,
        barycenter=bary,
        normals=nrm,
    )


def classify_boundary_facets(
    simplices: Sequence[Sequence[Hashable]] | Mapping[Hashable, Sequence[Hashable]],
    *,
    computational_facets: Sequence[Sequence[Hashable]] | None = None,
    orientation_seams: Sequence[Sequence[Hashable]] | None = None,
    config: DualFlowConfig | None = None,
) -> list[BoundaryClassification] | None:
    """SI S6.3 boundary-face taxonomy stub (proposal-path; #43).

    When ``enable_boundary_taxonomy`` is off, returns ``None``. When on:

    * facets owned by **exactly one** simplex are boundary candidates;
    * default label is :attr:`BoundaryType.TRUE_MANIFOLD` (no exterior flux);
    * facets listed in ``computational_facets`` → ``COMPUTATIONAL``;
    * facets listed in ``orientation_seams`` → ``ORIENTATION_SEAM``
      (hint wins over computational if both list the same facet).

    Interior facets (two or more owners) are omitted. ``facet_id`` is the
    enumeration index into the returned list (stable for a given input order),
    not a global face registry — full Stage-2 face ids remain future work.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_boundary_taxonomy:
        return None

    if isinstance(simplices, Mapping):
        items: list[tuple[Hashable, frozenset[Hashable]]] = [
            (sid, _as_vertex_frozenset(verts)) for sid, verts in simplices.items()
        ]
    else:
        items = [
            (i, _as_vertex_frozenset(verts)) for i, verts in enumerate(simplices)
        ]

    facet_owners: dict[frozenset[Hashable], list[Hashable]] = defaultdict(list)
    for sid, verts in items:
        for facet in _facets(verts):
            facet_owners[facet].append(sid)

    comp = {
        _as_vertex_frozenset(f) for f in (computational_facets or ())
    }
    seams = {
        _as_vertex_frozenset(f) for f in (orientation_seams or ())
    }

    out: list[BoundaryClassification] = []
    # Deterministic order: sort by repr of frozenset contents.
    for facet in sorted(facet_owners.keys(), key=lambda f: repr(sorted(f, key=repr))):
        owners = facet_owners[facet]
        if len(owners) != 1:
            continue
        if facet in seams:
            btype = BoundaryType.ORIENTATION_SEAM
        elif facet in comp:
            btype = BoundaryType.COMPUTATIONAL
        else:
            btype = BoundaryType.TRUE_MANIFOLD
        out.append(
            BoundaryClassification(facet_id=len(out), boundary_type=btype)
        )
    return out


def affected_subgraph_connected(
    dual_adjacency: DualAdjacency | None,
    affected_simplices: Sequence[Hashable],
) -> bool:
    """Induced dual-subgraph connectivity (SI S10.4 A2).

    Thin alias of :func:`proteus.evidence.gate.affected_dual_subgraph_connected`
    so Stage-2 call sites can import the producer+check from one module.
    """

    return affected_dual_subgraph_connected(dual_adjacency, affected_simplices)


def resolve_dual_connected(
    simplices: Sequence[Sequence[Hashable]] | Mapping[Hashable, Sequence[Hashable]],
    affected_simplices: Sequence[Hashable],
    *,
    config: DualFlowConfig | None = None,
) -> bool:
    """Build dual adjacency (if enabled) and return affected-subgraph connectivity.

    When the dual-adjacency flag is off, returns ``True`` (same conservative
    default as ``score_edit(..., dual_connected=True)`` / ``adj is None``).
    """

    adj = build_dual_adjacency(simplices, config=config)
    return affected_subgraph_connected(adj, affected_simplices)
