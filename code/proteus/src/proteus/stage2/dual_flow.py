"""Experimental Stage-2 dual / face-graph adjacency stub (SI S6 / S10.4; #43).

This module is a *proposal-path* producer for the evidence gate's affected
dual-subgraph connectivity check (SI S10.4 dynamic preservation A2). It builds
an undirected dual adjacency whose vertices are simplex ids and whose edges
join simplices that share a facet (codim-1 face) — the S6 face/factor graph
shape documented on :class:`proteus.evidence.gate.DualAdjacency`.

**What this stub is not.** Full SI S6 dual-flow remains M4 / OPEN_ISSUES #43:

* **S6.1** online face-pressure tallies — fractional residual → facet normals
  land behind ``enable_face_tallies`` (proposed; default off). Dry-run can
  demo-wire tallies via ``dry_run_dual_from_edit(..., samples=...)``. Live
  BMU routing harness lands behind ``enable_live_bmu_tally`` (proposed;
  default off; A5-T43) — still not acceptance-path Stage-1 wiring.
* **S6.2** loopy Gaussian BP conservative reconstruction (real factor-graph
  solve; this module sketches an identity / damped copy behind
  ``enable_conservative_bp``, an ``A_S`` residual / soft message-pass
  behind ``enable_as_message_pass``, and a whitened ``λ_f`` / ``μ_S``-
  weighted soft solve behind ``enable_mu_weighted_solve`` (eq.
  si-dual-flow-weight; A5-EXP-mu) with soft spectrum step-shrink and an
  ungated ``ε_flux`` health-check helper (A5-EXP-flux). Count-aware
  ``λ_f=1+n_f/(1+n̄)`` lands behind ``enable_count_aware_lambda``
  (A5-T46; baseline remains ``λ_f=1``). Multi-simplex patch
  ``Σ_S μ_S‖A_S p_S‖²`` soft solve lands behind ``enable_patch_mu_solve``
  (A5-T47 stub — not loopy BP). Remaining real-BP gaps: full loopy
  Gaussian BP on the multi-simplex face/factor graph; online tallies →
  offline solve schedule; true-manifold flux zeroing (S6.3).
* **S6.3** boundary-face taxonomy — manifold / computational / orientation
  seams land behind ``enable_boundary_taxonomy`` (proposed; default off).
  Heuristic single-owner → true-manifold; hint sets override. Seam stitch /
  ghost-reservoir sketches land behind ``enable_seam_ghost`` (A5-T45;
  default off) — not full Stage-2 face registry.
* **S6.4** simplex-local PL density — sketch behind ``enable_simplex_density``
  (proposed; default off). Does **not** flip density ``@awaiting`` tests.

Mass-conservation / density / benchmark ``@awaiting("stage2.dual_flow")``
(and ``stage2.density``) stay xfail until the full producer lands. This file
unblocks adjacency → gate wiring and experimental dry-run / BP / tally /
taxonomy / density sketches only.

Flags (proposal-path, SI S14.3 operational defaults — all default **off**):

* ``DualFlowConfig.enable_dual_adjacency`` — builders / dry-run return ``None``
  adjacency when off so
  :func:`proteus.evidence.gate.affected_dual_subgraph_connected` keeps its
  conservative ``True`` default (acceptance path unchanged).
* ``DualFlowConfig.enable_conservative_bp`` — when off,
  :func:`solve_conservative_pressures` returns ``None``; when on, returns an
  identity/damped sketch (``p ≈ hat p``), **not** the SI quadratic BP solve.
* ``DualFlowConfig.enable_face_tallies`` — when off,
  :func:`accumulate_face_pressure_tally` / dry-run tally field return ``None``.
* ``DualFlowConfig.enable_live_bmu_tally`` — when off,
  :func:`route_live_bmu_face_tallies` returns ``None`` (A5-T43 harness).
* ``DualFlowConfig.enable_stage1_bmu_wiring`` — when off,
  :func:`route_stage1_bmu_face_tallies` returns ``None`` (A5-T48 sketch).
* ``DualFlowConfig.enable_as_message_pass`` — when off,
  :func:`solve_as_message_pass` returns ``None``; when on, soft ``A_S``
  residual nudge (not full loopy BP).
* ``DualFlowConfig.enable_mu_weighted_solve`` — when off,
  :func:`solve_mu_weighted_pressures` returns ``None``; when on, soft
  quadratic with whitened ``λ_f`` + SI ``μ_S`` (not loopy BP).
* ``DualFlowConfig.enable_count_aware_lambda`` — when off,
  :func:`count_aware_lambda_f` is unused by the soft solve (baseline
  ``λ_f=1``); when on with ``face_hit_counts``, applies SI count-aware
  weights (A5-T46).
* ``DualFlowConfig.enable_patch_mu_solve`` — when off,
  :func:`solve_patch_mu_weighted_pressures` returns ``None`` (A5-T47).
* ``DualFlowConfig.enable_boundary_taxonomy`` — when off,
  :func:`classify_boundary_facets` returns ``None``.
* ``DualFlowConfig.enable_seam_ghost`` — when off, seam stitch / ghost
  reservoir helpers return ``None`` (A5-T45).
* ``DualFlowConfig.enable_simplex_density`` — when off,
  :func:`simplex_local_density` returns ``None``.
* Call sites that opt in (tests / experimental dry-runs) pass flags ``True``
  and feed results into the gate or diagnostics.

Acceptance-path plan (replace ``None`` ⇒ ``True``; A5-T42; do **not** flip yet)
---------------------------------------------------------------------------
Today ``affected_dual_subgraph_connected(None, ...)`` and flag-off dry-run /
``resolve_dual_connected`` conservatively treat A2 as open so Stage-1 edits
are not blocked by a missing Stage-2 producer. Closing #43 requires:

1. Default-on dual adjacency from a settled post-edit complex (or an
   equivalent always-available producer) so ``None`` is unreachable on the
   acceptance path — or an explicit fail-closed policy with a declared null.
2. Real S6.2 BP (not the identity sketch) feeding mass / density so
   ``@awaiting("stage2.dual_flow")`` / ``stage2.density`` can flip with green
   evidence — never by weakening tests.
3. Gate default ``apply_dual_adjacency=True`` only after (1)–(2) and SI S6.6
   promotion from proposed → acceptance; keep proposal flags off until then.
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
from proteus.stage2.flag_complex import simplex_volume
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
    "SimplexDensityResult",
    "LiveBmuTallyResult",
    "Stage1BmuTallyResult",
    "SeamStitchResult",
    "GhostReservoirResult",
    "MuWeightedSolveResult",
    "PatchMuSolveResult",
    "build_dual_adjacency",
    "build_dual_adjacency_from_complex",
    "dry_run_dual_from_edit",
    "solve_conservative_pressures",
    "simplex_outward_normals",
    "accumulate_face_pressure_tally",
    "locate_bmu_simplex",
    "route_live_bmu_face_tallies",
    "route_stage1_bmu_face_tallies",
    "build_divergence_stencil",
    "conservation_residual_r_cons",
    "epsilon_flux",
    "solve_as_message_pass",
    "whiten_empirical_pressures",
    "mu_S_weight",
    "count_aware_lambda_f",
    "solve_mu_weighted_pressures",
    "solve_patch_mu_weighted_pressures",
    "classify_boundary_facets",
    "stitch_orientation_seam_pressures",
    "apply_ghost_reservoir",
    "barycentric_coordinates",
    "vertex_weights_from_facet_pressures",
    "simplex_local_density",
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
        ``Δp̂_f ∝ max{0,(x-w̄_S)^T n_f}`` increments (proposal-path helper).
    enable_live_bmu_tally:
        When ``False`` (default), :func:`route_live_bmu_face_tallies` returns
        ``None``. When ``True``, experimental harness routes each sample to a
        winning simplex (containment, else nearest barycenter) and accumulates
        S6.1 tallies on that BMU only (A5-T43). Does **not** wire Stage-1
        routing; does not flip mass/density ``@awaiting``.
    enable_stage1_bmu_wiring:
        When ``False`` (default), :func:`route_stage1_bmu_face_tallies`
        returns ``None``. When ``True``, experimental sketch maps each
        sample's Stage-1 node BMU → candidate simplices incident on that
        node, then tallies on the winning simplex among those candidates
        (A5-T48). Still proposal-path; does not flip ``@awaiting``.
    enable_as_message_pass:
        When ``False`` (default), :func:`solve_as_message_pass` returns
        ``None``. When ``True``, soft-projects pressures toward ``ker(A_S)``
        while anchoring empirical tallies and reports nonzero ``r_cons``
        (A5-T44 sketch — **not** loopy Gaussian BP).
    enable_mu_weighted_solve:
        When ``False`` (default), :func:`solve_mu_weighted_pressures`
        returns ``None``. When ``True``, soft-minimizes the SI S6.2
        whitened ``λ_f`` data term plus ``μ_S‖A_S p‖²`` conservation
        (eq. si-dual-flow-weight; A5-EXP-mu). Still **not** loopy Gaussian
        BP on the face/factor graph.
    enable_count_aware_lambda:
        When ``False`` (default), soft solves keep baseline ``λ_f=1``
        after whitening. When ``True`` and ``face_hit_counts`` is supplied
        to :func:`solve_mu_weighted_pressures`, uses SI count-aware
        ``λ_f=1+n_f/(1+n̄)`` (A5-T46). Operational proposal-path only.
    enable_patch_mu_solve:
        When ``False`` (default), :func:`solve_patch_mu_weighted_pressures`
        returns ``None``. When ``True``, soft-minimizes a multi-simplex
        patch objective ``Σ λ(p-hat)² + Σ_S μ_S‖A_S p_S‖²`` with
        concatenated per-simplex face blocks (A5-T47 stub — **not** a
        shared face-registry / loopy BP graph).
    enable_boundary_taxonomy:
        When ``False`` (default), :func:`classify_boundary_facets` returns
        ``None``. When ``True``, labels single-owner facets via SI S6.3
        taxonomy (heuristic true-manifold + optional computational /
        orientation-seam hint sets).
    enable_seam_ghost:
        When ``False`` (default), :func:`stitch_orientation_seam_pressures`
        / :func:`apply_ghost_reservoir` return ``None``. When ``True``,
        applies SI S6.3 seam antisymmetry / weak ghost leak sketches (A5-T45).
    enable_simplex_density:
        When ``False`` (default), :func:`simplex_local_density` returns
        ``None``. When ``True``, evaluates the SI S6.4 PL profile sketch
        (proposal-path; does not flip density ``@awaiting`` tests).
    bp_damping:
        Operational damping in ``[0, 1]`` for the BP sketch
        (``p <- (1-d)*hat_p + d*p_prev``). Default ``0.5``.
    bp_max_iters:
        Sketch iteration count (default ``1``). Real S6.2 needs convergence
        monitoring via ``r_data`` / ``r_cons``; not implemented here.
    tally_scale:
        Operational scale on S6.1 increments (default ``1.0``). Not calibrated.
    volume_floor:
        Arithmetic safeguard on ``|S|_d`` for S6.4 (default ``1e-12``).
        Operational; not a shape diagnostic (SI S6.4).
    as_eps:
        ``ε_A`` arithmetic floor for ``‖A_S‖_F^2`` (SI S6.2; default
        ``1e-8``). Operational / numerical.
    as_step:
        Soft conservation gradient step for :func:`solve_as_message_pass`
        (default ``0.25``). Operational proposal-path only.
    mu_scale:
        Leading constant in SI ``μ_S = mu_scale * λ̄_S / (‖A_S‖_F² + ε_A)``
        (default ``0.1``; SI S6.2 / S14.3 operational). Tunable toward
        ``0.01``–``1.0`` when residual balance drifts.
    whiten_floor:
        Floor on running empirical std used to whiten ``hat p_f``
        (default ``1e-8``). Operational / numerical.
    spectrum_cond_cap:
        Soft spectrum-damping trigger for :func:`solve_mu_weighted_pressures`
        (default ``1e6``). When local Hessian ``cond`` exceeds this, the
        gradient step is halved each iteration (proposal-path stand-in for
        SI ``damping when spectra are poorly conditioned``).
    ghost_coupling:
        Weak leak fraction in ``[0, 1]`` from computational-boundary
        pressures into the ghost reservoir (default ``0.1``). Operational.
    """

    enable_dual_adjacency: bool = False
    enable_conservative_bp: bool = False
    enable_face_tallies: bool = False
    enable_live_bmu_tally: bool = False
    enable_stage1_bmu_wiring: bool = False
    enable_as_message_pass: bool = False
    enable_mu_weighted_solve: bool = False
    enable_count_aware_lambda: bool = False
    enable_patch_mu_solve: bool = False
    enable_boundary_taxonomy: bool = False
    enable_seam_ghost: bool = False
    enable_simplex_density: bool = False
    bp_damping: float = 0.5
    bp_max_iters: int = 1
    tally_scale: float = 1.0
    volume_floor: float = 1e-12
    as_eps: float = 1e-8
    as_step: float = 0.25
    mu_scale: float = 0.1
    whiten_floor: float = 1e-8
    spectrum_cond_cap: float = 1e6
    ghost_coupling: float = 0.1


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
    face_tallies:
        Optional per-affected-simplex S6.1 tally demo (``None`` when
        ``enable_face_tallies`` is off). When the flag is on, a (possibly
        empty) mapping from post-edit simplex id → :class:`FaceTallyResult`
        after accumulating ``samples`` on that simplex's vertices. Not live
        routing — demonstration wiring only (A5-T40).
    """

    dual_adjacency: DualAdjacencyDict | None
    affected_simplices: tuple[Hashable, ...]
    dual_connected: bool
    post_edit_complex: Complex
    face_tallies: Mapping[Hashable, FaceTallyResult] | None = None


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


@dataclass(frozen=True)
class SimplexDensityResult:
    """SI S6.4 simplex-local PL density sketch (proposal-path; #43 / A5-T41).

    ``density`` is ``p(x|S)``. When ``used_uniform_fallback`` is True,
    ``w_bar`` was zero and the evaluator fell back to ``m_S / |S|_d``.
    """

    density: float
    rho_tilde: float
    w_bar: float
    barycentric: np.ndarray
    volume: float
    used_uniform_fallback: bool
    note: str = (
        "sketch only: SI S6.4 PL profile; not wired to live density path; "
        "do not flip @awaiting(stage2.density / stage2.dual_flow)"
    )


@dataclass(frozen=True)
class LiveBmuTallyResult:
    """Harness output for live BMU face-tally routing (SI S6.1; A5-T43).

    ``tallies_by_simplex`` maps winning-simplex id → cumulative
    :class:`FaceTallyResult` after the sample pass. ``assignments`` lists the
    BMU id chosen for each input sample (same length as the sample sequence).
    """

    tallies_by_simplex: Mapping[Hashable, FaceTallyResult]
    assignments: tuple[Hashable, ...]
    note: str = (
        "sketch only: experimental BMU harness; not Stage-1 live wiring; "
        "do not flip @awaiting(stage2.dual_flow)"
    )


@dataclass(frozen=True)
class Stage1BmuTallyResult:
    """Stage-1 BMU → simplex tally wiring sketch (SI S6.1; A5-T48).

    ``node_bmus`` echoes the caller-supplied Stage-1 node BMU per sample.
    ``assignments`` are winning simplex ids among candidates incident on
    that node. Still proposal-path — does not replace Stage-1 routing or
    flip mass/density ``@awaiting``.
    """

    tallies_by_simplex: Mapping[Hashable, FaceTallyResult]
    node_bmus: tuple[Hashable, ...]
    assignments: tuple[Hashable, ...]
    note: str = (
        "sketch only: Stage-1 node BMU → incident-simplex tally; not "
        "acceptance-path wiring; do not flip @awaiting(stage2.dual_flow)"
    )


@dataclass(frozen=True)
class SeamStitchResult:
    """SI S6.3 orientation-seam pressure stitch sketch (A5-T45).

    After normal alignment, shared seam pressures obey ``p_a = -p_b``.
    """

    pressure_a: float
    pressure_b: float
    note: str = (
        "sketch only: antisymmetric average; no face-registry / patch graph"
    )


@dataclass(frozen=True)
class GhostReservoirResult:
    """SI S6.3 computational-boundary ghost reservoir sketch (A5-T45).

    ``adjusted`` is the interior-visible pressure vector after a weak leak
    ``γ`` into the ghost; ``ghost_load`` accumulates the leaked mass.
    """

    adjusted: np.ndarray
    ghost_load: float
    note: str = (
        "sketch only: weak leak on computational facets; not a full reservoir"
    )


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
    samples: Sequence[np.ndarray] | None = None,
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

    When ``enable_face_tallies`` is on, optionally demo-wires S6.1 tallies
    (A5-T40): each sample in ``samples`` is accumulated onto every affected
    simplex that has ``vertex_positions`` (naive all-to-affected routing —
    **not** BMU live routing). Flag off ⇒ ``face_tallies`` is ``None``.
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

    face_tallies: dict[Hashable, FaceTallyResult] | None = None
    if cfg.enable_face_tallies:
        face_tallies = {}
        positions = post_edit.vertex_positions
        if samples is not None and positions is not None:
            pos = np.asarray(positions, dtype=float)
            for sid in affected_t:
                s = post_simplices[int(sid)]
                vids = [int(v) for v in s.vertex_ids]
                P = pos[vids]
                prior = None
                last: FaceTallyResult | None = None
                for raw in samples:
                    last = accumulate_face_pressure_tally(
                        raw, P, prior_tallies=prior, config=cfg
                    )
                    if last is not None:
                        prior = last.tallies
                if last is not None:
                    face_tallies[sid] = last

    return DualDryRunResult(
        dual_adjacency=adj,
        affected_simplices=affected_t,
        dual_connected=connected,
        post_edit_complex=post_edit,
        face_tallies=face_tallies,
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


def locate_bmu_simplex(
    sample: np.ndarray,
    simplex_vertex_positions: Sequence[np.ndarray]
    | Mapping[Hashable, np.ndarray],
    *,
    eps: float = 1e-9,
) -> Hashable:
    """Winning simplex for a sample (experimental BMU locator; SI S6.1 / S7.5).

    Preference order:

    1. Simplices that contain ``sample`` (all barycentric coords ``>= -eps``),
       breaking ties by nearest barycenter.
    2. Otherwise the simplex whose barycenter is nearest to ``sample``.

    Ungated geometry helper used by :func:`route_live_bmu_face_tallies`.
    """

    if isinstance(simplex_vertex_positions, Mapping):
        items: list[tuple[Hashable, np.ndarray]] = [
            (sid, np.asarray(P, dtype=float))
            for sid, P in simplex_vertex_positions.items()
        ]
    else:
        items = [
            (i, np.asarray(P, dtype=float))
            for i, P in enumerate(simplex_vertex_positions)
        ]
    if not items:
        raise ValueError("simplex_vertex_positions must be non-empty")

    x = np.asarray(sample, dtype=float).reshape(-1)
    contained: list[tuple[float, Hashable]] = []
    nearest: list[tuple[float, Hashable]] = []
    for sid, P in items:
        if P.ndim != 2:
            raise ValueError("each simplex positions array must be 2-D")
        bary = P.mean(axis=0)
        dist = float(np.linalg.norm(x - bary))
        nearest.append((dist, sid))
        beta = barycentric_coordinates(x, P)
        if float(np.min(beta)) >= -eps:
            contained.append((dist, sid))
    pool = contained if contained else nearest
    pool.sort(key=lambda t: (t[0], repr(t[1])))
    return pool[0][1]


def route_live_bmu_face_tallies(
    samples: Sequence[np.ndarray],
    simplex_vertex_positions: Sequence[np.ndarray]
    | Mapping[Hashable, np.ndarray],
    *,
    prior_tallies: Mapping[Hashable, np.ndarray] | None = None,
    config: DualFlowConfig | None = None,
) -> LiveBmuTallyResult | None:
    """Live BMU face-tally routing harness (SI S6.1; proposal-path; A5-T43).

    When ``enable_live_bmu_tally`` is off, returns ``None``. When on, each
    sample is assigned to a winning simplex via :func:`locate_bmu_simplex` and
    :func:`accumulate_face_pressure_tally` runs **only** on that BMU (true
    winner-takes-routing, fractional face increments inside the winner).

    Requires face-tally math; internally forces tally accumulation even if
    ``enable_face_tallies`` is off so the live flag is self-contained.

    Does **not** flip mass/density ``@awaiting`` tests and does **not** replace
    Stage-1 sample routing.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_live_bmu_tally:
        return None

    if isinstance(simplex_vertex_positions, Mapping):
        pos_map: dict[Hashable, np.ndarray] = {
            sid: np.asarray(P, dtype=float)
            for sid, P in simplex_vertex_positions.items()
        }
    else:
        pos_map = {
            i: np.asarray(P, dtype=float)
            for i, P in enumerate(simplex_vertex_positions)
        }
    if not pos_map:
        raise ValueError("simplex_vertex_positions must be non-empty")

    tally_cfg = DualFlowConfig(
        enable_face_tallies=True,
        tally_scale=cfg.tally_scale,
        volume_floor=cfg.volume_floor,
    )
    priors: dict[Hashable, np.ndarray] = {}
    if prior_tallies is not None:
        priors = {k: np.asarray(v, dtype=float) for k, v in prior_tallies.items()}

    last_by_sid: dict[Hashable, FaceTallyResult] = {}
    assignments: list[Hashable] = []
    for raw in samples:
        sid = locate_bmu_simplex(raw, pos_map)
        assignments.append(sid)
        prior = priors.get(sid)
        result = accumulate_face_pressure_tally(
            raw, pos_map[sid], prior_tallies=prior, config=tally_cfg
        )
        if result is None:
            raise RuntimeError("tally accumulation unexpectedly disabled")
        priors[sid] = result.tallies
        last_by_sid[sid] = result

    return LiveBmuTallyResult(
        tallies_by_simplex=last_by_sid,
        assignments=tuple(assignments),
    )


def route_stage1_bmu_face_tallies(
    samples: Sequence[np.ndarray],
    stage1_node_bmus: Sequence[Hashable],
    node_to_simplices: Mapping[Hashable, Sequence[Hashable]],
    simplex_vertex_positions: Mapping[Hashable, np.ndarray],
    *,
    prior_tallies: Mapping[Hashable, np.ndarray] | None = None,
    config: DualFlowConfig | None = None,
) -> Stage1BmuTallyResult | None:
    """Stage-1 BMU → live face-tally wiring sketch (SI S6.1; A5-T48).

    When ``enable_stage1_bmu_wiring`` is off, returns ``None``. When on:

    1. Each sample carries a Stage-1 node BMU id (ANN winner).
    2. Candidate simplices are those listed in ``node_to_simplices[bmu]``
       (incident / starring the BMU node).
    3. Among candidates, :func:`locate_bmu_simplex` picks the winning
       simplex; S6.1 tallies accumulate on that winner only.

    Proposal-path bridge toward acceptance wiring — does **not** call into
    Stage-1 controllers, does **not** flip ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_stage1_bmu_wiring:
        return None

    if len(samples) != len(stage1_node_bmus):
        raise ValueError(
            f"samples length {len(samples)} != stage1_node_bmus "
            f"{len(stage1_node_bmus)}"
        )
    pos_map = {
        sid: np.asarray(P, dtype=float)
        for sid, P in simplex_vertex_positions.items()
    }
    if not pos_map:
        raise ValueError("simplex_vertex_positions must be non-empty")

    tally_cfg = DualFlowConfig(
        enable_face_tallies=True,
        tally_scale=cfg.tally_scale,
        volume_floor=cfg.volume_floor,
    )
    priors: dict[Hashable, np.ndarray] = {}
    if prior_tallies is not None:
        priors = {k: np.asarray(v, dtype=float) for k, v in prior_tallies.items()}

    last_by_sid: dict[Hashable, FaceTallyResult] = {}
    assignments: list[Hashable] = []
    node_ids: list[Hashable] = []
    for raw, node_bmu in zip(samples, stage1_node_bmus, strict=True):
        node_ids.append(node_bmu)
        candidates = list(node_to_simplices.get(node_bmu, ()))
        if not candidates:
            raise ValueError(
                f"no simplices mapped for Stage-1 BMU node {node_bmu!r}"
            )
        cand_pos = {sid: pos_map[sid] for sid in candidates if sid in pos_map}
        if not cand_pos:
            raise ValueError(
                f"candidate simplices for node {node_bmu!r} missing positions"
            )
        sid = locate_bmu_simplex(raw, cand_pos)
        assignments.append(sid)
        prior = priors.get(sid)
        result = accumulate_face_pressure_tally(
            raw, pos_map[sid], prior_tallies=prior, config=tally_cfg
        )
        if result is None:
            raise RuntimeError("tally accumulation unexpectedly disabled")
        priors[sid] = result.tallies
        last_by_sid[sid] = result

    return Stage1BmuTallyResult(
        tallies_by_simplex=last_by_sid,
        node_bmus=tuple(node_ids),
        assignments=tuple(assignments),
    )


def _intrinsic_basis(vertex_positions: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Orthonormal basis rows spanning the affine hull of ``P`` (shape ``(d, D)``)."""

    P = np.asarray(vertex_positions, dtype=float)
    n, D = P.shape
    d = n - 1
    if d <= 0:
        return np.zeros((0, D))
    edges = P[1:] - P[0]  # (d, D)
    # QR on edges^T → columns span the edge space; take first rank columns.
    q, _r = np.linalg.qr(edges.T, mode="reduced")  # (D, d)
    # Drop near-zero columns if degenerate.
    kept: list[np.ndarray] = []
    for j in range(q.shape[1]):
        col = q[:, j]
        if float(np.linalg.norm(col)) >= eps:
            kept.append(col)
    if not kept:
        return np.zeros((0, D))
    B = np.column_stack(kept)  # (D, rank)
    return B.T  # (rank, D)


def build_divergence_stencil(
    vertex_positions: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Build SI S6.2 divergence stencil ``A_S`` (ungated geometry helper).

    Columns are ``A_f n_f`` with facet ``(d-1)``-volumes ``A_f`` and outward
    unit normals projected into the simplex's intrinsic affine span, so
    ``A_S`` has shape ``(d, d+1)`` when the simplex is full-dimensional
    (``d = n_vertices - 1``). Degenerate simplices may yield fewer rows.
    """

    P = np.asarray(vertex_positions, dtype=float)
    if P.ndim != 2:
        raise ValueError("vertex_positions must be 2-D")
    n, _D = P.shape
    if n < 2:
        raise ValueError("simplex needs at least 2 vertices")
    nrm_amb = simplex_outward_normals(P, eps=eps)
    areas = np.array(
        [simplex_volume(np.delete(P, i, axis=0)) for i in range(n)],
        dtype=float,
    )
    basis = _intrinsic_basis(P, eps=eps)  # (d, D)
    d = basis.shape[0]
    A_S = np.zeros((d, n), dtype=float)
    for i in range(n):
        n_intr = basis @ nrm_amb[i]  # (d,)
        n_norm = float(np.linalg.norm(n_intr))
        if n_norm < eps:
            continue
        n_intr = n_intr / n_norm
        A_S[:, i] = areas[i] * n_intr
    return A_S


def conservation_residual_r_cons(
    divergence_stencil: np.ndarray,
    pressures: np.ndarray,
    *,
    eps_A: float = 1e-8,
    eps: float = 1e-12,
) -> float:
    """Single-simplex SI S6.2 ``r_cons`` contribution shape (eq. si-dual-flow-residuals).

    ``‖A_S p_S‖₂² / (‖A_S‖_F² + ε_A)`` normalized by ``‖p‖₂² + ε``.
    """

    A_S = np.asarray(divergence_stencil, dtype=float)
    p = np.asarray(pressures, dtype=float).reshape(-1)
    if A_S.ndim != 2 or A_S.shape[1] != p.shape[0]:
        raise ValueError(
            f"A_S shape {A_S.shape} incompatible with pressures length {p.shape[0]}"
        )
    flux = A_S @ p
    num = float(np.dot(flux, flux)) / (float(np.sum(A_S * A_S)) + float(eps_A))
    den = float(np.dot(p, p)) + float(eps)
    return num / den


def epsilon_flux(
    divergence_stencil: np.ndarray,
    pressures: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Single-simplex SI S6.2 ``ε_flux`` health-check shape (ungated).

    ``‖A_S p‖₂² / (‖p‖₂² + ε)`` — the global post-solve flux diagnostic
    (eq. after si-dual-flow-residuals). Distinct from ``r_cons``, which
    normalizes by ``‖A_S‖_F² + ε_A``. Multi-simplex summation remains a
    future face-graph wiring step.
    """

    A_S = np.asarray(divergence_stencil, dtype=float)
    p = np.asarray(pressures, dtype=float).reshape(-1)
    if A_S.ndim != 2 or A_S.shape[1] != p.shape[0]:
        raise ValueError(
            f"A_S shape {A_S.shape} incompatible with pressures length {p.shape[0]}"
        )
    flux = A_S @ p
    return float(np.dot(flux, flux)) / (float(np.dot(p, p)) + float(eps))


def solve_as_message_pass(
    empirical_pressures: np.ndarray,
    divergence_stencil: np.ndarray,
    *,
    config: DualFlowConfig | None = None,
) -> ConservativeBPResult | None:
    """Soft ``A_S`` residual / message-pass sketch (SI S6.2; A5-T44).

    When ``enable_as_message_pass`` is off, returns ``None``. When on:

    * anchors ``p`` to empirical tallies with ``bp_damping``;
    * takes gradient steps on ``‖A_S p‖₂²`` (soft projection toward
      conservation);
    * reports ``r_data`` and nonzero ``r_cons`` via
      :func:`conservation_residual_r_cons`.

    **Not** loopy Gaussian BP on the face/factor graph. Do **not** flip
    ``@awaiting("stage2.dual_flow")`` on this sketch.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_as_message_pass:
        return None

    hat = np.asarray(empirical_pressures, dtype=float).reshape(-1)
    A_S = np.asarray(divergence_stencil, dtype=float)
    if A_S.ndim != 2 or A_S.shape[1] != hat.shape[0]:
        raise ValueError(
            f"divergence_stencil shape {A_S.shape} incompatible with "
            f"pressures length {hat.shape[0]}"
        )
    damp = float(cfg.bp_damping)
    if not 0.0 <= damp <= 1.0:
        raise ValueError("bp_damping must be in [0, 1]")
    iters = int(cfg.bp_max_iters)
    if iters < 1:
        raise ValueError("bp_max_iters must be >= 1")
    step = float(cfg.as_step)
    if step < 0.0:
        raise ValueError("as_step must be >= 0")

    p = hat.copy()
    for _ in range(iters):
        p = (1.0 - damp) * hat + damp * p
        grad = A_S.T @ (A_S @ p)
        p = p - step * grad

    eps = 1e-12
    r_data = float(np.sum((p - hat) ** 2) / (np.sum(hat**2) + eps))
    r_cons = conservation_residual_r_cons(
        A_S, p, eps_A=float(cfg.as_eps), eps=eps
    )
    return ConservativeBPResult(
        empirical=hat,
        pressures=p,
        r_data=r_data,
        r_cons=r_cons,
        iters=iters,
        note=(
            "sketch only: soft A_S message-pass; full loopy Gaussian BP "
            "(SI S6.2) not implemented"
        ),
    )


@dataclass(frozen=True)
class MuWeightedSolveResult:
    """SI S6.2 ``μ_S``-weighted soft solve sketch (proposal-path; A5-EXP-mu).

    Single-simplex soft gradient on the whitened data + conservation
    objective. ``hessian_cond`` is ``cond(diag(λ) + μ A_Sᵀ A_S)`` (whitened
    scaling) for spectrum diagnostics; ``epsilon_flux`` is the SI S6.2
    post-solve flux health check on the unwhitened pressures.
    """

    empirical: np.ndarray
    empirical_whitened: np.ndarray
    pressures: np.ndarray
    lambda_f: np.ndarray
    mu_S: float
    r_data: float
    r_cons: float
    epsilon_flux: float
    iters: int
    hessian_cond: float
    spectrum_damped: bool
    note: str = (
        "sketch only: whitened λ_f + μ_S soft solve; full loopy Gaussian BP "
        "(SI S6.2) not implemented"
    )


def whiten_empirical_pressures(
    empirical_pressures: np.ndarray,
    running_std: np.ndarray | None = None,
    *,
    floor: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Whiten ``hat p_f`` by running empirical std (SI S6.2).

    Returns ``(whitened, std_used)``. When ``running_std`` is ``None``, uses
    ``max(|hat|, floor)`` per facet as a one-shot stand-in (proposal-path;
    online tallies would supply a real running std).
    """

    hat = np.asarray(empirical_pressures, dtype=float).reshape(-1)
    fl = float(floor)
    if fl <= 0.0:
        raise ValueError("whiten_floor must be > 0")
    if running_std is None:
        std = np.maximum(np.abs(hat), fl)
    else:
        std = np.asarray(running_std, dtype=float).reshape(-1)
        if std.shape != hat.shape:
            raise ValueError(
                f"running_std shape {std.shape} != pressures {hat.shape}"
            )
        std = np.maximum(std, fl)
    return hat / std, std


def mu_S_weight(
    divergence_stencil: np.ndarray,
    *,
    bar_lambda: float = 1.0,
    mu_scale: float = 0.1,
    eps_A: float = 1e-8,
) -> float:
    """SI S6.2 conservation weight ``μ_S`` (eq. si-dual-flow-weight).

    ``μ_S = mu_scale * λ̄_S / (‖A_S‖_F² + ε_A)`` with operational
    ``mu_scale=0.1`` (S14.3).
    """

    A_S = np.asarray(divergence_stencil, dtype=float)
    if A_S.ndim != 2:
        raise ValueError("divergence_stencil must be 2-D")
    fro2 = float(np.sum(A_S * A_S))
    return float(mu_scale) * float(bar_lambda) / (fro2 + float(eps_A))


def count_aware_lambda_f(
    face_hit_counts: np.ndarray,
) -> np.ndarray:
    """SI S6.2 count-aware data weights ``λ_f = 1 + n_f / (1 + n̄)``.

    Baseline remains ``λ_f=1`` (scale-stable early runs); this variant is
    gated by ``enable_count_aware_lambda`` (A5-T46). ``n̄`` is the mean of
    nonnegative hit counts ``n_f``.
    """

    n_f = np.asarray(face_hit_counts, dtype=float).reshape(-1)
    if np.any(n_f < 0.0):
        raise ValueError("face_hit_counts must be nonnegative")
    if n_f.size == 0:
        return n_f.copy()
    nbar = float(np.mean(n_f))
    return 1.0 + n_f / (1.0 + nbar)


def solve_mu_weighted_pressures(
    empirical_pressures: np.ndarray,
    divergence_stencil: np.ndarray,
    *,
    running_std: np.ndarray | None = None,
    lambda_f: np.ndarray | None = None,
    face_hit_counts: np.ndarray | None = None,
    config: DualFlowConfig | None = None,
) -> MuWeightedSolveResult | None:
    """Whitened ``λ_f`` + ``μ_S`` soft solve (SI S6.2; A5-EXP-mu / A5-T46).

    When ``enable_mu_weighted_solve`` is off, returns ``None``. When on,
    soft-minimizes

        Σ_f λ_f (p_f - hat̃_f)² + μ_S ‖A_S p‖₂²

    with baseline ``λ_f = 1`` after whitening (SI) and
    ``μ_S = 0.1 λ̄_S / (‖A_S‖_F² + ε_A)``. When
    ``enable_count_aware_lambda`` is on and ``face_hit_counts`` is given
    (and ``lambda_f`` is ``None``), uses ``λ_f=1+n_f/(1+n̄)``. Gradient
    steps with ``bp_damping`` / ``bp_max_iters`` / ``as_step`` — **not**
    loopy Gaussian BP. Do **not** flip ``@awaiting("stage2.dual_flow")``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_mu_weighted_solve:
        return None

    hat_raw = np.asarray(empirical_pressures, dtype=float).reshape(-1)
    A_S = np.asarray(divergence_stencil, dtype=float)
    if A_S.ndim != 2 or A_S.shape[1] != hat_raw.shape[0]:
        raise ValueError(
            f"divergence_stencil shape {A_S.shape} incompatible with "
            f"pressures length {hat_raw.shape[0]}"
        )

    hat_w, std = whiten_empirical_pressures(
        hat_raw, running_std, floor=float(cfg.whiten_floor)
    )
    n = hat_w.shape[0]
    if lambda_f is None:
        if cfg.enable_count_aware_lambda:
            if face_hit_counts is None:
                raise ValueError(
                    "enable_count_aware_lambda requires face_hit_counts "
                    "when lambda_f is not supplied"
                )
            lam = count_aware_lambda_f(face_hit_counts)
            if lam.shape != (n,):
                raise ValueError(
                    f"face_hit_counts length {lam.shape[0]} != ({n},)"
                )
        else:
            lam = np.ones(n, dtype=float)
    else:
        lam = np.asarray(lambda_f, dtype=float).reshape(-1)
        if lam.shape != (n,):
            raise ValueError(f"lambda_f shape {lam.shape} != ({n},)")
        if np.any(lam < 0.0):
            raise ValueError("lambda_f must be nonnegative")

    bar_lam = float(np.mean(lam))
    mu = mu_S_weight(
        A_S,
        bar_lambda=bar_lam,
        mu_scale=float(cfg.mu_scale),
        eps_A=float(cfg.as_eps),
    )
    damp = float(cfg.bp_damping)
    if not 0.0 <= damp <= 1.0:
        raise ValueError("bp_damping must be in [0, 1]")
    iters = int(cfg.bp_max_iters)
    if iters < 1:
        raise ValueError("bp_max_iters must be >= 1")
    step = float(cfg.as_step)
    if step < 0.0:
        raise ValueError("as_step must be >= 0")

    # Soft gradient on  Σ λ (p_w - hat_w)² + μ ‖A_S (p_w ⊙ std)‖²
    # Work in whitened coords; evaluate conservation on unwhitened pressures.
    p_w = hat_w.copy()
    AtA = A_S.T @ A_S
    scale = std.reshape(-1, 1) * AtA * std.reshape(1, -1)
    hess = np.diag(lam) + mu * scale
    try:
        cond = float(np.linalg.cond(hess))
    except np.linalg.LinAlgError:
        cond = float("inf")
    spectrum_damped = bool(cond > float(cfg.spectrum_cond_cap))
    eff_step = step
    if spectrum_damped:
        # SI: "damping when spectra are poorly conditioned" — soft stand-in.
        eff_step = step * 0.5

    for i in range(iters):
        p_w = (1.0 - damp) * hat_w + damp * p_w
        p_phys = p_w * std
        grad = lam * (p_w - hat_w) + mu * (std * (AtA @ p_phys))
        use_step = eff_step
        if spectrum_damped:
            use_step = eff_step / float(2 ** min(i, 8))
        p_w = p_w - use_step * grad

    p = p_w * std

    eps = 1e-12
    r_data = float(np.sum((p - hat_raw) ** 2) / (np.sum(hat_raw**2) + eps))
    r_cons = conservation_residual_r_cons(
        A_S, p, eps_A=float(cfg.as_eps), eps=eps
    )
    e_flux = epsilon_flux(A_S, p, eps=eps)
    note = (
        "sketch only: whitened λ_f + μ_S soft solve; full loopy Gaussian "
        "BP / multi-simplex face graph (SI S6.2) not implemented"
    )
    if cfg.enable_count_aware_lambda and lambda_f is None:
        note = (
            "sketch only: count-aware λ_f=1+n_f/(1+n̄) + μ_S soft solve; "
            "full loopy Gaussian BP (SI S6.2) not implemented"
        )
    return MuWeightedSolveResult(
        empirical=hat_raw,
        empirical_whitened=hat_w,
        pressures=p,
        lambda_f=lam,
        mu_S=mu,
        r_data=r_data,
        r_cons=r_cons,
        epsilon_flux=e_flux,
        iters=iters,
        hessian_cond=cond,
        spectrum_damped=spectrum_damped,
        note=note,
    )


@dataclass(frozen=True)
class PatchMuSolveResult:
    """Multi-simplex patch ``Σ_S μ_S`` soft solve stub (SI S6.2; A5-T47).

    Pressures are concatenated per-simplex face blocks (independent copies —
    shared-face identification / face registry is future work). ``mu_S`` maps
    simplex id → local conservation weight; ``mu_S_sum`` is their sum.
    """

    empirical: np.ndarray
    pressures: np.ndarray
    lambda_f: np.ndarray
    mu_S: Mapping[Hashable, float]
    mu_S_sum: float
    r_data: float
    r_cons: float
    epsilon_flux: float
    iters: int
    block_sizes: tuple[int, ...]
    simplex_ids: tuple[Hashable, ...]
    note: str = (
        "sketch only: block-concat patch Σ μ_S‖A_S p_S‖²; not shared "
        "face-registry / loopy Gaussian BP (SI S6.2)"
    )


def solve_patch_mu_weighted_pressures(
    empirical_by_simplex: Mapping[Hashable, np.ndarray],
    stencils_by_simplex: Mapping[Hashable, np.ndarray],
    *,
    face_hit_counts_by_simplex: Mapping[Hashable, np.ndarray] | None = None,
    config: DualFlowConfig | None = None,
) -> PatchMuSolveResult | None:
    """Multi-simplex patch soft solve (SI S6.2; A5-T47).

    When ``enable_patch_mu_solve`` is off, returns ``None``. When on,
    soft-minimizes

        Σ_f λ_f (p_f - hat_f)² + Σ_S μ_S ‖A_S p_S‖₂²

    over **block-concatenated** per-simplex face pressures (each simplex
    owns a private copy of its facet pressures — shared-face glue is not
    implemented). ``μ_S`` uses :func:`mu_S_weight` per stencil; reported
    ``mu_S_sum`` is ``Σ_S μ_S``. Optional count-aware ``λ_f`` when
    ``enable_count_aware_lambda`` and hit counts are supplied.

    Proposal-path stub only — do **not** flip ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_patch_mu_solve:
        return None

    if not empirical_by_simplex:
        raise ValueError("empirical_by_simplex must be non-empty")
    ids = tuple(empirical_by_simplex.keys())
    for sid in ids:
        if sid not in stencils_by_simplex:
            raise ValueError(f"missing divergence stencil for simplex {sid!r}")

    blocks_hat: list[np.ndarray] = []
    blocks_A: list[np.ndarray] = []
    blocks_lam: list[np.ndarray] = []
    mu_map: dict[Hashable, float] = {}
    block_sizes: list[int] = []

    for sid in ids:
        hat = np.asarray(empirical_by_simplex[sid], dtype=float).reshape(-1)
        A_S = np.asarray(stencils_by_simplex[sid], dtype=float)
        if A_S.ndim != 2 or A_S.shape[1] != hat.shape[0]:
            raise ValueError(
                f"stencil {sid!r} shape {A_S.shape} incompatible with "
                f"pressures length {hat.shape[0]}"
            )
        n = hat.shape[0]
        if cfg.enable_count_aware_lambda:
            if (
                face_hit_counts_by_simplex is None
                or sid not in face_hit_counts_by_simplex
            ):
                raise ValueError(
                    "enable_count_aware_lambda requires "
                    "face_hit_counts_by_simplex for every simplex"
                )
            lam = count_aware_lambda_f(face_hit_counts_by_simplex[sid])
            if lam.shape != (n,):
                raise ValueError(
                    f"hit counts for {sid!r} length {lam.shape[0]} != ({n},)"
                )
        else:
            lam = np.ones(n, dtype=float)
        bar_lam = float(np.mean(lam))
        mu = mu_S_weight(
            A_S,
            bar_lambda=bar_lam,
            mu_scale=float(cfg.mu_scale),
            eps_A=float(cfg.as_eps),
        )
        mu_map[sid] = mu
        blocks_hat.append(hat)
        blocks_A.append(A_S)
        blocks_lam.append(lam)
        block_sizes.append(n)

    hat_raw = np.concatenate(blocks_hat)
    lam_all = np.concatenate(blocks_lam)
    # Block-diagonal soft Hessian: per-simplex AtA scaled by μ_S.
    n_tot = hat_raw.shape[0]
    AtA_big = np.zeros((n_tot, n_tot), dtype=float)
    offset = 0
    for sid, A_S, n in zip(ids, blocks_A, block_sizes, strict=True):
        AtA = A_S.T @ A_S
        sl = slice(offset, offset + n)
        AtA_big[sl, sl] = float(mu_map[sid]) * AtA
        offset += n

    damp = float(cfg.bp_damping)
    if not 0.0 <= damp <= 1.0:
        raise ValueError("bp_damping must be in [0, 1]")
    iters = int(cfg.bp_max_iters)
    if iters < 1:
        raise ValueError("bp_max_iters must be >= 1")
    step = float(cfg.as_step)
    if step < 0.0:
        raise ValueError("as_step must be >= 0")

    # Whiten globally with one-shot |hat| floor (proposal-path).
    hat_w, std = whiten_empirical_pressures(
        hat_raw, None, floor=float(cfg.whiten_floor)
    )
    p_w = hat_w.copy()
    for _ in range(iters):
        p_w = (1.0 - damp) * hat_w + damp * p_w
        p_phys = p_w * std
        # AtA_big already folds μ_S into each block; whitened conservation
        # gradient is std ⊙ (AtA_big @ p_phys).
        grad = lam_all * (p_w - hat_w) + std * (AtA_big @ p_phys)
        p_w = p_w - step * grad

    p = p_w * std
    eps = 1e-12
    r_data = float(np.sum((p - hat_raw) ** 2) / (np.sum(hat_raw**2) + eps))

    # Aggregate r_cons / ε_flux over blocks.
    flux2 = 0.0
    cons_num = 0.0
    offset = 0
    for A_S, n in zip(blocks_A, block_sizes, strict=True):
        p_S = p[offset : offset + n]
        Ap = A_S @ p_S
        f2 = float(np.dot(Ap, Ap))
        fro2 = float(np.sum(A_S * A_S))
        flux2 += f2
        cons_num += f2 / (fro2 + float(cfg.as_eps))
        offset += n
    denom = float(np.sum(p * p)) + eps
    r_cons = cons_num / denom
    e_flux = flux2 / denom
    mu_sum = float(sum(mu_map.values()))

    return PatchMuSolveResult(
        empirical=hat_raw,
        pressures=p,
        lambda_f=lam_all,
        mu_S=mu_map,
        mu_S_sum=mu_sum,
        r_data=r_data,
        r_cons=r_cons,
        epsilon_flux=e_flux,
        iters=iters,
        block_sizes=tuple(block_sizes),
        simplex_ids=ids,
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


def stitch_orientation_seam_pressures(
    pressure_a: float,
    pressure_b: float,
    *,
    config: DualFlowConfig | None = None,
) -> SeamStitchResult | None:
    """SI S6.3 orientation-seam stitch sketch (proposal-path; A5-T45).

    When ``enable_seam_ghost`` is off, returns ``None``. When on, enforces
    antisymmetry after normal alignment: ``p_a' = (p_a - p_b) / 2``,
    ``p_b' = -p_a'`` so ``p_a' = -p_b'``. Does not maintain a face registry or
    patch graph — scalar sketch only.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_seam_ghost:
        return None
    p_a = float(pressure_a)
    p_b = float(pressure_b)
    stitched_a = 0.5 * (p_a - p_b)
    stitched_b = -stitched_a
    return SeamStitchResult(pressure_a=stitched_a, pressure_b=stitched_b)


def apply_ghost_reservoir(
    facet_pressures: np.ndarray,
    *,
    computational_mask: Sequence[bool] | np.ndarray,
    config: DualFlowConfig | None = None,
) -> GhostReservoirResult | None:
    """SI S6.3 computational-boundary ghost reservoir sketch (A5-T45).

    When ``enable_seam_ghost`` is off, returns ``None``. When on, leaks a
    fraction ``ghost_coupling`` of each computational-facet pressure into a
    scalar ghost load: ``p'_f = (1-γ) p_f`` on masked facets (true-manifold /
    seam facets unchanged). Weak coupling only — not a full exterior solve.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_seam_ghost:
        return None
    p = np.asarray(facet_pressures, dtype=float).reshape(-1)
    mask = np.asarray(computational_mask, dtype=bool).reshape(-1)
    if mask.shape != p.shape:
        raise ValueError(
            f"computational_mask length {mask.shape[0]} != pressures {p.shape[0]}"
        )
    gamma = float(cfg.ghost_coupling)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("ghost_coupling must be in [0, 1]")
    adjusted = p.copy()
    leaked = p[mask] * gamma
    adjusted[mask] = p[mask] * (1.0 - gamma)
    return GhostReservoirResult(
        adjusted=adjusted,
        ghost_load=float(np.sum(leaked)),
    )


def barycentric_coordinates(
    sample: np.ndarray,
    vertex_positions: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Affine barycentric coordinates of ``sample`` in simplex ``P`` (SI S6.4).

    Solves ``[P^T; 1] β = [x; 1]`` in least squares when ambient dim exceeds
    the simplex dim. Coordinates need not be nonnegative (sample may lie
    outside ``S``); the density sketch still evaluates the PL profile.
    """

    x = np.asarray(sample, dtype=float).reshape(-1)
    P = np.asarray(vertex_positions, dtype=float)
    if P.ndim != 2:
        raise ValueError("vertex_positions must be 2-D")
    n, D = P.shape
    if x.shape[0] != D:
        raise ValueError(f"sample dim {x.shape[0]} != vertex ambient dim {D}")
    if n < 1:
        raise ValueError("simplex needs at least one vertex")
    if n == 1:
        return np.array([1.0])

    # Stack affine constraint: sum β = 1.
    A = np.vstack([P.T, np.ones(n)])  # (D+1, n)
    b = np.concatenate([x, [1.0]])
    beta, *_rest = np.linalg.lstsq(A, b, rcond=None)
    # Renormalize tiny drift so sum is exactly 1 when solvable.
    s = float(np.sum(beta))
    if abs(s) > eps:
        beta = beta / s
    return np.asarray(beta, dtype=float)


def vertex_weights_from_facet_pressures(facet_pressures: np.ndarray) -> np.ndarray:
    """Vertex weights = sum of incident facet pressures (SI S6.4).

    Facet ``i`` is opposite vertex ``i``, so vertex ``i`` is incident to every
    facet except ``i``: ``w_i = Σ_{j≠i} p_j``.
    """

    p = np.asarray(facet_pressures, dtype=float).reshape(-1)
    total = float(np.sum(p))
    return np.asarray([total - float(p[i]) for i in range(p.shape[0])], dtype=float)


def simplex_local_density(
    sample: np.ndarray,
    vertex_positions: np.ndarray,
    *,
    mass: float,
    facet_pressures: np.ndarray,
    volume: float | None = None,
    config: DualFlowConfig | None = None,
) -> SimplexDensityResult | None:
    """SI S6.4 simplex-local PL density sketch (proposal-path; #43 / A5-T41).

    When ``enable_simplex_density`` is off, returns ``None``. When on:

        ρ̃_S(x) = Σ_i β_i(x) w_{v_i}^{(S)},
        w̄_S = (1/(d+1)) Σ_i w_{v_i}^{(S)},
        p(x|S) = (m_S / |S|_d) · (ρ̃_S / w̄_S)

    with ``w_v`` from :func:`vertex_weights_from_facet_pressures`. If
    ``w̄_S = 0``, falls back to the uniform profile ``m_S / |S|_d`` (SI S6.4
    graceful degradation). Volume floor is an arithmetic safeguard only.

    Does **not** flip ``@awaiting("stage2.density")`` / mass-conservation
    tests — live density path remains unwired.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_simplex_density:
        return None

    P = np.asarray(vertex_positions, dtype=float)
    beta = barycentric_coordinates(sample, P)
    w = vertex_weights_from_facet_pressures(facet_pressures)
    if w.shape != beta.shape:
        raise ValueError(
            f"facet_pressures length {w.shape[0]} != simplex vertex count {beta.shape[0]}"
        )
    w_bar = float(np.mean(w))
    rho_tilde = float(np.dot(beta, w))
    vol = float(volume) if volume is not None else float(simplex_volume(P))
    floor = float(cfg.volume_floor)
    if floor < 0.0:
        raise ValueError("volume_floor must be >= 0")
    vol_safe = max(vol, floor)
    if abs(w_bar) < 1e-15:
        dens = float(mass) / vol_safe
        return SimplexDensityResult(
            density=dens,
            rho_tilde=rho_tilde,
            w_bar=w_bar,
            barycentric=beta,
            volume=vol,
            used_uniform_fallback=True,
        )
    dens = (float(mass) / vol_safe) * (rho_tilde / w_bar)
    return SimplexDensityResult(
        density=dens,
        rho_tilde=rho_tilde,
        w_bar=w_bar,
        barycentric=beta,
        volume=vol,
        used_uniform_fallback=False,
    )


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
