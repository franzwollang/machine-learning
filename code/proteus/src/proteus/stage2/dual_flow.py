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
  (A5-T47 stub — not loopy BP). Shared-face antisymmetry soft glue for
  that patch solve lands behind ``enable_shared_face_glue`` (A5-EXP-glue;
  still not a global face registry / loopy BP). Global face-id soft
  solve (one pressure per unique facet, signed local incidence) lands
  behind ``enable_global_face_solve`` (A5-T49 stub — still **not** loopy
  Gaussian BP). A loopy Gaussian BP *message schedule* on the shared
  face/factor graph lands behind ``enable_loopy_bp_schedule``
  (A5-EXP-loopy-bp; cavity / factor-to-variable messages — still a
  sketch, not production BP with spectrum-safe convergence). Simplex
  mass normalization (``ε_mass``) lands behind
  ``enable_mass_normalization`` (A5-EXP-mass; ungated ``epsilon_mass``
  helper). Complex → node-star incidence + ANN BMU query for
  Stage-1 tally wiring lands behind ``enable_complex_ann_incidence``
  (A5-EXP-ann-inc). A BP spectrum-damping *probe* lands behind
  ``enable_bp_spectrum_damping_probe`` (A5-T55; documents step-shrink /
  loopy ridge on poorly conditioned spectra — still not production). An
  online-tallies→offline-solve *schedule* harness lands behind
  ``enable_online_offline_schedule`` (A5-T56). A *production damping
  policy* sketch (cond → damping + ridge decision) lands behind
  ``enable_bp_damping_policy`` (A5-T58). Online-tallies→offline *loopy*
  compose lands behind ``enable_online_offline_loopy_compose``
  (A5-T59). Wiring the damping policy into the loopy schedule lands
  behind ``enable_bp_policy_in_loopy`` (A5-T61). A residual-trajectory
  convergence *probe* lands behind ``enable_loopy_bp_convergence_probe``
  (A5-T62). A *certified residual-stop policy sketch* lands behind
  ``enable_loopy_bp_residual_stop`` (A5-T64; proposal-path — **not** a
  production certificate). The same flag also wires residual-stop
  *early-exit* into :func:`solve_loopy_bp_schedule` (A5-T67). A
  spectrum-safe residual-stop *certificate harness* lands behind
  ``enable_loopy_bp_spectrum_safe_cert`` (A5-T68; still a harness claim,
  not production). A mass-normalization × loopy-compose *probe*
  lands behind ``enable_mass_loopy_compose_probe`` (A5-T66). A
  policy × residual-stop compose *multi-iter residual pin* lands
  behind ``enable_policy_residual_compose_probe`` (A5-T69; proposal-path
  only). A spectrum-safe × policy_in_loopy *multi-cond pin* lands
  behind ``enable_spectrum_safe_policy_pin_probe`` (A5-T70; harness
  only).   A spectrum-safe × policy *cap-sweep residual trajectory*
  export lands behind ``enable_spectrum_safe_policy_traj_probe``
  (A5-T72; harness only). A residual-stop × mass_loopy compose
  *early-exit pin* lands behind
  ``enable_residual_mass_loopy_compose_probe`` (A5-T74; proposal-path;
  does not flip ``@awaiting``). A spectrum-safe × policy × mass_loopy
  *cap-sweep compose* lands behind
  ``enable_spectrum_safe_policy_mass_compose_probe`` (A5-T76; harness
  only; does not flip ``@awaiting``). A residual-stop × mass_loopy
  *patience sweep* lands behind
  ``enable_residual_mass_patience_sweep_probe`` (A5-T77; proposal-path;
  does not flip ``@awaiting``).   A spectrum-safe × policy × mass
  *cap-sweep residual trajectory export* lands behind
  ``enable_spectrum_safe_policy_mass_traj_probe`` (A5-T78; harness
  only; does not flip ``@awaiting``). A residual-stop × mass_loopy ×
  policy-in-loopy *patience compose* lands behind
  ``enable_residual_mass_policy_patience_probe`` (A5-T79; proposal-path;
  does not flip ``@awaiting``). Remaining real-BP gaps: true
  spectrum-safe production loopy BP certificate; true-manifold flux
  zeroing (S6.3).
* **S6.3** boundary-face taxonomy — manifold / computational / orientation
  seams land behind ``enable_boundary_taxonomy`` (proposed; default off).
  Heuristic single-owner → true-manifold; hint sets override. Seam stitch /
  ghost-reservoir sketches land behind ``enable_seam_ghost`` (A5-T45;
  default off) — not full Stage-2 face registry.
* **S6.4** simplex-local PL density — sketch behind ``enable_simplex_density``
  (proposed; default off). Live Complex/ANN density harness lands behind
  ``enable_live_density`` (A5-T50; default off). Does **not** flip density
  ``@awaiting`` tests.

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
* ``DualFlowConfig.enable_complex_ann_incidence`` — when off,
  :func:`build_node_to_simplices_from_complex` /
  :func:`query_stage1_ann_bmus` /
  :func:`route_stage1_from_complex` return ``None`` (A5-EXP-ann-inc;
  Complex star + ANN BMU bridge into the Stage-1 tally sketch).
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
* ``DualFlowConfig.enable_shared_face_glue`` — when off, patch solve keeps
  independent per-simplex face copies; when on (with ``simplices``), soft
  antisymmetry glue on shared facets (A5-EXP-glue).
* ``DualFlowConfig.enable_global_face_solve`` — when off,
  :func:`solve_global_face_mu_pressures` returns ``None``; when on,
  soft-solves one pressure per unique facet via a signed face registry
  (A5-T49 stub — **not** loopy Gaussian BP).
* ``DualFlowConfig.enable_loopy_bp_schedule`` — when off,
  :func:`solve_loopy_bp_schedule` returns ``None``; when on, runs a
  damped Gaussian factor-graph message schedule on the global face /
  simplex-factor graph (A5-EXP-loopy-bp sketch — **not** production BP).
* ``DualFlowConfig.enable_mass_normalization`` — when off,
  :func:`normalize_simplex_masses` returns ``None``; when on, rescales
  simplex masses to sum to 1 and reports ``ε_mass`` (A5-EXP-mass).
* ``DualFlowConfig.enable_boundary_taxonomy`` — when off,
  :func:`classify_boundary_facets` returns ``None``.
* ``DualFlowConfig.enable_seam_ghost`` — when off, seam stitch / ghost
  reservoir helpers return ``None`` (A5-T45).
* ``DualFlowConfig.enable_simplex_density`` — when off,
  :func:`simplex_local_density` returns ``None``.
* ``DualFlowConfig.enable_live_density`` — when off,
  :func:`route_live_density_from_complex` returns ``None`` (A5-T50
  harness: Complex/ANN BMU → S6.4 density per sample).
* ``DualFlowConfig.enable_bp_spectrum_damping_probe`` — when off,
  :func:`probe_bp_spectrum_damping` returns ``None``; when on, runs the
  A5-T55 spectrum step-shrink / loopy-ridge probe (proposal-path).
* ``DualFlowConfig.enable_bp_damping_policy`` — when off,
  :func:`propose_bp_damping_policy` returns ``None``; when on, maps
  Hessian / factor ``cond`` to recommended damping + ridge (A5-T58).
* ``DualFlowConfig.enable_online_offline_schedule`` — when off,
  :func:`run_online_offline_schedule` returns ``None``; when on, runs the
  A5-T56 online-tallies → offline-solve schedule sketch.
* ``DualFlowConfig.enable_online_offline_loopy_compose`` — when off,
  :func:`run_online_offline_loopy_compose` returns ``None``; when on,
  online tallies → offline loopy BP compose (A5-T59).
* ``DualFlowConfig.enable_bp_policy_in_loopy`` — when off, loopy BP uses
  fixed ``bp_damping`` + hard ``cond>cap`` ridge; when on, each factor
  consults :func:`propose_bp_damping_policy` (A5-T61).
* ``DualFlowConfig.enable_loopy_bp_convergence_probe`` — when off,
  :func:`probe_loopy_bp_convergence` returns ``None``; when on, records
  residual trajectories over increasing iters (A5-T62).
* ``DualFlowConfig.enable_loopy_bp_residual_stop`` — when off,
  :func:`propose_loopy_bp_residual_stop` returns ``None`` and
  :func:`solve_loopy_bp_schedule` runs all ``bp_max_iters``; when on,
  sketches a residual plateau / tolerance stop rule (A5-T64) **and**
  early-exits the loopy schedule when residuals plateau / hit tol
  (A5-T67; **not** certified production stop).
* ``DualFlowConfig.enable_loopy_bp_spectrum_safe_cert`` — when off,
  :func:`probe_loopy_bp_spectrum_safe_cert` returns ``None``; when on,
  runs loopy BP with residual-stop early-exit and reports a harness
  ``spectrum_safe_sketch_ok`` claim (A5-T68; **not** a production
  certificate).
* ``DualFlowConfig.enable_mass_loopy_compose_probe`` — when off,
  :func:`probe_mass_loopy_compose` returns ``None``; when on, runs
  mass-normalization together with online→offline loopy compose
  (A5-T66; proposal-path; does not flip ``@awaiting``).
* ``DualFlowConfig.enable_policy_residual_compose_probe`` — when off,
  :func:`probe_policy_residual_compose` returns ``None``; when on,
  pins multi-iter residuals under policy-in-loopy then compose with
  residual-stop (A5-T69; proposal-path).
* ``DualFlowConfig.enable_spectrum_safe_policy_pin_probe`` — when off,
  :func:`probe_spectrum_safe_policy_pin` returns ``None``; when on,
  pins spectrum-safe residual-stop harness outcomes across a
  ``spectrum_cond_cap`` grid with policy-in-loopy on (A5-T70; harness
  only — **not** a production certificate).
* ``DualFlowConfig.enable_spectrum_safe_policy_traj_probe`` — when off,
  :func:`probe_spectrum_safe_policy_traj` returns ``None``; when on,
  exports per-cap residual trajectories under policy-in-loopy plus the
  T70 harness sketch claim (A5-T72; harness only).
* ``DualFlowConfig.enable_residual_mass_loopy_compose_probe`` — when off,
  :func:`probe_residual_mass_loopy_compose` returns ``None``; when on,
  pins multi-iter residuals then runs mass-normalization together with
  online→offline loopy compose under residual-stop early-exit
  (A5-T74; proposal-path; does not flip ``@awaiting``).
* ``DualFlowConfig.enable_spectrum_safe_policy_mass_compose_probe`` —
  when off, :func:`probe_spectrum_safe_policy_mass_compose` returns
  ``None``; when on, cap-sweeps spectrum-safe×policy residual-stop
  sketches and mass×loopy compose under the same caps (A5-T76; harness
  only — **not** a production certificate; does not flip ``@awaiting``).
* ``DualFlowConfig.enable_residual_mass_patience_sweep_probe`` — when
  off, :func:`probe_residual_mass_patience_sweep` returns ``None``;
  when on, sweeps ``bp_residual_stop_patience`` under mass×loopy
  compose residual-stop early-exit (A5-T77; proposal-path; does not
  flip ``@awaiting``).
* ``DualFlowConfig.enable_spectrum_safe_policy_mass_traj_probe`` —
  when off, :func:`probe_spectrum_safe_policy_mass_traj` returns
  ``None``; when on, cap-sweeps residual trajectories under
  spectrum-safe×policy-in-loopy and mass×loopy compose under matching
  caps (A5-T78; harness only — **not** a production certificate; does
  not flip ``@awaiting``).
* ``DualFlowConfig.enable_residual_mass_policy_patience_probe`` — when
  off, :func:`probe_residual_mass_policy_patience` returns ``None``;
  when on, sweeps ``bp_residual_stop_patience`` under mass×loopy
  compose with residual-stop **and** ``enable_bp_policy_in_loopy``
  (A5-T79; proposal-path; does not flip ``@awaiting``).
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

A5-T54 :func:`probe_acceptance_none_open_default` locks a snapshot of the
current open-default matrix (None/flag-off ⇒ connected; flag-on detects
disconnect). Experiment / documentation only — does **not** flip defaults.

A5-T55 :func:`probe_bp_spectrum_damping` (flag-gated) documents SI S6.2
spectrum damping on the μ-soft solve and loopy BP ridge path. A5-T56
:func:`run_online_offline_schedule` sketches online tallies → offline
μ soft-solve. A5-T58 :func:`propose_bp_damping_policy` sketches a
production damping policy (cond → damping + ridge). A5-T59
:func:`run_online_offline_loopy_compose` wires online tallies → offline
loopy BP. A5-T57 :func:`probe_fail_closed_dual_adjacency_plan` documents the
path to replace None⇒True — still does **not** flip defaults.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from proteus.evidence.gate import (
    DualAdjacency,
    GateConfig,
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
    "LiveDensityResult",
    "SeamStitchResult",
    "GhostReservoirResult",
    "MuWeightedSolveResult",
    "PatchMuSolveResult",
    "GlobalFaceIncidence",
    "GlobalFaceRegistry",
    "GlobalFaceSolveResult",
    "LoopyBPScheduleResult",
    "MassNormalizationResult",
    "AcceptanceOpenDefaultProbe",
    "BpSpectrumDampingProbe",
    "BpDampingPolicyResult",
    "OnlineOfflineScheduleResult",
    "OnlineOfflineLoopyComposeResult",
    "FailClosedDualPlanProbe",
    "FailClosedGateSwitchProbe",
    "LoopyBPConvergenceProbe",
    "LoopyBPResidualStopPolicy",
    "LoopyBPSpectrumSafeCertProbe",
    "MassLoopyComposeProbe",
    "PolicyResidualComposeProbe",
    "SpectrumSafePolicyPinCase",
    "SpectrumSafePolicyPinProbe",
    "SpectrumSafePolicyTrajCase",
    "SpectrumSafePolicyTrajProbe",
    "ResidualMassLoopyComposeProbe",
    "SharedFacePair",
    "build_dual_adjacency",
    "build_dual_adjacency_from_complex",
    "build_global_face_registry",
    "build_shared_face_pairs",
    "dry_run_dual_from_edit",
    "solve_conservative_pressures",
    "simplex_outward_normals",
    "accumulate_face_pressure_tally",
    "locate_bmu_simplex",
    "route_live_bmu_face_tallies",
    "route_stage1_bmu_face_tallies",
    "build_node_to_simplices_from_complex",
    "build_simplex_positions_from_complex",
    "query_stage1_ann_bmus",
    "route_stage1_from_complex",
    "route_live_density_from_complex",
    "build_divergence_stencil",
    "conservation_residual_r_cons",
    "epsilon_flux",
    "epsilon_mass",
    "normalize_simplex_masses",
    "solve_as_message_pass",
    "whiten_empirical_pressures",
    "mu_S_weight",
    "count_aware_lambda_f",
    "solve_mu_weighted_pressures",
    "solve_patch_mu_weighted_pressures",
    "solve_global_face_mu_pressures",
    "solve_loopy_bp_schedule",
    "classify_boundary_facets",
    "stitch_orientation_seam_pressures",
    "apply_ghost_reservoir",
    "barycentric_coordinates",
    "vertex_weights_from_facet_pressures",
    "simplex_local_density",
    "affected_subgraph_connected",
    "resolve_dual_connected",
    "probe_acceptance_none_open_default",
    "probe_bp_spectrum_damping",
    "propose_bp_damping_policy",
    "run_online_offline_schedule",
    "run_online_offline_loopy_compose",
    "probe_loopy_bp_convergence",
    "propose_loopy_bp_residual_stop",
    "probe_loopy_bp_spectrum_safe_cert",
    "probe_mass_loopy_compose",
    "probe_policy_residual_compose",
    "probe_spectrum_safe_policy_pin",
    "probe_spectrum_safe_policy_traj",
    "probe_residual_mass_loopy_compose",
    "probe_spectrum_safe_policy_mass_compose",
    "probe_residual_mass_patience_sweep",
    "probe_spectrum_safe_policy_mass_traj",
    "probe_residual_mass_policy_patience",
    "probe_fail_closed_dual_adjacency_plan",
    "probe_gate_fail_closed_switch",
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
    enable_complex_ann_incidence:
        When ``False`` (default), :func:`build_node_to_simplices_from_complex`
        / :func:`build_simplex_positions_from_complex` /
        :func:`query_stage1_ann_bmus` / :func:`route_stage1_from_complex`
        return ``None``. When ``True``, builds node→incident-simplex maps
        from a :class:`~proteus.types.Complex` and queries Stage-1 ANN
        (or naive positions) BMUs to feed the Stage-1 tally sketch
        (A5-EXP-ann-inc). Proposal-path only; does not flip ``@awaiting``.
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
    enable_shared_face_glue:
        When ``False`` (default), the patch soft solve keeps independent
        per-simplex face copies. When ``True`` (and ``simplices`` is
        passed to :func:`solve_patch_mu_weighted_pressures`), adds a soft
        antisymmetry penalty ``Σ_shared (p_a + p_b)²`` on shared facets
        (A5-EXP-glue). Still **not** a global face variable / loopy BP.
    enable_global_face_solve:
        When ``False`` (default), :func:`solve_global_face_mu_pressures`
        returns ``None``. When ``True``, soft-minimizes the SI S6.2
        data + ``Σ_S μ_S‖A_S p_S‖²`` objective over **one pressure per
        unique facet** with signed local incidence (A5-T49 stub). Still
        **not** loopy Gaussian BP — gradient soft solve only.
    enable_loopy_bp_schedule:
        When ``False`` (default), :func:`solve_loopy_bp_schedule` returns
        ``None``. When ``True``, runs a damped Gaussian cavity / factor-
        to-variable message schedule on the global face/factor graph
        (A5-EXP-loopy-bp). Sketch only — not production loopy BP.
    enable_mass_normalization:
        When ``False`` (default), :func:`normalize_simplex_masses` returns
        ``None``. When ``True``, rescales simplex masses so ``Σ m_S = 1``
        and reports SI ``ε_mass`` (A5-EXP-mass). Does **not** flip
        ``@awaiting("stage2.dual_flow")``.
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
    enable_live_density:
        When ``False`` (default), :func:`route_live_density_from_complex`
        returns ``None``. When ``True``, routes samples via Complex/ANN
        incidence then evaluates S6.4 density on the winning simplex
        (A5-T50 harness). Still proposal-path; does not flip ``@awaiting``.
    enable_bp_spectrum_damping_probe:
        When ``False`` (default), :func:`probe_bp_spectrum_damping` returns
        ``None``. When ``True``, runs the A5-T55 spectrum step-shrink /
        loopy-ridge probe on a poorly conditioned fixture (proposal-path;
        does not change production defaults or ``@awaiting``).
    enable_bp_damping_policy:
        When ``False`` (default), :func:`propose_bp_damping_policy` returns
        ``None``. When ``True``, maps a reported Hessian / factor
        condition number to recommended ``bp_damping`` and whether
        factor ridge should apply (A5-T58 production-policy sketch;
        proposal-path only — not certified convergence).
    enable_online_offline_schedule:
        When ``False`` (default), :func:`run_online_offline_schedule`
        returns ``None``. When ``True``, runs the A5-T56 online face
        tallies → offline μ soft-solve schedule sketch (SI S6.2
        paragraph; proposal-path only).
    enable_online_offline_loopy_compose:
        When ``False`` (default), :func:`run_online_offline_loopy_compose`
        returns ``None``. When ``True``, online live-BMU tallies then
        offline loopy BP schedule on the shared face/factor graph
        (A5-T59; proposal-path compose wire).
    enable_bp_policy_in_loopy:
        When ``False`` (default), :func:`solve_loopy_bp_schedule` keeps
        the hard ``cond > spectrum_cond_cap`` ridge and fixed
        ``bp_damping``. When ``True``, each factor consults
        :func:`propose_bp_damping_policy` for per-update damping and
        ridge (A5-T61; proposal-path — not certified).
    enable_loopy_bp_convergence_probe:
        When ``False`` (default), :func:`probe_loopy_bp_convergence`
        returns ``None``. When ``True``, records ``r_data`` / ``r_cons``
        trajectories over increasing iteration counts (A5-T62;
        proposal-path harness — not a production certificate).
    enable_loopy_bp_residual_stop:
        When ``False`` (default), :func:`propose_loopy_bp_residual_stop`
        returns ``None`` and :func:`solve_loopy_bp_schedule` runs the
        full ``bp_max_iters``. When ``True``, sketches a residual-plateau
        / tolerance stop rule over increasing loopy iters (A5-T64) and
        early-exits the in-solver schedule on the same rule (A5-T67;
        proposal-path — **not** a production certificate).
    enable_loopy_bp_spectrum_safe_cert:
        When ``False`` (default), :func:`probe_loopy_bp_spectrum_safe_cert`
        returns ``None``. When ``True``, runs loopy BP with residual-stop
        early-exit and reports a harness ``spectrum_safe_sketch_ok``
        claim (A5-T68; proposal-path — **not** a production certificate).
    enable_mass_loopy_compose_probe:
        When ``False`` (default), :func:`probe_mass_loopy_compose`
        returns ``None``. When ``True``, runs mass normalization together
        with online→offline loopy compose (A5-T66; proposal-path; does
        not flip mass/density ``@awaiting``).
    enable_policy_residual_compose_probe:
        When ``False`` (default), :func:`probe_policy_residual_compose`
        returns ``None``. When ``True``, pins multi-iter residuals under
        ``enable_bp_policy_in_loopy`` and runs online→offline loopy
        compose with residual-stop early-exit (A5-T69; proposal-path;
        does not flip mass/density ``@awaiting``).
    enable_spectrum_safe_policy_pin_probe:
        When ``False`` (default), :func:`probe_spectrum_safe_policy_pin`
        returns ``None``. When ``True``, pins spectrum-safe residual-stop
        harness outcomes across a ``spectrum_cond_cap`` grid with
        ``enable_bp_policy_in_loopy`` on (A5-T70; harness only — **not**
        a production certificate; does not flip ``@awaiting``).
    enable_spectrum_safe_policy_traj_probe:
        When ``False`` (default), :func:`probe_spectrum_safe_policy_traj`
        returns ``None``. When ``True``, exports per-cap ``r_data`` /
        ``r_cons`` trajectories under policy-in-loopy and reports the
        spectrum-safe harness sketch claim (A5-T72; harness only — **not**
        a production certificate; does not flip ``@awaiting``).
    enable_residual_mass_loopy_compose_probe:
        When ``False`` (default), :func:`probe_residual_mass_loopy_compose`
        returns ``None``. When ``True``, pins multi-iter residuals then
        runs mass normalization with online→offline loopy compose under
        residual-stop early-exit (A5-T74; proposal-path; does not flip
        mass/density ``@awaiting``).
    enable_spectrum_safe_policy_mass_compose_probe:
        When ``False`` (default),
        :func:`probe_spectrum_safe_policy_mass_compose` returns ``None``.
        When ``True``, cap-sweeps spectrum-safe×policy residual-stop
        harness outcomes and mass×loopy compose under matching
        ``spectrum_cond_cap`` values (A5-T76; harness only — **not** a
        production certificate; does not flip ``@awaiting``).
    enable_residual_mass_patience_sweep_probe:
        When ``False`` (default),
        :func:`probe_residual_mass_patience_sweep` returns ``None``.
        When ``True``, sweeps ``bp_residual_stop_patience`` under
        mass×loopy compose with residual-stop early-exit (A5-T77;
        proposal-path; does not flip mass/density ``@awaiting``).
    enable_spectrum_safe_policy_mass_traj_probe:
        When ``False`` (default),
        :func:`probe_spectrum_safe_policy_mass_traj` returns ``None``.
        When ``True``, cap-sweeps residual trajectories under
        spectrum-safe×policy-in-loopy and mass×loopy compose under
        matching ``spectrum_cond_cap`` values (A5-T78; harness only —
        **not** a production certificate; does not flip ``@awaiting``).
    enable_residual_mass_policy_patience_probe:
        When ``False`` (default),
        :func:`probe_residual_mass_policy_patience` returns ``None``.
        When ``True``, sweeps ``bp_residual_stop_patience`` under
        mass×loopy compose with residual-stop early-exit **and**
        ``enable_bp_policy_in_loopy`` (A5-T79; proposal-path; does not
        flip mass/density ``@awaiting``).
    bp_residual_stop_tol:
        Absolute plateau tolerance on ``|Δr_data|`` / ``|Δr_cons|`` for
        the residual-stop sketch / early-exit (default ``1e-3``).
        Operational proposal-path only (SI S14.3).
    bp_residual_stop_patience:
        Consecutive plateau steps required before stopping (default
        ``2``). Operational proposal-path only.
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
        SI ``damping when spectra are poorly conditioned``). Also used by
        :func:`solve_loopy_bp_schedule` to ridge local factor precisions.
    shared_face_glue:
        Soft weight on shared-face antisymmetry residuals when
        ``enable_shared_face_glue`` is on (default ``1.0``). Operational
        proposal-path only (SI S14.3).
    ghost_coupling:
        Weak leak fraction in ``[0, 1]`` from computational-boundary
        pressures into the ghost reservoir (default ``0.1``). Operational.
    """

    enable_dual_adjacency: bool = False
    enable_conservative_bp: bool = False
    enable_face_tallies: bool = False
    enable_live_bmu_tally: bool = False
    enable_stage1_bmu_wiring: bool = False
    enable_complex_ann_incidence: bool = False
    enable_as_message_pass: bool = False
    enable_mu_weighted_solve: bool = False
    enable_count_aware_lambda: bool = False
    enable_patch_mu_solve: bool = False
    enable_shared_face_glue: bool = False
    enable_global_face_solve: bool = False
    enable_loopy_bp_schedule: bool = False
    enable_mass_normalization: bool = False
    enable_boundary_taxonomy: bool = False
    enable_seam_ghost: bool = False
    enable_simplex_density: bool = False
    enable_live_density: bool = False
    enable_bp_spectrum_damping_probe: bool = False
    enable_bp_damping_policy: bool = False
    enable_online_offline_schedule: bool = False
    enable_online_offline_loopy_compose: bool = False
    enable_bp_policy_in_loopy: bool = False
    enable_loopy_bp_convergence_probe: bool = False
    enable_loopy_bp_residual_stop: bool = False
    enable_loopy_bp_spectrum_safe_cert: bool = False
    enable_mass_loopy_compose_probe: bool = False
    enable_policy_residual_compose_probe: bool = False
    enable_spectrum_safe_policy_pin_probe: bool = False
    enable_spectrum_safe_policy_traj_probe: bool = False
    enable_residual_mass_loopy_compose_probe: bool = False
    enable_spectrum_safe_policy_mass_compose_probe: bool = False
    enable_residual_mass_patience_sweep_probe: bool = False
    enable_spectrum_safe_policy_mass_traj_probe: bool = False
    enable_residual_mass_policy_patience_probe: bool = False
    bp_residual_stop_tol: float = 1e-3
    bp_residual_stop_patience: int = 2
    bp_damping: float = 0.5
    bp_max_iters: int = 1
    tally_scale: float = 1.0
    volume_floor: float = 1e-12
    as_eps: float = 1e-8
    as_step: float = 0.25
    mu_scale: float = 0.1
    whiten_floor: float = 1e-8
    spectrum_cond_cap: float = 1e6
    shared_face_glue: float = 1.0
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
    stage1_route:
        Optional Complex/ANN Stage-1 BMU → tally bridge on the post-edit
        complex (``None`` when ``enable_complex_ann_incidence`` is off or
        ``samples`` is omitted). A5-T51 dry-run end-to-end wire.
    """

    dual_adjacency: DualAdjacencyDict | None
    affected_simplices: tuple[Hashable, ...]
    dual_connected: bool
    post_edit_complex: Complex
    face_tallies: Mapping[Hashable, FaceTallyResult] | None = None
    stage1_route: Stage1BmuTallyResult | None = None


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
class LiveDensityResult:
    """Live Complex/ANN → S6.4 density harness (SI S6.4; A5-T50).

    ``densities[i]`` is ``p(x_i|S*)`` on the winning simplex for sample
    ``i``. ``assignments`` / ``node_bmus`` mirror the Stage-1 incidence
    bridge. Still proposal-path — does not flip density ``@awaiting``.
    """

    densities: tuple[float, ...]
    assignments: tuple[Hashable, ...]
    node_bmus: tuple[Hashable, ...]
    per_sample: tuple[SimplexDensityResult, ...]
    pressures_by_simplex: Mapping[Hashable, np.ndarray]
    masses_by_simplex: Mapping[Hashable, float]
    note: str = (
        "sketch only: live Complex/ANN density harness; not acceptance-path "
        "density; do not flip @awaiting(stage2.density / stage2.dual_flow)"
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


@dataclass(frozen=True)
class SharedFacePair:
    """One shared facet between two simplices with local face indices.

    Local face index ``i`` is the facet opposite ordered vertex ``i``
    (same convention as :func:`simplex_outward_normals` /
    :func:`build_divergence_stencil` columns). ``facet`` is the unordered
    vertex frozenset of the shared codim-1 face.
    """

    simplex_a: Hashable
    local_face_a: int
    simplex_b: Hashable
    local_face_b: int
    facet: frozenset[Hashable]


def build_shared_face_pairs(
    simplices: Sequence[Sequence[Hashable]] | Mapping[Hashable, Sequence[Hashable]],
) -> tuple[SharedFacePair, ...]:
    """Enumerate shared-facet owner pairs with local face indices (SI S6.2).

    For each simplex with **ordered** vertex sequence ``V``, local face
    ``i`` excludes ``V[i]``. Facets owned by two or more simplices yield
    pairwise :class:`SharedFacePair` entries (first owner paired with each
    later owner). Ungated geometry helper — not a Stage-2 face registry.
    """

    if isinstance(simplices, Mapping):
        items: list[tuple[Hashable, tuple[Hashable, ...]]] = [
            (sid, tuple(verts)) for sid, verts in simplices.items()
        ]
    else:
        items = [(i, tuple(verts)) for i, verts in enumerate(simplices)]

    # facet frozenset -> list of (simplex_id, local_face_index)
    owners: dict[frozenset[Hashable], list[tuple[Hashable, int]]] = defaultdict(
        list
    )
    for sid, verts in items:
        if len(verts) < 2:
            continue
        for i, v in enumerate(verts):
            facet = frozenset(verts) - {v}
            owners[facet].append((sid, i))

    pairs: list[SharedFacePair] = []
    for facet, own in owners.items():
        if len(own) < 2:
            continue
        for j in range(1, len(own)):
            sa, ia = own[0]
            sb, ib = own[j]
            pairs.append(
                SharedFacePair(
                    simplex_a=sa,
                    local_face_a=ia,
                    simplex_b=sb,
                    local_face_b=ib,
                    facet=facet,
                )
            )
    return tuple(pairs)


@dataclass(frozen=True)
class GlobalFaceIncidence:
    """Signed local attachment of a simplex face to a global face id (SI S6.2).

    ``sign`` is ``+1`` for the first owner of a facet (stable simplex-id
    order) and ``-1`` for subsequent owners — a proposal-path stand-in for
    outward-normal antisymmetry on shared facets. Local face index matches
    :func:`simplex_outward_normals` (facet opposite ordered vertex ``i``).
    """

    simplex_id: Hashable
    local_face: int
    global_face: int
    sign: int
    facet: frozenset[Hashable]


@dataclass(frozen=True)
class GlobalFaceRegistry:
    """Unique-facet face registry with signed local incidences (SI S6.2).

    ``facets[g]`` is the unordered vertex frozenset for global face id
    ``g``. ``n_interior`` counts facets with two or more owners. Ungated
    geometry helper — not Stage-2 loopy BP.
    """

    facets: tuple[frozenset[Hashable], ...]
    incidences: tuple[GlobalFaceIncidence, ...]
    n_interior: int

    @property
    def n_faces(self) -> int:
        return len(self.facets)


def build_global_face_registry(
    simplices: Sequence[Sequence[Hashable]] | Mapping[Hashable, Sequence[Hashable]],
) -> GlobalFaceRegistry:
    """Build a global face-id registry from ordered simplex vertex lists.

    Each unique codim-1 facet becomes one global face. Owners are attached
    with signs ``(+1, -1, -1, ...)`` in stable ``(simplex_id, local_face)``
    order so a two-owner interior face is antisymmetric by construction.
    Ungated helper used by :func:`solve_global_face_mu_pressures` (A5-T49).
    """

    if isinstance(simplices, Mapping):
        items: list[tuple[Hashable, tuple[Hashable, ...]]] = [
            (sid, tuple(verts)) for sid, verts in simplices.items()
        ]
    else:
        items = [(i, tuple(verts)) for i, verts in enumerate(simplices)]

    # facet -> list of (simplex_id, local_face)
    owners: dict[frozenset[Hashable], list[tuple[Hashable, int]]] = defaultdict(
        list
    )
    for sid, verts in items:
        if len(verts) < 2:
            continue
        for i, v in enumerate(verts):
            facet = frozenset(verts) - {v}
            owners[facet].append((sid, i))

    # Stable facet order: sorted by sorted-tuple of vertex ids.
    facets = tuple(sorted(owners.keys(), key=lambda f: tuple(sorted(f, key=str))))
    incidences: list[GlobalFaceIncidence] = []
    n_interior = 0
    for g, facet in enumerate(facets):
        own = sorted(owners[facet], key=lambda t: (str(t[0]), t[1]))
        if len(own) >= 2:
            n_interior += 1
        for k, (sid, local_i) in enumerate(own):
            sign = 1 if k == 0 else -1
            incidences.append(
                GlobalFaceIncidence(
                    simplex_id=sid,
                    local_face=int(local_i),
                    global_face=g,
                    sign=sign,
                    facet=facet,
                )
            )
    return GlobalFaceRegistry(
        facets=facets,
        incidences=tuple(incidences),
        n_interior=n_interior,
    )


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

    When ``enable_complex_ann_incidence`` is on and ``samples`` is provided,
    also wires :func:`route_stage1_from_complex` on the post-edit complex
    into ``stage1_route`` (A5-T51 end-to-end dry-run bridge). Flag off or
    no samples ⇒ ``stage1_route`` is ``None``.
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

    stage1_route: Stage1BmuTallyResult | None = None
    if cfg.enable_complex_ann_incidence and samples is not None:
        if post_edit.vertex_positions is None:
            raise ValueError(
                "enable_complex_ann_incidence dry-run requires "
                "complex.vertex_positions"
            )
        if not post_edit.simplices:
            stage1_route = None
        else:
            stage1_route = route_stage1_from_complex(
                samples, post_edit, config=cfg
            )

    return DualDryRunResult(
        dual_adjacency=adj,
        affected_simplices=affected_t,
        dual_connected=connected,
        post_edit_complex=post_edit,
        face_tallies=face_tallies,
        stage1_route=stage1_route,
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


def build_node_to_simplices_from_complex(
    complex: Complex,
    *,
    config: DualFlowConfig | None = None,
) -> dict[Hashable, tuple[Hashable, ...]] | None:
    """Invert Complex incidence: node id → incident simplex ids (SI S6.1).

    When ``enable_complex_ann_incidence`` is off, returns ``None``. When on,
    each top-dimensional simplex index ``i`` is recorded under every
    ``vertex_ids`` entry (A5-EXP-ann-inc). Orphan nodes (no incident
    simplex) are omitted. Proposal-path helper for Stage-1 BMU → star
    candidate lists — does not flip ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_complex_ann_incidence:
        return None

    buckets: dict[Hashable, list[Hashable]] = defaultdict(list)
    for sid, simplex in enumerate(complex.simplices):
        for vid in simplex.vertex_ids:
            buckets[vid].append(sid)
    return {vid: tuple(sids) for vid, sids in buckets.items()}


def build_simplex_positions_from_complex(
    complex: Complex,
    *,
    config: DualFlowConfig | None = None,
) -> dict[Hashable, np.ndarray] | None:
    """Map simplex enumeration id → ``(d+1, D)`` vertex positions (SI S6.1).

    When ``enable_complex_ann_incidence`` is off, returns ``None``. Positions
    are sliced from ``complex.vertex_positions`` by each simplex's
    ``vertex_ids`` (A5-EXP-ann-inc).
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_complex_ann_incidence:
        return None

    V = np.asarray(complex.vertex_positions, dtype=float)
    out: dict[Hashable, np.ndarray] = {}
    for sid, simplex in enumerate(complex.simplices):
        vids = tuple(int(v) for v in simplex.vertex_ids)
        out[sid] = V[list(vids), :]
    return out


def query_stage1_ann_bmus(
    samples: Sequence[np.ndarray],
    *,
    ann: object | None = None,
    node_positions: np.ndarray | None = None,
    config: DualFlowConfig | None = None,
) -> tuple[Hashable, ...] | None:
    """Stage-1 ANN (or naive) BMU node ids for each sample (SI S6.1).

    When ``enable_complex_ann_incidence`` is off, returns ``None``. When on:

    * If ``ann`` exposes ``query_knn(point, k)``, uses ``k=1`` and takes the
      nearest index (duck-typed; works with :class:`proteus.ann.ANNIndex`).
    * Else if ``node_positions`` is ``(N, D)``, uses exact Euclidean argmin
      (naive Stage-1 stand-in).

    Proposal-path only — does not mutate Stage-1 controllers.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_complex_ann_incidence:
        return None

    if ann is None and node_positions is None:
        raise ValueError("provide ann or node_positions for BMU query")

    bmus: list[Hashable] = []
    if ann is not None:
        if not hasattr(ann, "query_knn"):
            raise TypeError("ann must provide query_knn(point, k)")
        for raw in samples:
            idx, _dists = ann.query_knn(np.asarray(raw, dtype=float), k=1)
            if len(idx) == 0:
                raise ValueError("ANN returned empty BMU set")
            bmus.append(int(idx[0]))
        return tuple(bmus)

    P = np.asarray(node_positions, dtype=float)
    if P.ndim != 2 or P.shape[0] == 0:
        raise ValueError("node_positions must be non-empty (N, D)")
    for raw in samples:
        x = np.asarray(raw, dtype=float).reshape(-1)
        if x.shape[0] != P.shape[1]:
            raise ValueError(
                f"sample dim {x.shape[0]} != node_positions dim {P.shape[1]}"
            )
        d2 = np.sum((P - x) ** 2, axis=1)
        bmus.append(int(np.argmin(d2)))
    return tuple(bmus)


def route_stage1_from_complex(
    samples: Sequence[np.ndarray],
    complex: Complex,
    *,
    ann: object | None = None,
    node_positions: np.ndarray | None = None,
    prior_tallies: Mapping[Hashable, np.ndarray] | None = None,
    config: DualFlowConfig | None = None,
) -> Stage1BmuTallyResult | None:
    """Complex + ANN BMU → Stage-1 face-tally wiring sketch (SI S6.1).

    When ``enable_complex_ann_incidence`` is off, returns ``None``. When on:

    1. Build ``node_to_simplices`` / simplex positions from ``complex``.
    2. Query Stage-1 node BMUs via :func:`query_stage1_ann_bmus`.
    3. Delegate to :func:`route_stage1_bmu_face_tallies` (forced on).

    A5-EXP-ann-inc bridge — still proposal-path; does not flip ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_complex_ann_incidence:
        return None

    node_map = build_node_to_simplices_from_complex(complex, config=cfg)
    pos_map = build_simplex_positions_from_complex(complex, config=cfg)
    if node_map is None or pos_map is None:
        raise RuntimeError("complex incidence unexpectedly disabled")
    if not node_map or not pos_map:
        raise ValueError("complex must contain at least one simplex")

    positions = (
        np.asarray(node_positions, dtype=float)
        if node_positions is not None
        else np.asarray(complex.vertex_positions, dtype=float)
    )
    node_bmus = query_stage1_ann_bmus(
        samples, ann=ann, node_positions=positions if ann is None else None, config=cfg
    )
    if node_bmus is None:
        raise RuntimeError("ANN BMU query unexpectedly disabled")

    tally_cfg = DualFlowConfig(
        enable_stage1_bmu_wiring=True,
        tally_scale=cfg.tally_scale,
        volume_floor=cfg.volume_floor,
    )
    return route_stage1_bmu_face_tallies(
        samples,
        node_bmus,
        node_map,
        pos_map,
        prior_tallies=prior_tallies,
        config=tally_cfg,
    )


def route_live_density_from_complex(
    samples: Sequence[np.ndarray],
    complex: Complex,
    *,
    pressures_by_simplex: Mapping[Hashable, np.ndarray] | None = None,
    masses_by_simplex: Mapping[Hashable, float] | None = None,
    ann: object | None = None,
    node_positions: np.ndarray | None = None,
    prior_tallies: Mapping[Hashable, np.ndarray] | None = None,
    config: DualFlowConfig | None = None,
) -> LiveDensityResult | None:
    """Live Complex/ANN BMU → S6.4 density harness (SI S6.4; A5-T50).

    When ``enable_live_density`` is off, returns ``None``. When on:

    1. Route samples via :func:`route_stage1_from_complex` (Complex star +
       ANN/naive BMU → winning simplex + face tallies).
    2. For each sample, evaluate :func:`simplex_local_density` on the
       winning simplex using ``pressures_by_simplex[S]`` when supplied,
       else the cumulative face tallies for ``S``, with mass
       ``masses_by_simplex.get(S, 1/n_simplices)``.

    Proposal-path only — does **not** flip ``@awaiting("stage2.density")``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_live_density:
        return None

    if not complex.simplices:
        raise ValueError("complex must contain at least one simplex")

    bridge_cfg = DualFlowConfig(
        enable_complex_ann_incidence=True,
        tally_scale=cfg.tally_scale,
        volume_floor=cfg.volume_floor,
    )
    routed = route_stage1_from_complex(
        samples,
        complex,
        ann=ann,
        node_positions=node_positions,
        prior_tallies=prior_tallies,
        config=bridge_cfg,
    )
    if routed is None:
        raise RuntimeError("complex ANN incidence unexpectedly disabled")

    n_S = len(complex.simplices)
    default_mass = 1.0 / float(n_S)
    dens_cfg = DualFlowConfig(
        enable_simplex_density=True,
        volume_floor=cfg.volume_floor,
    )

    # Resolve pressures: explicit map wins; else tallies; else ones.
    press_map: dict[Hashable, np.ndarray] = {}
    mass_map: dict[Hashable, float] = {}
    for sid, simp in enumerate(complex.simplices):
        n_faces = len(simp.vertex_ids)
        if pressures_by_simplex is not None and sid in pressures_by_simplex:
            press_map[sid] = np.asarray(
                pressures_by_simplex[sid], dtype=float
            ).reshape(-1)
        elif sid in routed.tallies_by_simplex:
            press_map[sid] = np.asarray(
                routed.tallies_by_simplex[sid].tallies, dtype=float
            ).reshape(-1)
        else:
            press_map[sid] = np.ones(n_faces, dtype=float)
        if masses_by_simplex is not None and sid in masses_by_simplex:
            mass_map[sid] = float(masses_by_simplex[sid])
        else:
            mass_map[sid] = default_mass

    pos_map = build_simplex_positions_from_complex(complex, config=bridge_cfg)
    if pos_map is None:
        raise RuntimeError("complex positions unexpectedly disabled")

    per_sample: list[SimplexDensityResult] = []
    densities: list[float] = []
    for sample, sid in zip(samples, routed.assignments, strict=True):
        if sid not in pos_map:
            raise ValueError(f"winning simplex {sid!r} missing positions")
        P = pos_map[sid]
        press = press_map[sid]
        if press.shape[0] != P.shape[0]:
            raise ValueError(
                f"pressures for simplex {sid!r} length {press.shape[0]} "
                f"!= vertex count {P.shape[0]}"
            )
        vol = None
        if 0 <= int(sid) < len(complex.simplices):
            vol = float(complex.simplices[int(sid)].volume)
        out = simplex_local_density(
            sample,
            P,
            mass=mass_map[sid],
            facet_pressures=press,
            volume=vol,
            config=dens_cfg,
        )
        if out is None:
            raise RuntimeError("simplex density unexpectedly disabled")
        per_sample.append(out)
        densities.append(out.density)

    return LiveDensityResult(
        densities=tuple(densities),
        assignments=routed.assignments,
        node_bmus=routed.node_bmus,
        per_sample=tuple(per_sample),
        pressures_by_simplex=press_map,
        masses_by_simplex=mass_map,
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


def epsilon_mass(
    masses: Mapping[Hashable, float],
    *,
    target: float = 1.0,
) -> float:
    """SI S6.2 ``ε_mass = |Σ_S m_S - target|`` (ungated helper).

    After :func:`normalize_simplex_masses`, this should be ``<= 1e-6``.
    Does **not** flip ``@awaiting("stage2.dual_flow")``.
    """

    if not masses:
        raise ValueError("masses must be non-empty")
    total = float(sum(float(v) for v in masses.values()))
    return abs(total - float(target))


def normalize_simplex_masses(
    masses: Mapping[Hashable, float],
    *,
    config: DualFlowConfig | None = None,
) -> MassNormalizationResult | None:
    """Rescale simplex masses so ``Σ m_S = 1`` (SI S6.2; A5-EXP-mass).

    When ``enable_mass_normalization`` is off, returns ``None``. When on,
    divides each mass by the pre-normalization total (must be ``> 0``)
    and reports ``ε_mass``. Proposal-path harness only — do **not** flip
    mass-conservation ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_mass_normalization:
        return None
    if not masses:
        raise ValueError("masses must be non-empty")
    total = float(sum(float(v) for v in masses.values()))
    if total <= 0.0:
        raise ValueError("mass total must be > 0 for normalization")
    scaled = {k: float(v) / total for k, v in masses.items()}
    return MassNormalizationResult(
        masses=scaled,
        total_before=total,
        epsilon_mass=epsilon_mass(scaled),
    )


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

    Pressures are concatenated per-simplex face blocks. When shared-face
    glue is off, blocks are independent copies (shared-face identification
    is optional via ``enable_shared_face_glue`` / A5-EXP-glue). ``mu_S``
    maps simplex id → local conservation weight; ``mu_S_sum`` is their sum.
    ``n_shared_faces`` counts glued shared-facet pairs (0 when glue off).
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
    n_shared_faces: int = 0
    shared_glue_residual: float = 0.0
    note: str = (
        "sketch only: block-concat patch Σ μ_S‖A_S p_S‖²; not shared "
        "face-registry / loopy Gaussian BP (SI S6.2)"
    )


def solve_patch_mu_weighted_pressures(
    empirical_by_simplex: Mapping[Hashable, np.ndarray],
    stencils_by_simplex: Mapping[Hashable, np.ndarray],
    *,
    face_hit_counts_by_simplex: Mapping[Hashable, np.ndarray] | None = None,
    simplices: Sequence[Sequence[Hashable]]
    | Mapping[Hashable, Sequence[Hashable]]
    | None = None,
    config: DualFlowConfig | None = None,
) -> PatchMuSolveResult | None:
    """Multi-simplex patch soft solve (SI S6.2; A5-T47 / A5-EXP-glue).

    When ``enable_patch_mu_solve`` is off, returns ``None``. When on,
    soft-minimizes

        Σ_f λ_f (p_f - hat_f)² + Σ_S μ_S ‖A_S p_S‖₂²

    over **block-concatenated** per-simplex face pressures. Optional
    count-aware ``λ_f`` when ``enable_count_aware_lambda`` and hit counts
    are supplied.

    When ``enable_shared_face_glue`` is on, ``simplices`` (ordered vertex
    ids per simplex) is required; shared facets contribute soft
    antisymmetry residuals ``(p_a + p_b)²`` weighted by
    ``shared_face_glue`` (outward normals on a shared face oppose, so
    pressures should cancel). This is **not** a global face variable or
    loopy BP — only a soft glue on private copies.

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

    glue_pairs: tuple[SharedFacePair, ...] = ()
    if cfg.enable_shared_face_glue:
        if simplices is None:
            raise ValueError(
                "enable_shared_face_glue requires simplices (ordered "
                "vertex ids per simplex)"
            )
        glue_pairs = build_shared_face_pairs(simplices)
        glue_w = float(cfg.shared_face_glue)
        if glue_w < 0.0:
            raise ValueError("shared_face_glue must be >= 0")

    blocks_hat: list[np.ndarray] = []
    blocks_A: list[np.ndarray] = []
    blocks_lam: list[np.ndarray] = []
    mu_map: dict[Hashable, float] = {}
    block_sizes: list[int] = []
    offsets: dict[Hashable, int] = {}

    offset = 0
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
        offsets[sid] = offset
        offset += n

    hat_raw = np.concatenate(blocks_hat)
    lam_all = np.concatenate(blocks_lam)
    # Block-diagonal soft Hessian: per-simplex AtA scaled by μ_S.
    n_tot = hat_raw.shape[0]
    AtA_big = np.zeros((n_tot, n_tot), dtype=float)
    for sid, A_S, n in zip(ids, blocks_A, block_sizes, strict=True):
        AtA = A_S.T @ A_S
        sl = slice(offsets[sid], offsets[sid] + n)
        AtA_big[sl, sl] = float(mu_map[sid]) * AtA

    # Resolve glue pair global indices (skip pairs whose simplex is absent).
    glue_idx: list[tuple[int, int]] = []
    for pair in glue_pairs:
        if pair.simplex_a not in offsets or pair.simplex_b not in offsets:
            continue
        ia = offsets[pair.simplex_a] + int(pair.local_face_a)
        ib = offsets[pair.simplex_b] + int(pair.local_face_b)
        if ia >= n_tot or ib >= n_tot:
            raise ValueError(
                f"shared-face local index out of range for pair "
                f"{pair.simplex_a!r}/{pair.simplex_b!r}"
            )
        glue_idx.append((ia, ib))

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
    glue_w = float(cfg.shared_face_glue) if cfg.enable_shared_face_glue else 0.0
    # Scale soft glue into whitened-data units so large |hat| faces do not
    # dominate / diverge the soft step (proposal-path operational).
    std2_bar = float(np.mean(std * std)) + float(cfg.whiten_floor)
    lam_bar = float(np.mean(lam_all)) + float(cfg.whiten_floor)
    glue_eff = glue_w * lam_bar / std2_bar
    for _ in range(iters):
        p_w = (1.0 - damp) * hat_w + damp * p_w
        p_phys = p_w * std
        # AtA_big already folds μ_S into each block; whitened conservation
        # gradient is std ⊙ (AtA_big @ p_phys).
        grad = lam_all * (p_w - hat_w) + std * (AtA_big @ p_phys)
        if glue_eff > 0.0 and glue_idx:
            # Soft ‖p_a + p_b‖² on physical pressures; chain-rule via std.
            for ia, ib in glue_idx:
                resid = float(p_phys[ia] + p_phys[ib])
                grad[ia] += glue_eff * resid * float(std[ia])
                grad[ib] += glue_eff * resid * float(std[ib])
        p_w = p_w - step * grad

    p = p_w * std
    # Hard antisymmetry projection locks shared faces (sketch stand-in for
    # identifying a single oriented face variable).
    if cfg.enable_shared_face_glue and glue_idx:
        for ia, ib in glue_idx:
            a = float(p[ia])
            b = float(p[ib])
            p[ia] = 0.5 * (a - b)
            p[ib] = 0.5 * (b - a)
    eps = 1e-12
    r_data = float(np.sum((p - hat_raw) ** 2) / (np.sum(hat_raw**2) + eps))

    # Aggregate r_cons / ε_flux over blocks.
    flux2 = 0.0
    cons_num = 0.0
    for sid, A_S, n in zip(ids, blocks_A, block_sizes, strict=True):
        off = offsets[sid]
        p_S = p[off : off + n]
        Ap = A_S @ p_S
        f2 = float(np.dot(Ap, Ap))
        fro2 = float(np.sum(A_S * A_S))
        flux2 += f2
        cons_num += f2 / (fro2 + float(cfg.as_eps))
    denom = float(np.sum(p * p)) + eps
    r_cons = cons_num / denom
    e_flux = flux2 / denom
    mu_sum = float(sum(mu_map.values()))

    glue_resid = 0.0
    for ia, ib in glue_idx:
        glue_resid += float(p[ia] + p[ib]) ** 2
    n_shared = len(glue_idx)

    note = (
        "sketch only: block-concat patch Σ μ_S‖A_S p_S‖²; not shared "
        "face-registry / loopy Gaussian BP (SI S6.2)"
    )
    if cfg.enable_shared_face_glue:
        note = (
            "sketch only: patch Σ μ_S + scaled soft shared-face glue + "
            "hard antisym projection; not global face registry / loopy "
            "Gaussian BP (SI S6.2)"
        )

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
        n_shared_faces=n_shared,
        shared_glue_residual=glue_resid,
        note=note,
    )


@dataclass(frozen=True)
class GlobalFaceSolveResult:
    """Global face-id soft solve stub (SI S6.2; A5-T49).

    ``pressures_global`` has one entry per unique facet. ``pressures_local``
    is the signed expansion onto per-simplex face slots (same block-concat
    layout as :class:`PatchMuSolveResult`). ``n_interior_faces`` counts
    facets with ≥2 owners. Still **not** loopy Gaussian BP.
    """

    empirical_local: np.ndarray
    empirical_global: np.ndarray
    pressures_global: np.ndarray
    pressures_local: np.ndarray
    lambda_f_global: np.ndarray
    mu_S: Mapping[Hashable, float]
    mu_S_sum: float
    r_data: float
    r_cons: float
    epsilon_flux: float
    iters: int
    block_sizes: tuple[int, ...]
    simplex_ids: tuple[Hashable, ...]
    n_faces: int
    n_interior_faces: int
    registry: GlobalFaceRegistry
    note: str = (
        "sketch only: global face-id soft solve (signed incidence); "
        "not loopy Gaussian BP (SI S6.2)"
    )


@dataclass(frozen=True)
class LoopyBPScheduleResult:
    """Loopy Gaussian BP message-schedule sketch (SI S6.2; A5-EXP-loopy-bp).

    Face variables are unique global facets; simplex factors encode
    ``μ_S‖A_S p_S‖²``. Cavity messages are 1-D Gaussian (precision,
    information). Still a proposal-path sketch — not production BP.
    """

    empirical_local: np.ndarray
    empirical_global: np.ndarray
    pressures_global: np.ndarray
    pressures_local: np.ndarray
    lambda_f_global: np.ndarray
    mu_S: Mapping[Hashable, float]
    mu_S_sum: float
    r_data: float
    r_cons: float
    epsilon_flux: float
    iters: int
    block_sizes: tuple[int, ...]
    simplex_ids: tuple[Hashable, ...]
    n_faces: int
    n_interior_faces: int
    n_factors: int
    message_updates: int
    registry: GlobalFaceRegistry
    spectrum_ridge_applied: bool = False
    policy_applied: bool = False
    max_policy_damping: float = 0.0
    residual_stop_enabled: bool = False
    residual_stop_reason: str | None = None
    note: str = (
        "sketch only: loopy Gaussian BP message schedule on face/factor "
        "graph; not production BP; do not flip @awaiting(stage2.dual_flow)"
    )


@dataclass(frozen=True)
class MassNormalizationResult:
    """Simplex-mass normalization harness (SI S6.2; A5-EXP-mass).

    After rescaling, ``Σ m_S = 1`` (up to float noise) and
    ``epsilon_mass = |Σ m_S - 1|``. Does **not** flip mass-conservation
    ``@awaiting``.
    """

    masses: Mapping[Hashable, float]
    total_before: float
    epsilon_mass: float
    note: str = (
        "sketch only: simplex-mass normalization; do not flip "
        "@awaiting(stage2.dual_flow)"
    )


def solve_global_face_mu_pressures(
    empirical_by_simplex: Mapping[Hashable, np.ndarray],
    stencils_by_simplex: Mapping[Hashable, np.ndarray],
    simplices: Sequence[Sequence[Hashable]]
    | Mapping[Hashable, Sequence[Hashable]],
    *,
    face_hit_counts_by_simplex: Mapping[Hashable, np.ndarray] | None = None,
    config: DualFlowConfig | None = None,
) -> GlobalFaceSolveResult | None:
    """Soft solve on unique facet pressures (SI S6.2; A5-T49).

    When ``enable_global_face_solve`` is off, returns ``None``. When on,
    builds a :class:`GlobalFaceRegistry` and soft-minimizes

        Σ_{S,i} λ (s_{S,i} p_{g(S,i)} - hat_{S,i})²
        + Σ_S μ_S ‖A_S p_S‖₂²

    with ``p_S[i] = s_{S,i} p_{g(S,i)}``. Optional count-aware ``λ_f``
    when ``enable_count_aware_lambda`` and hit counts are supplied
    (aggregated onto global faces by mean of owner λ). Gradient soft
    solve with damping — **not** loopy Gaussian BP.

    Proposal-path stub only — do **not** flip ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_global_face_solve:
        return None

    if not empirical_by_simplex:
        raise ValueError("empirical_by_simplex must be non-empty")
    ids = tuple(empirical_by_simplex.keys())
    for sid in ids:
        if sid not in stencils_by_simplex:
            raise ValueError(f"missing divergence stencil for simplex {sid!r}")

    registry = build_global_face_registry(simplices)
    # Incidence lookup: (sid, local_i) -> (g, sign)
    loc_map: dict[tuple[Hashable, int], tuple[int, int]] = {}
    for inc in registry.incidences:
        loc_map[(inc.simplex_id, inc.local_face)] = (
            inc.global_face,
            inc.sign,
        )

    blocks_hat: list[np.ndarray] = []
    blocks_A: list[np.ndarray] = []
    blocks_lam: list[np.ndarray] = []
    mu_map: dict[Hashable, float] = {}
    block_sizes: list[int] = []
    offsets: dict[Hashable, int] = {}

    offset = 0
    for sid in ids:
        hat = np.asarray(empirical_by_simplex[sid], dtype=float).reshape(-1)
        A_S = np.asarray(stencils_by_simplex[sid], dtype=float)
        if A_S.ndim != 2 or A_S.shape[1] != hat.shape[0]:
            raise ValueError(
                f"stencil {sid!r} shape {A_S.shape} incompatible with "
                f"pressures length {hat.shape[0]}"
            )
        n = hat.shape[0]
        for i in range(n):
            if (sid, i) not in loc_map:
                raise ValueError(
                    f"simplex {sid!r} local face {i} missing from face "
                    f"registry (check simplices keys match empirical ids)"
                )
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
        offsets[sid] = offset
        offset += n

    hat_local = np.concatenate(blocks_hat)
    lam_local = np.concatenate(blocks_lam)
    n_local = hat_local.shape[0]
    n_g = registry.n_faces
    if n_g < 1:
        raise ValueError("global face registry is empty")

    # Incidence matrix M: local = M @ p_global  (entries ±1).
    M = np.zeros((n_local, n_g), dtype=float)
    local_g: list[int] = [-1] * n_local
    local_s: list[float] = [0.0] * n_local
    for sid, n in zip(ids, block_sizes, strict=True):
        off = offsets[sid]
        for i in range(n):
            g, s = loc_map[(sid, i)]
            M[off + i, g] = float(s)
            local_g[off + i] = g
            local_s[off + i] = float(s)

    # Aggregate empirical / λ onto global faces (signed mean of owners).
    hat_g = np.zeros(n_g, dtype=float)
    lam_g = np.zeros(n_g, dtype=float)
    counts_g = np.zeros(n_g, dtype=float)
    for loc_i in range(n_local):
        g = local_g[loc_i]
        s = local_s[loc_i]
        hat_g[g] += s * float(hat_local[loc_i])
        lam_g[g] += float(lam_local[loc_i])
        counts_g[g] += 1.0
    counts_g = np.maximum(counts_g, 1.0)
    hat_g = hat_g / counts_g
    lam_g = lam_g / counts_g

    damp = float(cfg.bp_damping)
    if not 0.0 <= damp <= 1.0:
        raise ValueError("bp_damping must be in [0, 1]")
    iters = int(cfg.bp_max_iters)
    if iters < 1:
        raise ValueError("bp_max_iters must be >= 1")
    step = float(cfg.as_step)
    if step < 0.0:
        raise ValueError("as_step must be >= 0")

    # Whiten global empirics; soft-solve in whitened coords.
    hat_w, std = whiten_empirical_pressures(
        hat_g, None, floor=float(cfg.whiten_floor)
    )
    p_w = hat_w.copy()

    # Precompute A_S M_S blocks for conservation gradient.
    # p_local_phys = M @ (p_w * std); grad_w via chain rule.
    for _ in range(iters):
        p_w = (1.0 - damp) * hat_w + damp * p_w
        p_g = p_w * std
        p_loc = M @ p_g
        # Data term on whitened global aggregated hats.
        grad = lam_g * (p_w - hat_w)
        # Conservation: Σ_S μ_S ‖A_S p_S‖²; p_S = (M p_g)[block].
        cons_grad_g = np.zeros(n_g, dtype=float)
        for sid, A_S, n in zip(ids, blocks_A, block_sizes, strict=True):
            off = offsets[sid]
            p_S = p_loc[off : off + n]
            # d/dp_g  μ ‖A p_S‖² = 2 μ M_S^T A^T A p_S  (½ absorbed in step)
            AtAp = A_S.T @ (A_S @ p_S)
            for i in range(n):
                g = local_g[off + i]
                s = local_s[off + i]
                cons_grad_g[g] += float(mu_map[sid]) * s * float(AtAp[i])
        # Chain rule into whitened coords: ∂p_g/∂p_w = std.
        grad = grad + std * cons_grad_g
        p_w = p_w - step * grad

    p_g = p_w * std
    p_loc = M @ p_g

    eps = 1e-12
    r_data = float(
        np.sum((p_loc - hat_local) ** 2) / (np.sum(hat_local**2) + eps)
    )
    flux2 = 0.0
    cons_num = 0.0
    for sid, A_S, n in zip(ids, blocks_A, block_sizes, strict=True):
        off = offsets[sid]
        p_S = p_loc[off : off + n]
        Ap = A_S @ p_S
        f2 = float(np.dot(Ap, Ap))
        fro2 = float(np.sum(A_S * A_S))
        flux2 += f2
        cons_num += f2 / (fro2 + float(cfg.as_eps))
    denom = float(np.sum(p_g * p_g)) + eps
    r_cons = cons_num / denom
    e_flux = flux2 / denom
    mu_sum = float(sum(mu_map.values()))

    return GlobalFaceSolveResult(
        empirical_local=hat_local,
        empirical_global=hat_g,
        pressures_global=p_g,
        pressures_local=p_loc,
        lambda_f_global=lam_g,
        mu_S=mu_map,
        mu_S_sum=mu_sum,
        r_data=r_data,
        r_cons=r_cons,
        epsilon_flux=e_flux,
        iters=iters,
        block_sizes=tuple(block_sizes),
        simplex_ids=ids,
        n_faces=n_g,
        n_interior_faces=registry.n_interior,
        registry=registry,
    )


def solve_loopy_bp_schedule(
    empirical_by_simplex: Mapping[Hashable, np.ndarray],
    stencils_by_simplex: Mapping[Hashable, np.ndarray],
    simplices: Sequence[Sequence[Hashable]]
    | Mapping[Hashable, Sequence[Hashable]],
    *,
    face_hit_counts_by_simplex: Mapping[Hashable, np.ndarray] | None = None,
    config: DualFlowConfig | None = None,
) -> LoopyBPScheduleResult | None:
    """Loopy Gaussian BP message schedule on the face/factor graph (SI S6.2).

    When ``enable_loopy_bp_schedule`` is off, returns ``None``. When on:

    * Variables = unique global facet pressures (whitened).
    * Unary data factors: ``λ_f (p_f - hat_f)²``.
    * Simplex factors: ``μ_S ‖A_S p_S‖²`` with signed local incidence.
    * Each iteration updates factor→variable cavity messages
      (precision / information) with damping; beliefs = unary ⊕ messages.

    Proposal-path sketch (A5-EXP-loopy-bp) — **not** production BP with
    certified convergence. When ``enable_loopy_bp_residual_stop`` is also
    on, early-exits on residual plateau / absolute tolerance (A5-T67;
    still not a production certificate). Do **not** flip
    ``@awaiting("stage2.dual_flow")``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_loopy_bp_schedule:
        return None

    if not empirical_by_simplex:
        raise ValueError("empirical_by_simplex must be non-empty")
    ids = tuple(empirical_by_simplex.keys())
    for sid in ids:
        if sid not in stencils_by_simplex:
            raise ValueError(f"missing divergence stencil for simplex {sid!r}")

    registry = build_global_face_registry(simplices)
    loc_map: dict[tuple[Hashable, int], tuple[int, int]] = {}
    for inc in registry.incidences:
        loc_map[(inc.simplex_id, inc.local_face)] = (
            inc.global_face,
            inc.sign,
        )

    blocks_hat: list[np.ndarray] = []
    blocks_A: list[np.ndarray] = []
    blocks_lam: list[np.ndarray] = []
    mu_map: dict[Hashable, float] = {}
    block_sizes: list[int] = []
    offsets: dict[Hashable, int] = {}
    # Per-factor face index lists into global face ids / signs.
    factor_faces: dict[Hashable, list[int]] = {}
    factor_signs: dict[Hashable, list[float]] = {}

    offset = 0
    for sid in ids:
        hat = np.asarray(empirical_by_simplex[sid], dtype=float).reshape(-1)
        A_S = np.asarray(stencils_by_simplex[sid], dtype=float)
        if A_S.ndim != 2 or A_S.shape[1] != hat.shape[0]:
            raise ValueError(
                f"stencil {sid!r} shape {A_S.shape} incompatible with "
                f"pressures length {hat.shape[0]}"
            )
        n = hat.shape[0]
        g_ids: list[int] = []
        signs: list[float] = []
        for i in range(n):
            if (sid, i) not in loc_map:
                raise ValueError(
                    f"simplex {sid!r} local face {i} missing from face "
                    f"registry (check simplices keys match empirical ids)"
                )
            g, s = loc_map[(sid, i)]
            g_ids.append(g)
            signs.append(float(s))
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
        offsets[sid] = offset
        offset += n
        factor_faces[sid] = g_ids
        factor_signs[sid] = signs

    hat_local = np.concatenate(blocks_hat)
    lam_local = np.concatenate(blocks_lam)
    n_local = hat_local.shape[0]
    n_g = registry.n_faces
    if n_g < 1:
        raise ValueError("global face registry is empty")

    M = np.zeros((n_local, n_g), dtype=float)
    local_g: list[int] = [-1] * n_local
    local_s: list[float] = [0.0] * n_local
    for sid, n in zip(ids, block_sizes, strict=True):
        off = offsets[sid]
        for i in range(n):
            g, s = loc_map[(sid, i)]
            M[off + i, g] = float(s)
            local_g[off + i] = g
            local_s[off + i] = float(s)

    hat_g = np.zeros(n_g, dtype=float)
    lam_g = np.zeros(n_g, dtype=float)
    counts_g = np.zeros(n_g, dtype=float)
    for loc_i in range(n_local):
        g = local_g[loc_i]
        s = local_s[loc_i]
        hat_g[g] += s * float(hat_local[loc_i])
        lam_g[g] += float(lam_local[loc_i])
        counts_g[g] += 1.0
    counts_g = np.maximum(counts_g, 1.0)
    hat_g = hat_g / counts_g
    lam_g = lam_g / counts_g

    damp = float(cfg.bp_damping)
    if not 0.0 <= damp <= 1.0:
        raise ValueError("bp_damping must be in [0, 1]")
    iters = int(cfg.bp_max_iters)
    if iters < 1:
        raise ValueError("bp_max_iters must be >= 1")
    cond_cap = float(cfg.spectrum_cond_cap)
    if cond_cap <= 0.0:
        raise ValueError("spectrum_cond_cap must be > 0")

    hat_w, std = whiten_empirical_pressures(
        hat_g, None, floor=float(cfg.whiten_floor)
    )
    # Unary natural params for energy λ(p-hat)² ↔ ½ P p² - i p:
    # P_u = 2λ, i_u = 2λ hat.
    P_u = 2.0 * lam_g
    i_u = 2.0 * lam_g * hat_w

    # Messages: factor → face index within factor. Precision / info.
    msg_P: dict[Hashable, np.ndarray] = {
        sid: np.zeros(n, dtype=float) for sid, n in zip(ids, block_sizes)
    }
    msg_i: dict[Hashable, np.ndarray] = {
        sid: np.zeros(n, dtype=float) for sid, n in zip(ids, block_sizes)
    }

    # Face → list of (sid, local_idx_in_factor)
    face_owners: dict[int, list[tuple[Hashable, int]]] = defaultdict(list)
    for sid, n in zip(ids, block_sizes, strict=True):
        for li, g in enumerate(factor_faces[sid]):
            face_owners[g].append((sid, li))

    message_updates = 0
    spectrum_ridge_applied = False
    policy_applied = False
    max_policy_damping = 0.0
    residual_stop_enabled = bool(cfg.enable_loopy_bp_residual_stop)
    residual_stop_reason: str | None = None
    stop_tol = float(cfg.bp_residual_stop_tol)
    stop_patience = int(cfg.bp_residual_stop_patience)
    if residual_stop_enabled:
        if stop_tol < 0.0:
            raise ValueError("bp_residual_stop_tol must be >= 0")
        if stop_patience < 1:
            raise ValueError("bp_residual_stop_patience must be >= 1")
    prev_r_data: float | None = None
    prev_r_cons: float | None = None
    plateau = 0
    iters_executed = 0
    policy_cfg = DualFlowConfig(
        enable_bp_damping_policy=True,
        bp_damping=float(cfg.bp_damping),
        spectrum_cond_cap=float(cfg.spectrum_cond_cap),
    )

    def _current_residuals() -> tuple[float, float]:
        P_b_loc = P_u.copy()
        i_b_loc = i_u.copy()
        for sid_r, n_r in zip(ids, block_sizes, strict=True):
            for li_r, g_r in enumerate(factor_faces[sid_r]):
                P_b_loc[g_r] += float(msg_P[sid_r][li_r])
                i_b_loc[g_r] += float(msg_i[sid_r][li_r])
        p_w_loc = np.zeros(n_g, dtype=float)
        for g_r in range(n_g):
            if P_b_loc[g_r] > 0.0 and np.isfinite(P_b_loc[g_r]):
                p_w_loc[g_r] = float(i_b_loc[g_r]) / float(P_b_loc[g_r])
            else:
                p_w_loc[g_r] = float(hat_w[g_r])
        p_g_loc = p_w_loc * std
        p_loc_r = M @ p_g_loc
        eps_r = 1e-12
        rd = float(
            np.sum((p_loc_r - hat_local) ** 2)
            / (np.sum(hat_local**2) + eps_r)
        )
        cons_num_r = 0.0
        for sid_r, A_S_r, n_r in zip(ids, blocks_A, block_sizes, strict=True):
            off_r = offsets[sid_r]
            p_S_r = p_loc_r[off_r : off_r + n_r]
            Ap_r = A_S_r @ p_S_r
            f2_r = float(np.dot(Ap_r, Ap_r))
            fro2_r = float(np.sum(A_S_r * A_S_r))
            cons_num_r += f2_r / (fro2_r + float(cfg.as_eps))
        denom_r = float(np.sum(p_g_loc * p_g_loc)) + eps_r
        rc = cons_num_r / denom_r
        return rd, rc

    for _ in range(iters):
        for sid, A_S, n in zip(ids, blocks_A, block_sizes, strict=True):
            g_ids = factor_faces[sid]
            signs = factor_signs[sid]
            # Cavity = unary ⊕ other factors' messages.
            P_cav = np.zeros(n, dtype=float)
            i_cav = np.zeros(n, dtype=float)
            for li, g in enumerate(g_ids):
                P_cav[li] = float(P_u[g])
                i_cav[li] = float(i_u[g])
                for oid, o_li in face_owners[g]:
                    if oid == sid and o_li == li:
                        continue
                    P_cav[li] += float(msg_P[oid][o_li])
                    i_cav[li] += float(msg_i[oid][o_li])

            # Factor energy μ‖A p_S‖² with p_S[k] = sign_k * std_g * p_w[g].
            # B columns: B[:,k] = sign_k * std[g_k] * A[:,k]
            scale = np.array(
                [signs[k] * float(std[g_ids[k]]) for k in range(n)],
                dtype=float,
            )
            B = A_S * scale.reshape(1, -1)
            # Energy μ p^T B^T B p = ½ p^T (2μ B^T B) p
            J = np.diag(P_cav) + (2.0 * float(mu_map[sid])) * (B.T @ B)
            # Spectrum ridge when poorly conditioned (SI damping stand-in).
            try:
                cond = float(np.linalg.cond(J))
            except np.linalg.LinAlgError:
                cond = float("inf")
            factor_damp = damp
            apply_ridge = (not np.isfinite(cond)) or (cond > cond_cap)
            if cfg.enable_bp_policy_in_loopy:
                # A5-T61: consult production damping-policy sketch per factor.
                cond_for_policy = (
                    cond if np.isfinite(cond) else float(cond_cap) * 1e12
                )
                pol = propose_bp_damping_policy(
                    cond_for_policy, config=policy_cfg
                )
                if pol is not None:
                    policy_applied = True
                    factor_damp = float(pol.recommended_damping)
                    apply_ridge = bool(pol.apply_ridge)
                    if factor_damp > max_policy_damping:
                        max_policy_damping = factor_damp
            if apply_ridge:
                ridge = float(np.mean(np.abs(np.diag(J)))) + float(cfg.as_eps)
                J = J + ridge * np.eye(n)
                spectrum_ridge_applied = True
            try:
                mean = np.linalg.solve(J, i_cav)
            except np.linalg.LinAlgError:
                mean = np.linalg.lstsq(J, i_cav, rcond=None)[0]

            # Local marginals → messages = marg ⊖ cavity (damped).
            try:
                cov = np.linalg.inv(J)
            except np.linalg.LinAlgError:
                cov = np.linalg.pinv(J)
            for li in range(n):
                var = float(cov[li, li])
                if var <= 0.0 or not np.isfinite(var):
                    P_marg = float(P_cav[li]) + float(cfg.as_eps)
                else:
                    P_marg = 1.0 / var
                i_marg = P_marg * float(mean[li])
                P_new = P_marg - float(P_cav[li])
                i_new = i_marg - float(i_cav[li])
                # Keep messages finite; clamp tiny negative precision noise.
                if not np.isfinite(P_new):
                    P_new = 0.0
                if not np.isfinite(i_new):
                    i_new = 0.0
                if P_new < 0.0:
                    P_new = 0.0
                msg_P[sid][li] = (
                    (1.0 - factor_damp) * float(msg_P[sid][li])
                    + factor_damp * P_new
                )
                msg_i[sid][li] = (
                    (1.0 - factor_damp) * float(msg_i[sid][li])
                    + factor_damp * i_new
                )
                message_updates += 1

        iters_executed += 1
        # A5-T67: residual-stop early-exit when flag on.
        if residual_stop_enabled:
            rd_now, rc_now = _current_residuals()
            if rd_now <= stop_tol and rc_now <= stop_tol:
                residual_stop_reason = "abs_tol"
                break
            if prev_r_data is not None and prev_r_cons is not None:
                d_rd = abs(rd_now - prev_r_data)
                d_rc = abs(rc_now - prev_r_cons)
                if d_rd <= stop_tol and d_rc <= stop_tol:
                    plateau += 1
                else:
                    plateau = 0
                if plateau >= stop_patience:
                    residual_stop_reason = "plateau"
                    break
            prev_r_data = rd_now
            prev_r_cons = rc_now

    if residual_stop_enabled and residual_stop_reason is None:
        residual_stop_reason = "max_iters"

    # Final beliefs → whitened means; unwhiten to physical pressures.
    P_b = P_u.copy()
    i_b = i_u.copy()
    for sid, n in zip(ids, block_sizes, strict=True):
        for li, g in enumerate(factor_faces[sid]):
            P_b[g] += float(msg_P[sid][li])
            i_b[g] += float(msg_i[sid][li])
    p_w = np.zeros(n_g, dtype=float)
    for g in range(n_g):
        if P_b[g] > 0.0 and np.isfinite(P_b[g]):
            p_w[g] = float(i_b[g]) / float(P_b[g])
        else:
            p_w[g] = float(hat_w[g])
    p_g = p_w * std
    p_loc = M @ p_g

    eps = 1e-12
    r_data = float(
        np.sum((p_loc - hat_local) ** 2) / (np.sum(hat_local**2) + eps)
    )
    flux2 = 0.0
    cons_num = 0.0
    for sid, A_S, n in zip(ids, blocks_A, block_sizes, strict=True):
        off = offsets[sid]
        p_S = p_loc[off : off + n]
        Ap = A_S @ p_S
        f2 = float(np.dot(Ap, Ap))
        fro2 = float(np.sum(A_S * A_S))
        flux2 += f2
        cons_num += f2 / (fro2 + float(cfg.as_eps))
    denom = float(np.sum(p_g * p_g)) + eps
    r_cons = cons_num / denom
    e_flux = flux2 / denom

    return LoopyBPScheduleResult(
        empirical_local=hat_local,
        empirical_global=hat_g,
        pressures_global=p_g,
        pressures_local=p_loc,
        lambda_f_global=lam_g,
        mu_S=mu_map,
        mu_S_sum=float(sum(mu_map.values())),
        r_data=r_data,
        r_cons=r_cons,
        epsilon_flux=e_flux,
        iters=int(iters_executed),
        block_sizes=tuple(block_sizes),
        simplex_ids=ids,
        n_faces=n_g,
        n_interior_faces=registry.n_interior,
        n_factors=len(ids),
        message_updates=message_updates,
        registry=registry,
        spectrum_ridge_applied=bool(spectrum_ridge_applied),
        policy_applied=bool(policy_applied),
        max_policy_damping=float(max_policy_damping),
        residual_stop_enabled=bool(residual_stop_enabled),
        residual_stop_reason=residual_stop_reason,
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


@dataclass(frozen=True)
class AcceptanceOpenDefaultProbe:
    """Snapshot of the acceptance-path open-default matrix (SI S10.4 A2 / A5-T54).

    Documents that missing / flag-off dual adjacency still reports A2
    connected (``True``), while the proposal-path producer can detect a
    real disconnect. Experiment only — does **not** flip gate / dual-flow
    defaults or ``@awaiting`` markers.
    """

    gate_apply_dual_adjacency_default: bool
    dual_enable_dual_adjacency_default: bool
    none_adjacency_reports_connected: bool
    resolve_flag_off_reports_connected: bool
    dry_run_flag_off_dual_connected: bool
    flag_on_detects_endpoint_disconnect: bool
    note: str = (
        "open-default: None adj / flags off ⇒ dual_connected True; "
        "flag-on dual adj detects path-endpoint disconnect. "
        "Do not flip defaults until A5-T42 acceptance plan steps land."
    )


def probe_acceptance_none_open_default() -> AcceptanceOpenDefaultProbe:
    """Probe / document the current None⇒True A2 open-default (A5-T54).

    Uses a three-edge path complex whose middle removal leaves dual endpoints
    disconnected when adjacency is enabled. With defaults (flags off / ``None``
    adj) every check still reports connected — the conservative open default
    that keeps Stage-1 edits unblocked without a Stage-2 producer.

    Returns a frozen snapshot for tests / REQUEST_TRACKER notes. Does **not**
    change ``GateConfig.apply_dual_adjacency`` or ``DualFlowConfig`` defaults.
    """

    gate_default = GateConfig()
    dual_default = DualFlowConfig()

    # Path 0—1—2 of 1-simplices; affecting endpoints only ⇒ induced disconnect
    # when adjacency is actually built.
    path_simplices = [(0, 1), (1, 2), (2, 3)]
    endpoint_affected: tuple[Hashable, ...] = (0, 2)

    none_open = affected_dual_subgraph_connected(None, endpoint_affected)
    resolve_open = resolve_dual_connected(
        path_simplices, endpoint_affected, config=dual_default
    )

    complex_ = Complex(
        simplices=[
            Simplex(vertex_ids=(0, 1)),
            Simplex(vertex_ids=(1, 2)),
            Simplex(vertex_ids=(2, 3)),
        ],
        vertex_positions=np.zeros((4, 2)),
        intrinsic_dim=1,
    )
    dry_off = dry_run_dual_from_edit(
        complex_, remove_simplex_indices=[1], config=dual_default
    )
    dry_on = dry_run_dual_from_edit(
        complex_,
        remove_simplex_indices=[1],
        config=DualFlowConfig(enable_dual_adjacency=True),
    )

    return AcceptanceOpenDefaultProbe(
        gate_apply_dual_adjacency_default=bool(
            gate_default.apply_dual_adjacency
        ),
        dual_enable_dual_adjacency_default=bool(
            dual_default.enable_dual_adjacency
        ),
        none_adjacency_reports_connected=bool(none_open),
        resolve_flag_off_reports_connected=bool(resolve_open),
        dry_run_flag_off_dual_connected=bool(dry_off.dual_connected),
        flag_on_detects_endpoint_disconnect=bool(
            dry_on.dual_adjacency is not None and not dry_on.dual_connected
        ),
    )


@dataclass(frozen=True)
class BpSpectrumDampingProbe:
    """SI S6.2 spectrum-damping probe snapshot (A5-T55; proposal-path).

    Documents that ``spectrum_cond_cap`` triggers μ-soft step-shrink
    (``spectrum_damped``) and loopy-BP factor ridge
    (``spectrum_ridge_applied``) on a poorly conditioned fixture. Does
    **not** flip defaults or ``@awaiting``.
    """

    probe_flag_default_off: bool
    mu_spectrum_damped: bool
    mu_hessian_cond: float
    loopy_spectrum_ridge_applied: bool
    loopy_message_updates: int
    spectrum_cond_cap_used: float
    note: str = (
        "spectrum damping probe: μ soft-solve step-shrink + loopy ridge "
        "when Hessian/factor cond exceeds spectrum_cond_cap; sketch only"
    )


def probe_bp_spectrum_damping(
    *,
    config: DualFlowConfig | None = None,
) -> BpSpectrumDampingProbe | None:
    """Probe BP spectrum damping / ridge on a poorly conditioned fixture.

    When ``enable_bp_spectrum_damping_probe`` is off, returns ``None``.
    When on, forces ``spectrum_cond_cap=0`` on a triangle μ soft-solve and
    a two-simplex loopy BP schedule so both spectrum paths fire, then
    returns a frozen snapshot (A5-T55). Proposal-path only — keep mass /
    density ``@awaiting`` xfail.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_bp_spectrum_damping_probe:
        return None

    default_off = not DualFlowConfig().enable_bp_spectrum_damping_probe
    # Force spectrum paths: any reasonable cond exceeds a tiny cap.
    # (loopy BP requires spectrum_cond_cap > 0; μ soft-solve accepts 0.)
    probe_cap = 1e-12
    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    A_S = build_divergence_stencil(P)
    hat = np.array([2.0, 0.1, 0.1], dtype=float)
    mu_cfg = DualFlowConfig(
        enable_mu_weighted_solve=True,
        bp_max_iters=4,
        as_step=0.5,
        spectrum_cond_cap=probe_cap,
    )
    mu_out = solve_mu_weighted_pressures(hat, A_S, config=mu_cfg)
    if mu_out is None:
        raise RuntimeError("μ soft-solve unexpectedly None under probe cfg")

    left = P
    right = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]], dtype=float)
    A0 = build_divergence_stencil(left)
    A1 = build_divergence_stencil(right)
    hats = {
        0: np.array([2.0, 0.1, 0.1], dtype=float),
        1: np.array([0.2, 1.5, 0.2], dtype=float),
    }
    simplices = {0: (0, 1, 2), 1: (1, 3, 2)}
    loopy_cfg = DualFlowConfig(
        enable_loopy_bp_schedule=True,
        bp_damping=0.5,
        bp_max_iters=2,
        spectrum_cond_cap=probe_cap,
    )
    loopy_out = solve_loopy_bp_schedule(
        hats, {0: A0, 1: A1}, simplices, config=loopy_cfg
    )
    if loopy_out is None:
        raise RuntimeError("loopy BP unexpectedly None under probe cfg")

    return BpSpectrumDampingProbe(
        probe_flag_default_off=bool(default_off),
        mu_spectrum_damped=bool(mu_out.spectrum_damped),
        mu_hessian_cond=float(mu_out.hessian_cond),
        loopy_spectrum_ridge_applied=bool(loopy_out.spectrum_ridge_applied),
        loopy_message_updates=int(loopy_out.message_updates),
        spectrum_cond_cap_used=float(probe_cap),
    )


@dataclass(frozen=True)
class OnlineOfflineScheduleResult:
    """Online tallies → offline μ soft-solve schedule sketch (SI S6.2; A5-T56).

    Online phase writes face tallies during sample routing; offline phase
    solves the conservative field after tallies settle. Proposal-path only.
    """

    n_samples: int
    n_online_simplices: int
    n_offline_solves: int
    offline_r_cons_mean: float
    offline_spectrum_damped_any: bool
    note: str = (
        "sketch only: online live-BMU tallies → offline μ soft-solve; "
        "not production online→offline BP schedule"
    )


def run_online_offline_schedule(
    samples: Sequence[np.ndarray],
    simplex_positions: Mapping[Hashable, np.ndarray],
    *,
    config: DualFlowConfig | None = None,
) -> OnlineOfflineScheduleResult | None:
    """Online face tallies then offline μ soft-solve (SI S6.2; A5-T56).

    When ``enable_online_offline_schedule`` is off, returns ``None``. When
    on:

    1. **Online** — route samples via the live-BMU tally harness
       (internally enables that path for this call).
    2. **Offline** — after tallies settle, soft-solve whitened ``λ_f`` +
       ``μ_S`` on each BMU winner's face pressures.

    Does **not** flip mass/density ``@awaiting``. Production loopy BP
    online→offline remains future work.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_online_offline_schedule:
        return None
    if not samples:
        raise ValueError("samples must be non-empty")
    if not simplex_positions:
        raise ValueError("simplex_positions must be non-empty")

    tally_cfg = DualFlowConfig(
        enable_live_bmu_tally=True,
        tally_scale=float(cfg.tally_scale),
    )
    live = route_live_bmu_face_tallies(
        samples, simplex_positions, config=tally_cfg
    )
    if live is None:
        raise RuntimeError("live BMU tallies unexpectedly None")

    solve_cfg = DualFlowConfig(
        enable_mu_weighted_solve=True,
        bp_damping=float(cfg.bp_damping),
        bp_max_iters=max(int(cfg.bp_max_iters), 4),
        as_step=float(cfg.as_step),
        mu_scale=float(cfg.mu_scale),
        whiten_floor=float(cfg.whiten_floor),
        spectrum_cond_cap=float(cfg.spectrum_cond_cap),
        as_eps=float(cfg.as_eps),
        enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
    )
    r_cons_vals: list[float] = []
    spectrum_any = False
    n_solves = 0
    for sid, tally in live.tallies_by_simplex.items():
        if sid not in simplex_positions:
            continue
        A_S = build_divergence_stencil(
            np.asarray(simplex_positions[sid], dtype=float)
        )
        out = solve_mu_weighted_pressures(
            tally.tallies, A_S, config=solve_cfg
        )
        if out is None:
            continue
        n_solves += 1
        r_cons_vals.append(float(out.r_cons))
        spectrum_any = spectrum_any or bool(out.spectrum_damped)

    r_mean = float(np.mean(r_cons_vals)) if r_cons_vals else 0.0
    return OnlineOfflineScheduleResult(
        n_samples=len(samples),
        n_online_simplices=len(live.tallies_by_simplex),
        n_offline_solves=n_solves,
        offline_r_cons_mean=r_mean,
        offline_spectrum_damped_any=bool(spectrum_any),
    )


@dataclass(frozen=True)
class FailClosedDualPlanProbe:
    """Stub plan to replace None⇒True open-default (SI S10.4 A2; A5-T57).

    Documents the acceptance-path steps toward fail-closed dual adjacency
    without flipping ``GateConfig`` / ``DualFlowConfig`` defaults.
    """

    open_default_still_active: bool
    gate_apply_dual_adjacency_default: bool
    dual_enable_dual_adjacency_default: bool
    plan_steps: tuple[str, ...]
    note: str = (
        "fail-closed plan stub: keep None⇒True until dual producer + "
        "real S6.2 BP green; then gate apply_dual_adjacency=True"
    )


def probe_fail_closed_dual_adjacency_plan() -> FailClosedDualPlanProbe:
    """Document the fail-closed path replacing None⇒True (A5-T57).

    Always returns a frozen plan snapshot (stub / documentation). Does
    **not** flip defaults or ``@awaiting`` markers — acceptance-path
    promotion waits on A5-T42 steps and green dual-flow evidence.
    """

    gate_default = GateConfig()
    dual_default = DualFlowConfig()
    open_default = (
        (not gate_default.apply_dual_adjacency)
        and (not dual_default.enable_dual_adjacency)
        and affected_dual_subgraph_connected(None, (0, 1))
    )
    steps = (
        "Keep None adj / flags off ⇒ dual_connected True (current open default)",
        "Land default-on dual adjacency producer so None is unreachable on acceptance path",
        "Land real S6.2 BP (not identity sketch) with spectrum-safe convergence",
        "Green evidence for mass/density @awaiting before flipping markers",
        "Then set GateConfig.apply_dual_adjacency=True (fail-closed on disconnect)",
    )
    return FailClosedDualPlanProbe(
        open_default_still_active=bool(open_default),
        gate_apply_dual_adjacency_default=bool(
            gate_default.apply_dual_adjacency
        ),
        dual_enable_dual_adjacency_default=bool(
            dual_default.enable_dual_adjacency
        ),
        plan_steps=steps,
    )


@dataclass(frozen=True)
class BpDampingPolicyResult:
    """Production BP damping-policy sketch (SI S6.2; A5-T58; proposal-path).

    Maps a reported Hessian / factor condition number to recommended
    ``bp_damping`` and whether factor ridge should apply when
    ``cond > spectrum_cond_cap``. Sketch only — not certified convergence.
    """

    policy_flag_default_off: bool
    hessian_cond: float
    spectrum_cond_cap: float
    recommended_damping: float
    apply_ridge: bool
    overshoot_decades: float
    note: str = (
        "production damping policy sketch: cond>cap ⇒ raise damping toward "
        "1 and recommend factor ridge; flag off by default"
    )


def propose_bp_damping_policy(
    hessian_cond: float,
    *,
    config: DualFlowConfig | None = None,
) -> BpDampingPolicyResult | None:
    """Map spectrum condition number → damping + ridge decision (A5-T58).

    When ``enable_bp_damping_policy`` is off, returns ``None``. When on:

    * ``cond <= spectrum_cond_cap`` → keep ``bp_damping``, ``apply_ridge=False``.
    * ``cond > spectrum_cond_cap`` → interpolate damping toward ``1.0`` by
      ``min(1, log10(cond/cap) / 6)`` of the remaining gap, and
      ``apply_ridge=True`` (SI S6.2 "damping when spectra are poorly
      conditioned").

    Proposal-path only — does **not** flip mass/density ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_bp_damping_policy:
        return None

    default_off = not DualFlowConfig().enable_bp_damping_policy
    cond = float(hessian_cond)
    if not np.isfinite(cond) or cond < 0.0:
        raise ValueError("hessian_cond must be a finite non-negative float")
    cap = float(cfg.spectrum_cond_cap)
    if cap <= 0.0:
        raise ValueError("spectrum_cond_cap must be > 0 for damping policy")

    base = float(np.clip(cfg.bp_damping, 0.0, 1.0))
    if cond <= cap:
        return BpDampingPolicyResult(
            policy_flag_default_off=bool(default_off),
            hessian_cond=cond,
            spectrum_cond_cap=cap,
            recommended_damping=base,
            apply_ridge=False,
            overshoot_decades=0.0,
        )

    decades = float(np.log10(cond / cap))
    frac = float(min(1.0, max(0.0, decades / 6.0)))
    damping = float(base + (1.0 - base) * frac)
    return BpDampingPolicyResult(
        policy_flag_default_off=bool(default_off),
        hessian_cond=cond,
        spectrum_cond_cap=cap,
        recommended_damping=damping,
        apply_ridge=True,
        overshoot_decades=decades,
    )


@dataclass(frozen=True)
class OnlineOfflineLoopyComposeResult:
    """Online tallies → offline loopy BP compose (SI S6.2; A5-T59).

    Online phase writes face tallies; offline phase runs the loopy
    Gaussian BP message schedule on the shared face/factor graph.
    Proposal-path compose wire — not production online→offline BP.
    """

    n_samples: int
    n_online_simplices: int
    loopy_message_updates: int
    loopy_spectrum_ridge_applied: bool
    loopy_r_cons: float
    loopy_policy_applied: bool = False
    loopy_max_policy_damping: float = 0.0
    loopy_residual_stop_enabled: bool = False
    loopy_residual_stop_reason: str | None = None
    loopy_iters: int = 0
    note: str = (
        "sketch only: online live-BMU tallies → offline loopy BP schedule; "
        "not certified production compose"
    )


def run_online_offline_loopy_compose(
    samples: Sequence[np.ndarray],
    simplex_positions: Mapping[Hashable, np.ndarray],
    simplices: Sequence[Sequence[Hashable]]
    | Mapping[Hashable, Sequence[Hashable]],
    *,
    config: DualFlowConfig | None = None,
) -> OnlineOfflineLoopyComposeResult | None:
    """Online face tallies then offline loopy BP schedule (SI S6.2; A5-T59).

    When ``enable_online_offline_loopy_compose`` is off, returns ``None``.
    When on:

    1. **Online** — route samples via the live-BMU tally harness.
    2. **Offline** — build divergence stencils for winners and run
       :func:`solve_loopy_bp_schedule` on the shared face/factor graph
       (``simplices`` supplies the global face registry).

    When ``enable_bp_policy_in_loopy`` is also on, the offline loopy
    schedule consults :func:`propose_bp_damping_policy` per factor
    (A5-T63 forward). When ``enable_loopy_bp_residual_stop`` is on,
    residual-stop early-exit is forwarded into the offline schedule
    (A5-T69). Does **not** flip mass/density ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_online_offline_loopy_compose:
        return None
    if not samples:
        raise ValueError("samples must be non-empty")
    if not simplex_positions:
        raise ValueError("simplex_positions must be non-empty")

    tally_cfg = DualFlowConfig(
        enable_live_bmu_tally=True,
        tally_scale=float(cfg.tally_scale),
    )
    live = route_live_bmu_face_tallies(
        samples, simplex_positions, config=tally_cfg
    )
    if live is None:
        raise RuntimeError("live BMU tallies unexpectedly None")

    hats: dict[Hashable, np.ndarray] = {}
    stencils: dict[Hashable, np.ndarray] = {}
    for sid, tally in live.tallies_by_simplex.items():
        if sid not in simplex_positions:
            continue
        hats[sid] = np.asarray(tally.tallies, dtype=float)
        stencils[sid] = build_divergence_stencil(
            np.asarray(simplex_positions[sid], dtype=float)
        )
    if not hats:
        raise RuntimeError("online phase produced no simplex tallies")

    # Restrict face registry to winner simplices when a mapping is provided.
    if isinstance(simplices, Mapping):
        sub_simplices: Mapping[Hashable, Sequence[Hashable]] = {
            sid: simplices[sid] for sid in hats if sid in simplices
        }
        if len(sub_simplices) != len(hats):
            missing = set(hats) - set(sub_simplices)
            raise ValueError(
                f"simplices mapping missing winners {sorted(missing)!r}"
            )
        face_simplices: Sequence[Sequence[Hashable]] | Mapping[
            Hashable, Sequence[Hashable]
        ] = sub_simplices
    else:
        face_simplices = simplices

    loopy_cfg = DualFlowConfig(
        enable_loopy_bp_schedule=True,
        enable_bp_policy_in_loopy=bool(cfg.enable_bp_policy_in_loopy),
        enable_loopy_bp_residual_stop=bool(cfg.enable_loopy_bp_residual_stop),
        bp_damping=float(cfg.bp_damping),
        bp_max_iters=max(int(cfg.bp_max_iters), 2),
        bp_residual_stop_tol=float(cfg.bp_residual_stop_tol),
        bp_residual_stop_patience=int(cfg.bp_residual_stop_patience),
        mu_scale=float(cfg.mu_scale),
        as_eps=float(cfg.as_eps),
        whiten_floor=float(cfg.whiten_floor),
        spectrum_cond_cap=float(cfg.spectrum_cond_cap),
        enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
    )
    loopy_out = solve_loopy_bp_schedule(
        hats, stencils, face_simplices, config=loopy_cfg
    )
    if loopy_out is None:
        raise RuntimeError("loopy BP unexpectedly None under compose cfg")

    return OnlineOfflineLoopyComposeResult(
        n_samples=len(samples),
        n_online_simplices=len(live.tallies_by_simplex),
        loopy_message_updates=int(loopy_out.message_updates),
        loopy_spectrum_ridge_applied=bool(loopy_out.spectrum_ridge_applied),
        loopy_r_cons=float(loopy_out.r_cons),
        loopy_policy_applied=bool(loopy_out.policy_applied),
        loopy_max_policy_damping=float(loopy_out.max_policy_damping),
        loopy_residual_stop_enabled=bool(loopy_out.residual_stop_enabled),
        loopy_residual_stop_reason=loopy_out.residual_stop_reason,
        loopy_iters=int(loopy_out.iters),
    )


@dataclass(frozen=True)
class LoopyBPConvergenceProbe:
    """Residual trajectory harness for loopy BP (SI S6.2; A5-T62).

    Re-runs :func:`solve_loopy_bp_schedule` at increasing iteration
    counts and records ``r_data`` / ``r_cons``. Proposal-path only —
    **not** a production convergence certificate.
    """

    probe_flag_default_off: bool
    iters: tuple[int, ...]
    r_data: tuple[float, ...]
    r_cons: tuple[float, ...]
    policy_in_loopy_used: bool
    final_spectrum_ridge_applied: bool
    note: str = (
        "sketch only: residual trajectory over increasing loopy BP iters; "
        "not a certified convergence proof; do not flip @awaiting"
    )


def probe_loopy_bp_convergence(
    empirical_by_simplex: Mapping[Hashable, np.ndarray],
    stencils_by_simplex: Mapping[Hashable, np.ndarray],
    simplices: Sequence[Sequence[Hashable]]
    | Mapping[Hashable, Sequence[Hashable]],
    *,
    max_iters: int = 4,
    config: DualFlowConfig | None = None,
) -> LoopyBPConvergenceProbe | None:
    """Record loopy BP residual trajectory vs iteration count (A5-T62).

    When ``enable_loopy_bp_convergence_probe`` is off, returns ``None``.
    When on, runs the loopy schedule at ``bp_max_iters = 1..max_iters``
    (optionally with ``enable_bp_policy_in_loopy`` forwarded) and returns
    the residual sequences. Does **not** flip mass/density ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_loopy_bp_convergence_probe:
        return None
    n_max = int(max_iters)
    if n_max < 1:
        raise ValueError("max_iters must be >= 1")

    iters_out: list[int] = []
    r_data_out: list[float] = []
    r_cons_out: list[float] = []
    policy_used = False
    ridge_final = False
    for k in range(1, n_max + 1):
        run_cfg = DualFlowConfig(
            enable_loopy_bp_schedule=True,
            enable_bp_policy_in_loopy=bool(cfg.enable_bp_policy_in_loopy),
            bp_damping=float(cfg.bp_damping),
            bp_max_iters=k,
            mu_scale=float(cfg.mu_scale),
            as_eps=float(cfg.as_eps),
            whiten_floor=float(cfg.whiten_floor),
            spectrum_cond_cap=float(cfg.spectrum_cond_cap),
            enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
        )
        out = solve_loopy_bp_schedule(
            empirical_by_simplex,
            stencils_by_simplex,
            simplices,
            config=run_cfg,
        )
        if out is None:
            raise RuntimeError("loopy BP unexpectedly None under probe cfg")
        iters_out.append(k)
        r_data_out.append(float(out.r_data))
        r_cons_out.append(float(out.r_cons))
        policy_used = policy_used or bool(out.policy_applied)
        ridge_final = bool(out.spectrum_ridge_applied)

    return LoopyBPConvergenceProbe(
        probe_flag_default_off=not DualFlowConfig().enable_loopy_bp_convergence_probe,
        iters=tuple(iters_out),
        r_data=tuple(r_data_out),
        r_cons=tuple(r_cons_out),
        policy_in_loopy_used=bool(policy_used),
        final_spectrum_ridge_applied=bool(ridge_final),
    )


@dataclass(frozen=True)
class FailClosedGateSwitchProbe:
    """Documents GateConfig.fail_closed_dual_adjacency (A5-T60; default off).

    Acceptance-path switch stays off so None adj remains open-default.
    Flag-on + apply_dual_adjacency rejects missing dual producer.
    """

    switch_default_off: bool
    apply_dual_adjacency_default: bool
    open_default_none_still_connected: bool
    flag_on_none_rejects: bool
    note: str = (
        "GateConfig.fail_closed_dual_adjacency default False; "
        "do not flip until dual producer + real S6.2 BP green"
    )


def probe_gate_fail_closed_switch() -> FailClosedGateSwitchProbe:
    """Probe the GateConfig fail-closed dual-adjacency switch (A5-T60).

    Always returns a frozen snapshot. Does **not** flip ``GateConfig``
    defaults — documents that the switch is present and default-off, and
    that enabling it (with ``apply_dual_adjacency``) makes a missing
    adjacency fail closed on the evidence path.
    """

    gate_default = GateConfig()
    # Mirror score_edit dual-resolution for None adj under the stub switch.
    cfg_on = GateConfig(
        apply_dual_adjacency=True,
        fail_closed_dual_adjacency=True,
    )
    if cfg_on.apply_dual_adjacency and cfg_on.fail_closed_dual_adjacency:
        dual_connected_on_none = False
    else:
        dual_connected_on_none = True

    return FailClosedGateSwitchProbe(
        switch_default_off=not bool(gate_default.fail_closed_dual_adjacency),
        apply_dual_adjacency_default=bool(gate_default.apply_dual_adjacency),
        open_default_none_still_connected=bool(
            affected_dual_subgraph_connected(None, (0, 1))
        ),
        flag_on_none_rejects=not dual_connected_on_none,
    )


@dataclass(frozen=True)
class LoopyBPResidualStopPolicy:
    """Certified residual-stop *sketch* for loopy BP (SI S6.2; A5-T64).

    Walks increasing iteration counts and stops on residual plateau /
    absolute tolerance. Proposal-path only — ``sketch_certificate_ok``
    is a *harness claim*, **not** a production spectrum-safe certificate.
    """

    policy_flag_default_off: bool
    stopped_at_iters: int
    max_iters_scanned: int
    r_data_traj: tuple[float, ...]
    r_cons_traj: tuple[float, ...]
    stop_reason: str
    tol: float
    patience: int
    sketch_certificate_ok: bool
    policy_in_loopy_used: bool
    note: str = (
        "sketch only: residual-plateau / tol stop rule; "
        "NOT a certified production convergence proof; do not flip @awaiting"
    )


def propose_loopy_bp_residual_stop(
    empirical_by_simplex: Mapping[Hashable, np.ndarray],
    stencils_by_simplex: Mapping[Hashable, np.ndarray],
    simplices: Sequence[Sequence[Hashable]]
    | Mapping[Hashable, Sequence[Hashable]],
    *,
    max_iters: int = 6,
    config: DualFlowConfig | None = None,
) -> LoopyBPResidualStopPolicy | None:
    """Sketch a residual-plateau stop rule over loopy BP iters (A5-T64).

    When ``enable_loopy_bp_residual_stop`` is off, returns ``None``. When
    on, re-runs :func:`solve_loopy_bp_schedule` at ``bp_max_iters =
    1..max_iters`` and stops early when both ``|Δr_data|`` and
    ``|Δr_cons|`` stay below ``bp_residual_stop_tol`` for
    ``bp_residual_stop_patience`` consecutive steps, or when both
    residuals themselves fall below ``tol``.

    ``sketch_certificate_ok`` is true only when the chosen stop has
    finite non-negative residuals and a declared stop reason other than
    exhausting ``max_iters`` without plateau — still **not** a production
    certificate. Does **not** flip mass/density ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_loopy_bp_residual_stop:
        return None
    n_max = int(max_iters)
    if n_max < 1:
        raise ValueError("max_iters must be >= 1")
    tol = float(cfg.bp_residual_stop_tol)
    if tol < 0.0:
        raise ValueError("bp_residual_stop_tol must be >= 0")
    patience = int(cfg.bp_residual_stop_patience)
    if patience < 1:
        raise ValueError("bp_residual_stop_patience must be >= 1")

    r_data_out: list[float] = []
    r_cons_out: list[float] = []
    policy_used = False
    plateau = 0
    stop_reason = "max_iters"
    stopped_at = n_max
    for k in range(1, n_max + 1):
        run_cfg = DualFlowConfig(
            enable_loopy_bp_schedule=True,
            enable_bp_policy_in_loopy=bool(cfg.enable_bp_policy_in_loopy),
            bp_damping=float(cfg.bp_damping),
            bp_max_iters=k,
            mu_scale=float(cfg.mu_scale),
            as_eps=float(cfg.as_eps),
            whiten_floor=float(cfg.whiten_floor),
            spectrum_cond_cap=float(cfg.spectrum_cond_cap),
            enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
        )
        out = solve_loopy_bp_schedule(
            empirical_by_simplex,
            stencils_by_simplex,
            simplices,
            config=run_cfg,
        )
        if out is None:
            raise RuntimeError("loopy BP unexpectedly None under residual-stop cfg")
        rd = float(out.r_data)
        rc = float(out.r_cons)
        r_data_out.append(rd)
        r_cons_out.append(rc)
        policy_used = policy_used or bool(out.policy_applied)

        if rd <= tol and rc <= tol:
            stop_reason = "abs_tol"
            stopped_at = k
            break
        if k >= 2:
            d_rd = abs(r_data_out[-1] - r_data_out[-2])
            d_rc = abs(r_cons_out[-1] - r_cons_out[-2])
            if d_rd <= tol and d_rc <= tol:
                plateau += 1
            else:
                plateau = 0
            if plateau >= patience:
                stop_reason = "plateau"
                stopped_at = k
                break

    # Truncate trajectories to the stop horizon.
    r_data_t = tuple(r_data_out[:stopped_at])
    r_cons_t = tuple(r_cons_out[:stopped_at])
    finite_ok = all(np.isfinite(r) and r >= 0.0 for r in r_data_t + r_cons_t)
    sketch_ok = bool(finite_ok and stop_reason in ("abs_tol", "plateau"))

    return LoopyBPResidualStopPolicy(
        policy_flag_default_off=not DualFlowConfig().enable_loopy_bp_residual_stop,
        stopped_at_iters=int(stopped_at),
        max_iters_scanned=n_max,
        r_data_traj=r_data_t,
        r_cons_traj=r_cons_t,
        stop_reason=stop_reason,
        tol=tol,
        patience=patience,
        sketch_certificate_ok=sketch_ok,
        policy_in_loopy_used=bool(policy_used),
    )


@dataclass(frozen=True)
class LoopyBPSpectrumSafeCertProbe:
    """Spectrum-safe residual-stop certificate *harness* (SI S6.2; A5-T68).

    Combines residual-stop early-exit with a no-ridge spectrum check.
    ``spectrum_safe_sketch_ok`` is a *harness claim* only — **not** a
    production certificate. Do **not** flip ``@awaiting``.
    """

    probe_flag_default_off: bool
    residual_stop_reason: str | None
    iters_executed: int
    max_iters: int
    r_data: float
    r_cons: float
    spectrum_ridge_applied: bool
    spectrum_safe_sketch_ok: bool
    note: str = (
        "harness only: residual-stop early-exit + no-ridge spectrum check; "
        "NOT a certified production convergence proof; do not flip @awaiting"
    )


def probe_loopy_bp_spectrum_safe_cert(
    empirical_by_simplex: Mapping[Hashable, np.ndarray],
    stencils_by_simplex: Mapping[Hashable, np.ndarray],
    simplices: Sequence[Sequence[Hashable]]
    | Mapping[Hashable, Sequence[Hashable]],
    *,
    config: DualFlowConfig | None = None,
) -> LoopyBPSpectrumSafeCertProbe | None:
    """Harness: residual-stop early-exit + no-ridge spectrum claim (A5-T68).

    When ``enable_loopy_bp_spectrum_safe_cert`` is off, returns ``None``.
    When on, runs :func:`solve_loopy_bp_schedule` with residual-stop
    early-exit enabled and reports ``spectrum_safe_sketch_ok`` iff:

    * ``residual_stop_reason`` is ``abs_tol`` or ``plateau``,
    * ``spectrum_ridge_applied`` is ``False``,
    * residuals are finite and non-negative.

    Still **not** a production certificate. Does **not** flip
    mass/density ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_loopy_bp_spectrum_safe_cert:
        return None
    max_iters = max(int(cfg.bp_max_iters), 2)
    run_cfg = DualFlowConfig(
        enable_loopy_bp_schedule=True,
        enable_loopy_bp_residual_stop=True,
        enable_bp_policy_in_loopy=bool(cfg.enable_bp_policy_in_loopy),
        bp_damping=float(cfg.bp_damping),
        bp_max_iters=max_iters,
        bp_residual_stop_tol=float(cfg.bp_residual_stop_tol),
        bp_residual_stop_patience=int(cfg.bp_residual_stop_patience),
        mu_scale=float(cfg.mu_scale),
        as_eps=float(cfg.as_eps),
        whiten_floor=float(cfg.whiten_floor),
        spectrum_cond_cap=float(cfg.spectrum_cond_cap),
        enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
    )
    out = solve_loopy_bp_schedule(
        empirical_by_simplex,
        stencils_by_simplex,
        simplices,
        config=run_cfg,
    )
    if out is None:
        raise RuntimeError(
            "loopy BP unexpectedly None under spectrum-safe cert cfg"
        )
    rd = float(out.r_data)
    rc = float(out.r_cons)
    reason = out.residual_stop_reason
    finite_ok = bool(
        np.isfinite(rd) and np.isfinite(rc) and rd >= 0.0 and rc >= 0.0
    )
    sketch_ok = bool(
        finite_ok
        and reason in ("abs_tol", "plateau")
        and not bool(out.spectrum_ridge_applied)
    )
    return LoopyBPSpectrumSafeCertProbe(
        probe_flag_default_off=not DualFlowConfig().enable_loopy_bp_spectrum_safe_cert,
        residual_stop_reason=reason,
        iters_executed=int(out.iters),
        max_iters=max_iters,
        r_data=rd,
        r_cons=rc,
        spectrum_ridge_applied=bool(out.spectrum_ridge_applied),
        spectrum_safe_sketch_ok=sketch_ok,
    )


@dataclass(frozen=True)
class MassLoopyComposeProbe:
    """Mass-normalization × online→offline loopy compose (SI S6.2; A5-T66).

    Proposal-path combined probe — does **not** flip mass/density
    ``@awaiting`` markers.
    """

    probe_flag_default_off: bool
    n_samples: int
    n_online_simplices: int
    epsilon_mass: float
    mass_total_before: float
    loopy_message_updates: int
    loopy_r_cons: float
    loopy_spectrum_ridge_applied: bool
    note: str = (
        "sketch only: mass-norm + online→offline loopy compose probe; "
        "do not flip mass/density @awaiting"
    )


def probe_mass_loopy_compose(
    samples: Sequence[np.ndarray],
    simplex_positions: Mapping[Hashable, np.ndarray],
    simplices: Sequence[Sequence[Hashable]]
    | Mapping[Hashable, Sequence[Hashable]],
    *,
    masses: Mapping[Hashable, float] | None = None,
    config: DualFlowConfig | None = None,
) -> MassLoopyComposeProbe | None:
    """Run mass normalization together with loopy compose (A5-T66).

    When ``enable_mass_loopy_compose_probe`` is off, returns ``None``.
    When on:

    1. Online→offline loopy compose (internal compose flag on).
    2. Mass-normalize ``masses`` (default: unit mass per online winner
       simplex from the compose path's sample routing).

    Does **not** flip mass/density ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_mass_loopy_compose_probe:
        return None
    if not samples:
        raise ValueError("samples must be non-empty")
    if not simplex_positions:
        raise ValueError("simplex_positions must be non-empty")

    compose_cfg = DualFlowConfig(
        enable_online_offline_loopy_compose=True,
        enable_bp_policy_in_loopy=bool(cfg.enable_bp_policy_in_loopy),
        bp_damping=float(cfg.bp_damping),
        bp_max_iters=max(int(cfg.bp_max_iters), 2),
        tally_scale=float(cfg.tally_scale),
        mu_scale=float(cfg.mu_scale),
        as_eps=float(cfg.as_eps),
        whiten_floor=float(cfg.whiten_floor),
        spectrum_cond_cap=float(cfg.spectrum_cond_cap),
        enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
    )
    compose = run_online_offline_loopy_compose(
        samples, simplex_positions, simplices, config=compose_cfg
    )
    if compose is None:
        raise RuntimeError("loopy compose unexpectedly None under mass×loopy cfg")

    if masses is None:
        # Unit prior per online-routed simplex (stable sorted ids).
        # Re-run live BMU to discover winners without depending on compose guts.
        tally_cfg = DualFlowConfig(
            enable_live_bmu_tally=True,
            tally_scale=float(cfg.tally_scale),
        )
        live = route_live_bmu_face_tallies(
            samples, simplex_positions, config=tally_cfg
        )
        if live is None:
            raise RuntimeError("live BMU tallies unexpectedly None")
        mass_map: dict[Hashable, float] = {
            sid: 1.0 for sid in live.tallies_by_simplex
        }
    else:
        mass_map = {k: float(v) for k, v in masses.items()}
    if not mass_map:
        raise ValueError("masses must be non-empty for mass×loopy probe")

    mass_cfg = DualFlowConfig(enable_mass_normalization=True)
    mass_out = normalize_simplex_masses(mass_map, config=mass_cfg)
    if mass_out is None:
        raise RuntimeError("mass normalization unexpectedly None under probe cfg")

    return MassLoopyComposeProbe(
        probe_flag_default_off=not DualFlowConfig().enable_mass_loopy_compose_probe,
        n_samples=int(compose.n_samples),
        n_online_simplices=int(compose.n_online_simplices),
        epsilon_mass=float(mass_out.epsilon_mass),
        mass_total_before=float(mass_out.total_before),
        loopy_message_updates=int(compose.loopy_message_updates),
        loopy_r_cons=float(compose.loopy_r_cons),
        loopy_spectrum_ridge_applied=bool(compose.loopy_spectrum_ridge_applied),
    )


@dataclass(frozen=True)
class PolicyResidualComposeProbe:
    """Policy × residual-stop compose multi-iter residual pin (SI S6.2; A5-T69).

    Pins ``r_data`` / ``r_cons`` over increasing loopy iters under
    ``enable_bp_policy_in_loopy``, then runs online→offline loopy compose
    with residual-stop early-exit. Proposal-path only — **not** a
    production certificate. Do **not** flip mass/density ``@awaiting``.
    """

    probe_flag_default_off: bool
    pin_iters: tuple[int, ...]
    pin_r_data: tuple[float, ...]
    pin_r_cons: tuple[float, ...]
    pin_policy_applied: bool
    compose_n_samples: int
    compose_n_online_simplices: int
    compose_loopy_message_updates: int
    compose_loopy_r_cons: float
    compose_policy_applied: bool
    compose_residual_stop_enabled: bool
    compose_residual_stop_reason: str | None
    compose_loopy_iters: int
    compose_max_iters: int
    note: str = (
        "sketch only: policy-in-loopy multi-iter residual pin + "
        "compose residual-stop early-exit; not a production certificate; "
        "do not flip mass/density @awaiting"
    )


def probe_policy_residual_compose(
    samples: Sequence[np.ndarray],
    simplex_positions: Mapping[Hashable, np.ndarray],
    simplices: Sequence[Sequence[Hashable]]
    | Mapping[Hashable, Sequence[Hashable]],
    *,
    max_pin_iters: int = 4,
    config: DualFlowConfig | None = None,
) -> PolicyResidualComposeProbe | None:
    """Pin multi-iter residuals under policy, then compose+stop (A5-T69).

    When ``enable_policy_residual_compose_probe`` is off, returns ``None``.
    When on:

    1. **Multi-iter residual pin** — re-run
       :func:`solve_loopy_bp_schedule` at ``bp_max_iters = 1..max_pin_iters``
       with ``enable_bp_policy_in_loopy`` on (no in-solver residual-stop,
       so the pin covers the full horizon).
    2. **Compose** — online→offline loopy compose with policy-in-loopy
       and residual-stop early-exit both on.

    Does **not** flip mass/density ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_policy_residual_compose_probe:
        return None
    if not samples:
        raise ValueError("samples must be non-empty")
    if not simplex_positions:
        raise ValueError("simplex_positions must be non-empty")
    n_pin = int(max_pin_iters)
    if n_pin < 1:
        raise ValueError("max_pin_iters must be >= 1")

    # Online tallies → hats/stencils for the residual pin (same BMU path).
    tally_cfg = DualFlowConfig(
        enable_live_bmu_tally=True,
        tally_scale=float(cfg.tally_scale),
    )
    live = route_live_bmu_face_tallies(
        samples, simplex_positions, config=tally_cfg
    )
    if live is None:
        raise RuntimeError("live BMU tallies unexpectedly None")

    hats: dict[Hashable, np.ndarray] = {}
    stencils: dict[Hashable, np.ndarray] = {}
    for sid, tally in live.tallies_by_simplex.items():
        if sid not in simplex_positions:
            continue
        hats[sid] = np.asarray(tally.tallies, dtype=float)
        stencils[sid] = build_divergence_stencil(
            np.asarray(simplex_positions[sid], dtype=float)
        )
    if not hats:
        raise RuntimeError("online phase produced no simplex tallies")

    if isinstance(simplices, Mapping):
        face_simplices: Sequence[Sequence[Hashable]] | Mapping[
            Hashable, Sequence[Hashable]
        ] = {sid: simplices[sid] for sid in hats if sid in simplices}
        if len(face_simplices) != len(hats):
            missing = set(hats) - set(face_simplices)
            raise ValueError(
                f"simplices mapping missing winners {sorted(missing)!r}"
            )
    else:
        face_simplices = simplices

    pin_iters: list[int] = []
    pin_r_data: list[float] = []
    pin_r_cons: list[float] = []
    pin_policy = False
    for k in range(1, n_pin + 1):
        pin_cfg = DualFlowConfig(
            enable_loopy_bp_schedule=True,
            enable_bp_policy_in_loopy=True,
            enable_loopy_bp_residual_stop=False,
            bp_damping=float(cfg.bp_damping),
            bp_max_iters=k,
            mu_scale=float(cfg.mu_scale),
            as_eps=float(cfg.as_eps),
            whiten_floor=float(cfg.whiten_floor),
            spectrum_cond_cap=float(cfg.spectrum_cond_cap),
            enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
        )
        out = solve_loopy_bp_schedule(
            hats, stencils, face_simplices, config=pin_cfg
        )
        if out is None:
            raise RuntimeError("loopy BP unexpectedly None under pin cfg")
        pin_iters.append(k)
        pin_r_data.append(float(out.r_data))
        pin_r_cons.append(float(out.r_cons))
        pin_policy = pin_policy or bool(out.policy_applied)

    compose_max = max(int(cfg.bp_max_iters), n_pin, 2)
    compose_cfg = DualFlowConfig(
        enable_online_offline_loopy_compose=True,
        enable_bp_policy_in_loopy=True,
        enable_loopy_bp_residual_stop=True,
        bp_damping=float(cfg.bp_damping),
        bp_max_iters=compose_max,
        bp_residual_stop_tol=float(cfg.bp_residual_stop_tol),
        bp_residual_stop_patience=int(cfg.bp_residual_stop_patience),
        tally_scale=float(cfg.tally_scale),
        mu_scale=float(cfg.mu_scale),
        as_eps=float(cfg.as_eps),
        whiten_floor=float(cfg.whiten_floor),
        spectrum_cond_cap=float(cfg.spectrum_cond_cap),
        enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
    )
    compose = run_online_offline_loopy_compose(
        samples, simplex_positions, simplices, config=compose_cfg
    )
    if compose is None:
        raise RuntimeError(
            "loopy compose unexpectedly None under policy×residual cfg"
        )

    return PolicyResidualComposeProbe(
        probe_flag_default_off=not DualFlowConfig().enable_policy_residual_compose_probe,
        pin_iters=tuple(pin_iters),
        pin_r_data=tuple(pin_r_data),
        pin_r_cons=tuple(pin_r_cons),
        pin_policy_applied=bool(pin_policy),
        compose_n_samples=int(compose.n_samples),
        compose_n_online_simplices=int(compose.n_online_simplices),
        compose_loopy_message_updates=int(compose.loopy_message_updates),
        compose_loopy_r_cons=float(compose.loopy_r_cons),
        compose_policy_applied=bool(compose.loopy_policy_applied),
        compose_residual_stop_enabled=bool(compose.loopy_residual_stop_enabled),
        compose_residual_stop_reason=compose.loopy_residual_stop_reason,
        compose_loopy_iters=int(compose.loopy_iters),
        compose_max_iters=compose_max,
    )


@dataclass(frozen=True)
class SpectrumSafePolicyPinCase:
    """One ``spectrum_cond_cap`` cell of the spectrum-safe×policy pin (A5-T70)."""

    spectrum_cond_cap: float
    residual_stop_reason: str | None
    iters_executed: int
    max_iters: int
    r_data: float
    r_cons: float
    spectrum_ridge_applied: bool
    policy_applied: bool
    max_policy_damping: float
    spectrum_safe_sketch_ok: bool


@dataclass(frozen=True)
class SpectrumSafePolicyPinProbe:
    """Spectrum-safe × policy_in_loopy multi-cond pin (SI S6.2; A5-T70).

    Pins residual-stop + no-ridge harness outcomes across a
    ``spectrum_cond_cap`` grid with ``enable_bp_policy_in_loopy`` on.
    ``spectrum_safe_sketch_ok`` remains a *harness claim* only — **not** a
    production certificate. Do **not** flip mass/density ``@awaiting``.
    """

    probe_flag_default_off: bool
    caps: tuple[float, ...]
    cases: tuple[SpectrumSafePolicyPinCase, ...]
    note: str = (
        "harness only: spectrum-safe residual-stop × policy-in-loopy "
        "multi-cond pin; NOT a certified production convergence proof; "
        "do not flip mass/density @awaiting"
    )


def probe_spectrum_safe_policy_pin(
    empirical_by_simplex: Mapping[Hashable, np.ndarray],
    stencils_by_simplex: Mapping[Hashable, np.ndarray],
    simplices: Sequence[Sequence[Hashable]]
    | Mapping[Hashable, Sequence[Hashable]],
    *,
    spectrum_cond_caps: Sequence[float] | None = None,
    config: DualFlowConfig | None = None,
) -> SpectrumSafePolicyPinProbe | None:
    """Pin spectrum-safe harness across ``spectrum_cond_cap`` values (A5-T70).

    When ``enable_spectrum_safe_policy_pin_probe`` is off, returns ``None``.
    When on, for each cap in ``spectrum_cond_caps`` (default
    ``(1e-12, 1.0, 1e6, 1e12)``) runs :func:`solve_loopy_bp_schedule` with:

    * ``enable_loopy_bp_schedule`` / residual-stop early-exit on,
    * ``enable_bp_policy_in_loopy`` on,
    * that cell's ``spectrum_cond_cap``,

    and reports the same harness ``spectrum_safe_sketch_ok`` rule as
    :func:`probe_loopy_bp_spectrum_safe_cert` (abs_tol/plateau + no ridge
    + finite non-neg residuals). Still **not** a production certificate.
    Does **not** flip mass/density ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_spectrum_safe_policy_pin_probe:
        return None
    caps = tuple(
        float(c)
        for c in (
            spectrum_cond_caps
            if spectrum_cond_caps is not None
            else (1e-12, 1.0, 1e6, 1e12)
        )
    )
    if not caps:
        raise ValueError("spectrum_cond_caps must be non-empty")
    if any(not np.isfinite(c) or c <= 0.0 for c in caps):
        raise ValueError("spectrum_cond_caps must be finite and > 0")

    max_iters = max(int(cfg.bp_max_iters), 2)
    cases: list[SpectrumSafePolicyPinCase] = []
    for cap in caps:
        run_cfg = DualFlowConfig(
            enable_loopy_bp_schedule=True,
            enable_loopy_bp_residual_stop=True,
            enable_bp_policy_in_loopy=True,
            bp_damping=float(cfg.bp_damping),
            bp_max_iters=max_iters,
            bp_residual_stop_tol=float(cfg.bp_residual_stop_tol),
            bp_residual_stop_patience=int(cfg.bp_residual_stop_patience),
            mu_scale=float(cfg.mu_scale),
            as_eps=float(cfg.as_eps),
            whiten_floor=float(cfg.whiten_floor),
            spectrum_cond_cap=float(cap),
            enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
        )
        out = solve_loopy_bp_schedule(
            empirical_by_simplex,
            stencils_by_simplex,
            simplices,
            config=run_cfg,
        )
        if out is None:
            raise RuntimeError(
                "loopy BP unexpectedly None under spectrum-safe×policy pin cfg"
            )
        rd = float(out.r_data)
        rc = float(out.r_cons)
        reason = out.residual_stop_reason
        finite_ok = bool(
            np.isfinite(rd) and np.isfinite(rc) and rd >= 0.0 and rc >= 0.0
        )
        sketch_ok = bool(
            finite_ok
            and reason in ("abs_tol", "plateau")
            and not bool(out.spectrum_ridge_applied)
        )
        cases.append(
            SpectrumSafePolicyPinCase(
                spectrum_cond_cap=float(cap),
                residual_stop_reason=reason,
                iters_executed=int(out.iters),
                max_iters=max_iters,
                r_data=rd,
                r_cons=rc,
                spectrum_ridge_applied=bool(out.spectrum_ridge_applied),
                policy_applied=bool(out.policy_applied),
                max_policy_damping=float(out.max_policy_damping),
                spectrum_safe_sketch_ok=sketch_ok,
            )
        )

    return SpectrumSafePolicyPinProbe(
        probe_flag_default_off=not DualFlowConfig().enable_spectrum_safe_policy_pin_probe,
        caps=caps,
        cases=tuple(cases),
    )


@dataclass(frozen=True)
class SpectrumSafePolicyTrajCase:
    """One ``spectrum_cond_cap`` residual-trajectory export cell (A5-T72)."""

    spectrum_cond_cap: float
    iters: tuple[int, ...]
    r_data_traj: tuple[float, ...]
    r_cons_traj: tuple[float, ...]
    policy_applied_any: bool
    max_policy_damping: float
    spectrum_ridge_applied_final: bool
    residual_stop_reason_final: str | None
    iters_executed_final: int
    max_iters: int
    spectrum_safe_sketch_ok: bool


@dataclass(frozen=True)
class SpectrumSafePolicyTrajProbe:
    """Spectrum-safe × policy cap-sweep residual traj export (SI S6.2; A5-T72).

    For each ``spectrum_cond_cap``, exports the multi-iter residual
    trajectory under ``enable_bp_policy_in_loopy`` (no in-solver residual
    stop, so the horizon is fully observed) and reports the same harness
    ``spectrum_safe_sketch_ok`` claim as T70 on a residual-stop final run.
    Harness only — **not** a production certificate. Do **not** flip
    mass/density ``@awaiting``.
    """

    probe_flag_default_off: bool
    caps: tuple[float, ...]
    cases: tuple[SpectrumSafePolicyTrajCase, ...]
    note: str = (
        "harness only: spectrum-safe × policy-in-loopy cap-sweep residual "
        "traj export; NOT a certified production convergence proof; "
        "do not flip mass/density @awaiting"
    )


def probe_spectrum_safe_policy_traj(
    empirical_by_simplex: Mapping[Hashable, np.ndarray],
    stencils_by_simplex: Mapping[Hashable, np.ndarray],
    simplices: Sequence[Sequence[Hashable]]
    | Mapping[Hashable, Sequence[Hashable]],
    *,
    spectrum_cond_caps: Sequence[float] | None = None,
    max_traj_iters: int | None = None,
    config: DualFlowConfig | None = None,
) -> SpectrumSafePolicyTrajProbe | None:
    """Export residual trajectories across ``spectrum_cond_cap`` (A5-T72).

    When ``enable_spectrum_safe_policy_traj_probe`` is off, returns ``None``.
    When on, for each cap in ``spectrum_cond_caps`` (default
    ``(1e-12, 1.0, 1e6, 1e12)``):

    1. **Trajectory** — re-run :func:`solve_loopy_bp_schedule` at
       ``bp_max_iters = 1..max_traj_iters`` with policy-in-loopy on and
       residual-stop off (full horizon).
    2. **Final sketch** — one residual-stop early-exit run at
       ``max_traj_iters`` reporting T70's ``spectrum_safe_sketch_ok`` rule.

    Still **not** a production certificate. Does **not** flip
    mass/density ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_spectrum_safe_policy_traj_probe:
        return None
    caps = tuple(
        float(c)
        for c in (
            spectrum_cond_caps
            if spectrum_cond_caps is not None
            else (1e-12, 1.0, 1e6, 1e12)
        )
    )
    if not caps:
        raise ValueError("spectrum_cond_caps must be non-empty")
    if any(not np.isfinite(c) or c <= 0.0 for c in caps):
        raise ValueError("spectrum_cond_caps must be finite and > 0")
    n_max = int(
        max_traj_iters
        if max_traj_iters is not None
        else max(int(cfg.bp_max_iters), 2)
    )
    if n_max < 1:
        raise ValueError("max_traj_iters must be >= 1")

    cases: list[SpectrumSafePolicyTrajCase] = []
    for cap in caps:
        iters: list[int] = []
        r_data_t: list[float] = []
        r_cons_t: list[float] = []
        policy_any = False
        max_damp = 0.0
        for k in range(1, n_max + 1):
            traj_cfg = DualFlowConfig(
                enable_loopy_bp_schedule=True,
                enable_bp_policy_in_loopy=True,
                enable_loopy_bp_residual_stop=False,
                bp_damping=float(cfg.bp_damping),
                bp_max_iters=k,
                mu_scale=float(cfg.mu_scale),
                as_eps=float(cfg.as_eps),
                whiten_floor=float(cfg.whiten_floor),
                spectrum_cond_cap=float(cap),
                enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
            )
            out = solve_loopy_bp_schedule(
                empirical_by_simplex,
                stencils_by_simplex,
                simplices,
                config=traj_cfg,
            )
            if out is None:
                raise RuntimeError(
                    "loopy BP unexpectedly None under spectrum-safe×policy traj cfg"
                )
            iters.append(k)
            r_data_t.append(float(out.r_data))
            r_cons_t.append(float(out.r_cons))
            policy_any = policy_any or bool(out.policy_applied)
            max_damp = max(max_damp, float(out.max_policy_damping))

        final_cfg = DualFlowConfig(
            enable_loopy_bp_schedule=True,
            enable_loopy_bp_residual_stop=True,
            enable_bp_policy_in_loopy=True,
            bp_damping=float(cfg.bp_damping),
            bp_max_iters=n_max,
            bp_residual_stop_tol=float(cfg.bp_residual_stop_tol),
            bp_residual_stop_patience=int(cfg.bp_residual_stop_patience),
            mu_scale=float(cfg.mu_scale),
            as_eps=float(cfg.as_eps),
            whiten_floor=float(cfg.whiten_floor),
            spectrum_cond_cap=float(cap),
            enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
        )
        final = solve_loopy_bp_schedule(
            empirical_by_simplex,
            stencils_by_simplex,
            simplices,
            config=final_cfg,
        )
        if final is None:
            raise RuntimeError(
                "loopy BP unexpectedly None under spectrum-safe×policy final cfg"
            )
        rd = float(final.r_data)
        rc = float(final.r_cons)
        reason = final.residual_stop_reason
        finite_ok = bool(
            np.isfinite(rd) and np.isfinite(rc) and rd >= 0.0 and rc >= 0.0
        )
        sketch_ok = bool(
            finite_ok
            and reason in ("abs_tol", "plateau")
            and not bool(final.spectrum_ridge_applied)
        )
        cases.append(
            SpectrumSafePolicyTrajCase(
                spectrum_cond_cap=float(cap),
                iters=tuple(iters),
                r_data_traj=tuple(r_data_t),
                r_cons_traj=tuple(r_cons_t),
                policy_applied_any=bool(policy_any),
                max_policy_damping=float(max_damp),
                spectrum_ridge_applied_final=bool(final.spectrum_ridge_applied),
                residual_stop_reason_final=reason,
                iters_executed_final=int(final.iters),
                max_iters=n_max,
                spectrum_safe_sketch_ok=sketch_ok,
            )
        )

    return SpectrumSafePolicyTrajProbe(
        probe_flag_default_off=not DualFlowConfig().enable_spectrum_safe_policy_traj_probe,
        caps=caps,
        cases=tuple(cases),
    )


@dataclass(frozen=True)
class ResidualMassLoopyComposeProbe:
    """Residual-stop × mass_loopy compose early-exit pin (SI S6.2; A5-T74).

    Pins ``r_data`` / ``r_cons`` over increasing loopy iters (residual-stop
    off so the pin covers the full horizon), then runs mass normalization
    together with online→offline loopy compose under residual-stop
    early-exit. Proposal-path only — **not** a production certificate.
    Do **not** flip mass/density ``@awaiting``.
    """

    probe_flag_default_off: bool
    pin_iters: tuple[int, ...]
    pin_r_data: tuple[float, ...]
    pin_r_cons: tuple[float, ...]
    epsilon_mass: float
    mass_total_before: float
    compose_n_samples: int
    compose_n_online_simplices: int
    compose_loopy_message_updates: int
    compose_loopy_r_cons: float
    compose_residual_stop_enabled: bool
    compose_residual_stop_reason: str | None
    compose_loopy_iters: int
    compose_max_iters: int
    note: str = (
        "sketch only: multi-iter residual pin + mass-norm × loopy compose "
        "with residual-stop early-exit; not a production certificate; "
        "do not flip mass/density @awaiting"
    )


def probe_residual_mass_loopy_compose(
    samples: Sequence[np.ndarray],
    simplex_positions: Mapping[Hashable, np.ndarray],
    simplices: Sequence[Sequence[Hashable]]
    | Mapping[Hashable, Sequence[Hashable]],
    *,
    masses: Mapping[Hashable, float] | None = None,
    max_pin_iters: int = 4,
    config: DualFlowConfig | None = None,
) -> ResidualMassLoopyComposeProbe | None:
    """Pin multi-iter residuals, then mass×loopy compose+stop (A5-T74).

    When ``enable_residual_mass_loopy_compose_probe`` is off, returns
    ``None``. When on:

    1. **Multi-iter residual pin** — re-run
       :func:`solve_loopy_bp_schedule` at ``bp_max_iters = 1..max_pin_iters``
       with residual-stop off (full horizon).
    2. **Mass × compose** — mass-normalize online winners and run
       online→offline loopy compose with residual-stop early-exit on.

    Does **not** flip mass/density ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_residual_mass_loopy_compose_probe:
        return None
    if not samples:
        raise ValueError("samples must be non-empty")
    if not simplex_positions:
        raise ValueError("simplex_positions must be non-empty")
    n_pin = int(max_pin_iters)
    if n_pin < 1:
        raise ValueError("max_pin_iters must be >= 1")

    tally_cfg = DualFlowConfig(
        enable_live_bmu_tally=True,
        tally_scale=float(cfg.tally_scale),
    )
    live = route_live_bmu_face_tallies(
        samples, simplex_positions, config=tally_cfg
    )
    if live is None:
        raise RuntimeError("live BMU tallies unexpectedly None")

    hats: dict[Hashable, np.ndarray] = {}
    stencils: dict[Hashable, np.ndarray] = {}
    for sid, tally in live.tallies_by_simplex.items():
        if sid not in simplex_positions:
            continue
        hats[sid] = np.asarray(tally.tallies, dtype=float)
        stencils[sid] = build_divergence_stencil(
            np.asarray(simplex_positions[sid], dtype=float)
        )
    if not hats:
        raise RuntimeError("online phase produced no simplex tallies")

    if isinstance(simplices, Mapping):
        face_simplices: Sequence[Sequence[Hashable]] | Mapping[
            Hashable, Sequence[Hashable]
        ] = {sid: simplices[sid] for sid in hats if sid in simplices}
        if len(face_simplices) != len(hats):
            missing = set(hats) - set(face_simplices)
            raise ValueError(
                f"simplices mapping missing winners {sorted(missing)!r}"
            )
    else:
        face_simplices = simplices

    pin_iters: list[int] = []
    pin_r_data: list[float] = []
    pin_r_cons: list[float] = []
    for k in range(1, n_pin + 1):
        pin_cfg = DualFlowConfig(
            enable_loopy_bp_schedule=True,
            enable_loopy_bp_residual_stop=False,
            bp_damping=float(cfg.bp_damping),
            bp_max_iters=k,
            mu_scale=float(cfg.mu_scale),
            as_eps=float(cfg.as_eps),
            whiten_floor=float(cfg.whiten_floor),
            spectrum_cond_cap=float(cfg.spectrum_cond_cap),
            enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
        )
        out = solve_loopy_bp_schedule(
            hats, stencils, face_simplices, config=pin_cfg
        )
        if out is None:
            raise RuntimeError("loopy BP unexpectedly None under pin cfg")
        pin_iters.append(k)
        pin_r_data.append(float(out.r_data))
        pin_r_cons.append(float(out.r_cons))

    if masses is None:
        mass_map: dict[Hashable, float] = {
            sid: 1.0 for sid in live.tallies_by_simplex
        }
    else:
        mass_map = {k: float(v) for k, v in masses.items()}
    if not mass_map:
        raise ValueError("masses must be non-empty for residual×mass_loopy probe")

    mass_cfg = DualFlowConfig(enable_mass_normalization=True)
    mass_out = normalize_simplex_masses(mass_map, config=mass_cfg)
    if mass_out is None:
        raise RuntimeError("mass normalization unexpectedly None under probe cfg")

    compose_max = max(int(cfg.bp_max_iters), n_pin, 2)
    compose_cfg = DualFlowConfig(
        enable_online_offline_loopy_compose=True,
        enable_loopy_bp_residual_stop=True,
        bp_damping=float(cfg.bp_damping),
        bp_max_iters=compose_max,
        bp_residual_stop_tol=float(cfg.bp_residual_stop_tol),
        bp_residual_stop_patience=int(cfg.bp_residual_stop_patience),
        tally_scale=float(cfg.tally_scale),
        mu_scale=float(cfg.mu_scale),
        as_eps=float(cfg.as_eps),
        whiten_floor=float(cfg.whiten_floor),
        spectrum_cond_cap=float(cfg.spectrum_cond_cap),
        enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
    )
    compose = run_online_offline_loopy_compose(
        samples, simplex_positions, simplices, config=compose_cfg
    )
    if compose is None:
        raise RuntimeError(
            "loopy compose unexpectedly None under residual×mass_loopy cfg"
        )

    return ResidualMassLoopyComposeProbe(
        probe_flag_default_off=not DualFlowConfig().enable_residual_mass_loopy_compose_probe,
        pin_iters=tuple(pin_iters),
        pin_r_data=tuple(pin_r_data),
        pin_r_cons=tuple(pin_r_cons),
        epsilon_mass=float(mass_out.epsilon_mass),
        mass_total_before=float(mass_out.total_before),
        compose_n_samples=int(compose.n_samples),
        compose_n_online_simplices=int(compose.n_online_simplices),
        compose_loopy_message_updates=int(compose.loopy_message_updates),
        compose_loopy_r_cons=float(compose.loopy_r_cons),
        compose_residual_stop_enabled=bool(compose.loopy_residual_stop_enabled),
        compose_residual_stop_reason=compose.loopy_residual_stop_reason,
        compose_loopy_iters=int(compose.loopy_iters),
        compose_max_iters=compose_max,
    )


@dataclass(frozen=True)
class SpectrumSafePolicyMassComposeCase:
    """One ``spectrum_cond_cap`` cell of spectrum×policy×mass compose (A5-T76)."""

    spectrum_cond_cap: float
    residual_stop_reason: str | None
    iters_executed: int
    max_iters: int
    r_data: float
    r_cons: float
    spectrum_ridge_applied: bool
    policy_applied: bool
    max_policy_damping: float
    spectrum_safe_sketch_ok: bool
    epsilon_mass: float
    mass_total_before: float
    compose_n_samples: int
    compose_n_online_simplices: int
    compose_loopy_message_updates: int
    compose_loopy_r_cons: float
    compose_policy_applied: bool
    compose_spectrum_ridge_applied: bool
    compose_residual_stop_enabled: bool
    compose_residual_stop_reason: str | None
    compose_loopy_iters: int
    compose_max_iters: int


@dataclass(frozen=True)
class SpectrumSafePolicyMassComposeProbe:
    """Spectrum-safe × policy × mass_loopy cap-sweep compose (SI S6.2; A5-T76).

    For each ``spectrum_cond_cap``, pins the T70 spectrum-safe×policy
    residual-stop harness sketch and runs mass-normalization together
    with online→offline loopy compose under policy-in-loopy + residual-
    stop at that cap. Harness only — **not** a production certificate.
    Do **not** flip mass/density ``@awaiting``.
    """

    probe_flag_default_off: bool
    caps: tuple[float, ...]
    cases: tuple[SpectrumSafePolicyMassComposeCase, ...]
    note: str = (
        "harness only: spectrum-safe × policy-in-loopy × mass_loopy "
        "cap-sweep compose; NOT a certified production convergence "
        "proof; do not flip mass/density @awaiting"
    )


def probe_spectrum_safe_policy_mass_compose(
    samples: Sequence[np.ndarray],
    simplex_positions: Mapping[Hashable, np.ndarray],
    simplices: Sequence[Sequence[Hashable]]
    | Mapping[Hashable, Sequence[Hashable]],
    *,
    masses: Mapping[Hashable, float] | None = None,
    spectrum_cond_caps: Sequence[float] | None = None,
    config: DualFlowConfig | None = None,
) -> SpectrumSafePolicyMassComposeProbe | None:
    """Cap-sweep spectrum×policy sketch + mass×loopy compose (A5-T76).

    When ``enable_spectrum_safe_policy_mass_compose_probe`` is off,
    returns ``None``. When on, for each cap in ``spectrum_cond_caps``
    (default ``(1e-12, 1.0, 1e6, 1e12)``):

    1. **Spectrum-safe × policy pin** — :func:`solve_loopy_bp_schedule`
       with policy-in-loopy + residual-stop early-exit at that cap
       (same ``spectrum_safe_sketch_ok`` rule as T70).
    2. **Mass × compose** — mass-normalize online winners and run
       online→offline loopy compose with policy + residual-stop at
       that same cap.

    Does **not** flip mass/density ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_spectrum_safe_policy_mass_compose_probe:
        return None
    if not samples:
        raise ValueError("samples must be non-empty")
    if not simplex_positions:
        raise ValueError("simplex_positions must be non-empty")
    caps = tuple(
        float(c)
        for c in (
            spectrum_cond_caps
            if spectrum_cond_caps is not None
            else (1e-12, 1.0, 1e6, 1e12)
        )
    )
    if not caps:
        raise ValueError("spectrum_cond_caps must be non-empty")
    if any(not np.isfinite(c) or c <= 0.0 for c in caps):
        raise ValueError("spectrum_cond_caps must be finite and > 0")

    tally_cfg = DualFlowConfig(
        enable_live_bmu_tally=True,
        tally_scale=float(cfg.tally_scale),
    )
    live = route_live_bmu_face_tallies(
        samples, simplex_positions, config=tally_cfg
    )
    if live is None:
        raise RuntimeError("live BMU tallies unexpectedly None")

    hats: dict[Hashable, np.ndarray] = {}
    stencils: dict[Hashable, np.ndarray] = {}
    for sid, tally in live.tallies_by_simplex.items():
        if sid not in simplex_positions:
            continue
        hats[sid] = np.asarray(tally.tallies, dtype=float)
        stencils[sid] = build_divergence_stencil(
            np.asarray(simplex_positions[sid], dtype=float)
        )
    if not hats:
        raise RuntimeError("online phase produced no simplex tallies")

    if isinstance(simplices, Mapping):
        face_simplices: Sequence[Sequence[Hashable]] | Mapping[
            Hashable, Sequence[Hashable]
        ] = {sid: simplices[sid] for sid in hats if sid in simplices}
        if len(face_simplices) != len(hats):
            missing = set(hats) - set(face_simplices)
            raise ValueError(
                f"simplices mapping missing winners {sorted(missing)!r}"
            )
    else:
        face_simplices = simplices

    if masses is None:
        mass_map: dict[Hashable, float] = {
            sid: 1.0 for sid in live.tallies_by_simplex
        }
    else:
        mass_map = {k: float(v) for k, v in masses.items()}
    if not mass_map:
        raise ValueError(
            "masses must be non-empty for spectrum×policy×mass compose probe"
        )

    mass_cfg = DualFlowConfig(enable_mass_normalization=True)
    mass_out = normalize_simplex_masses(mass_map, config=mass_cfg)
    if mass_out is None:
        raise RuntimeError("mass normalization unexpectedly None under probe cfg")

    max_iters = max(int(cfg.bp_max_iters), 2)
    cases: list[SpectrumSafePolicyMassComposeCase] = []
    for cap in caps:
        pin_cfg = DualFlowConfig(
            enable_loopy_bp_schedule=True,
            enable_loopy_bp_residual_stop=True,
            enable_bp_policy_in_loopy=True,
            bp_damping=float(cfg.bp_damping),
            bp_max_iters=max_iters,
            bp_residual_stop_tol=float(cfg.bp_residual_stop_tol),
            bp_residual_stop_patience=int(cfg.bp_residual_stop_patience),
            mu_scale=float(cfg.mu_scale),
            as_eps=float(cfg.as_eps),
            whiten_floor=float(cfg.whiten_floor),
            spectrum_cond_cap=float(cap),
            enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
        )
        pin = solve_loopy_bp_schedule(
            hats, stencils, face_simplices, config=pin_cfg
        )
        if pin is None:
            raise RuntimeError(
                "loopy BP unexpectedly None under spectrum×policy×mass pin cfg"
            )
        rd = float(pin.r_data)
        rc = float(pin.r_cons)
        reason = pin.residual_stop_reason
        finite_ok = bool(
            np.isfinite(rd) and np.isfinite(rc) and rd >= 0.0 and rc >= 0.0
        )
        sketch_ok = bool(
            finite_ok
            and reason in ("abs_tol", "plateau")
            and not bool(pin.spectrum_ridge_applied)
        )

        compose_cfg = DualFlowConfig(
            enable_online_offline_loopy_compose=True,
            enable_loopy_bp_residual_stop=True,
            enable_bp_policy_in_loopy=True,
            bp_damping=float(cfg.bp_damping),
            bp_max_iters=max_iters,
            bp_residual_stop_tol=float(cfg.bp_residual_stop_tol),
            bp_residual_stop_patience=int(cfg.bp_residual_stop_patience),
            tally_scale=float(cfg.tally_scale),
            mu_scale=float(cfg.mu_scale),
            as_eps=float(cfg.as_eps),
            whiten_floor=float(cfg.whiten_floor),
            spectrum_cond_cap=float(cap),
            enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
        )
        compose = run_online_offline_loopy_compose(
            samples, simplex_positions, simplices, config=compose_cfg
        )
        if compose is None:
            raise RuntimeError(
                "loopy compose unexpectedly None under spectrum×policy×mass cfg"
            )

        cases.append(
            SpectrumSafePolicyMassComposeCase(
                spectrum_cond_cap=float(cap),
                residual_stop_reason=reason,
                iters_executed=int(pin.iters),
                max_iters=max_iters,
                r_data=rd,
                r_cons=rc,
                spectrum_ridge_applied=bool(pin.spectrum_ridge_applied),
                policy_applied=bool(pin.policy_applied),
                max_policy_damping=float(pin.max_policy_damping),
                spectrum_safe_sketch_ok=sketch_ok,
                epsilon_mass=float(mass_out.epsilon_mass),
                mass_total_before=float(mass_out.total_before),
                compose_n_samples=int(compose.n_samples),
                compose_n_online_simplices=int(compose.n_online_simplices),
                compose_loopy_message_updates=int(compose.loopy_message_updates),
                compose_loopy_r_cons=float(compose.loopy_r_cons),
                compose_policy_applied=bool(compose.loopy_policy_applied),
                compose_spectrum_ridge_applied=bool(
                    compose.loopy_spectrum_ridge_applied
                ),
                compose_residual_stop_enabled=bool(
                    compose.loopy_residual_stop_enabled
                ),
                compose_residual_stop_reason=compose.loopy_residual_stop_reason,
                compose_loopy_iters=int(compose.loopy_iters),
                compose_max_iters=max_iters,
            )
        )

    return SpectrumSafePolicyMassComposeProbe(
        probe_flag_default_off=not DualFlowConfig().enable_spectrum_safe_policy_mass_compose_probe,
        caps=caps,
        cases=tuple(cases),
    )


@dataclass(frozen=True)
class ResidualMassPatienceSweepCase:
    """One ``bp_residual_stop_patience`` cell of residual×mass sweep (A5-T77)."""

    patience: int
    epsilon_mass: float
    mass_total_before: float
    compose_n_samples: int
    compose_n_online_simplices: int
    compose_loopy_message_updates: int
    compose_loopy_r_cons: float
    compose_residual_stop_enabled: bool
    compose_residual_stop_reason: str | None
    compose_loopy_iters: int
    compose_max_iters: int


@dataclass(frozen=True)
class ResidualMassPatienceSweepProbe:
    """Residual-stop × mass_loopy patience sweep (SI S6.2; A5-T77).

    Mass-normalizes online winners once, then for each patience value
    runs online→offline loopy compose under residual-stop early-exit
    and reports iters / stop reason. Proposal-path only — **not** a
    production certificate. Do **not** flip mass/density ``@awaiting``.
    """

    probe_flag_default_off: bool
    patience_grid: tuple[int, ...]
    cases: tuple[ResidualMassPatienceSweepCase, ...]
    note: str = (
        "sketch only: mass-norm × loopy compose residual-stop patience "
        "sweep; not a production certificate; do not flip mass/density "
        "@awaiting"
    )


def probe_residual_mass_patience_sweep(
    samples: Sequence[np.ndarray],
    simplex_positions: Mapping[Hashable, np.ndarray],
    simplices: Sequence[Sequence[Hashable]]
    | Mapping[Hashable, Sequence[Hashable]],
    *,
    masses: Mapping[Hashable, float] | None = None,
    patience_grid: Sequence[int] | None = None,
    config: DualFlowConfig | None = None,
) -> ResidualMassPatienceSweepProbe | None:
    """Sweep residual-stop patience under mass×loopy compose (A5-T77).

    When ``enable_residual_mass_patience_sweep_probe`` is off, returns
    ``None``. When on:

    1. **Mass** — mass-normalize online winners.
    2. **Patience sweep** — for each patience in ``patience_grid``
       (default ``(1, 2, 4, 8)``), run online→offline loopy compose
       with residual-stop early-exit at that patience.

    Does **not** flip mass/density ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_residual_mass_patience_sweep_probe:
        return None
    if not samples:
        raise ValueError("samples must be non-empty")
    if not simplex_positions:
        raise ValueError("simplex_positions must be non-empty")
    grid = tuple(
        int(p)
        for p in (
            patience_grid if patience_grid is not None else (1, 2, 4, 8)
        )
    )
    if not grid:
        raise ValueError("patience_grid must be non-empty")
    if any(p < 1 for p in grid):
        raise ValueError("patience_grid values must be >= 1")

    tally_cfg = DualFlowConfig(
        enable_live_bmu_tally=True,
        tally_scale=float(cfg.tally_scale),
    )
    live = route_live_bmu_face_tallies(
        samples, simplex_positions, config=tally_cfg
    )
    if live is None:
        raise RuntimeError("live BMU tallies unexpectedly None")

    if masses is None:
        mass_map: dict[Hashable, float] = {
            sid: 1.0 for sid in live.tallies_by_simplex
        }
    else:
        mass_map = {k: float(v) for k, v in masses.items()}
    if not mass_map:
        raise ValueError(
            "masses must be non-empty for residual×mass patience sweep"
        )

    mass_cfg = DualFlowConfig(enable_mass_normalization=True)
    mass_out = normalize_simplex_masses(mass_map, config=mass_cfg)
    if mass_out is None:
        raise RuntimeError("mass normalization unexpectedly None under probe cfg")

    compose_max = max(int(cfg.bp_max_iters), max(grid) + 2, 4)
    cases: list[ResidualMassPatienceSweepCase] = []
    for patience in grid:
        compose_cfg = DualFlowConfig(
            enable_online_offline_loopy_compose=True,
            enable_loopy_bp_residual_stop=True,
            bp_damping=float(cfg.bp_damping),
            bp_max_iters=compose_max,
            bp_residual_stop_tol=float(cfg.bp_residual_stop_tol),
            bp_residual_stop_patience=int(patience),
            tally_scale=float(cfg.tally_scale),
            mu_scale=float(cfg.mu_scale),
            as_eps=float(cfg.as_eps),
            whiten_floor=float(cfg.whiten_floor),
            spectrum_cond_cap=float(cfg.spectrum_cond_cap),
            enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
        )
        compose = run_online_offline_loopy_compose(
            samples, simplex_positions, simplices, config=compose_cfg
        )
        if compose is None:
            raise RuntimeError(
                "loopy compose unexpectedly None under residual×mass patience cfg"
            )
        cases.append(
            ResidualMassPatienceSweepCase(
                patience=int(patience),
                epsilon_mass=float(mass_out.epsilon_mass),
                mass_total_before=float(mass_out.total_before),
                compose_n_samples=int(compose.n_samples),
                compose_n_online_simplices=int(compose.n_online_simplices),
                compose_loopy_message_updates=int(compose.loopy_message_updates),
                compose_loopy_r_cons=float(compose.loopy_r_cons),
                compose_residual_stop_enabled=bool(
                    compose.loopy_residual_stop_enabled
                ),
                compose_residual_stop_reason=compose.loopy_residual_stop_reason,
                compose_loopy_iters=int(compose.loopy_iters),
                compose_max_iters=compose_max,
            )
        )

    return ResidualMassPatienceSweepProbe(
        probe_flag_default_off=not DualFlowConfig().enable_residual_mass_patience_sweep_probe,
        patience_grid=grid,
        cases=tuple(cases),
    )


@dataclass(frozen=True)
class SpectrumSafePolicyMassTrajCase:
    """One ``spectrum_cond_cap`` cell of spectrum×policy×mass traj (A5-T78)."""

    spectrum_cond_cap: float
    iters: tuple[int, ...]
    r_data_traj: tuple[float, ...]
    r_cons_traj: tuple[float, ...]
    policy_applied_any: bool
    max_policy_damping: float
    spectrum_ridge_applied_final: bool
    residual_stop_reason_final: str | None
    iters_executed_final: int
    max_iters: int
    spectrum_safe_sketch_ok: bool
    epsilon_mass: float
    mass_total_before: float
    compose_n_samples: int
    compose_n_online_simplices: int
    compose_loopy_message_updates: int
    compose_loopy_r_cons: float
    compose_policy_applied: bool
    compose_spectrum_ridge_applied: bool
    compose_residual_stop_enabled: bool
    compose_residual_stop_reason: str | None
    compose_loopy_iters: int
    compose_max_iters: int


@dataclass(frozen=True)
class SpectrumSafePolicyMassTrajProbe:
    """Spectrum-safe × policy × mass_loopy traj export (SI S6.2; A5-T78).

    For each ``spectrum_cond_cap``, exports the multi-iter residual
    trajectory under ``enable_bp_policy_in_loopy`` (full horizon),
    reports the T70 harness ``spectrum_safe_sketch_ok`` claim on a
    residual-stop final run, and runs mass-normalization together with
    online→offline loopy compose under policy + residual-stop at that
    cap. Harness only — **not** a production certificate. Do **not**
    flip mass/density ``@awaiting``.
    """

    probe_flag_default_off: bool
    caps: tuple[float, ...]
    cases: tuple[SpectrumSafePolicyMassTrajCase, ...]
    note: str = (
        "harness only: spectrum-safe × policy-in-loopy × mass_loopy "
        "cap-sweep residual traj export; NOT a certified production "
        "convergence proof; do not flip mass/density @awaiting"
    )


def probe_spectrum_safe_policy_mass_traj(
    samples: Sequence[np.ndarray],
    simplex_positions: Mapping[Hashable, np.ndarray],
    simplices: Sequence[Sequence[Hashable]]
    | Mapping[Hashable, Sequence[Hashable]],
    *,
    masses: Mapping[Hashable, float] | None = None,
    spectrum_cond_caps: Sequence[float] | None = None,
    max_traj_iters: int | None = None,
    config: DualFlowConfig | None = None,
) -> SpectrumSafePolicyMassTrajProbe | None:
    """Cap-sweep residual traj + mass×loopy compose (A5-T78).

    When ``enable_spectrum_safe_policy_mass_traj_probe`` is off, returns
    ``None``. When on, for each cap in ``spectrum_cond_caps`` (default
    ``(1e-12, 1.0, 1e6, 1e12)``):

    1. **Trajectory** — re-run :func:`solve_loopy_bp_schedule` at
       ``bp_max_iters = 1..max_traj_iters`` with policy-in-loopy on and
       residual-stop off (full horizon).
    2. **Final sketch** — one residual-stop early-exit run at
       ``max_traj_iters`` reporting T70's ``spectrum_safe_sketch_ok``.
    3. **Mass × compose** — mass-normalize online winners and run
       online→offline loopy compose with policy + residual-stop at
       that same cap.

    Does **not** flip mass/density ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_spectrum_safe_policy_mass_traj_probe:
        return None
    if not samples:
        raise ValueError("samples must be non-empty")
    if not simplex_positions:
        raise ValueError("simplex_positions must be non-empty")
    caps = tuple(
        float(c)
        for c in (
            spectrum_cond_caps
            if spectrum_cond_caps is not None
            else (1e-12, 1.0, 1e6, 1e12)
        )
    )
    if not caps:
        raise ValueError("spectrum_cond_caps must be non-empty")
    if any(not np.isfinite(c) or c <= 0.0 for c in caps):
        raise ValueError("spectrum_cond_caps must be finite and > 0")
    n_max = int(
        max_traj_iters
        if max_traj_iters is not None
        else max(int(cfg.bp_max_iters), 2)
    )
    if n_max < 1:
        raise ValueError("max_traj_iters must be >= 1")

    tally_cfg = DualFlowConfig(
        enable_live_bmu_tally=True,
        tally_scale=float(cfg.tally_scale),
    )
    live = route_live_bmu_face_tallies(
        samples, simplex_positions, config=tally_cfg
    )
    if live is None:
        raise RuntimeError("live BMU tallies unexpectedly None")

    hats: dict[Hashable, np.ndarray] = {}
    stencils: dict[Hashable, np.ndarray] = {}
    for sid, tally in live.tallies_by_simplex.items():
        if sid not in simplex_positions:
            continue
        hats[sid] = np.asarray(tally.tallies, dtype=float)
        stencils[sid] = build_divergence_stencil(
            np.asarray(simplex_positions[sid], dtype=float)
        )
    if not hats:
        raise RuntimeError("online phase produced no simplex tallies")

    if isinstance(simplices, Mapping):
        face_simplices: Sequence[Sequence[Hashable]] | Mapping[
            Hashable, Sequence[Hashable]
        ] = {sid: simplices[sid] for sid in hats if sid in simplices}
        if len(face_simplices) != len(hats):
            missing = set(hats) - set(face_simplices)
            raise ValueError(
                f"simplices mapping missing winners {sorted(missing)!r}"
            )
    else:
        face_simplices = simplices

    if masses is None:
        mass_map: dict[Hashable, float] = {
            sid: 1.0 for sid in live.tallies_by_simplex
        }
    else:
        mass_map = {k: float(v) for k, v in masses.items()}
    if not mass_map:
        raise ValueError(
            "masses must be non-empty for spectrum×policy×mass traj probe"
        )

    mass_cfg = DualFlowConfig(enable_mass_normalization=True)
    mass_out = normalize_simplex_masses(mass_map, config=mass_cfg)
    if mass_out is None:
        raise RuntimeError("mass normalization unexpectedly None under probe cfg")

    cases: list[SpectrumSafePolicyMassTrajCase] = []
    for cap in caps:
        iters: list[int] = []
        r_data_t: list[float] = []
        r_cons_t: list[float] = []
        policy_any = False
        max_damp = 0.0
        for k in range(1, n_max + 1):
            traj_cfg = DualFlowConfig(
                enable_loopy_bp_schedule=True,
                enable_bp_policy_in_loopy=True,
                enable_loopy_bp_residual_stop=False,
                bp_damping=float(cfg.bp_damping),
                bp_max_iters=k,
                mu_scale=float(cfg.mu_scale),
                as_eps=float(cfg.as_eps),
                whiten_floor=float(cfg.whiten_floor),
                spectrum_cond_cap=float(cap),
                enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
            )
            out = solve_loopy_bp_schedule(
                hats, stencils, face_simplices, config=traj_cfg
            )
            if out is None:
                raise RuntimeError(
                    "loopy BP unexpectedly None under spectrum×policy×mass traj cfg"
                )
            iters.append(k)
            r_data_t.append(float(out.r_data))
            r_cons_t.append(float(out.r_cons))
            policy_any = policy_any or bool(out.policy_applied)
            max_damp = max(max_damp, float(out.max_policy_damping))

        final_cfg = DualFlowConfig(
            enable_loopy_bp_schedule=True,
            enable_loopy_bp_residual_stop=True,
            enable_bp_policy_in_loopy=True,
            bp_damping=float(cfg.bp_damping),
            bp_max_iters=n_max,
            bp_residual_stop_tol=float(cfg.bp_residual_stop_tol),
            bp_residual_stop_patience=int(cfg.bp_residual_stop_patience),
            mu_scale=float(cfg.mu_scale),
            as_eps=float(cfg.as_eps),
            whiten_floor=float(cfg.whiten_floor),
            spectrum_cond_cap=float(cap),
            enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
        )
        final = solve_loopy_bp_schedule(
            hats, stencils, face_simplices, config=final_cfg
        )
        if final is None:
            raise RuntimeError(
                "loopy BP unexpectedly None under spectrum×policy×mass final cfg"
            )
        rd = float(final.r_data)
        rc = float(final.r_cons)
        reason = final.residual_stop_reason
        finite_ok = bool(
            np.isfinite(rd) and np.isfinite(rc) and rd >= 0.0 and rc >= 0.0
        )
        sketch_ok = bool(
            finite_ok
            and reason in ("abs_tol", "plateau")
            and not bool(final.spectrum_ridge_applied)
        )

        compose_cfg = DualFlowConfig(
            enable_online_offline_loopy_compose=True,
            enable_loopy_bp_residual_stop=True,
            enable_bp_policy_in_loopy=True,
            bp_damping=float(cfg.bp_damping),
            bp_max_iters=n_max,
            bp_residual_stop_tol=float(cfg.bp_residual_stop_tol),
            bp_residual_stop_patience=int(cfg.bp_residual_stop_patience),
            tally_scale=float(cfg.tally_scale),
            mu_scale=float(cfg.mu_scale),
            as_eps=float(cfg.as_eps),
            whiten_floor=float(cfg.whiten_floor),
            spectrum_cond_cap=float(cap),
            enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
        )
        compose = run_online_offline_loopy_compose(
            samples, simplex_positions, simplices, config=compose_cfg
        )
        if compose is None:
            raise RuntimeError(
                "loopy compose unexpectedly None under spectrum×policy×mass traj cfg"
            )

        cases.append(
            SpectrumSafePolicyMassTrajCase(
                spectrum_cond_cap=float(cap),
                iters=tuple(iters),
                r_data_traj=tuple(r_data_t),
                r_cons_traj=tuple(r_cons_t),
                policy_applied_any=bool(policy_any),
                max_policy_damping=float(max_damp),
                spectrum_ridge_applied_final=bool(final.spectrum_ridge_applied),
                residual_stop_reason_final=reason,
                iters_executed_final=int(final.iters),
                max_iters=n_max,
                spectrum_safe_sketch_ok=sketch_ok,
                epsilon_mass=float(mass_out.epsilon_mass),
                mass_total_before=float(mass_out.total_before),
                compose_n_samples=int(compose.n_samples),
                compose_n_online_simplices=int(compose.n_online_simplices),
                compose_loopy_message_updates=int(compose.loopy_message_updates),
                compose_loopy_r_cons=float(compose.loopy_r_cons),
                compose_policy_applied=bool(compose.loopy_policy_applied),
                compose_spectrum_ridge_applied=bool(
                    compose.loopy_spectrum_ridge_applied
                ),
                compose_residual_stop_enabled=bool(
                    compose.loopy_residual_stop_enabled
                ),
                compose_residual_stop_reason=compose.loopy_residual_stop_reason,
                compose_loopy_iters=int(compose.loopy_iters),
                compose_max_iters=n_max,
            )
        )

    return SpectrumSafePolicyMassTrajProbe(
        probe_flag_default_off=not DualFlowConfig().enable_spectrum_safe_policy_mass_traj_probe,
        caps=caps,
        cases=tuple(cases),
    )


@dataclass(frozen=True)
class ResidualMassPolicyPatienceCase:
    """One patience cell of residual×mass×policy compose (A5-T79)."""

    patience: int
    spectrum_cond_cap: float
    epsilon_mass: float
    mass_total_before: float
    compose_n_samples: int
    compose_n_online_simplices: int
    compose_loopy_message_updates: int
    compose_loopy_r_cons: float
    compose_policy_applied: bool
    compose_spectrum_ridge_applied: bool
    compose_residual_stop_enabled: bool
    compose_residual_stop_reason: str | None
    compose_loopy_iters: int
    compose_max_iters: int


@dataclass(frozen=True)
class ResidualMassPolicyPatienceProbe:
    """Residual-stop × mass_loopy × policy patience compose (SI S6.2; A5-T79).

    Mass-normalizes online winners once, then for each patience value
    runs online→offline loopy compose under residual-stop early-exit
    with ``enable_bp_policy_in_loopy`` and reports iters / stop reason /
    policy engagement. Proposal-path only — **not** a production
    certificate. Do **not** flip mass/density ``@awaiting``.
    """

    probe_flag_default_off: bool
    patience_grid: tuple[int, ...]
    spectrum_cond_cap: float
    cases: tuple[ResidualMassPolicyPatienceCase, ...]
    note: str = (
        "sketch only: mass-norm × policy-in-loopy × loopy compose "
        "residual-stop patience sweep; not a production certificate; "
        "do not flip mass/density @awaiting"
    )


def probe_residual_mass_policy_patience(
    samples: Sequence[np.ndarray],
    simplex_positions: Mapping[Hashable, np.ndarray],
    simplices: Sequence[Sequence[Hashable]]
    | Mapping[Hashable, Sequence[Hashable]],
    *,
    masses: Mapping[Hashable, float] | None = None,
    patience_grid: Sequence[int] | None = None,
    config: DualFlowConfig | None = None,
) -> ResidualMassPolicyPatienceProbe | None:
    """Sweep residual-stop patience under mass×policy×loopy compose (A5-T79).

    When ``enable_residual_mass_policy_patience_probe`` is off, returns
    ``None``. When on:

    1. **Mass** — mass-normalize online winners.
    2. **Patience sweep** — for each patience in ``patience_grid``
       (default ``(1, 2, 4, 8)``), run online→offline loopy compose
       with residual-stop early-exit **and** ``enable_bp_policy_in_loopy``
       at ``config.spectrum_cond_cap``.

    Does **not** flip mass/density ``@awaiting``.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_residual_mass_policy_patience_probe:
        return None
    if not samples:
        raise ValueError("samples must be non-empty")
    if not simplex_positions:
        raise ValueError("simplex_positions must be non-empty")
    grid = tuple(
        int(p)
        for p in (
            patience_grid if patience_grid is not None else (1, 2, 4, 8)
        )
    )
    if not grid:
        raise ValueError("patience_grid must be non-empty")
    if any(p < 1 for p in grid):
        raise ValueError("patience_grid values must be >= 1")

    tally_cfg = DualFlowConfig(
        enable_live_bmu_tally=True,
        tally_scale=float(cfg.tally_scale),
    )
    live = route_live_bmu_face_tallies(
        samples, simplex_positions, config=tally_cfg
    )
    if live is None:
        raise RuntimeError("live BMU tallies unexpectedly None")

    if masses is None:
        mass_map: dict[Hashable, float] = {
            sid: 1.0 for sid in live.tallies_by_simplex
        }
    else:
        mass_map = {k: float(v) for k, v in masses.items()}
    if not mass_map:
        raise ValueError(
            "masses must be non-empty for residual×mass×policy patience"
        )

    mass_cfg = DualFlowConfig(enable_mass_normalization=True)
    mass_out = normalize_simplex_masses(mass_map, config=mass_cfg)
    if mass_out is None:
        raise RuntimeError("mass normalization unexpectedly None under probe cfg")

    cap = float(cfg.spectrum_cond_cap)
    compose_max = max(int(cfg.bp_max_iters), max(grid) + 2, 4)
    cases: list[ResidualMassPolicyPatienceCase] = []
    for patience in grid:
        compose_cfg = DualFlowConfig(
            enable_online_offline_loopy_compose=True,
            enable_loopy_bp_residual_stop=True,
            enable_bp_policy_in_loopy=True,
            bp_damping=float(cfg.bp_damping),
            bp_max_iters=compose_max,
            bp_residual_stop_tol=float(cfg.bp_residual_stop_tol),
            bp_residual_stop_patience=int(patience),
            tally_scale=float(cfg.tally_scale),
            mu_scale=float(cfg.mu_scale),
            as_eps=float(cfg.as_eps),
            whiten_floor=float(cfg.whiten_floor),
            spectrum_cond_cap=cap,
            enable_count_aware_lambda=bool(cfg.enable_count_aware_lambda),
        )
        compose = run_online_offline_loopy_compose(
            samples, simplex_positions, simplices, config=compose_cfg
        )
        if compose is None:
            raise RuntimeError(
                "loopy compose unexpectedly None under residual×mass×policy patience cfg"
            )
        cases.append(
            ResidualMassPolicyPatienceCase(
                patience=int(patience),
                spectrum_cond_cap=cap,
                epsilon_mass=float(mass_out.epsilon_mass),
                mass_total_before=float(mass_out.total_before),
                compose_n_samples=int(compose.n_samples),
                compose_n_online_simplices=int(compose.n_online_simplices),
                compose_loopy_message_updates=int(compose.loopy_message_updates),
                compose_loopy_r_cons=float(compose.loopy_r_cons),
                compose_policy_applied=bool(compose.loopy_policy_applied),
                compose_spectrum_ridge_applied=bool(
                    compose.loopy_spectrum_ridge_applied
                ),
                compose_residual_stop_enabled=bool(
                    compose.loopy_residual_stop_enabled
                ),
                compose_residual_stop_reason=compose.loopy_residual_stop_reason,
                compose_loopy_iters=int(compose.loopy_iters),
                compose_max_iters=compose_max,
            )
        )

    return ResidualMassPolicyPatienceProbe(
        probe_flag_default_off=not DualFlowConfig().enable_residual_mass_policy_patience_probe,
        patience_grid=grid,
        spectrum_cond_cap=cap,
        cases=tuple(cases),
    )
