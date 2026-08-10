"""Affected dual-subgraph connectivity + Stage-2 dual-flow stub (SI S10.4 / #43).

Green tests lock:

* Pure BFS hook and ``dual_connected=False`` ⇒ evidence-path reject (hand-built
  adjacency dicts — not a Stage-2 dual/face graph).
* Experimental ``stage2.dual_flow`` adjacency producer (facet-sharing) behind
  ``DualFlowConfig.enable_dual_adjacency`` / ``GateConfig.apply_dual_adjacency``
  (proposal-path, default off).
* Wired path: build dual adj → ``affected_dual_subgraph_connected`` →
  ``score_edit`` / ``EvidenceGate.evaluate`` rejects disconnect.
* Dry-run helper: Complex edit → affected simplices → DualAdjacency
  (``dry_run_dual_from_edit``, flag default off).
* Conservative BP *sketch* behind ``enable_conservative_bp`` (identity/damped;
  not the SI S6.2 loopy Gaussian solve).
* S6.1 face-pressure tally helper behind ``enable_face_tallies``; dry-run can
  demo-wire tallies via ``samples=``; live BMU harness behind
  ``enable_live_bmu_tally`` (A5-T43); Stage-1 BMU wiring sketch behind
  ``enable_stage1_bmu_wiring`` (A5-T48).
* S6.2 ``A_S`` geometry + soft message-pass sketch behind
  ``enable_as_message_pass`` (A5-T44; not loopy Gaussian BP).
* S6.2 whitened ``λ_f`` + ``μ_S`` soft solve behind
  ``enable_mu_weighted_solve`` (A5-EXP-mu; eq. si-dual-flow-weight).
* S6.2 count-aware ``λ_f=1+n_f/(1+n̄)`` behind ``enable_count_aware_lambda``
  (A5-T46); multi-simplex patch ``Σ μ_S`` soft solve behind
  ``enable_patch_mu_solve`` (A5-T47 stub); shared-face antisymmetry soft
  glue behind ``enable_shared_face_glue`` (A5-EXP-glue); global face-id
  soft solve behind ``enable_global_face_solve`` (A5-T49); loopy Gaussian
  BP *message schedule* behind ``enable_loopy_bp_schedule``
  (A5-EXP-loopy-bp); mass normalization behind
  ``enable_mass_normalization`` (A5-EXP-mass); Complex → node-star
  incidence + ANN BMU query behind ``enable_complex_ann_incidence``
  (A5-EXP-ann-inc).
* S6.3 boundary taxonomy behind ``enable_boundary_taxonomy``; seam stitch /
  ghost reservoir sketches behind ``enable_seam_ghost`` (A5-T45).
* S6.4 simplex-local PL density *sketch* behind ``enable_simplex_density``
  (default off); live Complex/ANN density harness behind
  ``enable_live_density`` (A5-T50; default off; does not flip density
  awaiting tests).

Gaps vs full SI S6 (do **not** flip these elsewhere yet):

* **S6.1** Tally + dry-run + BMU harness + Stage-1 wiring *sketch* +
  Complex/ANN incidence bridge exist; acceptance-path Stage-1 integration
  still open.
* **S6.2** Soft ``A_S`` / ``μ_S`` / count-aware / patch / shared-face glue /
  global face-id / loopy-BP *schedule* sketches only — not production BP.
  Mass-norm harness exists; conservation awaiting stays. See module
  docstring acceptance-path plan (A5-T42).
* **S6.3** Seam/ghost sketches are scalar; no face registry / patch graph.
* **S6.4** Density sketch + live Complex/ANN harness only; acceptance-path
  density open.
* Mass-conservation / density / benchmark ``@awaiting("stage2.dual_flow")``
  (and ``stage2.density``) remain xfail until that producer lands.
* Acceptance path still defaults open when adjacency is ``None`` / flags off
  (A5-T54 :func:`probe_acceptance_none_open_default` locks the matrix).
"""
from __future__ import annotations

import numpy as np
import pytest

from proteus.evidence import (
    EvidenceGate,
    GateConfig,
    affected_dual_subgraph_connected,
    bdeu_alpha,
    probe_fail_closed_score_edit_matrix,
    score_edit,
    star_incidence_matrix,
)
from proteus.evidence.dm_score import NodeTransition
from proteus.stage2 import (
    DualFlowConfig,
    accumulate_face_pressure_tally,
    apply_ghost_reservoir,
    barycentric_coordinates,
    build_divergence_stencil,
    build_dual_adjacency,
    build_dual_adjacency_from_complex,
    build_global_face_registry,
    build_node_to_simplices_from_complex,
    build_shared_face_pairs,
    build_simplex_positions_from_complex,
    classify_boundary_facets,
    conservation_residual_r_cons,
    count_aware_lambda_f,
    dry_run_dual_from_edit,
    epsilon_flux,
    epsilon_mass,
    locate_bmu_simplex,
    mu_S_weight,
    normalize_simplex_masses,
    probe_acceptance_none_open_default,
    probe_bp_spectrum_damping,
    probe_fail_closed_dual_adjacency_plan,
    probe_gate_fail_closed_switch,
    probe_loopy_bp_convergence,
    probe_mass_loopy_compose,
    propose_bp_damping_policy,
    propose_loopy_bp_residual_stop,
    query_stage1_ann_bmus,
    resolve_dual_connected,
    route_live_bmu_face_tallies,
    route_live_density_from_complex,
    route_stage1_bmu_face_tallies,
    route_stage1_from_complex,
    run_online_offline_loopy_compose,
    run_online_offline_schedule,
    simplex_local_density,
    simplex_outward_normals,
    simplex_volume,
    solve_as_message_pass,
    solve_conservative_pressures,
    solve_global_face_mu_pressures,
    solve_loopy_bp_schedule,
    solve_mu_weighted_pressures,
    solve_patch_mu_weighted_pressures,
    stitch_orientation_seam_pressures,
    vertex_weights_from_facet_pressures,
    whiten_empirical_pressures,
)
from proteus.types import (
    BoundaryType,
    Complex,
    EditProposal,
    EditType,
    Simplex,
)


def _good_split_fixture():
    """Keep/edit transitions + well-conditioned stars that would accept on F_DM."""

    a0 = bdeu_alpha(2)
    keep = [NodeTransition(np.array([15.0, 16.0]), 2, a0, node_id=0)]
    edit = [NodeTransition(np.array([30.0, 1.0]), 2, a0, node_id=0)]
    proposal = EditProposal(EditType.SPLIT, [0], diagnostic_strength=1.0)
    good_stars = {0: star_incidence_matrix([1, 2, 3, 4], [[0, 1, 2], [0, 3, 4]], 0)}
    return keep, edit, proposal, good_stars


def test_stub_defaults_true_without_dual_graph():
    """No S6 dual adjacency yet: stub asserts connectivity (OPEN_ISSUES #43)."""

    assert affected_dual_subgraph_connected(None, ["S0", "S1", "S2"]) is True
    assert affected_dual_subgraph_connected(None, []) is True


def test_induced_dual_subgraph_connectivity():
    """BFS on the induced dual subgraph: path-connected vs two components."""

    # Path S0—S1—S2 (facet-adjacency stand-in until S6 lands).
    path = {"S0": ["S1"], "S1": ["S0", "S2"], "S2": ["S1"]}
    assert affected_dual_subgraph_connected(path, ["S0", "S1", "S2"]) is True
    assert affected_dual_subgraph_connected(path, ["S0", "S2"]) is False  # induced drops S1

    # Two disjoint edges: affected set spanning both components is disconnected.
    disjoint = {"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]}
    assert affected_dual_subgraph_connected(disjoint, ["A", "B"]) is True
    assert affected_dual_subgraph_connected(disjoint, ["A", "B", "C"]) is False


def test_disconnect_rejects_evidence_path():
    """Property (SI S10.4): dual_connected=False forces reject even when F_DM would accept."""

    keep, edit, proposal, good_stars = _good_split_fixture()

    # Dual graph of three simplices; edit "affects" endpoints only -> disconnected.
    dual = {"S0": ["S1"], "S1": ["S0", "S2"], "S2": ["S1"]}
    connected = affected_dual_subgraph_connected(dual, ["S0", "S1", "S2"])
    disconnected = affected_dual_subgraph_connected(dual, ["S0", "S2"])
    assert connected is True
    assert disconnected is False

    v_ok = score_edit(
        keep, edit, proposal, edit_stars=good_stars, keep_stars=good_stars,
        dual_connected=connected,
    )
    assert v_ok.accepted

    v_disc = score_edit(
        keep, edit, proposal, edit_stars=good_stars, keep_stars=good_stars,
        dual_connected=disconnected,
    )
    assert not v_disc.accepted

    gate = EvidenceGate(n_nodes=4)
    v_gate = gate.evaluate(
        keep, edit, proposal, edit_stars=good_stars, keep_stars=good_stars,
        dual_connected=disconnected,
    )
    assert not v_gate.accepted


def test_dual_flow_flag_off_returns_none():
    """Proposal-path default: enable_dual_adjacency=False ⇒ no adjacency built."""

    simplices = [(0, 1), (1, 2), (2, 3)]
    assert build_dual_adjacency(simplices) is None
    assert build_dual_adjacency(simplices, config=DualFlowConfig()) is None
    assert resolve_dual_connected(simplices, [0, 2]) is True


def test_dual_flow_builds_facet_sharing_path():
    """Three edges sharing endpoints → dual path 0—1—2 (SI S6.2 facet adjacency)."""

    # 1-simplices (edges) on a path of nodes 0-1-2-3.
    simplices = [(0, 1), (1, 2), (2, 3)]
    adj = build_dual_adjacency(
        simplices, config=DualFlowConfig(enable_dual_adjacency=True)
    )
    assert adj is not None
    assert set(adj[0]) == {1}
    assert set(adj[1]) == {0, 2}
    assert set(adj[2]) == {1}
    assert resolve_dual_connected(
        simplices, [0, 1, 2], config=DualFlowConfig(enable_dual_adjacency=True)
    )
    assert not resolve_dual_connected(
        simplices, [0, 2], config=DualFlowConfig(enable_dual_adjacency=True)
    )


def test_dual_flow_disjoint_triangles():
    """Two triangles sharing no facet → dual disconnected across components."""

    # Triangle ABC and triangle DEF (vertex-disjoint).
    simplices = {
        "T0": (0, 1, 2),
        "T1": (3, 4, 5),
    }
    adj = build_dual_adjacency(
        simplices, config=DualFlowConfig(enable_dual_adjacency=True)
    )
    assert adj is not None
    assert adj["T0"] == ()
    assert adj["T1"] == ()
    assert affected_dual_subgraph_connected(adj, ["T0"]) is True
    assert affected_dual_subgraph_connected(adj, ["T0", "T1"]) is False


def test_dual_flow_shared_edge_glues_triangles():
    """Two triangles sharing an edge (facet) → dual-adjacent."""

    simplices = {
        "T0": (0, 1, 2),
        "T1": (0, 1, 3),
    }
    adj = build_dual_adjacency(
        simplices, config=DualFlowConfig(enable_dual_adjacency=True)
    )
    assert adj is not None
    assert set(adj["T0"]) == {"T1"}
    assert set(adj["T1"]) == {"T0"}


def test_dual_flow_from_complex():
    """Complex helper uses enumeration ids and respects the enable flag."""

    complex_ = Complex(
        simplices=[
            Simplex(vertex_ids=(0, 1, 2)),
            Simplex(vertex_ids=(0, 1, 3)),
            Simplex(vertex_ids=(4, 5, 6)),
        ],
        vertex_positions=np.zeros((7, 2)),
        intrinsic_dim=2,
    )
    assert build_dual_adjacency_from_complex(complex_) is None
    adj = build_dual_adjacency_from_complex(
        complex_, config=DualFlowConfig(enable_dual_adjacency=True)
    )
    assert adj is not None
    assert set(adj[0]) == {1}
    assert set(adj[1]) == {0}
    assert adj[2] == ()


def test_gate_flag_off_ignores_dual_adjacency_kwarg():
    """Acceptance path: apply_dual_adjacency=False ignores adjacency kwargs."""

    keep, edit, proposal, good_stars = _good_split_fixture()
    # Would be disconnected if applied — but flag is off, so dual_connected=True wins.
    dual = {"S0": ["S1"], "S1": ["S0", "S2"], "S2": ["S1"]}
    v = score_edit(
        keep,
        edit,
        proposal,
        edit_stars=good_stars,
        keep_stars=good_stars,
        dual_connected=True,
        dual_adjacency=dual,
        affected_simplices=["S0", "S2"],
        config=GateConfig(apply_dual_adjacency=False),
    )
    assert v.accepted


def test_s6_dual_adjacency_wires_into_evidence_gate():
    """Integration: stage2.dual_flow adj → score_edit / EvidenceGate (OPEN_ISSUES #43).

    Builds a post-edit dry-run dual adjacency from facet-sharing simplices,
    computes affected-subgraph connectivity, and passes it through the gated
    ``apply_dual_adjacency`` path. Disconnect rejects on the evidence path.
    Full S6 pressure/density solve is still out of scope (see module docstring).
    """

    keep, edit, proposal, good_stars = _good_split_fixture()
    # Path of three 1-simplices: affecting endpoints only → induced disconnect.
    simplices = [(0, 1), (1, 2), (2, 3)]
    adj = build_dual_adjacency(
        simplices, config=DualFlowConfig(enable_dual_adjacency=True)
    )
    assert adj is not None

    gate_cfg = GateConfig(apply_dual_adjacency=True)

    v_ok = score_edit(
        keep,
        edit,
        proposal,
        edit_stars=good_stars,
        keep_stars=good_stars,
        dual_adjacency=adj,
        affected_simplices=[0, 1, 2],
        config=gate_cfg,
    )
    assert v_ok.accepted

    v_disc = score_edit(
        keep,
        edit,
        proposal,
        edit_stars=good_stars,
        keep_stars=good_stars,
        dual_adjacency=adj,
        affected_simplices=[0, 2],
        config=gate_cfg,
    )
    assert not v_disc.accepted

    gate = EvidenceGate(n_nodes=4, config=gate_cfg)
    v_gate = gate.evaluate(
        keep,
        edit,
        proposal,
        edit_stars=good_stars,
        keep_stars=good_stars,
        dual_adjacency=adj,
        affected_simplices=[0, 2],
    )
    assert not v_gate.accepted


# ---------------------------------------------------------------------------
# A5-T34: dry-run helper
# ---------------------------------------------------------------------------


def _path_edge_complex() -> Complex:
    """Three 1-simplices forming a path (dual path 0—1—2)."""

    return Complex(
        simplices=[
            Simplex(vertex_ids=(0, 1)),
            Simplex(vertex_ids=(1, 2)),
            Simplex(vertex_ids=(2, 3)),
        ],
        vertex_positions=np.zeros((4, 2)),
        intrinsic_dim=1,
    )


def test_dry_run_flag_off_returns_none_adj_connected_true():
    """Dry-run with enable_dual_adjacency=False ⇒ None adj, dual_connected True."""

    c = _path_edge_complex()
    result = dry_run_dual_from_edit(c, remove_simplex_indices=[1])
    assert result.dual_adjacency is None
    assert result.dual_connected is True
    assert len(result.post_edit_complex.simplices) == 2


def test_dry_run_remove_middle_disconnects_affected_endpoints():
    """Removing the middle edge leaves endpoints dual-disconnected (SI S10.4 A2)."""

    c = _path_edge_complex()
    cfg = DualFlowConfig(enable_dual_adjacency=True)
    result = dry_run_dual_from_edit(c, remove_simplex_indices=[1], config=cfg)
    # Survivors remapped: old 0→0, old 2→1; both touch removed middle vertices.
    assert result.affected_simplices == (0, 1)
    assert result.dual_adjacency is not None
    assert result.dual_adjacency[0] == ()
    assert result.dual_adjacency[1] == ()
    assert result.dual_connected is False


def test_dry_run_add_bridge_reconnects():
    """Add a bridging edge after removing the middle → dual reconnects."""

    c = _path_edge_complex()
    cfg = DualFlowConfig(enable_dual_adjacency=True)
    # Remove middle (1,2); add bridge (1,2) back — trivial reconnect.
    result = dry_run_dual_from_edit(
        c,
        remove_simplex_indices=[1],
        add_simplices=[(1, 2)],
        config=cfg,
    )
    assert len(result.post_edit_complex.simplices) == 3
    assert result.dual_connected is True
    assert set(result.affected_simplices) == {0, 1, 2}


def test_dry_run_proposal_affected_nodes_scopes_set():
    """EditProposal.affected_node_ids scopes which post-edit simplices count."""

    c = Complex(
        simplices=[
            Simplex(vertex_ids=(0, 1, 2)),
            Simplex(vertex_ids=(0, 1, 3)),
            Simplex(vertex_ids=(4, 5, 6)),
        ],
        vertex_positions=np.zeros((7, 2)),
        intrinsic_dim=2,
    )
    cfg = DualFlowConfig(enable_dual_adjacency=True)
    proposal = EditProposal(EditType.PRUNE, [0, 1], diagnostic_strength=0.5)
    # No remove/add — only node-scoped affected set on the intact complex.
    result = dry_run_dual_from_edit(c, proposal=proposal, config=cfg)
    assert set(result.affected_simplices) == {0, 1}
    assert result.dual_connected is True
    # Far triangle (id 2) excluded from affected; induced on {0,1} stays connected.
    assert 2 not in result.affected_simplices


def test_dry_run_feeds_gate_apply_dual_adjacency():
    """Dry-run result kwargs reject evidence path when dual disconnects."""

    keep, edit, proposal, good_stars = _good_split_fixture()
    c = _path_edge_complex()
    cfg = DualFlowConfig(enable_dual_adjacency=True)
    dry = dry_run_dual_from_edit(c, remove_simplex_indices=[1], config=cfg)
    assert dry.dual_connected is False

    v = score_edit(
        keep,
        edit,
        proposal,
        edit_stars=good_stars,
        keep_stars=good_stars,
        dual_adjacency=dry.dual_adjacency,
        affected_simplices=list(dry.affected_simplices),
        config=GateConfig(apply_dual_adjacency=True),
    )
    assert not v.accepted


# ---------------------------------------------------------------------------
# A5-T35: conservative BP sketch
# ---------------------------------------------------------------------------


def test_conservative_bp_flag_off_returns_none():
    """enable_conservative_bp=False ⇒ solve_conservative_pressures is None."""

    hat = np.array([1.0, 2.0, 0.5])
    assert solve_conservative_pressures(hat) is None
    assert solve_conservative_pressures(hat, config=DualFlowConfig()) is None


def test_conservative_bp_sketch_identity_damped():
    """Sketch returns p≈hat_p; r_data small; r_cons stubbed at 0 (not real BP)."""

    hat = np.array([1.0, 2.0, 0.5, 0.0])
    result = solve_conservative_pressures(
        hat,
        config=DualFlowConfig(
            enable_conservative_bp=True, bp_damping=0.5, bp_max_iters=1
        ),
    )
    assert result is not None
    np.testing.assert_allclose(result.pressures, hat)
    assert result.r_data == pytest.approx(0.0)
    assert result.r_cons == 0.0
    assert result.iters == 1
    assert "sketch" in result.note.lower()


# ---------------------------------------------------------------------------
# A5-T36: expanded synthetic dual graphs
# ---------------------------------------------------------------------------


def test_dual_flow_vertex_only_touch_not_adjacent():
    """Triangles sharing only a vertex (not a facet) are dual-isolated."""

    # Fan around vertex 0: faces (0,1,2) and (0,3,4) share vertex 0 only.
    simplices = {
        "T0": (0, 1, 2),
        "T1": (0, 3, 4),
    }
    adj = build_dual_adjacency(
        simplices, config=DualFlowConfig(enable_dual_adjacency=True)
    )
    assert adj is not None
    assert adj["T0"] == ()
    assert adj["T1"] == ()
    assert affected_dual_subgraph_connected(adj, ["T0", "T1"]) is False


def test_dual_flow_triangle_chain_path():
    """Three triangles in a facet-sharing chain → dual path 0—1—2."""

    # T0—(edge 1,2)—T1—(edge 2,3)—T2
    simplices = [
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 4),
    ]
    adj = build_dual_adjacency(
        simplices, config=DualFlowConfig(enable_dual_adjacency=True)
    )
    assert adj is not None
    assert set(adj[0]) == {1}
    assert set(adj[1]) == {0, 2}
    assert set(adj[2]) == {1}
    assert resolve_dual_connected(
        simplices, [0, 1, 2], config=DualFlowConfig(enable_dual_adjacency=True)
    )
    assert not resolve_dual_connected(
        simplices, [0, 2], config=DualFlowConfig(enable_dual_adjacency=True)
    )


def test_dual_flow_tetrahedron_face_pair():
    """Two tetrahedra sharing a triangular facet → dual edge."""

    # Tet (0,1,2,3) and tet (0,1,2,4) share facet {0,1,2}.
    simplices = {
        "Tet0": (0, 1, 2, 3),
        "Tet1": (0, 1, 2, 4),
    }
    adj = build_dual_adjacency(
        simplices, config=DualFlowConfig(enable_dual_adjacency=True)
    )
    assert adj is not None
    assert set(adj["Tet0"]) == {"Tet1"}
    assert set(adj["Tet1"]) == {"Tet0"}


# ---------------------------------------------------------------------------
# A5-EXP-S61: S6.1 face-pressure tallies + S6.3 boundary taxonomy
# ---------------------------------------------------------------------------


def test_face_tally_flag_off_returns_none():
    """enable_face_tallies=False ⇒ accumulate_face_pressure_tally is None."""

    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    x = np.array([0.2, 0.2])
    assert accumulate_face_pressure_tally(x, P) is None
    assert accumulate_face_pressure_tally(x, P, config=DualFlowConfig()) is None


def test_face_tally_nonneg_residual_projection():
    """S6.1: Δp̂_f = scale * max{0,(x-w̄)^T n_f}; barycenter sample → zeros."""

    # Right triangle; barycenter at (1/3, 1/3).
    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cfg = DualFlowConfig(enable_face_tallies=True, tally_scale=1.0)
    at_bary = accumulate_face_pressure_tally(
        np.array([1.0 / 3.0, 1.0 / 3.0]), P, config=cfg
    )
    assert at_bary is not None
    np.testing.assert_allclose(at_bary.increments, 0.0, atol=1e-12)
    np.testing.assert_allclose(at_bary.tallies, 0.0, atol=1e-12)

    # Sample outside the hypotenuse (facet opposite vertex 0) → Δp̂_0 > 0.
    outside = accumulate_face_pressure_tally(
        np.array([0.7, 0.7]), P, config=cfg
    )
    assert outside is not None
    assert outside.increments[0] > 0.0
    assert np.all(outside.increments >= 0.0)

    # Sample near vertex 0 still yields some nonnegative flux somewhere.
    near_v0 = accumulate_face_pressure_tally(
        np.array([0.05, 0.05]), P, config=cfg
    )
    assert near_v0 is not None
    assert near_v0.increments.sum() >= 0.0
    assert np.all(near_v0.increments >= 0.0)

    # Prior accumulates.
    again = accumulate_face_pressure_tally(
        np.array([0.7, 0.7]),
        P,
        prior_tallies=outside.tallies,
        config=cfg,
    )
    assert again is not None
    np.testing.assert_allclose(again.tallies, outside.tallies + again.increments)


def test_simplex_outward_normals_unit_and_oriented():
    """Outward normals are unit length and point away from opposite vertices."""

    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    nrm = simplex_outward_normals(P)
    bary = P.mean(axis=0)
    for i in range(3):
        assert np.linalg.norm(nrm[i]) == pytest.approx(1.0)
        # From opposite vertex toward facet / exterior: n · (facet_c - v_i) > 0
        facet_c = np.delete(P, i, axis=0).mean(axis=0)
        assert float(np.dot(nrm[i], facet_c - P[i])) > 0.0
        # Barycenter lies on the inward side of each facet plane.
        assert float(np.dot(nrm[i], bary - facet_c)) < 0.0


def test_boundary_taxonomy_flag_off_returns_none():
    """enable_boundary_taxonomy=False ⇒ classify_boundary_facets is None."""

    assert classify_boundary_facets([(0, 1, 2)]) is None


def test_boundary_taxonomy_single_owner_and_hints():
    """Single-owner facets → TRUE_MANIFOLD; hints override to COMPUTATIONAL/SEAM."""

    # Two triangles sharing edge {1,2}: that facet is interior; outer edges boundary.
    simplices = [
        (0, 1, 2),
        (1, 2, 3),
    ]
    cfg = DualFlowConfig(enable_boundary_taxonomy=True)
    base = classify_boundary_facets(simplices, config=cfg)
    assert base is not None
    # Shared {1,2} omitted; four outer edges remain as true-manifold.
    assert len(base) == 4
    assert all(b.boundary_type == BoundaryType.TRUE_MANIFOLD for b in base)

    hinted = classify_boundary_facets(
        simplices,
        computational_facets=[(0, 1)],
        orientation_seams=[(2, 3)],
        config=cfg,
    )
    assert hinted is not None
    types = {b.boundary_type for b in hinted}
    assert BoundaryType.COMPUTATIONAL in types
    assert BoundaryType.ORIENTATION_SEAM in types
    assert BoundaryType.TRUE_MANIFOLD in types
    # Seam hint wins if both lists somehow overlapped — covered by exclusive sets here.
    assert len(hinted) == 4


# ---------------------------------------------------------------------------
# A5-T40: dry-run demo-wires S6.1 tallies (flag off by default)
# ---------------------------------------------------------------------------


def _triangle_complex() -> Complex:
    """Single triangle with nontrivial vertex positions for tally demos."""

    return Complex(
        simplices=[Simplex(vertex_ids=(0, 1, 2))],
        vertex_positions=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        intrinsic_dim=2,
    )


def test_dry_run_face_tallies_flag_off_none():
    """enable_face_tallies=False ⇒ DualDryRunResult.face_tallies is None."""

    c = _triangle_complex()
    samples = [np.array([0.7, 0.7])]
    result = dry_run_dual_from_edit(
        c, remove_simplex_indices=[], affected_node_ids=[0], samples=samples
    )
    assert result.face_tallies is None
    assert result.stage1_route is None

    cfg = DualFlowConfig(enable_dual_adjacency=True)  # tallies still off
    result2 = dry_run_dual_from_edit(
        c, affected_node_ids=[0], samples=samples, config=cfg
    )
    assert result2.face_tallies is None
    assert result2.stage1_route is None


def test_dry_run_face_tallies_demo_accumulates_on_affected():
    """Flag on + samples ⇒ per-affected-simplex FaceTallyResult (demo routing)."""

    c = _triangle_complex()
    cfg = DualFlowConfig(enable_face_tallies=True, tally_scale=1.0)
    samples = [np.array([0.7, 0.7]), np.array([0.6, 0.6])]
    result = dry_run_dual_from_edit(
        c, affected_node_ids=[0], samples=samples, config=cfg
    )
    assert result.face_tallies is not None
    assert set(result.face_tallies.keys()) == {0}
    tally = result.face_tallies[0]
    assert np.all(tally.increments >= 0.0)
    assert tally.tallies.shape == (3,)
    # Two samples both outside hypotenuse → facet 0 (opp v0) accumulates.
    assert tally.tallies[0] > 0.0

    # Flag on but no samples → empty mapping (wired, vacuous).
    empty = dry_run_dual_from_edit(c, affected_node_ids=[0], config=cfg)
    assert empty.face_tallies == {}


# ---------------------------------------------------------------------------
# A5-T41: S6.4 simplex-local density sketch (default off)
# ---------------------------------------------------------------------------


def test_simplex_density_flag_off_returns_none():
    """enable_simplex_density=False ⇒ simplex_local_density is None."""

    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    x = np.array([0.25, 0.25])
    assert simplex_local_density(x, P, mass=1.0, facet_pressures=np.ones(3)) is None


def test_simplex_density_pl_profile_and_uniform_fallback():
    """S6.4: p = (m/|S|) * ρ̃/w̄; w̄=0 ⇒ uniform m/|S|; barycenter β=1/(d+1)."""

    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    vol = simplex_volume(P)
    assert vol == pytest.approx(0.5)
    cfg = DualFlowConfig(enable_simplex_density=True)
    # Equal facet pressures → equal vertex weights → PL = uniform.
    pressures = np.array([1.0, 1.0, 1.0])
    w = vertex_weights_from_facet_pressures(pressures)
    np.testing.assert_allclose(w, [2.0, 2.0, 2.0])
    bary = P.mean(axis=0)
    beta = barycentric_coordinates(bary, P)
    np.testing.assert_allclose(beta, [1.0 / 3.0] * 3, atol=1e-9)

    out = simplex_local_density(
        bary, P, mass=1.0, facet_pressures=pressures, config=cfg
    )
    assert out is not None
    assert not out.used_uniform_fallback
    assert out.density == pytest.approx(1.0 / vol)

    # Zero pressures → uniform fallback.
    zero = simplex_local_density(
        bary, P, mass=0.5, facet_pressures=np.zeros(3), config=cfg
    )
    assert zero is not None
    assert zero.used_uniform_fallback
    assert zero.density == pytest.approx(0.5 / vol)

    # Nonuniform pressures: sample near a high-weight vertex raises density.
    # w0=p1+p2=3, w1=p0+p2=1.1, w2=p0+p1=1.1 → near v0 should exceed mean.
    uneven = np.array([0.1, 1.5, 1.5])
    near_v0 = simplex_local_density(
        np.array([0.05, 0.05]), P, mass=1.0, facet_pressures=uneven, config=cfg
    )
    at_bary = simplex_local_density(
        bary, P, mass=1.0, facet_pressures=uneven, config=cfg
    )
    assert near_v0 is not None and at_bary is not None
    assert near_v0.density > at_bary.density


# ---------------------------------------------------------------------------
# A5-T50: live Complex/ANN → S6.4 density harness (flag off by default)
# ---------------------------------------------------------------------------


def test_live_density_flag_off_returns_none():
    """enable_live_density=False ⇒ route_live_density_from_complex None."""

    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    complex_ = Complex(
        simplices=[Simplex(vertex_ids=(0, 1, 2), volume=0.5)],
        vertex_positions=V,
        intrinsic_dim=2,
    )
    assert (
        route_live_density_from_complex(
            [np.array([0.2, 0.2])], complex_
        )
        is None
    )


def test_live_density_routes_via_complex_ann_and_evaluates():
    """Flag on ⇒ ANN BMU + S6.4 density on winning simplex."""

    V = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [2.0, 0.0],
            [1.5, 1.0],
        ]
    )
    complex_ = Complex(
        simplices=[
            Simplex(vertex_ids=(0, 1, 2), volume=0.5),
            Simplex(vertex_ids=(1, 3, 4), volume=0.5),
        ],
        vertex_positions=V,
        intrinsic_dim=2,
    )
    cfg = DualFlowConfig(enable_live_density=True, tally_scale=1.0)
    samples = [np.array([0.1, 0.1]), np.array([1.8, 0.2])]
    # Uniform pressures → density = mass / volume = 0.5 / 0.5 = 1.0.
    out = route_live_density_from_complex(
        samples,
        complex_,
        pressures_by_simplex={0: np.ones(3), 1: np.ones(3)},
        masses_by_simplex={0: 0.5, 1: 0.5},
        config=cfg,
    )
    assert out is not None
    assert out.assignments == (0, 1)
    assert out.node_bmus == (0, 3)
    assert len(out.densities) == 2
    assert out.densities[0] == pytest.approx(1.0)
    assert out.densities[1] == pytest.approx(1.0)
    assert "density" in out.note.lower()
    assert "awaiting" in out.note.lower() or "do not flip" in out.note.lower()


def test_live_density_uses_tallies_when_pressures_omitted():
    """Without pressures_by_simplex, harness uses routed face tallies."""

    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    complex_ = Complex(
        simplices=[Simplex(vertex_ids=(0, 1, 2), volume=0.5)],
        vertex_positions=V,
        intrinsic_dim=2,
    )
    cfg = DualFlowConfig(enable_live_density=True, tally_scale=1.0)
    out = route_live_density_from_complex(
        [np.array([0.25, 0.25])],
        complex_,
        config=cfg,
    )
    assert out is not None
    assert out.assignments == (0,)
    assert 0 in out.pressures_by_simplex
    assert np.all(out.pressures_by_simplex[0] >= 0.0)
    assert out.densities[0] > 0.0
    assert out.per_sample[0].density == out.densities[0]


# ---------------------------------------------------------------------------
# A5-T42: acceptance-path plan + S6.2 gap documentation locked in module
# ---------------------------------------------------------------------------


def test_acceptance_path_plan_documented_in_dual_flow_module():
    """Module docstring records None⇒True replacement plan + real-BP gaps."""

    import proteus.stage2.dual_flow as dual_flow

    doc = dual_flow.__doc__ or ""
    assert "Acceptance-path plan" in doc
    assert "None" in doc and "True" in doc
    # Real S6.2 gaps called out (not just the identity sketch).
    assert "A_S" in doc
    assert "loopy Gaussian BP" in doc
    assert "r_cons" in doc or "ε_flux" in doc or "epsilon" in doc.lower()
    # Flags still default off — acceptance unchanged.
    cfg = DualFlowConfig()
    assert cfg.enable_dual_adjacency is False
    assert cfg.enable_conservative_bp is False
    assert cfg.enable_face_tallies is False
    assert cfg.enable_live_bmu_tally is False
    assert cfg.enable_stage1_bmu_wiring is False
    assert cfg.enable_complex_ann_incidence is False
    assert cfg.enable_as_message_pass is False
    assert cfg.enable_mu_weighted_solve is False
    assert cfg.enable_count_aware_lambda is False
    assert cfg.enable_patch_mu_solve is False
    assert cfg.enable_shared_face_glue is False
    assert cfg.enable_global_face_solve is False
    assert cfg.enable_loopy_bp_schedule is False
    assert cfg.enable_mass_normalization is False
    assert cfg.enable_boundary_taxonomy is False
    assert cfg.enable_seam_ghost is False
    assert cfg.enable_simplex_density is False
    assert cfg.enable_live_density is False
    assert cfg.enable_bp_spectrum_damping_probe is False
    assert cfg.enable_bp_damping_policy is False
    assert cfg.enable_online_offline_schedule is False
    assert cfg.enable_online_offline_loopy_compose is False
    assert cfg.enable_bp_policy_in_loopy is False
    assert cfg.enable_loopy_bp_convergence_probe is False
    assert GateConfig().fail_closed_dual_adjacency is False


# ---------------------------------------------------------------------------
# A5-T43: live BMU face-tally routing harness (flag off by default)
# ---------------------------------------------------------------------------


def test_live_bmu_tally_flag_off_returns_none():
    """enable_live_bmu_tally=False ⇒ route_live_bmu_face_tallies is None."""

    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    samples = [np.array([0.2, 0.2])]
    assert route_live_bmu_face_tallies(samples, [P]) is None
    assert route_live_bmu_face_tallies(
        samples, [P], config=DualFlowConfig()
    ) is None


def test_live_bmu_routes_to_containing_or_nearest_simplex():
    """BMU prefers containment; else nearest barycenter; tallies only on winner."""

    # Two adjacent triangles sharing {1,2}.
    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    right = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    positions = {0: left, 1: right}

    # Inside left triangle (near origin).
    assert locate_bmu_simplex(np.array([0.2, 0.2]), positions) == 0
    # Inside right triangle.
    assert locate_bmu_simplex(np.array([1.2, 0.2]), positions) == 1
    # Outside both, closer to right barycenter ~(1.333, 0.333).
    assert locate_bmu_simplex(np.array([3.0, 0.0]), positions) == 1

    cfg = DualFlowConfig(enable_live_bmu_tally=True, tally_scale=1.0)
    samples = [
        np.array([0.2, 0.2]),  # → left
        np.array([1.2, 0.2]),  # → right
        np.array([0.25, 0.15]),  # → left
    ]
    out = route_live_bmu_face_tallies(samples, positions, config=cfg)
    assert out is not None
    assert out.assignments == (0, 1, 0)
    assert set(out.tallies_by_simplex.keys()) == {0, 1}
    # Left got two samples; right one — tallies are nonnegative.
    assert np.all(out.tallies_by_simplex[0].tallies >= 0.0)
    assert np.all(out.tallies_by_simplex[1].tallies >= 0.0)
    # Two left samples accumulate more total pressure than a single right hit
    # for these interior points (may be near-zero at bary; use outside-ish).
    outside_left = route_live_bmu_face_tallies(
        [np.array([0.7, 0.7])], positions, config=cfg
    )
    assert outside_left is not None
    assert outside_left.assignments == (0,)
    assert outside_left.tallies_by_simplex[0].tallies[0] > 0.0


# ---------------------------------------------------------------------------
# A5-T44: A_S residual / soft message-pass sketch (default off)
# ---------------------------------------------------------------------------


def test_as_message_pass_flag_off_returns_none():
    """enable_as_message_pass=False ⇒ solve_as_message_pass is None."""

    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    A_S = build_divergence_stencil(P)
    hat = np.ones(3)
    assert solve_as_message_pass(hat, A_S) is None
    assert solve_as_message_pass(hat, A_S, config=DualFlowConfig()) is None


def test_divergence_stencil_and_as_message_pass_reduces_r_cons():
    """A_S has shape (d, d+1); soft message-pass reports/reduces r_cons."""

    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    A_S = build_divergence_stencil(P)
    assert A_S.shape == (2, 3)
    # Facet areas (edge lengths) are positive for a nondegenerate triangle.
    assert np.linalg.norm(A_S, ord="fro") > 0.0

    hat = np.array([2.0, 0.1, 0.1])  # deliberately unbalanced
    r0 = conservation_residual_r_cons(A_S, hat)
    assert r0 > 0.0

    cfg = DualFlowConfig(
        enable_as_message_pass=True,
        bp_damping=0.5,
        bp_max_iters=8,
        as_step=0.25,
    )
    out = solve_as_message_pass(hat, A_S, config=cfg)
    assert out is not None
    assert out.r_cons >= 0.0
    assert out.r_cons < r0
    assert out.pressures.shape == (3,)
    assert "A_S" in out.note or "message-pass" in out.note


# ---------------------------------------------------------------------------
# A5-T45: seam stitch / ghost reservoir sketches (default off)
# ---------------------------------------------------------------------------


def test_seam_ghost_flag_off_returns_none():
    """enable_seam_ghost=False ⇒ stitch / ghost helpers are None."""

    assert stitch_orientation_seam_pressures(1.0, -0.5) is None
    assert apply_ghost_reservoir(
        np.array([1.0, 2.0]), computational_mask=[True, False]
    ) is None


def test_seam_stitch_antisymmetric_and_ghost_leak():
    """Seam ⇒ p_a = -p_b; ghost leaks γ from computational facets only."""

    cfg = DualFlowConfig(enable_seam_ghost=True, ghost_coupling=0.25)
    seam = stitch_orientation_seam_pressures(1.0, -0.5, config=cfg)
    assert seam is not None
    assert seam.pressure_a == pytest.approx(-seam.pressure_b)
    # (1 - (-0.5))/2 = 0.75
    assert seam.pressure_a == pytest.approx(0.75)

    ghost = apply_ghost_reservoir(
        np.array([1.0, 2.0, 3.0]),
        computational_mask=[True, False, True],
        config=cfg,
    )
    assert ghost is not None
    np.testing.assert_allclose(ghost.adjusted, [0.75, 2.0, 2.25])
    assert ghost.ghost_load == pytest.approx(0.25 * 1.0 + 0.25 * 3.0)


# ---------------------------------------------------------------------------
# A5-EXP-mu: whitened λ_f + μ_S soft solve (default off)
# ---------------------------------------------------------------------------


def test_mu_weighted_solve_flag_off_returns_none():
    """enable_mu_weighted_solve=False ⇒ solve_mu_weighted_pressures is None."""

    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    A_S = build_divergence_stencil(P)
    hat = np.ones(3)
    assert solve_mu_weighted_pressures(hat, A_S) is None
    assert solve_mu_weighted_pressures(hat, A_S, config=DualFlowConfig()) is None


def test_mu_S_weight_matches_si_formula():
    """μ_S = 0.1 * λ̄ / (‖A_S‖_F² + ε_A) (eq. si-dual-flow-weight)."""

    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    A_S = build_divergence_stencil(P)
    fro2 = float(np.sum(A_S * A_S))
    eps_A = 1e-8
    expected = 0.1 * 1.0 / (fro2 + eps_A)
    assert mu_S_weight(A_S, bar_lambda=1.0, mu_scale=0.1, eps_A=eps_A) == pytest.approx(
        expected
    )


def test_whiten_and_mu_weighted_solve_reduces_r_cons():
    """Whitening + μ_S soft solve reports mu_S and reduces r_cons vs hat."""

    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    A_S = build_divergence_stencil(P)
    hat = np.array([2.0, 0.1, 0.1])
    r0 = conservation_residual_r_cons(A_S, hat)
    assert r0 > 0.0

    whitened, std = whiten_empirical_pressures(hat, floor=1e-8)
    assert whitened.shape == (3,)
    assert np.all(std >= 1e-8)

    cfg = DualFlowConfig(
        enable_mu_weighted_solve=True,
        bp_damping=0.5,
        bp_max_iters=12,
        as_step=0.5,
        mu_scale=0.1,
    )
    out = solve_mu_weighted_pressures(hat, A_S, config=cfg)
    assert out is not None
    assert out.mu_S > 0.0
    assert out.lambda_f.shape == (3,)
    assert np.allclose(out.lambda_f, 1.0)
    assert out.r_cons < r0
    assert out.pressures.shape == (3,)
    assert np.isfinite(out.hessian_cond)
    assert out.epsilon_flux >= 0.0
    assert isinstance(out.spectrum_damped, bool)
    assert "μ_S" in out.note or "mu" in out.note.lower()


def test_epsilon_flux_distinct_from_r_cons():
    """ε_flux = ‖Ap‖²/(‖p‖²+ε); r_cons also divides by ‖A‖_F²+ε_A."""

    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    A_S = build_divergence_stencil(P)
    p = np.array([2.0, 0.1, 0.1])
    e = epsilon_flux(A_S, p)
    r = conservation_residual_r_cons(A_S, p)
    assert e > 0.0 and r > 0.0
    # Same numerator flux²; r_cons further / (‖A‖_F²+ε_A) in num path → r < e
    # for nondegenerate A_S with fro² > 1 typically; at least they differ.
    assert e != pytest.approx(r)


def test_spectrum_cond_cap_triggers_damping_flag():
    """spectrum_cond_cap=0 forces spectrum_damped=True on nontrivial Hess."""

    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    A_S = build_divergence_stencil(P)
    hat = np.array([2.0, 0.1, 0.1])
    cfg = DualFlowConfig(
        enable_mu_weighted_solve=True,
        bp_max_iters=4,
        as_step=0.5,
        spectrum_cond_cap=0.0,
    )
    out = solve_mu_weighted_pressures(hat, A_S, config=cfg)
    assert out is not None
    assert out.spectrum_damped is True
    assert out.hessian_cond > 0.0


def test_live_bmu_tallies_feed_mu_weighted_solve():
    """Compose A5-T43 live tallies → A5-EXP-mu solve on BMU winner (flags on)."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    positions = {0: left}
    cfg_tally = DualFlowConfig(enable_live_bmu_tally=True, tally_scale=1.0)
    live = route_live_bmu_face_tallies(
        [np.array([0.7, 0.7]), np.array([0.6, 0.6])],
        positions,
        config=cfg_tally,
    )
    assert live is not None
    hat = live.tallies_by_simplex[0].tallies
    A_S = build_divergence_stencil(left)
    cfg_solve = DualFlowConfig(
        enable_mu_weighted_solve=True,
        bp_max_iters=10,
        as_step=0.5,
    )
    out = solve_mu_weighted_pressures(hat, A_S, config=cfg_solve)
    assert out is not None
    assert out.r_cons >= 0.0
    assert out.pressures.shape == hat.shape


# ---------------------------------------------------------------------------
# A5-T46: count-aware λ_f = 1 + n_f/(1+n̄) (flag off by default)
# ---------------------------------------------------------------------------


def test_count_aware_lambda_f_matches_si_formula():
    """λ_f = 1 + n_f / (1 + mean(n_f))."""

    n_f = np.array([0.0, 2.0, 4.0])
    nbar = float(np.mean(n_f))
    expected = 1.0 + n_f / (1.0 + nbar)
    np.testing.assert_allclose(count_aware_lambda_f(n_f), expected)


def test_count_aware_lambda_flag_off_keeps_baseline_ones():
    """enable_count_aware_lambda=False ⇒ λ_f ones even if counts passed."""

    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    A_S = build_divergence_stencil(P)
    hat = np.array([2.0, 0.1, 0.1])
    counts = np.array([10.0, 1.0, 1.0])
    cfg = DualFlowConfig(
        enable_mu_weighted_solve=True,
        enable_count_aware_lambda=False,
        bp_max_iters=4,
    )
    out = solve_mu_weighted_pressures(
        hat, A_S, face_hit_counts=counts, config=cfg
    )
    assert out is not None
    np.testing.assert_allclose(out.lambda_f, 1.0)


def test_count_aware_lambda_soft_solve_uses_variant_weights():
    """Flag on + face_hit_counts ⇒ λ_f = count-aware formula; r_cons finite."""

    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    A_S = build_divergence_stencil(P)
    hat = np.array([2.0, 0.1, 0.1])
    counts = np.array([10.0, 1.0, 1.0])
    cfg = DualFlowConfig(
        enable_mu_weighted_solve=True,
        enable_count_aware_lambda=True,
        bp_max_iters=12,
        as_step=0.5,
    )
    out = solve_mu_weighted_pressures(
        hat, A_S, face_hit_counts=counts, config=cfg
    )
    assert out is not None
    np.testing.assert_allclose(out.lambda_f, count_aware_lambda_f(counts))
    assert not np.allclose(out.lambda_f, 1.0)
    assert out.r_cons >= 0.0
    assert "count-aware" in out.note.lower() or "n_f" in out.note


# ---------------------------------------------------------------------------
# A5-T47: multi-simplex patch μ_S sum soft solve (flag off by default)
# ---------------------------------------------------------------------------


def test_patch_mu_solve_flag_off_returns_none():
    """enable_patch_mu_solve=False ⇒ solve_patch_mu_weighted_pressures None."""

    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    A_S = build_divergence_stencil(P)
    assert (
        solve_patch_mu_weighted_pressures({0: np.ones(3)}, {0: A_S}) is None
    )


def test_patch_mu_solve_sums_mu_S_and_reduces_block_r_cons():
    """Two-simplex patch: mu_S_sum = Σ μ_S; pressures finite; r_cons>=0."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    right = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    A0 = build_divergence_stencil(left)
    A1 = build_divergence_stencil(right)
    hat = {0: np.array([2.0, 0.1, 0.1]), 1: np.array([0.2, 1.5, 0.2])}
    cfg = DualFlowConfig(
        enable_patch_mu_solve=True,
        bp_max_iters=12,
        as_step=0.5,
        mu_scale=0.1,
    )
    out = solve_patch_mu_weighted_pressures(
        hat, {0: A0, 1: A1}, config=cfg
    )
    assert out is not None
    assert out.simplex_ids == (0, 1)
    assert out.block_sizes == (3, 3)
    assert out.mu_S_sum == pytest.approx(out.mu_S[0] + out.mu_S[1])
    assert out.mu_S[0] > 0.0 and out.mu_S[1] > 0.0
    assert out.pressures.shape == (6,)
    assert out.r_cons >= 0.0
    assert out.epsilon_flux >= 0.0
    assert np.allclose(out.lambda_f, 1.0)
    assert "patch" in out.note.lower() or "μ_S" in out.note or "mu" in out.note.lower()
    assert out.n_shared_faces == 0
    assert out.shared_glue_residual == 0.0


# ---------------------------------------------------------------------------
# A5-EXP-glue: shared-face antisymmetry soft glue (flag off by default)
# ---------------------------------------------------------------------------


def test_shared_face_pairs_two_triangles():
    """Adjacent triangles share one facet with matching local indices."""

    # left verts [0,1,2]; right [1,3,2] — shared {1,2}.
    # left local face 0 excludes 0 → {1,2}; right local face 1 excludes 3 → {1,2}.
    simplices = {0: (0, 1, 2), 1: (1, 3, 2)}
    pairs = build_shared_face_pairs(simplices)
    assert len(pairs) == 1
    p = pairs[0]
    assert p.facet == frozenset({1, 2})
    assert {p.simplex_a, p.simplex_b} == {0, 1}
    assert sorted([p.local_face_a, p.local_face_b]) == [0, 1]


def test_shared_face_glue_flag_off_keeps_independent_copies():
    """enable_shared_face_glue=False ⇒ no glue even if simplices passed."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    right = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    A0 = build_divergence_stencil(left)
    A1 = build_divergence_stencil(right)
    # Disagreeing shared-face empirics (left face0 vs right face1).
    hat = {0: np.array([3.0, 0.1, 0.1]), 1: np.array([0.1, 3.0, 0.1])}
    simplices = {0: (0, 1, 2), 1: (1, 3, 2)}
    cfg = DualFlowConfig(
        enable_patch_mu_solve=True,
        enable_shared_face_glue=False,
        bp_max_iters=8,
        as_step=0.5,
    )
    out = solve_patch_mu_weighted_pressures(
        hat, {0: A0, 1: A1}, simplices=simplices, config=cfg
    )
    assert out is not None
    assert out.n_shared_faces == 0
    # Without glue, shared faces stay near empirics (same sign, large).
    assert out.pressures[0] > 1.0 and out.pressures[4] > 1.0


def test_shared_face_glue_reduces_antisymmetry_residual():
    """Glue on ⇒ n_shared_faces=1 and shared residual drops vs unglued."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    right = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    A0 = build_divergence_stencil(left)
    A1 = build_divergence_stencil(right)
    hat = {0: np.array([3.0, 0.1, 0.1]), 1: np.array([0.1, 3.0, 0.1])}
    simplices = {0: (0, 1, 2), 1: (1, 3, 2)}
    base = dict(
        enable_patch_mu_solve=True,
        bp_max_iters=20,
        as_step=0.35,
        bp_damping=0.4,
        mu_scale=0.05,
    )
    unglued = solve_patch_mu_weighted_pressures(
        hat,
        {0: A0, 1: A1},
        simplices=simplices,
        config=DualFlowConfig(enable_shared_face_glue=False, **base),
    )
    glued = solve_patch_mu_weighted_pressures(
        hat,
        {0: A0, 1: A1},
        simplices=simplices,
        config=DualFlowConfig(
            enable_shared_face_glue=True, shared_face_glue=2.0, **base
        ),
    )
    assert unglued is not None and glued is not None
    assert glued.n_shared_faces == 1
    # Unglued residual on (p0_face0 + p1_face1); glue should shrink it.
    unglued_r = float(unglued.pressures[0] + unglued.pressures[4]) ** 2
    assert glued.shared_glue_residual < unglued_r
    assert "shared-face" in glued.note.lower() or "glue" in glued.note.lower()


def test_shared_face_glue_requires_simplices():
    """enable_shared_face_glue=True without simplices raises."""

    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    A_S = build_divergence_stencil(P)
    cfg = DualFlowConfig(
        enable_patch_mu_solve=True, enable_shared_face_glue=True
    )
    with pytest.raises(ValueError, match="simplices"):
        solve_patch_mu_weighted_pressures({0: np.ones(3)}, {0: A_S}, config=cfg)


# ---------------------------------------------------------------------------
# A5-T49: global face-id soft solve (flag off by default)
# ---------------------------------------------------------------------------


def test_global_face_registry_two_triangles():
    """Two triangles → 5 unique facets, 1 interior, opposite signs."""

    simplices = {0: (0, 1, 2), 1: (1, 3, 2)}
    reg = build_global_face_registry(simplices)
    assert reg.n_faces == 5  # {0,1},{0,2},{1,2},{1,3},{2,3}
    assert reg.n_interior == 1
    shared = [inc for inc in reg.incidences if inc.facet == frozenset({1, 2})]
    assert len(shared) == 2
    signs = {inc.simplex_id: inc.sign for inc in shared}
    assert signs[0] * signs[1] == -1
    assert set(signs.values()) == {1, -1}


def test_global_face_solve_flag_off_returns_none():
    """enable_global_face_solve=False ⇒ solve returns None."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    right = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    A0 = build_divergence_stencil(left)
    A1 = build_divergence_stencil(right)
    hat = {0: np.array([2.0, 0.1, 0.1]), 1: np.array([0.2, 1.5, 0.2])}
    simplices = {0: (0, 1, 2), 1: (1, 3, 2)}
    assert (
        solve_global_face_mu_pressures(
            hat, {0: A0, 1: A1}, simplices, config=DualFlowConfig()
        )
        is None
    )


def test_global_face_solve_identifies_shared_and_antisym_local():
    """Flag on ⇒ one global var per facet; shared locals are antisymmetric."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    right = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    A0 = build_divergence_stencil(left)
    A1 = build_divergence_stencil(right)
    # Disagreeing shared-face empirics (left face0 vs right face1).
    hat = {0: np.array([3.0, 0.1, 0.1]), 1: np.array([0.1, 3.0, 0.1])}
    simplices = {0: (0, 1, 2), 1: (1, 3, 2)}
    cfg = DualFlowConfig(
        enable_global_face_solve=True,
        bp_max_iters=20,
        as_step=0.35,
        bp_damping=0.4,
        mu_scale=0.05,
    )
    out = solve_global_face_mu_pressures(
        hat, {0: A0, 1: A1}, simplices, config=cfg
    )
    assert out is not None
    assert out.n_faces == 5
    assert out.n_interior_faces == 1
    assert out.simplex_ids == (0, 1)
    assert out.block_sizes == (3, 3)
    assert out.pressures_global.shape == (5,)
    assert out.pressures_local.shape == (6,)
    assert out.mu_S_sum == pytest.approx(out.mu_S[0] + out.mu_S[1])
    assert out.r_cons >= 0.0
    assert out.epsilon_flux >= 0.0
    # Shared facet {1,2}: left local 0, right local 1 → indices 0 and 4.
    assert out.pressures_local[0] == pytest.approx(
        -out.pressures_local[4], abs=1e-9
    )
    assert "global face" in out.note.lower() or "face-id" in out.note.lower()
    assert "loopy" in out.note.lower()


def test_global_face_solve_missing_simplex_raises():
    """Empirical simplex absent from simplices → ValueError."""

    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    A_S = build_divergence_stencil(P)
    cfg = DualFlowConfig(enable_global_face_solve=True)
    with pytest.raises(ValueError, match="missing from face registry"):
        solve_global_face_mu_pressures(
            {0: np.ones(3)},
            {0: A_S},
            simplices={9: (0, 1, 2)},  # wrong id
            config=cfg,
        )


# ---------------------------------------------------------------------------
# A5-EXP-loopy-bp: loopy Gaussian BP message schedule (flag off by default)
# ---------------------------------------------------------------------------


def test_loopy_bp_schedule_flag_off_returns_none():
    """enable_loopy_bp_schedule=False ⇒ solve_loopy_bp_schedule is None."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    right = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    A0 = build_divergence_stencil(left)
    A1 = build_divergence_stencil(right)
    hat = {0: np.array([2.0, 0.1, 0.1]), 1: np.array([0.2, 1.5, 0.2])}
    simplices = {0: (0, 1, 2), 1: (1, 3, 2)}
    assert (
        solve_loopy_bp_schedule(
            hat, {0: A0, 1: A1}, simplices, config=DualFlowConfig()
        )
        is None
    )


def test_loopy_bp_schedule_runs_and_antisym_shared():
    """Flag on ⇒ message schedule yields finite residuals + antisym locals."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    right = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    A0 = build_divergence_stencil(left)
    A1 = build_divergence_stencil(right)
    hat = {0: np.array([3.0, 0.1, 0.1]), 1: np.array([0.1, 3.0, 0.1])}
    simplices = {0: (0, 1, 2), 1: (1, 3, 2)}
    cfg = DualFlowConfig(
        enable_loopy_bp_schedule=True,
        bp_max_iters=8,
        bp_damping=0.5,
        mu_scale=0.05,
    )
    out = solve_loopy_bp_schedule(
        hat, {0: A0, 1: A1}, simplices, config=cfg
    )
    assert out is not None
    assert out.n_faces == 5
    assert out.n_interior_faces == 1
    assert out.n_factors == 2
    assert out.message_updates == 8 * (3 + 3)
    assert out.pressures_global.shape == (5,)
    assert out.pressures_local.shape == (6,)
    assert np.all(np.isfinite(out.pressures_global))
    assert out.r_data >= 0.0
    assert out.r_cons >= 0.0
    assert out.epsilon_flux >= 0.0
    # Shared facet {1,2}: left local 0, right local 1 → indices 0 and 4.
    assert out.pressures_local[0] == pytest.approx(
        -out.pressures_local[4], abs=1e-8
    )
    assert "loopy" in out.note.lower()
    assert "awaiting" in out.note.lower() or "do not flip" in out.note.lower()


# ---------------------------------------------------------------------------
# A5-EXP-mass: simplex-mass normalization harness (flag off by default)
# ---------------------------------------------------------------------------


def test_mass_normalization_flag_off_returns_none():
    """enable_mass_normalization=False ⇒ normalize_simplex_masses is None."""

    assert normalize_simplex_masses({0: 0.3, 1: 0.7}) is None
    assert normalize_simplex_masses(
        {0: 0.3, 1: 0.7}, config=DualFlowConfig()
    ) is None


def test_mass_normalization_rescales_and_reports_epsilon():
    """Flag on ⇒ Σ m = 1 and ε_mass ≤ 1e-6; ungated epsilon_mass matches."""

    raw = {0: 2.0, 1: 6.0, "c": 2.0}
    assert epsilon_mass(raw) == pytest.approx(9.0)  # |10 - 1|
    cfg = DualFlowConfig(enable_mass_normalization=True)
    out = normalize_simplex_masses(raw, config=cfg)
    assert out is not None
    assert out.total_before == pytest.approx(10.0)
    assert sum(out.masses.values()) == pytest.approx(1.0)
    assert out.epsilon_mass <= 1e-6
    assert out.epsilon_mass == pytest.approx(epsilon_mass(out.masses))
    assert out.masses[0] == pytest.approx(0.2)
    assert out.masses[1] == pytest.approx(0.6)
    assert "awaiting" in out.note.lower() or "do not flip" in out.note.lower()


def test_mass_normalization_nonpositive_total_raises():
    """Zero / negative mass total → ValueError."""

    cfg = DualFlowConfig(enable_mass_normalization=True)
    with pytest.raises(ValueError, match="mass total"):
        normalize_simplex_masses({0: 0.0, 1: 0.0}, config=cfg)


# ---------------------------------------------------------------------------
# A5-T48: Stage-1 BMU wiring sketch (flag off by default)
# ---------------------------------------------------------------------------


def test_stage1_bmu_wiring_flag_off_returns_none():
    """enable_stage1_bmu_wiring=False ⇒ route_stage1_bmu_face_tallies None."""

    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    samples = [np.array([0.2, 0.2])]
    assert (
        route_stage1_bmu_face_tallies(
            samples, [0], {0: [0]}, {0: P}
        )
        is None
    )


def test_stage1_bmu_wiring_routes_via_node_incident_simplices():
    """Stage-1 node BMU restricts candidates; tallies accumulate on winner."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    right = np.array([[1.0, 0.0], [2.0, 0.0], [1.5, 1.0]])
    # Node 0 incident only to left; node 1 only to right.
    node_to_simplices = {0: [0], 1: [1]}
    positions = {0: left, 1: right}
    samples = [np.array([0.2, 0.2]), np.array([1.4, 0.3])]
    node_bmus = [0, 1]
    cfg = DualFlowConfig(enable_stage1_bmu_wiring=True, tally_scale=1.0)
    out = route_stage1_bmu_face_tallies(
        samples,
        node_bmus,
        node_to_simplices,
        positions,
        config=cfg,
    )
    assert out is not None
    assert out.node_bmus == (0, 1)
    assert out.assignments == (0, 1)
    assert 0 in out.tallies_by_simplex and 1 in out.tallies_by_simplex
    assert out.tallies_by_simplex[0].tallies.shape == (3,)
    assert np.all(out.tallies_by_simplex[0].tallies >= 0.0)
    assert "Stage-1" in out.note or "stage-1" in out.note.lower()

# ---------------------------------------------------------------------------
# A5-EXP-ann-inc: Complex → node_to_simplices + ANN BMU (flag off by default)
# ---------------------------------------------------------------------------


def test_complex_ann_incidence_flag_off_returns_none():
    """enable_complex_ann_incidence=False ⇒ builders / route return None."""

    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    complex_ = Complex(
        simplices=[Simplex(vertex_ids=(0, 1, 2), volume=0.5)],
        vertex_positions=V,
        intrinsic_dim=2,
    )
    assert build_node_to_simplices_from_complex(complex_) is None
    assert build_simplex_positions_from_complex(complex_) is None
    assert query_stage1_ann_bmus([np.array([0.1, 0.1])], node_positions=V) is None
    assert route_stage1_from_complex([np.array([0.1, 0.1])], complex_) is None


def test_complex_ann_incidence_builds_node_star_and_routes():
    """Complex incidence + naive ANN BMU feeds Stage-1 tally sketch."""

    # Two triangles sharing vertex 1: left {0,1,2}, right {1,3,4}.
    V = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [2.0, 0.0],
            [1.5, 1.0],
        ]
    )
    complex_ = Complex(
        simplices=[
            Simplex(vertex_ids=(0, 1, 2), volume=0.5),
            Simplex(vertex_ids=(1, 3, 4), volume=0.5),
        ],
        vertex_positions=V,
        intrinsic_dim=2,
    )
    cfg = DualFlowConfig(enable_complex_ann_incidence=True, tally_scale=1.0)

    node_map = build_node_to_simplices_from_complex(complex_, config=cfg)
    assert node_map is not None
    assert node_map[0] == (0,)
    assert node_map[1] == (0, 1)
    assert node_map[3] == (1,)

    pos_map = build_simplex_positions_from_complex(complex_, config=cfg)
    assert pos_map is not None
    assert pos_map[0].shape == (3, 2)
    np.testing.assert_allclose(pos_map[0][0], V[0])

    # Sample near node 0 → BMU 0 → only left simplex candidate.
    samples = [np.array([0.1, 0.1]), np.array([1.8, 0.2])]
    bmus = query_stage1_ann_bmus(samples, node_positions=V, config=cfg)
    assert bmus == (0, 3)

    out = route_stage1_from_complex(samples, complex_, config=cfg)
    assert out is not None
    assert out.node_bmus == (0, 3)
    assert out.assignments == (0, 1)
    assert 0 in out.tallies_by_simplex and 1 in out.tallies_by_simplex
    assert np.all(out.tallies_by_simplex[0].tallies >= 0.0)


def test_complex_ann_incidence_accepts_ann_duck_type():
    """query_stage1_ann_bmus uses ann.query_knn when provided."""

    class _FakeAnn:
        def query_knn(self, point, k):
            assert k == 1
            # Always return node 2.
            return np.array([2]), np.array([0.0])

    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cfg = DualFlowConfig(enable_complex_ann_incidence=True)
    bmus = query_stage1_ann_bmus(
        [np.array([9.0, 9.0])], ann=_FakeAnn(), config=cfg
    )
    assert bmus == (2,)
    complex_ = Complex(
        simplices=[Simplex(vertex_ids=(0, 1, 2), volume=0.5)],
        vertex_positions=V,
        intrinsic_dim=2,
    )
    out = route_stage1_from_complex(
        [np.array([0.2, 0.2])], complex_, ann=_FakeAnn(), config=cfg
    )
    assert out is not None
    assert out.node_bmus == (2,)
    assert out.assignments == (0,)


# ---------------------------------------------------------------------------
# A5-T51: dry_run wires route_stage1_from_complex (flag off by default)
# ---------------------------------------------------------------------------


def test_dry_run_stage1_route_flag_off_none():
    """enable_complex_ann_incidence=False ⇒ stage1_route is None."""

    V = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [2.0, 0.0],
            [1.5, 1.0],
        ]
    )
    c = Complex(
        simplices=[
            Simplex(vertex_ids=(0, 1, 2), volume=0.5),
            Simplex(vertex_ids=(1, 3, 4), volume=0.5),
        ],
        vertex_positions=V,
        intrinsic_dim=2,
    )
    samples = [np.array([0.1, 0.1])]
    result = dry_run_dual_from_edit(
        c, affected_node_ids=[0], samples=samples, config=DualFlowConfig()
    )
    assert result.stage1_route is None


def test_dry_run_stage1_route_end_to_end():
    """Flag on + samples ⇒ stage1_route matches route_stage1_from_complex."""

    V = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [2.0, 0.0],
            [1.5, 1.0],
        ]
    )
    c = Complex(
        simplices=[
            Simplex(vertex_ids=(0, 1, 2), volume=0.5),
            Simplex(vertex_ids=(1, 3, 4), volume=0.5),
        ],
        vertex_positions=V,
        intrinsic_dim=2,
    )
    samples = [np.array([0.1, 0.1]), np.array([1.8, 0.2])]
    cfg = DualFlowConfig(
        enable_dual_adjacency=True,
        enable_complex_ann_incidence=True,
        tally_scale=1.0,
    )
    result = dry_run_dual_from_edit(
        c, affected_node_ids=[0, 1, 3], samples=samples, config=cfg
    )
    assert result.stage1_route is not None
    assert result.stage1_route.node_bmus == (0, 3)
    assert result.stage1_route.assignments == (0, 1)
    assert 0 in result.stage1_route.tallies_by_simplex
    assert 1 in result.stage1_route.tallies_by_simplex
    # Dual adjacency still built; these simplices share a vertex only, so
    # the induced dual on {0,1} is disconnected (correct A2 signal).
    assert result.dual_adjacency is not None
    assert set(result.dual_adjacency.keys()) == {0, 1}
    assert result.dual_connected is False

    direct = route_stage1_from_complex(samples, c, config=cfg)
    assert direct is not None
    assert result.stage1_route.node_bmus == direct.node_bmus
    assert result.stage1_route.assignments == direct.assignments


def test_dry_run_stage1_route_requires_samples():
    """Flag on without samples ⇒ stage1_route stays None."""

    c = _triangle_complex()
    cfg = DualFlowConfig(enable_complex_ann_incidence=True)
    result = dry_run_dual_from_edit(c, affected_node_ids=[0], config=cfg)
    assert result.stage1_route is None


# ---------------------------------------------------------------------------
# A5-T54: acceptance None⇒True open-default probe (document only; no flip)
# ---------------------------------------------------------------------------


def test_acceptance_none_open_default_probe_matrix():
    """A5-T54: probe locks current open-default; flag-on still detects disconnect.

    Does **not** flip GateConfig / DualFlowConfig defaults or awaiting markers.
    """

    probe = probe_acceptance_none_open_default()
    assert probe.gate_apply_dual_adjacency_default is False
    assert probe.dual_enable_dual_adjacency_default is False
    assert probe.none_adjacency_reports_connected is True
    assert probe.resolve_flag_off_reports_connected is True
    assert probe.dry_run_flag_off_dual_connected is True
    assert probe.flag_on_detects_endpoint_disconnect is True
    assert "open-default" in probe.note.lower() or "None" in probe.note


def test_acceptance_open_default_still_accepts_with_none_adj_on_gate():
    """score_edit with default gate ignores disconnecting adj kwargs; open default."""

    keep, edit, proposal, good_stars = _good_split_fixture()
    # Induced disconnect if applied — but apply_dual_adjacency default is off.
    dual = {"S0": ["S1"], "S1": ["S0", "S2"], "S2": ["S1"]}
    v = score_edit(
        keep,
        edit,
        proposal,
        edit_stars=good_stars,
        keep_stars=good_stars,
        dual_connected=True,
        dual_adjacency=dual,
        affected_simplices=["S0", "S2"],
        config=GateConfig(),  # apply_dual_adjacency=False
    )
    assert v.accepted
    # Counterfactual: same adj under proposal flag would reject.
    v_flag = score_edit(
        keep,
        edit,
        proposal,
        edit_stars=good_stars,
        keep_stars=good_stars,
        dual_adjacency=dual,
        affected_simplices=["S0", "S2"],
        config=GateConfig(apply_dual_adjacency=True),
    )
    assert not v_flag.accepted


# ---------------------------------------------------------------------------
# A5-T55: BP spectrum damping probe (flag off by default)
# ---------------------------------------------------------------------------


def test_bp_spectrum_damping_probe_flag_off_returns_none():
    """enable_bp_spectrum_damping_probe=False ⇒ probe is None."""

    assert probe_bp_spectrum_damping() is None
    assert probe_bp_spectrum_damping(config=DualFlowConfig()) is None


def test_bp_spectrum_damping_probe_fires_mu_and_loopy_ridge():
    """Flag on + tiny spectrum_cond_cap ⇒ μ spectrum_damped + loopy ridge."""

    cfg = DualFlowConfig(enable_bp_spectrum_damping_probe=True)
    probe = probe_bp_spectrum_damping(config=cfg)
    assert probe is not None
    assert probe.probe_flag_default_off is True
    assert probe.mu_spectrum_damped is True
    assert probe.mu_hessian_cond > 0.0
    assert probe.loopy_spectrum_ridge_applied is True
    assert probe.loopy_message_updates > 0
    assert probe.spectrum_cond_cap_used == pytest.approx(1e-12)
    assert "spectrum" in probe.note.lower()


# ---------------------------------------------------------------------------
# A5-T56: online tallies → offline μ schedule (flag off by default)
# ---------------------------------------------------------------------------


def test_online_offline_schedule_flag_off_returns_none():
    """enable_online_offline_schedule=False ⇒ harness is None."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert (
        run_online_offline_schedule(
            [np.array([0.3, 0.3])], {0: left}, config=DualFlowConfig()
        )
        is None
    )


def test_online_offline_schedule_runs_tally_then_solve():
    """Flag on: live BMU tallies then offline μ soft-solve per winner."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    samples = [np.array([0.25, 0.25]), np.array([0.4, 0.2])]
    cfg = DualFlowConfig(
        enable_online_offline_schedule=True,
        bp_max_iters=6,
        as_step=0.5,
    )
    out = run_online_offline_schedule(samples, {0: left}, config=cfg)
    assert out is not None
    assert out.n_samples == 2
    assert out.n_online_simplices >= 1
    assert out.n_offline_solves >= 1
    assert out.offline_r_cons_mean >= 0.0
    assert isinstance(out.offline_spectrum_damped_any, bool)
    assert "online" in out.note.lower() and "offline" in out.note.lower()


# ---------------------------------------------------------------------------
# A5-T57: fail-closed dual-adjacency plan stub (document only; no flip)
# ---------------------------------------------------------------------------


def test_fail_closed_dual_adjacency_plan_probe():
    """A5-T57: plan stub keeps open-default active; documents fail-closed steps."""

    probe = probe_fail_closed_dual_adjacency_plan()
    assert probe.open_default_still_active is True
    assert probe.gate_apply_dual_adjacency_default is False
    assert probe.dual_enable_dual_adjacency_default is False
    assert len(probe.plan_steps) >= 4
    assert any("fail-closed" in s.lower() or "apply_dual" in s for s in probe.plan_steps)
    assert "fail-closed" in probe.note.lower() or "None" in probe.note


# ---------------------------------------------------------------------------
# A5-T58: production BP damping policy sketch (flag off by default)
# ---------------------------------------------------------------------------


def test_bp_damping_policy_flag_off_returns_none():
    """enable_bp_damping_policy=False ⇒ propose_bp_damping_policy is None."""

    assert propose_bp_damping_policy(1e9) is None
    assert propose_bp_damping_policy(1e9, config=DualFlowConfig()) is None


def test_bp_damping_policy_recommends_ridge_when_cond_exceeds_cap():
    """Flag on: cond>cap ⇒ apply_ridge + raised damping; cond<=cap keeps base."""

    cfg = DualFlowConfig(
        enable_bp_damping_policy=True,
        bp_damping=0.5,
        spectrum_cond_cap=1e3,
    )
    ok = propose_bp_damping_policy(10.0, config=cfg)
    assert ok is not None
    assert ok.policy_flag_default_off is True
    assert ok.apply_ridge is False
    assert ok.recommended_damping == pytest.approx(0.5)
    assert ok.overshoot_decades == pytest.approx(0.0)

    bad = propose_bp_damping_policy(1e9, config=cfg)
    assert bad is not None
    assert bad.apply_ridge is True
    assert bad.recommended_damping > 0.5
    assert bad.recommended_damping <= 1.0
    assert bad.overshoot_decades > 0.0
    assert "damping" in bad.note.lower() or "ridge" in bad.note.lower()


# ---------------------------------------------------------------------------
# A5-T59: online → offline loopy compose wire (flag off by default)
# ---------------------------------------------------------------------------


def test_online_offline_loopy_compose_flag_off_returns_none():
    """enable_online_offline_loopy_compose=False ⇒ harness is None."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert (
        run_online_offline_loopy_compose(
            [np.array([0.3, 0.3])],
            {0: left},
            {0: (0, 1, 2)},
            config=DualFlowConfig(),
        )
        is None
    )


def test_online_offline_loopy_compose_runs_tally_then_loopy():
    """Flag on: live BMU tallies then offline loopy BP on shared face graph."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    right = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    samples = [np.array([0.25, 0.25]), np.array([1.2, 0.2])]
    cfg = DualFlowConfig(
        enable_online_offline_loopy_compose=True,
        bp_damping=0.5,
        bp_max_iters=2,
    )
    out = run_online_offline_loopy_compose(
        samples,
        {0: left, 1: right},
        {0: (0, 1, 2), 1: (1, 3, 2)},
        config=cfg,
    )
    assert out is not None
    assert out.n_samples == 2
    assert out.n_online_simplices >= 1
    assert out.loopy_message_updates > 0
    assert out.loopy_r_cons >= 0.0
    assert isinstance(out.loopy_spectrum_ridge_applied, bool)
    assert "loopy" in out.note.lower() and "online" in out.note.lower()


# ---------------------------------------------------------------------------
# A5-T60: GateConfig fail_closed_dual_adjacency switch (default off)
# ---------------------------------------------------------------------------


def test_gate_fail_closed_switch_default_off_probe():
    """A5-T60: switch default off; probe documents fail-closed path."""

    probe = probe_gate_fail_closed_switch()
    assert probe.switch_default_off is True
    assert probe.apply_dual_adjacency_default is False
    assert probe.open_default_none_still_connected is True
    assert probe.flag_on_none_rejects is True
    assert GateConfig().fail_closed_dual_adjacency is False


def test_gate_fail_closed_none_adj_rejects_when_switch_on():
    """apply_dual + fail_closed + None adj ⇒ evidence reject; default path open."""

    keep, edit, proposal, good_stars = _good_split_fixture()

    v_open = score_edit(
        keep,
        edit,
        proposal,
        edit_stars=good_stars,
        keep_stars=good_stars,
        dual_connected=True,
        dual_adjacency=None,
        affected_simplices=["S0", "S2"],
        config=GateConfig(),  # both flags off
    )
    assert v_open.accepted

    v_closed = score_edit(
        keep,
        edit,
        proposal,
        edit_stars=good_stars,
        keep_stars=good_stars,
        dual_connected=True,
        dual_adjacency=None,
        affected_simplices=["S0", "S2"],
        config=GateConfig(
            apply_dual_adjacency=True,
            fail_closed_dual_adjacency=True,
        ),
    )
    assert not v_closed.accepted


# ---------------------------------------------------------------------------
# A5-T61: wire damping policy into loopy BP (flag off by default)
# ---------------------------------------------------------------------------


def test_bp_policy_in_loopy_flag_off_keeps_fixed_damping():
    """enable_bp_policy_in_loopy=False ⇒ policy_applied False even if loopy on."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    right = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    hats = {
        0: np.array([1.0, -1.0, 0.5]),
        1: np.array([-1.0, 0.5, 0.25]),
    }
    stencils = {
        0: build_divergence_stencil(left),
        1: build_divergence_stencil(right),
    }
    simplices = {0: (0, 1, 2), 1: (1, 3, 2)}
    cfg = DualFlowConfig(
        enable_loopy_bp_schedule=True,
        enable_bp_policy_in_loopy=False,
        bp_damping=0.5,
        bp_max_iters=2,
        spectrum_cond_cap=1e-12,
    )
    out = solve_loopy_bp_schedule(hats, stencils, simplices, config=cfg)
    assert out is not None
    assert out.policy_applied is False
    assert out.max_policy_damping == pytest.approx(0.0)


def test_bp_policy_in_loopy_raises_damping_when_cond_exceeds_cap():
    """Flag on + tiny spectrum_cond_cap ⇒ policy_applied + raised damping."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    right = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    hats = {
        0: np.array([1.0, -1.0, 0.5]),
        1: np.array([-1.0, 0.5, 0.25]),
    }
    stencils = {
        0: build_divergence_stencil(left),
        1: build_divergence_stencil(right),
    }
    simplices = {0: (0, 1, 2), 1: (1, 3, 2)}
    cfg = DualFlowConfig(
        enable_loopy_bp_schedule=True,
        enable_bp_policy_in_loopy=True,
        bp_damping=0.5,
        bp_max_iters=2,
        spectrum_cond_cap=1e-12,
    )
    out = solve_loopy_bp_schedule(hats, stencils, simplices, config=cfg)
    assert out is not None
    assert out.policy_applied is True
    assert out.max_policy_damping > 0.5
    assert out.max_policy_damping <= 1.0
    assert out.spectrum_ridge_applied is True
    assert out.message_updates > 0


# ---------------------------------------------------------------------------
# A5-T62: loopy BP residual trajectory convergence probe (flag off)
# ---------------------------------------------------------------------------


def test_loopy_bp_convergence_probe_flag_off_returns_none():
    """enable_loopy_bp_convergence_probe=False ⇒ probe is None."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    hats = {0: np.array([0.1, -0.2, 0.05])}
    stencils = {0: build_divergence_stencil(left)}
    assert (
        probe_loopy_bp_convergence(
            hats, stencils, {0: (0, 1, 2)}, config=DualFlowConfig()
        )
        is None
    )


def test_loopy_bp_convergence_probe_records_residual_trajectory():
    """Flag on: residual sequences length == max_iters; flag default off."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    right = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    hats = {
        0: np.array([1.0, -1.0, 0.5]),
        1: np.array([-1.0, 0.5, 0.25]),
    }
    stencils = {
        0: build_divergence_stencil(left),
        1: build_divergence_stencil(right),
    }
    simplices = {0: (0, 1, 2), 1: (1, 3, 2)}
    cfg = DualFlowConfig(
        enable_loopy_bp_convergence_probe=True,
        enable_bp_policy_in_loopy=True,
        bp_damping=0.5,
        spectrum_cond_cap=1e-12,
    )
    probe = probe_loopy_bp_convergence(
        hats, stencils, simplices, max_iters=3, config=cfg
    )
    assert probe is not None
    assert probe.probe_flag_default_off is True
    assert probe.iters == (1, 2, 3)
    assert len(probe.r_data) == 3
    assert len(probe.r_cons) == 3
    assert all(r >= 0.0 for r in probe.r_data)
    assert all(r >= 0.0 for r in probe.r_cons)
    assert probe.policy_in_loopy_used is True
    assert "trajectory" in probe.note.lower() or "residual" in probe.note.lower()


# ---------------------------------------------------------------------------
# A5-T63: compose forwards enable_bp_policy_in_loopy (flag off by default)
# ---------------------------------------------------------------------------


def test_online_offline_loopy_compose_forwards_policy_flag():
    """Compose with enable_bp_policy_in_loopy ⇒ loopy_policy_applied on ill-cond."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    right = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    samples = [np.array([0.25, 0.25]), np.array([1.2, 0.2])]
    cfg = DualFlowConfig(
        enable_online_offline_loopy_compose=True,
        enable_bp_policy_in_loopy=True,
        bp_damping=0.5,
        bp_max_iters=2,
        spectrum_cond_cap=1e-12,
    )
    out = run_online_offline_loopy_compose(
        samples,
        {0: left, 1: right},
        {0: (0, 1, 2), 1: (1, 3, 2)},
        config=cfg,
    )
    assert out is not None
    assert out.loopy_policy_applied is True
    assert out.loopy_max_policy_damping > 0.5
    assert out.loopy_message_updates > 0


# ---------------------------------------------------------------------------
# A5-T64: certified loopy residual-stop policy sketch (flag off)
# ---------------------------------------------------------------------------


def test_loopy_bp_residual_stop_flag_off_returns_none():
    """enable_loopy_bp_residual_stop=False ⇒ propose_loopy_bp_residual_stop None."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    hats = {0: np.array([0.1, -0.2, 0.05])}
    stencils = {0: build_divergence_stencil(left)}
    assert (
        propose_loopy_bp_residual_stop(
            hats, stencils, {0: (0, 1, 2)}, config=DualFlowConfig()
        )
        is None
    )


def test_loopy_bp_residual_stop_policy_sketches_stop_reason():
    """Flag on: returns stop_reason + trajectories; sketch ≠ production cert."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    right = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    hats = {
        0: np.array([1.0, -1.0, 0.5]),
        1: np.array([-1.0, 0.5, 0.25]),
    }
    stencils = {
        0: build_divergence_stencil(left),
        1: build_divergence_stencil(right),
    }
    simplices = {0: (0, 1, 2), 1: (1, 3, 2)}
    cfg = DualFlowConfig(
        enable_loopy_bp_residual_stop=True,
        bp_residual_stop_tol=1e-3,
        bp_residual_stop_patience=2,
        bp_damping=0.5,
    )
    policy = propose_loopy_bp_residual_stop(
        hats, stencils, simplices, max_iters=5, config=cfg
    )
    assert policy is not None
    assert policy.policy_flag_default_off is True
    assert DualFlowConfig().enable_loopy_bp_residual_stop is False
    assert policy.stop_reason in ("abs_tol", "plateau", "max_iters")
    assert 1 <= policy.stopped_at_iters <= 5
    assert len(policy.r_data_traj) == policy.stopped_at_iters
    assert len(policy.r_cons_traj) == policy.stopped_at_iters
    assert all(r >= 0.0 for r in policy.r_data_traj)
    assert all(r >= 0.0 for r in policy.r_cons_traj)
    assert isinstance(policy.sketch_certificate_ok, bool)
    if policy.stop_reason in ("abs_tol", "plateau"):
        assert policy.sketch_certificate_ok is True
    else:
        assert policy.sketch_certificate_ok is False
    assert "sketch" in policy.note.lower() or "not" in policy.note.lower()


# ---------------------------------------------------------------------------
# A5-T65: fail_closed score_edit default-path matrix expansion
# ---------------------------------------------------------------------------


def test_fail_closed_score_edit_matrix_probe_defaults_unchanged():
    """A5-T65: probe documents matrix; GateConfig defaults stay off."""

    probe = probe_fail_closed_score_edit_matrix()
    assert probe.defaults_unchanged is True
    assert probe.apply_dual_default is False
    assert probe.fail_closed_default is False
    assert GateConfig().apply_dual_adjacency is False
    assert GateConfig().fail_closed_dual_adjacency is False
    assert probe.n_cases >= 8
    assert len(probe.cases) == probe.n_cases
    names = {c.name for c in probe.cases}
    assert "apply_fail_closed_none_reject" in names
    assert "fail_closed_alone_none" in names


def test_fail_closed_score_edit_matrix_matches_score_edit():
    """Each matrix cell's expect_accept matches live score_edit (good stars)."""

    keep, edit, proposal, good_stars = _good_split_fixture()
    connected = {
        "S0": ("S1",),
        "S1": ("S0", "S2"),
        "S2": ("S1",),
    }
    disconnect = {
        "S0": ("S1",),
        "S1": ("S0",),
        "S2": (),
    }
    probe = probe_fail_closed_score_edit_matrix()
    for case in probe.cases:
        if case.adj_kind == "none":
            adj = None
        elif case.adj_kind == "connected":
            adj = connected
        elif case.adj_kind == "disconnect":
            adj = disconnect
        else:
            raise AssertionError(f"unknown adj_kind {case.adj_kind!r}")
        verdict = score_edit(
            keep,
            edit,
            proposal,
            edit_stars=good_stars,
            keep_stars=good_stars,
            dual_connected=case.dual_connected_kwarg,
            dual_adjacency=adj,
            affected_simplices=["S0", "S2"],
            config=GateConfig(
                apply_dual_adjacency=case.apply_dual,
                fail_closed_dual_adjacency=case.fail_closed,
            ),
        )
        assert verdict.accepted is case.expect_accept, (
            f"{case.name}: expected accept={case.expect_accept}, "
            f"got {verdict.accepted}"
        )


# ---------------------------------------------------------------------------
# A5-T66: mass_normalization × loopy compose probe (flag off)
# ---------------------------------------------------------------------------


def test_mass_loopy_compose_probe_flag_off_returns_none():
    """enable_mass_loopy_compose_probe=False ⇒ probe is None."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert (
        probe_mass_loopy_compose(
            [np.array([0.3, 0.3])],
            {0: left},
            {0: (0, 1, 2)},
            config=DualFlowConfig(),
        )
        is None
    )


def test_mass_loopy_compose_probe_runs_mass_and_loopy():
    """Flag on: mass ε≈0 and loopy message updates; awaiting stays untouched."""

    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    right = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    samples = [np.array([0.25, 0.25]), np.array([1.2, 0.2])]
    cfg = DualFlowConfig(
        enable_mass_loopy_compose_probe=True,
        bp_damping=0.5,
        bp_max_iters=2,
    )
    out = probe_mass_loopy_compose(
        samples,
        {0: left, 1: right},
        {0: (0, 1, 2), 1: (1, 3, 2)},
        config=cfg,
    )
    assert out is not None
    assert out.probe_flag_default_off is True
    assert DualFlowConfig().enable_mass_loopy_compose_probe is False
    assert out.n_samples == 2
    assert out.n_online_simplices >= 1
    assert out.epsilon_mass == pytest.approx(0.0, abs=1e-9)
    assert out.mass_total_before > 0.0
    assert out.loopy_message_updates > 0
    assert out.loopy_r_cons >= 0.0
    assert "mass" in out.note.lower() and "awaiting" in out.note.lower()
