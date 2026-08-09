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
  ``enable_live_bmu_tally`` (A5-T43; still not Stage-1 wiring).
* S6.2 ``A_S`` geometry + soft message-pass sketch behind
  ``enable_as_message_pass`` (A5-T44; not loopy Gaussian BP).
* S6.3 boundary taxonomy behind ``enable_boundary_taxonomy``; seam stitch /
  ghost reservoir sketches behind ``enable_seam_ghost`` (A5-T45).
* S6.4 simplex-local PL density *sketch* behind ``enable_simplex_density``
  (default off; does not flip density awaiting tests).

Gaps vs full SI S6 (do **not** flip these elsewhere yet):

* **S6.1** Tally + dry-run + BMU harness exist; Stage-1 live wiring still open.
* **S6.2** Soft ``A_S`` message-pass only — no loopy Gaussian BP / ``μ_S`` /
  whitened ``λ_f``. See module docstring acceptance-path plan (A5-T42).
* **S6.3** Seam/ghost sketches are scalar; no face registry / patch graph.
* **S6.4** Density sketch only; live evaluator / mass normalization open.
* Mass-conservation / density / benchmark ``@awaiting("stage2.dual_flow")``
  (and ``stage2.density``) remain xfail until that producer lands.
* Acceptance path still defaults open when adjacency is ``None`` / flags off.
"""
from __future__ import annotations

import numpy as np
import pytest

from proteus.evidence import (
    EvidenceGate,
    GateConfig,
    affected_dual_subgraph_connected,
    bdeu_alpha,
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
    classify_boundary_facets,
    conservation_residual_r_cons,
    dry_run_dual_from_edit,
    locate_bmu_simplex,
    resolve_dual_connected,
    route_live_bmu_face_tallies,
    simplex_local_density,
    simplex_outward_normals,
    simplex_volume,
    solve_as_message_pass,
    solve_conservative_pressures,
    stitch_orientation_seam_pressures,
    vertex_weights_from_facet_pressures,
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

    cfg = DualFlowConfig(enable_dual_adjacency=True)  # tallies still off
    result2 = dry_run_dual_from_edit(
        c, affected_node_ids=[0], samples=samples, config=cfg
    )
    assert result2.face_tallies is None


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
    assert cfg.enable_as_message_pass is False
    assert cfg.enable_boundary_taxonomy is False
    assert cfg.enable_seam_ghost is False
    assert cfg.enable_simplex_density is False


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
