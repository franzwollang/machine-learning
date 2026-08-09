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
* S6.1 face-pressure tally helper behind ``enable_face_tallies`` (not wired
  into live sample routing).
* S6.3 boundary taxonomy behind ``enable_boundary_taxonomy`` (single-owner
  heuristic + hint sets; not full seam stitching).

Gaps vs full SI S6 (do **not** flip these elsewhere yet):

* **S6.1** Tally helper exists but is not wired into online sample routing.
* **S6.2** No real loopy Gaussian BP / ``A_S p_S`` conservation solve (sketch
  only; ``r_cons`` stubbed at 0).
* **S6.3** Taxonomy stub lacks ghost-reservoir / seam pressure stitching.
* **S6.4** No simplex-local PL density formula.
* Mass-conservation / density / benchmark ``@awaiting("stage2.dual_flow")``
  tests remain xfail until that producer lands.
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
    build_dual_adjacency,
    build_dual_adjacency_from_complex,
    classify_boundary_facets,
    dry_run_dual_from_edit,
    resolve_dual_connected,
    simplex_outward_normals,
    solve_conservative_pressures,
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
