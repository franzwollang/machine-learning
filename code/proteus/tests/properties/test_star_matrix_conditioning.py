"""Star-matrix conditioning invariants (SI S10.4 operational runtime form)."""
from __future__ import annotations

import numpy as np

from proteus.evidence import (
    NodeTransition,
    RHO_MIN_DEFAULT,
    bdeu_alpha,
    condition_ratio,
    evaluate_edit,
    f_dm,
    is_evidence_bearing,
    quarantined_nodes,
    score_edit,
    star_incidence_matrix,
)
from proteus.types import EditProposal, EditType


def test_incidence_proxy_matches_si_runtime_definition():
    """K_i^{inc}[j,S] = 1[{i,j} subset S] (SI S10.4 operational runtime matrix)."""

    K = star_incidence_matrix(
        out_edges=[1, 2, 3],
        star_simplices=[[0, 1, 2], [0, 1], [0, 3, 4]],
        node_id=0,
    )
    assert K.shape == (3, 3)
    # triangle {0,1,2}: edges to 1 and 2
    assert list(K[:, 0]) == [1.0, 1.0, 0.0]
    # edge {0,1}: only neighbor 1
    assert list(K[:, 1]) == [1.0, 0.0, 0.0]
    # triangle {0,3,4}: only neighbor 3 is an out-edge; 4 is outside out_edges
    assert list(K[:, 2]) == [0.0, 0.0, 1.0]


def test_conditioning_above_rho_min():
    """sigma_min(K_i^{inc})/sigma_max(K_i^{inc}) >= rho_min for an identifiable
    star (SI S10.4). Distinct simplices with distinct edge supports are well
    conditioned and evidence-bearing under the incidence proxy."""

    # Node 0 sits in two triangles sharing only the apex: edges to {1,2} and
    # {3,4}. The incidence map has orthogonal columns -> perfectly conditioned.
    K = star_incidence_matrix(
        out_edges=[1, 2, 3, 4],
        star_simplices=[[0, 1, 2], [0, 3, 4]],
        node_id=0,
    )
    rho = condition_ratio(K)
    assert rho >= RHO_MIN_DEFAULT
    assert rho == 1.0  # orthogonal columns of equal norm
    assert is_evidence_bearing(K)

    # A lone simplex is trivially identifiable for its single mass.
    K1 = star_incidence_matrix([1, 2], [[0, 1, 2]], 0)
    assert is_evidence_bearing(K1)


def test_underdetermined_star_not_evidence_bearing():
    """S10.4 operational count guard: more incident simplices than outgoing
    outcomes => not evidence-bearing, even when sigma_min/sigma_max looks fine."""

    # 2 outcomes, 3 simplices routing through overlapping edge sets.
    K = star_incidence_matrix(
        out_edges=[1, 2],
        star_simplices=[[0, 1, 2], [0, 1], [0, 2]],  # 3 simplices on 2 edges
        node_id=0,
    )
    assert K.shape == (2, 3)
    # The literal conditioning ratio is deceptively fine ...
    assert condition_ratio(K) >= RHO_MIN_DEFAULT
    # ... but the star is under-determined (n_simplices > n_outcomes).
    assert not is_evidence_bearing(K)


def test_ill_conditioned_stars_quarantined():
    """Stars below rho_min must not contribute to F_DM, and an edit touching any
    ill-conditioned affected star is not evidence-bearing (SI S10.4)."""

    # Two simplices routing through an identical edge set -> duplicate columns,
    # sigma_min = 0, so the masses cannot be told apart from transition counts.
    K_bad = star_incidence_matrix(
        out_edges=[1, 2],
        star_simplices=[[0, 1, 2], [0, 1, 2]],
        node_id=0,
    )
    assert condition_ratio(K_bad) < RHO_MIN_DEFAULT
    assert not is_evidence_bearing(K_bad)

    K_good = star_incidence_matrix([1, 2, 3, 4], [[5, 1, 2], [5, 3, 4]], 5)
    stars = {0: K_bad, 5: K_good}
    assert quarantined_nodes(stars) == {0}

    # Low-level F_DM quarantine primitive (used by the geometry-only path) drops
    # exactly the flagged node's likelihood term.
    a0 = bdeu_alpha(2)
    region = [
        NodeTransition(np.array([30.0, 1.0]), 2, a0, node_id=0),
        NodeTransition(np.array([10.0, 10.0]), 2, a0, node_id=5),
    ]
    only_good = f_dm([region[1]])
    assert abs(f_dm(region, quarantined={0}) - only_good) < 1e-12


def test_gate_rejects_edit_with_ill_conditioned_star():
    """All-or-nothing (SI S10.4): an edit whose F_DM would otherwise be accepted
    must be rejected when *any* affected post-edit star is ill-conditioned --
    partial evidence from the remaining conditioned nodes may not accept it."""

    a0 = bdeu_alpha(2)
    keep = [NodeTransition(np.array([15.0, 16.0]), 2, a0, node_id=0)]
    edit = [NodeTransition(np.array([30.0, 1.0]), 2, a0, node_id=0)]
    proposal = EditProposal(EditType.SPLIT, [0], diagnostic_strength=1.0)

    # With conditioned stars the concentrated edit clears the margin.
    good_stars = {0: star_incidence_matrix([1, 2, 3, 4], [[0, 1, 2], [0, 3, 4]], 0)}
    v_ok = score_edit(keep, edit, proposal, edit_stars=good_stars, keep_stars=good_stars)
    assert v_ok.accepted

    # A single ill-conditioned affected star blocks the evidence acceptance.
    bad_stars = {0: star_incidence_matrix([1, 2], [[0, 1, 2], [0, 1, 2]], 0)}
    v_ill = score_edit(keep, edit, proposal, edit_stars=bad_stars, keep_stars=good_stars)
    assert not v_ill.accepted

    # A disconnected affected dual subgraph likewise blocks it (S10.4).
    v_disc = score_edit(
        keep, edit, proposal, edit_stars=good_stars, keep_stars=good_stars,
        dual_connected=False,
    )
    assert not v_disc.accepted
