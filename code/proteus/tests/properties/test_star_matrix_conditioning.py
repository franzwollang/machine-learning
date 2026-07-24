"""Star-matrix conditioning invariants (SI S10.4)."""
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
    star_incidence_matrix,
)
from proteus.types import EditProposal, EditType


def test_conditioning_above_rho_min():
    """sigma_min(K_i)/sigma_max(K_i) must be >= rho_min for an identifiable star
    (SI S10.4). A generic simplicial star (distinct simplices spanning distinct
    edge sets) is well conditioned and evidence-bearing."""

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


def test_ill_conditioned_stars_quarantined():
    """Stars with conditioning below rho_min must not contribute to F_DM
    (SI S10.4 dynamic preservation rule)."""

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
    quarantined = quarantined_nodes(stars)
    assert quarantined == {0}
    assert 5 not in quarantined

    # The ill-conditioned node's likelihood term is excluded from F_DM: scoring
    # with the quarantine drops exactly that node's contribution.
    a0 = bdeu_alpha(2)
    region = [
        NodeTransition(np.array([30.0, 1.0]), 2, a0, node_id=0),  # ill-conditioned
        NodeTransition(np.array([10.0, 10.0]), 2, a0, node_id=5),
    ]
    full = f_dm(region)
    without = f_dm(region, quarantined={0})
    only_good = f_dm([region[1]])
    assert abs(without - only_good) < 1e-12
    assert without < full  # the quarantined term carried positive score

    # An edit that would only be "accepted" thanks to the ill-conditioned star's
    # concentrated counts must not be accepted once that star is quarantined.
    keep = [NodeTransition(np.array([15.0, 16.0]), 2, a0, node_id=0)]
    edit = [NodeTransition(np.array([30.0, 1.0]), 2, a0, node_id=0)]
    proposal = EditProposal(EditType.SPLIT, [0], diagnostic_strength=1.0)
    v_free = evaluate_edit(keep, edit, proposal, tau_bf=3.0)
    v_quar = evaluate_edit(keep, edit, proposal, tau_bf=3.0, quarantined={0})
    assert v_free.accepted
    assert not v_quar.accepted
    assert v_quar.log_bayes_factor == 0.0  # no evidence-bearing star remains
