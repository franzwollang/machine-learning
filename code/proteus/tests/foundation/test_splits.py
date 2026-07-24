"""Unit tests for variance-cap split helpers with shadow-moment inheritance."""

from __future__ import annotations

import numpy as np

from proteus.moments import variance_from_moments
from proteus.stage1 import Stage1Scaffold
from proteus.stage1.splits import SplitProposal, apply_split, propose_splits


def _one_node_scaffold() -> Stage1Scaffold:
    scaffold = Stage1Scaffold(dim=2, tau=1.0, min_nodes=1, ann_backend="naive")
    scaffold.init_from(np.array([[0.0, 0.0]]), n_seeds=1)
    node = scaffold.nodes[0]
    node.principal_dir = np.array([1.0, 0.0])
    node.residual_mean = np.array([1.0, 0.0])
    node.residual_sq = np.array([2.0, 0.0])
    node.nudge = np.array([0.25, -0.25])
    node.variance = 1.0
    node.hit_count = 10.0
    node.update_count = scaffold.prune_after
    # Seed shadow moments with known values
    node.m_pos = np.array([0.3, 0.1])
    node.s_pos = np.array([0.5, 0.2])
    node.h_pos = 6.0
    node.update_count_pos = 15
    node.m_neg = np.array([-0.2, 0.05])
    node.s_neg = np.array([0.4, 0.1])
    node.h_neg = 4.0
    node.update_count_neg = 10
    scaffold.tau_local = np.array([0.5])
    return scaffold


def test_propose_splits_over_cap_node() -> None:
    scaffold = _one_node_scaffold()

    proposals = propose_splits(scaffold)

    assert len(proposals) == 1
    assert proposals[0].node_id == 0
    np.testing.assert_allclose(
        proposals[0].axis_unit_vector,
        np.array([1.0, 0.0]),
    )


def test_propose_splits_uses_oja_not_residual_mean() -> None:
    """Split axis should always be the Oja direction, not the residual mean."""
    scaffold = _one_node_scaffold()
    node = scaffold.nodes[0]
    node.principal_dir = np.array([0.0, 1.0])
    node.residual_mean = np.array([1.0, 0.0])

    proposals = propose_splits(scaffold)

    assert len(proposals) == 1
    np.testing.assert_allclose(
        proposals[0].axis_unit_vector,
        np.array([0.0, 1.0]),
    )


def test_apply_split_inherits_shadow_moments_with_steiner_shift() -> None:
    scaffold = _one_node_scaffold()
    # Pre-compute un-shifted shadow variances for invariance check
    var_pos_pre = variance_from_moments(
        scaffold.nodes[0].m_pos, scaffold.nodes[0].s_pos
    )
    var_neg_pre = variance_from_moments(
        scaffold.nodes[0].m_neg, scaffold.nodes[0].s_neg
    )

    proposal = SplitProposal(0, np.array([1.0, 0.0]), 0.25)
    accepted = apply_split(scaffold, proposal)

    assert accepted
    assert len(scaffold.nodes) == 2
    parent, child = scaffold.nodes

    # Hit counts from shadow masses, flow-conserved
    assert np.isclose(parent.hit_count, 4.0)
    assert np.isclose(child.hit_count, 6.0)
    assert np.isclose(parent.hit_count + child.hit_count, 10.0)

    # Positions placed symmetrically along +/- u_i
    np.testing.assert_allclose(child.position, np.array([0.25, 0.0]))
    np.testing.assert_allclose(parent.position, np.array([-0.25, 0.0]))

    # Child inherits positive-side shadows with Steiner shift d = +[0.25, 0]
    # m' = m_pos - d = [0.3,0.1] - [0.25,0] = [0.05, 0.1]
    # s' = s_pos - 2*d*m_pos + d^2 = [0.5,0.2] - [0.15,0] + [0.0625,0]
    np.testing.assert_allclose(child.residual_mean, np.array([0.05, 0.1]))
    np.testing.assert_allclose(child.residual_sq, np.array([0.4125, 0.2]))

    # Parent inherits negative-side shadows with Steiner shift d = -[0.25, 0]
    # m' = m_neg - (-d) = m_neg + [0.25,0] = [-0.2,0.05] + [0.25,0] = [0.05, 0.05]
    # s' = s_neg + 2*[0.25,0]*m_neg + [0.0625,0] = [0.4,0.1] + [-0.1,0] + [0.0625,0]
    np.testing.assert_allclose(parent.residual_mean, np.array([0.05, 0.05]))
    np.testing.assert_allclose(parent.residual_sq, np.array([0.3625, 0.1]))

    # Variance is shift-invariant
    assert np.isclose(child.variance, var_pos_pre)
    assert np.isclose(parent.variance, var_neg_pre)

    # Both children have zeroed shadow pairs and nudge
    for node in (parent, child):
        np.testing.assert_allclose(node.m_pos, np.zeros(2))
        np.testing.assert_allclose(node.s_pos, np.zeros(2))
        np.testing.assert_allclose(node.m_neg, np.zeros(2))
        np.testing.assert_allclose(node.s_neg, np.zeros(2))
        np.testing.assert_allclose(node.nudge, np.zeros(2))
        assert node.h_pos == 0.0
        assert node.h_neg == 0.0
        assert node.update_count == 0
        assert node.update_count_pos == 0
        assert node.update_count_neg == 0

    # Principal dir inherited
    np.testing.assert_allclose(child.principal_dir, np.array([1.0, 0.0]))


def test_split_budget_guard_rejects_underfed_parent() -> None:
    scaffold = _one_node_scaffold()
    scaffold.nodes[0].hit_count = 0.0
    proposal = SplitProposal(0, np.array([1.0, 0.0]), 0.25)

    assert not apply_split(scaffold, proposal)
    assert len(scaffold.nodes) == 1
