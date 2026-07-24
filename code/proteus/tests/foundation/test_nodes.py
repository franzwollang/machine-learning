"""Unit tests for node-state helpers."""

from __future__ import annotations

import numpy as np

from proteus.moments import ewma_update, variance_from_moments
from proteus.nodes import accumulate_hit, make_node, update_node_moments


def test_make_node_shapes_and_defaults() -> None:
    node = make_node(np.array([1.0, 2.0, 3.0]), dim=3)

    assert node.position.shape == (3,)
    assert node.residual_mean.shape == (3,)
    assert node.residual_sq.shape == (3,)
    assert node.nudge.shape == (3,)
    assert node.hit_count == 0.0
    assert node.variance == 0.0
    assert np.isclose(np.linalg.norm(node.principal_dir), 1.0)


def test_make_node_normalizes_principal_dir() -> None:
    node = make_node(
        np.zeros(2),
        dim=2,
        principal_dir=np.array([3.0, 4.0]),
    )

    np.testing.assert_allclose(node.principal_dir, np.array([0.6, 0.8]))


def test_update_node_moments_bmu_updates_all() -> None:
    node = make_node(np.zeros(2), dim=2)
    e = np.array([2.0, -1.0])
    alpha = 0.25
    weight = 0.5

    m_expected, s_expected = ewma_update(
        node.residual_mean, node.residual_sq, e, alpha, weight
    )
    result = update_node_moments(node, e, alpha, weight, is_bmu=True)

    assert result is node
    np.testing.assert_allclose(node.residual_mean, m_expected)
    np.testing.assert_allclose(node.residual_sq, s_expected)
    assert np.isclose(node.variance, variance_from_moments(m_expected, s_expected))
    assert node.update_count == 1


def test_update_node_moments_non_bmu_updates_mean_only() -> None:
    node = make_node(np.zeros(2), dim=2)
    e = np.array([2.0, -1.0])
    alpha = 0.25
    weight = 0.5

    s_before = node.residual_sq.copy()
    var_before = node.variance
    update_node_moments(node, e, alpha, weight, is_bmu=False)

    gain = alpha * weight
    np.testing.assert_allclose(node.residual_mean, gain * e)
    np.testing.assert_allclose(node.residual_sq, s_before)
    assert node.variance == var_before
    assert node.update_count == 0


def test_accumulate_hit_is_additive() -> None:
    node = make_node(np.zeros(1), dim=1)

    accumulate_hit(node, 0.5)
    accumulate_hit(node, 0.25)

    assert np.isclose(node.hit_count, 0.75)


def test_accumulate_hit_rejects_negative_weights() -> None:
    node = make_node(np.zeros(1), dim=1)

    try:
        accumulate_hit(node, -1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative hit weights should raise ValueError")


def test_shadow_moments_partition_consistently() -> None:
    """Shadow hit mass should partition the aggregate hit mass by sign of
    the projection onto the Oja direction (SI S2.3.2)."""

    rng = np.random.default_rng(42)
    dim = 3
    u = np.array([1.0, 0.0, 0.0])
    node = make_node(np.zeros(dim), dim=dim, principal_dir=u)
    alpha = 0.1

    n_pos = 0
    n_neg = 0
    for _ in range(50):
        e = rng.standard_normal(dim)
        w = rng.uniform(0.1, 1.0)
        proj = float(np.dot(e, u))
        if proj > 0:
            n_pos += 1
        elif proj < 0:
            n_neg += 1
        update_node_moments(node, e, alpha, w, is_bmu=True)
        accumulate_hit(node, w)

    assert node.update_count_pos == n_pos
    assert node.update_count_neg == n_neg
    assert np.isclose(node.h_pos + node.h_neg, node.hit_count, rtol=0.05)
    assert node.h_pos > 0.0
    assert node.h_neg > 0.0
