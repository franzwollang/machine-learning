"""Unit tests for Proteus moment helpers."""

from __future__ import annotations

import numpy as np

from proteus.moments import ewma_update, incoherence_ratio, variance_from_moments


def test_ewma_update_known_values() -> None:
    m = np.array([0.0, 2.0])
    s = np.array([1.0, 4.0])
    e = np.array([2.0, -2.0])

    m_new, s_new = ewma_update(m, s, e, alpha=0.5, weight=0.25)
    gain = 0.125

    np.testing.assert_allclose(m_new, (1.0 - gain) * m + gain * e)
    np.testing.assert_allclose(s_new, (1.0 - gain) * s + gain * (e * e))


def test_ewma_update_zero_gain_is_identity() -> None:
    m = np.array([1.0, -1.0, 3.0])
    s = np.array([1.0, 1.0, 9.0])
    e = np.array([10.0, 10.0, 10.0])

    m_new, s_new = ewma_update(m, s, e, alpha=0.0, weight=1.0)

    np.testing.assert_allclose(m_new, m)
    np.testing.assert_allclose(s_new, s)


def test_variance_from_moments_clips_negative_noise() -> None:
    m = np.array([1.0, 2.0])
    s = np.array([0.99, 3.5])

    assert variance_from_moments(m, s) == 0.0


def test_variance_from_moments_trace() -> None:
    m = np.array([1.0, 2.0, 0.0])
    s = np.array([2.0, 7.0, 4.0])

    assert variance_from_moments(m, s) == 1.0 + 3.0 + 4.0


def test_incoherence_ratio() -> None:
    m = np.array([3.0, 4.0])

    assert np.isclose(incoherence_ratio(m, sigma=2.0, eps=0.0), 2.5)
