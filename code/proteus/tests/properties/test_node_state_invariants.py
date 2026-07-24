"""Node-state invariants: variance >= 0, hits >= 0, EWMA half-life (SI S2.2, S2.3)."""
from __future__ import annotations

import numpy as np

from proteus.moments import ewma_update
from proteus.nodes import accumulate_hit, make_node, update_node_moments


def test_variance_always_nonnegative():
    """sigma_i^2 = tr(s_i - m_i * m_i) must be >= 0 after any update."""
    rng = np.random.default_rng(0)
    node = make_node(np.zeros(4), dim=4)
    alpha = np.log(2.0) / 8.0
    for _ in range(1000):
        residual = rng.normal(size=4)
        weight = 2.0 ** (-int(rng.integers(0, 8)))
        update_node_moments(node, residual, alpha, weight)
        assert node.variance >= -1e-10


def test_hit_counts_nonnegative():
    """h_i must be >= 0 after any sequence of rank-weighted updates."""
    node = make_node(np.zeros(2), dim=2)
    for rank in range(8):
        accumulate_hit(node, 2.0 ** (-rank))
        assert node.hit_count >= 0.0


def test_ewma_half_life():
    """EWMA with alpha = ln2/k should halve a step response in k updates."""
    k = 128
    alpha = np.log(2.0) / k
    m = np.array([0.0])
    s = np.array([0.0])
    e = np.array([1.0])
    for _ in range(k):
        m, s = ewma_update(m, s, e, alpha=alpha, weight=1.0)
    assert np.isclose(m[0], 0.5, atol=0.01)
