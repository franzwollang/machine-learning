"""Rank-weighted directed link counter invariants (SI S2, S3.1)."""
from __future__ import annotations

import numpy as np

from proteus.links import LinkCounters


def test_rank_weights_sum():
    """Rank weights 2^{-(r-1)} for r=1..k must sum to 2(1-2^{-k})."""
    k = 8
    weights = np.array([2.0 ** (-rank) for rank in range(k)])
    assert np.isclose(weights.sum(), 2.0 * (1.0 - 2.0 ** (-k)))
    assert np.isclose(weights.sum(), 2.0 - 2.0 ** (-(k - 1)))


def test_directed_counters_accumulate():
    """C(i->j) increments by rank weight; all k neighbors get edges."""
    links = LinkCounters()
    neighbors = [0, 2, 3, 2]
    weights = [1.0, 0.5, 0.25, 0.125]

    links.record_neighborhood(neighbors, weights)

    summary = {(link.i, link.j): link for link in links.as_list()}
    assert np.isclose(summary[(0, 2)].count_ij, 0.5 + 0.125)
    assert summary[(0, 2)].lifted is True
    assert np.isclose(summary[(0, 3)].count_ij, 0.25)
    assert summary[(0, 3)].lifted is False
