"""Unit tests for degree-based intrinsic-dimension fallback."""

from __future__ import annotations

import numpy as np

from proteus.intrinsic_dim import estimate_d_final


def test_chain_graph_degree_proxy() -> None:
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}

    d_final = estimate_d_final(graph, ambient_dim=3)

    np.testing.assert_array_equal(d_final, np.array([1, 1, 1, 1]))


def test_complete_graph_clips_to_ambient_dim() -> None:
    graph = {i: [j for j in range(5) if j != i] for i in range(5)}

    d_final = estimate_d_final(graph, ambient_dim=2)

    np.testing.assert_array_equal(d_final, np.full(5, 2))


def test_star_graph_median_smoothing() -> None:
    graph = {0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0]}

    d_final = estimate_d_final(graph, ambient_dim=4)

    np.testing.assert_array_equal(d_final, np.array([1, 2, 2, 2, 2]))
