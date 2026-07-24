"""Flag-complex construction on a fitted Stage 1 scaffold (SI S4.1, S4.2, T3).

These are *construction-invariant* checks on a real (tissue-polluted) circle
scaffold: every stored simplex is a lifted clique, simplices are canonical and
deduplicated, the per-node incidence signature is consistent, and the complex
inherits the stabilized Stage 1 graph (S4.4 T3). Topology *recovery* (Betti
numbers) is a separate evaluation concern (SI S14.2) and is not asserted here;
see OPEN_ISSUES #25 (persistent-homology recovery) and #40 (operational
``d_final`` refresh for heterogeneous per-patch simplex dimension).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.stabilization import StabilizationConfig
from proteus.stage2.flag_complex import (
    _symmetric_adjacency,
    flag_complex_from_scaffold,
)
from tests.datasets.synthetic.circles import make_circle


@pytest.fixture(scope="module")
def circle_scaffold():
    dataset = make_circle(
        n_samples=1200, radius=1.0, noise=0.02, extrusion_dim=2, seed=21
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    config = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=8,
        ann_backend="naive",
        stabilization=StabilizationConfig(min_equilibrium_epochs=3, max_epochs=15),
        seed=77,
    )
    result = run_scale_search(dataset.points, dim=gt.ambient_dim, config=config)
    return result.scaffold_at_star


@pytest.mark.stage2
def test_scaffold_flag_complex_is_valid_construction(circle_scaffold) -> None:
    scaffold = circle_scaffold
    n_nodes = len(scaffold.nodes)
    res = flag_complex_from_scaffold(scaffold)

    # Vertex positions carried through verbatim (S4.4 T3 inherits the graph).
    assert res.complex.vertex_positions.shape[0] == n_nodes

    adj = _symmetric_adjacency(scaffold.neighbour_graph(), n_nodes)

    # Every stored simplex is a clique of lifted edges (S4.1 invariant).
    for sigma in res.simplices:
        for a, b in combinations(sigma, 2):
            assert b in adj[a], f"simplex {sigma} is not a lifted clique"

    # Canonical (sorted) and deduplicated (S13.4).
    assert all(tuple(sorted(s)) == s for s in res.simplices)
    assert len(res.simplices) == len(set(res.simplices))

    # Simplex dimension never exceeds the working dimension.
    assert res.complex.intrinsic_dim <= scaffold.nodes[0].position.shape[0]

    # There is a non-empty complex on a stabilized circle scaffold.
    assert len(res.simplices) > 0


@pytest.mark.stage2
def test_scaffold_incidence_signature_is_consistent(circle_scaffold) -> None:
    """Per-node incidence equals the number of stored simplices containing it."""
    scaffold = circle_scaffold
    n_nodes = len(scaffold.nodes)
    res = flag_complex_from_scaffold(scaffold)

    recount = np.zeros(n_nodes, dtype=int)
    for sigma in res.simplices:
        for v in sigma:
            recount[v] += 1
    assert np.array_equal(recount, res.incidence_counts)

    # Orphans are exactly the zero-incidence nodes.
    assert set(res.orphan_ids) == {i for i in range(n_nodes) if res.incidence_counts[i] == 0}
    # A well-connected circle scaffold leaves few orphans.
    assert len(res.orphan_ids) <= 0.1 * n_nodes
