"""Validate the intrinsic-dimension estimators against ground truth (SI S1.4.1).

These tests close OPEN_ISSUES #39: they quantify the degree-proxy estimator
against known intrinsic dimension and demonstrate the Levina--Bickel MLE
cross-check. Two regimes are covered:

* the uniform d-ball reference ensemble (the same ensemble that calibrates
  ``c_{d,k}`` / ``C_Q(d)``), where the degree proxy is *unbiased*; and
* a thin 1-D circle embedded in a higher ambient dimension, where the degree
  proxy over-reads and the Levina--Bickel estimator recovers the true value.
"""

from __future__ import annotations

import numpy as np
import pytest

from proteus.intrinsic_dim import estimate_d_final, estimate_d_final_mle
from proteus.stage1 import Stage1Scaffold
from proteus.stage1.calibration import _build_equilibrated_scaffold, sample_unit_ball
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.circles import make_circle


# --- estimate_d_final_mle: basic contract -------------------------------------


def test_mle_empty_and_singleton() -> None:
    assert estimate_d_final_mle(np.empty((0, 3))).size == 0
    np.testing.assert_array_equal(
        estimate_d_final_mle(np.zeros((1, 3)), dim_floor=2), np.array([2])
    )


def test_mle_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        estimate_d_final_mle(np.zeros((4, 2)), dim_floor=0)
    with pytest.raises(ValueError):
        estimate_d_final_mle(np.zeros(5))  # not 2-D


def test_mle_clips_to_ambient_dim() -> None:
    rng = np.random.default_rng(0)
    points = sample_unit_ball(400, 3, rng)
    d_final = estimate_d_final_mle(points, ambient_dim=2)
    assert int(np.max(d_final)) <= 2


# --- Levina--Bickel recovers dimension on well-sampled manifolds --------------


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_mle_recovers_dimension_on_dense_ball_samples(d: int) -> None:
    """On dense (locally Poissonian) uniform d-ball samples the MLE is exact."""

    points = sample_unit_ball(2000, d, np.random.default_rng(0))
    d_final = estimate_d_final_mle(points, ambient_dim=d)
    assert int(np.median(d_final)) == d


# --- Degree proxy is unbiased on the reference ensemble -----------------------


@pytest.mark.parametrize("d", [1, 2, 3])
def test_degree_proxy_unbiased_on_reference_ensemble(d: int) -> None:
    """The degree proxy median matches the true dimension on the uniform d-ball
    reference ensemble (the ensemble used to calibrate c_{d,k} / C_Q(d))."""

    scaffold, _ = _build_equilibrated_scaffold(
        d,
        8,
        n_samples=1200,
        target_nodes=60,
        min_nodes=4,
        max_nodes=200,
        max_epochs=15,
        ann_backend="naive",
        seed=0,
        ensemble=0,
    )
    d_final = estimate_d_final(scaffold.neighbour_graph(), ambient_dim=d)
    assert int(np.median(d_final)) == d


# --- Documented failure mode + MLE correction ---------------------------------


def test_mle_corrects_degree_proxy_upward_bias_on_thin_circle() -> None:
    """A 1-D circle embedded in 3-D: off-manifold lifted edges inflate node
    degree, so the degree proxy over-reads (median >= 2 vs true d = 1); the
    Levina--Bickel estimator on node positions recovers d = 1 (SI S1.4.1)."""

    dataset = make_circle(seed=0)
    signal = dataset.points[dataset.labels >= 0]
    scaffold = Stage1Scaffold(
        dim=signal.shape[1],
        tau=dataset.ground_truth.expected_tau,
        k=8,
        min_nodes=4,
        max_nodes=400,
        ann_backend="naive",
        rng=np.random.default_rng(0),
    )
    scaffold.init_from(signal, n_seeds=4)
    scaffold.run_until_stable(signal, StabilizationConfig(max_epochs=25))

    degree_median = int(
        np.median(estimate_d_final(scaffold.neighbour_graph(), ambient_dim=signal.shape[1]))
    )
    mle_median = int(
        np.median(estimate_d_final_mle(scaffold.node_positions(), ambient_dim=signal.shape[1]))
    )

    assert dataset.ground_truth.intrinsic_dim == 1
    assert degree_median >= 2  # documented upward bias on a thin curved manifold
    assert mle_median == 1  # Levina--Bickel recovers the true dimension
    assert mle_median < degree_median


# --- Scaffold estimator-method selection --------------------------------------


def test_scaffold_mle_method_populates_d_final() -> None:
    dataset = make_circle(seed=0)
    signal = dataset.points[dataset.labels >= 0]
    scaffold = Stage1Scaffold(
        dim=signal.shape[1],
        tau=dataset.ground_truth.expected_tau,
        k=8,
        min_nodes=4,
        max_nodes=400,
        ann_backend="naive",
        intrinsic_dim_method="mle",
        rng=np.random.default_rng(0),
    )
    scaffold.init_from(signal, n_seeds=4)
    scaffold.run_until_stable(signal, StabilizationConfig(max_epochs=20))
    scaffold.refresh_intrinsic_dim()

    assert all(int(node.d_final) >= 1 for node in scaffold.nodes)
    assert int(np.median([int(node.d_final) for node in scaffold.nodes])) == 1


def test_invalid_intrinsic_dim_method_rejected() -> None:
    with pytest.raises(ValueError):
        Stage1Scaffold(dim=2, tau=0.1, intrinsic_dim_method="bogus")
