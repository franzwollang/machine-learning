"""Regression tests for the repaired linked-tori benchmark (issue #45)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.datasets.synthetic.faded_density import TorusSurfaceFadedComponent
from tests.datasets.synthetic.linked_tori import make_linked_tori


def test_torus_surface_sampling_is_continuous_and_area_uniform() -> None:
    component = TorusSurfaceFadedComponent(
        major_radius=2.0,
        minor_radius=0.25,
        sigma=0.005,
        transition_radius=3.0,
        center=np.zeros(3),
        rotation=np.eye(3),
    )
    points = component.sample(20_000, np.random.default_rng(0))
    planar_radius = np.linalg.norm(points[:, :2], axis=1)
    tube_radius = np.sqrt(
        np.square(planar_radius - 2.0) + np.square(points[:, 2]),
    )
    cos_phi = (planar_radius - 2.0) / tube_radius

    # For torus-area measure p(phi) ∝ R + r cos(phi), hence
    # E[cos(phi)] = r / (2R). A uniform-angle or anchor-lattice sampler
    # instead tends to zero.
    assert np.isclose(cos_phi.mean(), 0.25 / 4.0, atol=0.015)
    theta = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    assert np.unique(np.round(theta, decimals=5)).size > 19_000


def test_linked_tori_default_has_declared_positive_gap() -> None:
    dataset = make_linked_tori(n_per_torus=128, seed=0)
    meta = dataset.metadata

    assert meta["sampling"] == "continuous_area_uniform"
    assert meta["centerline_gap"] > meta["surface_gap"] > 0.0
    assert meta["surface_gap"] > meta["lambda_half_gap"] > 0.0
    assert set(dataset.labels) == {-1, 0, 1}
    assert dataset.ground_truth.topology is not None
    assert dataset.ground_truth.topology.connected_components == 2


def test_linked_tori_rejects_overlapping_tubes() -> None:
    with pytest.raises(ValueError, match="surfaces overlap"):
        make_linked_tori(minor_radius=0.5)


def test_linked_tori_resolvability_metadata_tracks_sample_budget() -> None:
    sparse = make_linked_tori(n_per_torus=1_000, seed=0)
    dense = make_linked_tori(n_per_torus=4_000, seed=0)

    assert sparse.metadata["resolvable_k8"] is False
    assert dense.metadata["resolvable_k8"] is True
    assert (
        dense.metadata["expected_knn_radius_k8"]
        < dense.metadata["lambda_half_gap"]
    )
