"""Tests for synthetic ground-truth scale metadata."""

from __future__ import annotations

import numpy as np

from tests.datasets.ground_truth import (
    expected_tau_for_arc,
    expected_tau_for_surface,
    ideal_nodes_for_arc,
    ideal_nodes_for_surface,
)
from tests.datasets.synthetic.circles import make_circle
from tests.datasets.synthetic.dim_junctions import make_filament_sheet_junction
from tests.datasets.synthetic.hierarchical_gaussian import (
    make_hierarchical_gaussian,
)
from tests.datasets.synthetic.faded_density import AxisAlignedBoxFadedComponent
from tests.datasets.synthetic.linked_tori import make_linked_tori
from tests.datasets.synthetic.manifold_zoo import make_manifold_zoo
from tests.datasets.synthetic.mixed_dim import make_mixed_dim
from tests.datasets.synthetic.nested_spheres import make_nested_spheres
from tests.datasets.synthetic.swiss_roll import make_swiss_roll
from tests.datasets.synthetic.variable_density import (
    make_variable_density_sheet,
)


def test_expected_tau_for_arc_formula() -> None:
    tau = expected_tau_for_arc(perimeter=12.0, target_n_nodes=6)

    assert np.isclose(tau, (12.0 / 6.0) ** 2 / 12.0)


def test_expected_tau_for_surface_formula() -> None:
    tau = expected_tau_for_surface(surface_area=8.0 * np.pi, target_n_nodes=4)

    assert np.isclose(tau, 0.25)


def test_expected_tau_decreases_with_node_count() -> None:
    coarse = expected_tau_for_arc(perimeter=2.0 * np.pi, target_n_nodes=16)
    fine = expected_tau_for_arc(perimeter=2.0 * np.pi, target_n_nodes=32)

    assert fine < coarse


def test_ideal_nodes_for_arc_inverts_tau_formula() -> None:
    tau = expected_tau_for_arc(
        perimeter=2.0 * np.pi,
        target_n_nodes=32,
        noise_variance=0.01,
    )

    estimate = ideal_nodes_for_arc(2.0 * np.pi, tau, noise_variance=0.01)
    assert np.isclose(estimate, 32)


def test_ideal_nodes_for_surface_inverts_tau_formula() -> None:
    tau = expected_tau_for_surface(
        surface_area=4.0,
        target_n_nodes=64,
        noise_variance=0.01,
    )

    estimate = ideal_nodes_for_surface(4.0, tau, noise_variance=0.01)
    assert np.isclose(estimate, 64)


def test_all_synthetic_generators_populate_expected_tau() -> None:
    datasets = [
        make_circle(n_samples=64, extrusion_dim=2),
        make_swiss_roll(n_samples=64, extrusion_dim=1),
        make_nested_spheres(n_per_sphere=32, extrusion_dim=1),
        make_linked_tori(n_per_torus=32, extrusion_dim=1),
        make_variable_density_sheet(n_samples=64, extrusion_dim=1),
        make_mixed_dim(
            n_curve=32,
            n_sheet=32,
            curve_extrusion_dim=2,
            sheet_extrusion_dim=1,
        ),
        make_filament_sheet_junction(
            n_sheet=32,
            n_filament=32,
            filament_extrusion_dim=2,
            sheet_extrusion_dim=1,
        ),
        make_hierarchical_gaussian(n_samples=96),
        make_manifold_zoo(n_box=64, n_plane=48, n_segment=24, n_circle=32),
    ]

    for dataset in datasets:
        gt = dataset.ground_truth
        assert gt.expected_tau is not None, gt.name
        assert np.isfinite(gt.expected_tau), gt.name
        assert gt.expected_tau > 0.0, gt.name
        assert gt.expected_node_count is not None, gt.name
        assert gt.node_count_upper_bound is not None, gt.name
        assert gt.node_count_upper_bound >= gt.expected_node_count, gt.name
        assert gt.tau_grid_hint is not None, gt.name
        lo, hi = gt.tau_grid_hint
        assert lo < gt.expected_tau < hi, gt.name


def test_axis_aligned_box_component_contract() -> None:
    """Solid-box faded component: uniform inside, Gaussian fade in normals."""
    rng = np.random.default_rng(0)

    # Solid 3-box (no normal directions): samples stay in the box, interior
    # distance is zero, density is uniform inside and zero outside the span.
    solid = AxisAlignedBoxFadedComponent(
        lo=(0.0, 0.0, 0.0), hi=(1.0, 2.0, 3.0),
        ambient_dim=3, sigma=0.02, transition_radius=3.0,
    )
    assert solid.box_dim == 3
    pts = solid.sample(2000, rng)
    assert pts.shape == (2000, 3)
    assert np.all(pts >= np.array([0.0, 0.0, 0.0])) and np.all(pts <= np.array([1.0, 2.0, 3.0]))
    inside = np.array([[0.5, 1.0, 1.5]])
    outside = np.array([[5.0, 1.0, 1.5]])
    assert np.isclose(solid.distance(inside)[0], 0.0)
    assert solid.distance(outside)[0] > 0.0
    volume = 1.0 * 2.0 * 3.0
    assert np.isclose(solid.density(inside)[0], 1.0 / volume)
    assert np.isclose(solid.density(outside)[0], 0.0)

    # Box spanning a strict subset of the ambient axes fades in the normals.
    embedded = AxisAlignedBoxFadedComponent(
        lo=(0.0, 0.0), hi=(1.0, 1.0),
        ambient_dim=3, sigma=0.05, transition_radius=3.0,
    )
    assert embedded.box_dim == 2
    on_plane = np.array([[0.5, 0.5, 0.0]])
    off_plane = np.array([[0.5, 0.5, 0.4]])
    assert embedded.fade_weight(on_plane)[0] > embedded.fade_weight(off_plane)[0]
    samples = embedded.sample(500, np.random.default_rng(1))
    assert np.all(samples[:, 0] >= 0.0) and np.all(samples[:, 0] <= 1.0)
    assert np.abs(samples[:, 2]).mean() < 0.5  # concentrated near the plane
