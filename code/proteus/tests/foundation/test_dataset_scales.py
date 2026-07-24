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
from tests.datasets.synthetic.linked_tori import make_linked_tori
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
