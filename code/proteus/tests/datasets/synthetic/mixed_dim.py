"""Mixed-intrinsic-dimension generator for Proteus tests.

Separate 1D and 2D components that do NOT share a junction — tests
whether Proteus correctly estimates different intrinsic dimensions in
disjoint parts of the support without triggering the junction detector.
"""
from __future__ import annotations

import numpy as np

from ..ground_truth import (
    ClusterNode,
    GroundTruthManifold,
    SyntheticDataset,
    TopologyExpectation,
    expected_tau_for_arc,
    expected_tau_for_surface,
    ideal_nodes_for_arc,
    ideal_nodes_for_surface,
)
from .faded_density import (
    AxisAlignedSheetFadedComponent,
    CircleFadedComponent,
    FadedMixture,
    SupportBox,
    assign_labels_by_lambda,
    sample_faded_mixture,
)
from .tissue import (
    expected_tau_for_uniform_tissue_box,
    ideal_nodes_for_uniform_tissue_box,
)


def make_mixed_dim(
    n_curve: int = 600,
    n_sheet: int = 1200,
    curve_radius: float = 1.0,
    sheet_size: float = 2.0,
    separation: float = 5.0,
    noise: float = 0.02,
    target_n_nodes: int = 32,
    curve_extrusion_dim: int = 2,
    sheet_extrusion_dim: int = 1,
    curve_extrusion_sigma: float | None = None,
    sheet_extrusion_sigma: float | None = None,
    tissue_fraction: float = 0.03,
    seed: int = 0,
) -> SyntheticDataset:
    """Generate a thickened ring and sheet as exact faded densities.

    The circle lies in the xy-plane centered at (separation, 0, 0);
    the sheet lies in the xy-plane centered at the origin.  They are
    far enough apart that no junction should be detected.
    """
    if curve_extrusion_dim < 0:
        raise ValueError("curve_extrusion_dim must be non-negative")
    if sheet_extrusion_dim < 0:
        raise ValueError("sheet_extrusion_dim must be non-negative")

    rng = np.random.default_rng(seed)
    ambient_dim = 3 + max(curve_extrusion_dim - 2, sheet_extrusion_dim - 1, 0)
    if curve_extrusion_sigma is None:
        curve_sigma = float(noise / np.sqrt(max(curve_extrusion_dim, 1)))
    else:
        curve_sigma = float(curve_extrusion_sigma)
    if sheet_extrusion_sigma is None:
        sheet_sigma = float(noise / np.sqrt(max(sheet_extrusion_dim, 1)))
    else:
        sheet_sigma = float(sheet_extrusion_sigma)

    curve_component = CircleFadedComponent(
        radius=curve_radius,
        sigma=curve_sigma,
        transition_radius=3.0,
        center=np.array([separation, 0.0] + [0.0] * (ambient_dim - 2), dtype=float),
        weight=n_curve / max(n_curve + n_sheet, 1),
    )
    sheet_component = AxisAlignedSheetFadedComponent(
        u_range=(-sheet_size / 2.0, sheet_size / 2.0),
        v_range=(-sheet_size / 2.0, sheet_size / 2.0),
        ambient_dim=ambient_dim,
        sigma=sheet_sigma,
        transition_radius=3.0,
        weight=n_sheet / max(n_curve + n_sheet, 1),
    )
    curve = curve_component.sample(n_curve, np.random.default_rng(seed + 17))
    sheet = sheet_component.sample(n_sheet, np.random.default_rng(seed + 23))
    signal_points = np.vstack([curve, sheet])
    support = SupportBox.from_points(
        signal_points,
        padding_fraction=max(0.05, tissue_fraction),
        min_padding=0.05,
        extra_padding=3.0 * max(curve_sigma, sheet_sigma),
    )
    mixture = FadedMixture([curve_component, sheet_component], support)
    points, sampler_meta = sample_faded_mixture(mixture, n_curve + n_sheet, rng)
    labels = assign_labels_by_lambda(points, [curve_component, sheet_component], label_offsets=[0, 1])
    curve_noise_variance = ambient_dim * curve_sigma**2
    sheet_noise_variance = ambient_dim * sheet_sigma**2
    curve_tau = expected_tau_for_arc(
        perimeter=2.0 * np.pi * curve_radius,
        target_n_nodes=target_n_nodes,
        noise_variance=curve_noise_variance,
    )
    sheet_tau = expected_tau_for_surface(
        surface_area=sheet_size * sheet_size,
        target_n_nodes=target_n_nodes,
        noise_variance=sheet_noise_variance,
    )
    signal_tau = min(curve_tau, sheet_tau)
    tissue_bounds = support.bounds
    tissue_tau = expected_tau_for_uniform_tissue_box(
        tissue_bounds,
        target_n_nodes=target_n_nodes,
        noise_variance=max(curve_noise_variance, sheet_noise_variance),
    )
    expected_tau = max(signal_tau, tissue_tau)
    ideal_nodes = int(np.ceil(max(
        ideal_nodes_for_arc(
            perimeter=2.0 * np.pi * curve_radius,
            tau=expected_tau,
            noise_variance=curve_noise_variance,
        )
        + ideal_nodes_for_surface(
            surface_area=sheet_size * sheet_size,
            tau=expected_tau,
            noise_variance=sheet_noise_variance,
        ),
        ideal_nodes_for_uniform_tissue_box(
            tissue_bounds,
            tau=expected_tau,
            noise_variance=max(curve_noise_variance, sheet_noise_variance),
        ),
    )))

    gt = GroundTruthManifold(
        name="mixed_dim",
        ambient_dim=ambient_dim,
        intrinsic_dim=2,
        expected_scale_levels=2,
        cluster_hierarchy=[
            ClusterNode(
                cluster_id=0, level=0, parent_id=None, weight=1.0,
                center=signal_points.mean(axis=0),
                covariance=np.cov(signal_points, rowvar=False),
                is_leaf=False,
            ),
            ClusterNode(
                cluster_id=1, level=1, parent_id=0,
                weight=n_curve / (n_curve + n_sheet),
                center=curve.mean(axis=0),
                covariance=np.cov(curve, rowvar=False),
                is_leaf=True, intrinsic_dim=1,
            ),
            ClusterNode(
                cluster_id=2, level=1, parent_id=0,
                weight=n_sheet / (n_curve + n_sheet),
                center=sheet.mean(axis=0),
                covariance=np.cov(sheet, rowvar=False),
                is_leaf=True, intrinsic_dim=2,
            ),
        ],
        topology=TopologyExpectation(
            connected_components=2, betti_numbers=(2, 1), intrinsic_dim=2,
        ),
        per_component_topology=[
            TopologyExpectation(
                connected_components=1, betti_numbers=(1, 1), intrinsic_dim=1,
            ),
            TopologyExpectation(
                connected_components=1, betti_numbers=(1, 0), intrinsic_dim=2,
            ),
        ],
        expected_tau=expected_tau,
        expected_node_count=ideal_nodes,
        node_count_upper_bound=3 * ideal_nodes,
        noise_variance=max(curve_noise_variance, sheet_noise_variance),
        tau_grid_hint=(min(signal_tau, tissue_tau) / 8.0, max(curve_tau, sheet_tau, tissue_tau) * 8.0),
    )
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=gt,
        metadata={
            "curve_extrusion_dim": curve_extrusion_dim,
            "sheet_extrusion_dim": sheet_extrusion_dim,
            "curve_extrusion_sigma": curve_sigma if curve_extrusion_dim > 0 else 0.0,
            "sheet_extrusion_sigma": sheet_sigma if sheet_extrusion_dim > 0 else 0.0,
            "signal_expected_tau": float(signal_tau),
            "tissue_expected_tau": float(tissue_tau),
            "tissue_fraction_actual": float(np.mean(labels < 0)),
            "tissue_fraction_requested": tissue_fraction,
            "support_bounds_lo": tissue_bounds[0].tolist(),
            "support_bounds_hi": tissue_bounds[1].tolist(),
            **sampler_meta,
        },
    )
