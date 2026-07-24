"""Dimensionality-junction generator for Proteus tests.

A 1D filament meeting a 2D sheet at a shared junction region — tests
whether the junction detector (SI S8.4) correctly identifies stars that
straddle a dimensional contrast and whether the freeze rule prevents
oscillatory refinement.
"""
from __future__ import annotations

import numpy as np

from ..ground_truth import (
    ClusterNode,
    GroundTruthManifold,
    JunctionExpectation,
    SyntheticDataset,
    TopologyExpectation,
    expected_tau_for_arc,
    expected_tau_for_surface,
    ideal_nodes_for_arc,
    ideal_nodes_for_surface,
)
from .faded_density import (
    AxisAlignedSegmentFadedComponent,
    AxisAlignedSheetFadedComponent,
    FadedMixture,
    SupportBox,
    assign_labels_by_lambda,
    sample_faded_mixture,
)
from .tissue import (
    expected_tau_for_uniform_tissue_box,
    ideal_nodes_for_uniform_tissue_box,
)


def make_filament_sheet_junction(
    n_sheet: int = 1500,
    n_filament: int = 500,
    sheet_size: float = 2.0,
    filament_length: float = 2.0,
    noise: float = 0.02,
    target_n_nodes: int = 32,
    filament_extrusion_dim: int = 2,
    sheet_extrusion_dim: int = 1,
    filament_extrusion_sigma: float | None = None,
    sheet_extrusion_sigma: float | None = None,
    tissue_fraction: float = 0.03,
    seed: int = 0,
) -> SyntheticDataset:
    """Generate a thickened filament and sheet as exact faded densities.

    The filament extends from the origin along the x-axis; the sheet
    lies in the xy-plane centered at the origin.  They share a junction
    region near x=0.
    """
    if filament_extrusion_dim < 0:
        raise ValueError("filament_extrusion_dim must be non-negative")
    if sheet_extrusion_dim < 0:
        raise ValueError("sheet_extrusion_dim must be non-negative")

    rng = np.random.default_rng(seed)
    ambient_dim = 3 + max(filament_extrusion_dim - 2, sheet_extrusion_dim - 1, 0)
    if filament_extrusion_sigma is None:
        filament_sigma = float(noise / np.sqrt(max(filament_extrusion_dim, 1)))
    else:
        filament_sigma = float(filament_extrusion_sigma)
    if sheet_extrusion_sigma is None:
        sheet_sigma = float(noise / np.sqrt(max(sheet_extrusion_dim, 1)))
    else:
        sheet_sigma = float(sheet_extrusion_sigma)

    sheet_component = AxisAlignedSheetFadedComponent(
        u_range=(-sheet_size / 2.0, sheet_size / 2.0),
        v_range=(-sheet_size / 2.0, sheet_size / 2.0),
        ambient_dim=ambient_dim,
        sigma=sheet_sigma,
        transition_radius=3.0,
        weight=n_sheet / max(n_sheet + n_filament, 1),
    )
    filament_component = AxisAlignedSegmentFadedComponent(
        t_range=(0.0, filament_length),
        ambient_dim=ambient_dim,
        sigma=filament_sigma,
        transition_radius=3.0,
        weight=n_filament / max(n_sheet + n_filament, 1),
    )
    sheet = sheet_component.sample(n_sheet, np.random.default_rng(seed + 17))
    filament = filament_component.sample(n_filament, np.random.default_rng(seed + 23))
    signal_points = np.vstack([sheet, filament])
    support = SupportBox.from_points(
        signal_points,
        padding_fraction=max(0.05, tissue_fraction),
        min_padding=0.05,
        extra_padding=3.0 * max(sheet_sigma, filament_sigma),
    )
    mixture = FadedMixture([sheet_component, filament_component], support)
    points, sampler_meta = sample_faded_mixture(mixture, n_sheet + n_filament, rng)
    labels = assign_labels_by_lambda(points, [sheet_component, filament_component], label_offsets=[0, 1])
    sheet_noise_variance = ambient_dim * sheet_sigma**2
    filament_noise_variance = ambient_dim * filament_sigma**2

    junction_loc = np.zeros(ambient_dim)
    filament_tau = expected_tau_for_arc(
        perimeter=filament_length,
        target_n_nodes=target_n_nodes,
        noise_variance=filament_noise_variance,
    )
    sheet_tau = expected_tau_for_surface(
        surface_area=sheet_size * sheet_size,
        target_n_nodes=target_n_nodes,
        noise_variance=sheet_noise_variance,
    )
    signal_tau = min(filament_tau, sheet_tau)
    tissue_bounds = support.bounds
    tissue_tau = expected_tau_for_uniform_tissue_box(
        tissue_bounds,
        target_n_nodes=target_n_nodes,
        noise_variance=max(filament_noise_variance, sheet_noise_variance),
    )
    expected_tau = max(signal_tau, tissue_tau)
    ideal_nodes = int(np.ceil(max(
        ideal_nodes_for_arc(
            perimeter=filament_length,
            tau=expected_tau,
            noise_variance=filament_noise_variance,
        )
        + ideal_nodes_for_surface(
            surface_area=sheet_size * sheet_size,
            tau=expected_tau,
            noise_variance=sheet_noise_variance,
        ),
        ideal_nodes_for_uniform_tissue_box(
            tissue_bounds,
            tau=expected_tau,
            noise_variance=max(filament_noise_variance, sheet_noise_variance),
        ),
    )))

    gt = GroundTruthManifold(
        name="filament_sheet_junction",
        ambient_dim=ambient_dim,
        intrinsic_dim=2,
        expected_scale_levels=2,
        cluster_hierarchy=[
            ClusterNode(
                cluster_id=0, level=0, parent_id=None, weight=1.0,
                center=signal_points.mean(axis=0),
                covariance=np.cov(signal_points, rowvar=False),
                is_leaf=False, intrinsic_dim=2,
            ),
            ClusterNode(
                cluster_id=1, level=1, parent_id=0,
                weight=n_sheet / (n_sheet + n_filament),
                center=sheet.mean(axis=0),
                covariance=np.cov(sheet, rowvar=False),
                is_leaf=True, intrinsic_dim=2,
            ),
            ClusterNode(
                cluster_id=2, level=1, parent_id=0,
                weight=n_filament / (n_sheet + n_filament),
                center=filament.mean(axis=0),
                covariance=np.cov(filament, rowvar=False),
                is_leaf=True, intrinsic_dim=1,
            ),
        ],
        topology=TopologyExpectation(
            connected_components=1, betti_numbers=(1, 0), intrinsic_dim=2,
        ),
        junctions=[
            JunctionExpectation(
                location_hint=junction_loc,
                dim_low=1,
                dim_high=2,
                description="1D filament meets 2D sheet at origin",
            ),
        ],
        expected_tau=expected_tau,
        expected_node_count=ideal_nodes,
        node_count_upper_bound=4 * ideal_nodes,
        noise_variance=max(filament_noise_variance, sheet_noise_variance),
        tau_grid_hint=(min(signal_tau, tissue_tau) / 8.0, max(filament_tau, sheet_tau, tissue_tau) * 8.0),
    )
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=gt,
        metadata={
            "filament_extrusion_dim": filament_extrusion_dim,
            "sheet_extrusion_dim": sheet_extrusion_dim,
            "filament_extrusion_sigma": (
                filament_sigma if filament_extrusion_dim > 0 else 0.0
            ),
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
