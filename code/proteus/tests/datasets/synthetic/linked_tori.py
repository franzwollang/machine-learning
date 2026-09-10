"""Linked-torus generator for Proteus tests.

Two linked tori in R^3 — a classic topology-recovery benchmark where
the expected homology is non-trivial (each torus has b1=2, b2=1).
"""
from __future__ import annotations

import numpy as np

from ..ground_truth import (
    ClusterNode,
    GroundTruthManifold,
    SyntheticDataset,
    TopologyExpectation,
    expected_tau_for_surface,
    ideal_nodes_for_surface,
)
from .faded_density import (
    FadedMixture,
    SupportBox,
    TorusSurfaceFadedComponent,
    assign_labels_by_lambda,
    sample_faded_mixture,
    tissue_mass_metadata,
)
from .tissue import (
    expected_tau_for_uniform_tissue_box,
    ideal_nodes_for_uniform_tissue_box,
)

def make_linked_tori(
    n_per_torus: int = 1000,
    major_radius: float = 2.0,
    minor_radius: float = 0.25,
    noise: float = 0.02,
    target_n_nodes: int = 64,
    extrusion_dim: int = 1,
    extrusion_sigma: float | None = None,
    tissue_fraction: float = 0.03,
    tissue_mass: float | None = None,
    seed: int = 0,
) -> SyntheticDataset:
    """Generate two separated Hopf-linked tori as exact faded densities.

    The core-circle separation is
    ``major_radius * (sqrt(2) - 1)``.  The tube surfaces must leave a
    positive gap; the former ``minor_radius=0.5`` default violated this
    condition and welded the two labelled tori together.  Sampling is
    continuous and area-uniform on each torus rather than drawn from a
    sparse kernel-anchor lattice.

    ``tissue_fraction`` only pads the support box (historical name).
    Pass ``tissue_mass`` for an honest expected λ<0.5 background fraction;
    ``None`` keeps the legacy fade-balanced floor (~46–49% tissue).
    """
    if extrusion_dim < 0:
        raise ValueError("extrusion_dim must be non-negative")
    if n_per_torus <= 0:
        raise ValueError("n_per_torus must be positive")
    centerline_gap = float(major_radius * (np.sqrt(2.0) - 1.0))
    surface_gap = float(centerline_gap - 2.0 * minor_radius)
    if surface_gap <= 0.0:
        raise ValueError(
            "linked torus surfaces overlap: require "
            "2 * minor_radius < major_radius * (sqrt(2) - 1)",
        )

    rng = np.random.default_rng(seed)
    ambient_dim = 3 + max(extrusion_dim - 1, 0)
    if extrusion_sigma is None:
        tube_sigma = float(noise / np.sqrt(max(extrusion_dim, 1)))
    else:
        tube_sigma = float(extrusion_sigma)
    effective_noise_variance = (
        noise**2 if extrusion_dim == 0 else extrusion_dim * tube_sigma**2
    )
    lambda_half_radius = float(
        3.0 * tube_sigma * np.sqrt(2.0 * np.log(2.0)),
    )
    lambda_half_gap = float(surface_gap - 2.0 * lambda_half_radius)

    identity = np.eye(3)
    hopf_rotation = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    )
    component1 = TorusSurfaceFadedComponent(
        major_radius=major_radius,
        minor_radius=minor_radius,
        sigma=tube_sigma,
        transition_radius=3.0,
        center=np.zeros(ambient_dim),
        rotation=identity,
        weight=0.5,
    )
    component2 = TorusSurfaceFadedComponent(
        major_radius=major_radius,
        minor_radius=minor_radius,
        sigma=tube_sigma,
        transition_radius=3.0,
        center=np.r_[major_radius, np.zeros(ambient_dim - 1)],
        rotation=hopf_rotation,
        weight=0.5,
    )
    torus1 = component1.sample(n_per_torus, np.random.default_rng(seed + 17))
    torus2 = component2.sample(n_per_torus, np.random.default_rng(seed + 23))
    signal_points = np.vstack([torus1, torus2])
    support = SupportBox.from_points(
        signal_points,
        padding_fraction=max(0.05, tissue_fraction),
        min_padding=0.05,
        extra_padding=3.0 * tube_sigma,
    )
    mixture = FadedMixture(
        [component1, component2], support, tissue_mass=tissue_mass,
    )
    points, sampler_meta = sample_faded_mixture(mixture, 2 * n_per_torus, rng)
    labels = assign_labels_by_lambda(points, [component1, component2], label_offsets=[0, 1])

    torus_topo = TopologyExpectation(
        connected_components=1, betti_numbers=(1, 2, 1), intrinsic_dim=2,
    )
    surface_area = 4.0 * (np.pi ** 2) * major_radius * minor_radius
    expected_knn_radius_k8 = float(np.sqrt(
        8.0 * surface_area / (np.pi * n_per_torus),
    ))
    signal_tau = expected_tau_for_surface(
        surface_area=surface_area,
        target_n_nodes=target_n_nodes,
        noise_variance=effective_noise_variance,
    )
    tissue_bounds = support.bounds
    tissue_tau = expected_tau_for_uniform_tissue_box(
        tissue_bounds,
        target_n_nodes=target_n_nodes,
        noise_variance=effective_noise_variance,
    )
    expected_tau = max(signal_tau, tissue_tau)
    ideal_nodes = int(np.ceil(max(
        2.0 * ideal_nodes_for_surface(
            surface_area=surface_area,
            tau=expected_tau,
            noise_variance=effective_noise_variance,
        ),
        ideal_nodes_for_uniform_tissue_box(
            tissue_bounds,
            tau=expected_tau,
            noise_variance=effective_noise_variance,
        ),
    )))

    clusters = [
        ClusterNode(
            cluster_id=0, level=0, parent_id=None, weight=1.0,
            center=signal_points.mean(axis=0),
            covariance=np.cov(signal_points, rowvar=False),
            is_leaf=False, intrinsic_dim=2,
        ),
        ClusterNode(
            cluster_id=1, level=1, parent_id=0, weight=0.5,
            center=torus1.mean(axis=0), covariance=np.cov(torus1, rowvar=False),
            is_leaf=True, intrinsic_dim=2,
        ),
        ClusterNode(
            cluster_id=2, level=1, parent_id=0, weight=0.5,
            center=torus2.mean(axis=0), covariance=np.cov(torus2, rowvar=False),
            is_leaf=True, intrinsic_dim=2,
        ),
    ]

    gt = GroundTruthManifold(
        name="linked_tori",
        ambient_dim=ambient_dim,
        intrinsic_dim=2,
        expected_scale_levels=2,
        cluster_hierarchy=clusters,
        topology=TopologyExpectation(
            connected_components=2, betti_numbers=(2, 4, 2), intrinsic_dim=2,
        ),
        per_component_topology=[torus_topo, torus_topo],
        expected_tau=expected_tau,
        expected_node_count=ideal_nodes,
        node_count_upper_bound=3 * ideal_nodes,
        noise_variance=effective_noise_variance,
        tau_grid_hint=(min(signal_tau, tissue_tau) / 8.0, max(signal_tau, tissue_tau) * 8.0),
    )
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=gt,
        metadata={
            "extrusion_dim": extrusion_dim,
            "extrusion_sigma": tube_sigma if extrusion_dim > 0 else 0.0,
            "signal_expected_tau": float(signal_tau),
            "tissue_expected_tau": float(tissue_tau),
            "support_bounds_lo": tissue_bounds[0].tolist(),
            "support_bounds_hi": tissue_bounds[1].tolist(),
            "sampling": "continuous_area_uniform",
            "centerline_gap": centerline_gap,
            "surface_gap": surface_gap,
            "lambda_half_radius": lambda_half_radius,
            "lambda_half_gap": lambda_half_gap,
            "expected_knn_radius_k8": expected_knn_radius_k8,
            "resolvable_k8": bool(
                lambda_half_gap > 0.0
                and expected_knn_radius_k8 < lambda_half_gap
            ),
            **sampler_meta,
            **tissue_mass_metadata(
                tissue_fraction=tissue_fraction,
                tissue_mass=tissue_mass,
                labels=labels,
            ),
        },
    )
