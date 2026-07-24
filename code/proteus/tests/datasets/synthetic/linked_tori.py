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
    KernelMixtureFadedComponent,
    SupportBox,
    assign_labels_by_lambda,
    sample_faded_mixture,
)
from .tissue import (
    expected_tau_for_uniform_tissue_box,
    ideal_nodes_for_uniform_tissue_box,
)


def _sample_torus(
    n: int,
    R: float,
    r: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample uniformly on a torus with major radius R and minor radius r."""
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.stack([x, y, z], axis=1)


def make_linked_tori(
    n_per_torus: int = 1000,
    major_radius: float = 2.0,
    minor_radius: float = 0.5,
    noise: float = 0.02,
    target_n_nodes: int = 64,
    extrusion_dim: int = 1,
    extrusion_sigma: float | None = None,
    tissue_fraction: float = 0.03,
    seed: int = 0,
) -> SyntheticDataset:
    """Generate two linked thickened tori as exact faded densities."""
    if extrusion_dim < 0:
        raise ValueError("extrusion_dim must be non-negative")

    rng = np.random.default_rng(seed)
    ambient_dim = 3 + max(extrusion_dim - 1, 0)
    if extrusion_sigma is None:
        tube_sigma = float(noise / np.sqrt(max(extrusion_dim, 1)))
    else:
        tube_sigma = float(extrusion_sigma)
    effective_noise_variance = (
        noise**2 if extrusion_dim == 0 else extrusion_dim * tube_sigma**2
    )

    theta_grid = np.linspace(0.0, 2.0 * np.pi, num=24, endpoint=False)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, num=12, endpoint=False)
    theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid, indexing="ij")
    torus1_anchors = np.zeros((theta_mesh.size, ambient_dim), dtype=float)
    torus1_anchors[:, 0] = (major_radius + minor_radius * np.cos(phi_mesh.ravel())) * np.cos(theta_mesh.ravel())
    torus1_anchors[:, 1] = (major_radius + minor_radius * np.cos(phi_mesh.ravel())) * np.sin(theta_mesh.ravel())
    torus1_anchors[:, 2] = minor_radius * np.sin(phi_mesh.ravel())
    torus2_anchors = np.zeros_like(torus1_anchors)
    torus2_anchors[:, 0] = torus1_anchors[:, 2] + major_radius
    torus2_anchors[:, 1] = torus1_anchors[:, 1]
    torus2_anchors[:, 2] = torus1_anchors[:, 0]

    component1 = KernelMixtureFadedComponent(
        anchors=torus1_anchors,
        sigma=tube_sigma,
        transition_radius=3.0,
        weight=0.5,
    )
    component2 = KernelMixtureFadedComponent(
        anchors=torus2_anchors,
        sigma=tube_sigma,
        transition_radius=3.0,
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
    mixture = FadedMixture([component1, component2], support)
    points, sampler_meta = sample_faded_mixture(mixture, 2 * n_per_torus, rng)
    labels = assign_labels_by_lambda(points, [component1, component2], label_offsets=[0, 1])

    torus_topo = TopologyExpectation(
        connected_components=1, betti_numbers=(1, 2, 1), intrinsic_dim=2,
    )
    surface_area = 4.0 * (np.pi ** 2) * major_radius * minor_radius
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
            "tissue_fraction_actual": float(np.mean(labels < 0)),
            "tissue_fraction_requested": tissue_fraction,
            "support_bounds_lo": tissue_bounds[0].tolist(),
            "support_bounds_hi": tissue_bounds[1].tolist(),
            "anchor_count": int(torus1_anchors.shape[0] + torus2_anchors.shape[0]),
            **sampler_meta,
        },
    )
