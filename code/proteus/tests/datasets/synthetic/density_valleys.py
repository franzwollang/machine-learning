"""Connected-support density-valley scenes (OPEN_ISSUES #44 / #48).

These isolate Hartigan valleys from support disconnection:

* ``make_bimodal_circle`` — one circle, two angular modes. Topology is
  still one connected component; a superlevel set disconnects into two arcs.
* ``make_two_gaussians`` — two isotropic bumps whose valley depth is
  controlled by center separation in units of ``sigma``.
"""

from __future__ import annotations

import numpy as np

from ..ground_truth import (
    ClusterNode,
    GroundTruthManifold,
    SyntheticDataset,
    TopologyExpectation,
    expected_tau_for_arc,
)
from .faded_density import (
    BimodalCircleFadedComponent,
    FadedMixture,
    GaussianFadedComponent,
    SupportBox,
    assign_labels_by_lambda,
    sample_faded_mixture,
    tissue_mass_metadata,
)
from .tissue import expected_tau_for_uniform_tissue_box


def make_bimodal_circle(
    n_samples: int = 1500,
    radius: float = 1.0,
    noise: float = 0.02,
    kappa: float = 3.0,
    target_n_nodes: int = 32,
    extrusion_dim: int = 2,
    tissue_fraction: float = 0.03,
    tissue_mass: float | None = None,
    seed: int = 0,
    transition_radius: float = 3.0,
) -> SyntheticDataset:
    """Connected circle with a von Mises angular valley (should split).

    ``tissue_fraction`` only pads the support box (historical name).
    Pass ``tissue_mass`` for an honest expected λ<0.5 background fraction;
    ``None`` keeps the legacy fade-balanced floor (~46–49% tissue).
    """

    if extrusion_dim < 0:
        raise ValueError("extrusion_dim must be non-negative")
    rng = np.random.default_rng(seed)
    tube_sigma = float(noise / np.sqrt(max(extrusion_dim, 1)))
    ambient_dim = 2 + max(extrusion_dim - 1, 0)
    component = BimodalCircleFadedComponent(
        radius=radius,
        sigma=tube_sigma,
        transition_radius=transition_radius,
        center=np.zeros(ambient_dim),
        kappa=float(kappa),
    )
    manifold_angles = np.linspace(0.0, 2.0 * np.pi, num=128, endpoint=False)
    manifold_points = np.zeros((128, ambient_dim), dtype=float)
    manifold_points[:, 0] = radius * np.cos(manifold_angles)
    manifold_points[:, 1] = radius * np.sin(manifold_angles)
    support = SupportBox.from_points(
        manifold_points,
        padding_fraction=max(0.05, tissue_fraction),
        min_padding=0.05,
        extra_padding=3.0 * tube_sigma,
    )
    mixture = FadedMixture(
        components=[component], support=support, tissue_mass=tissue_mass,
    )
    points, sampler_meta = sample_faded_mixture(mixture, n_samples, rng)
    fade = component.fade_weight(points)
    labels = component.mode_labels(points)
    labels[fade < 0.5] = -1
    signal = labels >= 0
    signal_points = points[signal] if signal.any() else component.sample(
        n_samples, np.random.default_rng(seed + 17),
    )
    effective_noise_variance = ambient_dim * tube_sigma**2
    signal_tau = expected_tau_for_arc(
        perimeter=2.0 * np.pi * radius,
        target_n_nodes=target_n_nodes,
        noise_variance=effective_noise_variance,
    )
    tissue_tau = expected_tau_for_uniform_tissue_box(
        support.bounds,
        target_n_nodes=target_n_nodes,
        noise_variance=effective_noise_variance,
    )
    expected_tau = max(signal_tau, tissue_tau)
    gt = GroundTruthManifold(
        name="bimodal_circle",
        ambient_dim=ambient_dim,
        intrinsic_dim=1,
        expected_scale_levels=2,
        cluster_hierarchy=[
            ClusterNode(
                cluster_id=0, level=0, parent_id=None, weight=1.0,
                center=signal_points.mean(axis=0),
                covariance=np.cov(signal_points, rowvar=False),
                is_leaf=False, intrinsic_dim=1,
            ),
            ClusterNode(
                cluster_id=1, level=1, parent_id=0, weight=0.5,
                center=np.array([radius, 0.0, *np.zeros(ambient_dim - 2)]),
                covariance=np.eye(ambient_dim) * tube_sigma**2,
                is_leaf=True, intrinsic_dim=1,
            ),
            ClusterNode(
                cluster_id=2, level=1, parent_id=0, weight=0.5,
                center=np.array([-radius, 0.0, *np.zeros(ambient_dim - 2)]),
                covariance=np.eye(ambient_dim) * tube_sigma**2,
                is_leaf=True, intrinsic_dim=1,
            ),
        ],
        topology=TopologyExpectation(
            connected_components=1, betti_numbers=(1, 1), intrinsic_dim=1,
        ),
        expected_tau=expected_tau,
        expected_node_count=target_n_nodes,
        noise_variance=effective_noise_variance,
        tau_grid_hint=(min(signal_tau, tissue_tau) / 8.0, max(signal_tau, tissue_tau) * 8.0),
    )
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=gt,
        metadata={
            "kappa": float(kappa),
            "mode_angles": list(component.mode_angles),
            "connected_support": True,
            "expected_k": 2,
            **tissue_mass_metadata(
                tissue_fraction=tissue_fraction,
                tissue_mass=tissue_mass,
                tissue_mass_actual=float(np.mean(labels < 0)),
            ),
            **sampler_meta,
        },
    )


def make_two_gaussians(
    n_samples: int = 800,
    sigma: float = 0.25,
    separation: float = 2.5,
    ambient_dim: int = 2,
    tissue_fraction: float = 0.03,
    tissue_mass: float | None = None,
    seed: int = 0,
    transition_radius: float = 3.0,
) -> SyntheticDataset:
    """Two isotropic Gaussians. ``separation`` is center distance / sigma.

    ``separation=2.5`` is a weak-valley control (heavy overlap).
    ``separation=6.0`` is a clear Hartigan split.

    ``tissue_fraction`` only pads the support box (historical name).
    Pass ``tissue_mass`` for an honest expected λ<0.5 background fraction;
    ``None`` keeps the legacy fade-balanced floor (~46–49% tissue).
    """

    if separation <= 0.0:
        raise ValueError("separation must be positive")
    rng = np.random.default_rng(seed)
    half = 0.5 * float(separation) * float(sigma)
    centers = np.zeros((2, ambient_dim), dtype=float)
    centers[0, 0] = -half
    centers[1, 0] = half
    components = [
        GaussianFadedComponent(
            center=centers[i],
            sigma=float(sigma),
            transition_radius=transition_radius,
            weight=0.5,
        )
        for i in range(2)
    ]
    support = SupportBox.from_points(
        centers,
        padding_fraction=max(0.2, tissue_fraction),
        min_padding=3.0 * float(sigma),
        extra_padding=3.0 * float(sigma),
    )
    mixture = FadedMixture(
        components=components, support=support, tissue_mass=tissue_mass,
    )
    points, sampler_meta = sample_faded_mixture(mixture, n_samples, rng)
    labels = assign_labels_by_lambda(points, components, label_offsets=[0, 1])
    signal = labels >= 0
    signal_points = points[signal] if signal.any() else points
    cov = np.eye(ambient_dim) * (sigma ** 2)
    gt = GroundTruthManifold(
        name="two_gaussians",
        ambient_dim=ambient_dim,
        intrinsic_dim=ambient_dim,
        expected_scale_levels=1,
        cluster_hierarchy=[
            ClusterNode(
                cluster_id=0, level=0, parent_id=None, weight=1.0,
                center=centers.mean(axis=0),
                covariance=np.cov(signal_points, rowvar=False),
                is_leaf=False, intrinsic_dim=ambient_dim,
            ),
            ClusterNode(
                cluster_id=1, level=1, parent_id=0, weight=0.5,
                center=centers[0], covariance=cov,
                is_leaf=True, intrinsic_dim=ambient_dim,
            ),
            ClusterNode(
                cluster_id=2, level=1, parent_id=0, weight=0.5,
                center=centers[1], covariance=cov,
                is_leaf=True, intrinsic_dim=ambient_dim,
            ),
        ],
        topology=TopologyExpectation(
            connected_components=1 if separation < 6.0 else 2,
            betti_numbers=(1,),
            intrinsic_dim=ambient_dim,
        ),
        expected_tau=float(sigma ** 2 * ambient_dim),
        expected_node_count=32,
        noise_variance=0.0,
        tau_grid_hint=(0.05 * sigma ** 2, 8.0 * sigma ** 2 * ambient_dim),
    )
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=gt,
        metadata={
            "sigma": float(sigma),
            "separation": float(separation),
            "center_distance": float(separation * sigma),
            "expected_k": 2,
            "valley": "weak" if separation < 4.0 else "clear",
            **tissue_mass_metadata(
                tissue_fraction=tissue_fraction,
                tissue_mass=tissue_mass,
                tissue_mass_actual=float(np.mean(labels < 0)),
            ),
            **sampler_meta,
        },
    )
