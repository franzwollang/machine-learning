"""Nested-sphere generator for Proteus tests.

Produces concentric spheres of varying intrinsic dimension embedded in
a common ambient space — a standard manifold-learning benchmark for
testing hierarchical scale separation and topology recovery.
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
    SphereShellFadedComponent,
    SupportBox,
    assign_labels_by_lambda,
    sample_faded_mixture,
)
from .tissue import (
    expected_tau_for_uniform_tissue_box,
    ideal_nodes_for_uniform_tissue_box,
)


def _sample_sphere(
    n: int, dim: int, radius: float, rng: np.random.Generator,
) -> np.ndarray:
    """Sample uniformly on a ``dim``-sphere of given radius in R^{dim+1}."""
    raw = rng.normal(size=(n, dim + 1))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return raw / norms * radius


def make_nested_spheres(
    n_per_sphere: int = 800,
    radii: tuple[float, ...] = (1.0, 2.0),
    ambient_dim: int = 3,
    noise: float = 0.02,
    target_n_nodes: int = 64,
    extrusion_dim: int = 1,
    extrusion_sigma: float | None = None,
    tissue_fraction: float = 0.03,
    seed: int = 0,
) -> SyntheticDataset:
    """Generate concentric thickened spheres as exact faded densities.

    Each sphere is a (``ambient_dim - 1``)-sphere embedded in
    ``ambient_dim`` dimensions.
    """
    if extrusion_dim < 0:
        raise ValueError("extrusion_dim must be non-negative")

    rng = np.random.default_rng(seed)
    sphere_dim = ambient_dim - 1
    total_ambient_dim = ambient_dim + max(extrusion_dim - 1, 0)
    if extrusion_sigma is None:
        shell_sigma = float(noise / np.sqrt(max(extrusion_dim, 1)))
    else:
        shell_sigma = float(extrusion_sigma)
    effective_noise_variance = (
        noise**2 if extrusion_dim == 0 else extrusion_dim * shell_sigma**2
    )

    signal_samples: list[np.ndarray] = []
    clusters: list[ClusterNode] = []
    per_comp_topo: list[TopologyExpectation] = []
    components: list[SphereShellFadedComponent] = []

    root = ClusterNode(
        cluster_id=0, level=0, parent_id=None, weight=1.0,
        center=np.zeros(total_ambient_dim),
        covariance=np.eye(total_ambient_dim) * max(radii) ** 2,
        is_leaf=False, intrinsic_dim=sphere_dim,
    )
    clusters.append(root)

    for idx, r in enumerate(radii):
        component = SphereShellFadedComponent(
            radius=r,
            base_dim=ambient_dim,
            sigma=shell_sigma,
            transition_radius=3.0,
            center=np.zeros(total_ambient_dim),
            weight=1.0 / len(radii),
        )
        components.append(component)
        pts = component.sample(n_per_sphere, np.random.default_rng(seed + 17 + idx))
        signal_samples.append(pts)

        clusters.append(ClusterNode(
            cluster_id=idx + 1, level=1, parent_id=0, weight=1.0 / len(radii),
            center=pts.mean(axis=0),
            covariance=np.cov(pts, rowvar=False),
            is_leaf=True, intrinsic_dim=sphere_dim,
        ))
        betti = [0] * (sphere_dim + 1)
        betti[0] = 1
        betti[sphere_dim] = 1
        per_comp_topo.append(TopologyExpectation(
            connected_components=1,
            betti_numbers=tuple(betti),
            intrinsic_dim=sphere_dim,
        ))

    signal_points = np.vstack(signal_samples)
    support = SupportBox.from_points(
        signal_points,
        padding_fraction=max(0.05, tissue_fraction),
        min_padding=0.05,
        extra_padding=3.0 * shell_sigma,
    )
    mixture = FadedMixture(components, support)
    points, sampler_meta = sample_faded_mixture(mixture, n_per_sphere * len(radii), rng)
    labels = assign_labels_by_lambda(
        points,
        components,
        label_offsets=[idx + 1 for idx in range(len(radii))],
    )

    global_betti = [0] * (sphere_dim + 1)
    global_betti[0] = len(radii)
    global_betti[sphere_dim] = len(radii)

    shell_taus = [
        expected_tau_for_surface(
            surface_area=4.0 * np.pi * (r ** 2),
            target_n_nodes=target_n_nodes,
            noise_variance=effective_noise_variance,
        )
        for r in radii
    ]
    signal_tau = min(shell_taus)
    tissue_bounds = support.bounds
    tissue_tau = expected_tau_for_uniform_tissue_box(
        tissue_bounds,
        target_n_nodes=target_n_nodes,
        noise_variance=effective_noise_variance,
    )
    expected_tau = max(signal_tau, tissue_tau)
    ideal_nodes = int(np.ceil(max(
        sum(
            ideal_nodes_for_surface(
                surface_area=4.0 * np.pi * (r ** 2),
                tau=expected_tau,
                noise_variance=effective_noise_variance,
            )
            for r in radii
        ),
        ideal_nodes_for_uniform_tissue_box(
            tissue_bounds,
            tau=expected_tau,
            noise_variance=effective_noise_variance,
        ),
    )))

    gt = GroundTruthManifold(
        name="nested_spheres",
        ambient_dim=total_ambient_dim,
        intrinsic_dim=sphere_dim,
        expected_scale_levels=2,
        cluster_hierarchy=clusters,
        topology=TopologyExpectation(
            connected_components=len(radii),
            betti_numbers=tuple(global_betti),
            intrinsic_dim=sphere_dim,
        ),
        per_component_topology=per_comp_topo,
        expected_tau=expected_tau,
        expected_node_count=ideal_nodes,
        node_count_upper_bound=3 * ideal_nodes,
        noise_variance=effective_noise_variance,
        tau_grid_hint=(min(signal_tau, tissue_tau) / 8.0, max(max(shell_taus), tissue_tau) * 8.0),
    )
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=gt,
        metadata={
            "extrusion_dim": extrusion_dim,
            "extrusion_sigma": shell_sigma if extrusion_dim > 0 else 0.0,
            "base_ambient_dim": ambient_dim,
            "signal_expected_tau": float(signal_tau),
            "tissue_expected_tau": float(tissue_tau),
            "tissue_fraction_actual": float(np.mean(labels < 0)),
            "tissue_fraction_requested": tissue_fraction,
            "support_bounds_lo": tissue_bounds[0].tolist(),
            "support_bounds_hi": tissue_bounds[1].tolist(),
            **sampler_meta,
        },
    )
