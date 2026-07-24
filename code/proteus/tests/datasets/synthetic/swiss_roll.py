"""Gaussian-extruded Swiss-roll generator for Proteus tests."""
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


def make_swiss_roll(
    n_samples: int = 1500,
    height: float = 1.0,
    twists: float = 3.0,
    noise: float = 0.01,
    target_n_nodes: int = 48,
    extrusion_dim: int = 1,
    extrusion_sigma: float | None = None,
    tissue_fraction: float = 0.03,
    seed: int = 0,
    transition_radius: float = 3.0,
) -> SyntheticDataset:
    """Generate a Swiss roll as an exact faded density."""
    if extrusion_dim < 0:
        raise ValueError("extrusion_dim must be non-negative")

    rng = np.random.default_rng(seed)
    t_min = 1.5 * np.pi
    t_max = 1.5 * np.pi + twists * np.pi
    scale = t_max
    if extrusion_sigma is None:
        tube_sigma = float(noise / np.sqrt(max(extrusion_dim, 1)))
    else:
        tube_sigma = float(extrusion_sigma)
    ambient_dim = 3 + max(extrusion_dim - 1, 0)

    n_t = max(48, 4 * int(np.sqrt(target_n_nodes)))
    n_h = max(8, int(np.ceil(max(256, 8 * target_n_nodes) / n_t)))
    t_grid = np.linspace(t_min, t_max, num=n_t, endpoint=True)
    h_grid = np.linspace(0.0, height, num=n_h, endpoint=True)
    tt, hh = np.meshgrid(t_grid, h_grid, indexing="ij")
    anchors = np.zeros((tt.size, ambient_dim), dtype=float)
    anchors[:, 0] = (tt.ravel() * np.cos(tt.ravel())) / scale
    anchors[:, 1] = hh.ravel() / scale
    anchors[:, 2] = (tt.ravel() * np.sin(tt.ravel())) / scale

    component = KernelMixtureFadedComponent(
        anchors=anchors,
        sigma=tube_sigma,
        transition_radius=transition_radius,
        weight=1.0,
    )
    support = SupportBox.from_points(
        anchors,
        padding_fraction=max(0.05, tissue_fraction),
        min_padding=0.05,
        extra_padding=3.0 * tube_sigma,
    )
    mixture = FadedMixture(components=[component], support=support)
    points, sampler_meta = sample_faded_mixture(mixture, n_samples, rng)
    labels = assign_labels_by_lambda(points, [component], label_offsets=[0])
    signal_points = component.sample(n_samples, np.random.default_rng(seed + 17))
    effective_noise_variance = ambient_dim * tube_sigma**2

    def arc_primitive(value: float) -> float:
        return 0.5 * (value * np.sqrt(1.0 + value * value) + np.arcsinh(value))

    arc_length = float(arc_primitive(t_max) - arc_primitive(t_min)) / scale
    signal_tau = expected_tau_for_surface(
        surface_area=arc_length * height,
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
        ideal_nodes_for_surface(
            surface_area=arc_length * height,
            tau=expected_tau,
            noise_variance=effective_noise_variance,
        ),
        ideal_nodes_for_uniform_tissue_box(
            tissue_bounds,
            tau=expected_tau,
            noise_variance=effective_noise_variance,
        ),
    )))
    gt = GroundTruthManifold(
        name="swiss_roll",
        ambient_dim=points.shape[1],
        intrinsic_dim=2,
        expected_scale_levels=1,
        cluster_hierarchy=[
            ClusterNode(
                cluster_id=0, level=0, parent_id=None, weight=1.0,
                center=signal_points.mean(axis=0),
                covariance=np.cov(signal_points, rowvar=False),
                is_leaf=True, intrinsic_dim=2,
            ),
        ],
        topology=TopologyExpectation(
            connected_components=1, betti_numbers=(1, 0), intrinsic_dim=2,
        ),
        expected_tau=expected_tau,
        expected_node_count=ideal_nodes,
        node_count_upper_bound=3 * ideal_nodes,
        noise_variance=effective_noise_variance,
        tau_grid_hint=(min(signal_tau, tissue_tau) / 8.0, max(signal_tau, tissue_tau) * 8.0),
    )
    actual_tissue_fraction = float(np.mean(labels < 0))
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=gt,
        metadata={
            "extrusion_dim": extrusion_dim,
            "extrusion_sigma": tube_sigma if extrusion_dim > 0 else 0.0,
            "height": height,
            "twists": twists,
            "signal_expected_tau": float(signal_tau),
            "tissue_expected_tau": float(tissue_tau),
            "tissue_fraction_actual": actual_tissue_fraction,
            "tissue_fraction_requested": tissue_fraction,
            "support_bounds_lo": tissue_bounds[0].tolist(),
            "support_bounds_hi": tissue_bounds[1].tolist(),
            "anchor_count": int(anchors.shape[0]),
            **sampler_meta,
        },
    )
