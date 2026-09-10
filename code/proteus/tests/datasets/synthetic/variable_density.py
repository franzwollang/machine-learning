"""Variable-density manifold generator for Proteus tests.

A 2D sheet in R^3 with density varying spatially — tests whether
Proteus routes evidence correctly through regions of different local
sampling rates without over-splitting sparse regions or under-splitting
dense ones.
"""
from __future__ import annotations

import numpy as np

from ..ground_truth import (
    ClusterNode,
    DensityProfile,
    GroundTruthManifold,
    SyntheticDataset,
    TopologyExpectation,
    expected_tau_for_surface,
    ideal_nodes_for_surface,
)
from .faded_density import (
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


def make_variable_density_sheet(
    n_samples: int = 2000,
    size: float = 2.0,
    density_ratio: float = 5.0,
    noise: float = 0.01,
    target_n_nodes: int = 64,
    extrusion_dim: int = 1,
    extrusion_sigma: float | None = None,
    tissue_fraction: float = 0.03,
    seed: int = 0,
) -> SyntheticDataset:
    """Generate a variable-density sheet as an exact faded density.

    The left half of the sheet is sampled ``density_ratio`` times more
    densely than the right half.  Both halves share the same intrinsic
    dimension and topology; only the local sampling rate differs.
    """
    if extrusion_dim < 0:
        raise ValueError("extrusion_dim must be non-negative")

    rng = np.random.default_rng(seed)
    p_left = density_ratio / (density_ratio + 1.0)
    is_left = rng.random(n_samples) < p_left
    n_left = int(is_left.sum())
    n_right = n_samples - n_left
    if extrusion_sigma is None:
        tube_sigma = float(noise / np.sqrt(max(extrusion_dim, 1)))
    else:
        tube_sigma = float(extrusion_sigma)
    ambient_dim = 3 + max(extrusion_dim - 1, 0)
    effective_noise_variance = ambient_dim * tube_sigma**2

    left_component = AxisAlignedSheetFadedComponent(
        u_range=(0.0, size / 2.0),
        v_range=(0.0, size),
        ambient_dim=ambient_dim,
        sigma=tube_sigma,
        transition_radius=3.0,
        weight=p_left,
    )
    right_component = AxisAlignedSheetFadedComponent(
        u_range=(size / 2.0, size),
        v_range=(0.0, size),
        ambient_dim=ambient_dim,
        sigma=tube_sigma,
        transition_radius=3.0,
        weight=1.0 - p_left,
    )
    support_points = np.array([
        [0.0, 0.0] + [0.0] * (ambient_dim - 2),
        [size, 0.0] + [0.0] * (ambient_dim - 2),
        [0.0, size] + [0.0] * (ambient_dim - 2),
        [size, size] + [0.0] * (ambient_dim - 2),
    ], dtype=float)
    support = SupportBox.from_points(
        support_points,
        padding_fraction=max(0.05, tissue_fraction),
        min_padding=0.05,
        extra_padding=3.0 * tube_sigma,
    )
    mixture = FadedMixture([left_component, right_component], support)
    points, sampler_meta = sample_faded_mixture(mixture, n_samples, rng)
    labels = assign_labels_by_lambda(points, [left_component, right_component], label_offsets=[0, 1])
    signal_points = np.vstack([
        left_component.sample(n_left, np.random.default_rng(seed + 17)),
        right_component.sample(n_right, np.random.default_rng(seed + 23)),
    ])
    signal_tau = expected_tau_for_surface(
        surface_area=size * size / 2.0,
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
            surface_area=size * size / 2.0,
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
        name="variable_density_sheet",
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
        density_profiles=[
            DensityProfile("left_dense", relative_density=density_ratio),
            DensityProfile("right_sparse", relative_density=1.0),
            DensityProfile("tissue", relative_density=1.0 / max(density_ratio, 1.0)),
        ],
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
            "base_size": size,
            "signal_expected_tau": float(signal_tau),
            "tissue_expected_tau": float(tissue_tau),
            "tissue_fraction_actual": float(np.mean(labels < 0)),
            "tissue_fraction_requested": tissue_fraction,
            "support_bounds_lo": tissue_bounds[0].tolist(),
            "support_bounds_hi": tissue_bounds[1].tolist(),
            **sampler_meta,
        },
    )


# ---------------------------------------------------------------------------
# One-feature null geometries for φ-ceiling widen (A3-T2 / #48)
# Pure connected supports — no tissue floor — matching lone_gauss* style.
# ---------------------------------------------------------------------------


def make_uniform_disc(
    n_samples: int = 800,
    radius: float = 1.0,
    seed: int = 0,
) -> SyntheticDataset:
    """Uniform samples in a filled 2-D disc (one-feature null)."""

    rng = np.random.default_rng(int(seed))
    # Area-uniform: r = R * sqrt(U), theta ~ Unif[0, 2π).
    u = rng.random(int(n_samples))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=int(n_samples))
    r = float(radius) * np.sqrt(u)
    points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    labels = np.zeros(int(n_samples), dtype=int)
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=GroundTruthManifold(
            name="uniform_disc",
            ambient_dim=2,
            intrinsic_dim=2,
            topology=TopologyExpectation(
                connected_components=1, betti_numbers=(1, 0), intrinsic_dim=2,
            ),
        ),
        metadata={"radius": float(radius), "sampling": "area_uniform_disc"},
    )


def make_uniform_cube(
    n_samples: int = 800,
    half_extent: float = 1.0,
    dim: int = 3,
    seed: int = 0,
) -> SyntheticDataset:
    """Uniform samples in a filled axis-aligned cube (default 3-D)."""

    if int(dim) < 1:
        raise ValueError("dim must be >= 1")
    rng = np.random.default_rng(int(seed))
    lo = -float(half_extent)
    hi = float(half_extent)
    points = rng.uniform(lo, hi, size=(int(n_samples), int(dim)))
    labels = np.zeros(int(n_samples), dtype=int)
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=GroundTruthManifold(
            name=f"uniform_cube_{dim}d",
            ambient_dim=int(dim),
            intrinsic_dim=int(dim),
            topology=TopologyExpectation(
                connected_components=1,
                betti_numbers=(1,) + (0,) * int(dim),
                intrinsic_dim=int(dim),
            ),
        ),
        metadata={"half_extent": float(half_extent), "sampling": "uniform_cube"},
    )


def make_lone_gauss3d(
    n_samples: int = 800,
    sigma: float = 0.25,
    seed: int = 0,
) -> SyntheticDataset:
    """Isotropic 3-D Gaussian blob (one-feature null)."""

    rng = np.random.default_rng(int(seed))
    points = rng.normal(size=(int(n_samples), 3)) * float(sigma)
    labels = np.zeros(int(n_samples), dtype=int)
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=GroundTruthManifold(
            name="lone_gauss3d",
            ambient_dim=3,
            intrinsic_dim=3,
            topology=TopologyExpectation(
                connected_components=1, betti_numbers=(1, 0, 0), intrinsic_dim=3,
            ),
        ),
        metadata={"sigma": float(sigma)},
    )


def make_scurve_sheet(
    n_samples: int = 800,
    noise: float = 0.0,
    seed: int = 0,
) -> SyntheticDataset:
    """S-shaped 2-D sheet in R^3 (classic S-curve extruded along an axis).

    Parameterization: ``t ∈ [0, 1]`` along the S, ``s ∈ [0, 1]`` across the
    ribbon width. Area-uniform in ``(t, s)``.
    """

    rng = np.random.default_rng(int(seed))
    t = rng.random(int(n_samples))
    s = rng.random(int(n_samples))
    # Standard S-curve angle range ~ [1.5π .. 4.5π] mapped from t.
    angle = (1.5 * np.pi) * (1.0 + 2.0 * t)
    x = np.sin(angle)
    y = 2.0 * s
    z = np.sign(angle) * (np.cos(angle) - 1.0)
    points = np.column_stack([x, y, z]).astype(float)
    if float(noise) > 0.0:
        points = points + rng.normal(scale=float(noise), size=points.shape)
    labels = np.zeros(int(n_samples), dtype=int)
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=GroundTruthManifold(
            name="scurve_sheet",
            ambient_dim=3,
            intrinsic_dim=2,
            topology=TopologyExpectation(
                connected_components=1, betti_numbers=(1, 0), intrinsic_dim=2,
            ),
        ),
        metadata={"noise": float(noise), "sampling": "area_uniform_scurve_sheet"},
    )


def make_filled_ball(
    n_samples: int = 800,
    radius: float = 1.0,
    dim: int = 3,
    seed: int = 0,
) -> SyntheticDataset:
    """Uniform samples in a filled Euclidean ball (default 3-D)."""

    if int(dim) < 1:
        raise ValueError("dim must be >= 1")
    rng = np.random.default_rng(int(seed))
    # Direction ~ N(0,I), radius via U^{1/d}.
    direction = rng.normal(size=(int(n_samples), int(dim)))
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    direction = direction / norms
    u = rng.random(int(n_samples))
    r = float(radius) * (u ** (1.0 / float(dim)))
    points = direction * r[:, None]
    labels = np.zeros(int(n_samples), dtype=int)
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=GroundTruthManifold(
            name=f"filled_ball_{dim}d",
            ambient_dim=int(dim),
            intrinsic_dim=int(dim),
            topology=TopologyExpectation(
                connected_components=1,
                betti_numbers=(1,) + (0,) * int(dim),
                intrinsic_dim=int(dim),
            ),
        ),
        metadata={"radius": float(radius), "sampling": "volume_uniform_ball"},
    )
