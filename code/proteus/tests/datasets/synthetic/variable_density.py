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


# Default flat-strip aspect (length/width) at fixed area ``3π * 2`` (A3-T9/T16).
FLAT_STRIP_DEFAULT_ASPECT: float = 3.0 * np.pi / 2.0
FLAT_STRIP_AREA: float = 3.0 * np.pi * 2.0
# Labels like 4.71 / 9.42 map onto exact 3π/2 / 3π (A3-T16 ladder).
_FLAT_STRIP_ASPECT_LABEL_ATOL: float = 0.02


def canonicalize_flat_strip_aspect(aspect: float) -> float:
    """Map near-nominal ladder labels onto exact ``3π/2`` / ``3π``."""

    a = float(aspect)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError(f"aspect_ratio must be finite and positive, got {aspect!r}")
    if np.isclose(
        a, FLAT_STRIP_DEFAULT_ASPECT, rtol=0.0, atol=_FLAT_STRIP_ASPECT_LABEL_ATOL,
    ):
        return float(FLAT_STRIP_DEFAULT_ASPECT)
    double = float(2.0 * FLAT_STRIP_DEFAULT_ASPECT)
    if np.isclose(a, double, rtol=0.0, atol=_FLAT_STRIP_ASPECT_LABEL_ATOL):
        return double
    return a


def make_scurve_sheet(
    n_samples: int = 800,
    noise: float = 0.0,
    seed: int = 0,
    *,
    curvature_radius: float = 1.0,
    aspect_ratio: float | None = None,
) -> SyntheticDataset:
    """S-shaped 2-D sheet in R^3 (classic S-curve extruded along an axis).

    Fixed centerline arc length ``L = 3π`` and ribbon width ``2``.  Finite
    ``curvature_radius`` ``R`` uses the area-uniform S-curve
    ``θ ∈ [-L/(2R), L/(2R)]``, ``x = R sin(θ)``,
    ``z = R sign(θ)(cos(θ) - 1)``, ``y ∈ [0, 2]``.  Default ``R = 1`` is
    byte-identical to the post-A3-T6 generator.  ``R = ∞`` is a flat
    strip of the same arc length and width (mechanism control / A3-T9).

    For ``R = ∞`` only, ``aspect_ratio`` (length/width) may vary while
    holding area ``L·W = 6π`` and ``n_samples`` fixed (A3-T16).  Default
    ``None`` / ``3π/2`` is byte-identical to the A3-T9 strip.  Finite-``R``
    calls ignore ``aspect_ratio``.

    Per-sample arc coordinate (distance along the centerline from the
    low-θ end) is exposed in ``metadata["arc"]``.
    """

    rng = np.random.default_rng(int(seed))
    u = rng.random(int(n_samples))
    s = rng.random(int(n_samples))
    arc_len = 3.0 * np.pi
    width = 2.0
    R = float(curvature_radius)
    if R != R or R == 0.0:  # NaN or zero
        raise ValueError("curvature_radius must be positive or +inf")
    if R < 0.0:
        raise ValueError("curvature_radius must be positive or +inf")

    if np.isinf(R):
        # Flat strip in the xy-plane. Default aspect 3π/2 keeps L=3π, W=2.
        if aspect_ratio is None:
            aspect = float(FLAT_STRIP_DEFAULT_ASPECT)
        else:
            aspect = canonicalize_flat_strip_aspect(float(aspect_ratio))
        if np.isclose(aspect, FLAT_STRIP_DEFAULT_ASPECT, rtol=0.0, atol=0.0):
            # Exact A3-T9 geometry (byte-identical points for the same seed).
            arc_len = 3.0 * np.pi
            width = 2.0
            aspect = float(FLAT_STRIP_DEFAULT_ASPECT)
        else:
            # Area-preserving ladder rung: L = sqrt(aspect * area), W = area/L.
            arc_len = float(np.sqrt(aspect * FLAT_STRIP_AREA))
            width = float(FLAT_STRIP_AREA / arc_len)
        arc = arc_len * u
        x = arc - 0.5 * arc_len
        y = width * s
        z = np.zeros(int(n_samples), dtype=float)
        theta = np.full(int(n_samples), np.nan, dtype=float)
        theta_range = (float("-inf"), float("inf"))
        sampling = "area_uniform_flat_strip"
        gt_name = "flat_strip"
    elif R == 1.0:
        # Exact post-A3-T6 path (byte-identical points for the same seed).
        theta = arc_len * (u - 0.5)
        x = np.sin(theta)
        y = width * s
        z = np.sign(theta) * (np.cos(theta) - 1.0)
        arc = theta + 0.5 * arc_len
        theta_range = (-0.5 * arc_len, 0.5 * arc_len)
        sampling = "area_uniform_scurve_sheet"
        gt_name = "scurve_sheet"
    else:
        theta = (arc_len / R) * (u - 0.5)
        x = R * np.sin(theta)
        y = width * s
        z = R * np.sign(theta) * (np.cos(theta) - 1.0)
        arc = R * (theta + 0.5 * (arc_len / R))
        theta_range = (-0.5 * arc_len / R, 0.5 * arc_len / R)
        sampling = "area_uniform_scurve_sheet"
        gt_name = "scurve_sheet"

    points = np.column_stack([x, y, z]).astype(float)
    if float(noise) > 0.0:
        points = points + rng.normal(scale=float(noise), size=points.shape)
    labels = np.zeros(int(n_samples), dtype=int)
    meta: dict = {
        "noise": float(noise),
        "sampling": sampling,
        "curvature_radius": float(R),
        "arc_length": float(arc_len),
        "width": float(width),
        "theta_range": theta_range,
        "arc": np.asarray(arc, dtype=float),
        "theta": np.asarray(theta, dtype=float),
        "width_coord": np.asarray(width * s, dtype=float),
    }
    if np.isinf(R):
        meta["aspect_ratio"] = float(aspect)
        meta["area"] = float(arc_len * width)
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=GroundTruthManifold(
            name=gt_name,
            ambient_dim=3,
            intrinsic_dim=2,
            topology=TopologyExpectation(
                connected_components=1, betti_numbers=(1, 0), intrinsic_dim=2,
            ),
        ),
        metadata=meta,
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
