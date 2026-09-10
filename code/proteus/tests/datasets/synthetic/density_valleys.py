"""Connected-support density-valley scenes (OPEN_ISSUES #44 / #48).

These isolate Hartigan valleys from support disconnection:

* ``make_bimodal_circle`` — one circle, two angular modes. Topology is
  still one connected component; a superlevel set disconnects into two arcs.
* ``make_two_gaussians`` — two isotropic bumps whose valley depth is
  controlled by center separation in units of ``sigma``.

A4-T5 (#45) also exposes an analytic resolvability oracle: valley depth
(min/peak of the known mixture density on the mode-connecting path) and
the expected / realized sample count in a density valley band.
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

# Operational: valley band is densities within this fraction of the
# (peak - valley) gap above the path minimum (proposal-path covariate).
_DEFAULT_VALLEY_BAND_REL = 0.25
_EPS = 1e-12


def _path_valley_stats(
    path_points: np.ndarray,
    density: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return peak, valley, depth=(peak-valley)/peak, min/peak on a path."""
    dens = np.asarray(density, dtype=float)
    peak = float(np.max(dens))
    valley = float(np.min(dens))
    if peak <= _EPS:
        return peak, valley, 0.0, 1.0
    depth = float((peak - valley) / peak)
    return peak, valley, depth, float(valley / peak)


def _mc_valley_band_mass(
    mixture: FadedMixture,
    in_band,
    *,
    n_mc: int = 20000,
    seed: int = 0,
) -> float:
    """Importance-sample ∫ 1_band(x) p_mix(x) via mixture proposals."""
    rng = np.random.default_rng(int(seed) + 911)
    # Proposal ≈ component mixture (+ tissue floor when tissue_mass set).
    props = mixture.draw_proposals(n_mc, rng, proposal_signal_fraction=0.85)
    in_support = mixture.support.contains(props)
    props = props[in_support]
    if props.shape[0] == 0:
        return 0.0
    target = mixture.density(props)
    proposal = mixture.proposal_density(props, proposal_signal_fraction=0.85)
    weights = target / np.maximum(proposal, _EPS)
    band = in_band(props)
    # Self-normalized importance weight for the band indicator.
    denom = float(np.sum(weights))
    if denom <= _EPS:
        return 0.0
    return float(np.sum(weights[band]) / denom)


def two_gaussians_valley_oracle(
    mixture: FadedMixture,
    centers: np.ndarray,
    points: np.ndarray,
    *,
    n_samples: int,
    band_rel: float = _DEFAULT_VALLEY_BAND_REL,
    path_n: int = 256,
    seed: int = 0,
) -> dict[str, float | int | str]:
    """Analytic valley depth + valley-band counts for two equal Gaussians."""
    c0 = np.asarray(centers[0], dtype=float)
    c1 = np.asarray(centers[1], dtype=float)
    t = np.linspace(0.0, 1.0, num=int(path_n))
    path = c0[None, :] * (1.0 - t[:, None]) + c1[None, :] * t[:, None]
    dens = mixture.signal_density(path)
    peak, valley, depth, min_over_peak = _path_valley_stats(path, dens)
    thresh = valley + float(band_rel) * max(peak - valley, 0.0)

    axis = c1 - c0
    length = float(np.linalg.norm(axis))
    if length <= _EPS:
        raise ValueError("centers must be distinct")
    u = axis / length

    def in_band(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        proj = (arr - c0[None, :]) @ u
        between = (proj >= 0.0) & (proj <= length)
        # Lateral distance: stay near the connecting segment (1σ tube).
        closest = c0[None, :] + proj[:, None] * u[None, :]
        lateral = np.linalg.norm(arr - closest, axis=1)
        # Use half the center separation as a soft tube (operational).
        tube = 0.5 * length
        dens_x = mixture.signal_density(arr)
        return between & (lateral <= tube) & (dens_x <= thresh + _EPS)

    mass = _mc_valley_band_mass(mixture, in_band, seed=seed)
    realized = int(np.sum(in_band(points)))
    return {
        "valley_depth": depth,
        "valley_min_over_peak": min_over_peak,
        "valley_peak_density": peak,
        "valley_min_density": valley,
        "valley_band_rel": float(band_rel),
        "valley_band_mass": mass,
        "valley_band_expected_count": float(n_samples) * mass,
        "valley_band_count": realized,
        "valley_path": "segment",
    }


def bimodal_circle_valley_oracle(
    mixture: FadedMixture,
    component: BimodalCircleFadedComponent,
    points: np.ndarray,
    *,
    n_samples: int,
    band_rel: float = _DEFAULT_VALLEY_BAND_REL,
    path_n: int = 512,
    seed: int = 0,
) -> dict[str, float | int | str]:
    """Analytic valley depth + valley-band counts for the bimodal circle."""
    angles = np.linspace(0.0, 2.0 * np.pi, num=int(path_n), endpoint=False)
    path = np.zeros((path_n, component.dim), dtype=float)
    path[:, 0] = component.radius * np.cos(angles)
    path[:, 1] = component.radius * np.sin(angles)
    path += component.center[None, :]
    dens = mixture.signal_density(path)
    peak, valley, depth, min_over_peak = _path_valley_stats(path, dens)
    thresh = valley + float(band_rel) * max(peak - valley, 0.0)

    mode_a, mode_b = component.mode_angles
    # Valleys sit at angular midpoints between the two modes.
    mid_plus = 0.5 * (mode_a + mode_b)
    mid_minus = mid_plus + np.pi
    valley_angles = (float(mid_plus), float(mid_minus))
    # Half-width of each valley arc: quarter of the mode gap.
    half_gap = 0.25 * abs(float(mode_b - mode_a))
    if half_gap <= _EPS:
        half_gap = 0.25 * np.pi

    def _ang_dist(a: np.ndarray, b: float) -> np.ndarray:
        return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))

    def in_band(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        theta = component._theta(arr)
        near_valley = np.zeros(arr.shape[0], dtype=bool)
        for mu in valley_angles:
            near_valley |= _ang_dist(theta, mu) <= half_gap
        # Stay near the circle tube.
        near_tube = component.distance(arr) <= 2.0 * component.sigma
        dens_x = mixture.signal_density(arr)
        return near_valley & near_tube & (dens_x <= thresh + _EPS)

    mass = _mc_valley_band_mass(mixture, in_band, seed=seed)
    realized = int(np.sum(in_band(points)))
    return {
        "valley_depth": depth,
        "valley_min_over_peak": min_over_peak,
        "valley_peak_density": peak,
        "valley_min_density": valley,
        "valley_band_rel": float(band_rel),
        "valley_band_mass": mass,
        "valley_band_expected_count": float(n_samples) * mass,
        "valley_band_count": realized,
        "valley_path": "circle",
    }


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
    valley_meta = bimodal_circle_valley_oracle(
        mixture, component, points, n_samples=n_samples, seed=seed,
    )
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
            **sampler_meta,
            **tissue_mass_metadata(
                tissue_fraction=tissue_fraction,
                tissue_mass=tissue_mass,
                labels=labels,
            ),
            **valley_meta,
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
    component_only: bool = False,
    component_index: int = 0,
) -> SyntheticDataset:
    """Two isotropic Gaussians. ``separation`` is center distance / sigma.

    ``separation=2.5`` is a weak-valley control (heavy overlap).
    ``separation=6.0`` is a clear Hartigan split.

    ``tissue_fraction`` only pads the support box (historical name).
    Pass ``tissue_mass`` for an honest expected λ<0.5 background fraction;
    ``None`` keeps the legacy fade-balanced floor (~46–49% tissue).

    ``component_only=True`` (#45 A4-T6) emits pure samples from one bump
    (``component_index`` in ``{0,1}``) with no tissue and no sibling —
    a child-sized null (``n_samples`` typically 200–500).
    """

    if separation <= 0.0:
        raise ValueError("separation must be positive")
    if component_only and tissue_mass not in (None, 0.0):
        raise ValueError("component_only forbids nonzero tissue_mass")
    if component_only and int(component_index) not in (0, 1):
        raise ValueError("component_index must be 0 or 1")
    rng = np.random.default_rng(seed)
    half = 0.5 * float(separation) * float(sigma)
    centers = np.zeros((2, ambient_dim), dtype=float)
    centers[0, 0] = -half
    centers[1, 0] = half
    cov = np.eye(ambient_dim) * (sigma ** 2)

    if component_only:
        idx = int(component_index)
        component = GaussianFadedComponent(
            center=centers[idx],
            sigma=float(sigma),
            transition_radius=transition_radius,
            weight=1.0,
        )
        points = component.sample(int(n_samples), rng)
        labels = np.zeros(int(n_samples), dtype=int)
        gt = GroundTruthManifold(
            name="two_gaussians_component_only",
            ambient_dim=ambient_dim,
            intrinsic_dim=ambient_dim,
            expected_scale_levels=1,
            cluster_hierarchy=[
                ClusterNode(
                    cluster_id=0, level=0, parent_id=None, weight=1.0,
                    center=centers[idx], covariance=cov,
                    is_leaf=True, intrinsic_dim=ambient_dim,
                ),
            ],
            topology=TopologyExpectation(
                connected_components=1,
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
                "expected_k": 1,
                "valley": "none",
                "component_only": True,
                "null_scene": True,
                "component_index": idx,
                "parent_scene": "two_gaussians",
                **tissue_mass_metadata(
                    tissue_fraction=0.0,
                    tissue_mass=0.0,
                    labels=labels,
                ),
            },
        )

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
    valley_meta = two_gaussians_valley_oracle(
        mixture, centers, points, n_samples=n_samples, seed=seed,
    )
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
            "component_only": False,
            "null_scene": False,
            **sampler_meta,
            **tissue_mass_metadata(
                tissue_fraction=tissue_fraction,
                tissue_mass=tissue_mass,
                labels=labels,
            ),
            **valley_meta,
        },
    )
