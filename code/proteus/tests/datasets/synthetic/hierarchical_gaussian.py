"""Hierarchical faded-Gaussian mixture generator for Proteus tests.

Mathematical model
------------------
Each fine component i defines a *faded density*:

    f_i(x) = lambda_i(x) * N(x; mu_i, sigma_i^2 I) + (1 - lambda_i(x)) * U_Omega(x)

where:
    lambda_i(x) = exp(-0.5 * (||x - mu_i|| / (r * sigma_i))^2)
    U_Omega(x)  = 1 / Vol(B(c, R))   (uniform on coarse support ball)

The mixture at each hierarchical level is:
    p(x) = sum_i w_i * f_i(x)         (w_i = 1 / n_components)

Near mu_i the density is dominated by N(x; mu_i, sigma_i) (lambda -> 1);
far from all bumps it approaches uniform (all lambda_j -> 0) -- the
*connective tissue*.  There are no valleys or exclusion zones between
bumps; the tissue smoothly connects all components.

The parameter r (transition_radius) controls the Mahalanobis radius
at which each bump fades into the uniform background.

Sampling
--------
We sample via rejection against a mixture proposal:
    q(x) = p_g * sum_i w_i N(x; mu_i, sigma_i) + (1 - p_g) * U_Omega(x)
This yields i.i.d. draws from the exact faded density.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..ground_truth import (
    ClusterNode,
    GroundTruthManifold,
    SyntheticDataset,
    TopologyExpectation,
)
from .faded_density import (
    FadedMixture,
    GaussianFadedComponent,
    SupportBall,
    assign_labels_by_lambda,
)

_EPS = 1e-12


def _sample_faded_mixture_legacy(
    mixture: FadedMixture,
    n_samples: int,
    rng: np.random.Generator,
    max_rounds: int = 500,
) -> tuple[np.ndarray, int]:
    """Preserve the historical sampler behavior for hierarchy fixtures."""
    accepted: list[np.ndarray] = []
    total_drawn = 0
    remaining = int(n_samples)
    proposal_signal_fraction = 0.7
    envelope = None

    for _ in range(max_rounds):
        if remaining <= 0:
            break
        batch_n = max(512, 8 * remaining)
        proposals = mixture.draw_proposals(batch_n, rng, proposal_signal_fraction)
        total_drawn += batch_n

        target = mixture.density(proposals)
        proposal = mixture.proposal_density(proposals, proposal_signal_fraction)
        ratio = np.where(proposal > _EPS, target / proposal, 0.0)
        batch_max = float(np.max(ratio)) if ratio.size else 1.0
        if envelope is None:
            envelope = 1.05 * batch_max
        elif batch_max > envelope:
            envelope = 1.05 * batch_max

        accept_prob = np.minimum(ratio / max(envelope, _EPS), 1.0)
        keep = rng.random(batch_n) < accept_prob
        if keep.any():
            kept = proposals[keep][:remaining]
            accepted.append(kept)
            remaining -= kept.shape[0]

    if remaining > 0:
        accepted.append(mixture.support.sample_uniform(remaining, rng))
        total_drawn += remaining

    return np.vstack(accepted)[:n_samples], total_drawn


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

def make_hierarchical_gaussian(
    coarse_centers: Optional[np.ndarray] = None,
    coarse_covariances: Optional[list[np.ndarray]] = None,
    children_per_coarse: int = 2,
    child_spread: float = 0.3,
    fine_separation_factor: float = 2.0,
    coarse_ring_fraction: float = 0.0,
    transition_radius: float = 3.0,
    uniform_fraction: float = 0.15,
    n_samples: int = 3000,
    ambient_dim: int = 4,
    target_n_nodes_per_fine_cluster: int = 16,
    seed: int = 0,
) -> SyntheticDataset:
    """Generate a two-level hierarchical faded-Gaussian mixture.

    Samples are drawn from the exact faded density via rejection sampling.
    Labels are assigned by dominant fade weight: bump label 0..K*C-1 or
    background (-1) for connective tissue.

    Parameters
    ----------
    child_spread:
        Per-axis std of each fine Gaussian (sigma_i).
    fine_separation_factor:
        |offset| = fine_separation_factor * sigma * sqrt(D).
    transition_radius:
        Fade radius r in Mahalanobis units.  lambda_i(x) = exp(-d_M^2/(2r^2)).
    uniform_fraction:
        Legacy parameter (tissue fraction is determined by the density shape).
    """
    rng = np.random.default_rng(seed)

    if coarse_centers is None:
        K = 3
        coarse_centers = np.zeros((K, ambient_dim))
        for i in range(K):
            coarse_centers[i, min(i, ambient_dim - 1)] = 4.0 * (i + 1)
    else:
        coarse_centers = np.asarray(coarse_centers)
        K = coarse_centers.shape[0]
        ambient_dim = coarse_centers.shape[1]

    if coarse_covariances is None:
        coarse_covariances = [np.eye(ambient_dim) * 1.0 for _ in range(K)]

    child_std = child_spread * np.sqrt(ambient_dim)
    offset_magnitude = fine_separation_factor * child_std
    support_radius = offset_magnitude + 3.0 * child_spread

    # Build hierarchy metadata and faded components
    clusters: list[ClusterNode] = []
    cluster_id = 0

    root = ClusterNode(
        cluster_id=cluster_id, level=0, parent_id=None, weight=1.0,
        center=coarse_centers.mean(axis=0),
        covariance=np.eye(ambient_dim) * np.var(coarse_centers, axis=0).sum(),
        is_leaf=False, intrinsic_dim=ambient_dim,
    )
    clusters.append(root)
    cluster_id += 1

    all_group_components: list[list[GaussianFadedComponent]] = []
    fine_tau_values: list[float] = []
    child_cov = np.eye(ambient_dim) * (child_spread ** 2)

    for ci in range(K):
        coarse_id = cluster_id
        clusters.append(ClusterNode(
            cluster_id=coarse_id, level=1, parent_id=0,
            weight=1.0 / K, center=coarse_centers[ci],
            covariance=coarse_covariances[ci], is_leaf=False,
            intrinsic_dim=ambient_dim,
        ))
        cluster_id += 1

        direction = rng.normal(size=ambient_dim)
        direction /= np.linalg.norm(direction) + _EPS

        group_comps: list[GaussianFadedComponent] = []
        for fi in range(children_per_coarse):
            sign = 1.0 if fi % 2 == 0 else -1.0
            if children_per_coarse > 2:
                angle = 2.0 * np.pi * fi / children_per_coarse
                d2 = rng.normal(size=ambient_dim)
                d2 -= d2.dot(direction) * direction
                d2 /= np.linalg.norm(d2) + _EPS
                offset = (np.cos(angle) * direction + np.sin(angle) * d2) * offset_magnitude
            else:
                offset = sign * direction * offset_magnitude

            child_center = coarse_centers[ci] + offset
            fine_tau_values.append(
                float(np.trace(child_cov) / target_n_nodes_per_fine_cluster)
            )
            clusters.append(ClusterNode(
                cluster_id=cluster_id, level=2, parent_id=coarse_id,
                weight=1.0 / (K * children_per_coarse),
                center=child_center, covariance=child_cov,
                is_leaf=True, intrinsic_dim=ambient_dim,
            ))
            cluster_id += 1
            group_comps.append(GaussianFadedComponent(
                center=child_center,
                sigma=float(child_spread),
                transition_radius=float(transition_radius),
                weight=1.0 / children_per_coarse,
            ))

        all_group_components.append(group_comps)

    # Sample per coarse group
    all_points: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_drawn = 0
    samples_per_group = n_samples // K
    comp_idx_offset = 0

    for ci in range(K):
        n_group = samples_per_group if ci < K - 1 else (n_samples - samples_per_group * (K - 1))
        group_comps = all_group_components[ci]

        mixture = FadedMixture(
            components=group_comps,
            support=SupportBall(
                center=np.asarray(coarse_centers[ci], dtype=float),
                radius=float(support_radius),
            ),
        )

        pts, drawn = _sample_faded_mixture_legacy(mixture, n_group, rng)
        total_drawn += drawn
        all_points.append(pts)

        labels = assign_labels_by_lambda(
            pts,
            group_comps,
            label_offsets=[comp_idx_offset + i for i in range(children_per_coarse)],
        )
        all_labels.append(labels)
        comp_idx_offset += children_per_coarse

    points = np.vstack(all_points)
    labels = np.concatenate(all_labels)

    # Tau grid: the characteristic scale for detecting bumps above uniform.
    # sigma^2 * dim is the per-bump total variance — at this tau each bump
    # occupies ~1 scaffold resolution element.
    fine_cluster_tau = float(child_spread ** 2 * ambient_dim)

    n_tissue = int((labels < 0).sum())
    actual_tissue_frac = n_tissue / max(len(labels), 1)
    acceptance_rate = float(n_samples) / max(total_drawn, 1)

    gt = GroundTruthManifold(
        name="hierarchical_gaussian",
        ambient_dim=ambient_dim,
        intrinsic_dim=ambient_dim,
        expected_scale_levels=2,
        cluster_hierarchy=clusters,
        topology=TopologyExpectation(
            connected_components=K * children_per_coarse,
            betti_numbers=(K * children_per_coarse,),
            intrinsic_dim=ambient_dim,
        ),
        expected_tau=min(fine_tau_values),
        expected_node_count=target_n_nodes_per_fine_cluster * K * children_per_coarse,
        node_count_upper_bound=3 * target_n_nodes_per_fine_cluster * K * children_per_coarse,
        noise_variance=0.0,
        tau_grid_hint=(min(fine_tau_values) / 4.0, fine_cluster_tau),
    )
    return SyntheticDataset(
        points=points, labels=labels, ground_truth=gt,
        metadata={
            "children_per_coarse": children_per_coarse,
            "transition_radius": float(transition_radius),
            "uniform_fraction_actual": float(actual_tissue_frac),
            "support_radius": float(support_radius),
            "acceptance_rate": float(acceptance_rate),
            "total_proposal_draws": int(total_drawn),
        },
    )
