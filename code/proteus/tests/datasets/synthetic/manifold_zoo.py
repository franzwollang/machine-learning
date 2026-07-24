"""Manifold-zoo generator for Proteus tests (OPEN_ISSUES #26).

The classic Fritzke GNG benchmark: a single connected scene in ``R^3`` built
from components of four different intrinsic dimensions meeting at dimensional
junctions --- a 3D solid box, a 2D plane patch, a 1D segment, and a 1D circle.
It exercises per-patch mesh quality, ``d_final`` accuracy across a dimensional
contrast, junction detection (SI S8.4), and (via Stage 2) heterogeneous simplex
dimension (SI S4.2).

Layout (all axis-aligned, in ``R^3``)::

    box      x in [0,1], y in [0,1], z in [0,1]         (intrinsic dim 3)
    plane    x in [1,2], y in [0,1], z = 0              (intrinsic dim 2)
    segment  x in [2,3], y = 0.5,   z = 0              (intrinsic dim 1)
    circle   centre (3.5, 0.5, 0), radius 0.5, z = 0    (intrinsic dim 1)

The plane is coplanar with the box's ``z = 0`` face and abuts it along the edge
``x = 1`` (a 2<->3 junction); the segment extends from the plane's ``x = 2`` edge
(a 1<->2 junction); and the circle passes through the segment's ``x = 3`` end
point (a 1<->1 junction).  The circle contributes the scene's only loop, so the
whole zoo is one connected component with ``b_1 = 1``.

Scenario assertions (junction detection, per-patch intrinsic-dim recovery,
heterogeneous simplex dimension) are deferred until the S8.4 junction detector
and the S4.1/S4.2 flag complex land; this generator ships early as a diagnostic
fixture with full per-component ground truth (OPEN_ISSUES #26).
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
    AxisAlignedBoxFadedComponent,
    AxisAlignedSegmentFadedComponent,
    AxisAlignedSheetFadedComponent,
    CircleFadedComponent,
    FadedMixture,
    SupportBox,
    assign_labels_by_lambda,
    sample_faded_mixture,
)
from .tissue import (
    expected_tau_for_uniform_tissue_box,
    ideal_nodes_for_uniform_tissue_box,
)

# Component labels (tissue is -1).
LABEL_BOX = 0
LABEL_PLANE = 1
LABEL_SEGMENT = 2
LABEL_CIRCLE = 3


def make_manifold_zoo(
    n_box: int = 1400,
    n_plane: int = 800,
    n_segment: int = 300,
    n_circle: int = 400,
    box_size: float = 1.0,
    plane_length: float = 1.0,
    segment_length: float = 1.0,
    circle_radius: float = 0.5,
    noise: float = 0.02,
    target_n_nodes: int = 48,
    tissue_fraction: float = 0.03,
    seed: int = 0,
) -> SyntheticDataset:
    """Generate a connected circle+segment+plane+box manifold-zoo scene.

    All four components live in ``R^3``; each is drawn from an exact faded
    density and labeled by dominant fade weight.  Returns the sampled points,
    per-point component labels, and a :class:`GroundTruthManifold` carrying the
    per-component intrinsic dimensions, expected topology, and the three
    dimensional junctions.
    """
    if min(n_box, n_plane, n_segment, n_circle) <= 0:
        raise ValueError("all component sample counts must be positive")

    rng = np.random.default_rng(seed)
    ambient_dim = 3
    sigma = float(noise)

    # Component placement (see module docstring for the geometry).
    box_lo = (0.0, 0.0, 0.0)
    box_hi = (box_size, box_size, box_size)
    plane_u = (box_size, box_size + plane_length)
    plane_v = (0.0, box_size)
    segment_x = (box_size + plane_length, box_size + plane_length + segment_length)
    segment_yz = float(box_size / 2.0)
    circle_center = np.array(
        [segment_x[1] + circle_radius, segment_yz, 0.0], dtype=float,
    )

    box_component = AxisAlignedBoxFadedComponent(
        lo=box_lo, hi=box_hi, ambient_dim=ambient_dim, sigma=sigma,
        transition_radius=3.0, weight=n_box / max(n_box + n_plane + n_segment + n_circle, 1),
    )
    plane_component = AxisAlignedSheetFadedComponent(
        u_range=plane_u, v_range=plane_v, ambient_dim=ambient_dim, sigma=sigma,
        transition_radius=3.0, weight=n_plane / max(n_box + n_plane + n_segment + n_circle, 1),
    )
    segment_component = AxisAlignedSegmentFadedComponent(
        t_range=segment_x, ambient_dim=ambient_dim, sigma=sigma,
        transition_radius=3.0, offset=np.array([0.0, segment_yz, 0.0]),
        weight=n_segment / max(n_box + n_plane + n_segment + n_circle, 1),
    )
    circle_component = CircleFadedComponent(
        radius=circle_radius, sigma=sigma, transition_radius=3.0,
        center=circle_center,
        weight=n_circle / max(n_box + n_plane + n_segment + n_circle, 1),
    )
    components = [box_component, plane_component, segment_component, circle_component]

    box_pts = box_component.sample(n_box, np.random.default_rng(seed + 11))
    plane_pts = plane_component.sample(n_plane, np.random.default_rng(seed + 17))
    segment_pts = segment_component.sample(n_segment, np.random.default_rng(seed + 23))
    circle_pts = circle_component.sample(n_circle, np.random.default_rng(seed + 29))
    signal_points = np.vstack([box_pts, plane_pts, segment_pts, circle_pts])

    support = SupportBox.from_points(
        signal_points,
        padding_fraction=max(0.05, tissue_fraction),
        min_padding=0.05,
        extra_padding=3.0 * sigma,
    )
    mixture = FadedMixture(components, support)
    n_total = n_box + n_plane + n_segment + n_circle
    points, sampler_meta = sample_faded_mixture(mixture, n_total, rng)
    labels = assign_labels_by_lambda(
        points, components,
        label_offsets=[LABEL_BOX, LABEL_PLANE, LABEL_SEGMENT, LABEL_CIRCLE],
    )

    # Per-component noise variance (isotropic sigma over the normal directions).
    box_noise_var = 0.0                     # solid 3-box: no transverse fade
    plane_noise_var = 1.0 * sigma**2        # one normal direction (z)
    segment_noise_var = 2.0 * sigma**2      # two normal directions (y, z)
    circle_noise_var = 2.0 * sigma**2       # radial + one normal direction (z)

    box_tau = expected_tau_for_uniform_tissue_box(
        (np.asarray(box_lo), np.asarray(box_hi)),
        target_n_nodes=target_n_nodes, noise_variance=box_noise_var,
    )
    plane_tau = expected_tau_for_surface(
        surface_area=plane_length * box_size,
        target_n_nodes=target_n_nodes, noise_variance=plane_noise_var,
    )
    segment_tau = expected_tau_for_arc(
        perimeter=segment_length,
        target_n_nodes=target_n_nodes, noise_variance=segment_noise_var,
    )
    circle_tau = expected_tau_for_arc(
        perimeter=2.0 * np.pi * circle_radius,
        target_n_nodes=target_n_nodes, noise_variance=circle_noise_var,
    )
    signal_tau = min(box_tau, plane_tau, segment_tau, circle_tau)
    tissue_bounds = support.bounds
    tissue_tau = expected_tau_for_uniform_tissue_box(
        tissue_bounds, target_n_nodes=target_n_nodes,
        noise_variance=max(plane_noise_var, segment_noise_var, circle_noise_var),
    )
    expected_tau = max(signal_tau, tissue_tau)

    ideal_nodes = int(np.ceil(max(
        ideal_nodes_for_uniform_tissue_box(
            (np.asarray(box_lo), np.asarray(box_hi)),
            tau=expected_tau, noise_variance=box_noise_var,
        )
        + ideal_nodes_for_surface(
            surface_area=plane_length * box_size,
            tau=expected_tau, noise_variance=plane_noise_var,
        )
        + ideal_nodes_for_arc(
            perimeter=segment_length, tau=expected_tau, noise_variance=segment_noise_var,
        )
        + ideal_nodes_for_arc(
            perimeter=2.0 * np.pi * circle_radius,
            tau=expected_tau, noise_variance=circle_noise_var,
        ),
        ideal_nodes_for_uniform_tissue_box(
            tissue_bounds, tau=expected_tau,
            noise_variance=max(plane_noise_var, segment_noise_var, circle_noise_var),
        ),
    )))

    root = ClusterNode(
        cluster_id=0, level=0, parent_id=None, weight=1.0,
        center=signal_points.mean(axis=0),
        covariance=np.cov(signal_points, rowvar=False),
        is_leaf=False, intrinsic_dim=3,
    )
    leaves = [
        (LABEL_BOX, box_pts, 3, n_box),
        (LABEL_PLANE, plane_pts, 2, n_plane),
        (LABEL_SEGMENT, segment_pts, 1, n_segment),
        (LABEL_CIRCLE, circle_pts, 1, n_circle),
    ]
    cluster_hierarchy = [root]
    per_comp_topo: list[TopologyExpectation] = []
    for label, pts, d_int, n_c in leaves:
        cluster_hierarchy.append(ClusterNode(
            cluster_id=label + 1, level=1, parent_id=0, weight=n_c / n_total,
            center=pts.mean(axis=0), covariance=np.cov(pts, rowvar=False),
            is_leaf=True, intrinsic_dim=d_int,
        ))
        b1 = 1 if label == LABEL_CIRCLE else 0
        per_comp_topo.append(TopologyExpectation(
            connected_components=1, betti_numbers=(1, b1), intrinsic_dim=d_int,
        ))

    junctions = [
        JunctionExpectation(
            location_hint=np.array([box_size, segment_yz, 0.0]),
            dim_low=2, dim_high=3,
            description="2D plane meets 3D box along the x=box_size edge",
        ),
        JunctionExpectation(
            location_hint=np.array([box_size + plane_length, segment_yz, 0.0]),
            dim_low=1, dim_high=2,
            description="1D segment meets 2D plane at the x=(box+plane) edge",
        ),
        JunctionExpectation(
            location_hint=np.array([segment_x[1], segment_yz, 0.0]),
            dim_low=1, dim_high=1,
            description="1D segment meets 1D circle at the segment end point",
        ),
    ]

    gt = GroundTruthManifold(
        name="manifold_zoo",
        ambient_dim=ambient_dim,
        intrinsic_dim=3,
        expected_scale_levels=2,
        cluster_hierarchy=cluster_hierarchy,
        topology=TopologyExpectation(
            connected_components=1, betti_numbers=(1, 1), intrinsic_dim=3,
        ),
        per_component_topology=per_comp_topo,
        junctions=junctions,
        expected_tau=expected_tau,
        expected_node_count=ideal_nodes,
        node_count_upper_bound=4 * ideal_nodes,
        noise_variance=max(plane_noise_var, segment_noise_var, circle_noise_var),
        tau_grid_hint=(
            min(signal_tau, tissue_tau) / 8.0,
            max(box_tau, plane_tau, segment_tau, circle_tau, tissue_tau) * 8.0,
        ),
    )
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=gt,
        metadata={
            "box_expected_tau": float(box_tau),
            "plane_expected_tau": float(plane_tau),
            "segment_expected_tau": float(segment_tau),
            "circle_expected_tau": float(circle_tau),
            "signal_expected_tau": float(signal_tau),
            "tissue_expected_tau": float(tissue_tau),
            "tissue_fraction_actual": float(np.mean(labels < 0)),
            "tissue_fraction_requested": tissue_fraction,
            "circle_center": circle_center.tolist(),
            "support_bounds_lo": tissue_bounds[0].tolist(),
            "support_bounds_hi": tissue_bounds[1].tolist(),
            **sampler_meta,
        },
    )
