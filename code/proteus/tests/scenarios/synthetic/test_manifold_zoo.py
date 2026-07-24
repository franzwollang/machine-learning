"""Manifold-zoo scenario (OPEN_ISSUES #26).

The generator (circle + segment + plane + box meeting at dimensional
junctions) lands early as a diagnostic fixture; the mesh-quality, junction
detection, and heterogeneous simplex-dimension assertions are deferred until
the S8.4 junction detector and S4.1/S4.2 flag complex are implemented.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.datasets.synthetic.manifold_zoo import (
    LABEL_BOX,
    LABEL_CIRCLE,
    LABEL_PLANE,
    LABEL_SEGMENT,
    make_manifold_zoo,
)
from tests.harness.markers import awaiting


def test_manifold_zoo_ground_truth() -> None:
    """The fixture exposes four components of intrinsic dim {1,1,2,3}.

    Diagnostic-only: verifies the generator's per-component ground truth,
    labels, and junction metadata are internally consistent so downstream
    S8.4 / S4.2 scenario tests can rely on them.
    """
    dataset = make_manifold_zoo(seed=7)
    points = dataset.points
    labels = dataset.labels
    gt = dataset.ground_truth

    assert points.ndim == 2 and points.shape[1] == gt.ambient_dim == 3
    assert points.shape[0] == labels.shape[0]
    assert np.isfinite(points).all()

    # Every signal component plus tissue appears; no stray labels.
    present = set(int(v) for v in np.unique(labels))
    assert present <= {-1, LABEL_BOX, LABEL_PLANE, LABEL_SEGMENT, LABEL_CIRCLE}
    for label in (LABEL_BOX, LABEL_PLANE, LABEL_SEGMENT, LABEL_CIRCLE):
        assert label in present, f"component {label} produced no samples"

    # Four leaves carrying the heterogeneous intrinsic dimensions.
    assert gt.n_leaf_clusters == 4
    assert sorted(c.intrinsic_dim for c in gt.leaf_clusters) == [1, 1, 2, 3]
    assert gt.intrinsic_dim == 3

    # One connected scene whose only loop is the circle.
    assert gt.topology is not None
    assert gt.topology.connected_components == 1
    assert gt.topology.betti_numbers == (1, 1)

    # Per-component topology: only the circle carries b1 = 1.
    assert len(gt.per_component_topology) == 4
    b1_by_dim = [(t.intrinsic_dim, t.betti_numbers[1]) for t in gt.per_component_topology]
    assert (1, 1) in b1_by_dim  # the circle
    assert sum(b1 for _, b1 in b1_by_dim) == 1

    # Three dimensional junctions with the expected contrasts.
    contrasts = sorted((j.dim_low, j.dim_high) for j in gt.junctions)
    assert contrasts == [(1, 1), (1, 2), (2, 3)]
    for j in gt.junctions:
        assert j.location_hint.shape == (gt.ambient_dim,)

    # Scale metadata is well-formed and coarser than the finest signal scale.
    assert gt.expected_tau is not None and gt.expected_tau > 0.0
    assert gt.expected_node_count is not None and gt.expected_node_count > 0
    lo, hi = gt.tau_grid_hint
    assert 0.0 < lo < hi


def test_manifold_zoo_components_are_spatially_separated() -> None:
    """Component centroids are distinct and ordered along the scene's x-axis."""
    dataset = make_manifold_zoo(seed=1)
    points = dataset.points
    labels = dataset.labels
    centroids = {
        label: points[labels == label].mean(axis=0)
        for label in (LABEL_BOX, LABEL_PLANE, LABEL_SEGMENT, LABEL_CIRCLE)
    }
    xs = [centroids[label][0] for label in (LABEL_BOX, LABEL_PLANE, LABEL_SEGMENT, LABEL_CIRCLE)]
    # Box -> plane -> segment -> circle march along +x by construction.
    assert xs == sorted(xs)
    # The box is the only component with appreciable z-extent (it is solid 3D).
    box_z_extent = np.ptp(points[labels == LABEL_BOX][:, 2])
    plane_z_extent = np.ptp(points[labels == LABEL_PLANE][:, 2])
    assert box_z_extent > 5.0 * plane_z_extent


@awaiting("diagnostics.junction", si="S8.4")
def test_manifold_zoo_junction_detection() -> None:
    """Junction detector should fire at each dimensional contrast (S8.4)."""
    pytest.fail("Not implemented")


@awaiting("stage2.flag_complex", si="S4.2")
def test_manifold_zoo_heterogeneous_simplex_dimension() -> None:
    """Flag complex should assign per-patch simplex dimension {1,1,2,3} (S4.2)."""
    pytest.fail("Not implemented")
