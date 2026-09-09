"""Generator locks for connected-support density-valley scenes (#44 / #48)."""

from __future__ import annotations

import numpy as np

from tests.datasets.synthetic.density_valleys import (
    make_bimodal_circle,
    make_two_gaussians,
)


def test_bimodal_circle_is_connected_with_two_signal_modes() -> None:
    ds = make_bimodal_circle(n_samples=800, seed=0)
    assert ds.ground_truth.topology is not None
    assert ds.ground_truth.topology.connected_components == 1
    signal = ds.labels[ds.labels >= 0]
    assert set(signal.tolist()) == {0, 1}
    assert ds.metadata["connected_support"] is True
    assert ds.metadata["expected_k"] == 2
    # Modes sit on opposite sides of the circle.
    pts = ds.points[ds.labels >= 0]
    labs = ds.labels[ds.labels >= 0]
    c0 = pts[labs == 0].mean(axis=0)
    c1 = pts[labs == 1].mean(axis=0)
    assert c0[0] > 0.0
    assert c1[0] < 0.0


def test_two_gaussians_weak_valley_metadata_is_frozen() -> None:
    weak = make_two_gaussians(n_samples=400, separation=2.5, seed=0)
    clear = make_two_gaussians(n_samples=400, separation=6.0, seed=0)
    assert weak.metadata["valley"] == "weak"
    assert clear.metadata["valley"] == "clear"
    assert weak.metadata["center_distance"] == 2.5 * 0.25
    assert set(weak.labels[weak.labels >= 0].tolist()) == {0, 1}
    # Weak overlap still produces two majority labels, not a single blob.
    assert (weak.labels == 0).sum() > 40
    assert (weak.labels == 1).sum() > 40
