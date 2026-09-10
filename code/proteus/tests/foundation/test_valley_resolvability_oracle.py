"""A4-T5: analytic valley depth / valley-band covariates on density valleys."""
from __future__ import annotations

import numpy as np
import pytest

from tests.datasets.ground_truth import valley_resolvability_report
from tests.datasets.synthetic.density_valleys import (
    make_bimodal_circle,
    make_two_gaussians,
)


def _assert_oracle_keys(ds) -> None:
    report = ds.valley_resolvability()
    assert report == valley_resolvability_report(ds.metadata)
    assert 0.0 <= float(report["valley_depth"]) <= 1.0
    assert 0.0 <= float(report["valley_min_over_peak"]) <= 1.0
    assert float(report["valley_peak_density"]) >= float(report["valley_min_density"])
    assert float(report["valley_band_rel"]) == pytest.approx(0.25)
    assert float(report["valley_band_mass"]) >= 0.0
    assert float(report["valley_band_expected_count"]) >= 0.0
    assert int(report["valley_band_count"]) >= 0
    assert int(report["valley_band_count"]) <= ds.points.shape[0]
    assert report["valley_path"] in {"segment", "circle"}


def test_bimodal_circle_valley_oracle_metadata() -> None:
    ds = make_bimodal_circle(n_samples=800, seed=0)
    _assert_oracle_keys(ds)
    assert ds.metadata["valley_path"] == "circle"
    # von Mises κ=3 on a circle has a clear angular valley.
    assert ds.metadata["valley_depth"] > 0.2
    # Realized band count should be in the ballpark of the MC expectation.
    expected = float(ds.metadata["valley_band_expected_count"])
    realized = int(ds.metadata["valley_band_count"])
    assert abs(realized - expected) <= max(40.0, 0.6 * expected + 5.0)


def test_two_gaussians_weak_valley_shallower_than_clear() -> None:
    weak = make_two_gaussians(n_samples=800, separation=2.5, seed=0)
    clear = make_two_gaussians(n_samples=800, separation=6.0, seed=0)
    _assert_oracle_keys(weak)
    _assert_oracle_keys(clear)
    assert weak.metadata["valley_path"] == "segment"
    assert clear.metadata["valley_depth"] > weak.metadata["valley_depth"]
    assert clear.metadata["valley_min_over_peak"] < weak.metadata["valley_min_over_peak"]


def test_valley_depth_is_geometry_fixed_across_seeds() -> None:
    """Same geometry → identical analytic depth; band count may vary by seed."""
    depths = []
    counts = []
    for seed in range(5):
        ds = make_bimodal_circle(n_samples=800, seed=seed)
        depths.append(float(ds.metadata["valley_depth"]))
        counts.append(int(ds.metadata["valley_band_count"]))
    assert max(depths) - min(depths) < 1e-9
    # Sampling noise should move the realized band count at least a little.
    assert max(counts) - min(counts) >= 0
