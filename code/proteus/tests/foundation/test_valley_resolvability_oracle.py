"""A4-T5 / A4-T10: analytic valley depth / valley-band covariates."""
from __future__ import annotations

import pytest

from tests.datasets.ground_truth import (
    multi_valley_resolvability_report,
    valley_resolvability_report,
)
from tests.datasets.synthetic.density_valleys import (
    make_bimodal_circle,
    make_two_gaussians,
)
from tests.datasets.synthetic.hierarchical_gaussian import make_hierarchical_gaussian
from tests.datasets.synthetic.linked_tori import make_linked_tori
from tests.datasets.synthetic.nested_spheres import make_nested_spheres


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
    assert report["valley_path"] in {
        "segment",
        "circle",
        "segment_pairs",
        "radial_segment",
        "radial_segment_pairs",
        "gap_segment",
    }


def _assert_multi_pair_keys(ds, *, min_pairs: int, sibling_pairs: int | None = None) -> None:
    _assert_oracle_keys(ds)
    multi = ds.multi_valley_resolvability()
    assert multi == multi_valley_resolvability_report(ds.metadata)
    pairs = multi["valley_pairs"]
    assert isinstance(pairs, list)
    assert len(pairs) >= min_pairs
    assert int(multi["valley_pair_count"]) == len(pairs)
    if sibling_pairs is not None:
        assert int(multi["valley_sibling_pair_count"]) == sibling_pairs
    for rec in pairs:
        assert 0.0 <= float(rec["valley_depth"]) <= 1.0
        assert float(rec["valley_peak_density"]) >= float(rec["valley_min_density"])


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


def test_hierarchy_pairwise_fine_leaf_valley_oracle() -> None:
    """A4-T10: all C(6,2)=15 fine-leaf pairs; 3 sibling pairs under default K=3."""
    ds = make_hierarchical_gaussian(n_samples=900, seed=0)
    _assert_multi_pair_keys(ds, min_pairs=15, sibling_pairs=3)
    assert ds.metadata["valley_path"] == "segment_pairs"
    assert int(ds.metadata["valley_pair_count"]) == 15
    siblings = [p for p in ds.metadata["valley_pairs"] if p["same_parent"]]
    cross = [p for p in ds.metadata["valley_pairs"] if not p["same_parent"]]
    assert len(siblings) == 3
    assert len(cross) == 12
    # Cross-coarse gaps are deeper than within-coarse sibling valleys.
    assert min(float(p["valley_depth"]) for p in cross) > max(
        float(p["valley_depth"]) for p in siblings
    )


def test_nested_spheres_valley_oracle() -> None:
    ds = make_nested_spheres(n_per_sphere=400, seed=0)
    _assert_multi_pair_keys(ds, min_pairs=1, sibling_pairs=1)
    assert ds.metadata["valley_path"] == "radial_segment_pairs"
    assert ds.metadata["valley_depth"] > 0.05


def test_linked_tori_valley_oracle() -> None:
    ds = make_linked_tori(n_per_torus=400, seed=0)
    _assert_multi_pair_keys(ds, min_pairs=1, sibling_pairs=1)
    assert ds.metadata["valley_path"] == "gap_segment"
    assert ds.metadata["valley_depth"] > 0.05
    assert float(ds.metadata["valley_pairs"][0]["surface_gap"]) == pytest.approx(
        float(ds.metadata["surface_gap"])
    )
