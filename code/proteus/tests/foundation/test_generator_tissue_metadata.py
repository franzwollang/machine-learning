"""A4-T4: requested vs actual tissue mass and signal counts in generator metadata."""
from __future__ import annotations

import numpy as np
import pytest

from tests.datasets.ground_truth import tissue_partition_report
from tests.datasets.synthetic.circles import make_circle
from tests.datasets.synthetic.density_valleys import (
    make_bimodal_circle,
    make_two_gaussians,
)
from tests.datasets.synthetic.linked_tori import make_linked_tori
from tests.datasets.synthetic.nested_spheres import make_nested_spheres
from tests.datasets.synthetic.swiss_roll import make_swiss_roll


def _assert_partition_metadata(ds, *, tissue_mass: float | None) -> None:
    n = int(ds.labels.shape[0])
    tissue_actual = int(np.sum(ds.labels < 0))
    signal_actual = int(n - tissue_actual)
    report = ds.tissue_partition()
    assert report == tissue_partition_report(ds.metadata)
    assert report["tissue_mass_actual"] == pytest.approx(tissue_actual / n)
    assert report["signal_count_actual"] == signal_actual
    assert report["tissue_count_actual"] == tissue_actual
    assert ds.metadata["signal_count"] == signal_actual
    assert ds.metadata["tissue_count"] == tissue_actual
    if tissue_mass is None:
        assert report["tissue_mass_requested"] is None
        assert report["signal_count_requested"] is None
        assert report["tissue_count_requested"] is None
        assert report["tissue_mass_mode"] == "legacy_fade_balanced"
    else:
        assert report["tissue_mass_requested"] == pytest.approx(tissue_mass)
        n_tissue_req = min(max(int(np.round(tissue_mass * n)), 0), n)
        assert report["tissue_count_requested"] == n_tissue_req
        assert report["signal_count_requested"] == n - n_tissue_req
        assert report["tissue_mass_mode"] == "requested_mass"
        assert abs(report["tissue_mass_actual"] - tissue_mass) <= 0.03


def test_circle_tissue_partition_metadata() -> None:
    legacy = make_circle(n_samples=800, seed=0)
    _assert_partition_metadata(legacy, tissue_mass=None)
    honest = make_circle(n_samples=800, tissue_mass=0.20, seed=1)
    _assert_partition_metadata(honest, tissue_mass=0.20)


def test_swiss_roll_tissue_partition_metadata() -> None:
    legacy = make_swiss_roll(n_samples=800, seed=0)
    _assert_partition_metadata(legacy, tissue_mass=None)
    honest = make_swiss_roll(n_samples=800, tissue_mass=0.20, seed=1)
    _assert_partition_metadata(honest, tissue_mass=0.20)


def test_linked_tori_tissue_partition_metadata() -> None:
    legacy = make_linked_tori(n_per_torus=400, seed=0)
    _assert_partition_metadata(legacy, tissue_mass=None)
    honest = make_linked_tori(n_per_torus=400, tissue_mass=0.20, seed=1)
    _assert_partition_metadata(honest, tissue_mass=0.20)


def test_nested_spheres_tissue_partition_metadata() -> None:
    legacy = make_nested_spheres(n_per_sphere=400, seed=0)
    _assert_partition_metadata(legacy, tissue_mass=None)
    honest = make_nested_spheres(n_per_sphere=400, tissue_mass=0.20, seed=1)
    _assert_partition_metadata(honest, tissue_mass=0.20)


def test_bimodal_circle_tissue_partition_metadata() -> None:
    legacy = make_bimodal_circle(n_samples=800, seed=0)
    _assert_partition_metadata(legacy, tissue_mass=None)
    honest = make_bimodal_circle(n_samples=800, tissue_mass=0.20, seed=1)
    _assert_partition_metadata(honest, tissue_mass=0.20)


def test_two_gaussians_tissue_partition_metadata() -> None:
    legacy = make_two_gaussians(n_samples=800, seed=0)
    _assert_partition_metadata(legacy, tissue_mass=None)
    honest = make_two_gaussians(n_samples=800, tissue_mass=0.20, seed=1)
    _assert_partition_metadata(honest, tissue_mass=0.20)
