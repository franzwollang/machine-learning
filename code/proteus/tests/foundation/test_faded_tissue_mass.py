"""Honest tissue_mass control for faded-density generators (#45 / A4-T1)."""
from __future__ import annotations

import numpy as np
import pytest

from tests.datasets.synthetic.circles import make_circle
from tests.datasets.synthetic.density_valleys import make_bimodal_circle
from tests.datasets.synthetic.linked_tori import make_linked_tori
from tests.datasets.synthetic.nested_spheres import make_nested_spheres
from tests.datasets.synthetic.swiss_roll import make_swiss_roll


@pytest.mark.parametrize("tissue_mass", [0.05, 0.20, 0.46])
def test_circle_tissue_mass_matches_requested(tissue_mass: float) -> None:
    ds = make_circle(n_samples=4000, tissue_mass=tissue_mass, seed=0)
    actual = float(np.mean(ds.labels < 0))
    assert abs(actual - tissue_mass) <= 0.03
    assert ds.metadata["tissue_mass_requested"] == pytest.approx(tissue_mass)
    assert ds.metadata["tissue_mass_actual"] == pytest.approx(actual)
    assert ds.metadata["tissue_mass_mode"] == "requested_mass"
    assert ds.metadata["tissue_fraction_role"] == "support_box_padding"


def test_legacy_default_unchanged_near_half() -> None:
    """Default tissue_mass=None keeps the fade-balanced ~46–49% tissue."""
    ds = make_circle(n_samples=4000, tissue_fraction=0.03, seed=0)
    actual = float(np.mean(ds.labels < 0))
    assert 0.40 <= actual <= 0.55
    assert ds.metadata["tissue_mass_requested"] is None
    assert ds.metadata["tissue_mass_mode"] == "legacy_fade_balanced"
    assert ds.metadata["tissue_fraction_requested"] == pytest.approx(0.03)
    # Historical padding knob still does not set mass.
    ds_pad = make_circle(n_samples=2000, tissue_fraction=0.40, seed=1)
    assert abs(float(np.mean(ds_pad.labels < 0)) - 0.40) > 0.05


@pytest.mark.parametrize(
    "factory",
    [make_swiss_roll, make_linked_tori, make_nested_spheres, make_bimodal_circle],
)
def test_owned_generators_honour_tissue_mass(factory) -> None:
    kwargs = {"tissue_mass": 0.20, "seed": 2}
    if factory is make_linked_tori:
        kwargs["n_per_torus"] = 1000
    elif factory is make_nested_spheres:
        kwargs["n_per_sphere"] = 1000
    else:
        kwargs["n_samples"] = 2000
    ds = factory(**kwargs)
    actual = float(np.mean(ds.labels < 0))
    assert abs(actual - 0.20) <= 0.03
    assert ds.metadata["tissue_mass_requested"] == pytest.approx(0.20)
    assert ds.metadata["tissue_mass_actual"] == pytest.approx(actual)
