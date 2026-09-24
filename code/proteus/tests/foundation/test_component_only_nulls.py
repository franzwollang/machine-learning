"""A4-T6: component_only null scenes (one signal component, no tissue)."""
from __future__ import annotations

import pytest

from tests.datasets.synthetic.circles import make_circle
from tests.datasets.synthetic.density_valleys import make_two_gaussians
from tests.datasets.synthetic.nested_spheres import make_nested_spheres


@pytest.mark.parametrize(
    "factory",
    [
        lambda: make_circle(n_samples=300, seed=0, component_only=True),
        lambda: make_nested_spheres(
            n_per_sphere=300, seed=0, component_only=True, component_index=0,
        ),
        lambda: make_nested_spheres(
            n_per_sphere=300, seed=1, component_only=True, component_index=1,
        ),
        lambda: make_two_gaussians(
            n_samples=400, seed=0, component_only=True, component_index=0,
        ),
        lambda: make_two_gaussians(
            n_samples=400, seed=2, component_only=True, component_index=1,
        ),
    ],
)
def test_component_only_is_null_without_tissue(factory) -> None:
    ds = factory()
    assert ds.metadata["component_only"] is True
    assert ds.metadata["null_scene"] is True
    assert ds.metadata["tissue_mass_actual"] == pytest.approx(0.0)
    assert int(ds.metadata["tissue_count_actual"]) == 0
    assert int(ds.metadata["signal_count_actual"]) == ds.points.shape[0]
    assert set(ds.labels.tolist()) == {0}
    assert 200 <= ds.points.shape[0] <= 500
    assert ds.ground_truth.n_leaf_clusters == 1


def test_component_only_rejects_nonzero_tissue_mass() -> None:
    with pytest.raises(ValueError, match="tissue_mass"):
        make_circle(n_samples=300, component_only=True, tissue_mass=0.2)
    with pytest.raises(ValueError, match="tissue_mass"):
        make_nested_spheres(n_per_sphere=300, component_only=True, tissue_mass=0.1)
    with pytest.raises(ValueError, match="tissue_mass"):
        make_two_gaussians(n_samples=300, component_only=True, tissue_mass=0.05)


def test_default_generators_are_not_component_only_nulls() -> None:
    circle = make_circle(n_samples=400, seed=0)
    nested = make_nested_spheres(n_per_sphere=200, seed=0)
    gauss = make_two_gaussians(n_samples=400, seed=0)
    assert circle.metadata.get("component_only") is False
    assert nested.metadata.get("null_scene") is False
    assert gauss.metadata.get("null_scene") is False
