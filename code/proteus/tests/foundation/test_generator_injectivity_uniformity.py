"""A4-T7 (#48): injectivity and area-uniformity locks for A4-owned generators.

At n=20000 with near-zero tube noise, each generator's intrinsic sampling
measure must match its declared area/arc/volume law, and the manifold
parameterization must be injective (no double-covered arcs).

A3-owned nulls (``variable_density.make_scurve_sheet``, disc/cube/ball) are
audited in COORDINATION notes only — do not edit those files from A4.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.datasets.synthetic.circles import make_circle
from tests.datasets.synthetic.density_valleys import (
    make_bimodal_circle,
    make_two_gaussians,
)
from tests.datasets.synthetic.faded_density import (
    CircleFadedComponent,
    SphereShellFadedComponent,
    SwissRollSurfaceFadedComponent,
    TorusSurfaceFadedComponent,
    arc_primitive,
)
from tests.datasets.synthetic.linked_tori import make_linked_tori
from tests.datasets.synthetic.nested_spheres import make_nested_spheres
from tests.datasets.synthetic.swiss_roll import make_swiss_roll

_N = 20_000
_BINS = 20


def _max_min_ratio(values: np.ndarray, *, lo: float, hi: float, bins: int = _BINS) -> float:
    counts, _ = np.histogram(values, bins=int(bins), range=(float(lo), float(hi)))
    return float(counts.max()) / float(max(int(counts.min()), 1))


def test_circle_generator_theta_is_area_uniform_and_injective() -> None:
    """``circles.make_circle`` / ``CircleFadedComponent``: uniform angle."""
    component = CircleFadedComponent(
        radius=1.0,
        sigma=1e-6,
        transition_radius=3.0,
        center=np.zeros(2),
    )
    points = component.sample(_N, np.random.default_rng(0))
    theta = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    assert _max_min_ratio(theta, lo=0.0, hi=2.0 * np.pi, bins=36) < 1.35
    # Injectivity of the circle param: almost all samples get distinct θ.
    assert np.unique(np.round(theta, decimals=5)).size > 19_000

    ds = make_circle(
        n_samples=_N, noise=1e-6, component_only=True, seed=1,
    )
    theta_ds = np.mod(np.arctan2(ds.points[:, 1], ds.points[:, 0]), 2.0 * np.pi)
    assert _max_min_ratio(theta_ds, lo=0.0, hi=2.0 * np.pi, bins=36) < 1.35


def test_swiss_roll_generator_is_arc_length_uniform() -> None:
    """``swiss_roll`` sheet samples by arc-length CDF × uniform height."""
    t_min = 1.5 * np.pi
    t_max = 1.5 * np.pi + 3.0 * np.pi
    height = 1.0
    component = SwissRollSurfaceFadedComponent(
        t_min=t_min,
        t_max=t_max,
        height=height,
        sigma=1e-6,
        transition_radius=3.0,
        scale=t_max,
    )
    points = component.sample(_N, np.random.default_rng(2))
    t = component._project_t(points[:, [0, 2]])
    prim = arc_primitive(t)
    u = (prim - arc_primitive(t_min)) / (
        arc_primitive(t_max) - arc_primitive(t_min)
    )
    assert _max_min_ratio(u, lo=0.0, hi=1.0) < 1.35
    y = points[:, 1] * t_max  # sample stores h / scale
    assert _max_min_ratio(y, lo=0.0, hi=height) < 1.35
    # Spiral param is injective on [t_min, t_max] (no double-covered xz).
    assert float(t.min()) >= t_min - 1e-6
    assert float(t.max()) <= t_max + 1e-6
    assert np.unique(np.round(t, decimals=4)).size > 15_000

    ds = make_swiss_roll(n_samples=800, noise=1e-6, tissue_mass=0.0, seed=0)
    assert ds.metadata["sampling"] == "continuous_area_uniform"


def test_nested_spheres_shell_is_surface_uniform() -> None:
    """``nested_spheres`` / ``SphereShellFadedComponent``: Haar on the sphere."""
    component = SphereShellFadedComponent(
        radius=1.0,
        base_dim=3,
        sigma=1e-6,
        transition_radius=3.0,
        center=np.zeros(3),
    )
    points = component.sample(_N, np.random.default_rng(3))
    radii = np.linalg.norm(points, axis=1)
    assert float(np.mean(radii)) == pytest.approx(1.0, abs=1e-4)
    # Uniform sphere: E[x_i]≈0, E[x_i^2]=R^2/3.
    assert np.max(np.abs(points.mean(axis=0))) < 0.03
    assert np.allclose((points ** 2).mean(axis=0), 1.0 / 3.0, atol=0.02)

    ds = make_nested_spheres(
        n_per_sphere=_N,
        noise=1e-6,
        component_only=True,
        component_index=0,
        seed=4,
    )
    assert np.max(np.abs(ds.points[:, :3].mean(axis=0))) < 0.03


def test_linked_tori_generator_surface_measure() -> None:
    """``linked_tori`` / torus area measure: E[cos φ]=r/(2R)."""
    component = TorusSurfaceFadedComponent(
        major_radius=2.0,
        minor_radius=0.25,
        sigma=1e-6,
        transition_radius=3.0,
        center=np.zeros(3),
        rotation=np.eye(3),
    )
    points = component.sample(_N, np.random.default_rng(5))
    planar = np.linalg.norm(points[:, :2], axis=1)
    tube = np.sqrt(np.square(planar - 2.0) + np.square(points[:, 2]))
    cos_phi = (planar - 2.0) / np.maximum(tube, 1e-12)
    assert float(np.mean(cos_phi)) == pytest.approx(0.25 / 4.0, abs=0.015)
    theta = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    assert np.unique(np.round(theta, decimals=5)).size > 19_000

    ds = make_linked_tori(n_per_torus=256, seed=0)
    assert ds.metadata["sampling"] == "continuous_area_uniform"


def test_density_valleys_two_gaussians_equal_mode_mass() -> None:
    """``density_valleys.make_two_gaussians``: equal component weights, injective centers."""
    ds = make_two_gaussians(
        n_samples=_N, separation=6.0, tissue_mass=0.0, seed=0,
    )
    n0 = int((ds.labels == 0).sum())
    n1 = int((ds.labels == 1).sum())
    # Equal mixture → mode counts within ~3% at n=20000.
    assert abs(n0 - n1) / _N < 0.03
    c0 = ds.points[ds.labels == 0].mean(axis=0)
    c1 = ds.points[ds.labels == 1].mean(axis=0)
    # Centers remain separated (no collapsed / double-covered blob).
    assert float(np.linalg.norm(c0 - c1)) > 1.0


def test_density_valleys_bimodal_circle_modes_are_balanced() -> None:
    """``make_bimodal_circle``: intentional angular non-uniformity, equal modes."""
    ds = make_bimodal_circle(n_samples=_N, tissue_mass=0.0, seed=0)
    n0 = int((ds.labels == 0).sum())
    n1 = int((ds.labels == 1).sum())
    assert abs(n0 - n1) / _N < 0.05
    theta = np.mod(
        np.arctan2(ds.points[:, 1], ds.points[:, 0]), 2.0 * np.pi,
    )
    # Modes at 0 and π: both arcs receive mass; support still covers the circle.
    near0 = float(((theta < 0.6) | (theta > 2.0 * np.pi - 0.6)).mean())
    near_pi = float(((theta > np.pi - 0.6) & (theta < np.pi + 0.6)).mean())
    assert near0 > 0.2 and near_pi > 0.2
    assert abs(near0 - near_pi) < 0.05


def test_faded_density_sheet_and_circle_primitives_are_injective() -> None:
    """``faded_density`` primitives used across A4 generators stay injective."""
    # Circle: angle domain [0, 2π) with distinct samples.
    circle = CircleFadedComponent(
        radius=1.5, sigma=1e-6, transition_radius=3.0, center=np.zeros(2),
    )
    pts = circle.sample(_N, np.random.default_rng(7))
    theta = np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2.0 * np.pi)
    assert np.unique(np.round(theta, decimals=5)).size > 19_000

    # Swiss-roll grid uniqueness: xz path has no self-intersection on default twists.
    t_min, t_max = 1.5 * np.pi, 1.5 * np.pi + 3.0 * np.pi
    scale = t_max
    t_grid = np.linspace(t_min, t_max, num=3001)
    xz = np.column_stack([t_grid * np.cos(t_grid), t_grid * np.sin(t_grid)]) / scale
    uniq = np.unique(np.round(xz, decimals=5), axis=0).shape[0]
    assert uniq == t_grid.shape[0]
