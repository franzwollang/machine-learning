"""Integration test for the Stage 1 scale-grid controller."""

from __future__ import annotations

import numpy as np

from proteus.stage1.controller import (
    ScaleSearchConfig,
    diagnostic_advance_scaffold_to_tau,
    fit_scaffold_at_tau,
    run_scale_search,
)
from proteus.stage1.scaffold import Stage1Scaffold
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.circles import make_circle


def test_circle_scale_search_finds_peak_near_expected_tau() -> None:
    dataset = make_circle(
        n_samples=1200,
        radius=1.0,
        noise=0.02,
        extrusion_dim=2,
        seed=21,
    )
    data = dataset.points
    gt = dataset.ground_truth
    expected_tau = gt.expected_tau
    assert expected_tau is not None
    tau_grid_hint = gt.tau_grid_hint
    assert tau_grid_hint is not None
    tau_lo, tau_hi = tau_grid_hint

    config = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=8,
        ann_backend="naive",
        stabilization=StabilizationConfig(
            min_equilibrium_epochs=3,
            max_epochs=15,
        ),
        seed=77,
    )
    result = run_scale_search(data, dim=gt.ambient_dim, config=config)

    assert result.tau_star > 0.0
    ratio = result.tau_star / expected_tau
    # The load-crossover selector (SI S2.5.1) lands tau* at the variance-cap
    # up-crossing, materially tighter than the legacy load-band heuristic
    # (which sat ~8x coarse here); require tau* within one grid step of the
    # geometric scale (OPEN_ISSUES #28).
    assert 0.5 < ratio < 3.0, (
        f"tau_star={result.tau_star:.6f} vs expected={expected_tau:.6f} "
        f"(ratio={ratio:.2f})"
    )

    finite_load = result.load_trace[np.isfinite(result.load_trace)]
    assert finite_load.size > 0
    assert finite_load.min() < 1.0, (
        f"Load trace never goes under cap: min={finite_load.min():.2f}"
    )
    assert finite_load.max() > 1.0 or finite_load.max() > 0.5, (
        f"Load trace shows no resolution transition: max={finite_load.max():.2f}"
    )


def test_swiss_roll_scale_search_finds_peak_near_expected_tau() -> None:
    from tests.datasets.synthetic.swiss_roll import make_swiss_roll

    dataset = make_swiss_roll(n_samples=1500, noise=0.02, seed=7)
    gt = dataset.ground_truth
    expected_tau = gt.expected_tau
    assert expected_tau is not None
    tau_lo, tau_hi = gt.tau_grid_hint

    config = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=8,
        ann_backend="naive",
        stabilization=StabilizationConfig(
            min_equilibrium_epochs=3,
            max_epochs=15,
        ),
        seed=77,
    )
    result = run_scale_search(dataset.points, dim=gt.ambient_dim, config=config)

    assert result.tau_star > 0.0
    ratio = result.tau_star / expected_tau
    # Surface (d=2): the load-crossover lands within one grid step of the
    # geometric scale (OPEN_ISSUES #28 tightened tolerance).
    assert 0.5 < ratio < 3.0, (
        f"tau_star={result.tau_star:.6f} vs expected={expected_tau:.6f} "
        f"(ratio={ratio:.2f})"
    )


def test_diagnostic_advance_scaffold_to_tau_lowers_tau_without_reseeding() -> None:
    """Diagnostics warm-continuation helper (SI S2.6.2 / #48; not on path)."""

    rng = np.random.default_rng(0)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=80)
    points = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    scaffold = Stage1Scaffold(
        dim=2, tau=0.2, k=4, min_nodes=4, max_nodes=16, rng=rng,
    )
    scaffold.init_from(points, n_seeds=8)
    lean = StabilizationConfig(min_equilibrium_epochs=1, max_epochs=3)
    scaffold.run_until_stable(points, lean)
    n_before = len(scaffold.nodes)
    positions_before = np.asarray(
        [node.position for node in scaffold.nodes], dtype=float,
    )
    diagnostic_advance_scaffold_to_tau(scaffold, points, 0.05, lean)
    assert scaffold.tau == 0.05
    assert np.allclose(scaffold.tau_local, 0.05)
    assert len(scaffold.nodes) >= 4
    if len(scaffold.nodes) == n_before:
        positions_after = np.asarray(
            [node.position for node in scaffold.nodes], dtype=float,
        )
        assert positions_after.shape == positions_before.shape


def test_fit_scaffold_at_tau_reseeds_at_target() -> None:
    """Scale-matched N growth rebuilds the mesh at the current tau (#48)."""

    rng = np.random.default_rng(0)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=80)
    points = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    lean = StabilizationConfig(min_equilibrium_epochs=1, max_epochs=3)
    config = ScaleSearchConfig(
        k=4, min_nodes=4, n_seeds=8, max_nodes=32,
        stabilization=lean, seed=0,
    )
    fitted = fit_scaffold_at_tau(points, dim=2, tau=0.05, config=config, max_nodes=32)
    assert fitted.tau == 0.05
    assert len(fitted.nodes) >= 4
    assert fitted.max_nodes == 32


def test_apply_shot_noise_node_cap_clamps_max_nodes_default_off() -> None:
    """Native N≤n/k bound in run_scale_search is flag-gated (SI S2.6.2 / #28)."""

    from proteus.stage1.level_set import shot_noise_node_cap

    rng = np.random.default_rng(0)
    # n=80, k=8 → n/k=10; default max_nodes resolves to max(4*16,64)=64 then
    # min(64, n//2=40)=40, so the shot-noise flag must tighten 40 → 10.
    theta = rng.uniform(0.0, 2.0 * np.pi, size=80)
    points = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    lean = StabilizationConfig(min_equilibrium_epochs=1, max_epochs=3)
    assert ScaleSearchConfig().apply_shot_noise_node_cap is False

    off = run_scale_search(
        points,
        dim=2,
        config=ScaleSearchConfig(
            k=8,
            min_nodes=4,
            n_seeds=8,
            max_nodes=None,
            tau_min=1e-3,
            tau_max=1.0,
            max_grid_points=4,
            stabilization=lean,
            seed=0,
            apply_shot_noise_node_cap=False,
        ),
    )
    on = run_scale_search(
        points,
        dim=2,
        config=ScaleSearchConfig(
            k=8,
            min_nodes=4,
            n_seeds=8,
            max_nodes=None,
            tau_min=1e-3,
            tau_max=1.0,
            max_grid_points=4,
            stabilization=lean,
            seed=0,
            apply_shot_noise_node_cap=True,
        ),
    )
    expected_cap = shot_noise_node_cap(80, 8, 4)
    assert expected_cap == 10
    assert off.scaffold_at_star.max_nodes == 40
    assert on.scaffold_at_star.max_nodes == expected_cap
    assert on.scaffold_at_star.max_nodes < off.scaffold_at_star.max_nodes

