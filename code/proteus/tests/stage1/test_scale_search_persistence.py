"""Persistence-based characteristic-scale selection (SI S2.6.1, S2.6.2).

These integration tests exercise the ``selector="persistence"`` path of the
scale-search controller.  The discriminating claim of SI S2.6.1/S2.6.2 is that a
uniform manifold yields no multi-cluster partition that *persists* across
adjacent scales, whereas a genuinely multi-modal region does.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from proteus.stage1.controller import (
    ScaleSearchConfig,
    _THREE_QUARTER_LOAD_SCREEN_MIN,
    _resolve_persistence_tau_index,
    run_scale_search,
)
from proteus.stage1.persistence import PersistenceConfig, PersistenceResult
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.circles import make_circle
from tests.datasets.synthetic.hierarchical_gaussian import make_hierarchical_gaussian
from tests.datasets.synthetic.swiss_roll import make_swiss_roll


def test_uniform_manifold_has_no_persistent_split() -> None:
    dataset = make_circle(
        n_samples=1200, radius=1.0, noise=0.02, extrusion_dim=2, seed=21,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint

    config = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=8,
        ann_backend="naive",
        selector="persistence",
        stabilization=StabilizationConfig(min_equilibrium_epochs=3, max_epochs=15),
        seed=77,
    )
    result = run_scale_search(dataset.points, dim=gt.ambient_dim, config=config)

    assert result.persistence_result is not None
    # A ring is a single intrinsic feature: no multi-cluster partition persists.
    assert result.persistence_result.tau_star_index is None
    assert np.all(result.persistence_result.run_lengths == 0)
    # The controller still returns a positive tau* via the legacy fallback.
    assert result.tau_star > 0.0


def test_multimodal_region_yields_persistent_split() -> None:
    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=0,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint

    config = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=12,
        min_nodes=8,
        max_nodes=128,
        ann_backend="naive",
        selector="persistence",
        stabilization=StabilizationConfig(min_equilibrium_epochs=2, max_epochs=12),
        seed=42,
    )
    result = run_scale_search(dataset.points, dim=gt.ambient_dim, config=config)

    pr = result.persistence_result
    assert pr is not None
    # Three coarse blobs -> a multi-cluster partition persists across the grid.
    assert pr.tau_star_index is not None
    idx = pr.tau_star_index
    assert pr.run_lengths[idx] >= config.persistence.min_persistence
    assert result.partition_snapshots[idx].n_clusters >= 3
    # tau* was taken from the persistence signal, not the legacy fallback.
    assert result.tau_star == result.tau_grid[idx]


def test_cold_start_recheck_rejects_genuine_hierarchy_split() -> None:
    # Regression lock for the REFUTED cold-start recheck (SI S2.6.2, #27):
    # turning it on clears the coarse-anchored candidate that the warm sweep
    # (recheck off) correctly accepts on the hierarchical Gaussian, because
    # independent cold refits of the anchor interval resolve to different levels
    # (6-way vs 3-way).  This is why the flag ships default off; the test pins
    # the mechanism's behavior so it is not silently re-enabled on the
    # acceptance path.
    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=0,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    base = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=12,
        min_nodes=8,
        max_nodes=128,
        ann_backend="naive",
        selector="persistence",
        stabilization=StabilizationConfig(min_equilibrium_epochs=2, max_epochs=12),
        seed=42,
    )

    # Flag off (default): genuine split accepted.
    off = run_scale_search(
        dataset.points, dim=gt.ambient_dim,
        config=replace(base, persistence=PersistenceConfig(cold_start_recheck=False)),
    )
    assert off.persistence_result is not None
    assert off.persistence_result.tau_star_index is not None
    assert off.persistence_result.cold_start_rejected is False

    # Flag on: the recheck over-rejects the genuine split (documented failure).
    on = run_scale_search(
        dataset.points, dim=gt.ambient_dim,
        config=replace(base, persistence=PersistenceConfig(cold_start_recheck=True)),
    )
    assert on.persistence_result is not None
    assert on.persistence_result.tau_star_index is None
    assert on.persistence_result.tau_star is None
    assert on.persistence_result.cold_start_rejected is True


def test_default_selector_ignores_persistence() -> None:
    # With the default load_crossover selector, partitions are not recorded and
    # the persistence result is absent -- existing behavior is untouched.
    dataset = make_circle(
        n_samples=800, radius=1.0, noise=0.02, extrusion_dim=2, seed=21,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    config = ScaleSearchConfig(
        tau_min=tau_lo, tau_max=tau_hi, max_grid_points=6, k=8, n_seeds=8,
        ann_backend="naive",
        stabilization=StabilizationConfig(min_equilibrium_epochs=3, max_epochs=12),
        seed=77,
    )
    result = run_scale_search(dataset.points, dim=gt.ambient_dim, config=config)
    assert result.partition_snapshots is None
    assert result.persistence_result is None
    assert result.tau_star > 0.0


def test_resolve_within_interval_none_preserves_coarse_end_tau() -> None:
    # Flag-off parity (A6-T14): default resolve_within_interval="none" must keep
    # today's coarse-end persistence tau* (OPEN_ISSUES #28). Explicit "none"
    # must match an omitted / default PersistenceConfig.
    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=0,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    base = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=12,
        min_nodes=8,
        max_nodes=128,
        ann_backend="naive",
        selector="persistence",
        stabilization=StabilizationConfig(min_equilibrium_epochs=2, max_epochs=12),
        seed=42,
    )

    default = run_scale_search(dataset.points, dim=gt.ambient_dim, config=base)
    explicit_none = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(resolve_within_interval="none"),
        ),
    )
    assert default.persistence_result is not None
    assert default.persistence_result.tau_star_index is not None
    assert default.peak_index == default.persistence_result.tau_star_index
    assert default.tau_star == default.tau_grid[default.peak_index]
    assert explicit_none.peak_index == default.peak_index
    assert explicit_none.tau_star == default.tau_star
    assert PersistenceConfig().resolve_within_interval == "none"


def test_resolve_within_interval_load_crossover_stays_in_persistent_block() -> None:
    # Prototype path (default off): when enabled, tau* is load_crossover on the
    # accepted persistent subgrid only — still inside [i_lo, i_hi], may differ
    # from the coarse-end arbiter index. Does not flip any awaiting markers.
    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=0,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    base = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=12,
        min_nodes=8,
        max_nodes=128,
        ann_backend="naive",
        selector="persistence",
        stabilization=StabilizationConfig(min_equilibrium_epochs=2, max_epochs=12),
        seed=42,
    )

    off = run_scale_search(dataset.points, dim=gt.ambient_dim, config=base)
    on = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(resolve_within_interval="load_crossover"),
        ),
    )
    assert off.persistence_result is not None
    assert on.persistence_result is not None
    i_lo = off.persistence_result.tau_star_index
    assert i_lo is not None
    # Arbiter index unchanged; only ScaleSearchResult peak may move.
    assert on.persistence_result.tau_star_index == i_lo
    run_len = int(on.persistence_result.run_lengths[i_lo])
    i_hi = i_lo + run_len - 1
    assert i_lo <= on.peak_index <= i_hi
    assert on.tau_star == on.tau_grid[on.peak_index]
    # Default path still lands at the coarse end.
    assert off.peak_index == i_lo


def test_hierarchy_within_interval_hybrid_stays_coarse_vs_expected() -> None:
    # Regression lock for A6-T16 diagnose (OPEN_ISSUES #28): on hierarchical
    # Gaussian (seed=0, max_grid=8), default/none persistence tau* lands near
    # fine_cluster_tau (coarse feature scale), while resolve_within_interval=
    # load_crossover still remains many× expected_tau (fine-leaf packing). Do
    # not flip the default; this pins the category mismatch so a future default
    # change must be intentional.
    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=0,
    )
    gt = dataset.ground_truth
    assert gt.expected_tau is not None
    fine_cluster_tau = float(gt.tau_grid_hint[1])
    tau_lo, tau_hi = gt.tau_grid_hint
    base = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=12,
        min_nodes=8,
        max_nodes=128,
        ann_backend="naive",
        selector="persistence",
        stabilization=StabilizationConfig(min_equilibrium_epochs=2, max_epochs=12),
        seed=0,
    )

    none = run_scale_search(dataset.points, dim=gt.ambient_dim, config=base)
    hybrid = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(resolve_within_interval="load_crossover"),
        ),
    )
    assert none.persistence_result is not None
    assert none.persistence_result.tau_star_index is not None
    # Coarse-end acceptance: none tracks the bump-detect / fine_cluster scale.
    assert none.tau_star / fine_cluster_tau < 1.5
    assert none.tau_star / fine_cluster_tau > 0.5
    # Hybrid may refine within the block but remains ≫ fine-leaf expected_tau.
    assert hybrid.tau_star / gt.expected_tau > 5.0
    assert PersistenceConfig().resolve_within_interval == "none"


def test_resolve_within_interval_mid_vs_coarse() -> None:
    # Experimental mid_interval probe (A6-T28): default/none stays at coarse-end
    # arbiter; mid_interval lands at the integer midpoint of the persistent
    # block and is always inside [i_lo, i_hi]. Default remains "none".
    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=0,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    base = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=12,
        min_nodes=8,
        max_nodes=128,
        ann_backend="naive",
        selector="persistence",
        stabilization=StabilizationConfig(min_equilibrium_epochs=2, max_epochs=12),
        seed=42,
    )

    coarse = run_scale_search(dataset.points, dim=gt.ambient_dim, config=base)
    mid = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(resolve_within_interval="mid_interval"),
        ),
    )
    assert coarse.persistence_result is not None
    assert mid.persistence_result is not None
    i_lo = coarse.persistence_result.tau_star_index
    assert i_lo is not None
    assert mid.persistence_result.tau_star_index == i_lo
    run_len = int(mid.persistence_result.run_lengths[i_lo])
    i_hi = i_lo + run_len - 1
    expected_mid = (i_lo + i_hi) // 2
    assert mid.peak_index == expected_mid
    assert i_lo <= mid.peak_index <= i_hi
    assert coarse.peak_index == i_lo
    if run_len >= 2:
        assert mid.peak_index >= coarse.peak_index
    assert PersistenceConfig().resolve_within_interval == "none"


def test_resolve_within_interval_fine_end_vs_mid_coarse() -> None:
    # Experimental fine_end_of_block probe (A6-T31): lands at i_hi of the
    # accepted persistent block; contrasts vs mid_interval and coarse-end
    # (none). Default remains "none".
    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=0,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    base = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=12,
        min_nodes=8,
        max_nodes=128,
        ann_backend="naive",
        selector="persistence",
        stabilization=StabilizationConfig(min_equilibrium_epochs=2, max_epochs=12),
        seed=42,
    )

    coarse = run_scale_search(dataset.points, dim=gt.ambient_dim, config=base)
    mid = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(resolve_within_interval="mid_interval"),
        ),
    )
    fine = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(resolve_within_interval="fine_end_of_block"),
        ),
    )
    assert coarse.persistence_result is not None
    assert fine.persistence_result is not None
    i_lo = coarse.persistence_result.tau_star_index
    assert i_lo is not None
    assert fine.persistence_result.tau_star_index == i_lo
    run_len = int(fine.persistence_result.run_lengths[i_lo])
    i_hi = i_lo + run_len - 1
    assert fine.peak_index == i_hi
    assert i_lo <= fine.peak_index <= i_hi
    assert coarse.peak_index == i_lo
    assert mid.peak_index == (i_lo + i_hi) // 2
    if run_len >= 2:
        assert fine.peak_index >= mid.peak_index >= coarse.peak_index
    if run_len >= 3:
        assert fine.peak_index > mid.peak_index
    assert PersistenceConfig().resolve_within_interval == "none"


def test_resolve_within_interval_three_quarter_vs_mid_fine() -> None:
    # Experimental three_quarter_interval probe (A6-T34): lands 3/4 of the way
    # from i_lo toward i_hi; sits between mid_interval and fine_end_of_block.
    # Default remains "none".
    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=0,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    base = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=12,
        min_nodes=8,
        max_nodes=128,
        ann_backend="naive",
        selector="persistence",
        stabilization=StabilizationConfig(min_equilibrium_epochs=2, max_epochs=12),
        seed=42,
    )

    coarse = run_scale_search(dataset.points, dim=gt.ambient_dim, config=base)
    mid = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(resolve_within_interval="mid_interval"),
        ),
    )
    three_q = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(
                resolve_within_interval="three_quarter_interval"
            ),
        ),
    )
    fine = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(resolve_within_interval="fine_end_of_block"),
        ),
    )
    assert coarse.persistence_result is not None
    assert three_q.persistence_result is not None
    i_lo = coarse.persistence_result.tau_star_index
    assert i_lo is not None
    assert three_q.persistence_result.tau_star_index == i_lo
    run_len = int(three_q.persistence_result.run_lengths[i_lo])
    i_hi = i_lo + run_len - 1
    expected_tq = i_lo + (3 * (i_hi - i_lo)) // 4
    assert three_q.peak_index == expected_tq
    assert i_lo <= three_q.peak_index <= i_hi
    assert mid.peak_index == (i_lo + i_hi) // 2
    assert fine.peak_index == i_hi
    if run_len >= 2:
        assert (
            fine.peak_index
            >= three_q.peak_index
            >= mid.peak_index
            >= coarse.peak_index
        )
    if run_len >= 4:
        assert fine.peak_index > three_q.peak_index >= mid.peak_index
    assert PersistenceConfig().resolve_within_interval == "none"


def test_phi_within_interval_landing_vs_hierarchy_expected_tau() -> None:
    # Diagnostic probe (A6-T32/T35): correlate within-interval landing modes with
    # Phi_C(tau*) and hierarchy fine-leaf expected_tau. Reports a small table;
    # pins that load_crossover hybrid stays ≫ expected_tau while mid/3q/fine
    # refine (fine_end and often three_quarter can undershoot). Does not flip
    # acceptance defaults.
    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=0,
    )
    gt = dataset.ground_truth
    assert gt.expected_tau is not None
    fine_cluster_tau = float(gt.tau_grid_hint[1])
    tau_lo, tau_hi = gt.tau_grid_hint
    base = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=12,
        min_nodes=8,
        max_nodes=128,
        ann_backend="naive",
        selector="persistence",
        stabilization=StabilizationConfig(min_equilibrium_epochs=2, max_epochs=12),
        seed=0,
    )
    modes = (
        "none",
        "mid_interval",
        "three_quarter_interval",
        "fine_end_of_block",
        "load_crossover",
    )
    rows: list[dict[str, float | int | str]] = []
    for mode in modes:
        result = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=replace(
                base,
                persistence=PersistenceConfig(resolve_within_interval=mode),  # type: ignore[arg-type]
            ),
        )
        assert result.persistence_result is not None
        phi_star = float(result.phi_trace[result.peak_index])
        rows.append(
            {
                "mode": mode,
                "peak_index": int(result.peak_index),
                "tau_star": float(result.tau_star),
                "phi_star": phi_star,
                "tau_over_expected": float(result.tau_star / gt.expected_tau),
                "tau_over_fine_cluster": float(result.tau_star / fine_cluster_tau),
            }
        )

    # Human-readable diagnostic table (pytest -s).
    header = (
        f"{'mode':22s} {'idx':>3s} {'tau*':>10s} {'Phi*':>10s} "
        f"{'tau*/E[tau]':>12s} {'tau*/fine':>10s}"
    )
    print("\nA6-T35 Phi within-interval landing vs hierarchy expected_tau")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['mode']:22s} {row['peak_index']:3d} {row['tau_star']:10.4g} "
            f"{row['phi_star']:10.4g} {row['tau_over_expected']:12.3f} "
            f"{row['tau_over_fine_cluster']:10.3f}"
        )

    by_mode = {str(r["mode"]): r for r in rows}
    # Coarse-end (none) still tracks the bump-detect / fine_cluster scale.
    assert 0.5 < float(by_mode["none"]["tau_over_fine_cluster"]) < 1.5
    # load_crossover hybrid remains many× fine-leaf expected_tau (known #28 gap).
    assert float(by_mode["load_crossover"]["tau_over_expected"]) > 5.0
    # mid_interval refines toward expected_tau but stays above it on this fixture.
    assert float(by_mode["mid_interval"]["tau_star"]) < float(by_mode["none"]["tau_star"])
    assert float(by_mode["mid_interval"]["tau_over_expected"]) > 1.0
    # three_quarter sits between mid and fine_end on the grid (and in tau*).
    assert float(by_mode["three_quarter_interval"]["tau_star"]) <= float(
        by_mode["mid_interval"]["tau_star"]
    )
    assert float(by_mode["three_quarter_interval"]["tau_star"]) >= float(
        by_mode["fine_end_of_block"]["tau_star"]
    )
    # fine_end_of_block is the finest landing and can undershoot expected_tau
    # (overshoot risk — not an acceptance candidate without SI justification).
    assert float(by_mode["fine_end_of_block"]["tau_star"]) <= float(
        by_mode["three_quarter_interval"]["tau_star"]
    )
    assert float(by_mode["fine_end_of_block"]["tau_over_expected"]) < 1.0
    # Grid-index ordering: none ≤ mid ≤ three_quarter ≤ fine_end.
    assert int(by_mode["fine_end_of_block"]["peak_index"]) >= int(
        by_mode["three_quarter_interval"]["peak_index"]
    )
    assert int(by_mode["three_quarter_interval"]["peak_index"]) >= int(
        by_mode["mid_interval"]["peak_index"]
    )
    assert int(by_mode["mid_interval"]["peak_index"]) >= int(by_mode["none"]["peak_index"])
    # Phi at landing is finite for every mode (diagnostic usability).
    for mode in modes:
        assert np.isfinite(float(by_mode[mode]["phi_star"]))
    assert PersistenceConfig().resolve_within_interval == "none"


def test_phi_within_interval_landing_circle_and_swiss() -> None:
    # Diagnostic probe (A6-T37): same within-interval Phi/tau* table on
    # circle and swiss-roll manifolds (no persistent multi-cluster split).
    # All resolve_within_interval modes must agree (controller falls back to
    # load_crossover when persistence rejects), and default stays "none".
    modes = (
        "none",
        "mid_interval",
        "three_quarter_interval",
        "fine_end_of_block",
        "load_crossover",
    )
    fixtures = (
        (
            "circle",
            make_circle(
                n_samples=800, radius=1.0, noise=0.02, extrusion_dim=2, seed=21,
            ),
        ),
        ("swiss", make_swiss_roll(n_samples=800, seed=0)),
    )
    for name, dataset in fixtures:
        gt = dataset.ground_truth
        assert gt.expected_tau is not None
        tau_lo, tau_hi = gt.tau_grid_hint
        base = ScaleSearchConfig(
            tau_min=tau_lo,
            tau_max=tau_hi,
            max_grid_points=8,
            k=8,
            n_seeds=12,
            min_nodes=8,
            max_nodes=128,
            ann_backend="naive",
            selector="persistence",
            stabilization=StabilizationConfig(
                min_equilibrium_epochs=2, max_epochs=12,
            ),
            seed=0,
        )
        rows: list[dict[str, float | int | str]] = []
        for mode in modes:
            result = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=replace(
                    base,
                    persistence=PersistenceConfig(resolve_within_interval=mode),  # type: ignore[arg-type]
                ),
            )
            assert result.persistence_result is not None
            # Uniform / developable manifolds: no accepted persistent split.
            assert result.persistence_result.tau_star_index is None
            rows.append(
                {
                    "mode": mode,
                    "peak_index": int(result.peak_index),
                    "tau_star": float(result.tau_star),
                    "phi_star": float(result.phi_trace[result.peak_index]),
                    "load_star": float(result.load_trace[result.peak_index]),
                    "tau_over_expected": float(result.tau_star / gt.expected_tau),
                }
            )

        header = (
            f"{'mode':22s} {'idx':>3s} {'tau*':>10s} {'Phi*':>10s} "
            f"{'load*':>8s} {'tau*/E[tau]':>12s}"
        )
        print(f"\nA6-T37 Phi within-interval landing vs {name} expected_tau")
        print(header)
        print("-" * len(header))
        for row in rows:
            print(
                f"{row['mode']:22s} {row['peak_index']:3d} {row['tau_star']:10.4g} "
                f"{row['phi_star']:10.4g} {row['load_star']:8.3f} "
                f"{row['tau_over_expected']:12.3f}"
            )

        # No persistent block ⇒ within-interval modes cannot diverge.
        peak_indices = {int(r["peak_index"]) for r in rows}
        tau_stars = {float(r["tau_star"]) for r in rows}
        assert len(peak_indices) == 1
        assert len(tau_stars) == 1
        for row in rows:
            assert np.isfinite(float(row["phi_star"]))
            assert float(row["tau_star"]) > 0.0
    assert PersistenceConfig().resolve_within_interval == "none"


def test_three_quarter_load_screened_rejects_low_load_and_matches_raw_when_ok() -> None:
    # Experimental probe (A6-T38): three_quarter_load_screened falls back to
    # coarse-end when load at the three-quarter index is ≪ 1; otherwise matches
    # raw three_quarter_interval. Default resolve_within_interval stays "none".
    # Synthetic unit: low load at candidate → reject to i_lo.
    run_lengths = np.array([8, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    low_load = np.array([0.2, 0.3, 0.35, 0.4, 0.42, 0.45, 0.48, 0.49])
    high_load = np.array([0.7, 2.0, 4.0, 8.0, 12.0, 19.0, 30.0, 50.0])
    stabilized = [True] * 8
    pers = PersistenceResult(
        tau_star=1.0,
        tau_star_index=0,
        run_lengths=run_lengths,
        match_overlaps=np.ones(7),
    )
    cfg_screened = PersistenceConfig(
        resolve_within_interval="three_quarter_load_screened",
    )
    cfg_raw = PersistenceConfig(resolve_within_interval="three_quarter_interval")
    # i_lo=0, i_hi=7 → three_quarter index = 0 + 21//4 = 5.
    assert _THREE_QUARTER_LOAD_SCREEN_MIN == 0.5
    assert float(low_load[5]) < _THREE_QUARTER_LOAD_SCREEN_MIN
    assert float(high_load[5]) >= _THREE_QUARTER_LOAD_SCREEN_MIN
    assert (
        _resolve_persistence_tau_index(pers, low_load, stabilized, cfg_screened)
        == 0
    )
    assert (
        _resolve_persistence_tau_index(pers, low_load, stabilized, cfg_raw) == 5
    )
    assert (
        _resolve_persistence_tau_index(pers, high_load, stabilized, cfg_screened)
        == 5
    )
    assert (
        _resolve_persistence_tau_index(pers, high_load, stabilized, cfg_raw) == 5
    )

    # Integration contrast on hierarchy: three_quarter lands with load ≫ 1, so
    # the ≪1 screen does not fire — screened matches raw (undershoot is not a
    # low-load artifact).
    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=0,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    base = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=12,
        min_nodes=8,
        max_nodes=128,
        ann_backend="naive",
        selector="persistence",
        stabilization=StabilizationConfig(min_equilibrium_epochs=2, max_epochs=12),
        seed=0,
    )
    raw = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(
                resolve_within_interval="three_quarter_interval"
            ),
        ),
    )
    screened = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(
                resolve_within_interval="three_quarter_load_screened"
            ),
        ),
    )
    coarse = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(resolve_within_interval="none"),
        ),
    )
    assert raw.persistence_result is not None
    assert raw.persistence_result.tau_star_index is not None
    load_at_raw = float(raw.load_trace[raw.peak_index])
    print(
        "\nA6-T38 load-screened three_quarter contrast "
        f"(raw_load={load_at_raw:.3f}, screen_min={_THREE_QUARTER_LOAD_SCREEN_MIN})"
    )
    print(
        f"  raw idx={raw.peak_index} tau*={raw.tau_star:.4g} "
        f"screened idx={screened.peak_index} tau*={screened.tau_star:.4g} "
        f"coarse idx={coarse.peak_index} tau*={coarse.tau_star:.4g}"
    )
    assert load_at_raw >= _THREE_QUARTER_LOAD_SCREEN_MIN
    assert screened.peak_index == raw.peak_index
    assert screened.tau_star == raw.tau_star
    assert screened.peak_index > coarse.peak_index
    assert PersistenceConfig().resolve_within_interval == "none"


def test_default_selector_is_load_crossover() -> None:
    # Deletion-prep lock (A6-T29): acceptance-path default stays load_crossover.
    assert ScaleSearchConfig().selector == "load_crossover"


def test_deprecated_load_band_alias_redirects_to_load_crossover() -> None:
    # Deprecated path isolation: selector="load_band" warns and matches
    # load_crossover on the same data (OPEN_ISSUES #28).
    import warnings

    dataset = make_circle(
        n_samples=800, radius=1.0, noise=0.02, extrusion_dim=2, seed=21,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    base_kw = dict(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=6,
        k=8,
        n_seeds=8,
        ann_backend="naive",
        stabilization=StabilizationConfig(min_equilibrium_epochs=3, max_epochs=12),
        seed=77,
    )
    canonical = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=ScaleSearchConfig(selector="load_crossover", **base_kw),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=ScaleSearchConfig(selector="load_band", **base_kw),
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert any("load_band" in str(w.message) for w in caught)
    assert legacy.peak_index == canonical.peak_index
    assert legacy.tau_star == canonical.tau_star
    assert legacy.persistence_result is None
    assert ScaleSearchConfig().selector == "load_crossover"
