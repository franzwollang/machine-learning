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
    _WITHIN_INTERVAL_LOAD_SCREEN_MIN,
    _load_weighted_index,
    _mid_interval_index,
    _resolve_persistence_tau_index,
    _three_quarter_index,
    _two_thirds_index,
    run_scale_search,
)
from proteus.stage1.persistence import (
    EXPERIMENTAL_DENSIFY_OVERLAP_RECOVER_THRESHOLD,
    PersistenceConfig,
    PersistenceResult,
)
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


def test_resolve_within_interval_two_thirds_vs_mid_three_quarter() -> None:
    # Experimental two_thirds_interval probe (A6-T43): lands 2/3 of the way
    # from i_lo toward i_hi; sits between mid_interval and three_quarter.
    # Brackets the mid-overshoot / three_quarter-undershoot gap on hierarchy.
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
    two_thirds = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(
                resolve_within_interval="two_thirds_interval"
            ),
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
    assert coarse.persistence_result is not None
    assert two_thirds.persistence_result is not None
    i_lo = coarse.persistence_result.tau_star_index
    assert i_lo is not None
    assert two_thirds.persistence_result.tau_star_index == i_lo
    run_len = int(two_thirds.persistence_result.run_lengths[i_lo])
    i_hi = i_lo + run_len - 1
    expected_tt = i_lo + (2 * (i_hi - i_lo)) // 3
    assert two_thirds.peak_index == expected_tt
    assert i_lo <= two_thirds.peak_index <= i_hi
    assert mid.peak_index == (i_lo + i_hi) // 2
    assert three_q.peak_index == i_lo + (3 * (i_hi - i_lo)) // 4
    if run_len >= 2:
        assert (
            three_q.peak_index
            >= two_thirds.peak_index
            >= mid.peak_index
            >= coarse.peak_index
        )
    if run_len >= 4:
        # On a long enough block, 2/3 is strictly between mid and 3/4 or equal
        # to one endpoint under integer flooring — never outside that bracket.
        assert mid.peak_index <= two_thirds.peak_index <= three_q.peak_index
    assert PersistenceConfig().resolve_within_interval == "none"


def test_phi_within_interval_landing_vs_hierarchy_expected_tau() -> None:
    # Diagnostic probe (A6-T32/T35/T44): correlate within-interval landing modes
    # with Phi_C(tau*) and hierarchy fine-leaf expected_tau. Reports a small
    # table; pins that load_crossover hybrid stays ≫ expected_tau while
    # mid/2/3/3q/fine refine (fine_end and often three_quarter can undershoot).
    # Does not flip acceptance defaults.
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
        "mid_interval_load_screened",
        "two_thirds_interval",
        "two_thirds_load_screened",
        "three_quarter_interval",
        "three_quarter_load_screened",
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
        load_star = float(result.load_trace[result.peak_index])
        rows.append(
            {
                "mode": mode,
                "peak_index": int(result.peak_index),
                "tau_star": float(result.tau_star),
                "phi_star": phi_star,
                "load_star": load_star,
                "tau_over_expected": float(result.tau_star / gt.expected_tau),
                "tau_over_fine_cluster": float(result.tau_star / fine_cluster_tau),
            }
        )

    # Human-readable diagnostic table (pytest -s).
    header = (
        f"{'mode':28s} {'idx':>3s} {'tau*':>10s} {'Phi*':>10s} "
        f"{'load*':>8s} {'tau*/E[tau]':>12s} {'tau*/fine':>10s}"
    )
    print("\nA6-T44 Phi within-interval landing vs hierarchy expected_tau")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['mode']:28s} {row['peak_index']:3d} {row['tau_star']:10.4g} "
            f"{row['phi_star']:10.4g} {row['load_star']:8.3f} "
            f"{row['tau_over_expected']:12.3f} "
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
    # two_thirds sits between mid and three_quarter on the grid (and in tau*).
    assert float(by_mode["two_thirds_interval"]["tau_star"]) <= float(
        by_mode["mid_interval"]["tau_star"]
    )
    assert float(by_mode["two_thirds_interval"]["tau_star"]) >= float(
        by_mode["three_quarter_interval"]["tau_star"]
    )
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
    # Grid-index ordering: none ≤ mid ≤ two_thirds ≤ three_quarter ≤ fine_end.
    assert int(by_mode["fine_end_of_block"]["peak_index"]) >= int(
        by_mode["three_quarter_interval"]["peak_index"]
    )
    assert int(by_mode["three_quarter_interval"]["peak_index"]) >= int(
        by_mode["two_thirds_interval"]["peak_index"]
    )
    assert int(by_mode["two_thirds_interval"]["peak_index"]) >= int(
        by_mode["mid_interval"]["peak_index"]
    )
    assert int(by_mode["mid_interval"]["peak_index"]) >= int(by_mode["none"]["peak_index"])
    # A6-T41/T44: load ≫ 1 at mid/2/3/3q landings ⇒ screened modes match raw.
    assert float(by_mode["mid_interval"]["load_star"]) > 1.0
    assert float(by_mode["two_thirds_interval"]["load_star"]) > 1.0
    assert float(by_mode["three_quarter_interval"]["load_star"]) > 1.0
    assert by_mode["mid_interval_load_screened"]["peak_index"] == by_mode["mid_interval"][
        "peak_index"
    ]
    assert by_mode["mid_interval_load_screened"]["tau_star"] == by_mode["mid_interval"][
        "tau_star"
    ]
    assert by_mode["two_thirds_load_screened"]["peak_index"] == by_mode[
        "two_thirds_interval"
    ]["peak_index"]
    assert by_mode["two_thirds_load_screened"]["tau_star"] == by_mode[
        "two_thirds_interval"
    ]["tau_star"]
    assert by_mode["three_quarter_load_screened"]["peak_index"] == by_mode[
        "three_quarter_interval"
    ]["peak_index"]
    assert by_mode["three_quarter_load_screened"]["tau_star"] == by_mode[
        "three_quarter_interval"
    ]["tau_star"]
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
        "two_thirds_interval",
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
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == _THREE_QUARTER_LOAD_SCREEN_MIN
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


def test_mid_interval_load_screened_rejects_low_load_and_matches_raw_when_ok() -> None:
    # Experimental probe (A6-T40): mid_interval_load_screened falls back to
    # coarse-end when load at the mid index is ≪ 1; otherwise matches raw
    # mid_interval. Contrast vs three_quarter_load_screened; default stays
    # "none".
    run_lengths = np.array([8, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    # Mid index for [0,7] is 3; put low load only around mid for reject case.
    low_load = np.array([0.8, 0.7, 0.6, 0.4, 0.55, 0.65, 0.75, 0.85])
    high_load = np.array([0.7, 2.0, 4.0, 8.0, 12.0, 19.0, 30.0, 50.0])
    stabilized = [True] * 8
    pers = PersistenceResult(
        tau_star=1.0,
        tau_star_index=0,
        run_lengths=run_lengths,
        match_overlaps=np.ones(7),
    )
    cfg_screened = PersistenceConfig(
        resolve_within_interval="mid_interval_load_screened",
    )
    cfg_raw = PersistenceConfig(resolve_within_interval="mid_interval")
    cfg_tq = PersistenceConfig(
        resolve_within_interval="three_quarter_load_screened",
    )
    # i_lo=0, i_hi=7 → mid = 3; three_quarter = 5.
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5
    assert float(low_load[3]) < _WITHIN_INTERVAL_LOAD_SCREEN_MIN
    assert float(high_load[3]) >= _WITHIN_INTERVAL_LOAD_SCREEN_MIN
    assert (
        _resolve_persistence_tau_index(pers, low_load, stabilized, cfg_screened)
        == 0
    )
    assert (
        _resolve_persistence_tau_index(pers, low_load, stabilized, cfg_raw) == 3
    )
    assert (
        _resolve_persistence_tau_index(pers, high_load, stabilized, cfg_screened)
        == 3
    )
    assert (
        _resolve_persistence_tau_index(pers, high_load, stabilized, cfg_raw) == 3
    )
    # Shared screen floor: mid screened ≠ three_quarter screened on high load.
    assert (
        _resolve_persistence_tau_index(pers, high_load, stabilized, cfg_tq) == 5
    )

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
    raw_mid = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(resolve_within_interval="mid_interval"),
        ),
    )
    screened_mid = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(
                resolve_within_interval="mid_interval_load_screened"
            ),
        ),
    )
    raw_tq = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(
                resolve_within_interval="three_quarter_interval"
            ),
        ),
    )
    screened_tq = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(
                resolve_within_interval="three_quarter_load_screened"
            ),
        ),
    )
    assert raw_mid.persistence_result is not None
    assert raw_mid.persistence_result.tau_star_index is not None
    load_at_mid = float(raw_mid.load_trace[raw_mid.peak_index])
    load_at_tq = float(raw_tq.load_trace[raw_tq.peak_index])
    print(
        "\nA6-T40 load-screened mid contrast "
        f"(mid_load={load_at_mid:.3f}, tq_load={load_at_tq:.3f}, "
        f"screen_min={_WITHIN_INTERVAL_LOAD_SCREEN_MIN})"
    )
    print(
        f"  mid raw idx={raw_mid.peak_index} tau*={raw_mid.tau_star:.4g} "
        f"screened idx={screened_mid.peak_index} tau*={screened_mid.tau_star:.4g}"
    )
    print(
        f"  3q  raw idx={raw_tq.peak_index} tau*={raw_tq.tau_star:.4g} "
        f"screened idx={screened_tq.peak_index} tau*={screened_tq.tau_star:.4g}"
    )
    assert load_at_mid >= _WITHIN_INTERVAL_LOAD_SCREEN_MIN
    assert load_at_tq >= _WITHIN_INTERVAL_LOAD_SCREEN_MIN
    assert screened_mid.peak_index == raw_mid.peak_index
    assert screened_mid.tau_star == raw_mid.tau_star
    assert screened_tq.peak_index == raw_tq.peak_index
    assert screened_tq.tau_star == raw_tq.tau_star
    # Mid is coarser than three_quarter on this hierarchy fixture.
    assert screened_mid.peak_index <= screened_tq.peak_index
    assert PersistenceConfig().resolve_within_interval == "none"


def test_two_thirds_load_screened_rejects_low_load_and_matches_raw_when_ok() -> None:
    # Experimental probe (A6-T43): two_thirds_load_screened falls back to
    # coarse-end when load at the two-thirds index is ≪ 1; otherwise matches
    # raw two_thirds_interval. Contrast vs mid/three_quarter screened; default
    # stays "none".
    run_lengths = np.array([8, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    # i_lo=0, i_hi=7 → two_thirds = 4; put low load only around that index.
    low_load = np.array([0.8, 0.7, 0.6, 0.55, 0.4, 0.65, 0.75, 0.85])
    high_load = np.array([0.7, 2.0, 4.0, 8.0, 12.0, 19.0, 30.0, 50.0])
    stabilized = [True] * 8
    pers = PersistenceResult(
        tau_star=1.0,
        tau_star_index=0,
        run_lengths=run_lengths,
        match_overlaps=np.ones(7),
    )
    cfg_screened = PersistenceConfig(
        resolve_within_interval="two_thirds_load_screened",
    )
    cfg_raw = PersistenceConfig(resolve_within_interval="two_thirds_interval")
    cfg_mid = PersistenceConfig(
        resolve_within_interval="mid_interval_load_screened",
    )
    cfg_tq = PersistenceConfig(
        resolve_within_interval="three_quarter_load_screened",
    )
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5
    assert float(low_load[4]) < _WITHIN_INTERVAL_LOAD_SCREEN_MIN
    assert float(high_load[4]) >= _WITHIN_INTERVAL_LOAD_SCREEN_MIN
    assert (
        _resolve_persistence_tau_index(pers, low_load, stabilized, cfg_screened)
        == 0
    )
    assert (
        _resolve_persistence_tau_index(pers, low_load, stabilized, cfg_raw) == 4
    )
    assert (
        _resolve_persistence_tau_index(pers, high_load, stabilized, cfg_screened)
        == 4
    )
    assert (
        _resolve_persistence_tau_index(pers, high_load, stabilized, cfg_raw) == 4
    )
    # Shared screen floor: distinct landings under high load.
    assert (
        _resolve_persistence_tau_index(pers, high_load, stabilized, cfg_mid) == 3
    )
    assert (
        _resolve_persistence_tau_index(pers, high_load, stabilized, cfg_tq) == 5
    )

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
    raw_tt = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(
                resolve_within_interval="two_thirds_interval"
            ),
        ),
    )
    screened_tt = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(
                resolve_within_interval="two_thirds_load_screened"
            ),
        ),
    )
    raw_mid = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(resolve_within_interval="mid_interval"),
        ),
    )
    raw_tq = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(
                resolve_within_interval="three_quarter_interval"
            ),
        ),
    )
    assert raw_tt.persistence_result is not None
    assert raw_tt.persistence_result.tau_star_index is not None
    load_at_tt = float(raw_tt.load_trace[raw_tt.peak_index])
    print(
        "\nA6-T43 load-screened two_thirds contrast "
        f"(tt_load={load_at_tt:.3f}, "
        f"screen_min={_WITHIN_INTERVAL_LOAD_SCREEN_MIN})"
    )
    print(
        f"  2/3 raw idx={raw_tt.peak_index} tau*={raw_tt.tau_star:.4g} "
        f"screened idx={screened_tt.peak_index} tau*={screened_tt.tau_star:.4g}"
    )
    print(
        f"  mid idx={raw_mid.peak_index} tau*={raw_mid.tau_star:.4g} "
        f"3q idx={raw_tq.peak_index} tau*={raw_tq.tau_star:.4g}"
    )
    assert load_at_tt >= _WITHIN_INTERVAL_LOAD_SCREEN_MIN
    assert screened_tt.peak_index == raw_tt.peak_index
    assert screened_tt.tau_star == raw_tt.tau_star
    assert raw_mid.peak_index <= screened_tt.peak_index <= raw_tq.peak_index
    assert PersistenceConfig().resolve_within_interval == "none"


def test_halve_grid_steps_densifies_two_thirds_three_quarter_phi() -> None:
    # Experimental denser geometric grid (A6-T46): halve_grid_steps uses
    # sqrt(grid_ratio) and up to 2× max_grid_points so persistent blocks have
    # more tau candidates under two_thirds / three_quarter. Report Phi deltas
    # vs the standard grid; do not flip defaults.
    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=0,
    )
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
        stabilization=StabilizationConfig(min_equilibrium_epochs=2, max_epochs=12),
        seed=0,
    )
    assert ScaleSearchConfig().halve_grid_steps is False

    modes = ("two_thirds_interval", "three_quarter_interval")
    rows: list[dict[str, float | int | str | bool]] = []
    for dense in (False, True):
        for mode in modes:
            result = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=replace(
                    base,
                    halve_grid_steps=dense,
                    persistence=PersistenceConfig(resolve_within_interval=mode),
                ),
            )
            assert result.persistence_result is not None
            rows.append(
                {
                    "dense": dense,
                    "mode": mode,
                    "n_grid": int(len(result.tau_grid)),
                    "peak_index": int(result.peak_index),
                    "tau_star": float(result.tau_star),
                    "phi_star": float(result.phi_trace[result.peak_index]),
                    "tau_over_expected": float(result.tau_star / gt.expected_tau),
                }
            )

    header = (
        f"{'dense':5s} {'mode':22s} {'n':>3s} {'idx':>3s} "
        f"{'tau*':>10s} {'Phi*':>10s} {'tau*/E':>10s}"
    )
    print("\nA6-T46 denser grid (halve steps) under two_thirds / three_quarter")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{str(row['dense']):5s} {row['mode']:22s} {row['n_grid']:3d} "
            f"{row['peak_index']:3d} {row['tau_star']:10.4g} "
            f"{row['phi_star']:10.4g} {row['tau_over_expected']:10.3f}"
        )

    by = {(bool(r["dense"]), str(r["mode"])): r for r in rows}
    # Denser grid must actually add points (within tau_min/max budget).
    assert int(by[(True, "two_thirds_interval")]["n_grid"]) > int(
        by[(False, "two_thirds_interval")]["n_grid"]
    )
    # Ordering preserved on each grid: two_thirds coarser (larger tau*) than 3q.
    for dense in (False, True):
        assert float(by[(dense, "two_thirds_interval")]["tau_star"]) >= float(
            by[(dense, "three_quarter_interval")]["tau_star"]
        )
    # Pin densification effect (hierarchy seed=0):
    # - standard grid: three_quarter (~0.82×) closer to E[tau] than two_thirds
    #   (~1.49×) — discrete jump 1.49→0.82.
    # - half-step denser grid: two_thirds lands ≈1.00× (closest / exact on this
    #   fixture) while three_quarter undershoots further (~0.76×). Densifying
    #   does *not* preserve the standard-grid 3q-closest ranking.
    err_tt_std = abs(
        float(by[(False, "two_thirds_interval")]["tau_over_expected"]) - 1.0
    )
    err_tq_std = abs(
        float(by[(False, "three_quarter_interval")]["tau_over_expected"]) - 1.0
    )
    assert err_tq_std < err_tt_std
    ratio_tt_dense = float(by[(True, "two_thirds_interval")]["tau_over_expected"])
    ratio_tq_dense = float(by[(True, "three_quarter_interval")]["tau_over_expected"])
    err_tt_dense = abs(ratio_tt_dense - 1.0)
    err_tq_dense = abs(ratio_tq_dense - 1.0)
    assert err_tt_dense < err_tq_dense
    assert 0.9 < ratio_tt_dense < 1.1
    assert ratio_tq_dense < 1.0
    # Phi finite on both grids (diagnostic usability).
    for dense in (False, True):
        for mode in modes:
            assert np.isfinite(float(by[(dense, mode)]["phi_star"]))
    assert PersistenceConfig().resolve_within_interval == "none"
    assert ScaleSearchConfig().halve_grid_steps is False


def test_load_weighted_interval_picks_closest_load_and_rejects_low() -> None:
    # Experimental probe (A6-T47): load_weighted_interval selects the
    # persistent-block index with load closest to 1 among load >= screen floor;
    # all-below-floor ⇒ coarse-end. Contrast vs two_thirds / three_quarter.
    # Default resolve_within_interval stays "none".
    from proteus.stage1.controller import _load_weighted_index

    run_lengths = np.array([8, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    # Synthetic loads: indices 0..7; closest-to-1 among screened is idx 3
    # (load=1.05); idx 1 has load 0.2 ≪ floor and must be ignored.
    load = np.array([4.0, 0.2, 2.5, 1.05, 1.4, 1.8, 2.2, 3.0], dtype=float)
    pers = PersistenceResult(
        tau_star=1.0,
        tau_star_index=0,
        run_lengths=run_lengths,
        match_overlaps=np.ones(7),
    )
    cfg_w = PersistenceConfig(resolve_within_interval="load_weighted_interval")
    cfg_tt = PersistenceConfig(resolve_within_interval="two_thirds_interval")
    cfg_tq = PersistenceConfig(resolve_within_interval="three_quarter_interval")
    stab = [True] * 8
    assert _resolve_persistence_tau_index(pers, load, stab, cfg_w) == 3
    assert _load_weighted_index(0, 7, load) == 3
    # two_thirds = 0 + 14//3 = 4; three_quarter = 0 + 21//4 = 5.
    assert _resolve_persistence_tau_index(pers, load, stab, cfg_tt) == 4
    assert _resolve_persistence_tau_index(pers, load, stab, cfg_tq) == 5
    # All loads ≪ floor ⇒ fall back to i_lo.
    low = np.full(8, 0.1, dtype=float)
    assert _resolve_persistence_tau_index(pers, low, stab, cfg_w) == 0

    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=0,
    )
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
        stabilization=StabilizationConfig(min_equilibrium_epochs=2, max_epochs=12),
        seed=0,
    )
    weighted = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(resolve_within_interval="load_weighted_interval"),
        ),
    )
    two_thirds = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(resolve_within_interval="two_thirds_interval"),
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
    none = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=replace(
            base,
            persistence=PersistenceConfig(resolve_within_interval="none"),
        ),
    )
    assert weighted.persistence_result is not None
    i_lo = int(weighted.persistence_result.tau_star_index)  # type: ignore[arg-type]
    run_len = int(weighted.persistence_result.run_lengths[i_lo])
    i_hi = i_lo + run_len - 1
    assert i_lo <= weighted.peak_index <= i_hi
    load_w = float(weighted.load_trace[weighted.peak_index])
    assert load_w >= _WITHIN_INTERVAL_LOAD_SCREEN_MIN
    # On hierarchy seed-0, loads in the block are ≫ 1, so load-weighted tends
    # toward the smallest load in-block (closest to 1) — typically finer than
    # coarse-end; report contrast vs fractional landings.
    print(
        "\nA6-T47 load_weighted vs two_thirds / three_quarter "
        f"(hierarchy seed=0): weighted idx={weighted.peak_index} "
        f"tau*={weighted.tau_star:.4g} load={load_w:.3f} "
        f"tau*/E={weighted.tau_star / gt.expected_tau:.3f}; "
        f"2/3 idx={two_thirds.peak_index} tau*/E="
        f"{two_thirds.tau_star / gt.expected_tau:.3f}; "
        f"3q idx={three_q.peak_index} tau*/E="
        f"{three_q.tau_star / gt.expected_tau:.3f}; "
        f"none idx={none.peak_index}"
    )
    assert np.isfinite(float(weighted.phi_trace[weighted.peak_index]))
    assert PersistenceConfig().resolve_within_interval == "none"


def test_load_weighted_matches_coarse_across_hierarchy_seeds() -> None:
    # Throughput probe (A6-T49): on hierarchical Gaussians, load_weighted_interval
    # lands at the coarse-end persistence arbiter whenever a multi-cluster
    # interval is accepted. Root cause: within-block loads jump from
    # L(i_lo)~0.6–0.7 to L≫1 at the next grid point, so argmin|log L| among
    # L≥0.5 is always i_lo. Seeds 1–2 never accept a persistent split
    # (tau_star_index is None → LC fallback); seed-4 dense also rejects.
    # When persistence rejects, load_weighted is a no-op vs resolve=none.
    # Do not flip defaults. (A1 integrator harden: guard None tau_star_index.)
    seeds = (0, 1, 2, 3, 4)
    rows: list[dict[str, float | int | bool]] = []
    n_persist = 0
    for seed in seeds:
        for dense in (False, True):
            dataset = make_hierarchical_gaussian(
                children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
            )
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=seed,
                halve_grid_steps=dense,
            )
            none = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=replace(
                    base,
                    persistence=PersistenceConfig(resolve_within_interval="none"),
                ),
            )
            weighted = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=replace(
                    base,
                    persistence=PersistenceConfig(
                        resolve_within_interval="load_weighted_interval"
                    ),
                ),
            )
            assert none.persistence_result is not None
            assert weighted.persistence_result is not None
            assert none.peak_index is not None and weighted.peak_index is not None
            load_w = float(weighted.load_trace[weighted.peak_index])
            tau_idx = none.persistence_result.tau_star_index
            rows.append(
                {
                    "seed": seed,
                    "dense": dense,
                    "none_idx": int(none.peak_index),
                    "weighted_idx": int(weighted.peak_index),
                    "i_lo": -1 if tau_idx is None else int(tau_idx),
                    "load": load_w,
                    "tau_over_expected": float(
                        weighted.tau_star / gt.expected_tau
                    ),
                }
            )
            # Always: weighted reproduces the none landing (coarse or LC).
            assert weighted.peak_index == none.peak_index
            if tau_idx is None:
                # No accepted multi-cluster interval — LC fallback; modes agree.
                assert weighted.persistence_result.tau_star_index is None
                continue
            n_persist += 1
            i_lo = int(tau_idx)
            assert weighted.peak_index == i_lo
            assert load_w >= _WITHIN_INTERVAL_LOAD_SCREEN_MIN
            assert 0.5 <= load_w < 1.0

    # Non-vacuous: at least seeds 0/3 (std+dense) and seed-4 std persist.
    assert n_persist >= 5

    header = (
        f"{'seed':4s} {'dense':5s} {'idx':>3s} {'load':>7s} {'tau*/E':>8s}"
    )
    print("\nA6-T49 load_weighted ≡ coarse-end across hierarchy seeds")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{int(row['seed']):4d} {str(row['dense']):5s} "
            f"{int(row['weighted_idx']):3d} {float(row['load']):7.3f} "
            f"{float(row['tau_over_expected']):8.3f}"
        )
    assert PersistenceConfig().resolve_within_interval == "none"
    assert ScaleSearchConfig().halve_grid_steps is False


def test_dense_ranking_flip_is_seed_fragile() -> None:
    # Throughput probe (A6-T49): seed-0 densify flip (2/3≈1.00× beats 3q) is
    # not universal. Seed 4 matches seed 0 on the *standard* grid (3q closer)
    # but under halve_grid_steps persistence rejects (tau_star_index None) so
    # all fractional modes fall back to the same LC coarse landing — densify
    # fragility, not a preserved ranking flip. Seeds 1–2 are coarse-only
    # no-ops. Do not flip default. (A1: guard None tau_star_index.)
    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=4,
    )
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
        stabilization=StabilizationConfig(min_equilibrium_epochs=2, max_epochs=12),
        seed=4,
    )
    modes = ("two_thirds_interval", "three_quarter_interval", "mid_interval")
    by: dict[tuple[bool, str], dict[str, float | int | None]] = {}
    for dense in (False, True):
        for mode in modes:
            result = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=replace(
                    base,
                    halve_grid_steps=dense,
                    persistence=PersistenceConfig(resolve_within_interval=mode),
                ),
            )
            assert result.persistence_result is not None
            assert result.peak_index is not None
            by[(dense, mode)] = {
                "peak_index": int(result.peak_index),
                "tau_over_expected": float(result.tau_star / gt.expected_tau),
                "tau_star_index": result.persistence_result.tau_star_index,
            }

    # Standard grid seed-4: same 3q-closest pattern as seed-0; persist accepted.
    for mode in modes:
        assert by[(False, mode)]["tau_star_index"] is not None
    err_tt = abs(float(by[(False, "two_thirds_interval")]["tau_over_expected"]) - 1.0)
    err_tq = abs(
        float(by[(False, "three_quarter_interval")]["tau_over_expected"]) - 1.0
    )
    assert err_tq < err_tt
    assert abs(float(by[(False, "three_quarter_interval")]["tau_over_expected"]) - 0.82) < 0.05
    assert abs(float(by[(False, "two_thirds_interval")]["tau_over_expected"]) - 1.49) < 0.05

    # Dense seed-4: persistence rejects → all fractional modes share LC peak.
    peak_dense = int(by[(True, "two_thirds_interval")]["peak_index"])
    for mode in modes:
        assert by[(True, mode)]["tau_star_index"] is None
        assert int(by[(True, mode)]["peak_index"]) == peak_dense
        assert float(by[(True, mode)]["tau_over_expected"]) > 8.0

    print(
        "\nA6-T49 seed-4 ranking fragility: "
        f"std 2/3={by[(False, 'two_thirds_interval')]['tau_over_expected']:.3f} "
        f"3q={by[(False, 'three_quarter_interval')]['tau_over_expected']:.3f}; "
        f"dense persist-reject all→LC idx={peak_dense} "
        f"tau*/E={by[(True, 'two_thirds_interval')]['tau_over_expected']:.3f}"
    )
    assert PersistenceConfig().resolve_within_interval == "none"
    assert ScaleSearchConfig().halve_grid_steps is False


def test_export_multi_seed_phi_hierarchy_table() -> None:
    # EXPORT probe (A6-T50): multi-seed Phi / tau* hierarchy table for A3 SI
    # sync. Seeds 0..4 × standard vs halve_grid_steps densify; modes cover the
    # fractional + load_weighted probes. One scale-search per (seed, dense)
    # then offline _resolve_persistence_tau_index (matches full re-runs when
    # persist accepts; when reject, all modes share the LC fallback peak).
    # Default resolve stays "none"; do not flip acceptance.
    modes = (
        "none",
        "mid_interval",
        "two_thirds_interval",
        "three_quarter_interval",
        "fine_end_of_block",
        "load_weighted_interval",
    )
    seeds = (0, 1, 2, 3, 4)
    rows: list[dict[str, float | int | str | bool | None]] = []
    for seed in seeds:
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        assert gt.expected_tau is not None
        tau_lo, tau_hi = gt.tau_grid_hint
        for dense in (False, True):
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=seed,
                halve_grid_steps=dense,
            )
            none = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=replace(
                    base,
                    persistence=PersistenceConfig(resolve_within_interval="none"),
                ),
            )
            assert none.persistence_result is not None
            assert none.peak_index is not None
            pr = none.persistence_result
            persist_ok = pr.tau_star_index is not None
            for mode in modes:
                if persist_ok:
                    idx = _resolve_persistence_tau_index(
                        pr,
                        none.load_trace,
                        list(none.stabilized_flags),
                        PersistenceConfig(resolve_within_interval=mode),  # type: ignore[arg-type]
                    )
                else:
                    # Persistence reject → controller LC fallback; modes agree.
                    idx = int(none.peak_index)
                tau_star = float(none.tau_grid[idx])
                phi_star = float(none.phi_trace[idx])
                load_star = float(none.load_trace[idx])
                rows.append(
                    {
                        "seed": seed,
                        "dense": dense,
                        "mode": mode,
                        "persist": persist_ok,
                        "n_grid": int(len(none.tau_grid)),
                        "peak_index": int(idx),
                        "tau_star": tau_star,
                        "phi_star": phi_star,
                        "load_star": load_star,
                        "tau_over_expected": float(tau_star / gt.expected_tau),
                    }
                )

    header = (
        f"{'seed':4s} {'dense':5s} {'persist':7s} {'mode':24s} "
        f"{'n':>3s} {'idx':>3s} {'tau*/E':>8s} {'Phi*':>10s} {'load*':>8s}"
    )
    print("\nA6-T50 multi-seed Phi hierarchy export (for A3 SI)")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{int(row['seed']):4d} {str(row['dense']):5s} "
            f"{str(row['persist']):7s} {row['mode']:24s} "
            f"{int(row['n_grid']):3d} {int(row['peak_index']):3d} "
            f"{float(row['tau_over_expected']):8.3f} "
            f"{float(row['phi_star']):10.4g} {float(row['load_star']):8.3f}"
        )

    by = {
        (int(r["seed"]), bool(r["dense"]), str(r["mode"])): r for r in rows
    }
    # Seed-0 std: pin published hierarchy ratios (paper / prior A6 notes).
    assert abs(float(by[(0, False, "none")]["tau_over_expected"]) - 16.0) < 0.05
    assert abs(float(by[(0, False, "mid_interval")]["tau_over_expected"]) - 2.69) < 0.05
    assert (
        abs(float(by[(0, False, "two_thirds_interval")]["tau_over_expected"]) - 1.49)
        < 0.05
    )
    assert (
        abs(float(by[(0, False, "three_quarter_interval")]["tau_over_expected"]) - 0.82)
        < 0.05
    )
    assert (
        abs(float(by[(0, False, "fine_end_of_block")]["tau_over_expected"]) - 0.25)
        < 0.05
    )
    assert by[(0, False, "load_weighted_interval")]["peak_index"] == by[
        (0, False, "none")
    ]["peak_index"]
    # Seed-0 dense: densify flip (2/3≈1.00× beats 3q).
    assert 0.9 < float(by[(0, True, "two_thirds_interval")]["tau_over_expected"]) < 1.1
    assert float(by[(0, True, "three_quarter_interval")]["tau_over_expected"]) < 0.9
    # Persist accept/reject map (locks T49 systematics for SI export).
    for seed in (0, 3):
        assert by[(seed, False, "none")]["persist"] is True
        assert by[(seed, True, "none")]["persist"] is True
    for seed in (1, 2):
        assert by[(seed, False, "none")]["persist"] is False
        assert by[(seed, True, "none")]["persist"] is False
    assert by[(4, False, "none")]["persist"] is True
    assert by[(4, True, "none")]["persist"] is False
    # When persist rejects, all modes share the LC peak.
    for seed, dense in ((1, False), (1, True), (2, False), (2, True), (4, True)):
        peak = int(by[(seed, dense, "none")]["peak_index"])
        for mode in modes:
            assert int(by[(seed, dense, mode)]["peak_index"]) == peak
            assert by[(seed, dense, mode)]["persist"] is False
    # load_weighted ≡ none whenever persist accepts (hierarchy coarse-end).
    for seed in seeds:
        for dense in (False, True):
            if by[(seed, dense, "none")]["persist"]:
                assert by[(seed, dense, "load_weighted_interval")][
                    "peak_index"
                ] == by[(seed, dense, "none")]["peak_index"]
    # Phi finite wherever we land.
    for row in rows:
        assert np.isfinite(float(row["phi_star"]))
    assert PersistenceConfig().resolve_within_interval == "none"
    assert ScaleSearchConfig().halve_grid_steps is False


def test_load_weighted_x_halve_grid_combo_phi() -> None:
    # EXPERIMENT (A6-T51): load_weighted_interval × halve_grid_steps combo on
    # hierarchy. Contrast vs coarse-end (none) under std and dense grids.
    # Expect weighted ≡ none on both grids when persist accepts; when dense
    # reject (seed-4), both are LC no-ops. Flag defaults stay off.
    seeds = (0, 3, 4)
    rows: list[dict[str, float | int | bool | None]] = []
    for seed in seeds:
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        assert gt.expected_tau is not None
        tau_lo, tau_hi = gt.tau_grid_hint
        for dense in (False, True):
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=seed,
                halve_grid_steps=dense,
            )
            none = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=replace(
                    base,
                    persistence=PersistenceConfig(resolve_within_interval="none"),
                ),
            )
            weighted = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=replace(
                    base,
                    persistence=PersistenceConfig(
                        resolve_within_interval="load_weighted_interval"
                    ),
                ),
            )
            assert none.persistence_result is not None
            assert weighted.persistence_result is not None
            assert none.peak_index is not None and weighted.peak_index is not None
            persist_ok = none.persistence_result.tau_star_index is not None
            rows.append(
                {
                    "seed": seed,
                    "dense": dense,
                    "persist": persist_ok,
                    "n_grid": int(len(none.tau_grid)),
                    "none_idx": int(none.peak_index),
                    "weighted_idx": int(weighted.peak_index),
                    "none_tau_over_E": float(none.tau_star / gt.expected_tau),
                    "weighted_tau_over_E": float(
                        weighted.tau_star / gt.expected_tau
                    ),
                    "weighted_load": float(weighted.load_trace[weighted.peak_index]),
                }
            )
            assert weighted.peak_index == none.peak_index
            assert weighted.tau_star == none.tau_star
            if persist_ok:
                i_lo = int(none.persistence_result.tau_star_index)  # type: ignore[arg-type]
                assert weighted.peak_index == i_lo
                assert (
                    float(weighted.load_trace[weighted.peak_index])
                    >= _WITHIN_INTERVAL_LOAD_SCREEN_MIN
                )

    header = (
        f"{'seed':4s} {'dense':5s} {'persist':7s} {'n':>3s} "
        f"{'none':>4s} {'w':>4s} {'none/E':>8s} {'w/E':>8s} {'load_w':>8s}"
    )
    print("\nA6-T51 load_weighted × halve_grid_steps combo vs coarse")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{int(row['seed']):4d} {str(row['dense']):5s} "
            f"{str(row['persist']):7s} {int(row['n_grid']):3d} "
            f"{int(row['none_idx']):4d} {int(row['weighted_idx']):4d} "
            f"{float(row['none_tau_over_E']):8.3f} "
            f"{float(row['weighted_tau_over_E']):8.3f} "
            f"{float(row['weighted_load']):8.3f}"
        )

    by = {(int(r["seed"]), bool(r["dense"])): r for r in rows}
    # Dense grid actually densifies when persist path still runs.
    assert int(by[(0, True)]["n_grid"]) > int(by[(0, False)]["n_grid"])
    assert by[(0, False)]["persist"] is True
    assert by[(0, True)]["persist"] is True
    assert by[(4, False)]["persist"] is True
    assert by[(4, True)]["persist"] is False
    # Combo never refines past coarse/LC on this fixture.
    for row in rows:
        assert int(row["weighted_idx"]) == int(row["none_idx"])
    assert PersistenceConfig().resolve_within_interval == "none"
    assert ScaleSearchConfig().halve_grid_steps is False


def test_halve_grid_circle_swiss_within_interval_noop() -> None:
    # EXPERIMENT (A6-T53): circle / swiss under halve_grid_steps. Uniform /
    # developable manifolds still have no accepted persistent split on the
    # denser grid; within-interval modes remain identical LC fallbacks (only
    # the LC peak index may move with densify). Defaults stay off.
    modes = (
        "none",
        "mid_interval",
        "two_thirds_interval",
        "three_quarter_interval",
        "fine_end_of_block",
        "load_crossover",
        "load_weighted_interval",
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
    rows: list[dict[str, float | int | str | bool | None]] = []
    for name, dataset in fixtures:
        gt = dataset.ground_truth
        assert gt.expected_tau is not None
        tau_lo, tau_hi = gt.tau_grid_hint
        for dense in (False, True):
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=0,
                halve_grid_steps=dense,
            )
            # One scale-search; offline resolve (persist reject ⇒ LC peak).
            none = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=replace(
                    base,
                    persistence=PersistenceConfig(resolve_within_interval="none"),
                ),
            )
            assert none.persistence_result is not None
            assert none.peak_index is not None
            pr = none.persistence_result
            assert pr.tau_star_index is None
            lc_peak = int(none.peak_index)
            n_grid = int(len(none.tau_grid))
            # Spot-check one within-interval mode via a full controller re-run
            # (offline resolve is a no-op on persist-reject; this locks LC identity).
            mid = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=replace(
                    base,
                    persistence=PersistenceConfig(
                        resolve_within_interval="mid_interval"
                    ),
                ),
            )
            assert mid.persistence_result is not None
            assert mid.persistence_result.tau_star_index is None
            assert int(mid.peak_index) == lc_peak
            assert float(mid.tau_star) == float(none.tau_star)
            for mode in modes:
                rows.append(
                    {
                        "name": name,
                        "dense": dense,
                        "mode": mode,
                        "n_grid": n_grid,
                        "peak_index": lc_peak,
                        "tau_star": float(none.tau_grid[lc_peak]),
                        "phi_star": float(none.phi_trace[lc_peak]),
                        "tau_over_expected": float(
                            none.tau_grid[lc_peak] / gt.expected_tau
                        ),
                    }
                )

    header = (
        f"{'name':6s} {'dense':5s} {'mode':24s} "
        f"{'n':>3s} {'idx':>3s} {'tau*/E':>8s} {'Phi*':>10s}"
    )
    print("\nA6-T53 circle/swiss × halve_grid_steps (within-interval no-op)")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['name']:6s} {str(row['dense']):5s} {row['mode']:24s} "
            f"{int(row['n_grid']):3d} {int(row['peak_index']):3d} "
            f"{float(row['tau_over_expected']):8.3f} "
            f"{float(row['phi_star']):10.4g}"
        )

    by = {
        (str(r["name"]), bool(r["dense"])): r for r in rows if r["mode"] == "none"
    }
    # Densify actually doubles the geometric grid; LC peak may move.
    assert int(by[("circle", True)]["n_grid"]) > int(by[("circle", False)]["n_grid"])
    assert int(by[("swiss", True)]["n_grid"]) > int(by[("swiss", False)]["n_grid"])
    # All modes share one peak per (manifold, dense) cell.
    for name, _ in fixtures:
        for dense in (False, True):
            peaks = {
                int(r["peak_index"])
                for r in rows
                if r["name"] == name and bool(r["dense"]) is dense
            }
            assert len(peaks) == 1
    assert PersistenceConfig().resolve_within_interval == "none"
    assert ScaleSearchConfig().halve_grid_steps is False


def test_seed4_dense_persist_reject_is_first_step_overlap() -> None:
    # EXPERIMENT (A6-T54): mechanism for seed-4 densified persist-reject.
    # Standard grid: first adjacent matched Jaccard ≥ overlap_threshold so
    # coarse-anchored run_lengths[0] ≥ min_persistence and tau_star_index=0.
    # Half-step densify inserts a nearer neighbor whose overlap drops below
    # threshold (partition churn 5→3 clusters), so run_lengths[0]=1 and the
    # coarse-anchored rule rejects even though a long fineward run exists from
    # index 1. Do not flip defaults.
    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=4,
    )
    gt = dataset.ground_truth
    assert gt.expected_tau is not None
    tau_lo, tau_hi = gt.tau_grid_hint
    thr = float(PersistenceConfig().overlap_threshold)
    min_pers = int(PersistenceConfig().min_persistence)
    by: dict[bool, PersistenceResult] = {}
    for dense in (False, True):
        result = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=ScaleSearchConfig(
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=4,
                halve_grid_steps=dense,
                persistence=PersistenceConfig(resolve_within_interval="none"),
            ),
        )
        assert result.persistence_result is not None
        pr = result.persistence_result
        by[dense] = pr
        n_clusters = [int(s.n_clusters) for s in pr.snapshots]
        print(
            f"\nA6-T54 seed4 dense={dense}: n_grid={len(result.tau_grid)} "
            f"tsi={pr.tau_star_index} run0={int(pr.run_lengths[0])} "
            f"ov0={float(pr.match_overlaps[0]):.3f} "
            f"n_clusters[:3]={n_clusters[:3]}"
        )

    std_pr = by[False]
    dense_pr = by[True]
    assert std_pr.tau_star_index == 0
    assert int(std_pr.run_lengths[0]) >= min_pers
    assert float(std_pr.match_overlaps[0]) >= thr
    # Dense: first half-step breaks the coarse-end run under overlap_threshold.
    assert dense_pr.tau_star_index is None
    assert int(dense_pr.run_lengths[0]) == 1
    assert float(dense_pr.match_overlaps[0]) < thr
    assert int(dense_pr.snapshots[0].n_clusters) >= 2
    assert int(dense_pr.snapshots[1].n_clusters) >= 2
    # A long fineward run still exists from index 1 — discarded only because
    # coarse_anchored requires the *coarsest* multi-cluster point to persist.
    assert int(dense_pr.run_lengths[1]) >= min_pers
    assert PersistenceConfig().coarse_anchored is True
    assert PersistenceConfig().resolve_within_interval == "none"
    assert ScaleSearchConfig().halve_grid_steps is False


def test_seed3_short_persist_block_collapses_fractional_landings() -> None:
    # EXPERIMENT (A6-T55): seed-3 hierarchical Gaussian has a *short* accepted
    # persist block on the standard grid (run_lengths[0]=3 ⇒ [i_lo,i_hi]=[0,2]).
    # Integer mid / two-thirds / three-quarter all floor to the same interior
    # index 1, so mid≡2/3≡3q land at τ*/E≈8.83× — discrete short-block
    # quantization, not an intrinsic shared preference. Densifying expands the
    # block to the full half-step grid (len=16) and restores separated
    # fractional landings (incl. densify flip 2/3≈1.00× vs 3q undershoot).
    # Defaults stay off.
    from proteus.stage1.controller import (
        _mid_interval_index,
        _three_quarter_index,
        _two_thirds_index,
    )

    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=3,
    )
    gt = dataset.ground_truth
    assert gt.expected_tau is not None
    tau_lo, tau_hi = gt.tau_grid_hint
    modes = (
        "none",
        "mid_interval",
        "two_thirds_interval",
        "three_quarter_interval",
        "fine_end_of_block",
        "load_weighted_interval",
    )
    by: dict[tuple[bool, str], dict[str, float | int | bool]] = {}
    for dense in (False, True):
        none = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=ScaleSearchConfig(
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=3,
                halve_grid_steps=dense,
                persistence=PersistenceConfig(resolve_within_interval="none"),
            ),
        )
        assert none.persistence_result is not None
        pr = none.persistence_result
        assert pr.tau_star_index is not None
        i_lo = int(pr.tau_star_index)
        run = int(pr.run_lengths[i_lo])
        i_hi = min(i_lo + run - 1, len(none.load_trace) - 1)
        print(
            f"\nA6-T55 seed3 dense={dense}: n_grid={len(none.tau_grid)} "
            f"block=[{i_lo},{i_hi}] len={run} "
            f"ov0={float(pr.match_overlaps[0]):.3f}"
        )
        for mode in modes:
            idx = _resolve_persistence_tau_index(
                pr,
                none.load_trace,
                list(none.stabilized_flags),
                PersistenceConfig(resolve_within_interval=mode),  # type: ignore[arg-type]
            )
            ratio = float(none.tau_grid[idx] / gt.expected_tau)
            by[(dense, mode)] = {
                "peak_index": int(idx),
                "tau_over_expected": ratio,
                "run": run,
                "i_lo": i_lo,
                "i_hi": i_hi,
            }
            print(
                f"  {mode:24s} idx={idx} tau*/E={ratio:.3f}"
            )

    # Standard grid: short block forces mid≡2/3≡3q via integer floor.
    assert int(by[(False, "none")]["run"]) == 3
    assert int(by[(False, "none")]["i_lo"]) == 0
    assert int(by[(False, "none")]["i_hi"]) == 2
    assert _mid_interval_index(0, 2) == 1
    assert _two_thirds_index(0, 2) == 1
    assert _three_quarter_index(0, 2) == 1
    assert int(by[(False, "mid_interval")]["peak_index"]) == 1
    assert int(by[(False, "two_thirds_interval")]["peak_index"]) == 1
    assert int(by[(False, "three_quarter_interval")]["peak_index"]) == 1
    for mode in ("mid_interval", "two_thirds_interval", "three_quarter_interval"):
        assert abs(float(by[(False, mode)]["tau_over_expected"]) - 8.833) < 0.05
    assert int(by[(False, "fine_end_of_block")]["peak_index"]) == 2
    assert abs(float(by[(False, "none")]["tau_over_expected"]) - 16.0) < 0.05
    assert abs(float(by[(False, "fine_end_of_block")]["tau_over_expected"]) - 4.876) < 0.05
    # load_weighted still reproduces coarse-end on this fixture.
    assert int(by[(False, "load_weighted_interval")]["peak_index"]) == 0

    # Densify: full-grid persist (len=16) separates fractional landings.
    assert int(by[(True, "none")]["run"]) == 16
    assert int(by[(True, "mid_interval")]["peak_index"]) != int(
        by[(True, "two_thirds_interval")]["peak_index"]
    )
    assert 0.9 < float(by[(True, "two_thirds_interval")]["tau_over_expected"]) < 1.1
    assert float(by[(True, "three_quarter_interval")]["tau_over_expected"]) < 0.9
    assert int(by[(True, "load_weighted_interval")]["peak_index"]) == 0
    assert PersistenceConfig().resolve_within_interval == "none"
    assert ScaleSearchConfig().halve_grid_steps is False


def test_export_seed4_jaccard_half_step_table() -> None:
    # EXPORT (A6-T56): seed-4 matched-Jaccard / run_lengths table for A3 SI
    # S2.6.2 sync. Documents the densified first-half-step break
    # (ov0=0.39 < 0.5 ⇒ run0=1; long fineward run from idx1 discarded under
    # coarse_anchored). Standard grid remains fully agreeing (ov≥thr).
    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=4,
    )
    gt = dataset.ground_truth
    assert gt.expected_tau is not None
    tau_lo, tau_hi = gt.tau_grid_hint
    thr = float(PersistenceConfig().overlap_threshold)
    min_pers = int(PersistenceConfig().min_persistence)
    rows: list[dict[str, float | int | bool | None]] = []
    by_pr: dict[bool, PersistenceResult] = {}
    for dense in (False, True):
        result = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=ScaleSearchConfig(
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=4,
                halve_grid_steps=dense,
                persistence=PersistenceConfig(resolve_within_interval="none"),
            ),
        )
        assert result.persistence_result is not None
        pr = result.persistence_result
        by_pr[dense] = pr
        header = (
            f"{'dense':5s} {'i':>3s} {'tau':>10s} {'n_cl':>4s} "
            f"{'run':>4s} {'ov':>8s} {'agree':>5s}"
        )
        print(f"\nA6-T56 seed4 Jaccard half-step table (dense={dense})")
        print(header)
        print("-" * len(header))
        for i, snap in enumerate(pr.snapshots):
            ov = (
                float(pr.match_overlaps[i])
                if i < len(pr.match_overlaps)
                else float("nan")
            )
            agree = bool(np.isfinite(ov) and ov >= thr)
            rows.append(
                {
                    "dense": dense,
                    "i": i,
                    "tau": float(snap.tau),
                    "n_clusters": int(snap.n_clusters),
                    "run": int(pr.run_lengths[i]),
                    "overlap": ov,
                    "agree": agree,
                    "tsi": pr.tau_star_index,
                }
            )
            print(
                f"{str(dense):5s} {i:3d} {float(snap.tau):10.4g} "
                f"{int(snap.n_clusters):4d} {int(pr.run_lengths[i]):4d} "
                f"{ov:8.3f} {str(agree):>5}"
            )

    std_pr = by_pr[False]
    dense_pr = by_pr[True]
    # Standard: full coarse-anchored accept; every adjacent pair agrees.
    assert std_pr.tau_star_index == 0
    assert int(std_pr.run_lengths[0]) >= min_pers
    for i in range(len(std_pr.match_overlaps)):
        assert float(std_pr.match_overlaps[i]) >= thr
    # Dense: first half-step is the only disagreeing adjacent pair at coarse end.
    assert dense_pr.tau_star_index is None
    assert int(dense_pr.run_lengths[0]) == 1
    assert float(dense_pr.match_overlaps[0]) < thr
    assert abs(float(dense_pr.match_overlaps[0]) - 0.39) < 0.02
    assert int(dense_pr.snapshots[0].n_clusters) == 5
    assert int(dense_pr.snapshots[1].n_clusters) == 3
    assert int(dense_pr.run_lengths[1]) >= min_pers
    # Remaining dense adjacent pairs (from idx1 onward) agree.
    for i in range(1, len(dense_pr.match_overlaps)):
        assert float(dense_pr.match_overlaps[i]) >= thr
    assert PersistenceConfig().coarse_anchored is True
    assert PersistenceConfig().resolve_within_interval == "none"
    assert ScaleSearchConfig().halve_grid_steps is False
    assert len(rows) == 8 + 16


def test_lower_overlap_threshold_densify_recover_probe() -> None:
    # EXPERIMENT (A6-T58): flag-gated densify-overlap-recover.
    # Seed-4 × halve_grid_steps rejects under default overlap_threshold=0.5
    # (ov0≈0.39). densify_overlap_recover="lower_threshold" substitutes
    # EXPERIMENTAL_DENSIFY_OVERLAP_RECOVER_THRESHOLD (0.35) and recovers a
    # full coarse-anchored run (tsi=0, run0=16). Collateral: seed-1 also
    # flips reject→accept under the lower floor — probe only, do not flip
    # defaults / acceptance path.
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert EXPERIMENTAL_DENSIFY_OVERLAP_RECOVER_THRESHOLD == 0.35
    thr_default = float(PersistenceConfig().overlap_threshold)
    min_pers = int(PersistenceConfig().min_persistence)

    def _run(seed: int, recover: str) -> PersistenceResult:
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        tau_lo, tau_hi = gt.tau_grid_hint
        result = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=ScaleSearchConfig(
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=seed,
                halve_grid_steps=True,
                persistence=PersistenceConfig(
                    resolve_within_interval="none",
                    densify_overlap_recover=recover,  # type: ignore[arg-type]
                ),
            ),
        )
        assert result.persistence_result is not None
        return result.persistence_result

    none_pr = _run(4, "none")
    recover_pr = _run(4, "lower_threshold")
    print(
        f"\nA6-T58 seed4 dense recover: none ov0={float(none_pr.match_overlaps[0]):.3f} "
        f"tsi={none_pr.tau_star_index} run0={int(none_pr.run_lengths[0])}; "
        f"lower_threshold tsi={recover_pr.tau_star_index} "
        f"run0={int(recover_pr.run_lengths[0])}"
    )
    assert none_pr.tau_star_index is None
    assert int(none_pr.run_lengths[0]) == 1
    assert float(none_pr.match_overlaps[0]) < thr_default
    assert (
        float(none_pr.match_overlaps[0])
        >= EXPERIMENTAL_DENSIFY_OVERLAP_RECOVER_THRESHOLD
    )
    assert recover_pr.tau_star_index == 0
    assert int(recover_pr.run_lengths[0]) >= min_pers
    assert int(recover_pr.run_lengths[0]) == 16

    # Collateral: seed-1 densify also accepts under the lower floor.
    seed1_none = _run(1, "none")
    seed1_recover = _run(1, "lower_threshold")
    print(
        f"A6-T58 seed1 collateral: none tsi={seed1_none.tau_star_index} "
        f"ov0={float(seed1_none.match_overlaps[0]):.3f}; "
        f"lower_threshold tsi={seed1_recover.tau_star_index} "
        f"run0={int(seed1_recover.run_lengths[0])}"
    )
    assert seed1_none.tau_star_index is None
    assert seed1_recover.tau_star_index == 0
    assert int(seed1_recover.run_lengths[0]) >= min_pers
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().overlap_threshold == 0.5
    assert ScaleSearchConfig().halve_grid_steps is False


def test_export_multiseed_densify_jaccard_first_step_table() -> None:
    # EXPORT (A6-T60): multi-seed densify first-step matched-Jaccard table
    # (seeds 0..4 × standard vs halve_grid_steps) for A3 SI S2.6.2 sync.
    # Pins which seeds accept under default overlap_threshold and which
    # densified first half-steps break. Defaults stay off.
    thr = float(PersistenceConfig().overlap_threshold)
    min_pers = int(PersistenceConfig().min_persistence)
    rows: list[dict[str, float | int | bool | None]] = []
    by: dict[tuple[int, bool], PersistenceResult] = {}
    print("\nA6-T60 multi-seed densify Jaccard first-step table")
    header = (
        f"{'seed':>4s} {'dense':5s} {'ov0':>8s} {'run0':>4s} "
        f"{'tsi':>4s} {'n_cl0':>5s} {'n_cl1':>5s} {'accept':>6s}"
    )
    print(header)
    print("-" * len(header))
    for seed in range(5):
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        tau_lo, tau_hi = gt.tau_grid_hint
        for dense in (False, True):
            result = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=ScaleSearchConfig(
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
                        min_equilibrium_epochs=2, max_epochs=12
                    ),
                    seed=seed,
                    halve_grid_steps=dense,
                    persistence=PersistenceConfig(resolve_within_interval="none"),
                ),
            )
            assert result.persistence_result is not None
            pr = result.persistence_result
            by[(seed, dense)] = pr
            ov0 = float(pr.match_overlaps[0])
            accept = pr.tau_star_index is not None
            n0 = int(pr.snapshots[0].n_clusters)
            n1 = int(pr.snapshots[1].n_clusters)
            rows.append(
                {
                    "seed": seed,
                    "dense": dense,
                    "ov0": ov0,
                    "run0": int(pr.run_lengths[0]),
                    "tsi": pr.tau_star_index,
                    "n_cl0": n0,
                    "n_cl1": n1,
                    "accept": accept,
                }
            )
            print(
                f"{seed:4d} {str(dense):5s} {ov0:8.3f} "
                f"{int(pr.run_lengths[0]):4d} "
                f"{str(pr.tau_star_index):>4s} {n0:5d} {n1:5d} "
                f"{str(accept):>6}"
            )

    # Persist-accept under default thr: {0,3} both grids; {4} std only.
    for seed in (0, 3):
        for dense in (False, True):
            pr = by[(seed, dense)]
            assert pr.tau_star_index == 0
            assert int(pr.run_lengths[0]) >= min_pers
            assert float(pr.match_overlaps[0]) >= thr
    pr4_std = by[(4, False)]
    pr4_dense = by[(4, True)]
    assert pr4_std.tau_star_index == 0
    assert float(pr4_std.match_overlaps[0]) >= thr
    assert pr4_dense.tau_star_index is None
    assert float(pr4_dense.match_overlaps[0]) < thr
    assert abs(float(pr4_dense.match_overlaps[0]) - 0.39) < 0.02
    # Persist-reject both grids: seeds 1–2 (first-step ov below thr).
    for seed in (1, 2):
        for dense in (False, True):
            pr = by[(seed, dense)]
            assert pr.tau_star_index is None
            assert int(pr.run_lengths[0]) == 1
            assert float(pr.match_overlaps[0]) < thr
    # Pin approximate first-step overlaps for SI sync (tolerance 0.03).
    expected_ov0 = {
        (0, False): 0.504,
        (0, True): 0.504,
        (1, False): 0.471,
        (1, True): 0.364,
        (2, False): 0.200,
        (2, True): 0.340,
        (3, False): 0.503,
        (3, True): 0.503,
        (4, False): 0.557,
        (4, True): 0.390,
    }
    for key, want in expected_ov0.items():
        assert abs(float(by[key].match_overlaps[0]) - want) < 0.03
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().resolve_within_interval == "none"
    assert ScaleSearchConfig().halve_grid_steps is False
    assert len(rows) == 10


def test_circle_densify_x_load_weighted_lc_identity() -> None:
    # EXPERIMENT (A6-T57): circle × halve_grid_steps × load_weighted_interval.
    # No accepted persist split ⇒ load_weighted is a controller LC no-op and
    # must match resolve=none peak/tau* on both grids (densify may move the LC
    # peak; it must not create mode divergence). Defaults stay off.
    dataset = make_circle(
        n_samples=800, radius=1.0, noise=0.02, extrusion_dim=2, seed=21,
    )
    gt = dataset.ground_truth
    assert gt.expected_tau is not None
    tau_lo, tau_hi = gt.tau_grid_hint
    rows: list[dict[str, float | int | bool | None]] = []
    for dense in (False, True):
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
                min_equilibrium_epochs=2, max_epochs=12
            ),
            seed=0,
            halve_grid_steps=dense,
        )
        none = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=replace(
                base,
                persistence=PersistenceConfig(resolve_within_interval="none"),
            ),
        )
        weighted = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=replace(
                base,
                persistence=PersistenceConfig(
                    resolve_within_interval="load_weighted_interval"
                ),
            ),
        )
        assert none.persistence_result is not None
        assert weighted.persistence_result is not None
        assert none.persistence_result.tau_star_index is None
        assert weighted.persistence_result.tau_star_index is None
        assert none.peak_index is not None
        assert weighted.peak_index is not None
        assert int(weighted.peak_index) == int(none.peak_index)
        assert float(weighted.tau_star) == float(none.tau_star)
        rows.append(
            {
                "dense": dense,
                "n_grid": int(len(none.tau_grid)),
                "peak_index": int(none.peak_index),
                "tau_star": float(none.tau_star),
                "tau_over_expected": float(none.tau_star / gt.expected_tau),
            }
        )
        print(
            f"\nA6-T57 circle densify={dense}: n_grid={len(none.tau_grid)} "
            f"peak={none.peak_index} tau*/E="
            f"{float(none.tau_star / gt.expected_tau):.3f} "
            f"LW≡none"
        )

    assert int(rows[1]["n_grid"]) > int(rows[0]["n_grid"])
    # Densify moves the LC peak on this fixture (std idx2 → dense idx4).
    assert int(rows[1]["peak_index"]) != int(rows[0]["peak_index"])
    assert PersistenceConfig().resolve_within_interval == "none"
    assert ScaleSearchConfig().halve_grid_steps is False


def test_densify_recover_collateral_sweep_seeds0_4() -> None:
    # EXPERIMENT (A6-T61): full densify-overlap-recover collateral sweep
    # (seeds 0..4 × standard vs densify × none/lower_threshold). Pins the
    # accept map under thr=0.35 and the none→recover flips. Default stays
    # densify_overlap_recover="none"; do not flip acceptance.
    assert PersistenceConfig().densify_overlap_recover == "none"
    thr = float(PersistenceConfig().overlap_threshold)
    recover_thr = float(EXPERIMENTAL_DENSIFY_OVERLAP_RECOVER_THRESHOLD)
    min_pers = int(PersistenceConfig().min_persistence)
    assert recover_thr == 0.35
    assert thr == 0.5

    by: dict[tuple[int, bool, str], PersistenceResult] = {}
    print("\nA6-T61 densify-recover collateral sweep (seeds0..4)")
    header = (
        f"{'seed':>4s} {'dense':5s} {'recover':16s} {'ov0':>8s} "
        f"{'run0':>4s} {'tsi':>4s} {'accept':>6s}"
    )
    print(header)
    print("-" * len(header))
    for seed in range(5):
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        tau_lo, tau_hi = gt.tau_grid_hint
        for dense in (False, True):
            for recover in ("none", "lower_threshold"):
                result = run_scale_search(
                    dataset.points,
                    dim=gt.ambient_dim,
                    config=ScaleSearchConfig(
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
                            min_equilibrium_epochs=2, max_epochs=12
                        ),
                        seed=seed,
                        halve_grid_steps=dense,
                        persistence=PersistenceConfig(
                            resolve_within_interval="none",
                            densify_overlap_recover=recover,  # type: ignore[arg-type]
                        ),
                    ),
                )
                assert result.persistence_result is not None
                pr = result.persistence_result
                by[(seed, dense, recover)] = pr
                print(
                    f"{seed:4d} {str(dense):5s} {recover:16s} "
                    f"{float(pr.match_overlaps[0]):8.3f} "
                    f"{int(pr.run_lengths[0]):4d} "
                    f"{str(pr.tau_star_index):>4s} "
                    f"{str(pr.tau_star_index is not None):>6}"
                )

    # Accept under lower_threshold: {0,1,3,4} both grids; seed2 reject both.
    for seed in (0, 1, 3, 4):
        for dense in (False, True):
            pr = by[(seed, dense, "lower_threshold")]
            assert pr.tau_star_index == 0
            assert int(pr.run_lengths[0]) >= min_pers
    for dense in (False, True):
        pr = by[(2, dense, "lower_threshold")]
        assert pr.tau_star_index is None
        assert int(pr.run_lengths[0]) == 1
        assert float(pr.match_overlaps[0]) < recover_thr

    # none→recover flips: seed1 std+dense, seed4 dense only.
    flips = {
        (seed, dense)
        for seed in range(5)
        for dense in (False, True)
        if (by[(seed, dense, "none")].tau_star_index is None)
        != (by[(seed, dense, "lower_threshold")].tau_star_index is None)
    }
    assert flips == {(1, False), (1, True), (4, True)}
    for key in flips:
        assert by[(*key, "none")].tau_star_index is None
        assert by[(*key, "lower_threshold")].tau_star_index == 0

    # Seed-3 std collateral lengthening (short block expands under lower floor).
    assert int(by[(3, False, "none")].run_lengths[0]) == 3
    assert int(by[(3, False, "lower_threshold")].run_lengths[0]) == 5
    # Seed-4 dense recover pins full-grid run.
    assert int(by[(4, True, "lower_threshold")].run_lengths[0]) == 16
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert ScaleSearchConfig().halve_grid_steps is False


def test_seed3_dense_phi_matches_seed0_landing_indices() -> None:
    # EXPORT (A6-T62): seed~3 densify restores the same fractional landing
    # indices / tau*/E hierarchy as seed~0 densify (mid~2.30×, 2/3~1.00×,
    # 3q~0.76×, fine~0.25×). Contrasts the std short-block collapse where
    # mid≡2/3≡3q at ~8.83×. Phi magnitudes may differ; defaults stay off.
    modes = (
        "none",
        "mid_interval",
        "two_thirds_interval",
        "three_quarter_interval",
        "fine_end_of_block",
        "load_weighted_interval",
    )
    rows: dict[tuple[int, bool, str], dict[str, float | int]] = {}
    print("\nA6-T62 seed3 dense Phi pin vs seed0")
    header = (
        f"{'seed':>4s} {'dense':5s} {'mode':24s} "
        f"{'idx':>3s} {'tau*/E':>8s} {'Phi*':>10s}"
    )
    print(header)
    print("-" * len(header))
    for seed in (0, 3):
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        assert gt.expected_tau is not None
        tau_lo, tau_hi = gt.tau_grid_hint
        for dense in (False, True):
            result = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=ScaleSearchConfig(
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
                        min_equilibrium_epochs=2, max_epochs=12
                    ),
                    seed=seed,
                    halve_grid_steps=dense,
                    persistence=PersistenceConfig(resolve_within_interval="none"),
                ),
            )
            assert result.persistence_result is not None
            pr = result.persistence_result
            assert pr.tau_star_index is not None
            for mode in modes:
                idx = _resolve_persistence_tau_index(
                    pr,
                    result.load_trace,
                    list(result.stabilized_flags),
                    PersistenceConfig(resolve_within_interval=mode),  # type: ignore[arg-type]
                )
                tau_star = float(result.tau_grid[idx])
                phi_star = float(result.phi_trace[idx])
                ratio = float(tau_star / gt.expected_tau)
                rows[(seed, dense, mode)] = {
                    "idx": int(idx),
                    "ratio": ratio,
                    "phi": phi_star,
                }
                print(
                    f"{seed:4d} {str(dense):5s} {mode:24s} "
                    f"{idx:3d} {ratio:8.3f} {phi_star:10.4g}"
                )

    # Seed-3 std short-block: mid ≡ 2/3 ≡ 3q at idx1 / ~8.83×.
    for mode in ("mid_interval", "two_thirds_interval", "three_quarter_interval"):
        assert int(rows[(3, False, mode)]["idx"]) == 1
        assert abs(float(rows[(3, False, mode)]["ratio"]) - 8.833) < 0.05
    assert int(rows[(3, False, "fine_end_of_block")]["idx"]) == 2
    assert abs(float(rows[(3, False, "fine_end_of_block")]["ratio"]) - 4.876) < 0.05

    # Densify: seed3 matches seed0 landing indices and tau*/E hierarchy.
    for mode in modes:
        assert int(rows[(3, True, mode)]["idx"]) == int(rows[(0, True, mode)]["idx"])
        assert abs(
            float(rows[(3, True, mode)]["ratio"]) - float(rows[(0, True, mode)]["ratio"])
        ) < 0.05
        assert np.isfinite(float(rows[(3, True, mode)]["phi"]))
        assert np.isfinite(float(rows[(0, True, mode)]["phi"]))
    # Published dense hierarchy ratios (same as seed0 densify flip).
    assert abs(float(rows[(3, True, "none")]["ratio"]) - 16.0) < 0.05
    assert abs(float(rows[(3, True, "mid_interval")]["ratio"]) - 2.297) < 0.05
    assert abs(float(rows[(3, True, "two_thirds_interval")]["ratio"]) - 1.0) < 0.05
    assert abs(float(rows[(3, True, "three_quarter_interval")]["ratio"]) - 0.758) < 0.05
    assert abs(float(rows[(3, True, "fine_end_of_block")]["ratio"]) - 0.25) < 0.05
    assert int(rows[(3, True, "load_weighted_interval")]["idx"]) == int(
        rows[(3, True, "none")]["idx"]
    )
    assert PersistenceConfig().resolve_within_interval == "none"
    assert ScaleSearchConfig().halve_grid_steps is False


def test_densify_recover_threshold_sensitivity_seeds0_4() -> None:
    # EXPERIMENT (A6-T64): recover-thr sensitivity at 0.30 / 0.35 / 0.40
    # against the seeds0..4 × std/dense collateral map. Pins that 0.35 is
    # the narrow band recovering densified seed~4 without accepting densified
    # seed~2; 0.30 over-accepts seed~2 dense; 0.40 loses seed~1 dense and
    # seed~4 dense recovers. Default densify_overlap_recover stays "none".
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert EXPERIMENTAL_DENSIFY_OVERLAP_RECOVER_THRESHOLD == 0.35

    by: dict[tuple[float, int, bool], PersistenceResult] = {}
    print("\nA6-T64 densify-recover thr sensitivity (0.30/0.35/0.40)")
    header = (
        f"{'thr':>5s} {'seed':>4s} {'dense':5s} {'ov0':>8s} "
        f"{'run0':>4s} {'tsi':>4s} {'accept':>6s}"
    )
    print(header)
    print("-" * len(header))
    for thr in (0.30, 0.35, 0.40):
        for seed in range(5):
            dataset = make_hierarchical_gaussian(
                children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
            )
            gt = dataset.ground_truth
            tau_lo, tau_hi = gt.tau_grid_hint
            for dense in (False, True):
                result = run_scale_search(
                    dataset.points,
                    dim=gt.ambient_dim,
                    config=ScaleSearchConfig(
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
                            min_equilibrium_epochs=2, max_epochs=12
                        ),
                        seed=seed,
                        halve_grid_steps=dense,
                        persistence=PersistenceConfig(
                            resolve_within_interval="none",
                            densify_overlap_recover="lower_threshold",
                            densify_overlap_recover_threshold=thr,
                        ),
                    ),
                )
                assert result.persistence_result is not None
                pr = result.persistence_result
                by[(thr, seed, dense)] = pr
                print(
                    f"{thr:5.2f} {seed:4d} {str(dense):5s} "
                    f"{float(pr.match_overlaps[0]):8.3f} "
                    f"{int(pr.run_lengths[0]):4d} "
                    f"{str(pr.tau_star_index):>4s} "
                    f"{str(pr.tau_star_index is not None):>6}"
                )

    def _accept_pairs(thr: float) -> set[tuple[int, bool]]:
        return {
            (seed, dense)
            for seed in range(5)
            for dense in (False, True)
            if by[(thr, seed, dense)].tau_star_index is not None
        }

    # thr=0.35: published T61 map — accept {0,1,3,4} both grids; seed2 reject.
    assert _accept_pairs(0.35) == {
        (0, False), (0, True),
        (1, False), (1, True),
        (3, False), (3, True),
        (4, False), (4, True),
    }
    assert by[(0.35, 2, False)].tau_star_index is None
    assert by[(0.35, 2, True)].tau_star_index is None
    assert abs(float(by[(0.35, 2, True)].match_overlaps[0]) - 0.340) < 0.02
    assert int(by[(0.35, 3, False)].run_lengths[0]) == 5
    assert int(by[(0.35, 4, True)].run_lengths[0]) == 16

    # thr=0.30: additionally accepts densified seed~2 (ov0≈0.34 ≥ 0.30).
    assert _accept_pairs(0.30) == _accept_pairs(0.35) | {(2, True)}
    assert by[(0.30, 2, True)].tau_star_index == 0
    assert int(by[(0.30, 2, True)].run_lengths[0]) == 16
    assert by[(0.30, 2, False)].tau_star_index is None  # ov0≈0.20 still below

    # thr=0.40: loses seed~1 dense + seed~4 dense recovers; seed~3 std no longer
    # lengthens (run0 stays 3, same as default-threshold short block).
    assert _accept_pairs(0.40) == {
        (0, False), (0, True),
        (1, False),
        (3, False), (3, True),
        (4, False),
    }
    assert by[(0.40, 1, True)].tau_star_index is None
    assert abs(float(by[(0.40, 1, True)].match_overlaps[0]) - 0.364) < 0.02
    assert by[(0.40, 4, True)].tau_star_index is None
    assert abs(float(by[(0.40, 4, True)].match_overlaps[0]) - 0.390) < 0.02
    assert int(by[(0.40, 3, False)].run_lengths[0]) == 3

    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert EXPERIMENTAL_DENSIFY_OVERLAP_RECOVER_THRESHOLD == 0.35
    assert ScaleSearchConfig().halve_grid_steps is False


def test_seed3_densify_load_weighted_stays_coarse() -> None:
    # EXPERIMENT (A6-T65): densify × load_weighted_interval on seed~3.
    # LW stays at the coarse-end arbiter (idx0 / 16×) under std and densify,
    # with or without densify-overlap-recover — unlike mid/2/3/3q which move
    # finer as the block lengthens. Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert ScaleSearchConfig().halve_grid_steps is False

    modes = (
        "none",
        "mid_interval",
        "two_thirds_interval",
        "three_quarter_interval",
        "load_weighted_interval",
        "fine_end_of_block",
    )
    rows: dict[tuple[bool, str, str], dict[str, float | int]] = {}
    print("\nA6-T65 seed3 densify × load_weighted")
    header = (
        f"{'dense':5s} {'recover':16s} {'mode':24s} "
        f"{'idx':>3s} {'tau*/E':>8s} {'run0':>4s}"
    )
    print(header)
    print("-" * len(header))
    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=3,
    )
    gt = dataset.ground_truth
    assert gt.expected_tau is not None
    tau_lo, tau_hi = gt.tau_grid_hint
    for dense in (False, True):
        for recover in ("none", "lower_threshold"):
            result = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=ScaleSearchConfig(
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
                        min_equilibrium_epochs=2, max_epochs=12
                    ),
                    seed=3,
                    halve_grid_steps=dense,
                    persistence=PersistenceConfig(
                        resolve_within_interval="none",
                        densify_overlap_recover=recover,  # type: ignore[arg-type]
                    ),
                ),
            )
            assert result.persistence_result is not None
            pr = result.persistence_result
            assert pr.tau_star_index == 0
            run0 = int(pr.run_lengths[0])
            for mode in modes:
                idx = _resolve_persistence_tau_index(
                    pr,
                    result.load_trace,
                    list(result.stabilized_flags),
                    PersistenceConfig(resolve_within_interval=mode),  # type: ignore[arg-type]
                )
                ratio = float(result.tau_grid[idx]) / float(gt.expected_tau)
                rows[(dense, recover, mode)] = {
                    "idx": int(idx),
                    "ratio": ratio,
                    "run0": run0,
                }
                print(
                    f"{str(dense):5s} {recover:16s} {mode:24s} "
                    f"{idx:3d} {ratio:8.3f} {run0:4d}"
                )

    # LW ≡ none (coarse) in every densify/recover cell.
    for dense in (False, True):
        for recover in ("none", "lower_threshold"):
            assert int(rows[(dense, recover, "load_weighted_interval")]["idx"]) == 0
            assert abs(
                float(rows[(dense, recover, "load_weighted_interval")]["ratio"]) - 16.0
            ) < 0.05
            assert int(rows[(dense, recover, "none")]["idx"]) == 0

    # Densify (recover irrelevant — full-grid run already): fractional hierarchy.
    for recover in ("none", "lower_threshold"):
        assert int(rows[(True, recover, "mid_interval")]["idx"]) == 7
        assert abs(float(rows[(True, recover, "mid_interval")]["ratio"]) - 2.297) < 0.05
        assert int(rows[(True, recover, "two_thirds_interval")]["idx"]) == 10
        assert abs(
            float(rows[(True, recover, "two_thirds_interval")]["ratio"]) - 1.0
        ) < 0.05
        assert int(rows[(True, recover, "three_quarter_interval")]["idx"]) == 11
        assert abs(
            float(rows[(True, recover, "three_quarter_interval")]["ratio"]) - 0.758
        ) < 0.05
        assert int(rows[(True, recover, "fine_end_of_block")]["idx"]) == 15
        assert abs(
            float(rows[(True, recover, "fine_end_of_block")]["ratio"]) - 0.25
        ) < 0.05
        assert int(rows[(True, recover, "none")]["run0"]) == 16

    # Std + recover lengthens short block; mid/2/3/3q move, LW stays coarse.
    assert int(rows[(False, "none", "none")]["run0"]) == 3
    assert int(rows[(False, "lower_threshold", "none")]["run0"]) == 5
    assert int(rows[(False, "lower_threshold", "mid_interval")]["idx"]) == 2
    assert abs(
        float(rows[(False, "lower_threshold", "mid_interval")]["ratio"]) - 4.876
    ) < 0.05
    assert int(rows[(False, "lower_threshold", "two_thirds_interval")]["idx"]) == 2
    assert int(rows[(False, "lower_threshold", "three_quarter_interval")]["idx"]) == 3
    assert abs(
        float(rows[(False, "lower_threshold", "three_quarter_interval")]["ratio"])
        - 2.692
    ) < 0.05
    assert int(rows[(False, "lower_threshold", "fine_end_of_block")]["idx"]) == 4
    assert abs(
        float(rows[(False, "lower_threshold", "fine_end_of_block")]["ratio"]) - 1.486
    ) < 0.05

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert ScaleSearchConfig().halve_grid_steps is False


def test_densify_recover_thr_phi_export_seeds0_4() -> None:
    # EXPORT (burn follow-on pre-formal A6-T67): thr × Phi_C under
    # densify_overlap_recover_threshold ∈ {0.30, 0.35, 0.40}. Pins that the
    # override is a Jaccard accept/reject gate only — when the same densified
    # seed accepts under two floors, coarse-end and mid-interval Phi match
    # exactly. Defaults stay off. (A1 formal A6-T67 is densify×LW×thr combo.)
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert EXPERIMENTAL_DENSIFY_OVERLAP_RECOVER_THRESHOLD == 0.35

    phi_rows: dict[tuple[float, int], dict[str, float | int | None]] = {}
    print("\nA6-T67 thr × Phi densify export")
    header = (
        f"{'thr':>5s} {'seed':>4s} {'accept':>6s} {'run0':>4s} "
        f"{'Phi0':>10s} {'Phi_mid':>10s} {'idx_mid':>7s}"
    )
    print(header)
    print("-" * len(header))
    for thr in (0.30, 0.35, 0.40):
        for seed in range(5):
            dataset = make_hierarchical_gaussian(
                children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
            )
            gt = dataset.ground_truth
            tau_lo, tau_hi = gt.tau_grid_hint
            result = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=ScaleSearchConfig(
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
                        min_equilibrium_epochs=2, max_epochs=12
                    ),
                    seed=seed,
                    halve_grid_steps=True,
                    persistence=PersistenceConfig(
                        resolve_within_interval="none",
                        densify_overlap_recover="lower_threshold",
                        densify_overlap_recover_threshold=thr,
                    ),
                ),
            )
            assert result.persistence_result is not None
            pr = result.persistence_result
            accept = pr.tau_star_index is not None
            if accept:
                idx_mid = _resolve_persistence_tau_index(
                    pr,
                    result.load_trace,
                    list(result.stabilized_flags),
                    PersistenceConfig(resolve_within_interval="mid_interval"),
                )
                phi0 = float(result.phi_trace[0])
                phi_mid = float(result.phi_trace[idx_mid])
            else:
                idx_mid = None
                phi0 = float("nan")
                phi_mid = float("nan")
            phi_rows[(thr, seed)] = {
                "accept": int(accept),
                "run0": int(pr.run_lengths[0]),
                "phi0": phi0,
                "phi_mid": phi_mid,
                "idx_mid": idx_mid if idx_mid is not None else -1,
            }
            print(
                f"{thr:5.2f} {seed:4d} {str(accept):>6s} "
                f"{int(pr.run_lengths[0]):4d} {phi0:10.4g} "
                f"{phi_mid:10.4g} {str(idx_mid):>7s}"
            )

    # Accept map matches T64 densify column.
    for seed in (0, 1, 3, 4):
        assert phi_rows[(0.35, seed)]["accept"] == 1
    assert phi_rows[(0.35, 2)]["accept"] == 0
    assert phi_rows[(0.30, 2)]["accept"] == 1
    assert phi_rows[(0.40, 1)]["accept"] == 0
    assert phi_rows[(0.40, 4)]["accept"] == 0

    # Shared accepts: Phi0 / Phi_mid identical across thr (gate-only).
    for seed in (0, 3):
        for thr_a, thr_b in ((0.30, 0.35), (0.35, 0.40), (0.30, 0.40)):
            assert abs(
                float(phi_rows[(thr_a, seed)]["phi0"])
                - float(phi_rows[(thr_b, seed)]["phi0"])
            ) < 1e-9
            assert abs(
                float(phi_rows[(thr_a, seed)]["phi_mid"])
                - float(phi_rows[(thr_b, seed)]["phi_mid"])
            ) < 1e-9
            assert int(phi_rows[(thr_a, seed)]["idx_mid"]) == int(
                phi_rows[(thr_b, seed)]["idx_mid"]
            )
    for thr_a, thr_b in ((0.30, 0.35),):
        for seed in (1, 4):
            assert abs(
                float(phi_rows[(thr_a, seed)]["phi0"])
                - float(phi_rows[(thr_b, seed)]["phi0"])
            ) < 1e-9
            assert abs(
                float(phi_rows[(thr_a, seed)]["phi_mid"])
                - float(phi_rows[(thr_b, seed)]["phi_mid"])
            ) < 1e-9

    # seed2 contributes a Phi row only at thr=0.30 (over-accept).
    assert np.isfinite(float(phi_rows[(0.30, 2)]["phi0"]))
    assert int(phi_rows[(0.30, 2)]["idx_mid"]) == 7
    assert not np.isfinite(float(phi_rows[(0.35, 2)]["phi0"]))
    assert not np.isfinite(float(phi_rows[(0.40, 2)]["phi0"]))

    # Published dense mid landings (seed0/3 hierarchy).
    assert int(phi_rows[(0.35, 0)]["idx_mid"]) == 7
    assert int(phi_rows[(0.35, 3)]["idx_mid"]) == 7

    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_seed0_seed4_densify_load_weighted_stays_coarse() -> None:
    # EXPERIMENT (burn follow-on pre-formal A6-T68): densify × load_weighted on
    # seeds~0 and~4. Extends T65: LW stays coarse-end alias whenever the split
    # is accepted at the default recover floor; seed~0 is recover-invariant;
    # densified seed~4 needs recover to expose the densify hierarchy matching
    # seed~0. Defaults stay off. (A1 formal A6-T68 is thr0.30 seed2 ov0 pin.)
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert ScaleSearchConfig().halve_grid_steps is False

    modes = (
        "none",
        "mid_interval",
        "two_thirds_interval",
        "three_quarter_interval",
        "load_weighted_interval",
        "fine_end_of_block",
    )
    rows: dict[tuple[int, bool, str, str], dict[str, float | int] | None] = {}
    print("\nA6-T68 seed0/4 densify × load_weighted")
    header = (
        f"{'seed':>4s} {'dense':5s} {'recover':16s} {'mode':24s} "
        f"{'idx':>3s} {'tau*/E':>8s} {'run0':>4s}"
    )
    print(header)
    print("-" * len(header))
    for seed in (0, 4):
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        assert gt.expected_tau is not None
        tau_lo, tau_hi = gt.tau_grid_hint
        for dense in (False, True):
            for recover in ("none", "lower_threshold"):
                result = run_scale_search(
                    dataset.points,
                    dim=gt.ambient_dim,
                    config=ScaleSearchConfig(
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
                            min_equilibrium_epochs=2, max_epochs=12
                        ),
                        seed=seed,
                        halve_grid_steps=dense,
                        persistence=PersistenceConfig(
                            resolve_within_interval="none",
                            densify_overlap_recover=recover,  # type: ignore[arg-type]
                        ),
                    ),
                )
                assert result.persistence_result is not None
                pr = result.persistence_result
                if pr.tau_star_index is None:
                    for mode in modes:
                        rows[(seed, dense, recover, mode)] = None
                    print(
                        f"{seed:4d} {str(dense):5s} {recover:16s} "
                        f"{'(reject)':24s} {'—':>3s} {'—':>8s} "
                        f"{int(pr.run_lengths[0]):4d}"
                    )
                    continue
                assert pr.tau_star_index == 0
                run0 = int(pr.run_lengths[0])
                for mode in modes:
                    idx = _resolve_persistence_tau_index(
                        pr,
                        result.load_trace,
                        list(result.stabilized_flags),
                        PersistenceConfig(resolve_within_interval=mode),  # type: ignore[arg-type]
                    )
                    ratio = float(result.tau_grid[idx]) / float(gt.expected_tau)
                    rows[(seed, dense, recover, mode)] = {
                        "idx": int(idx),
                        "ratio": ratio,
                        "run0": run0,
                    }
                    print(
                        f"{seed:4d} {str(dense):5s} {recover:16s} {mode:24s} "
                        f"{idx:3d} {ratio:8.3f} {run0:4d}"
                    )

    # seed4 dense + none: persist-reject (recover required).
    assert rows[(4, True, "none", "none")] is None

    # LW ≡ none (coarse 16×) in every accepted cell.
    for seed in (0, 4):
        for dense in (False, True):
            for recover in ("none", "lower_threshold"):
                lw = rows[(seed, dense, recover, "load_weighted_interval")]
                none = rows[(seed, dense, recover, "none")]
                if none is None:
                    assert lw is None
                    continue
                assert lw is not None
                assert int(lw["idx"]) == 0
                assert abs(float(lw["ratio"]) - 16.0) < 0.05
                assert int(lw["idx"]) == int(none["idx"])

    # seed0 recover-invariant on both grids.
    for dense in (False, True):
        for mode in modes:
            a = rows[(0, dense, "none", mode)]
            b = rows[(0, dense, "lower_threshold", mode)]
            assert a is not None and b is not None
            assert int(a["idx"]) == int(b["idx"])
            assert abs(float(a["ratio"]) - float(b["ratio"])) < 0.05

    # Densify hierarchy (accepted): seed0 (any recover) and seed4+recover match.
    for key in ((0, True, "none"), (0, True, "lower_threshold"), (4, True, "lower_threshold")):
        assert abs(float(rows[(*key, "none")]["ratio"]) - 16.0) < 0.05  # type: ignore[index]
        assert abs(float(rows[(*key, "mid_interval")]["ratio"]) - 2.297) < 0.05  # type: ignore[index]
        assert abs(
            float(rows[(*key, "two_thirds_interval")]["ratio"]) - 1.0  # type: ignore[index]
        ) < 0.05
        assert abs(
            float(rows[(*key, "three_quarter_interval")]["ratio"]) - 0.758  # type: ignore[index]
        ) < 0.05
        assert abs(
            float(rows[(*key, "fine_end_of_block")]["ratio"]) - 0.25  # type: ignore[index]
        ) < 0.05
        assert int(rows[(*key, "none")]["run0"]) == 16  # type: ignore[index]

    # Std seed0/4: same fractional indices (run0=8); LW coarse.
    for seed in (0, 4):
        for recover in ("none", "lower_threshold"):
            assert int(rows[(seed, False, recover, "none")]["run0"]) == 8  # type: ignore[index]
            assert int(rows[(seed, False, recover, "mid_interval")]["idx"]) == 3  # type: ignore[index]
            assert abs(
                float(rows[(seed, False, recover, "mid_interval")]["ratio"]) - 2.692  # type: ignore[index]
            ) < 0.05
            assert int(rows[(seed, False, recover, "two_thirds_interval")]["idx"]) == 4  # type: ignore[index]
            assert int(rows[(seed, False, recover, "three_quarter_interval")]["idx"]) == 5  # type: ignore[index]
            assert int(rows[(seed, False, recover, "fine_end_of_block")]["idx"]) == 7  # type: ignore[index]

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert ScaleSearchConfig().halve_grid_steps is False


def test_densify_lw_recover_thr_combo_seeds0_4() -> None:
    # EXPERIMENT (A6-T67 formal): densify × load_weighted × recover-thr combo
    # on seeds0..4. Crosses halve_grid_steps × densify_overlap_recover_threshold
    # ∈ {0.30, 0.35, 0.40} and resolves none vs load_weighted_interval.
    # Pins: (1) accept map matches T64 thr sensitivity; (2) LW ≡ coarse-end
    # (idx0 / 16×) on every accepted cell *except* the thr=0.30 densified
    # seed~2 over-accept, where LW steps one index finer (~12.1×) while
    # none stays at coarse-end — the first LW≠coarse divergence under the
    # recover-thr lever. Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert EXPERIMENTAL_DENSIFY_OVERLAP_RECOVER_THRESHOLD == 0.35

    by: dict[tuple[float, int, bool], dict[str, float | int | None]] = {}
    print("\nA6-T67 densify × LW × recover-thr combo")
    header = (
        f"{'thr':>5s} {'seed':>4s} {'dense':5s} {'acc':>5s} {'ov0':>7s} "
        f"{'run0':>4s} {'LW':>3s} {'none':>4s} {'LW*':>7s}"
    )
    print(header)
    print("-" * len(header))
    for thr in (0.30, 0.35, 0.40):
        for seed in range(5):
            dataset = make_hierarchical_gaussian(
                children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
            )
            gt = dataset.ground_truth
            assert gt.expected_tau is not None
            tau_lo, tau_hi = gt.tau_grid_hint
            for dense in (False, True):
                result = run_scale_search(
                    dataset.points,
                    dim=gt.ambient_dim,
                    config=ScaleSearchConfig(
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
                            min_equilibrium_epochs=2, max_epochs=12
                        ),
                        seed=seed,
                        halve_grid_steps=dense,
                        persistence=PersistenceConfig(
                            resolve_within_interval="none",
                            densify_overlap_recover="lower_threshold",
                            densify_overlap_recover_threshold=thr,
                        ),
                    ),
                )
                assert result.persistence_result is not None
                pr = result.persistence_result
                accept = pr.tau_star_index is not None
                ov0 = float(pr.match_overlaps[0])
                run0 = int(pr.run_lengths[0])
                if accept:
                    idx_lw = _resolve_persistence_tau_index(
                        pr,
                        result.load_trace,
                        list(result.stabilized_flags),
                        PersistenceConfig(
                            resolve_within_interval="load_weighted_interval"
                        ),
                    )
                    idx_none = _resolve_persistence_tau_index(
                        pr,
                        result.load_trace,
                        list(result.stabilized_flags),
                        PersistenceConfig(resolve_within_interval="none"),
                    )
                    ratio = float(result.tau_grid[idx_lw]) / float(gt.expected_tau)
                else:
                    idx_lw = None
                    idx_none = None
                    ratio = float("nan")
                by[(thr, seed, dense)] = {
                    "accept": int(accept),
                    "ov0": ov0,
                    "run0": run0,
                    "idx_lw": idx_lw if idx_lw is not None else -1,
                    "idx_none": idx_none if idx_none is not None else -1,
                    "ratio": ratio,
                }
                print(
                    f"{thr:5.2f} {seed:4d} {str(dense):5s} {str(accept):>5s} "
                    f"{ov0:7.3f} {run0:4d} {str(idx_lw):>3s} "
                    f"{str(idx_none):>4s} {ratio:7.3f}"
                )

    def _accept_pairs(thr: float) -> set[tuple[int, bool]]:
        return {
            (seed, dense)
            for seed in range(5)
            for dense in (False, True)
            if by[(thr, seed, dense)]["accept"] == 1
        }

    # Accept map ≡ T64 thr sensitivity.
    assert _accept_pairs(0.35) == {
        (0, False), (0, True),
        (1, False), (1, True),
        (3, False), (3, True),
        (4, False), (4, True),
    }
    assert _accept_pairs(0.30) == _accept_pairs(0.35) | {(2, True)}
    assert _accept_pairs(0.40) == {
        (0, False), (0, True),
        (1, False),
        (3, False), (3, True),
        (4, False),
    }

    # LW ≡ coarse-end on every accepted cell except thr0.30 densified seed2.
    for thr in (0.30, 0.35, 0.40):
        for seed in range(5):
            for dense in (False, True):
                cell = by[(thr, seed, dense)]
                if cell["accept"] == 0:
                    assert int(cell["idx_lw"]) == -1
                    continue
                if (thr, seed, dense) == (0.30, 2, True):
                    continue  # divergence pinned below
                assert int(cell["idx_lw"]) == 0
                assert int(cell["idx_none"]) == 0
                assert abs(float(cell["ratio"]) - 16.0) < 0.05

    # First LW≠coarse divergence: thr=0.30 densified seed~2 over-accept.
    seed2 = by[(0.30, 2, True)]
    assert seed2["accept"] == 1
    assert int(seed2["idx_none"]) == 0
    assert int(seed2["idx_lw"]) == 1
    assert abs(float(seed2["ratio"]) - 12.126) < 0.05
    assert int(seed2["run0"]) == 16

    # Seed~3 densify×LW stays coarse across the thr band (T65/T69 pin).
    for thr in (0.30, 0.35, 0.40):
        for dense in (False, True):
            cell = by[(thr, 3, dense)]
            assert cell["accept"] == 1
            assert int(cell["idx_lw"]) == 0
            assert abs(float(cell["ratio"]) - 16.0) < 0.05

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_export_thr030_seed2_dense_accept_ov0_pin() -> None:
    # EXPORT (A6-T68 formal): thr=0.30 densified seed~2 accept mechanism for
    # A3 SI. Pins that densified seed~2 first-step Jaccard is ov0≈0.340
    # (invariant across thr floors) and only the 0.30 floor accepts it
    # (run0=16, coarse-end idx0); at 0.35/0.40 the same ov0 rejects
    # (run0=1). Standard-grid seed~2 stays reject (ov0≈0.200) at all three
    # floors. Defaults stay off.
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert EXPERIMENTAL_DENSIFY_OVERLAP_RECOVER_THRESHOLD == 0.35

    rows: dict[tuple[float, bool], dict[str, float | int | None]] = {}
    print("\nA6-T68 thr0.30 seed2 dense ov0 pin")
    header = (
        f"{'thr':>5s} {'dense':5s} {'ov0':>8s} {'run0':>4s} "
        f"{'tsi':>4s} {'accept':>6s}"
    )
    print(header)
    print("-" * len(header))
    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=2,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    for thr in (0.30, 0.35, 0.40):
        for dense in (False, True):
            result = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=ScaleSearchConfig(
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
                        min_equilibrium_epochs=2, max_epochs=12
                    ),
                    seed=2,
                    halve_grid_steps=dense,
                    persistence=PersistenceConfig(
                        resolve_within_interval="none",
                        densify_overlap_recover="lower_threshold",
                        densify_overlap_recover_threshold=thr,
                    ),
                ),
            )
            assert result.persistence_result is not None
            pr = result.persistence_result
            ov0 = float(pr.match_overlaps[0])
            run0 = int(pr.run_lengths[0])
            tsi = pr.tau_star_index
            accept = tsi is not None
            rows[(thr, dense)] = {
                "ov0": ov0,
                "run0": run0,
                "tsi": tsi if tsi is not None else -1,
                "accept": int(accept),
            }
            print(
                f"{thr:5.2f} {str(dense):5s} {ov0:8.3f} {run0:4d} "
                f"{str(tsi):>4s} {str(accept):>6s}"
            )

    # ov0 invariant across thr (gate-only); densify raises ~0.20 → ~0.34.
    for thr in (0.30, 0.35, 0.40):
        assert abs(float(rows[(thr, False)]["ov0"]) - 0.200) < 0.02
        assert abs(float(rows[(thr, True)]["ov0"]) - 0.340) < 0.02
        assert rows[(thr, False)]["accept"] == 0
        assert int(rows[(thr, False)]["run0"]) == 1

    # Only thr=0.30 densified accepts (ov0≈0.34 ≥ 0.30).
    assert rows[(0.30, True)]["accept"] == 1
    assert int(rows[(0.30, True)]["tsi"]) == 0
    assert int(rows[(0.30, True)]["run0"]) == 16
    assert rows[(0.35, True)]["accept"] == 0
    assert rows[(0.40, True)]["accept"] == 0
    assert int(rows[(0.35, True)]["run0"]) == 1
    assert int(rows[(0.40, True)]["run0"]) == 1

    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_seed2_thr030_dense_lw_load_mechanism() -> None:
    # EXPERIMENT (A6-T70-followon): export why thr=0.30 densified seed~2 is
    # the first LW≠coarse divergence. Pins load_trace at idx0/idx1 under the
    # closest-to-unit rule: L(0)≈0.614 under-unit vs L(1)≈1.562 over-unit,
    # with |log L(1)| < |log L(0)| so idx1 wins; both clear screen_min=0.5.
    # Defaults stay off; do not flip acceptance.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=2,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    result = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=ScaleSearchConfig(
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
                min_equilibrium_epochs=2, max_epochs=12
            ),
            seed=2,
            halve_grid_steps=True,
            persistence=PersistenceConfig(
                resolve_within_interval="none",
                densify_overlap_recover="lower_threshold",
                densify_overlap_recover_threshold=0.30,
            ),
        ),
    )
    assert result.persistence_result is not None
    pr = result.persistence_result
    assert pr.tau_star_index == 0
    assert int(pr.run_lengths[0]) == 16
    assert abs(float(pr.match_overlaps[0]) - 0.340) < 0.02

    load = np.asarray(result.load_trace, dtype=float)
    L0 = float(load[0])
    L1 = float(load[1])
    print("\nA6-T70 seed2 thr0.30 dense LW load mechanism")
    print(f"  L0={L0:.6f} L1={L1:.6f} screen={_WITHIN_INTERVAL_LOAD_SCREEN_MIN}")
    assert L0 >= _WITHIN_INTERVAL_LOAD_SCREEN_MIN
    assert L1 >= _WITHIN_INTERVAL_LOAD_SCREEN_MIN
    assert abs(L0 - 0.614) < 0.02
    assert abs(L1 - 1.562) < 0.05
    # Closest-to-unit: |log L1| < |log L0| ⇒ idx1 preferred over coarse-end.
    assert abs(float(np.log(L1))) < abs(float(np.log(L0)))
    assert _load_weighted_index(0, 15, load) == 1
    idx_lw = _resolve_persistence_tau_index(
        pr,
        load,
        list(result.stabilized_flags),
        PersistenceConfig(resolve_within_interval="load_weighted_interval"),
    )
    idx_none = _resolve_persistence_tau_index(
        pr,
        load,
        list(result.stabilized_flags),
        PersistenceConfig(resolve_within_interval="none"),
    )
    assert idx_none == 0
    assert idx_lw == 1
    assert abs(float(result.tau_grid[idx_lw]) / float(gt.expected_tau) - 12.126) < 0.05

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_seed1_densify_lw_stays_coarse_across_thr() -> None:
    # EXPERIMENT (A6-T71-followon): densify × load_weighted on recover-created
    # seed~1 across thr ∈ {0.30, 0.35, 0.40}. Contrast with thr0.30 densified
    # seed~2 LW≠coarse: seed~1 densified L(0)≈0.650 vs L(1)≈1.635 still favors
    # coarse-end (|log L0| < |log L1|). Accept map matches T64 for seed~1
    # (accept std+dense at 0.30/0.35; std-only at 0.40). Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False

    by: dict[tuple[float, bool], dict[str, float | int | None]] = {}
    print("\nA6-T71 seed1 densify × LW across thr")
    header = (
        f"{'thr':>5s} {'dense':5s} {'acc':>3s} {'ov0':>7s} {'run0':>4s} "
        f"{'LW':>3s} {'L0':>8s} {'L1':>8s}"
    )
    print(header)
    print("-" * len(header))
    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=1,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    for thr in (0.30, 0.35, 0.40):
        for dense in (False, True):
            result = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=ScaleSearchConfig(
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
                        min_equilibrium_epochs=2, max_epochs=12
                    ),
                    seed=1,
                    halve_grid_steps=dense,
                    persistence=PersistenceConfig(
                        resolve_within_interval="none",
                        densify_overlap_recover="lower_threshold",
                        densify_overlap_recover_threshold=thr,
                    ),
                ),
            )
            assert result.persistence_result is not None
            pr = result.persistence_result
            accept = pr.tau_star_index is not None
            ov0 = float(pr.match_overlaps[0])
            run0 = int(pr.run_lengths[0])
            L0 = float(result.load_trace[0])
            L1 = float(result.load_trace[1])
            if accept:
                idx_lw = _resolve_persistence_tau_index(
                    pr,
                    result.load_trace,
                    list(result.stabilized_flags),
                    PersistenceConfig(
                        resolve_within_interval="load_weighted_interval"
                    ),
                )
            else:
                idx_lw = None
            by[(thr, dense)] = {
                "accept": int(accept),
                "ov0": ov0,
                "run0": run0,
                "idx_lw": idx_lw if idx_lw is not None else -1,
                "L0": L0,
                "L1": L1,
            }
            print(
                f"{thr:5.2f} {str(dense):5s} {int(accept):3d} {ov0:7.3f} "
                f"{run0:4d} {str(idx_lw):>3s} {L0:8.4f} {L1:8.4f}"
            )

    # T64 seed~1 accept map under recover-thr.
    assert by[(0.30, False)]["accept"] == 1
    assert by[(0.30, True)]["accept"] == 1
    assert by[(0.35, False)]["accept"] == 1
    assert by[(0.35, True)]["accept"] == 1
    assert by[(0.40, False)]["accept"] == 1
    assert by[(0.40, True)]["accept"] == 0

    # ov0 invariant across thr; densify drops ~0.471 → ~0.364.
    for thr in (0.30, 0.35, 0.40):
        assert abs(float(by[(thr, False)]["ov0"]) - 0.471) < 0.02
        assert abs(float(by[(thr, True)]["ov0"]) - 0.364) < 0.02

    # Densified loads favor coarse-end (contrast seed~2 thr0.30).
    for thr in (0.30, 0.35):
        L0 = float(by[(thr, True)]["L0"])
        L1 = float(by[(thr, True)]["L1"])
        assert abs(L0 - 0.650) < 0.02
        assert abs(L1 - 1.635) < 0.05
        assert abs(float(np.log(L0))) < abs(float(np.log(L1)))
        assert int(by[(thr, True)]["idx_lw"]) == 0
        assert int(by[(thr, True)]["run0"]) == 16

    # Standard-grid accepts also stay coarse-end under LW.
    for thr in (0.30, 0.35, 0.40):
        assert int(by[(thr, False)]["idx_lw"]) == 0
        assert int(by[(thr, False)]["run0"]) == 8

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_seed2_thr030_dense_fractional_vs_lw() -> None:
    # EXPERIMENT (A6-T73-followon): thr=0.30 densified seed~2 mid / two-thirds /
    # three-quarter / fine-end vs load_weighted. Pins that the LW≠coarse
    # divergence (idx1 ~12.1×) is a *one-step* closest-to-unit nudge — not a
    # fractional-style within-block refinement (mid~2.30× / 2/3~1.00× /
    # 3/4~0.76× / fine~0.25×). Defaults stay off; do not flip acceptance.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False

    modes = (
        "none",
        "mid_interval",
        "two_thirds_interval",
        "three_quarter_interval",
        "load_weighted_interval",
        "fine_end_of_block",
    )
    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=2,
    )
    gt = dataset.ground_truth
    assert gt.expected_tau is not None
    tau_lo, tau_hi = gt.tau_grid_hint
    result = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=ScaleSearchConfig(
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
                min_equilibrium_epochs=2, max_epochs=12
            ),
            seed=2,
            halve_grid_steps=True,
            persistence=PersistenceConfig(
                resolve_within_interval="none",
                densify_overlap_recover="lower_threshold",
                densify_overlap_recover_threshold=0.30,
            ),
        ),
    )
    assert result.persistence_result is not None
    pr = result.persistence_result
    assert pr.tau_star_index == 0
    assert int(pr.run_lengths[0]) == 16
    assert abs(float(pr.match_overlaps[0]) - 0.340) < 0.02

    rows: dict[str, dict[str, float | int]] = {}
    print("\nA6-T73 seed2 thr0.30 dense fractional vs LW")
    header = f"{'mode':24s} {'idx':>3s} {'tau*/E':>8s}"
    print(header)
    print("-" * len(header))
    for mode in modes:
        idx = _resolve_persistence_tau_index(
            pr,
            result.load_trace,
            list(result.stabilized_flags),
            PersistenceConfig(resolve_within_interval=mode),  # type: ignore[arg-type]
        )
        ratio = float(result.tau_grid[idx]) / float(gt.expected_tau)
        rows[mode] = {"idx": int(idx), "ratio": ratio}
        print(f"{mode:24s} {idx:3d} {ratio:8.3f}")

    # Coarse-end vs LW one-step nudge (T70 mechanism).
    assert int(rows["none"]["idx"]) == 0
    assert abs(float(rows["none"]["ratio"]) - 16.0) < 0.05
    assert int(rows["load_weighted_interval"]["idx"]) == 1
    assert abs(float(rows["load_weighted_interval"]["ratio"]) - 12.126) < 0.05

    # Fractional landings match densify hierarchy (same indices as seed0/3).
    assert int(rows["mid_interval"]["idx"]) == 7
    assert abs(float(rows["mid_interval"]["ratio"]) - 2.297) < 0.05
    assert int(rows["two_thirds_interval"]["idx"]) == 10
    assert abs(float(rows["two_thirds_interval"]["ratio"]) - 1.0) < 0.05
    assert int(rows["three_quarter_interval"]["idx"]) == 11
    assert abs(float(rows["three_quarter_interval"]["ratio"]) - 0.758) < 0.05
    assert int(rows["fine_end_of_block"]["idx"]) == 15
    assert abs(float(rows["fine_end_of_block"]["ratio"]) - 0.25) < 0.05

    # LW is strictly coarser than every fractional landing.
    assert int(rows["load_weighted_interval"]["idx"]) < int(rows["mid_interval"]["idx"])
    assert int(rows["load_weighted_interval"]["idx"]) < int(
        rows["two_thirds_interval"]["idx"]
    )
    assert float(rows["load_weighted_interval"]["ratio"]) > float(
        rows["mid_interval"]["ratio"]
    )

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_thr030_dense_accept_load_vector_export() -> None:
    # EXPORT (A6-T74-followon): load vectors for all densified accepts under
    # thr=0.30 recover floor (seeds 0–4 all accept on this budget). Pins that
    # only seed~2 has |log L(1)| < |log L(0)| (LW→idx1); seeds 0/1/3/4 keep
    # |log L0| < |log L1| so LW≡coarse. Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    by: dict[int, dict[str, float | int]] = {}
    print("\nA6-T74 thr0.30 densified accept load vectors")
    header = (
        f"{'seed':>4s} {'ov0':>7s} {'run0':>4s} {'L0':>8s} {'L1':>8s} "
        f"{'|logL0|':>8s} {'|logL1|':>8s} {'LW':>3s}"
    )
    print(header)
    print("-" * len(header))
    for seed in range(5):
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        tau_lo, tau_hi = gt.tau_grid_hint
        result = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=ScaleSearchConfig(
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=seed,
                halve_grid_steps=True,
                persistence=PersistenceConfig(
                    resolve_within_interval="none",
                    densify_overlap_recover="lower_threshold",
                    densify_overlap_recover_threshold=0.30,
                ),
            ),
        )
        assert result.persistence_result is not None
        pr = result.persistence_result
        assert pr.tau_star_index == 0
        assert int(pr.run_lengths[0]) == 16
        load = np.asarray(result.load_trace, dtype=float)
        L0 = float(load[0])
        L1 = float(load[1])
        assert L0 >= _WITHIN_INTERVAL_LOAD_SCREEN_MIN
        assert L1 >= _WITHIN_INTERVAL_LOAD_SCREEN_MIN
        abs_log0 = abs(float(np.log(L0)))
        abs_log1 = abs(float(np.log(L1)))
        idx_lw = _resolve_persistence_tau_index(
            pr,
            load,
            list(result.stabilized_flags),
            PersistenceConfig(resolve_within_interval="load_weighted_interval"),
        )
        by[seed] = {
            "ov0": float(pr.match_overlaps[0]),
            "run0": int(pr.run_lengths[0]),
            "L0": L0,
            "L1": L1,
            "abs_log0": abs_log0,
            "abs_log1": abs_log1,
            "idx_lw": int(idx_lw),
        }
        print(
            f"{seed:4d} {float(pr.match_overlaps[0]):7.3f} "
            f"{int(pr.run_lengths[0]):4d} {L0:8.4f} {L1:8.4f} "
            f"{abs_log0:8.4f} {abs_log1:8.4f} {idx_lw:3d}"
        )

    # Published densified ov0 pins at thr0.30 (T64/T68 accept map).
    assert abs(float(by[0]["ov0"]) - 0.504) < 0.02
    assert abs(float(by[1]["ov0"]) - 0.364) < 0.02
    assert abs(float(by[2]["ov0"]) - 0.340) < 0.02
    assert abs(float(by[3]["ov0"]) - 0.503) < 0.02
    assert abs(float(by[4]["ov0"]) - 0.390) < 0.02

    # Load-vector pins: only seed2 flips |log| order → LW idx1.
    assert abs(float(by[0]["L0"]) - 0.732) < 0.02
    assert abs(float(by[0]["L1"]) - 1.764) < 0.05
    assert float(by[0]["abs_log0"]) < float(by[0]["abs_log1"])
    assert int(by[0]["idx_lw"]) == 0

    assert abs(float(by[1]["L0"]) - 0.650) < 0.02
    assert abs(float(by[1]["L1"]) - 1.635) < 0.05
    assert float(by[1]["abs_log0"]) < float(by[1]["abs_log1"])
    assert int(by[1]["idx_lw"]) == 0

    assert abs(float(by[2]["L0"]) - 0.614) < 0.02
    assert abs(float(by[2]["L1"]) - 1.562) < 0.05
    assert float(by[2]["abs_log1"]) < float(by[2]["abs_log0"])
    assert int(by[2]["idx_lw"]) == 1

    assert abs(float(by[3]["L0"]) - 0.722) < 0.02
    assert abs(float(by[3]["L1"]) - 1.785) < 0.05
    assert float(by[3]["abs_log0"]) < float(by[3]["abs_log1"])
    assert int(by[3]["idx_lw"]) == 0

    assert abs(float(by[4]["L0"]) - 0.692) < 0.02
    assert abs(float(by[4]["L1"]) - 1.903) < 0.05
    assert float(by[4]["abs_log0"]) < float(by[4]["abs_log1"])
    assert int(by[4]["idx_lw"]) == 0

    # Singleton LW≠coarse exception under thr0.30 densify.
    for seed in (0, 1, 3, 4):
        assert int(by[seed]["idx_lw"]) == 0
    assert int(by[2]["idx_lw"]) == 1

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_seed2_thr030_dense_phi_lw_vs_coarse_and_load_screened() -> None:
    # EXPERIMENT (A6-T76-followon): on thr=0.30 densified seed~2 (the sole
    # LW≠coarse cell), export Phi_C at coarse-end idx0 vs LW idx1, and contrast
    # load-screened mid / two-thirds / three-quarter vs their raw landings.
    # Expect: Phi *rises* one step finer with LW (not monotone along the block,
    # so LW is not a Phi-peak or Phi-descent rule); load-screened ≡ raw because
    # loads at fractional indices clear the ≪1 screen (0.5). Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=2,
    )
    gt = dataset.ground_truth
    assert gt.expected_tau is not None
    tau_lo, tau_hi = gt.tau_grid_hint
    result = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=ScaleSearchConfig(
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
                min_equilibrium_epochs=2, max_epochs=12
            ),
            seed=2,
            halve_grid_steps=True,
            persistence=PersistenceConfig(
                resolve_within_interval="none",
                densify_overlap_recover="lower_threshold",
                densify_overlap_recover_threshold=0.30,
            ),
        ),
    )
    assert result.persistence_result is not None
    pr = result.persistence_result
    assert pr.tau_star_index == 0
    assert int(pr.run_lengths[0]) == 16
    load = np.asarray(result.load_trace, dtype=float)
    phi = np.asarray(result.phi_trace, dtype=float)

    modes = (
        "none",
        "load_weighted_interval",
        "mid_interval",
        "mid_interval_load_screened",
        "two_thirds_interval",
        "two_thirds_load_screened",
        "three_quarter_interval",
        "three_quarter_load_screened",
    )
    rows: dict[str, dict[str, float | int]] = {}
    print("\nA6-T76 seed2 thr0.30 dense Phi LW vs coarse + load-screened")
    header = (
        f"{'mode':28s} {'idx':>3s} {'tau*/E':>8s} {'Phi*':>10s} {'load*':>8s}"
    )
    print(header)
    print("-" * len(header))
    for mode in modes:
        idx = _resolve_persistence_tau_index(
            pr,
            load,
            list(result.stabilized_flags),
            PersistenceConfig(resolve_within_interval=mode),  # type: ignore[arg-type]
        )
        ratio = float(result.tau_grid[idx]) / float(gt.expected_tau)
        phi_star = float(phi[idx])
        load_star = float(load[idx])
        rows[mode] = {
            "idx": int(idx),
            "ratio": ratio,
            "phi": phi_star,
            "load": load_star,
        }
        print(
            f"{mode:28s} {idx:3d} {ratio:8.3f} {phi_star:10.4f} {load_star:8.4f}"
        )

    # Coarse vs LW one-step: indices / ratios match T70/T73 pins.
    assert int(rows["none"]["idx"]) == 0
    assert abs(float(rows["none"]["ratio"]) - 16.0) < 0.05
    assert int(rows["load_weighted_interval"]["idx"]) == 1
    assert abs(float(rows["load_weighted_interval"]["ratio"]) - 12.126) < 0.05

    # Phi at LW idx1 is finite and *above* Phi at coarse idx0 — Phi is not
    # monotone along the accepted block, so the closest-to-unit load nudge is
    # not a Phi-descent (nor a global in-block Phi-peak: mid Phi sits between).
    phi0 = float(rows["none"]["phi"])
    phi1 = float(rows["load_weighted_interval"]["phi"])
    phi_mid = float(rows["mid_interval"]["phi"])
    assert np.isfinite(phi0) and np.isfinite(phi1) and np.isfinite(phi_mid)
    assert phi1 > phi0
    assert phi1 > phi_mid
    assert abs(phi0 - float(phi[0])) < 1e-12
    assert abs(phi1 - float(phi[1])) < 1e-12
    # Order-of-magnitude pins (diagnostic; response scale is fixture-local).
    assert 1e7 < phi0 < 1e8
    assert 1e8 < phi1 < 2e9
    assert 1e8 < phi_mid < 5e8

    # Fractional raw landings match densify hierarchy (T73).
    assert int(rows["mid_interval"]["idx"]) == 7
    assert int(rows["two_thirds_interval"]["idx"]) == 10
    assert int(rows["three_quarter_interval"]["idx"]) == 11

    # Load-screened ≡ raw: fractional loads clear ≪1 screen.
    for raw, screened in (
        ("mid_interval", "mid_interval_load_screened"),
        ("two_thirds_interval", "two_thirds_load_screened"),
        ("three_quarter_interval", "three_quarter_load_screened"),
    ):
        assert int(rows[screened]["idx"]) == int(rows[raw]["idx"])
        assert float(rows[screened]["phi"]) == float(rows[raw]["phi"])
        assert float(rows[raw]["load"]) >= _WITHIN_INTERVAL_LOAD_SCREEN_MIN
        assert float(rows[screened]["load"]) >= _WITHIN_INTERVAL_LOAD_SCREEN_MIN

    # Coarse/LW loads also clear the screen (T70 mechanism precondition).
    assert float(rows["none"]["load"]) >= _WITHIN_INTERVAL_LOAD_SCREEN_MIN
    assert float(rows["load_weighted_interval"]["load"]) >= _WITHIN_INTERVAL_LOAD_SCREEN_MIN
    assert abs(float(rows["none"]["load"]) - 0.614) < 0.02
    assert abs(float(rows["load_weighted_interval"]["load"]) - 1.562) < 0.05

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_seed2_thr030_dense_phi_peak_vs_lw_and_lc_hybrid() -> None:
    # EXPERIMENT (A6-T77-followon): thr=0.30 densified seed~2 full-block Phi
    # export + load_crossover hybrid vs LW. Pins that in-block argmax Phi lands
    # at the same unstabilized idx1 that LW picks (so LW≡Phi-peak on this
    # singleton cell by coincidence of definitions, not because LW uses Phi),
    # while resolve_within_interval="load_crossover" stays at coarse-end idx0
    # because the stabilization filter skips idx1 and the eligible straddle
    # 0↔2 returns the nearer-to-unit endpoint L(0). Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=2,
    )
    gt = dataset.ground_truth
    assert gt.expected_tau is not None
    tau_lo, tau_hi = gt.tau_grid_hint
    result = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=ScaleSearchConfig(
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
                min_equilibrium_epochs=2, max_epochs=12
            ),
            seed=2,
            halve_grid_steps=True,
            persistence=PersistenceConfig(
                resolve_within_interval="none",
                densify_overlap_recover="lower_threshold",
                densify_overlap_recover_threshold=0.30,
            ),
        ),
    )
    assert result.persistence_result is not None
    pr = result.persistence_result
    assert pr.tau_star_index == 0
    assert int(pr.run_lengths[0]) == 16
    load = np.asarray(result.load_trace, dtype=float)
    phi = np.asarray(result.phi_trace, dtype=float)
    stab = list(result.stabilized_flags)
    i_lo = 0
    i_hi = 15
    expected_tau = float(gt.expected_tau)

    print("\nA6-T77 seed2 thr0.30 dense full-block Phi + LC hybrid vs LW")
    header = (
        f"{'idx':>3s} {'tau*/E':>8s} {'Phi':>12s} {'load':>8s} {'stab':>4s}"
    )
    print(header)
    print("-" * len(header))
    for idx in range(i_lo, i_hi + 1):
        ratio = float(result.tau_grid[idx]) / expected_tau
        print(
            f"{idx:3d} {ratio:8.3f} {float(phi[idx]):12.4g} "
            f"{float(load[idx]):8.4f} {str(bool(stab[idx])):4s}"
        )

    finite = [
        idx for idx in range(i_lo, i_hi + 1) if np.isfinite(float(phi[idx]))
    ]
    assert len(finite) == 16
    phi_peak_idx = max(finite, key=lambda i: float(phi[i]))
    # In-block Phi jumps at idx1 then decreases toward fine-end.
    assert phi_peak_idx == 1
    assert float(phi[1]) > float(phi[0])
    for idx in range(1, i_hi):
        assert float(phi[idx]) > float(phi[idx + 1])

    modes = ("none", "load_weighted_interval", "load_crossover")
    rows: dict[str, dict[str, float | int | bool]] = {}
    print(
        f"\n{'mode':24s} {'idx':>3s} {'tau*/E':>8s} {'Phi*':>12s} {'load*':>8s}"
    )
    print("-" * 60)
    for mode in modes:
        idx = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval=mode),  # type: ignore[arg-type]
        )
        rows[mode] = {
            "idx": int(idx),
            "ratio": float(result.tau_grid[idx]) / expected_tau,
            "phi": float(phi[idx]),
            "load": float(load[idx]),
            "stab": bool(stab[idx]),
        }
        print(
            f"{mode:24s} {idx:3d} {float(rows[mode]['ratio']):8.3f} "
            f"{float(rows[mode]['phi']):12.4g} {float(rows[mode]['load']):8.4f}"
        )

    # Coarse / LW pins (T70 / T73 / T76).
    assert int(rows["none"]["idx"]) == 0
    assert abs(float(rows["none"]["ratio"]) - 16.0) < 0.05
    assert int(rows["load_weighted_interval"]["idx"]) == 1
    assert abs(float(rows["load_weighted_interval"]["ratio"]) - 12.126) < 0.05
    assert abs(float(rows["none"]["load"]) - 0.614) < 0.02
    assert abs(float(rows["load_weighted_interval"]["load"]) - 1.562) < 0.05

    # LW ≡ in-block Phi peak on this cell (both land at unstabilized idx1).
    assert int(rows["load_weighted_interval"]["idx"]) == phi_peak_idx
    assert bool(rows["load_weighted_interval"]["stab"]) is False
    assert bool(stab[1]) is False
    assert bool(stab[0]) is True
    assert bool(stab[2]) is True
    assert 1e8 < float(rows["load_weighted_interval"]["phi"]) < 2e9
    assert abs(
        float(rows["load_weighted_interval"]["phi"]) - float(phi[1])
    ) < 1e-6

    # load_crossover hybrid ≡ coarse: idx1 is unstabilized, so the eligible
    # straddle is 0↔2 and nearer-to-unit is L(0)≈0.614.
    assert int(rows["load_crossover"]["idx"]) == 0
    assert int(rows["load_crossover"]["idx"]) == int(rows["none"]["idx"])
    assert abs(float(rows["load_crossover"]["ratio"]) - 16.0) < 0.05
    assert bool(rows["load_crossover"]["stab"]) is True
    assert float(load[0]) <= 1.0 < float(load[2])
    assert abs(float(load[0]) - 1.0) <= abs(float(load[2]) - 1.0)

    # Contrast: LW ≠ LC hybrid on this singleton (stabilization filter).
    assert int(rows["load_weighted_interval"]["idx"]) != int(
        rows["load_crossover"]["idx"]
    )

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_thr030_dense_multiseed_phi_peak_vs_lw() -> None:
    # EXPERIMENT (A6-T78-followon): thr=0.30 densified seeds0..4 — does
    # in-block argmax Phi land at LW for accepts other than seed~2?
    # Answer: NO. Phi peaks at idx1 on every densified accept under this
    # floor, but LW stays at coarse-end idx0 for seeds0/1/3/4 (T74 load
    # vectors); only seed~2 has LW≡Phi-peak (T77 coincidence). LC hybrid
    # stays coarse on all five. Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    by: dict[int, dict[str, float | int | bool]] = {}
    print("\nA6-T78 thr0.30 densified multi-seed Phi-peak vs LW")
    header = (
        f"{'seed':>4s} {'peak':>4s} {'LW':>3s} {'LC':>3s} "
        f"{'phi0':>10s} {'phi1':>10s} {'L0':>8s} {'L1':>8s} {'stab1':>5s}"
    )
    print(header)
    print("-" * len(header))
    for seed in range(5):
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        assert gt.expected_tau is not None
        tau_lo, tau_hi = gt.tau_grid_hint
        result = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=ScaleSearchConfig(
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=seed,
                halve_grid_steps=True,
                persistence=PersistenceConfig(
                    resolve_within_interval="none",
                    densify_overlap_recover="lower_threshold",
                    densify_overlap_recover_threshold=0.30,
                ),
            ),
        )
        assert result.persistence_result is not None
        pr = result.persistence_result
        assert pr.tau_star_index == 0
        assert int(pr.run_lengths[0]) == 16
        load = np.asarray(result.load_trace, dtype=float)
        phi = np.asarray(result.phi_trace, dtype=float)
        stab = list(result.stabilized_flags)
        i_lo = 0
        i_hi = 15
        finite = [
            idx for idx in range(i_lo, i_hi + 1) if np.isfinite(float(phi[idx]))
        ]
        assert len(finite) == 16
        phi_peak_idx = max(finite, key=lambda i: float(phi[i]))
        idx_lw = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval="load_weighted_interval"),
        )
        idx_lc = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval="load_crossover"),
        )
        by[seed] = {
            "peak": int(phi_peak_idx),
            "idx_lw": int(idx_lw),
            "idx_lc": int(idx_lc),
            "phi0": float(phi[0]),
            "phi1": float(phi[1]),
            "L0": float(load[0]),
            "L1": float(load[1]),
            "stab1": bool(stab[1]),
        }
        print(
            f"{seed:4d} {phi_peak_idx:4d} {idx_lw:3d} {idx_lc:3d} "
            f"{float(phi[0]):10.4g} {float(phi[1]):10.4g} "
            f"{float(load[0]):8.4f} {float(load[1]):8.4f} "
            f"{str(bool(stab[1])):5s}"
        )

    # In-block Phi peaks at idx1 on every densified thr0.30 accept.
    for seed in range(5):
        assert int(by[seed]["peak"]) == 1
        assert float(by[seed]["phi1"]) > float(by[seed]["phi0"])
        assert int(by[seed]["idx_lc"]) == 0

    # Singleton LW≡Phi-peak only on seed2 (T74/T77); others LW≡coarse ≠ peak.
    for seed in (0, 1, 3, 4):
        assert int(by[seed]["idx_lw"]) == 0
        assert int(by[seed]["idx_lw"]) != int(by[seed]["peak"])
        assert bool(by[seed]["stab1"]) is True
    assert int(by[2]["idx_lw"]) == 1
    assert int(by[2]["idx_lw"]) == int(by[2]["peak"])
    assert bool(by[2]["stab1"]) is False

    # Load-vector pins match T74 (mechanism for LW landings).
    assert abs(float(by[0]["L0"]) - 0.732) < 0.02
    assert abs(float(by[1]["L0"]) - 0.650) < 0.02
    assert abs(float(by[2]["L0"]) - 0.614) < 0.02
    assert abs(float(by[2]["L1"]) - 1.562) < 0.05
    assert abs(float(by[3]["L0"]) - 0.722) < 0.02
    assert abs(float(by[4]["L0"]) - 0.692) < 0.02

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_seed2_thr030_dense_phi_at_lc_eligible_idx2_vs_peak() -> None:
    # EXPERIMENT (A6-T79): thr=0.30 densified seed~2 — Phi at the LC-eligible
    # fine straddle endpoint (idx2; stab=True) vs in-block Phi peak (idx1;
    # unstabilized; LW landing). T77 showed LC hybrid stays at coarse idx0
    # because the stab filter skips idx1 and nearer-to-unit among {0,2} is
    # L(0). This pins that the LC fine endpoint is still *Phi-near-peak*
    # (Phi2/Phi1≈0.78, Phi2≫Phi0) — so stab-skipping the peak leaves an
    # eligible fine candidate with high Phi, yet the load straddle rule
    # still picks coarse. Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=2,
    )
    gt = dataset.ground_truth
    assert gt.expected_tau is not None
    tau_lo, tau_hi = gt.tau_grid_hint
    result = run_scale_search(
        dataset.points,
        dim=gt.ambient_dim,
        config=ScaleSearchConfig(
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
                min_equilibrium_epochs=2, max_epochs=12
            ),
            seed=2,
            halve_grid_steps=True,
            persistence=PersistenceConfig(
                resolve_within_interval="none",
                densify_overlap_recover="lower_threshold",
                densify_overlap_recover_threshold=0.30,
            ),
        ),
    )
    assert result.persistence_result is not None
    pr = result.persistence_result
    assert pr.tau_star_index == 0
    assert int(pr.run_lengths[0]) == 16
    load = np.asarray(result.load_trace, dtype=float)
    phi = np.asarray(result.phi_trace, dtype=float)
    stab = list(result.stabilized_flags)
    expected_tau = float(gt.expected_tau)

    i_lo = 0
    i_hi = 15
    finite = [
        idx for idx in range(i_lo, i_hi + 1) if np.isfinite(float(phi[idx]))
    ]
    assert len(finite) == 16
    phi_peak_idx = max(finite, key=lambda i: float(phi[i]))
    assert phi_peak_idx == 1

    idx_lw = _resolve_persistence_tau_index(
        pr,
        load,
        stab,
        PersistenceConfig(resolve_within_interval="load_weighted_interval"),
    )
    idx_lc = _resolve_persistence_tau_index(
        pr,
        load,
        stab,
        PersistenceConfig(resolve_within_interval="load_crossover"),
    )

    print("\nA6-T79 seed2 thr0.30 dense Phi at LC-eligible idx2 vs peak")
    header = (
        f"{'idx':>3s} {'role':12s} {'tau*/E':>8s} {'Phi':>12s} "
        f"{'load':>8s} {'stab':>4s}"
    )
    print(header)
    print("-" * len(header))
    roles = {0: "coarse/LC", 1: "peak/LW", 2: "LC-elig"}
    for idx in (0, 1, 2):
        ratio = float(result.tau_grid[idx]) / expected_tau
        print(
            f"{idx:3d} {roles[idx]:12s} {ratio:8.3f} {float(phi[idx]):12.4g} "
            f"{float(load[idx]):8.4f} {str(bool(stab[idx])):4s}"
        )
    print(
        f"modes: LW={idx_lw} LC={idx_lc} peak={phi_peak_idx} "
        f"phi2/phi1={float(phi[2])/float(phi[1]):.4f} "
        f"phi2/phi0={float(phi[2])/float(phi[0]):.4f}"
    )

    # Stabilization topology that drives LC straddle 0↔2 (T77).
    assert bool(stab[0]) is True
    assert bool(stab[1]) is False
    assert bool(stab[2]) is True

    # Load pins: L0 under 1, L1/L2 over 1; nearer-to-unit among {0,2} is L0.
    assert abs(float(load[0]) - 0.614) < 0.02
    assert abs(float(load[1]) - 1.562) < 0.05
    assert abs(float(load[2]) - 2.059) < 0.05
    assert float(load[0]) <= 1.0 < float(load[2])
    assert abs(float(load[0]) - 1.0) < abs(float(load[2]) - 1.0)

    # Mode landings: LW≡peak at unstabilized idx1; LC≡coarse idx0.
    assert int(idx_lw) == 1
    assert int(idx_lw) == phi_peak_idx
    assert int(idx_lc) == 0
    assert abs(float(result.tau_grid[idx_lw]) / expected_tau - 12.126) < 0.05
    assert abs(float(result.tau_grid[idx_lc]) / expected_tau - 16.0) < 0.05

    # Phi at LC-eligible fine endpoint idx2 remains near the peak
    # (post-peak decay is shallow at +1 step) and far above coarse Phi —
    # so LC's load-straddle fine candidate is Phi-near-peak, yet LC still
    # lands coarse because |L0-1| < |L2-1| (not a Phi rule).
    phi0 = float(phi[0])
    phi1 = float(phi[1])
    phi2 = float(phi[2])
    assert np.isfinite(phi0) and np.isfinite(phi1) and np.isfinite(phi2)
    assert phi1 > phi2 > phi0
    assert 1e8 < phi1 < 2e9
    assert 1e8 < phi2 < 2e9
    assert 1e7 < phi0 < 1e8
    assert 0.70 < phi2 / phi1 < 0.85
    assert phi2 / phi0 > 20.0

    # Ratio pins for the three indices (densify grid).
    assert abs(float(result.tau_grid[0]) / expected_tau - 16.0) < 0.05
    assert abs(float(result.tau_grid[1]) / expected_tau - 12.126) < 0.05
    assert abs(float(result.tau_grid[2]) / expected_tau - 9.190) < 0.05

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_thr030_dense_multiseed_phi_first_stab_after_peak_vs_peak() -> None:
    # EXPERIMENT (A6-T81): thr=0.30 densified seeds0..4 — Phi at the first
    # stabilized index strictly after the in-block Phi peak, vs Phi at the
    # peak. T78 pinned peak=idx1 always; T79 pinned seed2's first stab-after
    # (idx2) is Phi-near-peak (~0.78) yet LC still rejects it. This extends
    # that comparison across all five densified accepts. Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    by: dict[int, dict[str, float | int | bool]] = {}
    print("\nA6-T81 thr0.30 densified multi-seed Phi first-stab-after-peak vs peak")
    header = (
        f"{'seed':>4s} {'peak':>4s} {'fsa':>3s} {'stabP':>5s} "
        f"{'phiP':>10s} {'phiF':>10s} {'ratio':>7s} {'LW':>3s} {'LC':>3s}"
    )
    print(header)
    print("-" * len(header))
    for seed in range(5):
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        assert gt.expected_tau is not None
        tau_lo, tau_hi = gt.tau_grid_hint
        result = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=ScaleSearchConfig(
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=seed,
                halve_grid_steps=True,
                persistence=PersistenceConfig(
                    resolve_within_interval="none",
                    densify_overlap_recover="lower_threshold",
                    densify_overlap_recover_threshold=0.30,
                ),
            ),
        )
        assert result.persistence_result is not None
        pr = result.persistence_result
        assert pr.tau_star_index == 0
        assert int(pr.run_lengths[0]) == 16
        load = np.asarray(result.load_trace, dtype=float)
        phi = np.asarray(result.phi_trace, dtype=float)
        stab = list(result.stabilized_flags)
        i_lo = 0
        i_hi = 15
        finite = [
            idx for idx in range(i_lo, i_hi + 1) if np.isfinite(float(phi[idx]))
        ]
        assert len(finite) == 16
        phi_peak_idx = max(finite, key=lambda i: float(phi[i]))
        fsa_idx = next(
            (idx for idx in range(phi_peak_idx + 1, i_hi + 1) if stab[idx]),
            None,
        )
        assert fsa_idx is not None
        idx_lw = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval="load_weighted_interval"),
        )
        idx_lc = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval="load_crossover"),
        )
        phi_p = float(phi[phi_peak_idx])
        phi_f = float(phi[fsa_idx])
        ratio = phi_f / phi_p
        by[seed] = {
            "peak": int(phi_peak_idx),
            "fsa": int(fsa_idx),
            "stab_peak": bool(stab[phi_peak_idx]),
            "phi_p": phi_p,
            "phi_f": phi_f,
            "ratio": float(ratio),
            "idx_lw": int(idx_lw),
            "idx_lc": int(idx_lc),
        }
        print(
            f"{seed:4d} {phi_peak_idx:4d} {fsa_idx:3d} "
            f"{str(bool(stab[phi_peak_idx])):5s} "
            f"{phi_p:10.4g} {phi_f:10.4g} {ratio:7.4f} "
            f"{idx_lw:3d} {idx_lc:3d}"
        )

    # Peak / first-stab-after topology is uniform: peak=idx1, fsa=idx2.
    for seed in range(5):
        assert int(by[seed]["peak"]) == 1
        assert int(by[seed]["fsa"]) == 2
        assert int(by[seed]["idx_lc"]) == 0
        assert float(by[seed]["phi_p"]) > float(by[seed]["phi_f"]) > 0.0

    # Seeds 0/1/3/4: peak itself is stabilized, so fsa is merely the next
    # grid step; Phi decay is shallow (ratio ≳ 0.84). Seed2: peak is
    # unstabilized (T77/T79), fsa=idx2 is the first LC-eligible fine
    # endpoint, still Phi-near-peak (~0.78) but deeper decay.
    for seed in (0, 1, 3, 4):
        assert bool(by[seed]["stab_peak"]) is True
        assert int(by[seed]["idx_lw"]) == 0
        assert 0.84 < float(by[seed]["ratio"]) < 0.98
    assert bool(by[2]["stab_peak"]) is False
    assert int(by[2]["idx_lw"]) == 1
    assert 0.70 < float(by[2]["ratio"]) < 0.85

    # Seed-specific ratio pins (shallow post-peak decay under densify).
    assert abs(float(by[0]["ratio"]) - 0.949) < 0.02
    assert abs(float(by[1]["ratio"]) - 0.847) < 0.02
    assert abs(float(by[2]["ratio"]) - 0.779) < 0.02
    assert abs(float(by[3]["ratio"]) - 0.925) < 0.02
    assert abs(float(by[4]["ratio"]) - 0.895) < 0.02

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_thr030_dense_multiseed_phi_stab_only_argmax_vs_lw() -> None:
    # EXPERIMENT (A6-T82): thr=0.30 densified seeds0..4 — Phi-argmax among
    # *stabilized-only* in-block indices vs LW landing. Contrasts with T78
    # (unfiltered Phi-peak ≡ LW only on seed2) and T81 (first-stab-after).
    # Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    by: dict[int, dict[str, float | int | bool]] = {}
    print("\nA6-T82 thr0.30 densified multi-seed stab-only Phi-argmax vs LW")
    header = (
        f"{'seed':>4s} {'peak':>4s} {'sArg':>4s} {'LW':>3s} {'LC':>3s} "
        f"{'stabP':>5s} {'phiP':>10s} {'phiS':>10s} {'ratio':>7s}"
    )
    print(header)
    print("-" * len(header))
    for seed in range(5):
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        assert gt.expected_tau is not None
        tau_lo, tau_hi = gt.tau_grid_hint
        result = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=ScaleSearchConfig(
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=seed,
                halve_grid_steps=True,
                persistence=PersistenceConfig(
                    resolve_within_interval="none",
                    densify_overlap_recover="lower_threshold",
                    densify_overlap_recover_threshold=0.30,
                ),
            ),
        )
        assert result.persistence_result is not None
        pr = result.persistence_result
        assert pr.tau_star_index == 0
        assert int(pr.run_lengths[0]) == 16
        load = np.asarray(result.load_trace, dtype=float)
        phi = np.asarray(result.phi_trace, dtype=float)
        stab = list(result.stabilized_flags)
        i_lo = 0
        i_hi = 15
        finite = [
            idx for idx in range(i_lo, i_hi + 1) if np.isfinite(float(phi[idx]))
        ]
        assert len(finite) == 16
        phi_peak_idx = max(finite, key=lambda i: float(phi[i]))
        stab_only = [idx for idx in finite if stab[idx]]
        assert len(stab_only) >= 1
        stab_argmax = max(stab_only, key=lambda i: float(phi[i]))
        idx_lw = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval="load_weighted_interval"),
        )
        idx_lc = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval="load_crossover"),
        )
        phi_p = float(phi[phi_peak_idx])
        phi_s = float(phi[stab_argmax])
        by[seed] = {
            "peak": int(phi_peak_idx),
            "stab_argmax": int(stab_argmax),
            "stab_peak": bool(stab[phi_peak_idx]),
            "phi_p": phi_p,
            "phi_s": phi_s,
            "ratio": float(phi_s / phi_p),
            "idx_lw": int(idx_lw),
            "idx_lc": int(idx_lc),
        }
        print(
            f"{seed:4d} {phi_peak_idx:4d} {stab_argmax:4d} "
            f"{idx_lw:3d} {idx_lc:3d} {str(bool(stab[phi_peak_idx])):5s} "
            f"{phi_p:10.4g} {phi_s:10.4g} {phi_s / phi_p:7.4f}"
        )

    # Unfiltered peak remains idx1 on every densified thr0.30 accept (T78).
    for seed in range(5):
        assert int(by[seed]["peak"]) == 1
        assert int(by[seed]["idx_lc"]) == 0

    # Seeds 0/1/3/4: peak is stabilized ⇒ stab-only argmax ≡ peak (=idx1),
    # but LW stays coarse (=idx0) — so stab-argmax ≠ LW.
    for seed in (0, 1, 3, 4):
        assert bool(by[seed]["stab_peak"]) is True
        assert int(by[seed]["stab_argmax"]) == 1
        assert int(by[seed]["stab_argmax"]) == int(by[seed]["peak"])
        assert int(by[seed]["idx_lw"]) == 0
        assert int(by[seed]["stab_argmax"]) != int(by[seed]["idx_lw"])
        assert abs(float(by[seed]["ratio"]) - 1.0) < 1e-12

    # Seed2: peak unstabilized ⇒ stab-only argmax steps to idx2 (T81 fsa),
    # while LW lands at the unstabilized peak idx1 — stab-argmax ≠ LW again.
    assert bool(by[2]["stab_peak"]) is False
    assert int(by[2]["stab_argmax"]) == 2
    assert int(by[2]["idx_lw"]) == 1
    assert int(by[2]["stab_argmax"]) != int(by[2]["idx_lw"])
    assert int(by[2]["stab_argmax"]) != int(by[2]["peak"])
    assert 0.70 < float(by[2]["ratio"]) < 0.85
    assert abs(float(by[2]["ratio"]) - 0.779) < 0.02

    # Universal negative: stab-only Phi-argmax never equals LW on this fixture.
    for seed in range(5):
        assert int(by[seed]["stab_argmax"]) != int(by[seed]["idx_lw"])

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_thr030_dense_multiseed_phi_post_peak_decay_curve() -> None:
    # EXPERIMENT (A6-T84): thr=0.30 densified seeds0..4 — export Phi ratios
    # at peak+1..peak+4 (decay curve beyond T81's single first-stab-after
    # ratio). Also pin near-peak vs fine-end stab-skip topology. Defaults
    # stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    by: dict[int, dict[str, object]] = {}
    print("\nA6-T84 thr0.30 densified multi-seed Phi post-peak decay (+1..+4)")
    header = (
        f"{'seed':>4s} {'peak':>4s} {'r+1':>7s} {'r+2':>7s} {'r+3':>7s} "
        f"{'r+4':>7s} {'skip':>18s} {'LC':>3s} {'LW':>3s}"
    )
    print(header)
    print("-" * len(header))
    for seed in range(5):
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        assert gt.expected_tau is not None
        tau_lo, tau_hi = gt.tau_grid_hint
        result = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=ScaleSearchConfig(
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=seed,
                halve_grid_steps=True,
                persistence=PersistenceConfig(
                    resolve_within_interval="none",
                    densify_overlap_recover="lower_threshold",
                    densify_overlap_recover_threshold=0.30,
                ),
            ),
        )
        assert result.persistence_result is not None
        pr = result.persistence_result
        assert pr.tau_star_index == 0
        assert int(pr.run_lengths[0]) == 16
        load = np.asarray(result.load_trace, dtype=float)
        phi = np.asarray(result.phi_trace, dtype=float)
        stab = list(result.stabilized_flags)
        i_lo = 0
        i_hi = 15
        finite = [
            idx for idx in range(i_lo, i_hi + 1) if np.isfinite(float(phi[idx]))
        ]
        assert len(finite) == 16
        phi_peak_idx = max(finite, key=lambda i: float(phi[i]))
        phi_p = float(phi[phi_peak_idx])
        ratios = [
            float(phi[phi_peak_idx + off]) / phi_p for off in (1, 2, 3, 4)
        ]
        skipped = [idx for idx in range(i_lo, i_hi + 1) if not stab[idx]]
        idx_lw = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval="load_weighted_interval"),
        )
        idx_lc = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval="load_crossover"),
        )
        by[seed] = {
            "peak": int(phi_peak_idx),
            "ratios": ratios,
            "skip": skipped,
            "stab_peak": bool(stab[phi_peak_idx]),
            "idx_lw": int(idx_lw),
            "idx_lc": int(idx_lc),
        }
        print(
            f"{seed:4d} {phi_peak_idx:4d} "
            f"{ratios[0]:7.4f} {ratios[1]:7.4f} {ratios[2]:7.4f} {ratios[3]:7.4f} "
            f"{str(skipped):>18s} {idx_lc:3d} {idx_lw:3d}"
        )

    # Peak / LC topology matches T78/T81; decay is strictly monotonic.
    for seed in range(5):
        assert int(by[seed]["peak"]) == 1
        assert int(by[seed]["idx_lc"]) == 0
        ratios = list(by[seed]["ratios"])  # type: ignore[arg-type]
        assert len(ratios) == 4
        assert all(0.0 < float(r) < 1.0 for r in ratios)
        assert float(ratios[0]) > float(ratios[1]) > float(ratios[2]) > float(
            ratios[3]
        )

    # Seed-specific decay pins (T81 +1 ratios preserved; +2..+4 extend).
    expected = {
        0: (0.949, 0.832, 0.710, 0.478),
        1: (0.847, 0.701, 0.558, 0.440),
        2: (0.779, 0.616, 0.466, 0.333),
        3: (0.925, 0.732, 0.613, 0.506),
        4: (0.895, 0.757, 0.594, 0.464),
    }
    for seed, pins in expected.items():
        ratios = list(by[seed]["ratios"])  # type: ignore[arg-type]
        for got, want in zip(ratios, pins, strict=True):
            assert abs(float(got) - float(want)) < 0.02

    # Stab-skip topology: only seed2 skips near the peak (idx1); others are
    # either fully stabilized or fine-end-only skips that do not reshape the
    # coarse LC straddle.
    assert list(by[0]["skip"]) == []  # type: ignore[arg-type]
    assert list(by[1]["skip"]) == [14, 15]  # type: ignore[arg-type]
    assert list(by[2]["skip"]) == [1, 13, 14, 15]  # type: ignore[arg-type]
    assert list(by[3]["skip"]) == [15]  # type: ignore[arg-type]
    assert list(by[4]["skip"]) == []  # type: ignore[arg-type]
    assert bool(by[2]["stab_peak"]) is False
    for seed in (0, 1, 3, 4):
        assert bool(by[seed]["stab_peak"]) is True
        assert 1 not in list(by[seed]["skip"])  # type: ignore[arg-type]

    # LW landings unchanged vs T74/T78.
    for seed in (0, 1, 3, 4):
        assert int(by[seed]["idx_lw"]) == 0
    assert int(by[2]["idx_lw"]) == 1

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_densify_stab_skip_x_thr_lc_eligible_set() -> None:
    # EXPERIMENT (A6-T85): densify × recover-thr ∈ {0.30, 0.35, 0.40} on
    # seeds0..4 — does thr change the stab-skip set / LC-eligible fine
    # endpoint? Answer: thr only gates accept/reject (T64 map); whenever a
    # seed accepts, skip topology + fsa=idx2 + LC≡coarse are thr-invariant.
    # The only near-peak skip that widens LC straddle to 0↔2 is seed2@0.30.
    # Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    # T64 densified accept map under recover-thr floors.
    expect_accept = {
        (0, 0.30): True,
        (1, 0.30): True,
        (2, 0.30): True,
        (3, 0.30): True,
        (4, 0.30): True,
        (0, 0.35): True,
        (1, 0.35): True,
        (2, 0.35): False,
        (3, 0.35): True,
        (4, 0.35): True,
        (0, 0.40): True,
        (1, 0.40): False,
        (2, 0.40): False,
        (3, 0.40): True,
        (4, 0.40): False,
    }
    # Stab-skip sets when accepted (thr-invariant per seed).
    expect_skip = {
        0: [],
        1: [14, 15],
        2: [1, 13, 14, 15],
        3: [15],
        4: [],
    }

    by: dict[tuple[int, float], dict[str, object]] = {}
    print("\nA6-T85 densify stab-skip × thr floors vs LC-eligible set")
    header = (
        f"{'seed':>4s} {'thr':>5s} {'acc':>5s} {'run0':>4s} {'peak':>4s} "
        f"{'skip':>18s} {'fsa':>3s} {'LC':>3s} {'LW':>3s} {'r_fsa':>7s}"
    )
    print(header)
    print("-" * len(header))
    for thr in (0.30, 0.35, 0.40):
        for seed in range(5):
            dataset = make_hierarchical_gaussian(
                children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
            )
            gt = dataset.ground_truth
            assert gt.expected_tau is not None
            tau_lo, tau_hi = gt.tau_grid_hint
            result = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=ScaleSearchConfig(
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
                        min_equilibrium_epochs=2, max_epochs=12
                    ),
                    seed=seed,
                    halve_grid_steps=True,
                    persistence=PersistenceConfig(
                        resolve_within_interval="none",
                        densify_overlap_recover="lower_threshold",
                        densify_overlap_recover_threshold=thr,
                    ),
                ),
            )
            pr = result.persistence_result
            accept = (
                pr is not None
                and pr.tau_star_index == 0
                and int(pr.run_lengths[0]) >= 2
            )
            run0 = int(pr.run_lengths[0]) if pr is not None else -1
            key = (seed, thr)
            assert accept is expect_accept[key]
            if not accept:
                assert run0 == 1
                by[key] = {"accept": False, "run0": run0}
                print(
                    f"{seed:4d} {thr:5.2f} {str(accept):>5s} {run0:4d} "
                    f"{'—':>4s} {'—':>18s} {'—':>3s} {'—':>3s} {'—':>3s} "
                    f"{'—':>7s}"
                )
                continue
            assert pr is not None
            assert run0 == 16
            load = np.asarray(result.load_trace, dtype=float)
            phi = np.asarray(result.phi_trace, dtype=float)
            stab = list(result.stabilized_flags)
            i_hi = run0 - 1
            finite = [
                idx
                for idx in range(0, i_hi + 1)
                if np.isfinite(float(phi[idx]))
            ]
            assert len(finite) == 16
            phi_peak_idx = max(finite, key=lambda i: float(phi[i]))
            skipped = [idx for idx in range(0, i_hi + 1) if not stab[idx]]
            fsa_idx = next(
                (
                    idx
                    for idx in range(phi_peak_idx + 1, i_hi + 1)
                    if stab[idx]
                ),
                None,
            )
            assert fsa_idx is not None
            idx_lw = _resolve_persistence_tau_index(
                pr,
                load,
                stab,
                PersistenceConfig(
                    resolve_within_interval="load_weighted_interval"
                ),
            )
            idx_lc = _resolve_persistence_tau_index(
                pr,
                load,
                stab,
                PersistenceConfig(resolve_within_interval="load_crossover"),
            )
            ratio = float(phi[fsa_idx]) / float(phi[phi_peak_idx])
            by[key] = {
                "accept": True,
                "run0": run0,
                "peak": int(phi_peak_idx),
                "skip": skipped,
                "fsa": int(fsa_idx),
                "idx_lc": int(idx_lc),
                "idx_lw": int(idx_lw),
                "ratio": float(ratio),
                "stab_peak": bool(stab[phi_peak_idx]),
            }
            print(
                f"{seed:4d} {thr:5.2f} {str(accept):>5s} {run0:4d} "
                f"{phi_peak_idx:4d} {str(skipped):>18s} {fsa_idx:3d} "
                f"{idx_lc:3d} {idx_lw:3d} {ratio:7.4f}"
            )

    # On every accept: peak=idx1, fsa=idx2, LC=coarse, skip≡expect_skip[seed].
    for (seed, thr), row in by.items():
        if not bool(row["accept"]):
            continue
        assert int(row["peak"]) == 1
        assert int(row["fsa"]) == 2
        assert int(row["idx_lc"]) == 0
        assert list(row["skip"]) == list(expect_skip[seed])  # type: ignore[arg-type]
        # Thr-invariance of skip topology: same seed, any other accepting thr.
        for thr2 in (0.30, 0.35, 0.40):
            other = by.get((seed, thr2))
            if other is None or not bool(other["accept"]):
                continue
            assert list(other["skip"]) == list(row["skip"])  # type: ignore[arg-type]
            assert int(other["fsa"]) == int(row["fsa"])
            assert int(other["idx_lc"]) == 0

    # Only seed2@0.30 has near-peak stab-skip (widens LC straddle to 0↔2);
    # still LC≡coarse. LW≡peak only on that singleton.
    assert bool(by[(2, 0.30)]["stab_peak"]) is False
    assert 1 in list(by[(2, 0.30)]["skip"])  # type: ignore[arg-type]
    assert int(by[(2, 0.30)]["idx_lw"]) == 1
    assert int(by[(2, 0.30)]["idx_lc"]) == 0
    assert 0.70 < float(by[(2, 0.30)]["ratio"]) < 0.85
    for seed in (0, 1, 3, 4):
        for thr in (0.30, 0.35, 0.40):
            row = by[(seed, thr)]
            if not bool(row["accept"]):
                continue
            assert bool(row["stab_peak"]) is True
            assert 1 not in list(row["skip"])  # type: ignore[arg-type]
            assert int(row["idx_lw"]) == 0

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_thr030_dense_multiseed_phi_half_life() -> None:
    # EXPERIMENT (A6-T87): thr=0.30 densified seeds0..4 — Phi "half-life"
    # index = first grid index after the in-block Phi peak where
    # Phi[i]/Phi[peak] <= 0.5, plus a linear-in-ratio fractional offset from
    # the peak. Extends T84's +1..+4 decay curve. Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    by: dict[int, dict[str, object]] = {}
    print("\nA6-T87 thr0.30 densified multi-seed Phi half-life (decay-to-0.5)")
    header = (
        f"{'seed':>4s} {'peak':>4s} {'half':>4s} {'off':>4s} {'fhalf':>7s} "
        f"{'r_hm1':>7s} {'r_h':>7s} {'LC':>3s} {'LW':>3s}"
    )
    print(header)
    print("-" * len(header))
    for seed in range(5):
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        assert gt.expected_tau is not None
        tau_lo, tau_hi = gt.tau_grid_hint
        result = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=ScaleSearchConfig(
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=seed,
                halve_grid_steps=True,
                persistence=PersistenceConfig(
                    resolve_within_interval="none",
                    densify_overlap_recover="lower_threshold",
                    densify_overlap_recover_threshold=0.30,
                ),
            ),
        )
        assert result.persistence_result is not None
        pr = result.persistence_result
        assert pr.tau_star_index == 0
        assert int(pr.run_lengths[0]) == 16
        load = np.asarray(result.load_trace, dtype=float)
        phi = np.asarray(result.phi_trace, dtype=float)
        stab = list(result.stabilized_flags)
        i_hi = 15
        finite = [
            idx for idx in range(0, i_hi + 1) if np.isfinite(float(phi[idx]))
        ]
        assert len(finite) == 16
        peak = max(finite, key=lambda i: float(phi[i]))
        phi_p = float(phi[peak])
        half_idx: int | None = None
        frac_off: float | None = None
        prev_r = 1.0
        for off in range(1, i_hi - peak + 1):
            r = float(phi[peak + off]) / phi_p
            if r <= 0.5:
                half_idx = peak + off
                t = (prev_r - 0.5) / (prev_r - r) if prev_r != r else 0.0
                frac_off = float(off - 1) + float(t)
                break
            prev_r = r
        assert half_idx is not None and frac_off is not None
        r_hm1 = float(phi[half_idx - 1]) / phi_p
        r_h = float(phi[half_idx]) / phi_p
        idx_lw = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval="load_weighted_interval"),
        )
        idx_lc = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval="load_crossover"),
        )
        by[seed] = {
            "peak": int(peak),
            "half": int(half_idx),
            "off": int(half_idx - peak),
            "frac": float(frac_off),
            "r_hm1": float(r_hm1),
            "r_h": float(r_h),
            "idx_lc": int(idx_lc),
            "idx_lw": int(idx_lw),
        }
        print(
            f"{seed:4d} {peak:4d} {half_idx:4d} {half_idx - peak:4d} "
            f"{frac_off:7.3f} {r_hm1:7.4f} {r_h:7.4f} {idx_lc:3d} {idx_lw:3d}"
        )

    # Peak / LC topology matches T78/T81/T84; half-life always exists in-block.
    for seed in range(5):
        assert int(by[seed]["peak"]) == 1
        assert int(by[seed]["idx_lc"]) == 0
        assert float(by[seed]["r_hm1"]) > 0.5
        assert float(by[seed]["r_h"]) <= 0.5

    # Absolute half-life index pins (seed2 fastest; seed3 slowest).
    expect_half = {0: 5, 1: 5, 2: 4, 3: 6, 4: 5}
    expect_frac = {0: 3.907, 1: 3.489, 2: 2.771, 3: 4.044, 4: 3.724}
    for seed, want in expect_half.items():
        assert int(by[seed]["half"]) == want
        assert int(by[seed]["off"]) == want - 1
        assert abs(float(by[seed]["frac"]) - float(expect_frac[seed])) < 0.05

    # Ordering: seed2 < seed1 < seed4 < seed0 < seed3 on fractional half-life.
    assert (
        float(by[2]["frac"])
        < float(by[1]["frac"])
        < float(by[4]["frac"])
        < float(by[0]["frac"])
        < float(by[3]["frac"])
    )

    # LW landings unchanged vs T74/T78.
    for seed in (0, 1, 3, 4):
        assert int(by[seed]["idx_lw"]) == 0
    assert int(by[2]["idx_lw"]) == 1

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_thr030_dense_multiseed_load_straddle_margin_vs_fsa() -> None:
    # EXPERIMENT (A6-T88): thr=0.30 densified seeds0..4 — export |L(0)-1|
    # vs |L(fsa)-1| (and |L(peak)-1|) to quantify why LC stays at coarse-end
    # even when first-stab-after-peak is Phi-near-peak (T81/T79). Defaults
    # stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    by: dict[int, dict[str, object]] = {}
    print("\nA6-T88 thr0.30 densified multi-seed |L0-1| vs |Lfsa-1| margin")
    header = (
        f"{'seed':>4s} {'peak':>4s} {'fsa':>3s} {'L0':>7s} {'Lfsa':>7s} "
        f"{'m0':>7s} {'mfsa':>7s} {'dm':>7s} {'mpk':>7s} {'LC':>3s} {'LW':>3s}"
    )
    print(header)
    print("-" * len(header))
    for seed in range(5):
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        assert gt.expected_tau is not None
        tau_lo, tau_hi = gt.tau_grid_hint
        result = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=ScaleSearchConfig(
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=seed,
                halve_grid_steps=True,
                persistence=PersistenceConfig(
                    resolve_within_interval="none",
                    densify_overlap_recover="lower_threshold",
                    densify_overlap_recover_threshold=0.30,
                ),
            ),
        )
        assert result.persistence_result is not None
        pr = result.persistence_result
        assert pr.tau_star_index == 0
        assert int(pr.run_lengths[0]) == 16
        load = np.asarray(result.load_trace, dtype=float)
        phi = np.asarray(result.phi_trace, dtype=float)
        stab = list(result.stabilized_flags)
        i_hi = 15
        finite = [
            idx for idx in range(0, i_hi + 1) if np.isfinite(float(phi[idx]))
        ]
        peak = max(finite, key=lambda i: float(phi[i]))
        fsa = next(
            idx for idx in range(peak + 1, i_hi + 1) if stab[idx]
        )
        L0 = float(load[0])
        Lfsa = float(load[fsa])
        Lpeak = float(load[peak])
        m0 = abs(L0 - 1.0)
        mfsa = abs(Lfsa - 1.0)
        mpk = abs(Lpeak - 1.0)
        idx_lw = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval="load_weighted_interval"),
        )
        idx_lc = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval="load_crossover"),
        )
        by[seed] = {
            "peak": int(peak),
            "fsa": int(fsa),
            "L0": L0,
            "Lfsa": Lfsa,
            "m0": m0,
            "mfsa": mfsa,
            "dm": mfsa - m0,
            "mpk": mpk,
            "idx_lc": int(idx_lc),
            "idx_lw": int(idx_lw),
        }
        print(
            f"{seed:4d} {peak:4d} {fsa:3d} {L0:7.4f} {Lfsa:7.4f} "
            f"{m0:7.4f} {mfsa:7.4f} {mfsa - m0:7.4f} {mpk:7.4f} "
            f"{idx_lc:3d} {idx_lw:3d}"
        )

    # Topology: peak=1, fsa=2, LC≡coarse always; LW≡peak only seed2.
    for seed in range(5):
        assert int(by[seed]["peak"]) == 1
        assert int(by[seed]["fsa"]) == 2
        assert int(by[seed]["idx_lc"]) == 0
        # LC prefers coarse because |L0-1| < |Lfsa-1| on every accept.
        assert float(by[seed]["m0"]) < float(by[seed]["mfsa"])
        assert float(by[seed]["dm"]) > 0.5

    expect_m0 = {0: 0.2685, 1: 0.3500, 2: 0.3858, 3: 0.2784, 4: 0.3078}
    expect_mfsa = {0: 1.2828, 1: 1.3395, 2: 1.0594, 3: 1.2816, 4: 1.4285}
    for seed in range(5):
        assert abs(float(by[seed]["m0"]) - expect_m0[seed]) < 0.02
        assert abs(float(by[seed]["mfsa"]) - expect_mfsa[seed]) < 0.02

    # Seed2 has the smallest (but still large) mfsa-m0 gap; still LC≡coarse.
    # Peak load is also farther from unit than coarse (|Lpeak-1|>|L0-1|), so
    # the LW≠coarse nudge is |log L|-driven, not |L-1|-driven.
    assert float(by[2]["dm"]) == min(float(by[s]["dm"]) for s in range(5))
    assert float(by[2]["mpk"]) > float(by[2]["m0"])
    assert int(by[2]["idx_lw"]) == 1
    for seed in (0, 1, 3, 4):
        assert int(by[seed]["idx_lw"]) == 0

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_phi_half_life_x_thr_floors_densify_multiseed() -> None:
    # EXPERIMENT (A6-T90): Phi half-life × densify recover thr ∈ {0.30, 0.35,
    # 0.40} on seeds0..4. Pins T64 accept map and shows half-life indices /
    # fractional offsets are thr-invariant on shared accepts (gate-only).
    # Defaults stay off; do not flip awaiting.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    by: dict[tuple[float, int], dict[str, object]] = {}
    print("\nA6-T90 Phi half-life × thr floors densify multi-seed")
    header = (
        f"{'thr':>5s} {'seed':>4s} {'acc':>3s} {'peak':>4s} {'half':>4s} "
        f"{'fhalf':>7s} {'LC':>3s} {'LW':>3s}"
    )
    print(header)
    print("-" * len(header))
    for thr in (0.30, 0.35, 0.40):
        for seed in range(5):
            dataset = make_hierarchical_gaussian(
                children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
            )
            gt = dataset.ground_truth
            assert gt.expected_tau is not None
            tau_lo, tau_hi = gt.tau_grid_hint
            result = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=ScaleSearchConfig(
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
                        min_equilibrium_epochs=2, max_epochs=12
                    ),
                    seed=seed,
                    halve_grid_steps=True,
                    persistence=PersistenceConfig(
                        resolve_within_interval="none",
                        densify_overlap_recover="lower_threshold",
                        densify_overlap_recover_threshold=thr,
                    ),
                ),
            )
            assert result.persistence_result is not None
            pr = result.persistence_result
            accept = pr.tau_star_index is not None
            row: dict[str, object] = {
                "accept": int(accept),
                "run0": int(pr.run_lengths[0]),
            }
            if accept:
                assert pr.tau_star_index == 0
                assert int(pr.run_lengths[0]) == 16
                load = np.asarray(result.load_trace, dtype=float)
                phi = np.asarray(result.phi_trace, dtype=float)
                stab = list(result.stabilized_flags)
                i_hi = 15
                finite = [
                    idx
                    for idx in range(0, i_hi + 1)
                    if np.isfinite(float(phi[idx]))
                ]
                assert len(finite) == 16
                peak = max(finite, key=lambda i: float(phi[i]))
                phi_p = float(phi[peak])
                half_idx: int | None = None
                frac_off: float | None = None
                prev_r = 1.0
                for off in range(1, i_hi - peak + 1):
                    r = float(phi[peak + off]) / phi_p
                    if r <= 0.5:
                        half_idx = peak + off
                        t = (
                            (prev_r - 0.5) / (prev_r - r)
                            if prev_r != r
                            else 0.0
                        )
                        frac_off = float(off - 1) + float(t)
                        break
                    prev_r = r
                assert half_idx is not None and frac_off is not None
                idx_lw = _resolve_persistence_tau_index(
                    pr,
                    load,
                    stab,
                    PersistenceConfig(
                        resolve_within_interval="load_weighted_interval"
                    ),
                )
                idx_lc = _resolve_persistence_tau_index(
                    pr,
                    load,
                    stab,
                    PersistenceConfig(
                        resolve_within_interval="load_crossover"
                    ),
                )
                row.update(
                    {
                        "peak": int(peak),
                        "half": int(half_idx),
                        "frac": float(frac_off),
                        "idx_lc": int(idx_lc),
                        "idx_lw": int(idx_lw),
                    }
                )
                print(
                    f"{thr:5.2f} {seed:4d} {1:3d} {peak:4d} {half_idx:4d} "
                    f"{frac_off:7.3f} {idx_lc:3d} {idx_lw:3d}"
                )
            else:
                print(f"{thr:5.2f} {seed:4d} {0:3d} {'-':>4} {'-':>4} {'-':>7} {'-':>3} {'-':>3}")
            by[(thr, seed)] = row

    # Accept map matches T64 densify column.
    for seed in range(5):
        assert by[(0.30, seed)]["accept"] == 1
    for seed in (0, 1, 3, 4):
        assert by[(0.35, seed)]["accept"] == 1
    assert by[(0.35, 2)]["accept"] == 0
    for seed in (0, 3):
        assert by[(0.40, seed)]["accept"] == 1
    for seed in (1, 2, 4):
        assert by[(0.40, seed)]["accept"] == 0

    # thr0.30 half-life pins match T87; thr-invariant on shared accepts.
    expect_half = {0: 5, 1: 5, 2: 4, 3: 6, 4: 5}
    expect_frac = {0: 3.907, 1: 3.489, 2: 2.771, 3: 4.044, 4: 3.724}
    for seed, want in expect_half.items():
        assert int(by[(0.30, seed)]["half"]) == want
        assert abs(float(by[(0.30, seed)]["frac"]) - expect_frac[seed]) < 0.05
    for thr in (0.35, 0.40):
        for seed in range(5):
            if by[(thr, seed)]["accept"] != 1:
                continue
            assert int(by[(thr, seed)]["peak"]) == 1
            assert int(by[(thr, seed)]["half"]) == expect_half[seed]
            assert abs(
                float(by[(thr, seed)]["frac"]) - expect_frac[seed]
            ) < 0.05
            assert int(by[(thr, seed)]["idx_lc"]) == 0

    # LW≠coarse remains the thr0.30 densified seed2 singleton only.
    assert int(by[(0.30, 2)]["idx_lw"]) == 1
    for seed in (0, 1, 3, 4):
        assert int(by[(0.30, seed)]["idx_lw"]) == 0
    for thr in (0.35, 0.40):
        for seed in range(5):
            if by[(thr, seed)]["accept"] == 1:
                assert int(by[(thr, seed)]["idx_lw"]) == 0

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_thr030_dense_lpeak_vs_l0_lw_mechanism_export() -> None:
    # EXPERIMENT (A6-T91): thr=0.30 densified seeds0..4 — export |Lpeak-1|
    # vs |L0-1| and |log Lpeak| vs |log L0| to lock the LW≠coarse mechanism
    # table (T70/T74/T88). Only seed2 has |log Lpeak| < |log L0|; every seed
    # has |Lpeak-1| > |L0-1|, so |L-1| alone cannot explain the LW nudge.
    # Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    by: dict[int, dict[str, object]] = {}
    print("\nA6-T91 thr0.30 densified |Lpeak-1| vs |L0-1| + |log L| export")
    header = (
        f"{'seed':>4s} {'peak':>4s} {'L0':>7s} {'Lpk':>7s} {'m0':>7s} "
        f"{'mpk':>7s} {'log0':>7s} {'logpk':>7s} {'LW':>3s}"
    )
    print(header)
    print("-" * len(header))
    for seed in range(5):
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        assert gt.expected_tau is not None
        tau_lo, tau_hi = gt.tau_grid_hint
        result = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=ScaleSearchConfig(
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=seed,
                halve_grid_steps=True,
                persistence=PersistenceConfig(
                    resolve_within_interval="none",
                    densify_overlap_recover="lower_threshold",
                    densify_overlap_recover_threshold=0.30,
                ),
            ),
        )
        assert result.persistence_result is not None
        pr = result.persistence_result
        assert pr.tau_star_index == 0
        assert int(pr.run_lengths[0]) == 16
        load = np.asarray(result.load_trace, dtype=float)
        phi = np.asarray(result.phi_trace, dtype=float)
        stab = list(result.stabilized_flags)
        i_hi = 15
        finite = [
            idx for idx in range(0, i_hi + 1) if np.isfinite(float(phi[idx]))
        ]
        peak = max(finite, key=lambda i: float(phi[i]))
        L0 = float(load[0])
        Lpeak = float(load[peak])
        m0 = abs(L0 - 1.0)
        mpk = abs(Lpeak - 1.0)
        log0 = abs(float(np.log(L0)))
        logpk = abs(float(np.log(Lpeak)))
        idx_lw = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval="load_weighted_interval"),
        )
        idx_lc = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval="load_crossover"),
        )
        by[seed] = {
            "peak": int(peak),
            "L0": L0,
            "Lpeak": Lpeak,
            "m0": m0,
            "mpk": mpk,
            "log0": log0,
            "logpk": logpk,
            "idx_lc": int(idx_lc),
            "idx_lw": int(idx_lw),
        }
        print(
            f"{seed:4d} {peak:4d} {L0:7.4f} {Lpeak:7.4f} {m0:7.4f} "
            f"{mpk:7.4f} {log0:7.4f} {logpk:7.4f} {idx_lw:3d}"
        )

    # Topology + LC coarse on all accepts.
    for seed in range(5):
        assert int(by[seed]["peak"]) == 1
        assert int(by[seed]["idx_lc"]) == 0
        # |L-1| favors coarse on every seed (peak is farther from unit).
        assert float(by[seed]["mpk"]) > float(by[seed]["m0"])

    expect_m0 = {0: 0.2685, 1: 0.3500, 2: 0.3858, 3: 0.2784, 4: 0.3078}
    expect_mpk = {0: 0.7641, 1: 0.6353, 2: 0.5625, 3: 0.7851, 4: 0.9029}
    expect_log0 = {0: 0.3127, 1: 0.4308, 2: 0.4874, 3: 0.3263, 4: 0.3679}
    expect_logpk = {0: 0.5676, 1: 0.4918, 2: 0.4463, 3: 0.5795, 4: 0.6434}
    for seed in range(5):
        assert abs(float(by[seed]["m0"]) - expect_m0[seed]) < 0.02
        assert abs(float(by[seed]["mpk"]) - expect_mpk[seed]) < 0.02
        assert abs(float(by[seed]["log0"]) - expect_log0[seed]) < 0.02
        assert abs(float(by[seed]["logpk"]) - expect_logpk[seed]) < 0.02

    # LW≠coarse iff |log Lpeak| < |log L0| — only seed2.
    assert float(by[2]["logpk"]) < float(by[2]["log0"])
    assert int(by[2]["idx_lw"]) == 1
    for seed in (0, 1, 3, 4):
        assert float(by[seed]["logpk"]) > float(by[seed]["log0"])
        assert int(by[seed]["idx_lw"]) == 0

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_phi_half_life_x_halve_grid_off_vs_on_thr030() -> None:
    # EXPERIMENT (A6-T93): Phi half-life × ``halve_grid_steps`` off vs on at
    # thr=0.30 densify-recover. Standard grid collapses half-life to the next
    # log-step (peak+1; frac≲1); densify reveals multi-step decay (T87 pins).
    # Physical τ_half/τ_peak is larger on the coarse grid (~0.55) than densified
    # (~0.25–0.44). Seed2 accepts only under densify. Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    by: dict[tuple[bool, int], dict[str, object]] = {}
    print("\nA6-T93 Phi half-life × halve_grid off vs on thr0.30")
    header = (
        f"{'dense':>5s} {'seed':>4s} {'acc':>3s} {'n':>3s} {'peak':>4s} "
        f"{'half':>4s} {'frac':>7s} {'tau_r':>7s} {'LC':>3s} {'LW':>3s}"
    )
    print(header)
    print("-" * len(header))
    for dense in (False, True):
        for seed in range(5):
            dataset = make_hierarchical_gaussian(
                children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
            )
            gt = dataset.ground_truth
            assert gt.expected_tau is not None
            tau_lo, tau_hi = gt.tau_grid_hint
            result = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=ScaleSearchConfig(
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
                        min_equilibrium_epochs=2, max_epochs=12
                    ),
                    seed=seed,
                    halve_grid_steps=dense,
                    persistence=PersistenceConfig(
                        resolve_within_interval="none",
                        densify_overlap_recover="lower_threshold",
                        densify_overlap_recover_threshold=0.30,
                    ),
                ),
            )
            assert result.persistence_result is not None
            pr = result.persistence_result
            accept = pr.tau_star_index is not None
            n = len(result.phi_trace)
            row: dict[str, object] = {
                "accept": int(accept),
                "n": int(n),
                "run0": int(pr.run_lengths[0]),
            }
            if accept:
                assert pr.tau_star_index == 0
                phi = np.asarray(result.phi_trace, dtype=float)
                load = np.asarray(result.load_trace, dtype=float)
                taus = np.asarray(result.tau_grid, dtype=float)
                stab = list(result.stabilized_flags)
                i_hi = n - 1
                finite = [
                    idx
                    for idx in range(0, i_hi + 1)
                    if np.isfinite(float(phi[idx]))
                ]
                peak = max(finite, key=lambda i: float(phi[i]))
                phi_p = float(phi[peak])
                half_idx: int | None = None
                frac_off: float | None = None
                prev_r = 1.0
                for off in range(1, i_hi - peak + 1):
                    r = float(phi[peak + off]) / phi_p
                    if r <= 0.5:
                        half_idx = peak + off
                        t = (
                            (prev_r - 0.5) / (prev_r - r)
                            if prev_r != r
                            else 0.0
                        )
                        frac_off = float(off - 1) + float(t)
                        break
                    prev_r = r
                assert half_idx is not None and frac_off is not None
                tau_r = float(taus[half_idx]) / float(taus[peak])
                idx_lw = _resolve_persistence_tau_index(
                    pr,
                    load,
                    stab,
                    PersistenceConfig(
                        resolve_within_interval="load_weighted_interval"
                    ),
                )
                idx_lc = _resolve_persistence_tau_index(
                    pr,
                    load,
                    stab,
                    PersistenceConfig(
                        resolve_within_interval="load_crossover"
                    ),
                )
                row.update(
                    {
                        "peak": int(peak),
                        "half": int(half_idx),
                        "frac": float(frac_off),
                        "tau_r": float(tau_r),
                        "idx_lc": int(idx_lc),
                        "idx_lw": int(idx_lw),
                    }
                )
                print(
                    f"{int(dense):5d} {seed:4d} {1:3d} {n:3d} {peak:4d} "
                    f"{half_idx:4d} {frac_off:7.3f} {tau_r:7.4f} "
                    f"{idx_lc:3d} {idx_lw:3d}"
                )
            else:
                print(
                    f"{int(dense):5d} {seed:4d} {0:3d} {n:3d} "
                    f"{'-':>4} {'-':>4} {'-':>7} {'-':>7} {'-':>3} {'-':>3}"
                )
            by[(dense, seed)] = row

    # Accept map: standard rejects seed2; densify accepts all (T64 thr0.30).
    for seed in (0, 1, 3, 4):
        assert by[(False, seed)]["accept"] == 1
        assert by[(False, seed)]["n"] == 8
    assert by[(False, 2)]["accept"] == 0
    assert by[(False, 2)]["run0"] == 1
    for seed in range(5):
        assert by[(True, seed)]["accept"] == 1
        assert by[(True, seed)]["n"] == 16
        assert by[(True, seed)]["run0"] == 16

    # Standard: half-life collapses to peak+1 (coarse log-step).
    expect_std_frac = {0: 0.997, 1: 0.941, 3: 0.876, 4: 0.950}
    for seed, want in expect_std_frac.items():
        assert int(by[(False, seed)]["peak"]) == 1
        assert int(by[(False, seed)]["half"]) == 2
        assert abs(float(by[(False, seed)]["frac"]) - want) < 0.05
        assert abs(float(by[(False, seed)]["tau_r"]) - 0.5520) < 0.02
        assert int(by[(False, seed)]["idx_lc"]) == 0
        assert int(by[(False, seed)]["idx_lw"]) == 0

    # Densify: multi-step half-life pins match T87.
    expect_half = {0: 5, 1: 5, 2: 4, 3: 6, 4: 5}
    expect_frac = {0: 3.907, 1: 3.489, 2: 2.771, 3: 4.044, 4: 3.724}
    expect_tau_r = {0: 0.3299, 1: 0.3299, 2: 0.4353, 3: 0.2500, 4: 0.3299}
    for seed, want in expect_half.items():
        assert int(by[(True, seed)]["peak"]) == 1
        assert int(by[(True, seed)]["half"]) == want
        assert abs(float(by[(True, seed)]["frac"]) - expect_frac[seed]) < 0.05
        assert abs(float(by[(True, seed)]["tau_r"]) - expect_tau_r[seed]) < 0.02
        assert int(by[(True, seed)]["idx_lc"]) == 0
    assert int(by[(True, 2)]["idx_lw"]) == 1
    for seed in (0, 1, 3, 4):
        assert int(by[(True, seed)]["idx_lw"]) == 0
        # Densify finds a deeper τ_half/τ_peak than the coarse-grid one-step.
        assert float(by[(True, seed)]["tau_r"]) < float(
            by[(False, seed)]["tau_r"]
        )

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_stab_skip_x_phi_half_life_correlation_thr030_dense() -> None:
    # EXPERIMENT (A6-T94): thr=0.30 densified seeds0..4 — correlate
    # stabilization skips with Phi half-life. Only seed2 has a near-peak
    # stab-skip (peak unstabilized) and is the fastest half-life; fine-end
    # skips on seeds1/3 do not predict half-life order. Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    by: dict[int, dict[str, object]] = {}
    print("\nA6-T94 thr0.30 densified stab-skip × Phi half-life correlation")
    header = (
        f"{'seed':>4s} {'peak':>4s} {'half':>4s} {'frac':>7s} {'nskp':>4s} "
        f"{'fsa':>3s} {'nskips':>6s} {'LC':>3s} {'LW':>3s}"
    )
    print(header)
    print("-" * len(header))
    for seed in range(5):
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        assert gt.expected_tau is not None
        tau_lo, tau_hi = gt.tau_grid_hint
        result = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=ScaleSearchConfig(
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=seed,
                halve_grid_steps=True,
                persistence=PersistenceConfig(
                    resolve_within_interval="none",
                    densify_overlap_recover="lower_threshold",
                    densify_overlap_recover_threshold=0.30,
                ),
            ),
        )
        assert result.persistence_result is not None
        pr = result.persistence_result
        assert pr.tau_star_index == 0
        assert int(pr.run_lengths[0]) == 16
        load = np.asarray(result.load_trace, dtype=float)
        phi = np.asarray(result.phi_trace, dtype=float)
        stab = list(result.stabilized_flags)
        i_hi = 15
        finite = [
            idx for idx in range(0, i_hi + 1) if np.isfinite(float(phi[idx]))
        ]
        peak = max(finite, key=lambda i: float(phi[i]))
        phi_p = float(phi[peak])
        half_idx: int | None = None
        frac_off: float | None = None
        prev_r = 1.0
        for off in range(1, i_hi - peak + 1):
            r = float(phi[peak + off]) / phi_p
            if r <= 0.5:
                half_idx = peak + off
                t = (prev_r - 0.5) / (prev_r - r) if prev_r != r else 0.0
                frac_off = float(off - 1) + float(t)
                break
            prev_r = r
        assert half_idx is not None and frac_off is not None
        skips = [i for i in range(16) if not stab[i]]
        near_peak_skip = int(not stab[peak])
        fsa = next(idx for idx in range(peak + 1, i_hi + 1) if stab[idx])
        idx_lw = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval="load_weighted_interval"),
        )
        idx_lc = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval="load_crossover"),
        )
        by[seed] = {
            "peak": int(peak),
            "half": int(half_idx),
            "frac": float(frac_off),
            "near": int(near_peak_skip),
            "skips": list(skips),
            "fsa": int(fsa),
            "idx_lc": int(idx_lc),
            "idx_lw": int(idx_lw),
        }
        print(
            f"{seed:4d} {peak:4d} {half_idx:4d} {frac_off:7.3f} "
            f"{near_peak_skip:4d} {fsa:3d} {str(skips):>6s} "
            f"{idx_lc:3d} {idx_lw:3d}"
        )

    # Half-life / topology pins match T87; fsa=2 always (T81).
    expect_half = {0: 5, 1: 5, 2: 4, 3: 6, 4: 5}
    expect_frac = {0: 3.907, 1: 3.489, 2: 2.771, 3: 4.044, 4: 3.724}
    expect_skips = {
        0: [],
        1: [14, 15],
        2: [1, 13, 14, 15],
        3: [15],
        4: [],
    }
    for seed in range(5):
        assert int(by[seed]["peak"]) == 1
        assert int(by[seed]["fsa"]) == 2
        assert int(by[seed]["half"]) == expect_half[seed]
        assert abs(float(by[seed]["frac"]) - expect_frac[seed]) < 0.05
        assert list(by[seed]["skips"]) == expect_skips[seed]
        assert int(by[seed]["idx_lc"]) == 0

    # Near-peak stab-skip is the seed2 singleton; coincides with fastest half-life.
    assert int(by[2]["near"]) == 1
    for seed in (0, 1, 3, 4):
        assert int(by[seed]["near"]) == 0
    assert float(by[2]["frac"]) == min(float(by[s]["frac"]) for s in range(5))
    # Fine-end skips alone do not predict half-life order (seed1 skips, seed3
    # skips one, but seed0/4 have empty skips and sit between them).
    assert (
        float(by[2]["frac"])
        < float(by[1]["frac"])
        < float(by[4]["frac"])
        < float(by[0]["frac"])
        < float(by[3]["frac"])
    )
    assert int(by[2]["idx_lw"]) == 1
    for seed in (0, 1, 3, 4):
        assert int(by[seed]["idx_lw"]) == 0

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_phi_half_life_x_halve_grid_x_thr_floors_densify() -> None:
    # EXPERIMENT (A6-T96): Phi half-life × ``halve_grid_steps`` off/on ×
    # densify-recover thr ∈ {0.30, 0.35, 0.40} on seeds0..4. Pins:
    # (1) densify accept map = T64/T90; (2) standard accept {0,1,3,4} is
    # thr-invariant (seed1 ov0≈0.47 clears even thr0.40); (3) half-life
    # indices/frac/tau_r thr-invariant within each densify mode on shared
    # accepts — std stays peak+1 (tau_r≈0.55), densify keeps T87 multi-step.
    # Defaults stay off; do not flip awaiting.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    by: dict[tuple[float, bool, int], dict[str, object]] = {}
    print("\nA6-T96 Phi half-life × halve_grid × thr floors densify")
    header = (
        f"{'thr':>5s} {'d':>1s} {'seed':>4s} {'acc':>3s} {'n':>3s} "
        f"{'peak':>4s} {'half':>4s} {'frac':>7s} {'tau_r':>7s} {'LW':>3s}"
    )
    print(header)
    print("-" * len(header))
    for thr in (0.30, 0.35, 0.40):
        for dense in (False, True):
            for seed in range(5):
                dataset = make_hierarchical_gaussian(
                    children_per_coarse=2,
                    n_samples=600,
                    ambient_dim=4,
                    seed=seed,
                )
                gt = dataset.ground_truth
                assert gt.expected_tau is not None
                tau_lo, tau_hi = gt.tau_grid_hint
                result = run_scale_search(
                    dataset.points,
                    dim=gt.ambient_dim,
                    config=ScaleSearchConfig(
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
                            min_equilibrium_epochs=2, max_epochs=12
                        ),
                        seed=seed,
                        halve_grid_steps=dense,
                        persistence=PersistenceConfig(
                            resolve_within_interval="none",
                            densify_overlap_recover="lower_threshold",
                            densify_overlap_recover_threshold=thr,
                        ),
                    ),
                )
                assert result.persistence_result is not None
                pr = result.persistence_result
                accept = pr.tau_star_index is not None
                n = len(result.phi_trace)
                row: dict[str, object] = {
                    "accept": int(accept),
                    "n": int(n),
                    "run0": int(pr.run_lengths[0]),
                }
                if accept:
                    assert pr.tau_star_index == 0
                    phi = np.asarray(result.phi_trace, dtype=float)
                    load = np.asarray(result.load_trace, dtype=float)
                    taus = np.asarray(result.tau_grid, dtype=float)
                    stab = list(result.stabilized_flags)
                    i_hi = n - 1
                    finite = [
                        idx
                        for idx in range(0, i_hi + 1)
                        if np.isfinite(float(phi[idx]))
                    ]
                    peak = max(finite, key=lambda i: float(phi[i]))
                    phi_p = float(phi[peak])
                    half_idx: int | None = None
                    frac_off: float | None = None
                    prev_r = 1.0
                    for off in range(1, i_hi - peak + 1):
                        r = float(phi[peak + off]) / phi_p
                        if r <= 0.5:
                            half_idx = peak + off
                            t = (
                                (prev_r - 0.5) / (prev_r - r)
                                if prev_r != r
                                else 0.0
                            )
                            frac_off = float(off - 1) + float(t)
                            break
                        prev_r = r
                    assert half_idx is not None and frac_off is not None
                    tau_r = float(taus[half_idx]) / float(taus[peak])
                    idx_lw = _resolve_persistence_tau_index(
                        pr,
                        load,
                        stab,
                        PersistenceConfig(
                            resolve_within_interval="load_weighted_interval"
                        ),
                    )
                    idx_lc = _resolve_persistence_tau_index(
                        pr,
                        load,
                        stab,
                        PersistenceConfig(
                            resolve_within_interval="load_crossover"
                        ),
                    )
                    row.update(
                        {
                            "peak": int(peak),
                            "half": int(half_idx),
                            "frac": float(frac_off),
                            "tau_r": float(tau_r),
                            "idx_lc": int(idx_lc),
                            "idx_lw": int(idx_lw),
                        }
                    )
                    print(
                        f"{thr:5.2f} {int(dense):1d} {seed:4d} {1:3d} {n:3d} "
                        f"{peak:4d} {half_idx:4d} {frac_off:7.3f} "
                        f"{tau_r:7.4f} {idx_lw:3d}"
                    )
                else:
                    print(
                        f"{thr:5.2f} {int(dense):1d} {seed:4d} {0:3d} {n:3d} "
                        f"{'-':>4} {'-':>4} {'-':>7} {'-':>7} {'-':>3}"
                    )
                by[(thr, dense, seed)] = row

    # Accept maps: densify = T64/T90; standard = {0,1,3,4} thr-invariant.
    for thr in (0.30, 0.35, 0.40):
        for seed in (0, 1, 3, 4):
            assert by[(thr, False, seed)]["accept"] == 1
            assert by[(thr, False, seed)]["n"] == 8
        assert by[(thr, False, 2)]["accept"] == 0
    for seed in range(5):
        assert by[(0.30, True, seed)]["accept"] == 1
        assert by[(0.30, True, seed)]["n"] == 16
    for seed in (0, 1, 3, 4):
        assert by[(0.35, True, seed)]["accept"] == 1
    assert by[(0.35, True, 2)]["accept"] == 0
    for seed in (0, 3):
        assert by[(0.40, True, seed)]["accept"] == 1
    for seed in (1, 2, 4):
        assert by[(0.40, True, seed)]["accept"] == 0

    # Standard half-life collapses to peak+1; thr-invariant on shared accepts.
    expect_std_frac = {0: 0.997, 1: 0.941, 3: 0.876, 4: 0.950}
    for thr in (0.30, 0.35, 0.40):
        for seed, want in expect_std_frac.items():
            assert int(by[(thr, False, seed)]["peak"]) == 1
            assert int(by[(thr, False, seed)]["half"]) == 2
            assert abs(float(by[(thr, False, seed)]["frac"]) - want) < 0.05
            assert abs(float(by[(thr, False, seed)]["tau_r"]) - 0.5520) < 0.02
            assert int(by[(thr, False, seed)]["idx_lc"]) == 0
            assert int(by[(thr, False, seed)]["idx_lw"]) == 0

    # Densify multi-step half-life pins; thr-invariant on shared accepts.
    expect_half = {0: 5, 1: 5, 2: 4, 3: 6, 4: 5}
    expect_frac = {0: 3.907, 1: 3.489, 2: 2.771, 3: 4.044, 4: 3.724}
    expect_tau_r = {0: 0.3299, 1: 0.3299, 2: 0.4353, 3: 0.2500, 4: 0.3299}
    for thr in (0.30, 0.35, 0.40):
        for seed in range(5):
            if by[(thr, True, seed)]["accept"] != 1:
                continue
            assert int(by[(thr, True, seed)]["peak"]) == 1
            assert int(by[(thr, True, seed)]["half"]) == expect_half[seed]
            assert abs(
                float(by[(thr, True, seed)]["frac"]) - expect_frac[seed]
            ) < 0.05
            assert abs(
                float(by[(thr, True, seed)]["tau_r"]) - expect_tau_r[seed]
            ) < 0.02
            assert int(by[(thr, True, seed)]["idx_lc"]) == 0
            if seed == 2:
                assert int(by[(thr, True, seed)]["idx_lw"]) == 1
            else:
                assert int(by[(thr, True, seed)]["idx_lw"]) == 0
                # Deeper physical ratio than coarse-grid one-step when both accept.
                if by[(thr, False, seed)]["accept"] == 1:
                    assert float(by[(thr, True, seed)]["tau_r"]) < float(
                        by[(thr, False, seed)]["tau_r"]
                    )

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_stab_skip_x_logL_seed2_joint_table_thr030_dense() -> None:
    # EXPERIMENT (A6-T97): thr=0.30 densified seeds0..4 — joint table of
    # near-peak stab-skip × |log Lpeak| vs |log L0| (plus |L-1| margins).
    # Pins seed2 as the unique cell with near-peak skip AND
    # |log Lpeak| < |log L0| AND LW≠coarse; |L-1| never favors the peak.
    # Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    by: dict[int, dict[str, object]] = {}
    print("\nA6-T97 thr0.30 densified stab-skip × |log L| joint table")
    header = (
        f"{'seed':>4s} {'peak':>4s} {'near':>4s} {'half':>4s} {'frac':>7s} "
        f"{'m0':>7s} {'mpk':>7s} {'log0':>7s} {'logpk':>7s} {'LW':>3s}"
    )
    print(header)
    print("-" * len(header))
    for seed in range(5):
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        assert gt.expected_tau is not None
        tau_lo, tau_hi = gt.tau_grid_hint
        result = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=ScaleSearchConfig(
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=seed,
                halve_grid_steps=True,
                persistence=PersistenceConfig(
                    resolve_within_interval="none",
                    densify_overlap_recover="lower_threshold",
                    densify_overlap_recover_threshold=0.30,
                ),
            ),
        )
        assert result.persistence_result is not None
        pr = result.persistence_result
        assert pr.tau_star_index == 0
        assert int(pr.run_lengths[0]) == 16
        load = np.asarray(result.load_trace, dtype=float)
        phi = np.asarray(result.phi_trace, dtype=float)
        stab = list(result.stabilized_flags)
        i_hi = 15
        finite = [
            idx for idx in range(0, i_hi + 1) if np.isfinite(float(phi[idx]))
        ]
        peak = max(finite, key=lambda i: float(phi[i]))
        phi_p = float(phi[peak])
        half_idx: int | None = None
        frac_off: float | None = None
        prev_r = 1.0
        for off in range(1, i_hi - peak + 1):
            r = float(phi[peak + off]) / phi_p
            if r <= 0.5:
                half_idx = peak + off
                t = (prev_r - 0.5) / (prev_r - r) if prev_r != r else 0.0
                frac_off = float(off - 1) + float(t)
                break
            prev_r = r
        assert half_idx is not None and frac_off is not None
        near = int(not stab[peak])
        skips = [i for i in range(16) if not stab[i]]
        L0 = float(load[0])
        Lpeak = float(load[peak])
        m0 = abs(L0 - 1.0)
        mpk = abs(Lpeak - 1.0)
        log0 = abs(float(np.log(L0)))
        logpk = abs(float(np.log(Lpeak)))
        log_favors_peak = int(logpk < log0)
        idx_lw = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval="load_weighted_interval"),
        )
        idx_lc = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval="load_crossover"),
        )
        by[seed] = {
            "peak": int(peak),
            "near": int(near),
            "skips": list(skips),
            "half": int(half_idx),
            "frac": float(frac_off),
            "m0": m0,
            "mpk": mpk,
            "log0": log0,
            "logpk": logpk,
            "log_fav": int(log_favors_peak),
            "idx_lc": int(idx_lc),
            "idx_lw": int(idx_lw),
        }
        print(
            f"{seed:4d} {peak:4d} {near:4d} {half_idx:4d} {frac_off:7.3f} "
            f"{m0:7.4f} {mpk:7.4f} {log0:7.4f} {logpk:7.4f} {idx_lw:3d}"
        )

    expect_half = {0: 5, 1: 5, 2: 4, 3: 6, 4: 5}
    expect_frac = {0: 3.907, 1: 3.489, 2: 2.771, 3: 4.044, 4: 3.724}
    expect_skips = {
        0: [],
        1: [14, 15],
        2: [1, 13, 14, 15],
        3: [15],
        4: [],
    }
    expect_m0 = {0: 0.2685, 1: 0.3500, 2: 0.3858, 3: 0.2784, 4: 0.3078}
    expect_mpk = {0: 0.7641, 1: 0.6353, 2: 0.5625, 3: 0.7851, 4: 0.9029}
    expect_log0 = {0: 0.3127, 1: 0.4308, 2: 0.4874, 3: 0.3263, 4: 0.3679}
    expect_logpk = {0: 0.5676, 1: 0.4918, 2: 0.4463, 3: 0.5795, 4: 0.6434}
    for seed in range(5):
        assert int(by[seed]["peak"]) == 1
        assert int(by[seed]["half"]) == expect_half[seed]
        assert abs(float(by[seed]["frac"]) - expect_frac[seed]) < 0.05
        assert list(by[seed]["skips"]) == expect_skips[seed]
        assert int(by[seed]["idx_lc"]) == 0
        assert float(by[seed]["mpk"]) > float(by[seed]["m0"])
        assert abs(float(by[seed]["m0"]) - expect_m0[seed]) < 0.02
        assert abs(float(by[seed]["mpk"]) - expect_mpk[seed]) < 0.02
        assert abs(float(by[seed]["log0"]) - expect_log0[seed]) < 0.02
        assert abs(float(by[seed]["logpk"]) - expect_logpk[seed]) < 0.02

    # Joint singleton: near-peak skip ≡ |log L| favors peak ≡ LW≠coarse ≡ seed2.
    assert int(by[2]["near"]) == 1
    assert int(by[2]["log_fav"]) == 1
    assert int(by[2]["idx_lw"]) == 1
    assert float(by[2]["frac"]) == min(float(by[s]["frac"]) for s in range(5))
    for seed in (0, 1, 3, 4):
        assert int(by[seed]["near"]) == 0
        assert int(by[seed]["log_fav"]) == 0
        assert int(by[seed]["idx_lw"]) == 0
        assert float(by[seed]["logpk"]) > float(by[seed]["log0"])

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_phi_half_life_x_fractional_landing_proximity_thr030_dense() -> None:
    # EXPERIMENT (A6-T99): thr=0.30 densified seeds0..4 — Phi half-life index
    # proximity to mid / two-thirds / three-quarter / fine-end landings.
    # Pins mid as uniquely closest on every accept; half-life always coarser
    # than mid; densify-flip two-thirds (~1.00×E[τ]) remains several steps
    # finer than half-life. Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    modes = (
        ("mid", "mid_interval", _mid_interval_index),
        ("tt", "two_thirds_interval", _two_thirds_index),
        ("tq", "three_quarter_interval", _three_quarter_index),
        ("fine", "fine_end_of_block", lambda i_lo, i_hi: i_hi),
    )
    by: dict[int, dict[str, object]] = {}
    print("\nA6-T99 thr0.30 densified Phi half-life × fractional landing proximity")
    header = (
        f"{'seed':>4s} {'peak':>4s} {'half':>4s} {'mid':>3s} {'tt':>3s} "
        f"{'tq':>3s} {'fine':>4s} {'d_mid':>5s} {'d_tt':>5s} {'d_tq':>5s} "
        f"{'d_f':>5s} {'closest':>8s}"
    )
    print(header)
    print("-" * len(header))
    for seed in range(5):
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        assert gt.expected_tau is not None
        tau_lo, tau_hi = gt.tau_grid_hint
        result = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=ScaleSearchConfig(
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=seed,
                halve_grid_steps=True,
                persistence=PersistenceConfig(
                    resolve_within_interval="none",
                    densify_overlap_recover="lower_threshold",
                    densify_overlap_recover_threshold=0.30,
                ),
            ),
        )
        assert result.persistence_result is not None
        pr = result.persistence_result
        assert pr.tau_star_index == 0
        assert int(pr.run_lengths[0]) == 16
        load = np.asarray(result.load_trace, dtype=float)
        phi = np.asarray(result.phi_trace, dtype=float)
        taus = np.asarray(result.tau_grid, dtype=float)
        stab = list(result.stabilized_flags)
        i_lo, i_hi = 0, 15
        finite = [
            idx for idx in range(i_lo, i_hi + 1) if np.isfinite(float(phi[idx]))
        ]
        peak = max(finite, key=lambda i: float(phi[i]))
        phi_p = float(phi[peak])
        half_idx: int | None = None
        frac_off: float | None = None
        prev_r = 1.0
        for off in range(1, i_hi - peak + 1):
            r = float(phi[peak + off]) / phi_p
            if r <= 0.5:
                half_idx = peak + off
                t = (prev_r - 0.5) / (prev_r - r) if prev_r != r else 0.0
                frac_off = float(off - 1) + float(t)
                break
            prev_r = r
        assert half_idx is not None and frac_off is not None
        land: dict[str, int] = {}
        for name, mode, fn in modes:
            idx = int(fn(i_lo, i_hi))
            resolved = _resolve_persistence_tau_index(
                pr,
                load,
                stab,
                PersistenceConfig(resolve_within_interval=mode),  # type: ignore[arg-type]
            )
            assert int(resolved) == idx
            land[name] = idx
        dists = {name: abs(int(half_idx) - idx) for name, idx in land.items()}
        closest = min(dists.values())
        tied = sorted(name for name, d in dists.items() if d == closest)
        E = float(gt.expected_tau)
        by[seed] = {
            "peak": int(peak),
            "half": int(half_idx),
            "frac": float(frac_off),
            "land": dict(land),
            "dists": dict(dists),
            "tied": list(tied),
            "tau_h_over_E": float(taus[half_idx]) / E,
            "tau_mid_over_E": float(taus[land["mid"]]) / E,
            "tau_tt_over_E": float(taus[land["tt"]]) / E,
        }
        print(
            f"{seed:4d} {peak:4d} {half_idx:4d} {land['mid']:3d} {land['tt']:3d} "
            f"{land['tq']:3d} {land['fine']:4d} {dists['mid']:5d} {dists['tt']:5d} "
            f"{dists['tq']:5d} {dists['fine']:5d} {','.join(tied):>8s}"
        )

    # Landing indices are densify-invariant (full-block [0,15]).
    expect_land = {"mid": 7, "tt": 10, "tq": 11, "fine": 15}
    expect_half = {0: 5, 1: 5, 2: 4, 3: 6, 4: 5}
    expect_d_mid = {0: 2, 1: 2, 2: 3, 3: 1, 4: 2}
    expect_tau_h = {0: 4.0, 1: 4.0, 2: 5.278, 3: 3.031, 4: 4.0}
    for seed in range(5):
        assert int(by[seed]["peak"]) == 1
        assert int(by[seed]["half"]) == expect_half[seed]
        assert by[seed]["land"] == expect_land
        assert by[seed]["tied"] == ["mid"]
        assert int(by[seed]["dists"]["mid"]) == expect_d_mid[seed]
        # Half-life always coarser than mid (and finer landings).
        assert int(by[seed]["half"]) < int(by[seed]["land"]["mid"])
        assert int(by[seed]["dists"]["mid"]) < int(by[seed]["dists"]["tt"])
        assert int(by[seed]["dists"]["tt"]) < int(by[seed]["dists"]["tq"])
        assert int(by[seed]["dists"]["tq"]) < int(by[seed]["dists"]["fine"])
        assert abs(float(by[seed]["tau_h_over_E"]) - expect_tau_h[seed]) < 0.05
        assert abs(float(by[seed]["tau_mid_over_E"]) - 2.297) < 0.05
        # Densify-flip two-thirds lands at ~1.00×E[τ] — farther from half-life
        # than mid despite being closer to fine-leaf E[τ].
        assert abs(float(by[seed]["tau_tt_over_E"]) - 1.0) < 0.05
        assert float(by[seed]["tau_h_over_E"]) > float(by[seed]["tau_mid_over_E"])
        assert float(by[seed]["tau_mid_over_E"]) > float(by[seed]["tau_tt_over_E"])

    # Seed3 is nearest half↔mid; seed2 (fastest half-life) is farthest.
    assert int(by[3]["dists"]["mid"]) == min(
        int(by[s]["dists"]["mid"]) for s in range(5)
    )
    assert int(by[2]["dists"]["mid"]) == max(
        int(by[s]["dists"]["mid"]) for s in range(5)
    )

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_phi_half_life_circle_swiss_x_halve_grid_no_persist() -> None:
    # EXPERIMENT (A6-T101): Phi half-life on circle / swiss-roll (no accepted
    # persist split) × ``halve_grid_steps`` off/on. Pins that half-life is
    # defined without a persistence block, and densify roughly doubles the
    # peak/half indices. Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert ScaleSearchConfig().halve_grid_steps is False

    fixtures = (
        (
            "circle",
            make_circle(
                n_samples=800, radius=1.0, noise=0.02, extrusion_dim=2, seed=21,
            ),
        ),
        ("swiss", make_swiss_roll(n_samples=800, seed=0)),
    )
    expect = {
        ("circle", False): {"n": 8, "peak": 4, "half": 6, "frac": 1.632},
        ("circle", True): {"n": 16, "peak": 8, "half": 12, "frac": 3.150},
        ("swiss", False): {"n": 8, "peak": 4, "half": 5, "frac": 0.960},
        ("swiss", True): {"n": 16, "peak": 8, "half": 10, "frac": 1.935},
    }
    by: dict[tuple[str, bool], dict[str, object]] = {}
    print("\nA6-T101 Phi half-life circle/swiss × halve_grid (no persist)")
    header = (
        f"{'name':>6s} {'dense':>5s} {'n':>3s} {'peak':>4s} {'half':>4s} "
        f"{'frac':>7s} {'LC':>3s}"
    )
    print(header)
    print("-" * len(header))
    for name, dataset in fixtures:
        gt = dataset.ground_truth
        assert gt.expected_tau is not None
        tau_lo, tau_hi = gt.tau_grid_hint
        for dense in (False, True):
            result = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=ScaleSearchConfig(
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
                    halve_grid_steps=dense,
                    persistence=PersistenceConfig(resolve_within_interval="none"),
                ),
            )
            assert result.persistence_result is not None
            assert result.persistence_result.tau_star_index is None
            phi = np.asarray(result.phi_trace, dtype=float)
            n = len(phi)
            finite = [
                idx for idx in range(n) if np.isfinite(float(phi[idx]))
            ]
            peak = max(finite, key=lambda i: float(phi[i]))
            phi_p = float(phi[peak])
            half_idx: int | None = None
            frac_off: float | None = None
            prev_r = 1.0
            for off in range(1, n - peak):
                r = float(phi[peak + off]) / phi_p
                if r <= 0.5:
                    half_idx = peak + off
                    t = (prev_r - 0.5) / (prev_r - r) if prev_r != r else 0.0
                    frac_off = float(off - 1) + float(t)
                    break
                prev_r = r
            assert half_idx is not None and frac_off is not None
            by[(name, dense)] = {
                "n": int(n),
                "peak": int(peak),
                "half": int(half_idx),
                "frac": float(frac_off),
                "lc": int(result.peak_index),
            }
            print(
                f"{name:>6s} {str(dense):>5s} {n:3d} {peak:4d} {half_idx:4d} "
                f"{frac_off:7.3f} {int(result.peak_index):3d}"
            )

    for key, want in expect.items():
        row = by[key]
        assert int(row["n"]) == int(want["n"])
        assert int(row["peak"]) == int(want["peak"])
        assert int(row["half"]) == int(want["half"])
        assert abs(float(row["frac"]) - float(want["frac"])) < 0.05
        assert int(row["half"]) > int(row["peak"])

    # Densify doubles grid length and roughly doubles peak/half indices.
    for name in ("circle", "swiss"):
        assert int(by[(name, True)]["n"]) == 2 * int(by[(name, False)]["n"])
        assert int(by[(name, True)]["peak"]) == 2 * int(by[(name, False)]["peak"])
        assert int(by[(name, True)]["half"]) == 2 * int(by[(name, False)]["half"])

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert ScaleSearchConfig().halve_grid_steps is False


def test_phi_half_life_x_load_weighted_proximity_thr030_dense() -> None:
    # EXPERIMENT (A6-T103): thr=0.30 densified seeds0..4 — Phi half-life index
    # proximity to load_weighted_interval vs mid. Pins that LW is farther from
    # half-life than mid on every accept except seed2 (LW≠coarse singleton),
    # where d(half,LW)=d(half,mid)=3; half-life proximity does not favor LW.
    # Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    by: dict[int, dict[str, object]] = {}
    print("\nA6-T103 thr0.30 densified Phi half-life × load-weighted proximity")
    header = (
        f"{'seed':>4s} {'peak':>4s} {'half':>4s} {'LW':>3s} {'mid':>3s} "
        f"{'d_LW':>4s} {'d_mid':>5s} {'L0':>6s} {'L1':>6s}"
    )
    print(header)
    print("-" * len(header))
    for seed in range(5):
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        assert gt.expected_tau is not None
        tau_lo, tau_hi = gt.tau_grid_hint
        result = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=ScaleSearchConfig(
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=seed,
                halve_grid_steps=True,
                persistence=PersistenceConfig(
                    resolve_within_interval="none",
                    densify_overlap_recover="lower_threshold",
                    densify_overlap_recover_threshold=0.30,
                ),
            ),
        )
        assert result.persistence_result is not None
        pr = result.persistence_result
        assert pr.tau_star_index == 0
        assert int(pr.run_lengths[0]) == 16
        load = np.asarray(result.load_trace, dtype=float)
        phi = np.asarray(result.phi_trace, dtype=float)
        stab = list(result.stabilized_flags)
        i_lo, i_hi = 0, 15
        finite = [
            idx for idx in range(i_lo, i_hi + 1) if np.isfinite(float(phi[idx]))
        ]
        peak = max(finite, key=lambda i: float(phi[i]))
        phi_p = float(phi[peak])
        half_idx: int | None = None
        frac_off: float | None = None
        prev_r = 1.0
        for off in range(1, i_hi - peak + 1):
            r = float(phi[peak + off]) / phi_p
            if r <= 0.5:
                half_idx = peak + off
                t = (prev_r - 0.5) / (prev_r - r) if prev_r != r else 0.0
                frac_off = float(off - 1) + float(t)
                break
            prev_r = r
        assert half_idx is not None and frac_off is not None
        lw = int(_load_weighted_index(i_lo, i_hi, load))
        mid = int(_mid_interval_index(i_lo, i_hi))
        resolved = _resolve_persistence_tau_index(
            pr,
            load,
            stab,
            PersistenceConfig(resolve_within_interval="load_weighted_interval"),
        )
        assert int(resolved) == lw
        d_lw = abs(int(half_idx) - lw)
        d_mid = abs(int(half_idx) - mid)
        by[seed] = {
            "peak": int(peak),
            "half": int(half_idx),
            "frac": float(frac_off),
            "lw": lw,
            "mid": mid,
            "d_lw": d_lw,
            "d_mid": d_mid,
            "L0": float(load[0]),
            "L1": float(load[1]),
        }
        print(
            f"{seed:4d} {peak:4d} {half_idx:4d} {lw:3d} {mid:3d} "
            f"{d_lw:4d} {d_mid:5d} {float(load[0]):6.3f} {float(load[1]):6.3f}"
        )

    expect_half = {0: 5, 1: 5, 2: 4, 3: 6, 4: 5}
    expect_lw = {0: 0, 1: 0, 2: 1, 3: 0, 4: 0}
    expect_d_lw = {0: 5, 1: 5, 2: 3, 3: 6, 4: 5}
    expect_d_mid = {0: 2, 1: 2, 2: 3, 3: 1, 4: 2}
    for seed in range(5):
        assert int(by[seed]["peak"]) == 1
        assert int(by[seed]["half"]) == expect_half[seed]
        assert int(by[seed]["lw"]) == expect_lw[seed]
        assert int(by[seed]["mid"]) == 7
        assert int(by[seed]["d_lw"]) == expect_d_lw[seed]
        assert int(by[seed]["d_mid"]) == expect_d_mid[seed]
        # Half-life always finer than LW (and coarser than / equal-gap mid).
        assert int(by[seed]["half"]) > int(by[seed]["lw"])
        if seed == 2:
            # LW≠coarse singleton: ties mid proximity, does not beat it.
            assert int(by[seed]["d_lw"]) == int(by[seed]["d_mid"])
        else:
            assert int(by[seed]["d_lw"]) > int(by[seed]["d_mid"])

    # Seed2 is the only LW≠coarse cell; still not closer than mid to half-life.
    assert int(by[2]["lw"]) == 1
    assert all(int(by[s]["lw"]) == 0 for s in (0, 1, 3, 4))
    assert abs(float(by[2]["L0"]) - 0.614) < 0.05
    assert abs(float(by[2]["L1"]) - 1.562) < 0.05

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_seed3_std_short_block_half_vs_fractional_collapse() -> None:
    # EXPERIMENT (A6-T104): seed-3 standard-grid short persist block
    # (run_lengths[0]=3 ⇒ [0,2]) collapses mid≡two-thirds≡three-quarter to
    # idx1 (~8.83×E[τ]), but Phi half-life does *not* join that collapse —
    # peak+1 half-life lands at fine-end idx2 (~4.88×). Densify separates
    # mid/tt and moves half-life finer. Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert ScaleSearchConfig().halve_grid_steps is False

    dataset = make_hierarchical_gaussian(
        children_per_coarse=2, n_samples=600, ambient_dim=4, seed=3,
    )
    gt = dataset.ground_truth
    assert gt.expected_tau is not None
    tau_lo, tau_hi = gt.tau_grid_hint
    E = float(gt.expected_tau)
    by: dict[bool, dict[str, object]] = {}
    print("\nA6-T104 seed3 std short-block half vs mid≡tt≡tq collapse")
    header = (
        f"{'dense':>5s} {'run':>3s} {'peak':>4s} {'half':>4s} {'mid':>3s} "
        f"{'tt':>3s} {'tq':>3s} {'fine':>4s} {'h==mid':>6s}"
    )
    print(header)
    print("-" * len(header))
    for dense in (False, True):
        result = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=ScaleSearchConfig(
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=3,
                halve_grid_steps=dense,
                persistence=PersistenceConfig(resolve_within_interval="none"),
            ),
        )
        assert result.persistence_result is not None
        pr = result.persistence_result
        assert pr.tau_star_index is not None
        i_lo = int(pr.tau_star_index)
        run = int(pr.run_lengths[i_lo])
        i_hi = min(i_lo + run - 1, len(result.load_trace) - 1)
        phi = np.asarray(result.phi_trace, dtype=float)
        taus = np.asarray(result.tau_grid, dtype=float)
        finite = [
            idx for idx in range(i_lo, i_hi + 1) if np.isfinite(float(phi[idx]))
        ]
        # Peak over the accepted block (matches densify half-life convention).
        peak = max(finite, key=lambda i: float(phi[i]))
        phi_p = float(phi[peak])
        half_idx: int | None = None
        frac_off: float | None = None
        prev_r = 1.0
        for off in range(1, i_hi - peak + 1):
            r = float(phi[peak + off]) / phi_p
            if r <= 0.5:
                half_idx = peak + off
                t = (prev_r - 0.5) / (prev_r - r) if prev_r != r else 0.0
                frac_off = float(off - 1) + float(t)
                break
            prev_r = r
        assert half_idx is not None and frac_off is not None
        mid = int(_mid_interval_index(i_lo, i_hi))
        tt = int(_two_thirds_index(i_lo, i_hi))
        tq = int(_three_quarter_index(i_lo, i_hi))
        fine = int(i_hi)
        by[dense] = {
            "run": run,
            "i_lo": i_lo,
            "i_hi": i_hi,
            "peak": int(peak),
            "half": int(half_idx),
            "frac": float(frac_off),
            "mid": mid,
            "tt": tt,
            "tq": tq,
            "fine": fine,
            "tau_h_over_E": float(taus[half_idx]) / E,
            "tau_mid_over_E": float(taus[mid]) / E,
            "tau_fine_over_E": float(taus[fine]) / E,
        }
        print(
            f"{str(dense):>5s} {run:3d} {peak:4d} {half_idx:4d} {mid:3d} "
            f"{tt:3d} {tq:3d} {fine:4d} {str(half_idx == mid):>6s}"
        )

    # Standard short block: mid≡tt≡tq collapse; half ≡ fine ≢ mid.
    std = by[False]
    assert int(std["run"]) == 3
    assert int(std["i_lo"]) == 0 and int(std["i_hi"]) == 2
    assert int(std["peak"]) == 1
    assert int(std["mid"]) == int(std["tt"]) == int(std["tq"]) == 1
    assert int(std["half"]) == int(std["fine"]) == 2
    assert int(std["half"]) != int(std["mid"])
    assert abs(float(std["tau_mid_over_E"]) - 8.833) < 0.05
    assert abs(float(std["tau_h_over_E"]) - 4.876) < 0.05
    assert abs(float(std["tau_fine_over_E"]) - 4.876) < 0.05
    assert abs(float(std["frac"]) - 0.876) < 0.05

    # Densify expands block; mid/tt separate; half moves finer than std fine.
    dens = by[True]
    assert int(dens["run"]) == 16
    assert int(dens["peak"]) == 1
    assert int(dens["half"]) == 6
    assert int(dens["mid"]) == 7
    assert int(dens["tt"]) == 10
    assert int(dens["half"]) != int(dens["mid"])
    assert int(dens["mid"]) != int(dens["tt"])
    assert int(dens["half"]) > int(std["half"])

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert ScaleSearchConfig().halve_grid_steps is False


def test_phi_half_life_x_lc_hybrid_proximity_thr030_dense() -> None:
    # EXPERIMENT (A6-T106): thr=0.30 densified seeds0..4 — Phi half-life index
    # proximity to load_crossover hybrid vs mid / LW. Pins that LC-hybrid stays
    # at the coarse-end arbiter on every accept (including the LW≠coarse
    # seed2 singleton), so d(half,LC) ≥ d(half,LW) and LC is always farther
    # from half-life than mid. Half-life proximity does not favor LC-hybrid.
    # Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    by: dict[int, dict[str, object]] = {}
    print("\nA6-T106 thr0.30 densified Phi half-life × LC-hybrid proximity")
    header = (
        f"{'seed':>4s} {'peak':>4s} {'half':>4s} {'LC':>3s} {'LW':>3s} "
        f"{'mid':>3s} {'d_LC':>4s} {'d_LW':>4s} {'d_mid':>5s}"
    )
    print(header)
    print("-" * len(header))
    for seed in range(5):
        dataset = make_hierarchical_gaussian(
            children_per_coarse=2, n_samples=600, ambient_dim=4, seed=seed,
        )
        gt = dataset.ground_truth
        assert gt.expected_tau is not None
        tau_lo, tau_hi = gt.tau_grid_hint
        result = run_scale_search(
            dataset.points,
            dim=gt.ambient_dim,
            config=ScaleSearchConfig(
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
                    min_equilibrium_epochs=2, max_epochs=12
                ),
                seed=seed,
                halve_grid_steps=True,
                persistence=PersistenceConfig(
                    resolve_within_interval="none",
                    densify_overlap_recover="lower_threshold",
                    densify_overlap_recover_threshold=0.30,
                ),
            ),
        )
        assert result.persistence_result is not None
        pr = result.persistence_result
        assert pr.tau_star_index == 0
        assert int(pr.run_lengths[0]) == 16
        load = np.asarray(result.load_trace, dtype=float)
        phi = np.asarray(result.phi_trace, dtype=float)
        stab = list(result.stabilized_flags)
        i_lo, i_hi = 0, 15
        finite = [
            idx for idx in range(i_lo, i_hi + 1) if np.isfinite(float(phi[idx]))
        ]
        peak = max(finite, key=lambda i: float(phi[i]))
        phi_p = float(phi[peak])
        half_idx: int | None = None
        frac_off: float | None = None
        prev_r = 1.0
        for off in range(1, i_hi - peak + 1):
            r = float(phi[peak + off]) / phi_p
            if r <= 0.5:
                half_idx = peak + off
                t = (prev_r - 0.5) / (prev_r - r) if prev_r != r else 0.0
                frac_off = float(off - 1) + float(t)
                break
            prev_r = r
        assert half_idx is not None and frac_off is not None
        lc = int(
            _resolve_persistence_tau_index(
                pr,
                load,
                stab,
                PersistenceConfig(resolve_within_interval="load_crossover"),
            )
        )
        lw = int(_load_weighted_index(i_lo, i_hi, load))
        mid = int(_mid_interval_index(i_lo, i_hi))
        d_lc = abs(int(half_idx) - lc)
        d_lw = abs(int(half_idx) - lw)
        d_mid = abs(int(half_idx) - mid)
        by[seed] = {
            "peak": int(peak),
            "half": int(half_idx),
            "frac": float(frac_off),
            "lc": lc,
            "lw": lw,
            "mid": mid,
            "d_lc": d_lc,
            "d_lw": d_lw,
            "d_mid": d_mid,
        }
        print(
            f"{seed:4d} {peak:4d} {half_idx:4d} {lc:3d} {lw:3d} "
            f"{mid:3d} {d_lc:4d} {d_lw:4d} {d_mid:5d}"
        )

    expect_half = {0: 5, 1: 5, 2: 4, 3: 6, 4: 5}
    expect_d_lc = {0: 5, 1: 5, 2: 4, 3: 6, 4: 5}
    expect_d_lw = {0: 5, 1: 5, 2: 3, 3: 6, 4: 5}
    expect_d_mid = {0: 2, 1: 2, 2: 3, 3: 1, 4: 2}
    for seed in range(5):
        assert int(by[seed]["peak"]) == 1
        assert int(by[seed]["half"]) == expect_half[seed]
        # LC-hybrid ≡ coarse-end on every densified thr0.30 accept (T78/T85).
        assert int(by[seed]["lc"]) == 0
        assert int(by[seed]["mid"]) == 7
        assert int(by[seed]["d_lc"]) == expect_d_lc[seed]
        assert int(by[seed]["d_lw"]) == expect_d_lw[seed]
        assert int(by[seed]["d_mid"]) == expect_d_mid[seed]
        assert int(by[seed]["half"]) > int(by[seed]["lc"])
        # LC never closer to half-life than LW; strictly farther on seed2.
        assert int(by[seed]["d_lc"]) >= int(by[seed]["d_lw"])
        # LC always farther from half-life than mid.
        assert int(by[seed]["d_lc"]) > int(by[seed]["d_mid"])

    # Seed2 LW≠coarse singleton: LC stays coarse while LW steps to idx1.
    assert int(by[2]["lw"]) == 1
    assert int(by[2]["lc"]) == 0
    assert int(by[2]["d_lc"]) == int(by[2]["d_lw"]) + 1
    assert all(int(by[s]["lw"]) == 0 for s in (0, 1, 3, 4))
    assert all(int(by[s]["lc"]) == int(by[s]["lw"]) for s in (0, 1, 3, 4))

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


def test_multiseed_std_half_eq_peak_plus_one_thr_pin_table() -> None:
    # EXPERIMENT (A6-T107): multi-seed standard-grid Phi half-life ≡ peak+1
    # pin table across densify-recover thr ∈ {0.30, 0.35, 0.40}. Pins that
    # on the coarse geometric grid every shared accept collapses half-life to
    # the next log-step (peak=1 → half=2; τ_half/τ_peak≈0.55), thr-invariant,
    # while seed2 remains reject. Defaults stay off (halve_grid_steps False;
    # recover lever only for thr probe, not acceptance default).
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False

    by: dict[tuple[float, int], dict[str, object]] = {}
    print("\nA6-T107 multi-seed std half≡peak+1 thr pin table")
    header = (
        f"{'thr':>5s} {'seed':>4s} {'acc':>3s} {'n':>3s} {'peak':>4s} "
        f"{'half':>4s} {'off':>3s} {'frac':>7s} {'tau_r':>7s}"
    )
    print(header)
    print("-" * len(header))
    for thr in (0.30, 0.35, 0.40):
        for seed in range(5):
            dataset = make_hierarchical_gaussian(
                children_per_coarse=2,
                n_samples=600,
                ambient_dim=4,
                seed=seed,
            )
            gt = dataset.ground_truth
            assert gt.expected_tau is not None
            tau_lo, tau_hi = gt.tau_grid_hint
            result = run_scale_search(
                dataset.points,
                dim=gt.ambient_dim,
                config=ScaleSearchConfig(
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
                        min_equilibrium_epochs=2, max_epochs=12
                    ),
                    seed=seed,
                    halve_grid_steps=False,
                    persistence=PersistenceConfig(
                        resolve_within_interval="none",
                        densify_overlap_recover="lower_threshold",
                        densify_overlap_recover_threshold=thr,
                    ),
                ),
            )
            assert result.persistence_result is not None
            pr = result.persistence_result
            accept = pr.tau_star_index is not None
            n = len(result.phi_trace)
            row: dict[str, object] = {
                "accept": int(accept),
                "n": int(n),
                "run0": int(pr.run_lengths[0]),
            }
            if accept:
                assert pr.tau_star_index == 0
                phi = np.asarray(result.phi_trace, dtype=float)
                taus = np.asarray(result.tau_grid, dtype=float)
                i_hi = n - 1
                finite = [
                    idx
                    for idx in range(0, i_hi + 1)
                    if np.isfinite(float(phi[idx]))
                ]
                peak = max(finite, key=lambda i: float(phi[i]))
                phi_p = float(phi[peak])
                half_idx: int | None = None
                frac_off: float | None = None
                prev_r = 1.0
                for off in range(1, i_hi - peak + 1):
                    r = float(phi[peak + off]) / phi_p
                    if r <= 0.5:
                        half_idx = peak + off
                        t = (
                            (prev_r - 0.5) / (prev_r - r)
                            if prev_r != r
                            else 0.0
                        )
                        frac_off = float(off - 1) + float(t)
                        break
                    prev_r = r
                assert half_idx is not None and frac_off is not None
                tau_r = float(taus[half_idx]) / float(taus[peak])
                row.update(
                    {
                        "peak": int(peak),
                        "half": int(half_idx),
                        "off": int(half_idx - peak),
                        "frac": float(frac_off),
                        "tau_r": float(tau_r),
                    }
                )
                print(
                    f"{thr:5.2f} {seed:4d} {1:3d} {n:3d} {peak:4d} "
                    f"{half_idx:4d} {half_idx - peak:3d} {frac_off:7.3f} "
                    f"{tau_r:7.4f}"
                )
            else:
                print(
                    f"{thr:5.2f} {seed:4d} {0:3d} {n:3d} {'-':>4} "
                    f"{'-':>4} {'-':>3} {'-':>7} {'-':>7}"
                )
            by[(thr, seed)] = row

    # Standard accept set {0,1,3,4} is thr-invariant; seed2 rejects.
    for thr in (0.30, 0.35, 0.40):
        for seed in (0, 1, 3, 4):
            assert by[(thr, seed)]["accept"] == 1
            assert by[(thr, seed)]["n"] == 8
        assert by[(thr, 2)]["accept"] == 0

    # Half-life ≡ peak+1 on every shared accept; thr-invariant frac/tau_r.
    expect_frac = {0: 0.997, 1: 0.941, 3: 0.876, 4: 0.950}
    for thr in (0.30, 0.35, 0.40):
        for seed, want in expect_frac.items():
            assert int(by[(thr, seed)]["peak"]) == 1
            assert int(by[(thr, seed)]["half"]) == 2
            assert int(by[(thr, seed)]["off"]) == 1
            assert abs(float(by[(thr, seed)]["frac"]) - want) < 0.05
            assert abs(float(by[(thr, seed)]["tau_r"]) - 0.5520) < 0.02

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False


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
