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
    # never refines past the default (``none``) landing across seeds. When an
    # accepted persist split exists, within-block loads jump from
    # L(i_lo)~0.6–0.7 to L≫1 at the next grid point, so argmin|log L| among
    # L≥0.5 is always i_lo (coarse-end). When no split is accepted,
    # within-interval modes are no-ops (controller LC fallback) and weighted
    # still matches ``none``. Dense densification does not help. Do not flip
    # defaults.
    seeds = (0, 1, 2, 3, 4)
    rows: list[dict[str, float | int | bool | str]] = []
    n_with_split = 0
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
            # Always identical to default landing (coarse-end or LC fallback).
            assert weighted.peak_index == none.peak_index
            assert weighted.tau_star == none.tau_star
            i_lo_raw = none.persistence_result.tau_star_index
            load_w = float(weighted.load_trace[weighted.peak_index])
            has_split = i_lo_raw is not None
            if has_split:
                n_with_split += 1
                i_lo = int(i_lo_raw)
                assert weighted.peak_index == i_lo
                assert load_w >= _WITHIN_INTERVAL_LOAD_SCREEN_MIN
                assert 0.5 <= load_w < 1.0
            rows.append(
                {
                    "seed": seed,
                    "dense": dense,
                    "split": has_split,
                    "weighted_idx": int(weighted.peak_index),
                    "load": load_w,
                    "tau_over_expected": float(
                        weighted.tau_star / gt.expected_tau
                    ),
                }
            )

    # At least seed-0 std/dense retain an accepted split (fixture sanity).
    assert n_with_split >= 2
    header = (
        f"{'seed':4s} {'dense':5s} {'split':5s} {'idx':>3s} "
        f"{'load':>7s} {'tau*/E':>8s}"
    )
    print("\nA6-T49 load_weighted ≡ none/coarse across hierarchy seeds")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{int(row['seed']):4d} {str(row['dense']):5s} "
            f"{str(row['split']):5s} {int(row['weighted_idx']):3d} "
            f"{float(row['load']):7.3f} "
            f"{float(row['tau_over_expected']):8.3f}"
        )
    assert PersistenceConfig().resolve_within_interval == "none"
    assert ScaleSearchConfig().halve_grid_steps is False


def test_dense_ranking_flip_is_seed_fragile() -> None:
    # Throughput probe (A6-T49): seed-0 densify flip (2/3≈1.00× beats 3q) is
    # not universal. Seed 4 matches seed 0 on the *standard* grid (3q closer)
    # but loses the accepted persist split under halve_grid_steps, so all
    # fractional modes fall back identically (coarse LC). Do not flip default.
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
    modes = ("none", "two_thirds_interval", "three_quarter_interval", "mid_interval")
    by: dict[tuple[bool, str], dict[str, float | int | bool]] = {}
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
            i_lo_raw = result.persistence_result.tau_star_index
            by[(dense, mode)] = {
                "peak_index": int(result.peak_index),
                "tau_over_expected": float(result.tau_star / gt.expected_tau),
                "has_split": i_lo_raw is not None,
            }

    # Standard grid seed-4: accepted split + same 3q-closest pattern as seed-0.
    assert by[(False, "none")]["has_split"] is True
    err_tt = abs(float(by[(False, "two_thirds_interval")]["tau_over_expected"]) - 1.0)
    err_tq = abs(
        float(by[(False, "three_quarter_interval")]["tau_over_expected"]) - 1.0
    )
    assert err_tq < err_tt
    assert abs(float(by[(False, "three_quarter_interval")]["tau_over_expected"]) - 0.82) < 0.05
    assert abs(float(by[(False, "two_thirds_interval")]["tau_over_expected"]) - 1.49) < 0.05

    # Dense seed-4: no accepted persist split ⇒ all within-interval modes
    # identical to ``none`` (LC fallback); densify flip from seed-0 does not
    # transfer.
    assert by[(True, "none")]["has_split"] is False
    none_idx = int(by[(True, "none")]["peak_index"])
    none_ratio = float(by[(True, "none")]["tau_over_expected"])
    for mode in modes:
        assert int(by[(True, mode)]["peak_index"]) == none_idx
        assert float(by[(True, mode)]["tau_over_expected"]) == none_ratio
    assert none_ratio > 8.0

    print(
        "\nA6-T49 seed-4 ranking fragility: "
        f"std 2/3={by[(False, 'two_thirds_interval')]['tau_over_expected']:.3f} "
        f"3q={by[(False, 'three_quarter_interval')]['tau_over_expected']:.3f}; "
        f"dense no-split all→idx={none_idx} tau*/E={none_ratio:.3f}"
    )
    assert PersistenceConfig().resolve_within_interval == "none"
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
