"""Phi half-life / stab-skip / LC-straddle densified persistence pins.

Moved from test_scale_search_persistence.py (A6-T4). Auto-marked slow
via conftest _SLOW_STAGE1_MODULES; assertions preserved verbatim.
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
    # defined without a persistence block. Densify doubles grid length;
    # circle peak/half indices still double, swiss does not (continuous
    # sheet moves the coarse peak off the mid-grid 2× correspondence).
    # Defaults stay off.
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
        # Continuous Swiss sheet (issue #48): the old 48×8 kernel lattice was
        # a comb of ridges, so phi peaked mid-grid (coarse peak=4, half=5,
        # frac=0.960; dense peak=8, half=10, frac=1.935).  Area-uniform
        # surface density moves the coarse peak to the fine end.
        ("swiss", False): {"n": 8, "peak": 1, "half": 3, "frac": 1.149},
        ("swiss", True): {"n": 16, "peak": 7, "half": 10, "frac": 2.351},
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

    # Densify doubles grid length. Circle peak/half indices still double
    # (mid-grid peak). Swiss no longer does: the continuous sheet moves the
    # coarse phi peak from mid-grid index 4 to fine-end index 1, so 2×coarse
    # is not a dense-grid correspondence (issue #48).
    for name in ("circle", "swiss"):
        assert int(by[(name, True)]["n"]) == 2 * int(by[(name, False)]["n"])
    assert int(by[("circle", True)]["peak"]) == 2 * int(by[("circle", False)]["peak"])
    assert int(by[("circle", True)]["half"]) == 2 * int(by[("circle", False)]["half"])

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

def test_phi_half_life_x_stab_only_phi_argmax_proximity_thr030_dense() -> None:
    # EXPERIMENT (A6-T109): thr=0.30 densified seeds0..4 — Phi half-life index
    # proximity to stab-only Phi-argmax (T82) vs mid. Pins that mid remains
    # closer on seeds 0/1/3/4; only the seed2 peak-unstabilized singleton has
    # d(half,sArg) < d(half,mid). Half-life proximity does not favor
    # stab-only Phi-argmax as a general landing. Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    by: dict[int, dict[str, object]] = {}
    print("\nA6-T109 thr0.30 densified Phi half-life × stab-only Phi-argmax")
    header = (
        f"{'seed':>4s} {'peak':>4s} {'half':>4s} {'sArg':>4s} {'mid':>3s} "
        f"{'d_s':>3s} {'d_mid':>5s} {'stabP':>5s}"
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
        phi = np.asarray(result.phi_trace, dtype=float)
        stab = list(result.stabilized_flags)
        i_lo, i_hi = 0, 15
        finite = [
            idx for idx in range(i_lo, i_hi + 1) if np.isfinite(float(phi[idx]))
        ]
        peak = max(finite, key=lambda i: float(phi[i]))
        phi_p = float(phi[peak])
        half_idx: int | None = None
        prev_r = 1.0
        for off in range(1, i_hi - peak + 1):
            r = float(phi[peak + off]) / phi_p
            if r <= 0.5:
                half_idx = peak + off
                break
            prev_r = r
        assert half_idx is not None
        stab_only = [idx for idx in finite if stab[idx]]
        assert len(stab_only) >= 1
        sarg = max(stab_only, key=lambda i: float(phi[i]))
        mid = int(_mid_interval_index(i_lo, i_hi))
        d_s = abs(int(half_idx) - int(sarg))
        d_mid = abs(int(half_idx) - mid)
        by[seed] = {
            "peak": int(peak),
            "half": int(half_idx),
            "sarg": int(sarg),
            "mid": mid,
            "d_s": d_s,
            "d_mid": d_mid,
            "stab_peak": bool(stab[peak]),
        }
        print(
            f"{seed:4d} {peak:4d} {half_idx:4d} {sarg:4d} {mid:3d} "
            f"{d_s:3d} {d_mid:5d} {str(bool(stab[peak])):5s}"
        )

    expect_half = {0: 5, 1: 5, 2: 4, 3: 6, 4: 5}
    expect_sarg = {0: 1, 1: 1, 2: 2, 3: 1, 4: 1}
    expect_d_s = {0: 4, 1: 4, 2: 2, 3: 5, 4: 4}
    expect_d_mid = {0: 2, 1: 2, 2: 3, 3: 1, 4: 2}
    for seed in range(5):
        assert int(by[seed]["peak"]) == 1
        assert int(by[seed]["half"]) == expect_half[seed]
        assert int(by[seed]["sarg"]) == expect_sarg[seed]
        assert int(by[seed]["mid"]) == 7
        assert int(by[seed]["d_s"]) == expect_d_s[seed]
        assert int(by[seed]["d_mid"]) == expect_d_mid[seed]

    # Seeds 0/1/3/4: peak stabilized ⇒ sArg≡peak; mid closer to half-life.
    for seed in (0, 1, 3, 4):
        assert bool(by[seed]["stab_peak"]) is True
        assert int(by[seed]["sarg"]) == int(by[seed]["peak"])
        assert int(by[seed]["d_mid"]) < int(by[seed]["d_s"])

    # Seed2 singleton: peak unstabilized ⇒ sArg=2; only cell where sArg
    # beats mid on half-life proximity.
    assert bool(by[2]["stab_peak"]) is False
    assert int(by[2]["sarg"]) == 2
    assert int(by[2]["d_s"]) < int(by[2]["d_mid"])

    # Universal negative: stab-only Phi-argmax is not closer than mid on
    # the majority of densified thr0.30 accepts.
    assert sum(int(by[s]["d_s"]) < int(by[s]["d_mid"]) for s in range(5)) == 1

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False

def test_lc_straddle_endpoint_vs_half_life_margins_thr030_dense() -> None:
    # EXPERIMENT (A6-T110): thr=0.30 densified seeds0..4 — LC load-straddle
    # endpoints vs Phi half-life. Pins that the fine straddle endpoint is
    # always closer to half-life than the coarse endpoint, yet |L-1| margins
    # still prefer coarse (so LC≡0). Relative to mid, only seed2's widened
    # 0↔2 straddle has fine closer to half-life; mid wins otherwise.
    # Defaults stay off.
    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False
    assert _WITHIN_INTERVAL_LOAD_SCREEN_MIN == 0.5

    by: dict[int, dict[str, object]] = {}
    print("\nA6-T110 thr0.30 densified LC straddle endpoints vs half-life")
    header = (
        f"{'seed':>4s} {'half':>4s} {'c':>2s} {'f':>2s} {'LC':>3s} {'mid':>3s} "
        f"{'d_c':>3s} {'d_f':>3s} {'d_mid':>5s} {'m0':>6s} {'mf':>6s}"
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
        for off in range(1, i_hi - peak + 1):
            r = float(phi[peak + off]) / phi_p
            if r <= 0.5:
                half_idx = peak + off
                break
        assert half_idx is not None
        eligible = [
            idx
            for idx in range(i_lo, i_hi + 1)
            if stab[idx] and np.isfinite(float(load[idx]))
        ]
        coarse_ep: int | None = None
        fine_ep: int | None = None
        for a, b in zip(eligible[:-1], eligible[1:]):
            if float(load[a]) <= 1.0 < float(load[b]):
                coarse_ep, fine_ep = int(a), int(b)
                break
        assert coarse_ep is not None and fine_ep is not None
        lc = int(
            _resolve_persistence_tau_index(
                pr,
                load,
                stab,
                PersistenceConfig(resolve_within_interval="load_crossover"),
            )
        )
        mid = int(_mid_interval_index(i_lo, i_hi))
        m0 = abs(float(load[coarse_ep]) - 1.0)
        mf = abs(float(load[fine_ep]) - 1.0)
        d_c = abs(int(half_idx) - int(coarse_ep))
        d_f = abs(int(half_idx) - int(fine_ep))
        d_mid = abs(int(half_idx) - mid)
        by[seed] = {
            "peak": int(peak),
            "half": int(half_idx),
            "coarse": int(coarse_ep),
            "fine": int(fine_ep),
            "lc": lc,
            "mid": mid,
            "d_c": d_c,
            "d_f": d_f,
            "d_mid": d_mid,
            "m0": m0,
            "mf": mf,
            "stab1": bool(stab[1]),
        }
        print(
            f"{seed:4d} {half_idx:4d} {coarse_ep:2d} {fine_ep:2d} {lc:3d} "
            f"{mid:3d} {d_c:3d} {d_f:3d} {d_mid:5d} {m0:6.3f} {mf:6.3f}"
        )

    expect_half = {0: 5, 1: 5, 2: 4, 3: 6, 4: 5}
    expect_fine = {0: 1, 1: 1, 2: 2, 3: 1, 4: 1}
    expect_d_f = {0: 4, 1: 4, 2: 2, 3: 5, 4: 4}
    expect_d_mid = {0: 2, 1: 2, 2: 3, 3: 1, 4: 2}
    expect_m0 = {0: 0.2685, 1: 0.3500, 2: 0.3858, 3: 0.2784, 4: 0.3078}
    expect_mf = {0: 0.764, 1: 0.635, 2: 1.059, 3: 0.785, 4: 0.903}
    for seed in range(5):
        assert int(by[seed]["peak"]) == 1
        assert int(by[seed]["half"]) == expect_half[seed]
        assert int(by[seed]["coarse"]) == 0
        assert int(by[seed]["fine"]) == expect_fine[seed]
        assert int(by[seed]["lc"]) == 0
        assert int(by[seed]["mid"]) == 7
        assert int(by[seed]["d_c"]) == expect_half[seed]
        assert int(by[seed]["d_f"]) == expect_d_f[seed]
        assert int(by[seed]["d_mid"]) == expect_d_mid[seed]
        # Fine endpoint always closer to half-life than coarse.
        assert int(by[seed]["d_f"]) < int(by[seed]["d_c"])
        # But |L-1| still prefers coarse ⇒ LC stays at coarse-end.
        assert float(by[seed]["m0"]) < float(by[seed]["mf"])
        assert abs(float(by[seed]["m0"]) - expect_m0[seed]) < 0.02
        assert abs(float(by[seed]["mf"]) - expect_mf[seed]) < 0.03

    # Seed2 widens straddle to 0↔2 (peak unstabilized); others 0↔1.
    assert bool(by[2]["stab1"]) is False
    assert int(by[2]["fine"]) == 2
    for seed in (0, 1, 3, 4):
        assert bool(by[seed]["stab1"]) is True
        assert int(by[seed]["fine"]) == 1
        assert int(by[seed]["d_mid"]) < int(by[seed]["d_f"])

    # Only seed2: rejected fine endpoint beats mid on half-life proximity.
    assert int(by[2]["d_f"]) < int(by[2]["d_mid"])
    assert sum(int(by[s]["d_f"]) < int(by[s]["d_mid"]) for s in range(5)) == 1

    assert PersistenceConfig().resolve_within_interval == "none"
    assert PersistenceConfig().densify_overlap_recover == "none"
    assert PersistenceConfig().densify_overlap_recover_threshold is None
    assert ScaleSearchConfig().halve_grid_steps is False

