"""Stage-1 seed sweep on denser fitted linked-tori (#41 / A4-T44).

A4-T42 found dataset seed=2 at ``n_per_torus=500`` / ``max_nodes=256`` yields
**both** tori Betti ``(1,2,0)`` under fixed Stage-1 seed=77 (b2 missing).
This harness fixes that denser dataset recipe and sweeps ``ScaleSearchConfig.seed``
to ask whether Stage-1 stochasticity alone unlocks ``b2`` / full ``(1,2,1)``.

Evidence-gathering only — does **not** flip ``@awaiting`` or SI defaults.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.linked_tori import make_linked_tori
from tests.metrics.persistent_homology import (
    FILTRATION_MULTIPLIER,
    nearest_data_labels,
    run_per_region_ph,
    sigma_star_from_tau,
)

EXPECTED_TORI: tuple[int, ...] = (1, 2, 1)
N_PER_TORUS: int = 500
MAX_NODES: int = 256
DATASET_SEED: int = 2
# Include baseline 77; probe nearby / distant Stage-1 seeds.
STAGE1_SEEDS: tuple[int, ...] = (0, 1, 7, 21, 42, 77, 99, 123, 256, 512)


@dataclass(frozen=True)
class Stage1SeedSweepRow:
    stage1_seed: int
    n_signal: int
    n_per_torus: dict[int, int]
    sigma_star: float
    all_match: bool | None
    betti: dict[int, tuple[int, ...]]
    max_b1: int
    max_b2: int
    any_b1_ge_2: bool
    any_b2: bool
    any_full_torus: bool
    both_tori_full: bool


@dataclass(frozen=True)
class LinkedToriStage1SeedSweepBundle:
    dataset_seed: int
    n_per_torus_data: int
    max_nodes: int
    stage1_seeds: tuple[int, ...]
    rows: tuple[Stage1SeedSweepRow, ...]
    n_seeds_with_b1_ge_2: int
    n_seeds_with_b2: int
    n_seeds_full_recover: int
    any_full_recover: bool
    any_b2: bool
    any_b1_ge_2: bool
    max_b1: int
    max_b2: int
    b2_cells: tuple[tuple[int, int, tuple[int, ...]], ...]
    full_cells: tuple[tuple[int, int, tuple[int, ...]], ...]
    baseline_seed77_both_partial: bool


@pytest.fixture(scope="module")
def linked_tori_stage1_seed_sweep_bundle() -> LinkedToriStage1SeedSweepBundle:
    """Fit denser seed=2 dataset across Stage-1 seeds; fixed_threshold SI fine."""
    dataset = make_linked_tori(
        n_per_torus=N_PER_TORUS,
        major_radius=2.0,
        minor_radius=0.5,
        noise=0.02,
        tissue_fraction=0.03,
        seed=DATASET_SEED,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint

    rows: list[Stage1SeedSweepRow] = []
    b2_cells: list[tuple[int, int, tuple[int, ...]]] = []
    full_cells: list[tuple[int, int, tuple[int, ...]]] = []

    for stage1_seed in STAGE1_SEEDS:
        config = ScaleSearchConfig(
            tau_min=tau_lo,
            tau_max=tau_hi,
            max_grid_points=8,
            k=8,
            n_seeds=8,
            max_nodes=MAX_NODES,
            ann_backend="naive",
            stabilization=StabilizationConfig(
                min_equilibrium_epochs=3, max_epochs=15,
            ),
            seed=int(stage1_seed),
        )
        result = run_scale_search(
            dataset.points, dim=gt.ambient_dim, config=config,
        )
        pos = result.scaffold_at_star.node_positions()
        sigma = sigma_star_from_tau(result.tau_star)
        node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
        signal_mask = np.isin(node_labels, [0, 1])
        signal_pos = pos[signal_mask]
        signal_labs = node_labels[signal_mask]
        ph = run_per_region_ph(
            signal_pos,
            signal_labs,
            sigma,
            scenario=f"linked_tori_stage1_seed_sweep_s{stage1_seed}",
            include_labels=[0, 1],
            reading="fixed_threshold",
            max_dim=2,
            filtration_mult=FILTRATION_MULTIPLIER,
            expected_betti=EXPECTED_TORI,
        )
        betti = {
            int(r.region_id): tuple(int(x) for x in r.betti) for r in ph.reports
        }
        max_b1 = max((int(b[1]) for b in betti.values() if len(b) > 1), default=0)
        max_b2 = max((int(b[2]) for b in betti.values() if len(b) > 2), default=0)
        any_b1 = any(len(b) > 1 and int(b[1]) >= 2 for b in betti.values())
        any_b2 = any(len(b) > 2 and int(b[2]) >= 1 for b in betti.values())
        any_full = any(b == EXPECTED_TORI for b in betti.values())
        both_full = all(
            betti.get(lab) == EXPECTED_TORI for lab in (0, 1)
        ) and set(betti.keys()) >= {0, 1}
        rows.append(
            Stage1SeedSweepRow(
                stage1_seed=int(stage1_seed),
                n_signal=int(signal_pos.shape[0]),
                n_per_torus={
                    int(lab): int(np.sum(signal_labs == lab)) for lab in (0, 1)
                },
                sigma_star=float(sigma),
                all_match=ph.all_match,
                betti=betti,
                max_b1=max_b1,
                max_b2=max_b2,
                any_b1_ge_2=any_b1,
                any_b2=any_b2,
                any_full_torus=any_full,
                both_tori_full=both_full,
            )
        )
        for lab, b in betti.items():
            if b == EXPECTED_TORI:
                full_cells.append((int(stage1_seed), int(lab), tuple(b)))
            if len(b) > 2 and int(b[2]) >= 1:
                b2_cells.append((int(stage1_seed), int(lab), tuple(b)))

    n_b1 = sum(1 for r in rows if r.any_b1_ge_2)
    n_b2 = sum(1 for r in rows if r.any_b2)
    n_full = sum(1 for r in rows if r.all_match is True)
    baseline = next(r for r in rows if r.stage1_seed == 77)
    baseline_both_partial = (
        baseline.betti.get(0) == (1, 2, 0)
        and baseline.betti.get(1) == (1, 2, 0)
    )
    return LinkedToriStage1SeedSweepBundle(
        dataset_seed=DATASET_SEED,
        n_per_torus_data=N_PER_TORUS,
        max_nodes=MAX_NODES,
        stage1_seeds=STAGE1_SEEDS,
        rows=tuple(rows),
        n_seeds_with_b1_ge_2=n_b1,
        n_seeds_with_b2=n_b2,
        n_seeds_full_recover=n_full,
        any_full_recover=n_full > 0,
        any_b2=n_b2 > 0,
        any_b1_ge_2=n_b1 > 0,
        max_b1=max((r.max_b1 for r in rows), default=0),
        max_b2=max((r.max_b2 for r in rows), default=0),
        b2_cells=tuple(b2_cells),
        full_cells=tuple(full_cells),
        baseline_seed77_both_partial=baseline_both_partial,
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_stage1_seed_sweep_harness_lands(
    linked_tori_stage1_seed_sweep_bundle,
) -> None:
    """Stage-1 seed sweep denser ladder lands; SI fine mult untouched."""
    bundle = linked_tori_stage1_seed_sweep_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.n_per_torus_data == N_PER_TORUS
    assert bundle.max_nodes == MAX_NODES
    assert bundle.stage1_seeds == STAGE1_SEEDS
    assert len(bundle.rows) == len(STAGE1_SEEDS)
    assert [r.stage1_seed for r in bundle.rows] == list(STAGE1_SEEDS)
    assert all(r.n_signal > 0 for r in bundle.rows)
    assert all(r.sigma_star > 0.0 for r in bundle.rows)
    # Baseline seed=77 should still show T42 both-tori partial interlocking.
    assert bundle.baseline_seed77_both_partial


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_stage1_seed_sweep_documents_gap(
    linked_tori_stage1_seed_sweep_bundle,
) -> None:
    """Document Stage-1 seed vs b2/full recovery; never flip awaiting.

    Soft: any seed with full ``(1,2,1)`` or any ``b2≥1`` is proposal-path
    evidence. Otherwise keep explicit ``max_b2 == 0``. Stage-1 seed alone ≠
    SI recovery.
    """
    bundle = linked_tori_stage1_seed_sweep_bundle
    if bundle.any_full_recover or bundle.any_b2:
        assert FILTRATION_MULTIPLIER == 1.5
        if bundle.any_full_recover:
            assert len(bundle.full_cells) >= 1
            assert bundle.n_seeds_full_recover >= 1
            assert bundle.max_b2 >= 1
        if bundle.any_b2:
            assert len(bundle.b2_cells) >= 1
            assert bundle.n_seeds_with_b2 >= 1
            assert bundle.max_b2 >= 1
    else:
        assert bundle.any_full_recover is False
        assert bundle.any_b2 is False
        assert bundle.max_b2 == 0
        assert bundle.n_seeds_with_b2 == 0
        assert bundle.n_seeds_full_recover == 0
        assert bundle.full_cells == ()
        assert bundle.b2_cells == ()
        # Partial interlocking may still appear on some Stage-1 seeds.
        assert bundle.any_b1_ge_2 or bundle.baseline_seed77_both_partial
