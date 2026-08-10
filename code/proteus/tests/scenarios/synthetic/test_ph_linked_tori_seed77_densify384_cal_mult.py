"""Densify max_nodes=384 × lifetime×cal-mult on seed77 (#41 / A4-T62-followon).

A4-T50: seed77 densify ladder keeps both-partial at 256/512 but **384 regresses**
to ``(1,1,0)`` (max_b2=0). A4-T59: primary/mild hollow at 384 does not restore
both-partial or unlock void Betti. A4-T49: hollow×lifetime×cal-mult on denser256
cleaned neither dirty-b2 nor full ``(1,2,1)``.

This harness freezes densify ``max_nodes=384`` and crosses
``lifetime_frac × filtration_mult`` on the **signal** cloud (no hollow — T59
showed hollow is a no-op / regressor at mid-regime), asking whether cal-mult
alone recovers ``b2`` or clean ``(1,2,1)`` at the mid-density scaffold.

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
    DEFAULT_LIFETIME_FRAC,
    FILTRATION_MULTIPLIER,
    format_lifetime_mult_grid_table,
    nearest_data_labels,
    run_per_region_ph,
    sigma_star_from_tau,
    sweep_lifetime_mult_grid_per_region,
)

EXPECTED_TORI: tuple[int, ...] = (1, 2, 1)
N_PER_TORUS: int = 500
MAX_NODES: int = 384
DATASET_SEED: int = 2
STAGE1_SEED: int = 77
FRAC_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MULT_GRID: tuple[float, ...] = (1.5, 3.0, 6.0, 8.0)


def _is_clean_b2(betti: tuple[int, ...]) -> bool:
    return (
        len(betti) > 2
        and int(betti[0]) == 1
        and int(betti[1]) >= 2
        and int(betti[2]) >= 1
    )


def _is_dirty_b2(betti: tuple[int, ...]) -> bool:
    return (
        len(betti) > 2
        and int(betti[2]) >= 1
        and not _is_clean_b2(betti)
    )


@dataclass(frozen=True)
class Seed77Densify384CalMultBundle:
    dataset_seed: int
    stage1_seed: int
    max_nodes: int
    n_signal: int
    sigma_star: float
    signal_fixed_betti: dict[int, tuple[int, ...]]
    signal_both_partial: bool
    signal_max_b1: int
    signal_max_b2: int
    fracs: tuple[float, ...]
    mults: tuple[float, ...]
    table: str
    any_full_match: bool
    any_clean_b2: bool
    any_dirty_b2: bool
    any_b1_ge_2: bool
    max_b1: int
    max_b2: int
    recovering_cells: tuple[tuple[int, float, float, tuple[int, ...]], ...]
    clean_b2_cells: tuple[tuple[int, float, float, tuple[int, ...]], ...]
    dirty_b2_cells: tuple[tuple[int, float, float, tuple[int, ...]], ...]


@pytest.fixture(scope="module")
def linked_tori_seed77_densify384_cal_mult_bundle() -> (
    Seed77Densify384CalMultBundle
):
    """Fit seed77 max_nodes=384; lifetime×cal-mult on signal nodes."""
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
        seed=STAGE1_SEED,
    )
    result = run_scale_search(dataset.points, dim=gt.ambient_dim, config=config)
    scaffold = result.scaffold_at_star
    pos = scaffold.node_positions()
    sigma = sigma_star_from_tau(result.tau_star)
    node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
    signal_mask = np.isin(node_labels, [0, 1])
    signal_pos = pos[signal_mask]
    signal_labs = node_labels[signal_mask]

    signal_fixed = run_per_region_ph(
        signal_pos,
        signal_labs,
        sigma,
        scenario="linked_tori_seed77_m384_signal_fixed",
        include_labels=[0, 1],
        reading="fixed_threshold",
        max_dim=2,
        filtration_mult=FILTRATION_MULTIPLIER,
        expected_betti=EXPECTED_TORI,
    )
    signal_fixed_betti = {
        int(r.region_id): tuple(int(x) for x in r.betti)
        for r in signal_fixed.reports
    }
    signal_both = (
        signal_fixed_betti.get(0) == (1, 2, 0)
        and signal_fixed_betti.get(1) == (1, 2, 0)
    )
    signal_max_b1 = max(
        (int(b[1]) for b in signal_fixed_betti.values() if len(b) > 1),
        default=0,
    )
    signal_max_b2 = max(
        (int(b[2]) for b in signal_fixed_betti.values() if len(b) > 2),
        default=0,
    )

    grid = sweep_lifetime_mult_grid_per_region(
        signal_pos,
        signal_labs,
        sigma,
        fracs=FRAC_GRID,
        mults=MULT_GRID,
        scenario="linked_tori_seed77_m384_lifetime_cal_mult",
        include_labels=[0, 1],
        max_dim=2,
        expected_betti=EXPECTED_TORI,
        b1_target=2,
    )

    recovering: list[tuple[int, float, float, tuple[int, ...]]] = []
    clean_cells: list[tuple[int, float, float, tuple[int, ...]]] = []
    dirty_cells: list[tuple[int, float, float, tuple[int, ...]]] = []
    max_b1 = 0
    max_b2 = signal_max_b2
    any_b1 = False
    for r in grid.rows:
        betti = tuple(int(x) for x in r.betti)
        rid = int(r.region_id)
        frac = float(r.lifetime_frac)
        mult = float(r.filtration_mult)
        max_b1 = max(max_b1, int(r.b1))
        b2 = int(betti[2]) if len(betti) > 2 else 0
        max_b2 = max(max_b2, b2)
        if betti == EXPECTED_TORI or r.match is True:
            recovering.append((rid, frac, mult, betti))
        if int(r.b1) >= 2:
            any_b1 = True
        if _is_clean_b2(betti):
            clean_cells.append((rid, frac, mult, betti))
        if _is_dirty_b2(betti):
            dirty_cells.append((rid, frac, mult, betti))

    return Seed77Densify384CalMultBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        max_nodes=MAX_NODES,
        n_signal=int(signal_pos.shape[0]),
        sigma_star=float(sigma),
        signal_fixed_betti=signal_fixed_betti,
        signal_both_partial=signal_both,
        signal_max_b1=signal_max_b1,
        signal_max_b2=signal_max_b2,
        fracs=FRAC_GRID,
        mults=MULT_GRID,
        table=format_lifetime_mult_grid_table(grid),
        any_full_match=bool(grid.any_full_match) or bool(recovering),
        any_clean_b2=bool(clean_cells),
        any_dirty_b2=bool(dirty_cells),
        any_b1_ge_2=any_b1,
        max_b1=max_b1,
        max_b2=max_b2,
        recovering_cells=tuple(recovering),
        clean_b2_cells=tuple(clean_cells),
        dirty_b2_cells=tuple(dirty_cells),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed77_densify384_cal_mult_harness_lands(
    linked_tori_seed77_densify384_cal_mult_bundle,
) -> None:
    """Seed77 densify384×lifetime×cal-mult lands; SI defaults untouched."""
    bundle = linked_tori_seed77_densify384_cal_mult_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert bundle.max_nodes == MAX_NODES
    assert bundle.n_signal > 0
    assert bundle.sigma_star > 0.0
    assert bundle.fracs == FRAC_GRID
    assert bundle.mults == MULT_GRID
    header = bundle.table.splitlines()[0]
    assert "frac" in header and "mult" in header and "b1" in header
    assert len(bundle.table.splitlines()) == 1 + 2 * len(FRAC_GRID) * len(MULT_GRID)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed77_densify384_cal_mult_documents_gap(
    linked_tori_seed77_densify384_cal_mult_bundle,
) -> None:
    """Document densify384×cal-mult vs clean (1,2,1); never flip awaiting.

    Soft: any full ``(1,2,1)`` or clean ``b2`` is proposal-path evidence.
    Otherwise keep documenting mid-regime regression / cal-mult gap.
    """
    bundle = linked_tori_seed77_densify384_cal_mult_bundle
    if bundle.any_full_match or bundle.any_clean_b2:
        assert FILTRATION_MULTIPLIER == 1.5
        if bundle.any_full_match:
            assert len(bundle.recovering_cells) >= 1
        if bundle.any_clean_b2:
            assert len(bundle.clean_b2_cells) >= 1
            assert bundle.max_b2 >= 1
    else:
        assert bundle.any_full_match is False
        assert bundle.any_clean_b2 is False
        assert bundle.recovering_cells == ()
        assert bundle.clean_b2_cells == ()
        # T50/T59: mid-regime often max_b2=0; cal-mult may or may not unlock dirty.
        assert bundle.any_dirty_b2 or bundle.max_b2 >= 0
        assert bundle.signal_max_b1 >= 0
