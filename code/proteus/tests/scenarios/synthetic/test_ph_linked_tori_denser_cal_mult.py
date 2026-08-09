"""Circle-calibrated mult=6 on denser fitted linked-tori (#41 follow-on).

A4-T38 partial ``b1=2`` used SI fine mult=1.5 at ``max_nodes=256``. Nested
recovery used cal-mult=6 on the outer shell. This harness asks whether
uniform cal-mult=6 (and crossed coarse/cal schedules) on denser fitted
scaffolds (``max_nodes∈{256,384}``, ``n_per_torus=500``) unlocks
``(1,2,1)``.

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
    format_scheduled_mult_ph_table,
    nearest_data_labels,
    run_per_region_ph,
    run_scheduled_mult_per_region_ph,
    sigma_star_from_tau,
)

COARSE_MULT: float = 3.0
CIRCLE_CALIBRATED_MULT: float = 6.0
EXPECTED_TORI: tuple[int, ...] = (1, 2, 1)
N_PER_TORUS: int = 500
MAX_NODES_LADDER: tuple[int, ...] = (256, 384)


@dataclass(frozen=True)
class DenserCalMultRow:
    max_nodes: int
    n_signal: int
    n_per_torus: dict[int, int]
    sigma_star: float
    fine_betti: dict[int, tuple[int, ...]]
    fine_all_match: bool | None
    fine_max_b1: int
    cal_betti: dict[int, tuple[int, ...]]
    cal_all_match: bool | None
    cal_max_b1: int
    schedule_a_betti: dict[int, tuple[int, ...]]
    schedule_b_betti: dict[int, tuple[int, ...]]
    schedule_a_all_match: bool | None
    schedule_b_all_match: bool | None
    schedule_a_table: str
    schedule_b_table: str
    any_recipe_recovers: bool
    any_b1_ge_2: bool
    max_b1: int


@dataclass(frozen=True)
class LinkedToriDenserCalMultBundle:
    n_per_torus_data: int
    rows: tuple[DenserCalMultRow, ...]
    any_full_recover: bool
    any_b1_ge_2: bool
    max_b1: int
    b1_ge_2_cells: tuple[tuple[int, str, int, tuple[int, ...]], ...]


def _betti_map(reports) -> dict[int, tuple[int, ...]]:
    return {
        int(r.region_id): tuple(int(x) for x in r.betti) for r in reports
    }


def _max_b1(betti: dict[int, tuple[int, ...]]) -> int:
    return max((int(b[1]) for b in betti.values() if len(b) > 1), default=0)


@pytest.fixture(scope="module")
def linked_tori_denser_cal_mult_bundle() -> LinkedToriDenserCalMultBundle:
    """Fit denser scaffolds; compare fine vs cal-mult=6 vs crossed schedules."""
    dataset = make_linked_tori(
        n_per_torus=N_PER_TORUS,
        major_radius=2.0,
        minor_radius=0.5,
        noise=0.02,
        tissue_fraction=0.03,
        seed=21,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint

    rows: list[DenserCalMultRow] = []
    cells: list[tuple[int, str, int, tuple[int, ...]]] = []
    for max_nodes in MAX_NODES_LADDER:
        config = ScaleSearchConfig(
            tau_min=tau_lo,
            tau_max=tau_hi,
            max_grid_points=8,
            k=8,
            n_seeds=8,
            max_nodes=int(max_nodes),
            ann_backend="naive",
            stabilization=StabilizationConfig(
                min_equilibrium_epochs=3, max_epochs=15,
            ),
            seed=77,
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

        fine = run_per_region_ph(
            signal_pos,
            signal_labs,
            sigma,
            scenario=f"linked_tori_denser_cal_fine_m{max_nodes}",
            include_labels=[0, 1],
            reading="fixed_threshold",
            max_dim=2,
            filtration_mult=FILTRATION_MULTIPLIER,
            expected_betti=EXPECTED_TORI,
        )
        cal = run_per_region_ph(
            signal_pos,
            signal_labs,
            sigma,
            scenario=f"linked_tori_denser_cal_uniform6_m{max_nodes}",
            include_labels=[0, 1],
            reading="fixed_threshold",
            max_dim=2,
            filtration_mult=CIRCLE_CALIBRATED_MULT,
            expected_betti=EXPECTED_TORI,
        )
        schedule_a = {0: COARSE_MULT, 1: CIRCLE_CALIBRATED_MULT}
        schedule_b = {0: CIRCLE_CALIBRATED_MULT, 1: COARSE_MULT}
        sched_a = run_scheduled_mult_per_region_ph(
            signal_pos,
            signal_labs,
            sigma,
            mult_by_region=schedule_a,
            scenario=f"linked_tori_denser_cal_sched_a_m{max_nodes}",
            reading="fixed_threshold",
            max_dim=2,
            expected_betti=EXPECTED_TORI,
        )
        sched_b = run_scheduled_mult_per_region_ph(
            signal_pos,
            signal_labs,
            sigma,
            mult_by_region=schedule_b,
            scenario=f"linked_tori_denser_cal_sched_b_m{max_nodes}",
            reading="fixed_threshold",
            max_dim=2,
            expected_betti=EXPECTED_TORI,
        )

        fine_betti = _betti_map(fine.reports)
        cal_betti = _betti_map(cal.reports)
        sched_a_betti = {int(r.region_id): tuple(r.betti) for r in sched_a.rows}
        sched_b_betti = {int(r.region_id): tuple(r.betti) for r in sched_b.rows}
        fine_max = _max_b1(fine_betti)
        cal_max = _max_b1(cal_betti)
        sched_max = max(_max_b1(sched_a_betti), _max_b1(sched_b_betti))
        max_b1 = max(fine_max, cal_max, sched_max)
        any_recover = bool(
            fine.all_match is True
            or cal.all_match is True
            or sched_a.all_match is True
            or sched_b.all_match is True
        )
        any_b1 = max_b1 >= 2
        for recipe, betti in (
            ("fine", fine_betti),
            ("cal6", cal_betti),
            ("sched_a", sched_a_betti),
            ("sched_b", sched_b_betti),
        ):
            for lab, b in betti.items():
                if len(b) > 1 and int(b[1]) >= 2:
                    cells.append((int(max_nodes), recipe, int(lab), tuple(b)))

        rows.append(
            DenserCalMultRow(
                max_nodes=int(max_nodes),
                n_signal=int(signal_pos.shape[0]),
                n_per_torus={
                    int(lab): int(np.sum(signal_labs == lab)) for lab in (0, 1)
                },
                sigma_star=float(sigma),
                fine_betti=fine_betti,
                fine_all_match=fine.all_match,
                fine_max_b1=fine_max,
                cal_betti=cal_betti,
                cal_all_match=cal.all_match,
                cal_max_b1=cal_max,
                schedule_a_betti=sched_a_betti,
                schedule_b_betti=sched_b_betti,
                schedule_a_all_match=sched_a.all_match,
                schedule_b_all_match=sched_b.all_match,
                schedule_a_table=format_scheduled_mult_ph_table(sched_a),
                schedule_b_table=format_scheduled_mult_ph_table(sched_b),
                any_recipe_recovers=any_recover,
                any_b1_ge_2=any_b1,
                max_b1=max_b1,
            )
        )

    return LinkedToriDenserCalMultBundle(
        n_per_torus_data=N_PER_TORUS,
        rows=tuple(rows),
        any_full_recover=any(r.any_recipe_recovers for r in rows),
        any_b1_ge_2=bool(cells),
        max_b1=max((r.max_b1 for r in rows), default=0),
        b1_ge_2_cells=tuple(cells),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_denser_cal_mult_harness_lands(
    linked_tori_denser_cal_mult_bundle,
) -> None:
    """Denser cal-mult tables land; SI fine mult constant untouched."""
    bundle = linked_tori_denser_cal_mult_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert CIRCLE_CALIBRATED_MULT == 6.0
    assert bundle.n_per_torus_data == N_PER_TORUS
    assert len(bundle.rows) == len(MAX_NODES_LADDER)
    assert [r.max_nodes for r in bundle.rows] == list(MAX_NODES_LADDER)
    assert all(r.n_signal > 0 for r in bundle.rows)
    assert "mult" in bundle.rows[0].schedule_a_table.splitlines()[0]


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_denser_cal_mult_documents_gap(
    linked_tori_denser_cal_mult_bundle,
) -> None:
    """Document cal-mult=6 denser fitted gap; never flip awaiting on partial.

    Soft: any recipe full ``(1,2,1)`` or ``b1≥2`` is proposal-path evidence.
    Partial b1 without both-tori SI match stays @awaiting.
    """
    bundle = linked_tori_denser_cal_mult_bundle
    if bundle.any_full_recover or bundle.any_b1_ge_2:
        assert FILTRATION_MULTIPLIER == 1.5
        assert bundle.max_b1 >= 2
        if bundle.any_b1_ge_2:
            assert len(bundle.b1_ge_2_cells) >= 1
        if not bundle.any_full_recover:
            assert all(not r.any_recipe_recovers for r in bundle.rows)
    else:
        assert bundle.any_full_recover is False
        assert bundle.any_b1_ge_2 is False
        assert bundle.max_b1 < 2
        assert bundle.b1_ge_2_cells == ()
        for row in bundle.rows:
            assert row.any_recipe_recovers is False
