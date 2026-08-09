"""Per-region filtration_mult schedule on linked-tori fitted scaffolds (#41).

Parallel to nested A4-T33 schedule ``{1: 3, 2: 6}``: probe whether asymmetric
per-torus mults recover fitted ``(1, 2, 1)`` under fixed_threshold. Tries
crossed coarse/calibrated schedules and uniform baselines on max_nodes=128.

Evidence-gathering only — does **not** flip ``test_linked_tori_betti_numbers``
``@awaiting`` or change SI ``FILTRATION_MULTIPLIER``.
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


@dataclass(frozen=True)
class LinkedToriPerRegionMultBundle:
    max_nodes: int
    n_signal: int
    sigma_star: float
    schedule_a_table: str
    schedule_b_table: str
    schedule_a_all_match: bool | None
    schedule_b_all_match: bool | None
    schedule_a_betti: dict[int, tuple[int, ...]]
    schedule_b_betti: dict[int, tuple[int, ...]]
    uniform_coarse_match: bool | None
    uniform_cal_match: bool | None
    any_schedule_recovers: bool


@pytest.fixture(scope="module")
def linked_tori_per_region_mult_bundle() -> LinkedToriPerRegionMultBundle:
    """Fit max_nodes=128; crossed coarse/cal schedules on tori 0/1."""
    dataset = make_linked_tori(
        n_per_torus=500,
        major_radius=2.0,
        minor_radius=0.5,
        noise=0.02,
        tissue_fraction=0.03,
        seed=21,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    max_nodes = 128
    config = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=8,
        max_nodes=max_nodes,
        ann_backend="naive",
        stabilization=StabilizationConfig(
            min_equilibrium_epochs=3, max_epochs=15,
        ),
        seed=77,
    )
    result = run_scale_search(dataset.points, dim=gt.ambient_dim, config=config)
    pos = result.scaffold_at_star.node_positions()
    sigma = sigma_star_from_tau(result.tau_star)
    node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
    # Linked-tori signal labels are 0/1 (tissue -1).
    signal_mask = np.isin(node_labels, [0, 1])
    signal_pos = pos[signal_mask]
    signal_labs = node_labels[signal_mask]

    schedule_a = {0: COARSE_MULT, 1: CIRCLE_CALIBRATED_MULT}
    schedule_b = {0: CIRCLE_CALIBRATED_MULT, 1: COARSE_MULT}

    sched_a = run_scheduled_mult_per_region_ph(
        signal_pos,
        signal_labs,
        sigma,
        mult_by_region=schedule_a,
        scenario="linked_tori_schedule_0c_1cal",
        reading="fixed_threshold",
        max_dim=2,
        expected_betti=EXPECTED_TORI,
    )
    sched_b = run_scheduled_mult_per_region_ph(
        signal_pos,
        signal_labs,
        sigma,
        mult_by_region=schedule_b,
        scenario="linked_tori_schedule_0cal_1c",
        reading="fixed_threshold",
        max_dim=2,
        expected_betti=EXPECTED_TORI,
    )
    uniform_coarse = run_per_region_ph(
        signal_pos,
        signal_labs,
        sigma,
        scenario="linked_tori_uniform_coarse",
        include_labels=[0, 1],
        reading="fixed_threshold",
        max_dim=2,
        filtration_mult=COARSE_MULT,
        expected_betti=EXPECTED_TORI,
    )
    uniform_cal = run_per_region_ph(
        signal_pos,
        signal_labs,
        sigma,
        scenario="linked_tori_uniform_cal",
        include_labels=[0, 1],
        reading="fixed_threshold",
        max_dim=2,
        filtration_mult=CIRCLE_CALIBRATED_MULT,
        expected_betti=EXPECTED_TORI,
    )

    any_recover = bool(sched_a.all_match is True or sched_b.all_match is True)
    return LinkedToriPerRegionMultBundle(
        max_nodes=max_nodes,
        n_signal=int(signal_pos.shape[0]),
        sigma_star=float(sigma),
        schedule_a_table=format_scheduled_mult_ph_table(sched_a),
        schedule_b_table=format_scheduled_mult_ph_table(sched_b),
        schedule_a_all_match=sched_a.all_match,
        schedule_b_all_match=sched_b.all_match,
        schedule_a_betti={
            int(r.region_id): tuple(r.betti) for r in sched_a.rows
        },
        schedule_b_betti={
            int(r.region_id): tuple(r.betti) for r in sched_b.rows
        },
        uniform_coarse_match=uniform_coarse.all_match,
        uniform_cal_match=uniform_cal.all_match,
        any_schedule_recovers=any_recover,
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_per_region_mult_harness_lands(
    linked_tori_per_region_mult_bundle,
) -> None:
    """Crossed schedule tables land; SI fine mult constant untouched."""
    bundle = linked_tori_per_region_mult_bundle
    assert bundle.max_nodes == 128
    assert bundle.n_signal > 0
    assert bundle.sigma_star > 0.0
    assert FILTRATION_MULTIPLIER == 1.5
    assert "mult" in bundle.schedule_a_table.splitlines()[0]
    assert "mult" in bundle.schedule_b_table.splitlines()[0]
    assert len(bundle.schedule_a_table.splitlines()) >= 3
    assert len(bundle.schedule_b_table.splitlines()) >= 3


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_per_region_mult_documents_gap(
    linked_tori_per_region_mult_bundle,
) -> None:
    """Document whether crossed coarse/cal schedules recover (1,2,1).

    Soft gate: any recovering schedule is proposal-path evidence. Otherwise
    keep explicit non-recovery. Never flip ``@awaiting``.
    """
    bundle = linked_tori_per_region_mult_bundle
    if bundle.any_schedule_recovers:
        assert FILTRATION_MULTIPLIER == 1.5
        assert (
            bundle.schedule_a_all_match is True
            or bundle.schedule_b_all_match is True
        )
    else:
        assert bundle.schedule_a_all_match is False
        assert bundle.schedule_b_all_match is False
        assert bundle.uniform_coarse_match is False
        assert bundle.uniform_cal_match is False
        assert "betti" in bundle.schedule_a_table
        assert "betti" in bundle.schedule_b_table
