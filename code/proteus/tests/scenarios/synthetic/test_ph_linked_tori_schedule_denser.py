"""Denser linked-tori per-region mult schedule probe (#41 follow-on).

Crossed coarse/cal schedules failed to recover ``(1,2,1)`` at max_nodes=128
(A4 tori mult harness). This transfers the same schedules to denser
``max_nodes=256`` and asks whether coverage + schedule jointly help.

Evidence-gathering only — does **not** flip ``@awaiting`` or SI mult.
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
    run_scheduled_mult_per_region_ph,
    sigma_star_from_tau,
)

COARSE_MULT: float = 3.0
CIRCLE_CALIBRATED_MULT: float = 6.0
EXPECTED_TORI: tuple[int, ...] = (1, 2, 1)
DENSER_MAX_NODES: int = 256


@dataclass(frozen=True)
class LinkedToriScheduleDenserBundle:
    max_nodes: int
    n_signal: int
    n_per_torus: dict[int, int]
    sigma_star: float
    schedule_a_table: str
    schedule_b_table: str
    schedule_a_all_match: bool | None
    schedule_b_all_match: bool | None
    schedule_a_betti: dict[int, tuple[int, ...]]
    schedule_b_betti: dict[int, tuple[int, ...]]
    any_schedule_recovers: bool


@pytest.fixture(scope="module")
def linked_tori_schedule_denser_bundle() -> LinkedToriScheduleDenserBundle:
    """Fit denser max_nodes=256; crossed coarse/cal schedules on tori 0/1."""
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
    config = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=8,
        max_nodes=DENSER_MAX_NODES,
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
        scenario="linked_tori_schedule_denser_0c_1cal",
        reading="fixed_threshold",
        max_dim=2,
        expected_betti=EXPECTED_TORI,
    )
    sched_b = run_scheduled_mult_per_region_ph(
        signal_pos,
        signal_labs,
        sigma,
        mult_by_region=schedule_b,
        scenario="linked_tori_schedule_denser_0cal_1c",
        reading="fixed_threshold",
        max_dim=2,
        expected_betti=EXPECTED_TORI,
    )
    any_recover = bool(sched_a.all_match is True or sched_b.all_match is True)
    return LinkedToriScheduleDenserBundle(
        max_nodes=DENSER_MAX_NODES,
        n_signal=int(signal_pos.shape[0]),
        n_per_torus={
            int(lab): int(np.sum(signal_labs == lab)) for lab in (0, 1)
        },
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
        any_schedule_recovers=any_recover,
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_schedule_denser_harness_lands(
    linked_tori_schedule_denser_bundle,
) -> None:
    """Denser tori schedule tables land; SI mult untouched."""
    bundle = linked_tori_schedule_denser_bundle
    assert bundle.max_nodes == DENSER_MAX_NODES
    assert bundle.n_signal > 0
    assert bundle.sigma_star > 0.0
    assert FILTRATION_MULTIPLIER == 1.5
    assert "mult" in bundle.schedule_a_table.splitlines()[0]
    assert len(bundle.schedule_a_table.splitlines()) >= 3


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_schedule_denser_documents_gap(
    linked_tori_schedule_denser_bundle,
) -> None:
    """Document denser+schedule transfer for tori (1,2,1); never flip awaiting."""
    bundle = linked_tori_schedule_denser_bundle
    if bundle.any_schedule_recovers:
        assert FILTRATION_MULTIPLIER == 1.5
        assert (
            bundle.schedule_a_all_match is True
            or bundle.schedule_b_all_match is True
        )
    else:
        assert bundle.schedule_a_all_match is False
        assert bundle.schedule_b_all_match is False
        assert "betti" in bundle.schedule_a_table
