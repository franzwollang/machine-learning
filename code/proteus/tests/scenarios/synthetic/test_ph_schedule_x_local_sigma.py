"""Schedule × local-sigma combo on nested + linked-tori fitted PH (#41).

Combines the nested per-shell mult schedule ``{1: 3, 2: 6}`` (and the tori
crossed coarse/cal schedules) with per-region median-NN local sigma — the two
levers previously tested separately.

Evidence-gathering only — does **not** flip nested/tori ``@awaiting`` tests or
change SI ``FILTRATION_MULTIPLIER``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.linked_tori import make_linked_tori
from tests.datasets.synthetic.nested_spheres import make_nested_spheres
from tests.metrics.persistent_homology import (
    FILTRATION_MULTIPLIER,
    format_scheduled_mult_ph_table,
    nearest_data_labels,
    per_region_median_nn_sigma,
    run_scheduled_mult_per_region_ph,
    sigma_star_from_tau,
)

COARSE_MULT: float = 3.0
CIRCLE_CALIBRATED_MULT: float = 6.0
NESTED_SCHEDULE: dict[int, float] = {1: COARSE_MULT, 2: CIRCLE_CALIBRATED_MULT}
TORI_SCHEDULE_A: dict[int, float] = {0: COARSE_MULT, 1: CIRCLE_CALIBRATED_MULT}
TORI_SCHEDULE_B: dict[int, float] = {0: CIRCLE_CALIBRATED_MULT, 1: COARSE_MULT}


@dataclass(frozen=True)
class ScheduleLocalSigmaComboBundle:
    nested_max_nodes: int
    nested_n_signal: int
    nested_global_table: str
    nested_local_table: str
    nested_global_all_match: bool | None
    nested_local_all_match: bool | None
    nested_local_sigmas: tuple[float, ...]
    nested_local_preserves: bool
    tori_max_nodes: int
    tori_n_signal: int
    tori_local_sigmas: tuple[float, ...]
    tori_a_global_all_match: bool | None
    tori_a_local_all_match: bool | None
    tori_b_global_all_match: bool | None
    tori_b_local_all_match: bool | None
    tori_a_local_table: str
    tori_b_local_table: str
    tori_any_local_recovers: bool
    tori_a_local_betti: dict[int, tuple[int, ...]]
    tori_b_local_betti: dict[int, tuple[int, ...]]


def _fit_signal(
    *,
    points: np.ndarray,
    labels: np.ndarray,
    ambient_dim: int,
    tau_lo: float,
    tau_hi: float,
    signal_labels: list[int],
    max_nodes: int = 128,
    seed: int = 77,
) -> tuple[np.ndarray, np.ndarray, float]:
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
        seed=seed,
    )
    result = run_scale_search(points, dim=ambient_dim, config=config)
    pos = result.scaffold_at_star.node_positions()
    sigma = sigma_star_from_tau(result.tau_star)
    node_labels = nearest_data_labels(pos, points, labels)
    signal_mask = np.isin(node_labels, signal_labels)
    return pos[signal_mask], node_labels[signal_mask], float(sigma)


@pytest.fixture(scope="module")
def schedule_local_sigma_combo_bundle() -> ScheduleLocalSigmaComboBundle:
    """Fit nested + tori; schedule under global vs local sigma."""
    nested_ds = make_nested_spheres(
        n_per_sphere=500,
        radii=(1.0, 2.0),
        ambient_dim=3,
        noise=0.02,
        tissue_fraction=0.03,
        seed=21,
    )
    n_gt = nested_ds.ground_truth
    n_tau_lo, n_tau_hi = n_gt.tau_grid_hint
    n_pos, n_labs, n_sigma = _fit_signal(
        points=nested_ds.points,
        labels=nested_ds.labels,
        ambient_dim=n_gt.ambient_dim,
        tau_lo=n_tau_lo,
        tau_hi=n_tau_hi,
        signal_labels=[1, 2],
    )
    n_local_labs, n_local_sigmas = per_region_median_nn_sigma(
        n_pos, n_labs, include_labels=[1, 2],
    )
    nested_g = run_scheduled_mult_per_region_ph(
        n_pos,
        n_labs,
        n_sigma,
        mult_by_region=NESTED_SCHEDULE,
        scenario="nested_schedule_global_sigma",
        reading="fixed_threshold",
        max_dim=2,
        expected_betti=(1, 0, 1),
    )
    nested_l = run_scheduled_mult_per_region_ph(
        n_pos,
        n_labs,
        n_local_sigmas,
        mult_by_region={lab: NESTED_SCHEDULE[lab] for lab in n_local_labs},
        scenario="nested_schedule_local_sigma",
        reading="fixed_threshold",
        max_dim=2,
        expected_betti=(1, 0, 1),
    )

    tori_ds = make_linked_tori(
        n_per_torus=500,
        major_radius=2.0,
        minor_radius=0.5,
        noise=0.02,
        tissue_fraction=0.03,
        seed=21,
    )
    t_gt = tori_ds.ground_truth
    t_tau_lo, t_tau_hi = t_gt.tau_grid_hint
    t_pos, t_labs, t_sigma = _fit_signal(
        points=tori_ds.points,
        labels=tori_ds.labels,
        ambient_dim=t_gt.ambient_dim,
        tau_lo=t_tau_lo,
        tau_hi=t_tau_hi,
        signal_labels=[0, 1],
    )
    t_local_labs, t_local_sigmas = per_region_median_nn_sigma(
        t_pos, t_labs, include_labels=[0, 1],
    )
    expected_tori = (1, 2, 1)

    def _tori_sched(schedule: dict[int, float], sigma, tag: str):
        return run_scheduled_mult_per_region_ph(
            t_pos,
            t_labs,
            sigma,
            mult_by_region=schedule,
            scenario=tag,
            reading="fixed_threshold",
            max_dim=2,
            expected_betti=expected_tori,
        )

    tori_a_g = _tori_sched(TORI_SCHEDULE_A, t_sigma, "tori_sched_a_global")
    tori_a_l = _tori_sched(
        {lab: TORI_SCHEDULE_A[lab] for lab in t_local_labs},
        t_local_sigmas,
        "tori_sched_a_local",
    )
    tori_b_g = _tori_sched(TORI_SCHEDULE_B, t_sigma, "tori_sched_b_global")
    tori_b_l = _tori_sched(
        {lab: TORI_SCHEDULE_B[lab] for lab in t_local_labs},
        t_local_sigmas,
        "tori_sched_b_local",
    )

    return ScheduleLocalSigmaComboBundle(
        nested_max_nodes=128,
        nested_n_signal=int(n_pos.shape[0]),
        nested_global_table=format_scheduled_mult_ph_table(nested_g),
        nested_local_table=format_scheduled_mult_ph_table(nested_l),
        nested_global_all_match=nested_g.all_match,
        nested_local_all_match=nested_l.all_match,
        nested_local_sigmas=tuple(float(s) for s in n_local_sigmas),
        nested_local_preserves=bool(
            nested_g.all_match is True and nested_l.all_match is True
        ),
        tori_max_nodes=128,
        tori_n_signal=int(t_pos.shape[0]),
        tori_local_sigmas=tuple(float(s) for s in t_local_sigmas),
        tori_a_global_all_match=tori_a_g.all_match,
        tori_a_local_all_match=tori_a_l.all_match,
        tori_b_global_all_match=tori_b_g.all_match,
        tori_b_local_all_match=tori_b_l.all_match,
        tori_a_local_table=format_scheduled_mult_ph_table(tori_a_l),
        tori_b_local_table=format_scheduled_mult_ph_table(tori_b_l),
        tori_any_local_recovers=bool(
            tori_a_l.all_match is True or tori_b_l.all_match is True
        ),
        tori_a_local_betti={
            int(r.region_id): tuple(r.betti) for r in tori_a_l.rows
        },
        tori_b_local_betti={
            int(r.region_id): tuple(r.betti) for r in tori_b_l.rows
        },
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_schedule_x_local_sigma_harness_lands(
    schedule_local_sigma_combo_bundle,
) -> None:
    """Combo tables land; local sigmas positive; SI mult untouched."""
    bundle = schedule_local_sigma_combo_bundle
    assert bundle.nested_max_nodes == 128
    assert bundle.tori_max_nodes == 128
    assert bundle.nested_n_signal > 0
    assert bundle.tori_n_signal > 0
    assert FILTRATION_MULTIPLIER == 1.5
    assert len(bundle.nested_local_sigmas) == 2
    assert len(bundle.tori_local_sigmas) == 2
    assert all(np.isfinite(s) and s > 0.0 for s in bundle.nested_local_sigmas)
    assert all(np.isfinite(s) and s > 0.0 for s in bundle.tori_local_sigmas)
    assert "mult" in bundle.nested_local_table.splitlines()[0]
    assert "mult" in bundle.tori_a_local_table.splitlines()[0]
    assert len(bundle.nested_local_table.splitlines()) >= 3
    assert len(bundle.tori_a_local_table.splitlines()) >= 3


@pytest.mark.scenario
@pytest.mark.synthetic
def test_schedule_x_local_sigma_documents_gap(
    schedule_local_sigma_combo_bundle,
) -> None:
    """Document nested preserve + tori combo gap; never flip awaiting.

    Soft gate: nested local preserving global schedule recovery is proposal-path
    evidence. Tori soft-gate on any local schedule full match.
    """
    bundle = schedule_local_sigma_combo_bundle
    # Nested: global schedule is the known recovering baseline (A4-T33).
    if bundle.nested_global_all_match:
        assert FILTRATION_MULTIPLIER == 1.5
        # Local-sigma either preserves recovery or we document the regression.
        if bundle.nested_local_all_match:
            assert bundle.nested_local_preserves is True
        else:
            assert bundle.nested_local_preserves is False
            assert "betti" in bundle.nested_local_table
    else:
        assert bundle.nested_global_all_match is False

    if bundle.tori_any_local_recovers:
        assert FILTRATION_MULTIPLIER == 1.5
        assert (
            bundle.tori_a_local_all_match is True
            or bundle.tori_b_local_all_match is True
        )
    else:
        assert bundle.tori_a_local_all_match is False
        assert bundle.tori_b_local_all_match is False
        assert bundle.tori_a_global_all_match is False
        assert bundle.tori_b_global_all_match is False
        assert "betti" in bundle.tori_a_local_table
        assert "betti" in bundle.tori_b_local_table
