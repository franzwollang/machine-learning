"""Nested schedule × lifetime_frac under global sigma (#41 / A4-T39).

Nested per-shell mult schedule ``{1:3, 2:6}`` recovers both shells under
``fixed_threshold`` + global σ (A4-T33). Local-σ × schedule regresses shell2
(A4-T36). This harness keeps **global σ** and crosses the recovering schedule
with a ``lifetime_frac`` ladder under the lifetime reading — asking whether
lifetime preserves schedule recovery or needs a different frac.

Evidence-gathering only — does **not** flip nested ``@awaiting`` or SI defaults.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.nested_spheres import make_nested_spheres
from tests.metrics.persistent_homology import (
    DEFAULT_LIFETIME_FRAC,
    FILTRATION_MULTIPLIER,
    format_scheduled_mult_ph_table,
    nearest_data_labels,
    run_scheduled_mult_per_region_ph,
    sigma_star_from_tau,
)

COARSE_INNER_MULT: float = 3.0
CIRCLE_CALIBRATED_MULT: float = 6.0
NESTED_SCHEDULE: dict[int, float] = {1: COARSE_INNER_MULT, 2: CIRCLE_CALIBRATED_MULT}
EXPECTED_SHELL: tuple[int, ...] = (1, 0, 1)
LIFETIME_FRACS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class NestedScheduleLifetimeRow:
    lifetime_frac: float
    all_match: bool | None
    betti: dict[int, tuple[int, ...]]
    table: str


@dataclass(frozen=True)
class NestedScheduleLifetimeBundle:
    max_nodes: int
    n_signal: int
    sigma_star: float
    fixed_threshold_all_match: bool | None
    fixed_threshold_betti: dict[int, tuple[int, ...]]
    lifetime_rows: tuple[NestedScheduleLifetimeRow, ...]
    lifetime_any_full_match: bool
    recovering_fracs: tuple[float, ...]


@pytest.fixture(scope="module")
def nested_schedule_x_lifetime_bundle() -> NestedScheduleLifetimeBundle:
    """Fit max_nodes=128; schedule × fixed_threshold vs lifetime frac ladder."""
    dataset = make_nested_spheres(
        n_per_sphere=500,
        radii=(1.0, 2.0),
        ambient_dim=3,
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
    signal_mask = np.isin(node_labels, [1, 2])
    signal_pos = pos[signal_mask]
    signal_labs = node_labels[signal_mask]

    fixed = run_scheduled_mult_per_region_ph(
        signal_pos,
        signal_labs,
        sigma,
        mult_by_region=NESTED_SCHEDULE,
        scenario="nested_schedule_fixed_threshold",
        reading="fixed_threshold",
        max_dim=2,
        expected_betti=EXPECTED_SHELL,
    )

    lifetime_rows: list[NestedScheduleLifetimeRow] = []
    recovering: list[float] = []
    for frac in LIFETIME_FRACS:
        sched = run_scheduled_mult_per_region_ph(
            signal_pos,
            signal_labs,
            sigma,
            mult_by_region=NESTED_SCHEDULE,
            scenario=f"nested_schedule_lifetime_f{frac:g}",
            reading="lifetime",
            max_dim=2,
            lifetime_frac=float(frac),
            expected_betti=EXPECTED_SHELL,
        )
        betti = {int(r.region_id): tuple(r.betti) for r in sched.rows}
        if sched.all_match is True:
            recovering.append(float(frac))
        lifetime_rows.append(
            NestedScheduleLifetimeRow(
                lifetime_frac=float(frac),
                all_match=sched.all_match,
                betti=betti,
                table=format_scheduled_mult_ph_table(sched),
            )
        )

    return NestedScheduleLifetimeBundle(
        max_nodes=max_nodes,
        n_signal=int(signal_pos.shape[0]),
        sigma_star=float(sigma),
        fixed_threshold_all_match=fixed.all_match,
        fixed_threshold_betti={
            int(r.region_id): tuple(r.betti) for r in fixed.rows
        },
        lifetime_rows=tuple(lifetime_rows),
        lifetime_any_full_match=bool(recovering),
        recovering_fracs=tuple(recovering),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_schedule_x_lifetime_harness_lands(
    nested_schedule_x_lifetime_bundle,
) -> None:
    """Harness lands; SI defaults untouched; schedule uses global σ."""
    bundle = nested_schedule_x_lifetime_bundle
    assert bundle.max_nodes == 128
    assert bundle.n_signal > 0
    assert bundle.sigma_star > 0.0
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert len(bundle.lifetime_rows) == len(LIFETIME_FRACS)
    assert set(bundle.fixed_threshold_betti.keys()) == {1, 2}


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_schedule_x_lifetime_documents_transfer(
    nested_schedule_x_lifetime_bundle,
) -> None:
    """fixed_threshold schedule stays green; document lifetime transfer.

    Soft: recovering lifetime fracs are proposal-path evidence for reading
    choice under the nested mult schedule. Never flip awaiting.
    """
    bundle = nested_schedule_x_lifetime_bundle
    # Baseline: global-σ schedule under fixed_threshold should still recover.
    assert bundle.fixed_threshold_all_match is True
    assert bundle.fixed_threshold_betti[1] == EXPECTED_SHELL
    assert bundle.fixed_threshold_betti[2] == EXPECTED_SHELL
    if bundle.lifetime_any_full_match:
        assert len(bundle.recovering_fracs) >= 1
        for frac in bundle.recovering_fracs:
            row = next(r for r in bundle.lifetime_rows if r.lifetime_frac == frac)
            assert row.all_match is True
            assert row.betti[1] == EXPECTED_SHELL
            assert row.betti[2] == EXPECTED_SHELL
    else:
        assert bundle.recovering_fracs == ()
        assert all(r.all_match is not True for r in bundle.lifetime_rows)
        # SI default lifetime under schedule does not recover both shells.
        si_row = next(
            r for r in bundle.lifetime_rows
            if r.lifetime_frac == DEFAULT_LIFETIME_FRAC
        )
        assert si_row.all_match is not True
