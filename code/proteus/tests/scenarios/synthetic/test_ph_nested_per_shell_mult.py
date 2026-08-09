"""Per-shell filtration_mult schedule on nested fitted scaffolds (#41 / A4-T33).

Prior probes (A4-T29 / A4-T32) found a scale split on nested shells:
  * coarse mult=3 recovers inner shell1 ``(1,0,1)`` but not shell2;
  * circle-calibrated mult=6 recovers outer shell2 but not shell1.

This harness applies that schedule jointly (inner=3, outer=6) and compares
against uniform mults. Evidence-gathering only — does **not** flip
``test_nested_spheres_topology`` ``@awaiting`` or change SI defaults.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.nested_spheres import make_nested_spheres
from tests.metrics.persistent_homology import (
    FILTRATION_MULTIPLIER,
    format_scheduled_mult_ph_table,
    nearest_data_labels,
    run_per_region_ph,
    run_scheduled_mult_per_region_ph,
    sigma_star_from_tau,
)

# Existence-proof mult from fitted-circle calibration (A4-T9 / A4-T32).
CIRCLE_CALIBRATED_MULT: float = 6.0
COARSE_INNER_MULT: float = 3.0


@dataclass(frozen=True)
class NestedPerShellMultBundle:
    max_nodes: int
    n_signal: int
    sigma_star: float
    schedule_table: str
    uniform_coarse_diag_match: bool | None
    uniform_cal_diag_match: bool | None
    schedule_all_match: bool | None
    schedule_matches: dict[int, bool | None]
    schedule_betti: dict[int, tuple[int, ...]]
    schedule_mults: dict[int, float]


@pytest.fixture(scope="module")
def nested_per_shell_mult_bundle() -> NestedPerShellMultBundle:
    """Fit max_nodes=128; compare scheduled vs uniform mults on shells 1/2."""
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

    schedule = {1: COARSE_INNER_MULT, 2: CIRCLE_CALIBRATED_MULT}
    scheduled = run_scheduled_mult_per_region_ph(
        signal_pos,
        signal_labs,
        sigma,
        mult_by_region=schedule,
        scenario="nested_per_shell_mult_schedule",
        reading="fixed_threshold",
        max_dim=2,
        expected_betti=(1, 0, 1),
    )
    uniform_coarse = run_per_region_ph(
        signal_pos,
        signal_labs,
        sigma,
        scenario="nested_uniform_coarse_mult",
        include_labels=[1, 2],
        reading="fixed_threshold",
        max_dim=2,
        filtration_mult=COARSE_INNER_MULT,
        expected_betti=(1, 0, 1),
    )
    uniform_cal = run_per_region_ph(
        signal_pos,
        signal_labs,
        sigma,
        scenario="nested_uniform_cal_mult",
        include_labels=[1, 2],
        reading="fixed_threshold",
        max_dim=2,
        filtration_mult=CIRCLE_CALIBRATED_MULT,
        expected_betti=(1, 0, 1),
    )

    return NestedPerShellMultBundle(
        max_nodes=max_nodes,
        n_signal=int(signal_pos.shape[0]),
        sigma_star=float(sigma),
        schedule_table=format_scheduled_mult_ph_table(scheduled),
        uniform_coarse_diag_match=uniform_coarse.all_match,
        uniform_cal_diag_match=uniform_cal.all_match,
        schedule_all_match=scheduled.all_match,
        schedule_matches={
            int(r.region_id): r.match for r in scheduled.rows
        },
        schedule_betti={
            int(r.region_id): tuple(r.betti) for r in scheduled.rows
        },
        schedule_mults={
            int(r.region_id): float(r.filtration_mult) for r in scheduled.rows
        },
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_per_shell_mult_harness_lands(nested_per_shell_mult_bundle) -> None:
    """Scheduled mult table lands; SI fine mult constant untouched."""
    bundle = nested_per_shell_mult_bundle
    assert bundle.max_nodes == 128
    assert bundle.n_signal > 0
    assert bundle.sigma_star > 0.0
    assert FILTRATION_MULTIPLIER == 1.5
    assert bundle.schedule_mults == {1: COARSE_INNER_MULT, 2: CIRCLE_CALIBRATED_MULT}
    assert "mult" in bundle.schedule_table.splitlines()[0]
    assert len(bundle.schedule_table.splitlines()) >= 3


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_per_shell_mult_documents_gap(
    nested_per_shell_mult_bundle,
) -> None:
    """Document whether inner=3/outer=6 recovers both shells; never flip awaiting.

    Soft gate: if the schedule recovers all shells, per-shell mult is viable
    evidence. Otherwise keep explicit non-recovery (uniform paths incomplete).
    """
    bundle = nested_per_shell_mult_bundle
    if bundle.schedule_all_match:
        assert FILTRATION_MULTIPLIER == 1.5
        assert all(bundle.schedule_matches.get(k) is True for k in (1, 2))
    else:
        assert bundle.schedule_all_match is False
        # Uniform single-mult baselines remain incomplete when schedule fails.
        assert bundle.uniform_coarse_diag_match is False
        assert bundle.uniform_cal_diag_match is False
        assert "betti" in bundle.schedule_table
