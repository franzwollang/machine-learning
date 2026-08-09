"""Denser max_nodes schedule transfer for nested fitted PH (#41 follow-on).

A4-T33 showed per-shell mult schedule ``{1: 3, 2: 6}`` recovers both nested
shells at ``max_nodes=128``. This harness asks whether the same schedule
still recovers (or improves occupancy) at denser ``max_nodes=256``, and
compares against uniform mults on the denser scaffold.

Evidence-gathering only — does **not** flip ``test_nested_spheres_topology``
``@awaiting`` or change SI ``FILTRATION_MULTIPLIER``.
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

CIRCLE_CALIBRATED_MULT: float = 6.0
COARSE_INNER_MULT: float = 3.0
BASELINE_MAX_NODES: int = 128
DENSER_MAX_NODES: int = 256


@dataclass(frozen=True)
class NestedScheduleDenserBundle:
    max_nodes_baseline: int
    max_nodes_dense: int
    n_signal_baseline: int
    n_signal_dense: int
    n_per_shell_dense: dict[int, int]
    sigma_star_dense: float
    schedule_table_dense: str
    schedule_all_match_dense: bool | None
    schedule_matches_dense: dict[int, bool | None]
    schedule_betti_dense: dict[int, tuple[int, ...]]
    uniform_coarse_match_dense: bool | None
    uniform_cal_match_dense: bool | None
    denser_preserves_schedule: bool


def _fit_nested_signal(max_nodes: int):
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
    result = run_scale_search(dataset.points, dim=gt.ambient_dim, config=config)
    pos = result.scaffold_at_star.node_positions()
    sigma = sigma_star_from_tau(result.tau_star)
    node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
    signal_mask = np.isin(node_labels, [1, 2])
    signal_pos = pos[signal_mask]
    signal_labs = node_labels[signal_mask]
    return signal_pos, signal_labs, float(sigma)


@pytest.fixture(scope="module")
def nested_schedule_denser_bundle() -> NestedScheduleDenserBundle:
    """Compare schedule {1:3, 2:6} at max_nodes=128 vs 256."""
    pos128, labs128, _sigma128 = _fit_nested_signal(BASELINE_MAX_NODES)
    pos256, labs256, sigma256 = _fit_nested_signal(DENSER_MAX_NODES)

    schedule = {1: COARSE_INNER_MULT, 2: CIRCLE_CALIBRATED_MULT}
    scheduled = run_scheduled_mult_per_region_ph(
        pos256,
        labs256,
        sigma256,
        mult_by_region=schedule,
        scenario="nested_schedule_denser_256",
        reading="fixed_threshold",
        max_dim=2,
        expected_betti=(1, 0, 1),
    )
    uniform_coarse = run_per_region_ph(
        pos256,
        labs256,
        sigma256,
        scenario="nested_uniform_coarse_denser",
        include_labels=[1, 2],
        reading="fixed_threshold",
        max_dim=2,
        filtration_mult=COARSE_INNER_MULT,
        expected_betti=(1, 0, 1),
    )
    uniform_cal = run_per_region_ph(
        pos256,
        labs256,
        sigma256,
        scenario="nested_uniform_cal_denser",
        include_labels=[1, 2],
        reading="fixed_threshold",
        max_dim=2,
        filtration_mult=CIRCLE_CALIBRATED_MULT,
        expected_betti=(1, 0, 1),
    )

    matches = {int(r.region_id): r.match for r in scheduled.rows}
    betti = {int(r.region_id): tuple(r.betti) for r in scheduled.rows}
    n_per_shell = {
        int(lab): int(np.sum(labs256 == lab)) for lab in (1, 2)
    }
    denser_preserves = scheduled.all_match is True

    return NestedScheduleDenserBundle(
        max_nodes_baseline=BASELINE_MAX_NODES,
        max_nodes_dense=DENSER_MAX_NODES,
        n_signal_baseline=int(pos128.shape[0]),
        n_signal_dense=int(pos256.shape[0]),
        n_per_shell_dense=n_per_shell,
        sigma_star_dense=float(sigma256),
        schedule_table_dense=format_scheduled_mult_ph_table(scheduled),
        schedule_all_match_dense=scheduled.all_match,
        schedule_matches_dense=matches,
        schedule_betti_dense=betti,
        uniform_coarse_match_dense=uniform_coarse.all_match,
        uniform_cal_match_dense=uniform_cal.all_match,
        denser_preserves_schedule=bool(denser_preserves),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_schedule_denser_harness_lands(
    nested_schedule_denser_bundle,
) -> None:
    """Denser schedule table lands; SI fine mult constant untouched."""
    bundle = nested_schedule_denser_bundle
    assert bundle.max_nodes_baseline == BASELINE_MAX_NODES
    assert bundle.max_nodes_dense == DENSER_MAX_NODES
    assert bundle.n_signal_dense > 0
    assert bundle.n_signal_dense >= bundle.n_signal_baseline
    assert bundle.sigma_star_dense > 0.0
    assert FILTRATION_MULTIPLIER == 1.5
    assert "mult" in bundle.schedule_table_dense.splitlines()[0]
    assert len(bundle.schedule_table_dense.splitlines()) >= 3


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_schedule_denser_documents_transfer(
    nested_schedule_denser_bundle,
) -> None:
    """Document whether schedule {1:3,2:6} transfers to denser coverage.

    Soft gate: denser recovery is green evidence for schedule robustness.
    Non-recovery keeps explicit gap (uniform paths incomplete). Never flip
    ``@awaiting``.
    """
    bundle = nested_schedule_denser_bundle
    if bundle.denser_preserves_schedule:
        assert FILTRATION_MULTIPLIER == 1.5
        assert all(
            bundle.schedule_matches_dense.get(k) is True for k in (1, 2)
        )
    else:
        assert bundle.schedule_all_match_dense is False
        # Uniform single-mult baselines remain incomplete when schedule fails.
        assert bundle.uniform_coarse_match_dense is False
        assert bundle.uniform_cal_match_dense is False
        assert "betti" in bundle.schedule_table_dense
