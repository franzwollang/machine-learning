"""Denser fitted-scaffold coverage probe on linked tori (#41, A4-T26).

Parallel to ``test_ph_nested_fitted_coverage``: vary
``ScaleSearchConfig.max_nodes`` on ``make_linked_tori``, label nodes by
nearest data, run per-torus VR-PH at SI ``1.5 σ*`` fixed_threshold toward
``(1, 2, 1)``.

Evidence-gathering only — does **not** flip ``test_linked_tori_betti_numbers``
``@awaiting``.
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


@dataclass(frozen=True)
class LinkedToriFittedCoverageRow:
    max_nodes: int
    n_nodes: int
    n_signal: int
    n_per_torus: dict[int, int]
    sigma_star: float
    betti_per_torus: dict[int, tuple[int, ...]]
    all_match_si: bool


def _fit_linked_tori_coverage(max_nodes: int) -> LinkedToriFittedCoverageRow:
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
    # Signal tori are labels 0 and 1 in make_linked_tori; tissue = -1.
    signal_mask = np.isin(node_labels, [0, 1])
    signal_pos = pos[signal_mask]
    signal_labs = node_labels[signal_mask]
    n_per = {int(lab): int(np.sum(signal_labs == lab)) for lab in (0, 1)}
    report = run_per_region_ph(
        signal_pos,
        signal_labs,
        [sigma, sigma],
        scenario=f"linked_tori_fitted_coverage_max_nodes_{max_nodes}",
        include_labels=[0, 1],
        reading="fixed_threshold",
        max_dim=2,
        filtration_mult=FILTRATION_MULTIPLIER,
        expected_betti=(1, 2, 1),
    )
    betti = {int(r.region_id): tuple(int(x) for x in r.betti) for r in report.reports}
    return LinkedToriFittedCoverageRow(
        max_nodes=int(max_nodes),
        n_nodes=int(pos.shape[0]),
        n_signal=int(signal_pos.shape[0]),
        n_per_torus=n_per,
        sigma_star=float(sigma),
        betti_per_torus=betti,
        all_match_si=bool(report.all_match),
    )


@pytest.fixture(scope="module")
def linked_tori_fitted_coverage_table() -> dict[int, LinkedToriFittedCoverageRow]:
    """Fit once per ``max_nodes`` for the linked-tori coverage table."""
    return {m: _fit_linked_tori_coverage(m) for m in (64, 128, 256)}


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_fitted_coverage_reports_node_betti_pairs(
    linked_tori_fitted_coverage_table,
) -> None:
    """Coverage table: denser max_nodes increases signal nodes; record Betti."""
    rows = linked_tori_fitted_coverage_table
    assert set(rows) == {64, 128, 256}
    for _m, row in rows.items():
        assert row.n_nodes > 0
        assert row.n_signal > 0
        assert set(row.betti_per_torus) == {0, 1}
        for lab in (0, 1):
            assert len(row.betti_per_torus[lab]) == 3
            assert row.n_per_torus[lab] >= 0
        assert FILTRATION_MULTIPLIER == 1.5

    assert rows[64].n_signal <= rows[128].n_signal <= rows[256].n_signal


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_fitted_coverage_documents_si_gap(
    linked_tori_fitted_coverage_table,
) -> None:
    """Document whether denser coverage alone recovers per-torus (1,2,1).

    Soft evidence gate — never flips ``@awaiting`` recovery tests.
    """
    rows = linked_tori_fitted_coverage_table
    recovered = [m for m, r in rows.items() if r.all_match_si]
    if recovered:
        assert max(recovered) >= min(recovered)
        assert any(rows[m].betti_per_torus[0] == (1, 2, 1) for m in recovered)
    else:
        for _m, row in rows.items():
            assert row.all_match_si is False
            assert any(b != (1, 2, 1) for b in row.betti_per_torus.values())
