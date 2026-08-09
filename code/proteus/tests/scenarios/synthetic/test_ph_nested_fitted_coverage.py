"""Denser fitted-scaffold coverage probe on nested spheres (#41, related A4-T19).

Varies ``ScaleSearchConfig.max_nodes`` on ``make_nested_spheres`` and reads
per-shell Betti via NN-to-data labels + SI ``1.5 σ*`` fixed_threshold.
Evidence-gathering only — does **not** flip ``test_nested_spheres_topology``
``@awaiting`` or change SI filtration defaults.

Circle finding (A4-T19): denser ``max_nodes≥128`` recovers SI on fitted
signal nodes. This probe asks whether the same coverage lever helps nested
S² shells toward per-shell ``(1,0,1)``.
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
    nearest_data_labels,
    run_per_region_ph,
    sigma_star_from_tau,
)


@dataclass(frozen=True)
class NestedFittedCoverageRow:
    max_nodes: int
    n_nodes: int
    n_signal: int
    n_per_shell: dict[int, int]
    sigma_star: float
    betti_per_shell: dict[int, tuple[int, ...]]
    all_match_si: bool


def _fit_nested_coverage(max_nodes: int) -> NestedFittedCoverageRow:
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
    # Signal shells are labels 1 and 2 in make_nested_spheres; tissue < 0.
    signal_mask = np.isin(node_labels, [1, 2])
    signal_pos = pos[signal_mask]
    signal_labs = node_labels[signal_mask]
    n_per_shell = {
        int(lab): int(np.sum(signal_labs == lab)) for lab in (1, 2)
    }
    # Per-shell sigma proxy: same tau* for both (scale-search single star);
    # clean-shell probes used sigma ∝ radius — here we document SI default
    # on fitted coverage only.
    report = run_per_region_ph(
        signal_pos,
        signal_labs,
        [sigma, sigma],
        scenario=f"nested_fitted_coverage_max_nodes_{max_nodes}",
        include_labels=[1, 2],
        reading="fixed_threshold",
        max_dim=2,
        filtration_mult=FILTRATION_MULTIPLIER,
        expected_betti=(1, 0, 1),
    )
    betti = {int(r.region_id): tuple(int(x) for x in r.betti) for r in report.reports}
    return NestedFittedCoverageRow(
        max_nodes=int(max_nodes),
        n_nodes=int(pos.shape[0]),
        n_signal=int(signal_pos.shape[0]),
        n_per_shell=n_per_shell,
        sigma_star=float(sigma),
        betti_per_shell=betti,
        all_match_si=bool(report.all_match),
    )


@pytest.fixture(scope="module")
def nested_fitted_coverage_table() -> dict[int, NestedFittedCoverageRow]:
    """Fit once per ``max_nodes`` for the nested coverage table."""
    return {m: _fit_nested_coverage(m) for m in (64, 128, 256)}


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_fitted_coverage_reports_node_betti_pairs(
    nested_fitted_coverage_table,
) -> None:
    """Coverage table: denser max_nodes increases signal nodes; record Betti.

    Does not require SI recovery yet — nested fitted shells remain harder
    than the circle (tissue + two radii). Assert structural reporting only
    and that denser caps do not shrink signal occupancy.
    """
    rows = nested_fitted_coverage_table
    assert set(rows) == {64, 128, 256}
    for m, row in rows.items():
        assert row.n_nodes > 0
        assert row.n_signal > 0
        assert set(row.betti_per_shell) == {1, 2}
        for lab in (1, 2):
            assert len(row.betti_per_shell[lab]) == 3
            assert row.n_per_shell[lab] >= 0
        # Filtration multiplier stays at SI default in the probe.
        assert FILTRATION_MULTIPLIER == 1.5

    # Monotone-ish signal occupancy under denser caps (allow plateaus).
    assert rows[64].n_signal <= rows[128].n_signal <= rows[256].n_signal


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_fitted_coverage_documents_si_gap(
    nested_fitted_coverage_table,
) -> None:
    """Document whether denser coverage alone recovers per-shell (1,0,1).

    Soft evidence gate: if any row recovers both shells, denser coverage is
    a viable reading path (mirrors circle A4-T19). If none recover, the
    table still lands — nested awaits more than max_nodes (lifetime /
    hollow-edge / dual-scale). Never flips ``@awaiting``.
    """
    rows = nested_fitted_coverage_table
    recovered = [m for m, r in rows.items() if r.all_match_si]
    # Always recordable; prefer denser when recovery appears.
    if recovered:
        assert max(recovered) >= min(recovered)
        assert any(rows[m].betti_per_shell[1] == (1, 0, 1) for m in recovered)
    else:
        # Explicit non-recovery documentation for tracker — still green.
        for m, row in rows.items():
            assert row.all_match_si is False
            # At least one shell fails SI target under current recipe.
            assert any(b != (1, 0, 1) for b in row.betti_per_shell.values())
