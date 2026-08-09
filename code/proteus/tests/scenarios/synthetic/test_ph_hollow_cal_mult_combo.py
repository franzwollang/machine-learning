"""Hollow-prune + circle-calibrated mult combo on nested/tori fitted PH (#41 / A4-T35).

Combines A4 primary hollow prune (mid=0.5, h0=0.7, no Gabriel) with the
circle-calibrated ``filtration_mult=6`` on denser fitted signal nodes.
Compares against SI mult and cal-mult-only baselines.

Evidence-gathering only — does **not** flip ``@awaiting`` recovery tests or
change SI ``FILTRATION_MULTIPLIER``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.edge_evidence import HollowEdgeConfig, prune_hollow_edges
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.linked_tori import make_linked_tori
from tests.datasets.synthetic.nested_spheres import make_nested_spheres
from tests.metrics.persistent_homology import (
    FILTRATION_MULTIPLIER,
    format_per_region_ph_diagnostics,
    nearest_data_labels,
    run_per_region_ph,
    sigma_star_from_tau,
)

CIRCLE_CALIBRATED_MULT: float = 6.0


def _undirected_edges_from_adj(adj: dict) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i, nbrs in adj.items():
        for j in nbrs:
            a, b = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
            if a != b:
                edges.add((a, b))
    return sorted(edges)


def _hollow_pruned_node_mask(
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    data: np.ndarray,
    *,
    config: HollowEdgeConfig | None = None,
) -> np.ndarray:
    """True for nodes that retain ≥1 non-hollow neighbour edge."""
    n = int(positions.shape[0])
    keep = np.zeros(n, dtype=bool)
    if not edges:
        return keep
    surviving = prune_hollow_edges(positions, edges, data, config=config)
    for i, j in surviving:
        keep[int(i)] = True
        keep[int(j)] = True
    return keep


@dataclass(frozen=True)
class HollowCalMultRow:
    scenario: str
    n_signal: int
    n_after_hollow: int
    hollow_fallback: bool
    sigma_star: float
    si_diag: str
    cal_diag: str
    hollow_cal_diag: str
    si_all_match: bool | None
    cal_all_match: bool | None
    hollow_cal_all_match: bool | None


def _fit_scaffold(
    points: np.ndarray,
    ambient_dim: int,
    tau_lo: float,
    tau_hi: float,
    max_nodes: int = 128,
):
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
    result = run_scale_search(points, dim=ambient_dim, config=config)
    scaffold = result.scaffold_at_star
    pos = scaffold.node_positions()
    sigma = sigma_star_from_tau(result.tau_star)
    return scaffold, pos, float(sigma)


@pytest.fixture(scope="module")
def hollow_cal_mult_rows() -> tuple[HollowCalMultRow, ...]:
    """Fit nested + tori; compare SI / cal / hollow+cal fixed_threshold PH."""
    hollow_cfg = HollowEdgeConfig(
        mid_radius_frac=0.5, h0=0.7, min_end_count=0.5, gabriel_fallback=False,
    )
    rows: list[HollowCalMultRow] = []

    # --- nested spheres ---
    nested = make_nested_spheres(
        n_per_sphere=500,
        radii=(1.0, 2.0),
        ambient_dim=3,
        noise=0.02,
        tissue_fraction=0.03,
        seed=21,
    )
    gt = nested.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    scaffold, pos, sigma = _fit_scaffold(
        nested.points, gt.ambient_dim, tau_lo, tau_hi,
    )
    node_labels = nearest_data_labels(pos, nested.points, nested.labels)
    include = [1, 2]
    expected = (1, 0, 1)
    signal_mask = np.isin(node_labels, include)
    signal_pos = pos[signal_mask]
    signal_labs = node_labels[signal_mask]

    adj = scaffold.links.neighbour_graph(pos.shape[0])
    edges = _undirected_edges_from_adj(adj)
    hollow_keep = _hollow_pruned_node_mask(
        pos, edges, nested.points, config=hollow_cfg,
    )
    hollow_signal = signal_mask & hollow_keep
    n_after = int(np.sum(hollow_signal))
    if n_after < 8 or not all(
        np.any(node_labels[hollow_signal] == lab) for lab in include
    ):
        hollow_pos, hollow_labs = signal_pos, signal_labs
        hollow_fallback = True
    else:
        hollow_pos = pos[hollow_signal]
        hollow_labs = node_labels[hollow_signal]
        hollow_fallback = False

    si = run_per_region_ph(
        signal_pos, signal_labs, sigma, scenario="nested_si",
        include_labels=include, reading="fixed_threshold", max_dim=2,
        filtration_mult=FILTRATION_MULTIPLIER, expected_betti=expected,
    )
    cal = run_per_region_ph(
        signal_pos, signal_labs, sigma, scenario="nested_cal",
        include_labels=include, reading="fixed_threshold", max_dim=2,
        filtration_mult=CIRCLE_CALIBRATED_MULT, expected_betti=expected,
    )
    hollow_cal = run_per_region_ph(
        hollow_pos, hollow_labs, sigma, scenario="nested_hollow_cal",
        include_labels=include, reading="fixed_threshold", max_dim=2,
        filtration_mult=CIRCLE_CALIBRATED_MULT, expected_betti=expected,
    )
    rows.append(
        HollowCalMultRow(
            scenario="nested_spheres",
            n_signal=int(signal_pos.shape[0]),
            n_after_hollow=n_after,
            hollow_fallback=bool(hollow_fallback),
            sigma_star=sigma,
            si_diag=format_per_region_ph_diagnostics(si),
            cal_diag=format_per_region_ph_diagnostics(cal),
            hollow_cal_diag=format_per_region_ph_diagnostics(hollow_cal),
            si_all_match=si.all_match,
            cal_all_match=cal.all_match,
            hollow_cal_all_match=hollow_cal.all_match,
        )
    )

    # --- linked tori ---
    tori = make_linked_tori(
        n_per_torus=500,
        major_radius=2.0,
        minor_radius=0.5,
        noise=0.02,
        tissue_fraction=0.03,
        seed=21,
    )
    gt = tori.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    scaffold, pos, sigma = _fit_scaffold(
        tori.points, gt.ambient_dim, tau_lo, tau_hi,
    )
    node_labels = nearest_data_labels(pos, tori.points, tori.labels)
    include = [0, 1]
    expected = (1, 2, 1)
    signal_mask = np.isin(node_labels, include)
    signal_pos = pos[signal_mask]
    signal_labs = node_labels[signal_mask]

    adj = scaffold.links.neighbour_graph(pos.shape[0])
    edges = _undirected_edges_from_adj(adj)
    hollow_keep = _hollow_pruned_node_mask(
        pos, edges, tori.points, config=hollow_cfg,
    )
    hollow_signal = signal_mask & hollow_keep
    n_after = int(np.sum(hollow_signal))
    if n_after < 8 or not all(
        np.any(node_labels[hollow_signal] == lab) for lab in include
    ):
        hollow_pos, hollow_labs = signal_pos, signal_labs
        hollow_fallback = True
    else:
        hollow_pos = pos[hollow_signal]
        hollow_labs = node_labels[hollow_signal]
        hollow_fallback = False

    si = run_per_region_ph(
        signal_pos, signal_labs, sigma, scenario="tori_si",
        include_labels=include, reading="fixed_threshold", max_dim=2,
        filtration_mult=FILTRATION_MULTIPLIER, expected_betti=expected,
    )
    cal = run_per_region_ph(
        signal_pos, signal_labs, sigma, scenario="tori_cal",
        include_labels=include, reading="fixed_threshold", max_dim=2,
        filtration_mult=CIRCLE_CALIBRATED_MULT, expected_betti=expected,
    )
    hollow_cal = run_per_region_ph(
        hollow_pos, hollow_labs, sigma, scenario="tori_hollow_cal",
        include_labels=include, reading="fixed_threshold", max_dim=2,
        filtration_mult=CIRCLE_CALIBRATED_MULT, expected_betti=expected,
    )
    rows.append(
        HollowCalMultRow(
            scenario="linked_tori",
            n_signal=int(signal_pos.shape[0]),
            n_after_hollow=n_after,
            hollow_fallback=bool(hollow_fallback),
            sigma_star=sigma,
            si_diag=format_per_region_ph_diagnostics(si),
            cal_diag=format_per_region_ph_diagnostics(cal),
            hollow_cal_diag=format_per_region_ph_diagnostics(hollow_cal),
            si_all_match=si.all_match,
            cal_all_match=cal.all_match,
            hollow_cal_all_match=hollow_cal.all_match,
        )
    )
    return tuple(rows)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_hollow_cal_mult_harness_lands(hollow_cal_mult_rows) -> None:
    """SI / cal / hollow+cal diagnostics land for nested and tori."""
    by = {r.scenario: r for r in hollow_cal_mult_rows}
    assert set(by) == {"nested_spheres", "linked_tori"}
    for row in hollow_cal_mult_rows:
        assert row.n_signal > 0
        assert row.sigma_star > 0.0
        assert "filtration_mult=1.5" in row.si_diag
        assert f"filtration_mult={CIRCLE_CALIBRATED_MULT:g}" in row.cal_diag
        assert f"filtration_mult={CIRCLE_CALIBRATED_MULT:g}" in row.hollow_cal_diag
        assert FILTRATION_MULTIPLIER == 1.5


@pytest.mark.scenario
@pytest.mark.synthetic
def test_hollow_cal_mult_documents_gap(hollow_cal_mult_rows) -> None:
    """Document whether hollow+cal recovers; never flip awaiting.

    Soft gate: if hollow+cal recovers all regions, combo is viable evidence.
    Otherwise assert SI / cal baselines remain non-recovering when combo fails.
    """
    for row in hollow_cal_mult_rows:
        if row.hollow_cal_all_match:
            assert FILTRATION_MULTIPLIER == 1.5
            assert CIRCLE_CALIBRATED_MULT == 6.0
        else:
            assert row.hollow_cal_all_match is False
            # When combo fails, SI path is still incomplete.
            assert row.si_all_match is False
