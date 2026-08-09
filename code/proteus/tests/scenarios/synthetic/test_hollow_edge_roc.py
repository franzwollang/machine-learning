"""ROC / adversarial-null tests for hollow-edge ``h_0`` calibration (#44, A4-T18).

Evidence-gathering harness only — does **not** wire production prepass flags
or flip linked_tori / nested_spheres ``@awaiting`` recovery tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.scenarios.synthetic.hollow_edge_nulls import (
    gabriel_is_hollow,
    generate_connected_sheet_null,
    generate_gap_tissue_cases,
    generate_gap_tissue_rate_sweep,
    hollowness_ratio,
    pooled_adversarial_roc,
    roc_from_cases,
    score_edge,
)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_hollowness_ratio_bridge_vs_within_at_zero_tissue() -> None:
    """Zero-tissue gap: bridges are near-hollow; within-blob edges are not."""
    cases = generate_gap_tissue_cases(tissue_rate=0.0, seed=3)
    bridges = [c for c in cases if c.kind == "gap_bridge"]
    withins = [c for c in cases if c.kind == "blob_within"]
    assert bridges and withins
    h_bridge = np.mean([score_edge(c) for c in bridges])
    h_within = np.mean([score_edge(c) for c in withins])
    assert h_bridge < 0.25
    assert h_within > 0.6
    assert h_bridge < h_within


@pytest.mark.scenario
@pytest.mark.synthetic
def test_connected_sheet_null_edges_are_not_hollow() -> None:
    """Density-gradient curved sheet: within-support edges stay O(1) H."""
    cases = generate_connected_sheet_null(seed=5, density_power=1.8, curvature=0.4)
    assert len(cases) >= 15
    scores = np.asarray([score_edge(c) for c in cases], dtype=float)
    assert float(np.median(scores)) > 0.45
    # Allow a sparse-tail under the density gradient; majority stay occupied.
    assert float(np.mean(scores > 0.25)) > 0.75


@pytest.mark.scenario
@pytest.mark.synthetic
def test_pooled_adversarial_roc_separates_sheet_from_bridges() -> None:
    """ROC AUC ≫ chance for sheet negatives vs zero-tissue bridges."""
    roc = pooled_adversarial_roc(sheet_seed=1, gap_seed=11, tissue_rate_for_positives=0.0)
    assert roc.labels_should_cut.sum() >= 4
    assert (1 - roc.labels_should_cut).sum() >= 10
    assert roc.auc >= 0.85
    # Best Youden-style point on the grid (not a seed-tuned fixture h_0).
    best = max(roc.points, key=lambda p: p.tpr - p.fpr)
    assert best.tpr >= 0.85
    assert best.fpr <= 0.25
    assert best.tpr - best.fpr >= 0.65

@pytest.mark.scenario
@pytest.mark.synthetic
def test_gap_tissue_rate_degrades_bridge_hollowness() -> None:
    """As gap tissue → signal density, bridge H rises (must-cut softens)."""
    sweep = generate_gap_tissue_rate_sweep(
        tissue_rates=(0.0, 0.25, 0.6, 1.0),
        seed=7,
    )
    mean_bridge_h: list[tuple[float, float]] = []
    for rate, cases in sweep.items():
        bridges = [c for c in cases if c.kind == "gap_bridge"]
        mean_h = float(np.mean([score_edge(c) for c in bridges]))
        mean_bridge_h.append((rate, mean_h))
    # Monotone-ish: last (tissue≈signal) clearly above first (empty gap).
    assert mean_bridge_h[0][1] < 0.25
    assert mean_bridge_h[-1][1] > mean_bridge_h[0][1] + 0.2


@pytest.mark.scenario
@pytest.mark.synthetic
def test_roc_harness_reports_threshold_grid() -> None:
    """Harness returns thresholds + confusion counts for calibration tables."""
    cases = generate_gap_tissue_cases(tissue_rate=0.0, seed=2) + generate_connected_sheet_null(
        seed=4, n_u=6, n_v=5, n_data=400
    )
    roc = roc_from_cases(cases, thresholds=[0.05, 0.15, 0.35, 0.6, 1.0])
    assert len(roc.points) == 5
    assert roc.points[0].threshold == pytest.approx(0.05)
    # Extreme: tiny h_0 → few cuts; huge h_0 → many cuts.
    assert roc.points[0].tp + roc.points[0].fp <= roc.points[-1].tp + roc.points[-1].fp


@pytest.mark.scenario
@pytest.mark.synthetic
def test_gabriel_fallback_agrees_on_empty_bridge() -> None:
    """Low-n fallback: empty-gap bridge is Gabriel-hollow; within-disk often not."""
    cases = generate_gap_tissue_cases(tissue_rate=0.0, seed=9, n_bridge=4, n_within=6)
    bridge_hollow = [
        gabriel_is_hollow(c.points, c.endpoint_i, c.endpoint_j)
        for c in cases
        if c.kind == "gap_bridge"
    ]
    within_h = [
        hollowness_ratio(c.points, c.endpoint_i, c.endpoint_j)
        for c in cases
        if c.kind == "blob_within"
    ]
    assert bridge_hollow and all(bridge_hollow)
    assert within_h and float(np.median(within_h)) > 0.4

@pytest.mark.scenario
@pytest.mark.synthetic
def test_mid_frac_035_roc_still_separates() -> None:
    """A2 operational mid_frac=0.35 keeps AUC ≫ chance on adversarial pool."""
    from tests.scenarios.synthetic.hollow_edge_nulls import (
        A2_MID_RADIUS_FRAC,
        mid_frac_roc_table,
    )

    table = mid_frac_roc_table(mid_fracs=(0.25, A2_MID_RADIUS_FRAC))
    assert table[0.25].auc >= 0.85
    assert table[A2_MID_RADIUS_FRAC].auc >= 0.85


@pytest.mark.scenario
@pytest.mark.synthetic
def test_a2_parity_decision_controls_fpr_on_sheet() -> None:
    """A2 (mid=0.35,h0=0.35)+Gabriel: high TPR on empty bridges, low sheet FPR."""
    from tests.scenarios.synthetic.hollow_edge_nulls import (
        A2_H0,
        A2_MID_RADIUS_FRAC,
        confusion_at_h0,
        generate_connected_sheet_null,
        generate_gap_tissue_cases,
    )

    sheet = generate_connected_sheet_null(seed=5, density_power=1.8, curvature=0.4)
    gap = generate_gap_tissue_cases(tissue_rate=0.0, seed=11)
    bridges = [c for c in gap if c.kind == "gap_bridge"]
    withins = [c for c in gap if c.kind == "blob_within"]
    pooled = list(sheet) + bridges + withins
    conf = confusion_at_h0(
        pooled,
        h0=A2_H0,
        mid_radius_frac=A2_MID_RADIUS_FRAC,
        use_a2_parity_decision=True,
    )
    assert conf.tpr >= 0.85
    assert conf.fpr <= 0.25


@pytest.mark.scenario
@pytest.mark.synthetic
def test_sheet_poisson_null_h0_below_lower_tail() -> None:
    """Sheet H quantiles: A2 h0 sits below empirical lower-tail null mass.

    Does not seed-tune a single fixture h0 — asserts structural ordering of
    the Poisson-ish sheet null vs the operational cut threshold (h0 under
    the 1% sheet H ⇒ near-zero H-only FPR on this adversarial null).
    """
    from tests.scenarios.synthetic.hollow_edge_nulls import (
        A2_H0,
        A2_MID_RADIUS_FRAC,
        sheet_null_h_quantiles,
    )

    null_l4 = sheet_null_h_quantiles(mid_radius_frac=0.25, seed=3)
    null_a2 = sheet_null_h_quantiles(mid_radius_frac=A2_MID_RADIUS_FRAC, seed=3)
    # Homogeneous-ish support: mean H is O(1), not near 0.
    assert null_l4.mean_h > 0.4
    assert null_a2.mean_h > 0.4
    assert null_a2.mean_end_mass > 0.5
    # Operational h0 should sit under the sheet median (null mass is O(1)).
    assert A2_H0 < null_a2.quantiles["q0.5"]
    assert null_a2.quantiles["q0.05"] < null_a2.quantiles["q0.5"]
    # Measured: A2 h0 sits *below* the empirical 1% sheet H — FPR≈0 via H
    # alone on this null (gabriel gate unused for decidable n_end).
    assert A2_H0 <= null_a2.quantiles["q0.01"]
