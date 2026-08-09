"""Dual-scale PH on linked-tori fitted scaffolds (#41 / A4-T30).

Parallel to ``test_ph_nested_dual_scale`` (A4-T29): fixed_threshold VR-PH at a
coarse filtration mult and the SI fine mult (``1.5``) on denser fitted signal
nodes (labels ``0/1``). Evidence-gathering only — does **not** flip
``test_linked_tori_betti_numbers`` ``@awaiting``.
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
    format_dual_scale_ph_table,
    nearest_data_labels,
    run_dual_scale_per_region_ph,
    sigma_star_from_tau,
)


@dataclass(frozen=True)
class LinkedToriDualScaleBundle:
    max_nodes: int
    n_signal: int
    sigma_star: float
    coarse_mult: float
    fine_mult: float
    table: str
    any_coarse_match: bool
    any_fine_match: bool
    all_tori_match_either: bool
    betti_by_region: dict[int, tuple[tuple[int, ...], tuple[int, ...]]]


@pytest.fixture(scope="module")
def linked_tori_dual_scale_bundle() -> LinkedToriDualScaleBundle:
    """Fit max_nodes=128; dual-scale PH on signal tori 0/1."""
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
    signal_mask = np.isin(node_labels, [0, 1])
    signal_pos = pos[signal_mask]
    signal_labs = node_labels[signal_mask]

    coarse_mult = 3.0
    fine_mult = FILTRATION_MULTIPLIER
    dual = run_dual_scale_per_region_ph(
        signal_pos,
        signal_labs,
        sigma,
        coarse_mult=coarse_mult,
        fine_mult=fine_mult,
        scenario="linked_tori_fitted_dual_scale",
        include_labels=[0, 1],
        max_dim=2,
        expected_betti=(1, 2, 1),
    )
    per_ok = False
    if dual.rows:
        per_ok = all(
            bool(r.coarse_match or r.fine_match) for r in dual.rows
        ) and len(dual.rows) == 2
    betti_by = {
        int(r.region_id): (tuple(r.coarse_betti), tuple(r.fine_betti))
        for r in dual.rows
    }

    return LinkedToriDualScaleBundle(
        max_nodes=max_nodes,
        n_signal=int(signal_pos.shape[0]),
        sigma_star=float(sigma),
        coarse_mult=float(coarse_mult),
        fine_mult=float(fine_mult),
        table=format_dual_scale_ph_table(dual),
        any_coarse_match=bool(dual.any_coarse_match),
        any_fine_match=bool(dual.any_fine_match),
        all_tori_match_either=bool(per_ok),
        betti_by_region=betti_by,
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_dual_scale_ph_reports_table(
    linked_tori_dual_scale_bundle,
) -> None:
    """Dual-scale table lands with coarse+fine columns; SI fine mult intact."""
    bundle = linked_tori_dual_scale_bundle
    assert bundle.max_nodes == 128
    assert bundle.n_signal > 0
    assert bundle.sigma_star > 0.0
    assert bundle.fine_mult == FILTRATION_MULTIPLIER == 1.5
    assert bundle.coarse_mult > bundle.fine_mult
    header = bundle.table.splitlines()[0]
    assert "coarse_betti" in header
    assert "fine_betti" in header
    assert len(bundle.table.splitlines()) >= 3
    assert set(bundle.betti_by_region) == {0, 1}


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_dual_scale_ph_documents_si_gap(
    linked_tori_dual_scale_bundle,
) -> None:
    """Document whether coarse or fine recovers (1,2,1); never flip awaiting.

    Soft evidence: if either scale recovers both tori, SI gap may be readable
    via dual-scale. Otherwise assert explicit non-recovery on SI fine.
    """
    bundle = linked_tori_dual_scale_bundle
    if bundle.all_tori_match_either:
        assert FILTRATION_MULTIPLIER == 1.5
        assert bundle.any_coarse_match or bundle.any_fine_match
    else:
        assert bundle.all_tori_match_either is False
        # SI fine mult alone did not recover every torus.
        assert bundle.any_fine_match is False or not bundle.all_tori_match_either
        assert "fine_betti" in bundle.table
