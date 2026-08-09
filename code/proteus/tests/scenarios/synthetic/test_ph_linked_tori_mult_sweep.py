"""Uniform filtration_mult sweep on linked-tori fitted scaffolds (#41).

Prior crossed schedules (3/6) and denser max_nodes left tori b1 stuck at 0|1.
This harness sweeps mults **below and above** the coarse=3 / cal=6 pair,
including SI 1.5 and sub-3 values, hunting for any region with ``b1=2`` or
full ``(1, 2, 1)`` under fixed_threshold.

Evidence-gathering only — does **not** flip ``test_linked_tori_betti_numbers``
``@awaiting`` or change SI ``FILTRATION_MULTIPLIER``.
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
    format_filtration_mult_sweep_table,
    nearest_data_labels,
    sigma_star_from_tau,
    sweep_filtration_mult_per_region,
)

EXPECTED_TORI: tuple[int, ...] = (1, 2, 1)
# Include sub-3 values suggested after A4-T34/T35 schedule failures.
MULT_GRID: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)


@dataclass(frozen=True)
class LinkedToriMultSweepBundle:
    max_nodes: int
    n_signal: int
    sigma_star: float
    mults: tuple[float, ...]
    table: str
    any_full_match: bool
    any_b1_eq_2: bool
    max_b1_seen: int
    betti_by_region_mult: dict[tuple[int, float], tuple[int, ...]]
    recovering_cells: tuple[tuple[int, float, tuple[int, ...]], ...]


@pytest.fixture(scope="module")
def linked_tori_mult_sweep_bundle() -> LinkedToriMultSweepBundle:
    """Fit max_nodes=128; sweep filtration_mult per torus under global sigma."""
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

    sweep = sweep_filtration_mult_per_region(
        signal_pos,
        signal_labs,
        sigma,
        mults=MULT_GRID,
        scenario="linked_tori_filtration_mult_sweep",
        include_labels=[0, 1],
        reading="fixed_threshold",
        max_dim=2,
        expected_betti=EXPECTED_TORI,
        b1_target=2,
    )

    betti_map: dict[tuple[int, float], tuple[int, ...]] = {
        (int(r.region_id), float(r.filtration_mult)): tuple(r.betti)
        for r in sweep.rows
    }
    recovering = tuple(
        (int(r.region_id), float(r.filtration_mult), tuple(r.betti))
        for r in sweep.rows
        if r.match is True or r.b1 == 2
    )
    max_b1 = max((int(r.b1) for r in sweep.rows), default=0)

    return LinkedToriMultSweepBundle(
        max_nodes=max_nodes,
        n_signal=int(signal_pos.shape[0]),
        sigma_star=float(sigma),
        mults=tuple(float(m) for m in MULT_GRID),
        table=format_filtration_mult_sweep_table(sweep),
        any_full_match=bool(sweep.any_full_match),
        any_b1_eq_2=bool(sweep.any_b1_target),
        max_b1_seen=int(max_b1),
        betti_by_region_mult=betti_map,
        recovering_cells=recovering,
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_mult_sweep_harness_lands(
    linked_tori_mult_sweep_bundle,
) -> None:
    """Sweep table lands across mult grid; SI fine mult constant untouched."""
    bundle = linked_tori_mult_sweep_bundle
    assert bundle.max_nodes == 128
    assert bundle.n_signal > 0
    assert bundle.sigma_star > 0.0
    assert FILTRATION_MULTIPLIER == 1.5
    assert bundle.mults == MULT_GRID
    assert "b1" in bundle.table.splitlines()[0]
    # 2 regions × len(MULT_GRID) rows + header
    assert len(bundle.table.splitlines()) == 1 + 2 * len(MULT_GRID)
    assert (0, 1.5) in bundle.betti_by_region_mult
    assert (1, 2.5) in bundle.betti_by_region_mult


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_mult_sweep_documents_gap(
    linked_tori_mult_sweep_bundle,
) -> None:
    """Document whether any mult yields b1=2 / (1,2,1); never flip awaiting.

    Soft gate: any recovering cell is proposal-path evidence. Otherwise keep
    explicit max_b1 < 2 across the sub-3..calibrated grid.
    """
    bundle = linked_tori_mult_sweep_bundle
    if bundle.any_full_match or bundle.any_b1_eq_2:
        assert FILTRATION_MULTIPLIER == 1.5
        assert len(bundle.recovering_cells) >= 1
        assert bundle.max_b1_seen >= 2
    else:
        assert bundle.any_full_match is False
        assert bundle.any_b1_eq_2 is False
        assert bundle.max_b1_seen < 2
        assert bundle.recovering_cells == ()
        assert "betti" in bundle.table
