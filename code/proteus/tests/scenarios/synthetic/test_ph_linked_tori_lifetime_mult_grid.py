"""lifetime_frac × filtration_mult grid on linked-tori fitted scaffolds (#41).

Prior mult-only sweeps (A4-T36) left ``max_b1=1`` under fixed_threshold.
Circle calibration needed ``mult≥6`` *and* ``lifetime_frac≥4`` for ``(1,1)`` —
this harness crosses both levers on fitted tori under the lifetime reading,
hunting for ``b1=2`` or full ``(1, 2, 1)``.

Evidence-gathering only — does **not** flip ``test_linked_tori_betti_numbers``
``@awaiting`` or change SI ``FILTRATION_MULTIPLIER`` / ``DEFAULT_LIFETIME_FRAC``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.linked_tori import make_linked_tori
from tests.metrics.persistent_homology import (
    DEFAULT_LIFETIME_FRAC,
    FILTRATION_MULTIPLIER,
    format_lifetime_mult_grid_table,
    nearest_data_labels,
    sigma_star_from_tau,
    sweep_lifetime_mult_grid_per_region,
)

EXPECTED_TORI: tuple[int, ...] = (1, 2, 1)
# Fracs span SI default through circle-recovery floor (≥4).
FRAC_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
# Mults span SI / coarse / circle-calibrated / above.
MULT_GRID: tuple[float, ...] = (1.5, 3.0, 6.0, 8.0)


@dataclass(frozen=True)
class LinkedToriLifetimeMultGridBundle:
    max_nodes: int
    n_signal: int
    sigma_star: float
    fracs: tuple[float, ...]
    mults: tuple[float, ...]
    table: str
    any_full_match: bool
    any_b1_eq_2: bool
    max_b1_seen: int
    betti_by_cell: dict[tuple[int, float, float], tuple[int, ...]]
    recovering_cells: tuple[tuple[int, float, float, tuple[int, ...]], ...]


@pytest.fixture(scope="module")
def linked_tori_lifetime_mult_grid_bundle() -> LinkedToriLifetimeMultGridBundle:
    """Fit max_nodes=128; cross lifetime_frac × filtration_mult per torus."""
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

    grid = sweep_lifetime_mult_grid_per_region(
        signal_pos,
        signal_labs,
        sigma,
        fracs=FRAC_GRID,
        mults=MULT_GRID,
        scenario="linked_tori_lifetime_mult_grid",
        include_labels=[0, 1],
        max_dim=2,
        expected_betti=EXPECTED_TORI,
        b1_target=2,
    )

    betti_map: dict[tuple[int, float, float], tuple[int, ...]] = {
        (int(r.region_id), float(r.lifetime_frac), float(r.filtration_mult)): (
            tuple(r.betti)
        )
        for r in grid.rows
    }
    recovering = tuple(
        (
            int(r.region_id),
            float(r.lifetime_frac),
            float(r.filtration_mult),
            tuple(r.betti),
        )
        for r in grid.rows
        if r.match is True or r.b1 == 2
    )
    max_b1 = max((int(r.b1) for r in grid.rows), default=0)

    return LinkedToriLifetimeMultGridBundle(
        max_nodes=max_nodes,
        n_signal=int(signal_pos.shape[0]),
        sigma_star=float(sigma),
        fracs=tuple(float(f) for f in FRAC_GRID),
        mults=tuple(float(m) for m in MULT_GRID),
        table=format_lifetime_mult_grid_table(grid),
        any_full_match=bool(grid.any_full_match),
        any_b1_eq_2=bool(grid.any_b1_target),
        max_b1_seen=int(max_b1),
        betti_by_cell=betti_map,
        recovering_cells=recovering,
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_lifetime_mult_grid_harness_lands(
    linked_tori_lifetime_mult_grid_bundle,
) -> None:
    """Grid table lands; SI defaults untouched."""
    bundle = linked_tori_lifetime_mult_grid_bundle
    assert bundle.max_nodes == 128
    assert bundle.n_signal > 0
    assert bundle.sigma_star > 0.0
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.fracs == FRAC_GRID
    assert bundle.mults == MULT_GRID
    header = bundle.table.splitlines()[0]
    assert "frac" in header and "mult" in header and "b1" in header
    # 2 regions × |fracs| × |mults| + header
    assert len(bundle.table.splitlines()) == 1 + 2 * len(FRAC_GRID) * len(MULT_GRID)
    assert (0, 0.5, 1.5) in bundle.betti_by_cell
    assert (1, 4.0, 6.0) in bundle.betti_by_cell


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_lifetime_mult_grid_documents_gap(
    linked_tori_lifetime_mult_grid_bundle,
) -> None:
    """Document whether any (frac, mult) yields b1=2 / (1,2,1); never flip awaiting.

    Soft gate: recovering cells are proposal-path evidence. Otherwise keep an
    explicit max_b1 < 2 across the crossed grid.
    """
    bundle = linked_tori_lifetime_mult_grid_bundle
    if bundle.any_full_match or bundle.any_b1_eq_2:
        assert FILTRATION_MULTIPLIER == 1.5
        assert DEFAULT_LIFETIME_FRAC == 0.5
        assert len(bundle.recovering_cells) >= 1
        assert bundle.max_b1_seen >= 2
    else:
        assert bundle.any_full_match is False
        assert bundle.any_b1_eq_2 is False
        assert bundle.max_b1_seen < 2
        assert bundle.recovering_cells == ()
        assert "betti" in bundle.table
