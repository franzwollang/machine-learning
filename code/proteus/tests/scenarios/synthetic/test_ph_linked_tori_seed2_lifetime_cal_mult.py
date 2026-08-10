"""lifetime × cal-mult grid on seed=2 denser fitted linked-tori (#41 / A4-T46).

A4-T42/T43: denser seed2 / max_nodes=256 yields both tori ``(1,2,0)`` under
SI fine fixed_threshold; lifetime alone never unlocked ``b2``. Prior
lifetime×mult grid (A4-T36) used max_nodes=128 / seed=21. This harness
crosses ``lifetime_frac × filtration_mult`` on the seed2 denser scaffold
toward ``b2`` / full ``(1,2,1)``.

Evidence-gathering only — does **not** flip ``@awaiting`` or SI defaults.
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
N_PER_TORUS: int = 500
MAX_NODES: int = 256
DATASET_SEED: int = 2
STAGE1_SEED: int = 77
FRAC_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
# SI fine / coarse / circle-calibrated / above.
MULT_GRID: tuple[float, ...] = (1.5, 3.0, 6.0, 8.0)


@dataclass(frozen=True)
class Seed2LifetimeCalMultBundle:
    dataset_seed: int
    max_nodes: int
    n_signal: int
    sigma_star: float
    fracs: tuple[float, ...]
    mults: tuple[float, ...]
    table: str
    any_full_match: bool
    any_b1_ge_2: bool
    any_b2: bool
    max_b1: int
    max_b2: int
    recovering_cells: tuple[tuple[int, float, float, tuple[int, ...]], ...]
    b2_cells: tuple[tuple[int, float, float, tuple[int, ...]], ...]


@pytest.fixture(scope="module")
def linked_tori_seed2_lifetime_cal_mult_bundle() -> Seed2LifetimeCalMultBundle:
    """Fit denser seed=2; cross lifetime_frac × filtration_mult per torus."""
    dataset = make_linked_tori(
        n_per_torus=N_PER_TORUS,
        major_radius=2.0,
        minor_radius=0.5,
        noise=0.02,
        tissue_fraction=0.03,
        seed=DATASET_SEED,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    config = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=8,
        max_nodes=MAX_NODES,
        ann_backend="naive",
        stabilization=StabilizationConfig(
            min_equilibrium_epochs=3, max_epochs=15,
        ),
        seed=STAGE1_SEED,
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
        scenario="linked_tori_seed2_lifetime_cal_mult",
        include_labels=[0, 1],
        max_dim=2,
        expected_betti=EXPECTED_TORI,
        b1_target=2,
    )

    recovering: list[tuple[int, float, float, tuple[int, ...]]] = []
    b2_cells: list[tuple[int, float, float, tuple[int, ...]]] = []
    max_b1 = 0
    max_b2 = 0
    any_b1 = False
    for r in grid.rows:
        betti = tuple(int(x) for x in r.betti)
        rid = int(r.region_id)
        frac = float(r.lifetime_frac)
        mult = float(r.filtration_mult)
        max_b1 = max(max_b1, int(r.b1))
        b2 = int(betti[2]) if len(betti) > 2 else 0
        max_b2 = max(max_b2, b2)
        if betti == EXPECTED_TORI or r.match is True:
            recovering.append((rid, frac, mult, betti))
        if int(r.b1) >= 2:
            any_b1 = True
        if b2 >= 1:
            b2_cells.append((rid, frac, mult, betti))

    return Seed2LifetimeCalMultBundle(
        dataset_seed=DATASET_SEED,
        max_nodes=MAX_NODES,
        n_signal=int(signal_pos.shape[0]),
        sigma_star=float(sigma),
        fracs=FRAC_GRID,
        mults=MULT_GRID,
        table=format_lifetime_mult_grid_table(grid),
        any_full_match=bool(grid.any_full_match) or bool(recovering),
        any_b1_ge_2=any_b1,
        any_b2=bool(b2_cells),
        max_b1=max_b1,
        max_b2=max_b2,
        recovering_cells=tuple(recovering),
        b2_cells=tuple(b2_cells),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed2_lifetime_cal_mult_harness_lands(
    linked_tori_seed2_lifetime_cal_mult_bundle,
) -> None:
    """Seed2 denser lifetime×cal-mult grid lands; SI defaults untouched."""
    bundle = linked_tori_seed2_lifetime_cal_mult_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.max_nodes == MAX_NODES
    assert bundle.n_signal > 0
    assert bundle.sigma_star > 0.0
    assert bundle.fracs == FRAC_GRID
    assert bundle.mults == MULT_GRID
    header = bundle.table.splitlines()[0]
    assert "frac" in header and "mult" in header and "b1" in header
    assert len(bundle.table.splitlines()) == 1 + 2 * len(FRAC_GRID) * len(MULT_GRID)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed2_lifetime_cal_mult_documents_gap(
    linked_tori_seed2_lifetime_cal_mult_bundle,
) -> None:
    """Document lifetime×cal-mult vs b2/full on seed2 denser; never flip awaiting.

    Soft: any full ``(1,2,1)`` or any ``b2≥1`` cell is proposal-path evidence.
    Otherwise keep explicit ``max_b2 == 0``. Grid alone ≠ SI recovery.
    """
    bundle = linked_tori_seed2_lifetime_cal_mult_bundle
    if bundle.any_full_match or bundle.any_b2:
        assert FILTRATION_MULTIPLIER == 1.5
        if bundle.any_full_match:
            assert len(bundle.recovering_cells) >= 1
        if bundle.any_b2:
            assert len(bundle.b2_cells) >= 1
            assert bundle.max_b2 >= 1
    else:
        assert bundle.any_full_match is False
        assert bundle.any_b2 is False
        assert bundle.max_b2 == 0
        assert bundle.recovering_cells == ()
        assert bundle.b2_cells == ()
        # Partial interlocking may still appear via b1≥2 under lifetime.
        assert bundle.any_b1_ge_2 or bundle.max_b1 >= 1
