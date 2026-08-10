"""lifetime_frac sweep on seed=2 denser fitted linked-tori (#41 follow-on).

A4-T42 found dataset seed=2 at ``n_per_torus=500`` / ``max_nodes=256`` yields
**both** tori Betti ``(1,2,0)`` under fixed_threshold SI fine mult (b2 missing).
Nested recovery needed ``lifetime_frac≥4``. This harness asks whether a
lifetime reading on that seed-fragile denser scaffold unlocks ``b2`` / full
``(1,2,1)``.

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
    format_lifetime_frac_sweep_table,
    nearest_data_labels,
    run_per_region_ph,
    sigma_star_from_tau,
    sweep_lifetime_frac_per_region,
)

EXPECTED_TORI: tuple[int, ...] = (1, 2, 1)
N_PER_TORUS: int = 500
MAX_NODES: int = 256
DATASET_SEED: int = 2
STAGE1_SEED: int = 77
# Span SI default through nested/circle recovery floor (≥4).
FRAC_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


@dataclass(frozen=True)
class Seed2LifetimeFracBundle:
    dataset_seed: int
    max_nodes: int
    n_signal: int
    sigma_star: float
    fixed_betti: dict[int, tuple[int, ...]]
    fixed_all_match: bool | None
    fracs: tuple[float, ...]
    table: str
    any_full_match: bool
    any_b2: bool
    any_b1_ge_2: bool
    max_b1: int
    max_b2: int
    recovering_cells: tuple[tuple[int, float, tuple[int, ...]], ...]
    b2_cells: tuple[tuple[int, float, tuple[int, ...]], ...]


@pytest.fixture(scope="module")
def linked_tori_seed2_lifetime_frac_bundle() -> Seed2LifetimeFracBundle:
    """Fit denser seed=2 scaffold; sweep lifetime_frac at SI fine mult."""
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

    fixed = run_per_region_ph(
        signal_pos,
        signal_labs,
        sigma,
        scenario="linked_tori_seed2_fixed_baseline",
        include_labels=[0, 1],
        reading="fixed_threshold",
        max_dim=2,
        filtration_mult=FILTRATION_MULTIPLIER,
        expected_betti=EXPECTED_TORI,
    )
    fixed_betti = {
        int(r.region_id): tuple(int(x) for x in r.betti) for r in fixed.reports
    }

    rows = sweep_lifetime_frac_per_region(
        signal_pos,
        signal_labs,
        sigma,
        fracs=FRAC_GRID,
        include_labels=[0, 1],
        max_dim=2,
        filtration_mult=FILTRATION_MULTIPLIER,
        target_betti=EXPECTED_TORI,
    )
    recovering: list[tuple[int, float, tuple[int, ...]]] = []
    b2_cells: list[tuple[int, float, tuple[int, ...]]] = []
    max_b1 = 0
    max_b2 = 0
    any_full = False
    any_b1 = False
    for row in rows:
        betti = tuple(int(x) for x in row.betti)
        rid = int(row.region_id) if row.region_id is not None else -1
        max_b1 = max(max_b1, int(betti[1]) if len(betti) > 1 else 0)
        max_b2 = max(max_b2, int(betti[2]) if len(betti) > 2 else 0)
        if betti == EXPECTED_TORI:
            any_full = True
            recovering.append((rid, float(row.lifetime_frac), betti))
        if len(betti) > 1 and int(betti[1]) >= 2:
            any_b1 = True
        if len(betti) > 2 and int(betti[2]) >= 1:
            b2_cells.append((rid, float(row.lifetime_frac), betti))

    return Seed2LifetimeFracBundle(
        dataset_seed=DATASET_SEED,
        max_nodes=MAX_NODES,
        n_signal=int(signal_pos.shape[0]),
        sigma_star=float(sigma),
        fixed_betti=fixed_betti,
        fixed_all_match=fixed.all_match,
        fracs=FRAC_GRID,
        table=format_lifetime_frac_sweep_table(rows),
        any_full_match=any_full,
        any_b2=bool(b2_cells),
        any_b1_ge_2=any_b1,
        max_b1=max_b1,
        max_b2=max_b2,
        recovering_cells=tuple(recovering),
        b2_cells=tuple(b2_cells),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed2_lifetime_frac_harness_lands(
    linked_tori_seed2_lifetime_frac_bundle,
) -> None:
    """Seed-2 lifetime_frac ladder lands; SI defaults untouched."""
    bundle = linked_tori_seed2_lifetime_frac_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.max_nodes == MAX_NODES
    assert bundle.n_signal > 0
    assert bundle.sigma_star > 0.0
    assert bundle.fracs == FRAC_GRID
    assert len(bundle.table) > 0
    # Fixed baseline should still show the T42 partial interlocking signal.
    assert any(
        len(b) > 1 and int(b[1]) >= 2 for b in bundle.fixed_betti.values()
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed2_lifetime_frac_documents_gap(
    linked_tori_seed2_lifetime_frac_bundle,
) -> None:
    """Document lifetime_frac vs b2 on seed2 denser; never flip awaiting.

    Soft: any full ``(1,2,1)`` or any ``b2≥1`` cell is proposal-path evidence.
    Otherwise keep explicit ``max_b2 == 0``. Lifetime alone ≠ SI recovery.
    """
    bundle = linked_tori_seed2_lifetime_frac_bundle
    if bundle.any_full_match or bundle.any_b2:
        assert FILTRATION_MULTIPLIER == 1.5
        if bundle.any_full_match:
            assert len(bundle.recovering_cells) >= 1
            assert bundle.max_b2 >= 1
        if bundle.any_b2:
            assert len(bundle.b2_cells) >= 1
            assert bundle.max_b2 >= 1
    else:
        assert bundle.any_full_match is False
        assert bundle.any_b2 is False
        assert bundle.max_b2 == 0
        assert bundle.recovering_cells == ()
        assert bundle.b2_cells == ()
        # Fixed baseline still documents partial b1 (T42); lifetime did not add b2.
        assert bundle.any_b1_ge_2 or any(
            len(b) > 1 and int(b[1]) >= 2 for b in bundle.fixed_betti.values()
        )
