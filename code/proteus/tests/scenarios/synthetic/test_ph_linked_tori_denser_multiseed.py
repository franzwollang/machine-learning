"""Multi-seed stability of denser fitted linked-tori partial b1=2 (#41 / A4-T42).

A4-T38 / denser clean-vs-fitted found interlocking torus0 Betti ``(1,2,0)``
at ``n_per_torus=500``, ``max_nodes=256``, dataset seed=21, Stage-1 seed=77.
This harness re-fits the same denser recipe across dataset seeds ``0..2``
(Stage-1 seed fixed at 77) and documents whether partial ``b1≥2`` is stable
or seed-fragile.

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
    FILTRATION_MULTIPLIER,
    nearest_data_labels,
    run_per_region_ph,
    sigma_star_from_tau,
)

EXPECTED_TORI: tuple[int, ...] = (1, 2, 1)
N_PER_TORUS: int = 500
MAX_NODES: int = 256
STAGE1_SEED: int = 77
DATASET_SEEDS: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class DenserMultiseedRow:
    dataset_seed: int
    n_signal: int
    n_per_torus: dict[int, int]
    sigma_star: float
    all_match: bool | None
    betti: dict[int, tuple[int, ...]]
    max_b1: int
    any_b1_ge_2: bool
    any_full_torus: bool


@dataclass(frozen=True)
class LinkedToriDenserMultiseedBundle:
    n_per_torus_data: int
    max_nodes: int
    stage1_seed: int
    rows: tuple[DenserMultiseedRow, ...]
    n_seeds_with_b1_ge_2: int
    n_seeds_full_recover: int
    any_full_recover: bool
    any_b1_ge_2: bool
    all_seeds_b1_ge_2: bool
    max_b1: int
    b1_ge_2_cells: tuple[tuple[int, int, tuple[int, ...]], ...]


@pytest.fixture(scope="module")
def linked_tori_denser_multiseed_bundle() -> LinkedToriDenserMultiseedBundle:
    """Fit denser max_nodes=256 tori across dataset seeds 0..2."""
    rows: list[DenserMultiseedRow] = []
    cells: list[tuple[int, int, tuple[int, ...]]] = []

    for dataset_seed in DATASET_SEEDS:
        dataset = make_linked_tori(
            n_per_torus=N_PER_TORUS,
            major_radius=2.0,
            minor_radius=0.5,
            noise=0.02,
            tissue_fraction=0.03,
            seed=int(dataset_seed),
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
        result = run_scale_search(
            dataset.points, dim=gt.ambient_dim, config=config,
        )
        pos = result.scaffold_at_star.node_positions()
        sigma = sigma_star_from_tau(result.tau_star)
        node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
        signal_mask = np.isin(node_labels, [0, 1])
        signal_pos = pos[signal_mask]
        signal_labs = node_labels[signal_mask]
        ph = run_per_region_ph(
            signal_pos,
            signal_labs,
            sigma,
            scenario=f"linked_tori_denser_multiseed_s{dataset_seed}",
            include_labels=[0, 1],
            reading="fixed_threshold",
            max_dim=2,
            filtration_mult=FILTRATION_MULTIPLIER,
            expected_betti=EXPECTED_TORI,
        )
        betti = {
            int(r.region_id): tuple(int(x) for x in r.betti) for r in ph.reports
        }
        max_b1 = max((int(b[1]) for b in betti.values() if len(b) > 1), default=0)
        any_b1 = any(len(b) > 1 and int(b[1]) >= 2 for b in betti.values())
        any_full = any(b == EXPECTED_TORI for b in betti.values())
        rows.append(
            DenserMultiseedRow(
                dataset_seed=int(dataset_seed),
                n_signal=int(signal_pos.shape[0]),
                n_per_torus={
                    int(lab): int(np.sum(signal_labs == lab)) for lab in (0, 1)
                },
                sigma_star=float(sigma),
                all_match=ph.all_match,
                betti=betti,
                max_b1=max_b1,
                any_b1_ge_2=any_b1,
                any_full_torus=any_full,
            )
        )
        for lab, b in betti.items():
            if len(b) > 1 and int(b[1]) >= 2:
                cells.append((int(dataset_seed), int(lab), tuple(b)))

    n_b1 = sum(1 for r in rows if r.any_b1_ge_2)
    n_full = sum(1 for r in rows if r.all_match is True)
    return LinkedToriDenserMultiseedBundle(
        n_per_torus_data=N_PER_TORUS,
        max_nodes=MAX_NODES,
        stage1_seed=STAGE1_SEED,
        rows=tuple(rows),
        n_seeds_with_b1_ge_2=n_b1,
        n_seeds_full_recover=n_full,
        any_full_recover=n_full > 0,
        any_b1_ge_2=n_b1 > 0,
        all_seeds_b1_ge_2=n_b1 == len(rows),
        max_b1=max((r.max_b1 for r in rows), default=0),
        b1_ge_2_cells=tuple(cells),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_denser_multiseed_harness_lands(
    linked_tori_denser_multiseed_bundle,
) -> None:
    """Multi-seed denser fitted ladder lands; SI fine mult untouched."""
    bundle = linked_tori_denser_multiseed_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert bundle.n_per_torus_data == N_PER_TORUS
    assert bundle.max_nodes == MAX_NODES
    assert bundle.stage1_seed == STAGE1_SEED
    assert len(bundle.rows) == len(DATASET_SEEDS)
    assert [r.dataset_seed for r in bundle.rows] == list(DATASET_SEEDS)
    assert all(r.n_signal > 0 for r in bundle.rows)
    assert all(r.sigma_star > 0.0 for r in bundle.rows)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_denser_multiseed_documents_stability(
    linked_tori_denser_multiseed_bundle,
) -> None:
    """Document partial-b1=2 seed stability; never flip awaiting.

    Soft: any seed with full ``(1,2,1)`` both tori, or any/all seeds with
    partial ``b1≥2``, is proposal-path evidence. Otherwise keep explicit
    ``max_b1 < 2``. Seed-fragile partial ≠ SI recovery.
    """
    bundle = linked_tori_denser_multiseed_bundle
    if bundle.any_full_recover or bundle.any_b1_ge_2:
        assert FILTRATION_MULTIPLIER == 1.5
        assert bundle.max_b1 >= 2
        if bundle.any_b1_ge_2:
            assert len(bundle.b1_ge_2_cells) >= 1
            assert bundle.n_seeds_with_b1_ge_2 >= 1
        if bundle.all_seeds_b1_ge_2:
            assert bundle.n_seeds_with_b1_ge_2 == len(DATASET_SEEDS)
        if not bundle.any_full_recover:
            assert all(r.all_match is not True for r in bundle.rows)
            assert bundle.n_seeds_full_recover == 0
    else:
        assert bundle.any_full_recover is False
        assert bundle.any_b1_ge_2 is False
        assert bundle.all_seeds_b1_ge_2 is False
        assert bundle.n_seeds_with_b1_ge_2 == 0
        assert bundle.n_seeds_full_recover == 0
        assert bundle.max_b1 < 2
        assert bundle.b1_ge_2_cells == ()
        for row in bundle.rows:
            assert row.all_match is not True
            assert row.any_b1_ge_2 is False
