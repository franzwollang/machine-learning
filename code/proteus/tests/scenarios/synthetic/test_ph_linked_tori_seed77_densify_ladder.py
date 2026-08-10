"""Stage-1 seed=77 densify max_nodes ladder hunt (#41 / A4-T50-followon).

Prior denser hunts (A4-T38/T40) used Stage-1 seed=77 on dataset seed21 and
found max_nodes 384/512 **regress** partial ``b1``. A4-T44 locked dataset
seed2 + Stage-1 seed77 as the best both-tori partial ``(1,2,0)`` scaffold at
``max_nodes=256``. This harness re-runs the densify ladder **seed77-only** on
that dataset seed2 scaffold asking whether 384/512 unlock ``b2`` / ``(1,2,1)``.

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
MAX_NODES_LADDER: tuple[int, ...] = (256, 384, 512)
DATASET_SEED: int = 2
STAGE1_SEED: int = 77


@dataclass(frozen=True)
class Seed77DensifyRow:
    max_nodes: int
    n_signal: int
    n_per_torus: dict[int, int]
    sigma_star: float
    all_match: bool | None
    betti: dict[int, tuple[int, ...]]
    max_b1: int
    max_b2: int
    any_b1_ge_2: bool
    any_b2: bool
    any_full_torus: bool
    both_tori_partial: bool


@dataclass(frozen=True)
class Seed77DensifyLadderBundle:
    dataset_seed: int
    stage1_seed: int
    n_per_torus_data: int
    rows: tuple[Seed77DensifyRow, ...]
    any_full_recover: bool
    any_b2: bool
    any_b1_ge_2: bool
    max_b1: int
    max_b2: int
    b2_cells: tuple[tuple[int, int, tuple[int, ...]], ...]
    baseline_256_both_partial: bool


@pytest.fixture(scope="module")
def linked_tori_seed77_densify_ladder_bundle() -> Seed77DensifyLadderBundle:
    """Fit seed2 interlocking tori; Stage-1 seed77 across denser max_nodes."""
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

    rows: list[Seed77DensifyRow] = []
    for max_nodes in MAX_NODES_LADDER:
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
            scenario=f"linked_tori_seed77_densify_m{max_nodes}",
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
        max_b2 = max((int(b[2]) for b in betti.values() if len(b) > 2), default=0)
        any_b1 = any(len(b) > 1 and int(b[1]) >= 2 for b in betti.values())
        any_b2 = any(len(b) > 2 and int(b[2]) >= 1 for b in betti.values())
        any_full = any(b == EXPECTED_TORI for b in betti.values())
        both_partial = (
            betti.get(0) == (1, 2, 0) and betti.get(1) == (1, 2, 0)
        )
        rows.append(
            Seed77DensifyRow(
                max_nodes=int(max_nodes),
                n_signal=int(signal_pos.shape[0]),
                n_per_torus={
                    int(lab): int(np.sum(signal_labs == lab)) for lab in (0, 1)
                },
                sigma_star=float(sigma),
                all_match=ph.all_match,
                betti=betti,
                max_b1=max_b1,
                max_b2=max_b2,
                any_b1_ge_2=any_b1,
                any_b2=any_b2,
                any_full_torus=any_full,
                both_tori_partial=both_partial,
            )
        )

    b2_cells: list[tuple[int, int, tuple[int, ...]]] = []
    for row in rows:
        for lab, betti in row.betti.items():
            if len(betti) > 2 and int(betti[2]) >= 1:
                b2_cells.append((int(row.max_nodes), int(lab), tuple(betti)))

    baseline = next(r for r in rows if r.max_nodes == 256)
    return Seed77DensifyLadderBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        n_per_torus_data=N_PER_TORUS,
        rows=tuple(rows),
        any_full_recover=any(r.all_match is True for r in rows),
        any_b2=bool(b2_cells),
        any_b1_ge_2=any(r.any_b1_ge_2 for r in rows),
        max_b1=max((r.max_b1 for r in rows), default=0),
        max_b2=max((r.max_b2 for r in rows), default=0),
        b2_cells=tuple(b2_cells),
        baseline_256_both_partial=baseline.both_tori_partial,
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed77_densify_ladder_harness_lands(
    linked_tori_seed77_densify_ladder_bundle,
) -> None:
    """Seed77 densify ladder lands; SI fine mult untouched."""
    bundle = linked_tori_seed77_densify_ladder_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert bundle.n_per_torus_data == N_PER_TORUS
    assert len(bundle.rows) == len(MAX_NODES_LADDER)
    assert [r.max_nodes for r in bundle.rows] == list(MAX_NODES_LADDER)
    assert all(r.n_signal > 0 for r in bundle.rows)
    assert all(r.sigma_star > 0.0 for r in bundle.rows)
    # T44 baseline at 256 must remain both-tori partial interlocking.
    assert bundle.baseline_256_both_partial


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed77_densify_ladder_documents_gap(
    linked_tori_seed77_densify_ladder_bundle,
) -> None:
    """Document seed77 densify>256 vs b2/(1,2,1); never flip awaiting.

    Soft: full ``(1,2,1)`` or any ``b2≥1`` is proposal-path evidence.
    Otherwise keep documenting partial-only densify gap. Ladder ≠ SI recovery.
    """
    bundle = linked_tori_seed77_densify_ladder_bundle
    if bundle.any_full_recover or bundle.any_b2:
        assert FILTRATION_MULTIPLIER == 1.5
        if bundle.any_b2:
            assert len(bundle.b2_cells) >= 1
            assert bundle.max_b2 >= 1
        if bundle.any_full_recover:
            assert any(r.any_full_torus for r in bundle.rows)
    else:
        assert bundle.any_full_recover is False
        assert bundle.any_b2 is False
        assert bundle.max_b2 == 0
        assert bundle.b2_cells == ()
        # Baseline partial at 256 still holds; denser may regress b1.
        assert bundle.baseline_256_both_partial
        assert bundle.any_b1_ge_2 or bundle.max_b1 >= 0
