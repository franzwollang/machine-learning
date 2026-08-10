"""Tissue×noise ablation on seed77 denser256 (#41 / A4-T58-followon).

Default denser recipe uses ``noise=0.02`` / ``tissue_fraction=0.03`` and
yields both-tori partial ``(1,2,0)`` (no void). This harness freezes
Stage-1 seed=77 / max_nodes=256 and crosses a compact tissue×noise grid,
asking whether cleaner tissue or quieter noise alone unlocks ``b2`` /
``(1,2,1)`` under fixed_threshold SI fine.

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
    nearest_data_labels,
    run_per_region_ph,
    sigma_star_from_tau,
)

EXPECTED_TORI: tuple[int, ...] = (1, 2, 1)
N_PER_TORUS: int = 500
MAX_NODES: int = 256
DATASET_SEED: int = 2
STAGE1_SEED: int = 77
TISSUE_GRID: tuple[float, ...] = (0.0, 0.03, 0.08)
NOISE_GRID: tuple[float, ...] = (0.01, 0.02, 0.05)


def _is_clean_b2(betti: tuple[int, ...]) -> bool:
    return (
        len(betti) > 2
        and int(betti[0]) == 1
        and int(betti[1]) >= 2
        and int(betti[2]) >= 1
    )


def _is_dirty_b2(betti: tuple[int, ...]) -> bool:
    return (
        len(betti) > 2
        and int(betti[2]) >= 1
        and not _is_clean_b2(betti)
    )


@dataclass(frozen=True)
class TissueNoiseRow:
    tissue_fraction: float
    noise: float
    n_signal: int
    sigma_star: float
    betti: dict[int, tuple[int, ...]]
    both_partial: bool
    any_full: bool
    any_clean_b2: bool
    any_dirty_b2: bool
    max_b1: int
    max_b2: int


@dataclass(frozen=True)
class Seed77TissueNoiseAblationBundle:
    dataset_seed: int
    stage1_seed: int
    max_nodes: int
    tissue_grid: tuple[float, ...]
    noise_grid: tuple[float, ...]
    rows: tuple[TissueNoiseRow, ...]
    n_cells_both_partial: int
    n_cells_with_b2: int
    n_cells_full: int
    n_cells_clean: int
    n_cells_dirty: int
    any_full_match: bool
    any_clean_b2: bool
    any_dirty_b2: bool
    max_b1: int
    max_b2: int
    baseline_both_partial: bool
    clean_cells: tuple[tuple[float, float, int, tuple[int, ...]], ...]
    dirty_cells: tuple[tuple[float, float, int, tuple[int, ...]], ...]
    table: str


@pytest.fixture(scope="module")
def linked_tori_seed77_tissue_noise_ablation_bundle() -> (
    Seed77TissueNoiseAblationBundle
):
    """Cross tissue×noise; fit seed77 denser256; fixed_threshold SI fine."""
    rows: list[TissueNoiseRow] = []
    clean_cells: list[tuple[float, float, int, tuple[int, ...]]] = []
    dirty_cells: list[tuple[float, float, int, tuple[int, ...]]] = []
    max_b1 = 0
    max_b2 = 0
    n_both = 0
    n_b2 = 0
    n_full = 0
    n_clean = 0
    n_dirty = 0
    any_full = False
    any_clean = False
    any_dirty = False
    baseline_both = False
    table_lines = ["tissue\tnoise\tn_signal\tbetti\tclean\tdirty"]

    for tissue in TISSUE_GRID:
        for noise in NOISE_GRID:
            dataset = make_linked_tori(
                n_per_torus=N_PER_TORUS,
                major_radius=2.0,
                minor_radius=0.5,
                noise=float(noise),
                tissue_fraction=float(tissue),
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
            result = run_scale_search(
                dataset.points, dim=gt.ambient_dim, config=config,
            )
            pos = result.scaffold_at_star.node_positions()
            sigma = float(sigma_star_from_tau(result.tau_star))
            node_labels = nearest_data_labels(
                pos, dataset.points, dataset.labels,
            )
            signal_mask = np.isin(node_labels, [0, 1])
            signal_pos = pos[signal_mask]
            signal_labs = node_labels[signal_mask]
            n_signal = int(np.sum(signal_mask))

            ph = run_per_region_ph(
                signal_pos,
                signal_labs,
                sigma,
                scenario=(
                    f"linked_tori_seed77_tissue{tissue:g}_noise{noise:g}"
                ),
                include_labels=[0, 1],
                reading="fixed_threshold",
                max_dim=2,
                filtration_mult=FILTRATION_MULTIPLIER,
                expected_betti=EXPECTED_TORI,
            )
            betti = {
                int(r.region_id): tuple(int(x) for x in r.betti)
                for r in ph.reports
            }
            both = betti.get(0) == (1, 2, 0) and betti.get(1) == (1, 2, 0)
            if both:
                n_both += 1
            if (
                abs(float(tissue) - 0.03) < 1e-12
                and abs(float(noise) - 0.02) < 1e-12
            ):
                baseline_both = both

            row_full = False
            row_clean = False
            row_dirty = False
            row_b1 = 0
            row_b2 = 0
            for rid, b in betti.items():
                b1 = int(b[1]) if len(b) > 1 else 0
                b2 = int(b[2]) if len(b) > 2 else 0
                row_b1 = max(row_b1, b1)
                row_b2 = max(row_b2, b2)
                if b == EXPECTED_TORI:
                    row_full = True
                    any_full = True
                if _is_clean_b2(b):
                    row_clean = True
                    any_clean = True
                    clean_cells.append(
                        (float(tissue), float(noise), int(rid), b)
                    )
                if _is_dirty_b2(b):
                    row_dirty = True
                    any_dirty = True
                    dirty_cells.append(
                        (float(tissue), float(noise), int(rid), b)
                    )

            if row_b2 >= 1:
                n_b2 += 1
            if row_full:
                n_full += 1
            if row_clean:
                n_clean += 1
            if row_dirty:
                n_dirty += 1
            max_b1 = max(max_b1, row_b1)
            max_b2 = max(max_b2, row_b2)

            table_lines.append(
                f"{tissue:g}\t{noise:g}\t{n_signal}\t{betti}\t"
                f"{row_clean}\t{row_dirty}"
            )
            rows.append(
                TissueNoiseRow(
                    tissue_fraction=float(tissue),
                    noise=float(noise),
                    n_signal=n_signal,
                    sigma_star=sigma,
                    betti=betti,
                    both_partial=both,
                    any_full=row_full,
                    any_clean_b2=row_clean,
                    any_dirty_b2=row_dirty,
                    max_b1=row_b1,
                    max_b2=row_b2,
                )
            )

    return Seed77TissueNoiseAblationBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        max_nodes=MAX_NODES,
        tissue_grid=TISSUE_GRID,
        noise_grid=NOISE_GRID,
        rows=tuple(rows),
        n_cells_both_partial=n_both,
        n_cells_with_b2=n_b2,
        n_cells_full=n_full,
        n_cells_clean=n_clean,
        n_cells_dirty=n_dirty,
        any_full_match=any_full,
        any_clean_b2=any_clean,
        any_dirty_b2=any_dirty,
        max_b1=max_b1,
        max_b2=max_b2,
        baseline_both_partial=baseline_both,
        clean_cells=tuple(clean_cells),
        dirty_cells=tuple(dirty_cells),
        table="\n".join(table_lines),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed77_tissue_noise_ablation_harness_lands(
    linked_tori_seed77_tissue_noise_ablation_bundle,
) -> None:
    """Tissue×noise ablation harness lands; SI defaults untouched."""
    bundle = linked_tori_seed77_tissue_noise_ablation_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert bundle.max_nodes == MAX_NODES
    assert bundle.tissue_grid == TISSUE_GRID
    assert bundle.noise_grid == NOISE_GRID
    assert len(bundle.rows) == len(TISSUE_GRID) * len(NOISE_GRID)
    header = bundle.table.splitlines()[0]
    assert "tissue" in header and "noise" in header and "betti" in header


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed77_tissue_noise_ablation_documents_gap(
    linked_tori_seed77_tissue_noise_ablation_bundle,
) -> None:
    """Document tissue×noise vs clean (1,2,1); never flip awaiting."""
    bundle = linked_tori_seed77_tissue_noise_ablation_bundle
    if bundle.any_full_match or bundle.any_clean_b2:
        assert FILTRATION_MULTIPLIER == 1.5
        if bundle.any_clean_b2:
            assert len(bundle.clean_cells) >= 1
            assert bundle.max_b2 >= 1
    else:
        assert bundle.any_full_match is False
        assert bundle.any_clean_b2 is False
        assert bundle.clean_cells == ()
        assert bundle.n_cells_full == 0
        assert bundle.n_cells_clean == 0
        # Default recipe cell should remain both-partial if still gap.
        assert bundle.baseline_both_partial
        assert bundle.max_b2 >= 0
