"""Circle mult3 residual × tissue∈{0.10,0.15} @ noise=0.22 (#41 / A4-T99).

A4-T93/T96: tissue=0.12 cal4 pin died@0.22 while mult3 alt stayed clean at
0.22 (then died@≥0.25). This harness freezes the same
``mult∈{3,4,5} × frac∈{2.5,3,3.5}`` neighborhood at noise=0.22 and asks
whether the mult3 residual transfers across neighboring tissue fractions
``{0.10, 0.15}`` — proposal-path only.

Evidence-gathering only — does **not** flip ``@awaiting`` or SI defaults.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.circles import make_circle
from tests.metrics.persistent_homology import (
    DEFAULT_LIFETIME_FRAC,
    FILTRATION_MULTIPLIER,
    lifetime_betti_numbers,
    nearest_data_labels,
    sigma_star_from_tau,
)

DATASET_SEED: int = 21
STAGE1_SEED: int = 77
N_SAMPLES: int = 1200
NOISE: float = 0.22
TISSUE_GRID: tuple[float, ...] = (0.10, 0.15)
MULT_ARMS: tuple[float, ...] = (3.0, 4.0, 5.0)
FRAC_GRID: tuple[float, ...] = (2.5, 3.0, 3.5)
EXPECTED_B1: int = 1
PIN_MULT: float = 4.0
PIN_FRAC: float = 3.0
ALT_PIN_MULT: float = 3.0


@dataclass(frozen=True)
class CircleTissue1015Mult3Noise022Row:
    tissue_fraction: float
    filtration_mult: float
    lifetime_frac: float
    n_signal: int
    sigma_star: float
    betti: tuple[int, ...]
    b0: int
    b1: int
    recovers_clean: bool
    is_pin_cell: bool
    is_alt_mult3: bool


@dataclass(frozen=True)
class CircleTissue1015Mult3Noise022Bundle:
    dataset_seed: int
    stage1_seed: int
    noise: float
    tissue_grid: tuple[float, ...]
    mult_arms: tuple[float, ...]
    frac_grid: tuple[float, ...]
    rows: tuple[CircleTissue1015Mult3Noise022Row, ...]
    pin_clean_tissues: tuple[float, ...]
    alt_mult3_clean_tissues: tuple[float, ...]
    any_pin_clean: bool
    any_alt_mult3_clean: bool
    any_clean: bool
    clean_cells: tuple[tuple[float, float, float, tuple[int, ...]], ...]
    residual_transfers_both_tissues: bool
    residual_tissues: tuple[float, ...]
    collapsed_all: bool
    table: str


@pytest.fixture(scope="module")
def circle_tissue10_15_mult3_noise022_bundle() -> (
    CircleTissue1015Mult3Noise022Bundle
):
    """Cal/mult3 neighborhood × tissue{0.10,0.15} at noise=0.22."""
    rows: list[CircleTissue1015Mult3Noise022Row] = []
    clean_cells: list[tuple[float, float, float, tuple[int, ...]]] = []
    pin_clean_tissues: list[float] = []
    alt_clean_tissues: list[float] = []
    any_pin = False
    any_alt = False
    any_clean = False
    table_lines = [
        "tissue\tmult\tfrac\tn_signal\tbetti\tb0\tb1\tclean\tpin\talt3"
    ]
    alt_clean_by_tissue: dict[float, bool] = {
        float(t): False for t in TISSUE_GRID
    }

    for tissue in TISSUE_GRID:
        dataset = make_circle(
            n_samples=N_SAMPLES,
            radius=1.0,
            noise=float(NOISE),
            extrusion_dim=2,
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
        node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
        signal_pts = pos[node_labels == 0]
        n_signal = int(signal_pts.shape[0])

        for mult in MULT_ARMS:
            for frac in FRAC_GRID:
                betti = tuple(
                    int(x)
                    for x in lifetime_betti_numbers(
                        signal_pts,
                        sigma,
                        max_dim=1,
                        filtration_mult=float(mult),
                        lifetime_frac=float(frac),
                    )
                )
                b0 = int(betti[0]) if len(betti) > 0 else 0
                b1 = int(betti[1]) if len(betti) > 1 else 0
                recovers = b1 >= EXPECTED_B1 and b0 == 1
                is_pin = (
                    abs(float(mult) - PIN_MULT) < 1e-12
                    and abs(float(frac) - PIN_FRAC) < 1e-12
                )
                is_alt = abs(float(mult) - ALT_PIN_MULT) < 1e-12
                if recovers:
                    any_clean = True
                    clean_cells.append(
                        (float(tissue), float(mult), float(frac), betti)
                    )
                    if is_pin:
                        any_pin = True
                        if float(tissue) not in pin_clean_tissues:
                            pin_clean_tissues.append(float(tissue))
                    if is_alt:
                        any_alt = True
                        alt_clean_by_tissue[float(tissue)] = True
                        if float(tissue) not in alt_clean_tissues:
                            alt_clean_tissues.append(float(tissue))
                rows.append(
                    CircleTissue1015Mult3Noise022Row(
                        tissue_fraction=float(tissue),
                        filtration_mult=float(mult),
                        lifetime_frac=float(frac),
                        n_signal=n_signal,
                        sigma_star=sigma,
                        betti=betti,
                        b0=b0,
                        b1=b1,
                        recovers_clean=recovers,
                        is_pin_cell=is_pin,
                        is_alt_mult3=is_alt,
                    )
                )
                table_lines.append(
                    f"{tissue:g}\t{mult:g}\t{frac:g}\t{n_signal}\t{betti}\t"
                    f"{b0}\t{b1}\t{int(recovers)}\t{int(is_pin)}\t{int(is_alt)}"
                )

    residual_tissues = tuple(
        float(t) for t in TISSUE_GRID if alt_clean_by_tissue[float(t)]
    )
    residual_both = len(residual_tissues) == len(TISSUE_GRID)

    return CircleTissue1015Mult3Noise022Bundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        noise=float(NOISE),
        tissue_grid=TISSUE_GRID,
        mult_arms=MULT_ARMS,
        frac_grid=FRAC_GRID,
        rows=tuple(rows),
        pin_clean_tissues=tuple(pin_clean_tissues),
        alt_mult3_clean_tissues=tuple(alt_clean_tissues),
        any_pin_clean=any_pin,
        any_alt_mult3_clean=any_alt,
        any_clean=any_clean,
        clean_cells=tuple(clean_cells),
        residual_transfers_both_tissues=residual_both,
        residual_tissues=residual_tissues,
        collapsed_all=not any_clean,
        table="\n".join(table_lines),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_tissue10_15_mult3_noise022_harness_lands(
    circle_tissue10_15_mult3_noise022_bundle,
) -> None:
    """Tissue{0.10,0.15} mult3×noise0.22 lands; SI defaults untouched."""
    bundle = circle_tissue10_15_mult3_noise022_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert abs(bundle.noise - NOISE) < 1e-12
    assert bundle.tissue_grid == TISSUE_GRID
    assert bundle.mult_arms == MULT_ARMS
    assert bundle.frac_grid == FRAC_GRID
    assert len(bundle.rows) == (
        len(TISSUE_GRID) * len(MULT_ARMS) * len(FRAC_GRID)
    )
    header = bundle.table.splitlines()[0]
    assert "tissue" in header and "mult" in header and "alt3" in header
    pin_rows = [r for r in bundle.rows if r.is_pin_cell]
    assert len(pin_rows) == len(TISSUE_GRID)
    alt_rows = [r for r in bundle.rows if r.is_alt_mult3]
    assert len(alt_rows) == len(TISSUE_GRID) * len(FRAC_GRID)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_tissue10_15_mult3_noise022_documents_gap(
    circle_tissue10_15_mult3_noise022_bundle,
) -> None:
    """Document mult3 residual vs tissue{0.10,0.15}@0.22; never flip awaiting.

    Soft: any pin/alt clean is proposal-path only.
    Otherwise document collapse away from the T93 tissue0.12@0.22 residual.
    """
    bundle = circle_tissue10_15_mult3_noise022_bundle
    if bundle.any_pin_clean or bundle.any_alt_mult3_clean or bundle.any_clean:
        assert FILTRATION_MULTIPLIER == 1.5
        assert len(bundle.clean_cells) >= 1
        assert bundle.collapsed_all is False
        assert all(t in TISSUE_GRID for t in bundle.residual_tissues)
    else:
        assert bundle.any_pin_clean is False
        assert bundle.any_alt_mult3_clean is False
        assert bundle.any_clean is False
        assert bundle.clean_cells == ()
        assert bundle.pin_clean_tissues == ()
        assert bundle.alt_mult3_clean_tissues == ()
        assert bundle.residual_tissues == ()
        assert bundle.residual_transfers_both_tissues is False
        assert bundle.collapsed_all is True
        assert all(r.n_signal >= 0 for r in bundle.rows)
