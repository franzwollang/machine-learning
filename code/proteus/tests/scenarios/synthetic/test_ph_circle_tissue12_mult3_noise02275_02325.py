"""Circle tissue=0.12 mult3 residual × noise{0.2275,0.2325} edge pin (#41 / A4-T108).

A4-T96: mult3 residual survived ONLY@0.22 and died@0.25/0.30 at tissue=0.12.
This harness freezes the same ``mult∈{3,4,5} × frac∈{2.5,3,3.5}`` neighborhood
and pins the NON-MONO dip edges with ``noise∈{0.22,0.225,0.2275,0.23,0.2325,0.235,0.24,0.25}`` — proposal-path only.

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
TISSUE: float = 0.12
MULT_ARMS: tuple[float, ...] = (3.0, 4.0, 5.0)
FRAC_GRID: tuple[float, ...] = (2.5, 3.0, 3.5)
# Pin collapse edge between T96 survive@0.22 and die@0.25.
NOISE_GRID: tuple[float, ...] = (0.22, 0.225, 0.2275, 0.23, 0.2325, 0.235, 0.24, 0.25)
EXPECTED_B1: int = 1
PIN_MULT: float = 4.0
PIN_FRAC: float = 3.0
ALT_PIN_MULT: float = 3.0


@dataclass(frozen=True)
class CircleTissue12Mult3Noise0227502325Row:
    noise: float
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
class CircleTissue12Mult3Noise0227502325Bundle:
    dataset_seed: int
    stage1_seed: int
    tissue_fraction: float
    noise_grid: tuple[float, ...]
    mult_arms: tuple[float, ...]
    frac_grid: tuple[float, ...]
    rows: tuple[CircleTissue12Mult3Noise0227502325Row, ...]
    pin_clean_noises: tuple[float, ...]
    alt_mult3_clean_noises: tuple[float, ...]
    any_pin_clean: bool
    any_alt_mult3_clean: bool
    any_clean: bool
    clean_cells: tuple[tuple[float, float, float, tuple[int, ...]], ...]
    residual_survives_022: bool
    residual_survives_0225: bool
    residual_survives_02275: bool
    residual_survives_023: bool
    residual_survives_02325: bool
    residual_survives_0235: bool
    residual_survives_024: bool
    residual_survives_025: bool
    residual_collapses_above_022: bool
    first_alt_collapse_noise: float | None
    noise_nonmono: bool
    collapsed_all: bool
    table: str


@pytest.fixture(scope="module")
def circle_tissue12_mult3_noise02275_02325_bundle() -> CircleTissue12Mult3Noise0227502325Bundle:
    """Cal/mult3 neighborhood × noise{0.22..0.25} fine-pin incl 0.2275/0.2325 at tissue=0.12."""
    rows: list[CircleTissue12Mult3Noise0227502325Row] = []
    clean_cells: list[tuple[float, float, float, tuple[int, ...]]] = []
    pin_clean_noises: list[float] = []
    alt_clean_noises: list[float] = []
    any_pin = False
    any_alt = False
    any_clean = False
    table_lines = [
        "noise\tmult\tfrac\tn_signal\tbetti\tb0\tb1\tclean\tpin\talt3"
    ]
    alt_clean_by_noise: dict[float, bool] = {
        float(n): False for n in NOISE_GRID
    }

    for noise in NOISE_GRID:
        dataset = make_circle(
            n_samples=N_SAMPLES,
            radius=1.0,
            noise=float(noise),
            extrusion_dim=2,
            tissue_fraction=TISSUE,
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
                        (float(noise), float(mult), float(frac), betti)
                    )
                    if is_pin:
                        any_pin = True
                        pin_clean_noises.append(float(noise))
                    if is_alt:
                        any_alt = True
                        alt_clean_by_noise[float(noise)] = True
                        if float(noise) not in alt_clean_noises:
                            alt_clean_noises.append(float(noise))
                rows.append(
                    CircleTissue12Mult3Noise0227502325Row(
                        noise=float(noise),
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
                    f"{noise:g}\t{mult:g}\t{frac:g}\t{n_signal}\t{betti}\t"
                    f"{b0}\t{b1}\t{int(recovers)}\t{int(is_pin)}\t{int(is_alt)}"
                )

    residual_022 = bool(alt_clean_by_noise.get(0.22, False))
    residual_0225 = bool(alt_clean_by_noise.get(0.225, False))
    residual_02275 = bool(alt_clean_by_noise.get(0.2275, False))
    residual_023 = bool(alt_clean_by_noise.get(0.23, False))
    residual_02325 = bool(alt_clean_by_noise.get(0.2325, False))
    residual_0235 = bool(alt_clean_by_noise.get(0.235, False))
    residual_024 = bool(alt_clean_by_noise.get(0.24, False))
    residual_025 = bool(alt_clean_by_noise.get(0.25, False))
    first_alt_collapse: float | None = None
    for n in NOISE_GRID:
        if float(n) <= 0.22 + 1e-12:
            continue
        if not alt_clean_by_noise[float(n)]:
            first_alt_collapse = float(n)
            break
    residual_collapses = residual_022 and first_alt_collapse is not None
    surv_seq = [
        residual_022, residual_0225, residual_02275, residual_023,
        residual_02325, residual_0235, residual_024, residual_025,
    ]
    # NON-MONO if any survive after a collapse (True after False in sequence).
    seen_fail = False
    noise_nonmono = False
    for s in surv_seq:
        if not s:
            seen_fail = True
        elif seen_fail:
            noise_nonmono = True
            break

    return CircleTissue12Mult3Noise0227502325Bundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        tissue_fraction=TISSUE,
        noise_grid=NOISE_GRID,
        mult_arms=MULT_ARMS,
        frac_grid=FRAC_GRID,
        rows=tuple(rows),
        pin_clean_noises=tuple(pin_clean_noises),
        alt_mult3_clean_noises=tuple(alt_clean_noises),
        any_pin_clean=any_pin,
        any_alt_mult3_clean=any_alt,
        any_clean=any_clean,
        clean_cells=tuple(clean_cells),
        residual_survives_022=residual_022,
        residual_survives_0225=residual_0225,
        residual_survives_02275=residual_02275,
        residual_survives_023=residual_023,
        residual_survives_02325=residual_02325,
        residual_survives_0235=residual_0235,
        residual_survives_024=residual_024,
        residual_survives_025=residual_025,
        residual_collapses_above_022=residual_collapses,
        first_alt_collapse_noise=first_alt_collapse,
        noise_nonmono=noise_nonmono,
        collapsed_all=not any_clean,
        table="\n".join(table_lines),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_tissue12_mult3_noise02275_02325_harness_lands(
    circle_tissue12_mult3_noise02275_02325_bundle,
) -> None:
    """Tissue0.12 mult3×noise{0.2275,0.2325} edge lands; SI defaults untouched."""
    bundle = circle_tissue12_mult3_noise02275_02325_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert abs(bundle.tissue_fraction - TISSUE) < 1e-12
    assert bundle.noise_grid == NOISE_GRID
    assert bundle.mult_arms == MULT_ARMS
    assert bundle.frac_grid == FRAC_GRID
    assert len(bundle.rows) == (
        len(NOISE_GRID) * len(MULT_ARMS) * len(FRAC_GRID)
    )
    header = bundle.table.splitlines()[0]
    assert "noise" in header and "mult" in header and "alt3" in header
    pin_rows = [r for r in bundle.rows if r.is_pin_cell]
    assert len(pin_rows) == len(NOISE_GRID)
    alt_rows = [r for r in bundle.rows if r.is_alt_mult3]
    assert len(alt_rows) == len(NOISE_GRID) * len(FRAC_GRID)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_tissue12_mult3_noise02275_02325_documents_gap(
    circle_tissue12_mult3_noise02275_02325_bundle,
) -> None:
    """Document mult3 residual edge vs noise{0.2275,0.2325}; never flip awaiting.

    Soft: any pin/alt clean is proposal-path only.
    Otherwise document collapse between T96@0.22 survive and @0.25 die.
    """
    bundle = circle_tissue12_mult3_noise02275_02325_bundle
    if bundle.any_pin_clean or bundle.any_alt_mult3_clean or bundle.any_clean:
        assert FILTRATION_MULTIPLIER == 1.5
        assert len(bundle.clean_cells) >= 1
        assert bundle.collapsed_all is False
        assert (
            bundle.first_alt_collapse_noise is None
            or bundle.first_alt_collapse_noise in NOISE_GRID
        )
    else:
        assert bundle.any_pin_clean is False
        assert bundle.any_alt_mult3_clean is False
        assert bundle.any_clean is False
        assert bundle.clean_cells == ()
        assert bundle.pin_clean_noises == ()
        assert bundle.alt_mult3_clean_noises == ()
        assert bundle.collapsed_all is True
        assert all(r.n_signal >= 0 for r in bundle.rows)
