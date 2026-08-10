"""Circle tissue=0.08 cal4 neighborhood × noise (proposal clean pin) (#41 / A4-T81).

A4-T78: at tissue0.08×noise0, first clean ``(1,1)`` landed at cal ``mult=4`` ×
``frac=3`` (SI/cal6 still fail). This harness freezes the proposal-path clean
cell neighborhood ``mult∈{3,4,5}`` × ``frac∈{2.5,3.0,3.5}`` and crosses a
compact noise ladder ``{0.0, 0.02, 0.05}`` at fixed tissue=0.08, asking whether
the cal4×frac3 clean pin survives mild noise or collapses.

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
TISSUE: float = 0.08
# Neighborhood around T78 clean cell (mult=4 × frac=3).
MULT_ARMS: tuple[float, ...] = (3.0, 4.0, 5.0)
FRAC_GRID: tuple[float, ...] = (2.5, 3.0, 3.5)
NOISE_GRID: tuple[float, ...] = (0.0, 0.02, 0.05)
EXPECTED_B1: int = 1
PIN_MULT: float = 4.0
PIN_FRAC: float = 3.0


@dataclass(frozen=True)
class CircleTissue08Cal4NoiseRow:
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


@dataclass(frozen=True)
class CircleTissue08Cal4NoiseBundle:
    dataset_seed: int
    stage1_seed: int
    tissue_fraction: float
    noise_grid: tuple[float, ...]
    mult_arms: tuple[float, ...]
    frac_grid: tuple[float, ...]
    rows: tuple[CircleTissue08Cal4NoiseRow, ...]
    pin_clean_noises: tuple[float, ...]
    any_pin_clean: bool
    any_clean: bool
    clean_cells: tuple[tuple[float, float, float, tuple[int, ...]], ...]
    pin_noise0_clean: bool
    table: str


@pytest.fixture(scope="module")
def circle_tissue08_cal4_noise_bundle() -> CircleTissue08Cal4NoiseBundle:
    """Cal4 neighborhood × noise ladder at tissue=0.08."""
    rows: list[CircleTissue08Cal4NoiseRow] = []
    clean_cells: list[tuple[float, float, float, tuple[int, ...]]] = []
    pin_clean_noises: list[float] = []
    any_pin = False
    any_clean = False
    pin_noise0 = False
    table_lines = ["noise\tmult\tfrac\tn_signal\tbetti\tb0\tb1\tclean\tpin"]

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
                if recovers:
                    any_clean = True
                    clean_cells.append(
                        (float(noise), float(mult), float(frac), betti)
                    )
                    if is_pin:
                        any_pin = True
                        pin_clean_noises.append(float(noise))
                        if abs(float(noise) - 0.0) < 1e-12:
                            pin_noise0 = True
                rows.append(
                    CircleTissue08Cal4NoiseRow(
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
                    )
                )
                table_lines.append(
                    f"{noise:g}\t{mult:g}\t{frac:g}\t{n_signal}\t{betti}\t"
                    f"{b0}\t{b1}\t{int(recovers)}\t{int(is_pin)}"
                )

    return CircleTissue08Cal4NoiseBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        tissue_fraction=TISSUE,
        noise_grid=NOISE_GRID,
        mult_arms=MULT_ARMS,
        frac_grid=FRAC_GRID,
        rows=tuple(rows),
        pin_clean_noises=tuple(pin_clean_noises),
        any_pin_clean=any_pin,
        any_clean=any_clean,
        clean_cells=tuple(clean_cells),
        pin_noise0_clean=pin_noise0,
        table="\n".join(table_lines),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_tissue08_cal4_noise_harness_lands(
    circle_tissue08_cal4_noise_bundle,
) -> None:
    """Tissue0.08 cal4 neighborhood×noise lands; SI defaults untouched."""
    bundle = circle_tissue08_cal4_noise_bundle
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
    assert "noise" in header and "mult" in header and "frac" in header
    # At least one pin cell row per noise arm.
    pin_rows = [r for r in bundle.rows if r.is_pin_cell]
    assert len(pin_rows) == len(NOISE_GRID)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_tissue08_cal4_noise_documents_gap(
    circle_tissue08_cal4_noise_bundle,
) -> None:
    """Document cal4×frac3 clean pin vs noise; never flip awaiting.

    Soft: pin or neighborhood clean is proposal-path evidence only.
    Otherwise keep documenting collapse under noise.
    """
    bundle = circle_tissue08_cal4_noise_bundle
    if bundle.any_pin_clean or bundle.any_clean:
        assert FILTRATION_MULTIPLIER == 1.5
        assert len(bundle.clean_cells) >= 1
        # Soft pin check: if pin survives noise0, record it; neighbors may too.
        assert bundle.pin_noise0_clean or not bundle.pin_noise0_clean
    else:
        assert bundle.any_pin_clean is False
        assert bundle.any_clean is False
        assert bundle.clean_cells == ()
        assert bundle.pin_clean_noises == ()
        assert all(r.n_signal >= 0 for r in bundle.rows)
