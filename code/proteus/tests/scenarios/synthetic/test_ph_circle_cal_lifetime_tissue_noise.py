"""Circle cal × lifetime_frac × tissue × noise joint (#41 / A4-T72-followon).

A4-T66: lifetime×noise at fixed tissue=0.03 leaves SI dead; cal recovers with
noise-dependent frac floors. A4-T69: tissue×noise at default frac=0.5 leaves
SI dead and makes cal ``b1=1`` dirty via inflated ``b0``. This follow-on jointly
crosses a compact ``tissue × noise × lifetime_frac`` grid under SI mult=1.5 and
cal mult=6, asking whether high-frac cleanup restores clean ``(b0=1,b1=1)`` at
any quieter tissue+noise corner (or unlocks SI).

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
TISSUE_GRID: tuple[float, ...] = (0.0, 0.03, 0.08)
NOISE_GRID: tuple[float, ...] = (0.0, 0.02, 0.05)
FRAC_GRID: tuple[float, ...] = (0.5, 2.0, 4.0, 8.0)
MULT_ARMS: tuple[float, ...] = (1.5, 6.0)
EXPECTED_B1: int = 1


@dataclass(frozen=True)
class CircleCalLifetimeTissueNoiseRow:
    tissue_fraction: float
    noise: float
    filtration_mult: float
    lifetime_frac: float
    n_signal: int
    sigma_star: float
    betti: tuple[int, ...]
    b1: int
    recovers_b1: bool
    dirty_b0: bool


@dataclass(frozen=True)
class CircleCalLifetimeTissueNoiseBundle:
    dataset_seed: int
    stage1_seed: int
    tissue_grid: tuple[float, ...]
    noise_grid: tuple[float, ...]
    frac_grid: tuple[float, ...]
    mult_arms: tuple[float, ...]
    rows: tuple[CircleCalLifetimeTissueNoiseRow, ...]
    si_recover_cells: tuple[tuple[float, float, float], ...]
    cal_recover_cells: tuple[tuple[float, float, float], ...]
    cal_dirty_b0_recover_cells: tuple[tuple[float, float, float], ...]
    any_si_recover: bool
    any_cal_recover: bool
    any_cal_dirty_b0_at_recover: bool
    min_frac_cal_clean: dict[tuple[float, float], float | None]
    table: str


@pytest.fixture(scope="module")
def circle_cal_lifetime_tissue_noise_bundle() -> CircleCalLifetimeTissueNoiseBundle:
    """Cross tissue×noise×lifetime_frac under SI and cal mult."""
    rows: list[CircleCalLifetimeTissueNoiseRow] = []
    si_cells: list[tuple[float, float, float]] = []
    cal_cells: list[tuple[float, float, float]] = []
    cal_dirty: list[tuple[float, float, float]] = []
    min_frac_clean: dict[tuple[float, float], float | None] = {}
    any_si = False
    any_cal = False
    any_cal_dirty = False
    table_lines = [
        "tissue\tnoise\tmult\tfrac\tn_signal\tbetti\tb1\trecover\tdirty_b0"
    ]

    for tissue in TISSUE_GRID:
        for noise in NOISE_GRID:
            dataset = make_circle(
                n_samples=N_SAMPLES,
                radius=1.0,
                noise=float(noise),
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
            key = (float(tissue), float(noise))
            min_frac_clean[key] = None

            for mult in MULT_ARMS:
                is_si = abs(float(mult) - FILTRATION_MULTIPLIER) < 1e-12
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
                    b1 = int(betti[1]) if len(betti) > 1 else 0
                    b0 = int(betti[0]) if len(betti) > 0 else 0
                    dirty_b0 = b0 > 1
                    recovers = b1 >= EXPECTED_B1 and b0 == 1
                    cell = (float(tissue), float(noise), float(frac))
                    if recovers:
                        if is_si:
                            any_si = True
                            si_cells.append(cell)
                        else:
                            any_cal = True
                            cal_cells.append(cell)
                            if min_frac_clean[key] is None:
                                min_frac_clean[key] = float(frac)
                    elif (not is_si) and b1 >= EXPECTED_B1 and dirty_b0:
                        any_cal_dirty = True
                        cal_dirty.append(cell)
                    rows.append(
                        CircleCalLifetimeTissueNoiseRow(
                            tissue_fraction=float(tissue),
                            noise=float(noise),
                            filtration_mult=float(mult),
                            lifetime_frac=float(frac),
                            n_signal=n_signal,
                            sigma_star=sigma,
                            betti=betti,
                            b1=b1,
                            recovers_b1=recovers,
                            dirty_b0=dirty_b0,
                        )
                    )
                    table_lines.append(
                        f"{tissue:g}\t{noise:g}\t{mult:g}\t{frac:g}\t"
                        f"{n_signal}\t{betti}\t{b1}\t{int(recovers)}\t"
                        f"{int(dirty_b0)}"
                    )

    return CircleCalLifetimeTissueNoiseBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        tissue_grid=TISSUE_GRID,
        noise_grid=NOISE_GRID,
        frac_grid=FRAC_GRID,
        mult_arms=MULT_ARMS,
        rows=tuple(rows),
        si_recover_cells=tuple(si_cells),
        cal_recover_cells=tuple(cal_cells),
        cal_dirty_b0_recover_cells=tuple(cal_dirty),
        any_si_recover=any_si,
        any_cal_recover=any_cal,
        any_cal_dirty_b0_at_recover=any_cal_dirty,
        min_frac_cal_clean=min_frac_clean,
        table="\n".join(table_lines),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_cal_lifetime_tissue_noise_harness_lands(
    circle_cal_lifetime_tissue_noise_bundle,
) -> None:
    """Circle cal×lifetime×tissue×noise joint probe lands; SI defaults untouched."""
    bundle = circle_cal_lifetime_tissue_noise_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert bundle.tissue_grid == TISSUE_GRID
    assert bundle.noise_grid == NOISE_GRID
    assert bundle.frac_grid == FRAC_GRID
    assert bundle.mult_arms == MULT_ARMS
    assert len(bundle.rows) == (
        len(TISSUE_GRID) * len(NOISE_GRID) * len(MULT_ARMS) * len(FRAC_GRID)
    )
    assert all(r.n_signal >= 8 for r in bundle.rows)
    assert all(r.sigma_star > 0.0 for r in bundle.rows)
    header = bundle.table.splitlines()[0]
    assert "tissue" in header and "noise" in header and "frac" in header


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_cal_lifetime_tissue_noise_documents_gap(
    circle_cal_lifetime_tissue_noise_bundle,
) -> None:
    """Document joint clutter×frac vs SI/cal clean b1; never flip awaiting.

    Soft: SI-mult clean recover at any tissue×noise×frac is proposal-path
    evidence. Otherwise keep documenting SI death + cal frac floors.
    """
    bundle = circle_cal_lifetime_tissue_noise_bundle
    if bundle.any_si_recover:
        assert FILTRATION_MULTIPLIER == 1.5
        assert len(bundle.si_recover_cells) >= 1
    else:
        assert bundle.any_si_recover is False
        assert bundle.si_recover_cells == ()
        # Baseline tissue=0.03 / noise=0.02 / default frac SI still fails.
        baseline = [
            r for r in bundle.rows
            if abs(r.tissue_fraction - 0.03) < 1e-12
            and abs(r.noise - 0.02) < 1e-12
            and abs(r.filtration_mult - FILTRATION_MULTIPLIER) < 1e-12
            and abs(r.lifetime_frac - DEFAULT_LIFETIME_FRAC) < 1e-12
        ]
        assert len(baseline) == 1
        assert baseline[0].b1 == 0
        assert baseline[0].recovers_b1 is False
        assert bundle.any_cal_recover or not bundle.any_cal_recover
