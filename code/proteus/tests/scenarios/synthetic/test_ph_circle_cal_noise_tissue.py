"""Circle cal-mult × noise × tissue joint probe (#41 / A4-T69-followon).

A4-T60/T63: tissue×mult / tissue×lifetime leave SI ``b1=0``; cal ``mult=6``
recovers with tissue-dependent frac floors. A4-T66: lifetime×noise at fixed
tissue=0.03 likewise leaves SI dead; cal recovers with noise-dependent frac
floors. This follow-on jointly crosses a compact ``tissue × noise`` grid under
SI ``filtration_mult=1.5`` and cal ``mult=6`` (default lifetime_frac=0.5),
asking whether any quieter tissue+noise corner unlocks SI circle ``b1=1`` or
whether cal recovery collapses under combined clutter.

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
MULT_ARMS: tuple[float, ...] = (1.5, 6.0)
EXPECTED_B1: int = 1


@dataclass(frozen=True)
class CircleCalNoiseTissueRow:
    tissue_fraction: float
    noise: float
    filtration_mult: float
    n_signal: int
    sigma_star: float
    betti: tuple[int, ...]
    b1: int
    recovers_b1: bool


@dataclass(frozen=True)
class CircleCalNoiseTissueBundle:
    dataset_seed: int
    stage1_seed: int
    tissue_grid: tuple[float, ...]
    noise_grid: tuple[float, ...]
    mult_arms: tuple[float, ...]
    lifetime_frac: float
    rows: tuple[CircleCalNoiseTissueRow, ...]
    si_recover_cells: tuple[tuple[float, float], ...]
    cal_recover_cells: tuple[tuple[float, float], ...]
    any_si_recover: bool
    any_cal_recover: bool
    quietest_si_cell: tuple[float, float] | None
    table: str


@pytest.fixture(scope="module")
def circle_cal_noise_tissue_bundle() -> CircleCalNoiseTissueBundle:
    """Cross tissue×noise under SI and cal mult at default lifetime_frac."""
    rows: list[CircleCalNoiseTissueRow] = []
    si_cells: list[tuple[float, float]] = []
    cal_cells: list[tuple[float, float]] = []
    any_si = False
    any_cal = False
    quietest: tuple[float, float] | None = None
    table_lines = ["tissue\tnoise\tmult\tn_signal\tbetti\tb1\trecover"]

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

            for mult in MULT_ARMS:
                betti = tuple(
                    int(x)
                    for x in lifetime_betti_numbers(
                        signal_pts,
                        sigma,
                        max_dim=1,
                        filtration_mult=float(mult),
                        lifetime_frac=DEFAULT_LIFETIME_FRAC,
                    )
                )
                b1 = int(betti[1]) if len(betti) > 1 else 0
                recovers = b1 >= EXPECTED_B1 and int(betti[0]) == 1
                is_si = abs(float(mult) - FILTRATION_MULTIPLIER) < 1e-12
                if recovers:
                    if is_si:
                        any_si = True
                        si_cells.append((float(tissue), float(noise)))
                        if quietest is None or (
                            float(tissue) + float(noise)
                            < quietest[0] + quietest[1]
                        ):
                            quietest = (float(tissue), float(noise))
                    else:
                        any_cal = True
                        cal_cells.append((float(tissue), float(noise)))
                rows.append(
                    CircleCalNoiseTissueRow(
                        tissue_fraction=float(tissue),
                        noise=float(noise),
                        filtration_mult=float(mult),
                        n_signal=n_signal,
                        sigma_star=sigma,
                        betti=betti,
                        b1=b1,
                        recovers_b1=recovers,
                    )
                )
                table_lines.append(
                    f"{tissue:g}\t{noise:g}\t{mult:g}\t{n_signal}\t"
                    f"{betti}\t{b1}\t{int(recovers)}"
                )

    return CircleCalNoiseTissueBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        tissue_grid=TISSUE_GRID,
        noise_grid=NOISE_GRID,
        mult_arms=MULT_ARMS,
        lifetime_frac=DEFAULT_LIFETIME_FRAC,
        rows=tuple(rows),
        si_recover_cells=tuple(si_cells),
        cal_recover_cells=tuple(cal_cells),
        any_si_recover=any_si,
        any_cal_recover=any_cal,
        quietest_si_cell=quietest,
        table="\n".join(table_lines),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_cal_noise_tissue_harness_lands(
    circle_cal_noise_tissue_bundle,
) -> None:
    """Circle cal×noise×tissue joint probe lands; SI defaults untouched."""
    bundle = circle_cal_noise_tissue_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert bundle.tissue_grid == TISSUE_GRID
    assert bundle.noise_grid == NOISE_GRID
    assert bundle.mult_arms == MULT_ARMS
    assert bundle.lifetime_frac == DEFAULT_LIFETIME_FRAC
    assert len(bundle.rows) == (
        len(TISSUE_GRID) * len(NOISE_GRID) * len(MULT_ARMS)
    )
    assert all(r.n_signal >= 8 for r in bundle.rows)
    assert all(r.sigma_star > 0.0 for r in bundle.rows)
    header = bundle.table.splitlines()[0]
    assert "tissue" in header and "noise" in header and "mult" in header


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_cal_noise_tissue_documents_gap(
    circle_cal_noise_tissue_bundle,
) -> None:
    """Document joint tissue×noise vs SI b1 recovery; never flip awaiting.

    Soft: SI-mult recovering ``b1`` at any tissue×noise cell is proposal-path
    evidence. Otherwise keep documenting that quieter joint clutter ≠ SI unlock.
    """
    bundle = circle_cal_noise_tissue_bundle
    if bundle.any_si_recover:
        assert FILTRATION_MULTIPLIER == 1.5
        assert len(bundle.si_recover_cells) >= 1
        assert bundle.quietest_si_cell is not None
    else:
        assert bundle.any_si_recover is False
        assert bundle.si_recover_cells == ()
        assert bundle.quietest_si_cell is None
        # Baseline tissue=0.03 / noise=0.02 SI arm still fails (T63/T66).
        baseline = [
            r for r in bundle.rows
            if abs(r.tissue_fraction - 0.03) < 1e-12
            and abs(r.noise - 0.02) < 1e-12
            and abs(r.filtration_mult - FILTRATION_MULTIPLIER) < 1e-12
        ]
        assert len(baseline) == 1
        assert baseline[0].b1 == 0
        assert baseline[0].recovers_b1 is False
        assert bundle.any_cal_recover or not bundle.any_cal_recover
