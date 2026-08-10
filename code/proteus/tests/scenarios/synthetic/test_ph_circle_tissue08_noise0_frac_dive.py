"""Circle cal × fine lifetime_frac dive at tissue=0.08 × noise=0 (#41 / A4-T75).

A4-T72: across tissue×noise×frac, cal cleaned most cells with min_frac floors,
but the ``tissue=0.08 × noise=0`` cell never reached clean ``(b0=1,b1=1)`` on
the coarse frac grid ``{0.5,2,4,8}``. This harness freezes that clutter cell
and sweeps a denser frac ladder under SI mult=1.5 and cal mult=6 (plus nearby
cal mults), asking whether any frac restores clean b1 or only rearranges
dirty ``b0``.

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
NOISE: float = 0.0
# Dense dive around the T72 null-clean cell (incl. very high frac).
FRAC_GRID: tuple[float, ...] = (
    0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0,
)
MULT_ARMS: tuple[float, ...] = (1.5, 3.0, 6.0, 8.0)
EXPECTED_B1: int = 1


@dataclass(frozen=True)
class CircleTissue08Noise0FracDiveRow:
    filtration_mult: float
    lifetime_frac: float
    n_signal: int
    sigma_star: float
    betti: tuple[int, ...]
    b0: int
    b1: int
    recovers_clean: bool
    dirty_b0: bool
    dirty_b1_only: bool


@dataclass(frozen=True)
class CircleTissue08Noise0FracDiveBundle:
    dataset_seed: int
    stage1_seed: int
    tissue_fraction: float
    noise: float
    frac_grid: tuple[float, ...]
    mult_arms: tuple[float, ...]
    n_signal: int
    sigma_star: float
    rows: tuple[CircleTissue08Noise0FracDiveRow, ...]
    si_clean_fracs: tuple[float, ...]
    cal_clean_fracs: tuple[float, ...]
    cal_dirty_b0_fracs: tuple[float, ...]
    any_si_clean: bool
    any_cal_clean: bool
    any_cal_dirty_b0_with_b1: bool
    min_frac_cal_clean: float | None
    best_cal_betti: tuple[int, ...] | None
    table: str


@pytest.fixture(scope="module")
def circle_tissue08_noise0_frac_dive_bundle() -> CircleTissue08Noise0FracDiveBundle:
    """Fine frac×mult dive at fixed tissue=0.08, noise=0."""
    dataset = make_circle(
        n_samples=N_SAMPLES,
        radius=1.0,
        noise=NOISE,
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
    result = run_scale_search(dataset.points, dim=gt.ambient_dim, config=config)
    pos = result.scaffold_at_star.node_positions()
    sigma = float(sigma_star_from_tau(result.tau_star))
    node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
    signal_pts = pos[node_labels == 0]
    n_signal = int(signal_pts.shape[0])

    rows: list[CircleTissue08Noise0FracDiveRow] = []
    si_clean: list[float] = []
    cal_clean: list[float] = []
    cal_dirty: list[float] = []
    any_si = False
    any_cal = False
    any_cal_dirty = False
    min_frac_clean: float | None = None
    best_cal_betti: tuple[int, ...] | None = None
    best_cal_score = -1
    table_lines = ["mult\tfrac\tn_signal\tbetti\tb0\tb1\tclean\tdirty_b0"]

    for mult in MULT_ARMS:
        is_si = abs(float(mult) - FILTRATION_MULTIPLIER) < 1e-12
        is_cal = abs(float(mult) - 6.0) < 1e-12 or float(mult) >= 3.0
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
            dirty_b0 = b0 > 1
            recovers = b1 >= EXPECTED_B1 and b0 == 1
            dirty_b1_only = b1 >= EXPECTED_B1 and dirty_b0
            if recovers:
                if is_si:
                    any_si = True
                    si_clean.append(float(frac))
                else:
                    any_cal = True
                    cal_clean.append(float(frac))
                    if min_frac_clean is None:
                        min_frac_clean = float(frac)
            elif (not is_si) and dirty_b1_only:
                any_cal_dirty = True
                cal_dirty.append(float(frac))
            if is_cal and not is_si:
                # Prefer clean; else prefer higher b1 with lower b0 inflate.
                score = (1000 if recovers else 0) + 10 * b1 - b0
                if score > best_cal_score:
                    best_cal_score = score
                    best_cal_betti = betti
            rows.append(
                CircleTissue08Noise0FracDiveRow(
                    filtration_mult=float(mult),
                    lifetime_frac=float(frac),
                    n_signal=n_signal,
                    sigma_star=sigma,
                    betti=betti,
                    b0=b0,
                    b1=b1,
                    recovers_clean=recovers,
                    dirty_b0=dirty_b0,
                    dirty_b1_only=dirty_b1_only,
                )
            )
            table_lines.append(
                f"{mult:g}\t{frac:g}\t{n_signal}\t{betti}\t{b0}\t{b1}\t"
                f"{int(recovers)}\t{int(dirty_b0)}"
            )

    return CircleTissue08Noise0FracDiveBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        tissue_fraction=TISSUE,
        noise=NOISE,
        frac_grid=FRAC_GRID,
        mult_arms=MULT_ARMS,
        n_signal=n_signal,
        sigma_star=sigma,
        rows=tuple(rows),
        si_clean_fracs=tuple(si_clean),
        cal_clean_fracs=tuple(cal_clean),
        cal_dirty_b0_fracs=tuple(cal_dirty),
        any_si_clean=any_si,
        any_cal_clean=any_cal,
        any_cal_dirty_b0_with_b1=any_cal_dirty,
        min_frac_cal_clean=min_frac_clean,
        best_cal_betti=best_cal_betti,
        table="\n".join(table_lines),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_tissue08_noise0_frac_dive_harness_lands(
    circle_tissue08_noise0_frac_dive_bundle,
) -> None:
    """Tissue0.08×noise0 frac dive lands; SI defaults untouched."""
    bundle = circle_tissue08_noise0_frac_dive_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert abs(bundle.tissue_fraction - TISSUE) < 1e-12
    assert abs(bundle.noise - NOISE) < 1e-12
    assert bundle.frac_grid == FRAC_GRID
    assert bundle.mult_arms == MULT_ARMS
    assert bundle.n_signal >= 8
    assert bundle.sigma_star > 0.0
    assert len(bundle.rows) == len(FRAC_GRID) * len(MULT_ARMS)
    header = bundle.table.splitlines()[0]
    assert "mult" in header and "frac" in header and "betti" in header


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_tissue08_noise0_frac_dive_documents_gap(
    circle_tissue08_noise0_frac_dive_bundle,
) -> None:
    """Document tissue0.08×noise0 frac dive vs clean b1; never flip awaiting.

    Soft: SI or cal clean recover at any frac is proposal-path evidence.
    Otherwise keep documenting the T72 null-clean cell under a denser ladder.
    """
    bundle = circle_tissue08_noise0_frac_dive_bundle
    if bundle.any_si_clean or bundle.any_cal_clean:
        assert FILTRATION_MULTIPLIER == 1.5
        if bundle.any_si_clean:
            assert len(bundle.si_clean_fracs) >= 1
        if bundle.any_cal_clean:
            assert len(bundle.cal_clean_fracs) >= 1
            assert bundle.min_frac_cal_clean is not None
    else:
        assert bundle.any_si_clean is False
        assert bundle.any_cal_clean is False
        assert bundle.si_clean_fracs == ()
        assert bundle.cal_clean_fracs == ()
        assert bundle.min_frac_cal_clean is None
        # Default-frac SI still fails on this clutter cell.
        baseline = [
            r for r in bundle.rows
            if abs(r.filtration_mult - FILTRATION_MULTIPLIER) < 1e-12
            and abs(r.lifetime_frac - DEFAULT_LIFETIME_FRAC) < 1e-12
        ]
        assert len(baseline) == 1
        assert baseline[0].recovers_clean is False
        assert bundle.any_cal_dirty_b0_with_b1 or bundle.best_cal_betti is not None
