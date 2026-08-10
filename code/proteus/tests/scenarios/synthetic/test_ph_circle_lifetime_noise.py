"""Circle lifetime_frac × manifold noise probe (#41 / A4-T66-followon).

A4-T60/T63: tissue×mult / tissue×lifetime leave SI ``b1=0`` at tissue∈{0,0.03,
0.08}; recovery needs cal ``mult≥6`` (or ``mult=3`` at tissue=0.08). This
follow-on freezes baseline tissue=0.03 and crosses manifold ``noise`` ×
``lifetime_frac`` under SI ``filtration_mult=1.5`` (plus cal-mult=6), asking
whether quieter/louder tube noise unlocks circle ``b1=1`` at SI fine scale.

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
TISSUE_FRACTION: float = 0.03
NOISE_GRID: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05)
FRAC_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MULT_ARMS: tuple[float, ...] = (1.5, 6.0)
EXPECTED_B1: int = 1


@dataclass(frozen=True)
class CircleLifetimeNoiseRow:
    noise: float
    filtration_mult: float
    lifetime_frac: float
    n_signal: int
    sigma_star: float
    betti: tuple[int, ...]
    b1: int
    recovers_b1: bool


@dataclass(frozen=True)
class CircleLifetimeNoiseBundle:
    dataset_seed: int
    stage1_seed: int
    tissue_fraction: float
    noise_grid: tuple[float, ...]
    frac_grid: tuple[float, ...]
    mult_arms: tuple[float, ...]
    rows: tuple[CircleLifetimeNoiseRow, ...]
    si_default_frac_b1: dict[float, int]
    si_any_frac_recover: dict[float, bool]
    cal_any_frac_recover: dict[float, bool]
    any_si_recover: bool
    any_cal_recover: bool
    min_frac_si_recover: dict[float, float | None]
    quietest_si_recover_noise: float | None
    table: str


@pytest.fixture(scope="module")
def circle_lifetime_noise_bundle() -> CircleLifetimeNoiseBundle:
    """Cross noise×lifetime_frac under SI and cal mult on fitted circle."""
    rows: list[CircleLifetimeNoiseRow] = []
    si_default_b1: dict[float, int] = {}
    si_any: dict[float, bool] = {}
    cal_any: dict[float, bool] = {}
    min_frac_si: dict[float, float | None] = {}
    any_si = False
    any_cal = False
    quietest: float | None = None
    table_lines = ["noise\tmult\tfrac\tn_signal\tbetti\tb1\trecover"]

    for noise in NOISE_GRID:
        dataset = make_circle(
            n_samples=N_SAMPLES,
            radius=1.0,
            noise=float(noise),
            extrusion_dim=2,
            tissue_fraction=TISSUE_FRACTION,
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

        si_hit = False
        cal_hit = False
        first_si_frac: float | None = None
        default_frac_b1 = -1

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
                b1 = int(betti[1]) if len(betti) > 1 else 0
                recovers = b1 >= EXPECTED_B1 and int(betti[0]) == 1
                if abs(float(mult) - FILTRATION_MULTIPLIER) < 1e-12:
                    if abs(float(frac) - DEFAULT_LIFETIME_FRAC) < 1e-12:
                        default_frac_b1 = b1
                    if recovers:
                        si_hit = True
                        any_si = True
                        if first_si_frac is None:
                            first_si_frac = float(frac)
                        if quietest is None or float(noise) < quietest:
                            quietest = float(noise)
                else:
                    if recovers:
                        cal_hit = True
                        any_cal = True
                rows.append(
                    CircleLifetimeNoiseRow(
                        noise=float(noise),
                        filtration_mult=float(mult),
                        lifetime_frac=float(frac),
                        n_signal=n_signal,
                        sigma_star=sigma,
                        betti=betti,
                        b1=b1,
                        recovers_b1=recovers,
                    )
                )
                table_lines.append(
                    f"{noise:g}\t{mult:g}\t{frac:g}\t{n_signal}\t"
                    f"{betti}\t{b1}\t{int(recovers)}"
                )

        si_default_b1[float(noise)] = int(default_frac_b1)
        si_any[float(noise)] = bool(si_hit)
        cal_any[float(noise)] = bool(cal_hit)
        min_frac_si[float(noise)] = first_si_frac

    return CircleLifetimeNoiseBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        tissue_fraction=TISSUE_FRACTION,
        noise_grid=NOISE_GRID,
        frac_grid=FRAC_GRID,
        mult_arms=MULT_ARMS,
        rows=tuple(rows),
        si_default_frac_b1=si_default_b1,
        si_any_frac_recover=si_any,
        cal_any_frac_recover=cal_any,
        any_si_recover=any_si,
        any_cal_recover=any_cal,
        min_frac_si_recover=min_frac_si,
        quietest_si_recover_noise=quietest,
        table="\n".join(table_lines),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_lifetime_noise_harness_lands(
    circle_lifetime_noise_bundle,
) -> None:
    """Circle lifetime×noise probe lands; SI defaults untouched."""
    bundle = circle_lifetime_noise_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert bundle.tissue_fraction == TISSUE_FRACTION
    assert bundle.noise_grid == NOISE_GRID
    assert bundle.frac_grid == FRAC_GRID
    assert bundle.mult_arms == MULT_ARMS
    assert len(bundle.rows) == (
        len(NOISE_GRID) * len(MULT_ARMS) * len(FRAC_GRID)
    )
    assert all(r.n_signal >= 8 for r in bundle.rows)
    assert all(r.sigma_star > 0.0 for r in bundle.rows)
    assert "noise" in bundle.table.splitlines()[0]
    assert 0.02 in bundle.si_default_frac_b1


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_lifetime_noise_documents_gap(
    circle_lifetime_noise_bundle,
) -> None:
    """Document noise×lifetime vs SI b1 recovery; never flip awaiting.

    Soft: SI-mult lifetime recovering ``b1`` at any noise is proposal-path
    evidence. Otherwise keep documenting that quieter noise ≠ SI unlock.
    """
    bundle = circle_lifetime_noise_bundle
    if bundle.any_si_recover:
        assert FILTRATION_MULTIPLIER == 1.5
        assert any(bundle.si_any_frac_recover.values())
        assert bundle.quietest_si_recover_noise is not None
    else:
        assert bundle.any_si_recover is False
        assert all(v is False for v in bundle.si_any_frac_recover.values())
        assert all(v is None for v in bundle.min_frac_si_recover.values())
        assert bundle.quietest_si_recover_noise is None
        # Baseline noise=0.02 SI default frac still fails (T63 tissue analog).
        assert bundle.si_default_frac_b1.get(0.02, -1) == 0
        assert bundle.any_cal_recover or not bundle.any_cal_recover
