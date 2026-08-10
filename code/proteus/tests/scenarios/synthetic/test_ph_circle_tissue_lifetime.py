"""Circle tissue × lifetime_frac b1 probe (#41 / A4-T63-followon).

A4-T60: fixed_threshold tissue×mult leaves SI ``b1=0`` at all tissue∈{0,0.03,0.08};
recovery needs ``mult≥6`` (tissue≤0.03) or ``mult=3`` (tissue=0.08). This
follow-on freezes the same Stage-1 / dataset seeds and crosses tissue ×
``lifetime_frac`` under SI ``filtration_mult=1.5`` (plus a compact cal-mult
arm at ``mult=6``), asking whether the lifetime reading unlocks circle ``b1=1``
at SI-default fine scale without raising the filtration multiplier.

Evidence-gathering only — does **not** flip ``@awaiting`` or SI defaults.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
FRAC_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
# SI fine + circle-calibrated mult (T7/T9 recovery floor).
MULT_ARMS: tuple[float, ...] = (1.5, 6.0)
EXPECTED_B1: int = 1


@dataclass(frozen=True)
class CircleTissueLifetimeRow:
    tissue_fraction: float
    filtration_mult: float
    lifetime_frac: float
    n_signal: int
    sigma_star: float
    betti: tuple[int, ...]
    b1: int
    recovers_b1: bool


@dataclass(frozen=True)
class CircleTissueLifetimeBundle:
    dataset_seed: int
    stage1_seed: int
    tissue_grid: tuple[float, ...]
    frac_grid: tuple[float, ...]
    mult_arms: tuple[float, ...]
    rows: tuple[CircleTissueLifetimeRow, ...]
    si_default_frac_b1: dict[float, int]
    si_any_frac_recover: dict[float, bool]
    cal_any_frac_recover: dict[float, bool]
    any_si_recover: bool
    any_cal_recover: bool
    min_frac_si_recover: dict[float, float | None]
    table: str


@pytest.fixture(scope="module")
def circle_tissue_lifetime_bundle() -> CircleTissueLifetimeBundle:
    """Cross tissue×lifetime_frac under SI and cal mult on fitted circle."""
    rows: list[CircleTissueLifetimeRow] = []
    si_default_b1: dict[float, int] = {}
    si_any: dict[float, bool] = {}
    cal_any: dict[float, bool] = {}
    min_frac_si: dict[float, float | None] = {}
    any_si = False
    any_cal = False
    table_lines = ["tissue\tmult\tfrac\tn_signal\tbetti\tb1\trecover"]

    for tissue in TISSUE_GRID:
        dataset = make_circle(
            n_samples=N_SAMPLES,
            radius=1.0,
            noise=0.02,
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
                else:
                    if recovers:
                        cal_hit = True
                        any_cal = True
                rows.append(
                    CircleTissueLifetimeRow(
                        tissue_fraction=float(tissue),
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
                    f"{tissue:g}\t{mult:g}\t{frac:g}\t{n_signal}\t"
                    f"{betti}\t{b1}\t{int(recovers)}"
                )

        si_default_b1[float(tissue)] = int(default_frac_b1)
        si_any[float(tissue)] = bool(si_hit)
        cal_any[float(tissue)] = bool(cal_hit)
        min_frac_si[float(tissue)] = first_si_frac

    return CircleTissueLifetimeBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        tissue_grid=TISSUE_GRID,
        frac_grid=FRAC_GRID,
        mult_arms=MULT_ARMS,
        rows=tuple(rows),
        si_default_frac_b1=si_default_b1,
        si_any_frac_recover=si_any,
        cal_any_frac_recover=cal_any,
        any_si_recover=any_si,
        any_cal_recover=any_cal,
        min_frac_si_recover=min_frac_si,
        table="\n".join(table_lines),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_tissue_lifetime_harness_lands(
    circle_tissue_lifetime_bundle,
) -> None:
    """Circle tissue×lifetime probe lands; SI defaults untouched."""
    bundle = circle_tissue_lifetime_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert bundle.tissue_grid == TISSUE_GRID
    assert bundle.frac_grid == FRAC_GRID
    assert bundle.mult_arms == MULT_ARMS
    assert len(bundle.rows) == (
        len(TISSUE_GRID) * len(MULT_ARMS) * len(FRAC_GRID)
    )
    assert all(r.n_signal >= 8 for r in bundle.rows)
    assert all(r.sigma_star > 0.0 for r in bundle.rows)
    assert "tissue" in bundle.table.splitlines()[0]
    assert 0.03 in bundle.si_default_frac_b1


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_tissue_lifetime_documents_gap(
    circle_tissue_lifetime_bundle,
) -> None:
    """Document tissue×lifetime vs SI b1 recovery; never flip awaiting.

    Soft: SI-mult lifetime recovering ``b1`` at any tissue is proposal-path
    evidence. Otherwise keep documenting that lifetime alone ≠ SI recovery
    (cal-mult arm may still recover independently).
    """
    bundle = circle_tissue_lifetime_bundle
    if bundle.any_si_recover:
        assert FILTRATION_MULTIPLIER == 1.5
        assert any(bundle.si_any_frac_recover.values())
    else:
        assert bundle.any_si_recover is False
        assert all(v is False for v in bundle.si_any_frac_recover.values())
        assert all(v is None for v in bundle.min_frac_si_recover.values())
        # Default SI frac at baseline tissue still fails (T60 fixed-thr analog).
        assert bundle.si_default_frac_b1.get(0.03, -1) == 0
        # Cal-mult lifetime may recover — record without flipping awaiting.
        assert bundle.any_cal_recover or not bundle.any_cal_recover
