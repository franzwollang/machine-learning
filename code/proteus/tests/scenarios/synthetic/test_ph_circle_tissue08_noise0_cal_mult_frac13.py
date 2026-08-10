"""Circle tissue=0.08 × noise=0 fine cal-mult × frac∈[1,3] (#41 / A4-T78).

A4-T75: at tissue0.08×noise0, cal6 shows a tradeoff — ``frac≤1`` keeps b1 with
dirty ``b0``, while ``frac≥3`` cleans ``b0`` but kills b1 (``(1,0)``). No joint
clean. This harness freezes that clutter cell and sweeps a denser cal-mult
ladder around the tradeoff fracs ``{1.0, 1.5, 2.0, 2.5, 3.0}``, asking whether
any intermediate mult×frac jointly cleans ``b0=1`` and keeps ``b1≥1``.

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
# Fine dive around T75 tradeoff band (dirty_b0+b1 ↔ clean_b0 no_b1).
FRAC_GRID: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)
MULT_ARMS: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
EXPECTED_B1: int = 1


@dataclass(frozen=True)
class CircleTissue08CalMultFrac13Row:
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
class CircleTissue08CalMultFrac13Bundle:
    dataset_seed: int
    stage1_seed: int
    tissue_fraction: float
    noise: float
    frac_grid: tuple[float, ...]
    mult_arms: tuple[float, ...]
    n_signal: int
    sigma_star: float
    rows: tuple[CircleTissue08CalMultFrac13Row, ...]
    any_si_clean: bool
    any_cal_clean: bool
    any_cal_dirty_b0_with_b1: bool
    any_joint_near: bool
    clean_cells: tuple[tuple[float, float, tuple[int, ...]], ...]
    dirty_b1_cells: tuple[tuple[float, float, tuple[int, ...]], ...]
    best_cal_betti: tuple[int, ...] | None
    table: str


@pytest.fixture(scope="module")
def circle_tissue08_cal_mult_frac13_bundle() -> CircleTissue08CalMultFrac13Bundle:
    """Fine cal-mult × frac∈[1,3] dive at tissue=0.08, noise=0."""
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

    rows: list[CircleTissue08CalMultFrac13Row] = []
    clean_cells: list[tuple[float, float, tuple[int, ...]]] = []
    dirty_b1_cells: list[tuple[float, float, tuple[int, ...]]] = []
    any_si = False
    any_cal = False
    any_cal_dirty = False
    any_joint_near = False
    best_cal_betti: tuple[int, ...] | None = None
    best_cal_score = -1
    table_lines = ["mult\tfrac\tn_signal\tbetti\tb0\tb1\tclean\tdirty_b0"]

    for mult in MULT_ARMS:
        is_si = abs(float(mult) - FILTRATION_MULTIPLIER) < 1e-12
        is_cal = float(mult) >= 2.0
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
            # Near-joint: b0≤2 and b1≥1 (soft evidence of proximity).
            if b0 <= 2 and b1 >= EXPECTED_B1:
                any_joint_near = True
            if recovers:
                if is_si:
                    any_si = True
                else:
                    any_cal = True
                clean_cells.append((float(mult), float(frac), betti))
            elif (not is_si) and dirty_b1_only:
                any_cal_dirty = True
                dirty_b1_cells.append((float(mult), float(frac), betti))
            if is_cal and not is_si:
                score = (1000 if recovers else 0) + 10 * b1 - b0
                if score > best_cal_score:
                    best_cal_score = score
                    best_cal_betti = betti
            rows.append(
                CircleTissue08CalMultFrac13Row(
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

    return CircleTissue08CalMultFrac13Bundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        tissue_fraction=TISSUE,
        noise=NOISE,
        frac_grid=FRAC_GRID,
        mult_arms=MULT_ARMS,
        n_signal=n_signal,
        sigma_star=sigma,
        rows=tuple(rows),
        any_si_clean=any_si,
        any_cal_clean=any_cal,
        any_cal_dirty_b0_with_b1=any_cal_dirty,
        any_joint_near=any_joint_near,
        clean_cells=tuple(clean_cells),
        dirty_b1_cells=tuple(dirty_b1_cells),
        best_cal_betti=best_cal_betti,
        table="\n".join(table_lines),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_tissue08_cal_mult_frac13_harness_lands(
    circle_tissue08_cal_mult_frac13_bundle,
) -> None:
    """Tissue0.08×noise0 cal-mult×frac1-3 dive lands; SI defaults untouched."""
    bundle = circle_tissue08_cal_mult_frac13_bundle
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
def test_circle_tissue08_cal_mult_frac13_documents_gap(
    circle_tissue08_cal_mult_frac13_bundle,
) -> None:
    """Document fine cal-mult×frac1-3 vs joint clean; never flip awaiting.

    Soft: SI or cal clean recover is proposal-path evidence.
    Otherwise keep documenting the T75 tradeoff under a denser mult ladder.
    """
    bundle = circle_tissue08_cal_mult_frac13_bundle
    if bundle.any_si_clean or bundle.any_cal_clean:
        assert FILTRATION_MULTIPLIER == 1.5
        assert len(bundle.clean_cells) >= 1
    else:
        assert bundle.any_si_clean is False
        assert bundle.any_cal_clean is False
        assert bundle.clean_cells == ()
        # Tradeoff residue: dirty_b0+b1 cells and/or near-joint soft evidence.
        assert (
            bundle.any_cal_dirty_b0_with_b1
            or bundle.any_joint_near
            or bundle.best_cal_betti is not None
        )
