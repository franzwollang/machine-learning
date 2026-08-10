"""Circle tissue×filtration_mult b1 probe (#41 / A4-T60-followon).

A4-T7/T9: fitted circle signal nodes need ``mult≈6`` for ``b1=1``; SI
``1.5 σ*`` yields ``b1=0``. Default recipe uses ``tissue_fraction=0.03``.
This harness freezes Stage-1 seed=77 / dataset seed=21 and crosses a compact
tissue grid × filtration_mult ladder on NN-filtered signal nodes, asking
whether cleaner tissue alone unlocks SI-default ``b1=1`` (or lowers the
recovery mult).

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
    FILTRATION_MULTIPLIER,
    betti_numbers,
    filtration_radius,
    nearest_data_labels,
    sigma_star_from_tau,
)

DATASET_SEED: int = 21
STAGE1_SEED: int = 77
N_SAMPLES: int = 1200
TISSUE_GRID: tuple[float, ...] = (0.0, 0.03, 0.08)
MULT_GRID: tuple[float, ...] = (1.5, 3.0, 6.0, 8.0, 10.0)
EXPECTED_B1: int = 1


@dataclass(frozen=True)
class CircleTissueMultRow:
    tissue_fraction: float
    filtration_mult: float
    n_signal: int
    sigma_star: float
    betti: tuple[int, ...]
    b1: int
    recovers_b1: bool


@dataclass(frozen=True)
class CircleTissueB1ProbeBundle:
    dataset_seed: int
    stage1_seed: int
    tissue_grid: tuple[float, ...]
    mult_grid: tuple[float, ...]
    rows: tuple[CircleTissueMultRow, ...]
    si_b1_by_tissue: dict[float, int]
    min_mult_recovering_b1: dict[float, float | None]
    any_si_b1: bool
    any_recover: bool
    baseline_tissue: float
    baseline_si_b1: int
    baseline_min_mult: float | None
    table: str


@pytest.fixture(scope="module")
def circle_tissue_b1_probe_bundle() -> CircleTissueB1ProbeBundle:
    """Cross tissue×mult on fitted circle signal nodes; fixed_threshold."""
    rows: list[CircleTissueMultRow] = []
    si_b1: dict[float, int] = {}
    min_mult: dict[float, float | None] = {}
    any_si = False
    any_recover = False
    table_lines = ["tissue\tmult\tn_signal\tbetti\tb1\trecover"]

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
        first_ok: float | None = None

        for mult in MULT_GRID:
            r = filtration_radius(sigma, multiplier=float(mult))
            betti = tuple(
                int(x) for x in betti_numbers(signal_pts, threshold=r, max_dim=1)
            )
            b1 = int(betti[1]) if len(betti) > 1 else 0
            recovers = b1 >= EXPECTED_B1 and int(betti[0]) == 1
            if recovers:
                any_recover = True
                if first_ok is None:
                    first_ok = float(mult)
            if abs(float(mult) - FILTRATION_MULTIPLIER) < 1e-12:
                si_b1[float(tissue)] = b1
                if recovers:
                    any_si = True
            rows.append(
                CircleTissueMultRow(
                    tissue_fraction=float(tissue),
                    filtration_mult=float(mult),
                    n_signal=n_signal,
                    sigma_star=sigma,
                    betti=betti,
                    b1=b1,
                    recovers_b1=recovers,
                )
            )
            table_lines.append(
                f"{tissue:g}\t{mult:g}\t{n_signal}\t{betti}\t{b1}\t{int(recovers)}"
            )
        min_mult[float(tissue)] = first_ok

    baseline = 0.03
    return CircleTissueB1ProbeBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        tissue_grid=TISSUE_GRID,
        mult_grid=MULT_GRID,
        rows=tuple(rows),
        si_b1_by_tissue=si_b1,
        min_mult_recovering_b1=min_mult,
        any_si_b1=any_si,
        any_recover=any_recover,
        baseline_tissue=baseline,
        baseline_si_b1=int(si_b1.get(baseline, -1)),
        baseline_min_mult=min_mult.get(baseline),
        table="\n".join(table_lines),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_tissue_b1_probe_harness_lands(
    circle_tissue_b1_probe_bundle,
) -> None:
    """Circle tissue×mult probe lands; SI fine mult untouched."""
    bundle = circle_tissue_b1_probe_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert bundle.tissue_grid == TISSUE_GRID
    assert bundle.mult_grid == MULT_GRID
    assert len(bundle.rows) == len(TISSUE_GRID) * len(MULT_GRID)
    assert all(r.n_signal >= 8 for r in bundle.rows)
    assert all(r.sigma_star > 0.0 for r in bundle.rows)
    assert "tissue" in bundle.table.splitlines()[0]
    # Baseline tissue must be present in SI map.
    assert bundle.baseline_tissue in bundle.si_b1_by_tissue


@pytest.mark.scenario
@pytest.mark.synthetic
def test_circle_tissue_b1_probe_documents_gap(
    circle_tissue_b1_probe_bundle,
) -> None:
    """Document tissue vs SI b1 recovery; never flip awaiting.

    Soft: SI-default ``b1=1`` at any tissue is proposal-path evidence.
    Otherwise keep documenting that tissue ablation alone ≠ SI recovery
    (min recovering mult still ≫ 1.5 when recovery exists).
    """
    bundle = circle_tissue_b1_probe_bundle
    if bundle.any_si_b1:
        assert FILTRATION_MULTIPLIER == 1.5
        assert any(b >= 1 for b in bundle.si_b1_by_tissue.values())
    else:
        assert bundle.any_si_b1 is False
        assert all(b == 0 for b in bundle.si_b1_by_tissue.values())
        # T7/T9: baseline still fails at SI; recovery only at larger mult if at all.
        assert bundle.baseline_si_b1 == 0
        if bundle.any_recover:
            assert bundle.baseline_min_mult is not None
            assert bundle.baseline_min_mult > FILTRATION_MULTIPLIER
        else:
            assert all(v is None for v in bundle.min_mult_recovering_b1.values())
