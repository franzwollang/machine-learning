"""Denser clean-grid vs fitted linked-tori PH density probe (#41 / A4-T38).

Clean torus grids recover ``(1,2,1)`` at modest n (24×12). Fitted Stage-1
scaffolds on interlocking tori still fail. This harness asks whether *denser*
clean grids stay green and whether denser dataset ``n_per_torus`` (with denser
``max_nodes``) moves fitted Betti toward ``b1=2``.

Evidence-gathering only — does **not** flip ``test_linked_tori_betti_numbers``
``@awaiting`` or change SI defaults.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.linked_tori import make_linked_tori
from tests.metrics.persistent_homology import (
    FILTRATION_MULTIPLIER,
    nearest_data_labels,
    run_per_region_ph,
    sigma_star_from_tau,
)
from tests.scenarios.synthetic.test_ph_reading_diagnostics import (
    _clean_torus_grid,
)

EXPECTED_TORI: tuple[int, ...] = (1, 2, 1)
CLEAN_SIGMA: float = 0.55
# (n_theta, n_phi) density ladder — keep VR clouds modest.
CLEAN_DENSITIES: tuple[tuple[int, int], ...] = (
    (24, 12),  # 288 pts / torus (known-green baseline)
    (32, 16),  # 512
    (40, 20),  # 800
)
FITTED_N_PER_TORUS: tuple[int, ...] = (500, 1000)
FITTED_MAX_NODES: int = 256


def _two_clean_tori(
    n_theta: int,
    n_phi: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    t0 = _clean_torus_grid(n_theta=n_theta, n_phi=n_phi, major=2.0, minor=0.5)
    t1 = _clean_torus_grid(n_theta=n_theta, n_phi=n_phi, major=2.0, minor=0.5)
    t1 = t1 + np.array([8.0, 0.0, 0.0])
    points = np.vstack([t0, t1])
    labels = np.array([0] * len(t0) + [1] * len(t1), dtype=int)
    return points, labels, CLEAN_SIGMA


@dataclass(frozen=True)
class CleanDensityRow:
    n_theta: int
    n_phi: int
    n_per_torus: int
    all_match: bool | None
    betti: dict[int, tuple[int, ...]]
    max_b1: int


@dataclass(frozen=True)
class FittedDensityRow:
    n_per_torus_data: int
    max_nodes: int
    n_signal: int
    n_fitted_per_torus: dict[int, int]
    all_match: bool | None
    betti: dict[int, tuple[int, ...]]
    max_b1: int


@dataclass(frozen=True)
class LinkedToriDenserCleanVsFittedBundle:
    clean_rows: tuple[CleanDensityRow, ...]
    fitted_rows: tuple[FittedDensityRow, ...]
    clean_all_recover: bool
    fitted_any_recover: bool
    fitted_max_b1: int


@pytest.fixture(scope="module")
def linked_tori_denser_clean_vs_fitted_bundle() -> LinkedToriDenserCleanVsFittedBundle:
    """Sweep clean density ladder + denser fitted n_per_torus / max_nodes."""
    clean_rows: list[CleanDensityRow] = []
    for n_th, n_ph in CLEAN_DENSITIES:
        pts, labs, sigma = _two_clean_tori(n_th, n_ph)
        result = run_per_region_ph(
            pts,
            labs,
            sigma,
            scenario=f"linked_tori_clean_{n_th}x{n_ph}",
            include_labels=[0, 1],
            reading="fixed_threshold",
            max_dim=2,
            filtration_mult=FILTRATION_MULTIPLIER,
            expected_betti=EXPECTED_TORI,
        )
        betti = {
            int(r.region_id): tuple(int(x) for x in r.betti) for r in result.reports
        }
        max_b1 = max((int(b[1]) for b in betti.values()), default=0)
        clean_rows.append(
            CleanDensityRow(
                n_theta=n_th,
                n_phi=n_ph,
                n_per_torus=n_th * n_ph,
                all_match=result.all_match,
                betti=betti,
                max_b1=max_b1,
            )
        )

    fitted_rows: list[FittedDensityRow] = []
    for n_data in FITTED_N_PER_TORUS:
        dataset = make_linked_tori(
            n_per_torus=n_data,
            major_radius=2.0,
            minor_radius=0.5,
            noise=0.02,
            tissue_fraction=0.03,
            seed=21,
        )
        gt = dataset.ground_truth
        tau_lo, tau_hi = gt.tau_grid_hint
        config = ScaleSearchConfig(
            tau_min=tau_lo,
            tau_max=tau_hi,
            max_grid_points=8,
            k=8,
            n_seeds=8,
            max_nodes=FITTED_MAX_NODES,
            ann_backend="naive",
            stabilization=StabilizationConfig(
                min_equilibrium_epochs=3, max_epochs=15,
            ),
            seed=77,
        )
        result = run_scale_search(
            dataset.points, dim=gt.ambient_dim, config=config,
        )
        pos = result.scaffold_at_star.node_positions()
        sigma = sigma_star_from_tau(result.tau_star)
        node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
        signal_mask = np.isin(node_labels, [0, 1])
        signal_pos = pos[signal_mask]
        signal_labs = node_labels[signal_mask]
        ph = run_per_region_ph(
            signal_pos,
            signal_labs,
            sigma,
            scenario=f"linked_tori_fitted_n{n_data}_m{FITTED_MAX_NODES}",
            include_labels=[0, 1],
            reading="fixed_threshold",
            max_dim=2,
            filtration_mult=FILTRATION_MULTIPLIER,
            expected_betti=EXPECTED_TORI,
        )
        betti = {
            int(r.region_id): tuple(int(x) for x in r.betti) for r in ph.reports
        }
        max_b1 = max((int(b[1]) for b in betti.values()), default=0)
        fitted_rows.append(
            FittedDensityRow(
                n_per_torus_data=n_data,
                max_nodes=FITTED_MAX_NODES,
                n_signal=int(signal_pos.shape[0]),
                n_fitted_per_torus={
                    int(lab): int(np.sum(signal_labs == lab)) for lab in (0, 1)
                },
                all_match=ph.all_match,
                betti=betti,
                max_b1=max_b1,
            )
        )

    return LinkedToriDenserCleanVsFittedBundle(
        clean_rows=tuple(clean_rows),
        fitted_rows=tuple(fitted_rows),
        clean_all_recover=all(r.all_match is True for r in clean_rows),
        fitted_any_recover=any(r.all_match is True for r in fitted_rows),
        fitted_max_b1=max((r.max_b1 for r in fitted_rows), default=0),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_denser_clean_vs_fitted_harness_lands(
    linked_tori_denser_clean_vs_fitted_bundle,
) -> None:
    """Density ladder lands; SI fine mult untouched."""
    bundle = linked_tori_denser_clean_vs_fitted_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert len(bundle.clean_rows) == len(CLEAN_DENSITIES)
    assert len(bundle.fitted_rows) == len(FITTED_N_PER_TORUS)
    assert bundle.clean_rows[0].n_per_torus == 24 * 12
    assert bundle.fitted_rows[0].max_nodes == FITTED_MAX_NODES
    assert bundle.fitted_rows[0].n_signal > 0


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_denser_clean_vs_fitted_documents_gap(
    linked_tori_denser_clean_vs_fitted_bundle,
) -> None:
    """Clean denser grids stay green; document fitted denser gap; no awaiting flip.

    Soft: if fitted recovers at denser n_per_torus, record as proposal-path
    evidence. Otherwise keep explicit fitted_max_b1 < 2 and clean_all_recover.
    """
    bundle = linked_tori_denser_clean_vs_fitted_bundle
    assert bundle.clean_all_recover is True
    for row in bundle.clean_rows:
        assert row.all_match is True
        assert row.max_b1 >= 2
        assert all(b == EXPECTED_TORI for b in row.betti.values())
    if bundle.fitted_any_recover:
        assert FILTRATION_MULTIPLIER == 1.5
        assert any(r.all_match is True for r in bundle.fitted_rows)
    else:
        assert bundle.fitted_any_recover is False
        assert bundle.fitted_max_b1 < 2
        for row in bundle.fitted_rows:
            assert row.all_match is not True
