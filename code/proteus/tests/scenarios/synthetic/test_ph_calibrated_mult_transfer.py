"""Transfer circle-calibrated filtration_mult onto nested/tori fitted PH (#41 / A4-T32).

Circle probes recover ``(b0,b1)=(1,1)`` at ``filtration_mult≥6`` (see
``test_ph_fitted_circle_calibration``). This harness asks whether that
calibrated mult transfers to nested shells ``(1,0,1)`` or linked tori
``(1,2,1)`` under fixed_threshold on denser fitted signal nodes.

Evidence-gathering only — does **not** flip ``@awaiting`` recovery tests or
change SI ``FILTRATION_MULTIPLIER=1.5``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.linked_tori import make_linked_tori
from tests.datasets.synthetic.nested_spheres import make_nested_spheres
from tests.metrics.persistent_homology import (
    FILTRATION_MULTIPLIER,
    format_per_region_ph_diagnostics,
    nearest_data_labels,
    run_per_region_ph,
    sigma_star_from_tau,
)

# Existence-proof mult from fitted-circle calibration (A4-T9); not a default.
CIRCLE_CALIBRATED_MULT: float = 6.0


@dataclass(frozen=True)
class CalibratedMultTransferRow:
    scenario: str
    n_signal: int
    sigma_star: float
    si_diag: str
    cal_diag: str
    si_all_match: bool | None
    cal_all_match: bool | None


def _fit_signal(
    points: np.ndarray,
    data_labels: np.ndarray,
    ambient_dim: int,
    tau_lo: float,
    tau_hi: float,
    include: list[int],
    max_nodes: int = 128,
):
    config = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=8,
        max_nodes=max_nodes,
        ann_backend="naive",
        stabilization=StabilizationConfig(
            min_equilibrium_epochs=3, max_epochs=15,
        ),
        seed=77,
    )
    result = run_scale_search(points, dim=ambient_dim, config=config)
    pos = result.scaffold_at_star.node_positions()
    sigma = sigma_star_from_tau(result.tau_star)
    node_labels = nearest_data_labels(pos, points, data_labels)
    mask = np.isin(node_labels, include)
    return pos[mask], node_labels[mask], float(sigma)


@pytest.fixture(scope="module")
def calibrated_mult_transfer_rows() -> tuple[CalibratedMultTransferRow, ...]:
    """Fit nested + tori once; compare SI mult vs circle-calibrated mult=6."""
    rows: list[CalibratedMultTransferRow] = []

    nested = make_nested_spheres(
        n_per_sphere=500,
        radii=(1.0, 2.0),
        ambient_dim=3,
        noise=0.02,
        tissue_fraction=0.03,
        seed=21,
    )
    gt = nested.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    npos, nlabs, nsig = _fit_signal(
        nested.points, nested.labels, gt.ambient_dim, tau_lo, tau_hi, [1, 2],
    )
    si_n = run_per_region_ph(
        npos, nlabs, nsig, scenario="nested_si_mult",
        include_labels=[1, 2], reading="fixed_threshold", max_dim=2,
        filtration_mult=FILTRATION_MULTIPLIER, expected_betti=(1, 0, 1),
    )
    cal_n = run_per_region_ph(
        npos, nlabs, nsig, scenario="nested_circle_cal_mult",
        include_labels=[1, 2], reading="fixed_threshold", max_dim=2,
        filtration_mult=CIRCLE_CALIBRATED_MULT, expected_betti=(1, 0, 1),
    )
    rows.append(
        CalibratedMultTransferRow(
            scenario="nested_spheres",
            n_signal=int(npos.shape[0]),
            sigma_star=nsig,
            si_diag=format_per_region_ph_diagnostics(si_n),
            cal_diag=format_per_region_ph_diagnostics(cal_n),
            si_all_match=si_n.all_match,
            cal_all_match=cal_n.all_match,
        )
    )

    tori = make_linked_tori(
        n_per_torus=500,
        major_radius=2.0,
        minor_radius=0.5,
        noise=0.02,
        tissue_fraction=0.03,
        seed=21,
    )
    gt = tori.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    tpos, tlabs, tsig = _fit_signal(
        tori.points, tori.labels, gt.ambient_dim, tau_lo, tau_hi, [0, 1],
    )
    si_t = run_per_region_ph(
        tpos, tlabs, tsig, scenario="tori_si_mult",
        include_labels=[0, 1], reading="fixed_threshold", max_dim=2,
        filtration_mult=FILTRATION_MULTIPLIER, expected_betti=(1, 2, 1),
    )
    cal_t = run_per_region_ph(
        tpos, tlabs, tsig, scenario="tori_circle_cal_mult",
        include_labels=[0, 1], reading="fixed_threshold", max_dim=2,
        filtration_mult=CIRCLE_CALIBRATED_MULT, expected_betti=(1, 2, 1),
    )
    rows.append(
        CalibratedMultTransferRow(
            scenario="linked_tori",
            n_signal=int(tpos.shape[0]),
            sigma_star=tsig,
            si_diag=format_per_region_ph_diagnostics(si_t),
            cal_diag=format_per_region_ph_diagnostics(cal_t),
            si_all_match=si_t.all_match,
            cal_all_match=cal_t.all_match,
        )
    )
    return tuple(rows)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_calibrated_mult_transfer_harness_lands(
    calibrated_mult_transfer_rows,
) -> None:
    """SI + circle-calibrated mult diagnostics land for nested and tori."""
    by = {r.scenario: r for r in calibrated_mult_transfer_rows}
    assert set(by) == {"nested_spheres", "linked_tori"}
    for row in calibrated_mult_transfer_rows:
        assert row.n_signal > 0
        assert row.sigma_star > 0.0
        assert "filtration_mult=1.5" in row.si_diag
        assert f"filtration_mult={CIRCLE_CALIBRATED_MULT:g}" in row.cal_diag
        assert FILTRATION_MULTIPLIER == 1.5


@pytest.mark.scenario
@pytest.mark.synthetic
def test_calibrated_mult_transfer_documents_gap(
    calibrated_mult_transfer_rows,
) -> None:
    """Document whether circle mult=6 transfers; never flip awaiting.

    Soft gate: if calibrated mult recovers all regions, transfer is viable
    evidence. Otherwise assert SI default remains non-recovering.
    """
    for row in calibrated_mult_transfer_rows:
        if row.cal_all_match:
            assert FILTRATION_MULTIPLIER == 1.5
            assert CIRCLE_CALIBRATED_MULT == 6.0
        else:
            assert row.cal_all_match is False
            # SI mult path still incomplete when transfer fails.
            assert row.si_all_match is False
