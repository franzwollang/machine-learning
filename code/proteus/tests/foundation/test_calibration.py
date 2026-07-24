"""Unit tests for the c_{d,k} calibration protocol (SI S2.5.5, OPEN_ISSUES #28)."""

from __future__ import annotations

import numpy as np

from proteus.stage1.calibration import (
    CDK_TABLE,
    CQ_TABLE,
    CDKCalibrationConfig,
    CQCalibrationConfig,
    c_dk,
    c_dk_analytic,
    c_q,
    c_q_analytic,
    calibrate_cdk,
    calibrate_cq,
    measure_cdk_from_scaffold,
    measure_cq_from_scaffold,
    sample_unit_ball,
)


def test_sample_unit_ball_inside_ball() -> None:
    rng = np.random.default_rng(0)
    pts = sample_unit_ball(2000, 3, rng)

    assert pts.shape == (2000, 3)
    radii = np.linalg.norm(pts, axis=1)
    assert np.all(radii <= 1.0 + 1e-9)


def test_sample_unit_ball_radial_density() -> None:
    """Uniform-in-ball => radial CDF is r^d, so median radius ~ 0.5^{1/d}."""

    rng = np.random.default_rng(1)
    d = 2
    pts = sample_unit_ball(20000, d, rng)
    radii = np.linalg.norm(pts, axis=1)
    expected_median = 0.5 ** (1.0 / d)
    assert abs(float(np.median(radii)) - expected_median) < 0.02


def test_c_dk_analytic_positive_and_decreasing_in_d() -> None:
    vals = [c_dk_analytic(d, 8) for d in (1, 2, 3, 4, 5)]
    assert all(v > 0.0 for v in vals)
    # The k-NN-radius-to-sqrt(tau) ratio shrinks as dimension grows.
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_c_dk_table_lookup_when_present() -> None:
    if not CDK_TABLE:
        return
    (d, k), v = next(iter(CDK_TABLE.items()))
    assert c_dk(d, k) == v


def test_c_dk_falls_back_to_nearest_k() -> None:
    if not CDK_TABLE:
        return
    ds = {d for (d, _k) in CDK_TABLE}
    d = min(ds)
    ks = sorted(k for (dd, k) in CDK_TABLE if dd == d)
    # An out-of-grid k resolves to the nearest tabulated k at that dimension.
    assert c_dk(d, ks[-1] + 100) == CDK_TABLE[(d, ks[-1])]


def test_c_dk_analytic_fallback_for_untabulated_dim() -> None:
    # Dimension 42 is not tabulated; lookup must return the analytic anchor.
    assert c_dk(42, 8) == c_dk_analytic(42, 8)


def test_calibrate_cdk_matches_analytic_order_of_magnitude() -> None:
    """A fast single-ensemble calibration should land within a small factor of
    the isotropic analytic anchor (validates the measurement pipeline)."""

    cfg = CDKCalibrationConfig(
        n_samples=1000,
        n_ensembles=1,
        target_nodes=50,
        max_nodes=150,
        max_epochs=8,
    )
    d, k = 2, 8
    c = calibrate_cdk(d, k, config=cfg, seed=0)
    anchor = c_dk_analytic(d, k)
    assert np.isfinite(c)
    assert 0.4 * anchor < c < 2.5 * anchor


def test_measure_cdk_positive_on_small_scaffold() -> None:
    from proteus.stage1 import Stage1Scaffold

    rng = np.random.default_rng(3)
    pts = sample_unit_ball(1500, 2, rng)
    scaffold = Stage1Scaffold(
        dim=2, tau=0.01, k=8, min_nodes=4, max_nodes=120,
        ann_backend="naive", rng=np.random.default_rng(4),
    )
    scaffold.init_from(pts, n_seeds=4)
    scaffold.run_epoch(pts)

    c = measure_cdk_from_scaffold(scaffold)
    assert np.isfinite(c)
    assert c > 0.0


# ---------------------------------------------------------------------------
# C_Q(d) interior star-radius constant (SI S3.3, OPEN_ISSUES #36)
# ---------------------------------------------------------------------------


def test_c_q_analytic_positive_and_decreasing_in_d() -> None:
    vals = [c_q_analytic(d) for d in (1, 2, 3, 4, 5)]
    assert all(v > 0.0 for v in vals)
    # The cap-fixed catchment radius per unit sqrt(tau) shrinks as d grows.
    assert all(a > b for a, b in zip(vals, vals[1:]))
    # Closed form sqrt((d+2)/d): d=1 -> sqrt(3), d=2 -> sqrt(2).
    assert abs(c_q_analytic(1) - np.sqrt(3.0)) < 1e-12
    assert abs(c_q_analytic(2) - np.sqrt(2.0)) < 1e-12


def test_c_q_table_lookup_when_present() -> None:
    if not CQ_TABLE:
        return
    d, v = next(iter(CQ_TABLE.items()))
    assert c_q(d) == v


def test_c_q_table_monotone_decreasing_in_d() -> None:
    ds = sorted(CQ_TABLE)
    vals = [CQ_TABLE[d] for d in ds]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_c_q_dominates_analytic_anchor_on_grid() -> None:
    """Calibrated C_Q runs above the cap-radius anchor (boundary-corner slack)
    but within a small factor, validating the derived anchor."""

    for d, v in CQ_TABLE.items():
        anchor = c_q_analytic(d)
        assert v >= anchor
        assert v < 2.0 * anchor


def test_c_q_analytic_fallback_for_untabulated_dim() -> None:
    assert c_q(42) == c_q_analytic(42)


def test_calibrate_cq_matches_analytic_order_of_magnitude() -> None:
    """A fast single-ensemble calibration should land within a small factor of
    the cap-radius anchor (validates the star-radius measurement pipeline)."""

    cfg = CQCalibrationConfig(
        n_samples=1000,
        n_ensembles=1,
        target_nodes=50,
        max_nodes=150,
        max_epochs=8,
    )
    d = 2
    c = calibrate_cq(d, config=cfg, seed=0)
    anchor = c_q_analytic(d)
    assert np.isfinite(c)
    assert anchor * 0.7 < c < anchor * 2.5


def test_measure_cq_positive_on_small_scaffold() -> None:
    from proteus.stage1 import Stage1Scaffold

    rng = np.random.default_rng(3)
    pts = sample_unit_ball(1500, 2, rng)
    scaffold = Stage1Scaffold(
        dim=2, tau=0.01, k=8, min_nodes=4, max_nodes=120,
        ann_backend="naive", rng=np.random.default_rng(4),
    )
    scaffold.init_from(pts, n_seeds=4)
    scaffold.run_epoch(pts)

    c = measure_cq_from_scaffold(scaffold, pts)
    assert np.isfinite(c)
    assert c > 0.0
