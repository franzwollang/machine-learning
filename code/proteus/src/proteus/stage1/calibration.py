"""Calibration of the variance-cap constant c_{d,k} (SI S2.5.5, OPEN_ISSUES #28).

The scale calibration constant ``c_{d,k}`` maps the operational variance cap
``tau`` to the expected k-nearest-neighbor radius of an equilibrated scaffold
under local isotropy: at equilibrium a cap-satisfied catchment locks its
effective radius to ``r_{k,i} ~ c_{d,k} * sqrt(tau)`` (SI S2.5, S2.5.4, S11.1).

``c_{d,k}`` is a **calibrated** constant, not an analytic one: it is measured on
a declared reference ensemble (uniform samples in the unit d-ball) by running the
fixed-tau Stage 1 scaffold to equilibrium and taking the median of
``r_{k,i} / sqrt(tau)`` over mature nodes, tabulated over ``(d, k)``. The shipped
table :data:`CDK_TABLE` was produced by :func:`calibrate_cdk` with the parameters
recorded in :data:`CDK_CALIBRATION_META`; :func:`c_dk` is the runtime lookup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Reference ensemble
# ---------------------------------------------------------------------------


def sample_unit_ball(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` samples uniformly from the unit ``d``-ball.

    Uses the standard normalized-Gaussian + radial-power method: a direction
    drawn uniformly on the sphere times a radius ``U^{1/d}`` for ``U ~ U(0,1)``
    gives a uniform density on the ball (SI S2.5.5 reference ensemble).
    """

    if int(n) < 1:
        raise ValueError("n must be positive")
    if int(d) < 1:
        raise ValueError("d must be positive")
    directions = rng.normal(size=(int(n), int(d)))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    directions /= norms
    radii = rng.uniform(size=(int(n), 1)) ** (1.0 / float(d))
    return directions * radii


# ---------------------------------------------------------------------------
# Measurement at equilibrium
# ---------------------------------------------------------------------------


def measure_cdk_from_scaffold(scaffold: object) -> float:
    """Median ``r_{k,i} / sqrt(tau)`` over mature nodes of an equilibrated scaffold.

    This is the empirical estimate of ``c_{d,k}`` for a single equilibrated
    scaffold (SI S2.5.5). Only mature nodes (``update_count >= prune_after``)
    contribute, matching the maturity gate used by the scale-response and
    stabilization diagnostics.
    """

    nodes = list(getattr(scaffold, "nodes"))
    n = len(nodes)
    if n < 2:
        return float("nan")
    tau = float(getattr(scaffold, "tau"))
    if tau <= 0.0:
        return float("nan")
    prune_after = int(getattr(scaffold, "prune_after", 0))
    positions = scaffold.node_positions()  # type: ignore[attr-defined]
    k = min(int(getattr(scaffold, "k")), n - 1)
    if k < 1:
        return float("nan")

    sqrt_tau = float(np.sqrt(tau))
    ratios: list[float] = []
    for i, node in enumerate(nodes):
        if int(getattr(node, "update_count", 0)) < prune_after:
            continue
        _, dists = scaffold.ann.query_knn(positions[i], k=k + 1)  # type: ignore[attr-defined]
        r_k = float(dists[k]) if len(dists) > k else float(dists[-1])
        ratios.append(r_k / sqrt_tau)
    if not ratios:
        # Fall back to all nodes if the maturity gate excluded everyone.
        for i in range(n):
            _, dists = scaffold.ann.query_knn(positions[i], k=k + 1)  # type: ignore[attr-defined]
            r_k = float(dists[k]) if len(dists) > k else float(dists[-1])
            ratios.append(r_k / sqrt_tau)
    return float(np.median(ratios))


# ---------------------------------------------------------------------------
# Calibration driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CDKCalibrationConfig:
    """Reference-ensemble parameters for a single ``(d, k)`` calibration point.

    ``tau`` is chosen per dimension so that the equilibrium node count on the
    unit d-ball lands near ``target_nodes`` (using the uniform-density estimate
    ``N ~ (d / ((d + 2) * tau))^{d/2}``), keeping enough mature nodes for a
    stable median while bounding runtime. All fields are operational calibration
    parameters (they set the measurement protocol, not model behavior).
    """

    n_samples: int = 3000
    n_ensembles: int = 4
    target_nodes: int = 120
    max_nodes: int = 400
    min_nodes: int = 4
    max_epochs: int = 25
    ann_backend: str = "auto"


def _tau_for_target(d: int, target_nodes: int) -> float:
    """Cap giving ~``target_nodes`` mature nodes on the unit d-ball.

    Inverts the uniform-density equilibrium estimate ``N ~ (V_ball / V_cell)``
    with cell radius ``R`` satisfying ``variance = d/(d+2) R^2 = tau``.
    """

    d_f = float(d)
    return (d_f / (d_f + 2.0)) * float(target_nodes) ** (-2.0 / d_f)


def calibrate_cdk(
    d: int,
    k: int,
    *,
    config: CDKCalibrationConfig | None = None,
    seed: int = 0,
) -> float:
    """Calibrate ``c_{d,k}`` on the uniform d-ball reference ensemble (SI S2.5.5).

    Runs ``config.n_ensembles`` independent uniform-d-ball scaffolds to
    equilibrium at a dimension-appropriate cap and returns the median of the
    per-scaffold ``median(r_{k,i} / sqrt(tau))`` estimates.
    """

    # Local import to avoid a package import cycle at module load time.
    from proteus.stage1 import Stage1Scaffold
    from proteus.stage1.stabilization import StabilizationConfig

    cfg = config if config is not None else CDKCalibrationConfig()
    tau = _tau_for_target(int(d), cfg.target_nodes)
    stab = StabilizationConfig(max_epochs=cfg.max_epochs)
    estimates: list[float] = []
    for e in range(cfg.n_ensembles):
        rng = np.random.default_rng(seed + e)
        points = sample_unit_ball(cfg.n_samples, int(d), rng)
        scaffold = Stage1Scaffold(
            dim=int(d),
            tau=tau,
            k=int(k),
            min_nodes=cfg.min_nodes,
            max_nodes=cfg.max_nodes,
            ann_backend=cfg.ann_backend,
            rng=np.random.default_rng(seed + e + 10_000),
        )
        scaffold.init_from(points, n_seeds=cfg.min_nodes)
        scaffold.run_until_stable(points, stab)
        est = measure_cdk_from_scaffold(scaffold)
        if np.isfinite(est):
            estimates.append(est)
    if not estimates:
        raise RuntimeError(f"calibration produced no finite estimate for d={d}, k={k}")
    return float(np.median(estimates))


# ---------------------------------------------------------------------------
# Shipped lookup table
# ---------------------------------------------------------------------------

# Metadata describing how CDK_TABLE was produced (reproducibility record).
CDK_CALIBRATION_META: dict[str, object] = {
    "protocol": "SI S2.5.5",
    "ensemble": "uniform unit d-ball",
    "statistic": "median over mature nodes of r_k / sqrt(tau), median over ensembles",
    "config": {
        "n_samples": 3000,
        "n_ensembles": 4,
        "target_nodes": 120,
        "max_nodes": 400,
        "max_epochs": 25,
        "ann_backend": "auto",
    },
    "grid": {"d": [1, 2, 3, 4], "k": [6, 8, 10, 12]},
    "seed": 0,
}

# Calibrated c_{d,k} values (see CDK_CALIBRATION_META). Regenerate with
# ``python -m proteus.stage1.calibration``.
CDK_TABLE: dict[tuple[int, int], float] = {
    (1, 6): 7.9362, (1, 8): 9.7148, (1, 10): 11.2209, (1, 12): 12.6680,
    (2, 6): 2.5532, (2, 8): 2.8373, (2, 10): 3.0735, (2, 12): 3.4924,
    (3, 6): 1.4963, (3, 8): 1.6144, (3, 10): 1.8086, (3, 12): 1.9452,
    (4, 6): 1.2759, (4, 8): 1.3244, (4, 10): 1.3816, (4, 12): 1.4626,
}


def c_dk(d: int, k: int) -> float:
    """Look up the calibrated ``c_{d,k}`` (SI S2.5.5).

    Falls back to the nearest calibrated ``k`` at the requested dimension, then
    to the isotropic analytic estimate :func:`c_dk_analytic` when the dimension
    is not tabulated. The analytic fallback keeps the scale-response diagnostic
    well-defined for dimensions outside the shipped table.
    """

    d_i, k_i = int(d), int(k)
    if (d_i, k_i) in CDK_TABLE:
        return CDK_TABLE[(d_i, k_i)]
    ks_at_d = sorted(kk for (dd, kk) in CDK_TABLE if dd == d_i)
    if ks_at_d:
        nearest_k = min(ks_at_d, key=lambda kk: abs(kk - k_i))
        return CDK_TABLE[(d_i, nearest_k)]
    return c_dk_analytic(d_i, k_i)


def c_dk_analytic(d: int, k: int) -> float:
    """Isotropic uniform-density estimate of ``c_{d,k}`` (SI S2.5.5 fallback).

    Under a locally uniform density with unit node spacing, the expected k-NN
    radius scales as ``(k / (rho V_d))^{1/d}`` while the variance cap fixes the
    catchment radius via ``d/(d+2) R^2 = tau``. Combining the two gives an
    order-of-magnitude ``c_{d,k}`` used only when a dimension is untabulated.
    """

    d_f, k_f = float(d), float(k)
    # Expected k-NN radius in units of the mean node spacing on a d-lattice.
    from scipy.special import gammaln

    v_d = float(np.pi ** (d_f / 2.0) / np.exp(gammaln(d_f / 2.0 + 1.0)))
    r_over_spacing = (k_f / v_d) ** (1.0 / d_f)
    # Cap-fixed catchment radius per unit variance: R = sqrt((d+2)/d).
    cap_radius = float(np.sqrt((d_f + 2.0) / d_f))
    return r_over_spacing * cap_radius


def _regenerate_table(
    ds: tuple[int, ...] = (1, 2, 3, 4),
    ks: tuple[int, ...] = (6, 8, 10, 12),
    *,
    config: CDKCalibrationConfig | None = None,
    seed: int = 0,
) -> dict[tuple[int, int], float]:
    """Recompute the full ``(d, k)`` table (used by ``__main__``)."""

    table: dict[tuple[int, int], float] = {}
    for d in ds:
        for k in ks:
            table[(d, k)] = calibrate_cdk(d, k, config=config, seed=seed)
    return table


if __name__ == "__main__":  # pragma: no cover - calibration entry point
    import json

    result = _regenerate_table()
    printable = {f"{d},{k}": round(v, 4) for (d, k), v in result.items()}
    print(json.dumps(printable, indent=2, sort_keys=True))
