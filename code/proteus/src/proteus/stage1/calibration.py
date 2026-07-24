"""Calibrated variance-cap constants on the uniform d-ball ensemble.

Two calibrated constants are measured on the same reference ensemble (uniform
samples in the unit d-ball, run to Stage 1 equilibrium):

``c_{d,k}`` (SI S2.5.5, OPEN_ISSUES #28) maps the operational variance cap
``tau`` to the expected k-nearest-neighbor radius of an equilibrated scaffold
under local isotropy: at equilibrium a cap-satisfied catchment locks its
effective radius to ``r_{k,i} ~ c_{d,k} * sqrt(tau)`` (SI S2.5, S2.5.4, S11.1).
It is measured as the median of ``r_{k,i} / sqrt(tau)`` over mature nodes,
tabulated over ``(d, k)`` in :data:`CDK_TABLE`.

``C_Q(d)`` (SI S3.3, OPEN_ISSUES #36) is the interior star-radius constant used
by the Stage-2 prune/merge guards: the worst-case reassignment distance of a
cap-equilibrated interior cell, normalized by ``sqrt(tau)``. It is measured as
the median of ``rho_prune / sqrt(tau)`` over the regular interior on the *same*
equilibrated scaffolds and tabulated over ``d`` in :data:`CQ_TABLE`.

Both are **calibrated** constants, not analytic ones. The shipped tables were
produced by :func:`calibrate_cdk` / :func:`calibrate_cq` with the parameters in
:data:`CDK_CALIBRATION_META` / :data:`CQ_CALIBRATION_META`; :func:`c_dk` and
:func:`c_q` are the runtime lookups. Regenerate both via
``python -m proteus.stage1.calibration``.
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


def _build_equilibrated_scaffold(
    d: int,
    k: int,
    *,
    n_samples: int,
    target_nodes: int,
    min_nodes: int,
    max_nodes: int,
    max_epochs: int,
    ann_backend: str,
    seed: int,
    ensemble: int,
) -> tuple[object, np.ndarray]:
    """Build one equilibrated uniform-d-ball scaffold (SI S2.5.5 ensemble).

    Shared by the ``c_{d,k}`` (k-NN radius) and ``C_Q(d)`` (star radius)
    calibrations so both constants are measured on the *identical* reference
    scaffolds. RNG seeding is deterministic in ``(seed, ensemble)``: the sample
    uses ``seed + ensemble`` and the scaffold uses ``seed + ensemble + 10_000``.
    """

    # Local import to avoid a package import cycle at module load time.
    from proteus.stage1 import Stage1Scaffold
    from proteus.stage1.stabilization import StabilizationConfig

    tau = _tau_for_target(int(d), target_nodes)
    points = sample_unit_ball(n_samples, int(d), np.random.default_rng(seed + ensemble))
    scaffold = Stage1Scaffold(
        dim=int(d),
        tau=tau,
        k=int(k),
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        ann_backend=ann_backend,
        rng=np.random.default_rng(seed + ensemble + 10_000),
    )
    scaffold.init_from(points, n_seeds=min_nodes)
    scaffold.run_until_stable(points, StabilizationConfig(max_epochs=max_epochs))
    return scaffold, points


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

    cfg = config if config is not None else CDKCalibrationConfig()
    estimates: list[float] = []
    for e in range(cfg.n_ensembles):
        scaffold, _ = _build_equilibrated_scaffold(
            d,
            k,
            n_samples=cfg.n_samples,
            target_nodes=cfg.target_nodes,
            min_nodes=cfg.min_nodes,
            max_nodes=cfg.max_nodes,
            max_epochs=cfg.max_epochs,
            ann_backend=cfg.ann_backend,
            seed=seed,
            ensemble=e,
        )
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
        "min_nodes": 4,
        "max_epochs": 25,
        "ann_backend": "auto",
    },
    "stabilization": "StabilizationConfig(max_epochs=25); other fields default",
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


# ---------------------------------------------------------------------------
# C_Q(d): interior star-radius constant (SI S3.3, OPEN_ISSUES #36)
# ---------------------------------------------------------------------------
#
# The Stage-2 prune/merge guards of SI S3.3 bound the worst-case reassignment
# distance of a node's Voronoi cell by ``C_Q(d) * sqrt(tau_local)``:
#
#   rho_prune := max_{x in V_j} min_{s != j} ||x - w_s||  <=  C_Q(d) * sqrt(tau).
#
# C_Q(d) is the "variance-cap star-radius constant in the regular interior": the
# typical value of that worst-case reassignment distance, normalized by
# sqrt(tau), for a cap-equilibrated cell away from the support boundary. It is a
# **calibrated** constant measured on the *same* uniform-d-ball ensemble as
# c_{d,k} (S2.5.5) -- the star radius rather than the k-NN radius on identical
# equilibrated scaffolds.


@dataclass(frozen=True)
class CQCalibrationConfig:
    """Reference-ensemble parameters for a single ``C_Q(d)`` calibration point.

    Shares the uniform-d-ball ensemble of :class:`CDKCalibrationConfig`. ``k`` is
    the representative neighbor count used to build the reference scaffolds (the
    S3.3 guards act on the operational scaffold, whose default is ``k = 8``).
    ``interior_fraction`` selects the "regular interior" as the inner fraction of
    mature nodes by distance from the ball center, which excludes boundary cells
    whose star radius is inflated by the missing exterior neighbors and adapts to
    dimension (in high d most nodes concentrate near the surface). All fields are
    operational calibration parameters (they set the measurement protocol).
    """

    n_samples: int = 3000
    n_ensembles: int = 4
    target_nodes: int = 120
    max_nodes: int = 400
    min_nodes: int = 4
    max_epochs: int = 25
    ann_backend: str = "auto"
    k: int = 8
    interior_fraction: float = 0.5


def measure_cq_from_scaffold(
    scaffold: object,
    points: np.ndarray,
    *,
    interior_fraction: float = 0.5,
) -> float:
    """Median interior ``rho_prune / sqrt(tau)`` of an equilibrated scaffold (SI S3.3).

    For each reference point ``x`` the two nearest nodes are queried; the nearest
    is the owner ``j`` and the second-nearest distance equals
    ``min_{s != j} ||x - w_s||`` (the reassignment distance if ``j`` is deleted).
    The per-node star radius ``rho_prune`` is the max of this over the points the
    node owns. Only mature nodes contribute; the "regular interior" keeps the
    inner ``interior_fraction`` of them by distance from the ball center. Returns
    ``NaN`` if the scaffold is too small to estimate.

    The max is taken over *sampled* points, so it is a lower-biased estimate of
    the continuous cell maximum that converges from below as sampling density
    grows; the bias is conservative for the S3.3 prune/merge guards it feeds.
    """

    nodes = list(getattr(scaffold, "nodes"))
    n = len(nodes)
    if n < 2:
        return float("nan")
    tau = float(getattr(scaffold, "tau"))
    if tau <= 0.0:
        return float("nan")
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 2:
        return float("nan")
    prune_after = int(getattr(scaffold, "prune_after", 0))
    positions = scaffold.node_positions()  # type: ignore[attr-defined]
    node_norms = np.linalg.norm(positions, axis=1)

    rho = np.zeros(n, dtype=float)  # per-node worst-case reassignment distance
    for x in points:
        idx, dists = scaffold.ann.query_knn(x, k=2)  # type: ignore[attr-defined]
        j = int(idx[0])
        d2 = float(dists[1]) if len(dists) > 1 else float(dists[0])
        if d2 > rho[j]:
            rho[j] = d2

    mature = [
        i
        for i, node in enumerate(nodes)
        if int(getattr(node, "update_count", 0)) >= prune_after and rho[i] > 0.0
    ]
    if len(mature) < 2:
        # Fall back to every node that owns at least one reference point.
        mature = [i for i in range(n) if rho[i] > 0.0]
    if len(mature) < 2:
        return float("nan")

    order = sorted(mature, key=lambda i: float(node_norms[i]))
    n_keep = max(2, int(np.ceil(float(interior_fraction) * len(order))))
    interior = order[:n_keep]

    sqrt_tau = float(np.sqrt(tau))
    ratios = [float(rho[i]) / sqrt_tau for i in interior]
    return float(np.median(ratios))


def calibrate_cq(
    d: int,
    *,
    config: CQCalibrationConfig | None = None,
    seed: int = 0,
) -> float:
    """Calibrate ``C_Q(d)`` on the uniform d-ball reference ensemble (SI S3.3).

    Runs ``config.n_ensembles`` independent uniform-d-ball scaffolds to
    equilibrium (the same ensemble as :func:`calibrate_cdk`) and returns the
    median of the per-scaffold interior ``median(rho_prune / sqrt(tau))``.
    """

    cfg = config if config is not None else CQCalibrationConfig()
    estimates: list[float] = []
    for e in range(cfg.n_ensembles):
        scaffold, points = _build_equilibrated_scaffold(
            d,
            cfg.k,
            n_samples=cfg.n_samples,
            target_nodes=cfg.target_nodes,
            min_nodes=cfg.min_nodes,
            max_nodes=cfg.max_nodes,
            max_epochs=cfg.max_epochs,
            ann_backend=cfg.ann_backend,
            seed=seed,
            ensemble=e,
        )
        est = measure_cq_from_scaffold(
            scaffold, points, interior_fraction=cfg.interior_fraction
        )
        if np.isfinite(est):
            estimates.append(est)
    if not estimates:
        raise RuntimeError(f"calibration produced no finite estimate for d={d}")
    return float(np.median(estimates))


# Metadata describing how CQ_TABLE was produced (reproducibility record).
CQ_CALIBRATION_META: dict[str, object] = {
    "protocol": "SI S3.3 (ensemble shared with S2.5.5)",
    "ensemble": "uniform unit d-ball",
    "statistic": (
        "median over the interior half of mature nodes of "
        "rho_prune / sqrt(tau), median over ensembles"
    ),
    "config": {
        "n_samples": 3000,
        "n_ensembles": 4,
        "target_nodes": 120,
        "max_nodes": 400,
        "min_nodes": 4,
        "max_epochs": 25,
        "ann_backend": "auto",
        "k": 8,
        "interior_fraction": 0.5,
    },
    "stabilization": "StabilizationConfig(max_epochs=25); other fields default",
    "grid": {"d": [1, 2, 3, 4]},
    "seed": 0,
}

# Calibrated C_Q(d) values (see CQ_CALIBRATION_META). Regenerate with
# ``python -m proteus.stage1.calibration``.
CQ_TABLE: dict[int, float] = {
    1: 2.3462,
    2: 1.6627,
    3: 1.4895,
    4: 1.4390,
}


def c_q(d: int) -> float:
    """Look up the calibrated interior star-radius constant ``C_Q(d)`` (SI S3.3).

    Falls back to the cap-fixed catchment-radius anchor :func:`c_q_analytic` for
    dimensions outside the tabulated grid.
    """

    d_i = int(d)
    if d_i in CQ_TABLE:
        return CQ_TABLE[d_i]
    return c_q_analytic(d_i)


def c_q_analytic(d: int, k: int = 8) -> float:
    """Inter-node-spacing anchor for ``C_Q(d)`` (SI S3.3 fallback).

    The worst-case reassignment distance is achieved near the deleted node's
    centroid, where the nearest *surviving* node sits one inter-node spacing
    ``s`` away; so ``C_Q(d)`` tracks the equilibrium spacing, **not** the cell
    radius. Under a locally uniform density the k-NN radius and the 1-NN spacing
    are related by ``r_{k} = s (k / V_d)^{1/d}``, hence
    ``s = r_{k} (V_d / k)^{1/d}``. Dividing by ``sqrt(tau)`` and substituting the
    calibrated k-NN radius ``r_{k} = c_{d,k} sqrt(tau)`` gives the anchor
    ``C_Q(d) ~ c_{d,k} (V_d / k)^{1/d}``.

    Using the calibrated ``c_{d,k}`` (rather than its analytic form) is what makes
    this track the measured table: the equilibrium catchment variance sits below
    the cap, which a pure cell-radius argument (``sqrt((d+2)/d)``) would miss. For
    dimensions outside the ``c_{d,k}`` grid this degrades to the analytic k-NN
    spacing via :func:`c_dk`. Used only as the runtime fallback for untabulated
    dimensions.
    """

    from scipy.special import gammaln

    d_f, k_f = float(d), float(k)
    v_d = float(np.pi ** (d_f / 2.0) / np.exp(gammaln(d_f / 2.0 + 1.0)))
    return c_dk(int(d), int(k)) * (v_d / k_f) ** (1.0 / d_f)


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


def _regenerate_cq_table(
    ds: tuple[int, ...] = (1, 2, 3, 4),
    *,
    config: CQCalibrationConfig | None = None,
    seed: int = 0,
) -> dict[int, float]:
    """Recompute the ``C_Q(d)`` table (used by ``__main__``)."""

    return {d: calibrate_cq(d, config=config, seed=seed) for d in ds}


if __name__ == "__main__":  # pragma: no cover - calibration entry point
    import json

    result = _regenerate_table()
    printable = {f"{d},{k}": round(v, 4) for (d, k), v in result.items()}
    cq_result = _regenerate_cq_table()
    cq_printable = {str(d): round(v, 4) for d, v in cq_result.items()}
    print(json.dumps({"c_dk": printable, "C_Q": cq_printable}, indent=2, sort_keys=True))
