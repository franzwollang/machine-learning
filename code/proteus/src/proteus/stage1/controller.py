"""Scale-grid search controller for Stage 1 (SI S2.5.1)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from proteus.stage1.persistence import (
    PartitionSnapshot,
    PersistenceConfig,
    PersistenceResult,
    compute_persistence,
    interval_is_persistent,
    route_samples_to_labels,
)
from proteus.stage1.scaffold import Stage1Scaffold
from proteus.stage1.scale_response import cluster_response, support_trace, variance_load
from proteus.stage1.stabilization import StabilizationConfig


@dataclass(frozen=True)
class ScaleSearchConfig:
    """Configuration for the geometric-grid scale search.

    ``selector`` chooses the characteristic-scale rule (SI S2.5.1 / S2.6.2):
    ``"load_crossover"`` (default) selects the grid point at the variance-load
    ``load approx 1`` up-crossing --- the coarsest scale at which the mean
    per-node variance first reaches the cap ``tau`` (SI S2.5.1).  ``"load_band"``
    is **DEPRECATED** (OPEN_ISSUES #28): the legacy ``0.65 <= load <= 1``
    coarsest-in-band heuristic, retained only behind this flag for regression
    bisection during the M2 transition; do not use for new work --- prefer
    ``load_crossover`` (resolution) or ``persistence`` (structural timing).
    ``"persistence"`` uses the Q-partition persistence signal --- the coarsest
    ``tau`` at which a multi-cluster partition persists across adjacent grid
    points --- for structural (recursion) timing, falling back to the
    ``load_crossover`` resolution scale when no split persists.  Persistence
    requires the per-grid-point partitions, so selecting it implies
    ``record_partitions``.
    """

    grid_ratio: float = 1.0 / np.sqrt(2.0)
    tau_min: float = 1e-5
    tau_max: float = 10.0
    max_grid_points: int = 12
    k: int = 8
    min_nodes: int = 4
    n_seeds: int = 8
    ann_backend: str = "naive"
    max_nodes: Optional[int] = None
    stabilization: StabilizationConfig = field(
        default_factory=StabilizationConfig,
    )
    selector: str = "load_crossover"
    record_partitions: bool = False
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    seed: int = 42


@dataclass
class ScaleSearchResult:
    """Output of a scale-grid search."""

    tau_grid: np.ndarray
    phi_trace: np.ndarray
    v_trace: np.ndarray
    load_trace: np.ndarray
    tau_star: float
    peak_index: int
    scaffold_at_star: Stage1Scaffold
    stabilized_flags: list[bool]
    node_count_trace: np.ndarray = field(default_factory=lambda: np.empty(0))
    epochs_at_tau_star: int = 0
    partition_snapshots: Optional[list[PartitionSnapshot]] = None
    persistence_result: Optional[PersistenceResult] = None


def _build_tau_grid(config: ScaleSearchConfig) -> np.ndarray:
    """Return a geometric grid of tau values from tau_max down to tau_min."""

    log_min = np.log(config.tau_min)
    log_max = np.log(config.tau_max)
    n_points = min(
        config.max_grid_points,
        max(3, int(np.ceil((log_max - log_min) / abs(np.log(config.grid_ratio)))) + 1),
    )
    return np.exp(np.linspace(log_max, log_min, n_points))


def run_scale_search(
    data: np.ndarray,
    dim: int,
    config: ScaleSearchConfig | None = None,
) -> ScaleSearchResult:
    """Sweep tau from coarse to fine on a single warm-started scaffold.

    The scaffold is seeded once at the coarsest tau, then the cap is
    lowered at each grid step and the scaffold is re-converged.  This
    lets the response trace ``Phi_C(tau)`` reflect the structural
    information the data exhibits at each scale, because the scaffold
    adapts incrementally rather than being rebuilt from scratch.
    """

    config = config if config is not None else ScaleSearchConfig()
    data_arr = np.asarray(data, dtype=float)
    tau_grid = _build_tau_grid(config)

    phi_trace = np.full(len(tau_grid), np.nan)
    v_trace = np.full(len(tau_grid), np.nan)
    stabilized = [False] * len(tau_grid)

    rng = np.random.default_rng(config.seed)
    if config.max_nodes is not None:
        max_nodes = min(int(config.max_nodes), data_arr.shape[0] // 2)
    else:
        max_nodes = min(max(config.min_nodes * 16, 64), data_arr.shape[0] // 2)

    scaffold = Stage1Scaffold(
        dim=dim,
        tau=float(tau_grid[0]),
        k=config.k,
        min_nodes=config.min_nodes,
        max_nodes=max_nodes,
        ann_backend=config.ann_backend,
        rng=rng,
    )
    n_seeds = min(config.n_seeds, data_arr.shape[0])
    scaffold.init_from(data_arr, n_seeds=n_seeds)

    best_scaffold: Optional[Stage1Scaffold] = None

    load_trace = np.full(len(tau_grid), np.nan)
    node_counts = np.zeros(len(tau_grid), dtype=float)

    record_partitions = config.record_partitions or config.selector == "persistence"
    snapshots: list[PartitionSnapshot] | None = [] if record_partitions else None

    last_history: dict[str, list[float]] | None = None
    for idx, tau in enumerate(tau_grid):
        scaffold.tau = float(tau)
        scaffold.tau_local = np.full(len(scaffold.nodes), float(tau), dtype=float)
        scaffold.delta_min_value = scaffold.kappa * (1.0 - scaffold.grid_ratio) * np.sqrt(float(tau))

        last_history = scaffold.run_until_stable(data_arr, config.stabilization)

        phi = cluster_response(scaffold, float(tau), dim)
        v = support_trace(scaffold, float(tau), dim)
        load = variance_load(scaffold, float(tau))

        phi_trace[idx] = phi
        v_trace[idx] = v
        load_trace[idx] = load
        node_counts[idx] = float(len(scaffold.nodes))
        cv_vals = last_history["cv"]
        stabilized[idx] = len(cv_vals) < config.stabilization.max_epochs

        if snapshots is not None:
            snapshots.append(
                _snapshot_partition(scaffold, data_arr, idx, float(tau), stabilized[idx]),
            )

    persistence_result: PersistenceResult | None = None
    if snapshots is not None:
        persistence_result = compute_persistence(snapshots, config.persistence)
        if (
            config.persistence.coarse_anchored
            and config.persistence.cold_start_recheck
            and persistence_result.tau_star_index is not None
        ):
            persistence_result = _cold_start_recheck(
                persistence_result, data_arr, dim, config, tau_grid, max_nodes,
            )

    if config.selector == "load_band":
        peak_idx = _select_load_band(
            load_trace, node_counts, tau_grid, stabilized,
        )
    else:
        peak_idx = _select_load_crossover(load_trace, stabilized)
    if (
        config.selector == "persistence"
        and persistence_result is not None
        and persistence_result.tau_star_index is not None
    ):
        peak_idx = int(persistence_result.tau_star_index)

    tau_star = float(tau_grid[peak_idx])
    epochs_at_star = 0
    if last_history is not None and abs(float(scaffold.tau) - tau_star) <= 1e-12:
        epochs_at_star = len(last_history["cv"])

    if abs(float(scaffold.tau) - tau_star) > 1e-12:
        scaffold.tau = tau_star
        scaffold.tau_local = np.full(len(scaffold.nodes), tau_star, dtype=float)
        scaffold.delta_min_value = (
            scaffold.kappa * (1.0 - scaffold.grid_ratio) * np.sqrt(tau_star)
        )
        last_history = scaffold.run_until_stable(data_arr, config.stabilization)
        epochs_at_star = len(last_history["cv"])

    if best_scaffold is None:
        best_scaffold = scaffold

    return ScaleSearchResult(
        tau_grid=tau_grid,
        phi_trace=phi_trace,
        v_trace=v_trace,
        load_trace=load_trace,
        tau_star=tau_star,
        peak_index=peak_idx,
        scaffold_at_star=best_scaffold,
        stabilized_flags=stabilized,
        node_count_trace=node_counts,
        epochs_at_tau_star=epochs_at_star,
        partition_snapshots=snapshots,
        persistence_result=persistence_result,
    )


def _snapshot_partition(
    scaffold: Stage1Scaffold,
    data: np.ndarray,
    grid_index: int,
    tau: float,
    stabilized: bool,
) -> PartitionSnapshot:
    """Cluster the current scaffold and record its sample-space partition.

    Used only when persistence tracking is enabled.  A scaffold too small to
    cluster is treated as a single-cluster (null) partition.
    """

    if len(scaffold.nodes) < 2:
        return PartitionSnapshot(
            grid_index=grid_index,
            tau=tau,
            labels=np.zeros(data.shape[0], dtype=int),
            n_clusters=1,
            partition_q_score=0.0,
            stabilized=stabilized,
        )

    # Local import avoids a module-load cycle (clustering has no controller dep,
    # but recursion imports both, so keep the edge one-directional at import time).
    from proteus.stage1.clustering import run_clustering

    cluster_result = run_clustering(scaffold)
    sample_labels = route_samples_to_labels(scaffold, data, cluster_result.labels)
    return PartitionSnapshot(
        grid_index=grid_index,
        tau=tau,
        labels=sample_labels,
        n_clusters=int(cluster_result.n_clusters),
        partition_q_score=float(cluster_result.partition_q_score),
        stabilized=stabilized,
    )


def _cold_start_snapshot(
    data: np.ndarray,
    dim: int,
    tau: float,
    grid_index: int,
    config: ScaleSearchConfig,
    max_nodes: int,
    seed: int,
) -> PartitionSnapshot:
    """Fit a fresh scaffold at ``tau`` (no warm carry-over) and snapshot its partition.

    Unlike the coarse-to-fine sweep in :func:`run_scale_search`, this seeds a new
    :class:`Stage1Scaffold` directly at ``tau`` on an independent RNG stream and
    converges it from scratch.  It is the primitive of the cold-start
    path-independence recheck (SI S2.6.2, ``PersistenceConfig.cold_start_recheck``).
    """

    rng = np.random.default_rng(seed)
    scaffold = Stage1Scaffold(
        dim=dim,
        tau=float(tau),
        k=config.k,
        min_nodes=config.min_nodes,
        max_nodes=max_nodes,
        ann_backend=config.ann_backend,
        rng=rng,
    )
    n_seeds = min(config.n_seeds, data.shape[0])
    scaffold.init_from(data, n_seeds=n_seeds)
    history = scaffold.run_until_stable(data, config.stabilization)
    stabilized = len(history["cv"]) < config.stabilization.max_epochs
    return _snapshot_partition(scaffold, data, grid_index, float(tau), stabilized)


def _cold_start_recheck(
    persistence_result: PersistenceResult,
    data: np.ndarray,
    dim: int,
    config: ScaleSearchConfig,
    tau_grid: np.ndarray,
    max_nodes: int,
) -> PersistenceResult:
    """Path-independence recheck of a coarse-anchored persistence candidate.

    Re-fits the grid points of the candidate interval
    ``[i0 .. i0 + min_persistence - 1]`` from cold-started scaffolds (independent
    RNG streams per grid point) and keeps the candidate only if that interval
    still persists.  On rejection returns a copy with ``tau_star_index`` /
    ``tau_star`` cleared and ``cold_start_rejected=True``.

    This is gated by ``PersistenceConfig.cold_start_recheck``, which is **off by
    default and refuted as an acceptance gate** (it over-rejects genuine
    multi-level features because cold single-``tau`` fits have high
    resolution-level variance); see that flag's docstring, SI S2.6.2, and
    OPEN_ISSUES #27.
    """

    i0 = persistence_result.tau_star_index
    if i0 is None:
        return persistence_result
    length = config.persistence.min_persistence
    interval = list(range(i0, min(i0 + length, len(tau_grid))))
    if len(interval) < length:
        # Too few fine-side grid points to re-verify path-independence: reject
        # conservatively rather than accept an unverifiable candidate.
        return replace(
            persistence_result,
            tau_star_index=None,
            tau_star=None,
            cold_start_rejected=True,
        )

    cold_snaps: list[PartitionSnapshot] = []
    for j in interval:
        # Distinct, deterministic stream per grid point, disjoint from the warm
        # sweep's ``config.seed`` so the refit shares no random trajectory.
        cold_seed = int(config.seed) + 9973 * (int(j) + 1)
        cold_snaps.append(
            _cold_start_snapshot(
                data, dim, float(tau_grid[j]), int(j), config, max_nodes, cold_seed,
            ),
        )

    if interval_is_persistent(cold_snaps, config.persistence):
        return persistence_result
    return replace(
        persistence_result,
        tau_star_index=None,
        tau_star=None,
        cold_start_rejected=True,
    )


def _select_load_crossover(
    load_trace: np.ndarray,
    stabilized: list[bool],
) -> int:
    """Select the characteristic scale at the variance-load ``load≈1`` up-crossing.

    The grid is in descending ``tau`` order (coarse to fine), so the mean
    per-node variance-to-cap ratio ``load = mean(sigma^2)/tau`` increases along
    the grid: at coarse scales the scaffold under-resolves the support and sits
    below the cap (``load < 1``); at fine scales the cap binds and ``load > 1``.
    The characteristic scale is the coarsest point at which the mean variance
    first reaches the cap --- the ``load = 1`` up-crossing (SI S2.5.1).

    Rule (among stabilized grid points only):

    * take the **coarsest** adjacent pair straddling ``load = 1`` (``load[i] <= 1
      < load[i+1]``) and return whichever endpoint is nearer to ``1``;
    * if the load never reaches ``1`` (over-coarse budget), return the finest
      stabilized point (largest load, most resolved);
    * if the load is always ``>= 1`` (over-fine grid), return the coarsest
      stabilized point.

    Unlike the legacy load-band heuristic this carries no ``band_lo`` constant
    and no one-step-coarser patch: the up-crossing is scale-invariant and
    self-locating (OPEN_ISSUES #28).
    """

    n = len(load_trace)
    if n == 0:
        return 0

    eligible = np.array(stabilized, dtype=bool)
    finite = np.where(np.isfinite(load_trace), load_trace, np.inf)
    idx = [i for i in range(n) if eligible[i] and np.isfinite(load_trace[i])]
    if not idx:
        return int(np.argmin(np.abs(finite - 1.0)))

    # Coarsest adjacent eligible pair straddling load = 1 (ascending load).
    for a, b in zip(idx[:-1], idx[1:]):
        if finite[a] <= 1.0 < finite[b]:
            return a if abs(finite[a] - 1.0) <= abs(finite[b] - 1.0) else b

    # No up-crossing: either always below the cap or always above it.
    loads = np.array([finite[i] for i in idx], dtype=float)
    if np.all(loads < 1.0):
        return int(idx[int(np.argmax(loads))])  # finest / most resolved
    return int(idx[0])  # coarsest stabilized (load already >= 1 everywhere)


def _select_load_band(
    load_trace: np.ndarray,
    node_counts: np.ndarray,
    tau_grid: np.ndarray,
    stabilized: list[bool],
) -> int:
    """DEPRECATED (OPEN_ISSUES #28): legacy load-band characteristic-scale rule.

    Retained only behind ``ScaleSearchConfig.selector="load_band"`` for
    regression bisection.  New code should use ``_select_load_crossover``
    (default resolution selector) or the persistence path.  Scheduled for
    deletion once ``load_crossover`` dominates every scenario/recursion
    regression (same issue: also drops dormant ``_legacy_slope_selector`` /
    ``_detect_peak``).

    The grid is in descending τ order (coarse to fine).  Among stabilized
    grid points with ``0.65 <= load <= 1.0``, take the **coarsest**
    (smallest index).  If exactly one index qualifies (the finest margin),
    prefer **one step coarser** when that neighbor is still stabilized with
    ``load <= 1`` — unstable interior grid points can otherwise leave a lone
    fine-scale cell as the only band member.  If the band is empty, fall back
    to the finest stabilized point with ``load <= 1.0`` and maximal load.
    """

    n = len(tau_grid)
    if n == 0:
        return 0

    eligible = np.array(stabilized, dtype=bool)
    if not eligible.any():
        return int(np.argmin(np.abs(load_trace - 1.0)))

    finite_load = np.where(np.isfinite(load_trace), load_trace, np.inf)

    # Prefer a slightly **coarser** τ among stabilized points with load in
    # ``[band_lo, 1]``, then take the **smallest index** (coarsest τ on this
    # descending grid).  ``band_lo`` is below 0.7 so the last sub-threshold
    # grid step (often ~0.65) is still eligible; using only ``>= 0.7`` leaves a
    # single candidate at the finest end and skews τ* below geometric scales.
    band_lo = 0.65
    band_candidates = [
        i
        for i in range(n)
        if eligible[i] and band_lo <= finite_load[i] <= 1.0
    ]
    if band_candidates:
        c = int(min(band_candidates))
        # If only the finest grid point qualifies, also allow one step coarser
        # when it is still variance-feasible (fixes unstable middle grid rows
        # that otherwise pin τ* to the finest cell with ratio << 0.5).
        if len(band_candidates) == 1 and c > 0:
            p = c - 1
            if eligible[p] and np.isfinite(finite_load[p]) and finite_load[p] <= 1.0:
                return p
        return c

    best_idx = -1
    best_load = -np.inf
    for i in range(n):
        if not eligible[i]:
            continue
        if finite_load[i] <= 1.0 and finite_load[i] > best_load:
            best_load = finite_load[i]
            best_idx = i

    if best_idx >= 0:
        return best_idx

    return int(np.argmin(np.abs(finite_load - 1.0)))


def _legacy_slope_selector(
    load_trace: np.ndarray,
    node_counts: np.ndarray,
    tau_grid: np.ndarray,
    stabilized: list[bool],
) -> int:
    """DEPRECATED / dormant (OPEN_ISSUES #28): slope-based selector.

    No callers remain; kept only until the load-band deletion pass removes it
    with ``_select_load_band`` and ``_detect_peak``.
    """

    n = len(tau_grid)
    if n < 3:
        return int(np.argmin(np.abs(load_trace - 0.5)))

    log_tau = np.log(tau_grid)
    log_n = np.log(np.maximum(node_counts, 1))
    slopes = np.abs(np.diff(log_n) / np.diff(log_tau))

    eligible_mask = np.array(stabilized, dtype=bool)
    mid_eligible = np.array([
        eligible_mask[i] and eligible_mask[i + 1]
        for i in range(n - 1)
    ])

    if not mid_eligible.any():
        return int(np.argmin(np.abs(load_trace - 0.5)))

    best_midpoint = -1
    best_cost = np.inf
    for i in range(n - 1):
        if not mid_eligible[i]:
            continue
        cost = abs(slopes[i] - 0.5) + abs(load_trace[i] - 0.5) + abs(load_trace[i + 1] - 0.5)
        if cost < best_cost:
            best_cost = cost
            best_midpoint = i

    if best_midpoint < 0:
        return int(np.argmin(np.abs(load_trace - 0.5)))
    if load_trace[best_midpoint] < load_trace[best_midpoint + 1]:
        return best_midpoint + 1
    return best_midpoint


def _detect_peak(phi: np.ndarray) -> int:
    """DEPRECATED / dormant (OPEN_ISSUES #28): peak finder for response traces.

    No callers remain in the active scale-search path; deletion is bundled
    with the legacy load-band selector removal.

    Uses centered second differences to find local maxima among eligible
    (non -inf) grid points.  Falls back to argmax if no interior peak
    is found.
    """

    n = len(phi)
    if n < 3:
        return int(np.argmax(phi))

    candidates = []
    for i in range(1, n - 1):
        if phi[i] > phi[i - 1] and phi[i] > phi[i + 1]:
            candidates.append(i)

    if candidates:
        return max(candidates, key=lambda i: phi[i])
    return int(np.argmax(phi))
