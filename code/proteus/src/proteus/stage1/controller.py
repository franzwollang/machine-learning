"""Scale-grid search controller for Stage 1 (SI S2.5.1)."""

from __future__ import annotations

import warnings
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
    per-node variance first reaches the cap ``tau`` (SI S2.5.1).
    ``"persistence"`` uses the Q-partition persistence signal --- the coarsest
    ``tau`` at which a multi-cluster partition persists across adjacent grid
    points --- for structural (recursion) timing, falling back to the
    ``load_crossover`` resolution scale when no split persists.  Persistence
    requires the per-grid-point partitions, so selecting it implies
    ``record_partitions``.  When a persistent split exists and
    ``persistence.resolve_within_interval`` is ``"load_crossover"`` or an
    experimental probe (``"mid_interval"``, ``"mid_interval_load_screened"``,
    ``"two_thirds_interval"``, ``"two_thirds_load_screened"``,
    ``"three_quarter_interval"``, ``"three_quarter_load_screened"``,
    ``"load_weighted_interval"``, ``"fine_end_of_block"``), persistence still
    decides accept/reject but ``tau*`` is re-picked inside that persistent
    subgrid (default ``"none"`` preserves coarse-end ``tau*``).

    ``halve_grid_steps`` (default ``False``) is an **experimental** denser
    geometric grid: effective ratio becomes ``sqrt(grid_ratio)`` (half the
    log-step) with up to ``2 * max_grid_points`` slots, so persistent blocks
    contain more candidate ``tau`` values under mid/two-thirds/three-quarter
    probes (OPEN_ISSUES #28).  Not an acceptance-path default.

    The legacy ``load_band`` selector (OPEN_ISSUES #28) is **deprecated**:
    passing it emits :class:`DeprecationWarning` and redirects to
    ``load_crossover``.  Unknown selector values raise ``ValueError``.
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
    # Experimental denser geometric grid (half log-step); default off (#28).
    halve_grid_steps: bool = False


# Deprecated ScaleSearchConfig.selector aliases (OPEN_ISSUES #28). Kept only so
# leftover callers fail soft with a warning instead of a hard break; do not use
# on the acceptance path. Map value is the replacement selector.
_DEPRECATED_SELECTORS: dict[str, str] = {
    "load_band": "load_crossover",
}


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


def _effective_grid_ratio(config: ScaleSearchConfig) -> float:
    """Geometric step ratio; ``sqrt(grid_ratio)`` when ``halve_grid_steps``."""

    ratio = float(config.grid_ratio)
    if config.halve_grid_steps:
        # Half the |log| step ⇒ denser within-block candidates (A6-T46).
        return float(np.sqrt(ratio))
    return ratio


def _effective_max_grid_points(config: ScaleSearchConfig) -> int:
    """Cap on grid length; doubled when ``halve_grid_steps`` is on."""

    cap = int(config.max_grid_points)
    if config.halve_grid_steps:
        return max(cap, 2 * cap)
    return cap


def _build_tau_grid(config: ScaleSearchConfig) -> np.ndarray:
    """Return a geometric grid of tau values from tau_max down to tau_min."""

    log_min = np.log(config.tau_min)
    log_max = np.log(config.tau_max)
    ratio = _effective_grid_ratio(config)
    n_points = min(
        _effective_max_grid_points(config),
        max(3, int(np.ceil((log_max - log_min) / abs(np.log(ratio)))) + 1),
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

    selector = _normalize_selector(config.selector)
    if selector == "load_crossover":
        peak_idx = _select_load_crossover(load_trace, stabilized)
    elif selector == "persistence":
        peak_idx = _select_load_crossover(load_trace, stabilized)
        if (
            persistence_result is not None
            and persistence_result.tau_star_index is not None
        ):
            peak_idx = _resolve_persistence_tau_index(
                persistence_result, load_trace, stabilized, config.persistence,
            )
    else:
        raise ValueError(
            f"Unknown ScaleSearchConfig.selector={config.selector!r}; "
            "expected 'load_crossover' or 'persistence' "
            "(legacy 'load_band' is a deprecated alias for 'load_crossover'; "
            "see OPEN_ISSUES #28)."
        )

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


def _normalize_selector(selector: str) -> str:
    """Resolve deprecated selector aliases; warn once per call site path."""

    replacement = _DEPRECATED_SELECTORS.get(selector)
    if replacement is None:
        return selector
    warnings.warn(
        f"ScaleSearchConfig.selector={selector!r} is deprecated "
        f"(OPEN_ISSUES #28); use {replacement!r} instead. "
        f"Redirecting to {replacement!r}.",
        DeprecationWarning,
        stacklevel=2,
    )
    return replacement


# Operational default (S14.3 / OPEN_ISSUES #28): variance-load floor below
# which an experimental within-interval landing is treated as ≪ 1 and rejected
# (fall back to coarse-end).  Shared by mid/two-thirds/three-quarter
# load-screened probes.  Not an acceptance-path constant.
_WITHIN_INTERVAL_LOAD_SCREEN_MIN: float = 0.5
# Backward-compatible alias (A6-T38 naming).
_THREE_QUARTER_LOAD_SCREEN_MIN: float = _WITHIN_INTERVAL_LOAD_SCREEN_MIN


def _three_quarter_index(i_lo: int, i_hi: int) -> int:
    """Three-quarters of the way from coarse ``i_lo`` toward fine ``i_hi``."""

    return i_lo + (3 * (i_hi - i_lo)) // 4


def _two_thirds_index(i_lo: int, i_hi: int) -> int:
    """Two-thirds of the way from coarse ``i_lo`` toward fine ``i_hi``."""

    return i_lo + (2 * (i_hi - i_lo)) // 3


def _mid_interval_index(i_lo: int, i_hi: int) -> int:
    """Integer midpoint of the accepted persistent block ``[i_lo, i_hi]``."""

    return (i_lo + i_hi) // 2


def _apply_load_screen(
    candidate: int,
    i_lo: int,
    load_trace: np.ndarray,
    *,
    screen_min: float = _WITHIN_INTERVAL_LOAD_SCREEN_MIN,
) -> int:
    """Keep ``candidate`` when ``load[candidate]`` is not ≪ 1; else ``i_lo``."""

    load_at = float(load_trace[candidate])
    if not np.isfinite(load_at) or load_at < screen_min:
        return i_lo
    return candidate


def _load_weighted_index(
    i_lo: int,
    i_hi: int,
    load_trace: np.ndarray,
    *,
    screen_min: float = _WITHIN_INTERVAL_LOAD_SCREEN_MIN,
) -> int:
    """Pick the persistent-block index whose load is closest to 1.

    Experimental probe (OPEN_ISSUES #28 / A6-T47): among indices in
    ``[i_lo, i_hi]`` with finite ``load >= screen_min``, maximize
    ``-abs(log(load))`` (unit load preferred).  If no index clears the
    ≪1 screen, fall back to coarse-end ``i_lo``.  Distinct from
    :func:`_select_load_crossover` (straddle-pair rule) and from fixed
    fractional landings.
    """

    best_idx = i_lo
    best_score = -np.inf
    found = False
    for idx in range(i_lo, i_hi + 1):
        load_at = float(load_trace[idx])
        if not np.isfinite(load_at) or load_at < screen_min:
            continue
        # Prefer load≈1; clamp away from 0 for log safety (screen already
        # enforces load >= screen_min > 0).
        score = -abs(float(np.log(load_at)))
        if score > best_score:
            best_score = score
            best_idx = idx
            found = True
    return best_idx if found else i_lo


def _resolve_persistence_tau_index(
    persistence_result: PersistenceResult,
    load_trace: np.ndarray,
    stabilized: list[bool],
    persistence: PersistenceConfig,
) -> int:
    """Map an accepted persistent split to a characteristic-scale grid index.

    Default (``resolve_within_interval="none"``): return the coarse-end arbiter
    index from :func:`compute_persistence`.  With
    ``resolve_within_interval="load_crossover"``, keep that interval as the
    accept/reject gate but re-pick ``tau*`` via :func:`_select_load_crossover`
    on the persistent subgrid only (OPEN_ISSUES #28 hybrid option).  With
    experimental ``"mid_interval"``, land at the integer midpoint of the
    accepted block; with experimental ``"mid_interval_load_screened"``, take
    that midpoint only when ``load[idx]`` is not ≪ 1 (else fall back to
    ``i_lo``); with experimental ``"two_thirds_interval"``, land two-thirds
    of the way from ``i_lo`` toward ``i_hi`` (between mid and three-quarter);
    with experimental ``"two_thirds_load_screened"``, take that two-thirds
    landing only when ``load[idx]`` is not ≪ 1 (else fall back to ``i_lo``);
    with experimental ``"three_quarter_interval"``, land three-quarters of
    the way from ``i_lo`` toward ``i_hi``; with experimental
    ``"three_quarter_load_screened"``, take that three-quarter landing only
    when ``load[idx]`` is not ≪ 1 (else fall back to ``i_lo``); with
    experimental ``"load_weighted_interval"``, pick the block index whose
    variance load is closest to 1 among those not ≪ 1 (else ``i_lo``); with
    experimental ``"fine_end_of_block"``, land at ``i_hi`` (probe only;
    default stays ``"none"``).
    """

    i_lo = int(persistence_result.tau_star_index)  # type: ignore[arg-type]
    mode = persistence.resolve_within_interval
    if mode == "none":
        return i_lo

    run_len = int(persistence_result.run_lengths[i_lo])
    if run_len < 1:
        return i_lo
    i_hi = min(i_lo + run_len - 1, len(load_trace) - 1)

    if mode == "mid_interval":
        # Integer midpoint of [i_lo, i_hi]; experimental coarse-vs-mid probe.
        return _mid_interval_index(i_lo, i_hi)

    if mode == "mid_interval_load_screened":
        # Same landing as mid_interval, but reject when load ≪ 1.
        return _apply_load_screen(
            _mid_interval_index(i_lo, i_hi), i_lo, load_trace
        )

    if mode == "two_thirds_interval":
        # Two-thirds from coarse toward fine; between mid and three-quarter.
        return _two_thirds_index(i_lo, i_hi)

    if mode == "two_thirds_load_screened":
        # Same landing as two_thirds_interval, but reject when load ≪ 1.
        return _apply_load_screen(
            _two_thirds_index(i_lo, i_hi), i_lo, load_trace
        )

    if mode == "three_quarter_interval":
        # Three-quarters from coarse toward fine; between mid and fine-end.
        return _three_quarter_index(i_lo, i_hi)

    if mode == "three_quarter_load_screened":
        # Same landing as three_quarter_interval, but reject when load ≪ 1.
        return _apply_load_screen(
            _three_quarter_index(i_lo, i_hi), i_lo, load_trace
        )

    if mode == "load_weighted_interval":
        # Closest-to-unit load in the block; reject ≪1 indices (A6-T47).
        return _load_weighted_index(i_lo, i_hi, load_trace)

    if mode == "fine_end_of_block":
        # Finest index of the accepted persistent block; experimental probe.
        return i_hi

    if mode != "load_crossover":
        raise ValueError(
            f"Unknown PersistenceConfig.resolve_within_interval={mode!r}; "
            "expected 'none', 'load_crossover', 'mid_interval', "
            "'mid_interval_load_screened', 'two_thirds_interval', "
            "'two_thirds_load_screened', 'three_quarter_interval', "
            "'three_quarter_load_screened', 'load_weighted_interval', "
            "or 'fine_end_of_block'."
        )

    sub_load = np.asarray(load_trace[i_lo : i_hi + 1], dtype=float)
    sub_stab = list(stabilized[i_lo : i_hi + 1])
    return i_lo + _select_load_crossover(sub_load, sub_stab)


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
