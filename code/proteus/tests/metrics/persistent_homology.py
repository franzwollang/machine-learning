"""Persistent homology metrics for Proteus evaluation (SI S14.2).

Canonical evaluation tool for Stage-2 topology recovery (#41, #25 residual):
Vietoris--Rips persistent homology on **node positions** (dense pairwise), not
the sparse lifted-graph flag complex (band holes are essential there; see #25).

Uses gudhi. Falls back to ImportError if gudhi is not installed — scenario
assertions that depend on this module stay ``@awaiting`` until a chosen
filtration/reading produces green evidence (do not flip tests by weakening).

Filtration / reading options (OPEN_ISSUES #41):
  1. ``fixed_threshold`` — Betti count at a single scale ``r = mult * sigma_star``
     (SI S14.2 default ``mult = 1.5``). Simple, but the true loop may be unborn
     or already filled at that cutoff on tissue-polluted scaffolds.
  2. ``lifetime`` — count ``H_k`` bars whose lifetime exceeds
     ``lifetime_frac * sigma_star``, plus essential (infinite-death) bars.
     More robust to short spurious loops from tissue nodes; the fraction is an
     **operational** proposal-path default until logged in SI S14.2 / S14.3.
  3. Per-region assembly — run (1) or (2) on each accepted cluster/region's
     node positions separately (nested spheres / linked tori; tissue pollution
     also pushes toward per-region rather than whole-scaffold PH).

Diagnostic helper ``compare_readings`` returns both (1) and (2) for the same
cloud so evidence-gathering tests can contrast them without choosing a default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import numpy as np

# SI S14.2: filtration up to 1.5 * sigma_star, sigma_star = sqrt(tau_star).
FILTRATION_MULTIPLIER: float = 1.5

# Proposed operational lifetime floor as a fraction of sigma_star (#41 item 1).
# Proposal-path only — not an acceptance-path constant; logged for calibration.
DEFAULT_LIFETIME_FRAC: float = 0.5

ReadingMode = Literal["fixed_threshold", "lifetime"]


def sigma_star_from_tau(tau_star: float) -> float:
    """Active scale ``sigma_star = sqrt(tau_star)`` (SI S14.2)."""
    return float(np.sqrt(tau_star))


def filtration_radius(
    sigma_star: float,
    *,
    multiplier: float = FILTRATION_MULTIPLIER,
) -> float:
    """Vietoris--Rips edge-length cutoff ``multiplier * sigma_star``."""
    return float(multiplier) * float(sigma_star)


def _require_gudhi():
    try:
        import gudhi
    except ImportError as e:
        raise ImportError(
            "gudhi is required for persistent homology metrics. "
            "Install with: pip install gudhi"
        ) from e
    return gudhi


def compute_persistence_diagrams(
    points: np.ndarray,
    max_dim: int = 2,
    max_edge_length: float = np.inf,
) -> list[np.ndarray]:
    """Compute persistence diagrams up to ``max_dim``.

    Returns a list of (n_features, 2) arrays, one per homology dimension.
    """
    gudhi = _require_gudhi()

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return [np.empty((0, 2)) for _ in range(max_dim + 1)]

    rips = gudhi.RipsComplex(points=pts, max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=max_dim + 1)
    st.compute_persistence()

    diagrams: list[np.ndarray] = []
    for dim in range(max_dim + 1):
        pairs = st.persistence_intervals_in_dimension(dim)
        if len(pairs) == 0:
            diagrams.append(np.empty((0, 2)))
        else:
            diagrams.append(np.array(pairs, dtype=float))
    return diagrams


def bottleneck_distance(
    dgm1: np.ndarray, dgm2: np.ndarray,
) -> float:
    """Bottleneck distance between two persistence diagrams."""
    try:
        import gudhi.bottleneck
    except ImportError as e:
        raise ImportError("gudhi is required for bottleneck distance.") from e

    if dgm1.size == 0 and dgm2.size == 0:
        return 0.0
    if dgm1.size == 0:
        dgm1 = np.empty((0, 2))
    if dgm2.size == 0:
        dgm2 = np.empty((0, 2))
    return float(gudhi.bottleneck.bottleneck_distance(dgm1, dgm2))


def wasserstein_distance(
    dgm1: np.ndarray,
    dgm2: np.ndarray,
    order: float = 2.0,
) -> float:
    """Wasserstein distance between two persistence diagrams."""
    try:
        import gudhi.wasserstein
    except ImportError as e:
        raise ImportError("gudhi is required for Wasserstein distance.") from e

    if dgm1.size == 0 and dgm2.size == 0:
        return 0.0
    if dgm1.size == 0:
        dgm1 = np.empty((0, 2))
    if dgm2.size == 0:
        dgm2 = np.empty((0, 2))
    return float(gudhi.wasserstein.wasserstein_distance(dgm1, dgm2, order=order))


def betti_numbers(
    points: np.ndarray,
    threshold: float,
    max_dim: int = 2,
) -> tuple[int, ...]:
    """Compute Betti numbers at a fixed filtration threshold (reading mode 1)."""
    diagrams = compute_persistence_diagrams(
        points, max_dim=max_dim, max_edge_length=threshold * 1.5,
    )
    result: list[int] = []
    for dim in range(max_dim + 1):
        dgm = diagrams[dim]
        if dgm.size == 0:
            result.append(0)
            continue
        alive = (dgm[:, 0] <= threshold) & (dgm[:, 1] > threshold)
        result.append(int(alive.sum()))
    return tuple(result)


def lifetime_betti_numbers(
    points: np.ndarray,
    sigma_star: float,
    *,
    max_dim: int = 2,
    filtration_mult: float = FILTRATION_MULTIPLIER,
    lifetime_frac: float = DEFAULT_LIFETIME_FRAC,
) -> tuple[int, ...]:
    """Betti counts via persistence-lifetime reading (reading mode 2).

    Computes VR persistence with ``max_edge_length = filtration_mult * sigma_star``
    and counts bars in each dimension whose lifetime ``death - birth`` exceeds
    ``lifetime_frac * sigma_star``. Essential features (non-finite death) always
    count. Proposal-path operational default for ``lifetime_frac`` — see #41.
    """
    r_max = filtration_radius(sigma_star, multiplier=filtration_mult)
    min_life = float(lifetime_frac) * float(sigma_star)
    diagrams = compute_persistence_diagrams(
        points, max_dim=max_dim, max_edge_length=r_max,
    )
    result: list[int] = []
    for dim in range(max_dim + 1):
        dgm = diagrams[dim]
        if dgm.size == 0:
            result.append(0)
            continue
        birth = dgm[:, 0]
        death = dgm[:, 1]
        essential = ~np.isfinite(death)
        life = np.where(essential, np.inf, death - birth)
        result.append(int(np.sum(essential | (life > min_life))))
    return tuple(result)


@dataclass(frozen=True)
class RegionTopologyReport:
    """Per-region PH summary for the topology-recovery harness."""

    region_id: int
    n_points: int
    betti: tuple[int, ...]
    reading: ReadingMode
    sigma_star: float
    filtration_radius: float


def region_betti_numbers(
    points: np.ndarray,
    sigma_star: float,
    *,
    reading: ReadingMode = "lifetime",
    max_dim: int = 2,
    filtration_mult: float = FILTRATION_MULTIPLIER,
    lifetime_frac: float = DEFAULT_LIFETIME_FRAC,
) -> tuple[int, ...]:
    """Recover Betti numbers for one region's node positions.

    Prefer ``reading="lifetime"`` on tissue-polluted scaffolds; ``fixed_threshold``
    matches the literal SI S14.2 single-cutoff statement.
    """
    r = filtration_radius(sigma_star, multiplier=filtration_mult)
    if reading == "fixed_threshold":
        return betti_numbers(points, threshold=r, max_dim=max_dim)
    if reading == "lifetime":
        return lifetime_betti_numbers(
            points,
            sigma_star,
            max_dim=max_dim,
            filtration_mult=filtration_mult,
            lifetime_frac=lifetime_frac,
        )
    raise ValueError(f"unknown reading mode: {reading!r}")


def per_region_topology(
    region_points: Sequence[np.ndarray],
    sigma_star: float | Sequence[float],
    *,
    reading: ReadingMode = "lifetime",
    max_dim: int = 2,
    filtration_mult: float = FILTRATION_MULTIPLIER,
    lifetime_frac: float = DEFAULT_LIFETIME_FRAC,
) -> list[RegionTopologyReport]:
    """Scaffold harness: one VR-PH summary per recovered region (#41 item 3).

    ``sigma_star`` may be a scalar (shared active scale) or one value per region.
    Empty regions yield zero Betti tuples without calling gudhi.
    """
    n = len(region_points)
    if np.isscalar(sigma_star):
        sigmas = [float(sigma_star)] * n  # type: ignore[arg-type]
    else:
        sigmas = [float(s) for s in sigma_star]  # type: ignore[arg-type]
        if len(sigmas) != n:
            raise ValueError(
                f"sigma_star length {len(sigmas)} != n_regions {n}"
            )

    reports: list[RegionTopologyReport] = []
    for i, (pts, sig) in enumerate(zip(region_points, sigmas)):
        arr = np.asarray(pts, dtype=float)
        r = filtration_radius(sig, multiplier=filtration_mult)
        if arr.size == 0 or arr.ndim != 2 or arr.shape[0] == 0:
            betti = tuple(0 for _ in range(max_dim + 1))
        else:
            betti = region_betti_numbers(
                arr,
                sig,
                reading=reading,
                max_dim=max_dim,
                filtration_mult=filtration_mult,
                lifetime_frac=lifetime_frac,
            )
        reports.append(
            RegionTopologyReport(
                region_id=i,
                n_points=int(arr.shape[0]) if arr.ndim == 2 else 0,
                betti=betti,
                reading=reading,
                sigma_star=sig,
                filtration_radius=r,
            )
        )
    return reports


def extract_region_node_positions(
    all_positions: np.ndarray,
    region_labels: np.ndarray,
    *,
    include_labels: Optional[Iterable[int]] = None,
) -> list[np.ndarray]:
    """Split scaffold node positions into per-region point clouds.

    ``region_labels[i]`` is the accepted cluster/region id of node ``i``.
    Tissue / noise labels can be omitted via ``include_labels``.
    """
    pos = np.asarray(all_positions, dtype=float)
    labels = np.asarray(region_labels)
    if pos.shape[0] != labels.shape[0]:
        raise ValueError("positions and labels must have the same length")

    if include_labels is None:
        uniq = sorted(int(x) for x in np.unique(labels))
    else:
        uniq = [int(x) for x in include_labels]

    return [pos[labels == lab] for lab in uniq]


def nearest_data_labels(
    node_positions: np.ndarray,
    data_points: np.ndarray,
    data_labels: np.ndarray,
) -> np.ndarray:
    """Label each scaffold node by its nearest data point (1-NN).

    Probe/harness helper for signal vs tissue filtering when Stage-1 cluster
    labels alone do not separate generative components (#41 fitted-circle
    probes). Does not change SI filtration defaults.
    """
    from sklearn.neighbors import NearestNeighbors

    pos = np.asarray(node_positions, dtype=float)
    pts = np.asarray(data_points, dtype=float)
    labs = np.asarray(data_labels)
    if pos.ndim != 2 or pts.ndim != 2:
        raise ValueError("node_positions and data_points must be 2-D")
    if pts.shape[0] != labs.shape[0]:
        raise ValueError("data_points and data_labels must have the same length")
    if pos.shape[0] == 0:
        return np.empty((0,), dtype=labs.dtype)
    nn = NearestNeighbors(n_neighbors=1).fit(pts)
    _, idx = nn.kneighbors(pos)
    return np.asarray(labs)[idx[:, 0]]


def topology_from_accepted_regions(
    all_positions: np.ndarray,
    region_labels: np.ndarray,
    sigma_star: float | Sequence[float],
    *,
    include_labels: Optional[Iterable[int]] = None,
    reading: ReadingMode = "lifetime",
    max_dim: int = 2,
    filtration_mult: float = FILTRATION_MULTIPLIER,
    lifetime_frac: float = DEFAULT_LIFETIME_FRAC,
) -> list[RegionTopologyReport]:
    """Scenario helper: accepted-region labels → per-region VR-PH reports (#41).

    Splits ``all_positions`` by ``region_labels`` (typically Stage-1 cluster /
    recursion-accepted region ids on scaffold nodes), then runs
    ``per_region_topology``. Pass ``include_labels`` to drop tissue / noise
    labels. Prefer ``reading="lifetime"`` on tissue-polluted scaffolds.

    Does not flip recovery scenario assertions — callers gather evidence first.
    """
    region_points = extract_region_node_positions(
        all_positions,
        region_labels,
        include_labels=include_labels,
    )
    # When include_labels is set, keep report region_id aligned with those labels.
    reports = per_region_topology(
        region_points,
        sigma_star,
        reading=reading,
        max_dim=max_dim,
        filtration_mult=filtration_mult,
        lifetime_frac=lifetime_frac,
    )
    if include_labels is not None:
        labs = [int(x) for x in include_labels]
        if len(labs) != len(reports):
            raise ValueError(
                f"include_labels length {len(labs)} != n_regions {len(reports)}"
            )
        return [
            RegionTopologyReport(
                region_id=lab,
                n_points=rep.n_points,
                betti=rep.betti,
                reading=rep.reading,
                sigma_star=rep.sigma_star,
                filtration_radius=rep.filtration_radius,
            )
            for lab, rep in zip(labs, reports)
        ]
    return reports


@dataclass(frozen=True)
class ReadingComparison:
    """Side-by-side fixed_threshold vs lifetime Betti readings (#41 diagnostics)."""

    fixed_threshold: tuple[int, ...]
    lifetime: tuple[int, ...]
    sigma_star: float
    filtration_radius: float
    lifetime_frac: float
    n_points: int


def compare_readings(
    points: np.ndarray,
    sigma_star: float,
    *,
    max_dim: int = 2,
    filtration_mult: float = FILTRATION_MULTIPLIER,
    lifetime_frac: float = DEFAULT_LIFETIME_FRAC,
) -> ReadingComparison:
    """Compare SI fixed-threshold reading against lifetime reading on one cloud.

    Diagnostic helper for #41 item 1 (filtration/persistence reading). Does not
    choose a production default — callers record both and decide with evidence.
    """
    arr = np.asarray(points, dtype=float)
    r = filtration_radius(sigma_star, multiplier=filtration_mult)
    n_points = int(arr.shape[0]) if arr.ndim == 2 else 0
    if arr.size == 0 or arr.ndim != 2 or arr.shape[0] == 0:
        zeros = tuple(0 for _ in range(max_dim + 1))
        return ReadingComparison(
            fixed_threshold=zeros,
            lifetime=zeros,
            sigma_star=float(sigma_star),
            filtration_radius=r,
            lifetime_frac=float(lifetime_frac),
            n_points=0,
        )
    fixed = region_betti_numbers(
        arr,
        sigma_star,
        reading="fixed_threshold",
        max_dim=max_dim,
        filtration_mult=filtration_mult,
    )
    life = region_betti_numbers(
        arr,
        sigma_star,
        reading="lifetime",
        max_dim=max_dim,
        filtration_mult=filtration_mult,
        lifetime_frac=lifetime_frac,
    )
    return ReadingComparison(
        fixed_threshold=fixed,
        lifetime=life,
        sigma_star=float(sigma_star),
        filtration_radius=r,
        lifetime_frac=float(lifetime_frac),
        n_points=n_points,
    )


@dataclass(frozen=True)
class LifetimeFracSweepRow:
    """One row of a ``lifetime_frac`` (or mult) sweep table (#41 / A4-T15)."""

    lifetime_frac: float
    filtration_mult: float
    betti: tuple[int, ...]
    n_points: int
    matches_target: bool | None = None
    region_id: int | None = None


def sweep_lifetime_frac(
    points: np.ndarray,
    sigma_star: float,
    *,
    fracs: Sequence[float],
    filtration_mult: float = FILTRATION_MULTIPLIER,
    max_dim: int = 2,
    target_betti: tuple[int, ...] | None = None,
    region_id: int | None = None,
) -> list[LifetimeFracSweepRow]:
    """Sweep lifetime reading over ``lifetime_frac`` values on one cloud.

    Returns a table artifact for calibration / documentation. Does not change
    SI defaults or flip recovery assertions — callers decide with evidence.
    """
    arr = np.asarray(points, dtype=float)
    n_points = int(arr.shape[0]) if arr.ndim == 2 else 0
    rows: list[LifetimeFracSweepRow] = []
    for frac in fracs:
        if n_points == 0:
            betti = tuple(0 for _ in range(max_dim + 1))
        else:
            betti = lifetime_betti_numbers(
                arr,
                sigma_star,
                max_dim=max_dim,
                filtration_mult=filtration_mult,
                lifetime_frac=float(frac),
            )
        match: bool | None = None
        if target_betti is not None:
            match = betti == tuple(target_betti)
        rows.append(
            LifetimeFracSweepRow(
                lifetime_frac=float(frac),
                filtration_mult=float(filtration_mult),
                betti=betti,
                n_points=n_points,
                matches_target=match,
                region_id=region_id,
            )
        )
    return rows


def sweep_lifetime_frac_per_region(
    all_positions: np.ndarray,
    region_labels: np.ndarray,
    sigma_star: float | Sequence[float],
    *,
    fracs: Sequence[float],
    include_labels: Optional[Iterable[int]] = None,
    filtration_mult: float = FILTRATION_MULTIPLIER,
    max_dim: int = 2,
    target_betti: tuple[int, ...] | None = None,
) -> list[LifetimeFracSweepRow]:
    """Per-region ``lifetime_frac`` sweep via accepted-region label split.

    Flattened table: one ``LifetimeFracSweepRow`` per (region, frac). Useful for
    nested-sphere / linked-tori clean-shell harnesses (#41).
    """
    region_points = extract_region_node_positions(
        all_positions,
        region_labels,
        include_labels=include_labels,
    )
    n = len(region_points)
    if np.isscalar(sigma_star):
        sigmas = [float(sigma_star)] * n  # type: ignore[arg-type]
    else:
        sigmas = [float(s) for s in sigma_star]  # type: ignore[arg-type]
        if len(sigmas) != n:
            raise ValueError(
                f"sigma_star length {len(sigmas)} != n_regions {n}"
            )
    if include_labels is None:
        region_ids = list(range(n))
    else:
        region_ids = [int(x) for x in include_labels]
        if len(region_ids) != n:
            raise ValueError(
                f"include_labels length {len(region_ids)} != n_regions {n}"
            )

    rows: list[LifetimeFracSweepRow] = []
    for rid, pts, sig in zip(region_ids, region_points, sigmas):
        rows.extend(
            sweep_lifetime_frac(
                pts,
                sig,
                fracs=fracs,
                filtration_mult=filtration_mult,
                max_dim=max_dim,
                target_betti=target_betti,
                region_id=rid,
            )
        )
    return rows


@dataclass(frozen=True)
class PerRegionPHRunResult:
    """Result of a per-region PH probe run (#41 / A4-T16 scaffolding)."""

    reports: tuple[RegionTopologyReport, ...]
    expected_betti: tuple[int, ...] | None
    all_match: bool | None
    reading: ReadingMode
    filtration_mult: float
    lifetime_frac: float
    scenario: str


def run_per_region_ph(
    all_positions: np.ndarray,
    region_labels: np.ndarray,
    sigma_star: float | Sequence[float],
    *,
    scenario: str = "generic",
    include_labels: Optional[Iterable[int]] = None,
    reading: ReadingMode = "lifetime",
    max_dim: int = 2,
    filtration_mult: float = FILTRATION_MULTIPLIER,
    lifetime_frac: float = DEFAULT_LIFETIME_FRAC,
    expected_betti: tuple[int, ...] | None = None,
) -> PerRegionPHRunResult:
    """Prototype runner: accepted-region labels → PH reports + optional match.

    Intended scaffolding for nested_spheres / linked_tori recovery paths.
    Callers may keep scenario assertions ``@awaiting`` / xfail until
    ``all_match`` is green on fitted regions — this helper never weakens tests.
    """
    reports = topology_from_accepted_regions(
        all_positions,
        region_labels,
        sigma_star,
        include_labels=include_labels,
        reading=reading,
        max_dim=max_dim,
        filtration_mult=filtration_mult,
        lifetime_frac=lifetime_frac,
    )
    match: bool | None = None
    if expected_betti is not None:
        target = tuple(expected_betti)
        match = all(rep.betti == target for rep in reports) and len(reports) > 0
    return PerRegionPHRunResult(
        reports=tuple(reports),
        expected_betti=(
            tuple(expected_betti) if expected_betti is not None else None
        ),
        all_match=match,
        reading=reading,
        filtration_mult=float(filtration_mult),
        lifetime_frac=float(lifetime_frac),
        scenario=str(scenario),
    )


def format_lifetime_frac_sweep_table(
    rows: Sequence[LifetimeFracSweepRow],
) -> str:
    """Render a compact text table for COORDINATION / artifact notes."""
    header = "region_id\tfrac\tmult\tn\tbetti\tmatch"
    lines = [header]
    for row in rows:
        rid = "" if row.region_id is None else str(row.region_id)
        match = "" if row.matches_target is None else str(row.matches_target)
        lines.append(
            f"{rid}\t{row.lifetime_frac:g}\t{row.filtration_mult:g}\t"
            f"{row.n_points}\t{row.betti}\t{match}"
        )
    return "\n".join(lines)
