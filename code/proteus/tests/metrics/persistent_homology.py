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
