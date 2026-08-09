"""Adversarial-null generators + ROC harness for hollow-edge ``h_0`` (#44 / A4-T18).

Calibration protocol from ``reference/empty_region_evidence_and_scale.md`` §6.3:

* (a) connected sheets with density gradients + curvature — must **not** cut;
* (b) two components with tissue in the gap at increasing rates — must cut
  until tissue ≈ signal.

Reports a ROC over the hollowness ratio ``H = n_mid / (n_end + eps)`` with
mid-ball radius ``r = L/4``. This is a **calibration harness** for the
acceptance-path threshold; it does not wire ``prefer_hollow_edge_prepass``
(A2) and does not flip ``@awaiting`` recovery tests.

No fixture-seed tuning of a single ``h_0`` — tests assert ROC structure /
separability across parameter sweeps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

# Proposal-path mid-ball fraction (theory note §3.1 / §7).
DEFAULT_MID_RADIUS_FRAC: float = 0.25
DEFAULT_EPS: float = 1e-9

# A2 operational HollowEdgeConfig defaults (proteus.stage1.edge_evidence;
# mirrored here so the ROC harness stays import-light for table sweeps).
A2_MID_RADIUS_FRAC: float = 0.35
A2_H0: float = 0.35
A2_MIN_END_COUNT: float = 0.5
A2_GABRIEL_FALLBACK: bool = True


@dataclass(frozen=True)
class HollowEdgeCase:
    """One labeled edge with data cloud for hollowness scoring."""

    points: np.ndarray
    endpoint_i: np.ndarray
    endpoint_j: np.ndarray
    should_cut: bool
    kind: str
    meta: dict[str, float | int | str]


@dataclass(frozen=True)
class ROCPoint:
    threshold: float
    tpr: float
    fpr: float
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass(frozen=True)
class ROCResult:
    thresholds: tuple[float, ...]
    points: tuple[ROCPoint, ...]
    auc: float
    scores: np.ndarray
    labels_should_cut: np.ndarray


def count_in_ball(points: np.ndarray, center: np.ndarray, radius: float) -> int:
    """Count samples within Euclidean ball ``(center, radius)``."""
    pts = np.asarray(points, dtype=float)
    c = np.asarray(center, dtype=float)
    if pts.size == 0 or radius < 0.0:
        return 0
    d2 = np.sum((pts - c) ** 2, axis=1)
    return int(np.count_nonzero(d2 <= radius * radius))


def hollowness_ratio(
    points: np.ndarray,
    endpoint_i: np.ndarray,
    endpoint_j: np.ndarray,
    *,
    mid_radius_frac: float = DEFAULT_MID_RADIUS_FRAC,
    eps: float = DEFAULT_EPS,
) -> float:
    """``H = n_mid / (n_end + eps)`` with ``r = mid_radius_frac * L`` (default L/4).

    When both endpoint balls are empty (endpoints far from data), returns
    ``0.0`` and relies on Gabriel fallback for low-n decisions — a ratio
    with ``n_end=0`` is not a calibrated density estimate.
    """
    xi = np.asarray(endpoint_i, dtype=float)
    xj = np.asarray(endpoint_j, dtype=float)
    L = float(np.linalg.norm(xj - xi))
    if L <= 0.0:
        return 1.0
    r = float(mid_radius_frac) * L
    mid = 0.5 * (xi + xj)
    n_mid = count_in_ball(points, mid, r)
    n_i = count_in_ball(points, xi, r)
    n_j = count_in_ball(points, xj, r)
    n_end = 0.5 * (n_i + n_j)
    if n_end <= 0.0:
        return 0.0 if n_mid == 0 else float(n_mid)  # uncalibrated; avoid 1/eps blow-up
    return float(n_mid / (n_end + float(eps)))


def _snap_to_nearest(data: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Replace each query row with its nearest neighbor in ``data``."""
    d2 = np.sum((data[:, None, :] - query[None, :, :]) ** 2, axis=2)
    nn = np.argmin(d2, axis=0)
    return data[nn]

def gabriel_is_hollow(
    points: np.ndarray,
    endpoint_i: np.ndarray,
    endpoint_j: np.ndarray,
    *,
    exclude_endpoints: bool = True,
) -> bool:
    """Gabriel criterion: hollow if diameter ball contains any other sample."""
    xi = np.asarray(endpoint_i, dtype=float)
    xj = np.asarray(endpoint_j, dtype=float)
    mid = 0.5 * (xi + xj)
    r = 0.5 * float(np.linalg.norm(xj - xi))
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return True
    d2 = np.sum((pts - mid) ** 2, axis=1)
    inside = d2 <= (r * r) + 1e-12
    if not exclude_endpoints:
        return bool(np.any(inside))
    # Exclude samples coincident with endpoints (scaffold/data coincidences).
    near_i = np.sum((pts - xi) ** 2, axis=1) <= 1e-18
    near_j = np.sum((pts - xj) ** 2, axis=1) <= 1e-18
    return bool(np.any(inside & ~near_i & ~near_j))


def score_edge(case: HollowEdgeCase, *, mid_radius_frac: float = DEFAULT_MID_RADIUS_FRAC) -> float:
    """Lower score ⇒ more hollow ⇒ more likely to cut under ``H < h_0``."""
    return hollowness_ratio(
        case.points,
        case.endpoint_i,
        case.endpoint_j,
        mid_radius_frac=mid_radius_frac,
    )


def endpoint_end_mass(
    points: np.ndarray,
    endpoint_i: np.ndarray,
    endpoint_j: np.ndarray,
    *,
    mid_radius_frac: float = DEFAULT_MID_RADIUS_FRAC,
) -> float:
    """Mean endpoint-ball occupancy ``n_end = 0.5*(n_i+n_j)`` at ``r = frac*L``."""
    xi = np.asarray(endpoint_i, dtype=float)
    xj = np.asarray(endpoint_j, dtype=float)
    L = float(np.linalg.norm(xj - xi))
    if L <= 0.0:
        return 0.0
    r = float(mid_radius_frac) * L
    n_i = count_in_ball(points, xi, r)
    n_j = count_in_ball(points, xj, r)
    return 0.5 * (n_i + n_j)


def decide_cut_a2_parity(
    case: HollowEdgeCase,
    *,
    mid_radius_frac: float = A2_MID_RADIUS_FRAC,
    h0: float = A2_H0,
    min_end_count: float = A2_MIN_END_COUNT,
    gabriel_fallback: bool = True,
) -> bool:
    """Mirror A2 ``hollow_edge_mask`` cut rule for one labeled case.

    When ``n_end >= min_end_count``: cut iff ``H < h0``.
    Else if ``gabriel_fallback``: cut iff Gabriel diameter ball is empty of
    *other* samples (harness helper; see ``gabriel_is_hollow``).
    Else: do not cut.

    Calibration-only — mirrors ``proteus.stage1.edge_evidence.hollow_edge_mask``
    without requiring a Stage-1 scaffold; production config lives in A2.
    """
    n_end = endpoint_end_mass(
        case.points,
        case.endpoint_i,
        case.endpoint_j,
        mid_radius_frac=mid_radius_frac,
    )
    if n_end >= float(min_end_count):
        return score_edge(case, mid_radius_frac=mid_radius_frac) < float(h0)
    if gabriel_fallback:
        return gabriel_is_hollow(case.points, case.endpoint_i, case.endpoint_j)
    return False


# ---------------------------------------------------------------------------
# Generators (a): connected curved sheet with density gradient — must NOT cut
# ---------------------------------------------------------------------------


def _embed_sheet(uv: np.ndarray, curvature: float) -> np.ndarray:
    """Map intrinsic ``(u,v)∈[0,1]^2`` to a curved sheet in R^3."""
    return np.column_stack(
        [
            uv[:, 0],
            uv[:, 1],
            float(curvature) * (uv[:, 0] ** 2),
        ]
    )


def make_curved_density_sheet(
    *,
    n_data: int = 2500,
    n_u: int = 6,
    n_v: int = 5,
    density_power: float = 1.5,
    curvature: float = 0.35,
    noise: float = 0.005,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Curved sheet with density ∝ u^p plus a coarse scaffold grid.

    Returns ``(data_points, scaffold_xyz)`` where scaffold nodes are snapped
    onto data so endpoint balls see mass; grid spacing is coarse enough that
    ``r = L/4`` mid-balls typically contain samples under the density gradient.
    """
    rng = np.random.default_rng(seed)
    # Dense data via rejection sampling (u-gradient — adversarial for absolute
    # density tests; ratio form should remain O(1) on within-support edges).
    pts_uv: list[np.ndarray] = []
    while len(pts_uv) < n_data:
        cand = rng.uniform(0.0, 1.0, size=(n_data * 3, 2))
        w = (0.2 + cand[:, 0]) ** float(density_power)
        w = w / w.max()
        keep = rng.random(len(cand)) < w
        pts_uv.extend(list(cand[keep]))
    data_uv = np.asarray(pts_uv[:n_data], dtype=float)
    data = _embed_sheet(data_uv, curvature)
    data = data + float(noise) * rng.normal(size=data.shape)

    us = np.linspace(0.1, 0.9, int(n_u))
    vs = np.linspace(0.1, 0.9, int(n_v))
    uu, vv = np.meshgrid(us, vs, indexing="xy")
    grid_uv = np.column_stack([uu.ravel(), vv.ravel()])
    scaffold = _embed_sheet(grid_uv, curvature)
    scaffold = _snap_to_nearest(data, scaffold)
    return data, scaffold


def sheet_within_support_edges(
    data: np.ndarray,
    scaffold: np.ndarray,
    *,
    n_u: int,
    n_v: int,
) -> list[HollowEdgeCase]:
    """4-connected grid edges on the scaffold — all ``should_cut=False``."""
    cases: list[HollowEdgeCase] = []

    def idx(iu: int, iv: int) -> int:
        return iv * n_u + iu

    for iv in range(n_v):
        for iu in range(n_u):
            i = idx(iu, iv)
            neighbors: list[int] = []
            if iu + 1 < n_u:
                neighbors.append(idx(iu + 1, iv))
            if iv + 1 < n_v:
                neighbors.append(idx(iu, iv + 1))
            for j in neighbors:
                # Drop collapsed snaps (identical / near-identical endpoints).
                if float(np.linalg.norm(scaffold[i] - scaffold[j])) < 1e-6:
                    continue
                cases.append(
                    HollowEdgeCase(
                        points=data,
                        endpoint_i=scaffold[i],
                        endpoint_j=scaffold[j],
                        should_cut=False,
                        kind="sheet_within",
                        meta={"i": i, "j": j},
                    )
                )
    return cases


def generate_connected_sheet_null(
    *,
    seed: int = 0,
    n_data: int = 2500,
    n_u: int = 6,
    n_v: int = 5,
    density_power: float = 1.5,
    curvature: float = 0.35,
) -> list[HollowEdgeCase]:
    """Adversarial null (a): density-gradient curved sheet — must not cut."""
    data, scaffold = make_curved_density_sheet(
        n_data=n_data,
        n_u=n_u,
        n_v=n_v,
        density_power=density_power,
        curvature=curvature,
        seed=seed,
    )
    return sheet_within_support_edges(data, scaffold, n_u=n_u, n_v=n_v)


# ---------------------------------------------------------------------------
# Generators (b): two components + gap tissue — must cut until tissue ~ signal
# ---------------------------------------------------------------------------


def make_two_blobs_with_gap_tissue(
    *,
    n_per_blob: int = 200,
    gap: float = 1.5,
    blob_radius: float = 0.55,
    tissue_rate: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two 2-D disks separated by ``gap``, optional tissue in the mid strip.

    ``tissue_rate`` is tissue density relative to mean blob areal density
    (1.0 ≈ same samples per unit area as a blob). Returns
    ``(points, labels, centers)`` with labels ``{0,1,-1}``.
    """
    rng = np.random.default_rng(seed)
    c0 = np.array([-0.5 * gap - blob_radius, 0.0])
    c1 = np.array([0.5 * gap + blob_radius, 0.0])

    def _disk(center: np.ndarray, n: int, s: int) -> np.ndarray:
        local = np.random.default_rng(s)
        r = blob_radius * np.sqrt(local.random(n))
        theta = local.uniform(0.0, 2.0 * np.pi, size=n)
        return center + np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    b0 = _disk(c0, n_per_blob, seed + 1)
    b1 = _disk(c1, n_per_blob, seed + 2)
    labels = np.array([0] * n_per_blob + [1] * n_per_blob, dtype=int)
    points = np.vstack([b0, b1])

    if tissue_rate > 0.0:
        blob_area = np.pi * blob_radius * blob_radius
        blob_density = n_per_blob / max(blob_area, 1e-12)
        x_lo, x_hi = -0.5 * gap, 0.5 * gap
        y_lo, y_hi = -1.2 * blob_radius, 1.2 * blob_radius
        band_area = max((x_hi - x_lo) * (y_hi - y_lo), 1e-12)
        n_tissue = int(np.round(tissue_rate * blob_density * band_area))
        if n_tissue > 0:
            tissue = np.column_stack(
                [
                    rng.uniform(x_lo, x_hi, size=n_tissue),
                    rng.uniform(y_lo, y_hi, size=n_tissue),
                ]
            )
            d0 = np.linalg.norm(tissue - c0, axis=1)
            d1 = np.linalg.norm(tissue - c1, axis=1)
            keep = (d0 > blob_radius) & (d1 > blob_radius)
            tissue = tissue[keep]
            if len(tissue):
                points = np.vstack([points, tissue])
                labels = np.concatenate(
                    [labels, np.full(len(tissue), -1, dtype=int)]
                )

    return points, labels, np.vstack([c0, c1])


def _data_pair_edges(
    points: np.ndarray,
    mask: np.ndarray,
    *,
    n_edges: int,
    min_sep: float,
    max_sep: float,
    seed: int,
    should_cut: bool,
    kind: str,
    meta: dict[str, float | int | str],
) -> list[HollowEdgeCase]:
    """Sample unordered pairs from ``points[mask]`` with separation in range."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return []
    cases: list[HollowEdgeCase] = []
    attempts = 0
    seen: set[tuple[int, int]] = set()
    while len(cases) < n_edges and attempts < n_edges * 80:
        attempts += 1
        a, b = rng.choice(idx, size=2, replace=False)
        i, j = (int(a), int(b)) if a < b else (int(b), int(a))
        if (i, j) in seen:
            continue
        sep = float(np.linalg.norm(points[i] - points[j]))
        if sep < min_sep or sep > max_sep:
            continue
        seen.add((i, j))
        cases.append(
            HollowEdgeCase(
                points=points,
                endpoint_i=points[i],
                endpoint_j=points[j],
                should_cut=should_cut,
                kind=kind,
                meta={**meta, "sep": sep},
            )
        )
    return cases


def generate_gap_tissue_cases(
    *,
    tissue_rate: float,
    seed: int = 0,
    n_per_blob: int = 200,
    gap: float = 1.5,
    blob_radius: float = 0.55,
    n_bridge: int = 10,
    n_within: int = 10,
) -> list[HollowEdgeCase]:
    """Adversarial null (b) at one tissue rate: bridge + within-blob edges."""
    points, labels, centers = make_two_blobs_with_gap_tissue(
        n_per_blob=n_per_blob,
        gap=gap,
        blob_radius=blob_radius,
        tissue_rate=tissue_rate,
        seed=seed,
    )
    cases: list[HollowEdgeCase] = []
    # Within-blob: medium chords inside each disk (occupied support).
    within_sep = (0.35 * blob_radius, 1.1 * blob_radius)
    for side in (0, 1):
        cases.extend(
            _data_pair_edges(
                points,
                labels == side,
                n_edges=n_within,
                min_sep=within_sep[0],
                max_sep=within_sep[1],
                seed=seed + 20 + side,
                should_cut=False,
                kind="blob_within",
                meta={"tissue_rate": float(tissue_rate), "side": side},
            )
        )

    # Bridge edges: one endpoint in each blob, spanning the gap.
    rng = np.random.default_rng(seed + 99)
    idx0 = np.flatnonzero(labels == 0)
    idx1 = np.flatnonzero(labels == 1)
    # Prefer near-facing samples (toward the gap) for cleaner mid voids.
    facing0 = idx0[points[idx0, 0] > centers[0, 0]]
    facing1 = idx1[points[idx1, 0] < centers[1, 0]]
    if len(facing0) < 2:
        facing0 = idx0
    if len(facing1) < 2:
        facing1 = idx1
    for _ in range(n_bridge):
        i = int(rng.choice(facing0))
        j = int(rng.choice(facing1))
        cases.append(
            HollowEdgeCase(
                points=points,
                endpoint_i=points[i],
                endpoint_j=points[j],
                should_cut=True,
                kind="gap_bridge",
                meta={
                    "tissue_rate": float(tissue_rate),
                    "gap": float(gap),
                    "center_sep": float(np.linalg.norm(centers[1] - centers[0])),
                },
            )
        )
    return cases


def generate_gap_tissue_rate_sweep(
    tissue_rates: Sequence[float] = (0.0, 0.05, 0.15, 0.35, 0.7, 1.0),
    *,
    seed: int = 0,
) -> dict[float, list[HollowEdgeCase]]:
    """Map tissue_rate → edge cases for ROC / separability tables."""
    return {
        float(r): generate_gap_tissue_cases(tissue_rate=float(r), seed=seed + int(10 * r))
        for r in tissue_rates
    }


# ---------------------------------------------------------------------------
# ROC harness
# ---------------------------------------------------------------------------


def roc_from_cases(
    cases: Iterable[HollowEdgeCase],
    *,
    thresholds: Sequence[float] | None = None,
    mid_radius_frac: float = DEFAULT_MID_RADIUS_FRAC,
) -> ROCResult:
    """ROC for predictor ``cut ⇔ H < h_0`` (lower H = more hollow)."""
    case_list = list(cases)
    if not case_list:
        raise ValueError("roc_from_cases requires at least one edge case")
    scores = np.asarray(
        [score_edge(c, mid_radius_frac=mid_radius_frac) for c in case_list],
        dtype=float,
    )
    y = np.asarray([1 if c.should_cut else 0 for c in case_list], dtype=int)

    if thresholds is None:
        # Dense grid over observed score range (calibration, not seed-tuned h_0).
        lo = float(np.min(scores))
        hi = float(np.max(scores))
        if hi <= lo:
            thresholds = [lo - 1e-6, lo + 1e-6]
        else:
            thresholds = list(np.linspace(lo - 1e-6, hi + 1e-6, num=41))

    thr_t = tuple(float(t) for t in thresholds)
    points: list[ROCPoint] = []
    for t in thr_t:
        pred = (scores < t).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        tn = int(np.sum((pred == 0) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        points.append(
            ROCPoint(threshold=t, tpr=tpr, fpr=fpr, tp=tp, fp=fp, tn=tn, fn=fn)
        )

    # Trapezoidal AUC in FPR–TPR space (sorted by FPR).
    order = sorted(points, key=lambda p: (p.fpr, p.tpr))
    auc = 0.0
    for a, b in zip(order, order[1:]):
        auc += 0.5 * (b.tpr + a.tpr) * (b.fpr - a.fpr)
    return ROCResult(
        thresholds=thr_t,
        points=tuple(points),
        auc=float(auc),
        scores=scores,
        labels_should_cut=y,
    )


def pooled_adversarial_roc(
    *,
    sheet_seed: int = 0,
    gap_seed: int = 10,
    tissue_rate_for_positives: float = 0.0,
    mid_radius_frac: float = DEFAULT_MID_RADIUS_FRAC,
) -> ROCResult:
    """Pool sheet negatives + low-tissue bridge positives for a calibration ROC."""
    negatives = generate_connected_sheet_null(seed=sheet_seed)
    gap_cases = generate_gap_tissue_cases(
        tissue_rate=tissue_rate_for_positives,
        seed=gap_seed,
    )
    # Keep within-blob as extra negatives; bridges as positives.
    return roc_from_cases(
        list(negatives) + list(gap_cases),
        mid_radius_frac=mid_radius_frac,
    )


@dataclass(frozen=True)
class FixedThresholdConfusion:
    """Confusion counts for a single ``(mid_frac, h0)`` decision rule."""

    mid_radius_frac: float
    h0: float
    tp: int
    fp: int
    tn: int
    fn: int
    tpr: float
    fpr: float


@dataclass(frozen=True)
class SheetNullQuantiles:
    """Empirical H distribution on sheet (must-not-cut) edges — Poisson-ish null."""

    mid_radius_frac: float
    n_edges: int
    quantiles: dict[str, float]
    mean_h: float
    mean_end_mass: float


def confusion_at_h0(
    cases: Iterable[HollowEdgeCase],
    *,
    h0: float,
    mid_radius_frac: float = DEFAULT_MID_RADIUS_FRAC,
    use_a2_parity_decision: bool = False,
    min_end_count: float = A2_MIN_END_COUNT,
    gabriel_fallback: bool = True,
) -> FixedThresholdConfusion:
    """Evaluate cut rule at a fixed ``h0`` (optionally A2-parity gabriel gate)."""
    case_list = list(cases)
    if not case_list:
        raise ValueError("confusion_at_h0 requires at least one edge case")
    y = np.asarray([1 if c.should_cut else 0 for c in case_list], dtype=int)
    if use_a2_parity_decision:
        pred = np.asarray(
            [
                1
                if decide_cut_a2_parity(
                    c,
                    mid_radius_frac=mid_radius_frac,
                    h0=h0,
                    min_end_count=min_end_count,
                    gabriel_fallback=gabriel_fallback,
                )
                else 0
                for c in case_list
            ],
            dtype=int,
        )
    else:
        scores = np.asarray(
            [score_edge(c, mid_radius_frac=mid_radius_frac) for c in case_list],
            dtype=float,
        )
        pred = (scores < float(h0)).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return FixedThresholdConfusion(
        mid_radius_frac=float(mid_radius_frac),
        h0=float(h0),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        tpr=tp / max(tp + fn, 1),
        fpr=fp / max(fp + tn, 1),
    )


def sheet_null_h_quantiles(
    *,
    mid_radius_frac: float = DEFAULT_MID_RADIUS_FRAC,
    seed: int = 0,
    qs: Sequence[float] = (0.01, 0.05, 0.1, 0.25, 0.5),
) -> SheetNullQuantiles:
    """Empirical H quantiles on density-gradient sheet edges (null for cuts).

    Under a locally homogeneous Poisson field, ``E[H]≈1`` for equal-radius
    balls; the lower tail of this sheet distribution is the practical
    Poisson-null reference for choosing ``h0`` without fixture-seed tuning.
    """
    cases = generate_connected_sheet_null(seed=seed)
    scores = np.asarray(
        [score_edge(c, mid_radius_frac=mid_radius_frac) for c in cases],
        dtype=float,
    )
    ends = np.asarray(
        [
            endpoint_end_mass(
                c.points,
                c.endpoint_i,
                c.endpoint_j,
                mid_radius_frac=mid_radius_frac,
            )
            for c in cases
        ],
        dtype=float,
    )
    qmap = {f"q{q:g}": float(np.quantile(scores, q)) for q in qs}
    return SheetNullQuantiles(
        mid_radius_frac=float(mid_radius_frac),
        n_edges=int(len(cases)),
        quantiles=qmap,
        mean_h=float(np.mean(scores)),
        mean_end_mass=float(np.mean(ends)),
    )


def mid_frac_roc_table(
    *,
    mid_fracs: Sequence[float] = (DEFAULT_MID_RADIUS_FRAC, A2_MID_RADIUS_FRAC),
    sheet_seed: int = 1,
    gap_seed: int = 11,
) -> dict[float, ROCResult]:
    """ROC AUC table across mid-ball fractions on the same adversarial pool."""
    return {
        float(f): pooled_adversarial_roc(
            sheet_seed=sheet_seed,
            gap_seed=gap_seed,
            tissue_rate_for_positives=0.0,
            mid_radius_frac=float(f),
        )
        for f in mid_fracs
    }


# ---------------------------------------------------------------------------
# Recommended HollowEdgeConfig export (A4-T24 → A2-T31)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HollowEdgeConfigRow:
    """One candidate ``(mid_frac, h0, min_end, gabriel)`` with ROC / null stats."""

    mid_radius_frac: float
    h0: float
    min_end_count: float
    gabriel_fallback: bool
    tpr: float
    fpr: float
    sheet_q01: float
    sheet_mean_h: float
    auc: float
    h0_at_or_below_sheet_q01: bool
    role: str


@dataclass(frozen=True)
class HollowEdgeConfigRecommendation:
    """Exported recommendation bundle for A2 ``HollowEdgeConfig`` (#44 / A4-T24).

    Sheet-null discipline: keep ``h0 ≤`` empirical sheet ``q01`` so H-only FPR
    on the density-gradient curved sheet stays near zero.  A2's nested probe
    found ``mid_frac=0.35`` often empty-ball (non-discriminative) while
    ``mid_frac≈0.5`` separates H but is not yet a lifted cut-set — both rows
    are exported; production choice stays with A2 / A1.
    """

    primary: HollowEdgeConfigRow
    alternates: tuple[HollowEdgeConfigRow, ...]
    table: tuple[HollowEdgeConfigRow, ...]
    note: str


def _pooled_sheet_bridge_cases(
    *,
    sheet_seed: int = 5,
    gap_seed: int = 11,
) -> list[HollowEdgeCase]:
    sheet = generate_connected_sheet_null(
        seed=sheet_seed, density_power=1.8, curvature=0.4,
    )
    gap = generate_gap_tissue_cases(tissue_rate=0.0, seed=gap_seed)
    bridges = [c for c in gap if c.kind == "gap_bridge"]
    withins = [c for c in gap if c.kind == "blob_within"]
    return list(sheet) + bridges + withins


def sweep_hollow_edge_config_table(
    *,
    mid_fracs: Sequence[float] = (0.25, 0.35, 0.5),
    h0_values: Sequence[float] = (0.25, 0.35, 0.5, 0.7),
    gabriel_options: Sequence[bool] = (False, True),
    min_end_count: float = A2_MIN_END_COUNT,
    sheet_seed: int = 5,
    gap_seed: int = 11,
    null_seed: int = 3,
    max_sheet_fpr: float = 0.05,
    min_bridge_tpr: float = 0.85,
) -> tuple[HollowEdgeConfigRow, ...]:
    """Grid sweep → rows that keep sheet FPR low and bridge TPR high.

    ``h0`` candidates that sit above the sheet ``q01`` are retained in the
    full table only when they still meet FPR/TPR gates (Gabriel or luck);
    the recommendation exporter prefers ``h0 ≤ q01``.
    """
    pooled = _pooled_sheet_bridge_cases(sheet_seed=sheet_seed, gap_seed=gap_seed)
    roc_by_mid = mid_frac_roc_table(
        mid_fracs=mid_fracs, sheet_seed=1, gap_seed=11,
    )
    rows: list[HollowEdgeConfigRow] = []
    for mid in mid_fracs:
        null = sheet_null_h_quantiles(mid_radius_frac=float(mid), seed=null_seed)
        q01 = float(null.quantiles["q0.01"])
        auc = float(roc_by_mid[float(mid)].auc) if float(mid) in roc_by_mid else float(
            pooled_adversarial_roc(
                sheet_seed=1, gap_seed=11, mid_radius_frac=float(mid),
            ).auc
        )
        for h0 in h0_values:
            for gab in gabriel_options:
                conf = confusion_at_h0(
                    pooled,
                    h0=float(h0),
                    mid_radius_frac=float(mid),
                    use_a2_parity_decision=True,
                    min_end_count=float(min_end_count),
                    gabriel_fallback=bool(gab),
                )
                if conf.fpr > max_sheet_fpr or conf.tpr < min_bridge_tpr:
                    continue
                rows.append(
                    HollowEdgeConfigRow(
                        mid_radius_frac=float(mid),
                        h0=float(h0),
                        min_end_count=float(min_end_count),
                        gabriel_fallback=bool(gab),
                        tpr=float(conf.tpr),
                        fpr=float(conf.fpr),
                        sheet_q01=q01,
                        sheet_mean_h=float(null.mean_h),
                        auc=auc,
                        h0_at_or_below_sheet_q01=bool(float(h0) <= q01),
                        role="candidate",
                    )
                )
    # Prefer low FPR, then high TPR, then h0 under q01, then no-Gabriel.
    rows.sort(
        key=lambda r: (
            r.fpr,
            -r.tpr,
            0 if r.h0_at_or_below_sheet_q01 else 1,
            0 if not r.gabriel_fallback else 1,
            r.mid_radius_frac,
            r.h0,
        )
    )
    return tuple(rows)


def recommend_hollow_edge_configs(
    *,
    max_sheet_fpr: float = 0.05,
    min_bridge_tpr: float = 0.85,
) -> HollowEdgeConfigRecommendation:
    """Pick primary + alternates for A2 ``HollowEdgeConfig`` (hand to A2-T31).

    Primary preference order among FPR/TPR-passing rows:
    1. ``h0 ≤`` sheet ``q01`` (Poisson-null discipline),
    2. ``gabriel_fallback=False`` (A2: Gabriel amplifies spurious K=2),
    3. ``mid_radius_frac ≥ 0.5`` when available (A2: 0.35 empty-ball on nested),
    4. else current A2 operational ``(0.35, 0.35)`` if it still passes gates.
    """
    table = sweep_hollow_edge_config_table(
        max_sheet_fpr=max_sheet_fpr, min_bridge_tpr=min_bridge_tpr,
    )
    if not table:
        raise RuntimeError("no HollowEdgeConfig candidate passed FPR/TPR gates")

    def _score(r: HollowEdgeConfigRow) -> tuple:
        return (
            0 if r.h0_at_or_below_sheet_q01 else 1,
            0 if not r.gabriel_fallback else 1,
            0 if r.mid_radius_frac >= 0.5 else 1,
            r.fpr,
            -r.tpr,
            abs(r.mid_radius_frac - 0.5),
            r.h0,
        )

    ranked = sorted(table, key=_score)
    primary = ranked[0]
    # Alternates: distinct mid_frac or gabriel setting from primary.
    alts: list[HollowEdgeConfigRow] = []
    seen = {(primary.mid_radius_frac, primary.h0, primary.gabriel_fallback)}
    for r in ranked[1:]:
        key = (r.mid_radius_frac, r.h0, r.gabriel_fallback)
        if key in seen:
            continue
        seen.add(key)
        alts.append(r)
        if len(alts) >= 3:
            break

    primary = HollowEdgeConfigRow(**{**primary.__dict__, "role": "primary"})
    alts_t = tuple(
        HollowEdgeConfigRow(**{**r.__dict__, "role": f"alternate_{i+1}"})
        for i, r in enumerate(alts)
    )
    note = (
        f"primary mid={primary.mid_radius_frac:g} h0={primary.h0:g} "
        f"gabriel={primary.gabriel_fallback} min_end={primary.min_end_count:g}; "
        f"sheet FPR={primary.fpr:.3f} bridge TPR={primary.tpr:.3f} "
        f"q01={primary.sheet_q01:.3f} AUC={primary.auc:.3f}. "
        "Sheet-null safe ≠ nested cut-set recovery (A2 ARI~0); "
        "do not flip awaiting."
    )
    return HollowEdgeConfigRecommendation(
        primary=primary,
        alternates=alts_t,
        table=table,
        note=note,
    )


def format_hollow_edge_config_table(
    rows: Sequence[HollowEdgeConfigRow],
) -> str:
    """Compact TSV for COORDINATION / A2 handoff."""
    header = (
        "role\tmid\th0\tmin_end\tgabriel\tTPR\tFPR\tq01\tmeanH\tAUC\th0<=q01"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"{r.role}\t{r.mid_radius_frac:g}\t{r.h0:g}\t{r.min_end_count:g}\t"
            f"{r.gabriel_fallback}\t{r.tpr:.3f}\t{r.fpr:.3f}\t"
            f"{r.sheet_q01:.3f}\t{r.sheet_mean_h:.3f}\t{r.auc:.3f}\t"
            f"{r.h0_at_or_below_sheet_q01}"
        )
    return "\n".join(lines)


def recommended_config_as_edge_evidence_kwargs(
    row: HollowEdgeConfigRow,
) -> dict[str, float | bool]:
    """Map a recommendation row to ``HollowEdgeConfig`` field kwargs (A2)."""
    return {
        "mid_radius_frac": float(row.mid_radius_frac),
        "h0": float(row.h0),
        "min_end_count": float(row.min_end_count),
        "gabriel_fallback": bool(row.gabriel_fallback),
    }


# ---------------------------------------------------------------------------
# Multi-tau / mid≥0.5 no-Gabriel ROC handoff refresh (A4-T27 → A2-T33)
# ---------------------------------------------------------------------------

# A4 primary export (A4-T24): sheet-safe mid≥0.5, no Gabriel.
A4_PRIMARY_MID_RADIUS_FRAC: float = 0.5
A4_PRIMARY_H0: float = 0.7
A4_PRIMARY_MIN_END_COUNT: float = 0.5
A4_PRIMARY_GABRIEL_FALLBACK: bool = False


@dataclass(frozen=True)
class MultiTauHollowRocRow:
    """One (mid_frac, h0, gabriel, density_scale) confusion + AUC row."""

    label: str
    mid_radius_frac: float
    h0: float
    gabriel_fallback: bool
    density_scale: float
    tpr: float
    fpr: float
    auc: float
    sheet_q01: float
    h0_at_or_below_sheet_q01: bool


@dataclass(frozen=True)
class MultiTauHollowRocHandoff:
    """Refreshed handoff: mid≥0.5 no-Gab vs A2 primary across density scales."""

    a4_primary: MultiTauHollowRocRow
    a2_primary: MultiTauHollowRocRow
    rows: tuple[MultiTauHollowRocRow, ...]
    note: str


def _thin_case_points(
    case: HollowEdgeCase,
    *,
    density_scale: float,
    rng: np.random.Generator,
) -> HollowEdgeCase:
    """Subsample cloud to approximate a coarser local sampling (tau-like).

    ``density_scale=1`` keeps all points; ``0.5`` keeps ~half. Endpoints are
    always retained via nearest-neighbor snap so the edge geometry is fixed.
    """
    scale = float(density_scale)
    if scale >= 0.999:
        return case
    pts = np.asarray(case.points, dtype=float)
    n = int(pts.shape[0])
    if n <= 8:
        return case
    keep_n = max(8, int(round(n * max(scale, 0.05))))
    keep_n = min(keep_n, n)
    idx = rng.choice(n, size=keep_n, replace=False)
    thin = pts[idx]
    # Ensure endpoints remain representable in the thinned cloud.
    ei = _snap_to_nearest(thin, np.asarray(case.endpoint_i, dtype=float).reshape(1, -1))[0]
    ej = _snap_to_nearest(thin, np.asarray(case.endpoint_j, dtype=float).reshape(1, -1))[0]
    return HollowEdgeCase(
        points=thin,
        endpoint_i=ei,
        endpoint_j=ej,
        should_cut=bool(case.should_cut),
        kind=str(case.kind),
        meta={**case.meta, "density_scale": float(scale)},
    )


def multi_tau_hollow_roc_handoff(
    *,
    mid_fracs: Sequence[float] = (0.5, 0.6, 0.7),
    density_scales: Sequence[float] = (1.0, 0.5, 0.25),
    h0_for_mid: dict[float, float] | None = None,
    min_end_count: float = A4_PRIMARY_MIN_END_COUNT,
    sheet_seed: int = 5,
    gap_seed: int = 11,
    null_seed: int = 3,
    thin_seed: int = 19,
) -> MultiTauHollowRocHandoff:
    """Compare mid≥0.5 no-Gabriel configs vs A2 primary across density scales.

    Density thinning is a tau-adjacent proxy: coarser sampling changes local
    occupancy relative to fixed edge length (isotropic scale leaves H invariant).
    Does not mutate production defaults or flip awaiting tests.
    """
    h0_map = {
        0.5: A4_PRIMARY_H0,
        0.6: 0.7,
        0.7: 0.75,
        A2_MID_RADIUS_FRAC: A2_H0,
    }
    if h0_for_mid:
        h0_map.update({float(k): float(v) for k, v in h0_for_mid.items()})

    pooled = _pooled_sheet_bridge_cases(sheet_seed=sheet_seed, gap_seed=gap_seed)
    rng = np.random.default_rng(thin_seed)
    rows: list[MultiTauHollowRocRow] = []

    def _eval(
        label: str,
        mid: float,
        h0: float,
        gab: bool,
        dens: float,
    ) -> MultiTauHollowRocRow:
        cases = [
            _thin_case_points(c, density_scale=dens, rng=rng) for c in pooled
        ]
        conf = confusion_at_h0(
            cases,
            h0=float(h0),
            mid_radius_frac=float(mid),
            use_a2_parity_decision=True,
            min_end_count=float(min_end_count),
            gabriel_fallback=bool(gab),
        )
        auc = float(
            roc_from_cases(cases, mid_radius_frac=float(mid)).auc
        )
        # Sheet q01 at full density for the mid_frac (Poisson-null reference).
        null = sheet_null_h_quantiles(mid_radius_frac=float(mid), seed=null_seed)
        q01 = float(null.quantiles["q0.01"])
        return MultiTauHollowRocRow(
            label=label,
            mid_radius_frac=float(mid),
            h0=float(h0),
            gabriel_fallback=bool(gab),
            density_scale=float(dens),
            tpr=float(conf.tpr),
            fpr=float(conf.fpr),
            auc=auc,
            sheet_q01=q01,
            h0_at_or_below_sheet_q01=bool(float(h0) <= q01),
        )

    # A4 primary continuum: mid≥0.5, gabriel=False.
    for mid in mid_fracs:
        mid_f = float(mid)
        if mid_f < 0.5 - 1e-12:
            continue
        h0 = float(h0_map.get(mid_f, A4_PRIMARY_H0))
        for dens in density_scales:
            rows.append(
                _eval(
                    f"a4_mid{mid_f:g}_d{float(dens):g}",
                    mid_f,
                    h0,
                    False,
                    float(dens),
                )
            )

    # A2 operational primary (with Gabriel) + no-Gab twin at each density.
    for dens in density_scales:
        rows.append(
            _eval(
                f"a2_primary_d{float(dens):g}",
                A2_MID_RADIUS_FRAC,
                A2_H0,
                True,
                float(dens),
            )
        )
        rows.append(
            _eval(
                f"a2_noGab_d{float(dens):g}",
                A2_MID_RADIUS_FRAC,
                A2_H0,
                False,
                float(dens),
            )
        )

    a4_full = next(
        r
        for r in rows
        if (
            r.mid_radius_frac == A4_PRIMARY_MID_RADIUS_FRAC
            and not r.gabriel_fallback
            and abs(r.density_scale - 1.0) < 1e-12
            and abs(r.h0 - A4_PRIMARY_H0) < 1e-12
        )
    )
    a2_full = next(
        r
        for r in rows
        if (
            r.mid_radius_frac == A2_MID_RADIUS_FRAC
            and r.gabriel_fallback
            and abs(r.density_scale - 1.0) < 1e-12
        )
    )

    note = (
        f"A4 primary mid={a4_full.mid_radius_frac:g} h0={a4_full.h0:g} "
        f"gabriel={a4_full.gabriel_fallback} @dens=1: "
        f"FPR={a4_full.fpr:.3f} TPR={a4_full.tpr:.3f} AUC={a4_full.auc:.3f}; "
        f"A2 primary mid={a2_full.mid_radius_frac:g} h0={a2_full.h0:g} "
        f"gabriel={a2_full.gabriel_fallback} @dens=1: "
        f"FPR={a2_full.fpr:.3f} TPR={a2_full.tpr:.3f} AUC={a2_full.auc:.3f}. "
        "Sheet-null safe ≠ nested ARI; no awaiting flip."
    )
    return MultiTauHollowRocHandoff(
        a4_primary=a4_full,
        a2_primary=a2_full,
        rows=tuple(rows),
        note=note,
    )


def format_multi_tau_hollow_roc_table(
    rows: Sequence[MultiTauHollowRocRow],
) -> str:
    """Compact TSV for COORDINATION / A2 multi-tau handoff."""
    header = (
        "label\tmid\th0\tgabriel\tdens\tTPR\tFPR\tAUC\tq01\th0<=q01"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"{r.label}\t{r.mid_radius_frac:g}\t{r.h0:g}\t"
            f"{r.gabriel_fallback}\t{r.density_scale:g}\t"
            f"{r.tpr:.3f}\t{r.fpr:.3f}\t{r.auc:.3f}\t"
            f"{r.sheet_q01:.3f}\t{r.h0_at_or_below_sheet_q01}"
        )
    return "\n".join(lines)
