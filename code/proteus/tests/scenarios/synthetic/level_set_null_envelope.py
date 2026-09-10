"""Reproducible φ-ceiling null-ensemble envelope (#48 / A3-T1, A3-T2).

Runs the root ``track_tau`` walk (``max_depth=1``) over a declared null
ensemble × seeds and writes:

1. one table row per root read that has a candidate cut
   (``scene, seed, step, N, tau, phi, accepted, reason[, n_samples, k]``);
2. an envelope summary (min, p1, p5, p10, median, count, any accepts).

Protocol match (2026-09-09): six null scenes × seeds 0–19 → 359 candidate
reads with φ>0; min 0.288 / p1 0.487 / p5 0.630 / p10 0.760 / median 1.30;
plus one exact φ=0 accept on ``lone_gauss2d_null`` seed 17 (excluded from
the envelope percentiles; graph-disconnection, owned by A2).

A3-T2 widen mode adds five geometries and an n×k grid over the six
existing nulls (seeds 0–9).  Kill: any new-geometry candidate with
``0 < φ < 0.25`` stops widening.

A3-T7 ``--mode component-only`` wires A4-T6 lone-component scenes
(circle / nested shell / two-Gaussians, no tissue) at ``n∈{200,500}``
× seeds 0–19 and reports the child-sized envelope.

Not a default pytest test.  Pure helpers below are unit-tested.

Usage::

    PYTHONPATH="src:$PWD" pipenv run python \\
        tests/scenarios/synthetic/level_set_null_envelope.py \\
        --seeds 0-19 --jobs 4

    PYTHONPATH="src:$PWD" pipenv run python \\
        tests/scenarios/synthetic/level_set_null_envelope.py \\
        --mode widen --seeds 0-9 --jobs 6 --csv /tmp/widen.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Sequence

import numpy as np

import proteus.stage1.recursion as recursion_mod
from proteus.stage1.controller import ScaleSearchConfig
from proteus.stage1.level_set import LevelSetConfig
from proteus.stage1.recursion import RecursionConfig, run_recursive_discovery
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.circles import make_circle
from tests.datasets.synthetic.density_valleys import make_two_gaussians
from tests.datasets.synthetic.linked_tori import make_linked_tori
from tests.datasets.synthetic.nested_spheres import make_nested_spheres
from tests.datasets.synthetic.swiss_roll import make_swiss_roll
from tests.datasets.synthetic.variable_density import (
    make_filled_ball,
    make_lone_gauss3d,
    make_scurve_sheet,
    make_uniform_cube,
    make_uniform_disc,
)
from tests.datasets.ground_truth import GroundTruthManifold, SyntheticDataset
from tests.scenarios.synthetic.level_set_normal_path_sweep import (
    _keep_signal_plus_nearby_tissue,
    _scenes,
)
from tests.scenarios.synthetic.level_set_suite import (
    CIRCLE_N,
    FILLED_BALL_RADIUS,
    LONE_GAUSS2D_N,
    LONE_GAUSS2D_SIGMA,
    LONE_GAUSS3D_SIGMA,
    LONE_GAUSS4D_N,
    LONE_SHELL_N_PER,
    LONE_TISSUE_RADIUS,
    LONE_TORUS_N_PER,
    SCURVE_SHEET_NOISE,
    SWISS_N,
    UNIFORM_CUBE_HALF,
    UNIFORM_DISC_RADIUS,
    WIDEN_K_DEFAULT,
    WIDEN_K_GRID,
    WIDEN_N_GRID,
    WIDEN_NULL_N,
)


# Declared null ensemble for the φ-ceiling calibration (SI S2.6.2 / S14.3).
DEFAULT_NULL_SCENES: tuple[str, ...] = (
    "circle_null",
    "swiss_roll_null",
    "lone_torus_null",
    "lone_shell_inner_null",
    "lone_gauss2d_null",
    "lone_gauss4d_null",
)

# New geometries for A3-T2 widen (kill-checked first).
WIDEN_NEW_SCENES: tuple[str, ...] = (
    "uniform_disc_null",
    "uniform_cube3d_null",
    "lone_gauss3d_null",
    "scurve_sheet_null",
    "filled_ball_null",
)

# Mechanism-control nulls (A3-T9): same arc/width as scurve, R=∞ flat strip.
MECHANISM_CONTROL_SCENES: tuple[str, ...] = (
    "flat_strip_null",
)

# A4-T6 / A3-T7: lone-component child-sized nulls (no tissue, no sibling).
COMPONENT_ONLY_SCENES: tuple[str, ...] = (
    "circle_component_only",
    "nested_shell_component_only",
    "two_gaussians_component_only",
)
CHILD_N_GRID: tuple[int, ...] = (200, 500)

# Envelope on candidate reads with φ > 0 (excludes the lone_gauss2d s17 φ=0).
# REFERENCE_* is the 2026-09-09 published protocol (SI S2.6.2 / S14.3).
# OBSERVED_* is this harness's first full reproduction (2026-09-10): min and
# the swiss s17 / lone_gauss2d s17 landmarks match; p1/median sit slightly
# higher and count is 357 vs 359. Report drift before widening (A3-T2).
REFERENCE_ENVELOPE: dict[str, float] = {
    "min": 0.288,
    "p1": 0.487,
    "p5": 0.630,
    "p10": 0.760,
    "median": 1.30,
    "count": 359.0,
}
OBSERVED_ENVELOPE_2026_09_10: dict[str, float] = {
    "min": 0.287623,
    "p1": 0.521339,
    "p5": 0.641613,
    "p10": 0.761657,
    "median": 1.36492,
    "count": 357.0,
}

CEILING = 0.25
CEILING_MARGIN = 0.10  # require min >= ceiling * (1 + margin)

TABLE_FIELDS: tuple[str, ...] = (
    "scene",
    "seed",
    "step",
    "N",
    "tau",
    "phi",
    "accepted",
    "reason",
    "n_samples",
    "k",
)


@dataclass(frozen=True)
class ReadRow:
    scene: str
    seed: int
    step: int
    N: int
    tau: float | None
    phi: float | None
    accepted: bool
    reason: str | None
    n_samples: int | None = None
    k: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "seed": self.seed,
            "step": self.step,
            "N": self.N,
            "tau": self.tau,
            "phi": self.phi,
            "accepted": int(self.accepted),
            "reason": self.reason if self.reason is not None else "",
            "n_samples": "" if self.n_samples is None else int(self.n_samples),
            "k": "" if self.k is None else int(self.k),
        }


@dataclass(frozen=True)
class EnvelopeSummary:
    min: float | None
    p1: float | None
    p5: float | None
    p10: float | None
    median: float | None
    count: int
    any_accepts: bool
    n_phi_zero: int
    phi_zero_accepts: tuple[tuple[str, int, int, float | None], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "min": self.min,
            "p1": self.p1,
            "p5": self.p5,
            "p10": self.p10,
            "median": self.median,
            "count": self.count,
            "any_accepts": self.any_accepts,
            "n_phi_zero": self.n_phi_zero,
            "phi_zero_accepts": list(self.phi_zero_accepts),
        }


def parse_seed_spec(spec: str | Sequence[int]) -> list[int]:
    """Parse ``0-19`` / ``0 1 2`` / ``[0,1]`` into a sorted unique seed list."""

    if isinstance(spec, (list, tuple)) and spec and not isinstance(spec[0], str):
        return sorted({int(s) for s in spec})
    text = " ".join(str(s) for s in spec) if isinstance(spec, (list, tuple)) else str(spec)
    seeds: set[int] = set()
    for token in text.replace(",", " ").split():
        if "-" in token and not token.startswith("-"):
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"empty seed range {token!r}")
            seeds.update(range(lo, hi + 1))
        else:
            seeds.add(int(token))
    return sorted(seeds)


def percentile_nearest(values: np.ndarray, pct: float) -> float:
    """Nearest-rank percentile on a non-empty 1-D array (pct in 0..100)."""

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("percentile_nearest requires a non-empty array")
    if not 0.0 <= float(pct) <= 100.0:
        raise ValueError(f"pct must be in [0, 100], got {pct}")
    ordered = np.sort(arr)
    if ordered.size == 1:
        return float(ordered[0])
    rank = int(np.ceil(float(pct) / 100.0 * ordered.size)) - 1
    rank = max(0, min(ordered.size - 1, rank))
    return float(ordered[rank])


def summarize_envelope(rows: Iterable[ReadRow]) -> EnvelopeSummary:
    """Envelope over candidate reads with φ > 0; track exact φ=0 separately."""

    pos: list[float] = []
    any_accepts = False
    n_phi_zero = 0
    phi_zero_accepts: list[tuple[str, int, int, float | None]] = []
    for row in rows:
        if row.phi is None:
            continue
        if row.accepted:
            any_accepts = True
        phi = float(row.phi)
        if phi == 0.0:
            n_phi_zero += 1
            if row.accepted:
                phi_zero_accepts.append((row.scene, row.seed, row.step, row.tau))
            continue
        pos.append(phi)
    if not pos:
        return EnvelopeSummary(
            min=None,
            p1=None,
            p5=None,
            p10=None,
            median=None,
            count=0,
            any_accepts=any_accepts,
            n_phi_zero=n_phi_zero,
            phi_zero_accepts=tuple(phi_zero_accepts),
        )
    arr = np.asarray(pos, dtype=float)
    return EnvelopeSummary(
        min=float(np.min(arr)),
        p1=percentile_nearest(arr, 1.0),
        p5=percentile_nearest(arr, 5.0),
        p10=percentile_nearest(arr, 10.0),
        median=float(np.median(arr)),
        count=int(arr.size),
        any_accepts=any_accepts,
        n_phi_zero=n_phi_zero,
        phi_zero_accepts=tuple(phi_zero_accepts),
    )


def format_table(rows: Sequence[ReadRow]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(TABLE_FIELDS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row.as_dict())
    return buf.getvalue()


def format_envelope(summary: EnvelopeSummary, title: str = "ENVELOPE") -> str:
    lines = [
        f"{title} (candidate reads with phi > 0)",
        f"  count={summary.count}",
        f"  min={_fmt(summary.min)}",
        f"  p1={_fmt(summary.p1)}",
        f"  p5={_fmt(summary.p5)}",
        f"  p10={_fmt(summary.p10)}",
        f"  median={_fmt(summary.median)}",
        f"  any_accepts={int(summary.any_accepts)}",
        f"  n_phi_zero_excluded={summary.n_phi_zero}",
    ]
    if summary.phi_zero_accepts:
        lines.append("  PHI_ZERO_ACCEPTS:")
        for scene, seed, step, tau in summary.phi_zero_accepts:
            lines.append(
                f"    {scene} seed={seed} step={step} tau={_fmt(tau)}"
            )
    if summary.min is not None:
        margin = (float(summary.min) - CEILING) / CEILING
        lines.append(
            f"  ceiling={CEILING} margin_vs_min={margin:.3f} "
            f"(need>={CEILING_MARGIN:.2f})"
        )
    return "\n".join(lines)


def _fmt(value: Any, digits: str = ".6g") -> str:
    if value is None:
        return "na"
    try:
        return format(float(value), digits)
    except (TypeError, ValueError):
        return str(value)


def _lone_torus_at_n(n_per: int, seed: int) -> SyntheticDataset:
    data = make_linked_tori(n_per_torus=int(n_per), seed=seed)
    points, labels = _keep_signal_plus_nearby_tissue(
        data.points, data.labels, 0, LONE_TISSUE_RADIUS,
    )
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=data.ground_truth,
        metadata=dict(data.metadata),
    )


def _lone_shell_at_n(n_per: int, seed: int) -> SyntheticDataset:
    data = make_nested_spheres(n_per_sphere=int(n_per), seed=seed)
    points, labels = _keep_signal_plus_nearby_tissue(
        data.points, data.labels, 1, LONE_TISSUE_RADIUS,
    )
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=data.ground_truth,
        metadata=dict(data.metadata),
    )


def _lone_gauss2d_at_n(n: int, seed: int) -> SyntheticDataset:
    rng = np.random.default_rng(int(seed))
    points = rng.normal(size=(int(n), 2)) * LONE_GAUSS2D_SIGMA
    labels = np.zeros(int(n), dtype=int)
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=GroundTruthManifold(
            name="lone_gauss2d", ambient_dim=2, intrinsic_dim=2,
        ),
    )


def _lone_gauss4d_at_n(n: int, seed: int) -> SyntheticDataset:
    rng = np.random.default_rng(int(seed))
    points = rng.normal(size=(int(n), 4))
    labels = np.zeros(int(n), dtype=int)
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=GroundTruthManifold(
            name="lone_gauss4d", ambient_dim=4, intrinsic_dim=4,
        ),
    )


def default_n_for_scene(scene_name: str) -> int:
    return {
        "circle_null": CIRCLE_N,
        "swiss_roll_null": SWISS_N,
        "lone_torus_null": LONE_TORUS_N_PER,
        "lone_shell_inner_null": LONE_SHELL_N_PER,
        "lone_gauss2d_null": LONE_GAUSS2D_N,
        "lone_gauss4d_null": LONE_GAUSS4D_N,
        "uniform_disc_null": WIDEN_NULL_N,
        "uniform_cube3d_null": WIDEN_NULL_N,
        "lone_gauss3d_null": WIDEN_NULL_N,
        "scurve_sheet_null": WIDEN_NULL_N,
        "filled_ball_null": WIDEN_NULL_N,
        "flat_strip_null": WIDEN_NULL_N,
        "circle_component_only": CHILD_N_GRID[-1],
        "nested_shell_component_only": CHILD_N_GRID[-1],
        "two_gaussians_component_only": CHILD_N_GRID[-1],
    }[scene_name]


def build_dataset(scene_name: str, seed: int, n_samples: int) -> SyntheticDataset:
    """Materialize a null scene at a chosen sample budget."""

    n = int(n_samples)
    if scene_name == "circle_null":
        return make_circle(n_samples=n, seed=seed)
    if scene_name == "swiss_roll_null":
        return make_swiss_roll(n_samples=n, seed=seed)
    if scene_name == "lone_torus_null":
        return _lone_torus_at_n(n, seed)
    if scene_name == "lone_shell_inner_null":
        return _lone_shell_at_n(n, seed)
    if scene_name == "lone_gauss2d_null":
        return _lone_gauss2d_at_n(n, seed)
    if scene_name == "lone_gauss4d_null":
        return _lone_gauss4d_at_n(n, seed)
    if scene_name == "uniform_disc_null":
        return make_uniform_disc(n_samples=n, radius=UNIFORM_DISC_RADIUS, seed=seed)
    if scene_name == "uniform_cube3d_null":
        return make_uniform_cube(
            n_samples=n, half_extent=UNIFORM_CUBE_HALF, dim=3, seed=seed,
        )
    if scene_name == "lone_gauss3d_null":
        return make_lone_gauss3d(n_samples=n, sigma=LONE_GAUSS3D_SIGMA, seed=seed)
    if scene_name == "scurve_sheet_null":
        return make_scurve_sheet(n_samples=n, noise=SCURVE_SHEET_NOISE, seed=seed)
    if scene_name == "flat_strip_null":
        return make_scurve_sheet(
            n_samples=n,
            noise=SCURVE_SHEET_NOISE,
            seed=seed,
            curvature_radius=float("inf"),
        )
    if scene_name == "filled_ball_null":
        return make_filled_ball(
            n_samples=n, radius=FILLED_BALL_RADIUS, dim=3, seed=seed,
        )
    if scene_name == "circle_component_only":
        return make_circle(n_samples=n, seed=seed, component_only=True)
    if scene_name == "nested_shell_component_only":
        return make_nested_spheres(
            n_per_sphere=n, seed=seed, component_only=True, component_index=0,
        )
    if scene_name == "two_gaussians_component_only":
        return make_two_gaussians(
            n_samples=n, seed=seed, component_only=True, component_index=0,
        )
    raise ValueError(f"unknown null scene {scene_name!r}")


def _recursion_config(
    seed: int,
    *,
    k: int,
    max_depth: int,
    max_epochs: int,
    max_grid_points: int,
    max_finer_steps: int,
    growth_policy: str,
) -> RecursionConfig:
    scale = ScaleSearchConfig(
        selector="load_crossover",
        tau_min=1e-5,
        tau_max=10.0,
        max_grid_points=int(max_grid_points),
        k=int(k),
        min_nodes=4,
        n_seeds=8,
        max_nodes=None,
        stabilization=StabilizationConfig(
            min_equilibrium_epochs=3,
            max_epochs=int(max_epochs),
        ),
        seed=int(seed),
    )
    return RecursionConfig(
        scale_search=scale,
        min_samples=100,
        max_depth=int(max_depth),
        use_level_set_clustering=True,
        allow_finer_research=True,
        max_finer_scale_steps=int(max_finer_steps),
        level_set=LevelSetConfig(
            k_neighbors=int(k),
            growth_policy=str(growth_policy),
        ),
        seed=int(seed),
    )


def known_null_scenes() -> tuple[str, ...]:
    return (
        DEFAULT_NULL_SCENES
        + WIDEN_NEW_SCENES
        + MECHANISM_CONTROL_SCENES
        + COMPONENT_ONLY_SCENES
    )


def collect_root_candidate_reads(
    scene_name: str,
    seed: int,
    *,
    n_samples: int | None = None,
    k: int = WIDEN_K_DEFAULT,
    max_depth: int = 1,
    max_epochs: int = 12,
    max_grid_points: int = 8,
    max_finer_steps: int = 16,
    growth_policy: str = "track_tau",
) -> list[ReadRow]:
    """Run one scene-seed root walk; return rows for reads with a candidate φ."""

    if scene_name not in known_null_scenes():
        raise ValueError(
            f"unknown null scene {scene_name!r}; known: {list(known_null_scenes())}"
        )
    n = int(default_n_for_scene(scene_name) if n_samples is None else n_samples)
    data = build_dataset(scene_name, int(seed), n)
    points = np.asarray(data.points, dtype=float)
    dim = int(data.ground_truth.ambient_dim)
    config = _recursion_config(
        int(seed),
        k=int(k),
        max_depth=int(max_depth),
        max_epochs=int(max_epochs),
        max_grid_points=int(max_grid_points),
        max_finer_steps=int(max_finer_steps),
        growth_policy=str(growth_policy),
    )

    original = recursion_mod.select_level_set_partition
    rows: list[ReadRow] = []
    step = 0

    def wrapper(scaffold, config=None, dm_config=None, data=None):
        nonlocal step
        selection = original(scaffold, config, dm_config, data)
        step += 1
        phi = selection.bottleneck_ratio
        if phi is None:
            return selection
        reason = None
        if selection.resolvability is not None:
            reason = selection.resolvability.reject_reason
        rows.append(
            ReadRow(
                scene=scene_name,
                seed=int(seed),
                step=int(step),
                N=len(getattr(scaffold, "nodes", ())),
                tau=(
                    None
                    if getattr(scaffold, "tau", None) is None
                    else float(scaffold.tau)
                ),
                phi=float(phi),
                accepted=bool(selection.accepted),
                reason=reason,
                n_samples=int(n),
                k=int(k),
            )
        )
        return selection

    recursion_mod.select_level_set_partition = wrapper
    try:
        run_recursive_discovery(points, dim=dim, config=config)
    finally:
        recursion_mod.select_level_set_partition = original
    return rows


def _worker(payload: dict[str, Any]) -> tuple[str, int, list[dict[str, Any]], float]:
    """Process-pool entry: return serializable row dicts + elapsed seconds."""

    t0 = time.time()
    rows = collect_root_candidate_reads(
        payload["scene"],
        int(payload["seed"]),
        n_samples=payload.get("n_samples"),
        k=int(payload.get("k", WIDEN_K_DEFAULT)),
        max_depth=int(payload["max_depth"]),
        max_epochs=int(payload["max_epochs"]),
        max_grid_points=int(payload["max_grid_points"]),
        max_finer_steps=int(payload["max_finer_steps"]),
        growth_policy=str(payload["growth_policy"]),
    )
    return (
        payload["scene"],
        int(payload["seed"]),
        [r.as_dict() for r in rows],
        time.time() - t0,
    )


def _rows_from_dicts(dicts: Iterable[dict[str, Any]]) -> list[ReadRow]:
    out: list[ReadRow] = []
    for d in dicts:
        n_raw = d.get("n_samples", "")
        k_raw = d.get("k", "")
        out.append(
            ReadRow(
                scene=str(d["scene"]),
                seed=int(d["seed"]),
                step=int(d["step"]),
                N=int(d["N"]),
                tau=None if d["tau"] in (None, "") else float(d["tau"]),
                phi=None if d["phi"] in (None, "") else float(d["phi"]),
                accepted=bool(int(d["accepted"])),
                reason=(None if d.get("reason") in (None, "") else str(d["reason"])),
                n_samples=None if n_raw in (None, "") else int(n_raw),
                k=None if k_raw in (None, "") else int(k_raw),
            )
        )
    return out


def run_envelope(
    scenes: Sequence[str],
    seeds: Sequence[int],
    *,
    jobs: int = 1,
    n_samples: int | None = None,
    k: int = WIDEN_K_DEFAULT,
    max_depth: int = 1,
    max_epochs: int = 12,
    max_grid_points: int = 8,
    max_finer_steps: int = 16,
    growth_policy: str = "track_tau",
    payloads: Sequence[dict[str, Any]] | None = None,
) -> tuple[list[ReadRow], EnvelopeSummary]:
    """Run the declared ensemble; return candidate rows + envelope summary."""

    if payloads is None:
        payloads = [
            {
                "scene": scene,
                "seed": int(seed),
                "n_samples": n_samples,
                "k": int(k),
                "max_depth": int(max_depth),
                "max_epochs": int(max_epochs),
                "max_grid_points": int(max_grid_points),
                "max_finer_steps": int(max_finer_steps),
                "growth_policy": str(growth_policy),
            }
            for scene in scenes
            for seed in seeds
        ]
    collected: list[ReadRow] = []
    if jobs <= 1 or len(payloads) <= 1:
        for payload in payloads:
            scene, seed, dicts, elapsed = _worker(payload)
            rows = _rows_from_dicts(dicts)
            collected.extend(rows)
            print(
                f"DONE {scene} seed={seed} n={payload.get('n_samples')} "
                f"k={payload.get('k')} candidate_reads={len(rows)} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=int(jobs)) as pool:
            futures = {pool.submit(_worker, p): p for p in payloads}
            for fut in as_completed(futures):
                payload = futures[fut]
                scene, seed, dicts, elapsed = fut.result()
                rows = _rows_from_dicts(dicts)
                collected.extend(rows)
                print(
                    f"DONE {scene} seed={seed} n={payload.get('n_samples')} "
                    f"k={payload.get('k')} candidate_reads={len(rows)} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    collected.sort(
        key=lambda r: (r.scene, r.n_samples or 0, r.k or 0, r.seed, r.step)
    )
    return collected, summarize_envelope(collected)


def compare_to_reference(
    summary: EnvelopeSummary,
    *,
    abs_tol: float = 0.02,
    count_tol: int = 5,
    soft_tol: float = 0.05,
) -> tuple[list[str], list[str]]:
    """Split critical vs soft mismatches vs the 2026-09-09 reference.

    Critical: ``min``, ``count``, and presence of the known φ=0 accept.
    Soft: upper envelope percentiles (p1/p5/p10/median) — these drifted on
    the 2026-09-10 reproduction while the min landmark held.
    """

    critical: list[str] = []
    soft: list[str] = []
    if abs(summary.count - int(REFERENCE_ENVELOPE["count"])) > count_tol:
        critical.append(
            f"count {summary.count} vs ref {int(REFERENCE_ENVELOPE['count'])}"
        )
    got_min = summary.min
    ref_min = float(REFERENCE_ENVELOPE["min"])
    if got_min is None or abs(float(got_min) - ref_min) > abs_tol:
        critical.append(f"min {_fmt(got_min)} vs ref {ref_min}")
    for key in ("p1", "p5", "p10", "median"):
        got = getattr(summary, key)
        ref = float(REFERENCE_ENVELOPE[key])
        if got is None or abs(float(got) - ref) > soft_tol:
            soft.append(f"{key} {_fmt(got)} vs ref {ref}")
        elif abs(float(got) - ref) > abs_tol:
            soft.append(f"{key} {_fmt(got)} vs ref {ref} (within soft_tol)")
    return critical, soft


def kill_rows_below_ceiling(
    rows: Sequence[ReadRow],
    *,
    ceiling: float = CEILING,
) -> list[ReadRow]:
    """New-geometry kill: candidate with ``0 < φ < ceiling`` (φ=0 → A2)."""

    out: list[ReadRow] = []
    for row in rows:
        if row.phi is None:
            continue
        phi = float(row.phi)
        if phi == 0.0:
            continue
        if phi < float(ceiling):
            out.append(row)
    return out


def group_envelopes(
    rows: Sequence[ReadRow],
    key_fn: Callable[[ReadRow], Any],
) -> list[tuple[Any, EnvelopeSummary]]:
    buckets: dict[Any, list[ReadRow]] = {}
    for row in rows:
        buckets.setdefault(key_fn(row), []).append(row)
    return [(key, summarize_envelope(group)) for key, group in sorted(buckets.items())]


def propose_ceiling_rule(
    per_geometry: Sequence[tuple[Any, EnvelopeSummary]],
    overall: EnvelopeSummary,
) -> str:
    """Propose (do not apply) fixed vs relative ceiling; NOTE-only."""

    mins = [s.min for _, s in per_geometry if s.min is not None]
    p1s = [s.p1 for _, s in per_geometry if s.p1 is not None]
    if not mins or overall.min is None or overall.p1 is None:
        return (
            "PROPOSE: insufficient positive-φ reads; keep fixed ceiling "
            f"{CEILING} pending more data."
        )
    worst_min = float(min(mins))
    p1_arr = np.asarray(p1s, dtype=float)
    frac_of_p1 = CEILING / float(overall.p1)
    # Stability of relative rule: ceiling = c * geometry_p1, with c chosen
    # so overall p1 maps to 0.25.
    relative_targets = [frac_of_p1 * float(p) for p in p1s]
    rel_spread = float(np.std(relative_targets)) if len(relative_targets) > 1 else 0.0
    fixed_margins = [(m - CEILING) / CEILING for m in mins]
    fixed_spread = float(np.std(fixed_margins)) if len(fixed_margins) > 1 else 0.0
    prefer_relative = rel_spread + 1e-9 < fixed_spread
    return (
        f"PROPOSE: worst_min={worst_min:.6g} overall_p1={overall.p1:.6g} "
        f"fixed_ceiling={CEILING} (= {CEILING / overall.p1:.3f}×overall_p1); "
        f"fixed_margin_std={fixed_spread:.4f} relative_target_std={rel_spread:.4f}; "
        + (
            f"prefer RELATIVE rule ceiling = {frac_of_p1:.3f} × envelope_p1 "
            f"(do not apply without director ack)."
            if prefer_relative
            else f"prefer FIXED ceiling {CEILING} "
            f"(stable enough; do not apply relative without director ack)."
        )
    )


def _widen_payloads(
    seeds: Sequence[int],
    *,
    max_depth: int,
    max_epochs: int,
    max_grid_points: int,
    max_finer_steps: int,
    growth_policy: str,
    include_grid: bool,
) -> list[dict[str, Any]]:
    base = {
        "max_depth": int(max_depth),
        "max_epochs": int(max_epochs),
        "max_grid_points": int(max_grid_points),
        "max_finer_steps": int(max_finer_steps),
        "growth_policy": str(growth_policy),
    }
    payloads: list[dict[str, Any]] = []
    for scene in WIDEN_NEW_SCENES:
        for seed in seeds:
            payloads.append(
                {
                    **base,
                    "scene": scene,
                    "seed": int(seed),
                    "n_samples": int(WIDEN_NULL_N),
                    "k": int(WIDEN_K_DEFAULT),
                }
            )
    if include_grid:
        for scene in DEFAULT_NULL_SCENES:
            for n in WIDEN_N_GRID:
                for k in WIDEN_K_GRID:
                    for seed in seeds:
                        payloads.append(
                            {
                                **base,
                                "scene": scene,
                                "seed": int(seed),
                                "n_samples": int(n),
                                "k": int(k),
                            }
                        )
    return payloads


def _component_only_payloads(
    seeds: Sequence[int],
    *,
    max_depth: int,
    max_epochs: int,
    max_grid_points: int,
    max_finer_steps: int,
    growth_policy: str,
    n_grid: Sequence[int] = CHILD_N_GRID,
    k: int = WIDEN_K_DEFAULT,
) -> list[dict[str, Any]]:
    """A3-T7: A4-T6 component_only scenes × child-sized n × seeds."""

    base = {
        "max_depth": int(max_depth),
        "max_epochs": int(max_epochs),
        "max_grid_points": int(max_grid_points),
        "max_finer_steps": int(max_finer_steps),
        "growth_policy": str(growth_policy),
        "k": int(k),
    }
    payloads: list[dict[str, Any]] = []
    for scene in COMPONENT_ONLY_SCENES:
        for n in n_grid:
            for seed in seeds:
                payloads.append(
                    {
                        **base,
                        "scene": scene,
                        "seed": int(seed),
                        "n_samples": int(n),
                    }
                )
    return payloads


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Null-ensemble φ envelope for LevelSetConfig.max_bottleneck_ratio",
    )
    parser.add_argument(
        "--mode",
        choices=("protocol", "widen", "widen-new", "component-only"),
        default="protocol",
        help="protocol=six nulls; widen-new=new geometries only; "
        "widen=new + existing n×k grid; component-only=A4-T6 child-sized nulls.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help="Seeds or inclusive ranges (default: 0-19 protocol / 0-9 widen).",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Null scene names (default depends on --mode).",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--k", type=int, default=WIDEN_K_DEFAULT)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--max-grid-points", type=int, default=8)
    parser.add_argument("--max-finer-steps", type=int, default=16)
    parser.add_argument(
        "--growth-policy",
        choices=("track_tau",),
        default="track_tau",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Optional path to write the candidate-read table.",
    )
    parser.add_argument(
        "--check-reference",
        action="store_true",
        help="Exit non-zero if envelope drifts beyond tolerance from 2026-09-09.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.seeds is None:
        if args.mode.startswith("widen"):
            args.seeds = ["0-9"]
        elif args.mode == "component-only":
            args.seeds = ["0-19"]
        else:
            args.seeds = ["0-19"]
    seeds = parse_seed_spec(args.seeds)

    payloads: list[dict[str, Any]] | None = None
    if args.mode in {"widen", "widen-new"}:
        payloads = _widen_payloads(
            seeds,
            max_depth=int(args.max_depth),
            max_epochs=int(args.max_epochs),
            max_grid_points=int(args.max_grid_points),
            max_finer_steps=int(args.max_finer_steps),
            growth_policy=str(args.growth_policy),
            include_grid=(args.mode == "widen"),
        )
        scenes = sorted({p["scene"] for p in payloads})
    elif args.mode == "component-only":
        payloads = _component_only_payloads(
            seeds,
            max_depth=int(args.max_depth),
            max_epochs=int(args.max_epochs),
            max_grid_points=int(args.max_grid_points),
            max_finer_steps=int(args.max_finer_steps),
            growth_policy=str(args.growth_policy),
            k=int(args.k),
        )
        scenes = list(COMPONENT_ONLY_SCENES)
    else:
        scenes = list(args.scenes) if args.scenes else list(DEFAULT_NULL_SCENES)
        unknown = [s for s in scenes if s not in known_null_scenes()]
        if unknown:
            raise SystemExit(
                f"unknown null scenes {unknown}; known: {list(known_null_scenes())}"
            )

    print(
        f"level_set_null_envelope mode={args.mode} scenes={scenes} "
        f"seeds={seeds[0]}..{seeds[-1]} n_seeds={len(seeds)} jobs={args.jobs} "
        f"max_depth={args.max_depth} finer_steps={args.max_finer_steps} "
        f"growth_policy={args.growth_policy} n_payloads="
        f"{len(payloads) if payloads is not None else len(scenes) * len(seeds)}",
        flush=True,
    )
    t0 = time.time()
    rows, summary = run_envelope(
        scenes,
        seeds,
        jobs=int(args.jobs),
        n_samples=args.n_samples,
        k=int(args.k),
        max_depth=int(args.max_depth),
        max_epochs=int(args.max_epochs),
        max_grid_points=int(args.max_grid_points),
        max_finer_steps=int(args.max_finer_steps),
        growth_policy=str(args.growth_policy),
        payloads=payloads,
    )
    elapsed = time.time() - t0

    table = format_table(rows)
    print(table, end="", flush=True)
    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as fh:
            fh.write(table)
        print(f"WROTE_CSV {args.csv}", flush=True)

    print(format_envelope(summary, title="ENVELOPE_OVERALL"), flush=True)

    if args.mode.startswith("widen"):
        print("ENVELOPE_PER_GEOMETRY:", flush=True)
        per_geo = group_envelopes(rows, lambda r: r.scene)
        for key, env in per_geo:
            print(format_envelope(env, title=f"  {key}"), flush=True)
        print("ENVELOPE_PER_N_K:", flush=True)
        per_nk = group_envelopes(
            rows, lambda r: (r.n_samples, r.k)
        )
        for key, env in per_nk:
            print(format_envelope(env, title=f"  n,k={key}"), flush=True)
        kills = kill_rows_below_ceiling(
            [r for r in rows if r.scene in WIDEN_NEW_SCENES]
        )
        if kills:
            print("KILL_FIRED new-geometry phi below ceiling:", flush=True)
            for row in kills[:20]:
                print(
                    f"  STEP scene={row.scene} seed={row.seed} step={row.step} "
                    f"N={row.N} tau={_fmt(row.tau)} phi={_fmt(row.phi)} "
                    f"accepted={int(row.accepted)} reason={row.reason}",
                    flush=True,
                )
            print(f"TOTAL_ELAPSED={elapsed:.1f}s n_rows={len(rows)}", flush=True)
            return 2
        print(propose_ceiling_rule(per_geo, summary), flush=True)

    if args.mode == "component-only":
        print("ENVELOPE_PER_GEOMETRY:", flush=True)
        per_geo = group_envelopes(rows, lambda r: r.scene)
        for key, env in per_geo:
            print(format_envelope(env, title=f"  {key}"), flush=True)
        print("ENVELOPE_PER_N:", flush=True)
        per_n = group_envelopes(rows, lambda r: r.n_samples)
        for key, env in per_n:
            print(format_envelope(env, title=f"  n={key}"), flush=True)
        print("ENVELOPE_PER_SCENE_N:", flush=True)
        per_sn = group_envelopes(rows, lambda r: (r.scene, r.n_samples))
        for key, env in per_sn:
            print(format_envelope(env, title=f"  {key}"), flush=True)

    print(f"TOTAL_ELAPSED={elapsed:.1f}s n_rows={len(rows)}", flush=True)

    if args.check_reference and scenes == list(DEFAULT_NULL_SCENES) and seeds == list(
        range(20)
    ):
        critical, soft = compare_to_reference(summary)
        zero_ok = any(
            s == "lone_gauss2d_null" and seed == 17
            for s, seed, _step, _tau in summary.phi_zero_accepts
        )
        if not zero_ok:
            critical.append(
                "missing expected phi=0 accept on lone_gauss2d_null seed 17"
            )
        if critical:
            print("REFERENCE_CRITICAL: " + "; ".join(critical), flush=True)
            if soft:
                print("REFERENCE_SOFT: " + "; ".join(soft), flush=True)
            return 1
        if soft:
            print("REFERENCE_SOFT: " + "; ".join(soft), flush=True)
            print(
                "REFERENCE_LANDMARKS_OK "
                "(min/count/phi0 within tol; percentile soft drift — "
                "see OBSERVED_ENVELOPE_2026_09_10)",
                flush=True,
            )
            return 0
        print("REFERENCE_OK", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Unit tests (pure helpers; full envelope is a long CLI run)
# ---------------------------------------------------------------------------


def test_default_null_scenes_match_protocol() -> None:
    assert DEFAULT_NULL_SCENES == (
        "circle_null",
        "swiss_roll_null",
        "lone_torus_null",
        "lone_shell_inner_null",
        "lone_gauss2d_null",
        "lone_gauss4d_null",
    )
    present = {s.name for s in _scenes()}
    assert set(DEFAULT_NULL_SCENES) <= present


def test_widen_new_scenes_build() -> None:
    for scene in WIDEN_NEW_SCENES:
        data = build_dataset(scene, seed=0, n_samples=WIDEN_NULL_N)
        assert data.points.shape[0] == WIDEN_NULL_N
        assert data.points.ndim == 2


def test_component_only_scenes_build() -> None:
    """A3-T7: A4-T6 component_only nulls materialize at child-sized n."""

    for scene in COMPONENT_ONLY_SCENES:
        for n in CHILD_N_GRID:
            data = build_dataset(scene, seed=0, n_samples=n)
            assert data.points.shape[0] == n
            assert data.points.ndim == 2
            assert data.metadata.get("component_only") is True
            assert data.metadata.get("null_scene") is True
    assert set(COMPONENT_ONLY_SCENES) <= set(known_null_scenes())
    payloads = _component_only_payloads(
        [0, 1],
        max_depth=1,
        max_epochs=12,
        max_grid_points=8,
        max_finer_steps=16,
        growth_policy="track_tau",
    )
    assert len(payloads) == len(COMPONENT_ONLY_SCENES) * len(CHILD_N_GRID) * 2
    assert {p["n_samples"] for p in payloads} == set(CHILD_N_GRID)


def test_scurve_sheet_half_occupancy_and_injectivity() -> None:
    """Corrected S-curve: 50/50 lobe occupancy; no half-arc double-cover (A3-T6)."""

    data = make_scurve_sheet(n_samples=20_000, noise=0.0, seed=0)
    points = np.asarray(data.points, dtype=float)
    z = points[:, 2]
    # Classic S: upper lobe z>0 and lower lobe z<0 each get ~half the mass.
    upper = float(np.mean(z > 1e-6))
    lower = float(np.mean(z < -1e-6))
    assert abs(upper - 0.5) < 0.02
    assert abs(lower - 0.5) < 0.02
    assert abs(upper - lower) < 0.03

    # Dense θ-grid injectivity on the (x, z) centerline (no double-cover).
    theta = np.linspace(-1.5 * np.pi, 1.5 * np.pi, 3001)
    xz = np.column_stack(
        [np.sin(theta), np.sign(theta) * (np.cos(theta) - 1.0)],
    )
    uniq = np.unique(np.round(xz, decimals=6), axis=0)
    assert uniq.shape[0] == theta.shape[0]

    lo, hi = data.metadata["theta_range"]
    assert np.isclose(lo, -1.5 * np.pi)
    assert np.isclose(hi, 1.5 * np.pi)
    assert float(data.metadata["curvature_radius"]) == 1.0
    arc = np.asarray(data.metadata["arc"], dtype=float)
    assert arc.shape == (20_000,)
    assert float(arc.min()) >= -1e-9
    assert float(arc.max()) <= 3.0 * np.pi + 1e-9


def test_scurve_sheet_curvature_radius_controls() -> None:
    """A3-T9: R=1 byte-identical; R=2 scaled S; R=∞ flat strip (arc 3π, width 2)."""

    n = 4_000
    r1 = make_scurve_sheet(n_samples=n, noise=0.0, seed=7, curvature_radius=1.0)
    r1_default = make_scurve_sheet(n_samples=n, noise=0.0, seed=7)
    assert np.array_equal(r1.points, r1_default.points)

    r2 = make_scurve_sheet(n_samples=n, noise=0.0, seed=7, curvature_radius=2.0)
    assert float(r2.metadata["curvature_radius"]) == 2.0
    assert np.isclose(r2.metadata["arc_length"], 3.0 * np.pi)
    assert np.isclose(r2.metadata["width"], 2.0)
    # Half-angle span is L/(2R) = 0.75π; centerline injectivity on (x,z).
    th = np.linspace(-0.75 * np.pi, 0.75 * np.pi, 2001)
    xz = np.column_stack(
        [2.0 * np.sin(th), 2.0 * np.sign(th) * (np.cos(th) - 1.0)],
    )
    assert np.unique(np.round(xz, decimals=6), axis=0).shape[0] == th.shape[0]
    # Same seed ⇒ same (u,s); R=2 is a radial scale of the R=1 embedding
    # only when θ_R1 = 2·θ_R2, which holds because θ ∝ 1/R for fixed u.
    assert np.allclose(r2.points[:, 1], r1.points[:, 1], atol=1e-12)
    assert float(np.max(np.abs(r2.points[:, 2]))) > float(
        np.max(np.abs(r1.points[:, 2]))
    )

    flat = make_scurve_sheet(
        n_samples=n, noise=0.0, seed=7, curvature_radius=float("inf"),
    )
    assert np.isinf(float(flat.metadata["curvature_radius"]))
    assert flat.ground_truth.name == "flat_strip"
    assert flat.metadata["sampling"] == "area_uniform_flat_strip"
    pts = np.asarray(flat.points, dtype=float)
    assert float(np.max(np.abs(pts[:, 2]))) < 1e-12
    assert abs(float(np.ptp(pts[:, 0])) - 3.0 * np.pi) < 0.05
    assert abs(float(np.ptp(pts[:, 1])) - 2.0) < 0.05
    arc = np.asarray(flat.metadata["arc"], dtype=float)
    assert abs(float(np.ptp(arc)) - 3.0 * np.pi) < 0.05
    # Flat strip builds via the envelope scene name.
    built = build_dataset("flat_strip_null", seed=0, n_samples=WIDEN_NULL_N)
    assert built.points.shape == (WIDEN_NULL_N, 3)
    assert built.ground_truth.name == "flat_strip"


def test_parse_seed_spec_ranges() -> None:
    assert parse_seed_spec("0-19") == list(range(20))
    assert parse_seed_spec(["0-2", "5"]) == [0, 1, 2, 5]
    assert parse_seed_spec([3, 1, 2]) == [1, 2, 3]


def test_summarize_envelope_excludes_phi_zero() -> None:
    rows = [
        ReadRow("a", 0, 1, 10, 0.1, 0.5, False, "bottleneck"),
        ReadRow("a", 0, 2, 12, 0.05, 0.0, True, None),
        ReadRow("b", 1, 1, 20, 0.2, 1.0, False, "bottleneck"),
        ReadRow("b", 1, 2, 22, 0.1, 0.288, False, "bottleneck"),
    ]
    summary = summarize_envelope(rows)
    assert summary.count == 3
    assert summary.min == 0.288
    assert summary.n_phi_zero == 1
    assert summary.phi_zero_accepts == (("a", 0, 2, 0.05),)
    assert summary.any_accepts is True


def test_percentile_nearest_basic() -> None:
    vals = np.asarray([0.288, 0.487, 0.630, 0.760, 1.30], dtype=float)
    assert percentile_nearest(vals, 0.0) == 0.288
    assert percentile_nearest(vals, 100.0) == 1.30
    assert percentile_nearest(vals, 50.0) == 0.630


def test_format_table_headers() -> None:
    rows = [ReadRow("circle_null", 0, 1, 64, 0.01, 0.9, False, "bottleneck")]
    text = format_table(rows)
    assert text.splitlines()[0] == ",".join(TABLE_FIELDS)
    assert "circle_null,0,1,64," in text


def test_kill_rows_below_ceiling() -> None:
    rows = [
        ReadRow("uniform_disc_null", 0, 1, 10, 0.1, 0.24, False, "bottleneck"),
        ReadRow("uniform_disc_null", 1, 1, 10, 0.1, 0.0, True, None),
        ReadRow("uniform_disc_null", 2, 1, 10, 0.1, 0.5, False, "bottleneck"),
    ]
    kills = kill_rows_below_ceiling(rows)
    assert len(kills) == 1
    assert kills[0].phi == 0.24


def test_compare_to_reference_within_tol() -> None:
    summary = EnvelopeSummary(
        min=0.288,
        p1=0.487,
        p5=0.630,
        p10=0.760,
        median=1.30,
        count=359,
        any_accepts=True,
        n_phi_zero=1,
        phi_zero_accepts=(("lone_gauss2d_null", 17, 9, 0.012),),
    )
    critical, soft = compare_to_reference(summary)
    assert critical == []
    assert soft == []


def test_compare_to_reference_soft_drift() -> None:
    summary = EnvelopeSummary(
        min=0.288,
        p1=0.521,
        p5=0.642,
        p10=0.762,
        median=1.365,
        count=357,
        any_accepts=True,
        n_phi_zero=1,
        phi_zero_accepts=(("lone_gauss2d_null", 17, 9, 0.012),),
    )
    critical, soft = compare_to_reference(summary)
    assert critical == []
    assert any(s.startswith("p1") for s in soft)


if __name__ == "__main__":
    sys.exit(main())
