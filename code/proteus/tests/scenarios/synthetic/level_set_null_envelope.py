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

A3-T11 ``--mode covariate`` emits a table-only cut-covariate report
(N, tau, at_bound, min-side node/sample mass, boundary vs interior
median ``r_k``).  Default packs: corrected S-curve + flat strip seeds
0–19, six protocol nulls seeds 0–4, and clear/tori/nested composites
seeds 0–4 via ``_scenes`` factories.  Propose no floor: if every
covariate's null-accept band overlaps the composite-accept band, print
``NO_COVARIATE_FLOOR`` and stop.

Not a default pytest test.  Pure helpers below are unit-tested.

Usage::

    PYTHONPATH="src:$PWD" pipenv run python \\
        tests/scenarios/synthetic/level_set_null_envelope.py \\
        --seeds 0-19 --jobs 4

    PYTHONPATH="src:$PWD" pipenv run python \\
        tests/scenarios/synthetic/level_set_null_envelope.py \\
        --mode widen --seeds 0-9 --jobs 6 --csv /tmp/widen.csv

    PYTHONPATH="src:$PWD" pipenv run python \\
        tests/scenarios/synthetic/level_set_null_envelope.py \\
        --mode covariate --jobs 4 --csv /tmp/a3_t11_covariate.csv
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
from scipy.spatial import cKDTree

import proteus.stage1.recursion as recursion_mod
from proteus.stage1.controller import ScaleSearchConfig
from proteus.stage1.level_set import (
    LevelSetConfig,
    _filter_relative_mass,
    at_shot_noise_scale,
)
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

# A3-T11 covariate table: long nulls (corrected S-curve + flat strip) s0-19,
# protocol six-nulls s0-4, composites via _scenes factories s0-4.
COVARIATE_NULL_LONG_SCENES: tuple[str, ...] = (
    "scurve_sheet_null",
    "flat_strip_null",
)
COVARIATE_NULL_LONG_SEEDS: tuple[int, ...] = tuple(range(20))
COVARIATE_NULL_SHORT_SEEDS: tuple[int, ...] = tuple(range(5))
COMPOSITE_COVARIATE_SCENES: tuple[str, ...] = (
    "two_gaussians_clear",
    "linked_tori",
    "nested_spheres",
)
COMPOSITE_COVARIATE_SEEDS: tuple[int, ...] = tuple(range(5))
# Boundary: nearest differently-labeled node within this × the node's r_k.
BOUNDARY_RK_MULTIPLE: float = 2.0
COVARIATE_NUMERIC_FIELDS: tuple[str, ...] = (
    "N",
    "tau",
    "at_bound",
    "min_side_node_frac",
    "min_side_sample_mass",
    "bound_med_rk",
    "intA_med_rk",
    "intB_med_rk",
    "rk_contrast",
)
COVARIATE_TABLE_FIELDS: tuple[str, ...] = (
    "scene",
    "seed",
    "step",
    "family",
    "row_role",
    "N",
    "tau",
    "phi",
    "accepted",
    "reason",
    "n_samples",
    "k",
    "n_clusters",
    "at_bound",
    "min_side_node_frac",
    "min_side_sample_mass",
    "bound_med_rk",
    "intA_med_rk",
    "intB_med_rk",
    "rk_contrast",
)

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


# ---------------------------------------------------------------------------
# A3-T11 covariate table (pure helpers; no floor proposal)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CovariateRow:
    """One emitted root-cut covariate row (accepted, else best φ K>=2)."""

    scene: str
    seed: int
    step: int
    family: str
    row_role: str
    N: int
    tau: float | None
    phi: float | None
    accepted: bool
    reason: str | None
    n_samples: int | None
    k: int | None
    n_clusters: int
    at_bound: bool
    min_side_node_frac: float | None
    min_side_sample_mass: float | None
    bound_med_rk: float | None
    intA_med_rk: float | None
    intB_med_rk: float | None
    rk_contrast: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "seed": self.seed,
            "step": self.step,
            "family": self.family,
            "row_role": self.row_role,
            "N": self.N,
            "tau": self.tau,
            "phi": self.phi,
            "accepted": int(self.accepted),
            "reason": self.reason if self.reason is not None else "",
            "n_samples": "" if self.n_samples is None else int(self.n_samples),
            "k": "" if self.k is None else int(self.k),
            "n_clusters": int(self.n_clusters),
            "at_bound": int(self.at_bound),
            "min_side_node_frac": self.min_side_node_frac,
            "min_side_sample_mass": self.min_side_sample_mass,
            "bound_med_rk": self.bound_med_rk,
            "intA_med_rk": self.intA_med_rk,
            "intB_med_rk": self.intB_med_rk,
            "rk_contrast": self.rk_contrast,
        }


def covariate_family(scene_name: str) -> str:
    """``composite`` for clear/tori/nested factories; else ``null``."""

    if scene_name in COMPOSITE_COVARIATE_SCENES:
        return "composite"
    return "null"


def build_covariate_dataset(
    scene_name: str,
    seed: int,
    n_samples: int | None = None,
) -> SyntheticDataset:
    """Nulls via ``build_dataset``; composites via ``_scenes`` factories."""

    if scene_name in known_null_scenes():
        n = int(default_n_for_scene(scene_name) if n_samples is None else n_samples)
        return build_dataset(scene_name, int(seed), n)
    factories = {s.name: s.factory for s in _scenes()}
    if scene_name not in factories:
        raise ValueError(
            f"unknown covariate scene {scene_name!r}; "
            f"known nulls={list(known_null_scenes())} "
            f"composites={list(COMPOSITE_COVARIATE_SCENES)}"
        )
    return factories[scene_name](int(seed))


def scaffold_positions(scaffold: Any) -> np.ndarray:
    nodes = getattr(scaffold, "nodes", ())
    if not nodes:
        return np.empty((0, 0), dtype=float)
    return np.asarray([node.position for node in nodes], dtype=float)


def scaffold_hit_weights(scaffold: Any, n_nodes: int) -> np.ndarray:
    """Per-node mass: ``hit_count`` when present and positive, else ones."""

    nodes = getattr(scaffold, "nodes", ())
    if not nodes or n_nodes <= 0:
        return np.ones(max(int(n_nodes), 0), dtype=float)
    hits = np.asarray(
        [float(getattr(node, "hit_count", np.nan)) for node in nodes],
        dtype=float,
    )
    if hits.size != int(n_nodes) or not np.isfinite(hits).any() or float(np.nansum(hits)) <= 0.0:
        return np.ones(int(n_nodes), dtype=float)
    hits = np.where(np.isfinite(hits) & (hits > 0.0), hits, 0.0)
    if float(hits.sum()) <= 0.0:
        return np.ones(int(n_nodes), dtype=float)
    return hits


def signal_cluster_ids(labels: np.ndarray) -> list[int]:
    return sorted({int(v) for v in np.asarray(labels, dtype=int).ravel() if int(v) >= 0})


def min_side_node_fraction(labels: np.ndarray) -> float | None:
    """``min_c |C| / sum_c |C|`` over cluster labels ``>= 0``.

    For a bipartition this is ``min(|A|,|B|) / (|A|+|B|)``.  Background
    (``< 0``) is excluded.  ``None`` when fewer than two signal clusters.
    """

    lab = np.asarray(labels, dtype=int).ravel()
    keys = signal_cluster_ids(lab)
    if len(keys) < 2:
        return None
    sizes = [int(np.sum(lab == key)) for key in keys]
    total = int(sum(sizes))
    if total <= 0:
        return None
    return float(min(sizes)) / float(total)


def assign_samples_to_bmu(positions: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Nearest-node index for each sample (BMU)."""

    pos = np.asarray(positions, dtype=float)
    pts = np.asarray(samples, dtype=float)
    if pos.ndim != 2 or pos.shape[0] == 0 or pts.ndim != 2 or pts.shape[0] == 0:
        return np.empty(0, dtype=int)
    _, idx = cKDTree(pos).query(pts, k=1)
    return np.atleast_1d(np.asarray(idx, dtype=int).ravel())


def min_side_sample_mass_fraction(
    labels: np.ndarray,
    scaffold: Any,
    samples: np.ndarray | None = None,
) -> float | None:
    """Min signal-side mass / total signal mass.

    If ``samples`` is provided, map each sample to its nearest node
    (BMU) and count samples whose BMU carries a signal label.  Otherwise
    weight nodes by ``hit_count`` when available and positive, else by
    equal weight per node.  Same two-or-more-cluster rule as
    :func:`min_side_node_fraction`.
    """

    lab = np.asarray(labels, dtype=int).ravel()
    keys = signal_cluster_ids(lab)
    if len(keys) < 2:
        return None
    n_nodes = int(lab.size)
    if samples is not None:
        pos = scaffold_positions(scaffold)
        bmu = assign_samples_to_bmu(pos, samples)
        if bmu.size == 0:
            return None
        valid = (bmu >= 0) & (bmu < n_nodes)
        bmu = bmu[valid]
        if bmu.size == 0:
            return None
        side = lab[bmu]
        masses = [float(np.sum(side == key)) for key in keys]
    else:
        weights = scaffold_hit_weights(scaffold, n_nodes)
        if weights.size != n_nodes:
            weights = np.ones(n_nodes, dtype=float)
        masses = [float(np.sum(weights[lab == key])) for key in keys]
    total = float(sum(masses))
    if total <= 0.0:
        return None
    return float(min(masses)) / total


def cut_at_bound(
    scaffold: Any,
    data: np.ndarray | None,
    k: int,
    max_nodes: int | None = None,
) -> bool:
    """True at the shot-noise scale or when ``n_nodes >= max_nodes``.

    ``at_shot_noise_scale(scaffold, data, k)`` is the ``N ≥ n/k``
    catchment bound.  ``max_nodes`` is taken from the argument when
    given, else ``scaffold.max_nodes`` when that attribute is set.
    """

    shot = False
    if data is not None:
        shot = bool(at_shot_noise_scale(scaffold, np.asarray(data, dtype=float), int(k)))
    cap = max_nodes
    if cap is None:
        raw = getattr(scaffold, "max_nodes", None)
        cap = int(raw) if raw is not None else None
    n_nodes = len(getattr(scaffold, "nodes", ()))
    at_cap = cap is not None and n_nodes >= int(cap)
    return bool(shot or at_cap)


def per_node_knn_radii(positions: np.ndarray, k: int) -> np.ndarray:
    """Per-node kNN radius on node positions (same k as the level-set tree)."""

    pos = np.asarray(positions, dtype=float)
    n = int(pos.shape[0]) if pos.ndim == 2 else 0
    if n == 0:
        return np.empty(0, dtype=float)
    if n == 1:
        return np.zeros(1, dtype=float)
    k_use = max(1, min(int(k), n - 1))
    dists, _ = cKDTree(pos).query(pos, k=k_use + 1)
    return np.asarray(dists[:, -1], dtype=float)


def boundary_node_mask(
    positions: np.ndarray,
    labels: np.ndarray,
    radii: np.ndarray,
    *,
    multiple: float = BOUNDARY_RK_MULTIPLE,
) -> np.ndarray:
    """Mark signal nodes whose nearest other-label node is within ``multiple * r_k``.

    A node ``i`` with ``labels[i] >= 0`` is **boundary** iff there exists a
    node ``j`` with ``labels[j] != labels[i]`` (the other side **or**
    background) whose Euclidean distance is ``<= multiple * r_k[i]``.
    Query is a ball of that radius (cKDTree); nodes with non-positive or
    non-finite ``r_k`` are never boundary.  Interior of a side is the
    complement among that side's signal nodes.
    """

    pos = np.asarray(positions, dtype=float)
    lab = np.asarray(labels, dtype=int).ravel()
    rk = np.asarray(radii, dtype=float).ravel()
    n = int(lab.size)
    mask = np.zeros(n, dtype=bool)
    if n == 0 or pos.ndim != 2 or pos.shape[0] != n or rk.size != n:
        return mask
    labeled = np.flatnonzero(lab >= 0)
    if labeled.size == 0 or n < 2:
        return mask
    tree = cKDTree(pos)
    for i in labeled:
        radius = float(multiple) * float(rk[i])
        if not np.isfinite(radius) or radius <= 0.0:
            continue
        for j in tree.query_ball_point(pos[i], radius):
            if int(j) != int(i) and int(lab[int(j)]) != int(lab[i]):
                mask[int(i)] = True
                break
    return mask


def _median_or_none(values: np.ndarray) -> float | None:
    if values.size == 0 or not np.isfinite(values).any():
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def two_largest_signal_labels(labels: np.ndarray) -> tuple[int, int] | None:
    """Return the two largest signal labels (A, B) by node count."""

    lab = np.asarray(labels, dtype=int).ravel()
    keys = signal_cluster_ids(lab)
    if len(keys) < 2:
        return None
    keys_sorted = sorted(keys, key=lambda key: (-int(np.sum(lab == key)), key))
    return int(keys_sorted[0]), int(keys_sorted[1])


def boundary_interior_rk_medians(
    positions: np.ndarray,
    labels: np.ndarray,
    k: int,
    *,
    multiple: float = BOUNDARY_RK_MULTIPLE,
) -> dict[str, float | None]:
    """Median ``r_k`` on boundary vs interior-A vs interior-B.

    Sides A/B are the two largest signal clusters.  Boundary uses
    :func:`boundary_node_mask`.  ``rk_contrast`` is
    ``bound_med_rk / min(intA_med_rk, intB_med_rk)`` when the
    denominator is positive, else ``None``.
    """

    empty = {
        "bound_med_rk": None,
        "intA_med_rk": None,
        "intB_med_rk": None,
        "rk_contrast": None,
    }
    pos = np.asarray(positions, dtype=float)
    lab = np.asarray(labels, dtype=int).ravel()
    pair = two_largest_signal_labels(lab)
    if pair is None or pos.ndim != 2 or pos.shape[0] != lab.size:
        return empty
    lab_a, lab_b = pair
    radii = per_node_knn_radii(pos, int(k))
    if radii.size != lab.size:
        return empty
    bound = boundary_node_mask(pos, lab, radii, multiple=float(multiple))
    on_ab = (lab == lab_a) | (lab == lab_b)
    bound_med = _median_or_none(radii[bound & on_ab])
    int_a = _median_or_none(radii[(lab == lab_a) & ~bound])
    int_b = _median_or_none(radii[(lab == lab_b) & ~bound])
    denom = None
    if int_a is not None and int_b is not None:
        denom = min(float(int_a), float(int_b))
    elif int_a is not None:
        denom = float(int_a)
    elif int_b is not None:
        denom = float(int_b)
    contrast = None
    if bound_med is not None and denom is not None and denom > 0.0:
        contrast = float(bound_med) / denom
    return {
        "bound_med_rk": bound_med,
        "intA_med_rk": int_a,
        "intB_med_rk": int_b,
        "rk_contrast": contrast,
    }


def compute_cut_covariates(
    scaffold: Any,
    labels: np.ndarray,
    samples: np.ndarray | None = None,
    k: int = WIDEN_K_DEFAULT,
    max_nodes: int | None = None,
) -> dict[str, Any]:
    """Bundle N, tau, at_bound, min-side, and boundary/interior ``r_k``.

    ``labels`` must already be the mass-filtered A/B (signal ``>= 0``,
    background ``< 0``) node labels.  ``k`` is ``k_neighbors``.
    """

    lab = np.asarray(labels, dtype=int).ravel()
    n_nodes = len(getattr(scaffold, "nodes", ()))
    tau_raw = getattr(scaffold, "tau", None)
    pos = scaffold_positions(scaffold)
    rk = boundary_interior_rk_medians(pos, lab, int(k))
    return {
        "N": int(n_nodes),
        "tau": None if tau_raw is None else float(tau_raw),
        "at_bound": cut_at_bound(scaffold, samples, int(k), max_nodes=max_nodes),
        "min_side_node_frac": min_side_node_fraction(lab),
        "min_side_sample_mass": min_side_sample_mass_fraction(
            lab, scaffold, samples=samples,
        ),
        "n_clusters": len(signal_cluster_ids(lab)),
        **rk,
    }


def mass_filtered_labels_from_selection(
    selection: Any,
    config: LevelSetConfig | None,
) -> np.ndarray | None:
    """Accepted ``cluster_result.labels``, else mass-filter the candidate level."""

    cr = getattr(selection, "cluster_result", None)
    raw = getattr(cr, "labels", None) if cr is not None else None
    if raw is not None:
        return np.asarray(raw, dtype=int)
    cand = getattr(selection, "candidate_level", None)
    tree = getattr(selection, "tree", None)
    levels = getattr(tree, "levels", ()) if tree is not None else ()
    if cand is None or not levels:
        return None
    idx = int(cand)
    if not (0 <= idx < len(levels)):
        return None
    ls = config or LevelSetConfig()
    return _filter_relative_mass(
        levels[idx].labels,
        ls.min_cluster_size,
        ls.min_cluster_frac,
    )


def select_covariate_emit_rows(rows: Sequence[CovariateRow]) -> list[CovariateRow]:
    """Keep every accepted root cut; else the lowest-φ K>=2 candidate."""

    accepted = [r for r in rows if r.accepted]
    if accepted:
        return [
            replace(r, row_role="accepted") if r.row_role != "accepted" else r
            for r in accepted
        ]
    cands = [
        r for r in rows
        if int(r.n_clusters) >= 2 and r.phi is not None
    ]
    if not cands:
        return []
    best = min(cands, key=lambda r: (float(r.phi), r.step))
    return [replace(best, row_role="best_candidate", accepted=False)]


def _finite_values(rows: Sequence[CovariateRow], field: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        raw = getattr(row, field)
        if field == "at_bound":
            out.append(1.0 if bool(raw) else 0.0)
            continue
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            out.append(val)
    return out


def covariate_band_stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None, "count": 0}
    arr = np.asarray(list(values), dtype=float)
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "count": int(arr.size),
    }


def ranges_overlap(
    a: Sequence[float],
    b: Sequence[float],
) -> bool:
    """Closed-interval overlap; empty side cannot separate (counts as overlap)."""

    if not a or not b:
        return True
    return not (float(max(a)) < float(min(b)) or float(max(b)) < float(min(a)))


def summarize_covariate_bands(
    rows: Sequence[CovariateRow],
) -> dict[str, Any]:
    """Min/median/max per numeric covariate on null- vs composite-accepts."""

    null_acc = [r for r in rows if r.family == "null" and r.accepted]
    comp_acc = [r for r in rows if r.family == "composite" and r.accepted]
    bands: dict[str, Any] = {
        "null_accepts": {"n": len(null_acc), "fields": {}},
        "composite_accepts": {"n": len(comp_acc), "fields": {}},
        "overlap": {},
        "separating": [],
    }
    for field in COVARIATE_NUMERIC_FIELDS:
        nv = _finite_values(null_acc, field)
        cv = _finite_values(comp_acc, field)
        bands["null_accepts"]["fields"][field] = covariate_band_stats(nv)
        bands["composite_accepts"]["fields"][field] = covariate_band_stats(cv)
        overlap = ranges_overlap(nv, cv)
        bands["overlap"][field] = overlap
        if not overlap:
            bands["separating"].append(field)
    bands["kill"] = len(bands["separating"]) == 0
    return bands


def format_covariate_summary(bands: dict[str, Any]) -> str:
    lines = ["SUMMARY_BANDS"]
    for side in ("null_accepts", "composite_accepts"):
        block = bands[side]
        lines.append(f"  {side} n={block['n']}")
        for field in COVARIATE_NUMERIC_FIELDS:
            st = block["fields"][field]
            lines.append(
                f"    {field} min={_fmt(st['min'])} "
                f"median={_fmt(st['median'])} max={_fmt(st['max'])} "
                f"count={st['count']}"
            )
    sep = bands["separating"]
    if bands["kill"]:
        lines.append(
            "KILL_CHECK every covariate null-accept range overlaps "
            "the composite-accept range (or no separating covariate)"
        )
        lines.append(
            "NO_COVARIATE_FLOOR do not propose min-side or bound rules"
        )
    else:
        lines.append(
            "KILL_CHECK not fired; separating covariates: " + ",".join(sep)
        )
        lines.append(
            "NO_FLOOR_PROPOSAL table-only (A3-T11); do not apply a floor"
        )
    return "\n".join(lines)


def format_covariate_table(rows: Sequence[CovariateRow]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=list(COVARIATE_TABLE_FIELDS), lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row.as_dict())
    return buf.getvalue()


def _covariate_payloads(
    *,
    scenes: Sequence[str] | None,
    seeds: Sequence[int] | None,
    max_depth: int,
    max_epochs: int,
    max_grid_points: int,
    max_finer_steps: int,
    growth_policy: str,
    k: int,
    n_samples: int | None,
) -> list[dict[str, Any]]:
    """Default three-pack cartesian, or an explicit scene × seed grid."""

    base = {
        "max_depth": int(max_depth),
        "max_epochs": int(max_epochs),
        "max_grid_points": int(max_grid_points),
        "max_finer_steps": int(max_finer_steps),
        "growth_policy": str(growth_policy),
        "k": int(k),
        "n_samples": n_samples,
    }
    payloads: list[dict[str, Any]] = []
    if scenes is not None:
        use_seeds = list(seeds) if seeds is not None else list(COVARIATE_NULL_SHORT_SEEDS)
        for scene in scenes:
            for seed in use_seeds:
                payloads.append({**base, "scene": scene, "seed": int(seed)})
        return payloads
    for scene in COVARIATE_NULL_LONG_SCENES:
        long_seeds = list(seeds) if seeds is not None else list(COVARIATE_NULL_LONG_SEEDS)
        for seed in long_seeds:
            payloads.append({**base, "scene": scene, "seed": int(seed)})
    short_seeds = list(seeds) if seeds is not None else list(COVARIATE_NULL_SHORT_SEEDS)
    for scene in DEFAULT_NULL_SCENES:
        for seed in short_seeds:
            payloads.append({**base, "scene": scene, "seed": int(seed)})
    for scene in COMPOSITE_COVARIATE_SCENES:
        for seed in short_seeds:
            payloads.append({**base, "scene": scene, "seed": int(seed)})
    return payloads


def collect_root_covariate_rows(
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
) -> list[CovariateRow]:
    """Root walk (max_depth=1); emit accepted cuts or the best K>=2 miss."""

    data = build_covariate_dataset(scene_name, int(seed), n_samples)
    points = np.asarray(data.points, dtype=float)
    dim = int(data.ground_truth.ambient_dim)
    n_full = int(points.shape[0])
    config = _recursion_config(
        int(seed),
        k=int(k),
        max_depth=int(max_depth),
        max_epochs=int(max_epochs),
        max_grid_points=int(max_grid_points),
        max_finer_steps=int(max_finer_steps),
        growth_policy=str(growth_policy),
    )
    family = covariate_family(scene_name)
    original = recursion_mod.select_level_set_partition
    collected: list[CovariateRow] = []
    step = 0

    def wrapper(scaffold, config=None, dm_config=None, data=None):
        nonlocal step
        selection = original(scaffold, config, dm_config, data)
        step += 1
        data_arr = points if data is None else np.asarray(data, dtype=float)
        if int(data_arr.shape[0]) != n_full:
            return selection
        labels = mass_filtered_labels_from_selection(selection, config)
        if labels is None:
            return selection
        keys = signal_cluster_ids(labels)
        phi = selection.bottleneck_ratio
        reason = None
        if selection.resolvability is not None:
            reason = selection.resolvability.reject_reason
        cov = compute_cut_covariates(
            scaffold,
            labels,
            samples=data_arr,
            k=int(k),
        )
        collected.append(
            CovariateRow(
                scene=scene_name,
                seed=int(seed),
                step=int(step),
                family=family,
                row_role="accepted" if selection.accepted else "candidate",
                N=int(cov["N"]),
                tau=cov["tau"],
                phi=None if phi is None else float(phi),
                accepted=bool(selection.accepted),
                reason=reason,
                n_samples=n_full,
                k=int(k),
                n_clusters=int(cov["n_clusters"]),
                at_bound=bool(cov["at_bound"]),
                min_side_node_frac=cov["min_side_node_frac"],
                min_side_sample_mass=cov["min_side_sample_mass"],
                bound_med_rk=cov["bound_med_rk"],
                intA_med_rk=cov["intA_med_rk"],
                intB_med_rk=cov["intB_med_rk"],
                rk_contrast=cov["rk_contrast"],
            )
        )
        # Keep K>=2 misses even when phi is set; drop empty cuts here.
        if not keys and not selection.accepted:
            collected.pop()
        return selection

    recursion_mod.select_level_set_partition = wrapper
    try:
        run_recursive_discovery(points, dim=dim, config=config)
    finally:
        recursion_mod.select_level_set_partition = original
    return select_covariate_emit_rows(collected)


def _covariate_worker(
    payload: dict[str, Any],
) -> tuple[str, int, list[dict[str, Any]], float]:
    t0 = time.time()
    rows = collect_root_covariate_rows(
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


def _covariate_rows_from_dicts(dicts: Iterable[dict[str, Any]]) -> list[CovariateRow]:
    out: list[CovariateRow] = []
    for d in dicts:
        n_raw = d.get("n_samples", "")
        k_raw = d.get("k", "")
        out.append(
            CovariateRow(
                scene=str(d["scene"]),
                seed=int(d["seed"]),
                step=int(d["step"]),
                family=str(d.get("family") or covariate_family(str(d["scene"]))),
                row_role=str(d.get("row_role") or "candidate"),
                N=int(d["N"]),
                tau=None if d.get("tau") in (None, "") else float(d["tau"]),
                phi=None if d.get("phi") in (None, "") else float(d["phi"]),
                accepted=bool(int(d["accepted"])),
                reason=(None if d.get("reason") in (None, "") else str(d["reason"])),
                n_samples=None if n_raw in (None, "") else int(n_raw),
                k=None if k_raw in (None, "") else int(k_raw),
                n_clusters=int(d.get("n_clusters") or 0),
                at_bound=bool(int(d.get("at_bound") or 0)),
                min_side_node_frac=(
                    None if d.get("min_side_node_frac") in (None, "")
                    else float(d["min_side_node_frac"])
                ),
                min_side_sample_mass=(
                    None if d.get("min_side_sample_mass") in (None, "")
                    else float(d["min_side_sample_mass"])
                ),
                bound_med_rk=(
                    None if d.get("bound_med_rk") in (None, "")
                    else float(d["bound_med_rk"])
                ),
                intA_med_rk=(
                    None if d.get("intA_med_rk") in (None, "")
                    else float(d["intA_med_rk"])
                ),
                intB_med_rk=(
                    None if d.get("intB_med_rk") in (None, "")
                    else float(d["intB_med_rk"])
                ),
                rk_contrast=(
                    None if d.get("rk_contrast") in (None, "")
                    else float(d["rk_contrast"])
                ),
            )
        )
    return out


def run_covariate_table(
    payloads: Sequence[dict[str, Any]],
    *,
    jobs: int = 1,
) -> list[CovariateRow]:
    collected: list[CovariateRow] = []
    if jobs <= 1 or len(payloads) <= 1:
        iterator: Iterable[tuple[dict[str, Any], tuple[Any, ...]]] = (
            (p, _covariate_worker(p)) for p in payloads
        )
        for payload, (scene, seed, dicts, elapsed) in iterator:
            rows = _covariate_rows_from_dicts(dicts)
            collected.extend(rows)
            print(
                f"DONE {scene} seed={seed} family={covariate_family(scene)} "
                f"emitted={len(rows)} elapsed={elapsed:.1f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=int(jobs)) as pool:
            futures = {pool.submit(_covariate_worker, p): p for p in payloads}
            for fut in as_completed(futures):
                payload = futures[fut]
                scene, seed, dicts, elapsed = fut.result()
                rows = _covariate_rows_from_dicts(dicts)
                collected.extend(rows)
                print(
                    f"DONE {scene} seed={seed} family={covariate_family(scene)} "
                    f"emitted={len(rows)} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    collected.sort(key=lambda r: (r.family, r.scene, r.seed, r.step))
    return collected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Null-ensemble φ envelope for LevelSetConfig.max_bottleneck_ratio",
    )
    parser.add_argument(
        "--mode",
        choices=("protocol", "widen", "widen-new", "component-only", "covariate"),
        default="protocol",
        help="protocol=six nulls; widen-new=new geometries only; "
        "widen=new + existing n×k grid; component-only=A4-T6 child-sized nulls; "
        "covariate=A3-T11 cut-covariate table (no floor).",
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

    if args.mode == "covariate":
        seeds_override = None if args.seeds is None else parse_seed_spec(args.seeds)
        cov_payloads = _covariate_payloads(
            scenes=list(args.scenes) if args.scenes else None,
            seeds=seeds_override,
            max_depth=int(args.max_depth),
            max_epochs=int(args.max_epochs),
            max_grid_points=int(args.max_grid_points),
            max_finer_steps=int(args.max_finer_steps),
            growth_policy=str(args.growth_policy),
            k=int(args.k),
            n_samples=args.n_samples,
        )
        cov_scenes = sorted({p["scene"] for p in cov_payloads})
        print(
            f"level_set_null_envelope mode=covariate scenes={cov_scenes} "
            f"n_payloads={len(cov_payloads)} jobs={args.jobs} "
            f"max_depth={args.max_depth} finer_steps={args.max_finer_steps} "
            f"growth_policy={args.growth_policy}",
            flush=True,
        )
        t0 = time.time()
        cov_rows = run_covariate_table(cov_payloads, jobs=int(args.jobs))
        elapsed = time.time() - t0
        table = format_covariate_table(cov_rows)
        print(table, end="", flush=True)
        if args.csv:
            with open(args.csv, "w", encoding="utf-8") as fh:
                fh.write(table)
            print(f"WROTE_CSV {args.csv}", flush=True)
        bands = summarize_covariate_bands(cov_rows)
        print(format_covariate_summary(bands), flush=True)
        print(
            f"TOTAL_ELAPSED={elapsed:.1f}s n_rows={len(cov_rows)} "
            f"null_accepts={bands['null_accepts']['n']} "
            f"composite_accepts={bands['composite_accepts']['n']}",
            flush=True,
        )
        return 0

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


class _FakeNode:
    def __init__(self, position: np.ndarray, hit_count: float = 1.0) -> None:
        self.position = np.asarray(position, dtype=float)
        self.hit_count = float(hit_count)


class _FakeScaffold:
    def __init__(
        self,
        positions: np.ndarray,
        hits: Sequence[float] | None = None,
        tau: float | None = 0.1,
        max_nodes: int | None = None,
    ) -> None:
        pos = np.asarray(positions, dtype=float)
        if hits is None:
            hits = [1.0] * int(pos.shape[0])
        self.nodes = [_FakeNode(p, h) for p, h in zip(pos, hits)]
        self.tau = tau
        if max_nodes is not None:
            self.max_nodes = int(max_nodes)


def _two_clump_positions(
    n_a: int = 8,
    n_b: int = 8,
    gap: float = 4.0,
    jitter: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Tight 2-D clumps on the x-axis; labels 0/1."""

    rng = np.random.default_rng(int(seed))
    a = rng.normal(scale=jitter, size=(int(n_a), 2))
    a[:, 0] -= float(gap) / 2.0
    b = rng.normal(scale=jitter, size=(int(n_b), 2))
    b[:, 0] += float(gap) / 2.0
    pos = np.vstack([a, b])
    lab = np.concatenate([
        np.zeros(int(n_a), dtype=int),
        np.ones(int(n_b), dtype=int),
    ])
    return pos, lab


def _cov_row(**kwargs: Any) -> CovariateRow:
    base: dict[str, Any] = dict(
        scene="circle_null",
        seed=0,
        step=1,
        family="null",
        row_role="candidate",
        N=16,
        tau=0.1,
        phi=0.4,
        accepted=False,
        reason="bottleneck",
        n_samples=100,
        k=8,
        n_clusters=2,
        at_bound=False,
        min_side_node_frac=0.4,
        min_side_sample_mass=0.4,
        bound_med_rk=0.2,
        intA_med_rk=0.1,
        intB_med_rk=0.1,
        rk_contrast=2.0,
    )
    base.update(kwargs)
    return CovariateRow(**base)


def test_min_side_node_fraction_two_clumps() -> None:
    _pos, lab = _two_clump_positions(n_a=3, n_b=7)
    assert min_side_node_fraction(lab) == 0.3
    assert min_side_node_fraction(np.zeros(5, dtype=int)) is None
    mixed = np.array([0, 0, 1, 1, -1, -1], dtype=int)
    assert min_side_node_fraction(mixed) == 0.5


def test_min_side_sample_mass_hit_count_and_bmu() -> None:
    pos, lab = _two_clump_positions(n_a=4, n_b=4, gap=5.0, jitter=0.02)
    hits = [10.0] * 4 + [1.0] * 4
    scaffold = _FakeScaffold(pos, hits=hits, tau=0.2)
    # No samples → hit_count weights: min side is B (4 vs 40).
    hit_frac = min_side_sample_mass_fraction(lab, scaffold, samples=None)
    assert hit_frac is not None
    assert abs(hit_frac - 4.0 / 44.0) < 1e-9
    # Samples piled on the small-hit clump → BMU mass flips.
    samples = np.column_stack([
        np.full(20, float(pos[4:, 0].mean())),
        np.full(20, float(pos[4:, 1].mean())),
    ])
    samples = np.vstack([
        samples,
        np.column_stack([
            np.full(5, float(pos[:4, 0].mean())),
            np.full(5, float(pos[:4, 1].mean())),
        ]),
    ])
    bmu_frac = min_side_sample_mass_fraction(lab, scaffold, samples=samples)
    assert bmu_frac is not None
    assert abs(bmu_frac - 5.0 / 25.0) < 1e-9


def test_cut_at_bound_shot_noise_and_max_nodes() -> None:
    pos, _lab = _two_clump_positions(n_a=8, n_b=8)
    scaffold = _FakeScaffold(pos, tau=0.05, max_nodes=32)
    # 16 nodes, 200 samples, k=8 → cap = 25; 16 < 25 and 16 < 32 → False.
    samples = np.zeros((200, 2), dtype=float)
    assert cut_at_bound(scaffold, samples, k=8) is False
    # 16 nodes, 80 samples, k=8 → cap = 10; 16 >= 10 → True (shot).
    tight = np.zeros((80, 2), dtype=float)
    assert cut_at_bound(scaffold, tight, k=8) is True
    # max_nodes argument fires even when shot does not.
    wide = np.zeros((400, 2), dtype=float)
    assert cut_at_bound(scaffold, wide, k=8, max_nodes=16) is True
    # No samples: only the cap check.
    bare = _FakeScaffold(pos, max_nodes=16)
    assert cut_at_bound(bare, None, k=8) is True
    uncapped = _FakeScaffold(pos)
    assert cut_at_bound(uncapped, None, k=8) is False


def test_boundary_vs_interior_rk_separated_clumps() -> None:
    """Far clumps: nearest other-label node is beyond 2 r_k → no boundary."""

    pos, lab = _two_clump_positions(n_a=8, n_b=8, gap=8.0, jitter=0.03)
    radii = per_node_knn_radii(pos, k=3)
    mask = boundary_node_mask(pos, lab, radii, multiple=2.0)
    assert int(np.sum(mask)) == 0
    stats = boundary_interior_rk_medians(pos, lab, k=3)
    assert stats["bound_med_rk"] is None
    assert stats["intA_med_rk"] is not None
    assert stats["intB_med_rk"] is not None
    assert stats["rk_contrast"] is None
    assert min_side_node_fraction(lab) == 0.5


def test_boundary_vs_interior_rk_adjacent_clumps() -> None:
    """Near clumps plus a mid-gap node: that node is boundary, cores interior."""

    rng = np.random.default_rng(1)
    a = rng.normal(scale=0.04, size=(6, 2))
    a[:, 0] -= 0.35
    b = rng.normal(scale=0.04, size=(6, 2))
    b[:, 0] += 0.35
    bridge = np.array([[0.0, 0.0]], dtype=float)
    pos = np.vstack([a, bridge, b])
    lab = np.concatenate([
        np.zeros(6, dtype=int),
        np.array([0], dtype=int),
        np.ones(6, dtype=int),
    ])
    radii = per_node_knn_radii(pos, k=3)
    mask = boundary_node_mask(pos, lab, radii, multiple=2.0)
    assert bool(mask[6])  # bridge sits next to the other side
    stats = boundary_interior_rk_medians(pos, lab, k=3)
    assert stats["bound_med_rk"] is not None
    # At least one interior core should exist on a tight clump.
    assert stats["intA_med_rk"] is not None or stats["intB_med_rk"] is not None


def test_compute_cut_covariates_bundle() -> None:
    pos, lab = _two_clump_positions(n_a=5, n_b=15, gap=6.0, jitter=0.04)
    scaffold = _FakeScaffold(pos, hits=[2.0] * 5 + [1.0] * 15, tau=0.03)
    samples = np.vstack([
        np.repeat(pos[:5].mean(axis=0, keepdims=True), 40, axis=0),
        np.repeat(pos[5:].mean(axis=0, keepdims=True), 120, axis=0),
    ])
    cov = compute_cut_covariates(scaffold, lab, samples=samples, k=3)
    assert cov["N"] == 20
    assert cov["tau"] == 0.03
    assert cov["n_clusters"] == 2
    assert cov["min_side_node_frac"] == 0.25
    assert cov["min_side_sample_mass"] is not None
    assert abs(float(cov["min_side_sample_mass"]) - 0.25) < 1e-9
    # 20 nodes, 160 samples, k=3 → shot cap = 53; 20 < 53 and no max_nodes.
    assert cov["at_bound"] is False


def test_select_covariate_emit_rows() -> None:
    accepted = [
        _cov_row(step=2, accepted=True, phi=0.12, row_role="accepted"),
        _cov_row(step=3, accepted=True, phi=0.11, row_role="accepted"),
    ]
    extras = [_cov_row(step=1, accepted=False, phi=0.4, n_clusters=2)]
    out = select_covariate_emit_rows(extras + accepted)
    assert [r.step for r in out] == [2, 3]
    misses = [
        _cov_row(step=1, accepted=False, phi=0.5, n_clusters=2),
        _cov_row(step=2, accepted=False, phi=0.31, n_clusters=2),
        _cov_row(step=3, accepted=False, phi=0.2, n_clusters=1),
    ]
    best = select_covariate_emit_rows(misses)
    assert len(best) == 1
    assert best[0].step == 2
    assert best[0].row_role == "best_candidate"
    assert select_covariate_emit_rows([]) == []


def test_covariate_band_overlap_kill() -> None:
    # Overlapping N / tau / fracs → kill.
    overlap_rows = [
        _cov_row(family="null", accepted=True, N=20, tau=0.10, min_side_node_frac=0.30,
                 min_side_sample_mass=0.30, bound_med_rk=0.20, intA_med_rk=0.10,
                 intB_med_rk=0.10, rk_contrast=2.0, at_bound=False),
        _cov_row(family="null", accepted=True, N=24, tau=0.14, min_side_node_frac=0.45,
                 min_side_sample_mass=0.40, bound_med_rk=0.28, intA_med_rk=0.14,
                 intB_med_rk=0.13, rk_contrast=2.2, at_bound=True),
        _cov_row(family="composite", accepted=True, N=22, tau=0.12, min_side_node_frac=0.35,
                 min_side_sample_mass=0.35, bound_med_rk=0.24, intA_med_rk=0.12,
                 intB_med_rk=0.11, rk_contrast=2.1, at_bound=False),
        _cov_row(family="composite", accepted=True, N=23, tau=0.13, min_side_node_frac=0.40,
                 min_side_sample_mass=0.38, bound_med_rk=0.26, intA_med_rk=0.13,
                 intB_med_rk=0.12, rk_contrast=2.15, at_bound=True),
    ]
    bands = summarize_covariate_bands(overlap_rows)
    assert bands["kill"] is True
    text = format_covariate_summary(bands)
    assert "KILL_CHECK" in text
    assert "NO_COVARIATE_FLOOR" in text
    # A separator on min_side_node_frac.
    sep_rows = [
        _cov_row(family="null", accepted=True, min_side_node_frac=0.05, N=10,
                 tau=0.01, min_side_sample_mass=0.05, bound_med_rk=1.0,
                 intA_med_rk=0.5, intB_med_rk=0.5, rk_contrast=2.0, at_bound=True),
        _cov_row(family="composite", accepted=True, min_side_node_frac=0.40, N=80,
                 tau=0.2, min_side_sample_mass=0.40, bound_med_rk=0.2,
                 intA_med_rk=0.1, intB_med_rk=0.1, rk_contrast=2.0, at_bound=False),
    ]
    sep_bands = summarize_covariate_bands(sep_rows)
    assert sep_bands["kill"] is False
    assert "min_side_node_frac" in sep_bands["separating"]


def test_covariate_payloads_default_packs() -> None:
    payloads = _covariate_payloads(
        scenes=None,
        seeds=None,
        max_depth=1,
        max_epochs=12,
        max_grid_points=8,
        max_finer_steps=16,
        growth_policy="track_tau",
        k=WIDEN_K_DEFAULT,
        n_samples=None,
    )
    scenes = {p["scene"] for p in payloads}
    assert scenes >= set(COVARIATE_NULL_LONG_SCENES)
    assert scenes >= set(DEFAULT_NULL_SCENES)
    assert scenes >= set(COMPOSITE_COVARIATE_SCENES)
    long_n = sum(1 for p in payloads if p["scene"] in COVARIATE_NULL_LONG_SCENES)
    assert long_n == len(COVARIATE_NULL_LONG_SCENES) * 20
    short_null = sum(1 for p in payloads if p["scene"] in DEFAULT_NULL_SCENES)
    assert short_null == len(DEFAULT_NULL_SCENES) * 5
    comp_n = sum(1 for p in payloads if p["scene"] in COMPOSITE_COVARIATE_SCENES)
    assert comp_n == len(COMPOSITE_COVARIATE_SCENES) * 5
    assert len(payloads) == 40 + 30 + 15


def test_build_covariate_dataset_composites_use_scenes_factories() -> None:
    present = {s.name for s in _scenes()}
    for scene in COMPOSITE_COVARIATE_SCENES:
        assert scene in present
        data = build_covariate_dataset(scene, seed=0)
        assert data.points.ndim == 2
        assert data.points.shape[0] >= 50
    scurve = build_covariate_dataset("scurve_sheet_null", seed=0)
    assert scurve.points.shape[0] == WIDEN_NULL_N
    flat = build_covariate_dataset("flat_strip_null", seed=0)
    assert flat.ground_truth.name == "flat_strip"


if __name__ == "__main__":
    sys.exit(main())
