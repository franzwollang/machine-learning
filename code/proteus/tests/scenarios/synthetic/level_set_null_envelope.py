"""Reproducible φ-ceiling null-ensemble envelope (#48 / A3-T1).

Runs the root ``track_tau`` walk (``max_depth=1``) over a declared null
ensemble × seeds and writes:

1. one table row per root read that has a candidate cut
   (``scene, seed, step, N, tau, phi, accepted, reason``);
2. an envelope summary (min, p1, p5, p10, median, count, any accepts).

Protocol match (2026-09-09): six null scenes × seeds 0–19 → 359 candidate
reads with φ>0; min 0.288 / p1 0.487 / p5 0.630 / p10 0.760 / median 1.30;
plus one exact φ=0 accept on ``lone_gauss2d_null`` seed 17 (excluded from
the envelope percentiles; graph-disconnection, owned by A2).

Not a default pytest test.  Pure helpers below are unit-tested.

Usage::

    PYTHONPATH="src:$PWD" pipenv run python \\
        tests/scenarios/synthetic/level_set_null_envelope.py \\
        --seeds 0-19 --jobs 4

    PYTHONPATH="src:$PWD" pipenv run python \\
        tests/scenarios/synthetic/level_set_null_envelope.py \\
        --seeds 17 --scenes lone_gauss2d_null
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np

import proteus.stage1.recursion as recursion_mod
from proteus.stage1.recursion import run_recursive_discovery
from tests.scenarios.synthetic.level_set_normal_path_sweep import _config, _scenes


# Declared null ensemble for the φ-ceiling calibration (SI S2.6.2 / S14.3).
DEFAULT_NULL_SCENES: tuple[str, ...] = (
    "circle_null",
    "swiss_roll_null",
    "lone_torus_null",
    "lone_shell_inner_null",
    "lone_gauss2d_null",
    "lone_gauss4d_null",
)

# Envelope on candidate reads with φ > 0 (excludes the lone_gauss2d s17 φ=0).
REFERENCE_ENVELOPE: dict[str, float] = {
    "min": 0.288,
    "p1": 0.487,
    "p5": 0.630,
    "p10": 0.760,
    "median": 1.30,
    "count": 359.0,
}

TABLE_FIELDS: tuple[str, ...] = (
    "scene",
    "seed",
    "step",
    "N",
    "tau",
    "phi",
    "accepted",
    "reason",
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
    # Nearest-rank: index = ceil(p/100 * n) - 1, clamped.
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


def format_envelope(summary: EnvelopeSummary) -> str:
    lines = [
        "ENVELOPE (candidate reads with phi > 0)",
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
    return "\n".join(lines)


def _fmt(value: Any, digits: str = ".6g") -> str:
    if value is None:
        return "na"
    try:
        return format(float(value), digits)
    except (TypeError, ValueError):
        return str(value)


def _null_scenes() -> dict[str, Any]:
    return {s.name: s for s in _scenes() if s.name in DEFAULT_NULL_SCENES}


def collect_root_candidate_reads(
    scene_name: str,
    seed: int,
    *,
    max_depth: int = 1,
    max_epochs: int = 12,
    max_grid_points: int = 8,
    max_finer_steps: int = 16,
    growth_policy: str = "track_tau",
) -> list[ReadRow]:
    """Run one scene-seed root walk; return rows for reads with a candidate φ."""

    scenes = _null_scenes()
    if scene_name not in scenes:
        known = sorted(_null_scenes())
        raise ValueError(f"unknown null scene {scene_name!r}; known: {known}")
    scene = scenes[scene_name]
    data = scene.factory(int(seed))
    points = np.asarray(data.points, dtype=float)
    dim = int(data.ground_truth.ambient_dim)
    args = argparse.Namespace(
        max_epochs=int(max_epochs),
        max_grid_points=int(max_grid_points),
        max_finer_steps=int(max_finer_steps),
        growth_policy=str(growth_policy),
    )
    config = replace(_config(args, int(seed)), max_depth=int(max_depth))

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
            )
        )
    return out


def run_envelope(
    scenes: Sequence[str],
    seeds: Sequence[int],
    *,
    jobs: int = 1,
    max_depth: int = 1,
    max_epochs: int = 12,
    max_grid_points: int = 8,
    max_finer_steps: int = 16,
    growth_policy: str = "track_tau",
) -> tuple[list[ReadRow], EnvelopeSummary]:
    """Run the declared ensemble; return candidate rows + envelope summary."""

    payloads = [
        {
            "scene": scene,
            "seed": int(seed),
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
                f"DONE {scene} seed={seed} candidate_reads={len(rows)} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=int(jobs)) as pool:
            futures = {pool.submit(_worker, p): p for p in payloads}
            for fut in as_completed(futures):
                scene, seed, dicts, elapsed = fut.result()
                rows = _rows_from_dicts(dicts)
                collected.extend(rows)
                print(
                    f"DONE {scene} seed={seed} candidate_reads={len(rows)} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    collected.sort(key=lambda r: (r.scene, r.seed, r.step))
    return collected, summarize_envelope(collected)


def compare_to_reference(
    summary: EnvelopeSummary,
    *,
    abs_tol: float = 0.02,
    count_tol: int = 5,
) -> list[str]:
    """Return human-readable mismatches vs the 2026-09-09 reference envelope."""

    mismatches: list[str] = []
    if abs(summary.count - int(REFERENCE_ENVELOPE["count"])) > count_tol:
        mismatches.append(
            f"count {summary.count} vs ref {int(REFERENCE_ENVELOPE['count'])}"
        )
    for key in ("min", "p1", "p5", "p10", "median"):
        got = getattr(summary, key)
        ref = float(REFERENCE_ENVELOPE[key])
        if got is None or abs(float(got) - ref) > abs_tol:
            mismatches.append(f"{key} {_fmt(got)} vs ref {ref}")
    return mismatches


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Null-ensemble φ envelope for LevelSetConfig.max_bottleneck_ratio",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["0-19"],
        help="Seeds or inclusive ranges (default: 0-19).",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=list(DEFAULT_NULL_SCENES),
        help="Null scene names (default: the six declared nulls).",
    )
    parser.add_argument("--jobs", type=int, default=4)
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

    seeds = parse_seed_spec(args.seeds)
    scenes = list(args.scenes)
    unknown = [s for s in scenes if s not in DEFAULT_NULL_SCENES]
    if unknown:
        raise SystemExit(
            f"unknown null scenes {unknown}; declared: {list(DEFAULT_NULL_SCENES)}"
        )

    print(
        f"level_set_null_envelope scenes={scenes} seeds={seeds[0]}..{seeds[-1]} "
        f"n_seeds={len(seeds)} jobs={args.jobs} max_depth={args.max_depth} "
        f"finer_steps={args.max_finer_steps} growth_policy={args.growth_policy}",
        flush=True,
    )
    t0 = time.time()
    rows, summary = run_envelope(
        scenes,
        seeds,
        jobs=int(args.jobs),
        max_depth=int(args.max_depth),
        max_epochs=int(args.max_epochs),
        max_grid_points=int(args.max_grid_points),
        max_finer_steps=int(args.max_finer_steps),
        growth_policy=str(args.growth_policy),
    )
    elapsed = time.time() - t0

    table = format_table(rows)
    print(table, end="", flush=True)
    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as fh:
            fh.write(table)
        print(f"WROTE_CSV {args.csv}", flush=True)

    print(format_envelope(summary), flush=True)
    print(f"TOTAL_ELAPSED={elapsed:.1f}s n_rows={len(rows)}", flush=True)

    if args.check_reference and scenes == list(DEFAULT_NULL_SCENES) and seeds == list(
        range(20)
    ):
        mismatches = compare_to_reference(summary)
        zero_ok = any(
            s == "lone_gauss2d_null" and seed == 17
            for s, seed, _step, _tau in summary.phi_zero_accepts
        )
        if not zero_ok:
            mismatches.append(
                "missing expected phi=0 accept on lone_gauss2d_null seed 17"
            )
        if mismatches:
            print("REFERENCE_MISMATCH: " + "; ".join(mismatches), flush=True)
            return 1
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
        phi_zero_accepts=(("lone_gauss2d_null", 17, 3, 0.012),),
    )
    assert compare_to_reference(summary) == []


if __name__ == "__main__":
    raise SystemExit(main())
