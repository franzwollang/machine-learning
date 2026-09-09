"""Measure r_k / sqrt(tau*) on lone uniforms vs composites (OPEN_ISSUES #48).

L=1 load-crossover only (no finer walk) except ``--children``, which runs
``run_recursive_discovery`` at ``max_depth=1`` on linked_tori / nested_spheres
and re-measures each non-background child.

Not a pytest test.

Usage::

    PYTHONPATH="src:$PWD" pipenv run python \\
        tests/scenarios/synthetic/level_set_scale_match_probe.py --seeds 0

    PYTHONPATH="src:$PWD" pipenv run python \\
        tests/scenarios/synthetic/level_set_scale_match_probe.py \\
        --children --scenes linked_tori nested_spheres --seeds 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.level_set import (
    LevelSetConfig,
    at_shot_noise_scale,
    mean_neighbor_radius,
    mesh_is_scale_matched,
    select_level_set_partition,
)
from proteus.stage1.recursion import RecursionConfig, run_recursive_discovery
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.ground_truth import expected_tau_for_arc, expected_tau_for_surface
from tests.datasets.synthetic.circles import make_circle
from tests.datasets.synthetic.density_valleys import (
    make_bimodal_circle,
    make_two_gaussians,
)
from tests.datasets.synthetic.hierarchical_gaussian import make_hierarchical_gaussian
from tests.datasets.synthetic.linked_tori import make_linked_tori
from tests.datasets.synthetic.nested_spheres import make_nested_spheres
from tests.datasets.synthetic.swiss_roll import make_swiss_roll
from tests.scenarios.synthetic.level_set_suite import (
    BIMODAL_CIRCLE_KAPPA,
    BIMODAL_CIRCLE_N,
    CIRCLE_N,
    HIERARCHY_N,
    NESTED_N_PER,
    SWISS_N,
    TORI_N_PER,
    TWO_GAUSSIANS_CLEAR_SEP,
    TWO_GAUSSIANS_N,
    TWO_GAUSSIANS_SIGMA,
)

K_NEIGHBORS = 8
MESH_RATIO = 2.0
CHILD_SCENES = ("linked_tori", "nested_spheres")
TISSUE_NEAR_M = 1.0
TORUS_MAJOR = 2.0
TORUS_MINOR = 0.25
NESTED_RADII = (1.0, 2.0)
JSON_PATH_TMPL = "/tmp/scale_match_probe_seed{seed}.json"
WORKER_JSON_TMPL = "/tmp/scale_match_probe_worker_{scene}_s{seed}.json"


@dataclass
class Geom:
    kind: str  # arc | surface | gauss | none
    perimeter: float | None = None
    surface_area: float | None = None
    sigma: float | None = None
    noise_variance: float = 0.0


@dataclass
class SceneBundle:
    name: str
    points: np.ndarray
    ambient_dim: int
    gt_intrinsic_dim: int
    geom: Geom


@dataclass
class Row:
    scene: str
    seed: int
    n_samples: int
    ambient_dim: int
    gt_intrinsic_dim: int
    n_nodes: int | None
    max_nodes: int | None
    tau_star: float | None
    sqrt_tau: float | None
    r_k: float | None
    ratio: float | None
    scale_matched: bool | None
    shot_noise: bool | None
    d_final_mean: float | None
    d_final_median: float | None
    d_final_estimator: str
    accepted: bool | None
    K: int | None
    verdict: str
    reject_reason: str
    bottleneck_ratio: float | None
    studentized_ratio: float | None
    log_bf: float | None
    rk_over_s: float | None
    sqrt_tau_over_sqrt_tau_geom: float | None
    sigma: float | None
    sqrt_tau_over_sigma: float | None
    elapsed_s: float
    error: str | None = None


def _swiss_surface_area(height: float = 1.0, twists: float = 3.0) -> float:
    t_min = 1.5 * np.pi
    t_max = 1.5 * np.pi + float(twists) * np.pi
    scale = t_max

    def arc_primitive(value: float) -> float:
        return 0.5 * (value * np.sqrt(1.0 + value * value) + np.arcsinh(value))

    arc_length = float(arc_primitive(t_max) - arc_primitive(t_min)) / scale
    return arc_length * float(height)


def _torus_area(major: float = TORUS_MAJOR, minor: float = TORUS_MINOR) -> float:
    return float(4.0 * (np.pi ** 2) * major * minor)


def _sphere_area(radius: float) -> float:
    return float(4.0 * np.pi * (radius ** 2))


def _keep_labels(points: np.ndarray, labels: np.ndarray, target: int) -> np.ndarray:
    mask = np.asarray(labels, dtype=int) == int(target)
    return np.asarray(points, dtype=float)[mask]


def _torus_plus_nearby_tissue(
    points: np.ndarray,
    labels: np.ndarray,
    target: int = 0,
    radius: float = TISSUE_NEAR_M,
) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    lab = np.asarray(labels, dtype=int)
    signal = arr[lab == int(target)]
    if signal.shape[0] == 0:
        return signal
    tissue = arr[lab < 0]
    if tissue.shape[0] == 0:
        return signal
    dists, _ = cKDTree(signal).query(tissue, k=1)
    near = tissue[np.asarray(dists, dtype=float) < float(radius)]
    return np.vstack([signal, near])


def _scale_cfg(args: argparse.Namespace, seed: int) -> ScaleSearchConfig:
    return ScaleSearchConfig(
        selector="load_crossover",
        tau_min=1e-5,
        tau_max=10.0,
        max_grid_points=int(args.max_grid_points),
        k=K_NEIGHBORS,
        min_nodes=4,
        n_seeds=8,
        max_nodes=None,
        stabilization=StabilizationConfig(
            min_equilibrium_epochs=3,
            max_epochs=int(args.max_epochs),
        ),
        seed=int(seed),
    )


def _recursion_cfg(args: argparse.Namespace, seed: int) -> RecursionConfig:
    """Sweep ``_config()`` with ``max_depth=1``."""

    return RecursionConfig(
        scale_search=_scale_cfg(args, seed),
        min_samples=100,
        max_depth=1,
        use_level_set_clustering=True,
        allow_finer_research=True,
        max_finer_scale_steps=16,
        level_set=LevelSetConfig(),
        seed=int(seed),
    )


def _fmt(value: float | None, width: int, prec: str = ".4g") -> str:
    if value is None:
        return f"{'na':>{width}}"
    if isinstance(value, float) and not np.isfinite(value):
        return f"{'inf' if value > 0 else '-inf':>{width}}"
    return f"{value:{width}{prec}}"


def _fmt_bool(value: bool | None, width: int) -> str:
    if value is None:
        return f"{'na':>{width}}"
    return f"{('Y' if value else 'N'):>{width}}"


def _fmt_int(value: int | None, width: int) -> str:
    if value is None:
        return f"{'na':>{width}}"
    return f"{value:{width}d}"


def format_header() -> str:
    return (
        f"{'scene':<22} {'sd':>2} {'n':>5} {'amb':>3} {'dgt':>3} "
        f"{'N':>4} {'maxN':>4} {'tau*':>10} {'sqrt_t':>8} {'r_k':>8} "
        f"{'ratio':>7} {'match':>5} {'shot':>4} {'dmu':>5} {'dmd':>4} "
        f"{'est':<6} {'acc':>3} {'K':>3} {'verdict':<16} {'reason':<16} "
        f"{'phi':>7} {'rho':>7} {'logBF':>8} {'rk/s':>7} {'t/tg':>7} "
        f"{'sig':>6} {'t/sig':>7} {'sec':>6}"
    )


def format_row(row: Row) -> str:
    if row.error:
        return (
            f"{row.scene:<22} {row.seed:2d} CRASH {row.error[:120]}"
        )
    return (
        f"{row.scene:<22} {row.seed:2d} {row.n_samples:5d} "
        f"{row.ambient_dim:3d} {row.gt_intrinsic_dim:3d} "
        f"{_fmt_int(row.n_nodes, 4)} {_fmt_int(row.max_nodes, 4)} "
        f"{_fmt(row.tau_star, 10)} {_fmt(row.sqrt_tau, 8)} {_fmt(row.r_k, 8)} "
        f"{_fmt(row.ratio, 7)} {_fmt_bool(row.scale_matched, 5)} "
        f"{_fmt_bool(row.shot_noise, 4)} "
        f"{_fmt(row.d_final_mean, 5, '.2f')} {_fmt(row.d_final_median, 4, '.1f')} "
        f"{row.d_final_estimator:<6} "
        f"{_fmt_bool(row.accepted, 3)} {_fmt_int(row.K, 3)} "
        f"{(row.verdict or 'na'):<16} {(row.reject_reason or 'na'):<16} "
        f"{_fmt(row.bottleneck_ratio, 7)} {_fmt(row.studentized_ratio, 7)} "
        f"{_fmt(row.log_bf, 8)} {_fmt(row.rk_over_s, 7)} "
        f"{_fmt(row.sqrt_tau_over_sqrt_tau_geom, 7)} "
        f"{_fmt(row.sigma, 6)} {_fmt(row.sqrt_tau_over_sigma, 7)} "
        f"{row.elapsed_s:6.1f}"
    )


def _d_final_stats(scaffold) -> tuple[float | None, float | None, str]:
    nodes = list(getattr(scaffold, "nodes", ()))
    if not nodes:
        return None, None, "none"
    existing = [getattr(node, "d_final", None) for node in nodes]
    populated = all(v is not None for v in existing)
    method = str(getattr(scaffold, "intrinsic_dim_method", "degree"))
    if not populated:
        scaffold.refresh_intrinsic_dim()
        method = str(getattr(scaffold, "intrinsic_dim_method", "degree"))
        values = np.asarray(
            [int(node.d_final) for node in scaffold.nodes], dtype=float,
        )
        return float(values.mean()), float(np.median(values)), method
    # Seeded to working dim; refresh so the row is an estimator, not the seed.
    ambient = int(getattr(scaffold, "dim", -1))
    as_int = np.asarray([int(v) for v in existing], dtype=float)
    if as_int.size and np.all(as_int == ambient):
        scaffold.refresh_intrinsic_dim()
        method = str(getattr(scaffold, "intrinsic_dim_method", "degree"))
        values = np.asarray(
            [int(node.d_final) for node in scaffold.nodes], dtype=float,
        )
        return float(values.mean()), float(np.median(values)), method
    return float(as_int.mean()), float(np.median(as_int)), f"{method}+pre"


def _tiling(geom: Geom, n_nodes: int, r_k: float, sqrt_tau: float) -> tuple[
    float | None, float | None, float | None, float | None,
]:
    sigma = geom.sigma
    t_over_sig = (sqrt_tau / sigma) if (sigma is not None and sigma > 0) else None
    if n_nodes <= 0:
        return None, None, sigma, t_over_sig
    noise = float(geom.noise_variance)
    if geom.kind == "arc" and geom.perimeter is not None and geom.perimeter > 0:
        spacing = float(geom.perimeter) / n_nodes
        tau_geom = expected_tau_for_arc(geom.perimeter, n_nodes, noise)
        rk_s = r_k / spacing if spacing > 0 else None
        t_tg = (
            sqrt_tau / float(np.sqrt(tau_geom))
            if tau_geom > 0 else None
        )
        return rk_s, t_tg, sigma, t_over_sig
    if geom.kind == "surface" and geom.surface_area is not None and geom.surface_area > 0:
        spacing = float(np.sqrt(geom.surface_area / n_nodes))
        tau_geom = expected_tau_for_surface(geom.surface_area, n_nodes, noise)
        rk_s = r_k / spacing if spacing > 0 else None
        t_tg = (
            sqrt_tau / float(np.sqrt(tau_geom))
            if tau_geom > 0 else None
        )
        return rk_s, t_tg, sigma, t_over_sig
    return None, None, sigma, t_over_sig


def measure_bundle(
    bundle: SceneBundle,
    seed: int,
    args: argparse.Namespace,
) -> Row:
    started = time.time()
    points = np.asarray(bundle.points, dtype=float)
    dim = int(bundle.ambient_dim)
    scale = _scale_cfg(args, seed)
    config = _recursion_cfg(args, seed)
    result = run_scale_search(points, dim, scale)
    scaffold = result.scaffold_at_star
    tau_star = float(result.tau_star) if result.tau_star is not None else None
    if scaffold is None or len(getattr(scaffold, "nodes", ())) < 2:
        return Row(
            scene=bundle.name,
            seed=seed,
            n_samples=int(points.shape[0]),
            ambient_dim=dim,
            gt_intrinsic_dim=int(bundle.gt_intrinsic_dim),
            n_nodes=0 if scaffold is None else len(getattr(scaffold, "nodes", ())),
            max_nodes=getattr(scaffold, "max_nodes", None) if scaffold else None,
            tau_star=tau_star,
            sqrt_tau=None,
            r_k=None,
            ratio=None,
            scale_matched=False,
            shot_noise=False,
            d_final_mean=None,
            d_final_median=None,
            d_final_estimator="none",
            accepted=None,
            K=None,
            verdict="na",
            reject_reason="empty_scaffold",
            bottleneck_ratio=None,
            studentized_ratio=None,
            log_bf=None,
            rk_over_s=None,
            sqrt_tau_over_sqrt_tau_geom=None,
            sigma=bundle.geom.sigma,
            sqrt_tau_over_sigma=None,
            elapsed_s=time.time() - started,
        )

    positions = np.asarray(scaffold.node_positions(), dtype=float)
    r_k = mean_neighbor_radius(positions, K_NEIGHBORS)
    sqrt_tau = float(np.sqrt(tau_star)) if tau_star is not None and tau_star > 0 else None
    ratio = (r_k / sqrt_tau) if sqrt_tau is not None and sqrt_tau > 0 else None
    matched = mesh_is_scale_matched(scaffold, float(tau_star), K_NEIGHBORS, MESH_RATIO)
    shot = at_shot_noise_scale(scaffold, points, K_NEIGHBORS)
    d_mean, d_med, estimator = _d_final_stats(scaffold)
    selection = select_level_set_partition(
        scaffold, LevelSetConfig(), config.dm_cluster,
    )
    rv = selection.resolvability
    verdict = rv.verdict.value if rv is not None else "na"
    reason = rv.reject_reason if rv is not None and rv.reject_reason else "na"
    k_val = (
        int(selection.cluster_result.n_clusters)
        if selection.cluster_result is not None else None
    )
    n_nodes = int(len(scaffold.nodes))
    rk_s, t_tg, sigma, t_sig = _tiling(bundle.geom, n_nodes, float(r_k), sqrt_tau or 0.0)
    if sqrt_tau is None:
        t_tg = None
        t_sig = None
    return Row(
        scene=bundle.name,
        seed=seed,
        n_samples=int(points.shape[0]),
        ambient_dim=dim,
        gt_intrinsic_dim=int(bundle.gt_intrinsic_dim),
        n_nodes=n_nodes,
        max_nodes=int(scaffold.max_nodes) if scaffold.max_nodes is not None else None,
        tau_star=tau_star,
        sqrt_tau=sqrt_tau,
        r_k=float(r_k),
        ratio=float(ratio) if ratio is not None else None,
        scale_matched=bool(matched),
        shot_noise=bool(shot),
        d_final_mean=d_mean,
        d_final_median=d_med,
        d_final_estimator=estimator,
        accepted=bool(selection.accepted),
        K=k_val,
        verdict=str(verdict),
        reject_reason=str(reason),
        bottleneck_ratio=(
            float(selection.bottleneck_ratio)
            if selection.bottleneck_ratio is not None else None
        ),
        studentized_ratio=(
            float(selection.studentized_ratio)
            if selection.studentized_ratio is not None else None
        ),
        log_bf=float(selection.log_bf),
        rk_over_s=rk_s,
        sqrt_tau_over_sqrt_tau_geom=t_tg,
        sigma=sigma,
        sqrt_tau_over_sigma=t_sig,
        elapsed_s=time.time() - started,
    )


def _circle(seed: int) -> SceneBundle:
    data = make_circle(n_samples=CIRCLE_N, seed=seed)
    return SceneBundle(
        "circle",
        data.points,
        int(data.ground_truth.ambient_dim),
        1,
        Geom("arc", perimeter=2.0 * np.pi * 1.0, noise_variance=data.ground_truth.noise_variance),
    )


def _swiss(seed: int) -> SceneBundle:
    data = make_swiss_roll(n_samples=SWISS_N, seed=seed)
    return SceneBundle(
        "swiss_roll",
        data.points,
        int(data.ground_truth.ambient_dim),
        2,
        Geom(
            "surface",
            surface_area=_swiss_surface_area(),
            noise_variance=data.ground_truth.noise_variance,
        ),
    )


def _lone_torus(seed: int) -> SceneBundle:
    data = make_linked_tori(n_per_torus=TORI_N_PER, seed=seed)
    points = _keep_labels(data.points, data.labels, 0)
    return SceneBundle(
        "lone_torus",
        points,
        int(data.ground_truth.ambient_dim),
        2,
        Geom("surface", surface_area=_torus_area(), noise_variance=data.ground_truth.noise_variance),
    )


def _lone_torus_tissue(seed: int) -> SceneBundle:
    data = make_linked_tori(n_per_torus=TORI_N_PER, seed=seed)
    points = _torus_plus_nearby_tissue(data.points, data.labels, 0, TISSUE_NEAR_M)
    return SceneBundle(
        "lone_torus_tissue",
        points,
        int(data.ground_truth.ambient_dim),
        2,
        Geom("surface", surface_area=_torus_area(), noise_variance=data.ground_truth.noise_variance),
    )


def _lone_shell(seed: int, which: str) -> SceneBundle:
    data = make_nested_spheres(n_per_sphere=NESTED_N_PER, seed=seed)
    # Generator uses label_offsets=[1, 2] for radii (1.0, 2.0); not 0/1.
    signal = np.unique(data.labels[data.labels >= 0])
    if which == "inner":
        target = int(signal.min()) if signal.size else 1
        radius = NESTED_RADII[0]
        name = "lone_shell_inner"
    else:
        target = int(signal.max()) if signal.size else 2
        radius = NESTED_RADII[-1]
        name = "lone_shell_outer"
    points = _keep_labels(data.points, data.labels, target)
    return SceneBundle(
        name,
        points,
        int(data.ground_truth.ambient_dim),
        2,
        Geom(
            "surface",
            surface_area=_sphere_area(radius),
            noise_variance=data.ground_truth.noise_variance,
        ),
    )


def _lone_gauss2d_400(seed: int) -> SceneBundle:
    data = make_two_gaussians(
        n_samples=TWO_GAUSSIANS_N,
        sigma=TWO_GAUSSIANS_SIGMA,
        separation=TWO_GAUSSIANS_CLEAR_SEP,
        seed=seed,
    )
    points = _keep_labels(data.points, data.labels, 0)
    return SceneBundle(
        "lone_gauss2d_400",
        points,
        int(data.ground_truth.ambient_dim),
        2,
        Geom("gauss", sigma=float(TWO_GAUSSIANS_SIGMA)),
    )


def _lone_gauss2d_800(seed: int) -> SceneBundle:
    rng = np.random.default_rng(int(seed))
    sigma = 0.25
    points = rng.normal(size=(800, 2)) * sigma
    return SceneBundle("lone_gauss2d_800", points, 2, 2, Geom("gauss", sigma=sigma))


def _lone_gauss4d_800(seed: int) -> SceneBundle:
    rng = np.random.default_rng(int(seed))
    points = rng.normal(size=(800, 4))
    return SceneBundle("lone_gauss4d_800", points, 4, 4, Geom("gauss", sigma=1.0))


def _lone_gauss4d_2000(seed: int) -> SceneBundle:
    rng = np.random.default_rng(int(seed))
    points = rng.normal(size=(2000, 4))
    return SceneBundle("lone_gauss4d_2000", points, 4, 4, Geom("gauss", sigma=1.0))


def _linked_tori(seed: int) -> SceneBundle:
    data = make_linked_tori(n_per_torus=TORI_N_PER, seed=seed)
    return SceneBundle(
        "linked_tori",
        data.points,
        int(data.ground_truth.ambient_dim),
        2,
        Geom(
            "surface",
            surface_area=2.0 * _torus_area(),
            noise_variance=data.ground_truth.noise_variance,
        ),
    )


def _nested_spheres(seed: int) -> SceneBundle:
    data = make_nested_spheres(n_per_sphere=NESTED_N_PER, seed=seed)
    area = sum(_sphere_area(r) for r in NESTED_RADII)
    return SceneBundle(
        "nested_spheres",
        data.points,
        int(data.ground_truth.ambient_dim),
        2,
        Geom("surface", surface_area=area, noise_variance=data.ground_truth.noise_variance),
    )


def _hierarchy(seed: int) -> SceneBundle:
    data = make_hierarchical_gaussian(n_samples=HIERARCHY_N, seed=seed)
    return SceneBundle(
        "hierarchy",
        data.points,
        int(data.ground_truth.ambient_dim),
        int(data.ground_truth.intrinsic_dim),
        Geom("none"),
    )


def _two_gaussians_clear(seed: int) -> SceneBundle:
    data = make_two_gaussians(
        n_samples=TWO_GAUSSIANS_N,
        sigma=TWO_GAUSSIANS_SIGMA,
        separation=TWO_GAUSSIANS_CLEAR_SEP,
        seed=seed,
    )
    return SceneBundle(
        "two_gaussians_clear",
        data.points,
        int(data.ground_truth.ambient_dim),
        int(data.ground_truth.intrinsic_dim),
        Geom("gauss", sigma=float(TWO_GAUSSIANS_SIGMA)),
    )


def _bimodal_circle(seed: int) -> SceneBundle:
    data = make_bimodal_circle(
        n_samples=BIMODAL_CIRCLE_N,
        kappa=BIMODAL_CIRCLE_KAPPA,
        seed=seed,
    )
    return SceneBundle(
        "bimodal_circle",
        data.points,
        int(data.ground_truth.ambient_dim),
        1,
        Geom("arc", perimeter=2.0 * np.pi * 1.0, noise_variance=data.ground_truth.noise_variance),
    )


SCENE_FACTORIES: dict[str, Callable[[int], SceneBundle]] = {
    "circle": _circle,
    "swiss_roll": _swiss,
    "lone_torus": _lone_torus,
    "lone_torus_tissue": _lone_torus_tissue,
    "lone_shell_inner": lambda seed: _lone_shell(seed, "inner"),
    "lone_shell_outer": lambda seed: _lone_shell(seed, "outer"),
    "lone_gauss2d_400": _lone_gauss2d_400,
    "lone_gauss2d_800": _lone_gauss2d_800,
    "lone_gauss4d_800": _lone_gauss4d_800,
    "lone_gauss4d_2000": _lone_gauss4d_2000,
    "linked_tori": _linked_tori,
    "nested_spheres": _nested_spheres,
    "hierarchy": _hierarchy,
    "two_gaussians_clear": _two_gaussians_clear,
    "bimodal_circle": _bimodal_circle,
}

SCENE_ORDER = tuple(SCENE_FACTORIES.keys())


def _crash_row(name: str, seed: int, n: int, dim: int, dgt: int, err: BaseException) -> Row:
    return Row(
        scene=name,
        seed=seed,
        n_samples=n,
        ambient_dim=dim,
        gt_intrinsic_dim=dgt,
        n_nodes=None,
        max_nodes=None,
        tau_star=None,
        sqrt_tau=None,
        r_k=None,
        ratio=None,
        scale_matched=None,
        shot_noise=None,
        d_final_mean=None,
        d_final_median=None,
        d_final_estimator="na",
        accepted=None,
        K=None,
        verdict="crash",
        reject_reason=type(err).__name__,
        bottleneck_ratio=None,
        studentized_ratio=None,
        log_bf=None,
        rk_over_s=None,
        sqrt_tau_over_sqrt_tau_geom=None,
        sigma=None,
        sqrt_tau_over_sigma=None,
        elapsed_s=0.0,
        error=f"{type(err).__name__}: {err}",
    )


def _row_to_json(row: Row) -> dict:
    payload = asdict(row)
    for key, value in list(payload.items()):
        if isinstance(value, float) and not np.isfinite(value):
            payload[key] = "inf" if value > 0 else "-inf"
    return payload


def _parse_json_value(value):
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    return value


def _row_from_json(payload: dict) -> Row:
    fields = {
        key: _parse_json_value(payload.get(key))
        for key in Row.__dataclass_fields__
    }
    return Row(**fields)


def _measure_children(
    bundle: SceneBundle,
    seed: int,
    args: argparse.Namespace,
) -> list[Row]:
    rows: list[Row] = []
    points = np.asarray(bundle.points, dtype=float)
    dim = int(bundle.ambient_dim)
    config = _recursion_cfg(args, seed)
    print(f"# recursive discovery {bundle.name} seed={seed} max_depth=1", flush=True)
    tree = run_recursive_discovery(points, dim, config)
    if not tree.nodes:
        return rows
    root = tree.nodes[0]
    child_i = 0
    for cid in root.children:
        child = tree.nodes[int(cid)]
        if child.is_background:
            continue
        idx = np.asarray(child.sample_indices, dtype=int)
        if idx.size == 0:
            continue
        child_bundle = SceneBundle(
            name=f"{bundle.name}_child{child_i}",
            points=points[idx],
            ambient_dim=dim,
            gt_intrinsic_dim=int(bundle.gt_intrinsic_dim),
            geom=bundle.geom,
        )
        child_i += 1
        try:
            rows.append(measure_bundle(child_bundle, seed, args))
        except Exception as exc:  # noqa: BLE001 — diagnostic; continue other children
            traceback.print_exc()
            rows.append(
                _crash_row(
                    child_bundle.name, seed, int(idx.size), dim,
                    int(bundle.gt_intrinsic_dim), exc,
                )
            )
    return rows


def run_one_scene(
    name: str,
    seed: int,
    args: argparse.Namespace,
    *,
    with_children: bool,
) -> list[Row]:
    factory = SCENE_FACTORIES[name]
    try:
        bundle = factory(int(seed))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return [_crash_row(name, seed, 0, 0, 0, exc)]
    rows: list[Row] = []
    try:
        rows.append(measure_bundle(bundle, seed, args))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        rows.append(
            _crash_row(
                name, seed, int(bundle.points.shape[0]),
                int(bundle.ambient_dim), int(bundle.gt_intrinsic_dim), exc,
            )
        )
    if with_children and name in CHILD_SCENES:
        try:
            rows.extend(_measure_children(bundle, seed, args))
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            rows.append(
                _crash_row(
                    f"{name}_children", seed, int(bundle.points.shape[0]),
                    int(bundle.ambient_dim), int(bundle.gt_intrinsic_dim), exc,
                )
            )
    return rows


def _run_as_parent(args: argparse.Namespace, wanted: list[str]) -> int:
    print(
        "scale-match probe  selector=load_crossover  k=8  "
        f"mesh_ratio={MESH_RATIO}  max_epochs={args.max_epochs}  "
        f"grid={args.max_grid_points}  children={int(args.children)}",
        flush=True,
    )
    print(
        "# ratio=r_k/sqrt(tau*)  match=mesh_is_scale_matched(...,2.0)  "
        "shot=at_shot_noise_scale  dest=d_final estimator (degree unless noted)  "
        "phi=bottleneck_ratio  rho=studentized_ratio  "
        "rk/s=r_k/ideal_spacing  t/tg=sqrt(tau*)/sqrt(tau_geom(N))",
        flush=True,
    )
    print(
        "# each scene is a subprocess so an ANN SIGSEGV cannot abort the rest",
        flush=True,
    )
    print(format_header(), flush=True)

    by_seed: dict[int, list[Row]] = {int(s): [] for s in args.seeds}
    script = str(Path(__file__).resolve())
    for seed in args.seeds:
        for name in wanted:
            print(f"# start {name} seed={seed}", flush=True)
            worker_path = WORKER_JSON_TMPL.format(scene=name, seed=int(seed))
            cmd = [
                sys.executable,
                script,
                "--worker",
                "--seeds",
                str(int(seed)),
                "--scenes",
                name,
                "--max-epochs",
                str(int(args.max_epochs)),
                "--max-grid-points",
                str(int(args.max_grid_points)),
            ]
            if args.children:
                cmd.append("--children")
            proc = subprocess.run(cmd, check=False)
            produced: list[Row] = []
            if Path(worker_path).is_file():
                with open(worker_path, encoding="utf-8") as handle:
                    produced = [_row_from_json(item) for item in json.load(handle)]
            if proc.returncode != 0 and not produced:
                err = RuntimeError(
                    f"worker exit {proc.returncode} (139=SIGSEGV) and no JSON"
                )
                produced = [_crash_row(name, int(seed), 0, 0, 0, err)]
            elif proc.returncode != 0:
                print(
                    f"# worker {name} seed={seed} exit={proc.returncode}",
                    flush=True,
                )
            by_seed[int(seed)].extend(produced)
            for row in produced:
                print(format_row(row), flush=True)
        path = JSON_PATH_TMPL.format(seed=int(seed))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([_row_to_json(r) for r in by_seed[int(seed)]], handle, indent=2)
        print(f"# wrote {path}", flush=True)

    n_crash = sum(1 for rows in by_seed.values() for r in rows if r.error)
    n_rows = sum(len(rows) for rows in by_seed.values())
    print(f"summary: {n_rows} rows, {n_crash} crashes", flush=True)
    return 1 if n_crash else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--max-grid-points", type=int, default=8)
    parser.add_argument(
        "--children",
        action="store_true",
        help="Also measure non-background children of linked_tori / nested_spheres.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    wanted = list(SCENE_ORDER) if args.scenes is None else list(args.scenes)
    unknown = [name for name in wanted if name not in SCENE_FACTORIES]
    if unknown:
        raise SystemExit(f"unknown scenes: {unknown}; known: {list(SCENE_ORDER)}")

    if not args.worker:
        return _run_as_parent(args, wanted)

    seed = int(args.seeds[0])
    name = wanted[0]
    produced = run_one_scene(name, seed, args, with_children=bool(args.children))
    path = WORKER_JSON_TMPL.format(scene=name, seed=seed)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([_row_to_json(r) for r in produced], handle, indent=2)
    return 1 if any(row.error for row in produced) else 0


if __name__ == "__main__":
    raise SystemExit(main())
