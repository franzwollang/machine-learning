"""Instrument root-region ``select_level_set_partition`` (OPEN_ISSUES #48).

Monkeypatches ``proteus.stage1.recursion.select_level_set_partition`` and
runs ``run_recursive_discovery`` at ``max_depth=1`` so only the root is
processed.  Prints one line per selection call (tau, N, hits, r_k, φ/ρ/BF).

Not a pytest test.

Usage::

    PYTHONPATH="src:$PWD" pipenv run python \\
        tests/scenarios/synthetic/level_set_root_accept_probe.py --seed 0
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from typing import Any

import numpy as np

import proteus.stage1.recursion as recursion_mod
from proteus.stage1.level_set import at_shot_noise_scale, mean_neighbor_radius
from proteus.stage1.recursion import run_recursive_discovery
from tests.scenarios.synthetic.level_set_normal_path_sweep import _config, _scenes


K_NEIGHBORS = 8
DEFAULT_SCENES = ("linked_tori", "nested_spheres")


def _fmt(value: Any, digits: str = ".6g") -> str:
    if value is None:
        return "na"
    try:
        return format(float(value), digits)
    except (TypeError, ValueError):
        return str(value)


def _hit_stats(scaffold) -> tuple[float | None, float | None, float | None]:
    nodes = getattr(scaffold, "nodes", ())
    hits = np.asarray(
        [float(getattr(node, "hit_count", np.nan)) for node in nodes],
        dtype=float,
    )
    if hits.size == 0 or not np.isfinite(hits).any():
        return None, None, None
    finite = hits[np.isfinite(hits)]
    return float(finite.mean()), float(finite.min()), float(np.median(finite))


def _node_rk(scaffold) -> float | None:
    positions = np.asarray(
        [node.position for node in getattr(scaffold, "nodes", ())],
        dtype=float,
    )
    if positions.ndim != 2 or positions.shape[0] < 2:
        return None
    return float(mean_neighbor_radius(positions, K_NEIGHBORS))


def run_scene(scene_name: str, args: argparse.Namespace) -> None:
    scenes = {s.name: s for s in _scenes()}
    if scene_name not in scenes:
        raise SystemExit(f"unknown scene {scene_name}; known: {sorted(scenes)}")
    scene = scenes[scene_name]
    seed = int(args.seed)
    data = scene.factory(seed)
    points = np.asarray(data.points, dtype=float)
    dim = int(data.ground_truth.ambient_dim)
    n_full = int(points.shape[0])
    config = replace(_config(args, seed), max_depth=int(args.max_depth))

    print("=" * 78, flush=True)
    print(
        f"SCENE {scene_name} seed={seed} n={n_full} dim={dim} "
        f"max_depth={config.max_depth} finer_steps={args.max_finer_steps}",
        flush=True,
    )
    print("=" * 78, flush=True)

    original = recursion_mod.select_level_set_partition
    records: list[dict[str, Any]] = []

    def wrapper(scaffold, config=None, dm_config=None, data=None):
        selection = original(scaffold, config, dm_config, data)
        data_arr = points if data is None else np.asarray(data, dtype=float)
        n_samples = int(data_arr.shape[0])
        n_nodes = len(getattr(scaffold, "nodes", ()))
        tau = getattr(scaffold, "tau", None)
        max_nodes = getattr(scaffold, "max_nodes", None)
        hits_mean, hits_min, hits_med = _hit_stats(scaffold)
        node_rk = _node_rk(scaffold)
        sample_rk = (
            float(mean_neighbor_radius(data_arr, K_NEIGHBORS))
            if n_samples >= 2 else None
        )
        shot = bool(at_shot_noise_scale(scaffold, data_arr, K_NEIGHBORS))
        cr = selection.cluster_result
        k_val = 0 if cr is None else int(cr.n_clusters)
        rv = selection.resolvability
        verdict = None if rv is None else rv.verdict.value
        reason = None if rv is None else rv.reject_reason
        accepted = bool(selection.accepted)
        mean_hits = (n_samples / n_nodes) if n_nodes else None
        region = "ROOT" if n_samples == n_full else "CHILD"
        rec = {
            "region": region,
            "n_samples": n_samples,
            "N": n_nodes,
            "max_nodes": max_nodes,
            "mean_hits_nN": mean_hits,
            "hits_mean": hits_mean,
            "hits_min": hits_min,
            "hits_med": hits_med,
            "tau": None if tau is None else float(tau),
            "shot": shot,
            "node_rk": node_rk,
            "sample_rk": sample_rk,
            "accepted": accepted,
            "K": k_val,
            "verdict": verdict,
            "reason": reason,
            "phi": selection.bottleneck_ratio,
            "rho": selection.studentized_ratio,
            "logBF": float(selection.log_bf),
        }
        records.append(rec)
        step = len(records)
        mark = "  <--- ACCEPTED" if accepted else ""
        print(
            f"STEP {step:02d} {region} n={n_samples} N={n_nodes} "
            f"max_nodes={_fmt(max_nodes, '.0f')} "
            f"n/N={_fmt(mean_hits)} "
            f"hits_mean={_fmt(hits_mean)} hits_min={_fmt(hits_min)} "
            f"hits_med={_fmt(hits_med)} "
            f"tau={_fmt(rec['tau'])} "
            f"at_shot_noise_scale={int(shot)} "
            f"node_rk={_fmt(node_rk)} sample_rk={_fmt(sample_rk)} "
            f"accepted={int(accepted)} K={k_val} "
            f"verdict={verdict} reason={reason} "
            f"phi={_fmt(rec['phi'])} rho={_fmt(rec['rho'])} "
            f"logBF={_fmt(rec['logBF'])}{mark}",
            flush=True,
        )
        return selection

    recursion_mod.select_level_set_partition = wrapper
    t0 = time.time()
    try:
        tree = run_recursive_discovery(points, dim=dim, config=config)
    finally:
        recursion_mod.select_level_set_partition = original
    elapsed = time.time() - t0

    root = tree.nodes[0] if tree.nodes else None
    root_k = 0 if root is None else int(root.n_clusters)
    accept = next((r for r in records if r["accepted"] and r["region"] == "ROOT"), None)
    if accept is None:
        accept = next((r for r in records if r["accepted"]), None)

    print("-" * 78, flush=True)
    if accept is None:
        print(
            f"SUMMARY {scene_name} root_K={root_k} accepted=0 "
            f"n_calls={len(records)} elapsed={elapsed:.1f}s",
            flush=True,
        )
    else:
        print(
            f"SUMMARY {scene_name} root_K={root_k} "
            f"accepted_N={accept['N']} n/N={_fmt(accept['mean_hits_nN'])} "
            f"hits_mean={_fmt(accept['hits_mean'])} "
            f"hits_min={_fmt(accept['hits_min'])} "
            f"phi={_fmt(accept['phi'])} rho={_fmt(accept['rho'])} "
            f"logBF={_fmt(accept['logBF'])} "
            f"at_shot={int(accept['shot'])} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    if root is None:
        print("ROOT_CHILDREN (none; empty tree)", flush=True)
    elif not root.children:
        print(
            f"ROOT_CHILDREN (none) root_leaf={int(root.is_leaf)} "
            f"tree_depth={tree.depth} n_nodes={len(tree.nodes)}",
            flush=True,
        )
    else:
        by_id = {int(n.region_id): n for n in tree.nodes}
        parts = []
        for cid in root.children:
            child = by_id[int(cid)]
            parts.append(
                f"id={int(child.region_id)} n={int(child.n_samples)} "
                f"bg={int(child.is_background)} leaf={int(child.is_leaf)} "
                f"tau*={child.tau_star} K={int(child.n_clusters)}"
            )
        print(
            f"ROOT_CHILDREN n={len(root.children)} "
            f"tree_depth={tree.depth} " + "; ".join(parts),
            flush=True,
        )
        child_searched = any(
            by_id[int(cid)].tau_star is not None for cid in root.children
        )
        print(
            f"max_depth={config.max_depth} child_run_scale_search="
            f"{int(child_searched)} "
            f"(tau*=None means child stopped before scale search)",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--max-grid-points", type=int, default=8)
    parser.add_argument("--max-finer-steps", type=int, default=16)
    parser.add_argument(
        "--growth-policy",
        choices=("track_tau",),
        default="track_tau",
    )
    args = parser.parse_args()
    known = [s.name for s in _scenes()]
    wanted = list(args.scenes)
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise SystemExit(f"unknown scenes: {unknown}; known: {known}")

    print(
        f"level_set_root_accept_probe seed={args.seed} "
        f"max_depth={args.max_depth} max_epochs={args.max_epochs} "
        f"grid={args.max_grid_points} finer_steps={args.max_finer_steps} "
        f"growth_policy={args.growth_policy} scenes={wanted}",
        flush=True,
    )
    for name in wanted:
        run_scene(name, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
