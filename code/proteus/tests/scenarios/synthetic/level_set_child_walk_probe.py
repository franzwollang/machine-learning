"""Dissect child-level splits on the normal path (OPEN_ISSUES #48).

Replays ``run_recursive_discovery`` for sweep scenes, prints the tree,
and for every internal non-root node re-derives the L=1 selection and
(if needed) the finer walk that accepted the child's split.

Not a pytest test.

Usage::

    PYTHONPATH="src:$PWD" pipenv run python \\
        tests/scenarios/synthetic/level_set_child_walk_probe.py \\
        --seed 0 --scenes bimodal_circle two_gaussians_weak

    PYTHONPATH="src:$PWD" pipenv run python \\
        tests/scenarios/synthetic/level_set_child_walk_probe.py \\
        --seed 0 --scenes nested_spheres
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import proteus.stage1.recursion as recursion_mod
from proteus.stage1.controller import run_scale_search
from proteus.stage1.level_set import (
    LevelSetConfig,
    at_shot_noise_scale,
    mean_neighbor_radius,
    select_level_set_partition,
)
from proteus.stage1.recursion import (
    RecursionConfig,
    RecursionNode,
    RecursionTree,
    _research_finer_split,
    assign_samples_to_clusters,
    run_recursive_discovery,
)
from tests.scenarios.synthetic.level_set_normal_path_sweep import _config, _scenes


K_NEIGHBORS = 8


def _gt_frac_str(sample_indices: np.ndarray, labels: np.ndarray) -> str:
    idx = np.asarray(sample_indices, dtype=int)
    if idx.size == 0:
        return "empty"
    lab = np.asarray(labels, dtype=int)[idx]
    n = float(lab.size)
    parts: list[str] = []
    for value in sorted({int(v) for v in lab if v >= 0}):
        count = int(np.sum(lab == value))
        parts.append(f"L{value}={count}({count / n:.3f})")
    n_bg = int(np.sum(lab < 0))
    parts.append(f"bg={n_bg}({n_bg / n:.3f})")
    return " ".join(parts)


def _print_tree(tree: RecursionTree, labels: np.ndarray, scene: str) -> None:
    print("=" * 78)
    print(f"TREE  {scene}")
    print("=" * 78)
    print(
        f"{'id':>4} {'lvl':>3} {'par':>4} {'n':>6} {'bg':>3} "
        f"{'tau*':>12} {'K':>3} {'ch':>3} {'leaf':>4}  gt",
        flush=True,
    )
    for node in tree.nodes:
        tau = "None" if node.tau_star is None else f"{float(node.tau_star):.6g}"
        parent = "-" if node.parent_id is None else str(int(node.parent_id))
        print(
            f"{int(node.region_id):4d} {int(node.level):3d} {parent:>4} "
            f"{int(node.n_samples):6d} {int(node.is_background):3d} "
            f"{tau:>12} {int(node.n_clusters):3d} {len(node.children):3d} "
            f"{int(node.is_leaf):4d}  {_gt_frac_str(node.sample_indices, labels)}",
            flush=True,
        )
    n_sig_leaves = sum(1 for leaf in tree.leaves if not leaf.is_background)
    print(
        f"nodes={len(tree.nodes)} depth={tree.depth} leaves={len(tree.leaves)} "
        f"signal_leaves={n_sig_leaves}",
        flush=True,
    )


def _selection_fields(selection, scaffold, points: np.ndarray) -> dict[str, Any]:
    cr = selection.cluster_result
    k_val = 0 if cr is None else int(cr.n_clusters)
    rv = selection.resolvability
    verdict = None if rv is None else rv.verdict.value
    reason = None if rv is None else rv.reject_reason
    tau = getattr(scaffold, "tau", None)
    n_nodes = 0 if scaffold is None else len(getattr(scaffold, "nodes", ()))
    positions = np.asarray(
        [node.position for node in getattr(scaffold, "nodes", ())],
        dtype=float,
    ) if scaffold is not None else np.empty((0, 1))
    r_k = (
        mean_neighbor_radius(positions, K_NEIGHBORS)
        if positions.shape[0] >= 2 else None
    )
    shot = bool(at_shot_noise_scale(scaffold, points, K_NEIGHBORS))
    return {
        "accepted": bool(selection.accepted),
        "K": k_val,
        "verdict": verdict,
        "reason": reason,
        "phi": selection.bottleneck_ratio,
        "rho": selection.studentized_ratio,
        "logBF": float(selection.log_bf),
        "N": n_nodes,
        "tau": None if tau is None else float(tau),
        "r_k": r_k,
        "at_shot_noise_scale": shot,
        "candidate_level": selection.candidate_level,
        "selected_level": selection.selected_level,
    }


def _fmt_sel(fields: dict[str, Any]) -> str:
    tau = fields["tau"]
    tau_s = "na" if tau is None else f"{tau:.6g}"
    phi = fields["phi"]
    rho = fields["rho"]
    r_k = fields["r_k"]
    return (
        f"accepted={int(fields['accepted'])} K={fields['K']} "
        f"verdict={fields['verdict']} reason={fields['reason']} "
        f"phi={phi if phi is None else f'{float(phi):.6g}'} "
        f"rho={rho if rho is None else f'{float(rho):.6g}'} "
        f"logBF={fields['logBF']:.6g} N={fields['N']} "
        f"tau={tau_s} "
        f"r_k={r_k if r_k is None else f'{float(r_k):.6g}'} "
        f"at_shot_noise_scale={int(fields['at_shot_noise_scale'])} "
        f"candL={fields['candidate_level']} selL={fields['selected_level']}"
    )


def _child_config(config: RecursionConfig, node: RecursionNode) -> RecursionConfig:
    return replace(
        config,
        scale_search=replace(config.scale_search, max_nodes=None),
        level_set=replace(config.level_set, grow_nodes_when_underresolved=False),
        allow_finer_research=config.allow_finer_research,
        seed=int(config.seed) + int(node.region_id),
    )


def _print_cluster_gt(
    name: str,
    points: np.ndarray,
    scaffold,
    selection,
    orig_indices: np.ndarray,
    labels: np.ndarray,
) -> None:
    if not selection.accepted or selection.cluster_result is None:
        print(f"{name}: no accepted clusters to map", flush=True)
        return
    sample_map = assign_samples_to_clusters(
        points, scaffold, selection.cluster_result.labels,
    )
    print(f"{name} accepted-cluster GT composition:", flush=True)
    for label, local_idx in sorted(sample_map.items()):
        local = np.asarray(local_idx, dtype=int)
        orig = np.asarray(orig_indices, dtype=int)[local]
        lab = np.asarray(labels, dtype=int)[orig]
        n1 = int(np.sum(lab == 1))
        n2 = int(np.sum(lab == 2))
        n_bg = int(np.sum(lab < 0))
        other = Counter(int(v) for v in lab if v >= 0 and v not in (1, 2))
        extra = "" if not other else f" other={dict(other)}"
        print(
            f"  cluster_label={int(label)} n={int(local.size)} "
            f"GT1={n1} GT2={n2} tissue={n_bg}{extra}",
            flush=True,
        )


def _instrumented_finer_split(
    points: np.ndarray,
    dim: int,
    child_config: RecursionConfig,
    parent_tau: float,
    parent_scaffold,
) -> None:
    original = recursion_mod.select_level_set_partition
    step_i = {"n": 0}

    def wrapper(scaffold, config=None, dm_config=None, data=None):
        selection = original(scaffold, config, dm_config, data)
        data_arr = points if data is None else np.asarray(data, dtype=float)
        fields = _selection_fields(selection, scaffold, data_arr)
        step_i["n"] += 1
        print(
            f"  FINER_STEP {step_i['n']:02d} {_fmt_sel(fields)}",
            flush=True,
        )
        return selection

    recursion_mod.select_level_set_partition = wrapper
    try:
        out = _research_finer_split(
            points, dim, child_config,
            parent_tau=float(parent_tau),
            parent_scaffold=parent_scaffold,
        )
    finally:
        recursion_mod.select_level_set_partition = original
    if out is None:
        print("  finer walk returned None (no accepted split)", flush=True)
    else:
        print("  finer walk returned an accepted split", flush=True)


def _dissect_internal(
    node: RecursionNode,
    parent: RecursionNode,
    points_full: np.ndarray,
    labels: np.ndarray,
    dim: int,
    config: RecursionConfig,
    scene: str,
) -> None:
    idx = np.asarray(node.sample_indices, dtype=int)
    points = np.asarray(points_full[idx], dtype=float)
    child_cfg = _child_config(config, node)
    parent_tau = parent.tau_star
    print("-" * 78)
    print(
        f"INTERNAL node id={int(node.region_id)} level={int(node.level)} "
        f"parent={int(parent.region_id)} n={int(node.n_samples)} "
        f"node.tau*={node.tau_star} parent.tau*={parent_tau} "
        f"n_clusters={int(node.n_clusters)} n_children={len(node.children)}",
        flush=True,
    )
    print(f"  gt {_gt_frac_str(idx, labels)}", flush=True)

    t0 = time.time()
    result = run_scale_search(points, int(dim), child_cfg.scale_search)
    scaffold = result.scaffold_at_star
    search_s = time.time() - t0
    print(
        f"  run_scale_search tau*={result.tau_star} t={search_s:.1f}s",
        flush=True,
    )
    if scaffold is None or len(getattr(scaffold, "nodes", ())) < 2:
        print("  empty/tiny scaffold; skip L=1 / finer", flush=True)
        return

    t1 = time.time()
    selection = select_level_set_partition(
        scaffold, LevelSetConfig(), config.dm_cluster, data=points,
    )
    fields = _selection_fields(selection, scaffold, points)
    print(f"  L=1 {_fmt_sel(fields)} select_t={time.time() - t1:.1f}s", flush=True)

    if scene == "nested_spheres" and node.parent_id == 0:
        _print_cluster_gt(
            f"  nested L=1 child id={int(node.region_id)}",
            points, scaffold, selection, idx, labels,
        )

    if selection.accepted:
        print("  split source: L=1 accept (not finer walk)", flush=True)
        return

    print("  L=1 not accepted; replaying finer walk", flush=True)
    tau_for_walk = node.tau_star
    if tau_for_walk is None:
        tau_for_walk = result.tau_star
    if tau_for_walk is None:
        print("  no tau* for finer walk; skip", flush=True)
        return
    _instrumented_finer_split(
        points, int(dim), child_cfg,
        parent_tau=float(tau_for_walk),
        parent_scaffold=scaffold,
    )


def _majority_gt(sample_indices: np.ndarray, labels: np.ndarray) -> int | None:
    idx = np.asarray(sample_indices, dtype=int)
    if idx.size == 0:
        return None
    lab = np.asarray(labels, dtype=int)[idx]
    signal = lab[lab >= 0]
    if signal.size == 0:
        return None
    values, counts = np.unique(signal, return_counts=True)
    return int(values[int(np.argmax(counts))])


def run_scene(scene_name: str, args: argparse.Namespace) -> None:
    scenes = {s.name: s for s in _scenes()}
    if scene_name not in scenes:
        raise SystemExit(
            f"unknown scene {scene_name}; known: {sorted(scenes)}"
        )
    scene = scenes[scene_name]
    seed = int(args.seed)
    data = scene.factory(seed)
    points = np.asarray(data.points, dtype=float)
    labels = np.asarray(data.labels, dtype=int)
    dim = int(data.ground_truth.ambient_dim)
    config = _config(args, seed)
    print(
        f"\n## scene={scene_name} seed={seed} n={points.shape[0]} dim={dim} "
        f"finer_steps={args.max_finer_steps}",
        flush=True,
    )
    t0 = time.time()
    tree = run_recursive_discovery(points, dim=dim, config=config)
    print(f"run_recursive_discovery t={time.time() - t0:.1f}s", flush=True)
    _print_tree(tree, labels, scene_name)

    if not tree.nodes:
        print("empty tree", flush=True)
        return

    by_id = {int(n.region_id): n for n in tree.nodes}
    internals = [
        n for n in tree.nodes
        if n.parent_id is not None and not n.is_leaf and not n.is_background
    ]
    print(
        f"internal non-root non-bg nodes: {len(internals)} "
        f"ids={[int(n.region_id) for n in internals]}",
        flush=True,
    )
    for node in internals:
        parent = by_id[int(node.parent_id)]
        _dissect_internal(node, parent, points, labels, dim, config, scene_name)

    if scene_name == "nested_spheres" and tree.nodes[0].children:
        root = tree.nodes[0]
        print("-" * 78)
        print("NESTED extra: root-child majority GT + L=1 cluster map", flush=True)
        for cid in root.children:
            child = by_id[int(cid)]
            maj = _majority_gt(child.sample_indices, labels)
            print(
                f"  root child id={int(child.region_id)} "
                f"bg={int(child.is_background)} leaf={int(child.is_leaf)} "
                f"majority_gt={maj} {_gt_frac_str(child.sample_indices, labels)}",
                flush=True,
            )
            if child.is_background or maj != 2:
                continue
            if child in internals:
                continue
            idx = np.asarray(child.sample_indices, dtype=int)
            child_pts = points[idx]
            child_cfg = _child_config(config, child)
            result = run_scale_search(child_pts, dim, child_cfg.scale_search)
            scaffold = result.scaffold_at_star
            if scaffold is None or len(scaffold.nodes) < 2:
                print("  outer-shell child: empty scaffold at L=1", flush=True)
                continue
            selection = select_level_set_partition(
                scaffold, LevelSetConfig(), config.dm_cluster, data=child_pts,
            )
            fields = _selection_fields(selection, scaffold, child_pts)
            print(f"  outer-shell L=1 {_fmt_sel(fields)}", flush=True)
            _print_cluster_gt(
                "  outer-shell child",
                child_pts, scaffold, selection, idx, labels,
            )


def _run_as_parent(args: argparse.Namespace, wanted: list[str]) -> int:
    script = str(Path(__file__).resolve())
    rc = 0
    for name in wanted:
        print(f"# start worker scene={name} seed={args.seed}", flush=True)
        cmd = [
            sys.executable,
            script,
            "--worker",
            "--seed",
            str(int(args.seed)),
            "--scenes",
            name,
            "--max-epochs",
            str(int(args.max_epochs)),
            "--max-grid-points",
            str(int(args.max_grid_points)),
            "--max-finer-steps",
            str(int(args.max_finer_steps)),
        ]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(
                f"# worker {name} seed={args.seed} exit={proc.returncode} "
                "(139=SIGSEGV)",
                flush=True,
            )
            rc = 1
    return rc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--max-grid-points", type=int, default=8)
    parser.add_argument("--max-finer-steps", type=int, default=16)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    known = [s.name for s in _scenes()]
    wanted = list(known) if args.scenes is None else list(args.scenes)
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise SystemExit(f"unknown scenes: {unknown}; known: {known}")

    print(
        f"level_set_child_walk_probe seed={args.seed} "
        f"max_epochs={args.max_epochs} grid={args.max_grid_points} "
        f"max_finer_scale_steps={args.max_finer_steps} "
        f"scenes={wanted} worker={int(args.worker)}",
        flush=True,
    )
    if not args.worker:
        return _run_as_parent(args, wanted)
    for name in wanted:
        run_scene(name, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
