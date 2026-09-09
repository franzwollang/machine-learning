"""Dissect false-positive splits of isotropic Gaussian children.

Normal-path ``two_gaussians_clear`` seed 0 finds root K=2 (correct) then
splits each Gaussian child again.  This probe replays the recursion tree
and mirrors ``select_level_set_partition`` on those children plus lone-
Gaussian controls.  Measurements only.

Usage::

    PYTHONPATH="src:$PWD" pipenv run python \\
        tests/scenarios/synthetic/level_set_gaussian_child_probe.py
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.level_set import (
    LevelSetConfig,
    _filter_relative_mass,
    _flow_bottleneck_ratio,
    _label_sets,
    apply_dm_sibling_collapse,
    apply_geometric_screens,
    at_shot_noise_scale,
    build_level_set_dag,
    build_level_set_tree,
    mean_neighbor_radius,
    null_bottleneck_ratio,
    select_level_set_partition,
    studentized_bottleneck,
)
from proteus.stage1.recursion import RecursionConfig, RecursionTree, run_recursive_discovery
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.density_valleys import make_two_gaussians
from tests.scenarios.synthetic.level_set_suite import (
    TWO_GAUSSIANS_CLEAR_SEP,
    TWO_GAUSSIANS_N,
    TWO_GAUSSIANS_SIGMA,
)


JSON_PATH = "/tmp/gaussian_child_probe_seed0.json"


def _config(args: argparse.Namespace, seed: int) -> RecursionConfig:
    scale = ScaleSearchConfig(
        selector="load_crossover",
        tau_min=1e-5,
        tau_max=10.0,
        max_grid_points=int(args.max_grid_points),
        k=8,
        min_nodes=4,
        n_seeds=8,
        max_nodes=None,
        stabilization=StabilizationConfig(
            min_equilibrium_epochs=3,
            max_epochs=int(args.max_epochs),
        ),
        seed=int(seed),
    )
    return RecursionConfig(
        scale_search=scale,
        min_samples=100,
        max_depth=5,
        use_level_set_clustering=True,
        allow_finer_research=True,
        max_finer_scale_steps=16,
        level_set=LevelSetConfig(),
        seed=int(seed),
    )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return str(v) if not np.isfinite(v) else v
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _gt_fractions(sample_indices: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    idx = np.asarray(sample_indices, dtype=int)
    if idx.size == 0:
        return {"frac0": 0.0, "frac1": 0.0, "frac_bg": 0.0, "n0": 0, "n1": 0, "n_bg": 0}
    lab = labels[idx]
    n = float(lab.size)
    n0 = int(np.sum(lab == 0))
    n1 = int(np.sum(lab == 1))
    n_bg = int(np.sum(lab < 0))
    return {
        "frac0": n0 / n,
        "frac1": n1 / n,
        "frac_bg": n_bg / n,
        "n0": n0,
        "n1": n1,
        "n_bg": n_bg,
    }


def _print_tree(tree: RecursionTree, labels: np.ndarray) -> None:
    print("=" * 78)
    print("PART 1 — recursion tree replay (two_gaussians_clear)")
    print("=" * 78)
    print(
        f"{'id':>4} {'lvl':>3} {'par':>4} {'n':>5} {'bg':>3} "
        f"{'tau*':>12} {'ch':>3} {'frac0':>7} {'frac1':>7} {'frac_bg':>8} "
        f"{'n0':>4} {'n1':>4} {'n_bg':>4} leaf",
        flush=True,
    )
    for node in tree.nodes:
        fr = _gt_fractions(node.sample_indices, labels)
        tau = "None" if node.tau_star is None else f"{float(node.tau_star):.6g}"
        parent = "-" if node.parent_id is None else str(int(node.parent_id))
        print(
            f"{int(node.region_id):4d} {int(node.level):3d} {parent:>4} "
            f"{int(node.n_samples):5d} {int(node.is_background):3d} "
            f"{tau:>12} {len(node.children):3d} "
            f"{fr['frac0']:7.3f} {fr['frac1']:7.3f} {fr['frac_bg']:8.3f} "
            f"{fr['n0']:4d} {fr['n1']:4d} {fr['n_bg']:4d} "
            f"{int(node.is_leaf)}",
            flush=True,
        )
    n_sig_leaves = sum(1 for leaf in tree.leaves if not leaf.is_background)
    print(
        f"nodes={len(tree.nodes)} depth={tree.depth} leaves={len(tree.leaves)} "
        f"signal_leaves={n_sig_leaves}",
        flush=True,
    )


def _cluster_sizes(labels: np.ndarray) -> dict[str, Any]:
    labs = np.asarray(labels, dtype=int)
    keys = sorted(set(int(v) for v in labs if v >= 0))
    sizes = [int(np.sum(labs == k)) for k in keys]
    return {
        "k": len(keys),
        "sizes": sizes,
        "n_background": int(np.sum(labs < 0)),
    }


def _dissect(
    name: str,
    points: np.ndarray,
    dim: int,
    config: RecursionConfig,
    *,
    max_epochs_override: int | None = None,
) -> dict[str, Any]:
    print("-" * 78)
    print(f"DISSECT  {name}  N={int(points.shape[0])} dim={int(dim)}")
    print("-" * 78, flush=True)
    scale = config.scale_search
    if max_epochs_override is not None:
        from dataclasses import replace
        scale = replace(
            scale,
            stabilization=replace(
                scale.stabilization,
                max_epochs=int(max_epochs_override),
            ),
        )
        print(f"NOTE: max_epochs reduced to {max_epochs_override} for this control", flush=True)

    t0 = time.time()
    result = run_scale_search(np.asarray(points, dtype=float), int(dim), scale)
    search_s = time.time() - t0
    scaffold = result.scaffold_at_star
    ls_cfg = config.level_set
    k_nn = int(ls_cfg.k_neighbors)

    if scaffold is None or len(scaffold.nodes) < 2:
        rec: dict[str, Any] = {
            "name": name,
            "n_samples": int(points.shape[0]),
            "dim": int(dim),
            "n_nodes": 0 if scaffold is None else len(scaffold.nodes),
            "max_nodes": None if scaffold is None else getattr(scaffold, "max_nodes", None),
            "tau_star": float(result.tau_star),
            "search_s": search_s,
            "empty_scaffold": True,
            "accepted": False,
        }
        print(f"empty/tiny scaffold after search t={search_s:.1f}s", flush=True)
        return rec

    positions = np.asarray([node.position for node in scaffold.nodes], dtype=float)
    n_nodes = int(positions.shape[0])
    max_nodes = getattr(scaffold, "max_nodes", None)
    tau_star = float(result.tau_star)
    rk = mean_neighbor_radius(positions, k_nn)
    at_shot = bool(at_shot_noise_scale(scaffold, np.asarray(points, dtype=float), k_nn))
    rk_data = mean_neighbor_radius(np.asarray(points, dtype=float), k_nn)

    t1 = time.time()
    selection = select_level_set_partition(scaffold, LevelSetConfig(), config.dm_cluster)
    select_s = time.time() - t1

    cr = selection.cluster_result
    cr_k = 0 if cr is None else int(cr.n_clusters)
    cr_sizes = None if cr is None else _cluster_sizes(cr.labels)
    rv = selection.resolvability
    verdict = None if rv is None else rv.verdict.value
    reject = None if rv is None else rv.reject_reason

    phi = selection.bottleneck_ratio
    phi_null = selection.null_bottleneck_ratio
    rho = selection.studentized_ratio
    if phi is not None and (phi_null is None or rho is None) and cr is not None:
        phi_null = null_bottleneck_ratio(scaffold, positions, cr.labels)
        rho = studentized_bottleneck(float(phi), phi_null)

    print(
        f"scaffold N_nodes={n_nodes} max_nodes={max_nodes} tau*={tau_star:.6g} "
        f"mean_r_k(k={k_nn})={rk:.6g} data_r_k={rk_data:.6g} "
        f"at_shot_noise_scale={at_shot} search_t={search_s:.1f}s",
        flush=True,
    )
    print(
        f"selection accepted={int(selection.accepted)} K={cr_k} "
        f"cluster_sizes={None if cr_sizes is None else cr_sizes['sizes']} "
        f"n_bg_nodes={None if cr_sizes is None else cr_sizes['n_background']} "
        f"candidate_level={selection.candidate_level} "
        f"selected_level={selection.selected_level}",
        flush=True,
    )
    phi_s = "na" if phi is None else f"{float(phi):.6g}"
    phi0_s = "na" if phi_null is None else f"{float(phi_null):.6g}"
    rho_s = "na" if rho is None else f"{float(rho):.6g}"
    ceil_phi = float(ls_cfg.max_bottleneck_ratio)
    print(
        f"phi={phi_s}  phi_0={phi0_s}  rho=phi/phi_0={rho_s}  "
        f"ceiling={ceil_phi}  log_bf={float(selection.log_bf):.6g}  "
        f"verdict={verdict} reject_reason={reject}  select_t={select_s:.1f}s",
        flush=True,
    )
    if phi is not None:
        print(
            f"  phi_below_ceiling={float(phi) <= ceil_phi}  "
            f"rho_below_ceiling="
            f"{'na' if rho is None else bool(float(rho) <= ceil_phi)}  "
            f"logBF_positive={float(selection.log_bf) > 0.0}",
            flush=True,
        )

    # Mirror select_level_set_partition helper order.
    ls_tree = build_level_set_tree(positions, ls_cfg)
    dag = build_level_set_dag(ls_tree)
    geom = apply_geometric_screens(dag.branches, ls_cfg)
    collapsed = apply_dm_sibling_collapse(
        ls_tree, dag, geom, scaffold, config.dm_cluster,
    )

    n_levels = len(ls_tree.levels)
    k_per_level = [int(lvl.n_clusters) for lvl in ls_tree.levels]
    n_dag = len(dag.branches)
    n_geom_live = sum(1 for b in geom if b.prune_reason is None)
    n_geom_pers = sum(1 for b in geom if b.prune_reason == "persistence")
    n_geom_mass = sum(1 for b in geom if b.prune_reason == "excess_mass")
    n_dm_live = sum(1 for b in collapsed if b.prune_reason is None)
    n_dm_pruned = sum(1 for b in collapsed if b.prune_reason == "dm")

    print(
        f"C-D tree: n_levels={n_levels} K_per_level={k_per_level}",
        flush=True,
    )
    print(
        f"DAG branches={n_dag}  survive_geometric_screens={n_geom_live} "
        f"(pruned persistence={n_geom_pers} excess_mass={n_geom_mass})  "
        f"after_dm_sibling_collapse live={n_dm_live} dm_pruned={n_dm_pruned}",
        flush=True,
    )

    floor = None
    if n_nodes > 0:
        from math import ceil
        floor = max(int(ls_cfg.min_cluster_size), int(ceil(float(ls_cfg.min_cluster_frac) * n_nodes)))
    print(
        f"mass filter min_cluster_frac={ls_cfg.min_cluster_frac} "
        f"min_cluster_size={ls_cfg.min_cluster_size} floor={floor}",
        flush=True,
    )
    print(
        f"{'lvl':>4} {'r':>10} {'K_raw':>6} {'K_mass':>7} {'sizes_mass':>20} "
        f"{'n_bg':>5} {'balanced':>8}",
        flush=True,
    )
    mass_rows: list[dict[str, Any]] = []
    coarsest_balanced: int | None = None
    for level_index in range(n_levels - 1, -1, -1):
        lvl = ls_tree.levels[level_index]
        filtered = _filter_relative_mass(
            lvl.labels, ls_cfg.min_cluster_size, ls_cfg.min_cluster_frac,
        )
        clusters, background = _label_sets(filtered)
        k_mass = len(clusters)
        sizes = [len(c) for c in clusters]
        balanced = k_mass >= 2
        if balanced:
            coarsest_balanced = level_index
        mass_rows.append(
            {
                "level": level_index,
                "radius": float(lvl.radius),
                "k_raw": int(lvl.n_clusters),
                "k_mass": k_mass,
                "sizes": sizes,
                "n_background": len(background),
                "balanced": balanced,
            }
        )
    # Print coarse-to-fine to match the selection walk.
    for row in mass_rows:
        print(
            f"{row['level']:4d} {row['radius']:10.5g} {row['k_raw']:6d} "
            f"{row['k_mass']:7d} {str(row['sizes']):>20} "
            f"{row['n_background']:5d} {int(row['balanced']):8d}",
            flush=True,
        )

    helper_phi = None
    helper_phi0 = None
    helper_rho = None
    helper_labels = None
    if coarsest_balanced is not None:
        cand = ls_tree.levels[coarsest_balanced]
        helper_labels = _filter_relative_mass(
            cand.labels, ls_cfg.min_cluster_size, ls_cfg.min_cluster_frac,
        )
        helper_phi = float(_flow_bottleneck_ratio(scaffold, helper_labels, positions))
        helper_phi0 = null_bottleneck_ratio(scaffold, positions, helper_labels)
        helper_rho = studentized_bottleneck(helper_phi, helper_phi0)
        print(
            f"helper coarsest mass-filtered K>=2 level={coarsest_balanced} "
            f"phi={helper_phi:.6g} phi_0="
            f"{'na' if helper_phi0 is None else f'{float(helper_phi0):.6g}'} "
            f"rho={'na' if helper_rho is None else f'{float(helper_rho):.6g}'}",
            flush=True,
        )
    else:
        print("helper: no mass-filtered K>=2 cut at any level", flush=True)

    rec = {
        "name": name,
        "n_samples": int(points.shape[0]),
        "dim": int(dim),
        "n_nodes": n_nodes,
        "max_nodes": None if max_nodes is None else int(max_nodes),
        "tau_star": tau_star,
        "mean_r_k": rk,
        "data_r_k": rk_data,
        "at_shot_noise_scale": at_shot,
        "accepted": bool(selection.accepted),
        "n_clusters": cr_k,
        "cluster_sizes": None if cr_sizes is None else cr_sizes["sizes"],
        "n_background_nodes": None if cr_sizes is None else cr_sizes["n_background"],
        "candidate_level": selection.candidate_level,
        "selected_level": selection.selected_level,
        "phi": phi,
        "phi_0": phi_null,
        "rho": rho,
        "log_bf": float(selection.log_bf),
        "verdict": verdict,
        "reject_reason": reject,
        "phi_below_ceiling": None if phi is None else bool(float(phi) <= ceil_phi),
        "rho_below_ceiling": None if rho is None else bool(float(rho) <= ceil_phi),
        "logBF_positive": bool(float(selection.log_bf) > 0.0),
        "ceiling": ceil_phi,
        "tree_n_levels": n_levels,
        "k_per_level": k_per_level,
        "dag_n_branches": n_dag,
        "survive_geometric_screens": n_geom_live,
        "geom_pruned_persistence": n_geom_pers,
        "geom_pruned_excess_mass": n_geom_mass,
        "dm_collapse_live": n_dm_live,
        "dm_collapse_pruned": n_dm_pruned,
        "mass_floor": floor,
        "coarsest_balanced_level": coarsest_balanced,
        "mass_rows": mass_rows,
        "helper_phi": helper_phi,
        "helper_phi_0": helper_phi0,
        "helper_rho": helper_rho,
        "search_s": search_s,
        "select_s": select_s,
        "empty_scaffold": False,
        "max_epochs_used": int(scale.stabilization.max_epochs),
    }
    return rec


def _label0_plus_nearby_tissue(
    points: np.ndarray,
    labels: np.ndarray,
    radius: float = 1.0,
) -> tuple[np.ndarray, dict[str, int]]:
    idx0 = np.where(labels == 0)[0]
    idx_tissue = np.where(labels < 0)[0]
    if idx0.size == 0 or idx_tissue.size == 0:
        keep_tissue = np.empty(0, dtype=int)
    else:
        dists, _ = cKDTree(points[idx0]).query(points[idx_tissue], k=1)
        keep_tissue = idx_tissue[np.asarray(dists) <= float(radius)]
    subset = np.concatenate([idx0, keep_tissue])
    meta = {
        "n_label0": int(idx0.size),
        "n_tissue_total": int(idx_tissue.size),
        "n_tissue_kept": int(keep_tissue.size),
        "n_subset": int(subset.size),
    }
    return points[subset], meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--max-grid-points", type=int, default=8)
    args = parser.parse_args()
    seed = int(args.seed)
    config = _config(args, seed)

    print(
        f"level_set_gaussian_child_probe seed={seed} "
        f"max_epochs={args.max_epochs} grid={args.max_grid_points} "
        f"max_finer_scale_steps=16 min_samples=100 max_depth=5",
        flush=True,
    )

    data = make_two_gaussians(
        n_samples=TWO_GAUSSIANS_N,
        sigma=TWO_GAUSSIANS_SIGMA,
        separation=TWO_GAUSSIANS_CLEAR_SEP,
        seed=seed,
    )
    points = np.asarray(data.points, dtype=float)
    labels = np.asarray(data.labels)
    dim = int(data.ground_truth.ambient_dim)
    n0 = int(np.sum(labels == 0))
    n1 = int(np.sum(labels == 1))
    n_bg = int(np.sum(labels < 0))
    print(
        f"scene two_gaussians_clear n={points.shape[0]} dim={dim} "
        f"GT n0={n0} n1={n1} n_tissue={n_bg} "
        f"sigma={TWO_GAUSSIANS_SIGMA} sep={TWO_GAUSSIANS_CLEAR_SEP}",
        flush=True,
    )

    t_tree = time.time()
    tree = run_recursive_discovery(points, dim=dim, config=config)
    print(f"run_recursive_discovery t={time.time() - t_tree:.1f}s", flush=True)
    _print_tree(tree, labels)

    print()
    print("=" * 78)
    print("PART 2 — stage dissection of level-1 non-background children")
    print("=" * 78, flush=True)
    if not tree.nodes:
        print("empty tree; no children", flush=True)
        part2: list[dict[str, Any]] = []
    else:
        root = tree.nodes[0]
        part2 = []
        for cid in root.children:
            child = tree.nodes[int(cid)]
            if child.is_background:
                print(
                    f"skip background child id={child.region_id} n={child.n_samples}",
                    flush=True,
                )
                continue
            idx = np.asarray(child.sample_indices, dtype=int)
            fr = _gt_fractions(idx, labels)
            print(
                f"child id={child.region_id} n={child.n_samples} "
                f"GT n0={fr['n0']} n1={fr['n1']} n_tissue={fr['n_bg']}",
                flush=True,
            )
            rec = _dissect(
                f"L1_child_{child.region_id}",
                points[idx],
                dim,
                config,
            )
            rec["region_id"] = int(child.region_id)
            rec["gt"] = fr
            part2.append(rec)

    print()
    print("=" * 78)
    print("PART 3 — controls (same dissection)")
    print("=" * 78, flush=True)
    rng = np.random.default_rng(seed)
    part3: dict[str, Any] = {}

    lone400 = rng.normal(size=(400, 2)) * TWO_GAUSSIANS_SIGMA
    part3["lone_400_2d"] = _dissect("control_a_lone_400_2d", lone400, 2, config)

    lone800 = rng.normal(size=(800, 2)) * TWO_GAUSSIANS_SIGMA
    part3["lone_800_2d"] = _dissect("control_b_lone_800_2d", lone800, 2, config)

    subset_pts, tissue_meta = _label0_plus_nearby_tissue(points, labels, radius=1.0)
    print(
        f"control_c label==0 + tissue within 1.0: {tissue_meta}",
        flush=True,
    )
    rec_c = _dissect("control_c_label0_plus_tissue", subset_pts, 2, config)
    rec_c["tissue_meta"] = tissue_meta
    part3["label0_plus_tissue"] = rec_c

    lone800_4d = rng.normal(size=(800, 4))
    part3["lone_800_4d"] = _dissect("control_d_lone_800_4d", lone800_4d, 4, config)

    print()
    print("=" * 78)
    print("PART 3 SUMMARY — which controls accept a split")
    print("=" * 78, flush=True)
    for key, rec in part3.items():
        acc = rec.get("accepted")
        print(
            f"{key:24s} accepted={acc} K={rec.get('n_clusters')} "
            f"phi={rec.get('phi')} rho={rec.get('rho')} "
            f"logBF={rec.get('log_bf')} verdict={rec.get('verdict')} "
            f"reason={rec.get('reject_reason')}",
            flush=True,
        )

    payload = {
        "seed": seed,
        "max_epochs": int(args.max_epochs),
        "max_grid_points": int(args.max_grid_points),
        "part2": part2,
        "part3": part3,
    }
    with open(JSON_PATH, "w") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
        fh.write("\n")
    print(f"\nJSON dumped to {JSON_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
