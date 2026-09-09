"""Root tissue leakage into signal children (OPEN_ISSUES #48 / #45).

Runs ``run_recursive_discovery`` at ``max_depth=1`` and wraps
``_descend_into_clusters`` to capture the root scaffold / cluster_result.
Reports how generator tissue (labels < 0) is mapped onto signal children.

Not a pytest test.

Usage::

    PYTHONPATH="src:$PWD" pipenv run python \\
        tests/scenarios/synthetic/level_set_root_tissue_probe.py --seed 0
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

import proteus.stage1.recursion as recursion_mod
from proteus.stage1.level_set import (
    LevelSetConfig,
    _filter_relative_mass,
    build_level_set_tree,
    select_level_set_partition,
)
from proteus.stage1.recursion import assign_samples_to_clusters, run_recursive_discovery
from tests.scenarios.synthetic.level_set_normal_path_sweep import _config, _scenes


DEFAULT_SCENES = ("bimodal_circle", "nested_spheres")
HIST_BINS = 10


def _gt_comp(lab: np.ndarray) -> str:
    lab = np.asarray(lab, dtype=int)
    if lab.size == 0:
        return "empty"
    n = float(lab.size)
    parts: list[str] = []
    for value in sorted({int(v) for v in lab if v >= 0}):
        count = int(np.sum(lab == value))
        parts.append(f"L{value}={count}({count / n:.3f})")
    n_bg = int(np.sum(lab < 0))
    parts.append(f"tissue={n_bg}({n_bg / n:.3f})")
    return " ".join(parts)


def _fmt(value: Any, digits: str = ".6g") -> str:
    if value is None:
        return "na"
    try:
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return "nan"
        return format(float(value), digits)
    except (TypeError, ValueError):
        return str(value)


def _bmu_ids_dists(scaffold, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample BMU index and distance via the scaffold ANN (k=1)."""

    n = int(points.shape[0])
    bmus = np.empty(n, dtype=int)
    dists = np.empty(n, dtype=float)
    for i in range(n):
        ids, ds = scaffold.ann.query_knn(points[i], k=1)
        bmus[i] = int(ids[0])
        dists[i] = float(ds[0])
    return bmus, dists


def _print_hist(name: str, values_a: np.ndarray, values_b: np.ndarray) -> None:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    pooled = np.concatenate([a, b]) if (a.size or b.size) else np.array([0.0, 1.0])
    lo = float(np.min(pooled)) if pooled.size else 0.0
    hi = float(np.max(pooled)) if pooled.size else 1.0
    if hi <= lo:
        hi = lo + 1e-12
    edges = np.linspace(lo, hi, HIST_BINS + 1)
    ha, _ = np.histogram(a, bins=edges)
    hb, _ = np.histogram(b, bins=edges)
    print(f"HIST {name}  n_sig_child_tissue={a.size} n_bg_child_tissue={b.size}", flush=True)
    print(
        f"  {'bin':>3} {'lo':>12} {'hi':>12} {'sig_child':>10} {'bg_child':>10}",
        flush=True,
    )
    for i in range(HIST_BINS):
        print(
            f"  {i:3d} {edges[i]:12.6g} {edges[i + 1]:12.6g} "
            f"{int(ha[i]):10d} {int(hb[i]):10d}",
            flush=True,
        )


def _background_criterion_report(scaffold, cluster_labels: np.ndarray) -> None:
    """Replay C-D + relative-mass to name the -1 rule and count sources."""

    cfg = LevelSetConfig()
    positions = np.asarray([node.position for node in scaffold.nodes], dtype=float)
    tree = build_level_set_tree(positions, cfg)
    print("BACKGROUND CRITERION (level-set reader → node label -1)", flush=True)
    print(
        "  1) build_level_set_tree: inactive if kNN core_radius > level radius "
        f"(k={cfg.k_neighbors}); runt if component size < min_cluster_size="
        f"{cfg.min_cluster_size}.",
        flush=True,
    )
    print(
        "  2) _filter_relative_mass: remaining cluster mapped to -1 if size < "
        f"max(min_cluster_size={cfg.min_cluster_size}, "
        f"ceil(min_cluster_frac={cfg.min_cluster_frac} * N)).",
        flush=True,
    )
    print(
        "  3) dm_partition_background_verdict does not assign -1; it only "
        "accepts/rejects the already-labelled partition. "
        "apply_dm_sibling_collapse prunes DAG branches for diagnostics, "
        "not the accepted ClusterResult labels.",
        flush=True,
    )
    print(
        f"  defaults: min_cluster_size={cfg.min_cluster_size} "
        f"min_cluster_frac={cfg.min_cluster_frac} "
        f"N={positions.shape[0]} "
        f"rel_mass_floor={max(cfg.min_cluster_size, int(np.ceil(cfg.min_cluster_frac * positions.shape[0])))}",
        flush=True,
    )

    selection = select_level_set_partition(scaffold, cfg, data=None)
    sel_level = selection.selected_level
    print(
        f"  recomputed selection: accepted={int(selection.accepted)} "
        f"selected_level={sel_level} K="
        f"{0 if selection.cluster_result is None else selection.cluster_result.n_clusters}",
        flush=True,
    )
    if sel_level is None or not tree.levels:
        return
    raw = np.asarray(tree.levels[sel_level].labels, dtype=int)
    filtered = _filter_relative_mass(raw, cfg.min_cluster_size, cfg.min_cluster_frac)
    level = tree.levels[sel_level]
    n_raw_bg = int(np.sum(raw < 0))
    n_filt_bg = int(np.sum(filtered < 0))
    n_relabel = int(np.sum((raw >= 0) & (filtered < 0)))
    match = bool(np.array_equal(filtered, np.asarray(cluster_labels, dtype=int)))
    print(
        f"  selected_level={sel_level} radius={level.radius:.6g} "
        f"n_inactive={level.n_inactive} n_runt={level.n_runt} "
        f"raw_n_background={n_raw_bg} after_rel_mass={n_filt_bg} "
        f"rel_mass_extra_to_-1={n_relabel} "
        f"matches_cluster_result={int(match)}",
        flush=True,
    )


def run_scene(scene_name: str, args: argparse.Namespace) -> None:
    scenes = {s.name: s for s in _scenes()}
    if scene_name not in scenes:
        raise SystemExit(f"unknown scene {scene_name}; known: {sorted(scenes)}")
    scene = scenes[scene_name]
    seed = int(args.seed)
    data = scene.factory(seed)
    points = np.asarray(data.points, dtype=float)
    labels = np.asarray(data.labels, dtype=int)
    dim = int(data.ground_truth.ambient_dim)
    config = replace(_config(args, seed), max_depth=1)

    print("=" * 78, flush=True)
    print(
        f"SCENE {scene_name} seed={seed} n={points.shape[0]} dim={dim} "
        f"max_depth={config.max_depth}",
        flush=True,
    )
    print("=" * 78, flush=True)

    captured: dict[str, Any] = {}
    original = recursion_mod._descend_into_clusters

    def wrapper(**kwargs):
        if "scaffold" not in captured:
            captured["scaffold"] = kwargs["scaffold"]
            captured["cluster_result"] = kwargs["cluster_result"]
            captured["data_arr"] = np.asarray(kwargs["data_arr"], dtype=float)
            captured["node"] = kwargs["node"]
        return original(**kwargs)

    recursion_mod._descend_into_clusters = wrapper
    t0 = time.time()
    try:
        tree = run_recursive_discovery(points, dim=dim, config=config)
    finally:
        recursion_mod._descend_into_clusters = original
    elapsed = time.time() - t0
    print(f"run_recursive_discovery elapsed={elapsed:.1f}s", flush=True)

    if "scaffold" not in captured:
        print("NO_ROOT_DESCENT (root did not call _descend_into_clusters)", flush=True)
        return

    scaffold = captured["scaffold"]
    cluster_result = captured["cluster_result"]
    node_labels = np.asarray(cluster_result.labels, dtype=int)
    n_nodes = int(node_labels.shape[0])
    hits = np.asarray(
        [float(node.hit_count) for node in scaffold.nodes],
        dtype=float,
    )
    positions = np.asarray(
        [node.position for node in scaffold.nodes],
        dtype=float,
    )

    print(
        f"ROOT accepted scaffold N={n_nodes} "
        f"cluster_result.n_clusters={int(cluster_result.n_clusters)} "
        f"node_label_set={sorted({int(v) for v in node_labels})}",
        flush=True,
    )

    # --- 1. Node-level -------------------------------------------------
    bmus, bmu_dists = _bmu_ids_dists(scaffold, points)
    assigned = node_labels[bmus]
    reproduced = assign_samples_to_clusters(points, scaffold, node_labels)

    print("-" * 78, flush=True)
    print("1. NODE-LEVEL", flush=True)
    print(f"  N_nodes={n_nodes} mean_hit_count={_fmt(float(np.mean(hits)))}", flush=True)
    for lab in sorted({int(v) for v in node_labels}):
        mask = node_labels == lab
        node_ids = np.flatnonzero(mask)
        catchment = np.flatnonzero(np.isin(bmus, node_ids))
        kind = "background" if lab < 0 else "signal"
        print(
            f"  node_label={lab} ({kind}) n_nodes={int(mask.sum())} "
            f"mean_hit={_fmt(float(np.mean(hits[mask])))} "
            f"n_nearest_samples={int(catchment.size)} "
            f"gt={_gt_comp(labels[catchment])}",
            flush=True,
        )

    # --- 2. Sample → child assignment ---------------------------------
    print("-" * 78, flush=True)
    print("2. SAMPLE→CHILD ASSIGNMENT", flush=True)
    print(
        "  rule: assign_samples_to_clusters — each sample's BMU is "
        "scaffold.ann.query_knn(x, k=1); child label = cluster_result.labels[bmu]. "
        "label < 0 is the background child (is_background=True).",
        flush=True,
    )
    repro_sizes = {int(k): int(np.asarray(v).size) for k, v in reproduced.items()}
    print(f"  reproduced sizes by node-label: {repro_sizes}", flush=True)
    print(
        f"  BMU-index assignment sizes: "
        f"{ {int(k): int(np.sum(assigned == k)) for k in sorted({int(v) for v in assigned})} }",
        flush=True,
    )

    by_id = {int(n.region_id): n for n in tree.nodes}
    root = tree.nodes[0]
    tree_sizes: list[int] = []
    print("  tree root children:", flush=True)
    for cid in root.children:
        child = by_id[int(cid)]
        n_c = int(child.n_samples)
        tree_sizes.append(n_c)
        cidx = np.asarray(child.sample_indices, dtype=int)
        print(
            f"    id={int(child.region_id)} n={n_c} "
            f"bg={int(child.is_background)} leaf={int(child.is_leaf)} "
            f"gt={_gt_comp(labels[cidx])}",
            flush=True,
        )
    expected = {
        "bimodal_circle": (303, 636, 561),
        "nested_spheres": (2186, 1940, 1874),
    }
    exp = expected.get(scene_name)
    if exp is not None:
        match = tuple(tree_sizes) == exp or tuple(sorted(tree_sizes)) == tuple(sorted(exp))
        print(
            f"  expected child sizes {exp}; tree={tuple(tree_sizes)} "
            f"match={int(match)}",
            flush=True,
        )
    n_tissue = int(np.sum(labels < 0))
    n_tissue_bg = 0
    for cid in root.children:
        child = by_id[int(cid)]
        if child.is_background:
            n_tissue_bg += int(np.sum(labels[np.asarray(child.sample_indices, dtype=int)] < 0))
    recall = (n_tissue_bg / n_tissue) if n_tissue else float("nan")
    print(
        f"  tissue_total={n_tissue} tissue_in_bg_child={n_tissue_bg} "
        f"bg_recall={_fmt(recall)}",
        flush=True,
    )

    # --- 3. Tissue that landed in SIGNAL children ---------------------
    print("-" * 78, flush=True)
    print("3. TISSUE IN SIGNAL CHILDREN", flush=True)
    signal_pts = points[labels >= 0]
    signal_tree = cKDTree(signal_pts) if signal_pts.shape[0] else None
    node_tree = cKDTree(positions) if positions.shape[0] else None
    bg_node_ids = np.flatnonzero(node_labels < 0)
    bg_tree = (
        cKDTree(positions[bg_node_ids]) if bg_node_ids.size else None
    )
    sig_node_mean_hit = float(np.mean(hits[node_labels >= 0])) if np.any(node_labels >= 0) else float("nan")

    tissue_sig_d_signal: list[float] = []
    tissue_bg_d_signal: list[float] = []

    for lab in sorted({int(v) for v in node_labels if v >= 0}):
        cluster_nodes = np.flatnonzero(node_labels == lab)
        cluster_tree = cKDTree(positions[cluster_nodes])
        tissue_in = np.flatnonzero((assigned == lab) & (labels < 0))
        n_t = int(tissue_in.size)
        if n_t == 0:
            print(f"  signal_label={lab}: no tissue samples", flush=True)
            continue
        d_node = np.asarray(cluster_tree.query(points[tissue_in], k=1)[0], dtype=float)
        if signal_tree is None:
            d_sig = np.full(n_t, np.nan)
        else:
            d_sig = np.asarray(signal_tree.query(points[tissue_in], k=1)[0], dtype=float)
        tissue_sig_d_signal.extend(float(x) for x in d_sig)
        assigned_hits = hits[bmus[tissue_in]]
        if bg_tree is None:
            nearer_bg = np.zeros(n_t, dtype=bool)
        else:
            d_bg = np.asarray(bg_tree.query(points[tissue_in], k=1)[0], dtype=float)
            nearer_bg = d_bg < d_node
        print(
            f"  signal_label={lab} n_tissue={n_t} "
            f"d_nearest_signal_sample mean={_fmt(float(np.mean(d_sig)))} "
            f"median={_fmt(float(np.median(d_sig)))} "
            f"d_nearest_assigned_cluster_node mean={_fmt(float(np.mean(d_node)))} "
            f"median={_fmt(float(np.median(d_node)))} "
            f"assigned_node_hit mean={_fmt(float(np.mean(assigned_hits)))} "
            f"median={_fmt(float(np.median(assigned_hits)))} "
            f"signal_node_mean_hit={_fmt(sig_node_mean_hit)} "
            f"frac_nearer_to_bg_node={float(np.mean(nearer_bg)):.4f} "
            f"({int(np.sum(nearer_bg))}/{n_t})",
            flush=True,
        )

    if signal_tree is not None:
        for lab in sorted({int(v) for v in node_labels if v < 0}):
            tissue_in_bg = np.flatnonzero((assigned == lab) & (labels < 0))
            if tissue_in_bg.size:
                d_sig_bg = signal_tree.query(points[tissue_in_bg], k=1)[0]
                tissue_bg_d_signal.extend(float(x) for x in np.asarray(d_sig_bg, dtype=float))
    _print_hist(
        "dist_to_nearest_signal_sample",
        np.asarray(tissue_sig_d_signal, dtype=float),
        np.asarray(tissue_bg_d_signal, dtype=float),
    )

    # --- 4. Node density contrast -------------------------------------
    print("-" * 78, flush=True)
    print("4. NODE DENSITY CONTRAST (signal-labelled nodes)", flush=True)
    n_tissue_nodes = 0
    n_empty = 0
    n_ge50_tissue = 0
    n_ge90_signal = 0
    hits_ge50_tissue: list[float] = []
    hits_ge90_signal: list[float] = []
    for nid in np.flatnonzero(node_labels >= 0):
        catchment = np.flatnonzero(bmus == int(nid))
        if catchment.size == 0:
            n_empty += 1
            continue
        tissue_frac = float(np.mean(labels[catchment] < 0))
        signal_frac = 1.0 - tissue_frac
        if tissue_frac >= 0.5:
            n_ge50_tissue += 1
            n_tissue_nodes += 1
            hits_ge50_tissue.append(float(hits[int(nid)]))
        if signal_frac >= 0.9:
            n_ge90_signal += 1
            hits_ge90_signal.append(float(hits[int(nid)]))
    print(
        f"  signal_nodes={int(np.sum(node_labels >= 0))} "
        f"empty_catchment={n_empty} "
        f"tissue_nodes(catchment>=50% tissue)={n_tissue_nodes} "
        f"nodes_ge90pct_signal={n_ge90_signal}",
        flush=True,
    )
    print(
        f"  mean_hit catchment>=50% tissue={_fmt(float(np.mean(hits_ge50_tissue)) if hits_ge50_tissue else None)} "
        f"n={len(hits_ge50_tissue)} "
        f"mean_hit catchment>=90% signal={_fmt(float(np.mean(hits_ge90_signal)) if hits_ge90_signal else None)} "
        f"n={len(hits_ge90_signal)}",
        flush=True,
    )

    # --- 5. Background criterion --------------------------------------
    print("-" * 78, flush=True)
    print("5. BACKGROUND LABEL SOURCE", flush=True)
    _background_criterion_report(scaffold, node_labels)
    _ = (node_tree, bmu_dists)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--max-grid-points", type=int, default=8)
    parser.add_argument("--max-finer-steps", type=int, default=16)
    args = parser.parse_args()
    known = [s.name for s in _scenes()]
    wanted = list(args.scenes)
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise SystemExit(f"unknown scenes: {unknown}; known: {known}")

    print(
        f"level_set_root_tissue_probe seed={args.seed} "
        f"max_depth=1 max_epochs={args.max_epochs} "
        f"grid={args.max_grid_points} finer_steps={args.max_finer_steps} "
        f"scenes={wanted}",
        flush=True,
    )
    for name in wanted:
        run_scene(name, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
