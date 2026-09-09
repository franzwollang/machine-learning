"""Root-only instrumented finer walk (OPEN_ISSUES #44 / #48).

Does **not** recurse into children.  Replays the *cold* shatter path
(fresh ``run_scale_search`` per step) so ``tau*`` can jump non-monotone.
Production level-set finer walk now re-seeds at each finer ``τ``
(``fit_scaffold_at_tau``, ``track_tau``); this probe still documents
the cold shatter path.  Prints per step: LC ``tau*``, ``tau_cap``,
node budget, hits, ``r_k``, trichotomy, bottleneck ratio, and DM log-BF.

Usage::

    PYTHONPATH="src:$PWD" pipenv run python \\
        tests/scenarios/synthetic/level_set_finer_walk_probe.py
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import time

import numpy as np
from scipy.spatial import cKDTree

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.level_set import LevelSetConfig, select_level_set_partition
from proteus.stage1.recursion import RecursionConfig
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.circles import make_circle


def _hit_stats(scaffold) -> tuple[float, float, float, float]:
    hits = np.asarray(
        [float(node.hit_count) for node in scaffold.nodes],
        dtype=float,
    )
    if hits.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = float(hits.mean())
    cv = float(hits.std() / max(mean, 1e-12))
    return mean, float(hits.min()), float(np.median(hits)), cv


def _rk_stats(scaffold, k: int = 8) -> tuple[float, float, float]:
    positions = np.asarray(
        [node.position for node in scaffold.nodes],
        dtype=float,
    )
    n = int(positions.shape[0])
    if n < 2:
        return 0.0, 0.0, 0.0
    k_use = max(1, min(int(k), n - 1))
    dists, _ = cKDTree(positions).query(positions, k=k_use + 1)
    r_k = dists[:, -1]
    mean = float(r_k.mean())
    cv = float(r_k.std() / max(mean, 1e-12))
    return mean, float(np.median(r_k)), cv


def _line(
    step: str,
    tau: float,
    scaffold,
    selection,
    elapsed: float,
    tau_cap: float | None = None,
) -> str:
    rv = selection.resolvability
    verdict = rv.verdict.value if rv is not None else "?"
    reason = rv.reject_reason if rv is not None else None
    ratio = selection.bottleneck_ratio
    ratio_s = f"{ratio:.4g}" if ratio is not None else "na"
    hits_mean, hits_min, _hits_med, hits_cv = _hit_stats(scaffold)
    rk_mean, _rk_med, rk_cv = _rk_stats(scaffold)
    k = (
        selection.cluster_result.n_clusters
        if selection.cluster_result is not None else 0
    )
    return (
        f"{step:6s} tau={tau:.5g}"
        f"{'' if tau_cap is None else f' cap={tau_cap:.5g}'} "
        f"nodes={len(scaffold.nodes):4d}/"
        f"{getattr(scaffold, 'max_nodes', '?')} "
        f"acc={int(selection.accepted)} K={k} {verdict:16s} "
        f"reason={reason} phi={ratio_s} logBF={selection.log_bf:.2f} "
        f"lvl={selection.candidate_level} "
        f"hits[mu={hits_mean:.1f} min={hits_min:.1f} cv={hits_cv:.2f}] "
        f"rk[mu={rk_mean:.4g} cv={rk_cv:.2f}] t={elapsed:.1f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=1500)
    parser.add_argument("--max-finer-steps", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--max-grid-points", type=int, default=8)
    args = parser.parse_args()

    data = make_circle(n_samples=int(args.n_samples), seed=int(args.seed))
    dim = int(data.ground_truth.ambient_dim)
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
        seed=int(args.seed),
    )
    config = RecursionConfig(
        scale_search=scale,
        min_samples=100,
        max_depth=1,
        use_level_set_clustering=True,
        allow_finer_research=True,
        max_finer_scale_steps=int(args.max_finer_steps),
        level_set=LevelSetConfig(),
        seed=int(args.seed),
    )

    print(
        f"circle n={args.n_samples} seed={args.seed} expected_tau="
        f"{data.ground_truth.expected_tau:.5g} "
        f"hint={data.ground_truth.tau_grid_hint}",
        flush=True,
    )

    t0 = time.time()
    result = run_scale_search(data.points, dim, scale)
    scaffold = result.scaffold_at_star
    assert scaffold is not None
    selection = select_level_set_partition(
        scaffold, config.level_set, config.dm_cluster,
    )
    print(
        _line("tau*", float(result.tau_star), scaffold, selection, time.time() - t0),
        flush=True,
    )
    if selection.accepted:
        print("accepted at tau*; no finer walk")
        return 0

    parent_tau = float(result.tau_star)
    ratio = float(config.finer_tau_cap_ratio)
    tau_min = float(scale.tau_min)
    tau_cap = parent_tau * ratio
    working_max_nodes = getattr(scaffold, "max_nodes", None)
    max_steps = int(args.max_finer_steps)

    for step in range(max_steps):
        if not (tau_min < tau_cap < parent_tau):
            print(f"stop: tau_cap={tau_cap:.5g} outside ({tau_min}, {parent_tau})")
            break
        step_cfg = (
            replace(scale, tau_max=tau_cap, max_nodes=int(working_max_nodes))
            if working_max_nodes is not None
            else replace(scale, tau_max=tau_cap)
        )
        t1 = time.time()
        step_result = run_scale_search(data.points, dim, step_cfg)
        sc = step_result.scaffold_at_star
        if sc is None or len(sc.nodes) < 2:
            print(f"step {step+1}: empty scaffold at tau_cap={tau_cap:.5g}", flush=True)
            tau_cap *= ratio
            continue
        sel = select_level_set_partition(sc, config.level_set, config.dm_cluster)
        if getattr(sc, "max_nodes", None) is not None:
            working_max_nodes = int(sc.max_nodes)
        print(
            _line(
                f"f{step+1}",
                float(step_result.tau_star),
                sc,
                sel,
                time.time() - t1,
            ),
            flush=True,
        )
        if sel.accepted:
            print(
                f"FIRST ACCEPT at finer step {step+1}; "
                f"tau*/parent={float(step_result.tau_star)/parent_tau:.4g}",
                flush=True,
            )
            return 0
        tau_cap *= ratio

    print("no accepted split in finer walk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
