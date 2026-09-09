"""Normal-path diagnostic: level-set + finer-research + cap-growth.

Unlike ``level_set_fitted_sweep.py`` this does **not** fit a hand-picked
fine ``tau``.  It runs ``run_recursive_discovery`` with the default
``load_crossover`` selector, ``use_level_set_clustering=True``,
``allow_finer_research=True``, and the #44 under-resolved cap-growth
knobs.  ``use_level_set_clustering`` stays default-off in production;
this script only opts in locally.

The measured nested/tori valleys sit ~80× below coarse ``L=1`` ``tau*``.
Default ``max_finer_scale_steps=8`` spans only 16× (``(1/sqrt(2))**8``),
so this diagnostic uses 16 steps unless overridden.  ``tau_min`` stays
the ScaleSearch default ``1e-5`` so the walk can reach ``tau_sep``;
GT ``tau_grid_hint`` lower bounds are too coarse (~``expected_tau/8``).

Seed-0 (2026-09-09, #48, with the ``N <= n/k`` bound): all six nulls
(circle, swiss, lone torus, lone inner shell, lone 2-d/4-d Gaussian)
one leaf; hierarchy root ``K=3`` / 6 fine leaves (GT 3×2); tori root
``K=2`` / 2 leaves; clear two-Gaussians ``K=2`` / 2; nested ``K=2`` /
5 and bimodal ``K=2`` / 4 (tissue-heavy children, #45). Valley-scene
ARI bars are not frozen. Not a default-flag flip.

Not a pytest test.  Nested/tori at the fitted-sweep ``n`` take minutes
to tens of minutes per seed.

Usage::

    PYTHONPATH="src:$PWD" pipenv run python \\
        tests/scenarios/synthetic/level_set_normal_path_sweep.py --seeds 0

    PYTHONPATH="src:$PWD" pipenv run python \\
        tests/scenarios/synthetic/level_set_normal_path_sweep.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from proteus.stage1.controller import ScaleSearchConfig
from proteus.stage1.level_set import LevelSetConfig
from proteus.stage1.recursion import RecursionConfig, RecursionTree, run_recursive_discovery
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.ground_truth import GroundTruthManifold, SyntheticDataset
from tests.datasets.synthetic.circles import make_circle
from tests.datasets.synthetic.hierarchical_gaussian import (
    make_hierarchical_gaussian,
)
from tests.datasets.synthetic.density_valleys import (
    make_bimodal_circle,
    make_two_gaussians,
)
from tests.datasets.synthetic.linked_tori import make_linked_tori
from tests.datasets.synthetic.nested_spheres import make_nested_spheres
from tests.datasets.synthetic.swiss_roll import make_swiss_roll
from tests.scenarios.synthetic.cd_level_set_probe import score
from tests.scenarios.synthetic.level_set_suite import (
    CIRCLE_N,
    HIERARCHY_K,
    HIERARCHY_MIN_ARI,
    HIERARCHY_MIN_COVERAGE,
    HIERARCHY_N,
    NESTED_K,
    NESTED_MIN_ARI,
    NESTED_MIN_COVERAGE,
    NESTED_N_PER,
    SWISS_N,
    TORI_K,
    TORI_MIN_ARI,
    TORI_MIN_COVERAGE,
    TORI_N_PER,
    BIMODAL_CIRCLE_N,
    BIMODAL_CIRCLE_KAPPA,
    BIMODAL_CIRCLE_K,
    TWO_GAUSSIANS_N,
    TWO_GAUSSIANS_SIGMA,
    TWO_GAUSSIANS_WEAK_SEP,
    TWO_GAUSSIANS_CLEAR_SEP,
    LONE_TORUS_N_PER,
    LONE_SHELL_N_PER,
    LONE_GAUSS2D_N,
    LONE_GAUSS2D_SIGMA,
    LONE_GAUSS4D_N,
    LONE_TISSUE_RADIUS,
)


@dataclass(frozen=True)
class Scene:
    name: str
    factory: Callable[[int], SyntheticDataset]
    expected_k: int | None
    min_ari: float = 0.0
    min_coverage: float = 0.0


@dataclass(frozen=True)
class Result:
    scene: str
    seed: int
    passed: bool
    tau_star: float | None
    n_signal_root: int
    n_signal_leaves: int
    n_leaves: int
    ari: float | None
    background_recall: float | None
    coverage: float | None
    elapsed: float
    note: str


def _keep_signal_plus_nearby_tissue(
    points: np.ndarray,
    labels: np.ndarray,
    target: int,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep ``labels == target`` plus tissue (``labels < 0``) within ``radius``."""

    arr = np.asarray(points, dtype=float)
    lab = np.asarray(labels, dtype=int)
    signal_idx = np.flatnonzero(lab == int(target))
    tissue_idx = np.flatnonzero(lab < 0)
    if signal_idx.size == 0:
        return arr[signal_idx], np.empty(0, dtype=int)
    if tissue_idx.size == 0:
        out_lab = np.zeros(signal_idx.size, dtype=int)
        return arr[signal_idx], out_lab
    dists, _ = cKDTree(arr[signal_idx]).query(arr[tissue_idx], k=1)
    near_idx = tissue_idx[np.asarray(dists, dtype=float) < float(radius)]
    keep = np.concatenate([signal_idx, near_idx])
    out_lab = np.concatenate([
        np.zeros(signal_idx.size, dtype=int),
        np.full(near_idx.size, -1, dtype=int),
    ])
    return arr[keep], out_lab


def _lone_torus_null(seed: int) -> SyntheticDataset:
    data = make_linked_tori(n_per_torus=LONE_TORUS_N_PER, seed=seed)
    points, labels = _keep_signal_plus_nearby_tissue(
        data.points, data.labels, 0, LONE_TISSUE_RADIUS,
    )
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=data.ground_truth,
        metadata=dict(data.metadata),
    )


def _lone_shell_inner_null(seed: int) -> SyntheticDataset:
    data = make_nested_spheres(n_per_sphere=LONE_SHELL_N_PER, seed=seed)
    points, labels = _keep_signal_plus_nearby_tissue(
        data.points, data.labels, 1, LONE_TISSUE_RADIUS,
    )
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=data.ground_truth,
        metadata=dict(data.metadata),
    )


def _lone_gauss2d_null(seed: int) -> SyntheticDataset:
    rng = np.random.default_rng(int(seed))
    points = rng.normal(size=(LONE_GAUSS2D_N, 2)) * LONE_GAUSS2D_SIGMA
    labels = np.zeros(LONE_GAUSS2D_N, dtype=int)
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=GroundTruthManifold(
            name="lone_gauss2d", ambient_dim=2, intrinsic_dim=2,
        ),
    )


def _lone_gauss4d_null(seed: int) -> SyntheticDataset:
    rng = np.random.default_rng(int(seed))
    points = rng.normal(size=(LONE_GAUSS4D_N, 4))
    labels = np.zeros(LONE_GAUSS4D_N, dtype=int)
    return SyntheticDataset(
        points=points,
        labels=labels,
        ground_truth=GroundTruthManifold(
            name="lone_gauss4d", ambient_dim=4, intrinsic_dim=4,
        ),
    )


def _scenes() -> tuple[Scene, ...]:
    return (
        Scene(
            "circle_null",
            lambda seed: make_circle(n_samples=CIRCLE_N, seed=seed),
            None,
        ),
        Scene(
            "swiss_roll_null",
            lambda seed: make_swiss_roll(n_samples=SWISS_N, seed=seed),
            None,
        ),
        Scene("lone_torus_null", _lone_torus_null, None),
        Scene("lone_shell_inner_null", _lone_shell_inner_null, None),
        Scene("lone_gauss2d_null", _lone_gauss2d_null, None),
        Scene("lone_gauss4d_null", _lone_gauss4d_null, None),
        Scene(
            "hierarchy",
            lambda seed: make_hierarchical_gaussian(n_samples=HIERARCHY_N, seed=seed),
            HIERARCHY_K,
            min_ari=HIERARCHY_MIN_ARI,
            min_coverage=HIERARCHY_MIN_COVERAGE,
        ),
        Scene(
            "linked_tori",
            lambda seed: make_linked_tori(n_per_torus=TORI_N_PER, seed=seed),
            TORI_K,
            min_ari=TORI_MIN_ARI,
            min_coverage=TORI_MIN_COVERAGE,
        ),
        Scene(
            "nested_spheres",
            lambda seed: make_nested_spheres(n_per_sphere=NESTED_N_PER, seed=seed),
            NESTED_K,
            min_ari=NESTED_MIN_ARI,
            min_coverage=NESTED_MIN_COVERAGE,
        ),
        Scene(
            "bimodal_circle",
            lambda seed: make_bimodal_circle(
                n_samples=BIMODAL_CIRCLE_N, kappa=BIMODAL_CIRCLE_KAPPA, seed=seed,
            ),
            BIMODAL_CIRCLE_K,
        ),
        Scene(
            "two_gaussians_weak",
            lambda seed: make_two_gaussians(
                n_samples=TWO_GAUSSIANS_N,
                sigma=TWO_GAUSSIANS_SIGMA,
                separation=TWO_GAUSSIANS_WEAK_SEP,
                seed=seed,
            ),
            2,
        ),
        Scene(
            "two_gaussians_clear",
            lambda seed: make_two_gaussians(
                n_samples=TWO_GAUSSIANS_N,
                sigma=TWO_GAUSSIANS_SIGMA,
                separation=TWO_GAUSSIANS_CLEAR_SEP,
                seed=seed,
            ),
            2,
        ),
    )


def _root_signal_labels(tree: RecursionTree, n: int) -> tuple[np.ndarray, int]:
    """Map samples to the root's non-background children (or 1 if terminal)."""

    pred = np.full(n, -1, dtype=int)
    if not tree.nodes:
        return pred, 0
    root = tree.nodes[0]
    if root.is_leaf:
        idx = np.asarray(root.sample_indices, dtype=int)
        if idx.size:
            pred[idx] = 0
        return pred, 0 if root.is_background else 1
    k = 0
    for cid in root.children:
        child = tree.nodes[int(cid)]
        idx = np.asarray(child.sample_indices, dtype=int)
        if child.is_background or idx.size == 0:
            continue
        pred[idx] = k
        k += 1
    return pred, k


def _signal_leaf_count(tree: RecursionTree) -> int:
    return sum(1 for leaf in tree.leaves if not leaf.is_background)


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
        max_finer_scale_steps=int(args.max_finer_steps),
        level_set=LevelSetConfig(),
        seed=int(seed),
    )


def run_scene(scene: Scene, seed: int, args: argparse.Namespace) -> Result:
    data = scene.factory(seed)
    if scene.name == "linked_tori" and not data.metadata["resolvable_k8"]:
        return Result(
            scene.name, seed, False, None, 0, 0, 0, None, None, None, 0.0,
            "generator metadata says k=8 cannot resolve the lambda=0.5 gap",
        )

    started = time.time()
    tree = run_recursive_discovery(
        data.points,
        dim=int(data.ground_truth.ambient_dim),
        config=_config(args, seed),
    )
    elapsed = time.time() - started
    n = int(data.points.shape[0])
    pred, root_k = _root_signal_labels(tree, n)
    n_sig_leaves = _signal_leaf_count(tree)
    tau_star = tree.nodes[0].tau_star if tree.nodes else None

    if scene.expected_k is None:
        passed = root_k <= 1
        return Result(
            scene.name, seed, passed, tau_star, root_k, n_sig_leaves,
            len(tree.leaves), None, None, None, elapsed,
            "connected-manifold null must stay one root feature",
        )

    ari, background_recall, coverage = score(data.labels, pred)
    passed = (
        root_k == scene.expected_k
        and ari >= scene.min_ari
        and coverage >= scene.min_coverage
    )
    return Result(
        scene.name, seed, passed, tau_star, root_k, n_sig_leaves,
        len(tree.leaves), float(ari), float(background_recall),
        float(coverage), elapsed,
        (
            f"require root K={scene.expected_k}, ARI>={scene.min_ari:.2f}, "
            f"coverage>={scene.min_coverage:.2f}"
        ),
    )


def _format(result: Result) -> str:
    verdict = "PASS" if result.passed else "FAIL"
    tau = f"tau*={result.tau_star:.4g}" if result.tau_star is not None else "tau*=None"
    metrics = ""
    if result.ari is not None:
        metrics = (
            f" ARI={result.ari:.3f} bg={result.background_recall:.3f}"
            f" cover={result.coverage:.3f}"
        )
    return (
        f"{verdict:4s} {result.scene:19s} s{result.seed} "
        f"{tau:14s} rootK={result.n_signal_root} "
        f"sigLeaves={result.n_signal_leaves} leaves={result.n_leaves}"
        f"{metrics} t={result.elapsed:6.1f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Subset of scene names (default: all five).",
    )
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--max-grid-points", type=int, default=8)
    parser.add_argument(
        "--max-finer-steps",
        type=int,
        default=16,
        help="Default 16 spans ~80x below tau*; production default is 8 (16x).",
    )
    args = parser.parse_args()

    wanted = None if args.scenes is None else set(args.scenes)
    results: list[Result] = []
    print(
        "normal-path level-set diagnostic "
        f"(load_crossover, finer_steps={args.max_finer_steps}, "
        f"max_epochs={args.max_epochs}, grid={args.max_grid_points})",
        flush=True,
    )
    for scene in _scenes():
        if wanted is not None and scene.name not in wanted:
            continue
        for seed in args.seeds:
            result = run_scene(scene, int(seed), args)
            results.append(result)
            print(_format(result), flush=True)

    passed = sum(result.passed for result in results)
    print(f"summary: {passed}/{len(results)} scene-seeds passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
