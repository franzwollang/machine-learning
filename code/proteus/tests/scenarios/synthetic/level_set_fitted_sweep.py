"""Multi-seed fitted-scaffold acceptance sweep for level-set extraction.

This is an explicit diagnostic, not a pytest test: fitting the full default
matrix takes several minutes.  It exercises the actual acceptance substrate
(equilibrated node spacing + Hebbian flows), never an expected-K selector.

Usage::

    PYTHONPATH="src:$PWD" pipenv run python \
        tests/scenarios/synthetic/level_set_fitted_sweep.py

    # Bounded smoke run:
    ... level_set_fitted_sweep.py --seeds 0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from typing import Callable

from scipy.spatial import cKDTree

from proteus.stage1.level_set import LevelSetConfig, select_level_set_partition
from tests.datasets.ground_truth import SyntheticDataset
from tests.datasets.synthetic.circles import make_circle
from tests.datasets.synthetic.hierarchical_gaussian import (
    make_hierarchical_gaussian,
)
from tests.datasets.synthetic.linked_tori import make_linked_tori
from tests.datasets.synthetic.nested_spheres import make_nested_spheres
from tests.datasets.synthetic.swiss_roll import make_swiss_roll
from tests.scenarios.synthetic.cd_level_set_probe import fit_scaffold, score


@dataclass(frozen=True)
class Scene:
    name: str
    factory: Callable[[int], SyntheticDataset]
    tau: Callable[[SyntheticDataset], float]
    max_nodes: int
    expected_k: int | None
    min_ari: float = 0.0
    min_coverage: float = 0.0


@dataclass(frozen=True)
class Result:
    scene: str
    seed: int
    passed: bool
    accepted: bool
    n_nodes: int
    n_clusters: int | None
    ari: float | None
    background_recall: float | None
    coverage: float | None
    elapsed: float
    note: str


def _scenes() -> tuple[Scene, ...]:
    return (
        Scene(
            "circle_null",
            lambda seed: make_circle(n_samples=1_500, seed=seed),
            lambda data: float(data.ground_truth.expected_tau) / 8.0,
            256,
            None,
        ),
        Scene(
            "swiss_roll_null",
            lambda seed: make_swiss_roll(n_samples=2_000, seed=seed),
            lambda data: float(data.ground_truth.expected_tau) / 8.0,
            256,
            None,
        ),
        Scene(
            "hierarchy",
            lambda seed: make_hierarchical_gaussian(n_samples=600, seed=seed),
            lambda data: float(data.ground_truth.expected_tau) / 3.0,
            256,
            3,
            min_ari=0.50,
            min_coverage=0.95,
        ),
        Scene(
            "linked_tori",
            lambda seed: make_linked_tori(n_per_torus=4_000, seed=seed),
            # Fine signal read. The global expected tau is tissue-dominated.
            lambda data: max(
                0.02, float(data.metadata["signal_expected_tau"]),
            ),
            768,
            2,
            min_ari=0.95,
            min_coverage=0.95,
        ),
        Scene(
            "nested_spheres",
            lambda seed: make_nested_spheres(
                n_per_sphere=3_000, seed=seed,
            ),
            lambda data: float(data.ground_truth.expected_tau) / 20.0,
            # 1024 truncates the fit and rejected 3/5 seeds; 1536 lets
            # stabilization settle at 1238--1392 nodes and recovered 5/5.
            1_536,
            2,
            min_ari=0.70,
            min_coverage=0.60,
        ),
    )


def run_scene(scene: Scene, seed: int) -> Result:
    data = scene.factory(seed)
    if scene.name == "linked_tori" and not data.metadata["resolvable_k8"]:
        return Result(
            scene.name, seed, False, False, 0, None, None, None, None, 0.0,
            "generator metadata says k=8 cannot resolve the lambda=0.5 gap",
        )

    started = time.time()
    scaffold = fit_scaffold(
        data.points,
        tau=scene.tau(data),
        max_nodes=scene.max_nodes,
        seed=seed,
    )
    selection = select_level_set_partition(scaffold, LevelSetConfig())
    elapsed = time.time() - started

    if scene.expected_k is None:
        passed = not selection.accepted
        return Result(
            scene.name, seed, passed, selection.accepted, len(scaffold.nodes),
            (
                selection.cluster_result.n_clusters
                if selection.cluster_result is not None else None
            ),
            None, None, None, elapsed,
            "connected-manifold null must reject",
        )

    if selection.cluster_result is None:
        return Result(
            scene.name, seed, False, False, len(scaffold.nodes), None,
            None, None, None, elapsed, "expected split rejected",
        )

    positions = scaffold.node_positions()
    _, bmu = cKDTree(positions).query(data.points, k=1)
    predicted = selection.cluster_result.labels[bmu]
    ari, background_recall, coverage = score(data.labels, predicted)
    k = selection.cluster_result.n_clusters
    passed = (
        k == scene.expected_k
        and ari >= scene.min_ari
        and coverage >= scene.min_coverage
    )
    return Result(
        scene.name, seed, passed, True, len(scaffold.nodes), k,
        float(ari), float(background_recall), float(coverage), elapsed,
        (
            f"require K={scene.expected_k}, ARI>={scene.min_ari:.2f}, "
            f"coverage>={scene.min_coverage:.2f}"
        ),
    )


def _format(result: Result) -> str:
    verdict = "PASS" if result.passed else "FAIL"
    split = (
        f"K={result.n_clusters}" if result.accepted
        else "REJECT"
    )
    metrics = ""
    if result.ari is not None:
        metrics = (
            f" ARI={result.ari:.3f} bg={result.background_recall:.3f}"
            f" cover={result.coverage:.3f}"
        )
    return (
        f"{verdict:4s} {result.scene:19s} s{result.seed} "
        f"nodes={result.n_nodes:4d} {split:7s}{metrics} "
        f"t={result.elapsed:5.1f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    args = parser.parse_args()

    results: list[Result] = []
    for scene in _scenes():
        for seed in args.seeds:
            result = run_scene(scene, int(seed))
            results.append(result)
            print(_format(result), flush=True)

    passed = sum(result.passed for result in results)
    print(f"summary: {passed}/{len(results)} scene-seeds passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
