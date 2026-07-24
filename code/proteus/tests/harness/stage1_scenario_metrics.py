"""Shared Stage 1 scenario diagnostics (circle, Swiss roll, hierarchy).

Stage 1 reconstruction is mean / max min-distance from data points to
learned prototype positions (``tests.metrics.reconstruction``).  Normalization
is dataset-specific so thresholds stay interpretable:

* **Circle:** ``mean_min_dist / radius`` (generator radius).
* **Swiss roll:** ``mean_min_dist / S`` where ``S`` is the axis-aligned bounding
  box diagonal of the generated point cloud (intrinsic sheet embedded in R^3).
* **Hierarchy:** ``mean_min_dist / S`` where ``S`` is the mean pairwise
  distance among a fixed-size random subsample of data rows (scale is
  data-driven, not tied to blob layout).

When Stage 2 adds density-based reconstruction, re-run the same normalized
Stage 1 metrics for apples-to-apples comparison (see OPEN_ISSUES / README).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from proteus.stage1 import Stage1Scaffold
from proteus.stage1.clustering import ClusterResult, run_clustering
from proteus.stage1.controller import ScaleSearchConfig, ScaleSearchResult, run_scale_search
from proteus.stage1.stabilization import StabilizationConfig
from tests.metrics.reconstruction import max_min_distance, mean_min_distance


@dataclass
class Stage1ScenarioReport:
    """Aggregated Stage 1 metrics after training (cluster fields optional)."""

    mean_min_dist: float
    max_min_dist: float
    n_nodes: int
    n_lifted_edges: int
    n_lifted_components: int
    n_isolated_lifted: int
    wall_seconds: float
    epochs_ran: int
    n_clusters: int | None = None
    partition_q_score: float | None = None


def _lifted_graph(scaffold: Any) -> dict[int, list[int]]:
    return scaffold.links.neighbour_graph(len(scaffold.nodes))


def count_lifted_components(n_nodes: int, graph: dict[int, list[int]]) -> int:
    """Number of connected components in the undirected lifted-edge graph."""

    visited = [False] * n_nodes
    components = 0
    for start in range(n_nodes):
        if visited[start]:
            continue
        components += 1
        if not graph.get(start):
            visited[start] = True
            continue
        stack = [start]
        visited[start] = True
        while stack:
            u = stack.pop()
            for v in graph.get(u, []):
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
    return components


def collect_stage1_scaffold_metrics(
    scaffold: Any,
    data: np.ndarray,
    *,
    cluster_result: ClusterResult | None = None,
    wall_seconds: float = 0.0,
    epochs_ran: int = 0,
) -> Stage1ScenarioReport:
    """Aggregate reconstruction, lifted-graph, and optional clustering stats."""

    reps = scaffold.node_positions()
    data_arr = np.asarray(data, dtype=float)
    graph = _lifted_graph(scaffold)
    n = len(scaffold.nodes)
    n_lifted = len(scaffold.links.lifted_links())
    n_isolated = sum(1 for i in range(n) if len(graph.get(i, [])) == 0)

    n_cl: int | None = None
    pq: float | None = None
    if cluster_result is not None:
        n_cl = int(cluster_result.n_clusters)
        pq = float(cluster_result.partition_q_score)

    return Stage1ScenarioReport(
        mean_min_dist=mean_min_distance(data_arr, reps),
        max_min_dist=max_min_distance(data_arr, reps),
        n_nodes=n,
        n_lifted_edges=n_lifted,
        n_lifted_components=count_lifted_components(n, graph),
        n_isolated_lifted=n_isolated,
        wall_seconds=float(wall_seconds),
        epochs_ran=int(epochs_ran),
        n_clusters=n_cl,
        partition_q_score=pq,
    )


DatasetKind = Literal["circle", "swiss_roll", "hierarchy"]


def swiss_roll_extent_scale(points: np.ndarray) -> float:
    """Axis-aligned bounding-box diagonal length (R^3 Swiss roll embedding)."""

    span = points.max(axis=0) - points.min(axis=0)
    return float(np.linalg.norm(span))


def hierarchy_pairwise_mean_scale(
    data: np.ndarray,
    *,
    n_pairs: int = 4096,
    rng: np.random.Generator | None = None,
) -> float:
    """Mean L2 distance over random point pairs (data-driven layout scale)."""

    rng = rng if rng is not None else np.random.default_rng(0)
    data_arr = np.asarray(data, dtype=float)
    n = data_arr.shape[0]
    if n < 2:
        return 1.0
    a = rng.integers(0, n, size=n_pairs)
    b = rng.integers(0, n, size=n_pairs)
    d = np.linalg.norm(data_arr[a] - data_arr[b], axis=1)
    return float(np.mean(d))


def normalize_stage1_reconstruction(
    report: Stage1ScenarioReport,
    kind: DatasetKind,
    *,
    radius: float | None = None,
    data: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Return ``mean_norm``, ``max_norm`` = raw distances / dataset scale."""

    if kind == "circle":
        if radius is None or radius <= 0:
            raise ValueError("circle normalization requires positive radius")
        scale = float(radius)
    elif kind == "swiss_roll":
        if data is None:
            raise ValueError("swiss_roll normalization requires data array")
        scale = swiss_roll_extent_scale(np.asarray(data, dtype=float))
        if scale <= 0:
            scale = 1.0
    elif kind == "hierarchy":
        if data is None:
            raise ValueError("hierarchy normalization requires data array")
        scale = hierarchy_pairwise_mean_scale(np.asarray(data, dtype=float), rng=rng)
        if scale <= 0:
            scale = 1.0
    else:
        raise ValueError(f"unknown dataset kind {kind!r}")

    return {
        "scale": scale,
        "mean_norm": float(report.mean_min_dist / scale),
        "max_norm": float(report.max_min_dist / scale),
    }


@dataclass
class FixedTauTrainResult:
    scaffold: Stage1Scaffold
    report: Stage1ScenarioReport
    wall_seconds: float
    epochs_ran: int
    cluster_result: ClusterResult | None = None


def run_fixed_tau_stable_and_report(
    data: np.ndarray,
    *,
    dim: int,
    tau: float,
    stabilization: StabilizationConfig,
    k: int = 8,
    min_nodes: int = 4,
    max_nodes: int = 128,
    n_seeds: int = 8,
    prune_after: int = 10,
    ann_backend: str = "naive",
    rng: np.random.Generator | None = None,
    cluster: bool = False,
) -> FixedTauTrainResult:
    """Train scaffold at fixed ``tau`` until stable; return metrics."""

    rng = rng if rng is not None else np.random.default_rng(0)
    data_arr = np.asarray(data, dtype=float)
    scaffold = Stage1Scaffold(
        dim=dim,
        tau=float(tau),
        k=k,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        prune_after=prune_after,
        ann_backend=ann_backend,
        rng=rng,
    )
    scaffold.init_from(data_arr, n_seeds=min(n_seeds, data_arr.shape[0]))
    t0 = time.perf_counter()
    history = scaffold.run_until_stable(data_arr, stabilization)
    wall = time.perf_counter() - t0
    epochs = len(history["cv"])
    cr = run_clustering(scaffold) if cluster else None
    report = collect_stage1_scaffold_metrics(
        scaffold,
        data_arr,
        cluster_result=cr,
        wall_seconds=wall,
        epochs_ran=epochs,
    )
    return FixedTauTrainResult(
        scaffold=scaffold,
        report=report,
        wall_seconds=wall,
        epochs_ran=epochs,
        cluster_result=cr,
    )


@dataclass
class ScaleSearchTrainResult:
    """Outcome of ``run_scale_search`` plus standardized metrics."""

    result: ScaleSearchResult
    report: Stage1ScenarioReport
    wall_seconds: float
    epochs_ran: int


def run_scale_search_and_report(
    data: np.ndarray,
    dim: int,
    config: ScaleSearchConfig,
    *,
    with_clustering: bool = True,
) -> ScaleSearchTrainResult:
    """Run scale search; collect Stage 1 metrics on ``scaffold_at_star``."""

    data_arr = np.asarray(data, dtype=float)
    t0 = time.perf_counter()
    result = run_scale_search(data_arr, dim, config)
    wall = time.perf_counter() - t0
    scaffold = result.scaffold_at_star
    cr = run_clustering(scaffold) if with_clustering else None
    epochs = int(getattr(result, "epochs_at_tau_star", 0))
    report = collect_stage1_scaffold_metrics(
        scaffold,
        data_arr,
        cluster_result=cr,
        wall_seconds=wall,
        epochs_ran=epochs,
    )
    return ScaleSearchTrainResult(
        result=result,
        report=report,
        wall_seconds=wall,
        epochs_ran=epochs,
    )
