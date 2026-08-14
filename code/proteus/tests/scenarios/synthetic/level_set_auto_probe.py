"""Automatic (expected-K-free) level-set extraction probe (#44).

Not a pytest test.  Builds a kNN lifted graph on the point cloud, runs
``select_level_set_partition``, and scores signal-only ARI / background
recall / coverage.  This asks whether the landed DAG prune recovers the
complex suite without an oracle cluster count.

Usage::

    PYTHONPATH="src:$PWD" pipenv run python \\
        tests/scenarios/synthetic/level_set_auto_probe.py
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree

from proteus.links import LinkCounters
from proteus.stage1.level_set import (
    LevelSetConfig,
    select_level_set_partition,
)
from tests.datasets.synthetic.circles import make_circle
from tests.datasets.synthetic.hierarchical_gaussian import (
    make_hierarchical_gaussian,
)
from tests.datasets.synthetic.linked_tori import make_linked_tori
from tests.datasets.synthetic.manifold_zoo import make_manifold_zoo
from tests.datasets.synthetic.nested_spheres import make_nested_spheres
from tests.datasets.synthetic.swiss_roll import make_swiss_roll
from tests.scenarios.synthetic.cd_level_set_probe import score
from tests.scenarios.synthetic.test_multiscale_dim_membership import (
    make_rods_sheets_slab,
)


class _Node:
    def __init__(self, position: np.ndarray, dim: int) -> None:
        self.position = np.asarray(position, dtype=float)
        self.hit_count = 1.0
        self.d_final = int(dim)
        self.variance = 0.1


class _KnnScaffold:
    """Synthetic scaffold: node = sample, lifted edges = undirected kNN.

    Edge counts follow a Gaussian kernel of edge length so that flows are
    density-proportional, as real Hebbian transition counts are: traffic
    between BMUs scales with local sample density, so long tissue edges
    carry near-zero flow.  A uniform-count graph would overweight tissue
    and defeat the flow-bottleneck guard the probe is meant to exercise.
    """

    def __init__(self, points: np.ndarray, k: int = 8) -> None:
        self.nodes = [_Node(p, points.shape[1]) for p in points]
        self.links = LinkCounters()
        self.tau = 1.0
        n = int(points.shape[0])
        k_use = max(1, min(int(k), n - 1))
        dist, idx = cKDTree(points).query(points, k=k_use + 1)
        sigma = float(np.median(dist[:, 1:]))
        for i, (drow, nbrs) in enumerate(zip(dist, idx)):
            for d, j in zip(drow[1:], nbrs[1:]):
                w = float(np.exp(-((float(d) / sigma) ** 2)))
                self.links.increment_directed(i, int(j), w, lift=True)
                self.links.increment_directed(int(j), i, w, lift=True)


def _run(
    name: str,
    points: np.ndarray,
    labels: np.ndarray,
    expect: str,
) -> None:
    t0 = time.time()
    scaffold = _KnnScaffold(points)
    selection = select_level_set_partition(scaffold, LevelSetConfig())
    elapsed = time.time() - t0
    ks = [lv.n_clusters for lv in selection.tree.levels]
    if selection.accepted and selection.cluster_result is not None:
        pred = selection.cluster_result.labels
        ari, bg, cov = score(labels, pred)
        k = selection.cluster_result.n_clusters
        line = (
            f"{name:18s} n={len(points):5d} K={k} "
            f"lvl={selection.selected_level} "
            f"sig_ARI={ari:.3f} cover={cov:.2f} bg_rec={bg:.2f} "
            f"logBF={selection.log_bf:.1f} t={elapsed:.1f}s"
        )
    else:
        line = (
            f"{name:18s} n={len(points):5d} REJECT "
            f"logBF={selection.log_bf:.1f} t={elapsed:.1f}s"
        )
    print(line)
    print(f"  expect {expect}; Ks={ks[:40]}{'...' if len(ks) > 40 else ''}")
    reasons = {}
    for branch in selection.branches:
        reasons[branch.prune_reason or "kept"] = (
            reasons.get(branch.prune_reason or "kept", 0) + 1
        )
    print(f"  branches={len(selection.branches)} prune={reasons}")


def main() -> None:
    print("=== automatic level-set extraction (kNN graph, no expected K) ===")
    ds = make_circle(n_samples=800, tissue_fraction=0.03, seed=0)
    _run("circle", ds.points, ds.labels, "reject / one feature")

    ds = make_swiss_roll(n_samples=1000, tissue_fraction=0.03, seed=0)
    _run("swiss_roll", ds.points, ds.labels, "reject / one feature")

    ds = make_manifold_zoo(tissue_fraction=0.03, seed=0)
    _run("manifold_zoo", ds.points, ds.labels, "reject / one connected scene")

    # n_per_sphere=800 is below the valley-resolution budget: the outer
    # shell never forms a large branch and the region correctly rejects.
    # 3000 per sphere is the resolvable regime (#44 node-budget caveat).
    ds = make_nested_spheres(n_per_sphere=3000, tissue_fraction=0.03, seed=0)
    _run("nested_spheres", ds.points, ds.labels, "K=2 shells")

    ds = make_hierarchical_gaussian(n_samples=600, seed=0)
    _run(
        "hierarchy",
        ds.points,
        ds.labels,
        "coarsest K=3 (fine K=6 in children)",
    )

    pts, rod, sheet = make_rods_sheets_slab(seed=0)
    _run("rods_sheets_slab", pts, sheet, "coarsest K=2 sheets (rods finer)")

    tori = make_linked_tori(n_per_torus=4_000, seed=0)
    assert tori.metadata["resolvable_k8"]
    _run("linked_tori", tori.points, tori.labels, "K=2 tori")


if __name__ == "__main__":
    main()
