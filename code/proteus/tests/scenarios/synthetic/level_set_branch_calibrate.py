"""Calibrate level-set geometric screens on connected-manifold nulls (#44).

Not a pytest test.  Builds C-D merge DAGs on tiny synthetic *node* clouds
(no Stage-1 equilibration) and reports, per region, the largest spurious
branch persistence and excess mass.  Thresholds should sit above the null
envelope and below true-split branches; when those ranges overlap the
screens stay conservative runt filters and DM remains the sibling arbiter.

Family-wise protocol: for each connected-manifold null, take the *largest*
non-root branch statistic in that region (not an expected-K count).  Power
is checked separately on two-blob / four-clump splits.  Full-scene
signal-only ARI, background recall, and coverage stay in
``cd_level_set_probe.py`` (hierarchy, nested spheres, repaired tori); this
script is the bounded geometric-screen study.

Usage::

    PYTHONPATH="src:$PWD" pipenv run python \\
        tests/scenarios/synthetic/level_set_branch_calibrate.py
"""

from __future__ import annotations

import numpy as np

from proteus.stage1.level_set import (
    LevelSetConfig,
    apply_geometric_screens,
    build_level_set_dag,
    build_level_set_tree,
)


def _ring(n: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = np.c_[np.cos(theta), np.sin(theta)]
    return pts + rng.normal(0.0, 0.01, pts.shape)


def _swiss_roll(n: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(1.5 * np.pi, 4.5 * np.pi, n)
    pts = np.c_[t * np.cos(t), t * np.sin(t)] / 8.0
    return pts + rng.normal(0.0, 0.01, pts.shape)


def _line(n: int = 48, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 4.0, n)
    pts = np.c_[t, np.zeros(n)]
    return pts + rng.normal(0.0, 0.01, pts.shape)


def _density_gradient(n: int = 48, seed: int = 0) -> np.ndarray:
    """Connected 1-manifold with monotone spacing (still one feature)."""

    rng = np.random.default_rng(seed)
    u = np.linspace(0.0, 1.0, n) ** 2
    pts = np.c_[4.0 * u, np.zeros(n)]
    return pts + rng.normal(0.0, 0.005, pts.shape)


def _disk(n: int = 80, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(0.0, 1.0, n))
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.c_[r * np.cos(th), r * np.sin(th)]


def _with_tissue(pts: np.ndarray, frac: float, seed: int) -> np.ndarray:
    if frac <= 0.0:
        return pts
    rng = np.random.default_rng(seed + 17)
    lo = pts.min(axis=0) - 0.5
    hi = pts.max(axis=0) + 0.5
    n_tissue = max(1, int(round(frac * len(pts))))
    tissue = rng.uniform(lo, hi, size=(n_tissue, pts.shape[1]))
    return np.vstack([pts, tissue])


def _two_blobs(n: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = max(4, n // 2)
    left = rng.normal((-2.0, 0.0), 0.05, size=(half, 2))
    right = rng.normal((2.0, 0.0), 0.05, size=(half, 2))
    tissue = rng.uniform((-1.0, -2.0), (1.0, 2.0), size=(4, 2))
    return np.vstack([left, right, tissue])


def _four_clumps(n: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    per = max(4, n // 4)
    centres = ((0.0, 0.0), (0.25, 0.0), (3.0, 0.0), (3.25, 0.0))
    clumps = [rng.normal(c, 0.02, size=(per, 2)) for c in centres]
    tissue = np.array([[12.0, 12.0], [12.0, -12.0], [-12.0, 12.0]])
    return np.vstack(clumps + [tissue])


def _region_stats(positions: np.ndarray, config: LevelSetConfig):
    tree = build_level_set_tree(positions, config)
    dag = build_level_set_dag(tree)
    roots = [
        b for b in dag.branches
        if b.parent_id is None and b.merge_level is None
    ]
    others = [b for b in dag.branches if b not in roots]
    max_p = max((b.persistence for b in others), default=0.0)
    max_m = max((b.excess_mass for b in others), default=0.0)
    other_ids = {b.branch_id for b in others}
    screened = apply_geometric_screens(dag.branches, config)
    n_kept = sum(
        1 for b in screened
        if b.prune_reason is None and b.branch_id in other_ids
    )
    return (
        len(tree.levels),
        [lv.n_clusters for lv in tree.levels],
        max_p,
        max_m,
        len(roots),
        n_kept,
    )


def main() -> None:
    config = LevelSetConfig(k_neighbors=4, min_cluster_size=4, n_levels=40)
    print(
        "protocol: family-wise max non-root branch per region; "
        f"screens P>={config.min_persistence} M>={config.min_excess_mass}"
    )
    print("scene n tissue seed levels max_spur_P max_spur_M n_roots n_unpruned_nonroot")
    nulls = [
        ("ring", _ring),
        ("swiss", _swiss_roll),
        ("line", _line),
        ("grad", _density_gradient),
        ("disk", _disk),
    ]
    positives = [("two_blobs", _two_blobs), ("four_clumps", _four_clumps)]
    null_p: list[float] = []
    null_m: list[float] = []
    pos_p: list[float] = []
    pos_m: list[float] = []
    for name, maker in nulls:
        for n in (32, 64):
            for tissue in (0.0, 0.05):
                for seed in range(3):
                    pts = _with_tissue(maker(n=n, seed=seed), tissue, seed)
                    nlev, _ks, p, m, nr, kept = _region_stats(pts, config)
                    null_p.append(p)
                    null_m.append(m)
                    print(
                        f"{name:12s} {n:3d} {tissue:4.2f} {seed} "
                        f"{nlev:3d} {p:.3f} {m:.3f} {nr} {kept}"
                    )
    print("--- positives (split-branch envelope; same family-wise max) ---")
    for name, maker in positives:
        for n in (32, 64):
            for seed in range(3):
                nlev, _ks, p, m, nr, kept = _region_stats(maker(n=n, seed=seed), config)
                pos_p.append(p)
                pos_m.append(m)
                print(
                    f"{name:12s} {n:3d} 0.00 {seed} "
                    f"{nlev:3d} {p:.3f} {m:.3f} {nr} {kept}"
                )
    print(
        "null max P/M "
        f"{max(null_p, default=0):.3f}/{max(null_m, default=0):.3f}  "
        "positive max P/M "
        f"{max(pos_p, default=0):.3f}/{max(pos_m, default=0):.3f}"
    )
    gap_p = min(pos_p, default=0) - max(null_p, default=0)
    gap_m = min(pos_m, default=0) - max(null_m, default=0)
    print(f"min-positive minus max-null: P {gap_p:.3f}  M {gap_m:.3f}")
    if gap_p <= 0.0 or gap_m <= 0.0:
        print(
            "overlap: keep operational floors "
            f"min_persistence={config.min_persistence} "
            f"min_excess_mass={config.min_excess_mass} "
            "(runt screens; DM sibling test is the connected-manifold guard)"
        )
    else:
        print(
            "suggested floors: "
            f"min_persistence>{max(null_p, default=0):.3f}  "
            f"min_excess_mass>{max(null_m, default=0):.3f}"
        )


if __name__ == "__main__":
    main()
