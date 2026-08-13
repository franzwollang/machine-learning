"""Chaudhuri-Dasgupta density level-set probes for issue #44 (not a test).

Runnable validation probes for the level-set pivot (see OPEN_ISSUES #44 and
``reference/empty_region_evidence_and_scale.md``): the target object for
Stage-1 separation is the Hartigan density cluster tree — connected
components of upper level sets ``{p >= lambda}`` with an explicit background
class — estimated by Chaudhuri-Dasgupta (2010) robust single linkage.

Two probes:

``batch``
    C-D robust single linkage on raw samples (the oracle). Sweeps activation
    levels ``r`` over quantiles of ``r_k`` (distance to the k-th neighbour),
    connects active points within ``alpha * r``, and reports the best level
    at the expected component count.

``scaffold``
    The scaffold-native read. Fit a ``Stage1Scaffold`` at a fine ``tau`` with
    a raised node cap, estimate per-node density from node spacing (the
    equalized code places nodes with density a monotone transform of ``p``;
    the cluster tree is invariant to monotone transforms), run the same C-D
    sweep over nodes (``k_node = 8``, ``alpha = 1.0``), and transfer node
    labels to samples via BMU.

Scoring (GT convention: labels < 0 = tissue and fade halo): signal-only ARI,
coverage of signal points, and background recall. Full-cloud ARI is
structurally capped because faded generators label ~half the samples as
halo; see OPEN_ISSUES #45 for the benchmark defects this probe surfaced
(linked_tori tube overlap at default minor_radius, anchor-clumped sampling).

Measured results (2026-08-12, seeds 0-2) are recorded in OPEN_ISSUES #44.

Usage::

    PYTHONPATH="src:$PWD" pipenv run python \
        tests/scenarios/synthetic/cd_level_set_probe.py batch
    PYTHONPATH="src:$PWD" pipenv run python \
        tests/scenarios/synthetic/cd_level_set_probe.py scaffold
"""

from __future__ import annotations

import sys
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score

from proteus.stage1.scaffold import Stage1Scaffold
from proteus.stage1.stabilization import StabilizationConfig

from tests.datasets.synthetic.circles import make_circle
from tests.datasets.synthetic.hierarchical_gaussian import (
    make_hierarchical_gaussian,
)
from tests.datasets.synthetic.manifold_zoo import make_manifold_zoo
from tests.datasets.synthetic.nested_spheres import make_nested_spheres
from tests.datasets.synthetic.swiss_roll import make_swiss_roll


# --------------------------------------------------------------------------
# corrected linked tori (see OPEN_ISSUES #45: the repo generator's tubes
# overlap at minor_radius=0.5 and its anchor-kernel sampling is clumpy)
# --------------------------------------------------------------------------

def uniform_torus(n: int, big_r: float, small_r: float, rng) -> np.ndarray:
    """Uniform area-correct sampling of a torus surface in R^3."""
    pts: list[np.ndarray] = []
    while len(pts) < n:
        th = rng.uniform(0, 2 * np.pi, n)
        ph = rng.uniform(0, 2 * np.pi, n)
        keep = rng.uniform(0, 1, n) < (
            (big_r + small_r * np.cos(ph)) / (big_r + small_r)
        )
        th, ph = th[keep], ph[keep]
        x = (big_r + small_r * np.cos(ph)) * np.cos(th)
        y = (big_r + small_r * np.cos(ph)) * np.sin(th)
        z = small_r * np.sin(ph)
        pts.extend(np.c_[x, y, z])
    return np.array(pts[:n])


def make_uniform_linked_tori(n_per=4000, minor_radius=0.25, seed=0):
    """Hopf-linked tori with a real gap and continuous surface sampling."""
    rng = np.random.default_rng(seed)
    big_r = 2.0
    t1 = uniform_torus(n_per, big_r, minor_radius, rng)
    t1 = t1 + rng.normal(0, 0.02, t1.shape)
    t2 = uniform_torus(n_per, big_r, minor_radius, rng)
    t2 = t2 + rng.normal(0, 0.02, t2.shape)
    t2 = np.c_[t2[:, 2] + big_r, t2[:, 1], t2[:, 0]]
    n_tissue = n_per // 16
    lo = np.vstack([t1, t2]).min(0) - 0.3
    hi = np.vstack([t1, t2]).max(0) + 0.3
    tissue = rng.uniform(lo, hi, (n_tissue, 3))
    points = np.vstack([t1, t2, tissue])
    labels = np.r_[
        np.zeros(n_per, int),
        np.ones(n_per, int),
        -np.ones(n_tissue, int),
    ]
    return points, labels


# --------------------------------------------------------------------------
# scoring and the C-D level sweep
# --------------------------------------------------------------------------

def score(y_true, lab):
    """(signal-only ARI, background recall, signal coverage)."""
    sig = y_true >= 0
    sig_ari = adjusted_rand_score(y_true[sig], lab[sig])
    bg = ~sig
    bg_rec = float((lab[bg] < 0).mean()) if bg.any() else 1.0
    cover = float((lab[sig] >= 0).mean())
    return sig_ari, bg_rec, cover


def cd_sweep(X, k, alpha, n_levels=120, min_size=10):
    """C-D robust single linkage level sweep.

    Yields ``(r, K, labels)`` per level: activate points with
    ``r_k <= r``, connect active pairs within ``alpha * r``; clusters are
    connected components with at least ``min_size`` members, everything
    else is background (-1).
    """
    n = X.shape[0]
    tree = cKDTree(X)
    dists, _ = tree.query(X, k=k + 1)
    r_k = dists[:, -1]
    out = []
    for r in np.unique(np.quantile(r_k, np.linspace(0.02, 1.0, n_levels))):
        active = np.where(r_k <= r)[0]
        if len(active) < min_size:
            continue
        sub = X[active]
        pairs = cKDTree(sub).query_pairs(alpha * r, output_type="ndarray")
        m = len(active)
        if len(pairs):
            g = csr_matrix(
                (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                shape=(m, m),
            )
        else:
            g = csr_matrix((m, m))
        _, comp = connected_components(g, directed=False)
        lab = np.full(n, -1, dtype=int)
        sizes = np.bincount(comp)
        big = np.where(sizes >= min_size)[0]
        remap = {c: i for i, c in enumerate(big)}
        for j, c in zip(active, comp):
            lab[j] = remap.get(c, -1)
        out.append((float(r), len(big), lab))
    return out


def best_at_k(levels, y, expect_k, project=None):
    """Best (by ARI x coverage) level with exactly ``expect_k`` clusters."""
    best = None
    hits = 0
    for _, big_k, lab in levels:
        if big_k != expect_k:
            continue
        hits += 1
        slab = lab if project is None else lab[project]
        ari, bg, cov = score(y, slab)
        if best is None or ari * min(cov, 1.0) > best[0]:
            best = (ari * min(cov, 1.0), ari, cov, bg)
    width = hits / max(len(levels), 1)
    return best, width


# --------------------------------------------------------------------------
# probe (a): batch oracle on raw samples
# --------------------------------------------------------------------------

def batch_scene(name, X, y, expect_k, extra_k=None):
    n, d = X.shape
    print(f"--- {name} (n={n}, d={d}, expect K={expect_k}) ---")
    for k in (max(5, int(round(d * np.log(n)))), 12):
        for alpha in (1.0, float(np.sqrt(2.0))):
            levels = cd_sweep(X, k=k, alpha=alpha)
            best, width = best_at_k(levels, y, expect_k)
            tag = f"k={k:3d} a={alpha:.2f} | "
            if best:
                _, ari, cov, bg = best
                tag += (
                    f"K={expect_k}: sig_ARI={ari:.3f} cover={cov:.2f} "
                    f"bg_rec={bg:.2f} width={width:.2f}"
                )
            else:
                tag += f"K={expect_k} never reached"
            if extra_k is not None:
                b2, _ = best_at_k(levels, y, extra_k)
                if b2:
                    tag += f" | K={extra_k}: sig_ARI={b2[1]:.3f}"
            print("  " + tag)


def run_batch(seed=0, tissue_fraction=0.03):
    print(f"=== batch C-D probe | seed={seed} tf={tissue_fraction} ===")
    ds = make_circle(n_samples=1500, tissue_fraction=tissue_fraction,
                     seed=seed)
    batch_scene("circle", ds.points, ds.labels, 1)
    ds = make_swiss_roll(n_samples=2000, tissue_fraction=tissue_fraction,
                         seed=seed)
    batch_scene("swiss_roll", ds.points, ds.labels, 1)
    ds = make_nested_spheres(tissue_fraction=tissue_fraction, seed=seed)
    batch_scene("nested_spheres", ds.points, ds.labels, 2)
    ds = make_manifold_zoo(tissue_fraction=tissue_fraction, seed=seed)
    batch_scene("manifold_zoo", ds.points, ds.labels, 1)
    ds = make_hierarchical_gaussian(seed=seed)
    n_fine = len(set(int(v) for v in ds.labels[ds.labels >= 0]))
    batch_scene("hier_gaussian", ds.points, ds.labels, n_fine, extra_k=3)
    X, y = make_uniform_linked_tori(n_per=8000, seed=seed)
    batch_scene("uniform_linked_tori", X, y, 2)


# --------------------------------------------------------------------------
# probe (b): scaffold-native node-spacing read
# --------------------------------------------------------------------------

def fit_scaffold(X, tau, max_nodes, seed=0, max_epochs=25):
    scaffold = Stage1Scaffold(
        dim=X.shape[1],
        tau=float(tau),
        k=8,
        min_nodes=8,
        max_nodes=int(max_nodes),
        ann_backend="naive",
        rng=np.random.default_rng(seed),
    )
    scaffold.init_from(X, n_seeds=8)
    scaffold.run_until_stable(X, StabilizationConfig(max_epochs=max_epochs))
    return scaffold


def scaffold_scene(name, X, y, tau, expect_k, max_nodes=512, extra_k=None,
                   seed=0):
    t0 = time.time()
    scaffold = fit_scaffold(X, tau, max_nodes, seed=seed)
    positions = scaffold.node_positions()
    _, bmu = cKDTree(positions).query(X, k=1)
    # k_node=8 spacing density + plain C-D linking at alpha=1.0: the single
    # reader that recovered every scene in the 2026-08-12 probe grid.
    levels = cd_sweep(positions, k=8, alpha=1.0, min_size=4)
    best, width = best_at_k(levels, y, expect_k, project=bmu)
    line = (
        f"{name:20s} s{seed} nodes={len(positions):4d} "
        f"fit={time.time() - t0:5.1f}s | "
    )
    if best:
        _, ari, cov, bg = best
        line += (
            f"K={expect_k}: sig_ARI={ari:.3f} cover={cov:.2f} "
            f"bg_rec={bg:.2f} width={width:.2f}"
        )
    else:
        line += f"K={expect_k} never reached"
    if extra_k is not None:
        b2, _ = best_at_k(levels, y, extra_k, project=bmu)
        if b2:
            line += f" | K={extra_k}: sig_ARI={b2[1]:.3f}"
    print(line, flush=True)


def run_scaffold(seeds=(0, 1, 2)):
    print("=== scaffold-spacing C-D probe ===")
    for seed in seeds:
        ds = make_circle(n_samples=1500, seed=seed)
        scaffold_scene(
            "circle", ds.points, ds.labels,
            ds.ground_truth.expected_tau / 8.0, 1, seed=seed,
        )
        ds = make_swiss_roll(n_samples=2000, seed=seed)
        scaffold_scene(
            "swiss_roll", ds.points, ds.labels,
            ds.ground_truth.expected_tau / 8.0, 1, seed=seed,
        )
        ds = make_nested_spheres(n_per_sphere=3000, seed=seed)
        scaffold_scene(
            "nested_spheres", ds.points, ds.labels,
            ds.ground_truth.expected_tau / 20.0, 2,
            max_nodes=1024, seed=seed,
        )
        ds = make_hierarchical_gaussian(seed=seed)
        n_fine = len(set(int(v) for v in ds.labels[ds.labels >= 0]))
        scaffold_scene(
            "hier_gaussian", ds.points, ds.labels,
            ds.ground_truth.expected_tau / 3.0, n_fine,
            extra_k=3, seed=seed,
        )
        X, y = make_uniform_linked_tori(n_per=4000, seed=seed)
        scaffold_scene(
            "uniform_linked_tori", X, y, 0.02, 2,
            max_nodes=768, seed=seed,
        )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "batch"
    if mode == "batch":
        run_batch(seed=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    elif mode == "scaffold":
        run_scaffold()
    else:
        raise SystemExit(f"unknown mode {mode!r} (use batch|scaffold)")
