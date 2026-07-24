"""Helpers to compare recursion trees to hierarchical Gaussian ground truth.

Each ``RecursionNode.sample_indices`` holds **original dataset row indices**
(see ``proteus.stage1.recursion``).  Leaf partitions vs ``fine_labels`` support
ARI checks.

**Primary (hierarchical GT):** ``assert_recursion_matches_gt_hierarchy_unimodal_levels``
builds a **bottom-up** τ-smoothed unimodal Gaussian per ``ClusterNode`` group in
``make_hierarchical_gaussian`` (fine → coarse pairs → root), using root
``tau_star`` for ``Σ_smooth = tau * I`` (see ``OPEN_ISSUES`` §32).  At each
``level``, same-level recursion nodes are matched to GT groups by **Hungarian
assignment on mean L2**; **mean** gates use a **chi-squared** threshold on a Hotelling statistic
under the τ-smoothed mixture (analytic ``Cov(X)`` for one draw, Gaussian reference for ``x̄``);
**covariance** gates compare sample covariance to the **raw** fine-leaf mixture second moment with a
**Frobenius** tolerance (no Monte Carlo).

Scenario tests require **full fine resolution**: one terminal leaf per fine
Gaussian (``assert_terminal_leaf_count_equals_fine_components``) and high
``ARI`` vs fine labels.  Optional ``required_levels`` forces the unimodal harness
to run at every listed GT depth (no silent skip when the tree lacks nodes).

**Legacy:** ``assert_recursion_gaussian_means_match_gt_hierarchy`` — raw
per-level centers without τ-unimodal construction.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from sklearn.metrics import adjusted_rand_score

from tests.datasets.ground_truth import ClusterNode


def nodes_by_region_id(tree: Any) -> dict[int, Any]:
    """``region_id -> RecursionNode``."""

    return {int(n.region_id): n for n in tree.nodes}


def resolve_tau_star(node: Any, tree: Any) -> float | None:
    """``tau_star`` on this node, else walk ``parent_id`` until one is set."""

    by_id = nodes_by_region_id(tree)
    visited: set[int] = set()
    cur: Any | None = node
    while cur is not None:
        rid = int(cur.region_id)
        if rid in visited:
            return None
        visited.add(rid)
        if cur.tau_star is not None:
            return float(cur.tau_star)
        pid = cur.parent_id
        if pid is None:
            return None
        cur = by_id.get(int(pid))
    return None


def smoothing_covariance_isotropic(tau: float, dim: int) -> np.ndarray:
    """Provisional map ``tau -> Σ_smooth`` for Gaussian convolution (ambient ``tau * I``).

    See ``OPEN_ISSUES`` §32: replace with subspace / anisotropic kernel once
    theory and Stage 1 agree on a canonical map.
    """

    t = float(tau)
    if t <= 0.0 or not np.isfinite(t):
        raise ValueError(f"tau must be finite and positive, got {tau}")
    return np.eye(int(dim), dtype=float) * t


def _sym(A: np.ndarray) -> np.ndarray:
    a = np.asarray(A, dtype=float)
    return 0.5 * (a + a.T)


def moment_matched_gaussian_from_components(
    means: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """First two moments of a finite Gaussian mixture (``means`` shape ``(K, D)``)."""

    w = np.asarray(weights, dtype=float).reshape(-1)
    w = w / max(float(w.sum()), 1e-15)
    mu = (w[:, None] * np.asarray(means, dtype=float)).sum(axis=0)
    d = int(mu.shape[0])
    cov = np.zeros((d, d), dtype=float)
    for k in range(int(means.shape[0])):
        dk = np.asarray(means[k], dtype=float).reshape(-1) - mu
        cov += float(w[k]) * (np.asarray(covs[k], dtype=float).reshape(d, d) + np.outer(dk, dk))
    return mu, cov


def mixture_observation_covariance(
    means: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """``Cov(X)`` for a single draw ``X`` from a Gaussian mixture (components ``N(μ_k, Σ_k)``)."""

    mm = np.asarray(means, dtype=float)
    k_tot, d = int(mm.shape[0]), int(mm.shape[1])
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = w / max(float(w.sum()), 1e-15)
    mu = (w[:, None] * mm).sum(axis=0)
    exxt = np.zeros((d, d), dtype=float)
    for k in range(k_tot):
        mk = mm[k].reshape(-1)
        sk = _sym(np.asarray(covs[k], dtype=float).reshape(d, d))
        exxt += float(w[k]) * (sk + np.outer(mk, mk))
    return _sym(exxt - np.outer(mu, mu))


def _hotelling_stat_mixture_mean(
    emp_mean: np.ndarray,
    mu_pop: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
    n: int,
    *,
    inv_reg: float,
) -> float:
    """``n * (x̄ - μ)^T Cov(X)^{-1} (x̄ - μ)`` with ``Cov(X)`` the mixture single-draw covariance."""

    cov_x = mixture_observation_covariance(means, covs, weights)
    d = int(cov_x.shape[0])
    reg = max(float(inv_reg), 1e-15 * float(np.trace(cov_x)) / max(d, 1))
    cov_r = cov_x + np.eye(d, dtype=float) * reg
    delta = np.asarray(emp_mean, dtype=float).reshape(-1) - np.asarray(mu_pop, dtype=float).reshape(-1)
    # n * delta^T Cov(X)^{-1} delta  ~  χ²_d  when X ~ single Gaussian; mixture: same scale as reference
    sol = np.linalg.solve(cov_r, delta)
    return float(n) * float(delta @ sol)


def _sample_cov_shrunk(x: np.ndarray, *, min_cov_diag: float = 1e-9) -> np.ndarray:
    """Unbiased sample covariance with tiny diagonal jitter (requires ``n >= 2``)."""

    x = np.asarray(x, dtype=float)
    n, d = x.shape
    if n < 2:
        return np.eye(d, dtype=float) * float(min_cov_diag)
    c = np.cov(x.T, bias=False)
    c = np.asarray(c, dtype=float).reshape(d, d)
    return c + np.eye(d, dtype=float) * float(min_cov_diag)


def children_by_parent_cluster_id(hierarchy: Sequence[ClusterNode]) -> dict[int, list[ClusterNode]]:
    """``parent_cluster_id -> [child ClusterNode, ...]`` sorted by ``cluster_id``."""

    ch: dict[int, list[ClusterNode]] = defaultdict(list)
    for c in hierarchy:
        if c.parent_id is None:
            continue
        ch[int(c.parent_id)].append(c)
    for k in ch:
        ch[k].sort(key=lambda x: int(x.cluster_id))
    return dict(ch)


def gt_analytic_unimodal_by_cluster_id(
    hierarchy: Sequence[ClusterNode],
    tau: float,
    dim: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Bottom-up τ-smoothed unimodal ``(μ, Σ)`` per GT cluster (fine → coarse → root).

    * **Fine (leaves):** ``Σ = Σ_gt + τ I`` around ``center``.
    * **Coarse (level 1):** moment-match two smoothed fine children (GT weights).
    * **Root (level 0):** moment-match three **coarse unimodals**, each with an
      extra ``τ I`` before merging (second smoothing pass at root scale).
    """

    sig = smoothing_covariance_isotropic(tau, dim)
    by_parent = children_by_parent_cluster_id(hierarchy)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    fine_nodes = [c for c in hierarchy if c.is_leaf]
    for c in sorted(fine_nodes, key=lambda x: int(x.cluster_id)):
        mu = np.asarray(c.center, dtype=float).reshape(-1)
        cov = _sym(np.asarray(c.covariance, dtype=float).reshape(dim, dim) + sig)
        out[int(c.cluster_id)] = (mu, cov)

    coarse_nodes = [c for c in hierarchy if int(c.level) == 1]
    for c in sorted(coarse_nodes, key=lambda x: int(x.cluster_id)):
        kids = by_parent[int(c.cluster_id)]
        w = np.asarray([float(k.weight) for k in kids], dtype=float)
        w = w / float(w.sum())
        means = np.stack([np.asarray(k.center, dtype=float).reshape(-1) for k in kids], axis=0)
        covs = np.stack(
            [_sym(np.asarray(k.covariance, dtype=float).reshape(dim, dim) + sig) for k in kids],
            axis=0,
        )
        mu_um, cov_um = moment_matched_gaussian_from_components(means, covs, w)
        out[int(c.cluster_id)] = (_sym(mu_um), _sym(cov_um))

    root = next(c for c in hierarchy if int(c.level) == 0)
    kids = by_parent[int(root.cluster_id)]
    w = np.asarray([float(k.weight) for k in kids], dtype=float)
    w = w / float(w.sum())
    means = np.stack([out[int(k.cluster_id)][0] for k in kids], axis=0)
    covs = np.stack([out[int(k.cluster_id)][1] + sig for k in kids], axis=0)
    mu_um, cov_um = moment_matched_gaussian_from_components(means, covs, w)
    out[int(root.cluster_id)] = (_sym(mu_um), _sym(cov_um))
    return out


def tau_smoothed_gt_mixture_components(
    c: ClusterNode,
    *,
    analytic_by_id: dict[int, tuple[np.ndarray, np.ndarray]],
    by_parent: dict[int, list[ClusterNode]],
    tau: float,
    dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(means, covs, weights)`` for the τ-smoothed Gaussian mixture at ``c`` (same law as analytic construction)."""

    sig = smoothing_covariance_isotropic(tau, dim)
    if c.is_leaf:
        mu = np.asarray(c.center, dtype=float).reshape(-1)
        cov = _sym(np.asarray(c.covariance, dtype=float).reshape(dim, dim) + sig)
        return mu.reshape(1, -1), cov.reshape(1, dim, dim), np.ones(1, dtype=float)

    if int(c.level) == 1:
        kids = by_parent[int(c.cluster_id)]
        w = np.asarray([float(k.weight) for k in kids], dtype=float)
        w = w / float(w.sum())
        means = np.stack([np.asarray(k.center, dtype=float).reshape(-1) for k in kids], axis=0)
        covs = np.stack(
            [_sym(np.asarray(k.covariance, dtype=float).reshape(dim, dim) + sig) for k in kids],
            axis=0,
        )
        return means, covs, w

    kids = by_parent[int(c.cluster_id)]
    w = np.asarray([float(k.weight) for k in kids], dtype=float)
    w = w / float(w.sum())
    means = np.stack([analytic_by_id[int(k.cluster_id)][0] for k in kids], axis=0)
    covs = np.stack([analytic_by_id[int(k.cluster_id)][1] + sig for k in kids], axis=0)
    return means, covs, w


def subtree_leaf_raw_mixture_params(
    c: ClusterNode,
    *,
    by_parent: dict[int, list[ClusterNode]],
    dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finest-leaf descendants of ``c`` as a raw Gaussian mixture (no ``τ``).

    Used for covariance comparison against ``make_hierarchical_gaussian`` draws.
    Means/covs come from leaf ``ClusterNode`` geometry; weights normalized.
    """

    stack = [c]
    leaves: list[ClusterNode] = []
    while stack:
        u = stack.pop()
        ch = by_parent.get(int(u.cluster_id), [])
        if not ch:
            if u.is_leaf:
                leaves.append(u)
        else:
            stack.extend(ch)
    leaves = sorted(leaves, key=lambda x: int(x.cluster_id))
    if not leaves:
        raise ValueError(f"no leaf descendants under cluster_id={c.cluster_id}")
    w = np.asarray([float(v.weight) for v in leaves], dtype=float)
    w = w / float(w.sum())
    means = np.stack([np.asarray(v.center, dtype=float).reshape(-1) for v in leaves], axis=0)
    covs = np.stack([np.asarray(v.covariance, dtype=float).reshape(dim, dim) for v in leaves], axis=0)
    return means, covs, w


def assert_recursion_matches_gt_hierarchy_unimodal_levels(
    data: np.ndarray,
    tree: Any,
    hierarchy: Sequence[ClusterNode],
    *,
    min_samples: int = 5,
    min_samples_cov: int | None = None,
    mean_chi2_alpha: float = 0.02,
    mean_inv_reg: float = 1e-7,
    cov_fro_atol: float = 0.55,
    cov_fro_rtol: float = 0.28,
    levels: set[int] | None = None,
    required_levels: Collection[int] | None = None,
    min_cov_diag: float = 1e-9,
) -> None:
    """Match recursion nodes to GT clusters by level (Hungarian on means), analytic gates.

    Skips any ``level`` where there are no recursion nodes or no GT nodes at
    that level **unless** ``required_levels`` contains that level, in which case
    missing prediction or GT nodes is an error (no silent skip).

    **Mean:** sample mean vs mixture mean under the τ-smoothed child law at ``c``;
    Hotelling statistic ``n (x̄-μ)^T Cov(X)^{-1}(x̄-μ)`` with ``Cov(X)`` the
    single-draw mixture covariance, compared to ``χ²_{d,1-α}`` (Gaussian reference;
    exact only for a single component).

    **Covariance:** Frobenius norm ``||S - Σ_ref||_F`` vs ``cov_fro_atol + cov_fro_rtol * ||Σ_ref||_F``
    where ``Σ_ref`` is the raw fine-leaf mixture second moment under the matched GT subtree.

    ``levels`` restricts which GT levels to evaluate; ``None`` means every
    level present in the union of GT and prediction keys.  ``required_levels``
    lists levels that must be evaluated (e.g. ``{0, 1, 2}`` for full depth);
    if the tree lacks nodes at a required level, this raises (no silent skip).
    """

    if not tree.nodes:
        raise AssertionError("empty recursion tree")

    req = None if required_levels is None else {int(x) for x in required_levels}
    if req is not None and levels is not None and not req <= {int(x) for x in levels}:
        raise ValueError(
            "required_levels must be contained in ``levels`` when levels is not None",
        )

    tau = resolve_tau_star(tree.nodes[0], tree)
    if tau is None:
        raise AssertionError("root has no resolved tau_star; cannot build τ-smoothed GT")

    data_arr = np.asarray(data, dtype=float)
    dim = int(data_arr.shape[1])
    if min_samples_cov is None:
        cov_n_min = max(int(min_samples), dim + 2)
    else:
        cov_n_min = int(min_samples_cov)

    analytic = gt_analytic_unimodal_by_cluster_id(hierarchy, float(tau), dim)
    by_parent = children_by_parent_cluster_id(hierarchy)
    pred_by = recursion_nodes_by_level(tree)
    gt_by = cluster_nodes_by_level(hierarchy)

    if req is not None:
        gt_levels = set(gt_by.keys())
        if not req <= gt_levels:
            raise ValueError(
                f"required_levels {sorted(req)} is not a subset of GT hierarchy levels "
                f"{sorted(gt_levels)}",
            )

    chi2_thr = float(chi2.ppf(1.0 - float(mean_chi2_alpha), dim))

    all_levels = sorted(set(gt_by.keys()) | set(pred_by.keys()))
    evaluated = 0

    for level in all_levels:
        if levels is not None and int(level) not in levels:
            continue

        pred_nodes = [
            n for n in pred_by.get(int(level), [])
            if int(n.n_samples) >= int(min_samples)
        ]
        gt_nodes = gt_by.get(int(level), [])
        if req is not None and int(level) in req:
            if not gt_nodes:
                raise AssertionError(
                    f"required_levels includes level={level} but GT hierarchy has no nodes "
                    f"at that level",
                )
            if not pred_nodes:
                raw = pred_by.get(int(level), [])
                counts = [int(n.n_samples) for n in raw]
                raise AssertionError(
                    f"required_levels includes level={level} but no recursion nodes pass "
                    f"min_samples={min_samples} (raw n_samples={counts})",
                )
        if not pred_nodes or not gt_nodes:
            continue

        emp_means: list[np.ndarray] = []
        for n in pred_nodes:
            idx = np.asarray(n.sample_indices, dtype=int)
            emp_means.append(data_arr[idx].mean(axis=0))

        p_means = np.stack(emp_means, axis=0)
        g_means = np.stack([analytic[int(g.cluster_id)][0] for g in gt_nodes], axis=0)
        cost = ((p_means[:, None, :] - g_means[None, :, :]) ** 2).sum(axis=-1)
        row_ind, col_ind = linear_sum_assignment(cost)

        for r, j in zip(row_ind, col_ind):
            node = pred_nodes[int(r)]
            gt = gt_nodes[int(j)]
            idx = np.asarray(node.sample_indices, dtype=int)
            n_s = int(idx.size)
            block = data_arr[idx]

            mu_star, _ = analytic[int(gt.cluster_id)]
            mm_s, covs_s, ww_s = tau_smoothed_gt_mixture_components(
                gt, analytic_by_id=analytic, by_parent=by_parent, tau=float(tau), dim=dim,
            )

            emp_mean = block.mean(axis=0)
            mu_pop, _ = moment_matched_gaussian_from_components(mm_s, covs_s, ww_s)
            stat_m = _hotelling_stat_mixture_mean(
                emp_mean, mu_pop, mm_s, covs_s, ww_s, n_s, inv_reg=float(mean_inv_reg),
            )
            if stat_m > chi2_thr:
                err_l2 = float(np.linalg.norm(emp_mean - mu_star))
                raise AssertionError(
                    f"level={level} region_id={int(node.region_id)} -> GT cluster_id={int(gt.cluster_id)}: "
                    f"Hotelling n(mu_hat-mu)^T Cov(X)^-1 (mu_hat-mu) = {stat_m:.4f} > "
                    f"chi2.ppf(1-{mean_chi2_alpha:.4g}, df={dim}) = {chi2_thr:.4f} "
                    f"(mean L2 vs analytic mu* = {err_l2:.4f}, n={n_s})",
                )

            if n_s >= cov_n_min and n_s >= 2:
                mm_r, covs_r, ww_r = subtree_leaf_raw_mixture_params(
                    gt, by_parent=by_parent, dim=dim,
                )
                _, cov_ref_raw = moment_matched_gaussian_from_components(mm_r, covs_r, ww_r)
                cov_ref_raw = _sym(cov_ref_raw)
                emp_cov = _sample_cov_shrunk(block, min_cov_diag=min_cov_diag)
                err_c = float(np.linalg.norm(_sym(emp_cov) - cov_ref_raw, ord="fro"))
                ref_norm = float(np.linalg.norm(cov_ref_raw, ord="fro"))
                thr_c = float(cov_fro_atol) + float(cov_fro_rtol) * max(ref_norm, 1e-15)
                if err_c > thr_c:
                    raise AssertionError(
                        f"level={level} region_id={int(node.region_id)} -> GT cluster_id={int(gt.cluster_id)}: "
                        f"cov Frobenius ||S-Sigma_ref||_F = {err_c:.4f} > atol + rtol*||Sigma_ref|| = {thr_c:.4f} "
                        f"(n={n_s})",
                    )

            evaluated += 1

    if evaluated == 0:
        raise AssertionError("no (level, Hungarian) pairs passed sample gates")


def leaf_partition_by_region_id(tree: Any) -> list[tuple[int, np.ndarray]]:
    """``(region_id, sample_indices)`` for each leaf, sorted by ``region_id``."""

    leaves = sorted(tree.leaves, key=lambda n: int(n.region_id))
    return [(int(n.region_id), np.asarray(n.sample_indices, dtype=int).copy()) for n in leaves]


def per_sample_leaf_labels(n_samples: int, leaf_partition: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """Integer leaf id ``0 .. L-1`` (sorted by ``region_id``) for each sample."""

    out = np.full(int(n_samples), -1, dtype=int)
    for lid, (_, idx) in enumerate(leaf_partition):
        out[np.asarray(idx, dtype=int)] = lid
    if (out < 0).any():
        raise ValueError("leaf partitions do not cover all samples or overlap")
    return out


def assert_leaf_partition_covers_dataset(tree: Any, n_samples: int) -> None:
    """Every original index appears exactly once across leaves."""

    leaves = tree.leaves
    if not leaves:
        raise AssertionError("expected at least one leaf")
    all_idx = np.sort(np.concatenate([np.asarray(n.sample_indices, dtype=int) for n in leaves]))
    expected = np.arange(int(n_samples), dtype=int)
    if all_idx.shape != expected.shape or not np.array_equal(all_idx, expected):
        raise AssertionError(
            f"leaf sample_indices must partition 0..n-1: got {all_idx.shape[0]} indices, "
            f"unique={len(np.unique(all_idx))}, n={n_samples}",
        )


def assert_terminal_leaf_count_equals_fine_components(tree: Any, *, n_fine: int) -> None:
    """Require one terminal region per fine component (e.g. six leaves for 3×2 hierarchy)."""

    leaves = tree.leaves
    n = len(leaves)
    if n != int(n_fine):
        raise AssertionError(
            f"expected {int(n_fine)} terminal leaves (one per fine GT component), got {n}; "
            f"leaf region_ids={[int(x.region_id) for x in leaves]}",
        )


def assert_fine_ari_at_least(
    leaf_labels: np.ndarray,
    fine_gt: np.ndarray,
    *,
    min_ari: float,
) -> None:
    """Adjusted Rand vs fine labels (permutation-invariant)."""

    ari = float(adjusted_rand_score(fine_gt, leaf_labels))
    if ari < float(min_ari):
        raise AssertionError(
            f"fine-label ARI {ari:.4f} < required {float(min_ari):.4f} "
            "(leaf partition must align with all fine Gaussians)",
        )


def adjusted_rand_vs_coarse_fine(
    leaf_labels: np.ndarray,
    coarse_gt: np.ndarray,
    fine_gt: np.ndarray,
) -> tuple[float, float]:
    """Return ``(ARI_coarse, ARI_fine)`` for the leaf partition vs GT labels."""

    return (
        float(adjusted_rand_score(coarse_gt, leaf_labels)),
        float(adjusted_rand_score(fine_gt, leaf_labels)),
    )


def gaussian_mle_full(
    points: np.ndarray,
    *,
    min_cov_diag: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Unweighted full-covariance Gaussian MLE (sample mean / sample covariance)."""

    x = np.asarray(points, dtype=float)
    if x.ndim != 2:
        raise ValueError("points must be 2D (n, d)")
    n, d = x.shape
    if n == 0:
        raise ValueError("empty point set")
    mean = x.mean(axis=0)
    if n == 1:
        cov = np.eye(d, dtype=float) * float(min_cov_diag)
        return mean, cov
    cov = np.cov(x.T, bias=False)
    cov = np.asarray(cov, dtype=float).reshape(d, d)
    cov = cov + np.eye(d, dtype=float) * float(min_cov_diag)
    return mean, cov


def cluster_nodes_by_level(hierarchy: Sequence[ClusterNode]) -> dict[int, list[ClusterNode]]:
    """``level -> [ClusterNode, ...]`` sorted by ``cluster_id``."""

    by: dict[int, list[ClusterNode]] = defaultdict(list)
    for c in hierarchy:
        by[int(c.level)].append(c)
    return {L: sorted(nodes, key=lambda n: int(n.cluster_id)) for L, nodes in by.items()}


def recursion_nodes_by_level(tree: Any) -> dict[int, list[Any]]:
    """``level -> [RecursionNode, ...]`` sorted by ``region_id``."""

    by: dict[int, list[Any]] = defaultdict(list)
    for n in tree.nodes:
        by[int(n.level)].append(n)
    return {L: sorted(nodes, key=lambda n: int(n.region_id)) for L, nodes in by.items()}


def assert_recursion_gaussian_means_match_gt_hierarchy(
    data: np.ndarray,
    tree: Any,
    hierarchy: Sequence[ClusterNode],
    *,
    min_samples: int = 5,
    max_mean_l2_by_level: dict[int, float] | None = None,
) -> None:
    """Fit a Gaussian MLE on each ``RecursionNode`` region and match GT nodes by level.

    At each hierarchy depth ``L`` present in both the recursion tree and
    ``cluster_hierarchy``, collect nodes with ``n_samples >= min_samples``,
    compute per-region sample means, and solve a linear assignment to GT
    centers at level ``L`` (``scipy.optimize.linear_sum_assignment`` on
    squared Euclidean distance).  Asserts the worst matched mean error is
    below ``max_mean_l2_by_level[L]`` when that level is configured.

    When there are fewer predicted nodes than GT nodes at a level (typical
    coarse-only recovery), each predicted mean is assigned to a distinct GT
    center; only the matched pairs are checked.
    """

    if max_mean_l2_by_level is None:
        max_mean_l2_by_level = {0: 0.35, 1: 0.45, 2: 1.25}

    data_arr = np.asarray(data, dtype=float)
    gt_by = cluster_nodes_by_level(hierarchy)
    pred_by = recursion_nodes_by_level(tree)

    for level, gt_nodes in sorted(gt_by.items()):
        thr = max_mean_l2_by_level.get(level)
        if thr is None:
            continue
        pred_nodes = pred_by.get(level, [])
        pred_means: list[np.ndarray] = []
        for n in pred_nodes:
            if int(n.n_samples) < int(min_samples):
                continue
            idx = np.asarray(n.sample_indices, dtype=int)
            mean, _ = gaussian_mle_full(data_arr[idx])
            pred_means.append(mean)
        if not pred_means:
            continue
        if not gt_nodes:
            continue
        p = np.stack(pred_means, axis=0)
        g = np.stack([np.asarray(c.center, dtype=float) for c in gt_nodes], axis=0)
        cost = ((p[:, None, :] - g[None, :, :]) ** 2).sum(axis=-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        dists = np.sqrt(np.maximum(cost[row_ind, col_ind], 0.0))
        worst = float(np.max(dists)) if dists.size else 0.0
        if worst > float(thr):
            raise AssertionError(
                f"Gaussian mean mismatch at hierarchy level {level}: "
                f"max L2 error after assignment = {worst:.4f} (threshold {thr}); "
                f"per-pair errors = {np.round(dists, 4)}",
            )
