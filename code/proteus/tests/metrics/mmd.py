"""Maximum Mean Discrepancy metric for Proteus evaluation."""
from __future__ import annotations

import numpy as np


def rbf_kernel(
    X: np.ndarray, Y: np.ndarray, bandwidth: float,
) -> np.ndarray:
    """RBF kernel matrix between rows of X and rows of Y."""
    sq = (
        np.sum(X ** 2, axis=1, keepdims=True)
        + np.sum(Y ** 2, axis=1, keepdims=True).T
        - 2.0 * X @ Y.T
    )
    return np.exp(-sq / (2.0 * bandwidth ** 2))


def median_heuristic(X: np.ndarray, Y: np.ndarray) -> float:
    """Median-distance bandwidth heuristic on the combined sample."""
    combined = np.vstack([X, Y])
    n = min(combined.shape[0], 2000)
    rng = np.random.default_rng(0)
    idx = rng.choice(combined.shape[0], size=n, replace=False)
    sub = combined[idx]
    dists = np.sqrt(np.maximum(
        np.sum(sub ** 2, axis=1, keepdims=True)
        + np.sum(sub ** 2, axis=1, keepdims=True).T
        - 2.0 * sub @ sub.T,
        0.0,
    ))
    med = float(np.median(dists[np.triu_indices(n, k=1)]))
    return max(med, 1e-8)


def mmd_squared(
    X: np.ndarray,
    Y: np.ndarray,
    bandwidth: float | None = None,
) -> float:
    """Biased estimate of MMD^2 with RBF kernel."""
    if bandwidth is None:
        bandwidth = median_heuristic(X, Y)
    Kxx = rbf_kernel(X, X, bandwidth)
    Kyy = rbf_kernel(Y, Y, bandwidth)
    Kxy = rbf_kernel(X, Y, bandwidth)
    m, n = X.shape[0], Y.shape[0]
    return (
        float(Kxx.sum()) / (m * m)
        - 2.0 * float(Kxy.sum()) / (m * n)
        + float(Kyy.sum()) / (n * n)
    )


def mmd_permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    bandwidth: float | None = None,
    n_permutations: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """Return (MMD^2, p-value) from a permutation test."""
    if bandwidth is None:
        bandwidth = median_heuristic(X, Y)
    observed = mmd_squared(X, Y, bandwidth)
    combined = np.vstack([X, Y])
    m = X.shape[0]
    rng = np.random.default_rng(seed)
    count_ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(combined.shape[0])
        Xp = combined[perm[:m]]
        Yp = combined[perm[m:]]
        if mmd_squared(Xp, Yp, bandwidth) >= observed:
            count_ge += 1
    p_value = (count_ge + 1) / (n_permutations + 1)
    return observed, p_value
