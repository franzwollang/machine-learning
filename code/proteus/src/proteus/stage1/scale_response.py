"""Lindeberg-normalized scale response for Stage 1 scaffolds (SI S2.5)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaln

from proteus.stage1.calibration import c_dk


def _unit_ball_volume(d: int) -> float:
    """Volume of the unit ball in R^d."""

    return float(np.pi ** (d / 2.0) / np.exp(gammaln(d / 2.0 + 1.0)))


def node_response(scaffold: Any, tau: float, d_working: int) -> np.ndarray:
    """Per-node Lindeberg-normalized response R_i(tau) (SI S2.5).

    Uses the scaffold's ANN index to find the k-th nearest scaffold
    neighbor distance for each node.
    """

    n = len(scaffold.nodes)
    if n < 2:
        return np.zeros(n, dtype=float)

    tau_f = float(tau)
    d = int(d_working)
    V_d = _unit_ball_volume(d)
    hits = np.array([node.hit_count for node in scaffold.nodes], dtype=float)
    positions = scaffold.node_positions()
    k = min(scaffold.k, n - 1)
    if k < 1:
        return np.zeros(n, dtype=float)

    N_C = float(hits.sum())
    if N_C <= 0.0:
        return np.zeros(n, dtype=float)

    r_k = np.empty(n, dtype=float)
    for i in range(n):
        _, dists = scaffold.ann.query_knn(positions[i], k=k + 1)
        if len(dists) > k:
            r_k[i] = float(dists[k])
        else:
            r_k[i] = float(dists[-1])
    r_k = np.maximum(r_k, 1e-12)

    rho_hat = hits * k * N_C / (V_d * r_k ** d)
    # Lindeberg calibration sigma = sqrt(tau) / c_{d,k} (SI S2.5, S2.5.5): the
    # calibrated constant converts the variance cap into the effective k-NN
    # bandwidth. Use the same effective k as r_k (clamped to n-1 in sparse
    # scaffolds) so the R_i = (sqrt(tau)/c_{d,k})^d * rho_hat identity is exact.
    c = c_dk(d, k)
    scale_factor = (np.sqrt(tau_f) / c) ** d
    return scale_factor * rho_hat


def cluster_response(scaffold: Any, tau: float, d_working: int) -> float:
    """Cluster-level scale response Phi_C(tau) = sum(R_i) (SI S2.5)."""

    return float(np.sum(node_response(scaffold, tau, d_working)))


def variance_load(scaffold: Any, tau: float) -> float:
    """Mean variance-to-cap ratio across mature nodes.

    This is a robust proxy for the scale response: at the characteristic
    scale the scaffold is "loaded" (mean sigma^2/tau near some target),
    while at over-fine scales the load is low and at over-coarse scales
    the scaffold cannot converge within the cap.
    """

    prune_after = getattr(scaffold, "prune_after", 0)
    mature = [n for n in scaffold.nodes if n.update_count >= prune_after]
    if not mature:
        return 0.0
    variances = np.array([n.variance for n in mature], dtype=float)
    tau_f = float(tau)
    if tau_f <= 0.0:
        return 0.0
    return float(np.mean(variances / tau_f))


def support_trace(scaffold: Any, tau: float, d_working: int) -> float:
    """Support volume trace V_C(tau) = sum(V_hat_i) (SI S2.5)."""

    n = len(scaffold.nodes)
    if n < 2:
        return 0.0

    d = int(d_working)
    V_d = _unit_ball_volume(d)
    hits = np.array([node.hit_count for node in scaffold.nodes], dtype=float)
    positions = scaffold.node_positions()
    k = min(scaffold.k, n - 1)
    if k < 1:
        return 0.0

    N_C = float(hits.sum())
    if N_C <= 0.0:
        return 0.0

    total = 0.0
    for i in range(n):
        _, dists = scaffold.ann.query_knn(positions[i], k=k + 1)
        r_ki = float(dists[k]) if len(dists) > k else float(dists[-1])
        r_ki = max(r_ki, 1e-12)
        V_hat_i = V_d * r_ki ** d / (k / N_C)
        total += V_hat_i
    return total
