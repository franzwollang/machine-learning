"""T2 parent-to-child recursion transfer (SI S1.4, S4.4 T2)."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np


@dataclass
class T2Result:
    """Output of a T2 PCA transfer for one child cluster."""

    child_data: np.ndarray
    child_dim: int
    pca_components: np.ndarray
    pca_mean: np.ndarray
    sample_indices: np.ndarray


def apply_t2_transfer(
    parent_data: np.ndarray,
    sample_indices: np.ndarray,
    parent_dim: int,
    d_hat_cluster: int,
    *,
    r_min: int = 3,
    explained_energy: float = 0.999,
) -> T2Result:
    """Apply S1.4 PCA transfer to a child cluster's samples.

    1. Extract and center child samples.
    2. PCA via SVD.
    3. Retain ``r_child = min(D_parent, max(r_ID, r_999))`` components.
    4. Project into the child PCA frame.
    """

    idx = np.asarray(sample_indices, dtype=int)
    child_raw = parent_data[idx]
    mean = child_raw.mean(axis=0)
    centered = child_raw - mean

    n, d = centered.shape
    if n < 2 or d < 1:
        return T2Result(
            child_data=centered,
            child_dim=d,
            pca_components=np.eye(d),
            pca_mean=mean,
            sample_indices=idx,
        )

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    var_explained = S ** 2
    cumulative = np.cumsum(var_explained)
    total = cumulative[-1] if cumulative.size else 1.0
    if total <= 0.0:
        total = 1.0

    r_999 = int(np.searchsorted(cumulative / total, explained_energy) + 1)
    r_999 = min(r_999, len(S))

    r_ID = max(int(r_min), 2 * int(ceil(float(d_hat_cluster))))

    r_child = min(int(parent_dim), max(r_ID, r_999))
    r_child = max(1, min(r_child, len(S)))

    components = Vt[:r_child]
    projected = centered @ components.T

    return T2Result(
        child_data=projected,
        child_dim=r_child,
        pca_components=components,
        pca_mean=mean,
        sample_indices=idx,
    )
