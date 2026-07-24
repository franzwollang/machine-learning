"""UMAP + KDE baseline for density and embedding comparison."""

from __future__ import annotations

from typing import Callable

import numpy as np


def fit_umap_kde(
    train: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    seed: int = 0,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Fit UMAP embedding and a KDE on the embedding space.

    Returns (embedding, log_density_fn) where log_density_fn takes
    (N, n_components) and returns (N,) log-densities.
    """
    try:
        import umap
        from scipy.stats import gaussian_kde
    except ImportError as e:
        raise ImportError(
            "umap-learn and scipy are required for the UMAP+KDE baseline."
        ) from e

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        random_state=seed,
    )
    embedding = reducer.fit_transform(train)
    kde = gaussian_kde(embedding.T)

    def log_density_fn(points: np.ndarray) -> np.ndarray:
        transformed = reducer.transform(points)
        return np.log(np.maximum(kde(transformed.T), 1e-300))

    return embedding, log_density_fn
