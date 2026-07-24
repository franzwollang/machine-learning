"""Shared uniform-tissue helpers for synthetic datasets."""
from __future__ import annotations

import numpy as np


def expected_tau_for_uniform_tissue_box(
    bounds: tuple[np.ndarray, np.ndarray],
    target_n_nodes: int,
    noise_variance: float = 0.0,
    eps: float = 1e-12,
) -> float:
    """Heuristic variance cap for a uniform ambient tissue box."""
    if target_n_nodes <= 0:
        raise ValueError("target_n_nodes must be positive")
    lo = np.asarray(bounds[0], dtype=float)
    hi = np.asarray(bounds[1], dtype=float)
    dim = int(lo.shape[0])
    span = np.maximum(hi - lo, eps)
    volume = float(np.prod(span))
    side = (volume / int(target_n_nodes)) ** (1.0 / dim)
    geometric = dim * side * side / 12.0
    return float(geometric + noise_variance)


def ideal_nodes_for_uniform_tissue_box(
    bounds: tuple[np.ndarray, np.ndarray],
    tau: float,
    noise_variance: float = 0.0,
    eps: float = 1e-12,
) -> float:
    """Invert the uniform tissue-box tau heuristic."""
    lo = np.asarray(bounds[0], dtype=float)
    hi = np.asarray(bounds[1], dtype=float)
    dim = int(lo.shape[0])
    span = np.maximum(hi - lo, eps)
    volume = float(np.prod(span))
    tau_geom = max(float(tau) - float(noise_variance), eps)
    side = np.sqrt(12.0 * tau_geom / dim)
    cell_volume = max(side**dim, eps)
    return float(volume / cell_volume)


def append_uniform_tissue(
    signal_points: np.ndarray,
    signal_labels: np.ndarray,
    *,
    rng: np.random.Generator,
    tissue_fraction: float = 0.03,
    padding_fraction: float = 0.05,
    min_padding: float = 0.05,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Append low-density uniform tissue and shuffle the combined sample.

    ``tissue_fraction`` is the desired fraction of tissue points in the final
    returned dataset. Tissue samples are labeled ``-1``.
    """
    if not 0.0 <= tissue_fraction < 1.0:
        raise ValueError("tissue_fraction must lie in [0, 1)")

    points = np.asarray(signal_points, dtype=float)
    labels = np.asarray(signal_labels, dtype=int)
    if points.ndim != 2:
        raise ValueError("signal_points must have shape (n, d)")
    if labels.shape != (points.shape[0],):
        raise ValueError("signal_labels must have shape (n,)")

    n_signal, dim = points.shape
    if tissue_fraction <= 0.0 or n_signal == 0:
        return points, labels, {
            "signal_count": int(n_signal),
            "tissue_count": 0,
            "tissue_fraction_actual": 0.0,
            "tissue_bounds_lo": points.min(axis=0).tolist() if n_signal else [0.0] * dim,
            "tissue_bounds_hi": points.max(axis=0).tolist() if n_signal else [0.0] * dim,
        }

    n_tissue = int(np.round(tissue_fraction * n_signal / max(1.0 - tissue_fraction, 1e-12)))
    n_tissue = max(n_tissue, 1)

    if bounds is None:
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        span = hi - lo
        padding = np.maximum(padding_fraction * span, min_padding)
        lo = lo - padding
        hi = hi + padding
    else:
        lo = np.asarray(bounds[0], dtype=float)
        hi = np.asarray(bounds[1], dtype=float)
        if lo.shape != (dim,) or hi.shape != (dim,):
            raise ValueError("bounds must match point ambient dimension")

    tissue = rng.uniform(lo, hi, size=(n_tissue, dim))
    tissue_labels = np.full(n_tissue, -1, dtype=int)

    all_points = np.vstack([points, tissue])
    all_labels = np.concatenate([labels, tissue_labels])
    perm = rng.permutation(all_points.shape[0])

    return all_points[perm], all_labels[perm], {
        "signal_count": int(n_signal),
        "tissue_count": int(n_tissue),
        "tissue_fraction_actual": float(n_tissue / max(all_points.shape[0], 1)),
        "tissue_bounds_lo": lo.tolist(),
        "tissue_bounds_hi": hi.tolist(),
    }
