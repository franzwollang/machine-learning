"""Nearest-neighbor backends for Proteus.

The naive backend is exact and serves as the oracle in tests.  The HNSW
backend wraps ``hnswlib`` but keeps a local point cache so the public API
can support updates/removals consistently during early implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


class ANNIndex(ABC):
    """Common interface for k-nearest-neighbor indices."""

    dim: int

    @abstractmethod
    def add(self, point: np.ndarray) -> int:
        """Add a point and return its integer index."""

    @abstractmethod
    def update(self, idx: int, point: np.ndarray) -> None:
        """Replace the point stored at ``idx``."""

    @abstractmethod
    def remove(self, idx: int) -> None:
        """Remove the point at ``idx`` from future queries."""

    @abstractmethod
    def build_from(self, points: np.ndarray) -> None:
        """Replace index contents with ``points``."""

    @abstractmethod
    def query_knn(self, point: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(indices, distances)`` for the k nearest neighbors."""


class NaiveBackend(ANNIndex):
    """Exact NumPy nearest-neighbor backend."""

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self._points: dict[int, np.ndarray] = {}
        self._next_id = 0

    def add(self, point: np.ndarray) -> int:
        point_arr = _validate_point(point, self.dim)
        idx = self._next_id
        self._next_id += 1
        self._points[idx] = point_arr.copy()
        return idx

    def update(self, idx: int, point: np.ndarray) -> None:
        if idx not in self._points:
            raise KeyError(f"Unknown ANN index {idx}")
        self._points[idx] = _validate_point(point, self.dim).copy()

    def remove(self, idx: int) -> None:
        self._points.pop(idx, None)

    def build_from(self, points: np.ndarray) -> None:
        points_arr = _validate_points(points, self.dim)
        self._points.clear()
        self._next_id = 0
        for point in points_arr:
            self.add(point)

    def query_knn(self, point: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k <= 0:
            raise ValueError("k must be positive")
        if not self._points:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        point_arr = _validate_point(point, self.dim)
        ids = np.array(sorted(self._points.keys()), dtype=int)
        data = np.vstack([self._points[int(idx)] for idx in ids])
        dists = np.linalg.norm(data - point_arr[None, :], axis=1)
        order = np.argsort(dists, kind="mergesort")[: min(k, ids.size)]
        return ids[order], dists[order]


class HNSWBackend(ANNIndex):
    """HNSW nearest-neighbor backend with a simple rebuild-on-change cache."""

    def __init__(
        self,
        dim: int,
        *,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
    ) -> None:
        self.dim = int(dim)
        self.M = int(M)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self._points: dict[int, np.ndarray] = {}
        self._next_id = 0
        self._index = None
        self._dirty = True

    def add(self, point: np.ndarray) -> int:
        point_arr = _validate_point(point, self.dim)
        idx = self._next_id
        self._next_id += 1
        self._points[idx] = point_arr.copy()
        self._dirty = True
        return idx

    def update(self, idx: int, point: np.ndarray) -> None:
        if idx not in self._points:
            raise KeyError(f"Unknown ANN index {idx}")
        self._points[idx] = _validate_point(point, self.dim).copy()
        self._dirty = True

    def remove(self, idx: int) -> None:
        if idx in self._points:
            del self._points[idx]
            self._dirty = True

    def build_from(self, points: np.ndarray) -> None:
        points_arr = _validate_points(points, self.dim)
        self._points.clear()
        self._next_id = 0
        for point in points_arr:
            idx = self._next_id
            self._next_id += 1
            self._points[idx] = point.copy()
        self._dirty = True
        self._rebuild()

    def query_knn(self, point: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k <= 0:
            raise ValueError("k must be positive")
        if not self._points:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        if self._dirty:
            self._rebuild()
        assert self._index is not None
        point_arr = _validate_point(point, self.dim)
        k_eff = min(k, len(self._points))
        labels, distances_sq = self._index.knn_query(point_arr[None, :], k=k_eff)
        return labels[0].astype(int), np.sqrt(np.maximum(distances_sq[0], 0.0))

    def _rebuild(self) -> None:
        try:
            import hnswlib
        except ImportError as e:
            raise ImportError(
                "hnswlib is required for HNSWBackend. Install via pipenv in "
                "code/proteus or use NaiveBackend."
            ) from e

        ids = np.array(sorted(self._points.keys()), dtype=int)
        data = np.vstack([self._points[int(idx)] for idx in ids]).astype(np.float32)
        index = hnswlib.Index(space="l2", dim=self.dim)
        index.init_index(
            max_elements=max(1, ids.size),
            ef_construction=self.ef_construction,
            M=self.M,
        )
        index.add_items(data, ids)
        index.set_ef(self.ef_search)
        self._index = index
        self._dirty = False


def make_ann(
    dim: int,
    backend: str = "auto",
    *,
    expected_size: Optional[int] = None,
) -> ANNIndex:
    """Create an ANN backend.

    ``backend='auto'`` uses HNSW only when ``expected_size`` is known and
    larger than 5000; otherwise it returns the exact naive backend.
    """

    if backend == "naive":
        return NaiveBackend(dim)
    if backend == "hnsw":
        return HNSWBackend(dim)
    if backend == "auto":
        if expected_size is not None and expected_size > 5000:
            return HNSWBackend(dim)
        return NaiveBackend(dim)
    raise ValueError(f"Unknown ANN backend {backend!r}")


def _validate_point(point: np.ndarray, dim: int) -> np.ndarray:
    point_arr = np.asarray(point, dtype=float)
    if point_arr.shape != (dim,):
        raise ValueError(f"point must have shape ({dim},), got {point_arr.shape}")
    return point_arr


def _validate_points(points: np.ndarray, dim: int) -> np.ndarray:
    points_arr = np.asarray(points, dtype=float)
    if points_arr.ndim != 2 or points_arr.shape[1] != dim:
        raise ValueError(f"points must have shape (N, {dim}), got {points_arr.shape}")
    return points_arr
