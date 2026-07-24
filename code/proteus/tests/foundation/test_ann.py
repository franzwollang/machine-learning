"""Unit tests for ANN backends."""

from __future__ import annotations

import numpy as np
import pytest

from proteus.ann import HNSWBackend, NaiveBackend, make_ann


def test_naive_backend_returns_exact_neighbors() -> None:
    points = np.array([[0.0], [1.0], [2.0], [3.0]])
    index = NaiveBackend(dim=1)
    index.build_from(points)

    ids, distances = index.query_knn(np.array([1.2]), k=2)

    np.testing.assert_array_equal(ids, np.array([1, 2]))
    np.testing.assert_allclose(distances, np.array([0.2, 0.8]))


def test_naive_backend_update_and_remove() -> None:
    index = NaiveBackend(dim=2)
    first = index.add(np.array([0.0, 0.0]))
    second = index.add(np.array([10.0, 0.0]))
    index.update(second, np.array([1.0, 0.0]))
    index.remove(first)

    ids, distances = index.query_knn(np.array([0.0, 0.0]), k=2)

    np.testing.assert_array_equal(ids, np.array([second]))
    np.testing.assert_allclose(distances, np.array([1.0]))


def test_make_ann_auto_uses_naive_for_small_expected_size() -> None:
    index = make_ann(dim=3, backend="auto", expected_size=100)

    assert isinstance(index, NaiveBackend)


def test_hnsw_backend_matches_naive_on_small_sample() -> None:
    pytest.importorskip("hnswlib")

    rng = np.random.default_rng(123)
    points = rng.normal(size=(250, 4))
    queries = rng.normal(size=(50, 4))

    naive = NaiveBackend(dim=4)
    hnsw = HNSWBackend(dim=4, ef_search=200)
    naive.build_from(points)
    hnsw.build_from(points)

    top1_matches = 0
    topk_overlap = 0.0
    k = 8
    for query in queries:
        naive_ids, _ = naive.query_knn(query, k=k)
        hnsw_ids, _ = hnsw.query_knn(query, k=k)
        top1_matches += int(naive_ids[0] == hnsw_ids[0])
        topk_overlap += len(set(naive_ids.tolist()) & set(hnsw_ids.tolist())) / k

    assert top1_matches == len(queries)
    assert topk_overlap / len(queries) >= 0.90
