"""Unit tests for T2 parent-to-child PCA transfer (SI S1.4)."""

from __future__ import annotations

import numpy as np

from proteus.stage1.transfer import apply_t2_transfer


def _make_4d_data_with_2d_intrinsic(n: int = 500, seed: int = 0) -> np.ndarray:
    """4D data that lives on a 2D plane plus small noise in the other 2 dims."""

    rng = np.random.default_rng(seed)
    intrinsic = rng.standard_normal((n, 2)) * 3.0
    noise = rng.standard_normal((n, 2)) * 0.01
    return np.column_stack([intrinsic, noise])


def test_t2_retains_correct_rank_for_2d_in_4d() -> None:
    data = _make_4d_data_with_2d_intrinsic()
    indices = np.arange(data.shape[0])

    result = apply_t2_transfer(data, indices, parent_dim=4, d_hat_cluster=2)

    assert result.child_dim >= 3
    assert result.child_dim <= 4
    assert result.child_data.shape == (data.shape[0], result.child_dim)


def test_t2_output_shape_matches_child_dim() -> None:
    rng = np.random.default_rng(1)
    data = rng.standard_normal((200, 6))
    indices = np.arange(100)

    result = apply_t2_transfer(data, indices, parent_dim=6, d_hat_cluster=3)

    assert result.child_data.shape[0] == 100
    assert result.child_data.shape[1] == result.child_dim


def test_t2_explained_variance_above_threshold() -> None:
    data = _make_4d_data_with_2d_intrinsic()
    indices = np.arange(data.shape[0])

    result = apply_t2_transfer(
        data, indices, parent_dim=4, d_hat_cluster=2, explained_energy=0.999,
    )

    centered = data - data.mean(axis=0)
    total_var = np.sum(np.var(centered, axis=0))
    projected_var = np.sum(np.var(result.child_data, axis=0))
    ratio = projected_var / total_var

    assert ratio >= 0.999, f"explained ratio {ratio:.4f} < 0.999"


def test_t2_pca_components_shape() -> None:
    data = _make_4d_data_with_2d_intrinsic()
    indices = np.arange(data.shape[0])

    result = apply_t2_transfer(data, indices, parent_dim=4, d_hat_cluster=2)

    assert result.pca_components.shape == (result.child_dim, 4)
    assert result.pca_mean.shape == (4,)
