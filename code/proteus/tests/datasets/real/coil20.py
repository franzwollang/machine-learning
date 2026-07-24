"""COIL-20 dataset loader.

Source: Columbia University (S. Nene, S. Nayar, H. Murase)
License: research use
Preprocessing: resize to 32x32, flatten, optionally PCA-reduce.
"""

from __future__ import annotations

import numpy as np

from .ground_truth import (
    ContinuousAxis,
    HierarchicalLabels,
    RealDataset,
    TopologySummary,
)


def load_coil20(pca_dim: int = 50, cache_dir: str | None = None) -> RealDataset:
    """Load COIL-20 images and preprocess for Proteus evaluation.

    Requires the dataset to be available in ``cache_dir`` (downloaded
    separately via the data-cache harness).
    """
    raise NotImplementedError(
        "COIL-20 loader requires the data-cache harness to download the "
        "dataset. Use tests.harness.data_cache.cached_dataset('coil20')."
    )
