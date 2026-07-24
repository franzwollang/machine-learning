"""CMU Motion Capture dataset loader (curated activity subset).

Source: CMU Graphics Lab Motion Capture Database
License: research use (free for academic purposes)
Preprocessing: parse .amc/.asf -> joint angles, standardize per joint.
"""

from __future__ import annotations

from .ground_truth import (
    ContinuousAxis,
    HierarchicalLabels,
    RealDataset,
    TopologySummary,
)


def load_cmu_mocap(cache_dir: str | None = None) -> RealDataset:
    """Load a curated CMU MoCap activity subset for Proteus evaluation.

    Requires the dataset to be available in ``cache_dir`` (downloaded
    separately via the data-cache harness).
    """
    raise NotImplementedError(
        "CMU MoCap loader requires the data-cache harness to download the "
        "dataset. Use tests.harness.data_cache.cached_dataset('cmu_mocap')."
    )
