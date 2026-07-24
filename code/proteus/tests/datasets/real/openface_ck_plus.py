"""OpenFace AU features on CK+ expression dataset.

Source: OpenFace 2.0 Action Unit extraction on Extended Cohn-Kanade (CK+)
License: CK+ requires a signed license from Pitt/CMU
Preprocessing: extract 17 AU intensities + 6 pose/gaze per frame, standardize.
"""
from __future__ import annotations

from .ground_truth import ContinuousAxis, HierarchicalLabels, RealDataset, TopologySummary


def load_openface_ck_plus(cache_dir: str | None = None) -> RealDataset:
    """Load OpenFace AU features on CK+ for Proteus evaluation.

    Requires pre-extracted AU features in ``cache_dir`` (extracted
    separately via OpenFace 2.0 and the data-cache harness).
    """
    raise NotImplementedError(
        "OpenFace CK+ loader requires pre-extracted AU features. "
        "Use tests.harness.data_cache.cached_dataset('openface_ck_plus')."
    )
