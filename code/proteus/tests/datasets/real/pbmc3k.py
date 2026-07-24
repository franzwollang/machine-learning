"""PBMC 3k single-cell RNA-seq dataset loader.

Source: 10x Genomics / scanpy.datasets.pbmc3k_processed()
License: CC0 (10x public dataset)
Preprocessing: filter, normalize, log-transform, scale, PCA to 50 dims.
"""
from __future__ import annotations

from .ground_truth import ContinuousAxis, HierarchicalLabels, RealDataset, TopologySummary


def load_pbmc3k() -> RealDataset:
    """Load and preprocess PBMC 3k for Proteus evaluation."""
    try:
        import scanpy as sc
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "scanpy is required for the PBMC 3k dataset. "
            "Install with: pip install 'proteus[real_data]'"
        ) from e

    adata = sc.datasets.pbmc3k_processed()
    features = np.asarray(adata.obsm["X_pca"][:, :50], dtype=float)

    cell_types = adata.obs["louvain"].astype(str).values
    unique_types = sorted(set(cell_types))
    type_to_int = {t: i for i, t in enumerate(unique_types)}
    fine_labels = np.array([type_to_int[t] for t in cell_types])

    coarse_map = {}
    for t in unique_types:
        if any(k in t.lower() for k in ["cd4", "cd8", "nk", "b ", "dendritic"]):
            coarse_map[t] = "lymphoid"
        else:
            coarse_map[t] = "myeloid"
    coarse_names = sorted(set(coarse_map.values()))
    coarse_to_int = {c: i for i, c in enumerate(coarse_names)}
    coarse_labels = np.array([coarse_to_int[coarse_map[t]] for t in cell_types])

    return RealDataset(
        features=features,
        hierarchical_labels=HierarchicalLabels(
            labels={"coarse": coarse_labels, "fine": fine_labels},
            hierarchy={"fine": "coarse"},
            label_names={
                "coarse": {v: k for k, v in coarse_to_int.items()},
                "fine": {v: k for k, v in type_to_int.items()},
            },
        ),
        topology=TopologySummary(
            connected_components=len(unique_types),
            intrinsic_dim=None,
        ),
        metadata={
            "source": "scanpy.datasets.pbmc3k_processed()",
            "license": "CC0",
            "n_pcs": 50,
        },
    )
