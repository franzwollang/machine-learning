"""AP cluster assignment and summary contracts (SI S2.6)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class APClusterAssignment:
    """Result of Affinity Propagation on the Hebbian graph (SI S2.6)."""
    node_labels: np.ndarray          # (N_nodes,) cluster label per node
    exemplar_indices: np.ndarray     # (K,) exemplar node index per cluster
    similarities: np.ndarray         # (N_edges,) smoothed-PMI similarities used


@dataclass
class ClusterSummary:
    """Per-cluster summary after AP and data routing (SI S2.6)."""
    cluster_id: int
    node_indices: np.ndarray
    sample_count: int
    centroid: np.ndarray
    scale_at_selection: float
