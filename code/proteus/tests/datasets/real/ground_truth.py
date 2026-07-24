"""Ground-truth schemas for real datasets (extends synthetic ground_truth)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class HierarchicalLabels:
    """Per-sample multi-level category labels with parent/child relations."""
    labels: dict[str, np.ndarray]    # level_name -> (N,) integer labels
    hierarchy: dict[str, str]        # child_level -> parent_level
    label_names: dict[str, dict[int, str]] = field(default_factory=dict)


@dataclass
class TopologySummary:
    """Expected topological summary for a real dataset or component."""
    connected_components: int
    expected_betti: Optional[tuple[int, ...]] = None
    intrinsic_dim: Optional[int] = None


@dataclass
class ContinuousAxis:
    """Per-sample continuous covariate (timestamp, angle, intensity)."""
    name: str
    values: np.ndarray               # (N,)
    is_cyclic: bool = False
    expected_smoothness: Optional[float] = None


@dataclass
class RealDataset:
    """A loaded real dataset with preprocessing applied."""
    features: np.ndarray             # (N, D)
    hierarchical_labels: Optional[HierarchicalLabels] = None
    topology: Optional[TopologySummary] = None
    continuous_axes: list[ContinuousAxis] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
