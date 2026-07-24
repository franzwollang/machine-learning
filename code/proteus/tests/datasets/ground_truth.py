"""Ground-truth schemas for Proteus dataset generators.

These dataclasses describe the known structure of a dataset so that test
scenarios can assert specific properties of the learned model against it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def expected_tau_for_arc(
    perimeter: float,
    target_n_nodes: int,
    noise_variance: float = 0.0,
) -> float:
    """Variance cap for a 1D arc-like support.

    The geometric term is the variance of a uniform interval whose length is
    the typical arc segment assigned to one node.
    """

    if perimeter <= 0.0:
        raise ValueError("perimeter must be positive")
    if target_n_nodes <= 0:
        raise ValueError("target_n_nodes must be positive")
    geometric = (float(perimeter) / int(target_n_nodes)) ** 2 / 12.0
    return float(geometric + noise_variance)


def expected_tau_for_surface(
    surface_area: float,
    target_n_nodes: int,
    noise_variance: float = 0.0,
) -> float:
    """Variance cap for a 2D surface-like support.

    The geometric term approximates isotropic local patch variance from the
    typical patch area assigned to one node.
    """

    if surface_area <= 0.0:
        raise ValueError("surface_area must be positive")
    if target_n_nodes <= 0:
        raise ValueError("target_n_nodes must be positive")
    geometric = float(surface_area) / (8.0 * np.pi * int(target_n_nodes))
    return float(geometric + noise_variance)


def ideal_nodes_for_arc(
    perimeter: float,
    tau: float,
    noise_variance: float = 0.0,
    eps: float = 1e-12,
) -> float:
    """Max-ent tiling estimate for a 1D arc-like support."""

    if perimeter <= 0.0:
        raise ValueError("perimeter must be positive")
    tau_geom = max(float(tau) - float(noise_variance), eps)
    return float(perimeter / np.sqrt(12.0 * tau_geom))


def ideal_nodes_for_surface(
    surface_area: float,
    tau: float,
    noise_variance: float = 0.0,
    eps: float = 1e-12,
) -> float:
    """Max-ent tiling estimate for a 2D surface-like support."""

    if surface_area <= 0.0:
        raise ValueError("surface_area must be positive")
    tau_geom = max(float(tau) - float(noise_variance), eps)
    return float(surface_area / (8.0 * np.pi * tau_geom))


@dataclass(frozen=True)
class ClusterNode:
    """A node in a hierarchical cluster tree."""
    cluster_id: int
    level: int
    parent_id: Optional[int]
    weight: float
    center: np.ndarray
    covariance: np.ndarray
    is_leaf: bool
    intrinsic_dim: Optional[int] = None

    @property
    def scale(self) -> float:
        return float(np.sqrt(np.trace(self.covariance)))


@dataclass(frozen=True)
class TopologyExpectation:
    """Expected topological summary for a dataset or component."""
    connected_components: int
    betti_numbers: tuple[int, ...]  # (b0, b1, b2, ...)
    intrinsic_dim: Optional[int] = None


@dataclass(frozen=True)
class DensityProfile:
    """Description of a density regime for a dataset region."""
    region_label: str
    relative_density: float  # 1.0 = nominal
    expected_node_density_ratio: float = 1.0


@dataclass(frozen=True)
class JunctionExpectation:
    """Expected dimensionality junction in a dataset."""
    location_hint: np.ndarray  # approximate spatial location
    dim_low: int
    dim_high: int
    description: str = ""


@dataclass
class GroundTruthManifold:
    """Complete ground-truth specification for a synthetic or real dataset.

    This is the single object that scenario tests consume to formulate
    assertions about scale recovery, clustering, topology, density, and
    junction behavior.
    """
    name: str
    ambient_dim: int
    intrinsic_dim: int
    expected_scale_levels: int = 1
    cluster_hierarchy: list[ClusterNode] = field(default_factory=list)
    topology: Optional[TopologyExpectation] = None
    per_component_topology: list[TopologyExpectation] = field(default_factory=list)
    density_profiles: list[DensityProfile] = field(default_factory=list)
    junctions: list[JunctionExpectation] = field(default_factory=list)
    expected_tau: Optional[float] = None
    expected_node_count: Optional[int] = None
    node_count_upper_bound: Optional[int] = None
    noise_variance: float = 0.0
    tau_grid_hint: Optional[tuple[float, float]] = None

    def max_ent_node_lower(self, multiplier: float = 0.5) -> int:
        """Return an explicit diagnostic node-count lower bound.

        Raises if the dataset does not support an ideal tiling estimate.
        """

        if self.expected_node_count is None:
            raise ValueError(f"{self.name} has no expected_node_count")
        return max(1, int(np.floor(multiplier * self.expected_node_count)))

    def max_ent_node_upper(self, multiplier: float = 2.0) -> int:
        """Return an explicit diagnostic node-count upper bound.

        Raises if the dataset does not support an ideal tiling estimate.
        """

        if self.expected_node_count is None:
            raise ValueError(f"{self.name} has no expected_node_count")
        return int(np.ceil(multiplier * self.expected_node_count))

    @property
    def leaf_clusters(self) -> list[ClusterNode]:
        return [c for c in self.cluster_hierarchy if c.is_leaf]

    @property
    def n_leaf_clusters(self) -> int:
        return len(self.leaf_clusters)

    @property
    def hierarchy_depth(self) -> int:
        if not self.cluster_hierarchy:
            return 0
        return max(c.level for c in self.cluster_hierarchy) + 1


@dataclass
class SyntheticDataset:
    """A generated synthetic dataset paired with its ground truth."""
    points: np.ndarray                        # (N, D)
    labels: np.ndarray                        # (N,) integer cluster labels
    ground_truth: GroundTruthManifold
    metadata: dict = field(default_factory=dict)
