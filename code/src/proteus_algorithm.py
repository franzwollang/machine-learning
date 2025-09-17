"""High-level orchestration of the Proteus Stage 1 scaffold builder.

This module wraps the low-level ``ProteusStage1`` implementation to
produce a simple, recursive hierarchy of mesh nodes.  It provides a
small amount of structure discovery on top of Stage 1 by repeatedly
running the scaffold builder on subsets of the data and checking for
statistically meaningful splits.  The resulting tree is intentionally
minimal: it is only aimed at supporting the unit tests and exploratory
visualisations that compare the learned hierarchy against synthetic
ground truth mixtures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .proteus.stage1 import ProteusStage1


def _mean_min_distance(points: np.ndarray, representatives: np.ndarray) -> float:
    """Return the mean distance from each point to its closest node."""
    if representatives.shape[0] == 0:
        return float("inf")
    deltas = points[:, None, :] - representatives[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    return float(distances.min(axis=1).mean())


@dataclass
class AlgorithmParameters:
    """Hyperparameters controlling the hierarchy search."""

    stage1_k_neighbors: int = 8
    stage1_alpha: Optional[float] = None
    stage1_eta_cent: float = 0.1
    stage1_eta_move: float = 0.08
    stage1_eta_oja: float = 0.05
    stage1_grid_ratio: float = 1.0 / np.sqrt(2.0)
    stage1_kappa: float = 0.5
    stage1_max_epochs: int = 45
    stage1_max_nodes: int = 128
    stage1_min_nodes: int = 10
    stage1_prune_after: int = 25
    split_improvement_threshold: float = 0.32
    split_balance_fraction: float = 0.22
    min_split_nodes: int = 10
    random_seed: int = 0


@dataclass
class HierarchyNode:
    """Node in the discovered hierarchy."""

    node_id: int
    level: int
    indices: np.ndarray
    centroid: np.ndarray
    covariance: np.ndarray
    stage1_nodes: np.ndarray
    neighbour_graph: Dict[int, Sequence[int]]
    stage1_cv: float
    quantisation_error: float
    data: np.ndarray
    parent: Optional["HierarchyNode"] = None
    children: List["HierarchyNode"] = field(default_factory=list)
    cv_history: Optional[List[float]] = None

    def add_child(self, child: "HierarchyNode") -> None:
        self.children.append(child)
        child.parent = self


class ProteusAlgorithm:
    """Recursive hierarchy builder that wraps ``ProteusStage1``."""

    def __init__(
        self,
        *,
        params: AlgorithmParameters,
        max_depth: int = 4,
        min_cluster_size: int = 200,
        run_stage2: bool = False,
    ) -> None:
        self.params = params
        self.max_depth = int(max_depth)
        self.min_cluster_size = int(min_cluster_size)
        self.run_stage2 = bool(run_stage2)

        self.all_nodes: List[HierarchyNode] = []
        self.leaf_nodes: List[HierarchyNode] = []
        self.hierarchy_root: Optional[HierarchyNode] = None
        self._node_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_complete_algorithm(self, data: np.ndarray) -> HierarchyNode:
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError("data must be a 2D array")
        if data.shape[0] < 2:
            raise ValueError("Proteus requires at least two samples")

        self.all_nodes.clear()
        self.leaf_nodes.clear()
        self._node_counter = 0

        indices = np.arange(data.shape[0])
        root = self._build_node(data, indices, level=0)
        self.hierarchy_root = root
        return root

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_node(
        self,
        data: np.ndarray,
        indices: np.ndarray,
        *,
        level: int,
    ) -> HierarchyNode:
        subset = data[indices]
        tau = float(np.var(subset, axis=0).mean())
        if tau <= 0.0:
            tau = 1e-6
        stage1 = ProteusStage1(
            k_neighbors=self.params.stage1_k_neighbors,
            alpha=self.params.stage1_alpha,
            eta_cent=self.params.stage1_eta_cent,
            eta_move=self.params.stage1_eta_move,
            eta_oja=self.params.stage1_eta_oja,
            grid_ratio=self.params.stage1_grid_ratio,
            kappa=self.params.stage1_kappa,
            max_epochs=self.params.stage1_max_epochs,
            max_nodes=self.params.stage1_max_nodes,
            min_nodes=self.params.stage1_min_nodes,
            prune_after=self.params.stage1_prune_after,
            rng=np.random.default_rng(self.params.random_seed + self._node_counter),
        )
        self._node_counter += 1
        stage1.fit(subset, tau=tau)

        positions = stage1.get_node_positions()
        quant_error = _mean_min_distance(subset, positions)
        covariance = np.cov(subset, rowvar=False)
        if covariance.ndim == 0:
            covariance = np.array([[float(covariance)]])

        node = HierarchyNode(
            node_id=len(self.all_nodes),
            level=level,
            indices=indices.copy(),
            centroid=subset.mean(axis=0),
            covariance=covariance,
            stage1_nodes=positions,
            neighbour_graph=stage1.neighbour_graph(),
            stage1_cv=stage1.last_cv,
            quantisation_error=quant_error,
            data=subset.copy(),
            cv_history=stage1.cv_history.copy(),
        )
        self.all_nodes.append(node)

        if level < self.max_depth - 1:
            splits = self._attempt_split(subset, indices, stage1)
            for child_indices in splits:
                child = self._build_node(data, child_indices, level=level + 1)
                node.add_child(child)

        if not node.children:
            self.leaf_nodes.append(node)
        return node

    def _attempt_split(
        self,
        subset: np.ndarray,
        indices: np.ndarray,
        stage1: ProteusStage1,
    ) -> List[np.ndarray]:
        if indices.size < 2 * self.min_cluster_size:
            return []
        positions = stage1.get_node_positions()
        if positions.shape[0] < max(self.params.min_split_nodes, 2):
            return []

        centres, labels = self._kmeans2(positions)
        if centres is None or labels is None:
            return []

        unique_labels = np.unique(labels)
        if unique_labels.size < 2:
            return []

        total_scatter = float(np.sum((positions - positions.mean(axis=0)) ** 2))
        within_scatter = 0.0
        for label in unique_labels:
            cluster_points = positions[labels == label]
            if cluster_points.size == 0:
                return []
            centre = centres[label]
            within_scatter += float(np.sum((cluster_points - centre) ** 2))
        improvement = 1.0 - within_scatter / (total_scatter + 1e-8)
        if improvement < self.params.split_improvement_threshold:
            return []

        assignments = self._assign_points_to_nodes(subset, positions)
        cluster_labels = labels[assignments]

        child_indices: List[np.ndarray] = []
        for label in unique_labels:
            mask = cluster_labels == label
            count = int(mask.sum())
            if count < self.min_cluster_size:
                return []
            fraction = count / float(indices.size)
            if fraction < self.params.split_balance_fraction:
                return []
            child_indices.append(indices[mask])

        if len(child_indices) < 2:
            return []
        return child_indices

    @staticmethod
    def _assign_points_to_nodes(points: np.ndarray, nodes: np.ndarray) -> np.ndarray:
        deltas = points[:, None, :] - nodes[None, :, :]
        distances = np.sum(deltas * deltas, axis=2)
        return np.argmin(distances, axis=1)

    def _kmeans2(self, points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        unique_points = np.unique(points, axis=0)
        if unique_points.shape[0] < 2:
            return None, None
        deltas = unique_points[:, None, :] - unique_points[None, :, :]
        dist_sq = np.sum(deltas * deltas, axis=2)
        i, j = np.unravel_index(np.argmax(dist_sq), dist_sq.shape)
        if i == j:
            return None, None
        centres = np.stack([unique_points[i], unique_points[j]], axis=0)
        labels = None
        for _ in range(50):
            distances = np.sum((points[:, None, :] - centres[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(distances, axis=1)
            new_centres = centres.copy()
            for centre_idx in range(centres.shape[0]):
                mask = new_labels == centre_idx
                if mask.any():
                    new_centres[centre_idx] = points[mask].mean(axis=0)
            if labels is not None and np.array_equal(new_labels, labels):
                centres = new_centres
                break
            centres = new_centres
            labels = new_labels
        if labels is None or np.unique(labels).size < 2:
            return None, None
        # Ensure deterministic ordering based on the first coordinate.
        order = np.argsort(centres[:, 0])
        centres = centres[order]
        relabelling = {old: new for new, old in enumerate(order)}
        remapped_labels = np.array([relabelling[label] for label in labels], dtype=int)
        return centres, remapped_labels

    def _calculate_max_depth(self, node: Optional[HierarchyNode]) -> int:
        if node is None:
            return 0
        if not node.children:
            return node.level
        return max(self._calculate_max_depth(child) for child in node.children)


__all__ = ["AlgorithmParameters", "HierarchyNode", "ProteusAlgorithm"]
