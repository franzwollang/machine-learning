"""Reference implementation of the Proteus Stage 1 scaffold builder.

This module implements the core loop from Algorithm~1 in the Proteus
supplement (docs/Proteus/paper_1_foundational/SI.tex).  The goal is to
produce a coarse geometric scaffold of the data manifold by running a
Growing-Neural-Gas-like procedure with principled EWMA statistics and a
thermal-equilibrium stopping rule.

The implementation intentionally prioritises clarity and faithfulness to
the documented algorithm over raw throughput.  It uses a naive
nearest-neighbour search (O(N) per query) which is sufficient for the
small synthetic datasets exercised in the unit tests.  The structure of
the code mirrors the description in the specification so that later
optimisation (e.g. swapping in hnswlib for ANN queries) is localised.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _unit_vector_like(vector: np.ndarray) -> np.ndarray:
    """Return a unit-norm version of ``vector`` (or a stable default)."""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        # Fall back to a canonical axis to keep the Oja update stable.
        e0 = np.zeros_like(vector)
        e0[0] = 1.0
        return e0
    return vector / norm


@dataclass
class ProteusNode:
    """State tracked for each node in the Stage 1 mesh."""

    position: np.ndarray
    residual_mean: np.ndarray
    residual_sq: np.ndarray
    nudge: np.ndarray
    principal_dir: np.ndarray
    hit_count: float = 0.0
    update_count: int = 0
    creation_index: int = 0
    matrix_index: int = 0

    def copy(self) -> "ProteusNode":
        return ProteusNode(
            position=self.position.copy(),
            residual_mean=self.residual_mean.copy(),
            residual_sq=self.residual_sq.copy(),
            nudge=self.nudge.copy(),
            principal_dir=self.principal_dir.copy(),
            hit_count=self.hit_count,
            update_count=self.update_count,
            creation_index=self.creation_index,
            matrix_index=self.matrix_index,
        )


@dataclass
class ProteusLink:
    """Undirected link with directional hit counters."""

    i: int
    j: int
    count_ij: float = 0.0
    count_ji: float = 0.0
    creation_index: int = 0
    protected_until: int = 0

    def total_hits(self) -> float:
        return self.count_ij + self.count_ji


class ProteusStage1:
    """Stage 1 scaffold builder following the specification.

    Parameters mirror the quantities described in the paper: ``k`` is the
    neighbourhood size, ``alpha`` controls the EWMA half life, ``eta_cent``
    is the deferred-nudge scaling, and ``tau`` is the current scale
    threshold.  The implementation maintains node/link objects in plain
    Python containers which keeps the code easy to inspect and extend.
    """

    def __init__(
        self,
        *,
        k_neighbors: int = 8,
        alpha: Optional[float] = None,
        eta_cent: float = 0.1,
        eta_move: float = 0.08,
        eta_oja: float = 0.05,
        grid_ratio: float = 1.0 / np.sqrt(2.0),
        kappa: float = 0.5,
        max_epochs: int = 40,
        max_nodes: int = 256,
        min_nodes: int = 6,
        prune_after: int = 30,
        min_link_strength: float = 4.0,
        link_protection: int = 25,
        cv_tolerance: float = 0.01,
        min_equilibrium_epochs: int = 3,
        samples_per_epoch: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.k_neighbors = int(k_neighbors)
        if alpha is None:
            alpha = float(np.log(2.0) / max(self.k_neighbors, 1))
        self.alpha = float(alpha)
        self.eta_cent = float(eta_cent)
        self.eta_move = float(eta_move)
        self.eta_oja = float(eta_oja)
        self.grid_ratio = float(grid_ratio)
        self.kappa = float(kappa)
        self.max_epochs = int(max_epochs)
        self.max_nodes = int(max_nodes)
        self.min_nodes = int(min_nodes)
        self.prune_after = int(prune_after)
        self.min_link_strength = float(min_link_strength)
        self.link_protection = int(link_protection)
        self.cv_tolerance = float(cv_tolerance)
        self.min_equilibrium_epochs = int(min_equilibrium_epochs)
        self.samples_per_epoch = None if samples_per_epoch is None else int(samples_per_epoch)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.nodes: List[ProteusNode] = []
        self.links: Dict[Tuple[int, int], ProteusLink] = {}
        self.dimension: int = 0
        self.tau: float = 0.0
        self.delta_min: float = 0.0
        self.iteration: int = 0
        self.cv_history: List[float] = []
        self.node_history: List[int] = []
        self._last_cv: float = np.inf
        self._eps: float = 1e-8
        self._positions_matrix: np.ndarray = np.zeros((0, 0), dtype=float)
        self._active_node_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        data: np.ndarray,
        *,
        tau: Optional[float] = None,
        d_subspace: Optional[int] = None,
    ) -> "ProteusStage1":
        """Run the Stage 1 loop until thermal equilibrium.

        Parameters
        ----------
        data:
            Array of shape (N, d) with the input samples.
        tau:
            Target variance threshold for the current scale.  If omitted we
            use the global average variance of ``data``.
        d_subspace:
            Estimate of the intrinsic dimensionality.  Defaults to the
            ambient dimensionality of ``data``.
        """

        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError("data must be a 2D array")
        n_samples, self.dimension = data.shape
        if n_samples < 2:
            raise ValueError("Stage 1 requires at least two samples")
        if tau is None:
            tau = float(np.var(data, axis=0).mean())
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        self.tau = float(tau)
        if d_subspace is None:
            d_subspace = self.dimension
        s_control = 1.0 - np.exp(-self.tau / float(d_subspace))
        # tau_local equals tau in the Stage 1 defaults; retaining the more
        # general expression keeps the mapping to the specification clear.
        # Without a per-node dimensionality estimate we default to a
        # conservative scaling that mirrors the Stage~1 intuition: the
        # intrinsic dimensionality is usually lower than the ambient one.
        self.tau_local = self.tau / max(float(d_subspace), 1.0)
        self.delta_min = self.kappa * (1.0 - self.grid_ratio) * np.sqrt(self.tau)

        self._initialise_nodes(data)

        for epoch in range(self.max_epochs):
            order = self.rng.permutation(n_samples)
            if self.samples_per_epoch is not None and self.samples_per_epoch < n_samples:
                epoch_indices = order[: self.samples_per_epoch]
            else:
                epoch_indices = order
            for idx in epoch_indices:
                x = data[idx]
                neighbor_indices = self._query_neighbors(x)
                self._apply_rank_weighted_updates(x, neighbor_indices)
                self._apply_deferred_nudges(neighbor_indices)
                self._check_splits(neighbor_indices)
                self.iteration += 1
            cv = self._compute_cv()
            self.cv_history.append(cv)
            self.node_history.append(len(self.nodes))
            self._last_cv = cv
            self._apply_pruning()
            if (
                epoch + 1 >= self.min_equilibrium_epochs
                and cv < self.cv_tolerance
            ):
                break
        return self

    # ------------------------------------------------------------------
    # Convenience accessors ------------------------------------------------
    @property
    def last_cv(self) -> float:
        return self._last_cv

    def get_node_positions(self) -> np.ndarray:
        if self._active_node_count == 0:
            return np.empty((0, self.dimension), dtype=float)
        return self._positions_matrix[: self._active_node_count].copy()

    def neighbour_graph(self) -> Dict[int, Sequence[int]]:
        adj = {i: [] for i in range(len(self.nodes))}
        for link in self.links.values():
            adj[link.i].append(link.j)
            adj[link.j].append(link.i)
        return adj

    # ------------------------------------------------------------------
    # Internal helpers ------------------------------------------------------
    def _initialise_nodes(self, data: np.ndarray) -> None:
        self.nodes.clear()
        self.links.clear()
        self.iteration = 0
        n_samples = data.shape[0]
        n_init = min(self.min_nodes, n_samples)
        indices = self.rng.choice(n_samples, size=n_init, replace=False)
        self._positions_matrix = np.zeros((self.max_nodes, self.dimension), dtype=float)
        for idx, sample_idx in enumerate(indices):
            sample = data[sample_idx]
            principal = self.rng.normal(size=self.dimension)
            node = ProteusNode(
                position=self._positions_matrix[idx],
                residual_mean=np.zeros_like(sample),
                residual_sq=np.zeros_like(sample),
                nudge=np.zeros_like(sample),
                principal_dir=_unit_vector_like(principal),
                creation_index=idx,
                matrix_index=idx,
            )
            np.copyto(self._positions_matrix[idx], sample)
            self.nodes.append(node)
        self._active_node_count = len(self.nodes)

    def _query_neighbors(self, x: np.ndarray) -> List[int]:
        if not self.nodes:
            raise RuntimeError("No nodes available for neighbour query")
        positions = self._positions_matrix[: self._active_node_count]
        deltas = positions - x
        distances = np.sum(deltas * deltas, axis=1)
        order = np.argsort(distances)
        k = min(self.k_neighbors, len(order))
        chosen = order[:k].tolist()
        if len(chosen) >= 2:
            self._register_link(chosen[0], chosen[1])
        return chosen

    def _register_link(self, i: int, j: int) -> None:
        if i == j:
            return
        if i > j:
            i, j = j, i
        key = (i, j)
        if key not in self.links:
            self.links[key] = ProteusLink(
                i=i,
                j=j,
                creation_index=self.iteration,
                protected_until=self.iteration + self.link_protection,
            )
        link = self.links[key]
        if i == key[0]:
            link.count_ij += 1.0
        else:
            link.count_ji += 1.0

    def _apply_rank_weighted_updates(self, x: np.ndarray, neighbors: Iterable[int]) -> None:
        for rank, idx in enumerate(neighbors):
            node = self.nodes[idx]
            weight = 2.0 ** (-rank)
            position = self._positions_matrix[node.matrix_index]
            error = x - position
            position += self.eta_move * weight * error
            node.residual_mean = (1.0 - self.alpha * weight) * node.residual_mean + self.alpha * weight * error
            node.residual_sq = (1.0 - self.alpha * weight) * node.residual_sq + self.alpha * weight * (error ** 2)
            variance_vec = np.maximum(node.residual_sq - node.residual_mean ** 2, 0.0)
            sigma = float(np.sqrt(np.mean(variance_vec)))
            rho = float(np.linalg.norm(node.residual_mean)) / (sigma + self._eps)
            node.nudge += self.eta_cent * rho * node.residual_mean
            # Oja's rule update for the principal direction
            delta_u = self.eta_oja * (
                error * float(np.dot(error, node.principal_dir))
                - (float(np.dot(error, error)) * node.principal_dir)
            )
            node.principal_dir = _unit_vector_like(node.principal_dir + delta_u)
            node.hit_count += weight
            node.update_count += 1

    def _apply_deferred_nudges(self, neighbors: Iterable[int]) -> None:
        for idx in neighbors:
            node = self.nodes[idx]
            magnitude = float(np.linalg.norm(node.nudge))
            if magnitude >= self.delta_min:
                position = self._positions_matrix[node.matrix_index]
                position += node.nudge
                node.position = position
                node.nudge.fill(0.0)

    def _check_splits(self, neighbors: Iterable[int]) -> None:
        if len(self.nodes) >= self.max_nodes:
            return
        for idx in neighbors:
            node = self.nodes[idx]
            variance_vec = np.maximum(node.residual_sq - node.residual_mean ** 2, 0.0)
            sigma_sq = float(np.mean(variance_vec))
            if sigma_sq > self.tau_local:
                self._split_node(idx)

    def _split_node(self, idx: int) -> None:
        if len(self.nodes) >= self.max_nodes:
            return
        parent = self.nodes[idx]
        offset = parent.residual_mean
        if float(np.linalg.norm(offset)) < 1e-6:
            offset = parent.principal_dir * np.sqrt(self.tau) * 0.5
        child_index = len(self.nodes)
        if child_index >= self.max_nodes:
            return
        parent_position = self._positions_matrix[parent.matrix_index]
        child_position = self._positions_matrix[child_index]
        np.copyto(child_position, parent_position)
        child_position += offset
        child = ProteusNode(
            position=child_position,
            residual_mean=0.5 * parent.residual_mean,
            residual_sq=parent.residual_sq.copy(),
            nudge=np.zeros_like(parent_position),
            principal_dir=parent.principal_dir.copy(),
            hit_count=parent.hit_count * 0.5,
            update_count=0,
            creation_index=self.iteration,
            matrix_index=child_index,
        )
        # Shrink parent statistics to reflect the split.
        parent.residual_mean *= 0.5
        parent.hit_count *= 0.5
        parent.position = parent_position
        self.nodes.append(child)
        self._active_node_count = len(self.nodes)
        # Existing links remain valid; neighbours will be picked up naturally.

    def _compute_cv(self) -> float:
        n_nodes = len(self.nodes)
        if n_nodes == 0:
            return float("inf")
        rho = np.zeros(n_nodes)
        for i, node in enumerate(self.nodes):
            variance_vec = np.maximum(node.residual_sq - node.residual_mean ** 2, 0.0)
            sigma = float(np.sqrt(np.mean(variance_vec)))
            rho[i] = float(np.linalg.norm(node.residual_mean)) / (sigma + self._eps)
        global_mean = float(np.mean(rho))
        rho_tilde = rho / (global_mean + self._eps)
        mean = float(np.mean(rho_tilde))
        if mean <= 0.0:
            return float("inf")
        std = float(np.std(rho_tilde))
        return std / mean

    def _apply_pruning(self) -> None:
        if len(self.nodes) <= self.min_nodes:
            return
        self._prune_links()
        self._prune_nodes()

    def _prune_links(self) -> None:
        to_remove = []
        for key, link in self.links.items():
            if self.iteration < link.protected_until:
                continue
            if link.total_hits() < self.min_link_strength:
                to_remove.append(key)
        for key in to_remove:
            del self.links[key]

    def _prune_nodes(self) -> None:
        if len(self.nodes) <= self.min_nodes:
            return
        neighbour_map = self.neighbour_graph()
        hits = np.array([node.hit_count for node in self.nodes])
        variances = np.array([
            float(np.mean(np.maximum(node.residual_sq - node.residual_mean ** 2, 0.0)))
            for node in self.nodes
        ])
        to_remove: List[int] = []
        for idx, node in enumerate(self.nodes):
            if node.update_count < self.prune_after:
                continue
            neighbours = neighbour_map[idx]
            if not neighbours:
                continue
            neighbour_mean_hits = float(np.mean(hits[neighbours]))
            if (
                node.hit_count < 0.5 * neighbour_mean_hits
                and variances[idx] < 0.5 * self.tau_local
            ):
                to_remove.append(idx)
        if not to_remove:
            return
        keep_mask = np.ones(len(self.nodes), dtype=bool)
        keep_mask[to_remove] = False
        new_index = np.cumsum(keep_mask) - 1
        self.nodes = [node for keep, node in zip(keep_mask, self.nodes) if keep]
        self._reindex_nodes()
        new_links: Dict[Tuple[int, int], ProteusLink] = {}
        for key, link in self.links.items():
            if not (keep_mask[link.i] and keep_mask[link.j]):
                continue
            i_new = int(new_index[link.i])
            j_new = int(new_index[link.j])
            if i_new > j_new:
                i_new, j_new = j_new, i_new
                link = ProteusLink(
                    i=i_new,
                    j=j_new,
                    count_ij=link.count_ji,
                    count_ji=link.count_ij,
                    creation_index=link.creation_index,
                    protected_until=link.protected_until,
                )
            else:
                link = ProteusLink(
                    i=i_new,
                    j=j_new,
                    count_ij=link.count_ij,
                    count_ji=link.count_ji,
                    creation_index=link.creation_index,
                    protected_until=link.protected_until,
                )
            new_links[(link.i, link.j)] = link
        self.links = new_links

    def _reindex_nodes(self) -> None:
        for idx, node in enumerate(self.nodes):
            np.copyto(self._positions_matrix[idx], node.position)
            node.matrix_index = idx
            node.position = self._positions_matrix[idx]
        self._active_node_count = len(self.nodes)


__all__ = ["ProteusStage1", "ProteusNode", "ProteusLink"]
