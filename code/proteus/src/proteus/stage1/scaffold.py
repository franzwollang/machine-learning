"""Fixed-tau Stage 1 scaffold loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from proteus.ann import ANNIndex, make_ann
from proteus.deferred import accumulate_nudge, apply_if_threshold
from proteus.intrinsic_dim import estimate_d_final
from proteus.links import LinkCounters
from proteus.nodes import accumulate_hit, make_node, update_node_moments
from proteus.oja import update_oja
from proteus.rates import delta_min, eta_cent
from proteus.stage1.pruning import (
    count_mature_lifted_isolated,
    prune_links,
    prune_nodes,
)
from proteus.stage1.routing_weights import routing_weights
from proteus.stage1.splits import apply_split, propose_splits
from proteus.stage1.stabilization import (
    StabilizationConfig,
    compute_neighbor_normalized_cv,
    compute_variance_cv,
    is_stable,
)
from tests.contracts.state import Link, NodeState
from tests.metrics.reconstruction import mean_min_distance


@dataclass
class RouteStats:
    """Per-sample update statistics."""

    deferred_fires: int = 0
    routed_nodes: int = 0


class Stage1Scaffold:
    """Fixed-tau Stage 1 scaffold."""

    def __init__(
        self,
        *,
        dim: int,
        tau: float,
        k: int = 8,
        kappa: float = 0.5,
        grid_ratio: float = 1.0 / np.sqrt(2.0),
        oja_lr: float = 0.05,
        ann_backend: str = "auto",
        min_nodes: int = 4,
        max_nodes: int = 128,
        prune_after: int = 25,
        prune_hit_fraction: float = 0.20,
        prune_beta: float = 0.5,
        link_protection: int = 25,
        enable_topology_edits: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if int(dim) < 1:
            raise ValueError("dim must be positive")
        if float(tau) <= 0.0:
            raise ValueError("tau must be positive")
        if int(k) < 1:
            raise ValueError("k must be >= 1")
        self.dim = int(dim)
        self.tau = float(tau)
        self.k = int(k)
        self.kappa = float(kappa)
        self.grid_ratio = float(grid_ratio)
        self.oja_lr = float(oja_lr)
        self.min_nodes = int(min_nodes)
        self.max_nodes = int(max_nodes)
        self.prune_after = int(prune_after)
        self.prune_hit_fraction = float(prune_hit_fraction)
        self.prune_beta = float(prune_beta)
        self.link_protection = int(link_protection)
        self.enable_topology_edits = bool(enable_topology_edits)
        self.alpha = float(np.log(2.0) / self.k)
        self.eta_cent_value = eta_cent(self.kappa, self.grid_ratio, self.k)
        self.delta_min_value = delta_min(self.kappa, self.grid_ratio, self.tau)
        self.s_control = 1.0 - float(np.exp(-self.tau / self.dim))
        self.rng = rng if rng is not None else np.random.default_rng()
        self.ann_backend = ann_backend
        self.ann: ANNIndex = make_ann(self.dim, backend=ann_backend)
        self.nodes: list[NodeState] = []
        self.links = LinkCounters()
        self.tau_local = np.empty(0, dtype=float)
        self.last_epoch_stats: dict[str, float] = {}
        self.cv_history: list[float] = []
        self.node_count_history: list[int] = []
        self.iteration = 0

    def init_from(self, points: np.ndarray, n_seeds: int) -> None:
        """Initialize nodes with deterministic farthest-point sampling."""

        points_arr = self._validate_points(points)
        if int(n_seeds) < 1:
            raise ValueError("n_seeds must be positive")
        if int(n_seeds) > points_arr.shape[0]:
            raise ValueError("n_seeds cannot exceed number of points")

        seed_indices = self._farthest_point_seed_indices(points_arr, int(n_seeds))
        self.nodes = [
            make_node(points_arr[idx], self.dim, d_final=self.dim)
            for idx in seed_indices
        ]
        self.links = LinkCounters()
        self.tau_local = np.full(len(self.nodes), self.tau, dtype=float)
        self.iteration = 0
        self.cv_history = []
        self.node_count_history = []
        self.rebuild_ann()

    def route_and_update(
        self, x: np.ndarray, *, _link_protect_base: int | None = None,
    ) -> RouteStats:
        """Route one sample through the fixed scaffold and update node state."""

        if not self.nodes:
            raise RuntimeError("Stage1Scaffold must be initialized before routing")
        x_arr = self._validate_point(x)
        neighbor_ids, distances = self.ann.query_knn(
            x_arr,
            k=min(self.k, len(self.nodes)),
        )
        weights = routing_weights(distances, self.tau, self.dim)
        stats = RouteStats(routed_nodes=len(neighbor_ids))

        for rank, (node_id, weight) in enumerate(zip(neighbor_ids, weights)):
            node = self.nodes[int(node_id)]
            residual = x_arr - node.position
            update_node_moments(
                node, residual, self.alpha, float(weight),
                is_bmu=(rank == 0),
            )
            accumulate_hit(node, float(weight))
            accumulate_nudge(node, node.residual_mean, self.eta_cent_value)
            node.principal_dir = update_oja(node.principal_dir, residual, self.oja_lr)

        base = self.iteration if _link_protect_base is None else int(_link_protect_base)
        protect_deadline = base + self.link_protection
        self.links.record_neighborhood(
            neighbor_ids.tolist(),
            weights.tolist(),
            protected_until=protect_deadline,
        )

        for node_id in neighbor_ids:
            node = self.nodes[int(node_id)]
            if apply_if_threshold(node, self.delta_min_value):
                self.ann.update(int(node_id), node.position)
                stats.deferred_fires += 1

        return stats

    def run_epoch(self, points: np.ndarray) -> dict[str, float]:
        """Run one shuffled epoch and return diagnostic statistics."""

        points_arr = self._validate_points(points)
        order = self.rng.permutation(points_arr.shape[0])
        deferred_fires = 0
        routed_nodes = 0
        epoch_start = self.iteration
        for pos, idx in enumerate(order):
            stats = self.route_and_update(
                points_arr[int(idx)], _link_protect_base=epoch_start + pos,
            )
            deferred_fires += stats.deferred_fires
            routed_nodes += stats.routed_nodes

        hit_counts = np.array([node.hit_count for node in self.nodes], dtype=float)
        variances = np.array([node.variance for node in self.nodes], dtype=float)
        fire_rate = deferred_fires / max(1, routed_nodes)
        self.last_epoch_stats = {
            "deferred_fire_rate": float(fire_rate),
            "deferred_fires": float(deferred_fires),
            "mean_hit_count": float(hit_counts.mean()) if hit_counts.size else 0.0,
            "mean_variance": float(variances.mean()) if variances.size else 0.0,
            "mean_min_distance": mean_min_distance(points_arr, self.node_positions()),
        }
        self.iteration = epoch_start + points_arr.shape[0]
        return dict(self.last_epoch_stats)

    def node_positions(self) -> np.ndarray:
        """Return an array of node positions in stable node-id order."""

        if not self.nodes:
            return np.empty((0, self.dim), dtype=float)
        return np.vstack([node.position for node in self.nodes])

    def link_summary(self) -> list[Link]:
        """Return directed link counters in stable order."""

        return self.links.as_list()

    def neighbour_graph(self) -> dict[int, list[int]]:
        """Return the current undirected scaffold adjacency."""

        return self.links.neighbour_graph(len(self.nodes))

    def rebuild_ann(self) -> None:
        """Rebuild the ANN index from current node positions."""

        self.ann = make_ann(
            self.dim,
            backend=self.ann_backend,
            expected_size=len(self.nodes),
        )
        self.ann.build_from(self.node_positions())

    def refresh_intrinsic_dim(self) -> None:
        """Refresh degree-based d_final estimates (diagnostic only).

        Per-node d_final is updated for junction diagnostics and
        scale-response normalization but does not modulate the cap.
        Within a single scaffold run, tau_local = tau uniformly.
        """

        d_final = estimate_d_final(
            self.neighbour_graph(),
            dim_floor=1,
            ambient_dim=self.dim,
        )
        if d_final.size != len(self.nodes):
            d_final = np.full(len(self.nodes), self.dim, dtype=int)
        for node, d_value in zip(self.nodes, d_final):
            node.d_final = int(d_value)
        self.tau_local = np.full(len(self.nodes), self.tau, dtype=float)

    def _propose_and_apply_splits(self) -> int:
        """Apply variance-cap splits and return the number accepted."""

        if not self.enable_topology_edits or len(self.nodes) >= self.max_nodes:
            return 0
        accepted = 0
        for proposal in propose_splits(self):
            if len(self.nodes) >= self.max_nodes:
                break
            if apply_split(self, proposal):
                accepted += 1
        if accepted:
            self.rebuild_ann()
        return accepted

    def apply_pruning_gauntlet(self) -> dict[str, int]:
        """Run link and node prune gauntlets."""

        if not self.enable_topology_edits:
            return {
                "link_verdicts": 0,
                "nodes_removed": 0,
                "lifted_isolated_mature": 0,
            }
        verdicts = prune_links(self)
        isolated = count_mature_lifted_isolated(self)
        removed = prune_nodes(self)
        return {
            "link_verdicts": len(verdicts),
            "nodes_removed": len(removed),
            "lifted_isolated_mature": isolated,
        }

    def run_until_stable(
        self,
        points: np.ndarray,
        config: StabilizationConfig | None = None,
    ) -> dict[str, list[float]]:
        """Run epochs until variance CV over mature nodes stabilizes."""

        config = config if config is not None else StabilizationConfig()
        self.cv_history = []
        self.node_count_history = []
        history: dict[str, list[float]] = {
            "cv": [],
            "incoherence_cv": [],
            "node_count": [],
            "mean_min_distance": [],
            "deferred_fire_rate": [],
            "splits": [],
            "nodes_pruned": [],
            "lifted_isolated_mature": [],
        }
        for epoch in range(config.max_epochs):
            stats = self.run_epoch(points)
            splits = self._propose_and_apply_splits()
            prune_stats = self.apply_pruning_gauntlet()
            cv = compute_variance_cv(self)

            self.cv_history.append(cv)
            self.node_count_history.append(len(self.nodes))
            history["cv"].append(cv)
            history["incoherence_cv"].append(
                compute_neighbor_normalized_cv(self)
            )
            history["node_count"].append(float(len(self.nodes)))
            history["mean_min_distance"].append(stats["mean_min_distance"])
            history["deferred_fire_rate"].append(stats["deferred_fire_rate"])
            history["splits"].append(float(splits))
            history["nodes_pruned"].append(float(prune_stats["nodes_removed"]))
            history["lifted_isolated_mature"].append(
                float(prune_stats["lifted_isolated_mature"]),
            )
            if is_stable(self.cv_history, config, scaffold=self):
                break
        return history

    def _farthest_point_seed_indices(self, points: np.ndarray, n_seeds: int) -> np.ndarray:
        first = int(self.rng.integers(0, points.shape[0]))
        selected = [first]
        min_dists = np.linalg.norm(points - points[first], axis=1)
        while len(selected) < n_seeds:
            next_idx = int(np.argmax(min_dists))
            selected.append(next_idx)
            dists = np.linalg.norm(points - points[next_idx], axis=1)
            min_dists = np.minimum(min_dists, dists)
        return np.array(selected, dtype=int)

    def _validate_points(self, points: np.ndarray) -> np.ndarray:
        points_arr = np.asarray(points, dtype=float)
        if points_arr.ndim != 2 or points_arr.shape[1] != self.dim:
            raise ValueError(
                f"points must have shape (N, {self.dim}), got {points_arr.shape}"
            )
        return points_arr

    def _validate_point(self, point: np.ndarray) -> np.ndarray:
        point_arr = np.asarray(point, dtype=float)
        if point_arr.shape != (self.dim,):
            raise ValueError(f"point must have shape ({self.dim},), got {point_arr.shape}")
        return point_arr
