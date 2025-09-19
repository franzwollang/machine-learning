#!/usr/bin/env python3
"""Compare Proteus hierarchy discovery against a synthetic ground truth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

from src.proteus_algorithm import AlgorithmParameters, ProteusAlgorithm

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


# ---------------------------------------------------------------------------
# Synthetic ground truth
# ---------------------------------------------------------------------------


def calculate_dimensionally_dependent_min_cluster_size(
    dimension: int,
) -> int:
    """Heuristic for the smallest cluster size a hierarchy node may represent."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    return max(int(dimension * 300), 400)


@dataclass
class GroundTruthCluster:
    cluster_id: int
    level: int
    parent_id: Optional[int]
    center: np.ndarray
    covariance: np.ndarray
    weight: float
    is_leaf: bool

    @property
    def scale(self) -> float:
        return float(np.sqrt(np.trace(self.covariance)))


@dataclass
class HierarchicalGroundTruth:
    clusters: List[GroundTruthCluster]
    hierarchy_depth: int

    def __post_init__(self) -> None:
        self._clusters_by_id: Dict[int, GroundTruthCluster] = {
            cluster.cluster_id: cluster for cluster in self.clusters
        }
        self._levels: Dict[int, List[GroundTruthCluster]] = {}
        for cluster in self.clusters:
            self._levels.setdefault(cluster.level, []).append(cluster)
        self._leaf_ids: List[int] = [
            cluster.cluster_id for cluster in self.clusters if cluster.is_leaf
        ]
        self._children_by_parent: Dict[int, List[GroundTruthCluster]] = {}
        for cluster in self.clusters:
            if cluster.parent_id is not None:
                self._children_by_parent.setdefault(cluster.parent_id, []).append(
                    cluster
                )
        self._parent_map: Dict[int, int] = {
            cluster.cluster_id: (
                cluster.parent_id if cluster.parent_id is not None else -1
            )
            for cluster in self.clusters
        }

    @property
    def leaf_ids(self) -> List[int]:
        return self._leaf_ids

    def cluster(self, cluster_id: int) -> GroundTruthCluster:
        return self._clusters_by_id[cluster_id]

    def clusters_by_level(self) -> Dict[int, List[GroundTruthCluster]]:
        return self._levels

    def children_of(self, parent_id: int) -> List[GroundTruthCluster]:
        return self._children_by_parent.get(parent_id, [])

    def generate_data(
        self,
        n_samples: int,
        *,
        random_state: int = 0,
        parent_noise_fraction: float = 0.08,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if parent_noise_fraction < 0.0:
            raise ValueError("parent_noise_fraction must be non-negative")
        rng = np.random.default_rng(random_state)
        leaves = [cluster for cluster in self.clusters if cluster.is_leaf]
        weights = np.array([cluster.weight for cluster in leaves], dtype=float)
        weights /= weights.sum()
        counts = rng.multinomial(n_samples, weights)

        pieces: List[np.ndarray] = []
        leaf_labels: List[np.ndarray] = []
        parent_labels: List[np.ndarray] = []
        leaf_counts: Dict[int, int] = {}
        for cluster, count in zip(leaves, counts):
            if count == 0:
                continue
            samples = rng.multivariate_normal(
                cluster.center, cluster.covariance, size=count
            )
            pieces.append(samples)
            leaf_labels.append(np.full(count, cluster.cluster_id, dtype=int))
            parent = cluster.parent_id if cluster.parent_id is not None else -1
            parent_labels.append(np.full(count, parent, dtype=int))
            leaf_counts[cluster.cluster_id] = (
                leaf_counts.get(cluster.cluster_id, 0) + count
            )

        dimension = leaves[0].center.shape[0] if leaves else 0

        # Inject sparse parent-level cover noise.
        if parent_noise_fraction > 0.0 and dimension > 0:
            for parent_cluster in [
                cluster for cluster in self.clusters if not cluster.is_leaf
            ]:
                children = self.children_of(parent_cluster.cluster_id)
                if not children:
                    continue
                child_counts = [
                    leaf_counts.get(child.cluster_id, 0) for child in children
                ]
                total_child_mass = int(sum(child_counts))
                if total_child_mass == 0:
                    continue
                mu_children = np.array([child.center for child in children])
                parent_mean = mu_children.mean(axis=0)
                radii = []
                for child in children:
                    eigvals = np.linalg.eigvalsh(child.covariance)
                    max_eig = float(np.max(eigvals))
                    offset = float(np.linalg.norm(child.center - parent_mean))
                    radii.append(offset + 2.0 * math.sqrt(max_eig))
                sigma_parent = max(radii) / 2.0 if radii else 0.0
                if sigma_parent <= 0.0:
                    sigma_parent = math.sqrt(
                        max(
                            float(np.max(np.linalg.eigvalsh(child.covariance)))
                            for child in children
                        )
                    )
                cov_scale = (sigma_parent**2) / max(float(dimension), 1.0)
                parent_cov = np.eye(dimension) * cov_scale
                parent_count = int(
                    math.ceil(parent_noise_fraction * float(total_child_mass))
                )
                if parent_count <= 0:
                    continue
                parent_samples = rng.multivariate_normal(
                    parent_mean, parent_cov, size=parent_count
                )
                pieces.append(parent_samples)
                leaf_labels.append(np.full(parent_count, -1, dtype=int))
                parent_labels.append(
                    np.full(parent_count, parent_cluster.cluster_id, dtype=int)
                )

        if not pieces:
            raise RuntimeError("Failed to sample any ground truth clusters")

        data = np.vstack(pieces)
        leaf_labels_arr = np.concatenate(leaf_labels)
        parent_labels_arr = np.concatenate(parent_labels)
        permutation = rng.permutation(data.shape[0])
        data = data[permutation]
        leaf_labels_arr = leaf_labels_arr[permutation]
        parent_labels_arr = parent_labels_arr[permutation]
        return data, leaf_labels_arr, parent_labels_arr


def _leaf_colour_lookup(
    ground_truth: HierarchicalGroundTruth,
) -> Tuple[np.ndarray, Dict[int, int]]:
    leaf_ids = ground_truth.leaf_ids
    if not leaf_ids:
        return np.zeros((0, 4)), {}
    colors = matplotlib.colormaps["tab10"](np.linspace(0.0, 1.0, len(leaf_ids)))
    mapping = {leaf_id: idx for idx, leaf_id in enumerate(leaf_ids)}
    return colors, mapping


def create_hierarchical_gaussian_mixture() -> HierarchicalGroundTruth:
    clusters: List[GroundTruthCluster] = []
    cluster_id = 0

    left_parent = GroundTruthCluster(
        cluster_id=cluster_id,
        level=1,
        parent_id=None,
        center=np.array([-3.0, 0.0]),
        covariance=np.array([[1.0, 0.0], [0.0, 0.9]]),
        weight=0.5,
        is_leaf=False,
    )
    clusters.append(left_parent)
    cluster_id += 1

    right_parent = GroundTruthCluster(
        cluster_id=cluster_id,
        level=1,
        parent_id=None,
        center=np.array([3.2, -0.2]),
        covariance=np.array([[1.1, 0.0], [0.0, 0.8]]),
        weight=0.5,
        is_leaf=False,
    )
    clusters.append(right_parent)
    cluster_id += 1

    leaves = [
        GroundTruthCluster(
            cluster_id=cluster_id,
            level=2,
            parent_id=left_parent.cluster_id,
            center=np.array([-4.2, 0.9]),
            covariance=np.array([[0.35, 0.0], [0.0, 0.22]]),
            weight=0.25,
            is_leaf=True,
        ),
        GroundTruthCluster(
            cluster_id=cluster_id + 1,
            level=2,
            parent_id=left_parent.cluster_id,
            center=np.array([-1.9, -1.1]),
            covariance=np.array([[0.4, 0.0], [0.0, 0.28]]),
            weight=0.25,
            is_leaf=True,
        ),
        GroundTruthCluster(
            cluster_id=cluster_id + 2,
            level=2,
            parent_id=right_parent.cluster_id,
            center=np.array([2.2, 1.3]),
            covariance=np.array([[0.3, 0.0], [0.0, 0.2]]),
            weight=0.25,
            is_leaf=True,
        ),
        GroundTruthCluster(
            cluster_id=cluster_id + 3,
            level=2,
            parent_id=right_parent.cluster_id,
            center=np.array([4.4, -1.2]),
            covariance=np.array([[0.32, 0.0], [0.0, 0.24]]),
            weight=0.25,
            is_leaf=True,
        ),
    ]
    clusters.extend(leaves)

    return HierarchicalGroundTruth(clusters=clusters, hierarchy_depth=2)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def plot_ground_truth_structure(
    ax: plt.Axes,
    ground_truth: HierarchicalGroundTruth,
    data: np.ndarray,
    leaf_labels: np.ndarray,
    title: str = "Ground Truth Structure",
) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold")
    color_map, leaf_to_idx = _leaf_colour_lookup(ground_truth)
    ax.scatter(
        data[:, 0],
        data[:, 1],
        c="lightgray",
        s=14,
        alpha=0.18,
        linewidths=0.0,
    )

    parent_mask = leaf_labels < 0
    leaf_mask = ~parent_mask
    if parent_mask.any():
        ax.scatter(
            data[parent_mask, 0],
            data[parent_mask, 1],
            color="dimgray",
            s=16,
            alpha=0.35,
            linewidths=0.0,
        )

    if leaf_mask.any():
        coloured_labels = leaf_labels[leaf_mask]
        colours = np.array(
            [color_map[leaf_to_idx[int(label)]] for label in coloured_labels]
        )
        ax.scatter(
            data[leaf_mask, 0],
            data[leaf_mask, 1],
            c=colours,
            s=18,
            alpha=0.6,
            linewidths=0.0,
        )

    for level, clusters in ground_truth.clusters_by_level().items():
        for cluster in clusters:
            eigenvals, eigenvecs = np.linalg.eigh(cluster.covariance)
            order = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[order]
            eigenvecs = eigenvecs[:, order]
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width, height = 2.0 * np.sqrt(eigenvals) * 2.0
            linestyle = "--" if level == 1 else "-"
            ellipse = patches.Ellipse(
                cluster.center,
                width,
                height,
                angle=angle,
                linewidth=1.8,
                edgecolor=(
                    "black"
                    if level == 1
                    else color_map[leaf_to_idx.get(cluster.cluster_id, 0)]
                ),
                facecolor="none",
                linestyle=linestyle,
                alpha=0.7,
            )
            ax.add_patch(ellipse)
            ax.plot(
                cluster.center[0],
                cluster.center[1],
                "o",
                color=(
                    "black"
                    if level == 1
                    else color_map[leaf_to_idx.get(cluster.cluster_id, 0)]
                ),
                markersize=6,
                markeredgewidth=0.0,
            )
            ax.annotate(
                f"L{level}",
                cluster.center,
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")


def plot_proteus_structure(
    ax: plt.Axes,
    proteus: ProteusAlgorithm,
    data: np.ndarray,
    title: str = "Proteus Discovered Structure",
) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold")
    if not proteus.all_nodes:
        ax.text(
            0.5,
            0.5,
            "No structure",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return

    max_level = max(node.level for node in proteus.all_nodes)
    colors = matplotlib.colormaps["viridis"](np.linspace(0.0, 1.0, max_level + 1))

    ax.scatter(data[:, 0], data[:, 1], c="lightgray", s=10, alpha=0.25, linewidths=0.0)

    for node in proteus.all_nodes:
        positions = node.stage1_nodes
        if positions.shape[0] == 0:
            continue
        colour = colors[node.level]
        graph = node.neighbour_graph
        for i, neighbours in graph.items():
            for j in neighbours:
                if j <= i:
                    continue
                p_i = positions[i]
                p_j = positions[j]
                if p_i.shape[0] >= 2 and p_j.shape[0] >= 2:
                    ax.plot(
                        [p_i[0], p_j[0]],
                        [p_i[1], p_j[1]],
                        color=colour,
                        linewidth=1.0,
                        alpha=0.7,
                    )
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=[colour],
            edgecolors="black",
            linewidths=0.6,
            s=28,
            alpha=0.85,
        )
        ax.annotate(
            f"L{node.level}",
            node.centroid,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=colour, alpha=0.9),
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[level],
            markersize=8,
            label=f"Level {level}",
        )
        for level in range(max_level + 1)
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")


def plot_proteus_structure_by_level(
    axes_by_level: Dict[int, plt.Axes], proteus: ProteusAlgorithm, data: np.ndarray
) -> None:
    """Render Proteus nodes for each hierarchy level in separate panels."""

    if not axes_by_level:
        return

    levels = sorted(axes_by_level.keys())
    colours = matplotlib.colormaps["viridis"](
        np.linspace(0.1, 0.9, max(len(levels), 1))
    )
    level_to_colour = {
        level: colours[idx if len(levels) > 1 else 0]
        for idx, level in enumerate(levels)
    }

    background_kwargs = dict(c="lightgray", s=12, alpha=0.18, linewidths=0.0)

    if not proteus.all_nodes:
        for level, ax in axes_by_level.items():
            ax.set_title(f"Proteus Level {level}", fontsize=12, fontweight="bold")
            ax.scatter(data[:, 0], data[:, 1], **background_kwargs)
            ax.text(
                0.5,
                0.5,
                "No nodes",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.grid(True, alpha=0.25)
            ax.axis("equal")
        return

    for level in levels:
        ax = axes_by_level[level]
        ax.set_title(f"Proteus Level {level}", fontsize=12, fontweight="bold")
        ax.scatter(data[:, 0], data[:, 1], **background_kwargs)
        nodes_at_level = [node for node in proteus.all_nodes if node.level == level]
        if not nodes_at_level:
            ax.text(
                0.5,
                0.5,
                "No nodes",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )
        colour = level_to_colour[level]
        for node in nodes_at_level:
            positions = node.stage1_nodes
            if positions.size == 0:
                continue
            graph = node.neighbour_graph
            for i, neighbours in graph.items():
                for j in neighbours:
                    if j <= i:
                        continue
                    p_i = positions[i]
                    p_j = positions[j]
                    ax.plot(
                        [p_i[0], p_j[0]],
                        [p_i[1], p_j[1]],
                        color=colour,
                        linewidth=1.0,
                        alpha=0.55,
                    )
            ax.scatter(
                positions[:, 0],
                positions[:, 1],
                c=[colour],
                edgecolors="black",
                linewidths=0.5,
                s=28,
                alpha=0.88,
            )
            ax.scatter(
                [node.centroid[0]],
                [node.centroid[1]],
                marker="*",
                color=colour,
                edgecolors="black",
                linewidths=0.6,
                s=70,
                alpha=0.95,
            )
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.grid(True, alpha=0.25)
        ax.axis("equal")


def create_hierarchy_tree_plot(
    ax: plt.Axes,
    proteus: ProteusAlgorithm,
    ground_truth: HierarchicalGroundTruth,
    title: str = "Hierarchy Comparison",
) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold")

    gt_levels = ground_truth.clusters_by_level()
    proteus_levels: Dict[int, List] = {}
    for node in proteus.all_nodes:
        proteus_levels.setdefault(node.level, []).append(node)

    max_depth = max(
        max(gt_levels.keys(), default=0),
        max((node.level for node in proteus.all_nodes), default=0),
    )
    ax.set_ylim(-0.5, max_depth + 0.5)
    ax.set_xlim(-0.6, 1.6)

    ax.text(
        0.0,
        max_depth + 0.3,
        "Ground Truth",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
    for level, clusters in gt_levels.items():
        y_pos = max_depth - level
        for idx, cluster in enumerate(clusters):
            x_pos = -0.15 + (idx - (len(clusters) - 1) / 2.0) * 0.15
            circle = patches.Circle((x_pos, y_pos), 0.04, color="steelblue", alpha=0.8)
            ax.add_patch(circle)
            ax.text(
                x_pos,
                y_pos,
                str(cluster.cluster_id),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )

    ax.text(
        1.0,
        max_depth + 0.3,
        "Proteus",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
    for level in range(max_depth + 1):
        nodes = proteus_levels.get(level, [])
        y_pos = max_depth - level
        for idx, node in enumerate(nodes):
            x_pos = 1.0 + (idx - (len(nodes) - 1) / 2.0) * 0.15
            circle = patches.Circle((x_pos, y_pos), 0.04, color="darkorange", alpha=0.8)
            ax.add_patch(circle)
            ax.text(
                x_pos,
                y_pos,
                str(len(node.indices)),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )

    for level in range(max_depth + 1):
        y_pos = max_depth - level
        ax.text(
            -0.45,
            y_pos,
            f"Level {level}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["GT", "Proteus"])
    ax.set_ylabel("Tree Level")
    ax.grid(True, alpha=0.2)


@dataclass
class ExperimentResult:
    quantisation_error: float
    quantisation_ratio: float
    mean_leaf_purity: float
    min_leaf_purity: float
    leaf_recall: float
    max_leaf_center_error_ratio: float
    proteus_leaf_count: int
    ground_truth_leaf_count: int


@dataclass
class ExperimentOutputs:
    ground_truth: HierarchicalGroundTruth
    proteus: ProteusAlgorithm
    data: np.ndarray
    leaf_labels: np.ndarray
    parent_labels: np.ndarray
    metrics: ExperimentResult
    figure_path: Path
    metrics_log_path: Path
    summary_log_path: Path


def evaluate_hierarchy(
    proteus: ProteusAlgorithm,
    ground_truth: HierarchicalGroundTruth,
    data: np.ndarray,
    leaf_labels: np.ndarray,
) -> ExperimentResult:
    if proteus.hierarchy_root is None:
        raise RuntimeError("Proteus hierarchy has not been built")

    data_scale = float(np.linalg.norm(data, axis=1).mean())
    quant_error = proteus.hierarchy_root.quantisation_error
    quant_ratio = quant_error / (data_scale + 1e-8)

    purities: List[float] = []
    dominant_labels: set[int] = set()
    max_leaf_id = max(ground_truth.leaf_ids, default=-1)
    for node in proteus.leaf_nodes:
        labels = leaf_labels[node.indices]
        valid_mask = labels >= 0
        if not np.any(valid_mask):
            continue
        valid_labels = labels[valid_mask]
        counts = np.bincount(valid_labels, minlength=max_leaf_id + 1)
        if counts.sum() == 0:
            continue
        dominant = int(np.argmax(counts))
        dominant_labels.add(dominant)
        purity = counts[dominant] / max(int(valid_mask.sum()), 1)
        purities.append(purity)

    if not purities:
        raise RuntimeError("Proteus did not produce any leaf nodes")

    gt_leaves = [ground_truth.cluster(leaf_id) for leaf_id in ground_truth.leaf_ids]
    centroid_errors: List[float] = []
    for cluster in gt_leaves:
        distances = [
            float(np.linalg.norm(node.centroid - cluster.center))
            for node in proteus.leaf_nodes
        ]
        centroid_errors.append(min(distances) / (cluster.scale + 1e-8))

    return ExperimentResult(
        quantisation_error=quant_error,
        quantisation_ratio=quant_ratio,
        mean_leaf_purity=float(np.mean(purities)),
        min_leaf_purity=float(np.min(purities)),
        leaf_recall=len(dominant_labels) / max(len(gt_leaves), 1),
        max_leaf_center_error_ratio=float(max(centroid_errors)),
        proteus_leaf_count=len(proteus.leaf_nodes),
        ground_truth_leaf_count=len(gt_leaves),
    )


def format_summary_statistics(
    metrics: ExperimentResult, proteus: ProteusAlgorithm
) -> str:
    depth = proteus._calculate_max_depth(proteus.hierarchy_root)
    return "\n".join(
        [
            f"Proteus nodes: {len(proteus.all_nodes)}",
            f"Proteus depth: {depth}",
            f"Leaf count: {metrics.proteus_leaf_count}",
            f"Quantisation ratio: {metrics.quantisation_ratio:.3f}",
            f"Min leaf purity: {metrics.min_leaf_purity:.3f}",
            f"Leaf recall: {metrics.leaf_recall:.3f}",
            f"Max centre error ratio: {metrics.max_leaf_center_error_ratio:.3f}",
        ]
    )


def render_summary(
    ax: plt.Axes, metrics: ExperimentResult, proteus: ProteusAlgorithm
) -> str:
    ax.set_title("Summary Statistics", fontsize=14, fontweight="bold")
    ax.axis("off")
    summary_text = format_summary_statistics(metrics, proteus)
    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        fontfamily="monospace",
    )
    return summary_text


def render_visualisation(
    output_dir: Path,
    ground_truth: HierarchicalGroundTruth,
    proteus: ProteusAlgorithm,
    data: np.ndarray,
    leaf_labels: np.ndarray,
    metrics: ExperimentResult,
) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    levels = sorted({node.level for node in proteus.all_nodes})
    level_count = len(levels)
    visual_panels = max(level_count + 1, 1)
    max_columns = 4
    n_columns = min(max_columns, max(visual_panels, 2))
    visual_rows = int(math.ceil(visual_panels / n_columns))
    total_rows = visual_rows + 1

    fig = plt.figure(
        figsize=(4.6 * n_columns, 3.8 * total_rows),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(
        total_rows,
        n_columns,
        height_ratios=[3.0] * visual_rows + [1.7],
        hspace=0.35,
        wspace=0.28,
    )
    fig.suptitle(
        "Proteus Hierarchy vs Ground Truth",
        fontsize=16,
        fontweight="bold",
    )

    axes_sequence: List[plt.Axes] = []
    for panel_idx in range(visual_panels):
        row = panel_idx // n_columns
        col = panel_idx % n_columns
        axes_sequence.append(fig.add_subplot(gs[row, col]))

    # Fill any unused cells in the top block with invisible axes to keep layout tidy.
    for filler_idx in range(visual_panels, visual_rows * n_columns):
        row = filler_idx // n_columns
        col = filler_idx % n_columns
        filler_ax = fig.add_subplot(gs[row, col])
        filler_ax.axis("off")

    ground_truth_ax = axes_sequence[0]
    plot_ground_truth_structure(ground_truth_ax, ground_truth, data, leaf_labels)

    axes_by_level: Dict[int, plt.Axes] = {}
    for offset, level in enumerate(levels, start=1):
        if offset < len(axes_sequence):
            axes_by_level[level] = axes_sequence[offset]

    if axes_by_level:
        plot_proteus_structure_by_level(axes_by_level, proteus, data)

        color_map, leaf_to_idx = _leaf_colour_lookup(ground_truth)
        leaf_mask = leaf_labels >= 0
        if leaf_mask.any() and color_map.size:
            coloured_labels = leaf_labels[leaf_mask]
            colours = np.array(
                [color_map[leaf_to_idx[int(label)]] for label in coloured_labels]
            )
            for ax in axes_by_level.values():
                ax.scatter(
                    data[leaf_mask, 0],
                    data[leaf_mask, 1],
                    c=colours,
                    s=16,
                    alpha=0.35,
                    linewidths=0.0,
                )

    tree_colspan = max(n_columns - 1, 1)
    tree_ax = fig.add_subplot(gs[-1, :tree_colspan])
    summary_ax = fig.add_subplot(gs[-1, tree_colspan:])

    create_hierarchy_tree_plot(tree_ax, proteus, ground_truth)
    summary_text = render_summary(summary_ax, metrics, proteus)

    figure_path = output_dir / "hierarchy_comparison.png"
    fig.savefig(str(figure_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path, summary_text


def create_comparison_visualization(
    output_dir: Optional[Path] = None,
    *,
    n_samples: int = 2600,
    random_state: int = 42,
) -> ExperimentOutputs:
    if output_dir is None:
        output_dir = ARTIFACTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = create_hierarchical_gaussian_mixture()
    data, leaf_labels, parent_labels = ground_truth.generate_data(
        n_samples=n_samples, random_state=random_state
    )

    params = AlgorithmParameters(
        random_seed=random_state,
        stage1_samples_per_epoch=1400,
        stage1_max_epochs=20,
    )
    estimated_local_dim = data.shape[1]
    min_cluster_size = calculate_dimensionally_dependent_min_cluster_size(
        estimated_local_dim
    )

    proteus = ProteusAlgorithm(
        params=params,
        max_depth=3,
        min_cluster_size=min_cluster_size,
        run_stage2=False,
    )
    proteus.run_complete_algorithm(data)
    metrics = evaluate_hierarchy(proteus, ground_truth, data, leaf_labels)

    depth = proteus._calculate_max_depth(proteus.hierarchy_root)
    metrics_log_path = output_dir / "hierarchy_comparison_metrics.txt"
    metrics_log_lines = [
        "Proteus hierarchy metrics",
        f"Samples: {data.shape[0]}",
        f"Proteus nodes: {len(proteus.all_nodes)}",
        f"Proteus depth: {depth}",
        f"Proteus leaves: {metrics.proteus_leaf_count}",
        f"Ground truth leaves: {metrics.ground_truth_leaf_count}",
        f"Quantisation error: {metrics.quantisation_error:.6f}",
        f"Quantisation ratio: {metrics.quantisation_ratio:.6f}",
        f"Mean leaf purity: {metrics.mean_leaf_purity:.6f}",
        f"Min leaf purity: {metrics.min_leaf_purity:.6f}",
        f"Leaf recall: {metrics.leaf_recall:.6f}",
        f"Max centre error ratio: {metrics.max_leaf_center_error_ratio:.6f}",
    ]
    metrics_log_path.write_text("\n".join(metrics_log_lines) + "\n")

    figure_path, summary_text = render_visualisation(
        output_dir, ground_truth, proteus, data, leaf_labels, metrics
    )
    summary_log_path = output_dir / "hierarchy_comparison_summary.txt"
    summary_log_path.write_text("Proteus hierarchy summary\n" + summary_text + "\n")
    return ExperimentOutputs(
        ground_truth=ground_truth,
        proteus=proteus,
        data=data,
        leaf_labels=leaf_labels,
        parent_labels=parent_labels,
        metrics=metrics,
        figure_path=figure_path,
        metrics_log_path=metrics_log_path,
        summary_log_path=summary_log_path,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_proteus_hierarchy_recovers_ground_truth() -> None:
    outputs = create_comparison_visualization()
    metrics = outputs.metrics
    assert metrics.quantisation_ratio < 0.22
    assert metrics.min_leaf_purity > 0.75
    assert metrics.leaf_recall >= 1.0
    assert metrics.max_leaf_center_error_ratio < 2.0
    assert metrics.proteus_leaf_count == metrics.ground_truth_leaf_count
    assert outputs.figure_path.exists()
    assert outputs.figure_path.parent == ARTIFACTS_DIR
    metrics_log_path = outputs.metrics_log_path
    summary_log_path = outputs.summary_log_path
    assert metrics_log_path.exists()
    assert metrics_log_path.read_text().strip()
    assert summary_log_path.exists()
    assert summary_log_path.read_text().strip()


if __name__ == "__main__":
    create_comparison_visualization()
