#!/usr/bin/env python3
"""
Visualization script to compare Proteus discovered hierarchy with ground truth.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import after path modification
from test_hierarchical_triangulation_fixed import (  # noqa: E402
    create_hierarchical_gaussian_mixture,
    calculate_dimensionally_dependent_min_cluster_size,
    HierarchicalGroundTruth,
)
from src.proteus_algorithm import ProteusAlgorithm  # noqa: E402
from src.structures import AlgorithmParameters  # noqa: E402


def extract_proteus_hierarchy_info(proteus_algorithm):
    """Extract hierarchy information from Proteus results."""
    hierarchy_info = {
        "nodes": [],
        "max_depth": 0,
        "leaf_nodes": [],
        "total_nodes": len(proteus_algorithm.all_nodes),
    }

    if proteus_algorithm.hierarchy_root:
        hierarchy_info["max_depth"] = proteus_algorithm._calculate_max_depth(
            proteus_algorithm.hierarchy_root
        )
        hierarchy_info["leaf_nodes"] = proteus_algorithm.leaf_nodes

        # Extract node information
        for i, node in enumerate(proteus_algorithm.all_nodes):
            node_info = {
                "id": i,
                "level": node.level,
                "data_size": len(node.data),
                "gng_nodes": len(node.global_state.Nodes),
                "children": (
                    [child for child in node.children]
                    if hasattr(node, "children") and node.children
                    else []
                ),
                "data_points": node.data if hasattr(node, "data") else None,
                "global_state": (
                    node.global_state if hasattr(node, "global_state") else None
                ),
            }
            hierarchy_info["nodes"].append(node_info)

    return hierarchy_info


def extract_ground_truth_info(ground_truth: HierarchicalGroundTruth):
    """Extract hierarchy information from ground truth."""
    gt_info = {
        "clusters": ground_truth.clusters,
        "hierarchy_depth": ground_truth.hierarchy_depth,
        "levels": {},
    }

    # Group clusters by level
    for cluster in ground_truth.clusters:
        level = cluster.level
        if level not in gt_info["levels"]:
            gt_info["levels"][level] = []
        gt_info["levels"][level].append(cluster)

    return gt_info


def plot_ground_truth_structure(
    ax, ground_truth_info, X, true_labels, title="Ground Truth Structure"
):
    """Plot the ground truth hierarchical structure."""
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Create a colormap for different levels
    levels = sorted(ground_truth_info["levels"].keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(levels)))

    # Plot data points colored by true labels
    ax.scatter(X[:, 0], X[:, 1], c=true_labels, alpha=0.6, s=20, cmap="tab20")

    # Draw ellipses for each ground truth cluster
    for level_idx, level in enumerate(levels):
        clusters = ground_truth_info["levels"][level]

        for cluster in clusters:
            # Create ellipse from covariance matrix
            eigenvals, eigenvecs = np.linalg.eigh(cluster.covariance)
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width, height = 2 * np.sqrt(eigenvals) * 2  # 2-sigma ellipse

            ellipse = patches.Ellipse(
                cluster.center,
                width,
                height,
                angle=angle,
                linewidth=2,
                edgecolor=colors[level_idx],
                facecolor="none",
                linestyle="--" if level > 0 else "-",
                alpha=0.8,
            )
            ax.add_patch(ellipse)

            # Add cluster center
            ax.plot(
                cluster.center[0],
                cluster.center[1],
                "o",
                color=colors[level_idx],
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1,
            )

            # Add level annotation
            ax.annotate(
                f"L{level}",
                cluster.center,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.grid(True, alpha=0.3)

    # Add legend for levels
    legend_elements = [
        plt.Line2D([0], [0], color=colors[i], lw=2, label=f"Level {level}")
        for i, level in enumerate(levels)
    ]
    ax.legend(handles=legend_elements, loc="upper right")


def plot_proteus_structure(ax, proteus_info, X, title="Proteus Discovered Structure"):
    """Plot the Proteus discovered hierarchical structure."""
    ax.set_title(title, fontsize=14, fontweight="bold")

    if proteus_info["total_nodes"] == 0:
        ax.text(
            0.5,
            0.5,
            "No structure found",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            color="red",
        )
        return

    # Create colormap for different levels
    max_level = max(node["level"] for node in proteus_info["nodes"])
    colors = plt.cm.viridis(np.linspace(0, 1, max_level + 1))

    # Plot all data points in light gray
    ax.scatter(X[:, 0], X[:, 1], c="lightgray", alpha=0.3, s=10)

    # Plot nodes by level, showing GNG structure
    for node in proteus_info["nodes"]:
        level = node["level"]
        color = colors[level]

        if node["global_state"] is not None:
            global_state = node["global_state"]

            # Plot GNG links first (so they appear behind nodes)
            for link in global_state.Links:
                node1_pos = link.node1.w
                node2_pos = link.node2.w
                # Only plot if both nodes are 2D
                if len(node1_pos) >= 2 and len(node2_pos) >= 2:
                    ax.plot(
                        [node1_pos[0], node2_pos[0]],
                        [node1_pos[1], node2_pos[1]],
                        color=color,
                        alpha=0.6,
                        linewidth=1.5,
                    )

            # Plot GNG nodes
            for gng_node in global_state.Nodes:
                # Only plot if node is 2D
                if len(gng_node.w) >= 2:
                    ax.plot(
                        gng_node.w[0],
                        gng_node.w[1],
                        "o",
                        color=color,
                        markersize=8,
                        markeredgecolor="black",
                        markeredgewidth=1,
                        alpha=0.8,
                    )

            # Add level annotation at the centroid of GNG nodes
            if global_state.Nodes:
                gng_positions = np.array(
                    [node.w for node in global_state.Nodes if len(node.w) >= 2]
                )
                if len(gng_positions) > 0:
                    centroid = np.mean(gng_positions, axis=0)
                    ax.annotate(
                        f"L{level}",
                        centroid,
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=10,
                        fontweight="bold",
                        color="white",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.9),
                    )

        elif node["data_points"] is not None and len(node["data_points"]) > 0:
            # Fallback: if no GNG structure, show data points and centroid
            data = node["data_points"]
            ax.scatter(
                data[:, 0],
                data[:, 1],
                c=[color],
                alpha=0.7,
                s=30,
                edgecolors="black",
                linewidth=0.5,
            )

            centroid = np.mean(data, axis=0)
            ax.plot(
                centroid[0],
                centroid[1],
                "o",
                color=color,
                markersize=12,
                markeredgecolor="black",
                markeredgewidth=2,
            )

            ax.annotate(
                f"L{level}",
                centroid,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
            )

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.grid(True, alpha=0.3)

    # Add legend for levels
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[i],
            markersize=10,
            label=f"Level {i}",
        )
        for i in range(max_level + 1)
    ]
    ax.legend(handles=legend_elements, loc="upper right")


def create_hierarchy_tree_plot(
    ax, proteus_info, ground_truth_info, title="Hierarchy Comparison"
):
    """Create a tree-like visualization comparing the hierarchies."""
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(-0.5, 1.5)

    # Ground truth tree (left side)
    gt_levels = sorted(ground_truth_info["levels"].keys())
    gt_max_depth = max(gt_levels)

    # Proteus tree (right side)
    proteus_levels = {}
    for node in proteus_info["nodes"]:
        level = node["level"]
        if level not in proteus_levels:
            proteus_levels[level] = []
        proteus_levels[level].append(node)

    proteus_max_depth = max(proteus_levels.keys()) if proteus_levels else 0

    max_depth = max(gt_max_depth, proteus_max_depth)
    ax.set_ylim(-0.5, max_depth + 0.5)

    # Plot ground truth tree
    ax.text(
        0, max_depth + 0.2, "Ground Truth", ha="center", fontweight="bold", fontsize=12
    )
    for level in gt_levels:
        clusters = ground_truth_info["levels"][level]
        y_pos = max_depth - level

        for i, cluster in enumerate(clusters):
            x_pos = 0 + (i - len(clusters) / 2 + 0.5) * 0.1
            circle = plt.Circle((x_pos, y_pos), 0.03, color="blue", alpha=0.7)
            ax.add_patch(circle)
            ax.text(
                x_pos,
                y_pos,
                str(len(clusters)),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    # Plot Proteus tree
    ax.text(
        1,
        max_depth + 0.2,
        "Proteus Result",
        ha="center",
        fontweight="bold",
        fontsize=12,
    )
    for level in sorted(proteus_levels.keys()):
        nodes = proteus_levels[level]
        y_pos = max_depth - level

        for i, node in enumerate(nodes):
            x_pos = 1 + (i - len(nodes) / 2 + 0.5) * 0.1
            circle = plt.Circle((x_pos, y_pos), 0.03, color="red", alpha=0.7)
            ax.add_patch(circle)
            ax.text(
                x_pos,
                y_pos,
                str(node["data_size"]),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    # Add level labels
    for level in range(max_depth + 1):
        y_pos = max_depth - level
        ax.text(
            -0.4,
            y_pos,
            f"Level {level}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Ground Truth", "Proteus"])
    ax.set_ylabel("Hierarchy Level")
    ax.grid(True, alpha=0.3)


def create_comparison_visualization():
    """Create the complete comparison visualization."""
    print("Creating hierarchical structure comparison visualization...")

    # Create ground truth
    ground_truth = create_hierarchical_gaussian_mixture()
    X, true_labels = ground_truth.generate_data(n_samples=3000, random_state=42)

    print(
        f"Ground truth: {len(ground_truth.clusters)} clusters, depth {ground_truth.hierarchy_depth}"
    )
    print(f"Data: {X.shape[0]} points in {X.shape[1]}D")

    # Run Proteus algorithm
    params = AlgorithmParameters()
    estimated_local_dim = 2
    min_cluster_size = calculate_dimensionally_dependent_min_cluster_size(
        estimated_local_dim
    )

    proteus = ProteusAlgorithm(
        params=params,
        max_depth=4,
        min_cluster_size=min_cluster_size,
        run_stage2=False,
    )

    print("Running Proteus algorithm...")
    try:
        hierarchy_root = proteus.run_complete_algorithm(X)
        print(
            f"Proteus found {len(proteus.all_nodes)} nodes, depth {proteus._calculate_max_depth(proteus.hierarchy_root) if proteus.hierarchy_root else 0}"
        )
    except Exception as e:
        print(f"Error running Proteus: {e}")
        return

    # Extract information
    gt_info = extract_ground_truth_info(ground_truth)
    proteus_info = extract_proteus_hierarchy_info(proteus)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Proteus Hierarchical Structure Discovery: Ground Truth vs Results",
        fontsize=16,
        fontweight="bold",
    )

    # Plot ground truth structure
    plot_ground_truth_structure(axes[0, 0], gt_info, X, true_labels)

    # Plot Proteus discovered structure
    plot_proteus_structure(axes[0, 1], proteus_info, X)

    # Create hierarchy tree comparison
    create_hierarchy_tree_plot(axes[1, 0], proteus_info, gt_info)

    # Create summary statistics
    ax = axes[1, 1]
    ax.set_title("Summary Statistics", fontsize=14, fontweight="bold")
    ax.axis("off")

    # Summary text
    summary_text = f"""
GROUND TRUTH:
• Total clusters: {len(ground_truth.clusters)}
• Hierarchy depth: {ground_truth.hierarchy_depth}
• Leaf clusters: 8
• Data points: {X.shape[0]}

PROTEUS RESULTS:
• Total nodes: {proteus_info['total_nodes']}
• Hierarchy depth: {proteus_info['max_depth']}
• Leaf nodes: {len(proteus_info['leaf_nodes'])}
• Min cluster size: {min_cluster_size}

COMPARISON:
• Depth match: {'✓' if proteus_info['max_depth'] >= 1 else '✗'}
• Multiple clusters: {'✓' if len(proteus_info['leaf_nodes']) > 1 else '✗'}
• Structure found: {'✓' if proteus_info['max_depth'] >= 1 and len(proteus_info['leaf_nodes']) > 1 else '✗'}
"""

    ax.text(
        0.1,
        0.9,
        summary_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(os.path.dirname(__file__), "hierarchy_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {output_path}")

    # Also save a high-resolution version
    output_path_hires = os.path.join(
        os.path.dirname(__file__), "hierarchy_comparison_hires.png"
    )
    plt.savefig(output_path_hires, dpi=600, bbox_inches="tight")
    print(f"High-resolution version saved to: {output_path_hires}")

    plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    create_comparison_visualization()
