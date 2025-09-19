#!/usr/bin/env python3
"""Investigatory diagnostics for Proteus Stage 1 on a single Gaussian."""

from __future__ import annotations

import csv
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

from src.proteus.stage1 import ProteusStage1

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def _mean_min_distance(
    data: np.ndarray, centres: np.ndarray
) -> Tuple[float, np.ndarray]:
    if centres.size == 0:
        return float("inf"), np.full(data.shape[0], np.inf)
    deltas = data[:, None, :] - centres[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    min_distances = distances.min(axis=1)
    return float(min_distances.mean()), min_distances


def _run_simple_kmeans(
    data: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise ValueError("k must be positive")
    n_samples = data.shape[0]
    if n_samples < k:
        raise ValueError("k cannot exceed the number of samples")
    indices = rng.choice(n_samples, size=k, replace=False)
    centres = data[indices].copy()
    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        deltas = data[:, None, :] - centres[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for idx in range(k):
            mask = labels == idx
            if not np.any(mask):
                centres[idx] = data[rng.integers(0, n_samples)]
                continue
            centres[idx] = data[mask].mean(axis=0)
    return centres, labels


def _moment_cover_baseline(
    data: np.ndarray,
    rng: np.random.Generator,
    *,
    tolerance: float = 0.05,
    max_k: int = 6,
) -> Optional[int]:
    true_mean = data.mean(axis=0)
    true_cov = np.cov(data, rowvar=False)
    true_cov_norm = float(np.linalg.norm(true_cov)) if true_cov.size else 0.0
    scale = float(np.sqrt(np.trace(true_cov))) if true_cov.size else 1.0

    for k in range(1, max_k + 1):
        centres, labels = _run_simple_kmeans(data, k, rng)
        weights = np.bincount(labels, minlength=k).astype(float)
        weight_total = weights.sum()
        if weight_total == 0.0:
            continue
        approx_mean = (weights[:, None] * centres).sum(axis=0) / weight_total
        centred = centres - approx_mean
        approx_cov = np.zeros_like(true_cov)
        for weight, diff in zip(weights, centred):
            approx_cov += weight * np.outer(diff, diff)
        approx_cov /= weight_total

        mean_error = float(np.linalg.norm(approx_mean - true_mean) / (scale + 1e-8))
        if true_cov_norm == 0.0:
            cov_error = 0.0
        else:
            cov_error = float(
                np.linalg.norm(approx_cov - true_cov) / (true_cov_norm + 1e-12)
            )
        if mean_error <= tolerance and cov_error <= tolerance:
            return k
    return None


def _geometric_cover_lower_bound(sigma: float, r_target: float) -> float:
    if r_target <= 0.0:
        return float("inf")
    packing_efficiency = 0.9069
    ratio = (sigma / r_target) ** 2
    return (4.0 / packing_efficiency) * ratio


def test_gaussian_node_efficiency() -> None:
    sigma = 1.0
    sample_sizes = (2_000, 5_000)
    parameter_configs = [
        {
            "name": "default",
            "k_neighbors": 8,
            "min_link_strength": 4.0,
            "prune_after": 25,
            "alpha_scale": 1.0,
        },
        {
            "name": "tight-links",
            "k_neighbors": 6,
            "min_link_strength": 2.0,
            "prune_after": 15,
            "alpha_scale": 1.0,
        },
        {
            "name": "loose-links",
            "k_neighbors": 12,
            "min_link_strength": 6.0,
            "prune_after": 35,
            "alpha_scale": 1.0,
        },
        {
            "name": "slow-alpha",
            "k_neighbors": 8,
            "min_link_strength": 4.0,
            "prune_after": 25,
            "alpha_scale": 0.5,
        },
    ]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = []
    representative_data: Optional[np.ndarray] = None
    representative_nodes: Optional[np.ndarray] = None
    representative_title: str = ""

    for n_samples in sample_sizes:
        data_rng = np.random.default_rng(42 + n_samples)
        data = data_rng.normal(loc=0.0, scale=sigma, size=(n_samples, 2))
        scale_mean = float(np.linalg.norm(data, axis=1).mean())
        for config_idx, config in enumerate(parameter_configs):
            k_neighbors = config["k_neighbors"]
            default_alpha = float(np.log(2.0) / max(k_neighbors, 1))
            alpha = default_alpha * config.get("alpha_scale", 1.0)
            stage_rng = np.random.default_rng(10_000 + n_samples + config_idx)
            stage1 = ProteusStage1(
                k_neighbors=k_neighbors,
                alpha=alpha,
                prune_after=config["prune_after"],
                min_link_strength=config["min_link_strength"],
                max_epochs=25,
                samples_per_epoch=None,
                rng=stage_rng,
            )
            stage1.fit(data, tau=sigma**2, d_subspace=2)
            centres = stage1.get_node_positions()
            node_count = centres.shape[0]
            r_bar, _ = _mean_min_distance(data, centres)
            r_ratio = r_bar / max(scale_mean, 1e-8)
            moment_rng = np.random.default_rng(20_000 + n_samples + config_idx)
            k_moment = _moment_cover_baseline(
                data, moment_rng, tolerance=0.05, max_k=6
            )
            geom_lower = _geometric_cover_lower_bound(sigma, max(r_bar, 1e-8))
            efficiency_ratio = (
                float(node_count / geom_lower)
                if np.isfinite(geom_lower) and geom_lower > 0.0
                else float("inf")
            )

            results.append(
                {
                    "sample_size": n_samples,
                    "config": config["name"],
                    "nodes": int(node_count),
                    "r_bar": float(r_bar),
                    "scale": scale_mean,
                    "r_ratio": float(r_ratio),
                    "moment_k": k_moment,
                    "geom_lower": float(geom_lower),
                    "efficiency": efficiency_ratio,
                    "k_neighbors": k_neighbors,
                    "alpha": float(alpha),
                    "min_link_strength": config["min_link_strength"],
                    "prune_after": config["prune_after"],
                }
            )

            if (
                representative_data is None
                and n_samples == max(sample_sizes)
                and config["name"] == "default"
            ):
                representative_data = data.copy()
                representative_nodes = centres.copy()
                representative_title = f"Stage 1 cover (N={n_samples})"

    assert results, "No efficiency runs recorded"

    for row in results:
        assert row["nodes"] > 0
        assert row["r_ratio"] < 0.2

    csv_path = ARTIFACTS_DIR / "gaussian_node_efficiency.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_size",
                "config",
                "nodes",
                "r_bar",
                "scale",
                "r_ratio",
                "moment_k",
                "geom_lower",
                "efficiency",
                "k_neighbors",
                "alpha",
                "min_link_strength",
                "prune_after",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row["sample_size"],
                    row["config"],
                    row["nodes"],
                    f"{row['r_bar']:.6f}",
                    f"{row['scale']:.6f}",
                    f"{row['r_ratio']:.6f}",
                    "" if row["moment_k"] is None else int(row["moment_k"]),
                    f"{row['geom_lower']:.3f}",
                    f"{row['efficiency']:.3f}",
                    row["k_neighbors"],
                    f"{row['alpha']:.4f}",
                    row["min_link_strength"],
                    row["prune_after"],
                ]
            )

    header = (
        f"{'N':>6} {'config':>12} {'K':>5} {'r/S':>7} {'k_mom':>7} "
        f"{'N_min':>8} {'K/N_min':>8} {'k':>4} {'alpha':>7} {'minL':>6} {'prune':>6}"
    )
    lines = [header, "-" * len(header)]
    for row in results:
        moment_display = (
            "--" if row["moment_k"] is None else f"{int(row['moment_k']):d}"
        )
        lines.append(
            f"{row['sample_size']:6d} {row['config']:>12} {row['nodes']:5d} "
            f"{row['r_ratio']:>7.4f} {moment_display:>7} "
            f"{row['geom_lower']:>8.2f} {row['efficiency']:>8.2f} "
            f"{row['k_neighbors']:4d} {row['alpha']:>7.4f} "
            f"{row['min_link_strength']:6.2f} {row['prune_after']:6d}"
        )

    log_path = ARTIFACTS_DIR / "gaussian_node_efficiency.txt"
    log_content = "Gaussian node efficiency summary\n" + "\n".join(lines) + "\n"
    log_path.write_text(log_content)

    print(log_content.rstrip())

    if representative_data is None:
        representative_data = data.copy()
        representative_nodes = centres.copy()
        representative_title = f"Stage 1 cover (N={n_samples})"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scatter_ax, summary_ax = axes
    scatter_ax.scatter(
        representative_data[:, 0],
        representative_data[:, 1],
        c="lightgray",
        s=10,
        alpha=0.18,
        linewidths=0.0,
    )
    if representative_nodes is not None and representative_nodes.size:
        scatter_ax.scatter(
            representative_nodes[:, 0],
            representative_nodes[:, 1],
            c="crimson",
            edgecolors="black",
            linewidths=0.5,
            s=36,
            alpha=0.85,
        )
    title_text = representative_title or "Stage 1 cover"
    scatter_ax.set_title(title_text, fontsize=12, fontweight="bold")
    scatter_ax.set_xlabel("X1")
    scatter_ax.set_ylabel("X2")
    scatter_ax.axis("equal")
    scatter_ax.grid(True, alpha=0.2)

    summary_ax.axis("off")
    summary_text = "Gaussian node efficiency\n\n" + "\n".join(lines)
    summary_ax.text(
        0.0,
        1.0,
        dedent(summary_text),
        transform=summary_ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        va="top",
    )

    efficiency_figure = ARTIFACTS_DIR / "gaussian_node_efficiency.png"
    fig.tight_layout()
    fig.savefig(efficiency_figure, dpi=300, bbox_inches="tight")
    plt.close(fig)

    assert efficiency_figure.exists()
    assert log_path.exists()
    assert csv_path.exists()
