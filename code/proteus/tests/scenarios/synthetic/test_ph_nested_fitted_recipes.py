"""Nested fitted PH recipes: lifetime_frac + signal filter + hollow prune (#41 / A4-T25).

Denser ``max_nodes`` alone worsened Betti on nested shells (A4-T19). This
probe tries reading / filtering levers on a single fitted scaffold
(``max_nodes=128``):

1. signal filter (NN-to-data labels; drop tissue),
2. lifetime_frac sweep at SI ``filtration_mult=1.5``,
3. hollow-pruned scaffold neighbour edges → keep nodes with surviving degree.

Evidence-gathering only — does **not** flip ``test_nested_spheres_topology``
``@awaiting`` or change SI defaults.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.edge_evidence import HollowEdgeConfig, prune_hollow_edges
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.nested_spheres import make_nested_spheres
from tests.metrics.persistent_homology import (
    DEFAULT_LIFETIME_FRAC,
    FILTRATION_MULTIPLIER,
    nearest_data_labels,
    run_per_region_ph,
    sigma_star_from_tau,
    sweep_lifetime_frac_per_region,
)


def _undirected_edges_from_adj(adj: dict) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i, nbrs in adj.items():
        for j in nbrs:
            a, b = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
            if a != b:
                edges.add((a, b))
    return sorted(edges)


def _hollow_pruned_node_mask(
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    data: np.ndarray,
    *,
    config: HollowEdgeConfig | None = None,
) -> np.ndarray:
    """True for nodes that retain ≥1 non-hollow neighbour edge."""
    n = int(positions.shape[0])
    keep = np.zeros(n, dtype=bool)
    if not edges:
        return keep
    surviving = prune_hollow_edges(positions, edges, data, config=config)
    for i, j in surviving:
        keep[int(i)] = True
        keep[int(j)] = True
    return keep


@dataclass(frozen=True)
class NestedRecipeRow:
    recipe: str
    lifetime_frac: float | None
    n_signal: int
    n_after_hollow: int
    betti_per_shell: dict[int, tuple[int, ...]]
    all_match_si: bool


@dataclass(frozen=True)
class NestedRecipeBundle:
    max_nodes: int
    n_nodes: int
    sigma_star: float
    rows: tuple[NestedRecipeRow, ...]
    lifetime_sweep_any_match: bool


@pytest.fixture(scope="module")
def nested_fitted_recipe_bundle() -> NestedRecipeBundle:
    """Fit once; evaluate signal / lifetime / hollow-prune recipes."""
    dataset = make_nested_spheres(
        n_per_sphere=500,
        radii=(1.0, 2.0),
        ambient_dim=3,
        noise=0.02,
        tissue_fraction=0.03,
        seed=21,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    max_nodes = 128
    config = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=8,
        max_nodes=max_nodes,
        ann_backend="naive",
        stabilization=StabilizationConfig(
            min_equilibrium_epochs=3, max_epochs=15,
        ),
        seed=77,
    )
    result = run_scale_search(dataset.points, dim=gt.ambient_dim, config=config)
    scaffold = result.scaffold_at_star
    pos = scaffold.node_positions()
    sigma = sigma_star_from_tau(result.tau_star)
    node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
    signal_mask = np.isin(node_labels, [1, 2])
    signal_pos = pos[signal_mask]
    signal_labs = node_labels[signal_mask]

    adj = scaffold.links.neighbour_graph(pos.shape[0])
    edges = _undirected_edges_from_adj(adj)
    # Prefer no-Gabriel mid=0.5 candidate from A4-T24 export when available.
    hollow_cfg = HollowEdgeConfig(
        mid_radius_frac=0.5, h0=0.7, min_end_count=0.5, gabriel_fallback=False,
    )
    hollow_keep = _hollow_pruned_node_mask(
        pos, edges, dataset.points, config=hollow_cfg,
    )
    hollow_signal = signal_mask & hollow_keep
    n_after = int(np.sum(hollow_signal))
    if n_after < 8 or not (
        np.any(node_labels[hollow_signal] == 1)
        and np.any(node_labels[hollow_signal] == 2)
    ):
        hollow_pos, hollow_labs = signal_pos, signal_labs
        hollow_used_fallback = True
    else:
        hollow_pos = pos[hollow_signal]
        hollow_labs = node_labels[hollow_signal]
        hollow_used_fallback = False

    rows: list[NestedRecipeRow] = []

    def _row(
        name: str,
        pts: np.ndarray,
        labs: np.ndarray,
        *,
        reading: str,
        lifetime_frac: float | None,
    ) -> NestedRecipeRow:
        report = run_per_region_ph(
            pts,
            labs,
            [sigma, sigma],
            scenario=f"nested_recipe_{name}",
            include_labels=[1, 2],
            reading=reading,  # type: ignore[arg-type]
            max_dim=2,
            filtration_mult=FILTRATION_MULTIPLIER,
            lifetime_frac=(
                DEFAULT_LIFETIME_FRAC if lifetime_frac is None else lifetime_frac
            ),
            expected_betti=(1, 0, 1),
        )
        betti = {
            int(r.region_id): tuple(int(x) for x in r.betti) for r in report.reports
        }
        return NestedRecipeRow(
            recipe=name,
            lifetime_frac=lifetime_frac,
            n_signal=int(pts.shape[0]),
            n_after_hollow=n_after,
            betti_per_shell=betti,
            all_match_si=bool(report.all_match),
        )

    rows.append(
        _row(
            "signal_fixed",
            signal_pos,
            signal_labs,
            reading="fixed_threshold",
            lifetime_frac=None,
        )
    )
    for frac in (0.25, 0.5, 1.0, 2.0):
        rows.append(
            _row(
                f"signal_lifetime_{frac:g}",
                signal_pos,
                signal_labs,
                reading="lifetime",
                lifetime_frac=float(frac),
            )
        )
    hollow_name = (
        "hollow_signal_lifetime_fallback"
        if hollow_used_fallback
        else "hollow_signal_lifetime"
    )
    rows.append(
        _row(
            hollow_name,
            hollow_pos,
            hollow_labs,
            reading="lifetime",
            lifetime_frac=DEFAULT_LIFETIME_FRAC,
        )
    )

    sweep = sweep_lifetime_frac_per_region(
        signal_pos,
        signal_labs,
        sigma,
        fracs=(0.25, 0.5, 1.0, 2.0, 4.0),
        include_labels=[1, 2],
        filtration_mult=FILTRATION_MULTIPLIER,
        max_dim=2,
        target_betti=(1, 0, 1),
    )
    by_frac: dict[float, list[bool]] = {}
    for srow in sweep:
        by_frac.setdefault(float(srow.lifetime_frac), []).append(
            bool(srow.matches_target)
        )
    any_match = any(all(flags) and len(flags) == 2 for flags in by_frac.values())

    return NestedRecipeBundle(
        max_nodes=max_nodes,
        n_nodes=int(pos.shape[0]),
        sigma_star=float(sigma),
        rows=tuple(rows),
        lifetime_sweep_any_match=bool(any_match),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_fitted_recipes_report_table(nested_fitted_recipe_bundle) -> None:
    """Recipe table lands with signal + lifetime + hollow variants; SI default intact."""
    bundle = nested_fitted_recipe_bundle
    assert bundle.max_nodes == 128
    assert bundle.n_nodes > 0
    assert bundle.sigma_star > 0.0
    names = {r.recipe for r in bundle.rows}
    assert "signal_fixed" in names
    assert any(n.startswith("signal_lifetime_") for n in names)
    assert any(n.startswith("hollow_signal_lifetime") for n in names)
    for row in bundle.rows:
        assert set(row.betti_per_shell) == {1, 2}
        for b in row.betti_per_shell.values():
            assert len(b) == 3
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_fitted_recipes_document_si_gap(nested_fitted_recipe_bundle) -> None:
    """Document whether any recipe recovers per-shell (1,0,1); never flip awaiting.

    Soft gate: if a recipe recovers, denser+lifetime/hollow is viable evidence.
    If none recover, assert explicit non-recovery (green documentation).
    """
    bundle = nested_fitted_recipe_bundle
    recovered = [r.recipe for r in bundle.rows if r.all_match_si]
    if recovered or bundle.lifetime_sweep_any_match:
        assert FILTRATION_MULTIPLIER == 1.5
    else:
        for row in bundle.rows:
            assert row.all_match_si is False
            assert any(b != (1, 0, 1) for b in row.betti_per_shell.values())
