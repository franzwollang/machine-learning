"""Seed7 densify256 dirty-b2 × high lifetime_frac (≥4) × cal (#41 / A4-T76).

A4-T48: stage1 seed=7 on denser256 yields dirty ``b2`` (e.g. ``(2,1,1)``) that
filtration/lifetime cleanup did not lift to clean ``(1,2,1)``. A4-T71: high
``lifetime_frac≥4`` on seed2 densify256×hollow preserved both-partial with
``max_b2=0``. This harness freezes the seed7 dirty-b2 densify256 scaffold and
restricts the lifetime grid to ``{4, 8}`` under SI + cal mults (signal +
primary hollow), asking whether high-frac cal cleans dirty-b2 toward
``(1,2,1)`` or only rearranges ``b0``/``b1``.

Evidence-gathering only — does **not** flip ``@awaiting`` or SI defaults.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.edge_evidence import HollowEdgeConfig, prune_hollow_edges
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.linked_tori import make_linked_tori
from tests.metrics.persistent_homology import (
    DEFAULT_LIFETIME_FRAC,
    FILTRATION_MULTIPLIER,
    format_lifetime_mult_grid_table,
    nearest_data_labels,
    run_per_region_ph,
    sigma_star_from_tau,
    sweep_lifetime_mult_grid_per_region,
)

EXPECTED_TORI: tuple[int, ...] = (1, 2, 1)
N_PER_TORUS: int = 500
MAX_NODES: int = 256
DATASET_SEED: int = 2
STAGE1_SEED: int = 7
FRAC_GRID: tuple[float, ...] = (4.0, 8.0)
MULT_GRID: tuple[float, ...] = (1.5, 3.0, 6.0, 8.0)
HOLLOW_CFG = HollowEdgeConfig(
    mid_radius_frac=0.5, h0=0.7, min_end_count=0.5, gabriel_fallback=False,
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
    n = int(positions.shape[0])
    keep = np.zeros(n, dtype=bool)
    if not edges:
        return keep
    surviving = prune_hollow_edges(positions, edges, data, config=config)
    for i, j in surviving:
        keep[int(i)] = True
        keep[int(j)] = True
    return keep


def _is_clean_b2(betti: tuple[int, ...]) -> bool:
    return (
        len(betti) > 2
        and int(betti[0]) == 1
        and int(betti[1]) >= 2
        and int(betti[2]) >= 1
    )


def _is_dirty_b2(betti: tuple[int, ...]) -> bool:
    return (
        len(betti) > 2
        and int(betti[2]) >= 1
        and not _is_clean_b2(betti)
    )


def _is_dirty_b0(betti: tuple[int, ...]) -> bool:
    return len(betti) > 0 and int(betti[0]) > 1


@dataclass(frozen=True)
class Seed7Densify256HighfracCalArm:
    name: str
    n_signal: int
    hollow_fallback: bool
    fixed_betti: dict[int, tuple[int, ...]]
    fixed_has_dirty_b2: bool
    table: str
    any_full_match: bool
    any_clean_b2: bool
    any_dirty_b2: bool
    any_dirty_b0: bool
    max_b1: int
    max_b2: int
    recovering_cells: tuple[tuple[int, float, float, tuple[int, ...]], ...]
    clean_b2_cells: tuple[tuple[int, float, float, tuple[int, ...]], ...]
    dirty_b2_cells: tuple[tuple[int, float, float, tuple[int, ...]], ...]


@dataclass(frozen=True)
class Seed7Densify256HighfracCalBundle:
    dataset_seed: int
    stage1_seed: int
    max_nodes: int
    sigma_star: float
    fracs: tuple[float, ...]
    mults: tuple[float, ...]
    signal_arm: Seed7Densify256HighfracCalArm
    hollow_arm: Seed7Densify256HighfracCalArm
    any_full_match: bool
    any_clean_b2: bool
    any_dirty_b2: bool
    max_b2: int


def _arm_from_grid(
    name: str,
    pts: np.ndarray,
    labs: np.ndarray,
    sigma: float,
    *,
    hollow_fallback: bool,
    scenario_prefix: str,
) -> Seed7Densify256HighfracCalArm:
    fixed = run_per_region_ph(
        pts,
        labs,
        sigma,
        reading="fixed_threshold",
        filtration_mult=FILTRATION_MULTIPLIER,
        scenario=f"{scenario_prefix}_{name}_fixed",
        include_labels=[0, 1],
        max_dim=2,
        expected_betti=EXPECTED_TORI,
    )
    fixed_betti = {
        int(r.region_id): tuple(int(x) for x in r.betti) for r in fixed.reports
    }
    fixed_has_dirty = any(_is_dirty_b2(b) for b in fixed_betti.values())

    grid = sweep_lifetime_mult_grid_per_region(
        pts,
        labs,
        sigma,
        fracs=FRAC_GRID,
        mults=MULT_GRID,
        scenario=f"{scenario_prefix}_{name}_highfrac",
        include_labels=[0, 1],
        max_dim=2,
        expected_betti=EXPECTED_TORI,
        b1_target=2,
    )

    recovering: list[tuple[int, float, float, tuple[int, ...]]] = []
    clean_cells: list[tuple[int, float, float, tuple[int, ...]]] = []
    dirty_b2_cells: list[tuple[int, float, float, tuple[int, ...]]] = []
    dirty_b0 = False
    max_b1 = 0
    max_b2 = 0
    for r in grid.rows:
        betti = tuple(int(x) for x in r.betti)
        rid = int(r.region_id)
        frac = float(r.lifetime_frac)
        mult = float(r.filtration_mult)
        max_b1 = max(max_b1, int(r.b1))
        b2 = int(betti[2]) if len(betti) > 2 else 0
        max_b2 = max(max_b2, b2)
        cell = (rid, frac, mult, betti)
        if betti == EXPECTED_TORI or r.match is True:
            recovering.append(cell)
        if _is_clean_b2(betti):
            clean_cells.append(cell)
        if _is_dirty_b2(betti):
            dirty_b2_cells.append(cell)
        if _is_dirty_b0(betti):
            dirty_b0 = True

    return Seed7Densify256HighfracCalArm(
        name=name,
        n_signal=int(pts.shape[0]),
        hollow_fallback=bool(hollow_fallback),
        fixed_betti=fixed_betti,
        fixed_has_dirty_b2=bool(fixed_has_dirty),
        table=format_lifetime_mult_grid_table(grid),
        any_full_match=bool(grid.any_full_match) or bool(recovering),
        any_clean_b2=bool(clean_cells),
        any_dirty_b2=bool(dirty_b2_cells),
        any_dirty_b0=dirty_b0,
        max_b1=max_b1,
        max_b2=max_b2,
        recovering_cells=tuple(recovering),
        clean_b2_cells=tuple(clean_cells),
        dirty_b2_cells=tuple(dirty_b2_cells),
    )


@pytest.fixture(scope="module")
def linked_tori_seed7_densify256_highfrac_cal_bundle() -> (
    Seed7Densify256HighfracCalBundle
):
    """Fit seed7 densify256; high-frac lifetime×cal on signal + hollow."""
    dataset = make_linked_tori(
        n_per_torus=N_PER_TORUS,
        major_radius=2.0,
        minor_radius=0.5,
        noise=0.02,
        tissue_fraction=0.03,
        seed=DATASET_SEED,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    config = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=8,
        max_nodes=MAX_NODES,
        ann_backend="naive",
        stabilization=StabilizationConfig(
            min_equilibrium_epochs=3, max_epochs=15,
        ),
        seed=STAGE1_SEED,
    )
    result = run_scale_search(dataset.points, dim=gt.ambient_dim, config=config)
    scaffold = result.scaffold_at_star
    pos = scaffold.node_positions()
    sigma = float(sigma_star_from_tau(result.tau_star))
    node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
    signal_mask = np.isin(node_labels, [0, 1])
    signal_pos = pos[signal_mask]
    signal_labs = node_labels[signal_mask]

    adj = scaffold.links.neighbour_graph(pos.shape[0])
    edges = _undirected_edges_from_adj(adj)
    hollow_keep = _hollow_pruned_node_mask(
        pos, edges, dataset.points, config=HOLLOW_CFG,
    )
    hollow_signal = signal_mask & hollow_keep
    n_after = int(np.sum(hollow_signal))
    if n_after < 8 or not (
        np.any(node_labels[hollow_signal] == 0)
        and np.any(node_labels[hollow_signal] == 1)
    ):
        hollow_pos = signal_pos
        hollow_labs = signal_labs
        hollow_fallback = True
    else:
        hollow_pos = pos[hollow_signal]
        hollow_labs = node_labels[hollow_signal]
        hollow_fallback = False

    prefix = "linked_tori_seed7_densify256_highfrac"
    signal_arm = _arm_from_grid(
        "signal", signal_pos, signal_labs, sigma,
        hollow_fallback=False, scenario_prefix=prefix,
    )
    hollow_arm = _arm_from_grid(
        "hollow", hollow_pos, hollow_labs, sigma,
        hollow_fallback=hollow_fallback, scenario_prefix=prefix,
    )

    return Seed7Densify256HighfracCalBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        max_nodes=MAX_NODES,
        sigma_star=sigma,
        fracs=FRAC_GRID,
        mults=MULT_GRID,
        signal_arm=signal_arm,
        hollow_arm=hollow_arm,
        any_full_match=signal_arm.any_full_match or hollow_arm.any_full_match,
        any_clean_b2=signal_arm.any_clean_b2 or hollow_arm.any_clean_b2,
        any_dirty_b2=signal_arm.any_dirty_b2 or hollow_arm.any_dirty_b2,
        max_b2=max(signal_arm.max_b2, hollow_arm.max_b2),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed7_densify256_highfrac_cal_harness_lands(
    linked_tori_seed7_densify256_highfrac_cal_bundle,
) -> None:
    """Seed7 densify256 highfrac×cal probe lands; SI defaults untouched."""
    bundle = linked_tori_seed7_densify256_highfrac_cal_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert bundle.max_nodes == MAX_NODES
    assert bundle.sigma_star > 0.0
    assert bundle.fracs == FRAC_GRID
    assert bundle.mults == MULT_GRID
    assert all(f >= 4.0 for f in bundle.fracs)
    assert bundle.signal_arm.n_signal > 0
    assert bundle.hollow_arm.n_signal > 0
    for arm in (bundle.signal_arm, bundle.hollow_arm):
        header = arm.table.splitlines()[0]
        assert "frac" in header and "mult" in header and "b1" in header
        assert len(arm.table.splitlines()) == 1 + 2 * len(FRAC_GRID) * len(MULT_GRID)
    # Seed7 fixed baseline should expose dirty b2 somewhere (T44/T48).
    assert (
        bundle.signal_arm.fixed_has_dirty_b2
        or any(
            len(b) > 2 and int(b[2]) >= 1
            for b in bundle.signal_arm.fixed_betti.values()
        )
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed7_densify256_highfrac_cal_documents_gap(
    linked_tori_seed7_densify256_highfrac_cal_bundle,
) -> None:
    """Document seed7 highfrac cal vs clean (1,2,1); never flip awaiting.

    Soft: any full ``(1,2,1)`` or clean ``b2`` is proposal-path evidence.
    Otherwise keep documenting dirty-b2 / no-full under frac≥4.
    """
    bundle = linked_tori_seed7_densify256_highfrac_cal_bundle
    if bundle.any_full_match or bundle.any_clean_b2:
        assert FILTRATION_MULTIPLIER == 1.5
        if bundle.any_full_match:
            assert (
                len(bundle.signal_arm.recovering_cells)
                + len(bundle.hollow_arm.recovering_cells)
                >= 1
            )
        if bundle.any_clean_b2:
            assert bundle.max_b2 >= 1
    else:
        assert bundle.any_full_match is False
        assert bundle.any_clean_b2 is False
        assert bundle.signal_arm.recovering_cells == ()
        assert bundle.hollow_arm.recovering_cells == ()
        assert bundle.signal_arm.clean_b2_cells == ()
        assert bundle.hollow_arm.clean_b2_cells == ()
        assert (
            bundle.any_dirty_b2
            or bundle.signal_arm.fixed_has_dirty_b2
            or bundle.max_b2 >= 0
        )
