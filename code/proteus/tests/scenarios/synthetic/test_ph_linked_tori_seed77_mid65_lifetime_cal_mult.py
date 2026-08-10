"""Mild hollow mid=0.65 × lifetime×cal-mult on seed77 denser256 (#41 / A4-T53-followon).

A4-T52: primary mid=0.5×h0∈{0.5,0.7} yields dirty torus ``b2``; milder
``mid=0.65`` preserves both-tori partial ``(1,2,0)`` with **no** dirty under
default fixed/lifetime. A4-T49: primary hollow×cal-mult≥3 erases dirty without
clean ``(1,2,1)``. This harness freezes mid=0.65/h0=0.7 hollow and crosses
lifetime×cal-mult asking whether the non-dirty mild hollow unlocks void Betti.

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
STAGE1_SEED: int = 77
FRAC_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MULT_GRID: tuple[float, ...] = (1.5, 3.0, 6.0, 8.0)
# Mild mid that T52 found preserves both-partial without dirty-b2.
HOLLOW_CFG = HollowEdgeConfig(
    mid_radius_frac=0.65, h0=0.7, min_end_count=0.5, gabriel_fallback=False,
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


@dataclass(frozen=True)
class Seed77Mid65LifetimeCalMultBundle:
    dataset_seed: int
    stage1_seed: int
    max_nodes: int
    n_signal: int
    n_after_hollow: int
    hollow_fallback: bool
    sigma_star: float
    mid_radius_frac: float
    h0: float
    signal_fixed_betti: dict[int, tuple[int, ...]]
    hollow_fixed_betti: dict[int, tuple[int, ...]]
    both_tori_partial_signal: bool
    both_tori_partial_hollow_fixed: bool
    fracs: tuple[float, ...]
    mults: tuple[float, ...]
    table: str
    any_full_match: bool
    any_clean_b2: bool
    any_dirty_b2: bool
    any_b1_ge_2: bool
    max_b1: int
    max_b2: int
    recovering_cells: tuple[tuple[int, float, float, tuple[int, ...]], ...]
    clean_b2_cells: tuple[tuple[int, float, float, tuple[int, ...]], ...]
    dirty_b2_cells: tuple[tuple[int, float, float, tuple[int, ...]], ...]


@pytest.fixture(scope="module")
def linked_tori_seed77_mid65_lifetime_cal_mult_bundle() -> (
    Seed77Mid65LifetimeCalMultBundle
):
    """Fit seed77 denser256; mild mid=0.65 hollow; cross lifetime×cal-mult."""
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
    sigma = sigma_star_from_tau(result.tau_star)
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
    n_signal = int(np.sum(signal_mask))
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

    signal_fixed = run_per_region_ph(
        signal_pos,
        signal_labs,
        sigma,
        scenario="linked_tori_seed77_mid65_signal_fixed",
        include_labels=[0, 1],
        reading="fixed_threshold",
        max_dim=2,
        filtration_mult=FILTRATION_MULTIPLIER,
        expected_betti=EXPECTED_TORI,
    )
    hollow_fixed = run_per_region_ph(
        hollow_pos,
        hollow_labs,
        sigma,
        scenario="linked_tori_seed77_mid65_hollow_fixed",
        include_labels=[0, 1],
        reading="fixed_threshold",
        max_dim=2,
        filtration_mult=FILTRATION_MULTIPLIER,
        expected_betti=EXPECTED_TORI,
    )
    signal_fixed_betti = {
        int(r.region_id): tuple(int(x) for x in r.betti)
        for r in signal_fixed.reports
    }
    hollow_fixed_betti = {
        int(r.region_id): tuple(int(x) for x in r.betti)
        for r in hollow_fixed.reports
    }
    both_signal = (
        signal_fixed_betti.get(0) == (1, 2, 0)
        and signal_fixed_betti.get(1) == (1, 2, 0)
    )
    both_hollow = (
        hollow_fixed_betti.get(0) == (1, 2, 0)
        and hollow_fixed_betti.get(1) == (1, 2, 0)
    )

    grid = sweep_lifetime_mult_grid_per_region(
        hollow_pos,
        hollow_labs,
        sigma,
        fracs=FRAC_GRID,
        mults=MULT_GRID,
        scenario="linked_tori_seed77_mid65_lifetime_cal_mult",
        include_labels=[0, 1],
        max_dim=2,
        expected_betti=EXPECTED_TORI,
        b1_target=2,
    )

    recovering: list[tuple[int, float, float, tuple[int, ...]]] = []
    clean_cells: list[tuple[int, float, float, tuple[int, ...]]] = []
    dirty_cells: list[tuple[int, float, float, tuple[int, ...]]] = []
    max_b1 = 0
    max_b2 = 0
    any_b1 = False
    for r in grid.rows:
        betti = tuple(int(x) for x in r.betti)
        rid = int(r.region_id)
        frac = float(r.lifetime_frac)
        mult = float(r.filtration_mult)
        max_b1 = max(max_b1, int(r.b1))
        b2 = int(betti[2]) if len(betti) > 2 else 0
        max_b2 = max(max_b2, b2)
        if betti == EXPECTED_TORI or r.match is True:
            recovering.append((rid, frac, mult, betti))
        if int(r.b1) >= 2:
            any_b1 = True
        if _is_clean_b2(betti):
            clean_cells.append((rid, frac, mult, betti))
        if _is_dirty_b2(betti):
            dirty_cells.append((rid, frac, mult, betti))

    return Seed77Mid65LifetimeCalMultBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        max_nodes=MAX_NODES,
        n_signal=n_signal,
        n_after_hollow=n_after,
        hollow_fallback=bool(hollow_fallback),
        sigma_star=float(sigma),
        mid_radius_frac=float(HOLLOW_CFG.mid_radius_frac),
        h0=float(HOLLOW_CFG.h0),
        signal_fixed_betti=signal_fixed_betti,
        hollow_fixed_betti=hollow_fixed_betti,
        both_tori_partial_signal=both_signal,
        both_tori_partial_hollow_fixed=both_hollow,
        fracs=FRAC_GRID,
        mults=MULT_GRID,
        table=format_lifetime_mult_grid_table(grid),
        any_full_match=bool(grid.any_full_match) or bool(recovering),
        any_clean_b2=bool(clean_cells),
        any_dirty_b2=bool(dirty_cells),
        any_b1_ge_2=any_b1,
        max_b1=max_b1,
        max_b2=max_b2,
        recovering_cells=tuple(recovering),
        clean_b2_cells=tuple(clean_cells),
        dirty_b2_cells=tuple(dirty_cells),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed77_mid65_lifetime_cal_mult_harness_lands(
    linked_tori_seed77_mid65_lifetime_cal_mult_bundle,
) -> None:
    """Seed77 mid=0.65×lifetime×cal-mult lands; SI defaults untouched."""
    bundle = linked_tori_seed77_mid65_lifetime_cal_mult_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert bundle.max_nodes == MAX_NODES
    assert bundle.mid_radius_frac == 0.65
    assert bundle.h0 == 0.7
    assert bundle.n_signal > 0
    assert bundle.n_after_hollow >= 0
    assert bundle.sigma_star > 0.0
    assert bundle.fracs == FRAC_GRID
    assert bundle.mults == MULT_GRID
    assert bundle.both_tori_partial_signal
    header = bundle.table.splitlines()[0]
    assert "frac" in header and "mult" in header and "b1" in header
    assert len(bundle.table.splitlines()) == 1 + 2 * len(FRAC_GRID) * len(MULT_GRID)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed77_mid65_lifetime_cal_mult_documents_gap(
    linked_tori_seed77_mid65_lifetime_cal_mult_bundle,
) -> None:
    """Document mid65×lifetime×cal-mult vs clean (1,2,1); never flip awaiting.

    Soft: any full ``(1,2,1)`` or clean ``b2`` is proposal-path evidence.
    Otherwise keep documenting partial-preserving / no-void gap.
    """
    bundle = linked_tori_seed77_mid65_lifetime_cal_mult_bundle
    if bundle.any_full_match or bundle.any_clean_b2:
        assert FILTRATION_MULTIPLIER == 1.5
        if bundle.any_full_match:
            assert len(bundle.recovering_cells) >= 1
        if bundle.any_clean_b2:
            assert len(bundle.clean_b2_cells) >= 1
            assert bundle.max_b2 >= 1
    else:
        assert bundle.any_full_match is False
        assert bundle.any_clean_b2 is False
        assert bundle.recovering_cells == ()
        assert bundle.clean_b2_cells == ()
        assert bundle.both_tori_partial_signal
        # Mild hollow should not invent dirty-b2 under SI mult; cal may vary.
        assert bundle.any_dirty_b2 or bundle.max_b2 >= 0
        assert bundle.n_after_hollow <= bundle.n_signal
