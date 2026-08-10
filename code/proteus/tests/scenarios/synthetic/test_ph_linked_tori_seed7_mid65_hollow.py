"""Stage-1 seed=7 mild mid=0.65 hollow on dirty-b2 scaffold (#41 / A4-T55-followon).

A4-T51: primary hollow ``mid=0.5/h0=0.7`` on seed7 denser256 leaves dirty
``(2,1,1)`` / partial without clean ``(1,2,1)``. A4-T52: milder mid=0.65 on
seed77 preserved both-partial without dirty. This harness applies mid=0.65
hollow to the **seed7 dirty** scaffold asking whether milder prune cleans
void signal toward ``(1,2,1)``.

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
    format_lifetime_frac_sweep_table,
    nearest_data_labels,
    run_per_region_ph,
    sigma_star_from_tau,
    sweep_lifetime_frac_per_region,
)

EXPECTED_TORI: tuple[int, ...] = (1, 2, 1)
DIRTY_B2_EXAMPLE: tuple[int, ...] = (2, 1, 1)
N_PER_TORUS: int = 500
MAX_NODES: int = 256
DATASET_SEED: int = 2
STAGE1_SEED: int = 7
FRAC_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
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


def _summarize_lifetime_rows(rows):
    recovering: list[tuple[int, float, tuple[int, ...]]] = []
    clean_cells: list[tuple[int, float, tuple[int, ...]]] = []
    dirty_cells: list[tuple[int, float, tuple[int, ...]]] = []
    max_b1 = 0
    max_b2 = 0
    any_full = False
    for row in rows:
        betti = tuple(int(x) for x in row.betti)
        rid = int(row.region_id) if row.region_id is not None else -1
        max_b1 = max(max_b1, int(betti[1]) if len(betti) > 1 else 0)
        max_b2 = max(max_b2, int(betti[2]) if len(betti) > 2 else 0)
        if betti == EXPECTED_TORI:
            any_full = True
            recovering.append((rid, float(row.lifetime_frac), betti))
        if _is_clean_b2(betti):
            clean_cells.append((rid, float(row.lifetime_frac), betti))
        if _is_dirty_b2(betti):
            dirty_cells.append((rid, float(row.lifetime_frac), betti))
    return (
        any_full,
        max_b1,
        max_b2,
        tuple(recovering),
        tuple(clean_cells),
        tuple(dirty_cells),
    )


@dataclass(frozen=True)
class Seed7Mid65HollowBundle:
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
    signal_fixed_dirty_b2: bool
    fracs: tuple[float, ...]
    signal_table: str
    hollow_table: str
    signal_any_full: bool
    signal_any_clean_b2: bool
    signal_any_dirty_b2: bool
    signal_max_b1: int
    signal_max_b2: int
    hollow_any_full: bool
    hollow_any_clean_b2: bool
    hollow_any_dirty_b2: bool
    hollow_max_b1: int
    hollow_max_b2: int
    any_full_match: bool
    any_clean_b2: bool
    any_dirty_b2: bool
    max_b2: int
    hollow_recovering: tuple[tuple[int, float, tuple[int, ...]], ...]
    hollow_clean_cells: tuple[tuple[int, float, tuple[int, ...]], ...]
    hollow_dirty_cells: tuple[tuple[int, float, tuple[int, ...]], ...]


@pytest.fixture(scope="module")
def linked_tori_seed7_mid65_hollow_bundle() -> Seed7Mid65HollowBundle:
    """Fit seed7 denser256; mild mid=0.65 hollow; lifetime contrast."""
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
    n_after = int(np.sum(hollow_signal))
    if n_after < 8 or not (
        np.any(node_labels[hollow_signal] == 0)
        and np.any(node_labels[hollow_signal] == 1)
    ):
        hollow_pos, hollow_labs = signal_pos, signal_labs
        hollow_fallback = True
    else:
        hollow_pos = pos[hollow_signal]
        hollow_labs = node_labels[hollow_signal]
        hollow_fallback = False

    signal_fixed = run_per_region_ph(
        signal_pos,
        signal_labs,
        sigma,
        scenario="linked_tori_seed7_mid65_signal_fixed",
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
        scenario="linked_tori_seed7_mid65_hollow_fixed",
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
    signal_fixed_dirty = any(
        _is_dirty_b2(b) for b in signal_fixed_betti.values()
    )

    signal_rows = sweep_lifetime_frac_per_region(
        signal_pos,
        signal_labs,
        sigma,
        fracs=FRAC_GRID,
        include_labels=[0, 1],
        max_dim=2,
        filtration_mult=FILTRATION_MULTIPLIER,
        target_betti=EXPECTED_TORI,
    )
    hollow_rows = sweep_lifetime_frac_per_region(
        hollow_pos,
        hollow_labs,
        sigma,
        fracs=FRAC_GRID,
        include_labels=[0, 1],
        max_dim=2,
        filtration_mult=FILTRATION_MULTIPLIER,
        target_betti=EXPECTED_TORI,
    )
    (
        sig_full, sig_max_b1, sig_max_b2, _, sig_clean, sig_dirty,
    ) = _summarize_lifetime_rows(signal_rows)
    (
        hol_full, hol_max_b1, hol_max_b2, hol_rec, hol_clean, hol_dirty,
    ) = _summarize_lifetime_rows(hollow_rows)

    any_full = sig_full or hol_full
    any_clean = bool(sig_clean) or bool(hol_clean)
    any_dirty = bool(sig_dirty) or bool(hol_dirty)
    max_b2 = max(sig_max_b2, hol_max_b2)

    return Seed7Mid65HollowBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        max_nodes=MAX_NODES,
        n_signal=int(signal_pos.shape[0]),
        n_after_hollow=n_after,
        hollow_fallback=bool(hollow_fallback),
        sigma_star=float(sigma),
        mid_radius_frac=float(HOLLOW_CFG.mid_radius_frac),
        h0=float(HOLLOW_CFG.h0),
        signal_fixed_betti=signal_fixed_betti,
        hollow_fixed_betti=hollow_fixed_betti,
        signal_fixed_dirty_b2=signal_fixed_dirty,
        fracs=FRAC_GRID,
        signal_table=format_lifetime_frac_sweep_table(signal_rows),
        hollow_table=format_lifetime_frac_sweep_table(hollow_rows),
        signal_any_full=sig_full,
        signal_any_clean_b2=bool(sig_clean),
        signal_any_dirty_b2=bool(sig_dirty),
        signal_max_b1=sig_max_b1,
        signal_max_b2=sig_max_b2,
        hollow_any_full=hol_full,
        hollow_any_clean_b2=bool(hol_clean),
        hollow_any_dirty_b2=bool(hol_dirty),
        hollow_max_b1=hol_max_b1,
        hollow_max_b2=hol_max_b2,
        any_full_match=any_full,
        any_clean_b2=any_clean,
        any_dirty_b2=any_dirty,
        max_b2=max_b2,
        hollow_recovering=hol_rec,
        hollow_clean_cells=hol_clean,
        hollow_dirty_cells=hol_dirty,
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed7_mid65_hollow_harness_lands(
    linked_tori_seed7_mid65_hollow_bundle,
) -> None:
    """Seed7 mid=0.65 hollow lands; SI defaults untouched."""
    bundle = linked_tori_seed7_mid65_hollow_bundle
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
    assert len(bundle.signal_table) > 0
    assert len(bundle.hollow_table) > 0
    # T44/T48/T51 baseline: seed7 fixed reading carries dirty b2 on torus0.
    assert bundle.signal_fixed_dirty_b2
    assert bundle.signal_fixed_betti.get(0) == DIRTY_B2_EXAMPLE


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed7_mid65_hollow_documents_gap(
    linked_tori_seed7_mid65_hollow_bundle,
) -> None:
    """Document seed7 mid65 hollow vs clean (1,2,1); never flip awaiting.

    Soft: any full ``(1,2,1)`` or clean ``b2`` is proposal-path evidence.
    Otherwise keep documenting dirty-only / mild-hollow gap.
    """
    bundle = linked_tori_seed7_mid65_hollow_bundle
    if bundle.any_full_match or bundle.any_clean_b2:
        assert FILTRATION_MULTIPLIER == 1.5
        if bundle.any_full_match:
            assert (
                bundle.signal_any_full
                or bundle.hollow_any_full
                or len(bundle.hollow_recovering) >= 1
            )
        if bundle.any_clean_b2:
            assert bundle.max_b2 >= 1
    else:
        assert bundle.any_full_match is False
        assert bundle.any_clean_b2 is False
        assert bundle.hollow_recovering == ()
        assert bundle.hollow_clean_cells == ()
        assert bundle.signal_fixed_dirty_b2
        assert bundle.any_dirty_b2 or bundle.max_b2 >= 0
        assert bundle.n_after_hollow <= bundle.n_signal
